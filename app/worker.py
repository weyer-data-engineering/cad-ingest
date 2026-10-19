"""Worker entry point: `python -m app.worker` runs a pool of MAX_CONCURRENT_CONVERSIONS RQ workers, each forking one
work-horse per job. `python -m app.worker check` is the health check (exit 0 if a worker of this host is registered).

The conversion runs in the work-horse, which RQ kills at the job timeout and which is the OOM killer's first victim and
address-space capped (see `limit_process`). What happens when a horse or a whole worker is lost:
- worker alive, horse killed by the kernel (SIGKILL) or crashed: the file's fault, the job fails at once;
- worker dead (container killed or restarted): RQ finds the job abandoned once its heartbeat has run out (about 70 s)
  and re-queues it while it has re-runs left, `on_abandoned` keeps the job document in step."""
import glob
import logging
import os
import resource
import shutil
import signal
import socket
import sys
import tempfile
from contextlib import suppress

import redis
from rq import Worker
from rq.exceptions import AbandonedJobError
from rq.utils import now
from rq.worker_pool import WorkerPool

from . import jobs, store
from .config import Settings
from .errors import JobError
from .plan2cad import library

log = logging.getLogger(__name__)


def limit_process() -> None:
    """Volunteers as the OOM killer's first victim (so the kernel never picks the worker) and caps the memory."""
    with suppress(OSError), open("/proc/self/oom_score_adj", "w") as f:
        f.write("1000")  # not Linux or not allowed: the memory limit still applies
    limit = Settings.from_env().worker_memory_mb() * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _rerun_or_fail(job, message: str) -> None:
    """The job was lost for a reason that is not the file's: RQ re-runs it if it has re-runs left, else it fails."""
    if job.retries_left:
        store.update(job.args[0], store.ACTIVE, status="queued")
    else:
        store.finish(job.args[0], "failed", error=JobError("internal", message, retryable=True).as_dict())


def on_abandoned(job, exc_type, exc_value, traceback) -> bool:
    """RQ exception handler. It is called for every failure; abandoned jobs (worker died) are the case it is for."""
    if exc_type is AbandonedJobError and job.origin != store.CALLBACKS:
        _rerun_or_fail(job, "The worker was lost while converting the file")
    return True


class CadWorker(Worker):
    def __init__(self, *args, **kwargs):
        # Timings tighter than RQ's defaults, so a lost worker's jobs are noticed within about two minutes.
        super().__init__(*args, worker_ttl=45, maintenance_interval=30, job_monitoring_interval=10, **kwargs)

    def main_work_horse(self, job, queue):
        limit_process()
        super().main_work_horse(job, queue)

    def handle_work_horse_killed(self, job, retpid, ret_val, rusage) -> None:
        """Called in the worker when a horse died without finishing, just before RQ decides about a re-run."""
        super().handle_work_horse_killed(job, retpid, ret_val, rusage)
        if job.origin == store.CALLBACKS:
            return  # RQ's retry covers it
        shutil.rmtree(jobs.workdir(job.args[0]), ignore_errors=True)
        if self._stopped_job_id == job.id:
            return  # cancelled by DELETE: the document says so already
        signalled = os.WTERMSIG(ret_val) if os.WIFSIGNALED(ret_val) else None
        timeout = job.timeout or 0
        if timeout and (now() - job.started_at).total_seconds() >= timeout:
            error = JobError("timeout", f"The conversion exceeded {timeout} seconds")
        elif signalled == signal.SIGKILL:  # the kernel's OOM killer or the address-space limit
            error = JobError("out_of_memory", "The file needs more memory than allowed")
        elif signalled in (None, signal.SIGTERM):
            return _rerun_or_fail(job, "The work-horse was lost")
        else:
            error = JobError("unreadable", f"The converter crashed on this file (signal {signalled})")
        job.retries_left = 0  # the file's fault: no automatic re-run
        store.finish(job.args[0], "failed", error=error.as_dict())


def registered() -> bool:
    """True if a worker of this host is registered in Redis."""
    try:
        return any(w.hostname == socket.gethostname() for w in Worker.all(connection=store.connection()))
    except redis.RedisError:
        return False


def load_libraries(settings: Settings) -> None:
    """Reads and checks the installed symbol libraries before the workers fork, so that every work-horse finds them
    parsed, and publishes them for the API. A broken library or an unknown default stops the worker."""
    try:
        available = library.installed(settings.symbol_libraries)
        defaults = library.select(folders=settings.symbol_libraries, defaults=settings.default_symbol_libraries)
    except library.LibraryError as e:
        log.error("Symbol libraries: %s", e)
        sys.exit(1)
    for lib in available.values():
        log.info("Symbol library %s %s: %s symbols from %s%s", lib.id, lib.version or "(no version)", len(lib.symbols),
                 lib.path, f", {len(lib.warnings)} notes (see `cad-ingest library check`)" if lib.warnings else "")
    store.publish_libraries({"libraries": [lib.summary() for lib in available.values()],
                             "default": [lib.id for lib in defaults]})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if sys.argv[1:] == ["check"]:
        sys.exit(0 if registered() else 1)
    settings = Settings.from_env()
    for stale in glob.glob(os.path.join(tempfile.gettempdir(), "cad-*")):  # left by horses that were killed
        shutil.rmtree(stale, ignore_errors=True)
    load_libraries(settings)
    # A connection of its own: RQ raises the socket timeout of the one it gets, for its blocking reads.
    WorkerPool([store.CALLBACKS, store.INTERACTIVE, store.BULK], redis.Redis.from_url(settings.redis_url),
               num_workers=settings.max_concurrent_conversions, worker_class=CadWorker,
               exception_handlers=[on_abandoned]).start()


if __name__ == "__main__":
    main()
