"""Job documents in Redis and their RQ jobs.

A job is a JSON document (`cad:job:<id>`) that is the API's source of truth, plus an RQ job that runs it. All changes
of the document go through `update`, a compare-and-set on the status, so a cancel, a finish and a re-run never
overwrite each other. `external_id` is the idempotency key (`cad:ext:<external_id>` -> job id)."""
import json
import uuid
from datetime import timedelta
from functools import lru_cache

import redis
from rq import Queue, Retry

from .config import Settings

INTERACTIVE, BULK, CALLBACKS = "cad-interactive", "cad-bulk", "cad-callbacks"
LIBRARIES = "cad:libraries"  # the symbol libraries the workers have installed (JSON), written when a worker starts
QUEUES = {"interactive": INTERACTIVE, "bulk": BULK}
ACTIVE = ("queued", "running")
# A job that waits in the queue for longer than this loses its document (and is dropped by the worker).
ACTIVE_TTL_SECONDS = 7 * 86_400
PUBLIC = ("id", "external_id", "status", "attempts", "operations", "kind", "format", "markdown",
          "preview_content_type", "result_content_type", "warnings", "error")


@lru_cache
def _connect(url: str) -> redis.Redis:
    return redis.Redis.from_url(url, socket_connect_timeout=3, socket_timeout=10, health_check_interval=30)


def connection() -> redis.Redis:
    return _connect(Settings.from_env().redis_url)


def _key(job_id: str) -> str:
    return f"cad:job:{job_id}"


def _save(pipe, doc: dict) -> None:
    ttl = ACTIVE_TTL_SECONDS if doc["status"] in ACTIVE else Settings.from_env().result_ttl_seconds
    pipe.set(_key(doc["id"]), json.dumps(doc), ex=ttl)


def get(job_id: str) -> dict | None:
    raw = connection().get(_key(job_id))
    return json.loads(raw) if raw else None


def get_many(job_ids: list[str]) -> list[dict | None]:
    """One round trip for many documents; None for unknown or expired ids."""
    if not job_ids:
        return []
    return [json.loads(raw) if raw else None for raw in connection().mget([_key(i) for i in job_ids])]


def view(doc: dict) -> dict:
    """The JSON of GET /jobs/<id> and of the callback: the public fields that are set."""
    return {k: doc[k] for k in PUBLIC if doc.get(k) is not None}


def create(doc: dict) -> tuple[dict, bool]:
    """Saves a new queued job unless a non-terminal job with the same external_id exists. Returns (job, created)."""
    r, ext_key = connection(), f"cad:ext:{doc['external_id']}"
    doc = {**doc, "id": uuid.uuid4().hex, "status": "queued", "attempts": 0, "warnings": [], "requeues": 0}
    doc["rq_id"] = doc["id"]

    def txn(pipe):
        existing_id = pipe.get(ext_key)
        raw = pipe.get(_key(existing_id.decode())) if existing_id else None
        if raw and json.loads(raw)["status"] in ACTIVE:
            return json.loads(raw), False
        pipe.multi()
        pipe.set(ext_key, doc["id"], ex=ACTIVE_TTL_SECONDS)
        _save(pipe, doc)
        return doc, True

    return r.transaction(txn, ext_key, value_from_callable=True)


def update(job_id: str, expect: tuple[str, ...], **fields) -> dict | None:
    """Sets fields on the job if its status is one of `expect`. Returns the new document, None if it did not apply."""
    def txn(pipe):
        raw = pipe.get(_key(job_id))
        if not raw or (doc := json.loads(raw))["status"] not in expect:
            return None
        doc.update(fields)
        pipe.multi()
        _save(pipe, doc)
        return doc

    return connection().transaction(txn, _key(job_id), value_from_callable=True)


def delete(job_id: str) -> None:
    connection().delete(_key(job_id))


def enqueue(doc: dict, delay: int = 0) -> None:
    """Puts the job on its priority queue. RQ re-runs it (with backoff) when it raises or its worker disappears; the
    number of re-runs is what is left of MAX_ATTEMPTS."""
    s = Settings.from_env()
    left = s.max_attempts - doc["attempts"] - 1
    delays = [s.retry_delay_seconds * 4**i for i in range(left)]
    queue = Queue(QUEUES[doc["priority"]], connection=connection())
    kwargs = {"job_id": doc["rq_id"], "job_timeout": s.conversion_timeout_seconds, "result_ttl": 0,
              "failure_ttl": s.result_ttl_seconds, "retry": Retry(max=left, interval=delays) if left > 0 else None}
    if delay:
        queue.enqueue_in(timedelta(seconds=delay), "app.jobs.run_job", doc["id"], **kwargs)
    else:
        queue.enqueue("app.jobs.run_job", doc["id"], **kwargs)


def requeue(doc: dict, delay: int) -> None:
    """Runs the job again later without it counting as an attempt (not enough free disk)."""
    doc = update(doc["id"], ("running",), status="queued", attempts=doc["attempts"] - 1,
                 requeues=doc["requeues"] + 1, rq_id=f"{doc['id']}-{doc['requeues'] + 1}")
    if doc:
        enqueue(doc, delay)


def finish(job_id: str, status: str, **fields) -> dict | None:
    """Ends a running job and sends the callback. Returns None if the job was cancelled meanwhile."""
    doc = update(job_id, ACTIVE, status=status, **fields)
    if doc and doc.get("callback_url"):
        s = Settings.from_env()
        delays = [s.retry_delay_seconds * 2**i for i in range(6)]  # about 30 minutes in total
        Queue(CALLBACKS, connection=connection()).enqueue(
            "app.jobs.send_callback", job_id, retry=Retry(max=len(delays), interval=delays), job_timeout=60,
            result_ttl=0, failure_ttl=s.result_ttl_seconds)
    return doc


def queued_count() -> dict[str, int]:
    r = connection()
    return {name: r.llen(Queue(queue, connection=r).key) for name, queue in QUEUES.items()}


def running_count() -> int:
    r = connection()
    return sum(r.zcard(f"rq:wip:{queue}") for queue in QUEUES.values())


def publish_libraries(catalogue: dict) -> None:
    """Stores the installed symbol libraries for GET /libraries and for checking the libraries a job names. All
    workers are expected to have the same libraries; the last one to start writes the list."""
    connection().set(LIBRARIES, json.dumps(catalogue))


def libraries() -> dict | None:
    """The catalogue of publish_libraries; None before any worker started."""
    raw = connection().get(LIBRARIES)
    return json.loads(raw) if raw else None
