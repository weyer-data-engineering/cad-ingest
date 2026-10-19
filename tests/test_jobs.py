"""The job service end to end: the API in-process, real Redis, real worker processes, a local HTTP server that plays
the storage and the client (see conftest.py)."""
import os
import signal
import socket
import subprocess
import sys

import pytest
from rq import Queue

from app import config, store
from app.config import Settings

from .conftest import REPO, gone, horses, kill_worker, submit, until, wait_for_status

TERMINAL = ("succeeded", "failed")


def test_succeeds_with_markdown_preview_and_callback(api, origin, worker):
    """One 2D and one 3D format, through the whole chain."""
    worker()
    for name, kind, fmt, text, preview_type, magic in (
            ("drawing.dxf", "DRAWING_2D", "dxf", "DWG-4711", "application/pdf", b"%PDF"),
            ("assembly.step", "MODEL_3D", "step", "Box part", "model/gltf-binary", b"glTF")):
        response = submit(api, origin, name)
        assert response.status_code == 202 and response.json()["status"] == "queued"
        job = wait_for_status(api, response.json()["id"], *TERMINAL)
        assert job["status"] == "succeeded" and job["attempts"] == 1 and "error" not in job
        assert (job["kind"], job["format"], job["operations"]) == (kind, fmt, ["extract", "preview"])
        assert text in job["markdown"] and job["preview_content_type"] == preview_type
        data, content_type = origin.previews[name]
        assert data.startswith(magic) and content_type == preview_type
        until(lambda: origin.callbacks_of(name), 10)
        assert origin.callbacks_of(name) == [(job, 200)]  # the callback carries what GET returns


def test_only_requested_operations_run(api, origin, worker):
    worker()
    job_id = submit(api, origin, "box.stl", operations=["extract"], preview_url=None).json()["id"]
    job = wait_for_status(api, job_id, *TERMINAL)
    assert "watertight" in job["markdown"] and "preview_content_type" not in job and not origin.previews
    job_id = submit(api, origin, "box.stl", "preview-only", operations=["preview"]).json()["id"]
    job = wait_for_status(api, job_id, *TERMINAL)
    assert "markdown" not in job and job["preview_content_type"] == "model/gltf-binary"


def test_same_external_id_returns_the_existing_job(api, origin):
    first = submit(api, origin, "drawing.dxf", "file1:task1")
    again = submit(api, origin, "drawing.dxf", "file1:task1")
    other = submit(api, origin, "drawing.dxf", "file1:task2")
    assert (first.status_code, again.status_code, other.status_code) == (202, 200, 202)
    assert again.json()["id"] == first.json()["id"] != other.json()["id"]
    assert again.json()["status"] == "queued"
    assert Queue(store.INTERACTIVE, connection=store.connection()).count == 2
    # once the job is terminal (here: cancelled) the same key starts a new job
    assert api.delete(f"/jobs/{first.json()['id']}").status_code == 204
    assert submit(api, origin, "drawing.dxf", "file1:task1").json()["id"] != first.json()["id"]


def test_status_of_many_jobs(api, origin, worker):
    worker()
    ids = [submit(api, origin, "box.stl", f"job{i}").json()["id"] for i in range(3)]
    for job_id in ids:
        wait_for_status(api, job_id, "succeeded")
    response = api.post("/jobs/status", json={"ids": [*ids, "unknown"]})
    body = response.json()
    assert response.status_code == 200 and body["missing"] == ["unknown"]
    assert [j["id"] for j in body["jobs"]] == ids
    assert all(j["status"] == "succeeded" and "markdown" not in j for j in body["jobs"])
    assert api.post("/jobs/status", json={"ids": []}).json() == {"jobs": [], "missing": []}
    assert api.post("/jobs/status", json={"ids": ["x"] * 501}).status_code == 400
    assert api.post("/jobs/status", json={"ids": ["x"] * 500}).status_code == 200


def test_submit_validation(api, origin, monkeypatch):
    assert submit(api, origin, "model.xyz").status_code == 415
    assert submit(api, origin, "noextension").status_code == 415
    assert submit(api, origin, "drawing.dxf", preview_url=None).status_code == 400  # preview needs a target
    assert submit(api, origin, "drawing.dxf", operations=[]).status_code == 400
    assert submit(api, origin, "drawing.dxf", operations=["print"]).status_code == 400
    assert submit(api, origin, "drawing.dxf", source_url="file:///etc/passwd").status_code == 400
    assert api.post("/jobs", json={}).status_code == 400
    assert api.get("/jobs/unknown").status_code == 404 and api.delete("/jobs/unknown").status_code == 404
    monkeypatch.setenv("API_TOKEN", "s3cret")
    assert submit(api, origin, "drawing.dxf").status_code == 401
    assert api.post("/jobs/status", json={"ids": []}).status_code == 401
    assert api.get("/jobs/unknown", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert api.get("/jobs/unknown", headers={"Authorization": "Bearer s3cret"}).status_code == 404
    assert api.get("/health").status_code == 200  # no token needed


def test_queue_full_is_503(api, origin, monkeypatch):
    monkeypatch.setenv("MAX_QUEUED_JOBS", "1")
    assert submit(api, origin, "drawing.dxf", "a").status_code == 202
    full = submit(api, origin, "drawing.dxf", "b")
    assert full.status_code == 503 and full.headers["retry-after"] == "15"


def test_redis_down_is_503(api, origin, monkeypatch):
    monkeypatch.setenv("REDIS_URL", f"redis://127.0.0.1:{_closed_port()}/0")
    response = submit(api, origin, "drawing.dxf")
    assert response.status_code == 503 and response.headers["retry-after"] == "15"
    assert api.get("/health").status_code == 503


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_health(api, origin, worker):
    submit(api, origin, "drawing.dxf", "a")
    submit(api, origin, "drawing.dxf", "b", priority="bulk")
    body = api.get("/health").json()
    assert (body["status"], body["redis"], body["workers"]) == ("ok", "ok", 0)
    assert body["queued"] == {"interactive": 1, "bulk": 1} and body["running"] == 0
    worker()
    assert api.get("/health").json()["workers"] >= 1


def test_broken_file_fails_without_a_rerun(api, origin, worker):
    origin.files["broken.step"] = b"this is not a CAD file\n" * 20
    worker()
    job = wait_for_status(api, submit(api, origin, "broken.step").json()["id"], *TERMINAL)
    assert job["status"] == "failed" and job["attempts"] == 1
    assert job["error"]["code"] == "unreadable" and job["error"]["retryable"] is False and job["error"]["message"]
    assert "markdown" not in job and not origin.previews
    until(lambda: origin.callbacks_of("broken.step"), 10)
    assert origin.callbacks_of("broken.step") == [(job, 200)]


def test_source_failure_is_run_again(api, origin, worker):
    origin.source_status["box.stl"] = [503]
    worker()
    job = wait_for_status(api, submit(api, origin, "box.stl").json()["id"], *TERMINAL)
    assert job["status"] == "succeeded" and job["attempts"] == 2 and origin.hits["box.stl"] == 2
    until(lambda: origin.callbacks_of("box.stl"), 10)
    assert len(origin.callbacks_of("box.stl")) == 1  # nothing was reported for the failed attempt


def test_source_failure_stops_after_max_attempts(api, origin, worker):
    origin.source_status["box.stl"] = [503] * 5
    worker()
    job = wait_for_status(api, submit(api, origin, "box.stl").json()["id"], *TERMINAL, seconds=90)
    assert (job["status"], job["attempts"], origin.hits["box.stl"]) == ("failed", 3, 3)
    assert job["error"]["code"] == "source_unavailable" and job["error"]["retryable"] is True
    until(lambda: origin.callbacks_of("box.stl"), 10)


def test_expired_source_url_is_not_run_again(api, origin, worker):
    origin.source_status["box.stl"] = [403]
    worker()
    job = wait_for_status(api, submit(api, origin, "box.stl").json()["id"], *TERMINAL)
    assert (job["status"], job["attempts"], origin.hits["box.stl"]) == ("failed", 1, 1)
    assert job["error"]["code"] == "source_unavailable" and job["error"]["retryable"] is True  # the client may resubmit


def test_preview_upload_failure_is_run_again(api, origin, worker):
    origin.preview_status["box.stl"] = [500]
    worker()
    job = wait_for_status(api, submit(api, origin, "box.stl").json()["id"], *TERMINAL)
    assert job["status"] == "succeeded" and job["attempts"] == 2 and "box.stl" in origin.previews


@pytest.mark.parametrize("chunked", [False, True])
def test_too_large(api, origin, worker, chunked):
    """Rejected from Content-Length before the download, or while streaming when there is none."""
    if chunked:
        origin.chunked.add("box.stl")
    worker(MAX_FILE_SIZE_BYTES=100)
    job = wait_for_status(api, submit(api, origin, "box.stl").json()["id"], *TERMINAL)
    assert job["status"] == "failed" and job["attempts"] == 1
    assert job["error"]["code"] == "too_large" and job["error"]["retryable"] is False


def test_priority_queues(api, origin, worker):
    submit(api, origin, "box.stl", "bulk-job", priority="bulk")
    submit(api, origin, "box.stl", "interactive-job")
    worker(MAX_CONCURRENT_CONVERSIONS=1)
    until(lambda: len(origin.callbacks) == 2)
    assert [name for name, _, _ in origin.callbacks] == ["interactive-job", "bulk-job"]


def test_not_enough_disk_waits_instead_of_failing(api, origin, worker):
    process = worker(MIN_FREE_DISK_MB=10**9)
    job_id = submit(api, origin, "box.stl").json()["id"]
    until(lambda: Queue(store.INTERACTIVE, connection=store.connection()).scheduled_job_registry.count
          or Queue(store.INTERACTIVE, connection=store.connection()).count, 20)
    job = api.get(f"/jobs/{job_id}").json()
    assert job["status"] == "queued" and job["attempts"] == 0 and not origin.hits  # no attempt used, no download
    kill_worker(process)
    worker()  # space is back
    assert wait_for_status(api, job_id, *TERMINAL)["status"] == "succeeded"


def test_delete_queued_job(api, origin, worker):
    job_id = submit(api, origin, "box.stl").json()["id"]
    assert api.delete(f"/jobs/{job_id}").status_code == 204
    assert api.get(f"/jobs/{job_id}").json()["status"] == "cancelled"
    assert Queue(store.INTERACTIVE, connection=store.connection()).count == 0
    worker()
    other = wait_for_status(api, submit(api, origin, "box.stl", "other").json()["id"], *TERMINAL)
    assert other["status"] == "succeeded" and not origin.hits["nothing"] and not origin.callbacks_of("box.stl")


def test_delete_finished_job_removes_the_result(api, origin, worker):
    worker()
    job_id = submit(api, origin, "box.stl").json()["id"]
    wait_for_status(api, job_id, "succeeded")
    assert api.delete(f"/jobs/{job_id}").status_code == 204
    assert api.get(f"/jobs/{job_id}").status_code == 404


def test_delete_running_job_kills_the_horse_and_sends_no_callback(api, origin, worker):
    process = worker()
    origin.gate.clear()  # the download hangs, so the job is certainly running
    job_id = submit(api, origin, "box.stl").json()["id"]
    until(lambda: origin.hits["box.stl"] == 1)
    horse = until(lambda: horses(process))[0]
    assert api.delete(f"/jobs/{job_id}").status_code == 204
    until(lambda: gone(horse), 3, "the work-horse is still alive 3 seconds after DELETE")
    assert api.get(f"/jobs/{job_id}").json()["status"] == "cancelled"
    origin.gate.set()
    other = wait_for_status(api, submit(api, origin, "box.stl", "other").json()["id"], *TERMINAL)  # the worker lives on
    assert other["status"] == "succeeded" and not origin.callbacks_of("box.stl") and not origin.previews.get("box.stl")


def test_horse_killed_by_the_kernel_fails_the_job(api, origin, worker):
    """The OOM killer (SIGKILL) takes the work-horse while the worker lives on: the file needs too much memory."""
    process = worker()
    origin.gate.clear()
    job_id = submit(api, origin, "box.stl").json()["id"]
    until(lambda: origin.hits["box.stl"] == 1)
    os.kill(until(lambda: horses(process))[0], signal.SIGKILL)
    job = wait_for_status(api, job_id, *TERMINAL)
    assert job["status"] == "failed" and job["attempts"] == 1
    assert job["error"]["code"] == "out_of_memory" and job["error"]["retryable"] is False
    until(lambda: origin.callbacks_of("box.stl"), 10)
    origin.gate.set()


def abandon(api, origin, worker, job_id: str, **env):
    """Kills the whole worker (not just the horse) in the middle of the job and lets its heartbeat run out."""
    origin.gate.clear()  # before the worker starts, or a fast worker downloads and finishes the file first
    process = worker(**env)
    until(lambda: origin.hits["box.stl"] == 1)
    until(lambda: api.get(f"/jobs/{job_id}").json()["status"] == "running")
    horse_pids = horses(process)
    kill_worker(process)
    for pid in horse_pids:
        os.kill(pid, signal.SIGKILL)
    assert api.get(f"/jobs/{job_id}").json()["status"] == "running"  # nobody knows yet
    registry = store.connection()
    members = registry.zrange(f"rq:wip:{store.INTERACTIVE}", 0, -1)
    assert members
    for member in members:  # RQ expires the entry after the heartbeat TTL (about 70 s); do not wait for it
        registry.zadd(f"rq:wip:{store.INTERACTIVE}", {member: 1}, xx=True)
    # The dead worker still holds the registry-cleanup and scheduler locks until they expire (30-60 s). In
    # production the next worker simply waits for that; here it would race the test's deadline, so expire them now.
    for pattern in ("rq:clean_registries:*", "rq:scheduler-lock:*"):
        for key in registry.scan_iter(pattern):
            registry.delete(key)
    origin.gate.set()


def test_abandoned_job_is_run_again(api, origin, worker):
    job_id = submit(api, origin, "box.stl").json()["id"]
    abandon(api, origin, worker, job_id)
    worker()  # a new worker (the restarted container) finds the abandoned job and re-queues it
    job = wait_for_status(api, job_id, *TERMINAL)
    assert job["status"] == "succeeded" and job["attempts"] == 2 and origin.hits["box.stl"] == 2
    until(lambda: origin.callbacks_of("box.stl"), 10)
    assert len(origin.callbacks_of("box.stl")) == 1


def test_abandoned_job_fails_when_no_run_is_left(api, origin, worker, monkeypatch):
    monkeypatch.setenv("MAX_ATTEMPTS", "1")
    job_id = submit(api, origin, "box.stl").json()["id"]
    abandon(api, origin, worker, job_id, MAX_ATTEMPTS=1)
    worker(MAX_ATTEMPTS=1)
    job = wait_for_status(api, job_id, *TERMINAL)
    assert job["status"] == "failed" and job["error"]["code"] == "internal" and job["error"]["retryable"] is True
    until(lambda: origin.callbacks_of("box.stl"), 10)


def test_callback_is_retried_on_5xx(api, origin, worker):
    origin.callback_status["box.stl"] = [500, 429]
    worker()
    wait_for_status(api, submit(api, origin, "box.stl").json()["id"], *TERMINAL)
    until(lambda: len(origin.callbacks_of("box.stl")) == 3, 30)
    assert [status for _, status in origin.callbacks_of("box.stl")] == [500, 429, 200]


def test_callback_accepts_202(api, origin, worker):
    origin.callback_status["box.stl"] = [202]
    worker()
    wait_for_status(api, submit(api, origin, "box.stl").json()["id"], *TERMINAL)
    until(lambda: origin.callbacks_of("box.stl"), 10)
    assert [status for _, status in origin.callbacks_of("box.stl")] == [202]  # accepted, not retried


def test_callback_stops_on_4xx(api, origin, worker):
    origin.callback_status["box.stl"] = [410, 200]  # 410: the client forgot the file; do not ask again
    worker()
    job_id = submit(api, origin, "box.stl").json()["id"]
    wait_for_status(api, job_id, *TERMINAL)
    until(lambda: origin.callbacks_of("box.stl"), 10)
    until(lambda: not Queue(store.CALLBACKS, connection=store.connection()).scheduled_job_registry.count, 10)
    assert [status for _, status in origin.callbacks_of("box.stl")] == [410]
    assert api.get(f"/jobs/{job_id}").json()["status"] == "succeeded"  # the result stays until it expires


def test_worker_limits_its_horses():
    """What the horse does first: oom_score_adj 1000 and the address-space cap."""
    code = ("import resource; from app.worker import limit_process; limit_process();"
            "print(resource.getrlimit(resource.RLIMIT_AS)[0], open('/proc/self/oom_score_adj').read().strip())")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                         env={**os.environ, "MEMORY_LIMIT_MB": "1234"}).stdout.split()
    assert out == [str(1234 * 2**20), "1000"]


def test_memory_limit_default_from_cgroup(tmp_path, monkeypatch):
    limit = tmp_path / "memory.max"
    monkeypatch.setattr(config, "CGROUP_MEMORY_FILES", (str(limit), str(tmp_path / "missing")))
    limit.write_text(f"{2 * 2**30}\n")
    assert Settings().worker_memory_mb() == int(2048 * 0.85)  # not divided by the horses
    assert Settings(max_concurrent_conversions=4).worker_memory_mb() == int(2048 * 0.85)
    assert Settings(memory_limit_mb=100).worker_memory_mb() == 100  # explicit value wins
    limit.write_text("max\n")
    assert Settings().worker_memory_mb() == 8192
    limit.write_text("9223372036854771712\n")  # cgroup v1 "unlimited"
    assert Settings().worker_memory_mb() == 8192
