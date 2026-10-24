"""Job API. Stateless: all state is in Redis (see store.py), so several api processes or replicas are fine."""
import hmac
import logging
import os
from contextlib import suppress
from pathlib import PurePosixPath
from typing import Annotated, Literal

import redis
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from rq import Worker
from rq.command import send_stop_job_command
from rq.exceptions import InvalidJobOperation, NoSuchJobError
from rq.job import Job

from . import store
from .config import RETRY_AFTER_SECONDS, VERSION, Settings
from .formats import FORMATS, SCAN_2D, formats_by_kind

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="cad-ingest", version=VERSION,
              description="Job API: CAD files to markdown and PDF/GLB previews, photos and scans of drawings to DXF.")


def _busy(message: str) -> HTTPException:
    return HTTPException(503, message, headers={"Retry-After": str(RETRY_AFTER_SECONDS)})


@app.exception_handler(RequestValidationError)
async def _invalid(_: Request, e: RequestValidationError) -> JSONResponse:
    detail = "; ".join(f"{'.'.join(map(str, err['loc'][1:]))}: {err['msg']}" for err in e.errors())
    return JSONResponse({"detail": detail}, status_code=400)


@app.exception_handler(redis.RedisError)
async def _redis_unavailable(_: Request, e: redis.RedisError) -> JSONResponse:
    logging.getLogger(__name__).error("Redis unavailable: %s", type(e).__name__)
    return JSONResponse({"detail": "Redis is unavailable, please retry later"}, status_code=503,
                        headers={"Retry-After": str(RETRY_AFTER_SECONDS)})


def _authorized(authorization: str = Header(default="")) -> None:
    token = Settings.from_env().api_token
    if token and not hmac.compare_digest(authorization.encode(), f"Bearer {token}".encode()):
        raise HTTPException(401, "Missing or wrong API token", headers={"WWW-Authenticate": "Bearer"})


jobs = APIRouter(prefix="/jobs", dependencies=[Depends(_authorized)])
LibraryId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]  # as plan2cad.library.ID
Url = Annotated[str, Field(pattern=r"^https?://")]


class VectorizeOptions(BaseModel):
    """How a photo or scan is read (operation "vectorize")."""

    paper: Literal["A0", "A1", "A2", "A3", "A4"] = "A3"  # the paper the drawing is printed on
    scale: int | None = Field(default=None, ge=1, le=10_000)  # e.g. 50 for 1:50; read from the title block if not set
    dpi: float | None = Field(default=None, gt=50, le=1200)  # scan resolution, only needed without visible paper edges
    drawing: Literal["auto", "building", "pid"] = "auto"  # floor plan or P&ID; recognised from the texts if "auto"
    # Installed symbol libraries (GET /libraries) to recognise symbols from, earlier ones first; the default ones if
    # neither these nor library_urls are given
    libraries: list[LibraryId] | None = Field(default=None, min_length=1, max_length=20)
    # Pages of a PDF or TIFF to convert, counted from 1; all of them (up to MAX_SCAN_PAGES) if not set
    pages: list[Annotated[int, Field(ge=1)]] | None = Field(default=None, min_length=1, max_length=1000)


class Reviewer(BaseModel):
    """A vision model behind an OpenAI-compatible endpoint, such as a LiteLLM proxy, that is asked about what the
    recognition of a photo or scan is unsure of (operation "vectorize"). It sees only crops of those places."""

    base_url: Url  # where /chat/completions is, e.g. https://llm.example.com/v1
    api_key: str = Field(min_length=1, max_length=1000, repr=False)
    model: str = Field(min_length=1, max_length=200)


class JobRequest(BaseModel):
    external_id: str = Field(min_length=1, max_length=255)
    file_name: str = Field(min_length=1, max_length=512)
    source_url: str = Field(pattern=r"^https?://")
    operations: list[Literal["extract", "preview", "vectorize"]] = Field(min_length=1)
    preview_url: str | None = Field(default=None, pattern=r"^https?://")
    result_url: str | None = Field(default=None, pattern=r"^https?://")
    options: VectorizeOptions | None = None
    # Presigned GETs of symbol libraries (a library ZIP or a DXF) that the job brings along; they take precedence over
    # the installed ones
    library_urls: list[Url] | None = Field(default=None, min_length=1, max_length=10)
    callback_url: str | None = Field(default=None, pattern=r"^https?://")
    priority: Literal["interactive", "bulk"] = "interactive"
    # Asked about the texts and symbols to check (vectorize); kept only until the job ends, never returned or logged
    reviewer: Reviewer | None = None

    @model_validator(mode="after")
    def _uploads_need_a_target(self) -> "JobRequest":
        if "preview" in self.operations and not self.preview_url:
            raise ValueError("preview_url is required for the operation 'preview'")
        if "vectorize" in self.operations and not self.result_url:
            raise ValueError("result_url is required for the operation 'vectorize'")
        if self.library_urls and "vectorize" not in self.operations:
            raise ValueError("library_urls are for the operation 'vectorize'")
        if self.reviewer and "vectorize" not in self.operations:
            raise ValueError("reviewer is for the operation 'vectorize'")
        return self


@jobs.post("", status_code=202)
def submit(body: JobRequest, response: Response) -> dict:
    """Queues a job. A non-terminal job with the same external_id is returned instead (200)."""
    name = os.path.basename(body.file_name.replace("\\", "/"))
    ext = PurePosixPath(name).suffix.lstrip(".").lower()
    if ext not in FORMATS:
        raise HTTPException(415, f"Unsupported file extension '{ext}'")
    scan = FORMATS[ext][0] == SCAN_2D
    if scan != ("vectorize" in body.operations):  # a photo is only read when asked for: it takes a while
        raise HTTPException(415, f"'{ext}' files need the operation 'vectorize'" if scan
                            else f"The operation 'vectorize' is for photos and scans, not for '{ext}' files")
    _check_libraries(body.options.libraries if body.options else None)
    if sum(store.queued_count().values()) >= Settings.from_env().max_queued_jobs:
        raise _busy("Too many jobs are queued, please retry later")
    doc, created = store.create({
        "external_id": body.external_id, "file_name": name, "kind": FORMATS[ext][0], "format": ext,
        "operations": sorted(set(body.operations)), "source_url": body.source_url, "preview_url": body.preview_url,
        "result_url": body.result_url, "options": body.options.model_dump() if body.options else None,
        "library_urls": body.library_urls, "callback_url": body.callback_url, "priority": body.priority,
        "reviewer": body.reviewer.model_dump() if body.reviewer else None})
    if not created:
        response.status_code = 200
        return store.view(doc)
    try:
        store.enqueue(doc)
    except BaseException:
        store.delete(doc["id"])  # no half-queued job that would answer this external_id forever
        raise
    return {"id": doc["id"], "status": "queued"}


def _check_libraries(ids: list[str] | None) -> None:
    """Rejects libraries that the workers do not have. Before any worker has started this is left to the worker."""
    catalogue = store.libraries() if ids else None
    if catalogue:
        known = [lib["id"] for lib in catalogue["libraries"]]
        if unknown := [i for i in ids if i not in known]:
            raise HTTPException(400, f"Unknown symbol library: {', '.join(unknown)} (installed: {', '.join(known)})")


class StatusRequest(BaseModel):
    ids: list[str] = Field(max_length=500)


@jobs.post("/status")
def job_status(body: StatusRequest) -> dict:
    """Status of many jobs in one request (for reconciliation): like GET /jobs/<id> but without the markdown."""
    docs = store.get_many(body.ids)
    return {"jobs": [{k: v for k, v in store.view(doc).items() if k != "markdown"} for doc in docs if doc],
            "missing": [i for i, doc in zip(body.ids, docs) if not doc]}


@jobs.get("/{job_id}")
def get_job(job_id: str) -> dict:
    doc = store.get(job_id)
    if not doc:
        raise HTTPException(404, "Unknown or expired job")
    return store.view(doc)


@jobs.delete("/{job_id}", status_code=204)
def cancel_job(job_id: str) -> Response:
    """Queued: removed from the queue. Running: its work-horse is killed. Finished: the result is deleted."""
    doc = store.get(job_id)
    if not doc:
        raise HTTPException(404, "Unknown or expired job")
    if doc["status"] not in store.ACTIVE:
        store.delete(job_id)
    elif store.update(job_id, store.ACTIVE, status="cancelled", reviewer=None):  # first: no result or callback follows
        _stop(doc["rq_id"])
    return Response(status_code=204)


def _stop(rq_id: str) -> None:
    connection = store.connection()
    try:
        job = Job.fetch(rq_id, connection=connection)
    except NoSuchJobError:
        return
    if job.is_started:
        with suppress(InvalidJobOperation):  # it ended just now
            send_stop_job_command(connection, rq_id)
    else:
        job.delete()  # waiting in the queue or for a re-run


app.include_router(jobs)


@app.get("/health")
def health() -> JSONResponse:
    connection = store.connection()
    try:
        connection.ping()
    except redis.RedisError:
        return JSONResponse({"status": "unavailable", "version": VERSION, "redis": "unavailable"}, status_code=503)
    return JSONResponse({"status": "ok", "version": VERSION, "redis": "ok", "queued": store.queued_count(),
                         "running": store.running_count(), "workers": Worker.count(connection=connection)})


@app.get("/formats")
def formats() -> dict:
    return formats_by_kind()


@app.get("/libraries", dependencies=[Depends(_authorized)])
def libraries() -> dict:
    """The symbol libraries the workers have installed, with their symbols, and the ones used by default."""
    catalogue = store.libraries()
    if catalogue is None:
        raise _busy("No worker has started yet")
    return catalogue
