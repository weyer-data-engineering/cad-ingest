"""The functions RQ runs in a work-horse: `run_job` (download, convert, upload the preview and result, store it) and
`send_callback`. They are queued by name (see store.enqueue), so the API process never imports the heavy parsers.

Failures that are the file's fault end the job at once. Transient ones (source, symbol library, preview or result
target unreachable) raise, which makes RQ run the job again with backoff until MAX_ATTEMPTS. Never log a full URL, it
carries a signature."""
import importlib
import logging
import os
import shutil
import tempfile
import time
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

import httpx
from rq import get_current_job
from rq.timeouts import JobTimeoutException

from . import store
from .config import Settings
from .errors import ConversionError, JobError
from .formats import FORMATS, PREVIEW, RESULT_CONTENT_TYPE
from .plan2cad import library

log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)  # its INFO lines carry the full URL, signature included
HTTP_TIMEOUT = httpx.Timeout(30.0, read=120.0)
CALLBACK_TIMEOUT = httpx.Timeout(30.0)  # connect and read; the client may need a moment to store the markdown
CHUNK = 1 << 20
# Further pages of a scan are left out where they would end after this share of the time limit: time for the uploads
PAGES_UNTIL = 0.9


class Requeue(Exception):
    """Not enough free disk right now: run the job again later, without counting an attempt."""


class CallbackFailed(Exception):
    """The callback target is unreachable or temporarily failing; RQ retries."""


def workdir(job_id: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"cad-{job_id}")


def _host(url: str) -> str:
    return urlsplit(url).hostname or "?"


def _transient(status: int) -> bool:
    return status in (408, 429) or status >= 500


def _retries_left() -> int:
    job = get_current_job()
    return (job.retries_left or 0) if job else 0


def _check_disk(extra: int, settings: Settings) -> None:
    if shutil.disk_usage(tempfile.gettempdir()).free - extra < settings.min_free_disk_mb * 2**20:
        raise Requeue


def run_job(job_id: str) -> None:
    """Runs one attempt of a job."""
    doc = store.get(job_id)
    if not doc or doc["status"] not in store.ACTIVE:
        return  # cancelled, expired or finished (a re-run after the result was stored)
    settings = Settings.from_env()
    doc = store.update(job_id, store.ACTIVE, status="running", attempts=doc["attempts"] + 1)
    if not doc:
        return
    try:
        result = _process(doc, settings)
    except Requeue:
        log.warning("Job %s: not enough free disk, trying again in %s s", job_id, settings.retry_delay_seconds)
        store.requeue(doc, settings.retry_delay_seconds)
        return
    except JobTimeoutException:
        error = JobError("timeout", f"The conversion exceeded {settings.conversion_timeout_seconds} seconds")
    except JobError as e:
        error = e
    else:
        if store.finish(job_id, "succeeded", **result):
            log.info("Job %s succeeded (attempt %s)", job_id, doc["attempts"])
        return
    if error.rerun and _retries_left() > 0:
        log.warning("Job %s attempt %s failed, running it again: %s", job_id, doc["attempts"], error.message)
        store.update(job_id, ("running",), status="queued")
        raise error
    log.warning("Job %s failed (%s): %s", job_id, error.code, error.message)
    store.finish(job_id, "failed", error=error.as_dict())


def _process(doc: dict, settings: Settings) -> dict:
    deadline = time.monotonic() + PAGES_UNTIL * settings.conversion_timeout_seconds
    ext = PurePosixPath(doc["file_name"]).suffix.lstrip(".").lower()
    if ext not in FORMATS:
        raise JobError("unsupported", f"Unsupported file extension '{ext}'")
    suffix, content_type = PREVIEW[FORMATS[ext][0]]
    directory = workdir(doc["id"])
    shutil.rmtree(directory, ignore_errors=True)  # left over from an attempt that was killed
    os.makedirs(directory)
    try:
        _check_disk(0, settings)
        path = os.path.join(directory, "input." + ext)
        _download(doc["source_url"], path, settings)
        preview = os.path.join(directory, "preview." + suffix)
        vectorized = os.path.join(directory, "result.zip")
        libraries = _libraries(doc.get("library_urls") or [], directory, settings)  # only with vectorize
        result = _convert(doc["operations"], ext, path, directory, doc["file_name"], preview, settings,
                          doc.get("options") or {}, vectorized, libraries, deadline, doc.get("reviewer"))
        if "vectorize" in doc["operations"]:
            _upload(doc["result_url"], vectorized, RESULT_CONTENT_TYPE, "result")
            result["result_content_type"] = RESULT_CONTENT_TYPE
        if "preview" in doc["operations"]:
            _upload(doc["preview_url"], preview, content_type, "preview")
            result["preview_content_type"] = content_type
        return result
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _http_error(code: str, what: str, host: str, status: int) -> JobError:
    return JobError(code, f"The {what} answered HTTP {status} ({host})", retryable=True, rerun=_transient(status))


def _download(url: str, path: str, settings: Settings, symbol_library: bool = False) -> None:
    host = _host(url)
    if symbol_library:
        limit, what, unavailable = settings.max_library_size_bytes, "symbol library", "library_unavailable"
        large = JobError("invalid_library", f"A symbol library is larger than {limit} bytes")
    else:
        limit, what, unavailable = settings.max_file_size_bytes, "source", "source_unavailable"
        large = JobError("too_large", f"File is larger than {limit} bytes")
    try:
        with httpx.stream("GET", url, timeout=HTTP_TIMEOUT, follow_redirects=True) as response:
            if response.status_code != 200:
                raise _http_error(unavailable, f"{what} download", host, response.status_code)
            length = int(response.headers.get("content-length") or 0)
            if length > limit:
                raise large
            _check_disk(length, settings)
            size = 0
            with open(path, "wb") as out:
                for chunk in response.iter_raw(CHUNK):  # raw: store exactly what the object holds
                    size += len(chunk)
                    if size > limit:
                        raise large
                    out.write(chunk)
    except httpx.HTTPError as e:
        raise JobError(unavailable, f"Could not download the {what} from {host} ({type(e).__name__})",
                       rerun=True) from None


def _libraries(urls: list[str], directory: str, settings: Settings) -> list[library.Library]:
    """Downloads and reads the symbol libraries a job brings along. A library's file name in its URL names it if it
    has no library.toml; nothing of the URL is used as a path."""
    out = []
    for n, url in enumerate(urls, 1):
        path = os.path.join(directory, f"library-{n}")
        _download(url, path, settings, symbol_library=True)
        with open(path, "rb") as f:
            data = f.read()
        try:
            out.append(library.from_bytes(data, PurePosixPath(unquote(urlsplit(url).path)).name or f"library-{n}"))
        except library.LibraryError as e:
            raise JobError("invalid_library", f"Symbol library {n} of library_urls: {e}") from None
    return out


def _convert(operations: list[str], ext: str, path: str, directory: str, name: str, preview: str,
             settings: Settings, options: dict | None = None, vectorized: str | None = None,
             libraries: list | tuple = (), deadline: float | None = None, reviewer: dict | None = None) -> dict:
    result: dict = {"warnings": []}
    try:
        handler = importlib.import_module(f".{FORMATS[ext][1]}", __package__)
        markdown, warnings = "", []
        if "vectorize" in operations:  # photos and scans: one recognition run serves all operations
            markdown, warnings = handler.vectorize(path, name, options or {}, vectorized,
                                                   preview if "preview" in operations else None, libraries, deadline,
                                                   reviewer)
        elif "extract" in operations:
            markdown, warnings = handler.extract(ext, path, directory, name)
        if "extract" in operations:
            if len(markdown) > settings.max_markdown_chars:
                markdown = markdown[:settings.max_markdown_chars] + "\n\n[Truncated: size limit reached]"
                warnings.append(f"Markdown was truncated to {settings.max_markdown_chars} characters")
            result.update(markdown=markdown)
        result["warnings"] += warnings
        if "preview" in operations and "vectorize" not in operations:
            result["warnings"] += handler.preview(ext, path, preview)
    except (JobTimeoutException, JobError):
        raise
    except ConversionError as e:
        raise JobError("unreadable", str(e)) from None
    except MemoryError:
        raise JobError("out_of_memory", "The file needs more memory than allowed") from None
    except Exception as e:  # parsers raise arbitrary exception types for broken input
        log.exception("Conversion failed")
        raise JobError("unreadable", f"The file could not be processed: {type(e).__name__}: {str(e)[:300]}") from None
    return result


def _upload(url: str, path: str, content_type: str, what: str) -> None:
    host = _host(url)

    def chunks():
        with open(path, "rb") as f:
            while chunk := f.read(CHUNK):
                yield chunk

    try:  # explicit Content-Length: S3 rejects chunked uploads
        response = httpx.put(url, content=chunks(), timeout=HTTP_TIMEOUT,
                             headers={"Content-Type": content_type, "Content-Length": str(os.path.getsize(path))})
    except httpx.HTTPError as e:
        raise JobError(f"{what}_upload_failed", f"Could not upload the {what} to {host} ({type(e).__name__})",
                       rerun=True) from None
    if not response.is_success:
        raise _http_error(f"{what}_upload_failed", f"{what} upload", host, response.status_code)


def send_callback(job_id: str) -> None:
    """POSTs the job JSON to the callback URL. Retried by RQ on connection errors, 5xx, 408 and 429; any other answer
    ends it: 2xx (200 and 202 alike) is delivered, another 4xx means the client does not want the result any more."""
    doc = store.get(job_id)
    if not doc or doc["status"] not in ("succeeded", "failed"):
        return  # deleted or cancelled meanwhile: no callback
    host = _host(doc["callback_url"])
    try:
        status = httpx.post(doc["callback_url"], json=store.view(doc), timeout=CALLBACK_TIMEOUT).status_code
        failure = f"HTTP {status}" if _transient(status) else None
    except httpx.HTTPError as e:
        status, failure = None, type(e).__name__
    if failure is None:
        log.info("Job %s: callback to %s answered HTTP %s", job_id, host, status)
        return
    if _retries_left() == 0:
        log.error("Job %s: giving up on the callback to %s (%s)", job_id, host, failure)
        return
    raise CallbackFailed(f"Callback to {host} failed ({failure})")
