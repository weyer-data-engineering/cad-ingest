# Job API

cad-ingest is driven by a small JSON API. A client submits a job that names the file and where to fetch it, the worker downloads it, converts it, uploads the preview or result to presigned URLs, keeps the markdown in Redis and calls the client back. Interactive documentation is served at `/docs` (OpenAPI).

## Endpoints

JSON. If `API_TOKEN` is set, every `/jobs` call needs `Authorization: Bearer <token>` (`401` otherwise).

`POST /jobs`

```json
{
  "external_id": "drawing-42:1",
  "file_name": "part.step",
  "source_url": "https://...",
  "operations": ["extract", "preview"],
  "preview_url": "https://...",
  "result_url": "https://...",
  "options": {"paper": "A3", "scale": 50, "dpi": 300, "drawing": "auto", "libraries": ["din-10628"]},
  "library_urls": ["https://..."],
  "callback_url": "https://...",
  "priority": "interactive"
}
```

| Field | |
|---|---|
| `external_id` | idempotency key: an existing job with the same key that is not finished (`queued`, `running`) is returned instead of a second one |
| `file_name` | its extension decides the format (see Formats) |
| `source_url` | presigned `GET` of the original |
| `operations` | `extract`, `preview` and/or `vectorize`, at least one. `vectorize` is required for photos and scans and allowed only for them |
| `preview_url` | presigned `PUT` target of the preview, required with `preview`. The converter sends `Content-Type` `application/pdf` (2D, photos and scans) or `model/gltf-binary` (3D) |
| `result_url` | presigned `PUT` target of the ZIP of `vectorize` (`Content-Type` `application/zip`), required with `vectorize` |
| `options` | optional, for `vectorize`: `paper` the drawing is printed on (`A0` ... `A4`, default `A3`), `scale` denominator (e.g. `50` for 1:50; read from the title block if not set), `dpi` of a scan without visible paper edges, `drawing`: `building` (floor plan), `pid` (P&ID) or `auto` (default, recognised from the texts), `libraries`: ids of installed symbol libraries to recognise symbols from (see `GET /libraries`; at most 20, earlier ones win on equal block names; default: `DEFAULT_SYMBOL_LIBRARIES`); each is used on the kind of drawing it is for (`drawing`: P&IDs, floor plans), `pages`: the pages of a PDF or TIFF to convert, counted from 1 (default: all, at most `MAX_SCAN_PAGES`) |
| `library_urls` | optional, only with `vectorize`: presigned `GET`s of symbol libraries the job brings along, a library ZIP or a DXF (at most 10, each up to `MAX_LIBRARY_SIZE_BYTES`). They come before the installed `libraries`; the default libraries are only used when the job names none at all. See [Symbol libraries](symbol-libraries.md) |
| `callback_url` | optional, see Callback |
| `priority` | `interactive` (default) or `bulk` |

Answers: `202 {"id","status":"queued"}`; `200` with the existing job for a repeated `external_id`; `400` invalid body
(`{"detail": "..."}`), also for a library in `options.libraries` that the workers do not have; `401`; `415` unsupported extension, or `vectorize` missing for a photo or given for a CAD file; `503` with `Retry-After: 15` when Redis is unavailable or
`MAX_QUEUED_JOBS` jobs are waiting.

`GET /jobs/{id}` -> `200`, `404` unknown or expired:

```json
{
  "id": "...", "external_id": "...", "status": "queued|running|succeeded|failed|cancelled",
  "attempts": 1, "operations": ["extract", "preview"], "kind": "DRAWING_2D|MODEL_3D|SCAN_2D", "format": "step",
  "markdown": "...", "preview_content_type": "model/gltf-binary", "result_content_type": "application/zip",
  "warnings": [],
  "error": {"code": "...", "message": "...", "retryable": false}
}
```

`markdown` is there when the job succeeded and `extract` was requested (it may be empty); `preview_content_type` when it
succeeded and `preview` was requested (the preview has been uploaded by then); `result_content_type` likewise for
`vectorize`; `error` when it failed. Results are kept
for `RESULT_TTL_SECONDS`. `attempts` counts the runs (re-runs after infrastructure failures included).

| `error.code` | Meaning | `retryable` |
|---|---|---|
| `unsupported` | extension not (or no longer) supported | no |
| `unreadable` | corrupt file, no geometry, parser error or crash | no |
| `too_large` | larger than `MAX_FILE_SIZE_BYTES` (from `Content-Length` or while streaming) | no |
| `timeout` | ran longer than `CONVERSION_TIMEOUT_SECONDS` | no |
| `out_of_memory` | needs more than `MEMORY_LIMIT_MB`, or the kernel killed the work-horse | no |
| `source_unavailable` | the download failed (after `MAX_ATTEMPTS` runs, or at once for a 4xx such as an expired URL) | yes |
| `preview_upload_failed` | the `PUT` failed (same rule) | yes |
| `result_upload_failed` | the `PUT` of the `vectorize` result failed (same rule) | yes |
| `library_unavailable` | the download of a library in `library_urls` failed (same rule as `source_unavailable`) | yes |
| `invalid_library` | a library in `library_urls` is not a usable symbol library or larger than `MAX_LIBRARY_SIZE_BYTES`, or a library in `options.libraries` is not installed on the worker; the message says why | no |
| `internal` | the worker was lost on the last attempt | yes |

`retryable` tells the client that submitting again (with fresh URLs) may help.

`DELETE /jobs/{id}` -> `204`, `404` unknown. Queued (or waiting for a re-run): removed from the queue. Running: the
work-horse and its children (e.g. `dwg2dxf`) are killed within about a second (RQ's stop-job command). Finished: the
stored result is deleted. A queued or running job is kept as `cancelled` for `RESULT_TTL_SECONDS` (GET shows it) and
frees its `external_id`. No result is stored and no callback is sent for a cancelled job.

`POST /jobs/status` with `{"ids": ["<job id>", ...]}` (at most 500 ids, `400` above) -> `200 {"jobs": [...], "missing":
["<id>", ...]}`: the same objects as `GET /jobs/{id}` but without `markdown`, in one Redis round trip, so a reconciler
can check thousands of open jobs with a few requests; `missing` are unknown or expired ids. Fetch the markdown with
`GET /jobs/{id}` for the jobs that succeeded.

`GET /health` -> `{"status":"ok","version":"...","redis":"ok","queued":{"interactive":n,"bulk":n},"running":n,"workers":n}`
(`running` counts running jobs, `workers` the registered RQ workers); `503` when Redis is unreachable.
`GET /formats` -> `{"DRAWING_2D":[...],"MODEL_3D":[...],"SCAN_2D":[...]}` (`SCAN_2D`: photos and scans, `vectorize` only).

`GET /libraries` (needs the API token like `/jobs`) -> the symbol libraries the workers have installed and the default
ones; `503` before any worker has started:

```json
{"libraries": [{"id": "din-10628", "name": "P&ID symbols after DIN EN ISO 10628-2 (starter library)",
                "version": "1.0.0", "licence": "Apache-2.0", "drawing": ["pid"], "size": "printed",
                "symbols": [{"name": "GATE_VALVE", "description": "Gate valve"}, ...]},
               {"id": "floor-plan", "name": "Floor plan fixtures (starter library)", "version": "1.0.0",
                "licence": "Apache-2.0", "drawing": ["building"], "size": "real",
                "symbols": [{"name": "WC", "description": "Toilet"}, ...]}],
 "default": ["din-10628", "floor-plan"]}
```

```bash
curl -X POST http://localhost:8080/jobs -H 'Content-Type: application/json' -d '{"external_id":"part-1",
  "file_name":"part.step","source_url":"https://storage/part.step?sig=...","operations":["extract"]}'
curl http://localhost:8080/jobs/<id>
curl -X POST http://localhost:8080/jobs -H 'Content-Type: application/json' -d '{"external_id":"plan-1",
  "file_name":"plan.heic","source_url":"https://storage/plan.heic?sig=...","operations":["vectorize","extract"],
  "result_url":"https://storage/plan.zip?sig=...","options":{"paper":"A3"}}'
curl -X POST http://localhost:8080/jobs -H 'Content-Type: application/json' -d '{"external_id":"pid-7",
  "file_name":"pid.jpg","source_url":"https://storage/pid.jpg?sig=...","operations":["vectorize"],
  "result_url":"https://storage/pid.zip?sig=...","library_urls":["https://storage/acme-pid.zip?sig=..."],
  "options":{"paper":"A1","libraries":["din-10628"]}}'
```

The `report.json` in the ZIP of `vectorize` lists the libraries that were used for the kind of drawing (`libraries`:
id and version) and, for every placed symbol, the library it came from.

PDF and TIFF scans may have several pages, and each page with a drawing is converted. The first one goes to the root
of the ZIP (`plan.dxf`, `plan.pdf`, `overlay.png`, `underlay.png`, `report.json`), every further page n to the folder
`page-n/` with the same files. Each `report.json` has its `page` and the `pages` converted (counted from 1). The
preview holds one page per drawing, and the markdown has a section `## Page n of m` for each one. A job converts at
most `MAX_SCAN_PAGES` pages and leaves out the pages it would not finish within `CONVERSION_TIMEOUT_SECONDS`; pages
without a drawing, pages left out and pages asked for that the file does not have are named in `warnings`. A single
image gives the same ZIP and markdown as before.

A PDF page drawn in vectors, such as a CAD export, is read as it is instead of being recognised on an image (see
[Vector PDFs](photos-and-scans.md#vector-pdfs)): the DXF keeps the exact geometry, colours, pen widths and CAD layers,
and the markdown lists the layers. `paper` and `dpi` do not apply, and no symbols are placed (`libraries` is empty).
Its `report.json` has `sheet` `page`, the page size in `sheet_size_mm` and `paper` (`null` where the page is no
standard size).

## Callback

When a job ends (`succeeded` or `failed`) the converter `POST`s the same JSON as `GET /jobs/{id}` to `callback_url`.
It is sent by a short follow-up job on the `cad-callbacks` queue, outside the memory-limited work-horse of the
conversion. Any 2xx (200 and 202 alike) counts as delivered; connect and read timeout are 30 s each. Connection errors, `5xx`, `408` and `429` are retried with doubling delays
(`RETRY_DELAY_SECONDS` x 1, 2, 4, ...: 30 s ... 16 min, six retries, about 30 minutes in total); any other answer, e.g.
`410` for a file the client no longer knows, stops it. For lost callbacks a client checks its open jobs with
`POST /jobs/status` now and then: the result stays readable with `GET` for `RESULT_TTL_SECONDS`.
