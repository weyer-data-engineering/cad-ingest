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
  "priority": "interactive",
  "reviewer": {"base_url": "https://llm.example.com/v1", "api_key": "...", "model": "..."}
}
```

| Field | |
|---|---|
| `external_id` | idempotency key: an existing job with the same key that is not finished (`queued`, `running`) is returned instead of a second one |
| `file_name` | its extension decides the format (see Formats); for `check_library` it is `.zip` or `.dxf` |
| `source_url` | presigned `GET` of the original |
| `operations` | `extract`, `preview` and/or `vectorize`, at least one. `vectorize` is required for photos and scans and allowed only for them. Or `check_library` alone: the file is a symbol library to read and describe (see [Checking a library](#checking-a-library)) |
| `preview_url` | presigned `PUT` target of the preview, required with `preview`. The converter sends `Content-Type` `application/pdf` (2D, photos and scans) or `model/gltf-binary` (3D) |
| `result_url` | presigned `PUT` target of the ZIP of `vectorize` (`Content-Type` `application/zip`), required with `vectorize` |
| `options` | optional, for `vectorize`: `paper` the drawing is printed on (`A0` ... `A4`, default `A3`), `scale` denominator (e.g. `50` for 1:50; read from the title block if not set), `dpi` of a scan without visible paper edges, `drawing`: `building` (floor plan, section), `pid` (P&ID), `electrical` (circuit diagram), `site` (site plan) or `auto` (default, recognised from the texts by the built-in [profiles](profiles.md)), `libraries`: ids of installed symbol libraries to recognise symbols from (see `GET /libraries`; at most 20, earlier ones win on equal block names; default: `DEFAULT_SYMBOL_LIBRARIES`); each is used on the kind of drawing it is for (`drawing`: P&IDs, floor plans), `pages`: the pages of a PDF or TIFF to convert, counted from 1 (default: all, at most `MAX_SCAN_PAGES`), `on` and `off`: steps of the recognition to switch on or off by name (see [Steps that can be switched](photos-and-scans.md#steps-that-can-be-switched); `400` for an unknown name). For `check_library`: `self_test` (default `false`) also tests the library on simulated scans and photos, which takes about half a minute per sheet of up to 48 symbols, and `scale` is then the drawing scale for a library of real-size blocks (default `50`) |
| `library_urls` | optional, only with `vectorize`: presigned `GET`s of symbol libraries the job brings along, a library ZIP or a DXF (at most 10, each up to `MAX_LIBRARY_SIZE_BYTES`). They come before the installed `libraries`; the default libraries are only used when the job names none at all. See [Symbol libraries](symbol-libraries.md) |
| `callback_url` | optional, see Callback |
| `priority` | `interactive` (default) or `bulk` |
| `reviewer` | optional, only with `vectorize`: a vision model behind an OpenAI-compatible endpoint (`base_url` up to `/v1`, `api_key`, `model`, optionally further `headers` such as the customer a LiteLLM proxy books the calls to), such as a LiteLLM proxy, that is asked about the texts and symbols to check. It sees only a crop around each of them (at most 40 per page). The key is kept only until the job ends; it is never returned or logged. See `review` below |

Answers: `202 {"id","status":"queued"}`; `200` with the existing job for a repeated `external_id`; `400` invalid body
(`{"detail": "..."}`), also for a library in `options.libraries` that the workers do not have and for `check_library`
with another operation; `401`; `415` unsupported extension, or `vectorize` missing for a photo or given for a CAD file,
or a file other than a ZIP or DXF for `check_library`; `503` with `Retry-After: 15` when Redis is unavailable or
`MAX_QUEUED_JOBS` jobs are waiting.

`GET /jobs/{id}` -> `200`, `404` unknown or expired:

```json
{
  "id": "...", "external_id": "...", "status": "queued|running|succeeded|failed|cancelled",
  "attempts": 1, "operations": ["extract", "preview"], "kind": "DRAWING_2D|MODEL_3D|SCAN_2D", "format": "step",
  "markdown": "...", "preview_content_type": "model/gltf-binary", "result_content_type": "application/zip",
  "library": {"id": "...", "...": "..."},
  "warnings": [],
  "error": {"code": "...", "message": "...", "retryable": false}
}
```

`markdown` is there when the job succeeded and `extract` was requested (it may be empty); `preview_content_type` when it
succeeded and `preview` was requested (the preview has been uploaded by then); `result_content_type` likewise for
`vectorize`; `library` when a `check_library` job succeeded (see [Checking a library](#checking-a-library)); `error`
when it failed. `kind` is not set for `check_library`, since a library is no drawing. Results are kept
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
| `library_unavailable` | the download of a library in `library_urls`, or of the library to check (`check_library`), failed (same rule as `source_unavailable`) | yes |
| `invalid_library` | a library in `library_urls` or the library to check is not a usable symbol library or larger than `MAX_LIBRARY_SIZE_BYTES`, or a library in `options.libraries` is not installed on the worker; the message says why | no |
| `internal` | the worker was lost on the last attempt | yes |

`retryable` tells the client that submitting again (with fresh URLs) may help.

`DELETE /jobs/{id}` -> `204`, `404` unknown. Queued (or waiting for a re-run): removed from the queue. Running: the
work-horse and its children (e.g. `dwg2dxf`) are killed within about a second (RQ's stop-job command). Finished: the
stored result is deleted. A queued or running job is kept as `cancelled` for `RESULT_TTL_SECONDS` (GET shows it) and
frees its `external_id`. No result is stored and no callback is sent for a cancelled job.

`POST /jobs/status` with `{"ids": ["<job id>", ...]}` (at most 500 ids, `400` above) -> `200 {"jobs": [...], "missing":
["<id>", ...]}`: the same objects as `GET /jobs/{id}` but without `markdown` and `library`, in one Redis round trip, so
a reconciler can check thousands of open jobs with a few requests; `missing` are unknown or expired ids. Fetch the
markdown and the library with `GET /jobs/{id}` for the jobs that succeeded.

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

What a person should check is listed under `review`, least confident first: texts that could not be read reliably
and symbols that may be the wrong ones, each with its `kind` (`text` or `symbol`), `what` was recognised (the text or
the symbol's name), `why` it is doubtful, where it is (`x`, `y` in sheet millimetres) and its `confidence` (0 to 100).
Every symbol has its `confidence` (the share of its outline found in the scan, in percent) and its `doubt` (`null`, or
why to check it); in the DXF a symbol to check is on the layer `P2C-SYMBOL-CHECK`, and the markdown has a section
`Symbols to check (uncertain recognition)`. The fields, the layer and the section are new; a client that does not know
them can ignore them.

A job with a `reviewer` asks that model about the elements to check, least confident first. For a text it sends a crop
of the text and the reading and asks for the text as printed; the answer is taken if it is one line about as long as the
reading (half to twice), and the text is then no longer uncertain. For a symbol it sends a crop of the symbol and the
library's drawing of it and asks yes or no; a yes confirms the symbol, a no stays to check with that as its `doubt`.
Everything else stays as it was. `report.json` then has `reviewer` with the `model`, how many elements were `asked`, and
how many were `confirmed`, `corrected`, `kept` or left `unanswered`; a model that cannot be reached gives a warning
naming its host, and the job still succeeds.

A large drawing printed small, such as an A1 P&ID on A3, is recognised at its drawn size (see [Print
size](photos-and-scans.md#how-it-works)): `report.json` has `printed`, the share of its drawn size the sheet shows it at
(1 for a drawing printed at its size, 0.5 for A1 on A3), and the DXF is at the drawn size, so its texts and symbols have
the sizes they were drawn with. Positions in `report.json` stay in sheet millimetres. The field is new; a client that
does not know it can ignore it.

Markups added by hand on a print (pen notes, clouds, strike-outs, highlighter in colour; notes, clouds and freehand
hatching in black or pencil) are kept apart from the drawing (see [Markups](photos-and-scans.md#markups)): the DXF has
them on the layers `P2C-MARKUP`, `P2C-HIGHLIGHT` and `P2C-MARKUP-TEXT`, `report.json` lists them under `markups`
(`kind` `note` or `highlight`, `box_mm`, `color`, the handwriting `reading` or `null`, the printed `texts` and symbol
tags they cover and their `image`), the ZIP holds an image `markup-k.png` of each one, and the markdown has a section
`Markups (added by hand)`. The fields and files are new; a client that does not know them can ignore them.
Handwriting is no longer read by default (`reading` is `null`): what Tesseract read in it was mostly wrong. The step
`read_handwriting` in `options.on` reads it as before.

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

## Checking a library

A platform whose users bring their own symbol libraries can have each one checked before it uses it in
`library_urls`. The job names the library, a ZIP or a DXF (see [Symbol libraries](symbol-libraries.md)), like a file:

```bash
curl -X POST http://localhost:8080/jobs -H 'Content-Type: application/json' -d '{"external_id":"library-7:3",
  "file_name":"acme-pid.zip","source_url":"https://storage/acme-pid.zip?sig=...","operations":["check_library"],
  "options":{"self_test":false},"callback_url":"https://app/cad-callback"}'
```

The worker downloads the library (up to `MAX_LIBRARY_SIZE_BYTES`) and reads it in its work-horse, under the time and
memory limits of a conversion, as it reads a library in `library_urls`; nothing in it is run, and the file is not
kept. The job then
has `library`: what `GET /libraries` lists for an installed library, the notes of `cad-ingest library check` on what
was ignored and what may keep a symbol from being found (`warnings`), and the results of the self-test (`self_test`,
`null` unless `options.self_test` asks for it):

```json
{"id": "...", "external_id": "library-7:3", "status": "succeeded", "attempts": 1, "operations": ["check_library"],
 "format": "zip",
 "library": {"id": "acme-pid", "name": "ACME P&ID symbols", "version": "2.1.0", "licence": "LicenseRef-ACME-internal",
             "drawing": ["pid"], "size": "printed",
             "symbols": [{"name": "MY_VALVE", "description": "Box valve"},
                         {"name": "VALVE_AT_PORT", "description": "Gate valve, inserted at a port"}],
             "warnings": ["VALVE_AT_PORT: the insertion point lies on a port; put it in the middle of the symbol, because a port is recognised by its direction from there"],
             "self_test": null},
 "warnings": []}
```

A bare DXF, or a ZIP with a `library.toml` at its top that has no `id`, is named after its file: here after
`file_name`, in `library_urls` after the last part of the URL's path. Give it the same file name in both, so that it
keeps its id.

The self-test draws every symbol twice, as in the block and turned and mirrored, on A3 sheets, and reads a simulated
200 dpi scan (`scan`) and a simulated 12 MP phone photo (`photo`) of each sheet with this library alone (see [Checking
a library](symbol-libraries.md#checking-a-library)). It takes about half a minute per sheet of up to 48 symbols; a
check without it takes a second or two. Sheets it would not finish within the time limit are left out, and the job's
`warnings` say how many:

```json
"self_test": {"sheets": 1, "scale": null, "drawn": 4, "found": {"scan": 4, "photo": 3},
              "extra": {"scan": 0, "photo": 0}, "seconds_per_sheet": {"scan": 9.5, "photo": 18.7},
              "symbol_seconds_per_sheet": {"scan": 0.4, "photo": 0.5},
              "symbols": [{"name": "MY_VALVE", "drawn": 2, "found": {"scan": 2, "photo": 2}, "mistaken_for": {}},
                          {"name": "VALVE_AT_PORT", "drawn": 2, "found": {"scan": 2, "photo": 1},
                           "mistaken_for": {"MY_VALVE": 1}}]}
```

| Field of `self_test` | |
|---|---|
| `sheets` | the A3 sheets drawn and read |
| `scale` | the drawing scale of the sheets for a library of real-size blocks (`options.scale`, default 50), else `null` |
| `drawn` | the symbols drawn on them |
| `found` | how many of them were found as themselves, in the scan and in the photo |
| `extra` | symbols found where none was drawn |
| `seconds_per_sheet`, `symbol_seconds_per_sheet` | the seconds to read one sheet, and of those the seconds of the symbol recognition |
| `symbols` | per symbol: how often it was `drawn` and `found`, and which symbols it was `mistaken_for` how often |

A library that cannot be used fails the job with `invalid_library` and the reason as the `message`, as a library in
`library_urls` does; one that cannot be downloaded with `library_unavailable`:

```json
{"id": "...", "external_id": "library-7:4", "status": "failed", "attempts": 1, "operations": ["check_library"],
 "format": "zip", "warnings": [],
 "error": {"code": "invalid_library", "message": "acme-pid.zip: library.toml: 'size' must be one of printed, real: 'paper'", "retryable": false}}
```

## Callback

When a job ends (`succeeded` or `failed`) the converter `POST`s the same JSON as `GET /jobs/{id}` to `callback_url`.
It is sent by a short follow-up job on the `cad-callbacks` queue, outside the memory-limited work-horse of the
conversion. Any 2xx (200 and 202 alike) counts as delivered; connect and read timeout are 30 s each. Connection errors, `5xx`, `408` and `429` are retried with doubling delays
(`RETRY_DELAY_SECONDS` x 1, 2, 4, ...: 30 s ... 16 min, six retries, about 30 minutes in total); any other answer, e.g.
`410` for a file the client no longer knows, stops it. For lost callbacks a client checks its open jobs with
`POST /jobs/status` now and then: the result stays readable with `GET` for `RESULT_TTL_SECONDS`.
