# Job API

cad-ingest is driven by a small JSON API. A client submits a job that names the file and where to fetch it, the worker
downloads it, converts it, uploads the preview or result to presigned URLs, keeps the markdown in Redis and calls the
client back. Interactive documentation is served at `/docs` (OpenAPI).

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
| `file_name` | its extension decides the format (see Formats) |
| `source_url` | presigned `GET` of the original |
| `operations` | `extract`, `preview` and/or `vectorize`, at least one. `vectorize` is required for photos and scans and allowed only for them |
| `preview_url` | presigned `PUT` target of the preview, required with `preview`. The converter sends `Content-Type` `application/pdf` (2D, photos and scans) or `model/gltf-binary` (3D) |
| `result_url` | presigned `PUT` target of the ZIP of `vectorize` (`Content-Type` `application/zip`), required with `vectorize` |
| `options` | optional, for `vectorize`: `paper` the drawing is printed on (`A0` ... `A4`, default `A3`), `scale` denominator (e.g. `50` for 1:50; read from the title block if not set), `dpi` of a scan without visible paper edges, `drawing`: `building` (floor plan, section), `pid` (P&ID), `electrical` (circuit diagram), `site` (site plan) or `auto` (default, recognised from the texts by the built-in [profiles](profiles.md)), `libraries`: ids of installed symbol libraries to recognise symbols from (see `GET /libraries`; at most 20, earlier ones win on equal block names; default: `DEFAULT_SYMBOL_LIBRARIES`); each is used on the kind of drawing it is for (`drawing`: P&IDs, floor plans), `pages`: the pages of a PDF or TIFF to convert, counted from 1 (default: all, at most `MAX_SCAN_PAGES`), `on` and `off`: steps of the recognition to switch on or off by name, at most 20 in each (listed below the table; `400` for an unknown name) |
| `library_urls` | optional, only with `vectorize`: presigned `GET`s of symbol libraries the job brings along, a library ZIP or a DXF (at most 10, each up to `MAX_LIBRARY_SIZE_BYTES`). They come before the installed `libraries`; the default libraries are only used when the job names none at all. See [Symbol libraries](symbol-libraries.md) |
| `callback_url` | optional, see Callback |
| `priority` | `interactive` (default) or `bulk` |
| `reviewer` | optional, only with `vectorize`: a vision model behind an OpenAI-compatible endpoint (`base_url` up to `/v1`, `api_key`, `model`, optionally further `headers` such as the customer a LiteLLM proxy books the calls to), such as a LiteLLM proxy, that is asked about the texts and symbols to check. It sees only a crop around each of them (at most 40 per page); with the step `review_handwriting` also the image of each handwritten note (at most 20 per page, up to 1600 px on its longer side), which can show a large part of the sheet. The key is kept only until the job ends; it is never returned or logged. See `review` below |

The steps of the recognition that `options.on` and `options.off` take (`vectorize` only; see [Steps that can be
switched](photos-and-scans.md#steps-that-can-be-switched) for what each does and why a step is off by default):

- on by default: `paper_edge`, `colour_guard`, `letter_rows`, `freehand`, `pencil_guard`, `merge_where_drawn`,
  `linetypes`, `oblique_text`, `text_direction`, `lookalike_tokens`, `hatching`, `text_on_pipes`, `vocabulary`,
  `dimensions`, `symbol_context`, `attributes_inside`, `dedupe`, `splines`, `square_lines`, `title_block`,
  `unclassified` and `width_calibration`;
- off by default: `freehand_strokes`, `leaders`, `symbols_first`, `framed_letters`, `short_strokes`, `sheet_sizes`,
  `pens`, `read_handwriting` and `review_handwriting`.

A name that is unknown, or in both lists, gives `400`. The profile of a kind of drawing switches some steps off on its
kind (`dimensions` on P&IDs and circuit diagrams, see below), unless the job names them in `options.on`.

Answers: `202 {"id","status":"queued"}`; `200` with the existing job for a repeated `external_id`; `400` invalid body
(`{"detail": "..."}`), also for a library in `options.libraries` that the workers do not have; `401`; `415`
unsupported extension, or `vectorize` missing for a photo or given for a CAD file; `503` with `Retry-After: 15` when
Redis is unavailable or `MAX_QUEUED_JOBS` jobs are waiting.

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
{"libraries": [{"id": "building-symbols", "name": "Marks of building drawings and site plans (starter library)",
                "version": "1.0.0", "licence": "Apache-2.0", "drawing": ["building", "site"], "size": "printed",
                "symbols": [{"name": "NORTH_ARROW", "description": "North arrow"}, ...]},
               {"id": "din-10628", "name": "P&ID symbols after DIN EN ISO 10628-2 (starter library)",
                "version": "1.1.0", "licence": "Apache-2.0", "drawing": ["pid"], "size": "printed",
                "symbols": [{"name": "GATE_VALVE", "description": "Gate valve"}, ...]},
               {"id": "floor-plan", "name": "Floor plan fixtures (starter library)", "version": "1.0.0",
                "licence": "Apache-2.0", "drawing": ["building"], "size": "real",
                "symbols": [{"name": "WC", "description": "Toilet"}, ...]}, ...],
 "default": ["din-10628", "floor-plan", "iec-60617", "building-symbols"]}
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
id and version) and, for every placed symbol, the library it came from. The default libraries include
`building-symbols` (north arrow, axis and section marker on floor plans, sections and site plans): a job that names
no libraries now gets these marks of building drawings and site plans as blocks too, with the axis or the section
letter as attributes. A job that names its libraries is unchanged.

What a person should check is listed under `review`, least confident first: texts that could not be read reliably,
symbols that may be the wrong ones and dimensions to check (see below), each with its `kind` (`text`, `symbol` or
`dimension`), `what` was recognised (the text, the symbol's name or the dimension's value), `why` it is doubtful,
where it is (`x`, `y` in sheet millimetres) and its `confidence` (0 to 100). Every symbol has its `confidence` (the
share of its outline found in the scan, in percent) and its `doubt` (`null`, or why to check it); in the DXF a symbol
to check is on the layer `P2C-SYMBOL-CHECK`, and the markdown has a section
`Symbols to check (uncertain recognition)`. The fields, the layer and the section are new; a client that does not know
them can ignore them.

Linework that looks like a symbol but was recognised as none of the libraries' (see [Symbols not
classified](photos-and-scans.md#symbols-not-classified)) stays linework, on the layer `P2C-SYMBOL-UNCLASSIFIED` of the
DXF: a symbol the libraries lack, or one of theirs that was missed. `report.json` lists each such cluster under
`unclassified` with its `box_mm` (x0, y0, x1, y1 on the sheet) and `why` it looks like a symbol (for example
`"4 strokes with 2 closed shapes, recognised as no symbol of the libraries"`; the box gives its size), the markdown has
a section `Symbols not classified (no library symbol recognised)`, and `warnings` says how many there are.
`options.off` with `unclassified` leaves the linework on its usual layers. The step `symbols_first` in `options.on`
matches symbols before the texts are taken as read (see [Symbols before text](photos-and-scans.md#symbols-before-text)),
and `short_strokes` in `options.on` keeps short strokes across a line, such as dimension ticks. The field, the layer,
the section, the warning and the three step names are new; a client that does not know them can ignore them.

A job with a `reviewer` asks that model about the elements to check, least confident first. For a text it sends a crop
of the text and the reading and asks for the text as printed; the answer is taken if it is one line about as long as the
reading (half to twice), and the text is then no longer uncertain. For a symbol it sends a crop of the symbol and the
library's drawing of it and asks yes or no; a yes confirms the symbol, a no stays to check with that as its `doubt`.
Everything else stays as it was. `report.json` then has `reviewer` with the `model`, how many elements were `asked`, and
how many were `confirmed`, `corrected`, `kept` or left `unanswered`; with the step `review_handwriting`, `asked`,
`kept` and `unanswered` count the notes too, and `read` the notes it read. A model that cannot be reached gives a
warning naming its host, and the job still succeeds.

A large drawing printed small, such as an A1 P&ID on A3, is recognised at its drawn size (see [Print
size](photos-and-scans.md#how-it-works)): `report.json` has `printed`, the share of its drawn size the sheet shows it at
(1 for a drawing printed at its size, 0.5 for A1 on A3), and the DXF is at the drawn size, so its texts and symbols have
the sizes they were drawn with. Positions in `report.json` stay in sheet millimetres. The field is new; a client that
does not know it can ignore it.

The title block of a photo or scan is read as a table (see [Output](photos-and-scans.md#output)). Where one was
found, `report.json` has `title_block` with its `box_mm` (x0, y0, x1, y1 in sheet millimetres), the `rows` and `cols`
of its grid and its `cells`, each with its `row` and `col` (from 0, top left), `row_span` and `col_span`, `box_mm`,
`text` (the texts in the cell, in reading order) and `uncertain` (a text in it is to be checked). The DXF has the
lines of the title block on the layer `P2C-TITLEBLOCK` instead of the layer of their kind (`P2C-LINE`, and on floor
plans and P&IDs often `P2C-WALL` or `P2C-PIPE`), and the markdown has a section `Title block` with the table. The
field, the layer and the section are new; a client that does not know them can ignore them. `options.off`
`["title_block"]` leaves the lines where they were.

On a scanner lid about as grey as its paper, the paper of a scan is found by its straight edges (step `paper_edge`).
A `dpi` given in `options` or stored in the file must then agree with the paper found within 1 %; otherwise the
`dpi` is used, as before.

Markups added by hand on a print (pen notes, clouds, strike-outs, highlighter in colour; notes, clouds and freehand
hatching in black or pencil) are kept apart from the drawing (see [Markups](photos-and-scans.md#markups)): the DXF has
them on the layers `P2C-MARKUP`, `P2C-HIGHLIGHT` and `P2C-MARKUP-TEXT`, `report.json` lists them under `markups`
(`kind` `note` or `highlight`, `box_mm`, `color`, the handwriting `reading` or `null`, the printed `texts` and symbol
tags they cover and their `image`), the ZIP holds an image `markup-k.png` of each one, and the markdown has a section
`Markups (added by hand)`. The fields and files are new; a client that does not know them can ignore them.
Handwriting is no longer read by default (`reading` is `null`): what Tesseract read in it was mostly wrong. The step
`read_handwriting` in `options.on` reads it as before. A job with a `reviewer` and the step `review_handwriting` in
`options.on` has the vision model read each note (at most 20 per page) from its image `markup-k.png`: the reading is
the note's `reading`, a text marked to check on `P2C-MARKUP-TEXT`, and the markdown line says "read by the reviewer
model, to check". An answer that starts with `UNREADABLE` or is not one line of at most 200 characters is no reading.
Each markup has `read_by`: `tesseract`, `reviewer` or `null` (not read), and `reviewer` in `report.json` counts the
notes `read`. The fields are new; a client that does not know them can ignore them.

Texts are read at any angle, not only horizontally and vertically: a street name or a parcel number turned to its
street or parcel on a site plan is read along it. Its `angle` in `report.json` (degrees, counter-clockwise; 0 reads
from left to right, 90 from bottom to top) can then be any value from -180 to 180, such as `-20.5`, or `-90` for a text
that reads from top to bottom, and the DXF text has that rotation. Before, it was 0 or 90 only. The steps
`oblique_text`, `text_direction` and `framed_letters` can be named in `options.on` and `options.off` (see [Steps that
can be switched](photos-and-scans.md#steps-that-can-be-switched)).

The steps `symbol_context` (on by default) and `sheet_sizes` (off by default) can be named in `options.on` and
`options.off` too: `symbol_context` places no orifice plate where a line runs on through it (the two bars of a flange)
and no flow arrow on strokes that are a hairline or part of a larger figure (the marks of a signal line, one triangle
of a valve); `sheet_sizes` holds each symbol against the size of its own kind on the sheet, for sheets that draw their
valves much smaller than their bubbles. It is off by default, as on a dense sheet with many small marks false arrows at
their own small size confirm each other's size. The `symbols` of `report.json` keep their shape; there are fewer wrong
ones.

Dimensions of floor plans and sections (a dimension line with its terminators, extension lines and value, see
[Dimensions](photos-and-scans.md#dimensions)) are linear DXF dimensions on the layer `P2C-DIMENSION`, with the value as
read as their text; those to check (a value read unreliably, missing, not fitting the length at the drawing's scale or
read with another decimal separator than the other numbers of the sheet) are on `P2C-DIMENSION-CHECK`, and a warning of
the job says how many there are. `report.json` lists them under `dimensions` (`value`, `ends_mm`, `measured_mm`,
`offset_mm`, `length_mm`, `angle`, `terminator`, `uncertain`, `doubt`), `counts` has `dimensions`, and those to check
are in `review` with the `kind` `dimension`. Their values are still listed in `texts`, counted in `counts.texts` and in
the markdown's texts; the markdown has the sections `Dimensions` and `Dimensions to check (uncertain recognition)` as
well. A value read unreliably is to check with its dimension: it is in `review` with the `kind` `dimension`, not
`text`, and counted in the job's warning about dimensions, not in the one about texts. The fields, layers and sections
are new; a client that does not know them can ignore them. With the step `dimensions` in `options.off`, they stay
lines and texts as before.

With the step `dimension_fit` (on by default), `measured_mm` are the points on the objects the dimension measures (a
wall face, a corner, the jamb of an opening) where its extension lines point at one within reach, and `offset_mm` is
how far from them its extension lines start (0 where they start at them; DIMEXO in the DXF). The objects are moved so
that each dimension measures its value exactly at the drawing's scale (`length_mm` is then the value at the scale);
one whose value does not fit, or does not add up with the others along it, moves nothing and is to check, the second
with the `doubt` "its value does not add up with the other dimensions along it". `offset_mm` is new; with the step in
`options.off`, `measured_mm` are where the extension lines start and `offset_mm` is 0, as before.

A profile can switch steps off for its kind of drawing (see [Profiles](profiles.md)): on P&IDs and circuit diagrams the
step `dimensions` is off, so their line numbers and values stay texts and lines. A job that names a step in
`options.on` keeps it on. `report.json` has `steps_off_by_kind`, the steps the kind switched off (`[]` where none),
and the markdown has a line `- Steps switched off for this kind of drawing: dimensions (options.on switches them on)`.
The step `pencil_guard` (on; `options.off` keeps the former behaviour) keeps print from being taken for pencil marks:
fewer false notes in `markups`. The field, the line and the step are new; a client that does not know them can
ignore them.

Hatching that the text recognition read as text ("IIII", "////"), such as the hatching of a cut slab or wall in a
section, is traced as lines (see [Text](photos-and-scans.md#how-it-works)): it is no longer a text to check on
`P2C-TEXT-CHECK` nor listed under `review`, and its lines join the outline they end on. The step `hatching` in
`options.off` keeps the former behaviour. Nearly parallel lines are joined where they are drawn, so dense hatching is
no longer moved across the line it ends on, and a long line keeps its angle beside a short piece of the same
direction; `merge_where_drawn` in `options.off` keeps the former behaviour.

The bars of a flange across a pipe are no longer read as letters ("H", or "HH" joined to a line label), and a valve read
as letters between two pipes can be placed as its symbol (step `text_on_pipes`); the number of an instrument bubble
is no longer taken from a text that runs out of its ring, nor from one read with the bars of a flange that touch the
ring from outside (step `attributes_inside`). Both steps can be named in `options.off` to keep the former behaviour
(see [Steps that can be switched](photos-and-scans.md#steps-that-can-be-switched)).

Lines that are not continuous are told apart (see [How it works](photos-and-scans.md#how-it-works)): in the DXF of
`vectorize` they have the linetype `DASHED`, `DASHDOT` or `DOT`, each scaled to the pattern measured on it (the
entity's linetype scale), still on the layer `P2C-DASH` (or the layer of their kind, such as `P2C-SIGNAL`). With the
step `linetypes` off they keep the linetype `P2C_DASH`, as before. Free curves traced as short straight pieces, such
as contour lines, are `SPLINE` entities through fit points on `P2C-CURVE`. The pen of each element is measured on the
grey image. The linetypes and splines are new; a client that cannot read splines switches the step `splines` off, and
such curves stay polylines.

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
conversion. Any 2xx (200 and 202 alike) counts as delivered; connect and read timeout are 30 s each. Connection
errors, `5xx`, `408` and `429` are retried with doubling delays (`RETRY_DELAY_SECONDS` x 1, 2, 4, ...: 30 s ... 16 min,
six retries, about 30 minutes in total); any other answer, e.g. `410` for a file the client no longer knows, stops it.
For lost callbacks a client checks its open jobs with `POST /jobs/status` now and then: the result stays readable with
`GET` for `RESULT_TTL_SECONDS`.
