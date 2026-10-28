# Changelog

All notable changes to cad-ingest are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/): the job API stays compatible within a major version.

## [Unreleased]

### Added

- Symbol libraries as add-ons (see `docs/symbol-libraries.md`). A library is a folder with a `library.toml` and a
  DXF, the same folder as a ZIP, or a bare DXF; units come from `$INSUNITS`, blocks are read around their base point,
  and splines, ellipses and nested blocks count towards a symbol's outline. The result DXF holds the library's
  original blocks.
  - Workers read the built-in starter library and those in `SYMBOL_LIBRARIES` when they start and stop on a broken
    one; `DEFAULT_SYMBOL_LIBRARIES` sets the libraries a job gets when it names none.
  - Job API: `options.libraries` chooses installed libraries (`400` for unknown ones), `library_urls` brings libraries
    along with a job, `GET /libraries` lists the installed ones. New error codes `invalid_library` and
    `library_unavailable`; new setting `MAX_LIBRARY_SIZE_BYTES`.
  - `report.json` lists the libraries used and, for every symbol, its library.
  - Command line: `--library` (an installed id or a path, repeatable), `cad-ingest library check` (format, notes and
    a self-test on simulated scans and photos) and `cad-ingest library build` (from DWG and DXF files).
- Symbols on floor plans: fixtures such as toilets, washbasins, bathtubs, shower trays, kitchen sinks, hobs and
  washing machines become blocks too, with their back on the wall.
  - Libraries say which drawings they are for (`drawing = ["building"]`) and whether their blocks are drawn at real
    size (`size = "real"`, like architectural CAD blocks); real-size blocks are shrunk by the drawing scale, and 1:50
    and 1:100 are tried while the scale is unknown. The DXF holds them at real size.
  - A second built-in library, `floor-plan`, with seven fixtures; `DEFAULT_SYMBOL_LIBRARIES` is now
    `din-10628,floor-plan`, and every library is used on the kind of drawing it is for.
  - Fixtures are found from their rectangular outline, also where a side is the face of the wall or the edge of a
    counter, and from two opposite sides where OCR read a small fixture at 1:100 as letters ("IOl"). A symbol never
    takes a piece out of a wall.
  - The benchmark has furnished floor plans at 1:50 and 1:100 (scan and phone photo), and `cad-ingest library check
    --scale` tests real-size libraries at a drawing scale.
- Every page of a PDF or multi-page TIFF scan is converted, not only the first one. The first page with a drawing stays
  at the root of the ZIP, every further page n goes to `page-n/` with the same files; each `report.json` names its
  `page` and all `pages` converted, the preview holds every page and the markdown has a section per page. Pages
  without a drawing are skipped with a warning.
  - Job API: `options.pages` chooses pages; new setting `MAX_SCAN_PAGES` (default 20). A job leaves out the pages
    it would not finish within `CONVERSION_TIMEOUT_SECONDS` and says so in a warning, rather than failing.
  - Command line: `--pages 1-3,5`.
- Vector PDFs, such as CAD exports, are read as they are instead of being recognised on an image: lines, arcs,
  circles, fills and texts keep their exact geometry, colour, pen width and CAD layer (from the PDF's optional
  content), and texts their string, angle, slant, height and length. Nothing is classified or replaced by symbols,
  so the DXF shows what the PDF shows. A PDF page that is a scan still takes the image route.
  - `report.json`: `sheet` is `page` for such a page, and `paper` is `null` where the page is no standard size.
  - The markdown lists the CAD layers with the number of elements on each.
- Markups on scanned prints: pen strokes and highlighter in colour on a black and white print are kept apart from the
  drawing instead of turning into stray lines and symbols and garbling the print under them (see
  `docs/photos-and-scans.md#markups`).
  - The DXF has pen strokes as polylines in their colour on `P2C-MARKUP`, highlighter as transparent fills on
    `P2C-HIGHLIGHT` and what Tesseract reads of the handwriting, as a guess, on `P2C-MARKUP-TEXT`.
  - `report.json` lists them under `markups` (kind, box, colour, reading and the printed texts and symbol tags they cover), the ZIP
    holds an image `markup-k.png` of each one, and the markdown has a section on them.
- Text spotting on photos and scans: a learned text detector and recogniser (PP-OCRv6 small, Apache-2.0 models run by
  RapidOCR on ONNX Runtime, on the CPU) reads the sheet after the character grouping and replaces the texts whose
  letters lie in the lines it finds. It finds letters that touch lines and bubble rings, and reads plotter fonts, the
  slashed zero and touching letters much better than Tesseract. New text needs a score of 0.8 (0.9 for one or two
  characters, which must also be no taller than the usual text), less sure replacements are marked for review, and
  characters both readers confuse go to the vote of their look-alikes. Lines cut by a tile edge are read again whole,
  and wide spaces the recogniser dropped are put back. The image fetches and checks the models when it is built. New
  dependencies: `rapidocr`, `onnxruntime`.
- Pipes and symbols on photos and scans are connected more reliably: a symbol beside the pipe that runs through two of
  its ports (a flow arrow) is moved onto it; flow arrows are tried where strokes stand beside a straight line and need
  their pipe to go on past both ends; strokes left wholly inside a symbol with ports are dropped, so they no longer
  join the pipes on either side; and a gate valve with a ball, filled or filled in by blur, is a ball valve.
- What to check on photos and scans: symbols that may be the wrong ones (less than 90 % of their outline found, or a
  symbol of another kind fits almost as well) go to the new layer `P2C-SYMBOL-CHECK` and appear in magenta in
  `overlay.png`; `report.json` has the new list `review` of the texts and symbols to check, least confident first, and
  each symbol its `confidence` and `doubt`; the markdown has a section `Symbols to check`. The benchmark counts the
  symbols marked and the wrong ones left unmarked.
- Reviewer model for `vectorize`: a job may name a vision model behind an OpenAI-compatible endpoint (`reviewer`:
  `base_url`, `api_key`, `model`, optional `headers`), such as a LiteLLM proxy. It is asked about the texts and symbols
  to check, sees only a crop around each, and its answers are taken where they fit; `report.json` counts them under
  `reviewer`. The key and headers are dropped when the job ends and never returned or logged.
- Sheet grammar for texts on photos and scans: inch marks read as foot marks or degree signs after pipe sizes, and the
  equals sign of key=value labels read as a colon, are put right where the sheet shows enough of these notations.
- Benchmark: exact texts by kind (codes such as tags and line numbers, numbers, words), errors left unmarked (texts
  read wrong without a review mark in their place), and connections between symbols (pairs joined by linework, found
  and correct).
- Large drawings printed small (an A1 P&ID on A3 or A4) are recognised at their drawn size, since the tolerances of
  recognition are made for drawings at their drawn size; the share they are printed at is measured on the height of
  the character-sized shapes of ink. On an image with fewer than 5 pixels per drawn millimetre (a phone photo of a
  small print) lines and texts are still found on the sheet. `report.json` has the new field `printed`. The benchmark
  has two such cases (A1 on A3, scanned; A1 on A4, photographed).
- Connected linework on photos and scans: once the symbols are placed, line ends that stop up to 2 mm short of other
  linework are joined to it (a branch to its pipe, a signal line to its bubble, a pipe to its valve or vessel), two
  loose ends whose lines meet nearby form a corner, a symbol port that faces a pipe or the next symbol's port within
  6 mm is piped to it, pieces that then meet end to end become one polyline, and loose pieces under 2 mm are dropped.
  The benchmark counts the ends that stop short and the loose crumbs.
- Fewer wrong symbols: a symbol needs 85 % of its outline instead of 80 % (on a real scan most symbols with less were
  wrong; their strokes stay as lines); a symbol whose strokes end freely in the library, such as the barbs of a flow
  arrow, is not placed where those strokes run on (the edges of a line number's frame, the cross in a box, letters);
  letters inside a symbol that holds none, such as a reducer, count against it; and what is left of letters that could
  not be read in an instrument bubble is dropped, the bubble marked to check ("its text could not be read").

### Changed

- The DXF of a drawing printed small is at its drawn size, so its texts and symbols have the sizes they were drawn
  with; positions in `report.json` stay in sheet millimetres.
- Texts in the DXF, the tags in symbols included, are fitted between the ends of their baseline as printed, so they
  are as long as on the paper in any font: a tag stays inside its bubble, a line number inside its frame. The preview
  draws a text that would come out longer smaller.

- The starter library moved from `app/plan2cad/symbols.dxf` to `libraries/din-10628`.
- A symmetric symbol (one that looks the same turned or mirrored) takes the pose in which most of its attribute
  positions have a text, so a tag below a symmetric valve is no longer missed, and then the pose with its back on the
  wall (a hob with its insertion point on the wall, not on its front edge). Poses whose outlines lie within 0.75 mm
  of each other count as the same.
- Text that is only strokes of a found symbol read as letters ("00" on a hob) is dropped, and text ink counts towards
  the circles and small closed shapes of a symbol.
- The simulated scans and photos of the benchmark moved to `app/plan2cad/simulate.py`.
- Lines in strong colour count as ink as dark as black ones (there the darkest colour channel is the grey value), so
  yellow and orange lines of a CAD plot are no longer lost. Photos keep their plain brightness.
- Text on scans is read with Tesseract's most accurate models (`tessdata_best` 4.1.0, German and English) instead of
  the fast ones Ubuntu ships. Characters are grouped into lines by similar height first, so two text lines beside a
  slanted stroke stay apart, and a character read unlike the glyphs most like it on the sheet is corrected by their
  vote (the slashed zero of a plotter font misread as 8, 9 or B). An O or I next to a digit in a code is a 0 or 1,
  and after "Maßstab" a scale read with letters ("1 SO") counts as 1:50. On the benchmark's P&ID scan, 93 % instead
  of 87 % of the texts are read exactly. The best models are slower: that scan takes about a tenth longer, the hard
  phone photo up to two thirds.
- Symbols are not taken from letters or from the frame around a text: a symbol may find at most 30 % of its outline
  on text that reads as words, and only a symbol that holds text (an instrument bubble) may be the box of a line
  number, an interlock diamond or the end of a label.
- Symbols drawn at the printed size keep to the size of most symbols on the sheet (at most 1.4 times larger or
  smaller), so the x marks of a capillary line are no longer taken for flow arrows, nor note markers for instruments.
- Symbols drawn at the printed size are looked for at the size of the sheet's text: on a sheet whose text is 3 mm
  high instead of about 2 mm, at one and a half times the library's size.
- Symbol recognition is faster on dense sheets. A symbol with a single connected port is only looked for at pipe
  ends (straight lines from 4 mm), not at the ends of the short strokes of letters, which also gives fewer false
  symbols on text; and candidates are checked with one search for nearby ink instead of two.
- Lines on scans and photos come out in one piece more often, and curves as true arcs, so that a conversion needs less
  rework (see `docs/photos-and-scans.md#how-it-works`):
  - A traced stroke that runs straight and then bends is cut into the fewest straight lines and arcs that follow it:
    a rounded corner, a vessel head or the round end of a line-number box is one arc or a few tangent ones, not a
    chain of short lines (a stroke that is round as a whole stays one arc, as before). Arcs that lie on one circle
    and were cut apart by a crossing or touching line become one arc again, also within a polyline.
  - A gap of up to 5 mm in a line is closed where the grey image shows a stroke along it (a faint stretch darker
    than the paper beside it, a crossing that cut it), and a branch from 4 mm that stops short of the pipe it points
    at is extended onto it. Gaps of bare paper, such as dashes, and the inside of a small symbol stay open; short
    pieces are joined only into a run of 8 mm or more (a dotted line), not the two sides of a valve.
  - Faint stretches of horizontal and vertical lines (a pale pen, a line that came out dotted) count as ink.
  - Of parallel pipes that end beside one symbol port, only the one pointing at it is snapped onto it, so the others
    are no longer turned onto the port.
  - Two or three dashes that run from one stroke to another, such as a short signal line between an instrument bubble
    and the symbol it acts on, become one dashed line; a short dash on its own is no longer dropped as a speck, and a
    dashed line is never dropped as a crumb.
- Text on scans and photos is read more completely:
  - A single character is read where it stands alone in a frame, such as the loop letter of an instrument bubble or
    an interlock diamond.
  - The text in circles that hold text is read a second time without the ring and the line across a shared display,
    so letters that touch them are no longer lost.
  - A text line read with low confidence stays a marked text also where the scan merged touching letters, as long as
    as many characters were read as its length holds.
  - A bar read at either end of a text line ("| LT"), the end of a leader beside it, stays linework.
- Symbols on scans and photos:
  - Nothing is placed in a strip of notes and tables below a line across the whole frame (revision table, issue
    record, title block).
  - A symbol drawn around a circle is centred on the circle found in the scan, and a symbol a little beside the pipes
    that end at its ports is moved onto them. A pipe end is snapped onto a port only where the port lies on its course,
    so that no pipe is turned askew.
  - A symbol drawn without curves, such as an off-page connector or a flow arrow, is not placed on the round end of a
    line-number box.
- Benchmark: new metrics for what makes rework tedious, `breaks_per_m` (places per metre of reference linework where
  the output stops although the line runs on), `curve_pieces` and `curves_as_arcs`, in a new report section.

### Fixed

- A P&ID with more vertical pipe labels than horizontal text was turned by 90 degrees. The rotation check now scores
  each rotation by horizontal text the right way up together with vertical text reading bottom to top.
- The round ends of a line-number box and rounded corners between two lines were read as brackets, so they were
  missing from the drawing and garbled the text next to them. They are now kept from text recognition.
- A circle in a box (a shared display instrument) was missed when OCR read its quarters as brackets, and the round
  end of a line-number box could be taken for a circle.
- Parallel lines close together, such as a pipe running into the box of its line number, could be pulled onto one
  averaged line when short strokes between them (letters touching the lines) chained them, so lines were shifted and
  box sides went missing. A piece now joins the line it continues at that line's own offset; pieces running beside
  one another less than 0.7 mm apart are still one stroke.
- A letter touching the round end of a line-number box was kept from text recognition with it.
- The frame of a long line-number box holding several texts was not found, so symbols could be placed on it.
- Letters in a small box, such as a line size, were read together with the box: the reading was garbled and the box
  went missing. A box or ring around letters is no longer taken for a character.
- A symbol could be placed on a small closed box whose text was not read, such as an instrument bubble on its sides
  or a valve on its letters. A symbol without a rectangle of its own no longer lies mostly on such a box.
- A P&ID listing areas in m² was taken for a floor plan. The drawing type now weighs the
  room areas and the building scale against nominal sizes, instrument bubbles and a title such as "R&I" or "R+I".

## [1.1.0] - unreleased

First public release under the name cad-ingest.

### Added

- Photos and scans of drawings to DXF (job operation `vectorize`): floor plans and P&IDs up to A0 from phone photos
  (JPEG, PNG, HEIC, WebP, TIFF, BMP) or scans (including PDF).
  - The sheet is found and straightened. Light is evened out, and linework, arcs, circles, dashed lines and filled
    areas are vectorised.
  - Texts are read with Tesseract OCR, and the scale is taken from the title block.
  - P&ID symbols are placed as blocks of a DXF symbol library, with their tags as attributes.
  - The result is a ZIP with the DXF in real-world millimetres, a PDF, a review overlay and a JSON report.
- `cad-ingest` command (`python -m app.convert`) that converts a single local file without Redis or a worker.
- Benchmark with synthetic drawings and ground truth (`bench/`).
- Multi-arch images (`linux/amd64`, `linux/arm64`) on GitHub Container Registry, with SBOM and build provenance; the
  test suite runs natively on both architectures.
- Open source project files: Apache-2.0 licence, contributing guide, code of conduct, security policy and
  `docker-compose.yml`.

## [1.0.0] - 2026-10-15

### Added

- Job service for CAD files: an HTTP API plus RQ workers on Redis, with queues, priorities, retries with backoff,
  callbacks, cancellation and restart safety.
- `extract`: searchable markdown from DXF, DWG, HPGL, STEP, IGES, BREP, IFC and mesh formats (STL, OBJ, PLY, OFF,
  3MF, DAE, glTF, GLB). It covers title blocks, texts, dimensions, layers, blocks, assembly trees, sizes and IFC
  property sets.
- `preview`: PDF for 2D drawings, binary glTF (GLB) for 3D models.
- Isolation of every job in its own process with a time and memory limit.

[1.1.0]: https://github.com/weyer-data-engineering/cad-ingest/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/weyer-data-engineering/cad-ingest/releases/tag/v1.0.0
