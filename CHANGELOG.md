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

### Changed

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

### Fixed

- A P&ID with more vertical pipe labels than horizontal text was turned by 90 degrees. The rotation check now scores
  each rotation by horizontal text the right way up together with vertical text reading bottom to top.
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
