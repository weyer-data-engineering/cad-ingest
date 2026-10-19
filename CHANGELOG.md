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
- Every page of a PDF or multi-page TIFF scan is converted, not only the first one. The first page with a drawing stays
  at the root of the ZIP, every further page n goes to `page-n/` with the same files; each `report.json` names its
  `page` and all `pages` converted, the preview holds every page and the markdown has a section per page. Pages
  without a drawing are skipped with a warning.
  - Job API: `options.pages` chooses pages; new setting `MAX_SCAN_PAGES` (default 20). A job leaves out the pages
    it would not finish within `CONVERSION_TIMEOUT_SECONDS` and says so in a warning, rather than failing.
  - Command line: `--pages 1-3,5`.

### Changed

- The starter library moved from `app/plan2cad/symbols.dxf` to `libraries/din-10628`.
- A symmetric symbol (one that looks the same turned half way or mirrored) takes the pose in which most of its
  attribute positions have a text, so a tag below a symmetric valve is no longer missed.
- The simulated scans and photos of the benchmark moved to `app/plan2cad/simulate.py`.

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
