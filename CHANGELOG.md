# Changelog

All notable changes to cad-ingest are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/): the job API stays compatible within a major version.

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
