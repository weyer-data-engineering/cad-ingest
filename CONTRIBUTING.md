# Contributing to cad-ingest

Thank you for helping. Bug reports, real-world failure cases, new symbols for the P&ID library, format support and
documentation fixes are all welcome.

## Reporting bugs and asking for features

Open an issue with the bug or feature template. For a file that converts badly, the most useful report contains the
file itself or a minimal version of it that shows the problem. If you cannot share the original, describe what is
on it (format, CAD program that wrote it, paper size, scan resolution) and attach the `overlay.png` and
`report.json` from the result ZIP.

Security problems go through private reporting, see [SECURITY.md](SECURITY.md).

## Development setup

Everything runs in Docker, with the same libraries as the published image:

```bash
docker build --target test .        # builds the image and runs the whole test suite inside it
docker build -t cad-ingest:dev .    # the runtime image
docker compose up --build           # Redis, the job API on http://localhost:8080 and a worker
```

Without Docker you need Python 3.12, `redis-server` and `tesseract` (with the `deu` and `eng` language data) on the
`PATH`:

```bash
pip install -r requirements-dev.txt
pytest                              # the DWG tests are skipped unless LibreDWG's dwgadd is on the PATH
```

The tests use a real Redis and real worker processes against a local HTTP server that plays the storage and the
client, so they cover the job service end to end.

The image is based on Ubuntu 24.04 (Python 3.12). Ubuntu ships no LibreDWG tools, so the `Dockerfile` builds
LibreDWG from a pinned release tarball (`LIBREDWG_VERSION`, `LIBREDWG_SHA256`). It stays on 0.13.3 on purpose:
0.13.4 writes every layer as switched off, so DWG previews come out blank. Check a real DWG preview before raising
the version. `cadquery-ocp-novtk` is used instead of `cadquery-ocp` because the latter pulls in VTK (about 640 MB).

## Changing the photo and scan pipeline

Changes to `app/plan2cad` are measured, not judged by eye. Run the benchmark before and after your change and put
both tables into the pull request:

```bash
python bench/run.py --out bench/out --quick     # a few minutes; without --quick all seeds, A0 and the resolution sweep
```

A change should not make any case worse without a good reason, and the reason belongs in the pull request.

## Pull requests

- One topic per pull request, with a test for every behaviour you add or fix.
- Follow the style of the surrounding code: small functions, names from the domain, comments that explain why rather
  than what.
- Keep the job API compatible. New fields are fine; renaming or removing fields, status codes or error codes needs a
  new major version.
- Check the licence before adding a dependency. Nothing under the AGPL (for example PyMuPDF), and GPL code only as a
  separate program, like LibreDWG's `dwg2dxf`.
- Update the README or the files in `docs/` when you change behaviour, configuration or the API, and add a line to
  [CHANGELOG.md](CHANGELOG.md).
- CI runs the test suite in the image on amd64 and arm64; both must pass.

AI-assisted contributions are welcome. This project was written with Claude Code, and [CLAUDE.md](CLAUDE.md) holds
the guidance for coding agents. You are responsible for every change you submit, however it was written.

## Releases

Every push to `main` publishes `ghcr.io/weyer-data-engineering/cad-ingest:main`. A Git tag `vX.Y.Z` publishes the
tags `vX.Y.Z`, `X.Y.Z`, `X.Y` and `latest`, with build provenance and a CycloneDX SBOM attached to the image and to
the GitHub release. Give the version its date in [CHANGELOG.md](CHANGELOG.md) before tagging.

## Licence

By contributing you agree that your contribution is licensed under the [Apache License 2.0](LICENSE), like the rest
of the project (see section 5 of the licence).

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Please be kind.
