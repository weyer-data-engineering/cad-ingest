# Guidance for coding agents

cad-ingest was written with Claude Code, and this file is the brief for any coding agent (or person) working on it.
Read `README.md` for what the service does and `docs/` for the job API, operations and the photo and scan pipeline.

## Layout

- `app/main.py`: the job API (FastAPI). Stateless; all state is in Redis (`app/store.py`).
- `app/worker.py`, `app/jobs.py`: RQ worker pool and the work-horse code (download, convert, upload, callback).
- `app/drawing.py`, `app/brep.py`, `app/mesh.py`, `app/ifc.py`: markdown and previews per CAD format family;
  `app/formats.py` maps file extensions to them.
- `app/scan.py` and `app/plan2cad/`: photos and scans to DXF (`vectorize`).
- `app/plan2cad/library.py` and `libraries/`: symbol libraries (format, reading, choosing); `libraries/din-10628` is the
  built-in starter library. `docs/symbol-libraries.md` is the format a library author relies on: keep it stable.
- `app/convert.py`: the `cad-ingest` command line for single local files; `app/library_tool.py`:
  `cad-ingest library check|build`.
- `bench/`: synthetic drawings with ground truth and the benchmark for `app/plan2cad`; the simulated scans and photos
  are in `app/plan2cad/simulate.py`, because `cad-ingest library check` uses them too.
- `tests/`: pytest; the job tests run a real Redis and real worker processes.

## Rules

- **Keep the job API stable.** Endpoints, status codes, error codes and the JSON shape are a contract with clients.
  Additive changes only within a major version; document every change in `docs/api.md` and `CHANGELOG.md`.
- **Treat every input as hostile.** Each job runs in its own work-horse process with a time and an address-space
  limit. Never use a job's file name as a path, and never log a URL (signatures), only its host. Symbol libraries
  are data: never run code from one, and read ZIPs in memory with a size limit.
- **Check licences.** No AGPL dependencies (for example PyMuPDF). GPL tools only as separate processes, like
  LibreDWG's `dwg2dxf`. Add every new bundled component to the licence table in the README.
- **Measure pipeline changes.** For `app/plan2cad`, run `python bench/run.py --quick` before and after and report
  both tables; do not trade one case for another silently.
- **Test in the image.** `docker build --target test .` runs the suite with the same native libraries as
  production. CI runs it on amd64 and arm64.
- **Write like the surrounding code.** Small functions, domain names, comments that explain why. Docs in plain,
  short English sentences.
