"""Command line: converts one local file without Redis, the job API or a worker, for trying cad-ingest out and for
scripts. CAD files give their markdown and a preview (PDF for 2D drawings, GLB for 3D models); photos and scans of
drawings also give the ZIP with the DXF. Unlike the work-horses of the job service, no time or memory limit applies.

    python -m app.convert part.step out/            # in the image: cad-ingest part.step out/
    python -m app.convert plan.jpg out/ --paper A3
"""
import argparse
import contextlib
import os
import sys
import tempfile
from pathlib import Path

from .config import Settings
from .errors import JobError
from .formats import FORMATS, PREVIEW, SCAN_2D
from .jobs import _convert


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cad-ingest", description="Converts a CAD file, or a photo or scan of a "
                                 "drawing, into markdown, a preview and (photos and scans) a DXF drawing.")
    ap.add_argument("source", help="CAD file (dxf, dwg, step, ifc, stl, ...) or photo/scan (jpg, png, heic, tif, pdf)")
    ap.add_argument("out", help="directory for <name>.md, the preview and, for photos and scans, <name>.zip")
    ap.add_argument("--no-preview", action="store_true", help="only extract the text")
    ap.add_argument("--paper", default="A3", choices=["A0", "A1", "A2", "A3", "A4"],
                    help="photos and scans: paper size of the sheet (default A3)")
    ap.add_argument("--scale", type=int, help="photos and scans: e.g. 50 for 1:50; read from the title block if not set")
    ap.add_argument("--dpi", type=float, help="scans without visible paper edges: their resolution")
    ap.add_argument("--drawing", default="auto", choices=["auto", "building", "pid"],
                    help="photos and scans: floor plan, P&ID or recognised from the texts (default)")
    args = ap.parse_args(argv)

    source = Path(args.source)
    ext = source.suffix.lstrip(".").lower()
    if ext not in FORMATS:
        ap.error(f"unsupported file extension '{ext}' (see the formats in the README)")
    kind = FORMATS[ext][0]
    operations = ["extract"] + ([] if args.no_preview else ["preview"]) + (["vectorize"] if kind == SCAN_2D else [])
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, source.stem)
    outputs = {"preview": f"{stem}.{PREVIEW[kind][0]}", "vectorize": f"{stem}.zip"}
    options = {"paper": args.paper, "scale": args.scale, "dpi": args.dpi, "drawing": args.drawing}

    # stdout carries only the paths of the results; libraries that print notices go to stderr
    with tempfile.TemporaryDirectory(prefix="cad-ingest-") as work, contextlib.redirect_stdout(sys.stderr):
        try:
            result = _convert(operations, ext, str(source), work, source.name, outputs["preview"], Settings.from_env(),
                              options, outputs["vectorize"])
        except JobError as e:
            print(f"{source.name}: {e.code}: {e.message}", file=sys.stderr)
            return 1
    with open(f"{stem}.md", "w", encoding="utf-8") as f:
        f.write(result.get("markdown", ""))
    for path in [f"{stem}.md"] + [outputs[op] for op in operations if op in outputs]:
        print(path)
    for warning in result["warnings"]:
        print(f"Warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
