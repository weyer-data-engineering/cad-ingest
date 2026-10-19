"""Photos and scans of drawings (operation "vectorize"): the sheet is rectified and its lines, arcs, circles, fills and
texts are recognised (see plan2cad.pipeline). One run serves all operations: "vectorize" writes a ZIP with the DXF,
"extract" returns the recognised texts as markdown, "preview" a PDF of the recognised drawing."""
import io
import json
import zipfile
from collections import Counter
from dataclasses import asdict

import cv2

from .config import Settings
from .drawing import _count_lines
from .errors import ConversionError, JobError
from .plan2cad import library, symbols
from .plan2cad.export import overlay, to_dxf, to_figure
from .plan2cad.imageio import load_image
from .plan2cad.pipeline import convert

KINDS = {"building": "floor plan", "pid": "P&ID"}
SHEETS = {"paper": "paper edges", "frame": "drawing frame", "image": "whole image, no sheet edge found"}


def vectorize(path: str, name: str, options: dict, result: str, preview: str | None,
              own_libraries: list[library.Library] | tuple = ()) -> tuple[str, list[str]]:
    """Recognises the drawing in the photo or scan at `path`, writes the ZIP to `result` and the PDF to `preview`.
    `own_libraries` are symbol libraries the job brought along. Returns the markdown and the warnings."""
    libraries = _libraries(options, own_libraries)
    with open(path, "rb") as f:
        try:
            img, dpi = load_image(f.read())
        except ValueError as e:
            raise ConversionError(str(e)) from None
    r = convert(img, paper=options.get("paper", "A3"), scale=options.get("scale"), dpi=options.get("dpi") or dpi,
                drawing=None if options.get("drawing", "auto") == "auto" else options["drawing"],
                templates=library.combine(libraries))
    r.info["libraries"] = [{"id": lib.id, "version": lib.version} for lib in libraries]
    d, counts = r.drawing, r.info["counts"]
    if not any(counts[k] for k in ("lines", "arcs", "circles", "polylines")):
        raise ConversionError("No drawing was recognised in the image")
    pdf = io.BytesIO()
    to_figure(d, pdf, "pdf")
    h, w = r.image.shape
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as z:
        dxf = io.StringIO()
        to_dxf(d, underlay=("underlay.png", (w, h))).write(dxf)  # the DXF shows the rectified photo underneath
        z.writestr("plan.dxf", dxf.getvalue())
        z.writestr("plan.pdf", pdf.getvalue())
        z.writestr("overlay.png", cv2.imencode(".png", overlay(r.image, d, r.ppm))[1].tobytes())
        z.writestr("underlay.png", cv2.imencode(".png", r.image, [cv2.IMWRITE_PNG_COMPRESSION, 6])[1].tobytes())
        z.writestr("report.json", json.dumps(r.info | _report(d), ensure_ascii=False, indent=1))
    if preview:
        with open(preview, "wb") as f:
            f.write(pdf.getvalue())
    warnings = []
    if r.info["sheet"] == "image":
        warnings.append("No sheet edge was found; the whole image was used and the sizes may be off")
    if unsure := sum(t.uncertain for t in d.texts):
        warnings.append(f"{unsure} texts could not be read reliably; they are on the layer P2C-TEXT-CHECK")
    return _markdown(name, r.info, d), warnings


def _libraries(options: dict, own: list[library.Library] | tuple) -> list[library.Library]:
    """The symbol libraries of the job: its own, then the installed ones it names, else the default ones."""
    s = Settings.from_env()
    try:
        return library.select(options.get("libraries"), own, s.symbol_libraries, s.default_symbol_libraries)
    except library.LibraryError as e:
        raise JobError("invalid_library", str(e)) from None


def _report(d) -> dict:
    """Texts, symbols with their library, attributes and ports, and the lines joining two symbol ports (by index)."""
    placed = [{"name": s.name, "library": s.template.library.id, "x": s.x, "y": s.y, "angle": s.angle,
               "mirror": s.mirror, "scale": s.scale, "attributes": {tag: t.text for tag, t in s.attributes.items()},
               "ports": s.ports} for s in d.symbols]
    return {"texts": [asdict(t) for t in d.texts], "symbols": placed, "connections": symbols.connections(d)}


def _markdown(name: str, info: dict, d) -> str:
    lines = [f"# Photo or scan of a drawing: {name}", "",
             f"- Recognised as: {KINDS.get(info['drawing_type'], 'technical drawing')}",
             f"- Paper: {info['paper']}, sheet {info['sheet_size_mm'][0]:g} x {info['sheet_size_mm'][1]:g} mm "
             f"(found by: {SHEETS[info['sheet']]})",
             f"- Scale: {'1:' + str(info['scale']) if info['scale'] else 'unknown'}",
             f"- Elements: {', '.join(f'{n} {k}' for k, n in info['counts'].items() if n)}"]
    reading_order = sorted(d.texts, key=lambda t: (round(t.y / 5), t.x))
    for title, unsure in (("Texts", False), ("Texts to check (uncertain reading)", True)):
        found = Counter(t.text for t in reading_order if t.uncertain == unsure)
        if found:
            lines += ["", f"## {title}", ""] + _count_lines(found)
    found = Counter(" ".join([s.template.description] + [t.text for t in s.attributes.values()])
                    for s in sorted(d.symbols, key=lambda s: (round(s.y / 5), s.x)))
    if found:
        lines += ["", "## Symbols", ""] + _count_lines(found)
    return "\n".join(lines) + "\n"


def main() -> None:
    """The operation vectorize on a local file, for trying it out: writes the ZIP and prints the markdown."""
    import argparse
    import os

    ap = argparse.ArgumentParser(prog="python -m app.scan", description="Turns a photo or scan of a drawing into DXF.")
    ap.add_argument("source", help="photo or scan (jpg, png, heic, tif, pdf, ...)")
    ap.add_argument("result", help="ZIP to write")
    ap.add_argument("--paper", default="A3", choices=["A0", "A1", "A2", "A3", "A4"])
    ap.add_argument("--scale", type=int, help="e.g. 50 for 1:50; read from the title block if not given")
    ap.add_argument("--dpi", type=float, help="scan resolution, only needed without visible paper edges")
    ap.add_argument("--drawing", default="auto", choices=["auto", "building", "pid"])
    a = ap.parse_args()
    options = {"paper": a.paper, "scale": a.scale, "dpi": a.dpi, "drawing": a.drawing}
    markdown, warnings = vectorize(a.source, os.path.basename(a.source), options, a.result, None)
    print(markdown + "".join(f"\nWarning: {w}" for w in warnings))


if __name__ == "__main__":
    main()
