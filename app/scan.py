"""Photos and scans of drawings (operation "vectorize"): the sheet is rectified and its lines, arcs, circles, fills and
texts are recognised (see plan2cad.pipeline), on every page of a PDF or TIFF. One run serves all operations:
"vectorize" writes a ZIP with the DXF, "extract" returns the recognised texts as markdown, "preview" a PDF of the
recognised drawing."""
import io
import json
import time
import zipfile
from collections import Counter
from dataclasses import asdict

import cv2
import pypdfium2

from .config import Settings
from .drawing import _count_lines
from .errors import ConversionError, JobError
from .plan2cad import library, symbols
from .plan2cad.export import overlay, to_dxf, to_figure
from .plan2cad.imageio import load_image, page_count
from .plan2cad.model import Drawing
from .plan2cad.pipeline import Result, convert

KINDS = {"building": "floor plan", "pid": "P&ID"}
SHEETS = {"paper": "paper edges", "frame": "drawing frame", "image": "whole image, no sheet edge found"}


def vectorize(path: str, name: str, options: dict, result: str, preview: str | None,
              own_libraries: list[library.Library] | tuple = (),
              deadline: float | None = None) -> tuple[str, list[str]]:
    """Recognises the drawing on each page of the photo or scan at `path` (a PDF or TIFF may have several), writes the
    ZIP to `result` and the PDF to `preview`. `own_libraries` are symbol libraries the job brought along. Pages that
    would end after the `deadline` (time.monotonic) are left out. Returns the markdown and the warnings.

    The first page with a drawing goes to the root of the ZIP, any further page n to the folder page-n/."""
    libraries = _libraries(options, own_libraries)
    with open(path, "rb") as f:
        data = f.read()
    try:
        total = page_count(data)
    except ValueError as e:
        raise ConversionError(str(e)) from None
    numbers, warnings = _pages(options.get("pages"), total, Settings.from_env().max_scan_pages)
    started, done, reports, pdfs = time.monotonic(), [], [], []
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as z:
        for k, n in enumerate(numbers):
            if deadline and k and time.monotonic() + (time.monotonic() - started) / k > deadline:
                warnings.append(f"Pages left out to stay within the time limit: {_listed(numbers[k:])}")
                break
            r = _page(data, n - 1, options, libraries)
            if r is None:
                if total == 1:
                    raise ConversionError("No drawing was recognised in the image")
                warnings.append(f"Page {n}: no drawing was recognised")
                continue
            folder = f"page-{n}/" if done else ""
            pdfs.append(_write(z, folder, r))
            reports.append((folder, r.info | _report(r.drawing) | {"page": n}))
            done.append((n, r.info, r.drawing))
            warnings += [f"Page {n}: {w}" if total > 1 else w for w in _warnings(r)]
        if not done:
            raise ConversionError("No drawing was recognised on any page")
        for folder, report in reports:
            z.writestr(folder + "report.json", json.dumps(report | {"pages": [n for n, _, _ in done]},
                                                          ensure_ascii=False, indent=1))
    if preview:
        with open(preview, "wb") as f:
            f.write(_merged(pdfs))
    return _markdown(name, done, total), warnings


def _pages(wanted: list[int] | None, total: int, limit: int) -> tuple[list[int], list[str]]:
    """The page numbers to convert (from 1): the ones asked for, else all, at most `limit`; and warnings."""
    numbers, warnings = sorted(set(wanted)) if wanted else list(range(1, total + 1)), []
    if missing := [n for n in numbers if n > total]:
        warnings.append(f"Pages left out that the file does not have (it has {total}): {_listed(missing)}")
        numbers = [n for n in numbers if n <= total]
    if not numbers:
        raise ConversionError(warnings[0])
    if len(numbers) > limit:
        warnings.append(f"Pages left out over the limit of {limit} per job (MAX_SCAN_PAGES): {_listed(numbers[limit:])}")
        numbers = numbers[:limit]
    return numbers, warnings


def _listed(numbers: list[int]) -> str:
    return ", ".join(map(str, numbers[:10])) + (", ..." if len(numbers) > 10 else "")


def _page(data: bytes, page: int, options: dict, libraries: list[library.Library]) -> Result | None:
    """The recognised drawing of a page (counted from 0), or None if it holds no drawing."""
    try:
        img, dpi = load_image(data, page)
    except ValueError as e:
        raise ConversionError(str(e)) from None
    r = convert(img, paper=options.get("paper", "A3"), scale=options.get("scale"), dpi=options.get("dpi") or dpi,
                drawing=None if options.get("drawing", "auto") == "auto" else options["drawing"],
                templates=library.combine(libraries))
    r.info["libraries"] = [{"id": lib.id, "version": lib.version} for lib in libraries]
    return r if any(r.info["counts"][k] for k in ("lines", "arcs", "circles", "polylines")) else None


def _write(z: zipfile.ZipFile, folder: str, r: Result) -> bytes:
    """Writes the DXF, PDF, overlay and underlay of a page to the folder of the ZIP; returns the PDF."""
    d, pdf = r.drawing, io.BytesIO()
    to_figure(d, pdf, "pdf")
    h, w = r.image.shape
    dxf = io.StringIO()
    to_dxf(d, underlay=("underlay.png", (w, h))).write(dxf)  # the DXF shows the rectified photo underneath
    z.writestr(folder + "plan.dxf", dxf.getvalue())
    z.writestr(folder + "plan.pdf", pdf.getvalue())
    z.writestr(folder + "overlay.png", cv2.imencode(".png", overlay(r.image, d, r.ppm))[1].tobytes())
    z.writestr(folder + "underlay.png", cv2.imencode(".png", r.image, [cv2.IMWRITE_PNG_COMPRESSION, 6])[1].tobytes())
    return pdf.getvalue()


def _warnings(r: Result) -> list[str]:
    out = []
    if r.info["sheet"] == "image":
        out.append("No sheet edge was found; the whole image was used and the sizes may be off")
    if unsure := sum(t.uncertain for t in r.drawing.texts):
        out.append(f"{unsure} texts could not be read reliably; they are on the layer P2C-TEXT-CHECK")
    return out


def _merged(pdfs: list[bytes]) -> bytes:
    """The pages of several PDFs as one."""
    if len(pdfs) == 1:
        return pdfs[0]
    out, buf = pypdfium2.PdfDocument.new(), io.BytesIO()
    for pdf in pdfs:
        out.import_pages(pypdfium2.PdfDocument(pdf))
    out.save(buf)
    return buf.getvalue()


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


def _markdown(name: str, pages: list[tuple[int, dict, Drawing]], total: int) -> str:
    lines = [f"# Photo or scan of a drawing: {name}"]
    for n, info, d in pages:
        if total > 1:
            lines += ["", f"## Page {n} of {total}"]
        lines += [""] + _page_markdown(info, d, "###" if total > 1 else "##")
    return "\n".join(lines) + "\n"


def _page_markdown(info: dict, d: Drawing, heading: str) -> list[str]:
    lines = [f"- Recognised as: {KINDS.get(info['drawing_type'], 'technical drawing')}",
             f"- Paper: {info['paper']}, sheet {info['sheet_size_mm'][0]:g} x {info['sheet_size_mm'][1]:g} mm "
             f"(found by: {SHEETS[info['sheet']]})",
             f"- Scale: {'1:' + str(info['scale']) if info['scale'] else 'unknown'}",
             f"- Elements: {', '.join(f'{n} {k}' for k, n in info['counts'].items() if n)}"]
    reading_order = sorted(d.texts, key=lambda t: (round(t.y / 5), t.x))
    for title, unsure in (("Texts", False), ("Texts to check (uncertain reading)", True)):
        found = Counter(t.text for t in reading_order if t.uncertain == unsure)
        if found:
            lines += ["", f"{heading} {title}", ""] + _count_lines(found)
    found = Counter(" ".join([s.template.description] + [t.text for t in s.attributes.values()])
                    for s in sorted(d.symbols, key=lambda s: (round(s.y / 5), s.x)))
    if found:
        lines += ["", f"{heading} Symbols", ""] + _count_lines(found)
    return lines


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
