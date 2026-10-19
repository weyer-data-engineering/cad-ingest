"""2D drawings: DXF (ezdxf), DWG (LibreDWG dwg2dxf -> ezdxf) and HPGL (ezdxf.addons.hpgl2)."""
import contextlib
import io
import os
import re
import subprocess
from collections import Counter

import ezdxf
from ezdxf import bbox, recover, units
from ezdxf.addons.drawing import Frontend, RenderContext, layout as page_layout
from ezdxf.addons.drawing.config import BackgroundPolicy, Configuration
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

# The hpgl2 add-on imports ezdxf's optional PyMuPDF backend, which prints a notice when PyMuPDF is missing. It is
# never used here (and is AGPL), so the notice would only confuse whoever reads the logs.
with contextlib.redirect_stdout(io.StringIO()):
    from ezdxf.addons.hpgl2 import api as hpgl

from .errors import ConversionError

MAX_LINES = 500  # per list in the markdown
PAGE_LONG_SIDE_MM = 420  # model space has no paper size (A3 long side); vector output scales losslessly
A3_LANDSCAPE_MM = (420, 297)
DIMENSION_TYPES = {0: "linear", 1: "aligned", 2: "angular", 3: "diameter", 4: "radius", 5: "angular", 6: "ordinate"}
HPGL_EXTS = {"plt", "hpgl", "hpg", "hp2"}


def extract(ext: str, path: str, workdir: str, name: str) -> tuple[str, list[str]]:
    if ext in HPGL_EXTS:
        return _extract_hpgl(path, name)
    warnings: list[str] = []
    doc = _load_dxf(ext, path, workdir, warnings)
    lines = [f"# CAD drawing: {name}", "",
             f"- Format: {'DWG' if ext == 'dwg' else 'DXF'}, version {doc.dxfversion} ({doc.acad_release})",
             f"- Drawing units: {units.decode(doc.header.get('$INSUNITS', 0)) or 'unspecified'}"]
    extmin, extmax = doc.header.get("$EXTMIN"), doc.header.get("$EXTMAX")
    if extmin and extmax and extmax[0] > extmin[0]:
        lines.append(f"- Model extents: {extmax[0] - extmin[0]:.6g} x {extmax[1] - extmin[1]:.6g}")
    lines.append(f"- Layouts: {', '.join(l.name for l in _layouts(doc))}")
    for layout in _layouts(doc):
        lines += _layout_section(layout)
    return "\n".join(lines) + "\n", warnings


def preview(ext: str, path: str, out: str) -> list[str]:
    warnings: list[str] = []
    if ext in HPGL_EXTS:
        with open(path, "rb") as f:
            data = f.read()
        if b"\x1b%" not in data:  # plain HPGL without the PCL wrapper that ezdxf looks for
            data = b"\x1b%1B" + data
        try:
            doc = hpgl.to_dxf(data)
        except hpgl.Hpgl2Error as e:
            raise ConversionError(f"Not a readable HPGL file: {e}") from None
        layouts = [doc.modelspace()]  # the generated paper space only holds a viewport onto it
    else:
        doc = _load_dxf(ext, path, os.path.dirname(out), warnings)
        layouts = _layouts(doc)
    # White paper, drawing colours kept; ezdxf swaps the "white/black" colour to black on white.
    config = Configuration(background_policy=BackgroundPolicy.WHITE)
    pages = 0
    with PdfPages(out) as pdf:
        for layout in layouts:
            if not _has_content(layout):
                continue
            try:
                fig = Figure(figsize=_size_inches(layout))  # no pyplot: no global state, freed after each page
                ax = fig.add_axes((0, 0, 1, 1))
                Frontend(RenderContext(doc), MatplotlibBackend(ax, adjust_figure=False), config).draw_layout(layout, finalize=True)
                pdf.savefig(fig, facecolor="white")
                pages += 1
            except Exception as e:  # one bad layout must not lose the others
                warnings.append(f"Layout '{layout.name}' could not be rendered: {type(e).__name__}: {e}")
    if pages == 0:
        raise ConversionError("The drawing contains no renderable geometry")
    return warnings


def _load_dxf(ext: str, path: str, workdir: str, warnings: list[str]) -> ezdxf.document.Drawing:
    if ext == "dwg":
        # LibreDWG is GPL: used as a separate process only.
        converted = os.path.join(workdir, "converted.dxf")
        proc = subprocess.run(["dwg2dxf", "-y", "-o", converted, path], capture_output=True, text=True)
        if not os.path.exists(converted) or os.path.getsize(converted) == 0:
            raise ConversionError(f"DWG file could not be converted: {proc.stderr.strip()[-300:] or 'unknown error'}")
        if proc.returncode:
            warnings.append("The DWG file was only partly readable; the content may be incomplete")
        path = converted
    try:
        return ezdxf.readfile(path)
    except Exception:
        try:
            doc, _ = recover.readfile(path)
        except Exception as e:
            raise ConversionError(f"Not a readable DXF file: {e}") from None
        warnings.append("The drawing was damaged and has been repaired; the content may be incomplete")
        return doc


def _has_content(layout) -> bool:
    # The default paper viewport (id 1) of every paper space layout does not count as content.
    return any(e.dxftype() != "VIEWPORT" or e.dxf.get("id", 0) > 1 for e in layout)


def _size_inches(layout) -> tuple[float, float]:
    """Paper size of a paper space layout; model space gets the A3 long side with the aspect ratio of its content."""
    if not layout.is_modelspace:
        page = page_layout.Page.from_dxf_layout(layout)
        if page.width_in_mm > 0 and page.height_in_mm > 0:
            return page.width_in_mm / 25.4, page.height_in_mm / 25.4
    extents = bbox.extents(layout, fast=True)
    if not extents.has_data or extents.size.x <= 0 or extents.size.y <= 0:
        return tuple(v / 25.4 for v in A3_LANDSCAPE_MM)
    ratio = min(max(extents.size.x / extents.size.y, 0.05), 20)
    width, height = (PAGE_LONG_SIDE_MM, PAGE_LONG_SIDE_MM / ratio) if ratio >= 1 else (PAGE_LONG_SIDE_MM * ratio, PAGE_LONG_SIDE_MM)
    return width / 25.4, height / 25.4


def _layouts(doc) -> list:
    return sorted(doc.layouts, key=lambda layout: layout.dxf_layout.dxf.taborder)


def _measurement(dimension) -> float | None:
    try:
        value = dimension.get_measurement()
    except Exception:  # unusual dimension geometry
        return None
    return value if isinstance(value, (int, float)) else None


_CODES = re.compile(r"%%([cdpCDP])|%%[uoUO]")
_SYMBOLS = {"c": "Ø", "d": "°", "p": "±"}


def _clean(text: str) -> str:
    """Resolves the TEXT control codes (%%c = diameter sign, ...) and collapses whitespace."""
    return " ".join(_CODES.sub(lambda m: _SYMBOLS.get((m.group(1) or "").lower(), ""), text).split())


def _count_lines(counter: Counter) -> list[str]:
    lines = [f"- {text}" + (f" (x{n})" if n > 1 else "") for text, n in counter.items()]
    return lines[:MAX_LINES] + ([f"- ... {len(lines) - MAX_LINES} more"] if len(lines) > MAX_LINES else [])


def _layout_section(layout) -> list[str]:
    texts, dimensions, blocks, layers = Counter(), Counter(), Counter(), set()
    attributes: dict[str, Counter] = {}  # block name -> Counter of "TAG: value" attribute sets
    for e in layout:
        kind = e.dxftype()
        layers.add(e.dxf.layer)
        if kind == "TEXT":
            texts[_clean(e.dxf.text)] += 1
        elif kind == "MTEXT":
            texts[_clean(e.plain_text())] += 1
        elif kind == "MULTILEADER":
            texts[_clean(e.get_mtext_content())] += 1
        elif kind == "DIMENSION":
            measurement = _measurement(e)
            label = DIMENSION_TYPES.get(e.dimtype & 7, "dimension")
            if measurement is not None:
                label += f" {measurement:.4f}".rstrip("0").rstrip(".")
            if e.dxf.get("text", "") not in ("", "<>", " "):
                label += f' (text "{_clean(e.dxf.text)}")'
            dimensions[label] += 1
        elif kind == "INSERT":
            if not e.dxf.name.startswith("*"):
                blocks[e.dxf.name] += 1
            values = "; ".join(f"{a.dxf.tag}: {_clean(a.dxf.text)}" for a in e.attribs if a.dxf.text.strip())
            if values:
                attributes.setdefault(e.dxf.name, Counter())[values] += 1
    texts.pop("", None)
    if not (texts or dimensions or blocks or attributes):
        return []
    lines = ["", f"## Layout: {layout.name}"]
    if attributes:
        lines += ["", "### Title block and block attributes", ""]
        for block, sets in attributes.items():
            lines += [f"- Block {block}:"] + [f"  {s}" for s in _count_lines(sets)]
    for title, counter in (("Texts", texts), ("Dimensions", dimensions)):
        if counter:
            lines += ["", f"### {title}", ""] + _count_lines(counter)
    if blocks:
        lines += ["", "### Blocks", ""] + _count_lines(Counter(dict(blocks.most_common())))
    lines += ["", f"Layers: {', '.join(sorted(layers)[:MAX_LINES])}"]
    return lines


def _extract_hpgl(path: str, name: str) -> tuple[str, list[str]]:
    with open(path, "rb") as f:
        raw = f.read().decode("latin-1")
    if not re.search(r"(?:IN|PU|PD|PA|SP)[\s\d;+.-]", raw):
        raise ConversionError("Not an HPGL plot file")
    labels = Counter(" ".join(m.split()) for m in re.findall(r"LB([^\x03]*)\x03", raw))
    labels.pop("", None)
    pens = sorted({int(p) for p in re.findall(r"SP(\d+)", raw)})
    lines = [f"# CAD drawing: {name}", "", "- Format: HPGL plot file",
             f"- Size: {len(raw)} bytes", f"- Pens used: {', '.join(map(str, pens)) or 'none'}"]
    if labels:
        lines += ["", "## Labels", ""] + _count_lines(labels)
    return "\n".join(lines) + "\n", []
