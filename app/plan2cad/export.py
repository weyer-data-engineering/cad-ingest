"""Writes a Drawing as DXF, PDF/PNG and as a review overlay."""

from __future__ import annotations

import math
from dataclasses import replace

import cv2
import ezdxf
import numpy as np
from ezdxf import xref
from ezdxf.enums import TextEntityAlignment
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Arc as MplArc
from matplotlib.patches import Circle as MplCircle
from matplotlib.patches import Polygon as MplPolygon

from .model import Drawing
from .render import CAP, DASH, GAP, dashes, render

LAYERS = {"P2C-LINE": 7, "P2C-DASH": 5, "P2C-CURVE": 3, "P2C-FILL": 8, "P2C-TEXT": 1, "P2C-UNDERLAY": 9,
          "P2C-WALL": 150, "P2C-DOOR": 30, "P2C-PIPE": 150, "P2C-EQUIPMENT": 94, "P2C-SIGNAL": 5, "P2C-FRAME": 8,
          "P2C-TEXT-CHECK": 6, "P2C-SYMBOL": 4}
KIND_LAYERS = {"wall": "P2C-WALL", "door": "P2C-DOOR", "pipe": "P2C-PIPE", "equipment": "P2C-EQUIPMENT",
               "signal": "P2C-SIGNAL", "frame": "P2C-FRAME"}
LINEWEIGHTS = (13, 18, 25, 35, 50, 70, 100, 140, 200)  # ISO pen widths in 1/100 mm
PT_PER_MM = 72 / 25.4


def lineweight(width_mm: float) -> int:
    return min(LINEWEIGHTS, key=lambda lw: abs(lw - width_mm * 100))


def to_dxf(d: Drawing, underlay: tuple[str, tuple[int, int]] | None = None) -> ezdxf.document.Drawing:
    """Builds a DXF document in real-world millimetres (sheet mm times the drawing scale).

    ``underlay`` optionally references the rectified photo (file name, size in pixels) as a
    raster image on its own layer, so the result can be checked and traced over in CAD.
    """
    s = d.scale or 1
    doc = ezdxf.new("R2018", units=ezdxf.units.MM)
    doc.header["$MEASUREMENT"] = 1
    doc.header["$LWDISPLAY"] = 1
    doc.linetypes.add("P2C_DASH", [(DASH + GAP) * s, DASH * s, -GAP * s], description="Dashed __ __ __")
    for name, color in LAYERS.items():
        doc.layers.add(name, color=color)
    msp = doc.modelspace()

    def pt(x: float, y: float) -> tuple[float, float]:
        return x * s, (d.height - y) * s

    for ln in d.lines:
        attribs = {"layer": "P2C-DASH" if ln.dashed else "P2C-LINE", "lineweight": lineweight(ln.width)}
        if ln.dashed:
            attribs["linetype"] = "P2C_DASH"
        msp.add_line(pt(ln.x1, ln.y1), pt(ln.x2, ln.y2), dxfattribs=attribs)
    for pl in d.polylines:  # y points up in DXF: an arc turning clockwise on screen has a negative bulge
        layer = KIND_LAYERS.get(pl.kind) or ("P2C-DASH" if pl.dashed else "P2C-LINE")
        attribs = {"layer": layer, "lineweight": lineweight(pl.width)}
        if pl.dashed:
            attribs["linetype"] = "P2C_DASH"
        e = msp.add_lwpolyline([(*pt(x, y), -b) for (x, y), b in zip(pl.points, pl.bulges + [0.0])], format="xyb",
                               close=pl.closed, dxfattribs=attribs)
        e.dxf.flags |= 128  # dash pattern runs on around corners
    for a in d.arcs:  # y flips, so clockwise screen angles become counter-clockwise DXF angles
        msp.add_arc(pt(a.cx, a.cy), a.r * s, (-a.end) % 360, (-a.start) % 360,
                    dxfattribs={"layer": "P2C-CURVE", "lineweight": lineweight(a.width)})
    for c in d.circles:
        msp.add_circle(pt(c.cx, c.cy), c.r * s, dxfattribs={"layer": "P2C-CURVE", "lineweight": lineweight(c.width)})
    for g in d.polygons:
        points = [pt(x, y) for x, y in g.points]
        if g.filled:
            hatch = msp.add_hatch(color=256, dxfattribs={"layer": "P2C-FILL"})
            hatch.paths.add_polyline_path(points, is_closed=True)
        else:
            msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "P2C-LINE", "lineweight": lineweight(g.width)})
    for t in d.texts:
        layer = "P2C-TEXT-CHECK" if t.uncertain else "P2C-TEXT"
        msp.add_text(t.text, height=t.height * s, rotation=t.angle, dxfattribs={"layer": layer}).set_placement(
            pt(t.x, t.y), align=TextEntityAlignment.MIDDLE_CENTER)
    _symbols(doc, d, pt, s)
    if underlay:
        name, size = underlay
        image_def = doc.add_image_def(filename=name, size_in_pixel=size)
        msp.add_image(image_def, insert=(0, 0), size_in_units=(d.width * s, d.height * s),
                      dxfattribs={"layer": "P2C-UNDERLAY"})
    return doc


def _symbols(doc: ezdxf.document.Drawing, d: Drawing, pt, s: float) -> None:
    """Symbols as references to the blocks of their symbol libraries, copied into the document unchanged (with their
    layers, colours and text styles), with their attributes where their texts stood. A block is drawn in its
    library's units at paper size, so the reference is scaled to paper millimetres and by the drawing scale."""
    used: dict = {}  # library -> names of its blocks in the drawing
    for sym in d.symbols:
        used.setdefault(sym.template.library, set()).add(sym.name)
    for lib, names in used.items():
        loader = xref.Loader(lib.doc, doc)
        for name in sorted(names):
            loader.load_block_layout(lib.doc.blocks.get(name))
        loader.execute()
    msp = doc.modelspace()
    for sym in d.symbols:  # y flips: clockwise on screen is a negative DXF rotation
        size = sym.scale * s * sym.template.library.unit
        ref = msp.add_blockref(sym.name, pt(sym.x, sym.y), dxfattribs={
            "layer": "P2C-SYMBOL", "rotation": -sym.angle % 360, "yscale": size,
            "xscale": size * (-1 if sym.mirror else 1)})
        for tag, t in sym.attributes.items():
            ref.add_attrib(tag, t.text, dxfattribs={"height": t.height * s, "layer": "P2C-TEXT"}).set_placement(
                pt(t.x, t.y), align=TextEntityAlignment.MIDDLE_CENTER)


def to_figure(d: Drawing, path, fmt: str = "pdf", dpi: float | None = None) -> None:
    """Plots the drawing true to scale on a sheet of its paper size (vector PDF or PNG preview)."""
    d = d.flat()
    fig = Figure(figsize=(d.width / 25.4, d.height / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, d.width)
    ax.set_ylim(d.height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    segments, widths = [], []
    for ln in d.lines:
        parts = dashes(ln) if ln.dashed else [((ln.x1, ln.y1), (ln.x2, ln.y2))]
        segments += parts
        widths += [ln.width * PT_PER_MM] * len(parts)
    ax.add_collection(LineCollection(segments, linewidths=widths, colors="k", capstyle="round"))
    for a in d.arcs:
        end = a.end if a.end > a.start else a.end + 360
        ax.add_patch(MplArc((a.cx, a.cy), 2 * a.r, 2 * a.r, theta1=a.start, theta2=end, lw=a.width * PT_PER_MM))
    for c in d.circles:
        ax.add_patch(MplCircle((c.cx, c.cy), c.r, fill=False, lw=c.width * PT_PER_MM))
    for g in d.polygons:
        ax.add_patch(MplPolygon(g.points, closed=True, fill=g.filled, color="k", lw=g.width * PT_PER_MM))
    for t in d.texts:
        rad = math.radians(t.angle)  # anchor on the baseline, half a cap height below the centre
        ax.text(t.x + math.sin(rad) * t.height / 2, t.y + math.cos(rad) * t.height / 2, t.text,
                fontsize=t.height / CAP * PT_PER_MM, rotation=t.angle, rotation_mode="anchor",
                ha="center", va="baseline", color="0.55" if t.uncertain else "k")
    fig.savefig(path, format=fmt, dpi=dpi, facecolor="white")


OVERLAY_COLORS = {"lines": (40, 40, 220), "dashed": (200, 90, 0), "curves": (0, 150, 0),
                  "polygons": (180, 0, 180), "texts": (0, 120, 255), "unsure": (200, 0, 200), "symbols": (150, 110, 0)}


def overlay(gray: np.ndarray, d: Drawing, ppm: float, max_side: int = 4000) -> np.ndarray:
    """Faded rectified photo with the recognised vectors drawn on top in colour (BGR)."""
    symbols = Drawing(d.width, d.height, symbols=d.symbols).flat()
    d = replace(d, symbols=[]).flat()
    f = min(1.0, max_side / max(gray.shape))
    base = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor((base * 0.3 + 178).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    parts = {
        "lines": Drawing(d.width, d.height, lines=[ln for ln in d.lines if not ln.dashed]),
        "dashed": Drawing(d.width, d.height, lines=[ln for ln in d.lines if ln.dashed]),
        "curves": Drawing(d.width, d.height, arcs=d.arcs, circles=d.circles),
        "polygons": Drawing(d.width, d.height, polygons=d.polygons),
        "texts": Drawing(d.width, d.height, texts=[t for t in d.texts if not t.uncertain]),
        "unsure": Drawing(d.width, d.height, texts=[t for t in d.texts if t.uncertain]),
        "symbols": symbols,
    }
    for key, part in parts.items():
        render(part, ppm * f, img, OVERLAY_COLORS[key])
    return img
