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

from .markup import HIGHLIGHT_LAYER, LAYER, READING_LAYER
from .model import Drawing
from .render import CAP, DASH, GAP, dashes, render

LAYERS = {"P2C-LINE": 7, "P2C-DASH": 5, "P2C-CURVE": 3, "P2C-FILL": 8, "P2C-TEXT": 1, "P2C-UNDERLAY": 9,
          "P2C-WALL": 150, "P2C-DOOR": 30, "P2C-PIPE": 150, "P2C-EQUIPMENT": 94, "P2C-SIGNAL": 5, "P2C-FRAME": 8,
          "P2C-TEXT-CHECK": 6, "P2C-SYMBOL": 4, "P2C-SYMBOL-CHECK": 6, LAYER: 1, HIGHLIGHT_LAYER: 3, READING_LAYER: 1}
KIND_LAYERS = {"wall": "P2C-WALL", "door": "P2C-DOOR", "pipe": "P2C-PIPE", "equipment": "P2C-EQUIPMENT",
               "signal": "P2C-SIGNAL", "frame": "P2C-FRAME"}
LINEWEIGHTS = (13, 18, 25, 35, 50, 70, 100, 140, 200)  # ISO pen widths in 1/100 mm
DXF_LINEWEIGHTS = (0, 5, 9, 13, 15, 18, 20, 25, 30, 35, 40, 50, 53, 60, 70, 80, 90, 100, 106, 120, 140, 158, 200, 211)
PT_PER_MM = 72 / 25.4


def lineweight(width_mm: float, exact: bool = False) -> int:
    """The pen for a width: the nearest ISO pen for a measured width, the nearest DXF lineweight for an exact one."""
    return min(DXF_LINEWEIGHTS if exact else LINEWEIGHTS, key=lambda lw: abs(lw - width_mm * 100))


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

    def on(layer: str, item) -> dict:
        """The layer (the item's own CAD layer, if it has one) and colour of an item."""
        if item.layer and item.layer not in doc.layers:
            doc.layers.add(item.layer)
        attribs = {"layer": item.layer or layer}
        if item.color:
            attribs["true_color"] = ezdxf.colors.rgb2int(item.color)
        return attribs

    def pen(layer: str, item) -> dict:
        return on(layer, item) | {"lineweight": lineweight(item.width, d.exact)}

    for ln in d.lines:
        attribs = pen("P2C-DASH" if ln.dashed else "P2C-LINE", ln)
        if ln.dashed:
            attribs["linetype"] = "P2C_DASH"
        msp.add_line(pt(ln.x1, ln.y1), pt(ln.x2, ln.y2), dxfattribs=attribs)
    for pl in d.polylines:  # y points up in DXF: an arc turning clockwise on screen has a negative bulge
        attribs = pen(KIND_LAYERS.get(pl.kind) or ("P2C-DASH" if pl.dashed else "P2C-LINE"), pl)
        if pl.dashed:
            attribs["linetype"] = "P2C_DASH"
        e = msp.add_lwpolyline([(*pt(x, y), -b) for (x, y), b in zip(pl.points, pl.bulges + [0.0])], format="xyb",
                               close=pl.closed, dxfattribs=attribs)
        e.dxf.flags |= 128  # dash pattern runs on around corners
    for a in d.arcs:  # y flips, so clockwise screen angles become counter-clockwise DXF angles
        msp.add_arc(pt(a.cx, a.cy), a.r * s, (-a.end) % 360, (-a.start) % 360, dxfattribs=pen("P2C-CURVE", a))
    for c in d.circles:
        msp.add_circle(pt(c.cx, c.cy), c.r * s, dxfattribs=pen("P2C-CURVE", c))
    for g in d.polygons:
        points = [pt(x, y) for x, y in g.points]
        if g.filled:
            hatch = msp.add_hatch(color=256, dxfattribs=on("P2C-FILL", g))
            hatch.paths.add_polyline_path(points, is_closed=True)
            if g.layer == HIGHLIGHT_LAYER:  # the print shows through highlighter
                hatch.transparency = 0.6
        else:
            msp.add_lwpolyline(points, close=True, dxfattribs=pen("P2C-LINE", g))
    for t in d.texts:
        attribs = on("P2C-TEXT-CHECK" if t.uncertain else "P2C-TEXT", t)
        if t.oblique:
            attribs["oblique"] = t.oblique
        text = msp.add_text(t.text, height=t.height * s, rotation=t.angle, dxfattribs=attribs)
        if t.length:  # stretched between the ends of its baseline, so it is as long as drawn in any font
            a = math.radians(t.angle)
            bx, by = t.x + math.sin(a) * t.height / 2, t.y + math.cos(a) * t.height / 2  # middle of the baseline
            ux, uy = math.cos(a) * t.length / 2, -math.sin(a) * t.length / 2
            text.set_placement(pt(bx - ux, by - uy), pt(bx + ux, by + uy), align=TextEntityAlignment.FIT)
        else:
            text.set_placement(pt(t.x, t.y), align=TextEntityAlignment.MIDDLE_CENTER)
    _symbols(doc, d, pt, s)
    if underlay:
        name, size = underlay
        image_def = doc.add_image_def(filename=name, size_in_pixel=size)
        msp.add_image(image_def, insert=(0, 0), size_in_units=(d.width * s, d.height * s),
                      dxfattribs={"layer": "P2C-UNDERLAY"})
    return doc


def _symbols(doc: ezdxf.document.Drawing, d: Drawing, pt, s: float) -> None:
    """Symbols as references to the blocks of their symbol libraries, copied into the document unchanged (with their
    layers, colours and text styles), with their attributes where their texts stood. The reference scales the block
    from its library's units to sheet millimetres (and real-size blocks by the drawing scale), then by the drawing
    scale to real-world millimetres."""
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
        size = sym.scale * s * sym.template.unit
        ref = msp.add_blockref(sym.name, pt(sym.x, sym.y), dxfattribs={
            "layer": "P2C-SYMBOL-CHECK" if sym.doubt else "P2C-SYMBOL", "rotation": -sym.angle % 360, "yscale": size,
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
    segments, widths, colors = [], [], []
    for ln in d.lines:
        parts = dashes(ln) if ln.dashed else [((ln.x1, ln.y1), (ln.x2, ln.y2))]
        segments += parts
        widths += [ln.width * PT_PER_MM] * len(parts)
        colors += [_rgb(ln.color)] * len(parts)
    ax.add_collection(LineCollection(segments, linewidths=widths, colors=colors or "k", capstyle="round"))
    for a in d.arcs:
        end = a.end if a.end > a.start else a.end + 360
        ax.add_patch(MplArc((a.cx, a.cy), 2 * a.r, 2 * a.r, theta1=a.start, theta2=end, lw=a.width * PT_PER_MM,
                            color=_rgb(a.color)))
    for c in d.circles:
        ax.add_patch(MplCircle((c.cx, c.cy), c.r, fill=False, lw=c.width * PT_PER_MM, color=_rgb(c.color)))
    for g in d.polygons:
        ax.add_patch(MplPolygon(g.points, closed=True, fill=g.filled, color=_rgb(g.color), lw=g.width * PT_PER_MM,
                                alpha=0.4 if g.layer == HIGHLIGHT_LAYER else None))
    for t in d.texts:
        rad = math.radians(t.angle)  # anchor on the baseline, half a cap height below the centre
        ax.text(t.x + math.sin(rad) * t.height / 2, t.y + math.cos(rad) * t.height / 2, t.text,
                fontsize=t.height / CAP * PT_PER_MM, rotation=t.angle, rotation_mode="anchor",
                ha="center", va="baseline",
                color=_rgb(t.color) if t.color else "0.55" if t.uncertain else "k")
    fig.savefig(path, format=fmt, dpi=dpi, facecolor="white")


def _rgb(color) -> tuple[float, float, float]:
    return tuple(v / 255 for v in color) if color else (0.0, 0.0, 0.0)


OVERLAY_COLORS = {"lines": (40, 40, 220), "dashed": (200, 90, 0), "curves": (0, 150, 0),
                  "polygons": (180, 0, 180), "texts": (0, 120, 255), "unsure": (200, 0, 200), "symbols": (150, 110, 0),
                  "doubtful": (200, 0, 200)}


def overlay(gray: np.ndarray, d: Drawing, ppm: float, max_side: int = 4000) -> np.ndarray:
    """Faded rectified photo with the recognised vectors drawn on top in colour (BGR)."""
    symbols = Drawing(d.width, d.height, symbols=[s for s in d.symbols if not s.doubt]).flat()
    doubtful = Drawing(d.width, d.height, symbols=[s for s in d.symbols if s.doubt]).flat()
    d = replace(d, symbols=[]).flat()
    marked = (LAYER, HIGHLIGHT_LAYER, READING_LAYER)
    marks = Drawing(d.width, d.height, [ln for ln in d.lines if ln.layer in marked],
                    polygons=[g for g in d.polygons if g.layer in marked],
                    texts=[t for t in d.texts if t.layer in marked])
    d = replace(d, lines=[ln for ln in d.lines if ln.layer not in marked],
                polygons=[g for g in d.polygons if g.layer not in marked],
                texts=[t for t in d.texts if t.layer not in marked])
    f = min(1.0, max_side / max(gray.shape))
    base = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor((base * 0.3 + 178).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for color in {item.color for item in marks.lines + marks.polygons + marks.texts}:  # markups in their colour
        r, g, b = color or (255, 0, 0)
        part = Drawing(d.width, d.height, [ln for ln in marks.lines if ln.color == color],
                       polygons=[p for p in marks.polygons if p.color == color],
                       texts=[t for t in marks.texts if t.color == color])
        layer = render(part, ppm * f, img.copy(), (b, g, r))
        img = cv2.addWeighted(img, 0.5, layer, 0.5, 0) if part.polygons and not part.lines else layer
    parts = {
        "lines": Drawing(d.width, d.height, lines=[ln for ln in d.lines if not ln.dashed]),
        "dashed": Drawing(d.width, d.height, lines=[ln for ln in d.lines if ln.dashed]),
        "curves": Drawing(d.width, d.height, arcs=d.arcs, circles=d.circles),
        "polygons": Drawing(d.width, d.height, polygons=d.polygons),
        "texts": Drawing(d.width, d.height, texts=[t for t in d.texts if not t.uncertain]),
        "unsure": Drawing(d.width, d.height, texts=[t for t in d.texts if t.uncertain]),
        "symbols": symbols,
        "doubtful": doubtful,  # in the colour of the uncertain texts: to check
    }
    for key, part in parts.items():
        render(part, ppm * f, img, OVERLAY_COLORS[key])
    return img
