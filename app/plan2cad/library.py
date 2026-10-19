"""Symbol library: the CAD blocks recognised symbols become (valves, pumps, instruments, ...).

A library is a DXF file; the starter library symbols.dxf is written by bench/make_library.py. Each block is one symbol
in paper millimetres around its insertion point, drawn with lines, arcs, circles and polylines; POINTs on the layer PORT
are where pipes connect, LINEs on the layer THROUGH a line that runs through the symbol (the pipe under a flow arrow:
required for a match, not part of the symbol), ATTDEFs its attributes (e.g. TAG), placed where their text stands."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import ezdxf

from .model import Arc, Circle, Line, Symbol

DEFAULT = Path(__file__).with_name("symbols.dxf")
PORT_LAYER, THROUGH_LAYER = "PORT", "THROUGH"


@dataclass
class Template:
    """A library symbol in its own coordinates: millimetres, y pointing down like the sheet, origin = insertion point."""

    name: str
    description: str
    parts: list[Line | Arc | Circle]
    groups: list[int]  # the drawing entity (polyline, circle, ...) each part comes from
    ports: list[tuple[float, float]]
    tags: dict[str, tuple[float, float]]  # attribute tag -> where its text stands
    through: list[Line]  # lines that run through the symbol in the drawing


@lru_cache
def document(path: Path = DEFAULT) -> ezdxf.document.Drawing:
    return ezdxf.readfile(path)


@lru_cache
def load(path: Path = DEFAULT) -> dict[str, Template]:
    """The symbols of the library at ``path`` by block name."""
    out = {}
    for block in document(path).blocks:
        if block.name.startswith("*"):  # model and paper space
            continue
        parts, groups, ports, tags, through = [], [], [], {}, []
        for e in block:
            kind = e.dxftype()
            if kind == "POINT" and e.dxf.layer.upper() == PORT_LAYER:
                ports.append(_down(e.dxf.location))
            elif kind == "LINE" and e.dxf.layer.upper() == THROUGH_LAYER:
                through.append(_part(e, e))
            elif kind == "ATTDEF":
                aligned = e.dxf.halign or e.dxf.valign
                tags[e.dxf.tag] = _down(e.dxf.align_point if aligned else e.dxf.insert)
            elif kind in ("LWPOLYLINE", "POLYLINE", "LINE", "ARC", "CIRCLE"):
                pieces = list(e.virtual_entities()) if kind.endswith("POLYLINE") else [e]
                parts += [_part(v, e) for v in pieces]
                groups += [len(set(groups))] * len(pieces)
        if parts:
            out[block.name] = Template(block.name, block.block.dxf.get("description", "") or block.name, parts, groups,
                                       ports, tags, through)
    return out


def _down(p) -> tuple[float, float]:
    return float(p[0]), 0.0 - float(p[1])  # 0.0 - avoids -0.0


def _part(e, owner) -> Line | Arc | Circle:
    """A DXF line, arc or circle in sheet orientation (y down: counter-clockwise DXF angles become clockwise ones)."""
    lw = owner.dxf.get("lineweight", -1)
    width = lw / 100 if lw > 0 else 0.35
    kind = e.dxftype()
    if kind == "LINE":
        return Line(*_down(e.dxf.start), *_down(e.dxf.end), width)
    if kind == "CIRCLE":
        return Circle(*_down(e.dxf.center), e.dxf.radius, width)
    return Arc(*_down(e.dxf.center), e.dxf.radius, (-e.dxf.end_angle) % 360, (-e.dxf.start_angle) % 360, width)


def transform(x: float, y: float, angle: float = 0.0, mirror: bool = False, scale: float = 1.0):
    """Maps symbol coordinates to the sheet: mirror across the symbol's vertical axis, scale, turn clockwise on screen
    by ``angle`` degrees, move the origin to (x, y)."""
    c, s, m = math.cos(math.radians(angle)) * scale, math.sin(math.radians(angle)) * scale, -1 if mirror else 1
    return lambda u, v: (x + c * m * u - s * v, y + s * m * u + c * v)


def place(t: Template, x: float, y: float, angle: float = 0.0, mirror: bool = False, scale: float = 1.0) -> Symbol:
    """The symbol ``t`` placed on the sheet."""
    f = transform(x, y, angle, mirror, scale)

    def moved(q):
        if isinstance(q, Line):
            return Line(*f(q.x1, q.y1), *f(q.x2, q.y2), q.width)
        if isinstance(q, Circle):
            return Circle(*f(q.cx, q.cy), q.r * scale, q.width)
        start, end = (180 - q.end, 180 - q.start) if mirror else (q.start, q.end)  # mirroring reverses the direction
        return Arc(*f(q.cx, q.cy), q.r * scale, (start + angle) % 360, (end + angle) % 360, q.width)

    return Symbol(t.name, x, y, angle % 360, mirror, scale, [moved(q) for q in t.parts], [f(*p) for p in t.ports])
