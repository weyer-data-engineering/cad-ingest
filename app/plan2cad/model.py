"""Vector primitives in sheet millimetres: origin top left, y pointing down (like the image)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

PAPER = {"A0": (1189, 841), "A1": (841, 594), "A2": (594, 420), "A3": (420, 297), "A4": (297, 210)}  # landscape mm
Color = tuple[int, int, int]  # RGB, 0 to 255; None draws in the colour of the layer
# Entities read from a vector PDF keep the CAD layer they were drawn on (``layer``), where the PDF names it.


@dataclass
class Line:
    x1: float
    y1: float
    x2: float
    y2: float
    width: float = 0.25  # pen width on paper in mm
    dashed: bool = False
    color: Color | None = None
    layer: str | None = None


@dataclass
class Arc:
    """Arc from ``start`` to ``end`` in degrees, measured clockwise on screen (y down)."""

    cx: float
    cy: float
    r: float
    start: float
    end: float
    width: float = 0.25
    color: Color | None = None
    layer: str | None = None


@dataclass
class Circle:
    cx: float
    cy: float
    r: float
    width: float = 0.25
    color: Color | None = None
    layer: str | None = None


@dataclass
class Polygon:
    points: list[tuple[float, float]]
    filled: bool = True
    width: float = 0.25
    color: Color | None = None
    layer: str | None = None


@dataclass
class Polyline:
    """Lines and arcs joined end to end into one object (a wall outline, a pipe run). ``bulges``
    holds one value per segment: 0 for a straight one, otherwise tan(sweep / 4) as in DXF,
    positive when the arc turns clockwise on screen."""

    points: list[tuple[float, float]]
    bulges: list[float]
    closed: bool = False
    width: float = 0.25
    dashed: bool = False
    kind: str = "line"  # what it stands for: "line", "wall" or "pipe"
    color: Color | None = None
    layer: str | None = None

    def parts(self) -> list[Line | Arc]:
        pts = self.points + self.points[:1] if self.closed else self.points
        out: list[Line | Arc] = []
        for (x1, y1), (x2, y2), b in zip(pts, pts[1:], self.bulges):
            chord = math.hypot(x2 - x1, y2 - y1)
            if b == 0 or chord == 0:
                out.append(Line(x1, y1, x2, y2, self.width, self.dashed, self.color, self.layer))
                continue
            sweep = 4 * math.atan(abs(b))
            r = chord / (2 * math.sin(sweep / 2))
            h = r * math.cos(sweep / 2) * (1 if b > 0 else -1) / chord  # centre offset along the left normal
            cx, cy = (x1 + x2) / 2 - (y2 - y1) * h, (y1 + y2) / 2 + (x2 - x1) * h
            a1, a2 = (math.degrees(math.atan2(y - cy, x - cx)) % 360 for x, y in ((x1, y1), (x2, y2)))
            out.append(Arc(cx, cy, r, *((a1, a2) if b > 0 else (a2, a1)), self.width, self.color, self.layer))
        return out


@dataclass
class Text:
    """Single text line; (x, y) is the centre of its capitals, angle 90 reads bottom to top."""

    x: float
    y: float
    height: float  # cap height in mm
    text: str
    angle: float = 0.0
    confidence: float = 100.0
    uncertain: bool = False  # looks like text but could not be read reliably: a guess to check
    length: float | None = None  # mm along the reading direction, where known exactly (vector PDFs): DXF fits to it
    oblique: float = 0.0  # degrees the letters lean forward (italics)
    color: Color | None = None
    layer: str | None = None


@dataclass
class Symbol:
    """A block of the symbol library placed on the sheet: turned by ``angle`` degrees (clockwise on screen) after being
    mirrored across its own vertical axis if ``mirror``. ``parts`` and ``ports`` are already in sheet millimetres."""

    name: str
    x: float
    y: float
    angle: float = 0.0
    mirror: bool = False
    scale: float = 1.0
    parts: list[Line | Arc | Circle] = field(default_factory=list)
    ports: list[tuple[float, float]] = field(default_factory=list)
    attributes: dict[str, Text] = field(default_factory=dict)  # attribute tag -> its text on the sheet
    template: Any = field(default=None, repr=False, compare=False)  # the library.Template it was placed from


@dataclass
class Markup:
    """Something added by hand in colour on a print (see markup): a pen note, cloud or strike-out ("note"), with
    what could be read of its handwriting, or a highlighted area ("highlight"); with the printed texts it covers."""

    kind: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in sheet mm
    color: Color
    reading: str | None = None
    texts: list[str] = field(default_factory=list)


@dataclass
class Drawing:
    width: float  # sheet size in mm
    height: float
    lines: list[Line] = field(default_factory=list)
    arcs: list[Arc] = field(default_factory=list)
    circles: list[Circle] = field(default_factory=list)
    polygons: list[Polygon] = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)
    scale: int | None = None  # denominator of the drawing scale, e.g. 100 for 1:100
    polylines: list[Polyline] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    exact: bool = False  # read from the vectors of a PDF, not recognised on an image: widths and colours are as drawn
    markups: list[Markup] = field(default_factory=list)  # their strokes, fills and readings are on their own layers

    def counts(self) -> dict[str, int]:
        return {k: len(getattr(self, k))
                for k in ("lines", "arcs", "circles", "polygons", "polylines", "symbols", "texts")}

    def flat(self) -> Drawing:
        """The same drawing with its polylines split into lines and arcs and its symbols into their parts and
        attribute texts (for drawing it)."""
        parts = [p for pl in self.polylines for p in pl.parts()] + [p for s in self.symbols for p in s.parts]
        return replace(self, lines=self.lines + [p for p in parts if isinstance(p, Line)],
                       arcs=self.arcs + [p for p in parts if isinstance(p, Arc)],
                       circles=self.circles + [p for p in parts if isinstance(p, Circle)],
                       texts=self.texts + [t for s in self.symbols for t in s.attributes.values()],
                       polylines=[], symbols=[])
