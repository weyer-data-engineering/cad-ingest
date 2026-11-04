"""Vector primitives in sheet millimetres: origin top left, y pointing down (like the image)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

PAPER = {"A0": (1189, 841), "A1": (841, 594), "A2": (594, 420), "A3": (420, 297), "A4": (297, 210)}  # landscape mm
Color = tuple[int, int, int]  # RGB, 0 to 255; None draws in the colour of the layer
LINETYPES = ("dashed", "dashdot", "dotted")  # of lines that are not continuous (``dashed``)
# Entities read from a vector PDF keep the CAD layer they were drawn on (``layer``), where the PDF names it.


@dataclass
class Line:
    x1: float
    y1: float
    x2: float
    y2: float
    width: float = 0.25  # pen width on paper in mm
    dashed: bool = False  # not continuous: dashed, dash-dot or dotted, see linetype
    color: Color | None = None
    layer: str | None = None
    linetype: str | None = None  # of a dashed line, one of LINETYPES; None is "dashed"
    pattern: float | None = None  # mm on paper: the length of one repeat of its dashes and gaps, where measured


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
    linetype: str | None = None  # as for Line
    pattern: float | None = None
    smooth: bool = False  # a free curve through its points (a spline in the DXF); its bulges are 0

    def parts(self) -> list[Line | Arc]:
        pts = self.points + self.points[:1] if self.closed else self.points
        out: list[Line | Arc] = []
        for (x1, y1), (x2, y2), b in zip(pts, pts[1:], self.bulges):
            chord = math.hypot(x2 - x1, y2 - y1)
            if b == 0 or chord == 0:
                out.append(Line(x1, y1, x2, y2, self.width, self.dashed, self.color, self.layer, self.linetype,
                                self.pattern))
                continue
            sweep = 4 * math.atan(abs(b))
            r = chord / (2 * math.sin(sweep / 2))
            h = r * math.cos(sweep / 2) * (1 if b > 0 else -1) / chord  # centre offset along the left normal
            cx, cy = (x1 + x2) / 2 - (y2 - y1) * h, (y1 + y2) / 2 + (x2 - x1) * h
            a1, a2 = (math.degrees(math.atan2(y - cy, x - cx)) % 360 for x, y in ((x1, y1), (x2, y2)))
            out.append(Arc(cx, cy, r, *((a1, a2) if b > 0 else (a2, a1)), self.width, self.color, self.layer))
        return out


def line_type(q) -> str:
    """The line type of a line or polyline: "continuous" or one of LINETYPES."""
    return (q.linetype or "dashed") if getattr(q, "dashed", False) else "continuous"


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
    length: float | None = None  # mm along the reading direction as printed, where measured: DXF fits the text to it
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
    confidence: float = 100.0  # share of its outline found in the scan, in percent
    doubt: str | None = None  # why to check it, as for an uncertain text: a symbol that may be the wrong one


@dataclass
class Dimension:
    """A linear dimension: the distance between the ends of its dimension line, (x1, y1) and (x2, y2), where its
    terminators sit. Its extension lines run from the measured points ``p1`` and ``p2`` to the dimension line and
    ``extension`` mm past it; a measured point at the end of the dimension line has none of its own (the line ends on
    the drawing's linework). ``value`` is its text as read, "" where none is written. A dimension recognised on a sheet
    keeps the strokes it was found from (``parts``) and its text (``text``), as printed, so that a preview draws it as
    printed; a reference dimension of the benchmark has neither, as its strokes and text are the drawing's own."""

    x1: float
    y1: float
    x2: float
    y2: float
    value: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    terminator: str = "tick"  # "tick" (an oblique stroke), "arrow" or "dot"
    size: float = 2.0  # mm: the tick's extent along the line, the arrowhead's length or the dot's width
    extension: float = 2.0
    width: float = 0.25  # pen width of the dimension line
    text: Text | None = None
    parts: list[Line | Arc | Circle | Polygon] = field(default_factory=list)
    doubt: str | None = None  # why to check it: its value is unsure, missing or does not fit its length

    @property
    def uncertain(self) -> bool:
        return self.doubt is not None


@dataclass
class Markup:
    """Something added by hand on a print, in colour (see markup) or in black or pencil (see freehand): a pen note,
    cloud or strike-out ("note"), with what was read of its handwriting if that was asked for and who read it
    ("tesseract", or "reviewer": the job's vision model), or a highlighted area ("highlight"); with the printed texts
    it covers."""

    kind: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in sheet mm
    color: Color
    reading: str | None = None
    texts: list[str] = field(default_factory=list)
    read_by: str | None = None


@dataclass
class Unclassified:
    """Linework that looks like a symbol but is none of the libraries' (see unclassified.py). It stays linework, on a
    layer of its own; ``box`` is x0, y0, x1, y1 in sheet mm, ``why`` says what makes it look like a symbol."""

    box: tuple[float, float, float, float]
    why: str


@dataclass
class Cell:
    """A cell of a title block: its first row and column in the grid of the block's rulings (from 0, top left), how
    many rows and columns it spans, its box and the texts in it in reading order (``uncertain`` if one of them is)."""

    row: int
    col: int
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in sheet mm
    text: str = ""
    rows: int = 1
    cols: int = 1
    uncertain: bool = False


@dataclass
class TitleBlock:
    """The title block as a table (see titleblock): its box and its cells."""

    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in sheet mm
    cells: list[Cell] = field(default_factory=list)

    def scaled(self, f: float) -> TitleBlock:
        """The same title block ``f`` times as large."""
        return TitleBlock(tuple(v * f for v in self.box),
                          [replace(c, box=tuple(v * f for v in c.box)) for c in self.cells])


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
    printed: float = 1.0  # the sheet shows the drawing at this share of its drawn size (an A1 plan printed on A3: 0.5)
    dimensions: list[Dimension] = field(default_factory=list)
    unclassified: list[Unclassified] = field(default_factory=list)  # their linework is on its own layer
    title_block: TitleBlock | None = None  # its linework is on its own layer (titleblock.LAYER)

    def scaled(self, f: float) -> Drawing:
        """The same drawing ``f`` times as large: positions, sizes, pen widths and texts (markups excepted)."""
        def at(q):
            if isinstance(q, Line):
                return replace(q, x1=q.x1 * f, y1=q.y1 * f, x2=q.x2 * f, y2=q.y2 * f, width=q.width * f,
                               pattern=q.pattern and q.pattern * f)
            if isinstance(q, (Arc, Circle)):
                return replace(q, cx=q.cx * f, cy=q.cy * f, r=q.r * f, width=q.width * f)
            if isinstance(q, Text):
                return replace(q, x=q.x * f, y=q.y * f, height=q.height * f, length=q.length and q.length * f)
            if isinstance(q, Polyline):
                return replace(q, points=[(x * f, y * f) for x, y in q.points], width=q.width * f,
                               pattern=q.pattern and q.pattern * f)
            return replace(q, points=[(x * f, y * f) for x, y in q.points], width=q.width * f)  # polygon

        symbols = [replace(s, x=s.x * f, y=s.y * f, scale=s.scale * f, parts=[at(q) for q in s.parts],
                           ports=[(x * f, y * f) for x, y in s.ports],
                           attributes={k: at(t) for k, t in s.attributes.items()}) for s in self.symbols]
        dimensions = [replace(m, x1=m.x1 * f, y1=m.y1 * f, x2=m.x2 * f, y2=m.y2 * f, p1=(m.p1[0] * f, m.p1[1] * f),
                              p2=(m.p2[0] * f, m.p2[1] * f), size=m.size * f, extension=m.extension * f,
                              width=m.width * f, text=m.text and at(m.text), parts=[at(q) for q in m.parts])
                      for m in self.dimensions]
        return replace(self, width=self.width * f, height=self.height * f, lines=[at(q) for q in self.lines],
                       arcs=[at(q) for q in self.arcs], circles=[at(q) for q in self.circles],
                       polygons=[at(q) for q in self.polygons], texts=[at(q) for q in self.texts],
                       polylines=[at(q) for q in self.polylines], symbols=symbols, dimensions=dimensions,
                       unclassified=[replace(u, box=tuple(v * f for v in u.box)) for u in self.unclassified],
                       title_block=self.title_block and self.title_block.scaled(f))

    def counts(self) -> dict[str, int]:
        """How many elements of each kind the drawing has; the values of its dimensions are texts of the sheet too."""
        out = {k: len(getattr(self, k))
               for k in ("lines", "arcs", "circles", "polygons", "polylines", "symbols", "texts", "dimensions")}
        out["texts"] += sum(m.text is not None for m in self.dimensions)
        return out

    def flat(self) -> Drawing:
        """The same drawing with its polylines split into lines and arcs, its symbols into their parts and attribute
        texts and its dimensions into their strokes and texts (for drawing it)."""
        parts = [p for pl in self.polylines for p in pl.parts()] + [p for s in self.symbols for p in s.parts]
        parts += [p for m in self.dimensions for p in m.parts]
        return replace(self, lines=self.lines + [p for p in parts if isinstance(p, Line)],
                       arcs=self.arcs + [p for p in parts if isinstance(p, Arc)],
                       circles=self.circles + [p for p in parts if isinstance(p, Circle)],
                       polygons=self.polygons + [p for p in parts if isinstance(p, Polygon)],
                       texts=self.texts + [t for s in self.symbols for t in s.attributes.values()]
                       + [m.text for m in self.dimensions if m.text],
                       polylines=[], symbols=[], dimensions=[])
