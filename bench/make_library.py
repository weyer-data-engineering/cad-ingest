"""Writes the DXFs of the built-in symbol libraries:

- libraries/din-10628: simplified P&ID symbols after DIN EN ISO 10628-2, at their printed size;
- libraries/floor-plan: fixtures of floor plans (sanitary objects, kitchen), at their real size in millimetres.

    python bench/make_library.py

Each block is one symbol around its insertion point (y up, as in any DXF); POINTs on the layer PORT are its
connection points, LINEs on the layer THROUGH a line that runs through the symbol (the pipe under a flow arrow),
ATTDEFs its attributes, placed where their text stands. A fixture has its insertion point in the middle of its back,
the side that stands against the wall, and reaches forward (down). Company libraries in the same form are added next
to them (docs/symbol-libraries.md).
"""

from __future__ import annotations

from pathlib import Path

import ezdxf
from ezdxf.enums import TextEntityAlignment

BOWTIE = [(-2.5, -1.6), (-2.5, 1.6), (2.5, -1.6), (2.5, 1.6), (-2.5, -1.6)]
LEFT = [(-2.5, -1.6), (-2.5, 1.6), (0, 0), (-2.5, -1.6)]
RIGHT = [(2.5, -1.6), (2.5, 1.6), (0, 0), (2.5, -1.6)]
DOWN = [(-1.6, -2.5), (1.6, -2.5), (0, 0), (-1.6, -2.5)]
INLINE = [(-2.5, 0), (2.5, 0)]
INSTRUMENT = {"FUNCTION": (0, 2.3), "NUMBER": (0, -2.4)}

# name: description, polylines, circles (x, y, r), arcs (x, y, r, start, end counter-clockwise), ports, attributes
SYMBOLS = {
    "GATE_VALVE": ("Gate valve", [BOWTIE], [], [], INLINE, {}),
    "BALL_VALVE": ("Ball valve", [BOWTIE], [(0, 0, 1.1)], [], INLINE, {}),
    "BUTTERFLY_VALVE": ("Butterfly valve", [BOWTIE, [(0, -1.6), (0, 1.6)]], [], [], INLINE, {}),
    "CHECK_VALVE": ("Check valve", [[(-2.5, -1.6), (-2.5, 1.6), (2.5, 0), (-2.5, -1.6)], [(2.5, -1.6), (2.5, 1.6)]],
                    [], [], INLINE, {}),
    "ANGLE_VALVE": ("Angle valve", [LEFT, DOWN], [], [], [(-2.5, 0), (0, -2.5)], {}),
    "THREE_WAY_VALVE": ("Three-way valve", [BOWTIE, DOWN], [], [], INLINE + [(0, -2.5)], {}),
    "CONTROL_VALVE": ("Control valve", [BOWTIE, [(0, 0), (0, 6)], [(-3.5, 6), (3.5, 6)]], [], [(0, 6, 3.5, 0, 180)],
                      INLINE, {}),
    "SAFETY_VALVE": ("Safety valve", [DOWN, RIGHT, [(0, 0), (0, 1.2), (-1.2, 1.7), (1.2, 2.5), (-1.2, 3.3),
                                                     (1.2, 4.1), (0, 4.6)]], [], [], [(0, -2.5), (2.5, 0)], {}),
    "PUMP": ("Centrifugal pump", [[(-4.2, -5), (-6.5, -9.5), (6.5, -9.5), (4.2, -5)]], [(0, 0, 6.5)], [],
             [(-6.5, 0), (0, 6.5)], {"TAG": (0, -15)}),
    "COMPRESSOR": ("Compressor", [[(-4.5, 4.7), (4.5, 2)], [(-4.5, -4.7), (4.5, -2)]], [(0, 0, 6.5)], [],
                   [(-6.5, 0), (6.5, 0)], {"TAG": (0, -10.5)}),
    "HEAT_EXCHANGER": ("Heat exchanger", [[(-6, -6.7), (-2, 3), (2, -3), (6, 6.7)]], [(0, 0, 9)], [],
                       [(-9, 0), (9, 0), (-6, -6.7), (6, 6.7)], {"TAG": (13, -11)}),
    "INSTRUMENT": ("Instrument, local", [], [(0, 0, 5)], [], [], INSTRUMENT),
    "INSTRUMENT_DCS": ("Instrument, control room", [[(-5, 0), (5, 0)]], [(0, 0, 5)], [], [], INSTRUMENT),
    "REDUCER": ("Reducer", [[(-2, -2), (-2, 2), (2, 1.2), (2, -1.2), (-2, -2)]], [], [], [(-2, 0), (2, 0)], {}),
    "FILTER": ("Filter", [[(-3, 0), (0, 3), (3, 0), (0, -3), (-3, 0)], [(0, -3), (0, 3)]], [], [],
               [(-3, 0), (3, 0)], {}),
    "ORIFICE": ("Orifice plate", [[(-0.6, -2.5), (-0.6, 2.5)], [(0.6, -2.5), (0.6, 2.5)]], [], [],
                [(-0.6, 0), (0.6, 0)], {}),
    "OFF_PAGE_CONNECTOR": ("Off-page connector", [[(-6.5, -3), (-6.5, 3), (3.5, 3), (6.5, 0), (3.5, -3), (-6.5, -3)]],
                           [], [], [(-6.5, 0), (6.5, 0)], {"TEXT": (-1.5, 0)}),
    "FLOW_ARROW": ("Flow direction", [[(-1.2, 1), (1.2, 0), (-1.2, -1)]], [], [], [(-1.2, 0), (1.2, 0)], {}),
}
HEAVY = {"PUMP", "COMPRESSOR", "HEAT_EXCHANGER"}  # drawn with the 0.5 mm pen like the pipes they sit in
THROUGH = {"FLOW_ARROW": [(-1.2, 0), (1.2, 0)], "ORIFICE": [(-0.6, 0), (0.6, 0)]}  # symbols sitting on a pipe


LIBRARIES = Path(__file__).parent.parent / "libraries"
BULGE = 0.4142135623730951  # tan(22.5 degrees): a quarter arc, counter-clockwise


def rounded(x0: float, y0: float, x1: float, y1: float, r: float) -> list[tuple[float, float, float]]:
    """A rectangle with rounded corners as LWPOLYLINE points (x, y, bulge), counter-clockwise."""
    return [(x0 + r, y0, 0), (x1 - r, y0, BULGE), (x1, y0 + r, 0), (x1, y1 - r, BULGE), (x1 - r, y1, 0),
            (x0 + r, y1, BULGE), (x0, y1 - r, 0), (x0, y0 + r, BULGE)]


def box(w: float, depth: float) -> list[tuple[float, float]]:
    """The outline of a fixture w wide and depth deep, its back on the wall (y = 0)."""
    return [(-w / 2, 0), (w / 2, 0), (w / 2, -depth), (-w / 2, -depth)]


# name: description, closed outlines (points, or points with bulges), lines, circles (x, y, r), ellipses (x, y, half
# width, half depth); millimetres, the back of the fixture on y = 0
FIXTURES = {
    "WC": ("Toilet", [box(400, 180)], [], [], [(0, -410, 180, 230)]),
    "WASHBASIN": ("Washbasin", [box(600, 450)], [], [], [(0, -240, 220, 150)]),
    "BATHTUB": ("Bathtub", [box(1700, 750), rounded(-775, -675, 775, -75, 150)], [], [], []),
    "SHOWER": ("Shower tray", [box(900, 900)], [[(-450, 0), (450, -900)], [(450, 0), (-450, -900)]], [], []),
    "KITCHEN_SINK": ("Kitchen sink", [box(860, 500), rounded(-360, -430, 40, -70, 60),
                                      rounded(100, -430, 360, -70, 40)], [], [], []),
    "HOB": ("Hob", [box(600, 600)], [], [(-150, -150, 90), (150, -150, 70), (-150, -450, 70), (150, -450, 90)], []),
    # No line for the control panel: next to a side, it would look like the second face of a wall at 1:100
    "WASHING_MACHINE": ("Washing machine", [box(600, 600)], [], [(0, -300, 200)], []),
}


def floor_plan(path: Path = LIBRARIES / "floor-plan" / "symbols.dxf") -> None:
    doc = ezdxf.new("R2018", units=ezdxf.units.MM)
    pen = {"lineweight": 25}
    for name, (description, outlines, lines, circles, ellipses) in FIXTURES.items():
        block = doc.blocks.new(name, dxfattribs={"description": description})
        for points in outlines:
            block.add_lwpolyline(points, format="xyb" if len(points[0]) == 3 else "xy", close=True, dxfattribs=pen)
        for points in lines:
            block.add_lwpolyline(points, dxfattribs=pen)
        for x, y, r in circles:
            block.add_circle((x, y), r, dxfattribs=pen)
        for x, y, half_width, half_depth in ellipses:  # the major axis is the longer one
            major, ratio = ((half_width, 0), half_depth / half_width) if half_width > half_depth else \
                ((0, half_depth), half_width / half_depth)
            block.add_ellipse((x, y), major_axis=major, ratio=ratio, dxfattribs=pen)
    doc.saveas(path)


def pid(path: Path = LIBRARIES / "din-10628" / "symbols.dxf") -> None:
    doc = ezdxf.new("R2018", units=ezdxf.units.MM)
    for layer in ("PORT", "THROUGH"):  # helpers for recognition, not part of the drawing
        doc.layers.add(layer, color=1).dxf.plot = 0
    for name, (description, polylines, circles, arcs, ports, attributes) in SYMBOLS.items():
        block = doc.blocks.new(name, dxfattribs={"description": description})
        pen = {"lineweight": 50 if name in HEAVY else 35}
        for points in polylines:
            closed = points[0] == points[-1]
            block.add_lwpolyline(points[:-1] if closed else points, close=closed, dxfattribs=pen)
        for x, y, r in circles:
            block.add_circle((x, y), r, dxfattribs=pen)
        for x, y, r, start, end in arcs:
            block.add_arc((x, y), r, start, end, dxfattribs=pen)
        for point in ports:
            block.add_point(point, dxfattribs={"layer": "PORT"})
        if name in THROUGH:
            block.add_line(*THROUGH[name], dxfattribs={"layer": "THROUGH"})
        for tag, point in attributes.items():
            block.add_attdef(tag, point, dxfattribs={"height": 2.0}).set_placement(
                point, align=TextEntityAlignment.MIDDLE_CENTER)
    doc.saveas(path)


if __name__ == "__main__":
    pid()
    floor_plan()
