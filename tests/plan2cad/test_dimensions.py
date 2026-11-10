import math

import numpy as np
import pytest
from ezdxf.entities import DimStyleOverride

from app import scan
from app.plan2cad import dimensions
from app.plan2cad.export import to_dxf
from app.plan2cad.features import Features
from app.plan2cad.model import Circle, Dimension, Drawing, Line, Polygon, Text
from app.plan2cad.pipeline import convert
from app.plan2cad.render import render
from app.plan2cad.simulate import plot

PPM = 12.0  # px per mm of the test images, about a 300 dpi scan
XS, Y = (50.0, 98.0, 122.0), 30.0  # a chain of two dimensions, 48 and 24 mm long: 2,40 and 1,20 m at 1:50


def chain(terminator: str = "tick", values=("2,40", "1,20"), overshoot: float = 2.0) -> Drawing:
    """A chain of dimensions as a plan at 1:50 has them: extension lines from 10 mm below the dimension line to
    ``overshoot`` above it, a terminator at each end and the values over the middle."""
    d = Drawing(200, 80, scale=50, texts=[Text((a + b) / 2, Y - 2.3, 2.5, v) for a, b, v in zip(XS, XS[1:], values)])
    d.lines = [Line(XS[0], Y, XS[-1], Y, 0.18)] + [Line(x, Y + 10, x, Y - overshoot, 0.18) for x in XS]
    for k, x in enumerate(XS):
        sides = [s for s in (1, -1) if (s == 1 and k < len(XS) - 1) or (s == -1 and k > 0)]  # into its dimensions
        if terminator == "tick":
            d.lines.append(Line(x - 1, Y + 1, x + 1, Y - 1, 0.35))
        elif terminator == "dot":
            ring = np.linspace(0, 2 * math.pi, 16, endpoint=False)
            d.polygons.append(Polygon([(x + 0.7 * math.cos(a), Y + 0.7 * math.sin(a)) for a in ring]))
        for s in sides if terminator in ("arrow", "open") else ():
            back, half = x + 3.0 * s, 0.8
            if terminator == "arrow":
                d.polygons.append(Polygon([(x, Y), (back, Y - half), (back, Y + half)]))
            else:
                d.lines += [Line(x, Y, back, Y - half, 0.18), Line(x, Y, back, Y + half, 0.18)]
    return d


def found(d: Drawing, ppm: float = PPM, mm: float = 1.0) -> tuple[list[Dimension], list[Line], list[Text]]:
    """What the finder makes of the drawing's own lines and texts, on its image (``mm`` units a millimetre as drawn)."""
    ink = (render(d, ppm) < 128).astype(np.uint8)
    return dimensions.find(d.lines, d.texts, ink, ppm, mm, scale=d.scale)


def test_a_chain_of_ticked_dimensions_becomes_dimension_objects():
    dims, lines, texts = found(chain())
    assert [(m.value, m.terminator, m.doubt) for m in dims] == [("2,40", "tick", None), ("1,20", "tick", None)]
    assert [(m.x1, m.y1, m.x2, m.y2) for m in dims] == [pytest.approx((50, 30, 98, 30), abs=0.1),
                                                        pytest.approx((98, 30, 122, 30), abs=0.1)]
    assert dims[0].p1 == pytest.approx((50, 40)) and dims[0].extension == pytest.approx(2.0, abs=0.1)
    assert dims[0].size == pytest.approx(2.0, abs=0.4)  # the tick's extent along the line
    assert not lines and not texts  # every stroke and text is a part of one dimension, once
    parts = [q for m in dims for q in m.parts]
    assert sum(isinstance(q, Line) for q in parts) == 1 + 3 + 3  # dimension line, extension lines, ticks
    assert [m.text.text for m in dims] == ["2,40", "1,20"]


@pytest.mark.parametrize("kind, terminator", [("arrow", "arrow"), ("open", "arrow"), ("dot", "dot")])
def test_arrowheads_and_dots_end_dimensions_too(kind, terminator):
    dims, lines, _ = found(chain(kind))
    assert [(m.value, m.terminator) for m in dims] == [("2,40", terminator), ("1,20", terminator)]
    filled = [q for m in dims for q in m.parts if isinstance(q, Polygon)]
    assert len(filled) == {"arrow": 4, "open": 0, "dot": 3}[kind]  # both heads where two dimensions meet
    assert not lines  # the sides of open heads as traced are parts, not loose lines as well


@pytest.mark.parametrize("kind", ["tick", "arrow", "open"])
def test_extension_lines_stopping_at_the_dimension_line_end_it_with_a_tick_or_an_arrowhead(kind):
    dims, lines, _ = found(chain(kind, overshoot=0.0))  # a style that does not run them on past it
    assert [m.value for m in dims] == ["2,40", "1,20"] and not lines


def test_a_line_nearer_to_the_values_than_their_dimension_line_does_not_hide_them():
    d = chain()
    d.lines.append(Line(20, Y - 4.0, 160, Y - 4.0, 0.5))  # a wall above the values, nearer than the dimension line
    dims, lines, _ = found(d)
    assert [m.value for m in dims] == ["2,40", "1,20"] and lines == [d.lines[-1]]


@pytest.mark.parametrize("start, end", [(100, 160), (40, 96)])  # over the second value only, over the first only
def test_a_line_nearer_to_some_values_of_a_chain_than_their_dimension_line_does_not_take_them(start, end):
    d = chain()
    d.lines.append(Line(start, Y - 4.0, end, Y - 4.0, 0.5))
    dims, lines, texts = found(d)
    assert [(m.value, m.doubt) for m in dims] == [("2,40", None), ("1,20", None)]
    assert lines == [d.lines[-1]] and not texts


def test_a_dimension_line_running_on_far_past_its_chain_leaves_the_rest_as_it_is():
    d = chain()
    d.lines[0] = Line(XS[0], Y, XS[-1] + 20, Y, 0.18)  # on as other linework, 20 mm past the last end
    dims, lines, _ = found(d)
    assert len(dims) == 2 and [(q.x1, q.x2) for q in lines] == [pytest.approx((XS[-1], XS[-1] + 20))]
    assert max(q.x2 for m in dims for q in m.parts) == pytest.approx(XS[-1] + 1, abs=0.1)  # the tick, not the line


def test_a_piece_of_a_tick_traced_across_the_dimension_line_is_no_end_of_its_own():
    d = chain()
    d.lines.append(Line(XS[1] + 0.7, Y, XS[1] + 0.7, Y - 0.8, 0.18))  # tracing broke it off the tick's arm
    dims, _, _ = found(d)
    assert [(m.value, round(m.x1), round(m.x2)) for m in dims] == [("2,40", 50, 98), ("1,20", 98, 122)]


def test_a_value_that_does_not_fit_its_length_is_marked_to_check():
    dims, _, _ = found(chain(values=("2,40", "1,70")))
    assert [m.doubt for m in dims] == [None, "its value does not fit its length at 1:50"]


def test_without_a_scale_a_value_is_checked_against_the_other_dimensions_from_three_on():
    def row(values: list[str]) -> list[Dimension]:  # 24 mm long each
        return [Dimension(24 * k, Y, 24 * k + 24, Y, v, (24 * k, Y + 10), (24 * k + 24, Y + 10))
                for k, v in enumerate(values)]

    dims, pair = row(["1,20", "1,20", "1,70", "1,20"]), row(["1,20", "1,70"])
    dimensions.check(dims, None)
    dimensions.check(pair, None)
    assert [m.doubt for m in dims] == [None, None, "its value does not fit its length as the other dimensions of the "
                                       "sheet do", None]
    assert [m.doubt for m in pair] == [None, None]  # two say nothing about the sheet


@pytest.mark.parametrize("ppm", [2 * PPM, PPM])
def test_a_sheet_printed_at_half_size_has_the_same_dimensions_at_half_size(ppm):
    dims, lines, texts = found(chain().scaled(0.5), ppm, mm=0.5)
    assert [(m.value, m.doubt) for m in dims] == [("2,40", None), ("1,20", None)]
    assert [(m.x1, m.x2) for m in dims] == [pytest.approx((25, 49), abs=0.1), pytest.approx((49, 61), abs=0.1)]
    assert not lines and not texts


def test_switched_off_dimensions_stay_lines_and_texts():
    d = chain()
    img = plot(Drawing(297, 210, lines=d.lines, polygons=d.polygons, texts=d.texts), 300)  # on an A4 sheet
    on, off = (convert(img, paper="A4", scale=50, dpi=300, features=f).drawing
               for f in (Features(), Features.of(off=["dimensions"])))
    assert [m.value for m in on.dimensions] == ["2,40", "1,20"] and not off.dimensions
    assert sorted(t.text for t in off.texts) == ["1,20", "2,40"] and len(off.lines) > len(on.lines)


def test_a_grid_with_numbers_is_no_dimension():
    d = chain()
    d.lines = d.lines[:4]  # the same lines and numbers without terminators, as in a table
    assert found(d) == ([], d.lines, d.texts)


def test_branches_with_junction_dots_are_no_dimension():
    d = chain("dot", overshoot=0.0)  # lines that end at the line, as wires branching off, do not cross it
    assert found(d)[0] == []


def test_dots_where_lines_run_on_both_ways_are_junctions_but_ticks_still_end_dimensions():
    assert found(chain("dot", overshoot=10.0))[0] == []  # wires running through junction dots
    assert len(found(chain("tick", overshoot=10.0))[0]) == 2  # a dimension from wall to wall


def wires(through: bool, pen: float, dot: tuple[float, float]) -> Drawing:
    """A wire of a circuit diagram between two junction dots (radius and pen ``dot``), where wires branch off that
    run on past it if ``through``, with its wire number over it."""
    d = Drawing(160, 80, texts=[Text(83.5, Y - 2.3, 2.0, "14")], lines=[Line(50, Y, 110, Y, pen)])
    for x in (50, 110):
        d.lines.append(Line(x, Y - 15, x, Y + (8 if through else 0), pen))
        d.circles.append(Circle(x, Y, *dot))
    return d


@pytest.mark.parametrize("through", [True, False])
@pytest.mark.parametrize("ppm, pen, dot", [(7.87, 0.35, (0.5, 0.8)), (PPM, 0.35, (0.5, 0.4)), (PPM, 0.7, (0.7, 0.8))])
def test_a_wire_number_between_junction_dots_is_no_dimension(through, ppm, pen, dot):
    d = wires(through, pen, dot)  # a dot is a blob as wide as it is long, a tick a slim stroke
    assert found(d, ppm) == ([], d.lines, d.texts)


@pytest.mark.parametrize("ppm", [7.87, PPM])
@pytest.mark.parametrize("pen, half", [(0.25, 0.6), (0.7, 0.6), (0.35, 1.5)])
def test_thin_thick_short_and_long_ticks_end_dimensions(ppm, pen, half):
    d = chain()
    d.lines[4:] = [Line(x - half, Y + half, x + half, Y - half, pen) for x in XS]
    assert [(m.value, m.terminator) for m in found(d, ppm)[0]] == [("2,40", "tick"), ("1,20", "tick")]


def test_the_linework_of_a_dimension_is_its_dimension_line_and_extension_lines_as_cad_draws_them():
    dims, _, _ = found(chain())
    assert [(q.x1, q.y1, q.x2, q.y2) for q in dimensions.linework(dims[0])] == [
        pytest.approx(v, abs=0.1) for v in ((50, 30, 98, 30), (50, 40, 50, 28), (98, 40, 98, 28))]  # no tick
    on_a_wall = Dimension(0, 0, 10, 0, "1,00", (0, 0), (10, 5), extension=1.0)  # its dimension line ends on one
    assert [(q.x1, q.y1, q.x2, q.y2) for q in dimensions.linework(on_a_wall)] == [(0, 0, 10, 0), (10, 5, 10, -1)]


SEPARATOR = "its decimal separator is not the one the other numbers of the sheet are written with"


@pytest.mark.parametrize("values, others, marked", [
    (("2.40", "1.20"), ["12,40 m²", "4,64"], [True, True]),  # a comma whose tail recognition missed reads as a point
    (("2,40", "1.20"), ["12.40 m²", "4.64"], [True, False]),
])
def test_a_value_read_with_another_decimal_separator_than_the_other_numbers_of_the_sheet_is_marked(values, others,
                                                                                                    marked):
    d = chain(values=values)
    d.texts += [Text(150, 50 + 8 * k, 2.5, s) for k, s in enumerate(others)]  # numbers of the sheet off the chain
    dims, _, texts = found(d)
    assert [m.doubt for m in dims] == [SEPARATOR if k else None for k in marked]
    assert [m.value for m in dims] == list(values) and [t.text for t in texts] == others  # all as read


@pytest.mark.parametrize("values, others", [
    (("2,40", "1,20"), ["12,40 m²", "4,64"]),
    (("2.40", "1.20"), ["12.40 m²", "4.64"]),
    (("2.40", "1.20"), ["12,40 m²", "4.64"]),  # a single comma may be a speck under a point
    (("2,40", "1,20"), ["16.10.2026"]),  # a date is no decimal number
    (("2.40", "1.20"), ["1,250", "1,500"]),  # nor a number whose three digits after the comma may be thousands
    (("2,40", "1,20"), []),  # the other value alone says nothing
])
def test_a_value_read_like_the_other_numbers_of_the_sheet_is_not_marked(values, others):
    d = chain(values=values)
    d.texts += [Text(150, 50 + 8 * k, 2.5, s) for k, s in enumerate(others)]
    assert [m.doubt for m in found(d)[0]] == [None, None]


def test_numbers_read_unreliably_say_nothing_about_the_separator_of_the_sheet():
    d = chain(values=("2.40", "1.20"))
    d.texts += [Text(150, 50, 2.5, "12,40 m²", uncertain=True), Text(150, 58, 2.5, "4,64", uncertain=True)]
    assert [m.doubt for m in found(d)[0]] == [None, None]


def test_a_value_marked_for_another_reason_keeps_that_reason():
    d = chain(values=("2.40", "1.70"))  # the second does not fit its length either
    d.texts += [Text(150, 50, 2.5, "12,40 m²"), Text(150, 58, 2.5, "4,64")]
    assert [m.doubt for m in found(d)[0]] == [SEPARATOR, "its value does not fit its length at 1:50"]


@pytest.mark.parametrize("text, expected", [("2,40", (2.4, 0.01, None)), ("240 mm", (240.0, 1.0, 1.0)),
                                            ("1,01⁵", (1.015, 0.001, None)), ("7'-6\"", (90.0, 1.0, 25.4)),
                                            ("12,40 m²", None), ("±0,00", None), ("Flur", None), ("1:50", None),
                                            ("0,00", None), ("3 1/0\"", None)])
def test_values_of_dimensions(text, expected):
    got = dimensions.value(text)
    assert got == (pytest.approx(expected) if expected else None)


def test_dimensions_are_dxf_dimensions_with_the_value_as_read():
    dims, _, _ = found(chain(values=("2,40", "1,70")))
    msp = to_dxf(Drawing(200, 80, scale=50, dimensions=dims)).modelspace()
    first, second = msp.query("DIMENSION")
    assert (first.dxf.layer, first.dxf.text, second.dxf.layer) == ("P2C-DIMENSION", "2,40", "P2C-DIMENSION-CHECK")
    assert first.get_measurement() == pytest.approx(2400, abs=10)  # real-world millimetres at 1:50
    assert DimStyleOverride(second).get("dimse1") == 1  # the extension line both share is drawn once
    ticks = [e for e in first.get_geometry_block() if e.dxftype() == "INSERT"]
    along = [[p.x for v in e.virtual_entities() for p in (v.dxf.start, v.dxf.end)] for e in ticks]
    assert [(max(x) - min(x)) / 50 for x in along] == [pytest.approx(2.0, abs=0.3)] * 2  # 2 mm as printed


def test_a_drawing_keeps_its_dimensions_when_scaled_and_draws_their_parts():
    dims, _, _ = found(chain())
    d = Drawing(200, 80, dimensions=dims).scaled(2.0)
    assert (d.dimensions[1].x2, d.dimensions[0].p1) == (pytest.approx(244, abs=0.2), pytest.approx((100, 80)))
    assert d.dimensions[0].text.height == pytest.approx(5.0)
    flat = d.flat()
    assert not flat.dimensions and len(flat.lines) == 7 and [t.text for t in flat.texts] == ["2,40", "1,20"]


def test_dimensions_go_into_the_report_and_the_markdown():
    dims, _, _ = found(chain(values=("2,40", "1,70")))
    d = Drawing(200, 80, scale=50, dimensions=dims)
    report = scan._report(d)
    assert [(m["value"], m["uncertain"], m["terminator"]) for m in report["dimensions"]] == [
        ("2,40", False, "tick"), ("1,70", True, "tick")]
    assert [t["text"] for t in report["texts"]] == ["2,40", "1,70"]  # the values are texts of the sheet as before
    assert d.counts()["texts"] == 2 and d.counts()["dimensions"] == 2
    assert [(c["kind"], c["what"]) for c in report["review"]] == [("dimension", "1,70")]
    info = {"drawing_type": "building", "paper": "A4", "sheet_size_mm": (200, 80), "sheet": "paper", "scale": 50,
            "counts": d.counts()}
    markdown = "\n".join(scan._page_markdown(info, d, "##"))
    assert "## Dimensions\n\n- 2,40" in markdown
    assert "## Dimensions to check (uncertain recognition)\n\n- 1,70 (its value does not fit its length" in markdown


def test_a_kind_without_dimensions_switches_them_off_and_says_so():
    d = chain()
    img = plot(Drawing(297, 210, lines=d.lines, polygons=d.polygons, texts=d.texts), 300)
    pid, asked = (convert(img, paper="A4", scale=50, dpi=300, drawing="pid", features=f)
                  for f in (Features(), Features.of(on=["dimensions"])))
    assert not pid.drawing.dimensions and pid.info["steps_off_by_kind"] == ["dimensions"]
    assert [m.value for m in asked.drawing.dimensions] == ["2,40", "1,20"] and asked.info["steps_off_by_kind"] == []
    said = [line for line in scan._page_markdown(pid.info, pid.drawing, "##") if "switched off" in line]
    assert said == ["- Steps switched off for this kind of drawing: dimensions (options.on switches them on)"]
    assert not [line for line in scan._page_markdown(asked.info, asked.drawing, "##") if "switched off" in line]
