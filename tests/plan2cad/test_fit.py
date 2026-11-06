import math

import pytest
from ezdxf.entities import DimStyleOverride

from app import scan
from app.plan2cad import fit
from app.plan2cad.export import to_dxf
from app.plan2cad.features import Features
from app.plan2cad.model import Dimension, Drawing, Line, Polyline, Text
from app.plan2cad.pipeline import convert
from app.plan2cad.simulate import plot


def plan(values=("2,40", "1,20"), jamb: float = 98.2, overall: str | None = None) -> Drawing:
    """A wall at 1:50 whose face steps back by 1 mm at a jamb drawn 0.2 mm right of where its chain of dimensions says
    (2,40 and 1,20 m: 48 and 24 mm), with a line ending on the jamb and the chain's extension lines starting 2 mm
    before the wall; with an ``overall`` dimension over the chain, further out."""
    d = Drawing(200, 100, scale=50)
    d.polylines = [Polyline([(50, 40), (jamb, 40), (jamb, 41), (122, 41), (122, 80), (50, 80)], [0.0] * 6, True,
                            kind="wall")]
    d.lines = [Line(jamb, 41, jamb, 70), Line(30, 60, 50, 60)]  # a wall face on from the jamb, a line to the wall
    d.dimensions = [Dimension(50, 30, jamb, 30, values[0], (50, 38), (jamb, 38)),
                    Dimension(jamb, 30, 122, 30, values[1], (jamb, 38), (122, 38))]
    if overall:
        d.dimensions.append(Dimension(50, 20, 122, 20, overall, (50, 28), (122, 28)))
    return d


def measured(m: Dimension) -> float:
    """The distance between a dimension's measured points along its dimension line, as CAD measures it."""
    u = ((m.x2 - m.x1) / math.dist((m.x1, m.y1), (m.x2, m.y2)), (m.y2 - m.y1) / math.dist((m.x1, m.y1), (m.x2, m.y2)))
    return abs((m.p2[0] - m.p1[0]) * u[0] + (m.p2[1] - m.p1[1]) * u[1])


def test_a_chain_drawn_off_its_values_comes_out_exact_and_connected():
    d = plan()
    fit.to_values(d)
    assert [measured(m) for m in d.dimensions] == [pytest.approx(48.0, abs=1e-9), pytest.approx(24.0, abs=1e-9)]
    assert [m.doubt for m in d.dimensions] == [None, None]
    wall, (face, line) = d.polylines[0].points, d.lines
    assert wall[2][0] - wall[0][0] == pytest.approx(48.0) and wall[3][0] - wall[2][0] == pytest.approx(24.0)
    assert wall[1][0] == wall[2][0] and (face.x1, face.y1) == wall[2]  # the jamb moved whole, the face on it with it
    assert (line.x2, line.y2) == (wall[5][0], 60)  # a line ending on a moved wall face stays on it
    assert d.dimensions[0].x2 == d.dimensions[1].x1 == wall[1][0]  # the chain moved with the drawing


def test_a_hatch_line_ending_on_a_moved_wall_slides_along_itself_and_keeps_its_direction():
    d = plan()
    d.lines.append(Line(78.2, 45, 98.2, 65))  # at 45 degrees to the face on from the jamb, which moves left
    fit.to_values(d)
    hatch, face = d.lines[2], d.lines[0]
    assert face.x1 < 98.2 and hatch.x2 == pytest.approx(face.x1)  # still on the face
    assert (hatch.x1, hatch.y1) == (78.2, 45) and hatch.y2 == pytest.approx(65 - (98.2 - face.x1))  # at 45 degrees


def test_measured_points_land_on_the_wall_with_the_gap_as_offset():
    d = plan()
    fit.to_values(d)
    first, second = d.dimensions
    assert (first.p1[1], first.p2[1], second.p2[1]) == (40, 40, 41)  # on the face, at the corner of the step
    assert first.p2 == second.p1 == d.polylines[0].points[1]
    assert (first.offset, second.offset) == (pytest.approx(2.0), pytest.approx(2.0))  # the smaller of its two gaps


def test_a_value_misread_beyond_the_tolerance_moves_nothing_and_is_marked():
    d = plan(values=("2,90", "1,20"))
    d.dimensions = d.dimensions[:1]  # 2,90 says 58 mm where 48.2 are drawn
    before = list(d.polylines[0].points)
    fit.to_values(d)
    assert d.dimensions[0].doubt == "its value does not fit its length at 1:50"
    assert d.polylines[0].points == before and measured(d.dimensions[0]) == pytest.approx(48.2)


def test_an_overall_dimension_that_does_not_add_up_with_its_chain_is_marked():
    d = plan(overall="3,62")  # 72.4 mm, within what tracing explains of the 72 drawn, but the chain says 72
    fit.to_values(d)
    assert [m.doubt for m in d.dimensions] == [None, None, fit.DISAGREE]
    assert [measured(m) for m in d.dimensions[:2]] == [pytest.approx(48.0, abs=1e-9), pytest.approx(24.0, abs=1e-9)]
    agreed = plan(overall="3,60")
    fit.to_values(agreed)
    assert [(m.doubt, round(measured(m), 9)) for m in agreed.dimensions] == [(None, 48), (None, 24), (None, 72)]


def test_a_chain_through_an_outer_wall_dimensions_the_inner_wall_behind_it():
    d = Drawing(200, 120, scale=50)  # an outer wall from y 20 down, 7 mm thick, and an inner wall drawn 0.2 mm off
    d.polylines = [Polyline([(50, 100), (50, 20), (120, 20)], [0.0, 0.0], kind="wall"),
                   Polyline([(57, 100), (57, 65), (110, 65), (110, 60.2), (57, 60.2), (57, 27), (120, 27)], [0.0] * 6,
                            kind="wall")]
    d.dimensions = [Dimension(40, 20, 40, 60.2, "2,00", (48, 20), (48, 60.2))]  # to the inner wall's upper face
    fit.to_values(d)
    (m,), inner = d.dimensions, d.polylines[1].points
    assert m.doubt is None and measured(m) == pytest.approx(40.0, abs=1e-9)
    assert m.p2 == (50, pytest.approx(inner[3][1])) and inner[3][1] == inner[4][1] < 60.2  # on the outer face
    assert inner[4][1] - d.polylines[0].points[1][1] == pytest.approx(40.0)  # the inner face is where the value says


def test_a_turned_chain_is_fitted_along_its_own_direction():
    d = plan()
    a = math.radians(30)

    def turn(x, y):
        c, s = math.cos(a), math.sin(a)
        return 100 + (x - 100) * c - (y - 50) * s, 50 + (x - 100) * s + (y - 50) * c

    d.polylines[0].points = [turn(*p) for p in d.polylines[0].points]
    d.lines = [Line(*turn(q.x1, q.y1), *turn(q.x2, q.y2)) for q in d.lines]
    d.dimensions = [Dimension(*turn(m.x1, m.y1), *turn(m.x2, m.y2), m.value, turn(*m.p1), turn(*m.p2))
                    for m in d.dimensions]
    fit.to_values(d)
    assert [measured(m) for m in d.dimensions] == [pytest.approx(48.0, abs=1e-9), pytest.approx(24.0, abs=1e-9)]
    assert math.dist(d.dimensions[0].p2, d.polylines[0].points[1]) < 1e-9


def test_the_dxf_dimension_measures_its_value_and_draws_the_gap_and_the_report_says_so():
    d = plan()
    fit.to_values(d)
    first, second = to_dxf(d).modelspace().query("DIMENSION")
    assert (first.get_measurement(), second.get_measurement()) == (pytest.approx(2400, abs=1e-6),
                                                                   pytest.approx(1200, abs=1e-6))
    assert DimStyleOverride(first).get("dimexo") == pytest.approx(2.0)
    report = scan._report(d)["dimensions"][0]
    assert (report["offset_mm"], report["measured_mm"][0], report["length_mm"]) == (2.0, [50.07, 40.0], 48.0)


def test_on_a_scan_the_step_puts_the_dimensions_on_the_wall_and_switched_off_leaves_them():
    d = plan(jamb=98.0)
    for m in d.dimensions:  # as printed: dimension line, extension lines up to 2 mm past it and ticks
        d.lines += [Line(m.x1, m.y1, m.x2, m.y2, 0.18)]
    d.lines += [Line(x, 38, x, 28, 0.18) for x in (50, 98, 122)]
    d.lines += [Line(x - 1, 31, x + 1, 29, 0.35) for x in (50, 98, 122)]
    d.texts = [Text(74, 27.7, 2.5, "2,40"), Text(110, 27.7, 2.5, "1,20")]
    for pl in d.polylines:
        pl.width = 0.5
    img = plot(Drawing(297, 210, lines=d.lines + [q for pl in d.polylines for q in pl.parts()], texts=d.texts), 300)
    on, off = (convert(img, paper="A4", scale=50, dpi=300, drawing="building", features=f).drawing
               for f in (Features(), Features.of(off=["dimension_fit"])))
    assert [m.value for m in on.dimensions] == [m.value for m in off.dimensions] == ["2,40", "1,20"]
    assert [round(measured(m), 6) for m in on.dimensions] == [48.0, 24.0]
    assert all(abs(p[1] - 38) < 0.3 for m in off.dimensions for p in (m.p1, m.p2))  # where the extension lines start
    assert [abs(m.p1[1] - 40) < 0.3 for m in on.dimensions] == [True, True]  # on the wall, 2 mm further
    assert all(m.offset == pytest.approx(2.0, abs=0.3) for m in on.dimensions)
