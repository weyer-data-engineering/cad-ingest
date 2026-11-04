import math

import pytest
from evaluate import _openings, crossings, openings

from app.plan2cad.model import Drawing, Line, Polyline


def curve(n: int = 10, step: float = 6.0) -> list[Line]:
    """A gentle curve drawn as n short straight lines, as a contour line is: each turns by 4 degrees."""
    pts = [(20.0, 50.0)]
    for k in range(n):
        a = math.radians(4 * k)
        pts.append((pts[-1][0] + step * math.cos(a), pts[-1][1] - step * math.sin(a)))
    return [Line(*p, *q, 0.18) for p, q in zip(pts, pts[1:])]


def test_a_curve_drawn_in_short_lines_bends_without_junctions():
    lines = curve()
    at = (lines[6].x2, lines[6].y2)
    branch = Line(*at, at[0] + 3, at[1] + 20, 0.25)  # ends on a vertex of the curve: a junction
    gt = Drawing(200, 100, lines=lines + [branch])
    smooth = Polyline([(lines[0].x1, lines[0].y1), (lines[3].x2, lines[3].y2), (lines[-1].x2, lines[-1].y2)],
                      [0.0, 0.0], width=0.18)  # the output follows it with fewer vertices
    assert crossings(gt, Drawing(200, 100, lines=[branch], polylines=[smooth])) == {"crossings_kept": None,
                                                                                    "junctions_kept": 1.0}
    short = Line(at[0] + 0.3, at[1] + 2, at[0] + 3, at[1] + 20, 0.25)  # the branch stops 2 mm short of the curve
    assert crossings(gt, Drawing(200, 100, lines=[short], polylines=[smooth]))["junctions_kept"] == 0.0


def test_a_corner_of_one_pen_is_still_a_junction():
    gt = Drawing(100, 100, lines=[Line(10, 10, 50, 10), Line(50, 10, 50, 50)])
    assert crossings(gt, Drawing(100, 100, lines=[Line(10, 10, 48, 10)]))["junctions_kept"] == 0.0  # no end near it
    assert crossings(gt, gt)["junctions_kept"] == 1.0


def test_a_crossing_kept_by_a_line_taken_for_dashed_still_counts():
    a, b = Line(10, 50, 90, 50, 0.25), Line(50, 10, 50, 90, 0.18)
    gt = Drawing(100, 100, lines=[a, b])
    dashed = Line(50, 10, 50, 90, 0.18, dashed=True)
    assert crossings(gt, Drawing(100, 100, lines=[a, dashed]))["crossings_kept"] == 1.0
    cut = [Line(10, 50, 50, 50), Line(50, 50, 90, 50)]  # joined at the crossing: two ends there
    assert crossings(gt, Drawing(100, 100, lines=cut + [dashed]))["crossings_kept"] == 0.0


def wall_with_door(gap: float = 18.0) -> list[Line]:
    """A wall 2.4 mm thick with a door: both faces broken off, the jambs closing them."""
    x0, x1, y0, y1 = 20.0, 120.0, 50.0, 52.4
    left, right = 60.0, 60.0 + gap
    return [Line(x0, y0, left, y0, 0.5), Line(left, y0, left, y1, 0.5), Line(left, y1, x0, y1, 0.5),
            Line(right, y0, x1, y0, 0.5), Line(right, y1, right, y0, 0.5), Line(x1, y1, right, y1, 0.5)]


def test_a_door_in_a_wall_is_an_opening_kept_open_or_closed():
    gt = Drawing(150, 100, lines=wall_with_door())
    assert len(_openings(gt)) == 2  # both faces
    assert openings(gt, gt) == {"openings_kept": 1.0}
    closed = Drawing(150, 100, lines=gt.lines + [Line(60, 50, 78, 50, 0.5)])  # one face drawn across the door
    assert openings(gt, closed) == {"openings_kept": 0.5}


def test_an_opening_without_its_wall_is_not_kept():
    gt = Drawing(150, 100, lines=wall_with_door())
    assert openings(gt, Drawing(150, 100)) == {"openings_kept": 0.0}
    assert openings(gt, Drawing(150, 100, lines=gt.lines[3:])) == {"openings_kept": 0.0}  # the wall left of it lost


def test_where_walls_join_there_is_no_opening():
    outer = [Line(20, 50, 120, 50, 0.5), Line(20, 52.4, 70, 52.4, 0.5), Line(72.4, 52.4, 120, 52.4, 0.5)]
    inner = [Line(70, 52.4, 70, 90, 0.5), Line(72.4, 52.4, 72.4, 90, 0.5)]  # its faces lead away from the face opposite
    assert _openings(Drawing(150, 100, lines=outer + inner + [Line(70, 90, 72.4, 90, 0.5)])) == []
    # an inner wall between two walls: each face it joins is broken at the same place, its faces join the two gaps
    other = [Line(20, 92.4, 120, 92.4, 0.5), Line(20, 90, 70, 90, 0.5), Line(72.4, 90, 120, 90, 0.5)]
    assert _openings(Drawing(150, 100, lines=outer + inner + other)) == []


def test_strokes_that_happen_to_lie_on_one_line_are_no_opening():
    def tick(x, y):  # a dimension tick, centred on the dimension point
        return Line(x - 1, y + 1, x + 1, y - 1, 0.35)

    # the end ticks of four dimension chains, two by two on one line, the gaps beside each other
    assert _openings(Drawing(100, 100, lines=[tick(50, 42), tick(62, 30), tick(42, 42), tick(62, 22)])) == []


def test_hatching_ending_on_an_outline_and_a_crack_are_no_openings():
    outline = [Line(10, 10, 90, 10, 0.5), Line(10, 30, 90, 30, 0.5)]
    hatch = [Line(20, 30, 40, 10, 0.18), Line(50, 0, 60, -10, 0.18)]  # on one line, ending on the outline
    assert _openings(Drawing(100, 100, lines=outline + hatch)) == []
    assert _openings(Drawing(100, 100, lines=wall_with_door(gap=1.0))) == []


@pytest.mark.parametrize("drawing", ["section", "site_plan", "building_plan", "schematic", "pid", "pid_details"])
def test_the_references_keep_all_of_their_own(drawing):
    import synth

    d = getattr(synth, drawing)(1)
    assert set(crossings(d, d).values()) <= {1.0, None}
    assert openings(d, d) == ({"openings_kept": 1.0} if drawing == "building_plan" else {})  # only plans have walls
