import math

import numpy as np
import pytest

from app.plan2cad.model import Dimension, Drawing, Line, Polyline, Symbol
from app.plan2cad.square import directions, square, tolerance


def angle(q: Line) -> float:
    return math.degrees(math.atan2(q.y2 - q.y1, q.x2 - q.x1)) % 180


def turned(x: float, y: float, a: float, cx: float = 100.0, cy: float = 100.0) -> tuple[float, float]:
    """(x, y) turned by ``a`` degrees about (cx, cy)."""
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c


def test_the_tolerance_grows_for_short_segments_between_a_floor_and_a_ceiling():
    assert math.degrees(tolerance(200.0)) == pytest.approx(0.5)
    assert math.degrees(tolerance(20.0)) == pytest.approx(math.degrees(math.atan(0.03)))
    assert math.degrees(tolerance(2.0)) == pytest.approx(5.0)


def test_a_slightly_turned_rectangle_comes_out_exact_and_closed():
    corners = [(20.0, 30.1), (120.2, 29.9), (120.0, 90.15), (19.85, 90.0)]  # traced: each side leans a little
    sides = [Line(*p, *q, 0.5) for p, q in zip(corners, corners[1:] + corners[:1])]
    d = Drawing(200, 150, lines=sides)
    square(d)
    assert all(q.x1 == q.x2 or q.y1 == q.y2 for q in d.lines)  # exactly along the axes
    for a, b in zip(d.lines, d.lines[1:] + d.lines[:1]):  # each corner still closed
        assert (a.x2, a.y2) == (b.x1, b.y1)
    assert all(math.dist(p, (q.x1, q.y1)) < 0.25 for p, q in zip(corners, d.lines))


def test_a_polyline_and_a_t_stay_joined():
    wall = Polyline([(10.0, 50.0), (150.0, 50.3), (150.1, 120.0)], [0.0, 0.0], width=0.5, kind="wall")
    branch = Line(80.0, 50.15, 80.4, 100.0, 0.35)  # ends on the wall between its ends
    d = Drawing(200, 150, lines=[branch], polylines=[wall])
    square(d)
    (x1, y1), (x2, y2), (x3, y3) = d.polylines[0].points
    assert y1 == y2 and x2 == x3
    assert d.lines[0].x1 == d.lines[0].x2  # the branch exactly vertical ...
    assert d.lines[0].y1 == y1  # ... and still on the wall
    assert abs(d.lines[0].x1 - 80.2) < 0.25


def test_a_lone_oblique_line_stays_as_drawn():
    roof = Line(40.0, 100.0, 120.0, 100.0 - 80.0 * math.tan(math.radians(30)), 0.35)
    frame = [Line(10, 10, 190, 10.1), Line(190, 10.1, 190, 140), Line(190, 140, 10, 140), Line(10, 140, 10, 10)]
    stair = Line(60.0, 130.0, 70.0, 130.0 - 10.0 * math.tan(math.radians(31)), 0.25)  # one more, not parallel to it
    d = Drawing(200, 150, lines=frame + [roof, stair])
    before = [Line(q.x1, q.y1, q.x2, q.y2, q.width) for q in (roof, stair)]
    square(d)
    assert d.lines[4:] == before
    angles = np.radians([angle(q) for q in d.lines])
    assert directions(angles, np.array([math.dist((q.x1, q.y1), (q.x2, q.y2)) for q in d.lines])) == [0.0]


def test_a_turned_building_with_its_hatching_keeps_its_direction():
    turn = 20.0
    corners = [turned(x, y, turn) for x, y in ((60, 70), (140, 70), (140, 130), (60, 130))]
    noise = [(0.1, -0.05), (-0.08, 0.1), (0.05, 0.08), (-0.1, -0.06)]
    corners = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(corners, noise)]
    house = Polyline(corners, [0.0] * 4, closed=True, width=0.7)
    hatching = []  # lines at 45 degrees to its sides, ending on its long sides
    for k in range(5):
        x = 62.0 + 4 * k
        a, b = turned(x, 70.0, turn), turned(x + 60.0, 130.0, turn)
        hatching.append(Line(a[0] + 0.05 * (-1) ** k, a[1], b[0], b[1] - 0.06 * (-1) ** k, 0.18))
    frame = [Line(10, 10, 190, 10), Line(190, 10, 190, 190), Line(190, 190, 10, 190), Line(10, 190, 10, 10)]
    d = Drawing(200, 200, lines=frame + hatching, polylines=[house])
    square(d)
    pts = d.polylines[0].points
    sides = [Line(*p, *q) for p, q in zip(pts, pts[1:] + pts[:1])]
    assert {round(angle(q) % 90, 9) for q in sides} == {round(angle(sides[0]) % 90, 9)}  # exactly square
    assert angle(sides[0]) == pytest.approx(turn, abs=0.2)
    assert len({round(angle(q), 9) for q in d.lines[4:]}) == 1  # the hatching exactly parallel ...
    assert angle(d.lines[4]) == pytest.approx(turn + 45, abs=0.3)
    for q in d.lines[4:]:  # ... and still ending on the sides of the house
        for p, side in (((q.x1, q.y1), sides[0]), ((q.x2, q.y2), sides[2])):
            n = (side.y1 - side.y2, side.x2 - side.x1)
            assert abs((p[0] - side.x1) * n[0] + (p[1] - side.y1) * n[1]) / math.hypot(*n) < 1e-6


def test_a_valve_moves_onto_its_pipe_and_its_ports_stay_on_the_pipe_ends():
    valve = Symbol("GATE_VALVE", 100.0, 50.04, parts=[Line(97.0, 48.54, 103.0, 51.54)],
                   ports=[(97.0, 50.04), (103.0, 50.04)])
    pipes = [Line(20.0, 50.0, 97.0, 50.04, 0.5), Line(103.0, 50.04, 180.0, 49.98, 0.5)]
    d = Drawing(200, 100, lines=pipes, symbols=[valve])
    square(d)
    assert all(q.y1 == q.y2 for q in d.lines)
    s = d.symbols[0]
    assert abs(s.y - 50.04) < 0.1 and s.parts[0].y1 - s.y == pytest.approx(-1.5)
    assert (d.lines[0].x2, d.lines[0].y2) == pytest.approx(s.ports[0], abs=1e-9)
    assert (d.lines[1].x1, d.lines[1].y1) == pytest.approx(s.ports[1], abs=1e-9)


def test_a_dimension_line_turns_with_the_sheet_and_its_extension_lines_meet_it_at_right_angles():
    wall = Line(20.0, 40.0, 180.0, 40.0, 0.5)
    m = Dimension(30.0, 30.0, 130.0, 30.2, "1,00", (30.0, 40.0), (130.0, 40.0), extension=2.0)
    d = Drawing(200, 100, lines=[wall], dimensions=[m])
    square(d)
    m = d.dimensions[0]
    assert m.y1 == m.y2 and (m.x1, m.x2) == (30.0, 130.0)  # the feet of its measured points
