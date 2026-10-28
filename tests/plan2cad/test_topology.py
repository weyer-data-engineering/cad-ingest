import math

import pytest

from app.plan2cad.library import load, place
from app.plan2cad.model import Arc, Circle, Drawing, Line, Polyline
from app.plan2cad.topology import connect, join


def test_rectangle_with_open_corners_becomes_one_closed_polyline():
    pls, lines, _ = join([Line(0, 0, 10, 0), Line(10, 0.3, 10, 8), Line(10, 8, 0.4, 8), Line(0, 8, 0, 0.2)], [])
    assert len(pls) == 1 and pls[0].closed and not lines
    assert sorted(pls[0].points) == [(0, 0), (0, 8), (10, 0), (10, 8)]


def test_t_junction_keeps_the_main_line_whole():
    pls, lines, _ = join([Line(0, 0, 5, 0), Line(5, 0, 10, 0), Line(5, 0, 5, 6)], [])
    assert not pls
    assert sorted((ln.x1, ln.x2) for ln in lines) == [(0, 10), (5, 5)]


def test_spurs_and_specks_are_dropped():
    pls, lines, _ = join([Line(0, 0, 10, 0), Line(5, 0, 5, 0.8), Line(20, 20, 20.5, 20)], [])
    assert not pls and lines == [Line(0, 0, 10, 0)]


def test_vessel_outline_becomes_one_closed_polyline():
    sides = [Line(-14, 0, -14, 50, 0.5), Line(14, 0, 14, 50, 0.5)]
    heads = [Arc(0, 0, 14, 180, 360, 0.5), Arc(0, 50, 14, 0, 180, 0.5)]
    pls, lines, arcs = join(sides, heads)
    assert len(pls) == 1 and pls[0].closed and not lines and not arcs
    centres = sorted((round(a.cx), round(a.cy), round(a.r)) for a in pls[0].parts() if isinstance(a, Arc))
    assert centres == [(0, 0, 14), (0, 50, 14)]


def test_two_arcs_of_one_circle_become_one_arc():
    pls, lines, arcs = join([Line(-10, 0, 0, 0), Line(10, 0, 20, 0)], [Arc(5, 0, 5, 180, 270), Arc(5, 0, 5, 270, 360)])
    assert len(pls) == 1 and not lines and not arcs
    curved = [q for q in pls[0].parts() if isinstance(q, Arc)]
    assert len(curved) == 1 and (curved[0].end - curved[0].start) % 360 == pytest.approx(180, abs=0.5)


def pipe(*points) -> Polyline:
    return Polyline(list(points), [0.0] * (len(points) - 1), kind="pipe")


def test_a_pipe_stopping_short_of_a_junction_or_circle_is_joined_to_it():
    d = Drawing(100, 100, polylines=[pipe((0, 50), (60, 50)), pipe((30, 51.2), (30, 80)), pipe((61, 50), (90, 50))],
                circles=[Circle(70, 20, 5)], lines=[Line(70, 78.8, 70, 26.4)])
    connect(d)
    ends = {pl.points[0] for pl in d.polylines} | {pl.points[-1] for pl in d.polylines}
    assert (30, 50) in ends  # the branch now meets the pipe it starts at
    assert any(math.isclose(ln.y2, 25) for ln in d.lines)  # the signal line reaches the circle
    assert [pl.points for pl in d.polylines if pl.points[-1][1] == 50] == [[(0, 50), (90, 50)]]  # one pipe again


def test_two_loose_ends_close_their_corner_and_become_one_polyline():
    d = Drawing(100, 100, polylines=[pipe((0, 0), (48.8, 0)), pipe((50, 1.5), (50, 40))])
    connect(d)
    assert [pl.points for pl in d.polylines] == [[(0, 0), (50, 0), (50, 40)]]


def test_a_pipe_drawn_broken_where_it_crosses_another_is_one_pipe_across_it_without_a_junction():
    d = Drawing(100, 100, polylines=[pipe((0, 50), (90, 50)), pipe((40, 0), (40, 47)), pipe((40, 53), (40, 90))])
    connect(d)
    assert sorted(pl.points for pl in d.polylines) == [[(0, 50), (90, 50)], [(40, 0), (40, 90)]]


def test_crumbs_that_touch_nothing_are_dropped_but_a_flange_mark_stays():
    d = Drawing(100, 100, polylines=[pipe((0, 0), (40, 0))],
                lines=[Line(20, -1, 20, 1), Line(70, 70, 71.2, 70.5)])
    connect(d)
    assert d.lines == [Line(20, -1, 20, 1)]


def test_a_valve_whose_short_pipe_to_the_junction_was_lost_is_piped_to_it():
    valve = place(load()["GATE_VALVE"], 60, 20)  # its ports at x 57.5 and 62.5
    d = Drawing(100, 100, polylines=[pipe((52, 0), (52, 40)), pipe((62.5, 20), (90, 20))], symbols=[valve])
    connect(d)
    assert [(57.5, 20), (52, 20)] in [pl.points for pl in d.polylines]
