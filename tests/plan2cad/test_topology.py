import pytest

from app.plan2cad.model import Arc, Line
from app.plan2cad.topology import join


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
