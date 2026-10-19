import numpy as np
import pytest

from app.plan2cad.model import Circle, Drawing, Line
from app.plan2cad.render import render
from app.plan2cad.vectorize import _circle_fit, axis_lines, circles, close_corners, merge_lines


def test_regular_dashes_merge_into_one_dashed_line():
    out = merge_lines([Line(x, 10, x + 3, 10.1) for x in np.arange(0, 45, 4.5)])
    assert len(out) == 1
    assert out[0].dashed
    assert (out[0].x1, out[0].x2) == pytest.approx((0, 43.5))


def test_broken_solid_line_stays_solid():
    out = merge_lines([Line(0, 5, 20, 5), Line(20.4, 5, 41, 5), Line(43, 5, 44, 5)])
    assert [ln.dashed for ln in out] == [False, False]
    assert (out[0].x1, out[0].x2) == pytest.approx((0, 41))


def test_parallel_lines_are_not_merged():
    assert len(merge_lines([Line(0, 0, 10, 0), Line(0, 1.4, 10, 1.4)])) == 2


def test_close_corners_snaps_ends_onto_perpendicular_lines():
    h, v = close_corners([Line(0.6, 0, 10, 0), Line(0, -0.4, 0, 10)])
    assert (h.x1, v.y1) == (0, 0)


def test_circle_fit():
    t = np.linspace(0, 2 * np.pi, 100)
    cx, cy, r, rms = _circle_fit(np.column_stack([5 + 3 * np.cos(t), 7 + 3 * np.sin(t)]))
    assert (cx, cy, r, rms) == pytest.approx((5, 7, 3, 0), abs=1e-6)


def test_lines_and_circle_from_raster():
    d = Drawing(100, 60, lines=[Line(10, 10, 90, 10, 0.35), Line(20, 15, 20, 55, 0.5)], circles=[Circle(60, 35, 6, 0.35)])
    ink = (render(d, 10) < 128).astype(np.uint8)
    lines, _, _ = axis_lines(ink, 10)
    ends = sorted((round(ln.x1), round(ln.y1), round(ln.x2), round(ln.y2)) for ln in lines)
    assert ends == [(10, 10, 90, 10), (20, 15, 20, 55)]
    found, _ = circles(ink, 10)
    assert len(found) == 1
    assert (found[0].cx, found[0].cy, found[0].r) == pytest.approx((60, 35, 6), abs=0.15)


def test_circle_survives_when_part_of_its_ring_was_read_as_text():
    ink = (render(Drawing(40, 40, circles=[Circle(20, 20, 6, 0.5)]), 10) < 128).astype(np.uint8)
    text = np.zeros_like(ink)
    text[:160] = ink[:160]  # the top quarter of the ring, as when OCR reads symbol arcs as letters
    found, _ = circles(ink, 10, text)
    assert len(found) == 1
    assert found[0].r == pytest.approx(6, abs=0.15)
