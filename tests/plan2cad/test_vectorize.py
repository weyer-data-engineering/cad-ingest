import math

import cv2
import numpy as np
import pytest

from app.plan2cad.model import Arc, Circle, Drawing, Line
from app.plan2cad.render import render
from app.plan2cad.vectorize import (_circle_fit, _pieces, axis_lines, bridge, circles, close_corners, joints, links,
                                    merge_arcs, merge_lines)


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


def test_rounded_corner_becomes_line_arc_line():
    ppm, r = 8.0, 40.0  # 5 mm radius
    t = np.linspace(0, np.pi / 2, 60)[1:]
    pts = np.vstack([np.column_stack([np.arange(0, 200.0), np.full(200, 100.0)]),
                     np.column_stack([200 + r * np.sin(t), 140 - r * np.cos(t)]),
                     np.column_stack([np.full(158, 240.0), np.arange(141, 299.0)])])
    pts += np.random.default_rng(1).normal(0, 0.3, pts.shape)
    out = _pieces(pts, False, ppm)
    assert [type(q).__name__ for q in out] == ["Line", "Arc", "Line"]
    assert out[1].r / ppm == pytest.approx(5, abs=0.2)


def test_closed_box_with_uneven_corners_keeps_its_sides():
    corners = [(0, 2), (1, 0), (40, 0), (40, 38), (38, 40), (2, 40), (0, 38), (0, 2)]  # a hand-drawn 5 mm box
    pts = np.vstack([np.linspace(p, q, 40, endpoint=False) for p, q in zip(corners, corners[1:])] + [[corners[0]]])
    out = _pieces(pts, True, 8.0)
    assert sum(isinstance(q, Line) and math.dist((q.x1, q.y1), (q.x2, q.y2)) > 30 for q in out) == 4


def test_half_ellipse_is_a_few_arcs_not_a_chain_of_lines():
    t = np.linspace(np.pi, 2 * np.pi, 400)
    pts = np.column_stack([300 + 170 * np.cos(t), 300 + 50 * np.sin(t)])
    pts += np.random.default_rng(2).normal(0, 0.3, pts.shape)
    out = _pieces(pts, False, 8.0)
    assert len(out) <= 5
    assert sum(isinstance(q, Arc) for q in out) >= 3


def test_bridge_closes_a_faint_stretch_but_not_a_gap_of_paper():
    ppm = 10
    norm = np.full((200, 400), 255, np.uint8)
    norm[50, 20:380] = 60  # a line with a faint stretch
    norm[50, 150:190] = 170
    norm[120, 20:380] = 60  # a line with a real gap
    norm[120, 150:190] = 255
    lines = [Line(2, 5, 15, 5), Line(19, 5, 38, 5), Line(2, 12, 15, 12), Line(19, 12, 38, 12)]
    out = bridge(lines, [], norm, ppm)
    assert sorted((ln.y1, round(ln.x1), round(ln.x2)) for ln in out) == [(5, 2, 38), (12, 2, 15), (12, 19, 38)]


def test_bridge_extends_a_branch_onto_the_pipe_it_meets_where_the_scan_shows_it():
    ppm = 10
    norm = np.full((200, 400), 255, np.uint8)
    norm[50, 20:380] = 60  # a pipe
    norm[50:150, 100] = 60  # a branch whose top stretch came out too faint
    norm[50:70, 100] = 170
    norm[50:150, 300] = 60  # a line that stops short of the pipe on paper
    norm[50:70, 300] = 255
    lines = [Line(2, 5, 38, 5), Line(10, 15, 10, 7), Line(30, 15, 30, 7)]
    out = bridge(lines, [], norm, ppm)
    assert [(ln.x1, ln.y1, ln.x2, round(ln.y2, 2)) for ln in out[1:]] == [(10, 15, 10, 5.0), (30, 15, 30, 7)]


def test_arcs_of_one_circle_split_by_a_line_become_one_arc_but_not_across_a_wide_gap():
    gap = math.degrees(1.0 / 5)  # 1 mm on a radius of 5 mm
    arcs, circles = merge_arcs([Arc(50, 50, 5, 180, 270 - gap / 2), Arc(50.1, 50, 5.05, 270 + gap / 2, 360),
                                Arc(10, 10, 5, 0, 60), Arc(10, 10, 5, 140, 180)])
    assert not circles
    assert sorted((round(a.cx), round(a.start), round(a.end)) for a in arcs) == [(10, 0, 60), (10, 140, 180),
                                                                                (50, 180, 0)]


def test_bridge_joins_a_dotted_line_but_not_the_two_sides_of_a_valve():
    ppm = 10
    norm = np.full((200, 400), 255, np.uint8)
    norm[50, 20:180] = 170  # a faint line that came out as dots
    norm[50, [35, 55, 75, 95, 115, 135, 155]] = 60
    norm[120, 200:260] = 60  # two short strokes meeting in the middle, like the sides of a valve
    dots = [Line(2 + 2 * k, 5, 3.4 + 2 * k, 5) for k in range(8)]
    out = bridge(dots + [Line(20, 12, 22.9, 12), Line(23.1, 12, 26, 12)], [], norm, ppm)
    assert sorted((round(ln.x1, 1), round(ln.x2, 1)) for ln in out) == [(2.0, 17.4), (20.0, 22.9), (23.1, 26.0)]


def test_merge_lines_keeps_parallel_lines_at_their_places():
    # a pipe running into the box of its line number, with short strokes of the letters in between: offsets that
    # chain from one line to the next must not pull the lines onto one averaged line
    box = [Line(10, 5.1, 26, 5.1), Line(10, 7.4, 26, 7.4)]
    pipe = [Line(0, 6.3, 9.5, 6.3), Line(26.5, 6.3, 40, 6.3)]
    letters = [Line(12 + k, 5.4 + 0.3 * k, 12.3 + k, 5.4 + 0.3 * k) for k in range(7)]
    out = merge_lines(box + pipe + letters)
    ys = sorted({round(ln.y1, 1) for ln in out if abs(ln.x2 - ln.x1) > 5})
    assert ys == [5.1, 6.3, 7.4]


def test_a_short_dashed_link_between_two_symbols_becomes_one_dashed_line():
    # two dashes from an instrument bubble to the side of a box: too few for merge_lines to call them dashed
    dashes = [Line(14.0, 10, 14.8, 10), Line(16.0, 10, 16.8, 10)]
    box, bubble = Line(18.5, 5, 18.5, 15), Circle(10, 10, 3)
    norm = render(Drawing(30, 20, lines=dashes + [box], circles=[bubble]), 10)
    out = links(dashes + [box], [], [bubble], norm, 10)
    link = [ln for ln in out if ln.dashed]
    assert len(out) == 2 and len(link) == 1
    assert (link[0].x1, link[0].x2, link[0].y1) == pytest.approx((13, 18.5, 10), abs=0.1)


def test_dashes_that_lead_nowhere_and_pipe_ends_at_a_flange_stay_as_they_are():
    dashes = [Line(14.0, 10, 14.8, 10), Line(16.0, 10, 16.8, 10)]
    norm = render(Drawing(30, 20, lines=dashes), 10)
    assert links(list(dashes), [], [], norm, 10) == dashes
    # two short pipe pieces between two strokes, with the bars of a flange across their ends
    pipe = [Line(10, 10, 12.5, 10), Line(13.3, 10, 15.8, 10), Line(9.5, 7, 9.5, 13), Line(16.3, 7, 16.3, 13)]
    flange = [Line(12.5, 9.2, 12.5, 10.8), Line(13.3, 9.2, 13.3, 10.8)]
    norm = render(Drawing(30, 20, lines=pipe + flange), 10)
    assert not any(ln.dashed for ln in links(pipe + flange, [], [], norm, 10))


def test_a_letter_touching_the_round_end_of_a_box_is_read_but_not_the_round_end():
    box = [Line(10, 10, 30, 10, 0.35), Line(10, 13, 30, 13, 0.35)]
    end = Arc(30, 11.5, 1.5, 270, 90, 0.35)
    letter = Line(29.7, 10.8, 29.7, 12.2, 0.35)  # the stem of a letter that runs into the round end
    ink = (render(Drawing(40, 20, lines=box + [letter], arcs=[end]), 10) < 128).astype(np.uint8)
    hv, used, _ = axis_lines(ink, 10)
    rem = ink & (1 - cv2.dilate(used, np.ones((3, 3), np.uint8)))
    mask = joints(rem, hv, 10)
    assert mask[110:120, 312:317].any()  # the round end
    assert not mask[110:120, 295:299].any()  # the letter
