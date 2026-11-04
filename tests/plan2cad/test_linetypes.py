import math
import time
from dataclasses import replace

import cv2
import numpy as np
import pytest
from ezdxf.math import fit_points_to_cad_cv
from scipy.spatial import cKDTree

from app.plan2cad import objects, vectorize
from app.plan2cad.export import to_dxf
from app.plan2cad.features import Features
from app.plan2cad.model import Arc, Dimension, Drawing, Line, Polygon, Polyline, line_type
from app.plan2cad.objects import curves
from app.plan2cad.render import dashes, render
from app.plan2cad.symbols import _split
from app.plan2cad.vectorize import calibrate_widths, dash_dots, dotted, line_types, merge_lines


def _sheet(*lines: Line, ppm: float = 10) -> np.ndarray:
    """The grey image of the lines on paper (``norm`` of the pipeline)."""
    return render(Drawing(200, 60, lines=list(lines)), ppm)


def test_the_pattern_along_a_dashed_line_tells_its_type_and_repeat():
    dashed, dashdot, dot = (Line(10, y, 190, y, 0.35, True, linetype=t) for y, t in
                            ((10, None), (25, "dashdot"), (40, "dotted")))
    norm = _sheet(dashed, dashdot, dot)
    found = line_types([Line(10, 10, 190, 10, 0.35, True), Line(10, 25, 190, 25, 0.35, True),
                        Line(10, 40, 190, 40, 0.35, True)], norm, 10)
    assert [q.linetype for q in found] == ["dashed", "dashdot", "dotted"]
    assert [q.pattern for q in found] == pytest.approx([4.5, 10.5, 1.5], abs=0.2)


def test_a_line_too_faint_to_tell_stays_dashed_and_solid_lines_are_left_alone():
    found = line_types([Line(10, 10, 190, 10, 0.35, True), Line(10, 30, 190, 30)], _sheet(), 10)
    assert [(line_type(q), q.pattern) for q in found] == [("dashed", None), ("continuous", None)]


def test_long_dashes_with_a_dot_between_become_one_dash_dot_line_but_a_line_crossing_the_gaps_is_no_dot():
    axis = Line(10, 20, 190, 20, 0.25, True, linetype="dashdot")
    pieces = [Line(a, 20, b, 20, 0.25) for (a, _), (b, _) in (seg for seg in dashes(axis) if seg[1][0] - seg[0][0] > 1)]
    out = dash_dots(list(pieces), _sheet(axis), 10)
    assert len(out) == 1 and (out[0].linetype, round(out[0].pattern, 1)) == ("dashdot", 10.5)
    assert (out[0].x1, out[0].x2) == pytest.approx((10, 190), abs=1.0)
    broken = [Line(10 + 12 * k, 40, 19 + 12 * k, 40, 0.25) for k in range(6)]  # with paths crossing the gaps
    across = [Line(20.5 + 12 * k, 30, 20.5 + 12 * k, 50, 0.25) for k in range(5)]
    assert not any(q.dashed for q in dash_dots(broken, _sheet(*broken, *across), 10))


def test_a_row_of_dots_is_a_dotted_line_but_a_grid_of_dots_is_not():
    ppm = 10
    row = Line(20, 20, 120, 20, 0.35, True, linetype="dotted")
    grid = [Line(140, y, 180, y, 0.35, True, linetype="dotted") for y in np.arange(30, 50, 1.5)]
    norm = _sheet(row, *grid, ppm=ppm)
    found, used = dotted((norm < 128).astype(np.uint8), norm, ppm)
    assert len(found) == 1 and found[0].linetype == "dotted" and found[0].pattern == pytest.approx(1.5, abs=0.05)
    assert (found[0].x1, found[0].x2) == pytest.approx((20, 120), abs=1.6) and used[195:206, 300:1100].any()


def test_dash_dot_and_dotted_lines_are_found_at_any_angle():
    for deg in (30, 63):
        u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        axis = Line(20, 20, *(np.array([20, 20]) + 150 * u), 0.25, True, linetype="dashdot")
        norm = render(Drawing(200, 160, lines=[axis]), 10)
        pieces = [Line(*a, *b, 0.25) for a, b in dashes(axis) if math.dist(a, b) > 1]
        assert [(q.linetype, round(q.pattern, 1)) for q in dash_dots(pieces, norm, 10)] == [("dashdot", 10.5)]
        row = Line(40, 10, *(np.array([40, 10]) + 100 * u), 0.35, True, linetype="dotted")
        norm = render(Drawing(200, 160, lines=[row]), 10)
        found = dotted((norm < 128).astype(np.uint8), norm, 10)[0]
        assert len(found) == 1 and found[0].pattern == pytest.approx(1.5, abs=0.05)
        assert math.dist((found[0].x2, found[0].y2), (row.x2, row.y2)) < 1.6


def test_regular_dashes_still_merge_into_a_dashed_line():
    out = merge_lines([Line(x, 10, x + 3, 10.1) for x in np.arange(0, 45, 4.5)])
    assert len(out) == 1 and out[0].dashed and line_type(out[0]) == "dashed"


def test_dashes_follow_the_line_type_and_the_measured_pattern():
    assert len(dashes(Line(0, 0, 45, 0, dashed=True))) == 10  # 3 mm dashes every 4.5 mm, as before
    dots = dashes(Line(0, 0, 15, 0, dashed=True, linetype="dotted"))
    assert len(dots) == 10 and all(b[0] - a[0] < 0.1 for a, b in dots)
    chain = dashes(Line(0, 0, 21, 0, dashed=True, linetype="dashdot", pattern=21.0))  # twice as long as usual
    assert [round(b[0] - a[0], 2) for a, b in chain] == [15.0, 0.05]


def test_dxf_has_the_line_types_scaled_to_the_measured_pattern_and_splines():
    d = Drawing(420, 297, lines=[Line(10, 20, 110, 20, 0.25, True, linetype="dashdot", pattern=21.0),
                                 Line(10, 40, 110, 40, 0.25, True),
                                 Line(10, 50, 110, 50, 0.25, True, linetype="dashed")],
                polylines=[Polyline([(10, 60), (20, 62), (30, 61), (40, 64)], [0.0] * 3, smooth=True)], scale=50)
    doc = to_dxf(d)
    lines = doc.modelspace().query("LINE")  # a dashed line never measured is drawn as before the line types
    assert [(e.dxf.linetype, round(e.dxf.ltscale, 2)) for e in lines] == [("DASHDOT", 2.0), ("P2C_DASH", 1.0),
                                                                          ("DASHED", 1.0)]
    assert doc.linetypes.get("DASHDOT").pattern_tags.tags  # defined in the document
    spline = doc.modelspace().query("SPLINE")[0]
    assert spline.dxf.layer == "P2C-CURVE" and len(spline.fit_points) == 4
    assert not doc.audit().has_errors


def test_widths_are_measured_on_the_grey_image_not_widened_by_blur():
    import cv2

    ppm = 10
    d = Drawing(200, 60, lines=[Line(20, 20, 180, 20, 0.7), Line(20, 40, 180, 40, 0.3)])
    norm = cv2.GaussianBlur(render(d, ppm), (0, 0), 1.5)  # blur spreads the ink, keeps its sum
    found = Drawing(200, 60, lines=[Line(20, 20, 180, 20, 1.2), Line(20, 40, 180, 40, 0.8)])  # as the binary showed
    calibrate_widths(found, norm, ppm)
    ink = [float((255 - render(Drawing(200, 60, lines=[q]), ppm)[:, 1000].astype(float)).sum() / 255 / ppm)
           for q in d.lines]  # what the renderer printed of each pen
    assert [q.width for q in found.lines] == pytest.approx(ink, rel=0.15)


def test_the_pens_of_dimensions_are_measured_too_and_their_terminators_take_them():
    ppm = 10
    line, tick = Line(20, 30, 180, 30, 0.25), Line(179, 31, 181, 29, 0.5)  # a chain of two dimensions
    extensions = [Line(100, 18, 100, 33, 0.35), Line(180, 18, 180, 33, 0.35)]
    sides = [Line(20, 30, 23, 29, 0.25), Line(20, 30, 23, 31, 0.25)]  # an open arrowhead at its start
    dot = Polygon([(100 + 0.6 * math.cos(a), 30 + 0.6 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 12, False)])
    wall = Line(20, 40, 180, 40, 0.7)  # whose ink shows how dark the ink is
    norm = cv2.GaussianBlur(render(Drawing(200, 60, lines=[line, tick, *extensions, *sides, wall], polygons=[dot]),
                                   ppm), (0, 0), 1.5)
    found = [replace(q, width=0.6) for q in (line, *extensions, *sides, tick)]  # as the binary image showed them
    first = Dimension(20, 30, 100, 30, "8,00", (20, 30), (100, 18), "arrow", 3.0, width=0.6,
                      parts=[found[0], found[1], found[3], found[4], dot])
    second = Dimension(100, 30, 180, 30, "8,00", (100, 18), (180, 18), "tick", 2.0, width=0.6,
                       parts=[found[2], found[5]])
    calibrate_widths(Drawing(200, 60, dimensions=[first, second]), norm, ppm)
    ink = [float((255 - render(Drawing(200, 60, lines=[replace(line, width=w)]), ppm)[:, 1000].astype(float)).sum()
                 / 255 / ppm) for w in (0.25, 0.35)]  # what the renderer printed of each pen
    assert [first.width, second.width, found[0].width] == pytest.approx([ink[0]] * 3, rel=0.15)
    assert [found[1].width, found[2].width] == pytest.approx([ink[1]] * 2, rel=0.15)  # the extension lines
    assert [q.width for q in found[3:]] == [first.width, first.width, second.width] and dot.width == 0.25


def test_a_smooth_run_of_pieces_on_one_circle_becomes_one_arc_and_a_contour_a_spline():
    r, c = 300.0, (200.0, 400.0)

    def at(deg):
        return c[0] + r * math.cos(math.radians(deg)), c[1] + r * math.sin(math.radians(deg))

    street = Polyline([at(a) for a in range(250, 291, 5)], [0.0] * 8, width=0.5)  # a street as short chords
    xs = np.linspace(30, 395, 60)
    contour = Polyline(list(zip(xs.tolist(), (50 + 6 * np.sin(xs / 40)).tolist())), [0.0] * 59, width=0.18)
    pipe = Polyline([(0, 0), (10, 0), (10, 10), (20, 10)], [0.0] * 3, kind="pipe")
    bend = Polyline([(0, 100), (10, 100), (10, 110), (20, 110)], [0.0] * 3)  # corners, not a curve
    d = Drawing(420, 297, polylines=[street, contour, pipe, bend])
    curves(d)
    assert len(d.arcs) == 1 and d.arcs[0].r == pytest.approx(r, rel=0.01)
    assert [(pl.smooth, pl.kind) for pl in d.polylines] == [(True, "line"), (False, "pipe"), (False, "line")]


def test_the_new_steps_are_features_and_can_be_switched_off():
    off = Features.of(off=["linetypes", "width_calibration", "splines"])
    assert Features().linetypes and Features().width_calibration and Features().splines
    assert not (off.linetypes or off.width_calibration or off.splines)


def test_line_type_of_lines_polylines_and_arcs():
    assert [line_type(q) for q in (Line(0, 0, 1, 0), Line(0, 0, 1, 0, dashed=True),
                                   Polyline([(0, 0), (1, 0)], [0.0], dashed=True, linetype="dotted"),
                                   Arc(0, 0, 1, 0, 90))] == ["continuous", "dashed", "dotted", "continuous"]


def test_the_benchmark_counts_line_types_found_right_and_splines_apart_from_arcs():
    from evaluate import continuity
    from evaluate import line_types as measured

    ref = Drawing(200, 60, lines=[Line(10, 10, 190, 10, dashed=True, linetype="dashdot"),
                                  Line(10, 30, 190, 30, dashed=True, linetype="dotted"), Line(10, 50, 190, 50)])
    out = Drawing(200, 60, lines=[Line(10, 10.3, 190, 10.3, dashed=True, linetype="dashdot"),
                                  Line(10, 30, 190, 30, dashed=True)])  # the dotted line taken for a dashed one
    assert measured(ref, out) == {"linetype_correct": 0.5, "linetype_precision": 1.0}
    assert measured(ref, Drawing(200, 60)) == {"linetype_correct": 0.0}
    specks = Drawing(200, 60, lines=[Line(10, 50, 100, 50, dashed=True, linetype="dotted")])  # on a solid line
    assert measured(Drawing(200, 60), out) == {"linetype_precision": 0.0} and measured(ref, specks) == {
        "linetype_correct": 0.0, "linetype_precision": 0.0}
    t = np.radians(np.linspace(200, 340, 30))
    arc = Drawing(200, 60, arcs=[Arc(100, 60, 40, 200, 340)])
    spline = Polyline([(100 + 40 * math.cos(a), 60 + 40 * math.sin(a)) for a in t], [0.0] * 29, smooth=True)
    m = continuity(arc, Drawing(200, 60, polylines=[spline]))
    assert (m["curves_as_arcs"], m["curves_as_splines"]) == (0.0, 1.0)  # a spline is no true arc
    m = continuity(arc, Drawing(200, 60, arcs=[Arc(100, 60, 40, 200, 340)]))
    assert (m["curves_as_arcs"], m["curves_as_splines"]) == (1.0, 0.0)


def test_dotted_lines_are_found_fast_in_dense_fields_of_dots():
    ppm = 300 / 25.4
    rng = np.random.default_rng(0)
    ink = np.zeros((round(60 * ppm), round(200 * ppm)), np.uint8)
    for x, y in zip(rng.integers(0, round(100 * ppm), 3000), rng.integers(0, round(30 * ppm), 3000)):
        ink[y:y + 2, x:x + 2] = 1  # specks, 1 per mm2
    for y in np.arange(0, 25, 0.34):  # a screened area: a dot every 0.34 mm
        for x in np.arange(110, 140, 0.34):
            ink[round(y * ppm):round(y * ppm) + 2, round(x * ppm):round(x * ppm) + 2] = 1
    norm = np.where(ink > 0, 0, 255).astype(np.uint8)
    start = time.perf_counter()
    found, _ = dotted(ink, norm, ppm)
    assert time.perf_counter() - start < 3.0  # was minutes, as the rows were joined over and over
    assert not any(q.x1 > 105 for q in found)  # the screen is no line


def test_rows_of_uneven_specks_or_with_ink_between_are_no_dotted_lines():
    ppm = 10
    row = Line(20, 20, 120, 20, 0.35, True, linetype="dotted")
    broken = cv2.min(_sheet(row, ppm=ppm), np.where(_sheet(Line(20, 20, 120, 20, 0.35), ppm=ppm) < 128, 150, 255)
                     .astype(np.uint8))  # a solid line whose ink broke up: a faint stroke joins the specks
    assert not dotted((broken < 128).astype(np.uint8), broken, ppm)[0]
    uneven = np.full((600, 2000), 255, np.uint8)
    for k, s in enumerate([1, 3, 1, 4, 1, 3, 1, 4, 1, 3, 1, 4]):  # specks of noise, 1.5 mm apart
        uneven[200:200 + s, 200 + 15 * k:200 + 15 * k + s] = 0
    assert not dotted((uneven < 128).astype(np.uint8), uneven, ppm)[0]
    norm = _sheet(row, ppm=ppm)
    assert len(dotted((norm < 128).astype(np.uint8), norm, ppm)[0]) == 1


def test_the_dashes_of_a_small_print_as_short_as_dots_give_its_repeat():
    ppm = 11.3  # a dashed line printed at a quarter: 0.75 mm dashes every 1.13 mm on the sheet
    for blur in (0.6, 1.2):
        norm = cv2.GaussianBlur(_sheet(Line(10, 20, 190, 20, 0.1, True, pattern=1.125), ppm=ppm), (0, 0), blur)
        found = line_types([Line(10, 20, 190, 20, 0.1, True)], norm, ppm)[0]
        assert (found.linetype, found.pattern) == ("dashed", pytest.approx(1.125, abs=0.1))


def test_a_straight_line_with_a_jog_and_a_road_running_into_a_bend_are_no_splines():
    jog = Polyline([(0, 0), (4.5, 0), (5.57, 0.25), (10.07, 0.27)], [0.0] * 3)  # turns -13 and +13 degrees
    zigzag = Polyline([(20 * k, 50 + 0.5 * (k % 2)) for k in range(7)], [0.0] * 6)
    r, sweep = 300.0, math.radians(20)  # a straight road, a tangent bend of 300 mm radius and a straight road
    end = (50 + r * math.sin(sweep), 100 + r - r * math.cos(sweep))
    road = Polyline([(0, 100), (50, 100), end, (end[0] + 50 * math.cos(sweep), end[1] + 50 * math.sin(sweep))],
                    [0.0, -math.tan(sweep / 4), 0.0])
    d = Drawing(420, 297, polylines=[jog, zigzag, road])
    curves(d)
    assert not d.arcs and [(pl.points, pl.smooth) for pl in d.polylines] == [
        (jog.points, False), (zigzag.points, False), (road.points, False)]


def test_a_repeated_point_or_a_fit_that_fails_keeps_the_conversion_going(monkeypatch):
    xs = [0, 5, 10, 10, 15, 20, 25, 30, 35, 40]  # a piece of no length
    wave = Polyline([(x, 50 + 2 * math.sin(x / 10)) for x in xs], [0.0] * 9)
    d = Drawing(100, 100, polylines=[wave])
    curves(d)
    spline = d.polylines[0]
    assert spline.smooth and len(set(spline.points)) == len(spline.points)
    dxf = np.array([(v.x, v.y) for v in fit_points_to_cad_cv(spline.points).approximate(2000)])
    assert cKDTree(dxf).query(objects._path(spline, 0.05))[0].max() < 0.08  # the preview draws what the DXF has

    def singular(points):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(objects, "fit_points_to_cad_cv", singular)
    d = Drawing(100, 100, polylines=[wave])
    curves(d)
    assert d.polylines == [wave]


def test_a_polyline_cut_by_a_symbol_keeps_its_line_type():
    pl = Polyline([(0, 0), (10, 0), (20, 0), (30, 0)], [0.0] * 3, dashed=True, color=(0, 0, 255), linetype="dashdot",
                  pattern=21.0)
    parts = _split(pl, {1})
    assert [(p.points, p.linetype, p.pattern, p.color) for p in parts] == [
        ([(0, 0), (10, 0)], "dashdot", 21.0, (0, 0, 255)), ([(20, 0), (30, 0)], "dashdot", 21.0, (0, 0, 255))]


def test_a_hatching_line_with_a_jog_where_an_axis_crosses_it_is_no_arc():
    # as traced on a phone photo of a section: bent towards the axis and back (turns of +9 and -9 degrees), which a
    # circle of 36 mm radius fits within the stroke, and so became an arc that lost the crossing
    hatch = Polyline([(277.79, 103.99), (275.53, 106.40), (274.25, 107.40), (270.57, 111.29)], [0.0] * 3, width=0.34)
    d = Drawing(420, 297, lines=[Line(274.36, 95.0, 274.36, 121.6, 0.5, True)], polylines=[replace(hatch)])
    curves(d)
    assert not d.arcs and d.polylines == [hatch]
    assert abs(objects._turning(objects._steps(hatch))) < math.radians(1)  # it turns to and fro, hardly at all


def test_an_arc_keeps_an_end_where_the_polyline_is_joined_and_a_free_end_may_move():
    r, c = 300.0, (200.0, 400.0)

    def at(deg, dr=0.0):
        return c[0] + (r + dr) * math.cos(math.radians(deg)), c[1] + (r + dr) * math.sin(math.radians(deg))

    # a street as chords, its last point pulled 0.3 mm in where it ends on a boundary across it
    street = Polyline([at(a) for a in range(250, 271, 4)] + [at(272, -0.3)], [0.0] * 6, width=0.5)
    first, last = street.points[0], street.points[-1]
    u = (math.cos(math.radians(272)), math.sin(math.radians(272)))
    boundary = Line(last[0] - 20 * u[0], last[1] - 20 * u[1], last[0] + 20 * u[0], last[1] + 20 * u[1], 0.35)

    def ends(lines: list[Line]) -> list[float]:
        d = Drawing(420, 297, lines=lines, polylines=[replace(street, points=list(street.points))])
        curves(d)
        (a,) = d.arcs
        tips = [(a.cx + a.r * math.cos(math.radians(t)), a.cy + a.r * math.sin(math.radians(t)))
                for t in (a.start, a.end)]
        return [min(math.dist(p, q) for q in tips) for p in (first, last)]

    free = ends([])
    assert 0.05 < max(free) < 0.25  # the circle that fits best, as before: both ends move a little onto it
    joined = ends([boundary])
    assert joined[1] < 1e-6 and joined[0] < 0.35  # the joined end stays on the boundary, the free one moves


def test_the_circle_through_fixed_points_passes_through_them_and_fits_the_rest():
    a = np.radians(np.linspace(200, 240, 30))
    pts = np.column_stack([100 + 50 * np.cos(a), 80 + 50 * np.sin(a)]) + np.random.default_rng(1).normal(0, 0.05,
                                                                                                         (30, 2))
    free = objects._circle_through(pts, pts[:0])
    assert free == pytest.approx(vectorize._circle_fit(pts)[:3], rel=1e-6)  # without fixed points: the best fit
    cx, cy, rad = objects._circle_through(pts, pts[[0, -1]])
    assert [math.hypot(x - cx, y - cy) for x, y in pts[[0, -1]]] == pytest.approx([rad, rad], abs=1e-9)
    assert rad == pytest.approx(50, abs=1.0)
