import io

import cv2
import ezdxf
import numpy as np
import pytest
from synth import Pen

from app import scan
from app.plan2cad import symbols
from app.plan2cad.export import to_dxf
from app.plan2cad.library import FLOOR_PLAN, combine, load, place, read
from app.plan2cad.model import Arc, Circle, Drawing, Line, Polyline, Text
from app.plan2cad.pipeline import convert
from app.plan2cad.render import render

from .conftest import write_library

LIBRARY = load()
FIXTURES = combine([read(FLOOR_PLAN)], "building", 50)  # floor plan fixtures on a 1:50 sheet
WALL = Polyline([(40, 80), (160, 80), (160, 82.4), (40, 82.4)], [0.0] * 4, closed=True, width=0.5, kind="wall")  # 24 cm at 1:100


def drawn(*placed, lines=(), texts=()) -> Drawing:
    """A drawing holding the strokes of the placed symbols as plain lines, arcs and circles, plus ``lines``."""
    strokes = Drawing(200, 100, symbols=list(placed)).flat()
    return Drawing(200, 100, lines=strokes.lines + list(lines), arcs=strokes.arcs, circles=strokes.circles,
                   texts=list(texts))


def test_library_has_symbols_with_ports_attributes_and_through_lines():
    assert {"GATE_VALVE", "BALL_VALVE", "PUMP", "INSTRUMENT", "FLOW_ARROW", "OFF_PAGE_CONNECTOR"} <= set(LIBRARY)
    pump = LIBRARY["PUMP"]
    assert pump.ports == [(-6.5, 0.0), (0.0, -6.5)] and "TAG" in pump.tags  # y points down, as on the sheet
    assert LIBRARY["FLOW_ARROW"].through and not LIBRARY["GATE_VALVE"].through  # an arrow sits on a pipe


def test_place_turns_clockwise_and_mirrors():
    assert place(LIBRARY["PUMP"], 100, 50, angle=90).ports == pytest.approx([(100, 43.5), (106.5, 50)])
    assert place(LIBRARY["PUMP"], 100, 50, mirror=True).ports[0] == pytest.approx((106.5, 50))


def test_valves_in_a_pipe_become_blocks_joined_by_the_pipe():
    gate, ball = place(LIBRARY["GATE_VALVE"], 100, 50), place(LIBRARY["BALL_VALVE"], 130, 50)
    pipes = [Line(60, 50, 97.5, 50, 0.5), Line(102.5, 50, 127.5, 50, 0.5), Line(132.5, 50, 170, 50, 0.5)]
    d = drawn(gate, ball, lines=pipes)
    symbols.recognise(d, LIBRARY)
    assert sorted((s.name, round(s.x), round(s.y)) for s in d.symbols) == [("BALL_VALVE", 130, 50),
                                                                          ("GATE_VALVE", 100, 50)]
    assert len(d.lines) == 3 and not d.circles  # the strokes are gone, the pipes stay
    assert len(symbols.connections(d)) == 1  # the middle pipe joins the two valves


def test_strokes_left_inside_a_symbol_do_not_join_it_to_itself():
    pipes = [Line(60, 50, 97.5, 50, 0.5), Line(102.5, 50, 140, 50, 0.5)]
    leftovers = [Line(98.2, 50, 101.6, 50), Line(97.9, 50.4, 98.6, 50.6)]  # across the valve and at one port
    d = drawn(place(LIBRARY["GATE_VALVE"], 100, 50), lines=pipes + leftovers)
    symbols.recognise(d, LIBRARY)
    assert [s.name for s in d.symbols] == ["GATE_VALVE"]
    assert len(d.lines) == 2 and not symbols.connections(d)


def test_instrument_takes_its_texts_as_attributes():
    bubble = place(LIBRARY["INSTRUMENT_DCS"], 100, 50)
    d = drawn(bubble, texts=[Text(100, 47.7, 2, "FIC"), Text(100, 52.4, 2, "101"), Text(150, 50, 2, "Note")])
    symbols.recognise(d, LIBRARY)
    assert [s.name for s in d.symbols] == ["INSTRUMENT_DCS"]
    assert {k: t.text for k, t in d.symbols[0].attributes.items()} == {"FUNCTION": "FIC", "NUMBER": "101"}
    assert [t.text for t in d.texts] == ["Note"]


def test_a_symmetric_symbol_takes_the_pose_with_its_tag_below_it(tmp_path):
    valves = load(write_library(tmp_path / "acme"))  # a box with a bar: the same when turned half way
    turned = place(valves["MY_VALVE"], 100, 50, angle=180)  # what the line ends alone may suggest
    pipes = [Line(60, 50, 97, 50, 0.5), Line(103, 50, 140, 50, 0.5)]
    d = drawn(turned, lines=pipes, texts=[Text(101, 56, 2.5, "V-101")])
    symbols.recognise(d, valves)
    assert [(s.name, s.attributes["TAG"].text) for s in d.symbols] == [("MY_VALVE", "V-101")] and not d.texts


def fixture(name: str):
    return next(t for t in FIXTURES.values() if t.name == name)


def on_wall(*placed) -> Drawing:
    """The strokes of the placed fixtures in a room whose bottom wall has its face along y = 80."""
    wall = Polyline([(40, 80), (160, 80), (160, 87), (40, 87)], [0.0] * 4, closed=True, width=0.5, kind="wall")
    strokes = drawn(*placed)
    return Drawing(200, 100, lines=strokes.lines, arcs=strokes.arcs, circles=strokes.circles, polylines=[wall])


def test_a_washbasin_on_a_wall_is_found_and_the_wall_stays():
    d = on_wall(place(fixture("WASHBASIN"), 100, 80, angle=180))  # its back on the wall face
    wall = list(d.polylines)
    symbols.recognise(d, FIXTURES)
    assert [(s.name, round(s.x), round(s.y), s.angle) for s in d.symbols] == [("WASHBASIN", 100, 80, 180)]
    assert d.polylines == wall and not d.lines and not d.arcs


def test_a_symmetric_fixture_takes_the_pose_with_its_back_on_the_wall():
    d = on_wall(place(fixture("HOB"), 100, 80, angle=180), place(fixture("SHOWER"), 140, 80, angle=180))
    symbols.recognise(d, FIXTURES)  # both look the same turned half way; the wall tells which side is the back
    assert sorted((s.name, round(s.x), round(s.y)) for s in d.symbols) == [("HOB", 100, 80), ("SHOWER", 140, 80)]


def test_two_opposite_sides_make_a_rectangle_where_ocr_took_the_others():
    wall_face, front = (40, 80, 160, 80), (97, 75.5, 103, 75.5)  # a washbasin at 1:100, its sides read as letters
    segments = np.array([wall_face, front], float)
    assert symbols._rectangles(segments, 1, 10, pairs=True).tolist() == [[100, 77.75, 6, 4.5]]
    assert not len(symbols._rectangles(segments, 1, 10))  # otherwise all four sides must be there


def test_a_small_washbasin_read_as_letters_is_found_and_the_letters_dropped():
    basin = combine([read(FLOOR_PLAN)], "building", 100)["WASHBASIN 1:100"]  # 6 x 4.5 mm on the sheet
    d = Drawing(297, 210, scale=100, polylines=[WALL], symbols=[place(basin, 100, 80, angle=180)])
    r = convert(cv2.cvtColor(render(d, 200 / 25.4), cv2.COLOR_GRAY2BGR), paper="A4", dpi=200, drawing="building",
                scale=100)
    assert [(s.name, round(s.x), round(s.y), s.angle) for s in r.drawing.symbols] == [("WASHBASIN", 100, 80, 180)]
    assert not r.drawing.texts  # its bowl and sides, read as "Ol"


def test_a_symbol_far_off_the_size_of_the_others_is_left_out():
    sizes = [1.0] * 12 + [0.6]  # a sheet's symbols are drawn at one size; the small one is something else
    at = [(25 + 50 * (k % 4), 15 + 20 * (k // 4)) for k in range(len(sizes))]
    valves = [place(LIBRARY["GATE_VALVE"], x, y, scale=f) for (x, y), f in zip(at, sizes)]
    pipes = [Line(x + side * 2.5 * f, y, x + side * (2.5 * f + 10), y, 0.5) for (x, y), f in zip(at, sizes)
             for side in (-1, 1)]
    d = drawn(*valves, lines=pipes)
    symbols.recognise(d, LIBRARY)
    assert sorted((round(s.x), round(s.y)) for s in d.symbols) == sorted(at[:12])


def test_lines_without_symbols_stay_lines():
    d = Drawing(200, 100, lines=[Line(10, 10, 60, 10), Line(60, 10, 60, 40), Line(20, 30, 50, 30)])
    symbols.recognise(d, LIBRARY)
    assert not d.symbols and len(d.lines) == 3


def test_a_symbol_at_one_port_is_tried_at_pipe_ends_only():
    pipe, letter = Line(10, 10, 60, 10), Line(80, 10, 82, 13)  # a stroke of a letter ends like a pipe does
    ends = symbols._ends(Drawing(200, 100, lines=[pipe, letter]), bend=45, shortest=symbols.STUB)
    assert sorted(tuple(p) for p, _ in ends) == [(10, 10), (60, 10)]


def test_nothing_is_placed_in_the_title_block():
    frame = Polyline([(10, 10), (287, 10), (287, 200), (10, 200)], [0] * 4, closed=True, kind="frame")
    block = [Line(207, 200, 207, 170), Line(207, 170, 287, 170)]
    pipes = [Line(x, y, x + 37.5, y, 0.5) for x, y in [(60, 100), (102.5, 100), (210, 185), (252.5, 185)]]
    strokes = Drawing(297, 210, symbols=[place(LIBRARY["GATE_VALVE"], 100, 100),
                                         place(LIBRARY["GATE_VALVE"], 250, 185)]).flat()
    d = Drawing(297, 210, lines=strokes.lines + block + pipes, polylines=[frame])
    symbols.recognise(d, LIBRARY)
    assert [(s.name, round(s.x), round(s.y)) for s in d.symbols] == [("GATE_VALVE", 100, 100)]


def test_nothing_is_placed_in_the_strip_of_tables_below_a_line_across_the_frame():
    frame = Polyline([(10, 10), (287, 10), (287, 200), (10, 200)], [0] * 4, closed=True, kind="frame")
    strip = [Line(10, 160, 287, 160), Line(100, 160, 100, 200)]  # notes and revision table below the drawing
    pipes = [Line(x, y, x + 37.5, y, 0.5) for x, y in [(60, 100), (102.5, 100), (150, 180), (192.5, 180)]]
    strokes = Drawing(297, 210, symbols=[place(LIBRARY["GATE_VALVE"], 100, 100),
                                         place(LIBRARY["GATE_VALVE"], 190, 180)]).flat()
    d = Drawing(297, 210, lines=strokes.lines + strip + pipes, polylines=[frame])
    symbols.recognise(d, LIBRARY)
    assert [(s.name, round(s.x), round(s.y)) for s in d.symbols] == [("GATE_VALVE", 100, 100)]


def test_a_symbol_beside_its_pipe_is_moved_onto_it_and_the_pipe_stays_straight():
    pipes = [Line(60, 50, 97.5, 50, 0.5), Line(102.5, 50, 140, 50, 0.5)]
    d = drawn(place(LIBRARY["GATE_VALVE"], 100, 50.4), lines=pipes)  # drawn a little off the pipe
    symbols.recognise(d, LIBRARY)
    assert [(s.name, round(s.y, 2)) for s in d.symbols] == [("GATE_VALVE", 50.0)]
    assert all(ln.y1 == ln.y2 == 50 for ln in d.lines)
    assert sorted((ln.x1, ln.x2) for ln in d.lines) == [(60, pytest.approx(97.5, abs=0.3)),
                                                         (pytest.approx(102.5, abs=0.3), 140)]


def test_a_flow_arrow_on_a_pipe_that_goes_on_is_found_and_moved_onto_it():
    d = drawn(place(LIBRARY["FLOW_ARROW"], 100, 50.6), lines=[Line(40, 50, 160, 50, 0.5)])  # a little off the pipe
    symbols.recognise(d, LIBRARY)
    assert [(s.name, round(s.y, 2)) for s in d.symbols] == [("FLOW_ARROW", 50.0)]


def test_a_flow_arrow_needs_its_pipe_to_go_on_past_both_ends():
    d = drawn(place(LIBRARY["FLOW_ARROW"], 100, 50), lines=[Line(98.8, 50, 160, 50, 0.5)])  # it starts at the tail
    symbols.recognise(d, LIBRARY)
    assert "FLOW_ARROW" not in [s.name for s in d.symbols]  # where an arc runs into a line, it looks like this


def test_a_stroke_left_inside_a_valve_does_not_join_the_pipes_on_either_side():
    left, stub = Line(60, 50, 97.5, 50, 0.5), Line(97.5, 50, 100, 50, 0.5)  # a pipe and a stroke up to the middle
    right = Polyline([(100, 50), (102.5, 50), (140, 50)], [0.0, 0.0], width=0.5)  # traced from the middle on
    d = drawn(place(LIBRARY["GATE_VALVE"], 100, 50), lines=[left, stub])
    d.polylines = [right]
    symbols.recognise(d, LIBRARY)
    assert [s.name for s in d.symbols] == ["GATE_VALVE"]
    ends = [x for ln in d.lines for x in (ln.x1, ln.x2)] + [p[0] for pl in d.polylines for p in pl.points]
    assert not any(98 < x < 102 for x in ends)  # nothing left between the ports


def test_a_symbol_with_a_ring_is_centred_on_the_circle_found():
    hit = symbols._Hit(symbols._shape(LIBRARY["INSTRUMENT"]), (100.3, 49.8, 0.0, False, 1.0), 1.0, np.array([]))
    ring = max((q for q in LIBRARY["INSTRUMENT"].parts if isinstance(q, Circle)), key=lambda q: q.r)
    x, y, _, _, scale = symbols._centred(hit, [Circle(100 + ring.cx, 50 + ring.cy, 1.05 * ring.r)]).pose
    assert (x, y, scale) == pytest.approx((100, 50, 1.05))


def test_a_straight_symbol_on_the_round_end_of_a_line_number_box_is_rejected():
    hit = symbols._Hit(symbols._shape(LIBRARY["OFF_PAGE_CONNECTOR"]), (100, 50, 0.0, False, 1.0), 1.0, np.array([]))
    end = Arc(103.5, 50, 3, 270, 90)  # the box ends in a half circle where the tip is
    t = np.radians(np.arange(270, 451, 5))
    points = np.column_stack([103.5 + 3 * np.cos(t), 50 + 3 * np.sin(t)])
    heading = np.column_stack([-np.sin(t), np.cos(t)])
    frame = symbols._Ink(points, heading, np.zeros(len(points)), np.zeros(len(points), bool), np.ones(len(points), bool))
    assert symbols._rounded(hit, [end], frame)
    stroke = symbols._Ink(points, heading, np.zeros(len(points)), np.zeros(len(points), bool))
    assert not symbols._rounded(hit, [end], stroke)  # a round stroke that frames no text
    assert not symbols._rounded(hit, [Arc(99, 50, 0.5, 270, 90)], frame)  # the curve of a letter inside


def test_ink_on_a_small_closed_box_is_marked_as_boxed():
    box = [Line(20, 20, 25, 20), Line(25, 20, 25, 25), Line(25, 25, 20, 25), Line(20, 25, 20, 20)]
    room = [Line(60, 20, 90, 20), Line(90, 20, 90, 50), Line(90, 50, 60, 50), Line(60, 50, 60, 20)]  # larger than BOX
    points = np.array([(22.5, 20.1), (24.9, 23), (22.5, 22.5), (75, 20), (40, 40)])
    ink = symbols._Ink(points, np.tile((1.0, 0.0), (5, 1)), np.zeros(5), np.zeros(5, bool))
    assert symbols._boxed(Drawing(100, 60, lines=box + room), ink).tolist() == [True, True, False, False, False]


def test_a_ball_valve_whose_ball_is_filled_in_is_no_gate_valve():
    d = Drawing(297, 210)
    p = Pen(d)
    p.line(20, 50, 72.5, 50, 0.5)
    p.symbol("GATE_VALVE", 75, 50)
    p.fill([(75 + 1.1 * np.cos(a), 50 + 1.1 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)])
    p.line(77.5, 50, 130, 50, 0.5)
    img = cv2.cvtColor(render(d, 150 / 25.4), cv2.COLOR_GRAY2BGR)  # coarse, as a photo: no ring left, a blob
    r = convert(img, paper="A4", dpi=150, drawing="pid")
    assert [s.name for s in r.drawing.symbols] == ["BALL_VALVE"]


def test_a_symbol_whose_outline_is_partly_missing_is_marked_to_check():
    pipes = [Line(60, 50, 97.5, 50, 0.5), Line(102.5, 50, 140, 50, 0.5)]
    whole, gappy = (drawn(place(LIBRARY["GATE_VALVE"], 100, 50), lines=pipes) for _ in range(2))
    del gappy.lines[0]  # one side of the valve not drawn
    for d in (whole, gappy):
        symbols.recognise(d, LIBRARY)
    assert [(s.confidence, s.doubt) for s in whole.symbols] == [(100.0, None)]
    assert [(s.name, s.doubt) for s in gappy.symbols] == [("GATE_VALVE", "only 87 % of its outline found")]


def test_a_symbol_to_check_goes_to_its_own_layer_and_into_the_review_list():
    sure, unsure = place(LIBRARY["PUMP"], 100, 50), place(LIBRARY["GATE_VALVE"], 150, 50)
    unsure.confidence, unsure.doubt = 88.0, "only 88 % of its outline found"
    d = Drawing(200, 100, symbols=[sure, unsure], texts=[Text(20, 20, 3, "P-1O1", confidence=40.0, uncertain=True)])
    assert [ref.dxf.layer for ref in to_dxf(d).modelspace().query("INSERT")] == ["P2C-SYMBOL", "P2C-SYMBOL-CHECK"]
    assert [(c["kind"], c["what"], c["why"]) for c in scan._report(d)["review"]] == [
        ("text", "P-1O1", "uncertain reading"), ("symbol", "GATE_VALVE", "only 88 % of its outline found")]
    info = {"drawing_type": "pid", "paper": "A4", "sheet_size_mm": (200, 100), "sheet": "paper", "scale": None,
            "counts": {}}
    markdown = "\n".join(scan._page_markdown(info, d, "##"))
    assert "## Symbols\n\n- Centrifugal pump" in markdown
    assert "## Symbols to check (uncertain recognition)\n\n- Gate valve (only 88 % of its outline found)" in markdown


def test_dxf_holds_the_library_blocks_with_attributes():
    s = place(LIBRARY["PUMP"], 100, 50, angle=90)
    s.attributes = {"TAG": Text(100, 65, 3, "P-101")}
    doc = to_dxf(Drawing(200, 100, symbols=[s], scale=10))
    ref = doc.modelspace().query("INSERT")[0]
    assert ref.dxf.name == "PUMP" and "PUMP" in doc.blocks
    assert ref.dxf.rotation == pytest.approx(270) and ref.dxf.xscale == ref.dxf.yscale == pytest.approx(10)
    assert ref.dxf.layer == "P2C-SYMBOL" and [(a.dxf.tag, a.dxf.text) for a in ref.attribs] == [("TAG", "P-101")]
    out = io.StringIO()
    doc.write(out)
    assert not ezdxf.read(io.StringIO(out.getvalue())).audit().has_errors  # written and read back


def test_symbols_are_found_on_a_scan():
    d = Drawing(297, 210)
    p = Pen(d)
    p.line(20, 100, 117.5, 100, 0.5)
    p.symbol("GATE_VALVE", 120, 100)
    p.line(122.5, 100, 143.5, 100, 0.5)
    p.symbol("PUMP", 150, 100, h=3.0, TAG="P-101")
    p.symbol("FLOW_ARROW", 70, 100)
    p.symbol("INSTRUMENT", 150, 60, FUNCTION="PI", NUMBER="102")
    p.line(150, 65, 150, 93.5, 0.25)
    img = cv2.cvtColor(render(d, 300 / 25.4), cv2.COLOR_GRAY2BGR)
    r = convert(img, paper="A4", dpi=300, drawing="pid")
    assert sorted(s.name for s in r.drawing.symbols) == ["FLOW_ARROW", "GATE_VALVE", "INSTRUMENT", "PUMP"]
    pump = next(s for s in r.drawing.symbols if s.name == "PUMP")
    assert pump.attributes["TAG"].text == "P-101"
