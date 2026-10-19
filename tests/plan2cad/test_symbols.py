import io

import cv2
import ezdxf
import pytest
from synth import Pen

from app.plan2cad import symbols
from app.plan2cad.export import to_dxf
from app.plan2cad.library import load, place
from app.plan2cad.model import Drawing, Line, Polyline, Text
from app.plan2cad.pipeline import convert
from app.plan2cad.render import render

LIBRARY = load()


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


def test_lines_without_symbols_stay_lines():
    d = Drawing(200, 100, lines=[Line(10, 10, 60, 10), Line(60, 10, 60, 40), Line(20, 30, 50, 30)])
    symbols.recognise(d, LIBRARY)
    assert not d.symbols and len(d.lines) == 3


def test_nothing_is_placed_in_the_title_block():
    frame = Polyline([(10, 10), (287, 10), (287, 200), (10, 200)], [0] * 4, closed=True, kind="frame")
    block = [Line(207, 200, 207, 170), Line(207, 170, 287, 170)]
    pipes = [Line(x, y, x + 37.5, y, 0.5) for x, y in [(60, 100), (102.5, 100), (210, 185), (252.5, 185)]]
    strokes = Drawing(297, 210, symbols=[place(LIBRARY["GATE_VALVE"], 100, 100),
                                         place(LIBRARY["GATE_VALVE"], 250, 185)]).flat()
    d = Drawing(297, 210, lines=strokes.lines + block + pipes, polylines=[frame])
    symbols.recognise(d, LIBRARY)
    assert [(s.name, round(s.x), round(s.y)) for s in d.symbols] == [("GATE_VALVE", 100, 100)]


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
