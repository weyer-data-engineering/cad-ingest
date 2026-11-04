"""The symbols of the starter libraries: found where they are drawn, with their attributes, and not where they are
not (bench/make_library.py writes them)."""
import pytest

from app.library_tool import advice
from app.plan2cad import symbols
from app.plan2cad.library import BUILT_IN, combine, installed, load, place
from app.plan2cad.model import Circle, Drawing, Line, Text
from app.plan2cad.pipeline import TEXT, _printed

PID = load()
CIRCUITS = load(BUILT_IN / "iec-60617")
MARKS = load(BUILT_IN / "building-symbols")


def drawn(*placed, lines=(), texts=()) -> Drawing:
    """A drawing holding the strokes of the placed symbols as plain lines, arcs and circles, plus ``lines``."""
    strokes = Drawing(200, 100, symbols=list(placed)).flat()
    return Drawing(200, 100, lines=strokes.lines + list(lines), arcs=strokes.arcs, circles=strokes.circles,
                   texts=list(texts))


def found(d: Drawing) -> list[tuple[str, int, int]]:
    return sorted((s.name, round(s.x), round(s.y)) for s in d.symbols)


def test_the_starter_libraries_need_no_notes():
    for lib in installed().values():
        assert not lib.warnings and not [note for t in lib.symbols.values() for note in advice(t)], lib.id


def test_circuit_symbols_have_a_tag_and_ports_where_their_wires_stop():
    assert {"TERMINAL", "CIRCUIT_BREAKER", "MOTOR_PROTECTION", "PUSH_BUTTON", "PROTECTIVE_EARTH"} <= set(CIRCUITS)
    assert all(t.ports and "TAG" in t.tags for t in CIRCUITS.values())
    assert CIRCUITS["NO_CONTACT"].ports == [(0, -2.5), (0, 2.5)]  # the ends of its gap: no leads along the wire


def test_contacts_switches_and_terminals_are_found_between_the_ends_of_their_wire():
    names = ["NO_CONTACT", "NC_CONTACT", "CIRCUIT_BREAKER", "MOTOR_PROTECTION", "PUSH_BUTTON", "TERMINAL"]
    placed = [place(CIRCUITS[name], 30 + 25 * k, 50) for k, name in enumerate(names)]
    wires = [Line(s.x, y, s.x, end, 0.35) for s in placed
             for y, end in zip((20, 80), sorted(p[1] for p in s.ports))]  # from the rails to the ports
    d = drawn(*placed, lines=wires)
    symbols.recognise(d, CIRCUITS)
    assert found(d) == sorted((name, 30 + 25 * k, 50) for k, name in enumerate(names))
    assert all(wire in d.lines for wire in wires)  # the wires stay, ending at the ports


def test_protective_earth_is_found_at_the_end_of_its_wire():
    d = drawn(place(CIRCUITS["PROTECTIVE_EARTH"], 100, 50), lines=[Line(100, 20, 100, 47, 0.35)])
    symbols.recognise(d, CIRCUITS)
    assert found(d) == [("PROTECTIVE_EARTH", 100, 50)]


def test_an_orifice_plate_in_a_pipe_is_found():
    pipes = [Line(20, 50, 99.2, 50, 0.5), Line(100.8, 50, 180, 50, 0.5)]  # ending at its sides
    d = drawn(place(PID["ORIFICE"], 100, 50), lines=pipes)
    symbols.recognise(d, PID)
    assert found(d) == [("ORIFICE", 100, 50)] and not d.symbols[0].doubt


@pytest.mark.parametrize("bar", [1.5, 2.5])  # a flange, an orifice plate drawn as two bars: neither is in the library
def test_two_bars_across_a_pipe_are_left_as_lines(bar):
    bars = [Line(x, 50 - bar, x, 50 + bar, 0.35) for x in (99.4, 100.6)]
    d = Drawing(200, 100, lines=[Line(20, 50, 180, 50, 0.5)] + bars)
    symbols.recognise(d, PID)
    assert not d.symbols


def test_marks_of_a_building_drawing_are_found_with_their_letter():
    axis, north = place(MARKS["AXIS"], 90, 40), place(MARKS["NORTH_ARROW"], 140, 50)
    section = place(MARKS["SECTION_MARK"], 175, 50, 270)
    texts = [Text(90, 40, 3.5, "A"), Text(175, 50, 3.5, "B")]
    d = drawn(axis, north, section, lines=[Line(90, 48, 90, 80, 0.18)], texts=texts)  # with the rest of the axis
    symbols.recognise(d, MARKS)
    assert found(d) == [("AXIS", 90, 40), ("NORTH_ARROW", 140, 50), ("SECTION_MARK", 175, 50)]
    assert {s.name: {k: t.text for k, t in s.attributes.items()} for s in d.symbols if s.attributes} == {
        "AXIS": {"AXIS": "A"}, "SECTION_MARK": {"LETTER": "B"}}
    assert not d.texts


def test_marks_on_a_sheet_lettered_larger_than_a_pid_are_looked_for_at_their_size():
    # 3.5 mm lettering, common on floor plans: against the TEXT of P&ID tags that would be a sheet printed 1.75 times
    # larger, but the marks go with the 2.5 mm text_height of their library
    texts = [Text(10 + 12 * (k % 5), 70 + 6 * (k // 5), 3.5, "Flur") for k in range(25)]
    marks = installed()["building-symbols"]
    assert (_printed(texts), _printed(texts, marks.text_height)) == (pytest.approx(1.75), pytest.approx(1.4))
    d = drawn(place(MARKS["NORTH_ARROW"], 140, 50), texts=texts)
    symbols.recognise(d, combine([marks], "building", 50, {marks.id: _printed(texts, marks.text_height)}))
    assert [s.name for s in d.symbols] == ["NORTH_ARROW"]
    assert combine([installed()["din-10628"]], "pid", None, {marks.id: 2.0})["PUMP"] is PID["PUMP"]  # unscaled
    assert _printed(texts, TEXT) == _printed(texts)  # a library without text_height: as before


def test_a_circle_with_a_letter_but_no_axis_line_is_no_axis():
    d = Drawing(200, 100, circles=[Circle(90, 40, 5)], texts=[Text(90, 40, 3.5, "A")])  # a tree, a bubble
    symbols.recognise(d, MARKS)
    assert not d.symbols
