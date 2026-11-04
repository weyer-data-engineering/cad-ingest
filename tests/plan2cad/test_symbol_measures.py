"""The benchmark's measures around symbols (bench/evaluate.py): symbols placed where there are none, wires that end
at the port of a symbol, and the marks of symbols not classified."""
from evaluate import continuity, symbols_apart
from evaluate import symbols as symbol_measures
from synth import section

from app.plan2cad.library import BUILT_IN, load, place
from app.plan2cad.model import Drawing, Line, Unclassified

PID, CIRCUITS = load(), load(BUILT_IN / "iec-60617")


def test_symbols_where_the_reference_has_none_are_counted_on_every_drawing():
    gt = Drawing(200, 100, symbols=[place(PID["GATE_VALVE"], 50, 50)])
    right, wrong_kind = place(PID["GATE_VALVE"], 50.5, 50), place(PID["BALL_VALVE"], 51, 50)
    nowhere = place(PID["PUMP"], 150, 50)
    m = symbol_measures(gt, Drawing(200, 100, symbols=[right, wrong_kind, nowhere]))
    assert (m["symbol_false"], m["symbol_recall"]) == (1, 1.0)  # the wrong kind is on a symbol, the pump on nothing
    assert symbol_measures(Drawing(200, 100), Drawing(200, 100, symbols=[nowhere])) == {"symbol_false": 1}
    assert symbol_measures(Drawing(200, 100), Drawing(200, 100)) == {}  # nothing to tell


def test_a_wire_ending_at_the_port_of_a_contact_is_no_break():
    contact = place(CIRCUITS["NO_CONTACT"], 100, 50)  # ports at y 47.5 and 52.5; its blade starts at 47.5
    wires = [Line(100, 10, 100, 47.5, 0.35), Line(100, 52.5, 100, 90, 0.35)]
    gt = Drawing(200, 100, lines=wires, symbols=[contact])
    assert continuity(gt, gt)["breaks_per_m"] == 0
    contact.ports = []  # the same end where the reference runs on in the same pen: a break
    assert continuity(gt, gt)["breaks_per_m"] > 0


def test_a_mark_on_a_symbol_of_no_library_that_the_reference_lists_is_right():
    level = Unclassified((64.4, 204.6, 67.6, 207.0), "level mark")  # a level mark's triangle
    gt = Drawing(420, 297, symbols=[place(PID["GATE_VALVE"], 50, 50)], unclassified=[level])
    marks = [Unclassified((63.2, 202.4, 67.4, 206.7), "5 strokes"), Unclassified((150, 150, 155, 155), "4 strokes")]
    pred = Drawing(420, 297, symbols=[place(PID["GATE_VALVE"], 50, 50)], unclassified=marks)
    m = symbols_apart(gt, pred)
    assert (m["unclassified_on_symbols"], m["unclassified_elsewhere"], m["unclassified_found"]) == (1, 1, 1.0)
    unmarked = symbols_apart(gt, Drawing(420, 297, symbols=gt.symbols))  # the valve placed, the level mark not marked
    assert (unmarked["unclassified_on_symbols"], unmarked["unclassified_found"]) == (0, 0.0)
    plain = symbols_apart(Drawing(420, 297), pred)  # no symbol in the reference, none of a library or of none
    assert plain == {"unclassified_on_symbols": 0, "unclassified_elsewhere": 2}


def test_a_section_lists_its_level_marks_as_symbols_of_no_library():
    d = section(1)
    assert [u.why for u in d.unclassified] == ["level mark"] * 4
    assert all(round(u.box[2] - u.box[0], 1) == 3.2 and round(u.box[3] - u.box[1], 1) == 2.4 for u in d.unclassified)
    assert d.scaled(0.5).unclassified[0].box == tuple(v / 2 for v in d.unclassified[0].box)  # printed smaller too
