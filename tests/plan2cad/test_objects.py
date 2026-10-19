import pytest

from app.plan2cad.model import Arc, Circle, Drawing, Line, Polyline, Text
from app.plan2cad.objects import classify, doors, drawing_type


def test_door_swing_is_one_object_and_the_jamb_stays_with_the_wall():
    leaf_and_jamb = Line(-3, 0, 10, 0, 0.35)  # merged where the leaf continues the wall end
    found, rest, arcs = doors([leaf_and_jamb], [Arc(0, 0, 10, 0, 90, 0.18)])
    assert len(found) == 1 and found[0].kind == "door" and not arcs
    assert found[0].points == pytest.approx([(0, 0), (10, 0), (0, 10)])
    assert rest == [Line(-3, 0, 0, 0, 0.35)]


def test_pid_lines_become_equipment_pipes_and_signal_lines():
    vessel = Polyline([(0, 0), (20, 0), (20, 40), (0, 40)], [0.0] * 4, closed=True)
    d = Drawing(200, 100, lines=[Line(20, 20, 60, 20), Line(25, 30, 45, 30)], polylines=[vessel],
                circles=[Circle(50, 30, 5)], texts=[Text(50, 28, 2, "PI")])
    classify(d, "pid")
    assert sorted(pl.kind for pl in d.polylines) == ["equipment", "pipe", "signal"] and not d.lines


@pytest.mark.parametrize("texts, scale, kind", [
    (["17,68 m²"], None, "building"), (["Blatt 1/1"], 50, "building"), (["DN80-P-1011"], None, "pid"),
    (["Blatt 1/1"], None, ""),
])
def test_drawing_type(texts, scale, kind):
    assert drawing_type([Text(0, 0, 2.5, t) for t in texts], [], scale) == kind
