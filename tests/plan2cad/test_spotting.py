import cv2
import numpy as np

from app.plan2cad import spotting
from app.plan2cad.model import Drawing, Line, Text
from app.plan2cad.ocr import Components, _spaced, find_texts, spotted
from app.plan2cad.render import render
from app.plan2cad.spotting import Spot, _once, _one_line, spot


def test_a_line_found_again_in_the_next_tile_is_kept_once():
    whole, part, other = Spot(100, 50, 300, 80, "LINE 101", 0.99), Spot(200, 50, 300, 80, "101", 0.9), \
        Spot(100, 200, 300, 230, "P-102", 0.95)
    assert _once([part, whole, other]) == [whole, other]


def test_texts_of_a_drawing_are_spotted_and_read():
    d = Drawing(80, 30, texts=[Text(40, 15, 3.0, "P-101")])
    gray = render(d, 10)
    found = spot(gray, 10)
    assert [s.text for s in found] == ["P-101"]
    assert found[0].x0 < 400 < found[0].x1 and found[0].y0 < 150 < found[0].y1
    assert type(found[0].x0) is float  # not numpy's: the texts end up in report.json


def sheet():
    """A text line, a pipe crossing its box and a lone stroke beside it, as the text pass leaves them."""
    d = Drawing(60, 30, lines=[Line(30, 5, 30, 25, 0.35), Line(45, 12, 45, 18, 0.35)],
                texts=[Text(20, 15, 2.5, "LINE 1O1")])
    gray = render(d, 10)
    ink = (gray < 128).astype(np.uint8)
    texts, labels, _ = find_texts(ink, gray, 10)
    return ink, gray, texts, labels


def test_a_spotted_line_replaces_the_text_it_covers_and_takes_its_letters():
    ink, gray, texts, labels = sheet()
    assert texts  # read before, perhaps wrongly
    line = Spot(60, 130, 340, 170, "LINE 101", 0.97)
    out, new = spotted(texts, labels, ink, gray, [line], 10)
    assert [t.text for t in out] == ["LINE 101"]
    assert new[135:165, 120:280].any()  # the letters are text
    assert not new[50:250, 298:303].any()  # the pipe crossing the box stays linework


def test_a_short_or_unsure_reading_where_nothing_was_read_is_no_text():
    ink, gray, texts, labels = sheet()
    stroke = Spot(430, 110, 470, 190, "I", 0.85)  # the lone stroke read as a letter
    unsure = Spot(430, 110, 470, 190, "IDI", 0.6)
    foreign = Spot(430, 110, 470, 190, "区区区", 0.99)
    for s in (stroke, unsure, foreign):
        out, new = spotted(texts, labels, ink, gray, [s], 10)
        assert [t.text for t in out] == [t.text for t in texts]
        assert not new[110:190, 440:460].any()


def test_a_sure_single_character_is_new_text():
    ink, gray, texts, labels = sheet()
    out, _ = spotted(texts, labels, ink, gray, [Spot(430, 110, 470, 190, "I", 0.97)], 10)
    assert "I" in [t.text for t in out]


def test_a_shorter_line_read_across_a_text_written_the_other_way_leaves_it():
    d = Drawing(60, 30, texts=[Text(30, 15, 2.5, "8,82", 90)])
    gray = render(d, 10)
    ink = (gray < 128).astype(np.uint8)
    texts, labels, _ = find_texts(ink, gray, 10)
    assert [(t.text, t.angle) for t in texts] == [("8,82", 90)]
    across = Spot(280, 100, 320, 150, "82", 0.96)
    out, new = spotted(texts, labels, ink, gray, [across], 10)
    assert out == texts and (new == labels).all()
    across.text = "AB12"  # longer than what was read first: that was the misreading
    out, _ = spotted(texts, labels, ink, gray, [across], 10)
    assert [t.text for t in out] == ["AB12"]


def test_pieces_of_one_line_are_level_and_side_by_side():
    a, b = Spot(100, 50, 300, 80, "Grundriss Erdges", 0.99, True), Spot(280, 51, 420, 79, "Erdgeschoss", 1.0)
    assert _one_line(a, b)
    assert not _one_line(a, Spot(280, 75, 420, 105, "Maßstab", 1.0))  # the next line
    assert not _one_line(a, Spot(320, 20, 340, 120, "8,82", 1.0))  # vertical


def test_wide_spaces_the_recogniser_dropped_are_put_back():
    d = Drawing(80, 20, texts=[Text(40, 10, 3.0, "AREA 12 m2")])
    gray = render(d, 10)
    cc = Components((gray < 128).astype(np.uint8), gray)
    near = np.arange(cc.n)
    assert _spaced(cc, near, False, "AREA12m2") == "AREA 12 m2"
    assert _spaced(cc, near, False, "AREA 12m2") == "AREA 12 m2"
    assert _spaced(cc, near, False, "AREA12") == "AREA12"  # characters and glyphs differ: left as read


def test_a_line_read_again_leaves_the_detector_on():
    gray = render(Drawing(80, 30, texts=[Text(40, 15, 3.0, "P-101")]), 10)
    spotting._engine()(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), use_det=False)  # as a cut line is read again
    assert [s.text for s in spot(gray, 10)] == ["P-101"]


def test_lines_found_but_read_as_nothing_are_no_text(monkeypatch):
    class Found:  # what RapidOCR answers then: the boxes of the detector, no texts
        boxes, scores = np.zeros((1, 4, 2)), (0.9,)

    monkeypatch.setattr(spotting, "_engine", lambda: lambda *a, **k: Found())
    assert spot(np.zeros((100, 100), np.uint8), 10) == []
