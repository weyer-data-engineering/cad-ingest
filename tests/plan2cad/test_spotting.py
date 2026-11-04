import math

import cv2
import numpy as np
import pytest

from app.plan2cad import spotting
from app.plan2cad.model import Drawing, Line, Text
from app.plan2cad.ocr import Components, _spaced, find_texts, spotted
from app.plan2cad.render import render
from app.plan2cad.spotting import Spot, _once, _one_line, _slant, spot


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


def test_a_box_turned_off_the_axes_holds_an_oblique_line():
    assert _slant(np.float32([[0, 0], [100, 0], [100, 20], [0, 20]])) is None
    a = math.radians(20)  # along the line, and across it towards its baseline
    along, across = np.array([math.cos(a), -math.sin(a)]), np.array([math.sin(a), math.cos(a)])
    quad = np.float32([(0, 0), 100 * along, 100 * along + 20 * across, 20 * across])
    assert _slant(quad) == pytest.approx(20)


def test_an_oblique_line_is_spotted_with_its_angle_and_becomes_a_text_along_it():
    gray = render(Drawing(80, 50, texts=[Text(40, 25, 3.0, "Lindenstraße", -20)]), 10)
    found = spot(gray, 10, oblique=True)
    assert [s.text for s in found] == ["Lindenstraße"] and found[0].angle % 360 == pytest.approx(340, abs=1.5)
    ink = (gray < 128).astype(np.uint8)
    texts, labels, _ = find_texts(ink, gray, 10)
    out, new = spotted(texts, labels, ink, gray, found, 10)
    t = next(t for t in out if t.text == "Lindenstraße")
    assert t.angle == pytest.approx(-20, abs=1.5) and (t.x, t.y) == pytest.approx((40, 25), abs=0.5)
    assert t.length == pytest.approx(26.3, abs=1.0) and (new[ink > 0] > 0).mean() > 0.95  # measured along it
    assert [s.angle for s in spot(gray, 10)] == [None]  # without the step: by its upright box, as before


def test_a_vertical_line_is_read_both_ways_round():
    gray = render(Drawing(40, 40, texts=[Text(20, 20, 2.5, "0,60", 90), Text(30, 20, 2.5, "Zulauf", -90)]), 10)
    up, down = sorted(spot(gray, 10, both=True), key=lambda s: s.x0)
    assert (up.text, up.angle) == ("0,60", 90) and up.turned and up.turned[0] != "0,60"
    assert (down.text, down.angle) == ("Zulauf", 270)  # read top to bottom
    assert spot(gray, 10)[0].angle is None  # by its box, as before


def test_a_short_vertical_line_is_told_by_its_characters_standing_one_above_the_other():
    from app.plan2cad.spotting import _stands

    texts = [Text(12, 15, 2.0, "12", 90), Text(30, 15, 2.0, "12")]  # a wire number and a path number
    gray = render(Drawing(40, 30, texts=texts), 10)
    assert [(s.text, s.angle) for s in sorted(spot(gray, 10, both=True), key=lambda s: s.x0)] == \
        [("12", 90), ("12", None)]
    gray = render(Drawing(40, 30, texts=[Text(10, 15, 2.0, "7", 90), Text(30, 15, 2.0, "7")]), 10)
    assert not _stands(gray, Spot(80, 130, 120, 165, "7", 1.0))  # one shape on its side may be a flow arrow too


def test_a_leaning_box_around_a_horizontal_line_gives_a_horizontal_text():
    gray = render(Drawing(80, 40, texts=[Text(40, 20, 3.0, "131,00")]), 10)
    ink = (gray < 128).astype(np.uint8)
    texts, labels, _ = find_texts(ink, gray, 10)
    a = math.radians(8)  # the box leans 8 degrees, as one beside a contour line may
    along, across = np.array([math.cos(a), -math.sin(a)]), np.array([math.sin(a), math.cos(a)])
    quad = [tuple(np.array([400, 200]) + u * 70 * along + v * 25 * across) for u, v in ((-1, -1), (1, -1), (1, 1),
                                                                                      (-1, 1))]
    xs, ys = zip(*quad)
    leaning = Spot(min(xs), min(ys), max(xs), max(ys), "131,00", 0.99, corners=tuple(quad), angle=8.0)
    out, _ = spotted(texts, labels, ink, gray, [leaning], 10)
    assert [(t.text, t.angle) for t in out] == [("131,00", 0.0)]
    # read again along the box, the same pixels read worse: the reading it was found with stays, about as sure
    again = Spot(min(xs), min(ys), max(xs), max(ys), "131,08", 0.92, corners=tuple(quad), angle=8.0,
                 found=("131,00", 0.85))
    out, _ = spotted(texts, labels, ink, gray, [again], 10)
    assert [(t.text, t.angle, t.confidence, t.uncertain) for t in out] == [("131,00", 0.0, 85.0, False)]


def test_a_line_whose_letters_run_level_keeps_its_reading_as_found_unless_the_one_along_its_box_is_clearly_surer():
    from dataclasses import replace

    from app.plan2cad.ocr import _level

    s = Spot(0, 0, 180, 90, "±0,08", 0.88, angle=355.8, found=("±0,00", 0.86))
    assert (_level(s, 0.0).text, _level(s, 0.0).score, _level(s, 0.0).angle) == ("±0,00", 0.86, 0.0)
    assert _level(replace(s, score=0.99), 0.0).text == "±0,08"  # surer by more than ABOUT along its box
    assert _level(replace(s, found=None), 0.0).text == "±0,08"  # not read again: as it is
    assert _level(replace(s, found=("—", 0.95)), 0.0).text == "±0,08"  # found as nothing to read


def test_an_oblique_line_read_again_the_way_its_box_runs_keeps_its_reading_as_found(monkeypatch):
    from dataclasses import replace

    from app.plan2cad.spotting import _directed

    def box(angle: float) -> Spot:  # a long box at ``angle`` degrees
        a = math.radians(angle)
        along, across = np.array([math.cos(a), -math.sin(a)]), np.array([math.sin(a), math.cos(a)])
        quad = [tuple(np.array([150.0, 250.0]) + u * along + v * across) for u, v in ((0, 0), (200, 0), (200, 30),
                                                                                    (0, 30))]
        xs, ys = zip(*quad)
        return Spot(min(xs), min(ys), max(xs), max(ys), "Lindenstrase", 0.8, corners=tuple(quad))

    s = box(20)
    monkeypatch.setattr(spotting, "_reread", lambda gray, f, s, angle: replace(s, text="Lindenstraße", score=0.9,
                                                                                angle=angle % 360))
    assert _directed(None, 1.0, s, oblique=True, both=True).found == ("Lindenstrase", 0.8)
    assert _directed(None, 1.0, box(60), oblique=True, both=True).found is None  # steeper: found read another way
    monkeypatch.setattr(spotting, "_reread", lambda gray, f, s, angle: replace(s, text="Lindenstraße", score=0.9,
                                                                                angle=(angle + 180) % 360))
    assert _directed(None, 1.0, s, oblique=True, both=True).found is None  # turned: as found, it read the other way
    assert _directed(None, 1.0, s, oblique=False, both=True) is s  # without the step: as found


def test_a_vertical_line_that_reading_again_only_made_surer_of_keeps_the_doubt_it_was_found_with():
    from dataclasses import replace

    from app.plan2cad.ocr import Readings

    gray = render(Drawing(30, 40, texts=[Text(15, 20, 2.5, "3,94", 90)]), 10)
    ink = (gray < 128).astype(np.uint8)
    line = Spot(129, 156, 170, 244, "", 0.0, angle=90.0)  # its box, read bottom to top
    none = np.zeros(gray.shape, np.uint8)

    def doubt(text: str, score: float, first: tuple[str, float] | None) -> float | None:
        readings = Readings()
        out, _ = spotted([], none, ink, gray, [replace(line, text=text, score=score, found=first)], 10, readings)
        return readings.doubt(out[0])

    assert doubt("3.94", 0.95, ("3.94", 0.79)) == 79.0  # the same reading, unsure as found
    assert doubt("3,94", 0.95, ("3.94", 0.79)) is None  # another reading: it stands on its own
    assert doubt("3.94", 0.95, ("3.94", 0.85)) is None  # sure as found already
    assert doubt("3.94", 0.95, None) is None  # not read again: without the step


def test_a_vertical_line_read_again_keeps_its_reading_as_found(monkeypatch):
    from dataclasses import replace

    from app.plan2cad.spotting import _directed

    s = Spot(100, 100, 130, 200, "3.94", 0.79)  # a tall box
    monkeypatch.setattr(spotting, "_reread", lambda gray, f, s, angle: replace(s, score=0.95, angle=90.0))
    assert _directed(None, 1.0, s, oblique=False, both=True).found == ("3.94", 0.79)
    monkeypatch.setattr(spotting, "_reread", lambda gray, f, s, angle: s)  # read as nothing again
    assert _directed(None, 1.0, s, oblique=False, both=True).found is None
    assert _directed(None, 1.0, s, oblique=True, both=False) is s  # without the step: as found


def test_pieces_of_one_vertical_line_found_as_overlapping_boxes_are_read_as_one():
    from app.plan2cad.spotting import _joined

    gray = render(Drawing(40, 40, texts=[Text(20, 20, 2.5, "8,82", 90)]), 10)  # from y = 15 to 25 mm
    top = Spot(180, 140, 225, 205, ",82", 0.99, angle=90.0)
    bottom = Spot(180, 195, 225, 260, "8,", 0.99)  # its box about as wide as high: taken for a horizontal one
    assert [(s.text, s.angle) for s in _joined(gray, 1.0, [top, bottom])] == [("8,82", 90)]
    apart = Spot(180, 300, 225, 360, "12", 0.99, angle=90.0)
    assert len(_joined(gray, 1.0, [top, apart])) == 2


def test_a_horizontal_tag_at_the_end_of_a_vertical_label_is_no_piece_of_it():
    from app.plan2cad.render import text_length

    label = Text(20, 30, 2.5, "Zulauf 120", 90)
    tag = Text(20, 30 - text_length(label) / 2 - 1.45, 2.5, "M1")  # its capitals 0.2 mm above the label's end
    gray = render(Drawing(40, 50, texts=[label, tag]), 10)
    assert sorted(s.text for s in spot(gray, 10, both=True)) == ["M1", "Zulauf 120"]


def test_pieces_whose_line_reads_as_nothing_whole_stay_as_found(monkeypatch):
    from dataclasses import replace

    from app.plan2cad.spotting import _joined

    # each piece reads well turned to vertical, the line they make reads as nothing
    monkeypatch.setattr(spotting, "_reread", lambda gray, f, s, angle: replace(s, angle=90.0) if s.text else s)
    top, bottom = Spot(180, 140, 225, 205, ",82", 0.99, angle=90.0), Spot(180, 195, 225, 260, "8,", 0.7)
    assert _joined(np.zeros((400, 400), np.uint8), 1.0, [top, bottom]) == [top, bottom]


def test_two_lines_of_an_oblique_label_are_told_apart_by_their_turned_boxes():
    a = math.radians(45)
    along, across = np.array([math.cos(a), -math.sin(a)]), np.array([math.sin(a), math.cos(a)])

    def line(shift: float) -> Spot:  # a long box at 45 degrees, ``shift`` px across it
        quad = [tuple(np.array([50.0, 250.0]) + (shift + v) * across + u * along) for u, v in ((0, 0), (280, 0),
                                                                                          (280, 30), (0, 30))]
        xs, ys = zip(*quad)
        return Spot(min(xs), min(ys), max(xs), max(ys), "Lindenstraße", 0.9, corners=tuple(quad))

    first, second = line(0), line(40)
    assert _once([first, second], oblique=True) == [first, second]
    assert len(_once([first, second])) == 1  # by their upright boxes, which overlap much, as before


def test_the_spotters_reading_turned_by_180_degrees_is_kept_for_the_kind_to_choose():
    from app.plan2cad.ocr import Readings

    gray = render(Drawing(30, 40, texts=[Text(15, 20, 2.5, "0,60", 90)]), 10)
    ink = (gray < 128).astype(np.uint8)
    readings, none = Readings(), np.zeros(gray.shape, np.uint8)
    out, _ = spotted([], none, ink, gray, spot(gray, 10, both=True), 10, readings)
    assert [(t.text, t.angle) for t in out] == [("0,60", 90.0)] and readings.other(out[0]).angle == -90
    out, _ = spotted([], none, ink, gray, spot(gray, 10), 10, readings)
    assert readings.other(out[0]) is None  # read as before: one way round


def test_a_single_character_spotted_where_one_stood_alone_in_a_frame_stays_one():
    from app.plan2cad.model import Circle
    from app.plan2cad.ocr import Readings

    d = Drawing(90, 40, circles=[Circle(20, 20, 5, 0.35)], texts=[
        Text(20, 18, 2.0, "FIC"), Text(20, 22.5, 2.0, "B"), Text(60, 32, 2.0, "LINE 101")])
    gray = render(d, 10)
    ink = (gray < 128).astype(np.uint8)
    readings = Readings()
    texts, labels, _ = find_texts(ink, gray, 10, frame=ink, readings=readings)
    out, _ = spotted(texts, labels, ink, gray, spot(gray, 10), 10, readings)
    assert [t.text for t in out if readings.alone(t)] == ["B"]


def test_the_end_of_a_spotted_line_a_pipe_runs_along_is_cut_off():
    # a vertical label (y 10 to 40) read with the two flanges and the valve below it: the pipe runs along the middle
    # of the box from beyond its lower end, past the valve between the flanges, up to the label's frame
    s = Spot(20, 10, 30, 64, "HH2-P-1", 0.99)
    pipe = [(25, 41, 25, 50), (25, 56, 25, 90)]
    assert spotting._uncovered(s, pipe, ppm=10) == (10, 41)
    frame = [(21, 5, 21, 45), (29, 5, 29, 45)]  # the sides of a frame run along the box, but not along its middle
    assert spotting._uncovered(s, frame, ppm=10) is None
    short = [(25, 64.5, 25, 90)]  # a pipe that only reaches the end of the box takes nothing of it
    assert spotting._uncovered(s, short, ppm=10) is None


def test_a_spotted_line_a_pipe_runs_through_from_end_to_end_is_dropped(monkeypatch):
    monkeypatch.setattr(spotting, "_reread", lambda *a: pytest.fail("nothing is left to read"))
    bars = Spot(20, 40, 35, 52, "H", 0.95)  # the bars of a flange across a horizontal pipe, read as "H"
    kept = Spot(20, 80, 60, 92, "P-101", 0.99)
    assert spotting.off_pipes(np.full((200, 200), 255, np.uint8), 10, [bars, kept], [(0, 46, 100, 46)]) == \
        ([kept], [bars])  # given apart, for the letters another reader found in it


def test_texts_read_alike_by_both_readers_come_out_the_same_with_the_vote_per_token():
    from app.plan2cad.ocr import Readings

    d = Drawing(120, 40, texts=[Text(30, 10, 2.5, "DN80-P-1011"), Text(30, 20, 2.5, "B-501"),
                                Text(80, 30, 2.5, "1080")])
    gray = render(d, 10)
    ink = (gray < 128).astype(np.uint8)
    out = []
    for by_token in (False, True):
        readings = Readings(by_token=by_token)
        texts, labels, _ = find_texts(ink, gray, 10, readings=readings)
        out.append(sorted((t.text, t.uncertain) for t in spotted(texts, labels, ink, gray, spot(gray, 10), 10,
                                                                   readings)[0]))
    assert out[0] == out[1] and ("B-501", False) in out[1]


RING = (100, 100, 50, 3)  # a bubble's drawn ring in pixels: centre, radius and pen (5 mm at 10 px per mm)


def test_a_spotted_line_inside_a_ring_is_cut_where_it_runs_out_of_it_at_one_end():
    # a number in the lower half of the bubble read with the bars of a flange against the ring outside it ("40201T")
    s = Spot(58, 112, 166, 136, "40201T", 0.9)
    start, end = spotting._inside(s, [RING])
    assert start == 58 and end == pytest.approx(100 + math.sqrt(48.5 ** 2 - 24 ** 2))  # at the stroke's inner edge
    standing = Spot(112, 58, 136, 166, "40201T", 0.9, angle=90.0)  # the same along a standing line
    assert spotting._inside(standing, [RING]) == pytest.approx((start, end))
    both = Spot(30, 112, 170, 136, "T40201T", 0.9)  # with the flanges on both sides
    assert spotting._inside(both, [RING]) == pytest.approx((200 - end, end))


def test_a_long_tag_close_to_the_ring_and_a_line_running_far_out_of_it_stay_whole():
    assert spotting._inside(Spot(56, 112, 144, 136, "40201", 0.99), [RING]) is None  # its box ends at the ring
    assert spotting._inside(Spot(53, 112, 150, 136, "40201", 0.99), [RING]) is None  # ... or just beyond it
    assert spotting._inside(Spot(140, 112, 220, 136, "P-101", 0.99), [RING]) is None  # its middle outside it
    assert spotting._inside(Spot(58, 112, 200, 136, "40201 P-101", 0.99), [RING]) is None  # far out: a text beside
    letter = (128, 124, 12, 3)  # the ring of the "O" of a large word, its middle in it
    assert spotting._inside(Spot(100, 112, 148, 136, "PO", 0.99), [letter]) is None


def test_the_part_inside_a_ring_is_read_again_and_a_line_whose_part_reads_as_nothing_stays(monkeypatch):
    from dataclasses import replace

    gray, s = np.full((200, 250), 255, np.uint8), Spot(58, 112, 166, 136, "40201T", 0.9)
    monkeypatch.setattr(spotting, "_reread", lambda gray, f, cut, angle: replace(cut, text="40201", score=0.99))
    (read,) = spotting.off_rings(gray, 10, [s], [RING])
    assert read.text == "40201" and read.x0 == 58 and read.x1 < 143
    monkeypatch.setattr(spotting, "_reread", lambda gray, f, cut, angle: cut)
    assert spotting.off_rings(gray, 10, [s], [RING]) == [s]
