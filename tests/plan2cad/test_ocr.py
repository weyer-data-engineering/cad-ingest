from dataclasses import replace

import numpy as np
import pytest

from app.plan2cad.model import Circle, Drawing, Line, Polyline, Text
from app.plan2cad.ocr import Components, _group, _voted, find_texts, fix, frames, grammar
from app.plan2cad.pipeline import detect_scale
from app.plan2cad.render import render


@pytest.mark.parametrize("raw, fixed", [
    ("17,68 m?", "17,68 m²"), ("11,56 m2", "11,56 m²"), ("15.62 m’", "15.62 m²"), ("DNSO-P-3011", "DN50-P-3011"), ("P-1O1", "P-101"),
    ("Küche", "Küche"), ("B-101", "B-101"),
    ("-PNI6", "-PN16"), ("-CS3I", "-CS31"), ("C=O1XY2Z", "C=01XY2Z"), ("PS=SSO6", "PS=SS06"), ("TIC", "TIC"),
    ("I/P", "I/P"),
])
def test_fix_typical_confusions(raw, fixed):
    assert fix(raw) == fixed


def test_reads_horizontal_and_vertical_text():
    d = Drawing(80, 40, texts=[Text(40, 10, 3.0, "Maßstab 1:50"), Text(15, 26, 2.5, "4,64", 90)])
    gray = render(d, 10)
    texts, mask, _ = find_texts((gray < 128).astype(np.uint8), gray, 10)
    assert sorted((t.text, t.angle) for t in texts) == [("4,64", 90.0), ("Maßstab 1:50", 0.0)]
    assert mask.any()
    assert detect_scale(texts) == 50


def test_sheet_of_vertical_pipe_labels_is_not_turned():
    words = ["Zulauf Kessel", "Kühlwasser", "Rücklauf", "Abgas Kamin", "Heizung Vorlauf", "Speisewasser", "Kondensat",
             "Abwasser"]  # more upright vertical text than horizontal text
    labels = [Text(10 + 8 * i, 60, 2.5, w, 90) for i, w in enumerate(words)]
    d = Drawing(140, 100, texts=labels + [Text(100, 10 + 6 * i, 2.5, f"Pumpe P-10{i}") for i in range(3)])
    gray = render(d, 10)
    assert find_texts((gray < 128).astype(np.uint8), gray, 10, True)[2] == 0


def test_text_lines_beside_a_slanted_stroke_stay_apart():
    # the slanted side of an off-page connector next to both lines of its text
    d = Drawing(80, 30, lines=[Line(37, 8.5, 41, 15.5, 0.35)],
                texts=[Text(22, 10, 2.0, "FROM E-4711"), Text(22, 14.5, 2.0, "DWG-04711-0002")])
    gray = render(d, 10)
    texts, _, _ = find_texts((gray < 128).astype(np.uint8), gray, 10)
    assert sorted(t.text for t in texts if not t.uncertain) == ["DWG-04711-0002", "FROM E-4711"]


def test_a_character_read_unlike_its_look_alikes_is_corrected():
    d = Drawing(200, 130, texts=[Text(25 + 45 * (k % 4), 10 + 12 * (k // 4), 2.5, "P-1000") for k in range(20)])
    gray = render(d, 10)
    cc = Components((gray < 128).astype(np.uint8), gray)
    groups, _ = _group(cc, np.arange(cc.n), horizontal=True)
    results = [("P-1000", 90.0)] * len(groups)
    results[3] = ("P-1080", 90.0)  # the third 0 read as 8, as the slashed zero of a plotter font often is
    assert [s for s, _ in _voted(cc, [(g, False) for g in groups], results)[0]] == ["P-1000"] * len(groups)


def test_a_rare_character_keeps_its_reading():
    d = Drawing(200, 130, texts=[Text(25 + 45 * (k % 4), 10 + 12 * (k // 4), 2.5, "P-1000") for k in range(19)]
                + [Text(160, 58, 2.5, "P-1006")])
    gray = render(d, 10)
    cc = Components((gray < 128).astype(np.uint8), gray)
    groups, _ = _group(cc, np.arange(cc.n), horizontal=True)
    results = [("P-1006" if cc.x[g].min() > 1400 and cc.y[g].min() > 500 else "P-1000", 90.0) for g in groups]
    assert sorted(s for s, _ in _voted(cc, [(g, False) for g in groups], results)[0]) == ["P-1000"] * 19 + ["P-1006"]


def _twenty_lines() -> tuple[Components, list[tuple[list[int], bool]]]:
    """Components of twenty lines "P-1000" and their groups, in the order they are drawn (rows of four)."""
    d = Drawing(200, 130, texts=[Text(25 + 45 * (k % 4), 10 + 12 * (k // 4), 2.5, "P-1000") for k in range(20)])
    gray = render(d, 10)
    cc = Components((gray < 128).astype(np.uint8), gray)
    groups, _ = _group(cc, np.arange(cc.n), horizontal=True)
    return cc, [(g, False) for g in sorted(groups, key=lambda g: (cc.y[g].min() // 50, cc.x[g].min()))]


def test_a_vote_without_a_clear_majority_changes_nothing_and_says_what_it_would_choose():
    cc, groups = _twenty_lines()
    results = [("P-1888" if k % 3 == 0 else "P-1000", 90.0) for k in range(20)]  # the zeros of 7 lines read as 8
    voted, every = _voted(cc, groups, results, careful=True)
    assert voted == results and set(every) == {"P-1000"}  # 13 lines to 7 is no clear majority: left to mark
    assert {s for s, _ in _voted(cc, groups, results)[0]} == {"P-1000"}  # not careful: chosen, as before


def test_half_of_the_look_alikes_read_one_way_do_not_turn_the_other_half():
    cc, groups = _twenty_lines()
    results = [("P-1888" if k % 2 == 0 else "P-1000", 90.0) for k in range(20)]
    assert {s for s, _ in _voted(cc, groups, results)[0]} == {"P-1888"}  # all 60 zeros as 8, on a vote of half
    assert _voted(cc, groups, results, careful=True)[0] == results


def test_a_character_two_readers_read_alike_is_not_voted_over():
    cc, groups = _twenty_lines()
    results = [("P-1000", 90.0)] * 20
    results[3] = ("P-1080", 90.0)
    assert _voted(cc, groups, results, frozenset({(3, 4)}), careful=True)[0][3][0] == "P-1080"
    assert _voted(cc, groups, results, careful=True)[0][3][0] == "P-1000"  # else a clear majority still corrects it


def test_the_characters_a_spotted_line_and_the_text_recognition_read_alike():
    from app.plan2cad.ocr import Readings, _agreed
    from app.plan2cad.spotting import Spot

    cc, groups = _twenty_lines()
    g = groups[0]
    text, readings = Text(25, 10, 2.5, "P-1080"), Readings(by_token=True)
    sure, unsure = Spot(0, 0, 1, 1, "P-1000", 0.95), Spot(0, 0, 1, 1, "P-1000", 0.5)
    given = [(text.text, 90.0), ("P-1000", 95.0)]
    assert _agreed(cc, [g, g], given, [text], readings, [sure]) == {(1, j) for j in (0, 1, 2, 3, 5)}
    assert _agreed(cc, [g, g], given, [text], readings, [unsure]) == set()
    readings.keep(text, "P-1000", "P-1000")  # the text recognition read 1000 before its own vote
    assert (1, 4) in _agreed(cc, [g, g], given, [text], readings, [sure])
    assert _agreed(cc, [g, g], given, [replace(text, uncertain=True)], readings, [sure]) == set()


def test_readings_before_and_after_the_vote_are_kept_where_they_differ_in_characters_only():
    from app.plan2cad.ocr import Readings

    t, u, readings = Text(0, 0, 2.5, "P-1000"), Text(0, 0, 2.5, "P-1000"), Readings(by_token=True)
    readings.keep(t, "P-1080", "P-1000")
    readings.keep(u, "P-1000", "P-1000")  # the vote had no say
    assert readings.votes(t) == ("P-1080", "P-1000") and readings.votes(u) is None
    readings.keep(u, "P-10 80", "P-1000")  # put right in more than characters after the vote
    assert readings.votes(u) is None and readings.votes(Text(0, 0, 2.5, "P-1000")) is None


def test_frames_are_the_strokes_around_a_text():
    box = Polyline([(10, 10), (40, 10), (40, 20), (10, 20)], [0.0] * 4, closed=True, width=0.35)
    d = Drawing(80, 40, polylines=[box], texts=[Text(25, 15, 2.5, "P-101"), Text(60, 30, 2.5, "T=EL")])
    gray = render(d, 10)
    ink = (gray < 128).astype(np.uint8)
    _, labels, _ = find_texts(ink, gray, 10)
    frame = frames(ink, labels, 10)
    assert frame[100, 250] and frame[150, 400]  # the box around P-101
    assert not frame[250:350, 450:750].any()  # T=EL stands free


def test_a_single_letter_is_read_in_a_frame_but_not_on_open_paper():
    d = Drawing(90, 40, circles=[Circle(20, 20, 5, 0.35)], texts=[
        Text(20, 18, 2.0, "FIC"), Text(20, 22.5, 2.0, "B"),  # an instrument bubble with its loop letter
        Text(60, 20, 2.0, "B"), Text(60, 32, 2.0, "LINE 101")])
    gray = render(d, 10)
    ink = (gray < 128).astype(np.uint8)
    texts, _, _ = find_texts(ink, gray, 10, frame=ink)
    assert sorted((t.text, round(t.x)) for t in texts if not t.uncertain) == [("B", 20), ("FIC", 20), ("LINE 101", 60)]


def test_a_stroke_beside_a_text_is_not_read_as_part_of_it():
    d = Drawing(60, 30, lines=[Line(25.5, 13.8, 25.5, 16.2, 0.3)], texts=[Text(30, 15, 2.0, "LT")])
    gray = render(d, 10)
    texts, labels, _ = find_texts((gray < 128).astype(np.uint8), gray, 10)
    assert [t.text for t in texts] == ["LT"]
    assert not labels[140:160, 253:258].any()  # the stroke stays linework


def test_letters_in_small_boxes_are_read_without_the_boxes():
    corners = [(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)]  # two 5 mm boxes side by side, as line sizes are drawn
    boxes = [Line(x + p[0], 12.5 + p[1], x + q[0], 12.5 + q[1], 0.35)
             for x in (15, 21.5) for p, q in zip(corners, corners[1:])]
    texts = [Text(17.5, 15, 1.4, "K7"), Text(24, 15, 1.4, "M3"), Text(45, 15, 1.8, "LINE 101")]
    d = Drawing(60, 30, lines=boxes, texts=texts)
    gray = render(d, 10)
    texts, labels, _ = find_texts((gray < 128).astype(np.uint8), gray, 10)
    assert sorted(t.text for t in texts) == ["K7", "LINE 101", "M3"]
    assert not labels[123:127].any()  # the boxes stay linework


@pytest.mark.parametrize("text, scale", [("Maßstab 1:50", 50), ("M 1:100", 100), ("Matstab 1 50", 50),
                                         ("Maßstab 1 SO", 50), ("Maßstab I:1O0", 100), ("1 SO", None),
                                         ("Blatt 1/1", None), ("PID 1 50", None),
                                         ("12,50 m", None), ("15.62 m’", None)])
def test_detect_scale(text, scale):
    assert detect_scale([Text(0, 0, 2.5, text)]) == scale


def read(*texts):
    return [t.text for t in grammar([Text(10, 10, 2.5, s) for s in texts])]


def test_inch_marks_are_put_right_on_a_sheet_that_writes_pipe_sizes_in_inches():
    sheet = ["2° -AB12", "4'x3'", '6" -CD34', "1/2'PIPE", "2'", "45°", "60°C", "12'-6\""]
    assert read(*sheet) == ['2" -AB12', '4"x3"', '6" -CD34', '1/2"PIPE', '2"', "45°", "60°C", "12'-6\""]
    assert read("2'", "45°", "4'x3'") == ["2'", "45°", "4'x3'"]  # one size is no evidence: a floor plan's feet


def test_a_colon_in_a_key_value_label_is_its_equals_sign_where_the_sheet_writes_such_labels():
    assert read("PN=16", "M=XY", "T:AB", "NOTE: SEE", "NAME : X") == ["PN=16", "M=XY", "T=AB", "NOTE: SEE", "NAME : X"]
    assert read("PN=16", "T:AB") == ["PN=16", "T:AB"]


def test_a_text_whose_letters_were_found_only_in_part_keeps_its_size():
    from app.plan2cad.ocr import _length

    assert _length(0, 120, 20, "AB=12XY", 10) == pytest.approx(12.0)
    assert _length(0, 30, 20, "AB=12XY", 10) is None  # the letters touching a line were not found: not fitted to that


@pytest.mark.parametrize("read, parts", [("AB-12 34", ["AB-12", "34"]), ("AB-1234", ["AB-12", "34"]),
                                         ("AB-2 34", ["AB-2", "34"])])  # one character not read: cut at the space
def test_a_line_read_across_the_end_of_a_frame_is_two_texts(read, parts):
    from app.plan2cad.ocr import _split

    d = Drawing(60, 20, lines=[Line(30, 4, 30, 16)], texts=[Text(20, 10, 2.5, "AB-12"), Text(36, 10, 2.5, "34")])
    gray = render(d, 10)
    cc = Components((gray < 128).astype(np.uint8), gray)
    letters = np.flatnonzero(cc.h < 40)  # all but the stroke
    assert [t for _, t in _split(cc, letters, False, read, gray < 128)] == parts
    assert [t for _, t in _split(cc, letters, False, "AB-12 34", np.zeros_like(gray, bool))] == ["AB-12 34"]


def test_a_line_through_a_text_close_between_two_characters_cuts_it_only_at_a_space():
    from app.plan2cad.ocr import _glyphs, _split

    gray = render(Drawing(60, 20, texts=[Text(20, 10, 2.5, "AB-12"), Text(27.8, 10, 2.5, "34")]), 10)
    cc = Components((gray < 128).astype(np.uint8), gray)
    letters = np.arange(cc.n)
    glyphs = _glyphs(cc, letters.tolist(), False)
    x = (max(cc.x[i] + cc.w[i] for i in glyphs[4]) + min(cc.x[i] for i in glyphs[5])) // 2  # between "2" and "3"
    dark = gray < 128
    dark[40:160, x] = True  # a wall drawn through a room's area
    assert [t for _, t in _split(cc, letters, False, "AB-1234", dark)] == ["AB-1234"]
    assert [t for _, t in _split(cc, letters, False, "AB-12 34", dark)] == ["AB-12", "34"]


@pytest.mark.parametrize("angle", [-20, 50, 160])
def test_a_text_at_any_angle_is_read_along_it(angle):
    d = Drawing(100, 70, texts=[Text(50, 35, 3.0, "Lindenstraße", angle), Text(20, 12, 2.5, "Flur 4")])
    gray = render(d, 10)
    ink = (gray < 128).astype(np.uint8)
    texts, labels, _ = find_texts(ink, gray, 10, oblique=True)
    turned = next(t for t in texts if t.text == "Lindenstraße")
    assert turned.angle == pytest.approx(angle, abs=1) and (turned.x, turned.y) == pytest.approx((50, 35), abs=0.5)
    assert next(t for t in texts if t.text == "Flur 4").angle == 0  # a horizontal line stays one
    around = (slice(220, 480), slice(320, 680))
    assert (labels[around][ink[around] > 0] > 0).mean() > 0.95  # its letters leave the linework
    plain, _, _ = find_texts(ink, gray, 10)
    assert "Lindenstraße" not in [t.text for t in plain if t.angle == pytest.approx(angle, abs=1)]


def test_characters_off_a_straight_line_are_no_oblique_line():
    from app.plan2cad.ocr import _straight

    d = Drawing(60, 40, texts=[Text(20, 20, 2.5, "A", -20), Text(24, 21.5, 2.5, "B", -20), Text(28, 26, 2.5, "C", -20)])
    gray = render(d, 10)
    cc = Components((gray < 128).astype(np.uint8), gray)
    assert _straight(cc, list(range(cc.n)), -20.0) is None  # the third one is a line lower
    assert _straight(cc, list(range(2)), -20.0) is None  # two are no line


def test_of_a_line_read_both_ways_round_the_surer_reading_stays_and_the_other_one_where_plausible():
    from app.plan2cad.ocr import _either

    def read(s, conf, uncertain=False):
        return Text(0, 0, 2.5, s, confidence=conf, uncertain=uncertain), [0]

    keep, other = _either(read("0,60", 92), read("09'0", 88))
    assert keep[0].text == "0,60" and other.text == "09'0"
    assert _either(read("0,60", 70), read("09'0", 95))[0][0].text == "09'0"  # surer by more than SURER
    assert _either(read("0,60", 92), read("t9'7", 40))[1] is None  # not plausible
    assert _either(None, read("0,60", 90))[0][0].text == "0,60"
    assert _either(read("AB", 60, True), read("CD", 55))[0][0].text == "CD"  # a sure reading before a guess
    keep, other = _either(read("501", 70), read("10%", 95), level=True)
    assert keep[0].text == "501" and other.text == "10%"  # a level line is not read upside down on confidence
    assert _either(None, read("Lindenstraße", 90), level=True)[0][0].text == "Lindenstraße"  # but upside down it is


def test_a_single_character_alone_in_a_frame_is_kept_for_the_kind_to_read():
    from app.plan2cad.ocr import Readings

    d = Drawing(90, 40, circles=[Circle(20, 20, 5, 0.35)], texts=[
        Text(20, 18, 2.0, "FIC"), Text(20, 22.5, 2.0, "B"), Text(60, 32, 2.0, "LINE 101")])
    gray = render(d, 10)
    ink = (gray < 128).astype(np.uint8)
    readings = Readings()
    texts, _, _ = find_texts(ink, gray, 10, frame=ink, readings=readings)
    assert [t.text for t in texts if readings.alone(t)] == ["B"]


def test_a_short_word_leaning_less_than_its_letters_can_tell_stays_level():
    d = Drawing(120, 60, texts=[Text(25, 15, 2.5, "KWR", -5), Text(70, 40, 3.0, "Lindenstraße", -8)])
    gray = render(d, 10)
    texts, _, _ = find_texts((gray < 128).astype(np.uint8), gray, 10, oblique=True)
    angles = {t.text: t.angle for t in texts}
    assert angles["KWR"] == 0  # three letters leaning 5 degrees, as on a photo: no direction of their own
    assert angles["Lindenstraße"] == pytest.approx(-8, abs=1)  # a long line tells 8 degrees well


def test_a_level_label_that_letters_of_an_oblique_text_beside_it_joined_stays_level():
    import math

    from app.plan2cad.render import text_length

    street = Text(0, 0, 3.0, "Lindenstraße", 40)
    a, half = math.radians(40), text_length(street) / 2  # the street name starts at (40, 60), 0.6 mm after the label
    street = Text(40 + half * math.cos(a), 60 - half * math.sin(a), 3.0, "Lindenstraße", 40)
    label = Text(40 - 0.6 - text_length(Text(0, 0, 3.0, "Flur 12")) / 2, 60, 3.0, "Flur 12")
    gray = render(Drawing(120, 80, texts=[street, label]), 10)
    texts, _, _ = find_texts((gray < 128).astype(np.uint8), gray, 10, oblique=True)
    assert [t.angle for t in texts if t.text.startswith("F")] == [0]  # it bends at its end: no oblique line


def test_a_vertical_text_is_read_top_to_bottom_too_and_measured_along_it():
    from app.plan2cad.ocr import Readings

    d = Drawing(60, 60, texts=[Text(20, 30, 2.5, "Zulauf 120", -90), Text(40, 30, 2.5, "0,60", 90),
                               Text(50, 30, 2.5, "12", -90)])
    gray = render(d, 10)
    ink = (gray < 128).astype(np.uint8)
    readings = Readings()
    texts, _, _ = find_texts(ink, gray, 10, both=True, readings=readings)
    down = next(t for t in texts if t.text == "Zulauf 120")
    assert down.angle == -90 and (down.x, down.y, down.height) == pytest.approx((20, 30, 2.5), abs=0.2)
    up = next(t for t in texts if t.text == "0,60")
    assert up.angle == 90 and readings.other(up).angle == -90  # "09'0" kept for the kind of drawing to choose
    assert [t.text for t in texts if t.angle == -90] == ["Zulauf 120"]  # two glyphs may be the barbs of an arrow
    plain, _, _ = find_texts(ink, gray, 10)
    assert all(t.angle != -90 for t in plain)
