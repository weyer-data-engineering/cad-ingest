import numpy as np
import pytest

from app.plan2cad.model import Circle, Drawing, Line, Polyline, Text
from app.plan2cad.ocr import Components, _group, _voted, find_texts, fix, frames
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
    assert [s for s, _ in _voted(cc, [(g, False) for g in groups], results)] == ["P-1000"] * len(groups)


def test_a_rare_character_keeps_its_reading():
    d = Drawing(200, 130, texts=[Text(25 + 45 * (k % 4), 10 + 12 * (k // 4), 2.5, "P-1000") for k in range(19)]
                + [Text(160, 58, 2.5, "P-1006")])
    gray = render(d, 10)
    cc = Components((gray < 128).astype(np.uint8), gray)
    groups, _ = _group(cc, np.arange(cc.n), horizontal=True)
    results = [("P-1006" if cc.x[g].min() > 1400 and cc.y[g].min() > 500 else "P-1000", 90.0) for g in groups]
    assert sorted(s for s, _ in _voted(cc, [(g, False) for g in groups], results)) == ["P-1000"] * 19 + ["P-1006"]


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


@pytest.mark.parametrize("text, scale", [("Maßstab 1:50", 50), ("M 1:100", 100), ("Matstab 1 50", 50),
                                         ("Maßstab 1 SO", 50), ("Maßstab I:1O0", 100), ("1 SO", None),
                                         ("Blatt 1/1", None), ("PID 1 50", None),
                                         ("12,50 m", None), ("15.62 m’", None)])
def test_detect_scale(text, scale):
    assert detect_scale([Text(0, 0, 2.5, text)]) == scale
