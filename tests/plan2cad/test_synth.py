"""The P&ID in ISA style with the details of printed sheets, and its slashed noughts (bench/synth.py)."""

import math

import pytest

import run  # bench/run.py; bench is on the path (conftest)
from app.plan2cad.library import STARTER, read
from app.plan2cad.model import Drawing, Text
from app.plan2cad.render import text_length
from synth import FLANGE_GAP, FLUSH, FLUSH_BAR, FLUSH_NUMBER, HAIRLINE, pid_details, slashed


@pytest.fixture(scope="module")
def sheet():
    details = []
    return pid_details(1, details), details


def test_valves_and_bubbles_in_a_pipe_stand_between_two_flanges(sheet):
    d, details = sheet
    flanges = [((x0 + x1) / 2, (y0 + y1) / 2, min(x1 - x0, y1 - y0)) for kind, (x0, y0, x1, y1) in details
               if kind == "flange"]
    assert {round(gap, 6) for _, _, gap in flanges} == {FLANGE_GAP}  # two bars 1 mm apart
    inline = [s for s in d.symbols if s.name.endswith("_VALVE")] + \
             [s for s in d.symbols if s.name == "INSTRUMENT" and s.attributes["FUNCTION"].text in ("FT", "FI")]
    assert len(inline) == 14
    for s in inline:
        assert sum(math.hypot(x - s.x, y - s.y) < 8 for x, y, _ in flanges) == 2, s.name


def test_flush_bubbles_in_a_pipe_are_drawn_smaller_with_their_flanges_against_the_ring():
    details = []
    d = pid_details(1, details, flush=True)
    flanges = [box for kind, box in details if kind == "flange"]
    bubbles = [s for s in d.symbols if s.name == "INSTRUMENT" and s.attributes["FUNCTION"].text in ("FT", "FI")]
    assert len(bubbles) == 3 and {s.scale for s in bubbles} == {FLUSH}
    for s in bubbles:
        beside = [(x0, x1, y1 - y0) for x0, y0, x1, y1 in flanges if y0 < s.y < y1 and abs((x0 + x1) / 2 - s.x) < 8]
        assert len(beside) == 2  # the pen of the inner bar (0.25 mm) against that of the ring (0.35 mm)
        assert all(min(abs(x0 - s.x), abs(x1 - s.x)) == pytest.approx(5 * FLUSH + 0.3) for x0, x1, _ in beside)
        assert all(bar == pytest.approx(FLUSH_BAR) for *_, bar in beside)  # reaching down beside the number
        assert s.attributes["NUMBER"].y == pytest.approx(s.y + FLUSH_NUMBER)  # in the lower half
        assert [t.height for t in s.attributes.values()] == pytest.approx([2.0 * FLUSH] * 2)  # lettered so


def test_its_symbols_are_of_the_starter_library_and_its_details_are_linework(sheet):
    d, details = sheet
    assert {s.name for s in d.symbols} <= set(read(STARTER).symbols)
    assert {kind for kind, _ in details} == {"flange", "label", "box", "diamond", "spec break", "zigzag", "cross",
                                             "double rule"}


def test_its_codes_begin_with_a_letter_and_hold_noughts(sheet):
    d, _ = sheet
    texts = [t.text for t in d.texts]
    assert "X1-01234-5678" in texts
    assert sum(t.startswith(("P-", '2"-P-', '3"-P-', '4"-P-')) and "0" in t for t in texts) >= 5
    assert "B-501" in texts and '2"x1"' in texts


def test_a_slash_strikes_each_nought_in_the_image_only():
    texts = [Text(50, 50, 2.5, "P-40208"), Text(20, 80, 2.0, "10", angle=90), Text(80, 50, 2.5, "ABC")]
    d = Drawing(100, 100, texts=texts)
    out = slashed(d)
    assert d.lines == [] and out.texts == d.texts  # the reference keeps its texts
    lying, standing = out.lines[:2], out.lines[2:]
    assert len(lying) == 2 and len(standing) == 1
    half = text_length(texts[0]) / 2
    middles = [(q.x1 + q.x2) / 2 for q in lying]
    assert 50 - half < middles[0] < middles[1] < 50 + half  # the noughts of 40208, left to right
    for q in lying:  # from low left to high right, inside the letters
        assert q.x1 < q.x2 and q.y1 > q.y2 and 48.75 < q.y2 < q.y1 < 51.25
    q = standing[0]  # the 0 of a text read from bottom to top is its upper character
    assert abs((q.x1 + q.x2) / 2 - 20) < 0.2 and (q.y1 + q.y2) / 2 < 80


def test_texts_without_noughts_get_no_slash():
    d = Drawing(100, 100, texts=[Text(50, 50, 2.5, "ABC-123")])
    assert slashed(d).lines == []


def test_the_benchmark_runs_it_as_a_scan_and_a_photo_of_its_a3_print():
    cases = [c for c in run.CASES if c[2] is pid_details]
    assert [(c[1], c[3], c[5]) for c in cases] == [(run.PID, "A3", [1, 2])] * 2
    assert [c[0].rsplit(", ", 1)[1] for c in cases] == ["scan 300 dpi", "phone photo 12 MP"]


def test_with_pen_marks_its_signal_lines_are_in_a_pen_and_end_in_arrowheads():
    details = []
    d, plain = pid_details(1, details, pen_marks=True), pid_details(1)
    assert d.symbols == plain.symbols and d.texts == plain.texts  # the same sheet
    widths = [q.width for q in d.lines + d.polylines]
    assert min(widths) == 0.25 and min(q.width for q in plain.lines + plain.polylines) == HAIRLINE
    assert sum(kind == "arrowhead" for kind, _ in details) == 6
    printed = [c for c in run.CASES if "marks in pen" in c[0]]
    assert [(c[1], c[3]) for c in printed] == [(run.PID, "A4")]
