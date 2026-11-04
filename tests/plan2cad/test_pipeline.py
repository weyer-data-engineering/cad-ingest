import cv2
import numpy as np
import pytest
from evaluate import evaluate
from synth import building_plan, photo, scan

from app.plan2cad.pipeline import convert
from app.plan2cad.render import render


def test_scan_recovers_geometry_text_and_scale(plan, plan_scan):
    r = convert(plan_scan, paper="A3")
    m = evaluate(plan, r.drawing)
    assert r.info["sheet"] == "paper"
    assert m["recall_1mm"] > 0.97 and m["precision_1mm"] > 0.97
    assert m["text_exact"] > 0.9
    assert r.drawing.scale == 50
    assert not r.drawing.symbols  # no fixtures or marks drawn, none found (building-symbols is used on floor plans)
    assert m["dimension_recall"] > 0.9 and m["dimension_precision"] > 0.9  # the dimension chains as dimensions


def test_fixtures_on_a_floor_plan_become_library_blocks():
    plan = building_plan(7, fixtures=True)
    r = convert(scan(plan, 7, dpi=200), paper="A3")
    m = evaluate(plan, r.drawing)
    assert r.info["drawing_type"] == "building"
    assert [lib["id"] for lib in r.info["libraries"]] == ["floor-plan", "building-symbols"]
    assert m["symbol_recall"] > 0.85 and m["symbol_precision"] > 0.85
    assert m["recall_1mm"] > 0.97  # the walls behind them stay


def test_phone_photo_is_rectified(plan):
    r = convert(photo(plan, 7), paper="A3")
    m = evaluate(plan, r.drawing)
    assert r.info["sheet"] == "paper"
    assert m["recall_1mm"] > 0.9 and m["precision_1mm"] > 0.9


def test_upside_down_scan_is_turned(plan, plan_scan):
    r = convert(np.ascontiguousarray(plan_scan[::-1, ::-1]), paper="A3")
    assert r.info["rotation_deg"] == 180
    assert evaluate(plan, r.drawing)["recall_1mm"] > 0.95


def test_scan_without_paper_edge_uses_its_resolution(plan):
    img = cv2.cvtColor(render(plan, 200 / 25.4), cv2.COLOR_GRAY2BGR)
    r = convert(img, paper="A3", dpi=200)
    assert r.info["sheet"] == "image"
    assert evaluate(plan, r.drawing)["recall_1mm"] > 0.97


def test_lines_in_colour_are_ink(plan):
    img = cv2.cvtColor(render(plan, 200 / 25.4), cv2.COLOR_GRAY2BGR)
    img[img[..., 0] < 128] = (0, 255, 255)  # yellow, as lines in CAD often are: pale grey in plain luminance
    assert evaluate(plan, convert(img, paper="A3", dpi=200).drawing)["recall_1mm"] > 0.97


@pytest.mark.parametrize("height, count, size", [(1.2, 30, 0.6), (2.1, 30, 1.0), (1.2, 5, 1.0), (5.0, 30, 2.0)])
def test_symbols_go_with_the_text_size_of_the_sheet(height, count, size):
    from app.plan2cad.model import Text
    from app.plan2cad.pipeline import _printed

    assert _printed([Text(0, 0, height, "P-101")] * count) == pytest.approx(size)


@pytest.mark.parametrize("height, share", [(2.5, 1.0), (1.25, 0.57), (0.8, 0.36)])
def test_a_drawing_printed_small_is_told_by_its_small_lettering(height, share):
    from app.plan2cad.model import Drawing, Text
    from app.plan2cad.pipeline import printed_size

    texts = [Text(20 + 30 * (k % 8), 10 + 8 * (k // 8), height, "P-101 AB") for k in range(80)]
    ink = (render(Drawing(260, 100, texts=texts), 12) < 128).astype(np.uint8)
    assert printed_size(ink, 12) == pytest.approx(share, abs=0.08)


def test_specks_of_dirt_are_no_lettering():
    from app.plan2cad.model import Drawing, Text
    from app.plan2cad.pipeline import printed_size

    texts = [Text(20 + 30 * (k % 8), 10 + 8 * (k // 8), 2.5, "P-101 AB") for k in range(40)]
    ink = (render(Drawing(260, 100, texts=texts), 12) < 128).astype(np.uint8)
    rng = np.random.default_rng(1)
    for x, y in rng.uniform((0, 0), ink.shape[::-1], (1500, 2)).astype(int):  # ragged specks of 0.5 mm, as on a
        ink[y + 2:y + 4, x:x + 6] = ink[y:y + 6, x + 2:x + 4] = 1  # poor scan
    assert printed_size(ink, 12) == 1.0
    assert printed_size(ink, 12, rows=False) < 0.4  # taken for lettering of 0.5 mm, as before
    assert printed_size(ink[:, :40], 12) == 1.0  # too few characters to tell


def test_a_drawing_printed_small_is_recognised_at_its_drawn_size():
    from synth import pid, printed_on

    d = printed_on(pid(3, "A2"), "A4")  # an A2 P&ID on A4: lettering of 1 mm instead of 2
    r = convert(scan(d, 3, dpi=300), paper="A4")
    m = evaluate(d, r.drawing)
    assert r.info["printed"] == pytest.approx(0.47, abs=0.08) and r.drawing.printed == r.info["printed"]
    assert m["symbol_recall"] > 0.85 and m["symbol_precision"] > 0.9 and m["recall_1mm"] > 0.95


def test_title_block_decides_orientation_when_text_is_unreadable(plan):
    from app.plan2cad.model import Line
    from app.plan2cad.pipeline import title_block_rotation

    W, H = plan.width, plan.height
    assert title_block_rotation(plan.lines, W, H) == 0
    flipped = [Line(W - ln.x1, H - ln.y1, W - ln.x2, H - ln.y2) for ln in plan.lines]
    assert title_block_rotation(flipped, W, H) == 180


def test_text_lying_on_a_circle_ring_is_dropped():
    from app.plan2cad.model import Text
    from app.plan2cad.pipeline import _off_rings

    labels = np.zeros((10, 10), np.uint8)
    labels[0, :4], labels[5, :4] = 1, 2
    rings = np.zeros((10, 10), np.uint8)
    rings[0] = 1
    texts, mask = _off_rings([Text(0, 0, 1, "N"), Text(0, 5, 1, "LIC")], labels, rings)
    assert [t.text for t in texts] == ["LIC"]
    assert mask.sum() == 4 and mask[5, :4].all()


def test_texts_of_one_size_are_lettered_alike_but_a_tight_one_stays_inside_its_frame():
    from app.plan2cad.model import Drawing, Text
    from app.plan2cad.pipeline import _lettering

    d = Drawing(100, 100, texts=[Text(10, 10, 1.1, "AB=CD", length=4.6), Text(10, 20, 1.2, "AB=CD", length=5.4),
                                 Text(10, 30, 1.25, "AB=CD"), Text(10, 40, 1.2, "EF-12", length=2.5),
                                 Text(10, 50, 3.3, "TITLE", length=14), Text(50, 60, 1.2, "AB=CD", length=7.0)])
    _lettering(d)
    same = [(t.height, t.length) for t in d.texts[:3] + d.texts[5:]]
    assert same[0] == same[1] == same[2] == same[3] and same[0][0] == pytest.approx(1.2, abs=0.05)
    assert d.texts[3].length == 2.5 and (d.texts[4].height, d.texts[4].length) == (3.3, 14)
    assert d.texts[5].x - d.texts[5].length / 2 == pytest.approx(50 - 7.0 / 2)  # spaced wider: starts where it did


def test_a_text_read_both_ways_round_takes_the_reading_that_is_a_word_of_the_kind():
    from dataclasses import replace

    from app.plan2cad.features import Features
    from app.plan2cad.model import Text
    from app.plan2cad.ocr import Readings
    from app.plan2cad.pipeline import _settled
    from app.plan2cad.profiles import installed

    def kind(drawing):
        return [p for p in installed() if p.drawing == drawing]

    misread, framed = Text(10, 10, 2.5, "09'0", -90.0), Text(20, 10, 2.5, "8")  # "0,60" upside down, a framed "B"
    readings = Readings()
    readings.turned[id(misread)] = (misread, Text(10, 10, 2.5, "0,60", 90.0))
    readings.framed[id(framed)] = framed
    texts = [misread, framed]
    assert [(t.text, t.angle) for t in _settled(texts, readings, kind("building"), Features())] == \
        [("0,60", 90.0), ("8", 0.0)]
    letters = [replace(p, framed="letter") for p in kind("pid")]  # a kind whose frames hold letters only
    assert [t.text for t in _settled(texts, readings, letters, Features.of(on=["framed_letters"]))] == ["09'0", "B"]
    assert [t.text for t in _settled(texts, readings, kind("pid"), Features.of(on=["framed_letters"]))] == ["09'0", "8"]
    assert [t.text for t in _settled(texts, readings, kind("building"), Features.of(off=["text_direction"]))] == \
        ["09'0", "8"]
    assert [t.text for t in _settled(texts, readings, [], Features())] == ["09'0", "8"]  # no kind: the surer one


def test_a_vertical_text_only_reading_it_again_made_sure_of_stays_marked_unless_the_kind_knows_it():
    from app.plan2cad.features import Features
    from app.plan2cad.model import Text
    from app.plan2cad.ocr import Readings
    from app.plan2cad.pipeline import _settled
    from app.plan2cad.profiles import installed

    building = [p for p in installed() if p.drawing == "building"]
    point, comma = Text(10, 10, 2.5, "3.94", 90.0, 94.9), Text(20, 10, 2.5, "4,64", 90.0, 94.7)
    readings = Readings()
    readings.doubted.update({id(point): (point, 79.9), id(comma): (comma, 78.0)})
    out = _settled([point, comma], readings, building, Features())
    # the kind writes decimal commas: "3.94" stays as unsure as the spotter found it, "4,64" is a word of it
    assert [(t.text, t.uncertain, t.confidence) for t in out] == [("3.94", True, 79.9), ("4,64", False, 94.7)]
    assert all(t.uncertain for t in _settled([point, comma], readings, [], Features()))  # no kind, no word of it
    assert _settled([point, comma], readings, building, Features.of(off=["text_direction"])) == [point, comma]


def test_a_token_the_vote_of_look_alikes_had_a_say_in_is_settled_by_the_vocabulary_or_marked():
    from app.plan2cad.features import Features
    from app.plan2cad.model import Text
    from app.plan2cad.ocr import Readings
    from app.plan2cad.pipeline import _settled
    from app.plan2cad.profiles import installed

    pid = [p for p in installed() if p.drawing == "pid"]
    readings = Readings(by_token=True)
    line, tag = Text(10, 10, 2.5, "0N80-P-1011"), Text(10, 20, 2.5, "8-501 Vorlage")  # D and B voted into digits
    readings.keep(line, "DN80-P-1011", "0N80-P-1011")
    readings.keep(tag, "B-501 Vorlage", "8-501 Vorlage")
    unclear, clear = Text(20, 10, 2.5, "1080"), Text(30, 10, 2.5, "1000")  # an 8 of a look-alike 0
    readings.keep(unclear, "1080", "1000")  # no clear majority for the 0
    readings.keep(clear, "1080", "1000")  # a clear one
    out = _settled([line, tag, unclear, clear], readings, pid, Features())
    assert [(t.text, t.uncertain) for t in out] == [("DN80-P-1011", False), ("B-501 Vorlage", False), ("1080", True),
                                                    ("1000", False)]
    assert [t.text for t in _settled([line, unclear], readings, [], Features())] == ["0N80-P-1011", "1080"]  # no kind
    assert _settled([line, unclear], Readings(), pid, Features()) == [line, unclear]  # the readings were not kept


def test_tokens_with_special_characters_read_exactly_are_measured_apart():
    from evaluate import texts

    from app.plan2cad.model import Drawing, Text

    want = [Text(20, 20, 2.5, "±0,00"), Text(20, 40, 2.5, "Ø 12"), Text(20, 60, 2.5, "Raum")]
    read = [Text(20, 20, 2.5, "+0,00"), Text(20, 40, 2.5, "Ø 12"), Text(20, 60, 2.5, "Raum")]
    assert texts(Drawing(100, 100, texts=want), Drawing(100, 100, texts=read))["special_tokens_exact"] == 0.5
    assert texts(Drawing(100, 100, texts=want[2:]), Drawing(100, 100, texts=read[2:]))["special_tokens_exact"] is None


def test_texts_and_spotted_lines_on_symbols_found_first_are_dropped():
    from app.plan2cad.model import Text
    from app.plan2cad.pipeline import _off_areas, _off_symbols
    from app.plan2cad.spotting import Spot

    labels = np.zeros((10, 10), np.uint8)
    labels[0, :4], labels[5, :4] = 1, 2
    areas = np.zeros((10, 10), np.uint8)
    areas[0, :3], areas[5, :1] = 1, 1
    texts, out = _off_symbols([Text(0, 0, 1, "X"), Text(0, 5, 1, "LIC")], labels, areas)
    assert [t.text for t in texts] == ["LIC"]  # mostly on a symbol: its strokes read as a letter
    assert not out[0].any() and out[5, :4].tolist() == [0, 1, 1, 1]
    spots = [Spot(0, 0, 3, 1, "X", 0.9), Spot(0, 5, 4, 6, "LIC", 0.9)]
    assert [s.text for s in _off_areas(spots, areas)] == ["LIC"] and _off_areas(spots, None) == spots


def test_symbols_first_and_short_strokes_on_and_unclassified_off_run_through_a_whole_conversion(plan, plan_scan):
    from app.plan2cad.features import Features
    from app.plan2cad.unclassified import LAYER

    r = convert(plan_scan, paper="A3", features=Features.of(["symbols_first", "short_strokes"], ["unclassified"]))
    m = evaluate(plan, r.drawing)
    assert m["recall_1mm"] > 0.97 and m["precision_1mm"] > 0.97 and m["text_exact"] > 0.9 and r.drawing.scale == 50
    assert not r.drawing.unclassified and LAYER not in {q.layer for q in r.drawing.lines + r.drawing.polylines}


def test_shapes_not_classified_are_counted_in_the_warnings():
    from app.plan2cad.model import Drawing, Unclassified
    from app.plan2cad.pipeline import Result
    from app.scan import _warnings

    d = Drawing(200, 100, unclassified=[Unclassified((1, 2, 6, 5), "4 strokes")] * 2)
    assert _warnings(Result(d, np.zeros((1, 1), np.uint8), 1.0, {"sheet": "paper"})) == [
        "2 shapes look like symbols but were recognised as none of the libraries'; they stay linework on the layer "
        "P2C-SYMBOL-UNCLASSIFIED"]


def test_small_lettering_that_blur_leaves_partly_alone_still_sets_the_print_size():
    from app.plan2cad.model import Drawing, Text
    from app.plan2cad.pipeline import printed_size

    small = [Text(20 + 30 * (k % 8), 10 + 8 * (k // 8), 0.8, "P-101 AB") for k in range(24)]
    alone = [Text(15 + 6 * (k % 40), 40 + 6 * (k // 40), 0.8, "8") for k in range(80)]  # neighbours merged by blur
    large = [Text(20 + 30 * (k % 8), 60 + 8 * (k // 8), 1.25, "P-101 AB") for k in range(30)]  # the title block's
    ink = (render(Drawing(260, 100, texts=small + alone + large), 12) < 128).astype(np.uint8)
    assert printed_size(ink, 12) == pytest.approx(0.36, abs=0.05)  # the small lettering: most of it stands in rows
    assert printed_size(ink, 12, rows=False) == pytest.approx(0.36, abs=0.05)


def test_line_types_are_told_apart_and_the_new_steps_are_skipped_when_switched_off(monkeypatch):
    from app.plan2cad import objects, vectorize
    from app.plan2cad.features import Features
    from app.plan2cad.model import Drawing, Line, Text, line_type
    from app.plan2cad.simulate import plot

    calls = []
    for module, name in ((vectorize, "dotted"), (vectorize, "dash_dots"), (vectorize, "line_types"),
                         (vectorize, "calibrate_widths"), (objects, "curves")):
        monkeypatch.setattr(module, name, lambda *a, _f=getattr(module, name), _n=name: calls.append(_n) or _f(*a))
    shapes = vectorize.shapes
    monkeypatch.setattr(vectorize, "shapes", lambda *a: calls.append(a[2]) or shapes(*a))  # the largest radius
    d = Drawing(297, 210, lines=[Line(20, 10, 287, 10, 0.7), Line(20, 200, 287, 200, 0.7), Line(20, 10, 20, 200, 0.7),
                                 Line(287, 10, 287, 200, 0.7), Line(40, 40, 200, 40, 0.25, True),
                                 Line(40, 70, 200, 70, 0.25, True, linetype="dashdot"),
                                 Line(40, 100, 200, 100, 0.35, True, linetype="dotted")],
                texts=[Text(150, 180, 3.5, "Line types")])
    img = plot(d, 250)
    on = convert(img, paper="A4", dpi=250).drawing
    assert set(calls) == {"dotted", "dash_dots", "line_types", "calibrate_widths", "curves", objects.LARGEST}
    calls.clear()
    off = convert(img, paper="A4", dpi=250, features=Features.of(off=["linetypes", "width_calibration", "splines"]))
    assert set(calls) == {vectorize.LARGEST}  # arcs of up to 250 mm radius, as before
    assert evaluate(d, on)["linetype_correct"] == 1.0
    assert {line_type(q) for q in off.drawing.lines if q.dashed} == {"dashed"}
    assert evaluate(d, off.drawing)["linetype_correct"] < 0.5


def test_curves_are_drawn_before_the_title_block_is_read_and_the_pens_measured_last(monkeypatch):
    from app.plan2cad import objects, titleblock, unclassified, vectorize
    from app.plan2cad.features import Features
    from app.plan2cad.model import Drawing, Line, Text
    from app.plan2cad.simulate import plot

    calls = []
    steps = ((objects, "curves"), (titleblock, "find"), (unclassified, "find"), (vectorize, "calibrate_widths"),
             (objects, "pens"))
    for module, name in steps:  # the chords of a curve in the title block would be rulings of it
        monkeypatch.setattr(module, name, lambda *a, _f=getattr(module, name), _n=name: calls.append(_n) or _f(*a))
    d = Drawing(297, 210, lines=[Line(20, 10, 287, 10, 0.7), Line(20, 200, 287, 200, 0.7), Line(20, 10, 20, 200, 0.7),
                                 Line(287, 10, 287, 200, 0.7)], texts=[Text(150, 100, 3.5, "Order of the steps")])
    convert(plot(d, 150), paper="A4", dpi=150, features=Features.of(on=["pens"]))
    assert calls == ["curves", "find", "find", "calibrate_widths", "pens"]  # the title block's, then the shapes'


def test_a_single_letter_on_bars_across_a_pipe_is_linework_but_a_digit_above_a_line_is_not():
    from app.plan2cad.model import Line, Text
    from app.plan2cad.pipeline import _bars

    ppm, ink = 10, np.zeros((400, 400), np.uint8)
    cv2.line(ink, (100, 0), (100, 399), 1, 5)  # a vertical pipe at x 10 mm ...
    for y in (195, 205):  # ... with the two bars of a flange across it, read along it as "H"
        cv2.line(ink, (83, y), (117, y), 1, 3)
    pipe = Line(10, 0, 10, 40, 0.5)
    # read from the bars right of the pipe only, as where the pipe cut them in two
    assert _bars(Text(11.1, 20, 1.3, "H", angle=90, length=1.5), [pipe], ink, ppm)
    assert not _bars(Text(11.1, 20, 1.3, "HH", angle=90, length=3.0), [pipe], ink, ppm)  # one character only
    short = Line(10, 0, 10, 20.9, 0.5)  # a pipe that stops at the bars does not run on past them
    assert not _bars(Text(11.1, 20, 1.3, "H", angle=90, length=1.5), [short], ink, ppm)
    above = np.zeros((400, 400), np.uint8)
    cv2.line(above, (0, 300), (399, 300), 1, 3)  # a dimension line at y 30 mm and the digit touching it from above
    cv2.line(above, (195, 280), (195, 298), 1, 3)
    assert not _bars(Text(19.5, 29.0, 2.0, "1", length=0.6), [Line(0, 30, 40, 30, 0.25)], above, ppm)


def test_a_text_a_pipe_runs_into_at_both_ends_is_piped_but_a_framed_label_is_not():
    from app.plan2cad.model import Line, Text
    from app.plan2cad.pipeline import _piped

    pipes = [Line(0, 50, 46.1, 50, 0.5), Line(53.9, 50, 100, 50, 0.5)]
    assert _piped(Text(50, 50, 2.0, "Hoeat", length=8.0), pipes)  # a valve with its flanges read as "Hoeat"
    assert not _piped(Text(50, 50, 2.0, "Hoeat", length=8.0), pipes[:1])  # a pipe at one end only
    framed = [Line(0, 50, 44.4, 50, 0.5), Line(55.6, 50, 100, 50, 0.5)]  # a label 1.6 mm inside its frame
    assert not _piped(Text(50, 50, 2.0, "P-101", length=8.0), framed)
    assert not _piped(Text(50, 47.5, 2.0, "P-101", length=8.0), pipes)  # a line number above the pipe
    assert not _piped(Text(50, 48.8, 2.0, "P-101", length=8.0), [Line(0, 50, 100, 50, 0.5)])  # ... touching it
    assert _piped(Text(50, 48.9, 1.3, "1m. ll", length=8.0), pipes)  # strokes above a pipe that cut them in two


def test_text_on_pipes_and_attributes_inside_are_skipped_when_switched_off(monkeypatch):
    from app.plan2cad import pipeline, spotting, symbols
    from app.plan2cad.features import Features
    from app.plan2cad.model import Drawing, Line, Text
    from app.plan2cad.simulate import plot

    calls, inside = [], []
    for module, name in ((spotting, "off_pipes"), (pipeline, "_bars"), (pipeline, "_piped"), (spotting, "off_rings")):
        monkeypatch.setattr(module, name, lambda *a, _f=getattr(module, name), _n=name: calls.append(_n) or _f(*a))
    recognise = symbols.recognise
    monkeypatch.setattr(symbols, "recognise", lambda *a: inside.append(a[3].attributes_inside) or recognise(*a))
    d = Drawing(297, 210, lines=[Line(20, 10, 287, 10, 0.7), Line(20, 200, 287, 200, 0.7),
                                 Line(20, 10, 20, 200, 0.7), Line(287, 10, 287, 200, 0.7), Line(150, 30, 150, 150, 0.5),
                                 Line(148.3, 89.5, 151.7, 89.5, 0.25), Line(148.3, 90.5, 151.7, 90.5, 0.25)],
                texts=[Text(100, 180, 3.5, "Pipe with a flange"), Text(60, 60, 2.5, "P-101")])
    img = plot(d, 250)
    convert(img, paper="A4", dpi=250, drawing="pid")
    assert {"off_pipes", "_bars", "_piped", "off_rings"} <= set(calls) and inside == [True]
    calls.clear(), inside.clear()
    convert(img, paper="A4", dpi=250, drawing="pid", features=Features.of(off=["text_on_pipes", "attributes_inside"]))
    assert not calls and inside == [False]
