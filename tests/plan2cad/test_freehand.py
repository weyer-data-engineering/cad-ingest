"""Marks by hand in black or pencil on a print, told apart from the drawing by their shape and their grey."""
import cv2
import numpy as np
import pytest
from evaluate import evaluate
from synth import building_plan, mark_up, pid, printed_on, schematic, section

from app.plan2cad import freehand
from app.plan2cad.features import Features
from app.plan2cad.model import Drawing
from app.plan2cad.pipeline import binarize, convert, normalize
from app.plan2cad.simulate import plot, scan

PPM = 300 / 25.4


def ink_of(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = normalize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), PPM)
    return binarize(norm, PPM), norm


@pytest.mark.parametrize("draw", [
    lambda m: cv2.rectangle(m, (40, 40), (560, 300), 1, 4),
    lambda m: (cv2.rectangle(m, (60, 40), (540, 300), 1, 4), cv2.circle(m, (300, 170), 60, 1, 4)),
    lambda m: cv2.ellipse(m, (300, 170), (220, 110), 15, 0, 360, 1, 4),  # a wash basin
    lambda m: [cv2.ellipse(m, c, (40, 40), 0, a, a + 90, 1, 4) for c, a in  # a bath: straight sides, round corners
               (((100, 100), 180), ((500, 100), 270), ((500, 260), 0), ((100, 260), 90))]
    + [cv2.line(m, a, b, 1, 4) for a, b in (((100, 60), (500, 60)), ((540, 100), (540, 260)),
                                            ((500, 300), (100, 300)), ((60, 260), (60, 100)))],
])
def test_cad_shapes_are_not_marks(draw):
    m = np.zeros((340, 600), np.uint8)
    draw(m)
    assert not freehand._drawn_by_hand(m, np.where(m > 0, 30, 255).astype(np.uint8), PPM, 30.0)


def test_handwriting_clouds_and_hatching_by_hand_are_marks():
    d = mark_up(Drawing(420, 297), 3, "black")
    ink, norm = ink_of(plot(d, 300))
    hand = freehand.find(ink, norm, PPM, 2.5 * PPM)
    assert hand.sum() > 0.6 * ink.sum()


@pytest.mark.parametrize("poor", [False, True])
@pytest.mark.parametrize("make", [lambda: building_plan(2, fixtures=True), lambda: printed_on(pid(1, "A1"), "A3"),
                                  lambda: schematic(1), lambda: section(1)])
def test_a_drawing_without_marks_keeps_all_its_lines(make, poor):
    d = make()
    ink, norm = ink_of(scan(d, 2, dpi=300, poor=True, skew=0.0) if poor else plot(d, 300))
    assert not freehand.find(ink, norm, PPM, 2.5 * PPM).any()
    assert not freehand.find(ink, norm, PPM, 2.5 * PPM, strokes=True, leaders=True).any()


def test_pencil_marks_go_to_the_markup_layer_and_are_not_read():
    d = mark_up(building_plan(1), 1, "pencil")
    img = plot(d, 200)
    on = convert(img, paper="A3", dpi=200)
    off = convert(img, paper="A3", dpi=200, features=Features.of(off=["freehand"]))
    m_on, m_off = evaluate(d, on.drawing), evaluate(d, off.drawing)
    assert m_on["markup_recall"] > 0.5 and m_on["markup_precision"] > 0.9
    assert m_on["precision_1mm"] > m_off["precision_1mm"] and m_on["recall_1mm"] >= m_off["recall_1mm"] - 0.005
    notes = on.drawing.markups
    assert notes and all(n.reading is None and n.color[0] > 60 for n in notes)  # grey
    assert len(on.crops) == len(notes)


def cloud(img: np.ndarray, x: float, y: float, w: float, h: float, bump: float = 5.0) -> None:
    """Draws a revision cloud (mm) into ``img``: half circles outwards along the sides of a box, in pen strokes of
    0.4 mm."""
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
        side = float(np.hypot(bx - ax, by - ay))
        n = max(2, round(side / bump))
        for i in range(n):
            mx, my = ax + (bx - ax) * (i + 0.5) / n, ay + (by - ay) * (i + 0.5) / n
            cv2.ellipse(img, (round(mx * PPM * 16), round(my * PPM * 16)), (round(side / n / 2 * PPM * 16),) * 2,
                        float(np.degrees(np.arctan2(by - ay, bx - ax))), 180, 360, 1, 5, cv2.LINE_AA, 4)


def strokes_sheet(marked: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """A 100 x 68 mm sheet (BGR) with a grid of lines across it, the drawing's network, and a revision cloud drawn
    across two of its lines; and the mask of the cloud."""
    gray = np.full((800, 1200), 255, np.uint8)
    for y in (200, 400, 600):
        cv2.line(gray, (50, y), (1150, y), 0, 4)
    for x in (300, 600, 900):
        cv2.line(gray, (x, 50), (x, 750), 0, 4)
    pen = np.zeros(gray.shape, np.uint8)
    if marked:
        cloud(pen, 38, 26, 30, 20)
        gray[pen > 0] = 0
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), pen


def test_a_mark_drawn_across_the_linework_is_taken_stroke_by_stroke():
    img, pen = strokes_sheet()
    ink, norm = ink_of(img)
    assert not freehand.find(ink, norm, PPM, 2.5 * PPM).any()  # part of the drawing's network: not judged whole
    hand = freehand.find(ink, norm, PPM, 2.5 * PPM, strokes=True)
    assert hand[pen > 0].mean() > 0.6
    lines = np.zeros_like(ink)
    cv2.line(lines, (50, 400), (1150, 400), 1, 4)
    cv2.line(lines, (600, 50), (600, 750), 1, 4)
    away = lines & (cv2.dilate(pen, np.ones((25, 25), np.uint8)) == 0)  # the lines it crosses, 1 mm from the cloud
    assert not hand[away > 0].any() and hand.sum() < 1.2 * pen.sum()


def test_the_linework_alone_is_not_taken_stroke_by_stroke():
    ink, norm = ink_of(strokes_sheet(marked=False)[0])
    assert not freehand.find(ink, norm, PPM, 2.5 * PPM, strokes=True, leaders=True).any()


def leader_sheet(barbs: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """A 150 x 100 mm sheet (px) with a revision cloud and a curved leader from it that ends in an arrowhead drawn
    by hand (or freely); and the mask of the leader."""
    img = np.full((round(100 * PPM), round(150 * PPM), 3), 255, np.uint8)
    marks = np.zeros(img.shape[:2], np.uint8)
    cloud(marks, 20, 20, 40, 25)
    img[marks > 0] = 0
    u = np.linspace(0, 1, 60)[:, None]
    a, m, b = np.array([64.0, 32]), np.array([90.0, 25]), np.array([110.0, 70])
    curve = (1 - u) ** 2 * a + 2 * (1 - u) * u * m + u * u * b
    pen = np.zeros(img.shape[:2], np.uint8)
    cv2.polylines(pen, [np.round(curve * PPM).astype(np.int32)], False, 1, 5)
    if barbs:
        back = np.arctan2(*(curve[-4] - b)[::-1])
        for turn in (-0.45, 0.45):
            tip = b + 3 * np.array([np.cos(back + turn), np.sin(back + turn)])
            cv2.line(pen, tuple(np.round(b * PPM).astype(int)), tuple(np.round(tip * PPM).astype(int)), 1, 5)
    img[pen > 0] = 0
    return img, pen


def test_a_leader_from_a_mark_with_its_arrowhead_is_a_mark():
    img, pen = leader_sheet()
    ink, norm = ink_of(img)
    assert freehand.find(ink, norm, PPM, 2.5 * PPM).any()  # the cloud
    assert freehand.find(ink, norm, PPM, 2.5 * PPM)[pen > 0].mean() < 0.1
    assert freehand.find(ink, norm, PPM, 2.5 * PPM, leaders=True)[pen > 0].mean() > 0.8


def test_a_stroke_from_a_mark_that_ends_freely_is_no_leader():
    ink, norm = ink_of(leader_sheet(barbs=False)[0])
    _, pen = leader_sheet(barbs=False)
    assert freehand.find(ink, norm, PPM, 2.5 * PPM, leaders=True)[pen > 0].mean() < 0.1


def text_sheet(through: float, marked: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A printed text with a pen stroke through it at ``through`` (0 its middle, 0.5 its baseline), and a revision
    cloud below it if ``marked``: the image, the mask of the stroke and that of the letters."""
    from app.plan2cad.model import Text
    from app.plan2cad.render import text_length

    t = Text(60, 30, 3.5, "PUMP STATION 12")
    letters = cv2.cvtColor(plot(Drawing(120, 90, texts=[t]), 300), cv2.COLOR_BGR2GRAY) < 128
    half, y = text_length(t) / 2 + 1.5, t.y + through * t.height
    pen = np.zeros(letters.shape, np.uint8)
    start, end = np.round(np.array([[t.x - half, y - 0.3], [t.x + half, y + 0.2]]) * PPM).astype(int)
    cv2.line(pen, tuple(start.tolist()), tuple(end.tolist()), 1, 5)
    img = np.full(letters.shape + (3,), 255, np.uint8)
    img[letters | (pen > 0)] = 0
    if marked:
        cloud(img, 40, 55, 40, 25)
    return img, pen, letters


@pytest.mark.parametrize("through", [0.0, -0.15, 0.15])
def test_a_text_struck_out_loses_the_stroke_and_keeps_its_letters(through):
    img, pen, letters = text_sheet(through)
    ink, norm = ink_of(img)
    assert not freehand.find(ink, norm, PPM, 2.5 * PPM)[pen > 0].any()
    hand = freehand.find(ink, norm, PPM, 2.5 * PPM, leaders=True)
    assert hand[pen > 0].mean() > 0.5  # the rest crosses the letters, which stay
    print_only = letters & (cv2.dilate(pen, np.ones((9, 9), np.uint8)) == 0)
    assert hand[print_only].mean() < 0.02


@pytest.mark.parametrize("through", [0.55, -0.6])
def test_an_underlined_or_overlined_text_is_not_struck_out(through):
    img, pen, letters = text_sheet(through)
    ink, norm = ink_of(img)
    assert not freehand.find(ink, norm, PPM, 2.5 * PPM, leaders=True)[(pen > 0) | letters].any()


def test_a_text_struck_out_on_a_sheet_without_other_marks_stays_print():
    ink, norm = ink_of(text_sheet(0.0, marked=False)[0])
    assert not freehand.find(ink, norm, PPM, 2.5 * PPM, strokes=True, leaders=True).any()


def mm(v: float) -> int:
    return round(v * PPM)


@pytest.mark.parametrize("draw", [
    lambda g: cv2.ellipse(g, (mm(40), mm(90)), (mm(60), mm(60)), 0, 255, 285, 0, 3),  # a shallow arc, 31 mm
    lambda g: cv2.rectangle(g, (mm(20), mm(20)), (mm(40), mm(22.4)), 0, -1),  # a filled wall, 20 x 2.4 mm
    lambda g: cv2.rectangle(g, (mm(20), mm(40)), (mm(50), mm(43.65)), 0, -1),  # a filled bar
    lambda g: [cv2.rectangle(g, (mm(10 + 10 * k), mm(70)), (mm(20 + 10 * k), mm(72)), 0, -1 if k % 2 else 2)
               for k in range(5)],  # a scale bar
    lambda g: (cv2.line(g, (mm(50), mm(50)), (mm(70), mm(50)), 0, 3),  # a text centred on a dimension line
               cv2.putText(g, "1250", (mm(54), mm(51.7)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)),
])
def test_print_on_a_sheet_with_marks_stays_print(draw):
    img = np.full((mm(100), mm(150)), 255, np.uint8)
    draw(img)
    cloud(img, 95, 60, 40, 25)
    ink, norm = ink_of(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    hand = freehand.find(ink, norm, PPM, 2.5 * PPM, strokes=True, leaders=True)
    assert hand.any() and not hand[:, :mm(90)].any()  # the cloud, nothing of the print


def test_the_benchmark_measures_each_part_of_the_markups_and_what_is_no_markup():
    from dataclasses import replace

    import run
    from evaluate import markups

    from app.plan2cad.model import Polyline

    gt = mark_up(Drawing(420, 297), 1, "black")
    notes = [replace(pl, layer="P2C-MARKUP") for pl in gt.polylines if pl.layer == "P2C-MARKUP-NOTE"]
    stray = Polyline([(10.0, 280.0), (50.0, 280.0)], [0.0], False, 0.5, False, "markup", None, "P2C-MARKUP")
    m = markups(gt, Drawing(420, 297, polylines=notes + [stray]))
    assert m["markup_found_note"] > 0.95 and m["markup_found_cloud"] < 0.05 and m["markup_found_leader"] < 0.05
    assert 35 <= m["markup_false_mm"] <= 45  # the stray line, 40 mm
    assert markups(Drawing(420, 297), Drawing(420, 297, polylines=[stray])) == {"markup_false_mm": m["markup_false_mm"]}
    case = dict.fromkeys(run.KEYS) | {"name": "Case", "kind": "floor plan", "markup_found_note": 1.0,
                                       "markup_false_mm": 40.0}
    assert "| Case | 100 % | – | – | – | – | 40 mm |" in run.report({"cases": [case], "sweep": [], "real": []})


def test_a_stroke_from_a_mark_that_ends_on_the_drawing_without_an_arrowhead_is_no_leader():
    img, pen = leader_sheet(barbs=False)
    cv2.line(img, (mm(100), mm(70)), (mm(130), mm(70)), (0, 0, 0), 4)  # it ends on a printed line
    ink, norm = ink_of(img)
    hand = freehand.find(ink, norm, PPM, 2.5 * PPM, leaders=True)
    assert hand[pen > 0].mean() < 0.1 and not hand[mm(69):mm(71), mm(112):mm(130)].any()


@pytest.mark.parametrize("on, off, passed", [
    ((), (), (False, False, True)), (("freehand_strokes", "leaders"), ("pencil_guard",), (True, True, False))])
def test_the_steps_reach_the_search_for_marks(monkeypatch, on, off, passed):
    seen = []

    def found(*args):
        seen.append(args[4:])
        raise StopIteration  # what follows is not needed

    monkeypatch.setattr(freehand, "find", found)
    with pytest.raises(StopIteration):
        convert(plot(Drawing(150, 100), 300), paper="A4", dpi=300, features=Features.of(on, off))
    assert seen == [passed]


def test_thin_print_is_as_grey_as_print_of_its_width_not_pencil():
    ink, gray = np.zeros((400, 400), np.uint8), np.full((400, 400), 255, np.uint8)
    for y in range(20, 380, 40):  # thick lines, black, and thin ones, grey as a scan shows them
        ink[y:y + 7, 20:380], gray[y:y + 7, 20:380] = 1, 30
        ink[y + 20, 20:380], gray[y + 20, 20:380] = 1, 140
    levels = freehand._print_levels(gray, ink, 20.0)  # the sheet's darkest print: 20
    assert (levels[2], levels[8], levels[4]) == (140, 30, 20)  # no print 4 pixels wide: the darkest
    thin, pencil = np.full(50, 140), np.full(50, 150)
    assert freehand._light(thin, 2, 20.0, None) and not freehand._light(thin, 2, 20.0, levels)
    assert freehand._light(pencil, 8, 20.0, levels)  # pencil of a usual width against black print of that width
    assert freehand._light(pencil, 4, 20.0, levels)


def test_specks_of_print_a_mark_leaves_are_no_notes():
    mask, gray = np.zeros((200, 400), np.uint8), np.full((200, 400), 255, np.uint8)
    cv2.circle(mask, (100, 100), 60, 1, 6)  # a printed circle, black, with light specks beside it
    gray[mask > 0] = 30
    for x in (90, 110):
        mask[95:99, x:x + 4], gray[95:99, x:x + 4] = 1, 200
    cv2.line(mask, (200, 40), (390, 160), 1, 5)  # and a long pencil stroke from it
    gray[(mask > 0) & (gray == 255)] = 200
    size, reach = 6.0 * PPM, 35
    loose, guarded = freehand._pencil(mask, gray, 30.0), freehand._pencil(mask, gray, 30.0, size, reach)
    assert (loose[95:99, 90:114] == 1).any() and not (guarded[95:99, 90:114] == 1).any()  # a mark only unguarded
    assert (guarded[95:99, 90:114] == freehand.SPECK).any()  # still out of the drawing, as before
    assert guarded[:, 200:].sum() == loose[:, 200:].sum() > 0  # the stroke stays either way


def test_a_thin_printed_line_as_grey_as_print_of_its_width_is_no_pencil():
    mask, gray = np.zeros((200, 400), np.uint8), np.full((200, 400), 255, np.uint8)
    cv2.circle(mask, (60, 100), 30, 1, 3)  # a printed circle, black, and a thin printed line from it, grey
    gray[mask > 0] = 30
    mask[100, 92:390], gray[100, 92:390] = 1, 150
    levels = np.full(freehand.WIDTHS + 1, 30.0)
    size, reach = 6.0 * PPM, 35
    assert freehand._pencil(mask, gray, 30.0, size, reach, levels)[100, 200:390].all()  # no thin print on the sheet
    levels[2] = 150  # thin print on the sheet is that grey
    assert not freehand._pencil(mask, gray, 30.0, size, reach, levels).any()
    thin = mask * 0  # the thin line alone, all of it light
    thin[100, 20:390] = 1
    grey = np.where(thin > 0, 150, 255).astype(np.uint8)
    assert not freehand._pencil(thin, grey, 30.0, size, reach, levels).any()
    assert (freehand._pencil(thin, grey, 30.0) == thin).all()  # without the guard: pencil, all of it

