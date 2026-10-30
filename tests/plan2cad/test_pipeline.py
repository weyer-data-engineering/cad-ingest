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
    assert not r.drawing.symbols  # no fixtures drawn, none found


def test_fixtures_on_a_floor_plan_become_library_blocks():
    plan = building_plan(7, fixtures=True)
    r = convert(scan(plan, 7, dpi=200), paper="A3")
    m = evaluate(plan, r.drawing)
    assert r.info["drawing_type"] == "building" and [lib["id"] for lib in r.info["libraries"]] == ["floor-plan"]
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
                                 Text(10, 50, 3.3, "TITLE", length=14)])
    _lettering(d)
    same = [(t.height, t.length) for t in d.texts[:3]]
    assert same[0] == same[1] == same[2] and same[0][0] == pytest.approx(1.2, abs=0.05)
    assert d.texts[3].length == 2.5 and (d.texts[4].height, d.texts[4].length) == (3.3, 14)
