import cv2
import numpy as np
from evaluate import evaluate
from synth import photo

from app.plan2cad.pipeline import convert
from app.plan2cad.render import render


def test_scan_recovers_geometry_text_and_scale(plan, plan_scan):
    r = convert(plan_scan, paper="A3")
    m = evaluate(plan, r.drawing)
    assert r.info["sheet"] == "paper"
    assert m["recall_1mm"] > 0.97 and m["precision_1mm"] > 0.97
    assert m["text_exact"] > 0.9
    assert r.drawing.scale == 50


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
