"""Marks by hand in black or pencil on a print, told apart from the drawing by their shape and their grey."""
import cv2
import numpy as np
import pytest
from evaluate import evaluate
from synth import building_plan, mark_up

from app.plan2cad import freehand
from app.plan2cad.features import Features
from app.plan2cad.model import Drawing
from app.plan2cad.pipeline import binarize, convert, normalize
from app.plan2cad.simulate import plot

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


def test_a_drawing_without_marks_keeps_all_its_lines():
    ink, norm = ink_of(plot(building_plan(2, fixtures=True), 300))
    assert not freehand.find(ink, norm, PPM, 2.5 * PPM).any()


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
