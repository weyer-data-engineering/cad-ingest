import numpy as np
import pytest

from app.plan2cad.model import Drawing, Text
from app.plan2cad.ocr import find_texts, fix
from app.plan2cad.pipeline import detect_scale
from app.plan2cad.render import render


@pytest.mark.parametrize("raw, fixed", [
    ("17,68 m?", "17,68 m²"), ("11,56 m2", "11,56 m²"), ("15.62 m’", "15.62 m²"), ("DNSO-P-3011", "DN50-P-3011"), ("P-1O1", "P-101"),
    ("Küche", "Küche"), ("B-101", "B-101"),
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


@pytest.mark.parametrize("text, scale", [("Maßstab 1:50", 50), ("M 1:100", 100), ("Matstab 1 50", 50),
                                         ("Blatt 1/1", None), ("PID 1 50", None),
                                         ("12,50 m", None), ("15.62 m’", None)])
def test_detect_scale(text, scale):
    assert detect_scale([Text(0, 0, 2.5, text)]) == scale
