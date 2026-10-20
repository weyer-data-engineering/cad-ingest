"""Markups by hand in colour on a black and white print: a pen cloud and note, and highlighter."""
import io
import json
import zipfile
from dataclasses import replace

import cv2
import numpy as np
import pytest
from evaluate import evaluate

from app import scan
from app.plan2cad import markup
from app.plan2cad.pipeline import convert
from app.plan2cad.render import render

PPM = 200 / 25.4
RED, GREEN = (40, 40, 220), (90, 235, 110)  # BGR: a red pen and a green highlighter


@pytest.fixture(scope="module")
def marked(plan):
    """The plan scanned at 200 dpi with a red cloud around "Wohnen", a red note below it and "Bad" highlighted."""
    img = cv2.cvtColor(render(plan, PPM), cv2.COLOR_GRAY2BGR)
    room = next(t for t in plan.texts if t.text == "Wohnen")
    x, y = int(room.x * PPM), int(room.y * PPM)
    cv2.ellipse(img, (x, y), (int(16 * PPM), int(7 * PPM)), 0, 0, 360, RED, 3)
    cv2.putText(img, "removed", (x - int(15 * PPM), y + int(17 * PPM)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, RED, 3)
    bath = next(t for t in plan.texts if t.text == "Bad")
    x0, y0, x1, y1 = (int(v * PPM) for v in (bath.x - 6, bath.y - 3, bath.x + 6, bath.y + 3))
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * (np.array(GREEN) / 255)).astype(np.uint8)  # the print shows through
    return img


def test_markups_are_kept_apart_from_the_drawing(plan, marked):
    r = convert(marked, paper="A3", dpi=200)
    d = r.drawing
    drawn = replace(d, polylines=[p for p in d.polylines if p.kind != "markup"])
    m = evaluate(plan, drawn)
    assert m["recall_1mm"] > 0.97 and m["precision_1mm"] > 0.97  # no pen strokes in the drawing's linework
    notes = [k for k in d.markups if k.kind == "note"]
    assert any("Wohnen" in k.texts for k in notes) and all(k.color[0] > 150 for k in notes)  # red, around the room
    assert any(k.kind == "highlight" and "Bad" in k.texts for k in d.markups)
    assert {"Wohnen", "Bad"} <= {t.text for t in d.texts}  # the print under the markups is still read
    strokes = [p for p in d.polylines if p.kind == "markup"]
    assert strokes and all(p.layer == markup.LAYER for p in strokes)
    assert len(r.crops) == len(d.markups)


def test_markups_in_the_result(marked, tmp_path):
    cv2.imwrite(str(tmp_path / "plan.png"), marked)
    markdown, _ = scan.vectorize(str(tmp_path / "plan.png"), "plan.png", {"paper": "A3", "dpi": 200},
                                 str(tmp_path / "plan.zip"), None)
    files = zipfile.ZipFile(io.BytesIO((tmp_path / "plan.zip").read_bytes()))
    report = json.loads(files.read("report.json"))
    assert {m["kind"] for m in report["markups"]} == {"note", "highlight"}
    assert all(m["image"] in files.namelist() for m in report["markups"])
    image = cv2.imdecode(np.frombuffer(files.read(report["markups"][0]["image"]), np.uint8), cv2.IMREAD_COLOR)
    assert np.median(image) > 200  # the markup on the faded print, mostly paper
    assert "## Markups (added by hand in colour)" in markdown and "Highlighted in green; covers: Bad" in markdown


def test_a_colour_plot_has_no_markups(plan):
    img = cv2.cvtColor(render(plan, PPM), cv2.COLOR_GRAY2BGR)
    img[img[..., 0] < 128] = (200, 60, 0)  # every line in blue: the colours are print
    gray, marks, _ = markup.split(img, PPM)
    assert marks is None and gray.max() > 200 and (gray < 128).sum() > 0.5 * (img[..., 0] == 200).sum()
