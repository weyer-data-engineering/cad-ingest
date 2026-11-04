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
    assert "## Markups (added by hand)" in markdown and "Highlighted in green; covers: Bad" in markdown


def test_a_colour_plot_has_no_markups(plan):
    img = cv2.cvtColor(render(plan, PPM), cv2.COLOR_GRAY2BGR)
    img[img[..., 0] < 128] = (200, 60, 0)  # every line in blue: the colours are print
    gray, marks, _ = markup.split(img, PPM)
    assert marks is None and gray.max() > 200 and (gray < 128).sum() > 0.5 * (img[..., 0] == 200).sum()


@pytest.fixture(scope="module")
def photographed(plan):
    """The plan with a red note, as a phone camera adds colour to it: a seam of red and one of blue a pixel wide along
    either side of the black print in a corner (the fringes of a lens) and patches of colour noise whose hue wanders
    (white balance in the shadows); and the mask of the note."""
    gray = render(plan, PPM)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    note = np.zeros(img.shape[:2], np.uint8)
    cv2.putText(note, "removed", (int(60 * PPM), int(40 * PPM)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, 1, 3)
    img[note > 0] = RED
    dark, paper = gray < 128, gray > 200
    corner = np.zeros_like(dark)
    corner[dark.shape[0] // 2:, 2 * dark.shape[1] // 3:] = True
    img[corner & paper & np.roll(dark, -1, axis=1)] = RED
    img[corner & paper & np.roll(dark, 1, axis=1)] = (220, 60, 40)
    rng = np.random.default_rng(1)
    for x, y in ((300, 80), (200, 200), (120, 260)):  # mm
        x0, y0, s = int(x * PPM), int(y * PPM), int(6 * PPM)
        hue = rng.integers(0, 180, (s, s)).astype(np.uint8)
        hsv = np.dstack([hue, np.full((s, s), 200, np.uint8), np.full((s, s), 200, np.uint8)])
        img[y0:y0 + s, x0:x0 + s] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img, note


def test_colour_a_camera_adds_is_no_pen(photographed):
    img, note = photographed
    near = cv2.dilate(note, np.ones((7, 7), np.uint8)) > 0
    _, off, _ = markup.split(img.copy(), PPM)
    _, on, _ = markup.split(img.copy(), PPM, guard=True)
    assert ((off >= markup.PEN) & ~near).sum() > 2000  # without the guard: fringes and noise as pen strokes
    assert ((on >= markup.PEN) & ~near).sum() < 0.05 * ((off >= markup.PEN) & ~near).sum()
    assert (on[note > 0] >= markup.PEN).mean() > 0.9  # the note, of one hue, stays


def test_colour_off_the_paper_is_its_shadow_not_a_pen(plan):
    img = cv2.cvtColor(render(plan, PPM), cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    cv2.line(img, (int(0.2 * w), 6), (int(0.8 * w), 6), RED, 4)  # along the edge, outside the sheet's corners
    corners = np.array([[20, 20], [w - 20, 20], [w - 20, h - 20], [20, h - 20]], float)
    assert markup.split(img.copy(), PPM, guard=True)[1] is not None
    assert markup.split(img.copy(), PPM, guard=True, paper=corners)[1] is None


def test_two_pens_that_cross_both_stay_pen_strokes(plan):
    img = cv2.cvtColor(render(plan, PPM), cv2.COLOR_GRAY2BGR)
    red, green = np.zeros(img.shape[:2], np.uint8), np.zeros(img.shape[:2], np.uint8)
    x, y = int(150 * PPM), int(100 * PPM)
    cv2.circle(red, (x, y), int(20 * PPM), 1, 4)
    cv2.line(green, (x - int(30 * PPM), y - int(25 * PPM)), (x + int(30 * PPM), y + int(25 * PPM)), 1, 4)
    img[red > 0], img[green > 0] = RED, (40, 150, 30)
    marks = markup.split(img.copy(), PPM, guard=True)[1]
    assert marks is not None
    for pen in (red & (green == 0), green):
        assert (marks[pen > 0] >= markup.PEN).mean() > 0.9


@pytest.mark.parametrize("dpi", [200, 300])
def test_a_pen_traced_over_a_printed_line_is_no_fringe(plan, dpi):
    ppm = dpi / 25.4
    img = cv2.cvtColor(render(plan, ppm), cv2.COLOR_GRAY2BGR)
    pen = np.zeros(img.shape[:2], np.uint8)
    a, b = (int(40 * ppm), int(285 * ppm)), (int(115 * ppm), int(285 * ppm))
    cv2.line(pen, a, b, 1, round(0.7 * ppm))  # a green pen 0.7 mm wide over a printed line 0.5 mm wide
    img[pen > 0] = (40, 150, 30)
    cv2.line(img, a, b, (0, 0, 0), round(0.5 * ppm))
    seams = (pen > 0) & (img[..., 1] == 150)
    assert (markup.split(img.copy(), ppm)[1][seams] >= markup.PEN).mean() > 0.9
    assert (markup.split(img.copy(), ppm, guard=True)[1][seams] >= markup.PEN).mean() > 0.9
