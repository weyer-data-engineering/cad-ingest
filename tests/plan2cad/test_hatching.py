import math

import cv2
import numpy as np
import pytest

from app.plan2cad import hatching, pipeline, vectorize
from app.plan2cad.features import Features
from app.plan2cad.model import Line, Text
from app.plan2cad.vectorize import merge_lines


def strokes(n: int, lean: float, length: int = 40, pitch: int = 14, width: int = 2) -> np.ndarray:
    """A row of n straight strokes, leaning ``lean`` degrees from upright (to the right), as a boolean mask."""
    img = np.zeros((length + 20, n * pitch + length + 20), np.uint8)
    dx, dy = length * math.sin(math.radians(lean)), length * math.cos(math.radians(lean))
    for k in range(n):
        x = 10 + k * pitch
        cv2.line(img, (x, 10 + round(dy)), (x + round(dx), 10), 1, width)
    return img > 0


@pytest.mark.parametrize("lean", [45, -45, 35])
def test_a_row_of_slanted_strokes_is_hatching(lean):
    assert hatching.is_hatching(strokes(12, lean), 0.0)
    assert hatching.is_hatching(strokes(12, lean).T, 90.0)  # a vertical line of them, read bottom to top


def test_lettering_upright_italic_or_dashed_is_no_hatching():
    word = np.zeros((60, 400), np.uint8)
    cv2.putText(word, "Wohnen 12,40", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 1, 3)
    assert not hatching.is_hatching(word > 0, 0.0)
    assert not hatching.is_hatching(strokes(12, 0), 0.0)  # "IIIIIIIIIIII"
    assert not hatching.is_hatching(strokes(12, 12), 0.0)  # in italics
    assert not hatching.is_hatching(strokes(12, 45), 45.0)  # strokes along the line: a row of dashes


def test_a_few_strokes_are_no_hatching():
    assert not hatching.is_hatching(strokes(4, 45), 0.0)


def test_hatching_read_as_text_goes_back_to_the_linework():
    rows = strokes(10, 45)
    labels = np.zeros((rows.shape[0] + 60, rows.shape[1]), np.uint16)
    labels[:rows.shape[0]][rows] = 1
    labels[-40:-20, 10:60] = 2  # a text of other pixels
    texts = [Text(10, 3, 3.0, "IIIIIIIIII", uncertain=True), Text(3, 10, 2.5, "Schnitt")]
    kept, out = hatching.dropped(texts, labels)
    assert [t.text for t in kept] == ["Schnitt"]
    assert out.dtype == labels.dtype and not out[:rows.shape[0]].any() and (out[-40:-20, 10:60] == 1).all()
    assert hatching.dropped(kept, out)[0] == kept  # nothing to drop: as it was


@pytest.mark.parametrize("on", [True, False])
def test_the_steps_can_be_switched_off(monkeypatch, on):
    calls, frames = [], []

    def spy(texts, labels):
        calls.append(len(texts))
        return texts, labels

    def merge(lines, **options):
        frames.append(options["one_frame"])
        return merge_lines(lines, **options)

    monkeypatch.setattr(hatching, "dropped", spy)
    monkeypatch.setattr(vectorize, "merge_lines", merge)
    ink = np.zeros((240, 360), np.uint8)
    cv2.rectangle(ink, (20, 100), (340, 140), 1, 3)  # a slab, 32 mm long and 4 mm high at 10 px/mm
    for x in range(30, 330, 15):
        cv2.line(ink, (x, 138), (x + 36, 102), 1, 2)
    norm = np.where(ink > 0, np.uint8(0), np.uint8(255))
    features = Features.of(off=[] if on else ["hatching", "merge_where_drawn"])
    pipeline._vectorize(ink, norm, 10.0, pipeline._Timer(), False, 1.0, features)
    assert bool(calls) == on and frames == [on]


def test_nearly_parallel_strokes_far_out_on_the_sheet_keep_their_places():
    def u(th):
        return np.array([math.cos(th), math.sin(th)])

    def offset(th, m):
        return -m[0] * math.sin(th) + m[1] * math.cos(th)

    # two strokes of a dense hatching, 2 mm apart and 0.8 degrees apart in angle, far from the corner of the sheet:
    # measured each at its own angle, their offsets are only 0.3 mm apart
    tp, tq, mp = math.radians(137.4), math.radians(136.6), np.array([264.0, 87.0])
    mq = mp + (offset(tp, mp) + 0.3 - offset(tq, mp)) / math.cos(tp - tq) * np.array([-math.sin(tp), math.cos(tp)])
    p, q = (Line(*(m - 2.1 * u(th)), *(m + 2.1 * u(th))) for m, th in ((mp, tp), (mq, tq)))

    def off(a: Line, b: Line) -> float:  # how far the ends of a lie from the line through b
        n = np.array([b.y1 - b.y2, b.x2 - b.x1]) / math.hypot(b.x2 - b.x1, b.y2 - b.y1)
        return max(abs(np.dot(np.array(e) - (b.x1, b.y1), n)) for e in ((a.x1, a.y1), (a.x2, a.y2)))

    out = merge_lines([p, q])
    assert len(out) == 2 and all(min(off(o, p), off(o, q)) < 0.06 for o in out)
    assert len(merge_lines([p, q], one_frame=False)) == 1  # as before: taken for one stroke, drawn between them


def _ends_off(a: Line, b: Line) -> float:
    """How far the ends of a lie from those of b, the nearer way round."""
    p, q, r, s = (a.x1, a.y1), (a.x2, a.y2), (b.x1, b.y1), (b.x2, b.y2)
    return min(max(math.dist(p, r), math.dist(q, s)), max(math.dist(p, s), math.dist(q, r)))


def _angle(a: Line) -> float:
    return math.degrees(math.atan2(a.y2 - a.y1, a.x2 - a.x1)) % 180


# a side of a roof far out on the sheet: a long line and, beyond a break of 12 mm, a short piece of it whose angle is
# 0.4 degrees off, as a short piece's is; the two fall into one group of merge_lines, the short piece first
ROOF, PIECE = Line(100.08, 84.25, 183.91, 32.51), Line(82.05, 95.43, 89.33, 90.86)


def test_a_long_line_keeps_its_angle_beside_a_short_piece_of_its_direction():
    out = merge_lines([ROOF, PIECE])
    # drawn at the short piece's angle, the long line was turned about its middle: its ends 0.4 mm off the ridge
    assert len(out) == 2
    assert min(_ends_off(o, ROOF) for o in out) < 0.01 and min(_ends_off(o, PIECE) for o in out) < 0.01


def test_pieces_of_one_line_are_drawn_at_their_angle_through_them():
    # a line drawn in two overlapping pieces 0.3 degrees apart, the short one first in its group: the line was drawn at
    # the short one's angle, its far end 0.13 mm off
    long_, short = Line(10.0, 10.0, 40.0, 30.0), Line(36.0, 27.25, 46.0, 34.0)
    out = merge_lines([long_, short])
    assert len(out) == 1
    weighted = (36.06 * _angle(long_) + 12.06 * _angle(short)) / (36.06 + 12.06)  # the angle of both, by length
    assert abs(_angle(out[0]) - weighted) < 0.01 and _ends_off(out[0], Line(10.0, 10.0, 46.0, 34.0)) < 0.05


def test_a_slanted_dashed_line_is_drawn_on_its_dashes():
    th = math.radians(30.0)
    dashes = []
    for k in range(12):  # 3 mm dashes, 1.5 mm gaps, each turned a little the one way or the other
        a = 20.0 + 4.5 * k
        tilt = math.radians(0.3 if k % 2 else -0.3)
        x, y = a * math.cos(th), a * math.sin(th)
        dashes.append(Line(x, y, x + 3 * math.cos(th + tilt), y + 3 * math.sin(th + tilt)))
    out = merge_lines(dashes)
    assert len(out) == 1 and out[0].dashed
    assert _ends_off(out[0], Line(dashes[0].x1, dashes[0].y1, dashes[-1].x2, dashes[-1].y2)) < 0.05


def test_without_the_step_the_long_line_turns_to_the_short_piece_as_before():
    out = merge_lines([ROOF, PIECE], one_frame=False)
    roof = max(out, key=lambda o: math.hypot(o.x2 - o.x1, o.y2 - o.y1))
    assert len(out) == 2 and abs(_angle(roof) - _angle(PIECE)) < 1e-9 and _ends_off(roof, ROOF) > 1.0


def test_a_single_piece_comes_out_as_drawn_and_level_ones_level():
    slanted, level = Line(310.0, 40.0, 318.0, 47.0), Line(10.0, 20.0, 30.0, 20.3)
    assert _ends_off(merge_lines([slanted])[0], slanted) < 1e-9
    out = merge_lines([level])[0]
    assert out.y1 == out.y2 and abs(out.y1 - 20.15) < 1e-9  # within AXIS_SNAP: level, as before


@pytest.mark.parametrize("length", [5.0, 7.5, 18.0, 20.5, 31.0])
def test_pieces_of_an_upright_line_merge_into_an_exactly_upright_line(length):
    # close_corners snaps the ends of exactly horizontal and vertical lines only, and a length-weighted mean of right
    # angles is not always exactly one (it was not for these lengths)
    out = merge_lines([Line(85.0, 40.0, 85.0, 40.0 + length), Line(85.1, 39.5 + length, 85.1, 60.0 + length)])
    assert len(out) == 1 and out[0].x1 == out[0].x2
