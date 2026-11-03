"""Strokes drawn by hand on a print, in black or grey ink: notes, revision clouds, freehand hatching, leaders. Colour
markups are told apart by their colour (markup.py); these by their shape and ink, with features that hold on any
kind of drawing:

- CAD draws exact straight lines, circular and elliptic arcs and smooth curves, joined at sharp corners. A hand bends
  all along and its bend changes: between its corners a stroke is none of these (HAND share of its length), or it
  turns so often that the pieces between its turns are too short to be anything (handwriting: SHORT share), and it
  has few straight pieces (STRAIGHT share at most).
- A revision cloud is a chain of arcs of about one size, all bulging outwards, meeting in points (CLOUD bumps).
- Pencil is lighter than print: the darkest part of its stroke stays grey (LIGHT); a light stroke needs less of the
  rest.

Only strokes larger than a few characters are judged (SIZE), and not those in a row of characters of their height:
printed characters bend all along too and are left to the text recognition. Neither are strokes that run across much
of the sheet (FAR): they are the drawing's own network, and a mark drawn into it stays in it. The marks found are kept
apart from the drawing on their own layer, as they are, not deleted."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .model import Color, Markup, Polyline, Text

SIZE = 3.0  # characters: a stroke at least this many characters long or wide is judged (a short word)
FAR = 0.5  # share of the sheet's longer side: a stroke larger than this is the drawing's network, not judged
STEP = 0.25  # mm between the samples along a stroke
CORNER = math.radians(12)  # a turn of at least this within CORNER_SPAN, three times as sharp as on either side of
CORNER_SPAN = 0.6  # it (mm), is a corner, where a CAD stroke changes direction; a hand bends more evenly
FIT = 0.1  # mm: a piece between corners within this of a straight line or an arc was drawn by CAD (and within a
# quarter of its stroke's width or 1.5 pixels, which the edges of a scanned stroke wander)
PIECE = 0.8  # mm: shorter pieces between corners are not judged (a joint, the tip of an arrow)
SMOOTH = 15.0  # mm: a piece that bends nowhere tighter than this radius is a smooth curve of CAD (a spline, a contour)
HAND = 0.45  # a stroke with at least this share of its length bent by hand is a mark
SHORT, STRAIGHT = 0.35, 0.4  # or with this share bent by hand or in short pieces, and at most that share straight,
BENT = 0.1  # and this share bent by hand or in arcs (round letters; not a symbol of lines and ellipses)
WIDE, NETWORK = 60.0, 0.25  # mm: a stroke wider than this with more than that share straight is linework, not a note
RUN = 3.0  # mm: straight runs at least this long in a piece that is bent between them: a CAD shape with round corners
ROW = 2  # neighbours of about its height beside it in a row: a large character of a text line, left to the text
CLOUD = 6  # arcs of a size in a row, bulging one way and meeting in points: a revision cloud (a coil has fewer)
LIGHT = 0.35  # a stroke whose darkest tenth is lighter than this share of the way from paper to print is pencil
KINDS = ("line", "arc", "curve", "hand", "short")


def find(ink: np.ndarray, gray: np.ndarray, ppm: float, letter: float) -> np.ndarray:
    """The mask of the strokes drawn by hand among ``ink``, at ``ppm`` pixels per mm with characters ``letter``
    pixels high; ``gray`` the normalised image (paper 255)."""
    out = np.zeros_like(ink)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    side = np.maximum(stats[1:, 2], stats[1:, 3])
    big = np.flatnonzero((side >= SIZE * letter) & (side <= FAR * max(ink.shape))) + 1
    if not len(big):
        return out
    darkest = _print_level(gray, ink)
    for i in big:
        x, y, w, h = stats[i, :4]
        mask = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        if not _in_row(stats, i) and _drawn_by_hand(mask, gray[y:y + h, x:x + w], ppm, darkest):
            out[y:y + h, x:x + w] |= _pencil(mask, gray[y:y + h, x:x + w], darkest)
    return out


def _pencil(mask: np.ndarray, gray: np.ndarray, darkest: float) -> np.ndarray:
    """Of a mark, what is not print: where a pencil line runs through printed text or lines (a text struck out), the
    print stays (the dark ink and its edge) and only the light strokes go. A mark in black is all of it."""
    light = (gray > darkest + LIGHT * (255 - darkest)) & (mask > 0)
    dark = (mask > 0) & ~light
    if dark.sum() < 0.2 * mask.sum() or light.sum() < 0.2 * mask.sum():
        return mask
    return (light & (cv2.dilate(dark.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)).astype(np.uint8)


def _in_row(stats: np.ndarray, i: int) -> bool:
    """Whether component i stands in a row with ROW components of about its height, as a character of a text line:
    its height within a third, beside it within its height, on about the same line."""
    x, y, w, h = stats[i, :4]
    other = stats[1:]
    alike = (np.abs(other[:, 3] - h) <= h / 3) & (np.abs(other[:, 1] + other[:, 3] - y - h) <= h / 4)
    gap = np.maximum(other[:, 0] - (x + w), x - (other[:, 0] + other[:, 2]))
    return int(np.sum(alike & (gap >= 0) & (gap <= h))) >= ROW


def marks(hand: np.ndarray, gray: np.ndarray, ppm: float, texts: list[Text]) -> tuple[list[Polyline], list[Markup]]:
    """The marks as polylines in mm on the markup layer, simplified to a tenth of a millimetre, each in the grey of its
    ink (black where it is about as dark as the print); and a note per group of them, with the printed ``texts`` it
    marks. Their handwriting is not read: what Tesseract reads in it is mostly made up."""
    from .markup import LAYER, notes
    from .vectorize import trace

    out = []
    darkest = _print_level(gray, (gray < 128).astype(np.uint8))
    for pts, width in trace(hand, ppm):
        if np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() < 0.6 * ppm:
            continue
        closed = len(pts) > 3 and np.allclose(pts[0], pts[-1])
        simple = cv2.approxPolyDP(pts.astype(np.float32), max(1.0, 0.1 * ppm), closed)[:, 0] / ppm
        if len(simple) < 2:
            continue
        points = [tuple(p) for p in simple.tolist()]
        out.append(Polyline(points, [0.0] * (len(points) - (0 if closed else 1)), closed, width / ppm, False,
                            "markup", _grey(gray, pts, darkest), LAYER))
    core = _grey(gray, np.column_stack(np.nonzero(hand)[::-1]), darkest)
    return out, notes(hand, ppm, core or (0, 0, 0), texts)


def _grey(gray: np.ndarray, pts: np.ndarray, darkest: float) -> Color | None:
    """The grey of the darkest tenth of the ink at ``pts`` (x, y pixels); None where it is about as dark as print."""
    ix = np.clip(np.round(pts).astype(int), 0, [gray.shape[1] - 1, gray.shape[0] - 1])
    core = float(np.percentile(gray[ix[:, 1], ix[:, 0]], 10))
    return None if core < darkest + 60 else (round(core),) * 3


def _print_level(gray: np.ndarray, ink: np.ndarray) -> float:
    """How dark the print is: the darkest tenth of the ink of the sheet."""
    values = gray[ink > 0]
    return float(np.percentile(values, 10)) if len(values) else 0.0


def _drawn_by_hand(mask: np.ndarray, gray: np.ndarray, ppm: float, darkest: float) -> bool:
    from .vectorize import trace

    width = 2 * float(np.median(cv2.distanceTransform(mask, cv2.DIST_L2, 3)[mask > 0]))
    tol = max(FIT, width / ppm / 4, 1.5 / ppm)
    kinds, arcs = dict.fromkeys(KINDS, 0.0), []
    for p, _ in trace(mask, ppm):
        if len(p) >= 2:
            arcs += _pieces(p, ppm, tol, kinds)
    total = sum(kinds.values())
    if total < 2 * PIECE:
        return False
    hand, loose, straight = kinds["hand"] / total, (kinds["hand"] + kinds["short"]) / total, kinds["line"] / total
    if hand >= HAND:
        return True
    if max(mask.shape) > WIDE * ppm and straight > NETWORK:  # a mark drawn into linework: a part of it, not all
        return False
    if loose >= SHORT and straight <= STRAIGHT and (kinds["hand"] + kinds["arc"]) / total >= BENT or _cloud(arcs):
        return True
    core = float(np.percentile(gray[mask > 0], 10))
    return core > darkest + LIGHT * (255 - darkest) and loose >= SHORT / 2  # pencil


def _pieces(path: np.ndarray, ppm: float, tol: float, kinds: dict[str, float]) -> list[tuple[float, int]]:
    """Adds the lengths of a traced stroke's pieces between corners to ``kinds`` by what they are (_fit; "short" for
    the corners and the pieces under PIECE mm) and returns its arcs in order: (radius in mm, the side they bulge to)."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)]) / ppm
    t = np.arange(0, s[-1], STEP)
    k = max(1, round(CORNER_SPAN / 2 / STEP))
    if len(t) < 4 * k + 2:
        kinds["short"] += float(s[-1])
        return []
    pts = np.column_stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])]) / ppm
    d = pts[2 * k:] - pts[:-2 * k]
    heading = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs((heading[k:] - heading[:-k] + np.pi) % (2 * np.pi) - np.pi)
    m = max(k + 1, round(1.0 / STEP))  # a millimetre on, the edges of a CAD corner run straight again
    pad = np.pad(turn, m)
    around = np.maximum(pad[:len(turn)], pad[2 * m:])
    corner = np.zeros(len(pts), bool)
    corner[2 * k:2 * k + len(turn)] = (turn > CORNER) & (turn > 3 * around)
    corner = np.convolve(corner, np.ones(2 * k + 1), "same") > 0  # with the samples the turn spans
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (~corner).astype(np.int8), [0]])) != 0)
    arcs, judged = [], 0.0
    for a, b in zip(edges[::2], edges[1::2]):  # runs between corners
        piece = pts[a:b]
        length = STEP * (len(piece) - 1)
        if length < PIECE:
            continue
        judged += length
        kind, radius, side = _fit(piece, tol)
        if kind == "hand" and (straight := _rounded(piece, tol)):  # lines with round corners: lines and arcs
            kinds["line"] += straight
            kinds["curve"] += length - straight
            continue
        kinds[kind] += length
        if kind == "arc":
            arcs.append((radius, side))
    kinds["short"] += max(0.0, float(s[-1]) - judged)
    return arcs


def _fit(pts: np.ndarray, tol: float = FIT) -> tuple[str, float, int]:
    """What a piece of a stroke between corners is: "line", "arc" (with its radius and the side it bulges to),
    "curve" (an elliptic arc, or bending nowhere tighter than SMOOTH) or "hand" (none of these, within ``tol`` mm)."""
    c = pts.mean(axis=0)
    _, sv, _ = np.linalg.svd(pts - c, full_matrices=False)
    if sv[1] / math.sqrt(len(pts)) <= tol / 2:
        return "line", 0.0, 0
    A = np.column_stack([2 * pts, np.ones(len(pts))])
    b = (pts ** 2).sum(axis=1)
    (cx, cy, k), *_ = np.linalg.lstsq(A, b, rcond=None)
    r = math.sqrt(max(k + cx * cx + cy * cy, 0.0))
    off = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
    if r > 0 and np.sqrt(np.mean(off ** 2)) <= tol / 2 and off.max() <= tol * 1.5:
        mid = pts[len(pts) // 2] - (pts[0] + pts[-1]) / 2
        chord = pts[-1] - pts[0]
        return "arc", r, 1 if chord[0] * mid[1] - chord[1] * mid[0] > 0 else -1
    if _elliptic(pts, tol):
        return "curve", 0.0, 0
    span = max(1, round(2.0 / STEP))  # the bend over 2 mm at a time: over less, the pixels of a thin line wander more
    if len(pts) > 2 * span:
        d = pts[span:] - pts[:-span]
        heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        if np.abs(np.diff(heading[::span])).max(initial=0) <= 2.0 / SMOOTH:
            return "curve", 0.0, 0
    return "hand", 0.0, 0


def _rounded(pts: np.ndarray, tol: float) -> float:
    """The length of the straight runs (at least RUN mm) of a piece if what lies between them is a circular or an
    elliptic arc each, as with the round corners of a bath or a slot; else 0."""
    n = max(4, round(RUN / STEP))
    if len(pts) < 2 * n:
        return 0.0
    straight = np.zeros(len(pts), bool)
    for i in range(0, len(pts) - n + 1, max(1, n // 4)):
        w = pts[i:i + n]
        if np.linalg.svd(w - w.mean(axis=0), compute_uv=False)[1] / math.sqrt(n) <= tol / 2:
            straight[i:i + n] = True
    if straight.mean() < 0.5:
        return 0.0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (~straight).astype(np.int8), [0]])) != 0)
    for a, b in zip(edges[::2], edges[1::2]):  # the bends between the straight runs
        bend = pts[max(0, a - 1):b + 1]
        if len(bend) >= 4 and _fit(bend, tol)[0] == "hand":
            return 0.0
    return float(straight.sum() * STEP)


def _elliptic(pts: np.ndarray, tol: float) -> bool:
    """Whether a piece lies on an ellipse within ``tol`` mm (a wash basin, a WC bowl, a vessel head), measured along
    the ray from the ellipse's centre: near enough to the true distance for a piece that lies on it."""
    if len(pts) < 8:
        return False
    try:
        (cx, cy), (w, h), angle = cv2.fitEllipseDirect(pts.astype(np.float32))
    except cv2.error:
        return False
    a, b = w / 2, h / 2
    if not (0.05 < a < 500 and 0.05 < b < 500):
        return False
    t = math.radians(angle)
    x = (pts[:, 0] - cx) * math.cos(t) + (pts[:, 1] - cy) * math.sin(t)
    y = (pts[:, 1] - cy) * math.cos(t) - (pts[:, 0] - cx) * math.sin(t)
    rho = np.maximum(np.hypot(x, y), 1e-9)
    on = 1 / np.sqrt((x / rho / a) ** 2 + (y / rho / b) ** 2)  # the ellipse's radius in the direction of each point
    off = np.abs(rho - on)
    return bool(np.sqrt(np.mean(off ** 2)) <= tol / 2 and off.max() <= tol * 1.5)


def _cloud(arcs: list[tuple[float, int]]) -> bool:
    """Whether a stroke's arcs are the bumps of a revision cloud: CLOUD or more in a row of about one size (within a
    factor 2), bulging to the same side."""
    run = 1
    for (r0, s0), (r1, s1) in zip(arcs, arcs[1:]):
        run = run + 1 if s0 == s1 and 0.5 <= r1 / max(r0, 1e-6) <= 2 and 0.8 <= r1 <= 8 else 1
        if run >= CLOUD:
            return True
    return False
