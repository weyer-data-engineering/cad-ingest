"""Hatching that the text recognition took for text. A narrow hatched area, such as a cut slab of a section, is
hatched with short slanted strokes as high as letters; once the outline of the area is taken out as lines, they stand
in a row like characters and read as "IIII" or "////", a text to check where the drawing has lines. No lettering is a
row of parallel straight strokes leaning far from upright: such a "text" is hatching, and its strokes go back to the
linework, so that they are traced and joined to the outline they end on."""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage

from .model import Text

STROKES = 5  # a row of hatching has at least this many strokes
SHARE = 0.8  # and they are at least this share of the pieces of the text line
SLIM = 4.0  # a straight stroke is at least this many times as long as it is wide
PARALLEL = math.radians(6)  # the strokes of one hatching differ in direction by at most this
SLANT = math.radians(25)  # hatching leans at least this far from upright against its line; italics lean less
FLAT = math.radians(20)  # and lies at least this far from along it, as a row of dashes does


def dropped(texts: list[Text], labels: np.ndarray) -> tuple[list[Text], np.ndarray]:
    """The texts without those that are hatching, and the label image of their pixels (0: no text, k: the k-th text)
    without the pixels of those, so that they are traced as lines."""
    boxes = ndimage.find_objects(labels, len(texts)) if texts else []
    hatched = {k for k, box in enumerate(boxes, 1) if box is not None and is_hatching(labels[box] == k,
                                                                                         texts[k - 1].angle)}
    if not hatched:
        return texts, labels
    kept = [k for k in range(1, len(texts) + 1) if k not in hatched]
    lut = np.zeros(len(texts) + 1, labels.dtype)
    lut[kept] = np.arange(1, len(kept) + 1)
    return [texts[k - 1] for k in kept], lut[labels]


def is_hatching(mask: np.ndarray, angle: float) -> bool:
    """Whether the pixels of a text line are hatching: at least STROKES straight strokes of one direction, SHARE of
    its pieces, leaning at least SLANT from upright against the line's reading direction ``angle`` (degrees,
    counter-clockwise as Text.angle) and at least FLAT from along it."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n - 1 < STROKES:
        return False
    direction, slim = _strokes(lab, n)
    straight = direction[slim]
    if len(straight) < max(STROKES, SHARE * (n - 1)):
        return False
    # the mean direction of the strokes (mod pi) through their doubled angles; then the share within PARALLEL of it
    mean = 0.5 * math.atan2(np.sin(2 * straight).sum(), np.cos(2 * straight).sum())
    off = np.abs((straight - mean + np.pi / 2) % np.pi - np.pi / 2)
    if np.sum(off <= PARALLEL) < max(STROKES, SHARE * (n - 1)):
        return False
    lean = abs((-mean - math.radians(angle)) % np.pi - np.pi / 2)  # from upright; image y points down, angle up
    return lean >= SLANT and np.pi / 2 - lean >= FLAT


def _strokes(lab: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The direction (radians, image axes, mod pi) of each component of a label image and whether it is a straight
    stroke (SLIM), from the second moments of its pixels."""
    ys, xs = np.nonzero(lab)
    k = lab[ys, xs]

    def total(w=None) -> np.ndarray:
        return np.bincount(k, weights=w, minlength=n)[1:]

    count = np.maximum(total(), 1)
    mx, my = total(xs) / count, total(ys) / count
    cxx, cyy, cxy = total(xs * xs) / count - mx ** 2, total(ys * ys) / count - my ** 2, total(xs * ys) / count - mx * my
    cxx, cyy = cxx + 1 / 12, cyy + 1 / 12  # a pixel is a unit square, not a point: a one pixel stroke has a width
    half = np.sqrt(((cxx - cyy) / 2) ** 2 + cxy ** 2)
    major, minor = (cxx + cyy) / 2 + half, (cxx + cyy) / 2 - half
    return 0.5 * np.arctan2(2 * cxy, cxx - cyy) % np.pi, major >= SLIM ** 2 * np.maximum(minor, 1e-9)
