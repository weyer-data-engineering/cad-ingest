"""Finds text lines among the small connected components and reads them with Tesseract.

Characters are grouped into lines geometrically; all line crops are scaled to the same cap height
and read in a few batch images ("mosaics"), which is much faster than one Tesseract call per line.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import cv2
import numpy as np
import pytesseract

from ..config import Settings
from .model import Text

os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # parallelism comes from the batches, not from Tesseract

LANG = Settings.from_env().ocr_languages
CAP_PX = 32  # cap height in pixels that Tesseract reads well
MOSAIC_H = 6000


class Components:
    """Connected components of the remaining ink with their boxes; ``gray`` is the normalised
    image the text crops are cut from (Tesseract binarises blurred glyphs better than we do)."""

    def __init__(self, ink: np.ndarray, gray: np.ndarray):
        self.gray = gray
        n, self.lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        self.x, self.y, self.w, self.h, self.area = (stats[1:, i].astype(np.int64) for i in range(5))
        self.n = n - 1


def find_texts(ink: np.ndarray, gray: np.ndarray, ppm: float, rotation_check: bool = False
               ) -> tuple[list[Text], np.ndarray, int | None]:
    """Returns recognised text lines (sheet mm), a label image of their pixels (0: no text, k: the
    k-th text line) and, if requested, the clockwise rotation in degrees (0, 90, 180, 270) that
    makes the text upright (None if the text is not readable enough to tell)."""
    cc = Components(ink, gray)
    cand = np.flatnonzero((np.maximum(cc.w, cc.h) <= 7.5 * ppm) & (cc.area >= 3))
    h_groups, singles = _group(cc, cand, horizontal=True)
    v_groups, _ = _group(cc, singles, horizontal=False)
    rotation: int | None = 0
    if rotation_check:  # vertical lines grouped from all characters, the sheet may be sideways
        rotation = _rotation(cc, h_groups, _group(cc, cand, horizontal=False)[0])
        if rotation:
            return [], np.zeros(ink.shape, np.uint8), rotation
    groups = [(g, False) for g in h_groups] + [(g, True) for g in v_groups]
    results = _read([_crop(cc, g, vertical) for g, vertical in groups])
    found, unsure = [], []
    for (g, vertical), (s, conf) in zip(groups, results):
        s = fix(" ".join(s.split()))
        if conf < 50 or not any(ch.isalnum() for ch in s) or (len(s) <= 3 and conf < 75) or (len(s) == 1 and conf < 90):
            letters = _letters(cc, g, vertical, ppm) if sum(ch.isalnum() for ch in s) >= 2 else []
            if letters:
                unsure.append((replace(_text(cc, letters, vertical, s, conf, ppm), uncertain=True), letters))
            continue
        found.append((_text(cc, g, vertical, s, conf, ppm), g))
    if len(found) >= 5:  # symbol parts misread as large letters: much taller than the sheet's usual text
        typical = float(np.median([t.height for t, _ in found]))
        found = [(t, g) for t, g in found if t.height <= 2.5 * typical or t.confidence >= 85]
    found += unsure  # unread text stays a (marked) text instead of turning into a heap of short strokes
    lut = np.zeros(cc.n + 1, np.min_scalar_type(len(found)))
    for k, (_, g) in enumerate(found, 1):
        lut[np.array(g, np.int64) + 1] = k
    return [t for t, _ in found], lut[cc.lab], rotation


def _letters(cc: Components, g: list[int], vertical: bool, ppm: float) -> list[int]:
    """The character-like members of a group that looks like a line of text: at least three of
    similar height (0.8 to 6 mm), none much longer than high (dashes, symbol parts). Empty if the
    group does not look like text."""
    g = np.array(g)
    size = (cc.w if vertical else cc.h)[g].astype(np.float64)
    along = (cc.h if vertical else cc.w)[g]
    med = float(np.median(size))
    ok = (np.abs(size - med) <= 0.4 * med) & (along <= 1.5 * size)
    if ok.sum() < 3 or ok.mean() < 0.6 or not 0.8 * ppm <= med <= 6 * ppm:
        return []
    return g[ok].tolist()


_DIGITS = str.maketrans("OoSIlBZ", "0051182")


def fix(s: str) -> str:
    """Typical OCR confusions in drawings: superscript units and letters inside numbers."""
    s = re.sub(r"(\d) ?m[?2’'`]$", r"\1 m²", s)
    s = re.sub(r"(\d) ?m[3]$", r"\1 m³", s)
    s = re.sub(r"\bDN([0-9OoSIlBZ]+)\b", lambda m: "DN" + m.group(1).translate(_DIGITS), s)
    return re.sub(r"(?<=\d)[OoIl]+(?=\d)|(?<=\d)[Oo]+\b", lambda m: m.group(0).translate(_DIGITS), s)


def _group(cc: Components, idx: np.ndarray, horizontal: bool) -> tuple[list[list[int]], np.ndarray]:
    """Links neighbouring characters of one text line (union-find). Returns groups of at least two
    components and the remaining single components."""
    if not len(idx):
        return [], idx
    a, b, la, lb = (cc.x, cc.y, cc.w, cc.h) if horizontal else (cc.y, cc.x, cc.h, cc.w)
    order = idx[np.argsort(a[idx])]
    parent = {int(i): int(i) for i in idx}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    reach = int(lb[idx].max()) + 1
    for k, i in enumerate(order):
        end_i = a[i] + la[i]
        for j in order[k + 1:]:
            gap = a[j] - end_i
            if gap > reach:
                break
            size = max(lb[i], lb[j])
            if gap > 1.0 * size or gap < -0.5 * min(la[i], la[j]):
                continue
            overlap = min(b[i] + lb[i], b[j] + lb[j]) - max(b[i], b[j])
            centres = abs((2 * b[i] + lb[i]) - (2 * b[j] + lb[j])) / 2
            if overlap >= 0.3 * min(lb[i], lb[j]) and centres <= 0.6 * size:
                parent[root(int(i))] = root(int(j))
    groups: dict[int, list[int]] = {}
    for i in idx:
        groups.setdefault(root(int(i)), []).append(int(i))
    multi = [g for g in groups.values() if len(g) > 1]
    singles = np.array([g[0] for g in groups.values() if len(g) == 1], np.int64)
    return multi, singles


def _metrics(cc: Components, g: list[int], vertical: bool) -> tuple[float, float, float, float]:
    """Top and baseline across the reading direction plus the extent along it (pixels)."""
    g = np.array(g)
    if vertical:  # reads bottom to top: "up" points to -x, the baseline is on the right
        size, top, bottom, lo, hi = cc.w[g], cc.x[g], cc.x[g] + cc.w[g], cc.y[g].min(), (cc.y[g] + cc.h[g]).max()
    else:
        size, top, bottom, lo, hi = cc.h[g], cc.y[g], cc.y[g] + cc.h[g], cc.x[g].min(), (cc.x[g] + cc.w[g]).max()
    big = size >= 0.6 * size.max()
    return float(top[big].min()), float(np.median(bottom[big])), float(lo), float(hi)


def _crop(cc: Components, g: list[int], vertical: bool, turn: int = 0) -> np.ndarray:
    """Black-on-white crop of the group's own pixels, upright and scaled to CAP_PX."""
    x0 = min(cc.x[i] for i in g)
    y0 = min(cc.y[i] for i in g)
    x1 = max(cc.x[i] + cc.w[i] for i in g)
    y1 = max(cc.y[i] + cc.h[i] for i in g)
    lut = np.zeros(cc.n + 1, bool)
    lut[np.array(g) + 1] = True
    own = cv2.dilate(lut[cc.lab[y0:y1, x0:x1]].astype(np.uint8), np.ones((3, 3), np.uint8))
    img = np.where(own > 0, cc.gray[y0:y1, x0:x1], 255).astype(np.uint8)
    if vertical:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if turn:
        img = cv2.rotate(img, cv2.ROTATE_180)
    top, base, _, _ = _metrics(cc, g, vertical)
    f = CAP_PX / max(base - top, 1.0)
    img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)


def _read(crops: list[np.ndarray]) -> list[tuple[str, float]]:
    """Reads one text line per crop. Crops are stacked into batch images for Tesseract."""
    batches, cur, height = [], [], 0
    for i, c in enumerate(crops):
        if cur and height + c.shape[0] + 16 > MOSAIC_H:
            batches.append(cur)
            cur, height = [], 0
        cur.append(i)
        height += c.shape[0] + 16
    if cur:
        batches.append(cur)
    words: list[list[tuple[int, str, float]]] = [[] for _ in crops]

    def run(batch: list[int]) -> None:
        width = max(crops[i].shape[1] for i in batch)
        img = np.full((sum(crops[i].shape[0] + 16 for i in batch), width), 255, np.uint8)
        spans, y = [], 0
        for i in batch:
            h, w = crops[i].shape
            img[y:y + h, :w] = crops[i]
            spans.append((y, y + h, i))
            y += h + 16
        data = pytesseract.image_to_data(img, lang=LANG, config="--psm 6", output_type=pytesseract.Output.DICT)
        for text, conf, left, top, h in zip(data["text"], data["conf"], data["left"], data["top"], data["height"]):
            if not text.strip() or float(conf) < 0:
                continue
            yc = top + h / 2
            for y0, y1, i in spans:
                if y0 <= yc < y1:
                    words[i].append((left, text, float(conf)))
                    break

    def single(i: int) -> None:  # a line the batch read badly gets a second, separate read
        data = pytesseract.image_to_data(crops[i], lang=LANG, config="--psm 7", output_type=pytesseract.Output.DICT)
        alone = [(left, t, float(c)) for t, c, left in zip(data["text"], data["conf"], data["left"])
                 if t.strip() and float(c) >= 0]
        if _joined(alone)[1] > _joined(words[i])[1]:
            words[i] = alone

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as pool:
        list(pool.map(run, batches))
        list(pool.map(single, [i for i, ws in enumerate(words) if _joined(ws)[1] < 70]))
    return [_joined(ws) for ws in words]


def _joined(ws: list[tuple[int, str, float]]) -> tuple[str, float]:
    ws = sorted(ws)
    n = sum(len(t) for _, t, _ in ws)
    return " ".join(t for _, t, _ in ws), (sum(c * len(t) for _, t, c in ws) / n if n else 0.0)


def _text(cc: Components, g: list[int], vertical: bool, s: str, conf: float, ppm: float) -> Text:
    top, base, lo, hi = _metrics(cc, g, vertical)
    mid_across, mid_along = (top + base) / 2, (lo + hi) / 2
    x, y = (mid_across, mid_along) if vertical else (mid_along, mid_across)
    return Text(x / ppm, y / ppm, (base - top) / ppm, s, 90.0 if vertical else 0.0, round(conf, 1))


def _rotation(cc: Components, h_groups: list, v_groups: list) -> int | None:
    """Clockwise rotation that makes the text upright, of four. Each text line is read both ways round, which gives
    the number of lines read with high confidence as horizontal text the right way up (0) or upside down (180) and
    as vertical text reading bottom to top (90) or top to bottom (270). An upright sheet shows horizontal text the
    right way up and vertical text bottom to top, as drawing standards have it; a sheet that needs turning by r
    shows both turned by r. So each rotation is scored by its pair, and a sheet full of vertical pipe labels is no
    longer taken for a sideways one. Ties go to the rotation with more horizontal text, None if the text is too
    poor to decide."""
    good = {}
    for vertical, groups in ((False, h_groups), (True, v_groups)):
        groups = sorted(groups, key=len, reverse=True)[:60]
        for turn in (0, 1):
            res = _read([_crop(cc, g, vertical, turn) for g in groups]) if len(groups) >= 3 else []
            # vertical text reading bottom-to-top is upright after a 90 degree clockwise turn
            good[(90 if turn == 0 else 270) if vertical else (0 if turn == 0 else 180)] = sum(
                c >= 80 and sum(ch.isalnum() for ch in s) >= 3 for s, c in res)
    score = {r: (good[r] + good[(r + 90) % 360], good[r]) for r in (0, 90, 180, 270)}
    ranked = sorted(score, key=score.get, reverse=True)
    best, second = score[ranked[0]], score[ranked[1]]
    if best[0] < 2 or best == second or best[0] - second[0] < 2 and best[1] - second[1] < 2:
        return None
    return ranked[0]
