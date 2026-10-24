"""Finds text lines among the small connected components and reads them with Tesseract.

Characters are grouped into lines geometrically; all line crops are scaled to the same cap height
and read in a few batch images ("mosaics"), which is much faster than one Tesseract call per line.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import cv2
import numpy as np
import pytesseract
from scipy.ndimage import find_objects
from scipy.spatial import cKDTree

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


def find_texts(ink: np.ndarray, gray: np.ndarray, ppm: float, rotation_check: bool = False,
               frame: np.ndarray | None = None) -> tuple[list[Text], np.ndarray, int | None]:
    """Returns recognised text lines (sheet mm), a label image of their pixels (0: no text, k: the
    k-th text line) and, if requested, the clockwise rotation in degrees (0, 90, 180, 270) that
    makes the text upright (None if the text is not readable enough to tell). ``frame``: all strokes of the sheet,
    which may enclose a single letter (see _framed)."""
    cc = Components(ink, gray)
    cand = np.flatnonzero((np.maximum(cc.w, cc.h) <= 7.5 * ppm) & (cc.area >= 3))
    cand = np.setdiff1d(cand, _enclosing(cc, cand, ppm))
    h_groups, singles = _group(cc, cand, horizontal=True)
    framed = _framed(cc, singles, h_groups, ppm, frame)  # before vertical lines: a bubble's letter above its stem
    v_groups, _ = _group(cc, np.setdiff1d(singles, framed), horizontal=False)
    rotation: int | None = 0
    if rotation_check:  # vertical lines grouped from all characters, the sheet may be sideways
        rotation = _rotation(cc, h_groups, _group(cc, cand, horizontal=False)[0])
        if rotation:
            return [], np.zeros(ink.shape, np.uint8), rotation
    groups = [(g, False) for g in h_groups] + [(g, True) for g in v_groups]
    groups += [([i], False) for i in framed]
    results = _voted(cc, groups, _read([_crop(cc, g, vertical) for g, vertical in groups]))
    found, unsure = [], []
    for (g, vertical), (s, conf) in zip(groups, results):
        s, g = _unbarred(cc, g, vertical, fix(" ".join(s.split())))
        if conf < 50 or not any(ch.isalnum() for ch in s) or (len(s) <= 3 and conf < 75) or (len(s) == 1 and conf < 85):
            letters = _letters(cc, g, vertical, ppm, s) if sum(ch.isalnum() for ch in s) >= 2 else []
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


def _enclosing(cc: Components, cand: np.ndarray, ppm: float) -> np.ndarray:
    """The candidates that enclose a letter-sized one at most half their height, such as the box around a line size
    or the ring of a bubble: a frame, not a character, and read together with its letters it spoils their reading."""
    x0, y0, x1, y1 = cc.x[cand], cc.y[cand], cc.x[cand] + cc.w[cand], cc.y[cand] + cc.h[cand]
    centres = np.column_stack([x0 + x1, y0 + y1]) / 2
    tree = cKDTree(centres)
    out = []
    for k in np.flatnonzero(cc.h[cand] >= ppm):
        near = np.array(tree.query_ball_point(centres[k], max(cc.w[cand[k]], cc.h[cand[k]])))
        inner = (x0[near] > x0[k]) & (x1[near] < x1[k]) & (y0[near] > y0[k]) & (y1[near] < y1[k])
        sized = (cc.h[cand[near]] >= 0.5 * ppm) & (2 * cc.h[cand[near]] <= cc.h[cand[k]])
        if (inner & sized).any():
            out.append(cand[k])
    return np.array(out, np.int64)


def _framed(cc: Components, alone: np.ndarray, groups: list[list[int]], ppm: float,
            frame: np.ndarray | None = None) -> list[int]:
    """The components among ``alone`` that look like a character of the sheet's text lines and sit in a frame of their
    own, such as the loop letter of an instrument bubble: a single character is read only there, since elsewhere a
    lone stroke of that size is as likely part of a symbol."""
    lines = [g for g in groups if len(g) >= 3]
    if not lines or not len(alone):
        return []
    h = float(np.median([cc.h[i] for g in lines for i in g]))
    reach = round(FRAME * ppm)
    out = []
    for i in alone:
        if not (0.75 * h <= cc.h[i] <= 1.5 * h and 0.4 * h <= cc.w[i] <= 1.2 * h):
            continue  # not letter-sized, or a bar or a dash
        y0, x0 = max(0, cc.y[i] - reach), max(0, cc.x[i] - reach)
        y1, x1 = min(cc.lab.shape[0], cc.y[i] + cc.h[i] + reach), min(cc.lab.shape[1], cc.x[i] + cc.w[i] + reach)
        lab = cc.lab[y0:y1, x0:x1]
        window = ((lab > 0) & (lab != i + 1)).astype(np.uint8)
        gap = round(h)
        row = window[cc.y[i] - y0:cc.y[i] - y0 + cc.h[i], max(0, cc.x[i] - x0 - gap):cc.x[i] - x0 + cc.w[i] + gap]
        if row.any():
            continue  # a letter of a word whose other letters touch something: not a single character
        if frame is not None:
            window |= frame[y0:y1, x0:x1] & (lab != i + 1)
        seed = np.argwhere(lab == i + 1)[0]
        fill = np.zeros((y1 - y0 + 2, x1 - x0 + 2), np.uint8)
        cv2.floodFill(window, fill, (int(seed[1]), int(seed[0])), 1, 0, 0, 4 | cv2.FLOODFILL_MASK_ONLY | (1 << 8))
        inside = fill[1:-1, 1:-1]
        if not (inside[0].any() or inside[-1].any() or inside[:, 0].any() or inside[:, -1].any()):
            out.append(int(i))
    return out


BARS = set("|ı")  # what Tesseract reads for a bare stroke


def _unbarred(cc: Components, g: list[int], vertical: bool, s: str) -> tuple[str, list[int]]:
    """Drops a bar read at either end of a line, apart from the next character ("| LT", "ıLT"): the end of a leader or
    a pipe beside the text, which stays linework."""
    glyphs = _glyphs(cc, g, vertical)
    a, la = (cc.y, cc.h) if vertical else (cc.x, cc.w)
    top, base, _, _ = _metrics(cc, g, vertical)

    def apart(p: list[int], q: list[int]) -> bool:  # the gap between two glyphs, at least a third of a capital
        return max(a[i] for i in q) - min(a[i] + la[i] for i in p) >= (base - top) / 3 or \
            min(a[i] for i in p) - max(a[i] + la[i] for i in q) >= (base - top) / 3

    drop: list[int] = []
    s = s.strip()
    if len(s) > 1 and s[0] in BARS and len(glyphs) > 1 and apart(glyphs[0], glyphs[1]):
        drop, s, glyphs = drop + glyphs[0], s[1:].lstrip(), glyphs[1:]
    if len(s) > 1 and s[-1] in BARS and len(glyphs) > 1 and apart(glyphs[-2], glyphs[-1]):
        drop, s = drop + glyphs[-1], s[:-1].rstrip()
    return s, [i for i in g if i not in drop]


def frames(ink: np.ndarray, labels: np.ndarray, ppm: float) -> np.ndarray:
    """The strokes that enclose a text line: the outline around a tag, such as an instrument bubble, the box of a
    line number or an off-page connector (``labels`` as from find_texts). The paper around the text is filled
    from it; if the fill stays within FRAME mm of the text, the strokes it meets are the frame."""
    out = np.zeros(ink.shape, np.uint8)
    blocked = ((ink > 0) & (labels == 0)).astype(np.uint8)  # strokes other than letters stop the fill
    reach = round(FRAME * ppm)
    stroke = np.ones((2 * round(0.4 * ppm) + 1,) * 2, np.uint8)  # the whole width of the frame's stroke
    for sl in find_objects(labels):
        if sl is None:
            continue
        # a long line-number box holds several texts: along the text the frame may be further away
        for along in (reach, 4 * reach):
            y0, x0 = max(0, sl[0].start - reach), max(0, sl[1].start - along)
            y1, x1 = min(ink.shape[0], sl[0].stop + reach), min(ink.shape[1], sl[1].stop + along)
            window = blocked[y0:y1, x0:x1]
            seed = ((sl[1].start + sl[1].stop) // 2 - x0, (sl[0].start + sl[0].stop) // 2 - y0)
            if window[seed[1], seed[0]]:
                break
            fill = np.zeros((y1 - y0 + 2, x1 - x0 + 2), np.uint8)
            cv2.floodFill(window, fill, seed, 1, 0, 0, 4 | cv2.FLOODFILL_MASK_ONLY | (1 << 8))
            inside = fill[1:-1, 1:-1]
            if not (inside[0].any() or inside[-1].any() or inside[:, 0].any() or inside[:, -1].any()):
                out[y0:y1, x0:x1] |= cv2.dilate(inside, stroke) & window
                break  # otherwise the paper runs on: no frame, or a longer one
    return out


def _voted(cc: Components, groups: list[tuple[list[int], bool]], results: list[tuple[str, float]]
           ) -> list[tuple[str, float]]:
    """Corrects characters Tesseract confuses (0 and 8, 5 and S, ...) by a vote of the glyphs that look most alike
    on the sheet. A drawing is written in one font, and a glyph read differently from most of its look-alikes was
    misread: the slashed zero of a plotter font reads as 8, 9 or B now and then, but as 0 mostly. Only lines whose
    characters map one to one to their glyphs take part."""
    places, images = [], []
    for k, ((g, vertical), (s, conf)) in enumerate(zip(groups, results)):
        glyphs = _glyphs(cc, g, vertical)
        chars = [j for j, ch in enumerate(s) if ch != " "]
        if conf >= 30 and len(glyphs) == len(chars):
            places += [(k, j) for j in chars]
            images += [_glyph(cc, parts, vertical) for parts in glyphs]
    if len(images) <= LOOK_ALIKES:
        return results
    dist, near = cKDTree(np.array(images)).query(np.array(images), k=LOOK_ALIKES + 1)
    read = [results[k][0][j] for k, j in places]

    def nearest(n: int, ch: str) -> float:  # the distance to the nearest look-alike read as ch
        return next((d for d, m in zip(dist[n][1:], near[n][1:]) if read[m] == ch), np.inf)

    spread = {ch: float(np.median([nearest(n, ch) for n, c in enumerate(read) if c == ch])) for ch in set(read)}
    out = [list(s) for s, _ in results]
    for n, (k, j) in enumerate(places):
        ch = read[n]
        alike = next((c for c in CONFUSED if ch in c), None)
        if alike is None:
            continue
        best, votes = Counter(read[m] for m in near[n][1:]).most_common(1)[0]
        # and the glyph looks like those of that kind: a rare 6 among many 0 is just as near them as to anything else
        if best != ch and best in alike and votes >= LOOK_ALIKES // 2 and nearest(n, best) <= SPREAD * spread[best]:
            out[k][j] = best
    return [("".join(chars), conf) for chars, (_, conf) in zip(out, results)]


def _glyphs(cc: Components, g: list[int], vertical: bool) -> list[list[int]]:
    """The components of a text line merged into glyphs (parts across one another, like the dots of a colon), in
    reading order."""
    a, la = (cc.y, cc.h) if vertical else (cc.x, cc.w)
    out: list[list[int]] = []
    for i in sorted(g, key=lambda i: a[i]):
        j = out[-1][0] if out else i
        if j != i and min(a[i] + la[i], a[j] + la[j]) - max(a[i], a[j]) > 0.5 * min(la[i], la[j]):
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _glyph(cc: Components, parts: list[int], vertical: bool) -> np.ndarray:
    """A glyph's image, upright, scaled to 10 x 14 pixels and normalised (for comparing glyphs)."""
    x0, y0 = min(cc.x[i] for i in parts), min(cc.y[i] for i in parts)
    x1, y1 = max(cc.x[i] + cc.w[i] for i in parts), max(cc.y[i] + cc.h[i] for i in parts)
    img = cc.gray[y0:y1, x0:x1].astype(np.float32)
    if vertical:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    f = cv2.resize(img, (10, 14), interpolation=cv2.INTER_AREA).ravel()
    return (f - f.mean()) / (f.std() + 1e-6)


def _letters(cc: Components, g: list[int], vertical: bool, ppm: float, s: str) -> list[int]:
    """The character-like members of a group that looks like a line of text read as ``s``: at least three of
    similar height (0.8 to 6 mm), none longer than three characters (a scan merges touching letters), and as many
    characters read as their length holds: a symbol, such as a valve with its stem, reads as two or three large
    letters. Empty if the group does not look like text."""
    g = np.array(g)
    size = (cc.w if vertical else cc.h)[g].astype(np.float64)
    start, along = ((cc.y, cc.h) if vertical else (cc.x, cc.w))
    med = float(np.median(size))
    ok = (np.abs(size - med) <= 0.4 * med) & (along[g] <= 3.5 * size)
    if ok.sum() < 3 or ok.mean() < 0.6 or not 0.8 * ppm <= med <= 6 * ppm:
        return []
    extent = (start[g[ok]] + along[g[ok]]).max() - start[g[ok]].min()
    if extent > 1.3 * med * len(s.replace(" ", "")):
        return []
    return g[ok].tolist()


LOOK_ALIKES = 16  # glyphs most like a glyph that vote on what it is
SPREAD = 1.5  # a glyph is like a kind if as near to it as its own glyphs usually are to one another, give or take
# characters Tesseract mistakes for one another in drawing fonts; not I and l, which differ only in height
CONFUSED = ("0OD89B6Q", "17TJ", "5S", "2Z")
FRAME = 12.0  # mm: the frame around a text (a tag's bubble or box) is at most this far from it
SAME_HEIGHT = 1.5  # characters of one text line differ in height by at most this factor (a lower-case e and an f)
PIECES = 0.25  # gap between the pieces of a broken character at most, in character heights (not a dashed line)
SHAPES = set("0oO°1lI|[](){}.,:;-_= ")  # what the circles and bars of a symbol read as: two hotplates as "oo"
_DIGITS = str.maketrans("OoSIlBZ", "0051182")


def digits(s: str) -> str:
    """A number with the letters OCR reads for digits turned into them (O as 0, S as 5, ...)."""
    return s.translate(_DIGITS)


def worded(s: str) -> bool:
    """Whether a text reads as words or a code, not only as the circles and bars of a symbol ("oo", "101")."""
    return not set(s) <= SHAPES


def fix(s: str) -> str:
    """Typical OCR confusions in drawings: superscript units and letters inside numbers."""
    s = re.sub(r"(\d) ?m[?2’'`]$", r"\1 m²", s)
    s = re.sub(r"(\d) ?m[3]$", r"\1 m³", s)
    s = re.sub(r"\bDN([0-9OoSIlBZ]+)\b", lambda m: "DN" + m.group(1).translate(_DIGITS), s)
    # in a code an O or I next to a digit is a 0 or 1: PN16, CS31, C=01XY2Z (not the I of I/P or TIC)
    return re.sub(r"(?<=\d)[OoIl]+|[OoIl]+(?=\d)", lambda m: m.group(0).translate(_DIGITS), s)


def _group(cc: Components, idx: np.ndarray, horizontal: bool) -> tuple[list[list[int]], np.ndarray]:
    """Links neighbouring characters of one text line (union-find). Characters of similar height are linked first;
    then small marks (hyphens, dots, inch marks, the pieces of a broken character) join a line of two or more
    characters they sit in. So a tall
    stroke, such as the slanted side of an off-page connector next to two lines of text, links neither line to the
    other nor itself to them. Returns groups of at least two components and the remaining single components."""
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

    def neighbours():
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
                    yield int(i), int(j), gap

    pairs = list(neighbours())
    for i, j, _ in pairs:
        if max(lb[i], lb[j]) <= SAME_HEIGHT * min(lb[i], lb[j]):
            parent[root(i)] = root(j)
    members: dict[int, list[int]] = {}
    for i in idx:
        members.setdefault(root(int(i)), []).append(int(i))
    height = {r: float(np.median(lb[m])) for r, m in members.items() if len(m) > 1}  # of a line's characters
    letters = {i for m in members.values() if len(m) > 1 for i in m}
    for i, j, gap in pairs:  # left to right, so the pieces of a broken character join one after the other
        for small, big in ((i, j), (j, i)):
            h = height.get(root(big), 0.0)
            middle = b[small] + lb[small] / 2  # of the small mark, beside the one it joins
            near = big in letters or gap <= PIECES * h  # a mark that joined before: only if it nearly touches
            if near and lb[small] < SAME_HEIGHT * h and b[big] - 0.2 * h <= middle <= b[big] + lb[big] + 0.2 * h:
                parent[root(small)] = root(big)
                break
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
