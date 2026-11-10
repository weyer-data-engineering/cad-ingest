"""Finds text lines among the small connected components and reads them with Tesseract.

Characters are grouped into lines geometrically; all line crops are scaled to the same cap height
and read in a few batch images ("mosaics"), which is much faster than one Tesseract call per line.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
import pytesseract
from scipy import ndimage
from scipy.ndimage import find_objects
from scipy.spatial import cKDTree

from ..config import Settings
from .model import Text
from .spotting import ABOUT, LEVEL, TILT, Spot, lean, leeway

os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # parallelism comes from the batches, not from Tesseract

LANG = Settings.from_env().ocr_languages
CAP_PX = 32  # cap height in pixels that Tesseract reads well
MOSAIC_H = 6000
UNSURE = 0.8  # spotted lines the recogniser is less sure of are marked for review
SHORT = 2  # characters: a spotted line this short is new text only if the recogniser is SURE of it
GLYPH = 7.5  # mm: the largest character on a sheet
SURE = 0.9
TALL = 1.6  # nor much higher than the usual text of the sheet (two stacked circles read as "8")
SPACE = 2.5
NARROW = 0.35  # cap heights a character takes along the line at least, on average ("1", "I" and "." with their gaps)
LATIN = 0x24F  # last code point of the Latin scripts; the recogniser also knows Chinese, Korean, ...
DIAGONAL = 30.0  # degrees off the axis it was grouped along from which an oblique line is likely in pieces
STRAIGHT = 0.25  # cap heights the centres of an oblique line's characters lie off a straight line at most (spread)
SURER = 10  # confidence by which a reading turned by 180 degrees must beat the usual way round to replace it
PLAUSIBLE = 60  # confidence from which a reading turned by 180 degrees stays a candidate (see Readings)


class Components:
    """Connected components of the remaining ink with their boxes; ``gray`` is the normalised
    image the text crops are cut from (Tesseract binarises blurred glyphs better than we do)."""

    def __init__(self, ink: np.ndarray, gray: np.ndarray):
        self.gray = gray
        n, self.lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        self.x, self.y, self.w, self.h, self.area = (stats[1:, i].astype(np.int64) for i in range(5))
        self.n = n - 1


class Tilted:
    """The boxes of components as seen along a text line at ``angle`` degrees (counter-clockwise, as Text.angle):
    turned so that the line reads from left to right with its letters upright, where the measures of horizontal
    lines apply (_group, _metrics, _letters, ...). Pixels; only the components ``idx`` have a box (NaN elsewhere)."""

    def __init__(self, cc: Components, idx, angle: float):
        a = math.radians(angle)
        self.angle, self.cos, self.sin, self.n = angle, math.cos(a), math.sin(a), cc.n
        self.x, self.y, self.w, self.h = (np.full(cc.n, np.nan) for _ in range(4))
        for i in np.asarray(idx, np.int64):  # pixel k covers k to k + 1, as the boxes of Components
            ys, xs = np.nonzero(cc.lab[cc.y[i]:cc.y[i] + cc.h[i], cc.x[i]:cc.x[i] + cc.w[i]] == i + 1)
            along, across = self.turn(xs + cc.x[i] + 0.5, ys + cc.y[i] + 0.5)
            self.x[i], self.y[i] = along.min() - 0.5, across.min() - 0.5
            self.w[i], self.h[i] = along.max() + 0.5 - self.x[i], across.max() + 0.5 - self.y[i]

    def turn(self, x, y):
        """Pixels of the sheet to pixels along and across the line (across grows towards its baseline)."""
        return x * self.cos - y * self.sin, x * self.sin + y * self.cos

    def back(self, along, across):
        """Pixels along and across the line to pixels of the sheet."""
        return along * self.cos + across * self.sin, -along * self.sin + across * self.cos

    def turned(self) -> Tilted:
        """The same boxes seen the other way round, for reading the line turned by 180 degrees."""
        out = object.__new__(Tilted)
        out.angle, out.cos, out.sin, out.n = (self.angle + 180) % 360, -self.cos, -self.sin, self.n
        out.x, out.y, out.w, out.h = -(self.x + self.w), -(self.y + self.h), self.w, self.h
        return out


@dataclass
class Readings:
    """What the readers leave open until the kind of drawing is known (pipeline._settled): of a text read vertically or
    at an angle, the reading turned by 180 degrees where that is plausible too ("0,60" and "09'0"), the single
    characters read alone in a frame, which may be letters or digits (O or 0), the vertical lines that only reading
    them again both ways round made sure of the reading the spotter was unsure of (see spotted), and, with
    ``by_token``, the readings of a text the vote of look-alike glyphs had a say in (see _voted): as read, and with
    every character the vote would change changed. Kept by the id() of the text as read, with the text itself, so a
    text replaced later (by a spotted line, by grammar) drops out."""

    turned: dict[int, tuple[Text, Text]] = field(default_factory=dict)  # text -> it read turned by 180 degrees
    framed: dict[int, Text] = field(default_factory=dict)
    doubted: dict[int, tuple[Text, float]] = field(default_factory=dict)  # text -> the confidence it was found with
    by_token: bool = False  # the vote decides per token (Features.lookalike_tokens)
    voted: dict[int, tuple[Text, str, str]] = field(default_factory=dict)  # text -> it as read, as voted everywhere

    def other(self, t: Text) -> Text | None:
        """The reading turned by 180 degrees of ``t``, if one is kept."""
        kept = self.turned.get(id(t))
        return kept[1] if kept and kept[0] is t else None

    def doubt(self, t: Text) -> float | None:
        """The confidence the spotter found ``t`` with, where only reading it again made sure of it."""
        kept = self.doubted.get(id(t))
        return kept[1] if kept and kept[0] is t else None

    def alone(self, t: Text) -> bool:
        """Whether ``t`` is a single character read alone in a frame."""
        return self.framed.get(id(t)) is t

    def votes(self, t: Text) -> tuple[str, str] | None:
        """``t`` as read before the vote of look-alikes and as the vote would have it everywhere, if it had a say."""
        kept = self.voted.get(id(t))
        return kept[1:] if kept and kept[0] is t else None

    def keep(self, t: Text, read: str, best: str) -> None:
        """Keeps the readings of ``t`` before and after the vote, where they differ from it only in characters (the
        same length and spaces); a text they differ from in more was put right after the vote and stays as it is."""
        if (read != t.text or best != t.text) and all(_aligned(t.text, s) for s in (read, best)):
            self.voted[id(t)] = (t, read, best)


def _aligned(a: str, b: str) -> bool:
    """Whether two readings differ in characters only: of the same length, with their spaces in the same places."""
    return len(a) == len(b) and all((x == " ") == (y == " ") for x, y in zip(a, b))


def find_texts(ink: np.ndarray, gray: np.ndarray, ppm: float, rotation_check: bool = False,
               frame: np.ndarray | None = None, printed: float = 1.0, oblique: bool = False, both: bool = False,
               readings: Readings | None = None) -> tuple[list[Text], np.ndarray, int | None]:
    """Returns recognised text lines (mm at ``ppm`` pixels each), a label image of their pixels (0: no text, k: the
    k-th text line) and, if requested, the clockwise rotation in degrees (0, 90, 180, 270) that
    makes the text upright (None if the text is not readable enough to tell). ``frame``: all strokes of the sheet,
    which may enclose a single letter (see _framed). ``printed``: the share of its drawn size the sheet shows the
    drawing at, where ``ppm`` counts the pixels per millimetre as drawn. ``oblique``: lines of characters at other
    angles are read too, both ways round (see _oblique). ``both``: vertical lines are read top to bottom as well.
    ``readings``: what is left open until the kind of drawing is known is kept there (see Readings)."""
    cc = Components(ink, gray)
    # a character is at most GLYPH mm on the sheet, whatever size it is drawn at: on a small print blur runs the
    # letters of a word together into one shape, which is read as well
    cand = np.flatnonzero((np.maximum(cc.w, cc.h) <= GLYPH * ppm / printed) & (cc.area >= 3))
    cand = np.setdiff1d(cand, _enclosing(cc, cand, ppm))
    h_groups, singles = _group(cc, cand, horizontal=True)
    framed = _framed(cc, singles, h_groups, ppm, frame)  # before vertical lines: a bubble's letter above its stem
    v_groups, rest = _group(cc, np.setdiff1d(singles, framed), horizontal=False)
    rotation: int | None = 0
    if rotation_check:  # vertical lines grouped from all characters, the sheet may be sideways
        rotation = _rotation(cc, h_groups, _group(cc, cand, horizontal=False)[0])
        if rotation:
            return [], np.zeros(ink.shape, np.uint8), rotation
    lines: list[tuple[list[int], Tilted]] = []
    if oblique:
        h_groups, v_groups, lines = _oblique(cc, h_groups, v_groups, rest)
    groups = [(g, False) for g in h_groups] + [(g, True) for g in v_groups]
    groups += [([i], False) for i in framed]
    crops = [_crop(cc, g, vertical) for g, vertical in groups]
    crops += [_tilted_crop(cc, t, g, turn) for g, t in lines for turn in (0, 1)]  # oblique lines both ways round
    down = [len(h_groups) + k for k in range(len(v_groups))] if both else []
    read = _read(crops + [_crop(cc, groups[k][0], True, 1) for k in down])  # vertical lines read top to bottom too
    careful = readings is not None and readings.by_token
    results, every = _voted(cc, groups, read[:len(groups)], careful=careful)
    # (the reading, the one turned by 180 degrees, whether the line is LEVEL)
    pairs = [(_line(cc, g, vertical, s, conf, ppm), None, False) for (g, vertical), (s, conf) in zip(groups, results)]
    for (g, vertical), (first, _), best, (line, *_) in zip(groups, read, every, pairs) if careful else ():
        if line is not None:  # put right as the line was (see _line)
            readings.keep(line[0], *(_unbarred(cc, g, vertical, fix(" ".join(s.split())))[0] for s in (first, best)))
    for k, (s, conf) in zip(down, read[len(groups) + 2 * len(lines):]):
        # of three glyphs at least, and a character for each: the two barbs of a flow arrow on a horizontal pipe, one
        # above the other, read top to bottom as letters ("A", "AA"); measured along it, its baseline on the left
        if len(s.replace(" ", "")) >= len(_glyphs(cc, g := groups[k][0], True)) >= 3:
            pairs[k] = pairs[k][0], _line(Tilted(cc, g, -90.0), g, False, s, conf, ppm), False
    sure = {"ppm": ppm, "guess": False}  # an oblique line read badly is no guess: broken hatching lines up like letters
    for k, (g, t) in enumerate(lines):  # their two readings follow those of the other lines
        first, turned = read[len(groups) + 2 * k], read[len(groups) + 2 * k + 1]
        pairs.append((_line(t, g, False, *first, **sure), _line(t.turned(), g, False, *turned, **sure),
                      abs(t.angle) < LEVEL))
    found, unsure, others = [], [], {}
    for first, second, level in pairs:
        keep, other = _either(first, second, level)
        if keep is not None:
            (unsure if keep[0].uncertain else found).append(keep)
            others[id(keep[0])] = other
    if len(found) >= 5:  # symbol parts misread as large letters: much taller than the sheet's usual text
        typical = float(np.median([t.height for t, _ in found]))
        found = [(t, g) for t, g in found if t.height <= 2.5 * typical or t.confidence >= 85]
    found += unsure  # unread text stays a (marked) text instead of turning into a heap of short strokes
    if readings is not None:
        alone = set(framed)
        for t, g in found:
            if others.get(id(t)) is not None:
                readings.turned[id(t)] = (t, others[id(t)])
            if len(g) == 1 and g[0] in alone and not t.uncertain:
                readings.framed[id(t)] = t
    lut = np.zeros(cc.n + 1, np.min_scalar_type(len(found)))
    for k, (_, g) in enumerate(found, 1):
        lut[np.array(g, np.int64) + 1] = k
    return [t for t, _ in found], lut[cc.lab], rotation


def _line(cc: Components | Tilted, g: list[int], vertical: bool, s: str, conf: float, ppm: float,
          guess: bool = True) -> tuple[Text, list[int]] | None:
    """The text a line of characters read as ``s`` makes, with the characters it is made of: as read where the reader
    is sure enough of it, a guess to check where it still looks like text (see _letters; not unless ``guess``), else
    none."""
    s, g = _unbarred(cc, g, vertical, fix(" ".join(s.split())))
    if conf < 50 or not any(ch.isalnum() for ch in s) or (len(s) <= 3 and conf < 75) or (len(s) == 1 and conf < 85):
        letters = _letters(cc, g, vertical, ppm, s) if guess and sum(ch.isalnum() for ch in s) >= 2 else []
        if not letters:
            return None
        t, g = replace(_text(cc, letters, vertical, s, conf, ppm), uncertain=True), letters
    else:
        t = _text(cc, g, vertical, s, conf, ppm)
    return t, g


def _either(first: tuple[Text, list[int]] | None, second: tuple[Text, list[int]] | None, level: bool = False
            ) -> tuple[tuple[Text, list[int]] | None, Text | None]:
    """Of a line read both ways round (``first`` the way drawings write it, readable from the bottom or the right),
    the reading to keep and the other one where that is plausible too: a sure reading before a guess to check, and the
    one turned by 180 degrees only where it is SURER, and where the line is ``level`` only if the first one reads as
    nothing: an upright sheet writes it from left to right ("501" leaning a little reads upside down as "10%" now and
    then), but upside down it is read too."""
    def rank(p) -> float:
        return -np.inf if p is None else p[0].confidence - (100 if p[0].uncertain else 0)

    turn = rank(second) > rank(first) + SURER and (not level or first is None)
    keep, other = (second, first) if turn else (first, second)
    plausible = other is not None and not other[0].uncertain and other[0].confidence >= PLAUSIBLE
    return keep, other[0] if plausible and keep is not None else None


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


def spotted(texts: list[Text], labels: np.ndarray, ink: np.ndarray, gray: np.ndarray, spots: list[Spot], ppm: float,
            readings: Readings | None = None) -> tuple[list[Text], np.ndarray]:
    """The texts with the spotted lines (see spotting.py): a line replaces the texts whose letters lie in its box and
    takes those letters as its pixels; texts no line covers stay as they were read. Letters are the components of
    ``ink`` wholly in the box and as high as most of them, so a pipe or the ring of a bubble that touches the line, a
    dash or a line that crosses it stays linework. Where no text was found before, a reading must not be UNSURE, and
    one of one or two characters must be SURE and no TALLer than the usual text: such a reading is as likely a symbol,
    a valve read as two letters or a fixture as "8". Characters both readers confuse are put to the vote of their
    look-alikes on the sheet (see _voted), with the first reading of the texts a line replaces among them: the readers
    make different mistakes. Wide spaces the recogniser dropped are put back (see _spaced). A line that runs across part
    of a longer text written the other way is a misreading and leaves it; where the first reading is the shorter one, it
    is usually the misreading (a vertical "mm" read as "33"). A line the spotter read along its turned box (an oblique
    one) is measured along it (see Tilted), unless its letters run along an axis after all (see _axis): then it keeps
    the reading it was found with where that is about as sure (see _level). ``readings``: the spotter's reading turned
    by 180 degrees is kept there where it is plausible too, a single character read where one stood alone in a frame
    stays one so read, and a vertical line that reading it again only made surer of keeps its doubt (see _doubted)."""
    cc = Components(ink, gray)
    old = np.zeros(cc.n, np.int64)  # the text of each component
    if cc.n:
        old[:] = ndimage.maximum(labels, cc.lab, np.arange(1, cc.n + 1))
    taken = np.zeros(cc.n, bool)
    usual = float(np.median([t.height for t in texts])) * ppm if texts else np.inf
    lines, claimed = [], set()
    for s in spots:
        text = fix(" ".join(s.text.split()))
        if not _readable(text):
            continue  # nothing to read, or read in a script the sheet is not written in
        seen, box = _seen(cc, s)
        near, letters, cap = _in_line(seen, box, old, taken)
        if seen is not cc and (axis := _axis(cc, letters, s)) is not None:  # a box askew of its letters
            s = _level(s, axis)
            text = fix(" ".join(s.text.split()))
            (seen, box), (near, letters, cap) = (cc, s), _in_line(cc, s, old, taken)
        owners = set(old[letters].tolist()) - {0}
        n, angle = sum(ch.isalnum() for ch in text), _reads(s)
        if any(_other_way(texts[k - 1].angle, angle) and sum(ch.isalnum() for ch in texts[k - 1].text) > n
               and not np.isin(np.flatnonzero(old == k), letters).all() for k in owners):
            continue  # a shorter reading across part of a text written the other way: a vertical "8,82" as "82", "8"
        short = n <= SHORT
        if not owners and (s.score < UNSURE or short and (s.score < SURE or cap > TALL * usual)):
            continue
        taken[letters] = True
        claimed |= owners
        framed = n == 1 and readings is not None and any(readings.alone(texts[k - 1]) for k in owners)
        lines.append((s, text, near, letters, cap, seen, box, framed))
    # oblique lines and lines read top to bottom do not vote: their glyphs are not cut upright
    axes = {0.0: False, 90.0: True}  # whether a text that reads along an axis is vertical
    groups = [(np.flatnonzero(old == k).tolist(), axes.get(t.angle)) for k, t in enumerate(texts, 1)]
    groups += [(near.tolist(), axes.get(_reads(s))) for s, _, near, *_ in lines]
    given = [(t.text, t.confidence) for t in texts] + [(t, 100 * s.score) for s, t, *_ in lines]
    careful = readings is not None and readings.by_token
    agreed = _agreed(cc, groups, given, texts, readings, [s for s, *_ in lines]) if careful else frozenset()
    read, every = _voted(cc, groups, given, agreed, careful)
    keep = [k for k in range(1, len(texts) + 1) if k not in claimed]
    dark = gray < 128
    parts = []  # (spotted line, its letters, their text, cap height, whether the line is whole, seen, box, framed,
    # the text as spotted and as the vote would have it, or None)
    for (s, first, near, letters, cap, seen, box, framed), (text, _), best in zip(lines, read[len(texts):],
                                                                                 every[len(texts):]):
        uncut = box.vertical or seen is not cc  # _split looks for strokes across a horizontal line only

        def cut(reading: str) -> list[tuple[np.ndarray, str]]:
            return _split(seen, letters, uncut, _spaced(seen, near, box.vertical, reading), dark)

        split = cut(text)
        votes = [None] * len(split)
        if careful and (text != first or text != best):  # the readings before and after the vote, cut alike
            first_cut, best_cut = cut(first), cut(best)
            if len(first_cut) == len(best_cut) == len(split):
                votes = [(a, b) for (_, a), (_, b) in zip(first_cut, best_cut)]
        parts += [(s, part, piece, cap, len(split) == 1, seen, box, framed, vote)
                  for (part, piece), vote in zip(split, votes)]
    kind = np.min_scalar_type(len(keep) + len(parts))
    ids = np.zeros(len(texts) + 1, kind)
    ids[keep] = np.arange(1, len(keep) + 1)
    line = np.zeros(cc.n + 1, kind)
    out = [texts[k - 1] for k in keep]
    for s, letters, text, cap, whole, seen, box, framed, vote in parts:
        t = _spotted_text(seen, box, letters, text, cap, whole, s, ppm)
        out.append(t)
        line[letters + 1] = len(out)
        if vote is not None:
            readings.keep(t, *vote)
        if readings is not None and s.turned and whole and s.turned[1] >= UNSURE and not t.uncertain:
            other = replace(t, text=fix(" ".join(s.turned[0].split())), angle=_degrees(t.angle + 180),
                            confidence=round(100 * s.turned[1], 1))
            readings.turned[id(t)] = (t, other)
        if readings is not None and _doubted(s) and not t.uncertain:
            readings.doubted[id(t)] = (t, round(100 * s.found[1], 1))
        if framed:
            readings.framed[id(t)] = t
    take = line[cc.lab]
    return out, np.where(take > 0, take, ids[labels])


def _agreed(cc: Components, groups: list[tuple[list[int], bool | None]], given: list[tuple[str, float]],
            texts: list[Text], readings: Readings, spots: list[Spot]) -> frozenset[tuple[int, int]]:
    """The characters (line, place) of the spotted lines, which follow the ``texts`` in ``groups`` and ``given``, that
    the text recognition read alike in the text the glyph is part of, before its own vote, both readers surely (a text
    not marked, a line spotted at UNSURE at least): two readers that see a glyph in different ways agree on it."""
    first: dict[tuple[int, ...], str] = {}  # a glyph of a text (its components) -> its character as read
    for (g, vertical), t in zip(groups, texts):
        if vertical is None or t.uncertain:
            continue
        glyphs, chars = _glyphs(cc, g, vertical), [ch for ch in (readings.votes(t) or (t.text,))[0] if ch != " "]
        if len(glyphs) == len(chars):
            first.update(zip((tuple(sorted(p)) for p in glyphs), chars))
    out = set()
    for k, ((g, vertical), (s, _), spot) in enumerate(zip(groups[len(texts):], given[len(texts):], spots), len(texts)):
        if vertical is None or spot.score < UNSURE:
            continue
        glyphs, places = _glyphs(cc, g, vertical), [j for j, ch in enumerate(s) if ch != " "]
        if len(glyphs) == len(places):
            out.update((k, j) for p, j in zip(glyphs, places) if first.get(tuple(sorted(p))) == s[j])
    return frozenset(out)


def _spotted_text(seen: Components | Tilted, box: Spot, letters: np.ndarray, text: str, cap: float, whole: bool,
                  s: Spot, ppm: float) -> Text:
    """The text of a spotted line: centred on its letters along the line (on its box where only some of them were
    found), across it on its box."""
    span = (box.y0, box.y1) if box.vertical else (box.x0, box.x1)
    lo, hi = _metrics(seen, letters.tolist(), box.vertical)[2:] if len(letters) else span  # its letters along it
    if (length := _length(lo, hi, cap, text, ppm)) is None and whole:  # only some of its letters found: on its box
        lo, hi = span
    x, y = ((box.x0 + box.x1) / 2, (lo + hi) / 2) if box.vertical else ((lo + hi) / 2, (box.y0 + box.y1) / 2)
    if isinstance(seen, Tilted):
        x, y = seen.back(x, y)
    return Text(x / ppm, y / ppm, cap / ppm, text, _reads(s), round(100 * s.score, 1), s.score < UNSURE, length)


def _seen(cc: Components, s: Spot) -> tuple[Components | Tilted, Spot]:
    """The components and the box of a spotted line as seen along it: as they are where it reads horizontally or
    vertically; turned with it where it is oblique, with the box turned with it too."""
    if not s.oblique:
        return cc, s
    m = 0.1 * min(s.x1 - s.x0, s.y1 - s.y0)  # what lies in the turned box lies in the upright one around it
    near = np.flatnonzero((cc.x >= s.x0 - m) & (cc.x + cc.w <= s.x1 + m) & (cc.y >= s.y0 - m)
                          & (cc.y + cc.h <= s.y1 + m))
    seen = Tilted(cc, near, s.angle)
    along, across = seen.turn(s.quad[:, 0].astype(float), s.quad[:, 1].astype(float))
    return seen, Spot(float(along.min()), float(across.min()), float(along.max()), float(across.max()), s.text, s.score)


def _axis(cc: Components, letters: np.ndarray, s: Spot) -> float | None:
    """Where a line found in a turned box runs along an axis after all: its letters do, within what they can tell
    (see _direction), as the box of a level line next to a contour line may lean towards it; or, of fewer than three
    letters, its box leans no further than such a box can tell (a short word on a photo, letters run together). The
    direction of that axis nearest to the one the line reads in, else None."""
    if len(letters) >= 3 and (d := _direction(cc, letters.tolist())) is not None:
        level, oblique = abs(d[0]) <= 45, _off(d) > d[2]
    else:
        angle, along, across = lean(s.quad)
        level, oblique = abs(angle) <= 45, min(abs(angle), 90 - abs(angle)) > leeway(along, across)
    if oblique:
        return None
    return min((0.0, 180.0) if level else (90.0, 270.0), key=lambda a: abs((a - s.angle + 180) % 360 - 180))


def _level(s: Spot, axis: float) -> Spot:
    """A line read along a turned box whose letters run along the ``axis`` after all, at that axis: with the reading it
    was found with where that one is about as sure (ABOUT) as the one along the box. The box leaned for another reason
    than its line (a descender, a level mark beside it, the paper of a photo), and the same pixels read again read worse
    now and then ("±0,00" as "±0,08", "Kühler" as "Kuhler"); as found, RapidOCR read them along the box as well."""
    if s.found and _readable(s.found[0]) and s.found[1] >= s.score - ABOUT:
        return replace(s, text=s.found[0], score=s.found[1], angle=axis)
    return replace(s, angle=axis)


def _doubted(s: Spot) -> bool:
    """Whether a vertical line read again both ways round reads as the spotter found it, unsure of it: read again it
    is surer, but of the same reading, which tells nothing new ("3.94" for "3,94" either way round, its comma blurred
    into a point). Unsure means marked, unless the kind's vocabulary knows the reading (pipeline._settled)."""
    found = s.found
    return s.vertical and found is not None and found[1] < UNSURE <= s.score and found[0].split() == s.text.split()


def _readable(text: str) -> bool:
    """Whether a reading holds something to read, in the script the sheet is written in."""
    return any(ch.isalnum() for ch in text) and not any(ord(ch) > LATIN and ch not in "€–’“”" for ch in text)


def _reads(s: Spot) -> float:
    """The direction a spotted line reads in, as Text.angle."""
    return round(_degrees(s.angle), 1) if s.angle is not None else 90.0 if s.vertical else 0.0


def _other_way(a: float, b: float) -> bool:
    """Whether two texts read in clearly different directions (degrees, as Text.angle)."""
    return abs((a - b + 180) % 360 - 180) > 2 * TILT


def _degrees(a: float) -> float:
    """An angle in degrees from -180 (excluded) to 180."""
    a = (a + 180) % 360 - 180
    return 180.0 if a == -180 else a


def _split(cc: Components, letters: np.ndarray, vertical: bool, text: str, dark: np.ndarray
           ) -> list[tuple[np.ndarray, str]]:
    """A spotted line cut into the texts on either side of a stroke that crosses it between two glyphs from above its
    letters to below them: the end of a line number's frame or the ring of the bubble next to it, a break between two
    labels. The two glyphs must be a letter's height apart or the reading must have a space there, since a wall may run
    through a room's area between two digits. Its characters go with its glyphs one to one; where a character or two
    were not read, across a single cut the shorter side takes as many characters from its end as it has glyphs (at a
    space the reader put next to it). As it is if it reads vertically or that does not work out."""
    glyphs = _glyphs(cc, letters.tolist(), vertical)
    chars = [j for j, ch in enumerate(text) if ch != " "]
    if vertical or len(glyphs) < 2:
        return [(letters, text)]
    top, base = int(cc.y[letters].min()), int((cc.y[letters] + cc.h[letters]).max())
    reach = round(0.3 * (base - top))  # a curve, as the end of a frame, runs this far aside above and below the letters
    band = ndimage.maximum_filter1d(dark[max(0, top - reach):base + reach], 2 * reach + 1, axis=1)
    cuts = {n: x1 - x0 >= base - top for n, (p, g) in enumerate(zip(glyphs, glyphs[1:]), 1)
            if (x0 := max(cc.x[i] + cc.w[i] for i in p)) < (x1 := min(cc.x[i] for i in g))
            and (band[:, x0:x1].mean(axis=0) >= 0.9).any()}  # cut: whether wide
    if not cuts or len(glyphs) != len(chars) and (len(cuts) > 1 or abs(len(glyphs) - len(chars)) > 2):
        return [(letters, text)]
    if len(glyphs) == len(chars):
        at = {n: chars[n] for n in cuts}
    else:  # the shorter side is read whole more likely
        n = next(iter(cuts))
        k = n if n <= len(glyphs) - n else len(chars) - (len(glyphs) - n)
        if not 0 < k < len(chars):
            return [(letters, text)]
        spaces = [j for j in range(chars[max(0, k - 2)], chars[min(len(chars) - 1, k + 1)] + 1) if text[j] == " "]
        at = {n: min(spaces, key=lambda j: abs(j - chars[k])) if spaces else chars[k]}
    at = {n: j for n, j in at.items() if cuts[n] or " " in text[j - 1:j + 1]}
    if not at:
        return [(letters, text)]
    glyph_bounds, text_bounds = [0, *at, len(glyphs)], [0, *at.values(), len(text)]
    return [(np.array([i for g in glyphs[a:b] for i in g]), text[c:e].strip())
            for a, b, c, e in zip(glyph_bounds, glyph_bounds[1:], text_bounds, text_bounds[1:])]


def _spaced(cc: Components, near: np.ndarray, vertical: bool, text: str) -> str:
    """The reading with a space added where the gap between two glyphs is wide (the recogniser drops some wide spaces:
    "11,56m²"), if its characters map one to one to the glyphs. Wide is SPACE times the line's usual gap."""
    glyphs = _glyphs(cc, near.tolist(), vertical)
    chars = [(j > 0 and text[j - 1] == " ", ch) for j, ch in enumerate(text) if ch != " "]  # (space before, char)
    if len(glyphs) != len(chars) or len(chars) < 3:
        return text
    a, la = (cc.y, cc.h) if vertical else (cc.x, cc.w)
    gaps = [min(a[i] for i in g) - max(a[i] + la[i] for i in p) for p, g in zip(glyphs, glyphs[1:])]
    wide = max(SPACE * max(float(np.median(gaps)), 1.0), 0.25 * float(np.median([max(cc.w[i], cc.h[i]) for i in near])))
    return chars[0][1] + "".join((" " if space or gap > wide else "") + ch for gap, (space, ch) in zip(gaps, chars[1:]))


def _in_line(cc: Components, s: Spot, old: np.ndarray, taken: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The components wholly in a spotted line's box (not given to another line), its letters among them, and their
    height across the line (pixels)."""
    across, size = (s.x1 - s.x0, cc.w) if s.vertical else (s.y1 - s.y0, cc.h)
    m = 0.1 * across
    near = np.flatnonzero((cc.x >= s.x0 - m) & (cc.x + cc.w <= s.x1 + m) & (cc.y >= s.y0 - m)
                          & (cc.y + cc.h <= s.y1 + m) & ~taken)
    sizes = size[near]
    tall = sizes[sizes >= 0.25 * across]
    cap = float(np.median(tall)) if len(tall) else across / 2
    return near, near[((sizes >= 0.5 * cap) & (sizes <= 1.5 * cap)) | (old[near] > 0)], cap  # and what was text


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


def _voted(cc: Components, groups: list[tuple[list[int], bool]], results: list[tuple[str, float]],
           agreed: frozenset[tuple[int, int]] = frozenset(), careful: bool = False
           ) -> tuple[list[tuple[str, float]], list[str]]:
    """Corrects characters Tesseract confuses (0 and 8, 5 and S, ...) by a vote of the glyphs that look most alike
    on the sheet. A drawing is written in one font, and a glyph read differently from most of its look-alikes was
    misread: the slashed zero of a plotter font reads as 8, 9 or B now and then, but as 0 mostly. Only lines whose
    characters map one to one to their glyphs take part, and that read horizontally or bottom to top (``vertical``
    None for the others, whose glyphs are not cut upright). Returns the readings with the votes taken, and the
    readings as the vote would have them everywhere. ``careful``: only a CLEAR majority of the look-alikes changes a
    character; a vote without one, which flips with a few glyphs more or less elsewhere on the sheet, changes it in
    the second readings only, so the text can be marked (see Readings). Characters in ``agreed`` ((line, place)), which
    two readers read alike, are not voted over, though they vote on the others."""
    places, images = [], []
    for k, ((g, vertical), (s, conf)) in enumerate(zip(groups, results)):
        if vertical is None:
            continue
        glyphs = _glyphs(cc, g, vertical)
        chars = [j for j, ch in enumerate(s) if ch != " "]
        if conf >= 30 and len(glyphs) == len(chars):
            places += [(k, j) for j in chars]
            images += [_glyph(cc, parts, vertical) for parts in glyphs]
    if len(images) <= LOOK_ALIKES:
        return results, [s for s, _ in results]
    dist, near = cKDTree(np.array(images)).query(np.array(images), k=LOOK_ALIKES + 1)
    read = [results[k][0][j] for k, j in places]

    def nearest(n: int, ch: str) -> float:  # the distance to the nearest look-alike read as ch
        return next((d for d, m in zip(dist[n][1:], near[n][1:]) if read[m] == ch), np.inf)

    spread = {ch: float(np.median([nearest(n, ch) for n, c in enumerate(read) if c == ch])) for ch in set(read)}
    out, every = [list(s) for s, _ in results], [list(s) for s, _ in results]
    for n, (k, j) in enumerate(places):
        ch = read[n]
        alike = next((c for c in CONFUSED if ch in c), None)
        if alike is None or (k, j) in agreed:
            continue
        best, votes = Counter(read[m] for m in near[n][1:]).most_common(1)[0]
        # and the glyph looks like those of that kind: a rare 6 among many 0 is just as near them as to anything else
        if best != ch and best in alike and votes >= LOOK_ALIKES // 2 and nearest(n, best) <= SPREAD * spread[best]:
            every[k][j] = best
            if not careful or votes >= CLEAR:
                out[k][j] = best
    return [("".join(chars), conf) for chars, (_, conf) in zip(out, results)], ["".join(chars) for chars in every]


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
CLEAR = 12  # of the LOOK_ALIKES make a clear majority, which a few glyphs more or less elsewhere do not overturn
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


# A pipe size before its specification ("2" -AB12") or in a reducer ("4"x3""): there the mark after the number is an
# inch mark, whatever was read for it. A sheet with a few of these writes pipe sizes in inches, so there a nominal size
# with any mark on its own ("2'", "1/2°") is one too; other numbers keep theirs ("45°", feet and inches "12'-6"").
MARK = r"(?:''|[\"'’°])"
SIZED = re.compile(r"\d\s?" + MARK + r"\s?(?:-\s?[A-Z]{2}\d|[x/]\s?\d)")
SIZE = re.compile(r"(\d+(?:/\d+)?)\s?" + MARK + r"(?!\s?[CF]\b)(?!\s?-\s?\d)(?=(\s?(?:-\s?[A-Z]{2}\d|[x/]\s?\d))?)")
NPS = {"1/8", "1/4", "3/8", "1/2", "3/4", "1", "2", "3", "4", "5", "6", "8", "10", "12", "14", "16", "18", "20", "22",
       "24", "26", "28", "30", "32", "34", "36", "42", "48"}  # nominal pipe sizes in inches
KEYED = re.compile(r"\b[A-Z]{1,3}=[A-Z0-9]")  # a key and its value ("PN=16"), as attribute labels are written
COLON = re.compile(r"\b([A-Z]{1,3}):(?=[A-Z0-9])")  # ... and its equals sign read as a colon
EVIDENCE = (3, 2)  # texts that show the sheet writes inch sizes, key=value labels


def grammar(texts: list[Text]) -> list[Text]:
    """Marks the readers confuse, put right where the sheet itself shows how it writes them: inch marks read as a
    foot mark or a degree sign ("2°-AB12", "4'x3'"), and the equals sign of a key=value label read as a colon
    ("PN:16"). A rule applies only on a sheet with enough texts that show the notation (EVIDENCE), so the angles of a
    floor plan and the colons of a note stay as they are; temperatures ("60°C") always do."""
    inches = sum(bool(SIZED.search(t.text)) for t in texts) >= EVIDENCE[0]
    keyed = sum(bool(KEYED.search(t.text)) for t in texts) >= EVIDENCE[1]

    def inch(m: re.Match) -> str:
        return m.group(1) + '"' if m.group(2) or m.group(1) in NPS else m.group(0)

    out = []
    for t in texts:
        s = SIZE.sub(inch, t.text) if inches else t.text
        s = COLON.sub(r"\1=", s) if keyed else s
        out.append(t if s == t.text else replace(t, text=s))
    return out


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


def _oblique(cc: Components, h_groups: list[list[int]], v_groups: list[list[int]], rest: np.ndarray
             ) -> tuple[list[list[int]], list[list[int]], list[tuple[list[int], Tilted]]]:
    """Lines of characters at another angle than horizontal or vertical, such as street names and parcel numbers on a
    site plan. Up to DIAGONAL off an axis, the grouping along it links their characters already, as their boxes overlap
    across it; such a line is oblique where it runs further off the axis than its letters can tell (see _direction,
    _off). Nearer the diagonals it comes in pieces, which are grouped anew with the characters left over, along the two
    diagonals. An oblique line has at least three characters on a straight line (see _straight). Returns the
    horizontal and vertical lines left and the oblique ones, seen along them; a piece that joins no oblique line stays
    as it was."""
    lines, kept, pieces = [], ([], []), []
    for axis, groups in enumerate((h_groups, v_groups)):
        for g in groups:
            d = _direction(cc, g)  # None for two characters near a diagonal too
            off = None if d is None else _off(d, 90.0 * axis)
            if off is not None and off <= d[2]:
                kept[axis].append(g)
            elif off is not None and off < DIAGONAL and (t := _straight(cc, g, d[0])):
                lines.append((g, t))
            else:
                pieces.append((axis, g))
    pool = np.union1d(rest, [i for _, g in pieces for i in g]).astype(np.int64)
    for diagonal in (45.0, -45.0):
        if len(pool) < 3:
            break
        grouped, left = _group(Tilted(cc, pool, diagonal), pool, horizontal=True)
        for g in grouped:
            d = _direction(cc, g)
            if d is not None and _off(d) > d[2] and (t := _straight(cc, g, d[0])):
                lines.append((g, t))
            else:
                left = np.concatenate([left, g])
        pool = left.astype(np.int64)
    used = {i for g, _ in lines for i in g}
    for axis, g in pieces:
        if not used.intersection(g):
            kept[axis].append(g)
    return kept[0], kept[1], lines


def _direction(cc: Components, g: list[int]) -> tuple[float, float, float] | None:
    """The angle a line of characters runs at (-90 to 90 degrees, counter-clockwise): along the longer side of the
    smallest rectangle around its characters (see _characters), and along the straight line through their centres;
    and how far off an axis it may run although it lies along it (spotting.leeway). None where that rectangle is too
    square to tell. The two angles differ where a descender tilts the rectangle of a short word, or where a letter of
    a line nearby joined it, which tilts the rectangle but hardly the line through the centres."""
    chars = np.array(g)[_characters(np.maximum(cc.w[g], cc.h[g]).astype(float))]
    if not len(chars):  # two shapes of quite different size
        return None
    pts = np.vstack([np.argwhere(cc.lab[cc.y[i]:cc.y[i] + cc.h[i], cc.x[i]:cc.x[i] + cc.w[i]] == i + 1)
                     + (cc.y[i], cc.x[i]) for i in chars])
    box = cv2.boxPoints(cv2.minAreaRect(pts[:, ::-1].astype(np.float32)))
    a, b = box[1] - box[0], box[2] - box[1]
    along, across = (a, b) if np.hypot(*a) >= np.hypot(*b) else (b, a)
    if np.hypot(*along) < 1.25 * np.hypot(*across):
        return None
    angle, through = _angle(along[0], -along[1]), None
    if len(chars) >= 2:  # the main axis of the centres (y up, as the angles)
        centres = np.column_stack([cc.x[chars] + cc.w[chars] / 2, -(cc.y[chars] + cc.h[chars] / 2)])
        through = _angle(*np.linalg.svd(centres - centres.mean(0), full_matrices=False)[2][0])
    size = float(np.median(np.maximum(cc.w[chars], cc.h[chars])))
    return angle, angle if through is None else through, leeway(float(np.hypot(*along)), size)


def _angle(dx: float, dy: float) -> float:
    """The angle of a direction (y up), from -90 (excluded) to 90 degrees."""
    a = (math.degrees(math.atan2(dy, dx)) + 90) % 180 - 90
    return 90.0 if a == -90 else a


def _off(d: tuple[float, float, float], axis: float | None = None) -> float:
    """How far a line runs off the ``axis`` (0 or 90 degrees; None: the nearer one) by the nearer of its two angles
    (see _direction): it is oblique only where both say so."""
    def off(a: float) -> float:
        return min(abs(a), 90 - abs(a)) if axis is None else abs((a - axis + 90) % 180 - 90)

    return min(off(d[0]), off(d[1]))


def _straight(cc: Components, g: list[int], angle: float) -> Tilted | None:
    """``g`` seen along ``angle`` where it is a line of at least three characters whose larger ones are centred on a
    straight line, within STRAIGHT of their height (two lines of text side by side are not), and whose halves run
    along it too, as far as their letters tell (a level label that the first letters of an oblique text beside it
    joined bends at its end), else None."""
    if len(g) < 3:
        return None
    t = Tilted(cc, g, angle)
    members = np.array(g)
    chars = members[_characters(t.h[members])]
    if len(chars) < 3 or float(np.std(t.y[chars] + t.h[chars] / 2)) > STRAIGHT * float(np.median(t.h[chars])):
        return None
    order, n = chars[np.argsort(t.x[chars])], len(chars) // 2
    for half in ((order[:n], order[-n:]) if n >= 3 else ()):  # halves of fewer letters tell too little
        if (d := _direction(cc, half.tolist())) is not None and _off(d, angle) > d[2]:
            return None
    return t


def _characters(size: np.ndarray) -> np.ndarray:
    """Which components of a line are its characters, by their size (pixels): about the usual one, not the dots and
    dashes beside them, nor a character run into a stroke that hangs off the line (much larger). The usual size is
    that of the components not much smaller than most, so that a line of few characters with a hyphen and a comma
    has it too."""
    usual = float(np.median(size[size >= 0.5 * np.median(size)]))
    return (size >= 0.6 * usual) & (size <= 1.5 * usual)


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


def _tilted_crop(cc: Components, t: Tilted, g: list[int], turn: int = 0) -> np.ndarray:
    """Black-on-white crop of an oblique line's own pixels as _crop, turned to read from left to right (``turn``: the
    other way round) and scaled to CAP_PX."""
    x0, y0 = min(cc.x[i] for i in g), min(cc.y[i] for i in g)
    x1, y1 = max(cc.x[i] + cc.w[i] for i in g), max(cc.y[i] + cc.h[i] for i in g)
    lut = np.zeros(cc.n + 1, bool)
    lut[np.array(g) + 1] = True
    own = cv2.dilate(lut[cc.lab[y0:y1, x0:x1]].astype(np.uint8), np.ones((3, 3), np.uint8))
    img = np.where(own > 0, cc.gray[y0:y1, x0:x1], 255).astype(np.uint8)
    members = np.array(g)
    a0, c0 = float(t.x[members].min()) - 2, float(t.y[members].min()) - 2  # with the dilated rim
    a1, c1 = float((t.x + t.w)[members].max()) + 2, float((t.y + t.h)[members].max()) + 2
    # from the pixels of the crop (along and across the line) back to those of the image
    (ox, oy) = t.back(a0 + 0.5, c0 + 0.5)
    m = np.float32([[t.cos, t.sin, ox - 0.5 - x0], [-t.sin, t.cos, oy - 0.5 - y0]])
    img = cv2.warpAffine(img, m, (math.ceil(a1 - a0), math.ceil(c1 - c0)),
                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=255)
    if turn:
        img = cv2.rotate(img, cv2.ROTATE_180)
    top, base, _, _ = _metrics(t, g, False)
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


def _text(cc: Components | Tilted, g: list[int], vertical: bool, s: str, conf: float, ppm: float) -> Text:
    top, base, lo, hi = _metrics(cc, g, vertical)
    mid_across, mid_along = (top + base) / 2, (lo + hi) / 2
    if isinstance(cc, Tilted):  # an oblique line, measured along it
        (x, y), angle = cc.back(mid_along, mid_across), round(_degrees(cc.angle), 1)
    else:
        (x, y), angle = (mid_across, mid_along) if vertical else (mid_along, mid_across), 90.0 if vertical else 0.0
    return Text(x / ppm, y / ppm, (base - top) / ppm, s, angle, round(conf, 1),
                length=_length(lo, hi, base - top, s, ppm))


def _length(lo: float, hi: float, cap: float, text: str, ppm: float) -> float | None:
    """The length of a text line as printed, in mm, from the extent of its letters along the line (in pixels); none
    where they are too short for the text, as when letters that touch a line were not found as letters: fitted to
    that, the text would shrink to a fraction of its size."""
    return (hi - lo) / ppm if hi - lo >= NARROW * cap * sum(not ch.isspace() for ch in text) else None


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
