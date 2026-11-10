"""Strokes drawn by hand on a print, in black or grey ink: notes, revision clouds, freehand hatching, leaders. Colour
markups are told apart by their colour (markup.py); these by their shape and ink, with features that hold on any
kind of drawing:

- CAD draws exact straight lines, circular and elliptic arcs and smooth curves, joined at sharp corners. A hand bends
  all along and its bend changes: between its corners a stroke is none of these (HAND share of its length), or it
  turns so often that the pieces between its turns are too short to be anything (handwriting: SHORT share), and it
  has few straight pieces (STRAIGHT share at most).
- A revision cloud is a chain of arcs of about one size, all bulging outwards, meeting in points (CLOUD bumps).
- Pencil is lighter than print: the darkest part of its stroke stays grey (LIGHT); a light stroke needs less of the
  rest. A thin printed line is grey in a scan too, as it covers its pixels only in part: with the guard, a stroke is
  light against the print of its width on the same sheet (_print_levels), and what a mark through print leaves of the
  print, specks smaller than a mark, is no note (_pencil).

Only strokes larger than a few characters are judged (SIZE), not those in a row of characters of their height and not
those no taller across than a line of print (LINE), the words of a blurred photo whose letters ran together: printed
characters bend all along too and are left to the text recognition. Strokes that run across much of the sheet (FAR),
and large ones that are mostly straight, are the drawing's own network, and a mark drawn into the linework is part of
one: these are traced into strokes between their junctions and judged stroke by stroke, conservatively
(features.freehand_strokes). A leader from a mark found whole that ends in an arrowhead, and on a sheet with such
marks a stroke through a line of printed text (struck out), are marks too (features.leaders). The marks found are
kept apart from the drawing on their own layer, as they are, not deleted."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .markup import NOTE, _odd
from .model import Color, Markup, Polyline, Text

SIZE = 3.0  # characters: a stroke at least this long or wide is judged (a short word)
MARK = 6.0  # mm: and at least this, as specks of dirt on a poor scan can make the print look smaller than it is
LINE = 1.3  # characters: a stroke no wider across than this is a line of print whose letters ran together (blur)
ACROSS = 3.0  # mm: or no wider than this
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
SHADE = 0.2  # and lighter than this share of the way from print of its own width to paper: pencil is about half as
# dark as print of its width, both lightened alike where blur spreads a thin stroke over more pixels
WIDTHS = 12  # px: the print level is measured per stroke width up to this; wider strokes go with the widest
SAMPLES = 200  # pixels along the middle of strokes of a width that its print level needs (else the darkest print)
SPECK = 2  # value in the mask of find: a speck of print a mark leaves, no mark, but out of the drawing as before
KINDS = ("line", "arc", "curve", "hand", "short")
TURN = math.radians(35)  # a stroke leaving a junction within this of straight on from another one continues it
JUNCTION = 0.5  # mm beyond the widest stroke at a junction within which a traced skeleton bends towards it
JUDGED = 2.0  # mm: in a stroke traced through the linework, shorter parts between junctions are not judged
BAND = 1.0  # mm: a stroke traced through the linework that stays within this of a straight line is straight
BUMP = (0.8, 8.0, 0.25)  # mm: the bump of a cloud is a round piece of a stroke of this radius, within this of a circle
LINK = 3.0  # mm: the bumps of a cloud traced through the linework end within this of the next one (cut back by
# the margins of the junction they meet at), or a bump further where a line cut the bump between into short pieces
CAD = 4.0  # mm: a stroke the vectoriser draws in pieces this long on average (lines and arcs joined at tangents, a
# vessel with its heads) is CAD, though its pieces between corners are none of a line, an arc or an ellipse
LEADER = 4.0  # mm: a stroke that starts this close to a mark found and ends in an arrowhead is a leader from it
# (about a letter and a half: a leader starts beside its note)
BARB = (0.8, 6.0)  # mm: the barbs of an arrowhead drawn by hand are this long,
BARB_TURN = (math.radians(10), math.radians(60))  # and leave its tip back along the leader at this angle to it;
CUSP = math.radians(110)  # a stroke turning back this sharply runs on into a barb (over a millimetre either side)
STRUCK, SPAN, SIDE = 2.5, 0.85, 0.2  # a text struck out: a component at most STRUCK characters high across with a
# nearly straight stroke along SPAN of its length and SIDE of the rest of its ink (its letters) on either side of it,
REACH, THIN, COVER = 0.25, 0.3, 0.7  # reaching at least REACH characters away from it on either side (not the specks
# of a ragged line), the stroke at most THIN as wide as the letters are high across it (not a word whose letters ran
# together), and the letters along COVER of the stroke at least (a printed line through a text runs on beyond it)


def find(ink: np.ndarray, gray: np.ndarray, ppm: float, letter: float, strokes: bool = False,
         leaders: bool = False, guard: bool = False) -> np.ndarray:
    """The mask of the strokes drawn by hand among ``ink``, at ``ppm`` pixels per mm with characters ``letter``
    pixels high; ``gray`` the normalised image (paper 255). With ``strokes``, the components that are the drawing's
    own network (too large to be judged whole, or large and mostly straight), which a mark drawn into the linework is
    part of, are judged stroke by stroke (_strokes). With ``leaders``, leaders drawn from the marks found whole and
    strokes through printed text are marks too (_leaders, _strike_outs), on a sheet with marks found whole only: the
    strokes judged one by one in the linework are less sure, and a stroke through a text alone is too little to go
    by (a printed line between two rows of text looks the same). With ``guard``, a stroke is light against print of
    its width (_print_levels), and specks of print left of a mark are no mark (_pencil): SPECK in the mask, left out
    of the drawing as before, as a speck between the dashes of a line would join them."""
    out = np.zeros_like(ink)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    side, across = np.maximum(stats[1:, 2], stats[1:, 3]), np.minimum(stats[1:, 2], stats[1:, 3])
    long, far = side >= max(SIZE * letter, MARK * ppm), side > FAR * max(ink.shape)
    big = np.flatnonzero(long & ~far & (across > max(LINE * letter, ACROSS * ppm))) + 1
    network = list(np.flatnonzero(long & far) + 1) if strokes else []
    traced: dict[int, _Traced] = {}  # the components traced stroke by stroke, by label
    darkest = _print_level(gray, ink) if len(big) or network else 0.0
    levels = _print_levels(gray, ink, darkest) if guard and (len(big) or network) else None
    mark = (max(SIZE * letter, MARK * ppm), _odd(NOTE * ppm), levels) if guard else ()  # least size, reach of a mark
    for i in big:
        x, y, w, h = stats[i, :4]
        if _in_row(stats, i):
            continue
        mask = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        hand = _drawn_by_hand(mask, gray[y:y + h, x:x + w], ppm, darkest, levels)
        if hand:
            out[y:y + h, x:x + w] |= _pencil(mask, gray[y:y + h, x:x + w], darkest, *mark)
        elif hand is None and strokes:
            network.append(i)
    whole = (out == 1).astype(np.uint8)
    for i in network:
        x, y, w, h = stats[i, :4]
        mask = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        traced[i] = t = _traced(mask, ppm)
        _strokes(t, gray[y:y + h, x:x + w], ppm, letter, darkest, stats, (x, y), levels)
        out[y:y + h, x:x + w] |= _pixels(mask, t)
    if leaders and whole.any():
        out |= _leaders(ink, lab, stats, out, whole, traced, ppm)
        out |= _strike_outs(ink, lab, stats, out, ppm, letter)
    return out


@dataclass
class _Traced:
    """A component traced into strokes: its paths (vectorize.trace; pixels from the component's corner, each with its
    width), the strokes they make (_chains: the paths of each in order, and whether each runs backwards), the
    junctions or free ends at the ends of each path, the spurs, the margins of the junctions, and per point of each
    path whether it was drawn by hand."""
    paths: list[tuple[np.ndarray, float]]
    chains: list[list[tuple[int, bool]]]
    nodes: list[tuple]
    spurs: dict[int, tuple]
    margin: dict[tuple, float]
    hand: list[np.ndarray]

    def points(self, chain: list[tuple[int, bool]]) -> np.ndarray:
        """The points of a stroke in order."""
        return np.vstack([self.paths[k][0][::-1] if rev else self.paths[k][0] for k, rev in chain])


def _traced(mask: np.ndarray, ppm: float) -> _Traced:
    from .vectorize import trace

    # with a pixel of paper around it: a component that fills its box (a filled bar) has no paper to measure its width
    # to, and its width would be infinite
    paths = [(p - 1, w) for p, w in trace(np.pad(mask, 1), ppm) if len(p) >= 2]
    return _Traced(paths, *_chains(paths, ppm), [np.zeros(len(p), bool) for p, _ in paths])


def _strokes(t: _Traced, gray: np.ndarray, ppm: float, letter: float, darkest: float, stats: np.ndarray,
             at: tuple[int, int], levels: np.ndarray | None = None) -> None:
    """Marks the strokes drawn by hand into the linework of a component of the drawing's network (whose corner is
    pixel ``at`` of the sheet, whose components have ``stats``), traced into strokes between its junctions: a stroke is
    followed on through a junction where it runs on smoothly (a pen crossing a wall, _chains), and each stroke is
    judged by the rules of a whole mark (_judged); the bumps of a cloud, which meet in points, are found as a ring of
    them (_clouds)."""
    for chain in t.chains:
        parts = [_inner(t.paths[k][0][::-1] if rev else t.paths[k][0], t.margin,
                        t.nodes[k][::-1] if rev else t.nodes[k]) for k, rev in chain]
        width = float(np.median([t.paths[k][1] for k, _ in chain]))
        if _judged(parts, width, gray, ppm, letter, darkest, stats, at, levels):
            for k, _ in chain:
                t.hand[k][:] = True
    for k, a, b in _clouds(t.paths, t.chains, t.nodes, t.margin, ppm):
        t.hand[k][a:b + 1] = True


def _pixels(mask: np.ndarray, t: _Traced) -> np.ndarray:
    """The pixels of the strokes of a traced component marked as drawn by hand, with the spurs at their junctions (the
    tip of a pen's turn, such as a cloud's point): each at its width, leaving the print it crosses."""
    if not any(h.any() for h in t.hand):
        return np.zeros_like(mask)
    marked = {n for k, h in enumerate(t.hand) if h.any() for n in t.nodes[k]}
    hand = [h | (t.spurs.get(k) in marked) for k, h in enumerate(t.hand)]
    taken, kept = np.zeros_like(mask), np.zeros_like(mask)
    for (p, w), h in zip(t.paths, hand):  # each at its width and a pixel more either side: the blurred edge
        for img, sel in ((taken, h), (kept, ~h)):
            edges = np.flatnonzero(np.diff(np.concatenate([[0], sel.astype(np.int8), [0]])))
            runs = [np.round(p[a:b]).astype(np.int32) for a, b in zip(edges[::2], edges[1::2])]
            cv2.polylines(img, runs, False, 1, round(w) + 2)
    return mask & taken & (1 - kept)


def _leaders(ink: np.ndarray, lab: np.ndarray, stats: np.ndarray, hand: np.ndarray, whole: np.ndarray,
             traced: dict[int, _Traced], ppm: float) -> np.ndarray:
    """The leaders drawn from the marks found whole (``whole``; ``hand`` all marks found): strokes of the ink not
    taken yet that start at such a mark (within LEADER) and end in an arrowhead (_arrowhead), with its barbs. A stroke
    that ends on the drawing without one is no clear leader: a hand bends as smoothly as a printed spline there.
    ``traced``: the components traced already."""
    reach = 2 * round(LEADER * ppm) + 1
    near = cv2.dilate(whole, np.ones((reach, reach), np.uint8))
    out = np.zeros_like(hand)
    for i in np.unique(lab[(near > 0) & (hand == 0) & (ink > 0)]):
        x, y, w, h = stats[i, :4]
        if i == 0 or max(w, h) < MARK * ppm or i not in traced and max(w, h) > FAR * max(ink.shape):
            continue  # the drawing's network is traced only stroke by stroke (_strokes)
        mask = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        t = traced.get(i) or _traced(mask, ppm)
        degree = Counter(n for ends in t.nodes for n in ends)
        found = [k for chain in t.chains
                 for k in _leader(t, chain, near[y:y + h, x:x + w], hand[y:y + h, x:x + w], degree, ppm)]
        for k in found:
            t.hand[k][:] = True
        if found:
            out[y:y + h, x:x + w] |= _pixels(mask, t)
    return out


def _leader(t: _Traced, chain: list[tuple[int, bool]], near: np.ndarray, hand: np.ndarray, degree: Counter,
            ppm: float) -> list[int]:
    """The paths of a leader if a stroke of a traced component is one (see _leaders), with the barbs of its
    arrowhead; else none. ``near``: within LEADER of a mark found, ``hand``: the marks found, ``degree``: the number of
    paths ending at each junction."""
    pts = t.points(chain)
    if _length(pts) < MARK * ppm or _at(hand, pts).mean() >= 0.5:  # too short, or a mark already
        return []
    starts = _at(near, pts[[0, -1]])
    for far in (1, 0):
        if not starts[1 - far]:
            continue
        run = chain if far else [(k, not rev) for k, rev in chain[::-1]]  # from the mark to the far end
        if barbs := _arrowhead(t, run, degree, ppm):
            return [k for k, _ in chain] + barbs
    return []


def _arrowhead(t: _Traced, run: list[tuple[int, bool]], degree: Counter, ppm: float) -> list[int]:
    """The barbs of the arrowhead a stroke (its paths in order, from its start) ends in: one on either side of it, of
    BARB length, leaving its tip back along it within BARB_TURN; or none. A barb is a path that ends freely with its
    other end at the tip, or the end of the stroke itself where it turns back at its tip (CUSP): the skeleton of a
    pointed arrowhead runs on into one of its barbs or splits it off."""
    pts = _resampled(t.points(run), ppm)
    k = max(1, round(1.0 / STEP))
    if len(pts) < 2 * k + 1:
        return []
    tip, sides = len(pts) - 1, {}
    if (cusp := _cusp(pts, k)) is not None:  # the stroke runs on into a barb
        tip = cusp
        sides[_side(pts[tip - k] - pts[tip], pts[min(len(pts) - 1, tip + 2 * k)] - pts[tip])] = None
    u, at = pts[tip - k] - pts[tip], pts[tip] * ppm  # back along the stroke from its tip; the tip in pixels
    on = {k for k, _ in run}
    for j, (b, w) in enumerate(t.paths):
        if j in on or not BARB[0] * ppm <= _length(b) <= BARB[1] * ppm:
            continue
        for e in (0, 1):  # the barb's end at the tip, its other end free
            if np.linalg.norm(b[-e] - at) <= max(2 * w, ppm) and degree[t.nodes[j][1 - e]] == 1:
                v = _away(b, e, w, ppm)
                if BARB_TURN[0] <= _turn(u, v) <= BARB_TURN[1]:
                    sides.setdefault(_side(u, v), j)
    return [j for j in sides.values() if j is not None] if len(sides) == 2 else []


def _cusp(pts: np.ndarray, k: int) -> int | None:
    """The sample of a stroke (points STEP mm apart, in mm) where it turns back by at least CUSP within its last
    BARB mm, leaving a barb of BARB length after it; None if it does not."""
    turns = [(_turn(pts[i] - pts[i - k], pts[i + k] - pts[i]), i)
             for i in range(max(k, len(pts) - 1 - round(BARB[1] / STEP)), len(pts) - max(k, round(BARB[0] / STEP)))]
    turn, i = max(turns, default=(0.0, None))
    return i if turn >= CUSP else None


def _turn(a: np.ndarray, b: np.ndarray) -> float:
    """The angle (radians) between two directions."""
    return math.acos(min(1.0, max(-1.0, float(a @ b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9))))


def _side(u: np.ndarray, v: np.ndarray) -> bool:
    """Whether ``v`` lies to the left of ``u`` (on screen, y down: clockwise)."""
    return bool(u[0] * v[1] - u[1] * v[0] > 0)


def _strike_outs(ink: np.ndarray, lab: np.ndarray, stats: np.ndarray, hand: np.ndarray, ppm: float,
                 letter: float) -> np.ndarray:
    """Strokes through a line of printed text, the text struck out: in a component of the size of a text line (at
    least SIZE characters long, at most STRUCK high across and three times as long), a nearly straight stroke through
    the middle of its letters from one end to the other, thin beside them (_strike; no linework runs on beyond them).
    The stroke goes, the letters stay."""
    out = np.zeros_like(ink)
    side, across = np.maximum(stats[:, 2], stats[:, 3]), np.minimum(stats[:, 2], stats[:, 3])
    for i in np.flatnonzero((side >= max(SIZE * letter, MARK * ppm)) & (across <= STRUCK * letter)
                            & (side >= 3 * across)):
        x, y, w, h = stats[i, :4]
        if i == 0 or hand[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i].any():
            continue
        mask = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        t = _traced(mask, ppm)
        for k in _strike(t, mask, side[i], letter, ppm):
            t.hand[k][:] = True
        out[y:y + h, x:x + w] |= _pixels(mask, t)
    return out


def _strike(t: _Traced, mask: np.ndarray, side: float, letter: float, ppm: float) -> list[int]:
    """The paths of a stroke that strikes out the letters of a component the size of a text line (``mask``, ``side``
    pixels long), or none: the paths along the line of its longest straight stroke (at least two characters long),
    within a quarter of BAND and half a pen's width of it, that together run along SPAN of the component with no gap
    of more than two characters, with SIDE of the rest of its ink (its letters, which the stroke cuts into pieces) on
    either side of the line, reaching REACH characters away from it and along COVER of it, and no wider than THIN of
    the letters across it.
    Where it crosses a letter, the stroke is traced in short pieces that do not join into one stroke (_chains), so
    they are gathered along the line, and the line is fitted again to all of them."""
    band = BAND * ppm / 4 + float(np.median([w for _, w in t.paths])) / 2  # its skeleton, wobbling a little
    seed = max((p for p in map(t.points, t.chains) if _straight(p, band)), key=_length, default=None)
    if seed is None or _length(seed) < 2 * letter:
        return []
    on: list[int] = []
    for pts in (seed, None):
        pts = np.vstack([t.paths[k][0] for k in on]) if pts is None else pts
        c = pts.mean(axis=0)
        axis, normal = np.linalg.svd(pts - c, full_matrices=False)[2]
        on = [k for k, (p, _) in enumerate(t.paths) if np.abs((p - c) @ normal).max() <= band]
        if not on:  # a stroke straight within the band whose paths each wander off the line fitted to all
            return []
    along = np.sort(np.concatenate([(t.paths[k][0] - c) @ axis for k in on]))
    if along[-1] - along[0] < SPAN * side or np.diff(along).max(initial=0) > 2 * letter:
        return []
    rest = mask.copy()
    for k in on:
        cv2.polylines(rest, [np.round(t.paths[k][0]).astype(np.int32)], False, 0, round(t.paths[k][1]) + 2)
    ys, xs = np.nonzero(rest)
    offside = (np.column_stack([xs, ys]) - c) @ normal
    above, below = offside[offside > 0], -offside[offside < 0]
    if min(len(above), len(below)) < SIDE * max(len(xs), 1):  # an underline: letters on one side
        return []
    reach, width = min(above.max(), below.max()), float(np.median([t.paths[k][1] for k in on]))
    letters = (np.column_stack([xs, ys]) - c) @ axis
    cover = (letters.max() - letters.min()) / max(along[-1] - along[0], 1.0)
    return on if reach >= REACH * letter and width <= THIN * (above.max() + below.max()) and cover >= COVER else []


def _inner(p: np.ndarray, margin: dict, ends: tuple) -> np.ndarray:
    """A path without its ends at junctions (``margin`` of each, see _chains), where the skeleton of a stroke bends
    towards the junction's centre: a straight pen stroke across a wall would wobble there."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    keep = (s >= margin.get(ends[0], 0.0)) & (s <= s[-1] - margin.get(ends[1], 0.0))
    return p[keep]


def _judged(parts: list[np.ndarray], width: float, gray: np.ndarray, ppm: float, letter: float, darkest: float,
            stats: np.ndarray, at: tuple[int, int], levels: np.ndarray | None = None) -> bool:
    """Whether a stroke traced in the linework (its paths' ``parts`` away from junctions, in pixels, ``width`` wide)
    was drawn by hand: the rules of a whole mark (_hand) on its parts of at least JUDGED (shorter ones lie between
    junctions, in symbols and letters), and conservatively so, as it is part of the print: as large as a whole mark
    (SIZE, MARK, not as thin across as a line of text, not in a row of characters), not straight (BAND, STRAIGHT) and
    not drawn by the vectoriser in long pieces, as lines and arcs joined at tangents are (CAD)."""
    from .vectorize import _pieces as drawn  # the lines and arcs the vectoriser would draw a stroke with

    parts = [q for q in parts if len(q) >= 2 and _length(q) >= JUDGED * ppm]
    if not parts:
        return False
    pts = np.vstack(parts)
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    w, h = x1 - x0, y1 - y0
    if max(w, h) < max(SIZE * letter, MARK * ppm) or min(w, h) <= max(LINE * letter, ACROSS * ppm):
        return False
    if _row(stats, x0 + at[0], y0 + at[1], w, h) or _straight(pts, BAND * ppm):
        return False
    kinds = dict.fromkeys(KINDS, 0.0)
    for q in parts:
        _pieces(q, ppm, max(FIT, width / ppm / 4, 1.5 / ppm), kinds)
    total = sum(kinds.values())
    if total < MARK or kinds["line"] > STRAIGHT * total:
        return False
    if total >= CAD * sum(len(drawn(q, False, ppm, width)) for q in parts):
        return False
    ix = np.clip(np.round(pts).astype(int), 0, [gray.shape[1] - 1, gray.shape[0] - 1])
    return _hand(kinds, _light(gray[ix[:, 1], ix[:, 0]], width, darkest, levels))


def _clouds(paths: list[tuple[np.ndarray, float]], chains: list[list[tuple[int, bool]]], nodes: list[tuple],
            margin: dict, ppm: float) -> list[tuple[int, int, int]]:
    """The bumps of revision clouds drawn into the linework, as (path, its first and last point): round strokes or
    pieces of a stroke between its corners (BUMP; a bump that a line crosses is one stroke, _chains), the end of each
    meeting the end of the next (LINK; in a point, where the pen turned, or where both touch a line, or a bump further
    where a line cut the one between into pieces too short to judge) as neighbours in a cloud do (_neighbours), at
    least CLOUD of them in a row."""
    bumps = []  # its points [(path, first, last)], centre, radius, ends and middle in mm
    for chain in chains:
        if not BUMP[0] * 2 <= sum(_length(paths[k][0]) for k, _ in chain) / ppm <= 2 * math.pi * BUMP[1]:
            continue
        parts = []
        for k, rev in chain:
            p = paths[k][0][::-1] if rev else paths[k][0]
            q = _inner(p, margin, nodes[k][::-1] if rev else nodes[k])
            if len(q) >= 2:
                parts.append((k, rev, q, _resampled(q, ppm)))
        if not parts or _straight(np.vstack([q for _, _, q, _ in parts]), 0.3 * ppm):
            continue
        pts = np.vstack([r for *_, r in parts])
        if whole := _bump(pts):  # a bump between two points or junctions
            bumps.append(([(k, 0, len(paths[k][0]) - 1) for k, _ in chain], *whole, pts[0], pts[-1],
                          pts[len(pts) // 2]))
            continue
        for k, rev, q, r in parts:  # or the bumps between the corners of a stroke
            s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]) / ppm
            for a, b in _runs(r):
                if fit := _bump(r[a:b]):
                    first, last = (int(np.searchsorted(s, v)) for v in (a * STEP, (b - 1) * STEP))
                    first, last = _on_path(paths[k][0], q, rev, first, min(last, len(q) - 1))
                    bumps.append(([(k, first, last)], *fit, r[a], r[b - 1], r[(a + b) // 2]))
    if len(bumps) < CLOUD:
        return []
    root = list(range(len(bumps)))

    def find(i: int) -> int:
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    ends = np.array([e for b in bumps for e in (b[3], b[4])])
    for i, j in cKDTree(ends).query_pairs(LINK + 2 * max(b[2] for b in bumps)):  # i < j: ends of two bumps, or of one
        a, b = bumps[i // 2], bumps[j // 2]
        if i // 2 < j // 2 and np.linalg.norm(ends[i] - ends[j]) <= LINK + 2 * max(a[2], b[2]) and _neighbours(a, b):
            root[find(i // 2)] = find(j // 2)
    groups: dict[int, list] = {}
    for i, b in enumerate(bumps):
        groups.setdefault(find(i), []).append(b)
    return [m for group in groups.values() if len(group) >= CLOUD for b in group for m in b[0]]


def _on_path(p: np.ndarray, q: np.ndarray, rev: bool, first: int, last: int) -> tuple[int, int]:
    """Points ``first`` to ``last`` of ``q``, a part of path ``p`` (run backwards if ``rev``), as points of ``p``."""
    at = int(np.flatnonzero(np.all((p[::-1] if rev else p) == q[0], axis=1))[0])
    first, last = at + first, at + last
    return (len(p) - 1 - last, len(p) - 1 - first) if rev else (first, last)


def _neighbours(a: tuple, b: tuple) -> bool:
    """Whether two bumps whose ends meet are neighbours in a cloud: about one size (within a factor 2), with centres
    of their own (not two arcs of one circle that a line cut apart), bulging to the same side of the line through
    their centres (outwards, not in a wave)."""
    (ca, ra, ma), (cb, rb, mb) = (a[1], a[2], a[5]), (b[1], b[2], b[5])
    along = cb - ca
    if not 0.5 <= ra / rb <= 2 or np.linalg.norm(along) < 0.5 * min(ra, rb):
        return False
    def side(v: np.ndarray) -> float:
        return float(along[0] * v[1] - along[1] * v[0])

    return side(ma - ca) * side(mb - cb) > 0


def _bump(arc: np.ndarray) -> tuple[np.ndarray, float] | None:
    """The centre and radius of a piece of a stroke (mm) that may be the bump of a cloud: within BUMP of a circle of
    BUMP radius, round over one to five radians of it."""
    if len(arc) < 4 or (fit := _circle(arc)) is None:
        return None
    centre, r, rms = fit
    sweep = _length(arc) / r
    return (np.array(centre), r) if BUMP[0] <= r <= BUMP[1] and rms <= BUMP[2] and 1.0 <= sweep <= 5.0 else None


def _circle(pts: np.ndarray) -> tuple[tuple[float, float], float, float] | None:
    """The circle through points (centre, radius, RMS distance from it), fitted algebraically."""
    A = np.column_stack([2 * pts, np.ones(len(pts))])
    (cx, cy, k), *_ = np.linalg.lstsq(A, (pts ** 2).sum(axis=1), rcond=None)
    r2 = k + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    return (float(cx), float(cy)), r, float(np.sqrt(np.mean((np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) ** 2)))


def _chains(paths: list[tuple[np.ndarray, float]], ppm: float
            ) -> tuple[list[list[tuple[int, bool]]], list[tuple], dict[int, tuple], dict[tuple, float]]:
    """The traced paths of a component joined into strokes, each as its paths in order (index, reversed): where
    paths meet at a junction, one that leaves it within TURN of straight on from another continues it, as a pen
    stroke across a line does; one that turns there, as at a corner or a branch, ends. Junctions joined by a path no
    longer than a stroke is wide are one: two strokes
    that cross at a narrow angle meet in two. Also returns the junctions (or free ends) at the ends of each path, the
    spurs (short paths that end freely) with their junction and the margin of each junction within which the skeleton
    bends towards it: the width of the widest stroke there and JUNCTION more."""
    def key(q: np.ndarray) -> tuple[float, float]:
        return round(float(q[0]), 2), round(float(q[1]), 2)

    count: dict = {}
    for p, _ in paths:
        for q in (p[0], p[-1]):
            count[key(q)] = count.get(key(q), 0) + 1
    root = {k: k for k in count}

    def find(k):
        while root[k] != k:
            root[k] = root[root[k]]
            k = root[k]
        return k

    joints = set()
    for i, (p, w) in enumerate(paths):
        a, b = key(p[0]), key(p[-1])
        if a != b and count[a] > 1 and count[b] > 1 and _length(p) <= max(1.5 * w, 0.5 * ppm):
            root[find(a)] = find(b)
            joints.add(i)
    nodes = [(find(key(p[0])), find(key(p[-1]))) for p, _ in paths]
    spurs = {i: nodes[i][e] for i, (p, w) in enumerate(paths) for e in (0, 1)
             if count[key(p[-e])] > 1 and count[key(p[e - 1])] == 1 and _length(p) <= max(2 * w, ppm)}
    ends: dict = {}
    margin: dict = {}
    for i, (p, w) in enumerate(paths):
        for q, n in zip((p[0], p[-1]), nodes[i]):
            if count[key(q)] > 1 or n != key(q):
                margin[n] = max(margin.get(n, 0.0), w + JUNCTION * ppm)
        if i in joints or np.array_equal(p[0], p[-1]):
            continue
        for e in (0, 1):
            ends.setdefault(nodes[i][e], []).append((i, e, _away(p, e, w, ppm)))
    partner = {}
    for here in ends.values():
        turns = sorted((math.acos(min(1.0, max(-1.0, -float(np.dot(u, v))))), (i, e), (j, f))
                       for n, (i, e, u) in enumerate(here) for j, f, v in here[n + 1:])
        for turn, a, b in turns:
            if turn <= TURN and a not in partner and b not in partner:
                partner[a], partner[b] = b, a
    chains, seen = [], set()
    for i in range(len(paths)):
        if i in seen or i in joints:
            continue
        seen.add(i)
        chain = [(i, False)]
        for e, step in ((1, 1), (0, -1)):  # on from its end, then back from its start
            end = (i, e)
            while end in partner and partner[end][0] not in seen:
                j, f = partner[end]
                seen.add(j)
                link = (j, f == 1) if step == 1 else (j, f == 0)  # entered at its end: run backwards
                chain = chain + [link] if step == 1 else [link] + chain
                end = (j, 1 - f)
        chains.append(chain)
    return chains, nodes, spurs, margin


def _away(p: np.ndarray, e: int, width: float, ppm: float) -> np.ndarray:
    """The direction in which a path leaves its end ``e`` (0 its start, 1 its end), measured beyond the junction it
    may start at, where the skeleton bends towards the junction's centre."""
    q = p if e == 0 else p[::-1]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    near = min(max(width, 0.4 * ppm), s[-1] / 2)
    far = min(near + max(2 * width, ppm), s[-1])
    v = np.array([np.interp(far, s, q[:, 0]) - np.interp(near, s, q[:, 0]),
                  np.interp(far, s, q[:, 1]) - np.interp(near, s, q[:, 1])])
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _at(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Whether ``mask`` is set at points (x, y pixels)."""
    ix = np.clip(np.round(pts).astype(int), 0, [mask.shape[1] - 1, mask.shape[0] - 1])
    return mask[ix[:, 1], ix[:, 0]] > 0


def _straight(pts: np.ndarray, band: float) -> bool:
    """Whether points lie within ``band`` of a straight line (their main axis)."""
    c = pts - pts.mean(axis=0)
    axis = np.linalg.svd(c, full_matrices=False)[2][1]  # the normal of the main axis
    return float(np.abs(c @ axis).max()) <= band


def _pencil(mask: np.ndarray, gray: np.ndarray, darkest: float, size: float = 0.0, reach: int = 1,
            levels: np.ndarray | None = None) -> np.ndarray:
    """Of a mark, what is not print: where a pencil line runs through printed text or lines (a text struck out), the
    print stays (the dark ink and its edge) and only the light strokes go. A mark in black is all of it. With a
    ``size`` (pixels), only groups of light strokes (within ``reach`` pixels of each other) at least that long or wide
    go, and with ``levels`` only those light against print of their width (_light): the light edges of printed letters
    fused with a printed circle and the marks of a thin printed line are specks each smaller than a mark (SPECK), and
    a thin printed line beside thicker ones is as grey as print of its width (back to the drawing)."""
    light = (gray > darkest + LIGHT * (255 - darkest)) & (mask > 0)
    dark = (mask > 0) & ~light
    if light.sum() < 0.2 * mask.sum():
        return mask
    if dark.sum() < 0.2 * mask.sum():
        return mask if levels is None or _light(gray[mask > 0], _width(mask), darkest, levels) else mask * 0
    kept = (light & (cv2.dilate(dark.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)).astype(np.uint8)
    if not size:
        return kept
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(kept, np.ones((reach, reach), np.uint8)))
    out = np.zeros_like(kept)
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        part = kept[y:y + h, x:x + w] * (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        if max(w, h) - (reach - 1) < size:
            out[y:y + h, x:x + w] |= part * SPECK
        elif levels is None or _light(gray[y:y + h, x:x + w][part > 0], _width(part), darkest, levels):
            out[y:y + h, x:x + w] |= part
    return out


def _light(values: np.ndarray, width: float, darkest: float, levels: np.ndarray | None) -> bool:
    """Whether ink of the grey ``values`` (a stroke's pixels, ``width`` pixels wide) is pencil: its darkest tenth is
    lighter than LIGHT of the way from the sheet's darkest print to paper, and with ``levels`` (_print_levels) also
    SHADE of the way from print of that width to paper. A thin printed line, as grey as print of its width, is none."""
    core = float(np.percentile(values, 10))
    if levels is None or core <= darkest + LIGHT * (255 - darkest):
        return core > darkest + LIGHT * (255 - darkest)
    level = float(levels[min(max(round(width), 0), WIDTHS)])
    return core > level + SHADE * (255 - level)


def _print_levels(gray: np.ndarray, ink: np.ndarray, darkest: float) -> np.ndarray:
    """How dark the print of each stroke width is on the sheet (by the width in pixels, up to WIDTHS): the median grey
    along the middle of its strokes, and never darker than its darkest print. A thin printed line covers its pixels
    only in part, so it is grey in a scan, as light as pencil of a usual width can be; pencil is lighter than print of
    its own width. A width with too little ink (SAMPLES) has the darkest print."""
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    middle = (ink > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)))  # along the middle of each stroke
    widths, values = np.minimum(np.round(2 * dist[middle]).astype(int), WIDTHS), gray[middle]
    levels = np.full(WIDTHS + 1, darkest)
    for w in range(WIDTHS + 1):
        if len(sel := values[widths == w]) >= SAMPLES:
            levels[w] = max(darkest, float(np.median(sel)))
    return levels


def _width(mask: np.ndarray) -> float:
    """The width of a component's strokes in pixels, measured along their middle as _print_levels measures it."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    middle = (mask > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)))
    return float(np.median(np.round(2 * dist[middle]))) if middle.any() else 0.0


def _in_row(stats: np.ndarray, i: int) -> bool:
    """Whether component i stands in a row with ROW components of about its height, as a character of a text line:
    its height within a third, beside it within its height, on about the same line."""
    return _row(stats, *stats[i, :4])


def _row(stats: np.ndarray, x: float, y: float, w: float, h: float) -> bool:
    """Whether the box x, y, w, h stands in a row with ROW components of about its height (see _in_row)."""
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


def _drawn_by_hand(mask: np.ndarray, gray: np.ndarray, ppm: float, darkest: float,
                   levels: np.ndarray | None = None) -> bool | None:
    """Whether a component was drawn by hand (_hand); None where it is part of the drawing's network, large and
    mostly straight, whose marks are judged stroke by stroke if at all (_strokes). ``levels``: see _light."""
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
    if kinds["hand"] < HAND * total and max(mask.shape) > WIDE * ppm and kinds["line"] > NETWORK * total:
        return None  # a mark drawn into linework: a part of it, not all
    light = _light(gray[mask > 0], _width(mask) if levels is not None else 0.0, darkest, levels)
    return _cloud(arcs) or _hand(kinds, light)


def _hand(kinds: dict[str, float], light: bool) -> bool:
    """The rules of a mark by hand on the lengths of a stroke's pieces by ``kinds`` (_pieces) and whether its ink is
    ``light`` (pencil)."""
    total = sum(kinds.values())
    if total < 2 * PIECE:
        return False
    hand, loose, straight = kinds["hand"] / total, (kinds["hand"] + kinds["short"]) / total, kinds["line"] / total
    bent = (kinds["hand"] + kinds["arc"]) / total
    if hand >= HAND or loose >= SHORT and straight <= STRAIGHT and bent >= BENT:
        return True
    # pencil; a thin printed line blurred in a poor scan is light too, but it is straight
    return light and loose >= SHORT / 2 and bent >= BENT


def _pieces(path: np.ndarray, ppm: float, tol: float, kinds: dict[str, float]) -> list[tuple[float, int]]:
    """Adds the lengths of a traced stroke's pieces between corners to ``kinds`` by what they are (_fit; "short" for
    the corners and the pieces under PIECE mm) and returns its arcs in order: (radius in mm, the side they bulge to)."""
    pts, stroke = _resampled(path, ppm), _length(path) / ppm
    if len(pts) < 4 * max(1, round(CORNER_SPAN / 2 / STEP)) + 2:
        kinds["short"] += stroke
        return []
    arcs, judged = [], 0.0
    for a, b in _runs(pts):
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
    kinds["short"] += max(0.0, stroke - judged)
    return arcs


def _resampled(path: np.ndarray, ppm: float) -> np.ndarray:
    """A traced stroke (pixels) as points STEP mm apart along it, in mm."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]) / ppm
    t = np.arange(0, s[-1], STEP)
    return np.column_stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])]) / ppm


def _runs(pts: np.ndarray) -> list[tuple[int, int]]:
    """The runs of a resampled stroke between its corners (_pieces), as (first, end) sample indices."""
    k = max(1, round(CORNER_SPAN / 2 / STEP))
    if len(pts) < 4 * k + 2:
        return []
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
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


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
