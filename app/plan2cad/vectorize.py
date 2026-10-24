"""Turns the binary ink image (pixels) into lines, arcs, circles and filled areas (sheet mm)."""

from __future__ import annotations

import math
from collections import Counter

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .model import Arc, Circle, Line, Polygon

AXIS_SNAP = math.radians(2.0)


def fills(ink: np.ndarray, ppm: float) -> tuple[list[Polygon], np.ndarray]:
    """Solid areas (e.g. concrete walls, chimneys): strokes thicker than 1.5 mm and at least
    3.5 mm across, so blurred small text and arrowheads stay strokes."""
    r = max(2.0, 0.75 * ppm)
    core = (cv2.distanceTransform(ink, cv2.DIST_L2, 3) > r).astype(np.uint8)
    keep = np.zeros_like(ink)
    if not core.any():
        return [], keep
    grown = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(r) + 3,) * 2)) & ink
    polys = []
    for c in cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
        if min(cv2.boundingRect(c)[2:]) >= 3.5 * ppm:
            pts = cv2.approxPolyDP(c, max(1.0, 0.2 * ppm), True)[:, 0, :] / ppm
            polys.append(Polygon([(float(x), float(y)) for x, y in pts]))
            cv2.drawContours(keep, [c], -1, 1, cv2.FILLED)
    return polys, keep & grown


def axis_lines(ink: np.ndarray, ppm: float) -> tuple[list[Line], np.ndarray, np.ndarray]:
    """Horizontal and vertical strokes longer than 6 mm, found with a morphological opening.

    Text, symbols and curves have no straight runs that long, so they stay in the remainder. Returns
    the lines, the mask of their pixels and, within it, the pixels of strokes that cross or meet a
    line (the side of a valve on a pipe): the curve tracing gets those back so that the crossing
    stroke is not cut in two, while text recognition sees them cut."""
    length = max(12, round(6 * ppm))
    half = cv2.distanceTransform(ink, cv2.DIST_L2, 3)  # half the stroke width at the centre line
    out, masks = [], []
    for horizontal in (True, False):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1) if horizontal else (1, length))
        lines, mask = _fit_components(cv2.morphologyEx(ink, cv2.MORPH_OPEN, k), half, ppm, horizontal)
        out += lines
        masks.append(mask)
    # strokes running across a line, clearly longer than a line is thick
    across = max(5, round(1.2 * ppm), round(2.5 * ppm * float(np.median([ln.width for ln in out] or [0]))))
    h, v = masks
    runs_h, runs_v = (cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
                      for k in ((across, 1), (1, across)))
    used = h | v
    return out, used, used & (1 - ((h & (1 - runs_v)) | (v & (1 - runs_h)) | (h & v)))


def _fit_components(mask: np.ndarray, half: np.ndarray, ppm: float, horizontal: bool) -> tuple[list[Line], np.ndarray]:
    """Fits a line to each straight run and returns the lines with the mask of their strokes. A run
    much thinner than its stroke whose stroke goes on beyond both ends is the flat top of a curve
    (e.g. a vessel head): it is left to the curve tracing."""
    n, lab = cv2.connectedComponents(mask, connectivity=8)
    if n <= 1:
        return [], np.zeros_like(mask)
    ys, xs = np.nonzero(lab)
    lb = lab[ys, xs]
    along, across = (xs, ys) if horizontal else (ys, xs)
    along, across = along.astype(np.float64), across.astype(np.float64)
    idx = np.arange(1, n)
    cnt = np.bincount(lb, minlength=n)[1:]
    lo = ndimage.minimum(along, lb, idx)
    hi = ndimage.maximum(along, lb, idx)
    ma = np.bincount(lb, along, n)[1:] / cnt
    mc = np.bincount(lb, across, n)[1:] / cnt
    saa = np.bincount(lb, along * along, n)[1:] / cnt - ma ** 2
    sac = np.bincount(lb, along * across, n)[1:] / cnt - ma * mc
    slope = sac / np.maximum(saa, 1e-9)
    width = cnt / (hi - lo + 1)
    stroke = np.maximum(2 * ndimage.maximum(half[ys, xs], lb, idx) - 1, 1)
    lines, keep = [], np.zeros(n, np.uint8)
    for i, a0, a1, m_a, m_c, k, w, sw in zip(idx, lo, hi, ma, mc, slope, width, stroke):
        a0, a1 = a0 + w / 2, a1 - w / 2  # round pen caps reach half a stroke beyond the end points
        if a1 <= a0 or w < 0.6 * sw and all(_inked(half, a, m_c, sw + ppm, horizontal) for a in
                                            (a0 - ppm, a0 - 2 * ppm, a1 + ppm, a1 + 2 * ppm)):
            continue
        keep[i] = 1
        c0, c1 = (m_c, m_c) if abs(k) < math.tan(AXIS_SNAP) else (m_c + k * (a0 - m_a), m_c + k * (a1 - m_a))
        x1, y1, x2, y2 = (a0, c0, a1, c1) if horizontal else (c0, a0, c1, a1)
        lines.append(Line(x1 / ppm, y1 / ppm, x2 / ppm, y2 / ppm, float(w / ppm)))
    return lines, keep[lab]


def _inked(half: np.ndarray, along: float, across: float, reach: float, horizontal: bool) -> bool:
    """Any ink within ``reach`` pixels across the run direction at position ``along``."""
    a, c0, c1 = round(along), max(0, round(across - reach)), round(across + reach) + 1
    if not 0 <= a < half.shape[1 if horizontal else 0]:
        return False
    return bool((half[c0:c1, a] if horizontal else half[a, c0:c1]).any())


JOINT = 8.0  # mm: the largest stroke between two line ends that is kept from text recognition


def joints(rem: np.ndarray, lines: list[Line], ppm: float) -> np.ndarray:
    """The mask of the small strokes in ``rem`` that join the ends of two of the axis ``lines``: the round ends of a
    line-number box or a rounded corner. Without their lines they look like brackets, and text recognition would
    take them for letters. Only the part beyond the two line ends: a letter that touches the round end from inside
    the box is still read."""
    tips = np.zeros(rem.shape, np.int32)
    for i, ln in enumerate(lines, 1):
        for x, y in ((ln.x1, ln.y1), (ln.x2, ln.y2)):
            cv2.circle(tips, (round(x * ppm), round(y * ppm)), round((0.5 + ln.width) * ppm), i, -1)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(rem, connectivity=8)
    hit = (lab > 0) & (tips > 0)
    touching: dict[int, list[int]] = {}
    for c, i in np.unique(np.stack([lab[hit], tips[hit]]), axis=1).T:
        touching.setdefault(int(c), []).append(int(i) - 1)
    out = np.zeros(rem.shape, np.uint8)
    for c, ids in touching.items():
        x0, y0, w, h = stats[c, :4]
        if len(ids) < 2 or max(w, h) > JOINT * ppm:
            continue
        (a, ua), (b, ub) = (_tip(lines[i], cent[c] / ppm) for i in ids[:2])
        normal = np.array([a[1] - b[1], b[0] - a[0]])
        ys, xs = np.nonzero(lab[y0:y0 + h, x0:x0 + w] == c)
        side = (np.column_stack([xs + x0, ys + y0]) / ppm - a) @ normal * np.sign(normal @ (ua + ub) or 1.0)
        margin = (0.5 * max(lines[ids[0]].width, lines[ids[1]].width) + 0.15) * float(np.hypot(*normal))
        keep = side <= margin  # beyond the chord between the line ends, away from the lines, or on it
        out[ys[keep] + y0, xs[keep] + x0] = 1
    return out


def _tip(ln: Line, near: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The end of the line nearest to ``near`` (mm) and the direction from it into the line."""
    p, q = np.array([ln.x1, ln.y1]), np.array([ln.x2, ln.y2])
    if np.hypot(*(q - near)) < np.hypot(*(p - near)):
        p, q = q, p
    return p, (q - p) / max(float(np.hypot(*(q - p))), 1e-9)


def circles(ink: np.ndarray, ppm: float, text: np.ndarray | None = None,
            lines: list[Line] | None = None) -> tuple[list[Circle], np.ndarray]:
    """Circles of 1.5 to 40 mm radius (instrument bubbles, pumps, heat exchangers): Hough candidates,
    refined on the ink and accepted if ink covers at least 85 % of the circumference while the
    inside is not filled. Candidates are searched without the ``text`` pixels (a letter O is not a
    circle), then on all ink: OCR misreads symbol arcs as letters, such as the quarters of a circle in a box as
    brackets. A ring found only there must not be a single letter. The ring is checked on all ink, and the inside must
    be free of ink other than text (a filled dot is not a ring). The round end of a line-number box between two of
    the axis ``lines`` is not a circle. Returns the circles and the mask of their rings."""
    search = ink if text is None else ink & (1 - text)
    out, mask = [], np.zeros_like(ink)
    h, w = ink.shape
    thick = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    ends = _line_ends(lines or [])
    for src, letters in [(search, None)] + ([] if text is None else [(ink, ink & text)]):
        for x, y, r in _hough(src, ppm):
            band = max(2.0, 0.6 * ppm)
            x0, y0 = max(0, int(x - 1.3 * r - band)), max(0, int(y - 1.3 * r - band))
            x1, y1 = min(w, int(x + 1.3 * r + band) + 1), min(h, int(y + 1.3 * r + band) + 1)
            ys, xs = np.nonzero(src[y0:y1, x0:x1])
            xs, ys = xs + x0, ys + y0
            dist = np.hypot(xs - x, ys - y)
            r = _ring_radius(dist, r)
            near = np.abs(dist - r) < band
            if near.sum() < 20:
                continue
            fit = _circle_fit(np.column_stack([xs[near], ys[near]]).astype(np.float64))
            if fit is None:
                continue
            cx, cy, R, _ = fit
            if _coverage(thick, cx, cy, R) < 0.85 or _coverage(search, cx, cy, 0.6 * R) > 0.5 \
                    or any(math.hypot(cx - c.cx * ppm, cy - c.cy * ppm) < 0.5 * R for c in out) \
                    or letters is not None and _one_glyph(letters, cx, cy, R) \
                    or _slot_end(ends, cx / ppm, cy / ppm, R / ppm):
                continue
            width = max(1.0, near.sum() / (2 * np.pi * R))
            out.append(Circle(cx / ppm, cy / ppm, R / ppm, float(width / ppm)))
            cv2.circle(mask, (round(cx), round(cy)), round(R), 1, round(width + 2 * max(1.0, 0.1 * ppm)))
    return out, mask


def _ring_radius(dist: np.ndarray, r: float) -> float:
    """The radius between 0.7 and 1.3 times the Hough radius ``r`` at which most ink lies (per unit of circumference):
    the Hough radius of a circle in a box is drawn towards the box."""
    bins = np.arange(int(0.7 * r), int(1.3 * r) + 2)
    if len(bins) < 3:
        return r
    count, _ = np.histogram(dist, bins)
    k = int(np.argmax(count / (bins[:-1] + 0.5)))
    return float(bins[k] + 0.5)


def _hough(ink: np.ndarray, ppm: float) -> np.ndarray:
    """Circle candidates (x, y, r in pixels), searched at about 4 px/mm."""
    f = min(1.0, 4.0 / ppm)
    px = ppm * f
    small = cv2.GaussianBlur(cv2.resize(ink * 255, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (0, 0), 1.0)
    found = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT_ALT, 1.5, max(4.0, 2 * px), param1=200, param2=0.7,
                             minRadius=round(1.5 * px), maxRadius=round(40 * px))
    return np.zeros((0, 3)) if found is None else found.reshape(-1, 3) / f


def _one_glyph(letters: np.ndarray, cx: float, cy: float, r: float) -> bool:
    """True if most of the ring at (cx, cy, r) (pixels) is one connected stroke of the ``letters``: a letter O."""
    pad = 3
    x0, y0 = max(0, int(cx - r) - pad), max(0, int(cy - r) - pad)
    window = letters[y0:int(cy + r) + pad + 1, x0:int(cx + r) + pad + 1]
    _, lab = cv2.connectedComponents(window, connectivity=8)
    lab = cv2.dilate(lab.astype(np.float32), np.ones((3, 3), np.uint8))
    t = np.linspace(0, 2 * np.pi, max(36, int(2 * np.pi * r)), endpoint=False)
    u = np.clip(np.round(cx - x0 + r * np.cos(t)).astype(int), 0, lab.shape[1] - 1)
    v = np.clip(np.round(cy - y0 + r * np.sin(t)).astype(int), 0, lab.shape[0] - 1)
    hit = lab[v, u][lab[v, u] > 0]
    return len(hit) > 0 and np.bincount(hit.astype(int)).max() > 0.6 * len(t)


def _line_ends(lines: list[Line]) -> np.ndarray:
    """Both ends of each line with the unit vector into the line: rows x, y, ux, uy."""
    rows = []
    for ln in lines:
        length = math.hypot(ln.x2 - ln.x1, ln.y2 - ln.y1)
        if length > 0:
            ux, uy = (ln.x2 - ln.x1) / length, (ln.y2 - ln.y1) / length
            rows += [(ln.x1, ln.y1, ux, uy), (ln.x2, ln.y2, -ux, -uy)]
    return np.array(rows, np.float64).reshape(-1, 4)


def _slot_end(ends: np.ndarray, cx: float, cy: float, r: float) -> bool:
    """True if two parallel lines end at the circle (mm) on either side of its centre, about its diameter apart, and
    leave it the same way: the round end of a slot-shaped line-number box, not a circle."""
    d = ends[:, :2] - (cx, cy)
    dist = np.hypot(d[:, 0], d[:, 1])
    at = ends[(dist > 0.5 * r) & (dist < r + max(0.5, 0.15 * r))]
    for i, (x, y, ux, uy) in enumerate(at):
        for x2, y2, ux2, uy2 in at[i + 1:]:
            sides = (ux * (cy - y) - uy * (cx - x)) * (ux2 * (cy - y2) - uy2 * (cx - x2))
            apart = abs(ux * (y2 - y) - uy * (x2 - x))
            if ux * ux2 + uy * uy2 > 0.95 and sides < 0 and 1.3 * r <= apart <= 2.3 * r:
                return True
    return False


def _coverage(ink: np.ndarray, cx: float, cy: float, r: float) -> float:
    """Share of points on the circle that hit ink."""
    t = np.linspace(0, 2 * np.pi, max(36, int(2 * np.pi * r)), endpoint=False)
    u = np.clip(np.round(cx + r * np.cos(t)).astype(int), 0, ink.shape[1] - 1)
    v = np.clip(np.round(cy + r * np.sin(t)).astype(int), 0, ink.shape[0] - 1)
    return float(ink[v, u].mean())


def on_circle(ln: Line, cs: list[Circle]) -> bool:
    """True for short line pieces that lie on a circle (e.g. the flat top of a pump circle)."""
    pts = [(ln.x1 + (ln.x2 - ln.x1) * t, ln.y1 + (ln.y2 - ln.y1) * t) for t in (0, 0.25, 0.5, 0.75, 1)]
    return any(all(abs(math.hypot(x - c.cx, y - c.cy) - c.r) < max(0.4, c.width) for x, y in pts) for c in cs)


# --------------------------------------------------------------------------- skeleton tracing

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]  # clockwise ring


def _neighbours(img: np.ndarray) -> list[np.ndarray]:
    p = np.pad(img, 1)
    h, w = img.shape
    return [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in _NB]


def trace(ink: np.ndarray, ppm: float) -> list[tuple[np.ndarray, float]]:
    """Skeletonises the remaining strokes and returns ordered centre polylines (pixels) with
    their stroke width (pixels). Polylines end at junction centres so that strokes stay connected."""
    sk = skeletonize(ink.astype(bool)).astype(np.uint8)
    nbs = _neighbours(sk)
    count = sum(nbs)
    crossings = sum((nbs[i] == 0) & (nbs[(i + 1) % 8] == 1) for i in range(8))
    junction = (sk == 1) & (((count >= 3) & (crossings >= 3)) | (count >= 4))
    chains = sk & ~junction
    ends = (chains == 1) & (sum(_neighbours(chains)) == 1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(chains, connectivity=8)
    nj, jlab, _, jcent = cv2.connectedComponentsWithStats(junction.astype(np.uint8), connectivity=8)
    jpad = np.pad(jlab, 1)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    out = []
    for i in range(1, n):
        x0, y0, w, h, area = stats[i]
        sub = (lab[y0:y0 + h, x0:x0 + w] == i).astype(np.uint8)
        path = _ordered(sub, ends[y0:y0 + h, x0:x0 + w] & (sub == 1))
        if path is None:
            continue
        path = path + (x0, y0)
        width = 2 * float(np.median(dist[path[:, 1].astype(int), path[:, 0].astype(int)]))
        if not np.array_equal(path[0], path[-1]):  # open chain: connect ends to adjacent junctions
            pts = [path]
            for k, (ex, ey) in ((0, path[0]), (-1, path[-1])):
                j = jpad[int(ey):int(ey) + 3, int(ex):int(ex) + 3].max()
                if j:
                    c = jcent[j][None, :]
                    pts = [c] + pts if k == 0 else pts + [c]
            path = np.vstack(pts)
        out.append((path.astype(np.float64), width))
    return out


def _ordered(sub: np.ndarray, ends: np.ndarray) -> np.ndarray | None:
    """Orders the pixels of a one pixel wide chain. The outer contour of a thin chain runs along
    it and back again, so the part between the two end pixels is the ordered path."""
    c = cv2.findContours(np.pad(sub, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if not c:
        return None
    pts = c[0][:, 0, :] - 1
    if len(pts) < 2:
        return None
    if not ends.any():  # closed loop
        return np.vstack([pts, pts[:1]])
    hit = np.flatnonzero(ends[pts[:, 1], pts[:, 0]])
    if len(hit) < 2:
        return pts[hit[0]:] if len(hit) else pts
    return pts[hit[0]:hit[1] + 1]


# --------------------------------------------------------------------------- shapes

def shapes(polylines: list[tuple[np.ndarray, float]], ppm: float) -> tuple[list[Line], list[Arc], list[Circle]]:
    """Fits a circle or an arc where a whole polyline is round; otherwise splits it into the fewest straight segments
    and arcs that stay within a tenth of a millimetre or so of it (see _pieces), so that a curve comes out as one or
    a few arcs instead of a chain of short lines."""
    lines, arcs, circles = [], [], []
    eps = max(1.2, 0.15 * ppm)
    sag = max(2.0, 0.25 * ppm)  # an arc must bulge at least 0.25 mm from its chord
    tol = max(0.6, 0.05 * ppm)
    min_len = max(3.0, 0.4 * ppm)
    ends = Counter((round(x), round(y)) for pts, _ in polylines for x, y in (pts[0], pts[-1]))
    for pts, width in polylines:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        alone = ends[round(pts[0][0]), round(pts[0][1])] == ends[round(pts[-1][0]), round(pts[-1][1])] == 1
        if seg.sum() + (0.5 * width if alone else 0.0) < min_len:  # a short dash on its own is longer than its skeleton
            continue
        w = width / ppm
        closed = len(pts) > 3 and np.allclose(pts[0], pts[-1])
        simple = cv2.approxPolyDP(pts.astype(np.float32), eps, closed)[:, 0, :].astype(np.float64)
        if len(simple) <= (4 if closed else 3):  # a stroke, a corner or a triangle, such as a dash with a hooked end
            ring = np.vstack([simple, simple[:1]]) if closed else simple  # or a valve's side: no arcs
            lines += [Line(x1 / ppm, y1 / ppm, x2 / ppm, y2 / ppm, w) for (x1, y1), (x2, y2) in zip(ring, ring[1:])]
            continue
        if len(pts) >= 8:
            fit = _circle_fit(pts)
            if fit and fit[3] < tol and 0.5 * ppm <= fit[2] <= 250 * ppm:
                cx, cy, r, _ = fit
                a = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
                sweep = _sweep(a)
                if closed or sweep >= 330:
                    circles.append(Circle(cx / ppm, cy / ppm, r / ppm, w))
                    continue
                if sweep >= 25 and r * (1 - math.cos(math.radians(min(sweep, 180)) / 2)) >= sag:
                    start, end = (a[0], a[-1]) if _increasing(a) else (a[-1], a[0])
                    arcs.append(Arc(cx / ppm, cy / ppm, r / ppm, start % 360, end % 360, w))
                    continue
        for piece in _pieces(pts, closed, ppm, width):
            if isinstance(piece, Line):
                lines.append(Line(piece.x1 / ppm, piece.y1 / ppm, piece.x2 / ppm, piece.y2 / ppm, w))
            else:
                arcs.append(Arc(piece.cx / ppm, piece.cy / ppm, piece.r / ppm, piece.start, piece.end, w))
    return lines, arcs, circles


def _pieces(pts: np.ndarray, closed: bool, ppm: float, width: float = 0.0) -> list[Line | Arc]:
    """Splits a traced centre line (pixels; ``width`` its stroke width) into the fewest straight segments and arcs
    that each stay close to it.

    The pieces start and end at corners of a fine polygon approximation; a dynamic programme picks the split with
    the fewest pieces, a straight segment before an arc where both fit. A curve that a coarse polygon would cut
    into many short lines (a vessel head, a bend, the round end of a slot) becomes one arc or a few tangent ones."""
    fine = max(0.7, 0.06 * ppm)
    smallest = max(ppm, 2 * width, 6.0)  # a tighter bend is a corner of the traced centre line, rounded by the
    # stroke's width or the blur of a photo, not a drawn arc
    line_tol = max(1.2, 0.15 * ppm)  # as far as the old polygon approximation went
    arc_rms, arc_max = max(0.6, 0.05 * ppm), max(1.5, 0.15 * ppm)
    p32 = pts.astype(np.float32)
    if closed:  # start at the sharpest corner, so that no piece has to bend around it
        ring = pts[:-1]
        ahead, back = np.roll(ring, -3, axis=0) - ring, ring - np.roll(ring, 3, axis=0)
        turn = np.abs(np.remainder(np.arctan2(ahead[:, 1], ahead[:, 0]) - np.arctan2(back[:, 1], back[:, 0]) + np.pi,
                                   2 * np.pi) - np.pi)
        k = int(np.argmax(turn))
        pts = np.vstack([ring[k:], ring[:k], ring[k:k + 1]])
        p32 = pts.astype(np.float32)
    corners = cv2.approxPolyDP(p32, fine, False)[:, 0, :]
    # indices of the corners along the path; on a closed path the end, being the start, is not among the corners
    idx = [0]
    for c in corners[1:]:
        j = idx[-1] + int(np.argmin(np.hypot(*(pts[idx[-1]:] - c).T)))
        if j > idx[-1]:
            idx.append(j)
    if idx[-1] != len(pts) - 1:
        idx.append(len(pts) - 1)
    n = len(idx)
    best = [math.inf] * n
    best[0], choice = 0.0, [None] * n
    for j in range(1, n):
        misses = 0
        for i in range(j - 1, max(-1, j - 61), -1):
            piece, cost = _fit_piece(pts[idx[i]:idx[j] + 1], line_tol, arc_rms, arc_max, smallest, ppm)
            if piece is None:
                misses += 1
                if misses > 3:  # spans a corner: reaching further back only gets worse
                    break
                continue
            misses = 0
            if best[i] + cost < best[j]:
                best[j], choice[j] = best[i] + cost, (i, piece)
    out, j = [], n - 1
    while j > 0:
        if choice[j] is None:  # cannot happen: two neighbouring corners always fit a line
            break
        i, piece = choice[j]
        out.append(piece)
        j = i
    return out[::-1]


def _fit_piece(seg: np.ndarray, line_tol: float, arc_rms: float, arc_max: float, smallest: float, ppm: float):
    """A straight segment or an arc (radius ``smallest`` or more) through the ends of ``seg`` (pixels) if it stays
    close to all of its points;
    returns (piece, cost) or (None, None). Arcs cost slightly more, so that a straight run stays a line, and must
    bulge at least 0.25 mm from their chord. A flat arc (under 45 degrees) costs as much as two lines: it is only
    worth it where it replaces more, not where it rounds off a corner of the traced centre line."""
    sag = max(2.0, 0.25 * ppm)
    p, q = seg[0], seg[-1]
    chord = q - p
    h = float(np.hypot(*chord)) / 2
    if h < 1e-6:
        return None, None
    u = chord / (2 * h)
    nrm = np.array([-u[1], u[0]])
    rel = seg - (p + q) / 2
    a, v = rel @ u, rel @ nrm  # along the chord from its middle, and across it
    if np.max(np.abs(v)) <= line_tol:
        return Line(float(p[0]), float(p[1]), float(q[0]), float(q[1])), 1.0
    if len(seg) < 6:
        return None, None
    # circle through both ends with its centre on the bisector, at (0, t): u^2 + v^2 - 2 t v - h^2 = 0
    den = 2 * float(np.sum(v * v))
    if den <= 0:
        return None, None
    t = float(np.sum(v * (a * a + v * v - h * h))) / den
    r = math.hypot(h, t)
    if not smallest <= r <= 250 * ppm:
        return None, None
    dev = np.hypot(a, v - t) - r
    if float(np.sqrt(np.mean(dev ** 2))) > arc_rms or float(np.max(np.abs(dev))) > arc_max:
        return None, None
    side = 1.0 if float(np.mean(v)) > 0 else -1.0  # the arc bulges to this side of the chord
    sweep = 2 * math.asin(min(1.0, h / r))
    if side * t > 0:  # the centre lies on the bulge's side: more than a half circle
        sweep = 2 * math.pi - sweep
    if sweep < math.radians(8) or sweep > math.radians(330) or r * (1 - math.cos(min(sweep, math.pi) / 2)) < sag:
        return None, None  # too flat: a slightly bent dash stays straight
    c = (p + q) / 2 + t * nrm
    a0, a1 = (math.degrees(math.atan2(y - c[1], x - c[0])) % 360 for x, y in (p, q))
    mid = seg[len(seg) // 2] - c
    am = math.degrees(math.atan2(mid[1], mid[0])) % 360
    start, end = (a0, a1) if (am - a0) % 360 <= (a1 - a0) % 360 else (a1, a0)  # angles increase from start to end
    return Arc(float(c[0]), float(c[1]), r, start, end), 1.05 if sweep >= math.radians(45) else 2.05


def _circle_fit(pts: np.ndarray) -> tuple[float, float, float, float] | None:
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, -(x ** 2 + y ** 2), rcond=None)
    cx, cy = -sol[0] / 2, -sol[1] / 2
    r2 = cx ** 2 + cy ** 2 - sol[2]
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    return cx, cy, r, float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))


def _sweep(a: np.ndarray) -> float:
    """Angular span covered by a sequence of angles (degrees), following the sequence."""
    return float(np.abs(np.sum((np.diff(a) + 180) % 360 - 180)))


def _increasing(a: np.ndarray) -> bool:
    return float(np.sum((np.diff(a) + 180) % 360 - 180)) > 0


def merge_arcs(arcs: list[Arc]) -> tuple[list[Arc], list[Circle]]:
    """Arcs of one circle that were split where lines cross or touch it become a circle again if they go almost all
    the way round, otherwise one arc where they lie closely on one circle (see _chained)."""
    groups: list[list[Arc]] = []
    for a in sorted(arcs, key=lambda a: a.r):
        for g in groups:
            b = g[0]
            if math.hypot(a.cx - b.cx, a.cy - b.cy) < max(0.4, 0.06 * b.r) and abs(a.r - b.r) < max(0.3, 0.06 * b.r):
                g.append(a)
                break
        else:
            groups.append([a])
    out_arcs, circles = [], []
    for g in groups:
        covered = np.zeros(360, bool)
        for a in g:
            s = int(math.floor(a.start))
            covered[np.arange(s, s + int(math.ceil((a.end - a.start) % 360)) + 1) % 360] = True
        if len(g) > 1 and covered.sum() >= 300:
            wsum = sum((a.end - a.start) % 360 for a in g)
            avg = [sum(getattr(a, k) * ((a.end - a.start) % 360) for a in g) / wsum for k in ("cx", "cy", "r")]
            circles.append(Circle(*avg, max(a.width for a in g)))
        else:
            out_arcs += _chained(g)
    return out_arcs, circles


def _chained(arcs: list[Arc]) -> list[Arc]:
    """Arcs that lie on one circle within 3 % of its radius become one arc across the gaps between them. A tangent
    line takes the stroke of an arc over a longer stretch than a crossing one, so the gap that is closed grows with
    the radius. Arcs that do not lie that closely on one circle (two halves of a vessel head drawn a little apart)
    stay as they are, so that they keep meeting their tangent lines."""
    out, rest = [], sorted(arcs, key=lambda a: -((a.end - a.start) % 360))
    while rest:
        a = rest.pop(0)
        tol = max(0.2, 0.03 * a.r)
        close = [b for b in rest if math.hypot(b.cx - a.cx, b.cy - a.cy) <= tol and abs(b.r - a.r) <= tol]
        rest = [b for b in rest if b not in close]
        group = [a] + close
        weight = [(b.end - b.start) % 360 or 360 for b in group]
        cx, cy, r = (float(np.average([getattr(b, k) for b in group], weights=weight)) for k in ("cx", "cy", "r"))
        for start, extent, members in _runs(group, math.degrees(max(1.5, 1.8 * math.sqrt(r)) / r)):
            if len(members) == 1:
                out.append(members[0])
            else:
                out.append(Arc(cx, cy, r, start % 360, (start + extent) % 360, max(b.width for b in members)))
    return out


def _runs(arcs: list[Arc], gap: float) -> list[tuple[float, float, list[Arc]]]:
    """The stretches (start, extent in degrees, arcs) that arcs of one circle cover, with gaps up to ``gap`` degrees
    closed."""
    runs: list[list] = []
    for a in sorted(arcs, key=lambda a: a.start % 360):
        sweep = (a.end - a.start) % 360 or 360
        if runs and (a.start - runs[-1][0]) % 360 <= runs[-1][1] + gap:
            runs[-1][1] = max(runs[-1][1], (a.start - runs[-1][0]) % 360 + sweep)
            runs[-1][2].append(a)
        else:
            runs.append([a.start % 360, sweep, [a]])
    if len(runs) > 1 and (runs[0][0] - runs[-1][0]) % 360 <= runs[-1][1] + gap:  # round past 0 degrees
        last, first = runs.pop(), runs[0]
        first[1] = max(last[1], (first[0] - last[0]) % 360 + first[1])
        first[0] = last[0]
        first[2] += last[2]
    return [(start, min(extent, 360.0), members) for start, extent, members in runs]


# --------------------------------------------------------------------------- line clean-up

def merge_lines(lines: list[Line], tol: float = 0.35, gap: float = 0.6, dash_gap: float = 2.6) -> list[Line]:
    """Joins collinear pieces: overlapping or nearly touching ones into one line, regular short
    pieces with gaps into one dashed line. Works on (angle, normal offset, position) per line."""
    items = []
    for ln in lines:
        dx, dy = ln.x2 - ln.x1, ln.y2 - ln.y1
        lateral = max(math.tan(AXIS_SNAP) * math.hypot(dx, dy), 0.25)  # short pieces have noisy angles
        if abs(dy) <= lateral and abs(dx) >= abs(dy):
            th = 0.0
        elif abs(dx) <= lateral:
            th = math.pi / 2
        else:
            th = math.atan2(dy, dx) % math.pi
        c, s = math.cos(th), math.sin(th)
        mx, my = (ln.x1 + ln.x2) / 2, (ln.y1 + ln.y2) / 2
        t1, t2 = sorted((ln.x1 * c + ln.y1 * s, ln.x2 * c + ln.y2 * s))
        items.append((th, -mx * s + my * c, t1, t2, ln))
    items.sort(key=lambda it: (round(math.degrees(it[0])), it[1]))
    out, i = [], 0
    while i < len(items):
        j = i + 1
        while j < len(items) and abs(items[j][0] - items[i][0]) < math.radians(1.0) \
                and abs(items[j][1] - items[j - 1][1]) < 2 * tol:  # neighbours, so pixel steps do not split a line
            j += 1
        out += _merge_group(items[i:j], gap, dash_gap, tol)
        i = j
    return out


def _merge_group(group, gap, dash_gap, tol: float) -> list[Line]:
    """Joins the parallel pieces of one direction into runs along their length. A piece joins a run that it continues
    (within ``dash_gap``) at the offset of the run's last piece from 1 mm (within ``tol``), or that it runs beside
    within twice ``tol`` (two strokes that close are one). So a line on a photo may drift a little along its length,
    while two lines a millimetre apart, such as a pipe and the box of its line number, stay apart and keep their
    places, and short strokes between them, such as letters touching both, do not pull them together. Each run takes
    the length-weighted offset of its pieces."""
    th = group[0][0]
    c, s = math.cos(th), math.sin(th)
    runs: list[list] = []  # [pieces, weighted offset sum, length sum, end, offset of its last piece from 1 mm]
    for g in sorted(group, key=lambda g: g[2]):
        length = g[3] - g[2] + 1e-6
        fits = [r for r in runs if g[2] - r[3] <= dash_gap and abs(g[1] - r[4]) < (2 * tol if g[2] < r[3] else tol)]
        if fits:
            r = min(fits, key=lambda r: abs(g[1] - r[4]))
            r[0].append(g)
            r[1] += g[1] * length
            r[2] += length
            r[3] = max(r[3], g[3])
            r[4] = g[1] if length >= 1.0 else r[4]
        else:
            runs.append([[g], g[1] * length, length, g[3], g[1]])
    out = []
    for members, weighted, total, _, _ in runs:
        rho = weighted / total
        pieces = _solid_pieces(members, gap)
        if _dashed(pieces):
            pieces = [(pieces[0][0], pieces[-1][1], max(p[2] for p in pieces), True)]
        for t1, t2, w, dashed in pieces:
            out.append(Line(t1 * c - rho * s, t1 * s + rho * c, t2 * c - rho * s, t2 * s + rho * c, w, dashed))
    return out


def _dashed(pieces) -> bool:
    """At least three short dashes with regular lengths and gaps; broken solid lines are irregular."""
    if len(pieces) < 3:
        return False
    lens = np.array([p[1] - p[0] for p in pieces])
    gaps = np.array([b[0] - a[1] for a, b in zip(pieces, pieces[1:])])
    inner = lens[1:-1] if len(lens) > 3 else lens  # the first and last dash may be cut short
    regular = (np.mean(np.abs(inner - np.median(inner)) <= 0.35 * np.median(inner)) >= 0.7
               and np.mean(np.abs(gaps - np.median(gaps)) <= 0.5 * np.median(gaps)) >= 0.7)
    return bool(np.median(lens) <= 6.0 and np.median(gaps) >= 0.4 and regular)


def _solid_pieces(run, gap) -> list[tuple[float, float, float, bool]]:
    pieces = []
    for _, _, t1, t2, ln in run:
        if pieces and t1 - pieces[-1][1] <= gap:
            p = pieces[-1]
            pieces[-1] = (p[0], max(p[1], t2), max(p[2], ln.width), p[3] or ln.dashed)
        else:
            pieces.append((t1, t2, ln.width, ln.dashed))
    return pieces


BRIDGE = 5.0  # mm: the longest gap between two pieces of one line that is closed where the scan shows its stroke
BRIDGE_ANGLE = math.radians(12)
BRANCH = 4.0  # mm: lines from this length are extended onto a line they stop short of (pipes, not symbol strokes)


def bridge(lines: list[Line], arcs: list[Arc], norm: np.ndarray, ppm: float) -> list[Line]:
    """Closes gaps in lines that the scan shows as continuous: two line or arc ends that face each other up to
    BRIDGE mm apart, or a line end that points at another line, are joined where the grey image has the stroke's
    ink all along the gap (a faint or broken stretch, a crossing that cut the stroke). A gap of bare paper stays
    open, so dashes, line jumps and deliberate breaks are kept. Collinear lines become one line; otherwise the
    line end is moved onto the other piece. Returns the lines (arcs are not changed)."""
    if not lines:
        return lines
    ends, dirs, owner, width, length = [], [], [], [], []
    for i, ln in enumerate(lines):
        if ln.dashed:
            continue
        for k, (x, y, ox, oy) in enumerate(((ln.x1, ln.y1, ln.x2, ln.y2), (ln.x2, ln.y2, ln.x1, ln.y1))):
            ends.append((x, y))
            dirs.append(math.atan2(y - oy, x - ox))
            owner.append((i, k))
            width.append(ln.width)
            length.append(math.hypot(ln.x2 - ln.x1, ln.y2 - ln.y1))
    for i, a in enumerate(arcs):
        for k, t in enumerate((a.start, a.end)):
            r = math.radians(t)
            ends.append((a.cx + a.r * math.cos(r), a.cy + a.r * math.sin(r)))
            dirs.append(r - math.pi / 2 if k == 0 else r + math.pi / 2)  # leaving the arc along its tangent
            owner.append((len(lines) + i, k))
            width.append(a.width)
            length.append(math.radians((a.end - a.start) % 360) * a.r)
    if not ends:
        return lines
    E, D = np.array(ends), np.array(dirs)
    darkness = 255.0 - norm.astype(np.float32)
    pairs = []
    for i, j in cKDTree(E).query_pairs(BRIDGE):
        if owner[i][0] == owner[j][0] or min(width[i], width[j]) < 0.6 * max(width[i], width[j]):
            continue  # one piece, or two different pens
        gap = E[j] - E[i]
        g = float(np.hypot(*gap))
        if abs(math.remainder(D[i] - D[j] - math.pi, 2 * math.pi)) > BRIDGE_ANGLE:
            continue  # not facing each other
        if g > 0.3 and (abs(math.remainder(math.atan2(gap[1], gap[0]) - D[i], 2 * math.pi)) > BRIDGE_ANGLE
                        and abs(math.sin(math.atan2(gap[1], gap[0]) - D[i])) * g > 0.35):
            continue  # side by side, not in line
        pairs.append((max(length[i], length[j]) < BRANCH, g, i, j))
    used, merged, dots = set(), {}, []
    out = list(lines)
    for short, g, i, j in sorted(pairs):
        if i in used or j in used:
            continue
        a, b = owner[i], owner[j]
        if short and (a[0] >= len(lines) or b[0] >= len(lines)):
            continue
        if not _stroked(darkness, E[i], E[j], D[i], min(width[i], width[j]), ppm):
            continue
        if short:  # two short strokes: only as part of a longer run, see _dotted
            dots.append((i, j))
            continue
        used |= {i, j}
        if a[0] < len(lines) and b[0] < len(lines) and _in_line(out[a[0]], out[b[0]]):
            merged[a] = b
        elif a[0] < len(lines) and b[0] < len(lines):  # meet half way, the polylines join them
            _move(out[a[0]], a[1], (E[i] + E[j]) / 2)
            _move(out[b[0]], b[1], (E[i] + E[j]) / 2)
        elif a[0] < len(lines):  # a line runs on into an arc: move its end onto the arc's end
            _move(out[a[0]], a[1], E[j])
        elif b[0] < len(lines):
            _move(out[b[0]], b[1], E[i])
    for i, j in _dotted(dots, merged, owner, out):
        used |= {i, j}
        if _in_line(out[owner[i][0]], out[owner[j][0]]):
            merged[owner[i]] = owner[j]
        else:
            _move(out[owner[i][0]], owner[i][1], (E[i] + E[j]) / 2)
            _move(out[owner[j][0]], owner[j][1], (E[i] + E[j]) / 2)
    free = [i for i in range(len(E)) if i not in used and owner[i][0] < len(lines)]
    _branches(out, [(owner[i], E[i], D[i]) for i in free], darkness, ppm)
    # collinear lines that bridge a gap become one line through their outer ends
    parent = list(range(len(out)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (i, ki), (j, kj) in merged.items():
        if root(i) != root(j):
            parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(out)):
        groups.setdefault(root(i), []).append(i)
    result = []
    for members in groups.values():
        if len(members) == 1:
            result.append(out[members[0]])
            continue
        result.append(_joined([out[i] for i in members]))
    return result


def _dotted(dots: list[tuple[int, int]], merged: dict, owner: list, lines: list[Line]) -> list[tuple[int, int]]:
    """The gaps between short pieces in line (ends i, j) that are closed: those of a run that is at least twice
    BRANCH long once joined, such as a faint line that came out dotted, but not the two sides of a valve that meet in
    its middle."""
    parent = list(range(len(lines)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (i, _), (j, _) in merged.items():
        parent[root(i)] = root(j)
    taken = set()
    for i, j in dots:
        if i not in taken and j not in taken and root(owner[i][0]) != root(owner[j][0]):
            taken |= {i, j}
            parent[root(owner[i][0])] = root(owner[j][0])
    length: dict[int, float] = {}
    for k, ln in enumerate(lines):
        length[root(k)] = length.get(root(k), 0.0) + math.hypot(ln.x2 - ln.x1, ln.y2 - ln.y1)
    return [(i, j) for i, j in dots if i in taken and length[root(owner[i][0])] >= 2 * BRANCH]


def _branches(lines: list[Line], ends: list, darkness: np.ndarray, ppm: float) -> None:
    """Extends each of the line ``ends`` ((line, end), point, heading) that stops short of another line it points at
    (a branch off a pipe, a leader) up to BRIDGE mm onto that line, where the grey image has the stroke along the
    gap. Only lines of BRANCH mm or more: the short strokes of a symbol keep their ends, so the symbol still fits
    them."""
    ends = [e for e in ends if math.hypot(lines[e[0][0]].x2 - lines[e[0][0]].x1,
                                          lines[e[0][0]].y2 - lines[e[0][0]].y1) >= BRANCH]
    if not ends:
        return
    pts, of = [], []
    for k, ln in enumerate(lines):
        n = max(2, int(math.hypot(ln.x2 - ln.x1, ln.y2 - ln.y1)) + 2)  # about one sample per mm
        pts.append(np.linspace((ln.x1, ln.y1), (ln.x2, ln.y2), n))
        of.append(np.full(n, k))
    tree, of = cKDTree(np.vstack(pts)), np.concatenate(of)
    for (i, k), p, heading in ends:
        u = np.array([math.cos(heading), math.sin(heading)])
        hits = [(t, m) for m in set(of[tree.query_ball_point(p, BRIDGE + 1.0)].tolist()) if m != i
                and (t := _ray_hit(p, u, lines[m])) is not None and 0.05 < t <= BRIDGE]
        if hits:
            t, _ = min(hits)
            if _stroked(darkness, p, p + u * t, heading, lines[i].width, ppm):
                _move(lines[i], k, p + u * t)


def _ray_hit(p: np.ndarray, u: np.ndarray, ln: Line) -> float | None:
    """How far along the ray from p in direction u it meets the line (mm), if it meets it at 30 degrees or more."""
    a = np.array([ln.x1, ln.y1])
    v = np.array([ln.x2 - ln.x1, ln.y2 - ln.y1])
    cross = u[0] * v[1] - u[1] * v[0]
    if abs(cross) < 0.5 * math.hypot(*v):
        return None
    w = a - p
    t, s = (w[0] * v[1] - w[1] * v[0]) / cross, (w[0] * u[1] - w[1] * u[0]) / cross
    return float(t) if t > 0 and -0.02 <= s <= 1.02 else None


def _stroked(darkness: np.ndarray, p: np.ndarray, q: np.ndarray, heading: float, width: float, ppm: float) -> bool:
    """True if the grey image has the stroke (``width`` mm) all along p to q (mm): every sample within a pixel of the
    path at least 40 % as dark as the stroke just before p, with no clear stretch longer than 0.25 mm, and where it is
    faint, darker than the paper beside it. The grey that blur leaves inside a small symbol, such as a valve on a
    photo, is no stroke."""
    h, w = darkness.shape
    u = np.array([math.cos(heading), math.sin(heading)])
    before = p - u * np.linspace(0.3, 1.5, 6)[:, None]  # on the stroke, behind its end
    g = float(np.hypot(*(q - p)))
    n = max(2, int(g * ppm * 2))
    path = p + (q - p) * np.linspace(0, 1, n)[:, None]

    def dark(points: np.ndarray, spread: int = 1) -> np.ndarray:
        px = points * ppm
        best = np.zeros(len(points), np.float32)
        for dx in range(-spread, spread + 1):
            for dy in range(-spread, spread + 1):
                x = np.clip(np.round(px[:, 0]).astype(int) + dx, 0, w - 1)
                y = np.clip(np.round(px[:, 1]).astype(int) + dy, 0, h - 1)
                best = np.maximum(best, darkness[y, x])
        return best

    ref = float(np.median(dark(before)))
    if ref < 40:  # the stroke itself is too faint to tell
        return False
    along = dark(path)
    ok = along >= 0.4 * ref
    if ok.mean() < 0.85:
        return False
    run = longest = 0
    for v in ok:
        run = 0 if v else run + 1
        longest = max(longest, run)
    if longest * g / n > 0.25:
        return False
    faint = along < 0.7 * ref
    if not faint.any():
        return True
    side = np.array([-u[1], u[0]]) * (width / 2 + 2.0 / ppm)
    beside = np.maximum(dark(path[faint] + side, 0), dark(path[faint] - side, 0))
    return bool(np.mean(along[faint] - beside >= 0.15 * ref) >= 0.8)


LINK_DASH = 3.0  # mm: the longest dash of a short dashed link
LINK_GAP = (0.3, 2.0)  # mm: the gaps between its dashes


def links(lines: list[Line], arcs: list[Arc], circles: list[Circle], norm: np.ndarray, ppm: float) -> list[Line]:
    """Short dashed links, such as a signal line of two or three dashes between an instrument and the symbol it acts
    on: too few dashes for merge_lines to tell them from a broken line. Two or more short pieces in line with gaps of
    bare paper between them become one dashed line if the run leads from one stroke to another: each end reaches the
    stroke it points at within two gaps and a dash (the last dash often runs into it). Runs that lead nowhere, such as
    the strokes of unread letters, stay as they are."""
    short = [i for i, ln in enumerate(lines) if not ln.dashed and 0.3 <= _length(ln) <= LINK_DASH]
    if len(short) < 2:
        return lines
    runs = _runs_of_dashes(lines, short)
    if not runs:
        return lines
    pts, of = _samples(lines, arcs, circles)
    tree = cKDTree(pts)
    darkness = 255.0 - norm.astype(np.float32)
    out, dropped = [], set()
    for run in runs:
        link = _link(lines, run, tree, of, darkness, ppm)
        if link is not None:
            out.append(link)
            dropped |= set(run)
    return [ln for i, ln in enumerate(lines) if i not in dropped] + out


def _runs_of_dashes(lines: list[Line], short: list[int]) -> list[list[int]]:
    """Groups of two or more short pieces that follow one another in line with a gap of LINK_GAP between them."""
    ends = np.array([[(lines[i].x1, lines[i].y1), (lines[i].x2, lines[i].y2)] for i in short]).reshape(-1, 2)
    parent = list(range(len(short)))

    def root(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for a, b in cKDTree(ends).query_pairs(LINK_GAP[1]):
        i, j = a // 2, b // 2
        if i == j or not LINK_GAP[0] <= math.dist(ends[a], ends[b]) or not _in_row(lines[short[i]], lines[short[j]]):
            continue
        if min(math.dist(p, q) for p in ends[2 * i:2 * i + 2] for q in ends[2 * j:2 * j + 2]) < LINK_GAP[0]:
            continue  # touching or overlapping: one stroke, not two dashes
        parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for k in range(len(short)):
        groups.setdefault(root(k), []).append(short[k])
    return [g for g in groups.values() if len(g) >= 2]


def _in_row(a: Line, b: Line, tol: float = 0.35) -> bool:
    """Each end of one piece within ``tol`` mm of the other's axis, both ways (pieces in one row)."""
    def near(p: Line, q: Line) -> bool:
        ux, uy = (p.x2 - p.x1) / _length(p), (p.y2 - p.y1) / _length(p)
        return all(abs((x - p.x1) * uy - (y - p.y1) * ux) <= tol for x, y in ((q.x1, q.y1), (q.x2, q.y2)))
    return near(a, b) and near(b, a)


def _link(lines: list[Line], run: list[int], tree: cKDTree, of: np.ndarray, darkness: np.ndarray, ppm: float
          ) -> Line | None:
    """The dashed line of a run of dashes from the stroke before it to the stroke after it, or None."""
    pts = np.array([p for i in run for p in ((lines[i].x1, lines[i].y1), (lines[i].x2, lines[i].y2))])
    c = pts.mean(axis=0)
    u = np.linalg.svd(pts - c)[2][0]
    if abs(u[0]) >= math.cos(AXIS_SNAP) or abs(u[1]) >= math.cos(AXIS_SNAP):
        u = np.round(u)
    t = (pts - c) @ u
    if np.abs((pts - c) @ np.array([-u[1], u[0]])).max() > 0.3:
        return None  # not one straight row
    spans = sorted((min(t[2 * k], t[2 * k + 1]), max(t[2 * k], t[2 * k + 1])) for k in range(len(run)))
    gaps = [b[0] - a[1] for a, b in zip(spans, spans[1:])]
    if min(gaps) < LINK_GAP[0] or max(gaps) > LINK_GAP[1]:
        return None
    others = np.isin(of, run, invert=True)

    n = np.array([-u[1], u[0]])

    def touched(m: np.ndarray) -> bool:  # another stroke at m, or ink beside it (a flange mark across a pipe end)
        return any(others[k] for k in tree.query_ball_point(m, 0.35)) or \
            max(_dark_at(darkness, m + n * s, ppm) for s in (-0.6, -0.45, 0.45, 0.6)) > 100

    for a, b in zip(spans, spans[1:]):  # a gap of bare paper between free dash ends, not two pipe ends at a flange
        m = c + u * (a[1] + b[0]) / 2
        if _dark_at(darkness, m, ppm) > 100 or touched(m) or touched(c + u * a[1]) or touched(c + u * b[0]):
            return None
    reach = 2 * float(np.median(gaps)) + float(np.median([b - a for a, b in spans]))
    start = _reach(c + u * spans[0][0], -u, reach, tree, others)
    end = _reach(c + u * spans[-1][1], u, reach, tree, others)
    if start is None or end is None:
        return None
    width = float(np.median([lines[i].width for i in run]))
    return Line(float(start[0]), float(start[1]), float(end[0]), float(end[1]), width, True)


def _reach(p: np.ndarray, u: np.ndarray, reach: float, tree: cKDTree, others: np.ndarray) -> np.ndarray | None:
    """Where the ray from p along u first meets another stroke within ``reach`` mm (on the stroke's centre), or None."""
    for t in np.arange(0.0, reach + 1e-6, 0.05):
        near = [k for k in tree.query_ball_point(p + u * t, 0.2) if others[k]]
        if near:
            return p + u * max(t, float((tree.data[near] - p).mean(axis=0) @ u))
    return None


def _samples(lines: list[Line], arcs: list[Arc], circles: list[Circle], step: float = 0.1
             ) -> tuple[np.ndarray, np.ndarray]:
    """Points every ``step`` mm along the linework and the index of the line each comes from (-1: an arc or circle)."""
    pts, of = [], []
    for k, ln in enumerate(lines):
        n = max(2, int(_length(ln) / step) + 1)
        pts.append(np.linspace((ln.x1, ln.y1), (ln.x2, ln.y2), n))
        of.append(np.full(n, k))
    for cx, cy, r, a0, sweep in [(a.cx, a.cy, a.r, a.start, (a.end - a.start) % 360) for a in arcs] + \
            [(c.cx, c.cy, c.r, 0.0, 360.0) for c in circles]:
        t = np.radians(a0 + np.linspace(0, sweep, max(2, int(math.radians(sweep) * r / step) + 1)))
        pts.append(np.c_[cx + r * np.cos(t), cy + r * np.sin(t)])
        of.append(np.full(len(t), -1))
    return np.vstack(pts), np.concatenate(of)


def _dark_at(darkness: np.ndarray, p: np.ndarray, ppm: float) -> float:
    """The darkest grey within a pixel of p (mm)."""
    x, y = int(round(p[0] * ppm)), int(round(p[1] * ppm))
    return float(darkness[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max(initial=0.0))


def _length(ln: Line) -> float:
    return math.hypot(ln.x2 - ln.x1, ln.y2 - ln.y1)


def _in_line(a: Line, b: Line) -> bool:
    """Two lines on one straight: directions within 2 degrees and each end of one within 0.35 mm of the other's axis."""
    da, db = math.atan2(a.y2 - a.y1, a.x2 - a.x1), math.atan2(b.y2 - b.y1, b.x2 - b.x1)
    if abs(math.remainder(da - db, math.pi)) > math.radians(2):
        return False
    ux, uy = math.cos(da), math.sin(da)
    return all(abs((x - a.x1) * uy - (y - a.y1) * ux) <= 0.35 for x, y in ((b.x1, b.y1), (b.x2, b.y2)))


def _move(ln: Line, k: int, p: np.ndarray) -> None:
    if k == 0:
        ln.x1, ln.y1 = float(p[0]), float(p[1])
    else:
        ln.x2, ln.y2 = float(p[0]), float(p[1])


def _joined(group: list[Line]) -> Line:
    """One line through the outermost ends of collinear pieces, along their length-weighted direction."""
    pts = np.array([(ln.x1, ln.y1) for ln in group] + [(ln.x2, ln.y2) for ln in group])
    lengths = np.array([math.hypot(ln.x2 - ln.x1, ln.y2 - ln.y1) for ln in group])
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    d = vt[0]
    t = (pts - c) @ d
    p, q = c + d * t.min(), c + d * t.max()
    horizontal = all(ln.y1 == ln.y2 for ln in group)
    vertical = all(ln.x1 == ln.x2 for ln in group)
    if horizontal:
        y = float(np.average([ln.y1 for ln in group], weights=lengths + 1e-6))
        p, q = (min(pts[:, 0]), y), (max(pts[:, 0]), y)
    elif vertical:
        x = float(np.average([ln.x1 for ln in group], weights=lengths + 1e-6))
        p, q = (x, min(pts[:, 1])), (x, max(pts[:, 1]))
    width = float(np.average([ln.width for ln in group], weights=lengths + 1e-6))
    return Line(float(p[0]), float(p[1]), float(q[0]), float(q[1]), width, group[0].dashed)


def close_corners(lines: list[Line], tol: float = 1.0) -> list[Line]:
    """Snaps ends of horizontal and vertical lines onto perpendicular lines within ``tol`` mm, so
    that corners and T-junctions are closed in CAD."""
    hs = [ln for ln in lines if ln.y1 == ln.y2 and ln.x1 != ln.x2]
    vs = [ln for ln in lines if ln.x1 == ln.x2 and ln.y1 != ln.y2]
    for a, b in ((hs, vs), (vs, hs)):
        horizontal = a is hs
        if not b:
            continue
        pos = np.array([ln.x1 if horizontal else ln.y1 for ln in b])  # position of the perpendicular lines
        lo = np.array([min(ln.y1, ln.y2) if horizontal else min(ln.x1, ln.x2) for ln in b])
        hi = np.array([max(ln.y1, ln.y2) if horizontal else max(ln.x1, ln.x2) for ln in b])
        for ln in a:
            level = ln.y1 if horizontal else ln.x1
            for end in ("1", "2"):
                v = getattr(ln, ("x" if horizontal else "y") + end)
                d = np.abs(pos - v)
                ok = (d <= tol) & (lo - tol <= level) & (level <= hi + tol)
                if ok.any():
                    k = int(np.argmin(np.where(ok, d, np.inf)))
                    setattr(ln, ("x" if horizontal else "y") + end, float(pos[k]))
    return [ln for ln in lines if math.hypot(ln.x2 - ln.x1, ln.y2 - ln.y1) > 0.2]
