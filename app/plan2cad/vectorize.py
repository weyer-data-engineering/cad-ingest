"""Turns the binary ink image (pixels) into lines, arcs, circles and filled areas (sheet mm)."""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage
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


def circles(ink: np.ndarray, ppm: float, text: np.ndarray | None = None) -> tuple[list[Circle], np.ndarray]:
    """Circles of 1.5 to 40 mm radius (instrument bubbles, pumps, heat exchangers): Hough candidates,
    refined on the ink and accepted if ink covers at least 85 % of the circumference while the
    inside is not filled. Candidates are searched without the ``text`` pixels (a letter O is not a
    circle), but the ring is checked on all ink: on sharp scans OCR misreads symbol arcs as letters.
    The inside must be free of ink other than text (a filled dot is not a ring).
    Returns the circles and the mask of their rings."""
    search = ink if text is None else ink & (1 - text)
    f = min(1.0, 4.0 / ppm)  # search at about 4 px/mm
    px = ppm * f
    small = cv2.GaussianBlur(cv2.resize(search * 255, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (0, 0), 1.0)
    found = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT_ALT, 1.5, max(4.0, 2 * px), param1=200, param2=0.7,
                             minRadius=round(1.5 * px), maxRadius=round(40 * px))
    out, mask = [], np.zeros_like(ink)
    if found is None:
        return out, mask
    h, w = ink.shape
    thick = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    for x, y, r in found.reshape(-1, 3) / f:
        band = max(2.0, 0.6 * ppm)
        x0, y0 = max(0, int(x - r - band)), max(0, int(y - r - band))
        x1, y1 = min(w, int(x + r + band) + 1), min(h, int(y + r + band) + 1)
        ys, xs = np.nonzero(search[y0:y1, x0:x1])
        xs, ys = xs + x0, ys + y0
        near = np.abs(np.hypot(xs - x, ys - y) - r) < band
        if near.sum() < 20:
            continue
        fit = _circle_fit(np.column_stack([xs[near], ys[near]]).astype(np.float64))
        if fit is None:
            continue
        cx, cy, R, _ = fit
        if _coverage(thick, cx, cy, R) < 0.85 or _coverage(search, cx, cy, 0.6 * R) > 0.5 \
                or any(math.hypot(cx - c.cx * ppm, cy - c.cy * ppm) < 0.5 * R for c in out):
            continue
        width = max(1.0, near.sum() / (2 * np.pi * R))
        out.append(Circle(cx / ppm, cy / ppm, R / ppm, float(width / ppm)))
        cv2.circle(mask, (round(cx), round(cy)), round(R), 1, round(width + 2 * max(1.0, 0.1 * ppm)))
    return out, mask


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
    """Fits arcs and circles where a polyline is round, otherwise simplifies it to straight segments."""
    lines, arcs, circles = [], [], []
    eps = max(1.2, 0.15 * ppm)
    sag = max(2.0, 0.25 * ppm)  # an arc must bulge at least 0.25 mm from its chord
    tol = max(0.6, 0.05 * ppm)
    min_len = max(3.0, 0.4 * ppm)
    for pts, width in polylines:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if seg.sum() < min_len:
            continue
        w = width / ppm
        closed = len(pts) > 3 and np.allclose(pts[0], pts[-1])
        simple = cv2.approxPolyDP(pts.astype(np.float32), eps, closed)[:, 0, :].astype(np.float64)
        if (closed or len(simple) > 3) and len(pts) >= 8:  # a corner (three points) is no arc
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
        if closed:
            simple = np.vstack([simple, simple[:1]])
        for (x1, y1), (x2, y2) in zip(simple, simple[1:]):
            lines.append(Line(x1 / ppm, y1 / ppm, x2 / ppm, y2 / ppm, w))
    return lines, arcs, circles


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
    """Arcs of one circle that were split where lines touch it become a circle again."""
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
            out_arcs += g
    return out_arcs, circles


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
                and abs(items[j][1] - items[j - 1][1]) < tol:  # neighbours, so pixel steps do not split a line
            j += 1
        out += _merge_group(items[i:j], gap, dash_gap)
        i = j
    return out


def _merge_group(group, gap, dash_gap) -> list[Line]:
    th = group[0][0]
    c, s = math.cos(th), math.sin(th)
    length = np.array([g[3] - g[2] for g in group]) + 1e-6
    rho = float(np.average([g[1] for g in group], weights=length))
    group = sorted(group, key=lambda g: g[2])
    runs, cur = [], [group[0]]
    for g in group[1:]:
        if g[2] - max(x[3] for x in cur) <= dash_gap:
            cur.append(g)
        else:
            runs.append(cur)
            cur = [g]
    runs.append(cur)
    out = []
    for run in runs:
        pieces = _solid_pieces(run, gap)
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
