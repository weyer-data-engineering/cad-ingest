"""Finds the paper (or, if the paper is cut off, the drawing frame) and straightens it."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import map_coordinates, median_filter

FRAME_MARGIN = 10.0  # assumed paper margin around the drawing frame in mm
LID_DEPTH = 0.3  # share of the image across each side in which the edge of paper on a scanner lid is looked for
STEP = 2.0  # mm: the grey on either side of a paper edge is the median over this width
STRIP = 5.0  # mm along a side that one profile across it is averaged over
PROFILES = 24  # profiles across each side
FIT = 0.01  # paper found on a lid has the aspect ratio of its size within this share, and the scale of a known dpi
FILLS = 0.9  # share of the image that paper found on a lid with a side of the image's own fills at most
SKEW = 8.0  # degrees a paper edge on a lid may lean against the side of the image


@dataclass
class Sheet:
    corners: np.ndarray  # top left, top right, bottom right, bottom left in image pixels
    size: tuple[float, float]  # sheet width, height in mm
    method: str  # "paper" or "frame"
    inset: float = 0.0  # distance of the corners from the sheet edge in mm (frame mode)


def locate(img: np.ndarray, paper_mm: tuple[float, float], lid: bool = False, dpi: float | None = None
           ) -> Sheet | None:
    """Finds the sheet in a photo. Returns None if no paper edge is visible (e.g. a scan that is
    cropped to the sheet); falls back to the drawing frame if the paper runs out of the image. Where the border of the
    image looks like paper, the paper may lie on a scanner lid as grey as it: with ``lid``, it is looked for by its
    straight edges there (on_lid), which must agree with the scan's ``dpi`` where that is known."""
    h, w = img.shape[:2]
    f = 800 / max(h, w)
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    b = 3  # thin border band, also the background band for GrabCut
    if _paper_fills_image(small, b):
        return on_lid(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), paper_mm, dpi=dpi) if lid else None
    # GrabCut separates paper from desk by colour and edges, which also works under a shadow
    # that darkens part of the sheet below the brightness of the desk.
    mask = np.full(small.shape[:2], cv2.GC_PR_FGD, np.uint8)
    mask[:b], mask[-b:], mask[:, :b], mask[:, -b:] = cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD
    cv2.setRNGSeed(0)  # GrabCut starts from random colour clusters: same photo, same sheet on every request
    cv2.grabCut(small, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    contours = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.15 * fg.size:
        return None
    quad = _quad(c)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # A border band that looks like the paper means the paper runs out of the picture on that side.
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    paper = np.median(lab[fg > 0], axis=0)
    cut = max(float((np.linalg.norm(band.reshape(-1, 3) - paper, axis=1) < 25).mean())
              for band in (lab[:b], lab[-b:], lab[:, :b], lab[:, -b:]))
    if quad is not None and cut < 0.25:
        corners = _refine(gray, quad / f, max(16.0, 12 / f))
        return Sheet(corners, oriented(corners, paper_mm), "paper")
    # A border that looks like paper: the paper lies on a lid as light as it, with a straight edge on every side, or
    # it runs out of the picture.
    if lid and (on := on_lid(gray, paper_mm, every_side=True, dpi=dpi)) is not None:
        return on
    # The paper runs out of the picture: use the printed frame inside the visible paper instead.
    # GrabCut is unreliable here (its background band contains paper), so the frame is searched in
    # everything that is bright like paper.
    gs = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (0, 0), 2)
    bright = cv2.threshold(gs, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    corners = find_frame(gray, cv2.resize(bright | fg, (w, h), interpolation=cv2.INTER_NEAREST))
    if corners is None or not cv2.isContourConvex(corners.astype(np.float32)) \
            or cv2.contourArea(corners.astype(np.float32)) < 0.25 * h * w:
        return None
    aspect = rect_aspect(corners, (w, h))
    long = max(paper_mm) - 2 * FRAME_MARGIN
    edges = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    landscape = edges[0] + edges[2] >= edges[1] + edges[3]
    if aspect is None or not 0.85 < (aspect if landscape else 1 / aspect) / math.sqrt(2) < 1.15:
        aspect = math.sqrt(2) if landscape else 1 / math.sqrt(2)
    w_, h_ = (long, long / aspect) if landscape else (long * aspect, long)
    return Sheet(corners, (w_ + 2 * FRAME_MARGIN, h_ + 2 * FRAME_MARGIN), "frame", FRAME_MARGIN)


def _paper_fills_image(small: np.ndarray, m: int) -> bool:
    border = np.concatenate([small[:m].reshape(-1, 3), small[-m:].reshape(-1, 3),
                             small[:, :m].reshape(-1, 3), small[:, -m:].reshape(-1, 3)]).astype(np.float32)
    h, w = small.shape[:2]
    centre = small[h // 4:3 * h // 4, w // 4:3 * w // 4].reshape(-1, 3).astype(np.float32)
    paper = np.median(centre[centre.sum(1) >= np.percentile(centre.sum(1), 50)], axis=0)
    return float(np.linalg.norm(np.median(border, axis=0) - paper)) < 25


def on_lid(gray: np.ndarray, paper_mm: tuple[float, float], every_side: bool = False, dpi: float | None = None
           ) -> Sheet | None:
    """The paper of a scan on a scanner lid about as grey as the paper, which colour does not tell apart: the long
    straight step of grey near each side of the image (_lid_side). A side without one is the image's own edge, where
    the paper lies against it or runs out of it, unless ``every_side`` must have one; there the image must be paper,
    not lid, out to its border (_paper_out_to). The paper found must have the aspect ratio of its size and, where the
    scan's ``dpi`` is known, its scale (FIT); with a side of the image's own and most of the image filled (FILLS), it
    must fit that aspect ratio better than the image itself, which is otherwise the likelier sheet (a copy cropped to
    it with a dark band along an edge). None otherwise, so a scan cropped to its sheet, whose sides have no such step,
    stays as it is. Its corners may lie a little outside the image: a skewed sheet's corners are often not scanned."""
    h, w = gray.shape
    ppm = max(h, w) / max(paper_mm)  # a little high where a lid shows: enough to size the windows by
    found = [_lid_side(gray, side, ppm) for side in range(4)]
    edges = [f for f in found if f is not None]
    if not edges or every_side and len(edges) < 4:
        return None
    lid, paper = (float(np.median([f[k] for f in edges])) for k in (1, 2))
    if any(f is None and not _paper_out_to(gray, side, ppm, lid, paper) for side, f in enumerate(found)):
        return None
    sides = [((0.0, 0.0), (0.0, 1.0)), ((0.0, 0.0), (1.0, 0.0)), ((w - 1.0, 0.0), (0.0, 1.0)),
             ((0.0, h - 1.0), (1.0, 0.0))]  # the image's own: left, top, right, bottom
    left, top, right, bottom = [tuple(map(np.array, s)) if f is None else f[0] for f, s in zip(found, sides)]
    corners = np.array([_intersect(top, left), _intersect(top, right), _intersect(bottom, right),
                        _intersect(bottom, left)])
    ratio = max(paper_mm) / min(paper_mm)
    misfit = abs(_aspect(corners) / ratio - 1)
    near = np.all((corners >= -0.1 * np.array([w, h])) & (corners <= 1.1 * np.array([w, h])))
    if not near or misfit > FIT:
        return None
    if len(edges) < 4 and cv2.contourArea(corners.astype(np.float32)) > FILLS * w * h \
            and abs(max(w, h) / min(w, h) / ratio - 1) <= misfit:
        return None
    on = Sheet(corners, oriented(corners, paper_mm), "paper")
    return None if dpi and abs(source_ppm(on) * 25.4 / dpi - 1) > FIT else on


def _aspect(corners: np.ndarray) -> float:
    """The long sides of a quadrilateral against its short ones."""
    e = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    return max(e[0] + e[2], e[1] + e[3]) / min(e[0] + e[2], e[1] + e[3])


def _turned(gray: np.ndarray, side: int) -> np.ndarray:
    """The image with its rows along one side (0 left, 1 top, 2 right, 3 bottom) and its columns inwards from it."""
    return (gray, gray.T, gray[:, ::-1], gray[::-1].T)[side]


def _profiles(g: np.ndarray, depth: int, ppm: float) -> list[tuple[int, np.ndarray]]:
    """PROFILES profiles ``depth`` pixels inwards across a side (``g`` turned to it), with their place along it: each
    the mean of the middle half of a strip STRIP mm wide, so no specks or lines, and not in whole steps of grey."""
    half, out = max(1, round(STRIP / 2 * ppm)), []
    for at in np.linspace(0.12, 0.88, PROFILES) * g.shape[0]:
        strip = np.sort(g[max(0, int(at) - half):int(at) + half, :depth], axis=0)
        q = len(strip) // 4
        out.append((int(at), strip[q:len(strip) - q].mean(axis=0)))
    return out


def _paper_out_to(gray: np.ndarray, side: int, ppm: float, lid: float, paper: float) -> bool:
    """Whether the image is paper, not lid, out to a side without a paper edge: the grey of its outermost STEP mm is
    nearer that of the paper than that of the lid (at the paper edges found on the other sides) at three in four places
    along it at least. A small sheet near the corner of a large scanner bed shows lid there."""
    n = max(1, round(STEP * ppm))
    greys = np.array([np.median(prof) for _, prof in _profiles(_turned(gray, side), n, ppm)])
    return float(np.mean(np.abs(greys - paper) < np.abs(greys - lid))) >= 0.75


def _lid_side(gray: np.ndarray, side: int, ppm: float) -> tuple[tuple[np.ndarray, np.ndarray], float, float] | None:
    """The paper edge along one side of the image (0 left, 1 top, 2 right, 3 bottom) as a line (point, direction), with
    the grey of the lid and of the paper there: the steps of PROFILES profiles across the side (_lid_step), all of one
    sign (the lid is darker or lighter than the paper all along) and most on one straight line, nearly parallel to the
    side; or None."""
    g = _turned(gray, side)  # rows along the side, columns inwards from it
    length, depth = g.shape[0], round(LID_DEPTH * g.shape[1])
    width = max(3, round(STEP * ppm)) | 1
    k = width // 2 + max(1, round(0.4 * ppm))  # half a window and the blur of the edge
    if depth <= 4 * k + round(ppm):
        return None
    profiles = []
    for at, prof in _profiles(g, depth, ppm):
        med = median_filter(prof, size=width, mode="nearest")
        step = np.zeros_like(med)
        step[k:-k] = med[2 * k:] - med[:-2 * k]  # the median further in less the one further out
        profiles.append((at, prof, med, step))
    # the noise of the scan, from all profiles: the print that some of them cross (a title block) does not raise it
    noise = float(np.median([1.4826 * np.median(np.abs(s[k:-k] - np.median(s[k:-k]))) for *_, s in profiles]))
    found = [(at, *edge) for at, prof, med, s in profiles
             if (edge := _lid_step(prof, med, s, k, ppm, max(4.0, 6 * noise))) is not None]
    if len(found) < PROFILES // 2:
        return None
    pts = np.array(found)
    pts = pts[np.sign(pts[:, 2]) == np.sign(np.median(pts[:, 2]))]
    fit = _straight(pts[:, 0], pts[:, 1], max(1.5, 0.3 * ppm))
    if fit is None or fit[2] < PROFILES // 2 or abs(fit[1]) > math.tan(math.radians(SKEW)):
        return None
    a, b, _ = fit
    h, w = gray.shape

    def image(along: float, inwards: float) -> np.ndarray:
        return np.array([(inwards, along), (along, inwards), (w - 1 - inwards, along), (along, h - 1 - inwards)][side])

    p, q = image(0, a), image(length, a + b * length)
    return (p, (q - p) / np.linalg.norm(q - p)), float(np.median(pts[:, 3])), float(np.median(pts[:, 4]))


def _lid_step(prof: np.ndarray, med: np.ndarray, s: np.ndarray, k: int, ppm: float, need: float
              ) -> tuple[float, float, float, float] | None:
    """Where a profile coming in from a side of the image steps from the even grey of a lid to the paper: the
    position in pixels, the step in grey and the grey of the lid and of the paper on either side, or None. ``med`` is
    the profile's median over STEP mm and ``s`` the difference of the medians ``k`` pixels further in and further out,
    so a thin printed line, such as the drawing frame, is no step. A step needs ``need`` grey; the lid's side must be
    even out to the side of the image and the paper's side as bright as the paper further in, so neither edge of a
    broad band of print is one either. Only the first step from outside counts: further in is the drawing."""
    strong = np.flatnonzero(np.abs(s) >= need)
    strong = strong[strong >= k + round(ppm)]  # a millimetre of lid at least
    if not len(strong):
        return None
    c = int(strong[0] + np.argmax(np.abs(s[strong[0]:strong[0] + 2 * k + 1])))  # within k of the edge
    lid, inner = float(med[c - k]), float(med[c + k])
    paper = float(np.percentile(prof[c + k:c + k + round(15 * ppm)], 75))
    outside = float(np.median(prof[:c - k]))  # c is within k of the edge: up to there, all lid
    tol = max(3.0, 0.35 * abs(float(s[c])))
    if abs(outside - lid) > tol or abs(inner - paper) > tol or abs(paper - lid) < need:
        return None
    # A median filter keeps an edge where it is: the edge is where the filtered profile crosses the middle grey
    seg = med[c - k:c + k + 1] - (lid + inner) / 2
    cross = np.flatnonzero(np.sign(seg[:-1]) != np.sign(seg[1:]))
    if not len(cross):
        return None
    return float(np.mean(c - k + cross + seg[cross] / (seg[cross] - seg[cross + 1]))), float(s[c]), lid, inner


def _straight(t: np.ndarray, v: np.ndarray, tol: float) -> tuple[float, float, int] | None:
    """The straight line v = a + b t through the most points (within ``tol``), fitted to them: a, b and their number.
    Every pair of points proposes a line."""
    best = None
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            if t[j] != t[i]:
                b = (v[j] - v[i]) / (t[j] - t[i])
                near = np.abs(v - (v[i] + b * (t - t[i]))) <= tol
                if best is None or near.sum() > best.sum():
                    best = near
    if best is None:
        return None
    b, a = np.polyfit(t[best], v[best], 1)
    return float(a), float(b), int(best.sum())


def _quad(contour: np.ndarray) -> np.ndarray | None:
    """Four corners of a sheet-shaped contour: a rough quadrilateral of the convex hull, then a
    straight line fitted to the contour along each side, so rounded corners do not pull it in."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.1, 19):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) <= 4:
            break
    if len(approx) != 4:
        return None
    q = order_corners(approx.reshape(4, 2).astype(np.float64))
    pts = contour[:, 0, :].astype(np.float64)
    sides = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        length = np.linalg.norm(b - a)
        u = (b - a) / length
        t, off = (pts - a) @ u, (pts - a) @ np.array([-u[1], u[0]])
        near = (t > 0.15 * length) & (t < 0.85 * length) & (np.abs(off) < 0.04 * length)
        if near.sum() < 10:
            return q
        vx, vy, x0, y0 = cv2.fitLine(pts[near].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        sides.append((np.array([x0, y0]), np.array([vx, vy])))
    return np.array([_intersect(sides[i - 1], sides[i]) for i in range(4)])


def order_corners(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(1), pts[:, 1] - pts[:, 0]
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def _intersect(l1, l2) -> np.ndarray:
    (p1, d1), (p2, d2) = l1, l2
    t = np.linalg.solve(np.column_stack([d1, -d2]), p2 - p1)
    return p1 + t[0] * d1


def _refine(gray: np.ndarray, quad: np.ndarray, r: float) -> np.ndarray:
    """Moves each edge onto the paper edge at full resolution. Coming from outside, the paper edge
    is where the profile becomes paper-bright for several pixels in a row; printed lines (like the
    drawing frame a few millimetres inside) only follow later."""
    g = gray.astype(np.float32)
    ss = np.arange(-r, r + 0.5, 0.5)  # sample positions across the edge, positive towards the sheet
    lines = []
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        d = (q - p) / np.linalg.norm(q - p)
        n = np.array([-d[1], d[0]])
        pts = p + np.linspace(0.08, 0.92, 48)[:, None] * (q - p)
        xy = pts[:, None, :] + ss[None, :, None] * n
        prof = map_coordinates(g, [xy[..., 1].ravel(), xy[..., 0].ravel()], order=1, mode="nearest").reshape(len(pts), -1)
        found = [row + pos * n for row, pr in zip(pts, prof) if (pos := _paper_edge(pr, ss)) is not None]
        if len(found) < 6:  # keep the rough edge
            lines.append((p, d))
            continue
        vx, vy, x0, y0 = cv2.fitLine(np.array(found, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    return np.array([_intersect(lines[i - 1], lines[i]) for i in range(4)])


def _paper_edge(prof: np.ndarray, ss: np.ndarray, w: int = 12) -> float | None:
    """Position along ``ss`` where the profile turns from background into paper, or None."""
    n = len(prof)
    paper_level = float(np.percentile(prof[n // 2:], 75))
    background = float(np.median(prof[: n // 4]))
    if abs(paper_level - background) < 15:
        return None
    is_paper = np.abs(prof - paper_level) < max(10.0, 0.3 * abs(paper_level - background))
    window = np.convolve(is_paper.astype(np.int32), np.ones(w, np.int32), "valid") == w
    starts = np.flatnonzero(window[1:] & ~is_paper[: len(window) - 1]) + 1
    if not len(starts):
        return None
    k0 = max(0, starts[0] - w)
    seg = prof[k0:starts[0] + 1] - (paper_level + background) / 2  # crossing of the middle level
    cross = np.flatnonzero(np.diff(np.sign(seg)))
    if not len(cross):
        return float(ss[starts[0]])
    k = cross[-1]
    return float(ss[k0 + k] + seg[k] / (seg[k] - seg[k + 1]) * (ss[1] - ss[0]))


def find_frame(gray: np.ndarray, paper: np.ndarray) -> np.ndarray | None:
    """Corners of the outermost long printed lines (the drawing frame) inside the visible paper."""
    h, w = gray.shape
    f = 1600 / max(h, w)
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    region = cv2.resize(paper, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    region = cv2.erode(region, np.ones((15, 15), np.uint8))
    ink = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10) & (region * 255)
    sh, sw = small.shape
    segs = cv2.HoughLinesP(ink, 1, np.pi / 720, 80, minLineLength=0.15 * min(sh, sw), maxLineGap=6)
    if segs is None:
        return None
    segs = segs.reshape(-1, 4).astype(np.float64)
    d = segs[:, 2:] - segs[:, :2]
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180
    length = np.hypot(d[:, 0], d[:, 1])
    sides = []
    for horizontal in (True, False):
        sel = ((ang < 30) | (ang > 150)) if horizontal else ((ang > 60) & (ang < 120))
        s, dd, ln = segs[sel], d[sel], length[sel]
        if not len(s):
            return None
        if horizontal:  # position: y at the image centre column
            pos = s[:, 1] + (sw / 2 - s[:, 0]) * dd[:, 1] / np.where(dd[:, 0] == 0, 1e-9, dd[:, 0])
        else:
            pos = s[:, 0] + (sh / 2 - s[:, 1]) * dd[:, 0] / np.where(dd[:, 1] == 0, 1e-9, dd[:, 1])
        need = 0.3 * (sw if horizontal else sh)
        for order in (np.argsort(pos), np.argsort(-pos)):  # outermost first: top/left, then bottom/right
            for k in order:
                near = np.abs(pos - pos[k]) < 0.006 * max(sh, sw)
                if ln[near].sum() >= need:
                    pts = np.vstack([s[near, :2], s[near, 2:]]).astype(np.float32)
                    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
                    sides.append((np.array([x0, y0]) / f, np.array([vx, vy])))
                    break
            else:
                return None
    top, bottom, left, right = sides
    corners = np.array([_intersect(top, left), _intersect(top, right), _intersect(bottom, right),
                        _intersect(bottom, left)])
    return corners


def rect_aspect(corners: np.ndarray, image_size: tuple[int, int]) -> float | None:
    """Width/height of the real rectangle seen as ``corners``, assuming square pixels and the
    principal point in the image centre (Zhang & He, "Whiteboard scanning", 2007)."""
    u0, v0 = image_size[0] / 2, image_size[1] / 2
    tl, tr, br, bl = (np.array([x, y, 1.0]) for x, y in corners)
    m1, m2, m3, m4 = tl, tr, bl, br
    den2, den3 = np.dot(np.cross(m2, m4), m3), np.dot(np.cross(m3, m4), m2)
    if abs(den2) < 1e-9 or abs(den3) < 1e-9:
        return None
    k2, k3 = np.dot(np.cross(m1, m4), m3) / den2, np.dot(np.cross(m1, m4), m2) / den3
    n2, n3 = k2 * m2 - m1, k3 * m3 - m1
    if abs(n2[2] * n3[2]) < 1e-12:
        return None
    f2 = -((n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 ** 2)
           + (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 ** 2)) / (n2[2] * n3[2])
    if f2 <= 0:
        return None
    Ai = np.linalg.inv(np.array([[math.sqrt(f2), 0, u0], [0, math.sqrt(f2), v0], [0, 0, 1]]))
    a, b = Ai @ n2, Ai @ n3
    return float(math.sqrt((a @ a) / (b @ b)))


def oriented(corners: np.ndarray, paper_mm: tuple[float, float]) -> tuple[float, float]:
    """Paper size in the orientation of the photographed sheet."""
    e = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    long, short = max(paper_mm), min(paper_mm)
    return (long, short) if e[0] + e[2] >= e[1] + e[3] else (short, long)


def source_ppm(sheet: Sheet) -> float:
    w, h = sheet.size[0] - 2 * sheet.inset, sheet.size[1] - 2 * sheet.inset
    return math.sqrt(float(cv2.contourArea(sheet.corners.astype(np.float32))) / (w * h))


def rectify(gray: np.ndarray, sheet: Sheet, ppm: float, flags: int = cv2.INTER_CUBIC) -> np.ndarray:
    W, H = sheet.size
    i = sheet.inset
    dst = np.float32([[i, i], [W - i, i], [W - i, H - i], [i, H - i]]) * ppm
    M = cv2.getPerspectiveTransform(sheet.corners.astype(np.float32), dst)
    return cv2.warpPerspective(gray, M, (round(W * ppm), round(H * ppm)), flags=flags,
                               borderMode=cv2.BORDER_REPLICATE)


def deskew(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Rotates a scan without visible paper edges so that its long lines are axis-parallel; returns it with the
    rotation matrix (None if it needs none)."""
    f = 1500 / max(gray.shape)
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ink = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    segs = cv2.HoughLinesP(ink, 1, np.pi / 720, 80, minLineLength=small.shape[1] // 8, maxLineGap=4)
    if segs is None:
        return gray, None
    segs = segs.reshape(-1, 4)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    ang = (ang + 45) % 90 - 45
    ang = ang[np.abs(ang) < 10]
    if not len(ang) or abs(np.median(ang)) < 0.05:
        return gray, None
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(np.median(ang)), 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), M
