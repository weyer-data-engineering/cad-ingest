"""Compares a recognised drawing with its ground truth (all in sheet millimetres)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

from app.plan2cad.model import Arc, Circle, Drawing, Line, Polygon, Text
from app.plan2cad.render import text_length

STEP = 0.25  # mm between sample points along the linework


def _line(x1, y1, x2, y2) -> np.ndarray:
    n = max(2, int(math.hypot(x2 - x1, y2 - y1) / STEP) + 1)
    t = np.linspace(0, 1, n)[:, None]
    return np.array([x1, y1]) + t * np.array([x2 - x1, y2 - y1])


def _arc(cx, cy, r, start, end) -> np.ndarray:
    sweep = (end - start) % 360 or 360
    n = max(3, int(math.radians(sweep) * r / STEP) + 1)
    a = np.radians(start + np.linspace(0, sweep, n))
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def samples(d: Drawing, dashed_only: bool = False) -> np.ndarray:
    """Points every STEP mm along all linework (dashed lines count as continuous)."""
    pts = [_line(ln.x1, ln.y1, ln.x2, ln.y2) for ln in d.lines if ln.dashed or not dashed_only]
    if not dashed_only:
        pts += [_arc(a.cx, a.cy, a.r, a.start, a.end) for a in d.arcs]
        pts += [_arc(c.cx, c.cy, c.r, 0, 360) for c in d.circles]
        for g in d.polygons:
            ring = g.points + g.points[:1]
            pts += [_line(*a, *b) for a, b in zip(ring, ring[1:])]
    return np.vstack(pts) if pts else np.zeros((0, 2))


def geometry(gt: Drawing, pred: Drawing) -> dict:
    g, p = samples(gt), samples(pred)
    if not len(p):
        return {"recall_1mm": 0.0, "precision_1mm": 0.0, "f1_1mm": 0.0, "recall_05mm": 0.0, "precision_05mm": 0.0}
    dg, dp = cKDTree(p).query(g)[0], cKDTree(g).query(p)[0]
    out = {}
    for tol, key in ((1.0, "1mm"), (0.5, "05mm")):
        r, pr = float(np.mean(dg <= tol)), float(np.mean(dp <= tol))
        out[f"recall_{key}"], out[f"precision_{key}"] = r, pr
        out[f"f1_{key}"] = 2 * r * pr / (r + pr) if r + pr else 0.0
    out["deviation_mm"] = float(np.mean(dp[dp <= 1.0])) if np.any(dp <= 1.0) else None
    gd, pd = samples(gt, True), samples(pred, True)
    if len(gd):
        out["dashed_recall_1mm"] = float(np.mean(cKDTree(pd).query(gd)[0] <= 1.0)) if len(pd) else 0.0
    return out


def _sampled(parts: list) -> tuple[np.ndarray, np.ndarray]:
    """Sample points of lines, arcs, circles and polygon outlines with their direction (mod pi)."""
    pts, dirs = [], []
    for q in parts:
        if isinstance(q, Line):
            pts.append(_line(q.x1, q.y1, q.x2, q.y2))
            dirs.append(np.full(len(pts[-1]), math.atan2(q.y2 - q.y1, q.x2 - q.x1)))
        elif isinstance(q, (Arc, Circle)):
            pts.append(_arc(q.cx, q.cy, q.r, *((q.start, q.end) if isinstance(q, Arc) else (0, 360))))
            dirs.append(np.arctan2(pts[-1][:, 1] - q.cy, pts[-1][:, 0] - q.cx) + np.pi / 2)
        elif isinstance(q, Polygon):
            ring = q.points + q.points[:1]
            pts.append(np.vstack([_line(*a, *b) for a, b in zip(ring, ring[1:])]))
            dirs.append(np.zeros(len(pts[-1])))  # filled areas match in any direction
    return (np.vstack(pts), np.concatenate(dirs) % np.pi) if pts else (np.zeros((0, 2)), np.zeros(0))


def _reference_objects(d: Drawing) -> list[list]:
    """What a CAD user would draw as one object: lines and arcs of the same pen that meet end to end
    with nothing else at that point form one polyline; circles and filled areas count singly."""
    pieces = [(q, (q.x1, q.y1), (q.x2, q.y2)) for q in d.lines]
    for a in d.arcs:
        ends = [(a.cx + a.r * math.cos(math.radians(t)), a.cy + a.r * math.sin(math.radians(t))) for t in (a.start, a.end)]
        pieces.append((a, *ends))
    at = defaultdict(list)
    for i, (_, p, q) in enumerate(pieces):
        for x, y in (p, q):
            at[(round(x, 3), round(y, 3))].append(i)
    parent = list(range(len(pieces)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def pen(q):
        return q.width, getattr(q, "dashed", False)

    for ids in at.values():
        if len(ids) == 2 and pen(pieces[ids[0]][0]) == pen(pieces[ids[1]][0]):
            parent[root(ids[0])] = root(ids[1])
    groups = defaultdict(list)
    for i, (q, _, _) in enumerate(pieces):
        groups[root(i)].append(q)
    return list(groups.values()) + [[c] for c in d.circles] + [[g] for g in d.polygons] + [s.parts for s in d.symbols]


def _entities(d: Drawing) -> list[list]:
    """The pieces of each output entity; a polyline and a symbol (a block reference) are one entity each."""
    return [[q] for q in d.lines + d.arcs + d.circles + d.polygons] + [pl.parts() for pl in d.polylines] + \
        [s.parts for s in d.symbols]


def _length(parts: list) -> float:
    return sum(math.hypot(q.x2 - q.x1, q.y2 - q.y1) if isinstance(q, Line)
               else math.radians((q.end - q.start) % 360 or 360) * q.r for q in parts if isinstance(q, (Line, Arc)))


def objects(gt: Drawing, pred: Drawing) -> dict:
    """How many output entities make up each reference object (1 = in one piece, as drawn in CAD),
    and how many tiny pieces (under 1.5 mm) the output contains."""
    ents = _entities(pred)
    sampled = [_sampled(e) for e in ents]
    if not ents or not sum(len(p) for p, _ in sampled):
        return {}
    P = np.vstack([p for p, _ in sampled])
    D = np.concatenate([a for _, a in sampled])
    E = np.concatenate([np.full(len(p), i) for i, (p, _) in enumerate(sampled)])
    tree = cKDTree(P)
    pieces, whole = [], 0
    for obj in _reference_objects(gt):
        pts, dirs = _sampled(obj)
        dist, idx = tree.query(pts, k=8, distance_upper_bound=1.0)
        hit = idx < len(P)
        diff = np.abs(D[np.minimum(idx, len(P) - 1)] - dirs[:, None]) % np.pi
        ok = hit & ((np.minimum(diff, np.pi - diff) < math.radians(20)) | isinstance(obj[0], Polygon))
        first = np.argmax(ok, axis=1)
        found = E[idx[np.arange(len(pts)), first][ok.any(axis=1)]]
        if len(found) < 0.5 * len(pts):
            continue  # mostly missing: a recognition error, not a question of pieces
        counts = Counter(found.tolist())
        pieces.append(sum(k >= 2 for k in counts.values()))
        whole += max(counts.values()) >= 0.9 * len(found)
    lengths = [_length(e) for e in ents if not isinstance(e[0], (Circle, Polygon))]
    return {"pieces_per_object": float(np.mean(pieces)), "objects_whole": whole / len(pieces),
            "entities": len(ents), "reference_objects": len(_reference_objects(gt)),
            "short_pieces": sum(x < 1.5 for x in lengths)}


def circles(gt: Drawing, pred: Drawing) -> dict:
    def match(a, b):
        return math.hypot(a.cx - b.cx, a.cy - b.cy) < max(0.5, 0.15 * a.r) and abs(a.r - b.r) < max(0.4, 0.15 * a.r)

    reference = [c for c in gt.circles if c.r >= 1.5]  # smaller dots are outside the detection range
    if not reference:
        return {}
    hit = sum(any(match(c, q) for q in pred.circles) for c in reference)
    ok = sum(any(match(c, q) for c in gt.circles) for q in pred.circles)
    return {"circle_recall": hit / len(reference), "circle_precision": ok / max(1, len(pred.circles))}


def _cer(a: str, b: str) -> float:
    """Character error rate: Levenshtein distance of b to the reference a, divided by len(a)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(a))


def _box(t: Text, margin: float) -> tuple[float, float, float, float]:
    half_l, half_h = text_length(t) / 2 + margin, t.height / 2 + margin
    return (t.x - half_h, t.y - half_l, t.x + half_h, t.y + half_l) if t.angle == 90 else \
        (t.x - half_l, t.y - half_h, t.x + half_l, t.y + half_h)


def texts(gt: Drawing, pred: Drawing) -> dict:
    """Per ground-truth text, all recognised texts whose centre lies in its box are joined in
    reading order and compared. Recognised texts that match nothing count as false positives."""
    if not gt.texts:
        return {}
    review = sum(t.uncertain for t in pred.texts)  # marked as unread: not counted as read text
    pred = Drawing(pred.width, pred.height, texts=[t for t in pred.texts if not t.uncertain])
    used, cers, exact = set(), [], 0
    for t in gt.texts:
        x0, y0, x1, y1 = _box(t, 1.0)
        inside = [i for i, q in enumerate(pred.texts) if x0 <= q.x <= x1 and y0 <= q.y <= y1 and q.angle == t.angle]
        used.update(inside)
        key = (lambda i: -pred.texts[i].y) if t.angle == 90 else (lambda i: pred.texts[i].x)
        read = " ".join(pred.texts[i].text for i in sorted(inside, key=key))
        cers.append(min(1.0, _cer(t.text, read)))
        exact += read == t.text
    return {"text_cer": float(np.mean(cers)), "text_exact": exact / len(gt.texts),
            "text_false_positives": len(pred.texts) - len(used), "text_review": review}


def symbols(gt: Drawing, pred: Drawing) -> dict:
    """Symbols placed as library blocks: the right one within 1.5 mm of the reference, and its attributes read."""
    if not gt.symbols:
        return {}

    def match(a, b):
        return a.name == b.name and math.hypot(a.x - b.x, a.y - b.y) <= 1.5

    hit = sum(any(match(s, q) for q in pred.symbols) for s in gt.symbols)
    ok = sum(any(match(s, q) for s in gt.symbols) for q in pred.symbols)
    attributes = [(s, tag, t.text) for s in gt.symbols for tag, t in s.attributes.items()]
    read = sum(any(match(s, q) and getattr(q.attributes.get(tag), "text", None) == text for q in pred.symbols)
               for s, tag, text in attributes)
    return {"symbol_recall": hit / len(gt.symbols), "symbol_precision": ok / max(1, len(pred.symbols)),
            "symbol_attributes": read / len(attributes) if attributes else None}


def evaluate(gt: Drawing, pred: Drawing) -> dict:
    flat_gt, flat_pred = gt.flat(), pred.flat()
    out = geometry(flat_gt, flat_pred) | circles(flat_gt, flat_pred) | texts(flat_gt, flat_pred) | objects(gt, pred)
    out |= symbols(gt, pred)
    out["scale_ok"] = pred.scale == gt.scale if gt.scale else None
    return out
