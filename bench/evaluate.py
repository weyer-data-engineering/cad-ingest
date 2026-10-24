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


def _nodes(d: Drawing) -> np.ndarray:
    """Where reference linework ends or branches: line and arc ends that are not the inner vertex of a run
    (exactly two pieces of the same pen meeting)."""
    pieces = [(q, ((q.x1, q.y1), (q.x2, q.y2))) for q in d.lines]
    pieces += [(a, tuple((a.cx + a.r * math.cos(math.radians(t)), a.cy + a.r * math.sin(math.radians(t)))
                         for t in (a.start, a.end))) for a in d.arcs]
    at = defaultdict(list)
    for q, ends in pieces:
        for x, y in ends:
            at[(round(x, 3), round(y, 3))].append(q)
    pen = {id(q): (q.width, getattr(q, "dashed", False)) for q, _ in pieces}
    nodes = [p for p, qs in at.items() if len(qs) != 2 or pen[id(qs[0])] != pen[id(qs[1])]]
    return np.array(nodes, np.float64).reshape(-1, 2)


def continuity(gt: Drawing, pred: Drawing) -> dict:
    """What makes reworking a conversion tedious. ``breaks_per_m``: places per metre of reference linework where an
    output entity stops although the reference runs on (a gap or two pieces where one is drawn), counted from the
    output's open ends that lie on the reference and run along it, away from its ends and branches. ``curve_pieces``:
    output entities per reference arc (1 = in one piece; circles are scored by ``circles``). ``curves_as_arcs``: share
    of the reference arcs that the output draws as arcs or circles, not as chains of short straight segments."""
    flat_gt = gt.flat()
    ents = _entities(pred)
    ref = samples(flat_gt)
    if not ents or not len(ref):
        return {}
    pts, dirs, ent, curved = [], [], [], []
    for i, e in enumerate(ents):
        for q in e:
            p, a = _sampled([q])
            pts.append(p)
            dirs.append(a)
            ent.append(np.full(len(p), i))
            curved.append(np.full(len(p), isinstance(q, (Arc, Circle))))
    out = {}
    ends, heading = _open_ends(pred)
    lw, lw_dir = _sampled(flat_gt.lines + flat_gt.arcs + flat_gt.circles)
    if len(ends) and len(lw):
        dist, near = cKDTree(lw).query(ends)
        turn = np.abs(heading - lw_dir[near]) % np.pi
        on_ref = (dist <= 0.5) & (np.minimum(turn, np.pi - turn) < math.radians(25))  # running along it
        nodes = _nodes(flat_gt)
        free = cKDTree(nodes).query(ends)[0] > 1.5 if len(nodes) else np.ones(len(ends), bool)
        length = len(ref) * STEP / 1000
        out["breaks_per_m"] = float(np.sum(on_ref & free) / 2 / length)
    P, D, E, C = np.vstack(pts), np.concatenate(dirs), np.concatenate(ent), np.concatenate(curved)
    tree = cKDTree(P)
    pieces, as_arc, total = [], 0, 0
    for q in flat_gt.arcs:
        if q.r < 1.0:
            continue
        s, a = _sampled([q])
        dist, idx = tree.query(s, k=8, distance_upper_bound=0.5)
        hit = idx < len(P)
        diff = np.abs(D[np.minimum(idx, len(P) - 1)] - a[:, None]) % np.pi
        ok = hit & (np.minimum(diff, np.pi - diff) < math.radians(25))
        if ok.any(axis=1).mean() < 0.5:
            continue  # mostly missing: a recognition error, not a question of pieces
        first = idx[np.arange(len(s)), np.argmax(ok, axis=1)][ok.any(axis=1)]
        counts = Counter(E[first].tolist())
        pieces.append(sum(k >= 2 for k in counts.values()))
        as_arc += int(C[first].sum())
        total += len(first)
    if pieces:
        out["curve_pieces"] = float(np.mean(pieces))
        out["curves_as_arcs"] = as_arc / total
    return out


def _open_ends(d: Drawing) -> tuple[np.ndarray, np.ndarray]:
    """The two ends of every line, arc and open polyline (symbols are blocks and have none) with the direction of the
    entity there (mod pi)."""
    at = [(q, (q.x1, q.y1)) for q in d.lines] + [(q, (q.x2, q.y2)) for q in d.lines]
    at += [(a, (a.cx + a.r * math.cos(math.radians(t)), a.cy + a.r * math.sin(math.radians(t))))
           for a in d.arcs for t in (a.start, a.end)]
    for pl in d.polylines:
        if not pl.closed:
            parts = pl.parts()
            at += [(parts[0], pl.points[0]), (parts[-1], pl.points[-1])]
    dirs = [math.atan2(q.y2 - q.y1, q.x2 - q.x1) if isinstance(q, Line)
            else math.atan2(y - q.cy, x - q.cx) + math.pi / 2 for q, (x, y) in at]
    return np.array([p for _, p in at], np.float64).reshape(-1, 2), np.array(dirs) % np.pi


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


def _kind(s: str) -> str:
    """codes (tags, line numbers: letters and digits), numbers (sizes, dimensions) or words (notes, names)."""
    digits, letters = any(c.isdigit() for c in s), any(c.isalpha() for c in s)
    return "codes" if digits and letters else "numbers" if digits else "words"


def texts(gt: Drawing, pred: Drawing) -> dict:
    """Per ground-truth text, all recognised texts whose centre lies in its box are joined in
    reading order and compared. Recognised texts that match nothing count as false positives. A text read wrong is
    flagged if a text marked for review lies in its box; the others are errors nobody is pointed to."""
    if not gt.texts:
        return {}
    review = sum(t.uncertain for t in pred.texts)  # marked as unread: not counted as read text
    marked = [t for t in pred.texts if t.uncertain]
    pred = Drawing(pred.width, pred.height, texts=[t for t in pred.texts if not t.uncertain])
    used, cers, exact, unflagged = set(), [], 0, 0
    kinds = defaultdict(lambda: [0, 0])  # kind: [exact, all]
    for t in gt.texts:
        x0, y0, x1, y1 = _box(t, 1.0)
        inside = [i for i, q in enumerate(pred.texts) if x0 <= q.x <= x1 and y0 <= q.y <= y1 and q.angle == t.angle]
        used.update(inside)
        key = (lambda i: -pred.texts[i].y) if t.angle == 90 else (lambda i: pred.texts[i].x)
        read = " ".join(pred.texts[i].text for i in sorted(inside, key=key))
        cers.append(min(1.0, _cer(t.text, read)))
        exact += read == t.text
        kinds[_kind(t.text)][0] += read == t.text
        kinds[_kind(t.text)][1] += 1
        unflagged += read != t.text and not any(x0 <= q.x <= x1 and y0 <= q.y <= y1 for q in marked)
    return {"text_cer": float(np.mean(cers)), "text_exact": exact / len(gt.texts),
            "text_false_positives": len(pred.texts) - len(used), "text_review": review,
            "text_unflagged": unflagged / len(gt.texts)} | \
        {f"text_exact_{k}": kinds[k][0] / kinds[k][1] if kinds[k][1] else None for k in ("codes", "numbers", "words")}


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


def _linked(d: Drawing, tol: float = 0.5) -> set[frozenset]:
    """Pairs of symbols joined by linework. Lines, arcs and polylines that meet end to end, or with an end on another
    one (a branch), form one net; a symbol belongs to every net that reaches one of its ports."""
    nets = [[q] for q in d.lines + d.arcs] + [pl.parts() for pl in d.polylines if pl.kind != "frame"]
    sampled = [_sampled(parts)[0] for parts in nets]
    if not d.symbols or not any(len(p) for p in sampled):
        return set()
    points = np.vstack([p for p in sampled if len(p)])
    owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(sampled) if len(p)])
    tree = cKDTree(points)
    parent = list(range(len(nets)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, p in enumerate(sampled):
        for end in (p[:1], p[-1:]) if len(p) else ():
            for j in tree.query_ball_point(end[0], tol):
                parent[root(owner[j])] = root(i)
    members = defaultdict(set)
    for k, s in enumerate(d.symbols):
        for port in s.ports:
            for j in tree.query_ball_point(port, tol):
                members[root(owner[j])].add(k)
    return {frozenset((a, b)) for m in members.values() for a in m for b in m if a < b}


def connectivity(gt: Drawing, pred: Drawing) -> dict:
    """Symbols joined by linework in the reference and in the output; output symbols count as the reference symbol of
    the same name within 1.5 mm. A broken pipe loses a connection, a wrong junction adds one."""
    ref = _linked(gt)
    if not ref:
        return {}
    same = {}
    for k, q in enumerate(pred.symbols):
        near = [(math.hypot(s.x - q.x, s.y - q.y), i) for i, s in enumerate(gt.symbols) if s.name == q.name]
        near = [n for n in near if n[0] <= 1.5]
        same[k] = min(near)[1] if near else -1 - k
    found = {frozenset(same[k] for k in pair) for pair in _linked(pred)}
    found = {pair for pair in found if len(pair) == 2}
    hit = len(ref & found)
    return {"connection_recall": hit / len(ref), "connection_precision": hit / max(1, len(found))}


def evaluate(gt: Drawing, pred: Drawing) -> dict:
    flat_gt, flat_pred = gt.flat(), pred.flat()
    out = geometry(flat_gt, flat_pred) | circles(flat_gt, flat_pred) | texts(flat_gt, flat_pred) | objects(gt, pred)
    out |= continuity(gt, pred) | symbols(gt, pred) | connectivity(gt, pred)
    out["scale_ok"] = pred.scale == gt.scale if gt.scale else None
    return out
