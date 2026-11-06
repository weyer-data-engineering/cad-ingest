"""Compares a recognised drawing with its ground truth (all in sheet millimetres)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from dataclasses import replace

import cv2
import numpy as np
from scipy.spatial import cKDTree

from app.plan2cad.dimensions import INCH, UNITS, value
from app.plan2cad.fit import ASIDE, Linework
from app.plan2cad.model import Arc, Circle, Drawing, Line, Polygon, Text, line_type
from app.plan2cad.simulate import MARKUP
from app.plan2cad.render import text_length

STEP = 0.25  # mm between sample points along the linework
SPECIAL = set("±Ø°²³‰×′″")  # characters drawings write that a reader easily drops or confuses: "±0,00", "Ø 12", "m²"


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


def pens(gt: Drawing, pred: Drawing) -> dict:
    """The share of the reference linework drawn with the right pen: the ISO pen of the output's linework within 1 mm
    (the pen of the DXF) is the pen of the reference."""
    from app.plan2cad.export import lineweight

    def widths(d: Drawing) -> tuple[np.ndarray, np.ndarray]:
        parts = [q for q in d.lines + d.arcs + d.circles if q.width > 0]
        sampled = [_sampled([q])[0] for q in parts]
        if not parts:
            return np.zeros((0, 2)), np.zeros(0)
        return np.vstack(sampled), np.concatenate([np.full(len(s), lineweight(q.width)) for q, s in zip(parts, sampled)])

    (g, gw), (p, pw) = widths(gt), widths(pred)
    if not len(g) or not len(p):
        return {}
    dist, idx = cKDTree(p).query(g)
    near = dist <= 1.0
    return {"pens_correct": float(np.mean(gw[near] == pw[idx[near]])) if near.any() else None}


def line_types(gt: Drawing, pred: Drawing) -> dict:
    """``linetype_correct``: the share of the reference lines that are not continuous (dashed, dash-dot, dotted) found
    with their line type: most of the line has output linework of that line type along it within 1 mm.
    ``linetype_precision``: the share of the length of the output's dash-dot and dotted lines that runs along reference
    linework of its line type within 1 mm; a row of specks or a broken solid line taken for one counts against it."""
    out = {}
    if ref := [q for q in gt.lines if q.dashed]:
        index = _typed_index([q for q in pred.lines if q.dashed])
        out["linetype_correct"] = float(np.mean([np.mean(_typed(q, index)) > 0.5 for q in ref]))
    if new := [q for q in pred.lines if line_type(q) in ("dashdot", "dotted")]:
        index = _typed_index(gt.lines + gt.arcs + gt.circles)
        out["linetype_precision"] = float(np.mean(np.concatenate([_typed(q, index) for q in new])))
    return out


def _typed_index(items: list) -> tuple | None:
    """Sample points of lines, arcs and circles with their directions and line types, and a tree over them."""
    if not items:
        return None
    sampled = [_sampled([q]) for q in items]
    P = np.vstack([p for p, _ in sampled])
    kind = np.concatenate([np.full(len(p), line_type(q)) for q, (p, _) in zip(items, sampled)])
    return cKDTree(P), np.concatenate([a for _, a in sampled]), kind


def _typed(q: Line, index: tuple | None) -> np.ndarray:
    """For each sample point of the line, whether the nearest linework of the index along it (within 1 mm, directions
    within 20 degrees) is of its line type."""
    pts, dirs = _sampled([q])
    if index is None:
        return np.zeros(len(pts), bool)
    tree, D, kind = index
    dist, idx = tree.query(pts, k=4, distance_upper_bound=1.0)
    hit = idx < len(D)
    diff = np.abs(D[np.minimum(idx, len(D) - 1)] - dirs[:, None]) % np.pi
    ok = hit & (np.minimum(diff, np.pi - diff) < math.radians(20))
    first = idx[np.arange(len(pts)), np.argmax(ok, axis=1)]
    return ok.any(axis=1) & (kind[np.minimum(first, len(D) - 1)] == line_type(q))


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
    """The pieces of each output entity; a polyline, a chain of dimensions and a symbol (a block reference) are one
    entity each (the symbols last)."""
    return [[q] for q in d.lines + d.arcs + d.circles + d.polygons] + [pl.parts() for pl in d.polylines] + \
        _chains(d.dimensions) + [s.parts for s in d.symbols]


def _chains(dims: list) -> list[list]:
    """The strokes of each chain of dimensions: dimensions that follow one another on one dimension line share it and
    their extension lines and terminators, of which each has only some among its parts."""
    out: list[list] = []
    for before, m in zip([None] + dims, dims):
        if before and math.dist((before.x2, before.y2), (m.x1, m.y1)) < 1e-6:
            out[-1] += m.parts
        else:
            out.append(list(m.parts))
    return [parts for parts in out if parts]


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
    of the reference arcs that the output draws as arcs or circles, not as chains of short straight segments.
    ``curves_as_splines``: the share it draws as splines (free curves) instead."""
    flat_gt = gt.flat()
    ents = _entities(pred)
    ref = samples(flat_gt)
    if not ents or not len(ref):
        return {}
    pts, dirs, ent, curved, spline = [], [], [], [], []
    smooth = [False] * len(pred.lines + pred.arcs + pred.circles + pred.polygons) + [pl.smooth for pl in pred.polylines]
    smooth += [False] * (len(ents) - len(smooth))  # dimensions and symbols, after the polylines (_entities)
    for i, e in enumerate(ents):
        for q in e:
            p, a = _sampled([q])
            pts.append(p)
            dirs.append(a)
            ent.append(np.full(len(p), i))
            curved.append(np.full(len(p), isinstance(q, (Arc, Circle))))
            spline.append(np.full(len(p), smooth[i]))
    out = {}
    ends, heading = _open_ends(pred)
    lw, lw_dir = _sampled(flat_gt.lines + flat_gt.arcs + flat_gt.circles)
    if len(ends) and len(lw):
        dist, near = cKDTree(lw).query(ends)
        turn = np.abs(heading - lw_dir[near]) % np.pi
        on_ref = (dist <= 0.5) & (np.minimum(turn, np.pi - turn) < math.radians(25))  # running along it
        # a wire that ends at the port of a symbol ends there, although it runs on into the symbol in the same pen
        ports = np.array([p for s in gt.symbols for p in s.ports], np.float64).reshape(-1, 2)
        nodes = np.vstack([_nodes(flat_gt), ports])
        free = cKDTree(nodes).query(ends)[0] > 1.5 if len(nodes) else np.ones(len(ends), bool)
        length = len(ref) * STEP / 1000
        out["breaks_per_m"] = float(np.sum(on_ref & free) / 2 / length)
    P, D, E, C = np.vstack(pts), np.concatenate(dirs), np.concatenate(ent), np.concatenate(curved)
    S = np.concatenate(spline)
    tree = cKDTree(P)
    pieces, as_arc, as_spline, total = [], 0, 0, 0
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
        as_spline += int(S[first].sum())
        total += len(first)
    if pieces:
        out["curve_pieces"] = float(np.mean(pieces))
        out["curves_as_arcs"] = as_arc / total
        out["curves_as_splines"] = as_spline / total
    return out


def _open_ends(d: Drawing) -> tuple[np.ndarray, np.ndarray]:
    """The two ends of every line, arc and open polyline (symbols and dimensions are one entity and have none) with the
    direction of the entity there (mod pi)."""
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


def _in(t: Text, q: Text, margin: float = 1.0) -> bool:
    """Whether the centre of q lies in the box of t (turned with t) and q reads the same way."""
    if abs((q.angle - t.angle + 180) % 360 - 180) > 5:
        return False
    a = math.radians(t.angle)
    dx, dy = q.x - t.x, q.y - t.y
    along, across = dx * math.cos(a) - dy * math.sin(a), dx * math.sin(a) + dy * math.cos(a)
    return abs(along) <= text_length(t) / 2 + margin and abs(across) <= t.height / 2 + margin


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
    used, cers, exact, unflagged, tokens, found, special, special_found = set(), [], 0, 0, 0, 0, 0, 0
    kinds = defaultdict(lambda: [0, 0])  # kind: [exact, all]
    for t in gt.texts:
        inside = [i for i, q in enumerate(pred.texts) if _in(t, q)]
        used.update(inside)
        a = math.radians(t.angle)
        read = " ".join(pred.texts[i].text for i in sorted(  # in reading order: along the text's direction
            inside, key=lambda i: pred.texts[i].x * math.cos(a) - pred.texts[i].y * math.sin(a)))
        cers.append(min(1.0, _cer(t.text, read)))
        exact += read == t.text
        kinds[_kind(t.text)][0] += read == t.text
        kinds[_kind(t.text)][1] += 1
        unflagged += read != t.text and not any(_in(t, q) for q in marked)
        want, got = Counter(t.text.split()), Counter(read.split())
        tokens += sum(want.values())
        found += sum((want & got).values())
        signs = Counter({w: n for w, n in want.items() if SPECIAL & set(w)})
        special += sum(signs.values())
        special_found += sum((signs & got).values())
    return {"text_cer": float(np.mean(cers)), "text_exact": exact / len(gt.texts),
            "token_exact": found / max(1, tokens),
            "special_tokens_exact": special_found / special if special else None,
            "text_false_positives": len(pred.texts) - len(used), "text_review": review,
            "text_unflagged": unflagged / len(gt.texts)} | \
        {f"text_exact_{k}": kinds[k][0] / kinds[k][1] if kinds[k][1] else None for k in ("codes", "numbers", "words")}


def symbols(gt: Drawing, pred: Drawing) -> dict:
    """Symbols placed as library blocks: the right one within 1.5 mm of the reference, and its attributes read. And
    the symbols marked to check, and the wrong ones left unmarked (share of the reference symbols): nobody is pointed
    to those. ``symbol_false``: symbols placed where the reference has none, outside every reference symbol."""
    boxes = [_extent(s) for s in gt.symbols]
    false = sum(not any(x0 <= q.x <= x1 and y0 <= q.y <= y1 for x0, y0, x1, y1 in boxes) for q in pred.symbols)
    if not gt.symbols:
        return {"symbol_false": false} if pred.symbols else {}

    def match(a, b):
        return a.name == b.name and math.hypot(a.x - b.x, a.y - b.y) <= 1.5

    hit = sum(any(match(s, q) for q in pred.symbols) for s in gt.symbols)
    ok = sum(any(match(s, q) for s in gt.symbols) for q in pred.symbols)
    attributes = [(s, tag, t.text) for s in gt.symbols for tag, t in s.attributes.items()]
    read = sum(any(match(s, q) and getattr(q.attributes.get(tag), "text", None) == text for q in pred.symbols)
               for s, tag, text in attributes)
    unflagged = sum(not q.doubt and not any(match(s, q) for s in gt.symbols) for q in pred.symbols)
    return {"symbol_recall": hit / len(gt.symbols), "symbol_precision": ok / max(1, len(pred.symbols)),
            "symbol_attributes": read / len(attributes) if attributes else None,
            "symbol_review": sum(bool(q.doubt) for q in pred.symbols), "symbol_unflagged": unflagged / len(gt.symbols),
            "symbol_false": false}


def _extent(s, margin: float = 1.5) -> tuple[float, float, float, float]:
    """The box around a placed symbol's strokes, ``margin`` mm larger."""
    xs, ys = [s.x], [s.y]
    for q in s.parts:
        if isinstance(q, Line):
            xs += [q.x1, q.x2]
            ys += [q.y1, q.y2]
        else:  # an arc as its whole circle
            xs += [q.cx - q.r, q.cx + q.r]
            ys += [q.cy - q.r, q.cy + q.r]
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


def _parts_box(s) -> tuple[float, float, float, float]:
    """The box of the strokes of a placed symbol (x0, y0, x1, y1)."""
    pts, _ = _sampled(s.parts)
    return (*pts.min(axis=0), *pts.max(axis=0))


def _overlap(a: tuple, b: tuple) -> float:
    """The share of the smaller of two boxes (x0, y0, x1, y1) that the other covers."""
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return max(w, 0) * max(h, 0) / smaller if smaller > 0 else 0.0


def symbols_apart(gt: Drawing, pred: Drawing) -> dict:
    """What happens around the symbols of the reference. ``symbol_phantom_texts``: texts of the output on a reference
    symbol where the reference has no text, its strokes read as letters. ``unclassified_on_symbols`` and
    ``unclassified_elsewhere``: the boxes the output marks as a symbol not classified on a reference symbol it did not
    place right, or on a symbol of no library that the reference lists (its ``unclassified``, such as a level mark),
    and elsewhere; ``unclassified_found``: the share of those symbols it marked so."""
    unlisted = [u.box for u in getattr(gt, "unclassified", [])]
    marked = [u.box for u in getattr(pred, "unclassified", [])]
    if not gt.symbols and not unlisted:  # all that is marked lies elsewhere
        return {"unclassified_on_symbols": 0, "unclassified_elsewhere": len(marked)}
    known = gt.flat().texts
    boxes = [_parts_box(s) for s in gt.symbols]
    phantoms = sum(1 for t in pred.texts if any(b[0] - 0.5 <= t.x <= b[2] + 0.5 and b[1] - 0.5 <= t.y <= b[3] + 0.5
                                               for b in boxes) and not any(_in(k, t) for k in known))
    missed = [b for s, b in zip(gt.symbols, boxes) if not any(
        q.name == s.name and math.hypot(q.x - s.x, q.y - s.y) <= 1.5 for q in pred.symbols)] + unlisted
    on = sum(any(_overlap(m, b) >= 0.5 for b in missed) for m in marked)
    return {"symbol_phantom_texts": phantoms, "unclassified_on_symbols": on, "unclassified_elsewhere": len(marked) - on,
            "unclassified_found": sum(any(_overlap(m, b) >= 0.5 for m in marked) for b in missed) / len(missed)
            if missed else None}


def short_strokes(gt: Drawing, pred: Drawing, longest: float = 3.0) -> dict:
    """The share of the short straight strokes of the reference (0.5 to ``longest`` mm, solid: ticks, arrowheads,
    the bars and barbs of symbols) that the output has, as linework or in a placed symbol: three quarters of the
    stroke within 0.5 mm of output linework running the same way."""
    short = [q for q in gt.lines if not q.dashed and 0.5 <= math.hypot(q.x2 - q.x1, q.y2 - q.y1) <= longest]
    p, a = _sampled(pred.lines + pred.arcs + pred.circles + pred.polygons)
    if not short:
        return {}
    if not len(p):
        return {"short_strokes_kept": 0.0}
    tree, kept = cKDTree(p), 0
    for q in short:
        s, d = _sampled([q])
        dist, idx = tree.query(s, k=8, distance_upper_bound=0.5)
        hit = idx < len(p)
        diff = np.abs(a[np.minimum(idx, len(p) - 1)] - d[:, None]) % np.pi
        kept += np.mean((hit & (np.minimum(diff, np.pi - diff) < math.radians(25))).any(axis=1)) >= 0.75
    return {"short_strokes_kept": kept / len(short)}


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


def loose(d: Drawing) -> dict:
    """What has to be joined or cleaned up by hand in CAD, without a reference: ``gap_ends``, ends of lines, arcs and
    open polylines that stop short of other linework by up to 2 mm without touching it (a pipe that does not reach its
    junction, valve or vessel); ``crumbs``, pieces under 2 mm that touch no other linework."""
    ents = [e for e in _entities(d) if e]
    sampled = [_sampled(e)[0] for e in ents]
    if not sampled:
        return {"gap_ends": 0, "crumbs": 0}
    owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(sampled)])
    tree = cKDTree(np.vstack(sampled))

    def nearest_other(points: np.ndarray, i: int) -> np.ndarray:
        dist, idx = tree.query(points, k=12, distance_upper_bound=2.0)
        dist[(idx < len(owner)) & (owner[np.minimum(idx, len(owner) - 1)] == i)] = np.inf
        return dist.min(axis=1)

    touch = STEP / 2 + 0.05
    gaps = crumbs = 0
    for i, e in enumerate(ents[:len(ents) - len(d.symbols)]):  # the symbols are blocks
        if isinstance(e[0], (Circle, Polygon)) or _closed(e):
            continue
        ends = np.array([_ends(e[0])[0], _ends(e[-1])[1]])
        dist = nearest_other(ends, i)
        gaps += int(np.sum((dist > touch) & (dist <= 2.0)))
        crumbs += _length(e) < 2.0 and bool(nearest_other(sampled[i], i).min() > touch)
    return {"gap_ends": gaps, "crumbs": crumbs}


def _ends(q) -> tuple[tuple[float, float], tuple[float, float]]:
    if isinstance(q, Line):
        return (q.x1, q.y1), (q.x2, q.y2)
    return tuple((q.cx + q.r * math.cos(math.radians(t)), q.cy + q.r * math.sin(math.radians(t)))
                 for t in (q.start, q.end))


def _closed(parts: list) -> bool:
    return len(parts) > 1 and math.dist(_ends(parts[0])[0], _ends(parts[-1])[1]) < 1e-6


def _segments(d: Drawing, dashed: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The straight pieces of the solid lines and polylines of a drawing (not of its symbols; with ``dashed``, of the
    dashed ones too) as rows x1, y1, x2, y2, the number of the line or polyline each belongs to, the ends of the open
    ones (rows x, y) and the pen width of each piece."""
    rows, owner, ends, width = [], [], [], []
    for i, q in enumerate(d.lines + d.polylines):
        parts = [p for p in (q.parts() if hasattr(q, "parts") else [q]) if isinstance(p, Line)]
        if q.dashed and not dashed or not parts:
            continue
        rows += [(p.x1, p.y1, p.x2, p.y2) for p in parts]
        owner += [i] * len(parts)
        width += [q.width] * len(parts)
        if not getattr(q, "closed", False):
            ends += [(parts[0].x1, parts[0].y1), (parts[-1].x2, parts[-1].y2)]
    return np.array(rows, float).reshape(-1, 4), np.array(owner, int), np.array(ends, float).reshape(-1, 2), \
        np.array(width, float)


BEND = math.radians(20)  # two pieces of one pen meeting end to end that go on within this of straight are one line


def _bends(seg: np.ndarray, owner: np.ndarray, width: np.ndarray) -> set[frozenset[int]]:
    """Pairs of pieces (rows of seg) that are one line bending: two pieces of different lines and the same pen that
    meet end to end, alone at that point, and go on within BEND of straight, such as the short straight lines a contour
    line or another curve is drawn with. Where they meet is a bend of one line, not a junction."""
    at = defaultdict(list)
    for i, (x1, y1, x2, y2) in enumerate(seg):
        at[(round(x1, 3), round(y1, 3))].append((i, math.atan2(y2 - y1, x2 - x1)))
        at[(round(x2, 3), round(y2, 3))].append((i, math.atan2(y1 - y2, x1 - x2)))
    a, d = seg[:, :2], seg[:, 2:] - seg[:, :2]
    out = set()
    for p, here in at.items():
        if len(here) != 2:
            continue
        (i, u), (j, v) = here
        if owner[i] == owner[j] or width[i] != width[j] or math.pi - abs(math.remainder(u - v, 2 * math.pi)) > BEND:
            continue
        t = np.clip(((p[0] - a[:, 0]) * d[:, 0] + (p[1] - a[:, 1]) * d[:, 1]) / np.maximum((d ** 2).sum(axis=1), 1e-12),
                    0, 1)
        if np.sum(np.hypot(a[:, 0] + t * d[:, 0] - p[0], a[:, 1] + t * d[:, 1] - p[1]) <= 1e-3) == 2:  # nothing else
            out.add(frozenset((i, j)))
    return out


def _meets(seg: np.ndarray, owner: np.ndarray, ends: np.ndarray, dots: list[Circle],
           bends: set[frozenset[int]] = frozenset()) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Where two lines cross, both running on (no end of a line within 1.5 mm, no dot), and where they join: where
    they meet at an end (within 0.5 mm), or at a dot; not where two pieces of one line bend (``bends``, _bends).
    Places with an end in between are neither."""
    crossings, junctions = [], [(c.cx, c.cy) for c in dots]
    tree = cKDTree(ends) if len(ends) else None
    a, d = seg[:, :2], seg[:, 2:] - seg[:, :2]
    for i in range(len(seg)):
        p, r = a[i], d[i]
        denom = r[0] * d[:, 1] - r[1] * d[:, 0]
        ok = (np.abs(denom) > 1e-12) & (owner > owner[i])
        safe = np.where(ok, denom, 1.0)
        t = ((a[:, 0] - p[0]) * d[:, 1] - (a[:, 1] - p[1]) * d[:, 0]) / safe
        u = ((a[:, 0] - p[0]) * r[1] - (a[:, 1] - p[1]) * r[0]) / safe
        for j in np.flatnonzero(ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)):
            if frozenset((i, int(j))) in bends:
                continue
            x = (float(p[0] + t[j] * r[0]), float(p[1] + t[j] * r[1]))
            near = len(tree.query_ball_point(x, 1.5)) if tree is not None else 0
            if tree is not None and tree.query_ball_point(x, 0.5) or any(math.dist(x, (c.cx, c.cy)) <= 1.0 for c in dots):
                junctions.append(x)
            elif not near:
                crossings.append(x)
    return crossings, junctions


def crossings(gt: Drawing, pred: Drawing) -> dict:
    """Lines that cross without joining, and lines that join. Of the places where two solid lines of the reference
    cross and run on (no dot), the share where two lines of the output run through as well, neither ending within
    1.5 mm (crossings kept); of the places where they join (at an end, at a dot; not where a line drawn in short
    straight pieces bends), the share where a line of the output ends, turns (a corner of a closed outline) or a dot is
    within 1 mm (junctions kept). A crossing closed into a junction loses the one, a junction left with a gap the
    other. In the output, dashed lines count as well: a line taken for dashed has the wrong type, but crosses or joins
    as drawn."""
    seg, owner, ends, width = _segments(gt)
    if len(seg) < 2:
        return {}
    across, joins = _meets(seg, owner, ends, [c for c in gt.circles if c.r <= 1.0], _bends(seg, owner, width))
    pseg, powner, pends, _ = _segments(pred, dashed=True)
    if not len(pseg) or not len(pends):
        return {"crossings_kept": 0.0 if across else None, "junctions_kept": 0.0 if joins else None}
    tree = cKDTree(pends)
    end_owner = np.repeat([o for o in dict.fromkeys(powner.tolist())
                           if not getattr((pred.lines + pred.polylines)[o], "closed", False)], 2)
    a, d = pseg[:, :2], pseg[:, 2:] - pseg[:, :2]
    length2 = np.maximum((d ** 2).sum(axis=1), 1e-12)

    def through(x) -> int:  # output lines passing within 0.75 mm of x without an end within 1.5 mm of it
        t = np.clip(((x[0] - a[:, 0]) * d[:, 0] + (x[1] - a[:, 1]) * d[:, 1]) / length2, 0, 1)
        near = set(powner[np.hypot(a[:, 0] + t * d[:, 0] - x[0], a[:, 1] + t * d[:, 1] - x[1]) <= 0.75].tolist())
        ending = set(end_owner[tree.query_ball_point(x, 1.5)].tolist())
        return len(near - ending)

    kept = sum(through(x) >= 2 for x in across)
    corners = cKDTree(np.vstack([pseg[:, :2], pseg[:, 2:]]))  # the ends of every straight piece
    met = sum(bool(corners.query_ball_point(x, 1.0)) or any(math.dist(x, (c.cx, c.cy)) <= 1.0 for c in pred.circles)
              for x in joins)
    return {"crossings_kept": kept / len(across) if across else None,
            "junctions_kept": met / len(joins) if joins else None}


OPENING = (2.0, 60.0)  # mm: a gap this long in a line is an opening, such as a door or a window in a wall


def _along(d: Drawing) -> tuple[cKDTree | None, np.ndarray]:
    """All linework of a drawing (dashed lines and symbols too) as sample points with their direction (mod pi)."""
    flat = d.flat()
    pts, dirs = _sampled(flat.lines + flat.arcs + flat.circles)
    return (cKDTree(pts) if len(pts) else None), dirs


def _covered(pts: np.ndarray, heading: float, tree: cKDTree | None, dirs: np.ndarray, tol: float) -> np.ndarray:
    """Which of the points have linework within ``tol`` mm that runs along ``heading`` (within 20 degrees)."""
    if tree is None or not len(pts):
        return np.zeros(len(pts), bool)
    _, idx = tree.query(pts, k=8, distance_upper_bound=tol)
    diff = np.abs(dirs[np.minimum(idx, len(dirs) - 1)] - heading % np.pi) % np.pi
    return ((idx < len(dirs)) & (np.minimum(diff, np.pi - diff) < math.radians(20))).any(axis=1)


def _gaps(seg: np.ndarray, width: np.ndarray, tree: cKDTree | None) -> list[tuple[np.ndarray, np.ndarray]]:
    """The gaps between the ends of two pieces (rows of seg) of the same pen on one line, OPENING long and clear of all
    linework (``tree``, _along), each as its two ends."""
    a, b = seg[:, :2], seg[:, 2:]
    length = np.hypot(*(b - a).T)
    u = (b - a) / np.maximum(length, 1e-12)[:, None]
    out = []
    for i in np.flatnonzero(length > 0.1):
        side = [np.abs((e[:, 0] - a[i, 0]) * u[i, 1] - (e[:, 1] - a[i, 1]) * u[i, 0]) < 0.05 for e in (a, b)]
        for j in np.flatnonzero((np.arange(len(seg)) > i) & side[0] & side[1] & (width == width[i]) & (length > 0.1)):
            lo, hi = sorted(((a[j] - a[i]) @ u[i], (b[j] - a[i]) @ u[i]))
            gap = (length[i], lo) if lo >= length[i] else (hi, 0.0) if hi <= 0 else None  # j beyond either end of i
            if gap is None or not OPENING[0] <= gap[1] - gap[0] <= OPENING[1]:
                continue
            p, q = a[i] + gap[0] * u[i], a[i] + gap[1] * u[i]
            w = 0.75 * (q - p) / np.hypot(*(q - p))  # clear of the ends of the two pieces and of what they meet there
            inside = _line(*(p + w), *(q - w))
            if tree is None or not np.isfinite(tree.query(inside, distance_upper_bound=0.5)[0]).any():
                out.append((p, q))  # else something lies in the gap: hatching, or a symbol a line is broken for
    return out


def _openings(d: Drawing) -> list[tuple[np.ndarray, np.ndarray]]:
    """Where a wall of the reference is broken off for an opening, such as a door or a window: a gap in one face of
    the wall (_gaps), the same gap in the other face, and at each side of it a line from the one face to the other, the
    jamb, shorter than the gap is wide (a wall is thinner than its doors and windows are wide). Not where another wall
    joins (the face opposite runs on, and the faces of the joining wall lead away from it), nor between strokes that
    happen to lie on one line, such as the ticks of two dimension chains. Each face as the two ends of its gap."""
    seg, _, _, width = _segments(d)
    gaps = _gaps(seg, width, _along(d)[0])
    if not gaps:
        return []
    tree, other = cKDTree(np.vstack([seg[:, :2], seg[:, 2:]])), np.vstack([seg[:, 2:], seg[:, :2]])
    middles = cKDTree([(p + q) / 2 for p, q in gaps])

    def jambed(p: np.ndarray, q: np.ndarray) -> bool:  # a line from each end, both to the ends of the same gap
        for v in (other[k] - p for k in tree.query_ball_point(p, 0.1)):
            if np.hypot(*v) < np.hypot(*(q - p)) and middles.query_ball_point((p + q) / 2 + v, 0.1) and \
                    any(np.hypot(*(other[k] - q - v)) < 0.1 for k in tree.query_ball_point(q, 0.1)):
                return True
        return False

    return [(p, q) for p, q in gaps if jambed(p, q)]


def openings(gt: Drawing, pred: Drawing) -> dict:
    """Of the openings in the walls of the reference (doors and windows, each face, see _openings), the share the
    output keeps: its wall stops at both sides of the gap, an end or a corner of its lines within 1 mm of each end, and
    no output line runs along the middle half of the gap, within 0.5 mm, over at least half of it. A wall closed across
    its door loses one, and so does a wall that is lost."""
    gaps = _openings(gt)
    if not gaps:
        return {}
    pseg = _segments(pred, dashed=True)[0]
    if not len(pseg):
        return {"openings_kept": 0.0}
    corners = cKDTree(np.vstack([pseg[:, :2], pseg[:, 2:]]))  # the ends of every straight piece
    tree, dirs = _along(pred)
    kept = 0
    for p, q in gaps:
        middle = _line(*(p + 0.25 * (q - p)), *(p + 0.75 * (q - p)))
        across = _covered(middle, math.atan2(q[1] - p[1], q[0] - p[0]), tree, dirs, 0.5).mean() >= 0.5
        kept += not across and all(corners.query_ball_point(e, 1.0) for e in (p, q))
    return {"openings_kept": kept / len(gaps)}


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


MARKUP_LAYERS = {"P2C-MARKUP", "P2C-MARKUP-TEXT", "P2C-HIGHLIGHT"}  # where the pipeline puts what was added by hand


def _is_markup(q) -> bool:
    return getattr(q, "kind", None) == MARKUP or q.layer in MARKUP_LAYERS


def content(d: Drawing) -> Drawing:
    """The drawing without what was added by hand (markups are measured on their own)."""
    return replace(d, polylines=[pl for pl in d.polylines if not _is_markup(pl)],
                   lines=[q for q in d.lines if not _is_markup(q)], arcs=[q for q in d.arcs if not _is_markup(q)],
                   polygons=[g for g in d.polygons if not _is_markup(g)], texts=[t for t in d.texts if not _is_markup(t)])


def markups(gt: Drawing, pred: Drawing) -> dict:
    """How well what was added by hand is kept apart from the drawing: of the markup strokes of the reference, the
    share the pipeline put on a markup layer (recall; also of each part, by the layer synth.mark_up puts it on) and
    the share it took for the drawing (leaked; where no line of the drawing lies under them); of what it put on a
    markup layer, the share that is markup (precision) and the length that is not (in mm, also on drawings without
    markups); and the texts of the drawing it read on a markup (phantoms) and the fills it made there."""
    marks = [pl for pl in gt.polylines if pl.kind == MARKUP]
    found = Drawing(pred.width, pred.height, polylines=[pl for pl in pred.polylines if _is_markup(pl)],
                    lines=[q for q in pred.lines if _is_markup(q)])
    m = samples(found.flat())
    g = np.vstack([_sampled(pl.parts())[0] for pl in marks]) if marks else np.zeros((0, 2))
    false = cKDTree(g).query(m)[0] > 1.5 if len(g) and len(m) else np.ones(len(m), bool)
    if not marks:
        return {"markup_false_mm": float(false.sum()) * STEP}
    parts: dict[str, list] = {}
    for pl in marks:
        parts.setdefault((pl.layer or "").rsplit("-", 1)[-1].lower(), []).append(pl)
    drawn = samples(content(gt).flat())
    clear = g[cKDTree(drawn).query(g)[0] > 1.0] if len(drawn) else g  # markup away from the print
    rest = content(pred).flat()
    c = samples(rest)
    tree = cKDTree(g)
    out = {"markup_recall": float(np.mean(cKDTree(m).query(g)[0] <= 1.5)) if len(m) else 0.0,
           "markup_precision": float(np.mean(tree.query(m)[0] <= 1.5)) if len(m) else None,
           "markup_leaked": float(np.mean(cKDTree(c).query(clear)[0] <= 0.5)) if len(c) and len(clear) else 0.0,
           "markup_false_mm": float(false.sum()) * STEP}
    for part, pls in parts.items():
        q = np.vstack([_sampled(pl.parts())[0] for pl in pls])
        out[f"markup_found_{part}"] = float(np.mean(cKDTree(m).query(q)[0] <= 1.5)) if len(m) else 0.0
    known = content(gt).flat().texts
    out["markup_phantom_texts"] = sum(1 for t in rest.texts if tree.query((t.x, t.y))[0] <= 2.0
                                      and not any(_in(k, t) for k in known))
    out["markup_fills"] = sum(1 for p in rest.polygons if p.filled
                              and tree.query(np.mean(p.points, axis=0))[0] <= 3.0)
    return out


def dissolved(d: Drawing) -> Drawing:
    """The drawing with its dimensions taken apart into the strokes and texts they were recognised from, as loose
    lines and texts: the measures of linework and text see them where they are, as before they made a dimension."""
    parts = [q for m in d.dimensions for q in m.parts]
    return replace(d, lines=d.lines + [q for q in parts if isinstance(q, Line)],
                   arcs=d.arcs + [q for q in parts if isinstance(q, Arc)],
                   circles=d.circles + [q for q in parts if isinstance(q, Circle)],
                   polygons=d.polygons + [q for q in parts if isinstance(q, Polygon)],
                   texts=d.texts + [m.text for m in d.dimensions if m.text], dimensions=[])


def dimensions(gt: Drawing, pred: Drawing) -> dict:
    """Dimensions recognised as dimension objects: a reference dimension is found where one of the output has both
    ends of its dimension line within 1 mm of the reference's and its value exactly (none where none is written).
    And the dimensions marked to check, and those on a drawing without dimensions."""
    if not gt.dimensions:
        return {"dimension_false": len(pred.dimensions)} if pred.dimensions else {}

    hit = sum(any(_dimension_match(r, q) for q in pred.dimensions) for r in gt.dimensions)
    ok = sum(any(_dimension_match(r, q) for r in gt.dimensions) for q in pred.dimensions)
    return {"dimension_recall": hit / len(gt.dimensions),
            "dimension_precision": ok / len(pred.dimensions) if pred.dimensions else None,
            "dimension_review": sum(q.uncertain for q in pred.dimensions)}


def _dimension_match(a, b) -> bool:
    """Whether output dimension ``b`` is reference dimension ``a``: both ends of its dimension line within 1 mm and its
    value exactly (none where none is written)."""
    ends = ((a.x1, a.y1), (a.x2, a.y2))
    return a.value == b.value and any(
        math.dist(ends[0], p) <= 1.0 and math.dist(ends[1], q) <= 1.0 for p, q in
        (((b.x1, b.y1), (b.x2, b.y2)), ((b.x2, b.y2), (b.x1, b.y1))))


def _unit(gt: Drawing) -> float:
    """The unit (in mm) the reference's dimensions are written in, where they do not name it: the one most of them
    fit at its scale."""
    rows = [(v, math.dist((m.x1, m.y1), (m.x2, m.y2)) / gt.printed) for m in gt.dimensions if (v := value(m.value))]
    return max((*UNITS.values(), INCH), key=lambda unit: (sum(
        abs(number * (own or unit) / gt.scale - length) <= 0.05 for (number, _, own), length in rows), -unit))


def dimensions_true(gt: Drawing, pred: Drawing) -> dict:
    """Dimensions that dimension the drawing. True to their value: the share of the output dimensions with a value
    whose measured points lie as far apart along the dimension line as the value says, within 1 mm in reality at the
    reference's scale (what measuring the DXF gives). On the object: the share of the measured points of the output
    dimensions that are reference ones that lie within 0.1 mm of the object the reference dimension measures, the
    reference linework its extension line meets first (fit.Linework.ahead; the reference's extension lines start
    about 2 mm off the wall), or that its dimension line ends on where it has no extension line, up to ASIDE beyond
    the end of such a line: the reference draws a wall face up to a pixel of its grid (1 cm) off where its dimensions
    say, so a wall corner moved to its value lies beside the reference's. Ends whose extension line meets nothing
    within reach (a chain of levels beside a section) do not count."""
    if not gt.dimensions or not gt.scale:
        return {}
    unit = _unit(gt)
    true = []
    for m in pred.dimensions:
        if v := value(m.value):
            u = np.array((m.x2 - m.x1, m.y2 - m.y1)) / max(math.dist((m.x1, m.y1), (m.x2, m.y2)), 1e-9)
            measured = abs(float((np.array(m.p2) - np.array(m.p1)) @ u)) / gt.printed * gt.scale
            true.append(abs(measured - v[0] * (v[2] or unit)) <= 1.0)
    work = Linework(gt)
    on = []
    for r in gt.dimensions:
        objects = [_object(work, r, k) for k in (0, 1)]
        for q in (q for q in pred.dimensions if _dimension_match(r, q)):
            flip = math.dist((r.x1, r.y1), (q.x1, q.y1)) > math.dist((r.x1, r.y1), (q.x2, q.y2))
            for k, segs in enumerate(objects):
                p = np.array((q.p2, q.p1)[k] if flip else (q.p1, q.p2)[k], float)
                if len(segs):
                    on.append(bool(work.on(p, segs, 0.1, beyond=ASIDE).any()))
    return {"dimension_true": float(np.mean(true)) if true else None,
            "dimension_on_object": float(np.mean(on)) if on else None}


def _object(work: Linework, r, k: int) -> np.ndarray:
    """The reference segments that end ``k`` of reference dimension ``r`` measures (see dimensions_true)."""
    e, p = np.array(((r.x1, r.y1), (r.x2, r.y2))[k], float), np.array((r.p1, r.p2)[k], float)
    far = float(np.hypot(*(p - e)))
    u = np.array((r.x2 - r.x1, r.y2 - r.y1)) / max(math.dist((r.x1, r.y1), (r.x2, r.y2)), 1e-9)
    at = work.ahead(p, (p - e) / far) if far > 1e-6 else work.across(p, u)
    if at is None or not len(work.seg):
        return np.zeros(0, int)
    every = np.arange(len(work.seg))
    return every[work.on(at, every, 0.01)]


def uncertain(pred: Drawing) -> dict:
    """The share of the texts and symbols the pipeline marked to check."""
    n = len(pred.texts) + len(pred.symbols)
    flagged = sum(t.uncertain for t in pred.texts) + sum(s.doubt is not None for s in pred.symbols)
    return {"uncertain_share": flagged / n if n else None}


def deviations(gt: Drawing, pred: Drawing, ppm: float = 4.0) -> dict:
    """Areas where the recognised linework and the reference differ, as on an overlay of the two: ink of either
    that has none of the other within 1 mm, in patches of at least 2 mm² (texts left out)."""
    from app.plan2cad.render import render

    a = render(gt, ppm, texts=False) < 128
    b = render(replace(pred, width=gt.width, height=gt.height), ppm, texts=False) < 128  # on the same canvas
    k = np.ones((2 * round(ppm) + 1,) * 2, np.uint8)
    off = (a & ~(cv2.dilate(b.astype(np.uint8), k) > 0)) | (b & ~(cv2.dilate(a.astype(np.uint8), k) > 0))
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(off.astype(np.uint8), k), connectivity=8)
    return {"deviation_areas": int(np.sum(stats[1:, cv2.CC_STAT_AREA] >= 2 * ppm * ppm))}


def title_cells(gt: Drawing, pred: Drawing) -> dict:
    """The cells of the reference's title block that the output's title block has (a cell whose box overlaps the
    reference cell's by half of the two together at least: ``title_cells_found``), and those of them read exactly,
    spaces aside, and not marked to check (``title_cells_exact``)."""
    if gt.title_block is None:
        return {}

    def overlap(a, b) -> float:
        w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
        both = w * h if w > 0 and h > 0 else 0.0
        return both / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - both)

    cells = pred.title_block.cells if pred.title_block else []
    found = exact = 0
    for c in gt.title_block.cells:
        best = max(cells, key=lambda q: overlap(c.box, q.box), default=None)
        if best is not None and overlap(c.box, best.box) >= 0.5:
            found += 1
            exact += best.text.split() == c.text.split() and not best.uncertain
    n = len(gt.title_block.cells)
    return {"title_cells_found": found / n, "title_cells_exact": exact / n}


def evaluate(gt: Drawing, pred: Drawing) -> dict:
    """The measures of the output against the reference. Those of linework and texts see the strokes and texts of
    the dimensions where they are; those of the entities of the DXF (objects in one piece, breaks, ends stopping short)
    take a chain of dimensions as one entity with nothing to join, as a symbol is."""
    dims = dimensions(gt, pred) | dimensions_true(gt, pred)
    whole = content(pred)
    gt, pred = dissolved(gt), dissolved(pred)
    marks = markups(gt, pred)
    gt, pred = content(gt), content(pred)
    flat_gt, flat_pred = gt.flat(), pred.flat()
    out = geometry(flat_gt, flat_pred) | circles(flat_gt, flat_pred) | texts(flat_gt, flat_pred) | objects(gt, whole)
    out |= continuity(gt, whole) | symbols(gt, pred) | connectivity(gt, pred) | crossings(gt, pred)
    out |= loose(replace(whole, dimensions=[])) | marks | openings(gt, pred)
    out |= uncertain(pred) | deviations(flat_gt, flat_pred) | pens(flat_gt, flat_pred) | dims
    out |= symbols_apart(gt, pred) | short_strokes(flat_gt, flat_pred) | title_cells(gt, pred)
    out |= line_types(flat_gt, flat_pred)
    out["scale_ok"] = pred.scale == gt.scale if gt.scale else None
    # the print size measured on the sheet (pipeline.printed_size) within 15 % of the share it is printed at
    out["printed_ok"] = abs(pred.printed / gt.printed - 1) <= 0.15
    return out
