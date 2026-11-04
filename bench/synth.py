"""Synthetic drawings with ground truth; `scan` and `photo` (app.plan2cad.simulate) turn them into simulated scans
and phone photos. The drawings carry German labels (room names, title block), like the real plans the pipeline was
first built for; OCR reads them with `deu`.

All drawings are generated as vector primitives first; the raster images are rendered from them,
so every test image comes with an exact reference solution.
"""

from __future__ import annotations

import math
from dataclasses import replace

import cv2
import numpy as np

from app.plan2cad.library import BUILT_IN, FLOOR_PLAN, load, place, scaled, transform
from app.plan2cad.model import (PAPER, Arc, Cell, Circle, Dimension, Drawing, Line, Polygon, Polyline, Text,
                                TitleBlock, Unclassified)
from app.plan2cad.render import text_length
from app.plan2cad.simulate import MARKUP, photo, plot, scan  # noqa: F401 (part of the benchmark's API)

# P&ID, circuit diagram and building drawing symbols, at their printed size
LIBRARY = load() | load(BUILT_IN / "iec-60617") | load(BUILT_IN / "building-symbols")
FIXTURES = load(FLOOR_PLAN)  # real size in mm

ROOMS = ["Wohnen", "Küche", "Bad", "Schlafen", "Kind", "Flur", "Büro", "Essen", "Gäste-WC", "Abstellraum", "Hobby"]


class Pen:
    """Small helper to add primitives to a drawing."""

    def __init__(self, d: Drawing):
        self.d = d

    def line(self, x1, y1, x2, y2, w=0.25, dashed=False, linetype=None):
        """A line; ``dashed`` with a ``linetype`` ("dashed", "dashdot", "dotted") drawn at its usual pattern."""
        self.d.lines.append(Line(float(x1), float(y1), float(x2), float(y2), w, dashed or bool(linetype),
                                 linetype=linetype))

    def poly(self, pts, w=0.25, closed=False, dashed=False, linetype=None):
        pts = list(pts) + ([pts[0]] if closed else [])
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            self.line(x1, y1, x2, y2, w, dashed, linetype)

    def rect(self, x, y, w, h, lw=0.25):
        self.poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], lw, closed=True)

    def text(self, x, y, s, h=2.5, angle=0.0, align="center"):
        if align == "left":
            x += text_length(Text(0, 0, h, s)) / 2
        self.d.texts.append(Text(float(x), float(y), h, s, angle))

    def arc(self, cx, cy, r, start, end, w=0.25):
        self.d.arcs.append(Arc(float(cx), float(cy), r, start, end, w))

    def circle(self, cx, cy, r, w=0.25):
        self.d.circles.append(Circle(float(cx), float(cy), r, w))

    def fill(self, pts):
        self.d.polygons.append(Polygon([(float(x), float(y)) for x, y in pts]))

    def dimension(self, a, b, value, p1, p2, extension):
        """Records a dimension as a reference: from ``a`` to ``b`` on its dimension line, measured from ``p1`` and
        ``p2``. It draws nothing: its lines and text are drawn as lines and a text, so the images stay as they are."""
        self.d.dimensions.append(Dimension(*map(float, (*a, *b)), value, tuple(map(float, p1)),
                                           tuple(map(float, p2)), extension=extension, width=0.18))

    def symbol(self, name, x, y, angle=0.0, scale=1.0, h=2.0, mirror=False, **attributes):
        """A symbol of the library; its attribute texts stand where the library puts them."""
        s = place(LIBRARY[name], float(x), float(y), angle, mirror, scale)
        f = transform(x, y, angle, mirror, scale)
        s.attributes = {tag: Text(*map(float, f(*LIBRARY[name].tags[tag])), h, v) for tag, v in attributes.items()}
        self.d.symbols.append(s)
        return s


def frame(p: Pen, rows: list[list[tuple[str, float]]], tb_w: float = 170, tb_h: float = 40):
    """Sheet frame with a 20 mm filing margin and a title block in the bottom right corner, open at its top and left
    side; its cells (row, column and span in the grid of all its rulings, box, text) are the reference of the title
    block (Drawing.title_block)."""
    W, H = p.d.width, p.d.height
    p.rect(20, 10, W - 30, H - 20, 0.7)
    x0, y0 = W - 10 - tb_w, H - 10 - tb_h
    rh = tb_h / len(rows)
    grid = sorted({round(x0 + tb_w * j / len(row), 6) for row in rows for j in range(len(row) + 1)})
    cells = []
    for i, row in enumerate(rows):
        if i:
            p.line(x0, y0 + i * rh, x0 + tb_w, y0 + i * rh, 0.35)
        cw = tb_w / len(row)
        for j, (s, h) in enumerate(row):
            if j:
                p.line(x0 + j * cw, y0 + i * rh, x0 + j * cw, y0 + (i + 1) * rh, 0.35)
            p.text(x0 + j * cw + 3, y0 + (i + 0.5) * rh, s, h, align="left")
            a, b = (grid.index(round(x0 + tb_w * k / len(row), 6)) for k in (j, j + 1))
            cells.append(Cell(i, a, (x0 + j * cw, y0 + i * rh, x0 + (j + 1) * cw, y0 + (i + 1) * rh), s, cols=b - a))
    p.d.title_block = TitleBlock((x0, y0, x0 + tb_w, y0 + tb_h), cells)


def de(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


def printed_on(d: Drawing, paper: str) -> Drawing:
    """The drawing printed smaller on ``paper``, as a large P&ID is printed on A3 or A4 to take along: everything,
    pens and texts included, shrinks by the same factor."""
    f = min(PAPER[paper][0] / d.width, PAPER[paper][1] / d.height)
    return replace(d.scaled(f), printed=f)


# --------------------------------------------------------------------------- building plan

def building_plan(seed: int, fixtures: bool = False, scale: int = 50, marks: bool = True) -> Drawing:
    """Ground floor plan of a house on A3 at 1:50 (or 1:100) with walls, doors, windows, stairs, dimensions, the axes
    of two outer walls (dash-dot) and, with ``marks``, a north arrow and the markers of a section. With ``fixtures``,
    kitchen, bathroom and toilet get their fixtures from the floor plan library (the plan is otherwise the same)."""
    rng = np.random.default_rng(seed)
    blocked = []  # cm: door swings and stairs, where no fixture may stand
    cm = 10 / scale  # paper mm per real cm
    d = Drawing(*PAPER["A3"], scale=scale)
    p = Pen(d)
    frame(p, [[("Muster GmbH Architekten", 3.0)], [("Neubau Einfamilienhaus", 3.0)],
              [("Grundriss Erdgeschoss", 5.0)],
              [(f"Maßstab 1:{scale}", 2.5), ("Plan-Nr. A-1%02d" % (seed % 100), 2.5), ("16.10.2026", 2.5)]])
    bw, bh, to = int(rng.integers(1000, 1300)), int(rng.integers(780, 980)), 36  # cm
    ox, oy = 62.0, 42.0

    def mm(cx, cy):
        return ox + cx * cm, oy + cy * cm

    grid = np.ones((bh, bw), np.uint8)
    grid[to:bh - to, to:bw - to] = 0
    yh = int(rng.integers(int(bh * 0.42), int(bh * 0.58)))
    grid[yh - 12:yh + 12, to:bw - to] = 1

    def splits(n):
        lo, hi = to + 280, bw - to - 280
        while True:
            xs = sorted(int(x) for x in rng.integers(lo, hi, n))
            if all(b - a > 280 for a, b in zip(xs, xs[1:])):
                return xs

    top_x, bot_x = splits(int(rng.integers(1, 3))), splits(int(rng.integers(1, 3)))
    for x in top_x:
        grid[to:yh, x - 6:x + 6] = 1
    for x in bot_x:
        grid[yh:bh - to, x - 6:x + 6] = 1
    rooms = [(a, to, b, yh - 12) for a, b in zip([to] + [x + 6 for x in top_x], [x - 6 for x in top_x] + [bw - to])]
    rooms += [(a, yh + 12, b, bh - to) for a, b in zip([to] + [x + 6 for x in bot_x], [x - 6 for x in bot_x] + [bw - to])]

    def door_v(x, ya, yb, side):  # door in a vertical wall between ya and yb, swinging to side (+1/-1)
        w = 88
        y0 = int(rng.integers(ya + 25, yb - w - 25))
        grid[y0:y0 + w, x - 6:x + 6] = 0
        top = bool(rng.integers(2))
        xf, yhg = x + 6 * side, (y0 if top else y0 + w)
        blocked.append((min(xf, xf + side * w), y0, max(xf, xf + side * w), y0 + w))
        p.line(*mm(xf, yhg), *mm(xf + side * w, yhg), 0.35)
        start = {(1, True): 0, (1, False): 270, (-1, True): 90, (-1, False): 180}[(side, top)]
        p.arc(*mm(xf, yhg), w * cm, start, start + 90, 0.18)

    def door_h(y, half, xa, xb, side, w=88):  # door in a horizontal wall, swinging down (+1) or up (-1)
        x0 = int(rng.integers(xa + 25, xb - w - 25))
        grid[y - half:y + half, x0:x0 + w] = 0
        left = bool(rng.integers(2))
        yf, xhg = y + half * side, (x0 if left else x0 + w)
        blocked.append((x0, min(yf, yf + side * w), x0 + w, max(yf, yf + side * w)))
        p.line(*mm(xhg, yf), *mm(xhg, yf + side * w), 0.35)
        start = {(1, True): 0, (1, False): 90, (-1, True): 270, (-1, False): 180}[(side, left)]
        p.arc(*mm(xhg, yf), w * cm, start, start + 90, 0.18)
        return x0, x0 + w

    for x in top_x:
        door_v(x, to, yh - 12, 1 if rng.integers(2) else -1)
    for x in bot_x:
        door_v(x, yh + 12, bh - to, 1 if rng.integers(2) else -1)
    walls = sorted({to, bw - to} | {x - 6 for x in top_x + bot_x} | {x + 6 for x in top_x + bot_x})
    gaps = [(a, b) for a, b in zip(walls, walls[1:]) if b - a > 160]
    a, b = gaps[int(rng.integers(len(gaps)))]
    door_h(yh, 12, a, b, 1 if rng.integers(2) else -1)
    front_room = rooms[len(top_x) + 1 + int(rng.integers(len(bot_x) + 1))]
    front = door_h(bh - to // 2, to // 2, front_room[0], front_room[2], -1, w=101)

    def window(axis, pos, a, b):  # window centred in [a, b] on an outer wall
        w = min(int(rng.choice([101, 126, 151, 176, 201])), (b - a) - 80)
        c = (a + b) // 2 + int(rng.integers(-20, 21))
        w0 = c - w // 2
        if axis == "h":
            if pos == bh - to and front[0] - 30 < w0 + w and w0 < front[1] + 30:
                return None
            grid[pos:pos + to, w0:w0 + w] = 0
            for f in (0.4, 0.6):
                p.line(*mm(w0, pos + to * f), *mm(w0 + w, pos + to * f), 0.18)
        else:
            grid[w0:w0 + w, pos:pos + to] = 0
            for f in (0.4, 0.6):
                p.line(*mm(pos + to * f, w0), *mm(pos + to * f, w0 + w), 0.18)
        return w0, w0 + w

    top_windows = [window("h", 0, x0, x1) for x0, _, x1, _ in rooms[:len(top_x) + 1]]
    for x0, _, x1, _ in rooms[len(top_x) + 1:]:
        window("h", bh - to, x0, x1)
    window("v", 0, to, yh - 12)
    window("v", bw - to, yh + 12, bh - to)

    # Stairs centred along the outer wall of the largest room, away from the doors in its side walls.
    big = max(range(len(rooms)), key=lambda i: (rooms[i][2] - rooms[i][0]) * (rooms[i][3] - rooms[i][1]))
    rx0, ry0, rx1, ry1 = rooms[big]
    top_band = ry0 == to
    n = max(3, min(12, (rx1 - rx0 - 200) // 26))
    sx, sy = (rx0 + rx1) // 2 - 13 * n, (ry0 + 10 if top_band else ry1 - 110)
    p.rect(*mm(sx, sy), 26 * n * cm, 100 * cm, 0.35)
    blocked.append((sx, sy, sx + 26 * n, sy + 100))
    for k in range(1, n):
        p.line(*mm(sx + 26 * k, sy), *mm(sx + 26 * k, sy + 100), 0.25)
    p.line(*mm(sx + 13, sy + 50), *mm(sx + 26 * n - 13, sy + 50), 0.25)
    tip = mm(sx + 26 * n - 13, sy + 50)
    p.line(*tip, tip[0] - 2.0, tip[1] - 1.0, 0.25)
    p.line(*tip, tip[0] - 2.0, tip[1] + 1.0, 0.25)
    p.circle(*mm(sx + 13, sy + 50), 0.8, 0.25)
    # Chimney as a filled block (poché) in an outer corner of the building.
    cx0 = to if big != 0 else bw - to - 38
    p.fill([mm(cx0, to), mm(cx0 + 38, to), mm(cx0 + 38, to + 38), mm(cx0, to + 38)])

    contours, _ = cv2.findContours(grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        p.poly([mm(float(x), float(y)) for x, y in c[:, 0, :]], 0.5, closed=True)

    names = list(rng.permutation(ROOMS))
    if fixtures:  # rooms that get fixtures, after the largest one (which holds the stairs)
        others = [i for i in range(len(rooms)) if i != big]
        for i, name in zip(others, ["Küche", "Bad", "Gäste-WC", "Abstellraum"]):
            names[names.index(name)], names[i] = names[i], name
        furnish(p, rooms, names, blocked, mm, cm, np.random.default_rng(seed + 7919))
    for i, (x0, y0, x1, y1) in enumerate(rooms):
        shift = (60 if top_band else -60) if i == big else 0
        x, y = mm((x0 + x1) / 2, (y0 + y1) / 2 + shift)
        p.text(x, y - 3.0, names[i], 3.5)
        p.text(x, y + 3.0, de((x1 - x0) * (y1 - y0) / 1e4) + " m²", 2.5)

    def hchain(xs, y_dim):  # a chain of dimensions; an interval of 50 cm or less has no value written
        for x in xs:
            p.line(*mm(x, -10), *mm(x, y_dim - 8), 0.18)
            x_, y_ = mm(x, y_dim)
            p.line(x_ - 1.0, y_ + 1.0, x_ + 1.0, y_ - 1.0, 0.35)
        p.line(*mm(xs[0], y_dim), *mm(xs[-1], y_dim), 0.18)
        for a, b in zip(xs, xs[1:]):
            if b - a > 50:
                x_, y_ = mm((a + b) / 2, y_dim)
                p.text(x_, y_ - 2.3, de((b - a) / 100), 2.5)
            p.dimension(mm(a, y_dim), mm(b, y_dim), de((b - a) / 100) if b - a > 50 else "", mm(a, -10), mm(b, -10),
                        8 * cm)

    def vchain(ys, x_dim):
        for y in ys:
            p.line(*mm(-10, y), *mm(x_dim - 8, y), 0.18)
            x_, y_ = mm(x_dim, y)
            p.line(x_ - 1.0, y_ + 1.0, x_ + 1.0, y_ - 1.0, 0.35)
        p.line(*mm(x_dim, ys[0]), *mm(x_dim, ys[-1]), 0.18)
        for a, b in zip(ys, ys[1:]):
            if b - a > 50:
                x_, y_ = mm(x_dim, (a + b) / 2)
                p.text(x_ - 2.3, y_, de((b - a) / 100), 2.5, angle=90)
            p.dimension(mm(x_dim, a), mm(x_dim, b), de((b - a) / 100) if b - a > 50 else "", mm(-10, a), mm(-10, b),
                        8 * cm)

    hchain(sorted({0, bw} | {v for w in top_windows if w for v in w}), -60)
    hchain([0, bw], -100)
    vchain([0, yh - 12, yh + 12, bh], -60)
    vchain([0, bh], -100)
    for y in (to / 2, bh - to / 2):  # the axes of the outer walls at the top and the bottom, out to the right
        x_, y_ = mm(bw, y)
        p.line(x_ + 2.0, y_, x_ + 30.0, y_, 0.18, linetype="dashdot")
    if not marks:
        return d
    # North arrow at the top right, and the section A-A cut across the lower rooms, its markers left and right of the
    # building, looking up (symbols of the library, drawn at their printed size)
    p.symbol("NORTH_ARROW", d.width - 40, 40, h=3.5, LETTER="N")
    y_cut = mm(0, (yh + 12 + bh - to) / 2)[1]
    p.symbol("SECTION_MARK", 27, y_cut, 270, h=3.5, LETTER="A")
    p.symbol("SECTION_MARK", mm(bw, 0)[0] + 20, y_cut, 90, h=3.5, mirror=True, LETTER="A")
    return d


FURNITURE = {"Küche": ["KITCHEN_SINK", "HOB"], "Bad": ["BATHTUB", "WC", "WASHBASIN", "SHOWER"],
             "Gäste-WC": ["WC", "WASHBASIN"], "Abstellraum": ["WASHING_MACHINE"]}


def furnish(p: Pen, rooms: list, names: list, blocked: list, mm, cm: float, rng: np.random.Generator):
    """Puts the fixtures of each room against its walls, clear of doors, stairs, the room name and each other. Each
    stands with its back on the wall face, as in a real plan."""
    taken = list(blocked)
    for (x0, y0, x1, y1), name in zip(rooms, names):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        taken.append((cx - 110, cy - 100, cx + 110, cy + 100))  # room name and area
        for fixture in FURNITURE.get(name, []):
            t = FIXTURES[fixture]
            w, depth = (max(q) - min(q) for q in ([v / 10 for q in t.parts for v in (_xs(q))],
                                                  [v / 10 for q in t.parts for v in (_ys(q))]))
            for _ in range(300):
                side = rng.choice(["top", "bottom", "left", "right"])
                along = (x0, x1) if side in ("top", "bottom") else (y0, y1)
                if along[1] - along[0] < w:
                    continue
                a = float(rng.uniform(along[0], along[1] - w))
                box, at, angle = {"top": ((a, y0, a + w, y0 + depth), (a + w / 2, y0), 0),
                                  "bottom": ((a, y1 - depth, a + w, y1), (a + w / 2, y1), 180),
                                  "left": ((x0, a, x0 + depth, a + w), (x0, a + w / 2), 270),
                                  "right": ((x1 - depth, a, x1, a + w), (x1, a + w / 2), 90)}[side]
                if all(box[2] + 5 <= b[0] or b[2] + 5 <= box[0] or box[3] + 5 <= b[1] or b[3] + 5 <= box[1]
                       for b in taken):
                    taken.append(box)
                    p.d.symbols.append(place(scaled(t, cm / 10), *mm(*at), angle, bool(rng.integers(2))))
                    break


def _xs(q) -> list[float]:
    return [q.x1, q.x2] if hasattr(q, "x1") else [q.cx - q.r, q.cx + q.r]


def _ys(q) -> list[float]:
    return [q.y1, q.y2] if hasattr(q, "y1") else [q.cy - q.r, q.cy + q.r]


# --------------------------------------------------------------------------- P&ID

def pid(seed: int, paper: str = "A1") -> Drawing:
    """Piping and instrumentation diagram with repeated process units (vessel, pump, valves, cooler)."""
    rng = np.random.default_rng(seed)
    d = Drawing(*PAPER[paper])
    p = Pen(d)
    frame(p, [[("Muster GmbH Anlagenbau", 3.0)], [("Anlage D-100 Destillation", 3.0)], [("R&I-Fließbild", 5.0)],
              [("Maßstab ohne", 2.5), ("PID-100-%02d" % (seed % 100), 2.5), ("Blatt 1/1", 2.5)]], tb_h=44)
    cols, rows = {"A0": (4, 3), "A1": (3, 2), "A2": (2, 1), "A3": (1, 1)}[paper]
    x_lo, y_lo, x_hi, y_hi = 25, 15, d.width - 12, d.height - 62
    cw, ch = (x_hi - x_lo) / cols, (y_hi - y_lo) / rows
    for k in range(rows * cols):
        ux = x_lo + (k % cols) * cw + (cw - 220) / 2
        uy = y_lo + (k // cols) * ch + (ch - 190) / 2
        unit(p, ux, uy, k + 1, rng)
    return d


VALVES = ["GATE_VALVE", "BALL_VALVE", "BUTTERFLY_VALVE"]


def unit(p: Pen, ux: float, uy: float, n: int, rng: np.random.Generator):
    """One process unit inside a 220 x 190 mm box: feed vessel with safety valve and drain, pump with check valve,
    control valve station with block and bypass valves, an orifice plate, cooler, reducer, off-page connectors and
    instruments. Symbols come from the symbol library, valves of a random type and size."""
    t = f"{n}0"
    vs = float(rng.uniform(0.85, 1.2))  # valve size
    h = 2.5 * vs  # half the length of a valve

    def valve(x, y, vertical=False):
        p.symbol(str(rng.choice(VALVES)), x, y, 90 if vertical else 0, scale=vs)

    vx, vy0, vy1, rv = ux + 45, uy + 46, uy + 104, 14
    p.line(vx - rv, vy0, vx - rv, vy1, 0.5)
    p.line(vx + rv, vy0, vx + rv, vy1, 0.5)
    p.arc(vx, vy0, rv, 180, 360, 0.5)
    p.arc(vx, vy1, rv, 0, 180, 0.5)
    p.text(vx, uy + 10, f"B-{t}1", 3.0)
    p.text(vx, uy + 14.5, rng.choice(["Vorlage", "Puffer", "Sammler"]), 2.5)
    p.line(vx, vy0 - rv, vx, uy + 26 + h, 0.35)  # safety valve blowing off to the right
    p.symbol("SAFETY_VALVE", vx, uy + 26, scale=vs)
    p.line(vx + h, uy + 26, vx + 12, uy + 26, 0.35)
    # Feed from the left, from another sheet
    p.symbol("OFF_PAGE_CONNECTOR", ux + 6.5, vy0 + 8, TEXT=f"{n}A")
    p.line(ux + 13, vy0 + 8, vx - rv, vy0 + 8, 0.5)
    p.symbol("FLOW_ARROW", ux + 20.8, vy0 + 8)
    p.text(ux + 22, vy0 + 3.5, f"DN50-P-{t}10", 2.0)
    # Level control: impulse line to the vessel, signal to the flow controller
    lx, ly = vx + 30, uy + 82
    p.symbol("INSTRUMENT_DCS", lx, ly, FUNCTION="LIC", NUMBER=f"{t}1")
    p.line(lx - 5, ly, vx + rv, ly, 0.25)
    fx, fy = ux + 128, uy + 96
    p.poly([(lx, ly - 5), (lx, uy + 4), (fx, uy + 4), (fx, fy - 5)], 0.25, dashed=True)
    # Suction line with drain and valve to the pump
    px, py, rp = ux + 82, uy + 150, 6.5
    gx = (vx + px - rp) / 2 + 4
    p.line(vx, vy1 + rv, vx, py, 0.5)
    p.poly([(vx, py - 12), (vx - 10, py - 12), (vx - 10, py - 6 - h)], 0.35)
    valve(vx - 10, py - 6, vertical=True)
    p.line(vx - 10, py - 6 + h, vx - 10, py + 2, 0.35)
    p.text(vx - 10, py + 6, "Ablauf", 2.0)
    p.line(vx, py, gx - h, py, 0.5)
    valve(gx, py)
    p.line(gx + h, py, px - rp, py, 0.5)
    p.symbol("FLOW_ARROW", vx + 6.8, py)
    p.text(gx, py + 5, f"DN80-P-{t}11", 2.0)
    p.symbol("PUMP", px, py, h=3.0, TAG=f"P-{t}1")
    p.text(px, py + 19.5, "Pumpe", 2.5)
    # Discharge with check valve and pressure gauge to the control valve station
    dx_, dy_ = px + 16, uy + 124
    cx = dx_ + 8
    p.poly([(px, py - rp), (dx_, py - rp), (dx_, dy_), (cx - h, dy_)], 0.5)
    p.symbol("CHECK_VALVE", cx, dy_, scale=vs)
    p.line(cx + h, dy_, fx - 10 - h, dy_, 0.5)
    p.symbol("INSTRUMENT", px, uy + 128, FUNCTION="PI", NUMBER=f"{t}2")
    p.line(px + 5, uy + 128, dx_, uy + 128, 0.25)
    for vx_ in (fx - 10, fx + 10):  # block valves
        valve(vx_, dy_)
    p.line(fx - 10 + h, dy_, fx - h, dy_, 0.5)
    p.line(fx + h, dy_, fx + 10 - h, dy_, 0.5)
    p.symbol("CONTROL_VALVE", fx, dy_, scale=vs)
    p.poly([(fx - 16, dy_), (fx - 16, dy_ + 10), (fx - h, dy_ + 10)], 0.5)  # bypass
    valve(fx, dy_ + 10)
    p.poly([(fx + h, dy_ + 10), (fx + 16, dy_ + 10), (fx + 16, dy_)], 0.5)
    p.text(fx, dy_ + 15, f"DN50-P-{t}12", 2.0)
    p.symbol("INSTRUMENT_DCS", fx, fy, FUNCTION="FIC", NUMBER=f"{t}3")
    p.line(fx, fy + 5, fx, dy_ - 9.5 * vs, 0.25, dashed=True)
    # Cooler, reducer and outlet to another sheet
    hx, hy, rh = ux + 176, dy_, 9
    ox = ux + 160  # an orifice plate in the pipe, which ends at either side of it (its ports)
    (left, _), (right, _) = sorted(p.symbol("ORIFICE", ox, dy_).ports)
    p.line(fx + 10 + h, dy_, left, dy_, 0.5)
    p.line(right, dy_, hx - rh, dy_, 0.5)
    p.symbol("FLOW_ARROW", fx + 22.8, dy_)
    p.symbol("HEAT_EXCHANGER", hx, hy, h=3.0, TAG=f"W-{t}1")
    p.line(hx - 6, hy + 6.7, hx - 6, uy + 176, 0.35)
    p.line(hx + 6, hy - 6.7, hx + 6, uy + 100, 0.35)
    p.text(hx - 6, uy + 180, "KW", 2.0)
    p.text(hx + 6, uy + 96, "KWR", 2.0)
    p.text(hx + 13, hy + 15.5, "Kühler", 2.5)
    rx = hx + 16
    p.line(hx + rh, hy, rx - 2 * vs, hy, 0.5)
    p.symbol("REDUCER", rx, hy, scale=vs)
    p.line(rx + 2 * vs, hy, ux + 207, hy, 0.5)
    p.symbol("FLOW_ARROW", ux + 196.8, hy)
    p.symbol("OFF_PAGE_CONNECTOR", ux + 213.5, hy, TEXT=f"{n}B")
    tx, ty = ux + 202, uy + 136
    p.symbol("INSTRUMENT", tx, ty, FUNCTION="TI", NUMBER=f"{t}4")
    p.line(tx, ty - 5, tx, hy, 0.25)


# --------------------------------------------------------------------------- P&ID with the details of printed ones

FLANGE_GAP, FLANGE_BAR = 1.0, 3.4  # mm: the two bars of a flange across a pipe, as an A3 print shows them
HAIRLINE = 0.1  # mm: the pen of signal lines and their marks, which a print shows grey on a scan of 300 dpi
# A bubble in a pipe whose flanges stand against its ring: its size against the library's, the mm its number stands
# below its middle and the length of the bars, which reach down beside the number
FLUSH, FLUSH_NUMBER, FLUSH_BAR = 0.95, 1.7, 4.4


def pid_details(seed: int, details: list | None = None, pen_marks: bool = False, flush: bool = False) -> Drawing:
    """P&ID on A3 in ISA style with the details that printed sheets carry and the other P&IDs lack: flanges (two short
    bars across the pipe, 1 mm apart) at the valves and on each side of the instrument bubbles that sit in a pipe, ball
    and gate valves lying and standing between them, a control valve with its actuator, line labels in rounded frames
    in the pipes (lying and standing), line numbers between two flanges, reducers with their sizes (3"x2"), spec breaks,
    an electric signal line (thin, with small zigzags), a capillary line (thin, with small crosses), dashed signal
    lines, interlock diamonds (function, number and letter), boxes where lines end, an equipment tag in a box, a legend
    under a double rule (two long lines 1 mm apart), and codes and line numbers that begin with a letter and hold
    noughts (X1-01234-5678, P-40208). Valves, bubbles, reducers, flow arrows and the orifice plate are symbols of the
    library; flanges, frames, boxes, diamonds and the marks of the signal lines are linework of no library. Where
    ``details`` is given, it gets (kind, box) for each of those, the box (x0, y0, x1, y1) in mm, to tell what the
    recognition put on them. With ``pen_marks``, the signal lines and their marks are drawn in the pen of the thin
    lines, not a hairline, and small arrowheads stand where signal lines end at an instrument, a box or an
    interlock, as other prints draw them. ``flush``: the bubbles in a pipe are drawn a little smaller than the
    library's and lettered so, their number nearer the middle, with the bars of their flanges against the ring and
    reaching down beside the number, as some prints show them."""
    rng = np.random.default_rng(seed)
    d = Drawing(*PAPER["A3"])
    p = Pen(d)
    frame(p, [[("Muster GmbH Anlagenbau", 3.0)], [("Anlage X1 Dosierung", 3.0)], [("R&I-Fließbild", 5.0)],
              [("Maßstab ohne", 2.5), (f"X1-0{1233 + seed % 100}-5678", 2.5), ("Blatt 1/1", 2.5)]], tb_h=44)

    def number(digits: int = 5) -> str:  # plant numbering with noughts: 40208, 0107
        return (f"{rng.integers(1, 5)}0{rng.integers(1, 3)}0{rng.integers(1, 10)}" if digits == 5
                else f"0{rng.integers(1, 3)}0{rng.integers(1, 10)}")

    def note(kind, x0, y0, x1, y1):
        if details is not None:
            details.append((kind, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))))

    def xy(at, s, vertical):  # a place on a pipe: ``s`` along it, ``at`` across
        return (at, s) if vertical else (s, at)

    def pipe(at, a, b, gaps=(), vertical=False):
        """A pipe from a to b, broken where a symbol or a frame stands in it (``gaps``: from, to along it)."""
        s = a
        for g0, g1 in sorted(g for g in gaps if g) + [(b, b)]:
            if g0 > s:
                p.line(*xy(at, s, vertical), *xy(at, g0, vertical), 0.5)
            s = max(s, g1)

    def flange(at, s, vertical=False, bar=FLANGE_BAR):
        for u in (s - FLANGE_GAP / 2, s + FLANGE_GAP / 2):
            p.line(*xy(at - bar / 2, u, vertical), *xy(at + bar / 2, u, vertical), 0.25)
        note("flange", *xy(at - bar / 2, s - FLANGE_GAP / 2, vertical), *xy(at + bar / 2, s + FLANGE_GAP / 2, vertical))

    def valve(name, at, s, vertical=False):
        """A valve between two flanges, drawn small against the lettering as on many sheets; its gap in the pipe."""
        vs = float(rng.uniform(0.7, 0.85))
        p.symbol(name, *xy(at, s, vertical), 90 if vertical else 0, scale=vs)
        for side in (-1, 1):
            flange(at, s + side * (2.5 * vs + 0.8 + FLANGE_GAP / 2), vertical)
        return s - 2.5 * vs, s + 2.5 * vs

    def control_valve(at, s):
        """A control valve with its actuator, between two flanges, set by a controller above it."""
        gap = valve("CONTROL_VALVE", at, s)
        p.symbol("INSTRUMENT_DCS", s, at - 26, FUNCTION="FIC", NUMBER=number())
        p.line(s, at - 21, s, at - 9.5 * (gap[1] - gap[0]) / 5, 0.25, dashed=True)  # to the top of the actuator
        arrowhead(s, at - 9.5 * (gap[1] - gap[0]) / 5, 0, 1)
        return gap

    def inline(function, at, s):
        """An instrument bubble in a lying pipe, as a flow meter sits in it, with a flange on each side: 0.5 mm from
        the ring, or touching it from outside where ``flush``."""
        if not flush:
            p.symbol("INSTRUMENT", s, at, FUNCTION=function, NUMBER=number())
            for side in (-1, 1):
                flange(at, s + side * 6.0)
            return s - 5, s + 5
        bubble = p.symbol("INSTRUMENT", s, at, scale=FLUSH, h=2.0 * FLUSH, FUNCTION=function, NUMBER=number())
        bubble.attributes["NUMBER"].y = at + FLUSH_NUMBER
        for side in (-1, 1):  # the inner bar's pen against the ring's
            flange(at, s + side * (5 * FLUSH + 0.3 + FLANGE_GAP / 2), bar=FLUSH_BAR)
        return s - 5 * FLUSH, s + 5 * FLUSH

    def rounded(x0, y0, x1, y1, r=1.2):
        for a, b in (((x0 + r, y0), (x1 - r, y0)), ((x0 + r, y1), (x1 - r, y1)), ((x0, y0 + r), (x0, y1 - r)),
                     ((x1, y0 + r), (x1, y1 - r))):
            p.line(*a, *b, 0.25)
        for cx, cy, start in ((x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90)):
            p.arc(cx, cy, r, start, start + 90)

    def label(text, at, s, vertical=False):
        """A line label in a rounded frame that stands in the pipe."""
        half = text_length(Text(0, 0, 2.0, text)) / 2 + 1.6
        x, y = xy(at, s, vertical)
        p.text(x, y, text, 2.0, angle=90 if vertical else 0)
        hx, hy = xy(2.2, half, vertical)
        rounded(x - hx, y - hy, x + hx, y + hy)
        note("label", x - hx, y - hy, x + hx, y + hy)
        return s - half, s + half

    def reducer(sizes, at, s, vertical=False):
        vs = float(rng.uniform(0.9, 1.1))
        x, y = xy(at, s, vertical)
        p.symbol("REDUCER", x, y, 90 if vertical else 0, scale=vs)
        if vertical:
            p.text(x + 4.5, y, sizes, 2.0, align="left")
        else:
            p.text(x, y - 4.5, sizes, 2.0)
        return s - 2 * vs, s + 2 * vs

    def arrow(at, s, vertical=False):
        p.symbol("FLOW_ARROW", *xy(at, s, vertical), 90 if vertical else 0)

    def spec_break(at, s, left, right):
        """A bar across the pipe where its specification changes, with an arrow and the class on each side."""
        p.line(s, at - 4, s, at + 4, 0.35)
        for side, name in ((-1, left), (1, right)):
            y, tip = at - 3.0, s + side * 4.8
            p.line(s + side * 0.8, y, tip - side * 1.2, y, 0.25)
            p.fill([(tip, y), (tip - side * 1.4, y - 0.6), (tip - side * 1.4, y + 0.6)])
            p.text(s + side * 9.5, y, name, 2.0)
        note("spec break", s - 13, at - 4, s + 13, at + 4)

    def numbered(at, s, text):
        """A line number above the pipe between two flanges, as a spool is drawn."""
        for side in (-1, 1):
            flange(at, s + side * 11)
        p.text(s, at - 2.3, text, 2.0)

    def box(x, y, text, w, h, size=2.5):
        p.rect(x - w / 2, y - h / 2, w, h, 0.35)
        p.text(x, y, text, size)
        note("box", x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    def interlock(x, y, function, letter):
        """An interlock: a diamond with its function, number and letter in three rows."""
        p.poly([(x, y - 8), (x + 7, y), (x, y + 8), (x - 7, y)], 0.25, closed=True)
        for dy, s in ((-4, function), (0, number(4)), (4, letter)):
            p.text(x, y + dy, s, 2.0)
        note("diamond", x - 7, y - 8, x + 7, y + 8)

    marks = 0.25 if pen_marks else HAIRLINE

    def arrowhead(x, y, ux, uy):
        """A small open arrowhead with its tip where a signal line ends (pointing ux, uy), with pen_marks only."""
        if pen_marks:
            for side in (-1, 1):
                p.line(x, y, x - 1.2 * ux - side * 0.45 * uy, y - 1.2 * uy + side * 0.45 * ux, 0.25)
            note("arrowhead", x - 1.3, y - 1.3, x + 1.3, y + 1.3)

    def marked(x0, y0, x1, y1, kind, step):
        """A thin signal line with its marks at even spacing: small zigzags across it ("zigzag", electric) or small
        crosses on it ("cross", capillary)."""
        p.line(x0, y0, x1, y1, marks)
        n = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / n, (y1 - y0) / n
        shape = [[(-0.9, 0), (-0.45, 1.1), (0.45, -1.1), (0.9, 0)]] if kind == "zigzag" else \
            [[(-0.8, -0.8), (0.8, 0.8)], [(-0.8, 0.8), (0.8, -0.8)]]
        for s in np.arange(step / 2, n - step / 4, step):
            cx, cy = x0 + ux * s, y0 + uy * s
            for stroke in shape:
                p.poly([(cx + a * ux - c * uy, cy + a * uy + c * ux) for a, c in stroke], marks)
            note(kind, cx - 1.1, cy - 1.1, cx + 1.1, cy + 1.1)

    def legend(x, y):
        """A small legend in a box: its heading over a double rule, then each kind of line with its name."""
        p.rect(x, y, 115, 66, 0.35)
        p.text(x + 4, y + 6, "Legende", 3.0, align="left")
        for dy in (10.5, 11.5):
            p.line(x + 2, y + dy, x + 113, y + dy, 0.25)
        note("double rule", x + 2, y + 10.5, x + 113, y + 11.5)
        names = ["Prozessleitung", "Signal elektrisch", "Kapillare", "Wirkleitung", "Mantelrohr", "Flansch"]
        for k, name in enumerate(names):
            yy = y + 18 + 8 * k
            p.text(x + 36, yy, name, 2.5, align="left")
            if k == 1:
                marked(x + 5, yy, x + 30, yy, "zigzag", 8.0)
            elif k == 2:
                marked(x + 5, yy, x + 30, yy, "cross", 6.0)
            elif k == 4:  # a jacketed pipe: two lines 1 mm apart
                for dy in (-0.5, 0.5):
                    p.line(x + 5, yy + dy, x + 30, yy + dy, 0.35)
                note("double rule", x + 5, yy - 0.5, x + 30, yy + 0.5)
            else:
                p.line(x + 5, yy, x + 30, yy, 0.25 if k == 3 else 0.5, dashed=k == 3)
                if k == 5:
                    flange(yy, x + 17.5)

    # The feed from the left to the vessel, and two branches down to the return line
    y = 50.0
    spec_break(y, 38, "A1A", "B2C")
    p.text(28, y + 5, "Dosierung", 2.5, align="left")
    arrow(y, 165)
    pipe(y, 25, 262, [valve("GATE_VALVE", y, 56), inline("FT", y, 96), reducer('3"x2"', y, 118),
                      label(f'2"-P-{number()}', y, 142), valve("BALL_VALVE", y, 182), control_valve(y, 208)])
    pipe(75, y, 120, [valve("BALL_VALVE", 75, 66, True), label(f'2"-P-{number()}', 75, 87, True),
                      valve("GATE_VALVE", 75, 105, True)], vertical=True)
    arrow(228, 95, True)
    pipe(228, y, 120, [valve("BALL_VALVE", 228, 75, True), reducer('2"x1"', 228, 108, True)], vertical=True)
    # The return line, from another plant (a code) to the second branch
    y = 120.0
    p.text(28, y - 4, f"X1-0{1233 + seed % 100}-5678", 2.0, align="left")
    numbered(y, 99, f"P-{number()}")
    numbered(y, 195, number())
    (left, _), (right, _) = sorted(p.symbol("ORIFICE", 150, y).ports)
    pipe(y, 25, 228, [inline("FI", y, 130), (left, right), valve("GATE_VALVE", y, 168)])
    # The vessel with its tag, a level transmitter on a capillary, an electric signal to its controller and the
    # interlock to the control system
    p.rect(262, 32, 28, 50, 0.5)
    note("box", 262, 32, 290, 82)
    p.text(276, 57, "Vorlage", 2.5)
    box(276, 24, "B-501", 18, 7, 3.0)
    p.symbol("INSTRUMENT", 323, 45, FUNCTION="LT", NUMBER=number())
    marked(290, 45, 318, 45, "cross", 6.0)
    arrowhead(318, 45, 1, 0)
    marked(323, 50, 323, 85, "zigzag", 10.0)
    p.symbol("INSTRUMENT_DCS", 323, 90, FUNCTION="LIC", NUMBER=number())
    p.line(328, 90, 346, 90, 0.25, dashed=True)
    arrowhead(346, 90, 1, 0)
    interlock(353, 90, "I", "A")
    p.line(360, 90, 372, 90, 0.25, dashed=True)
    arrowhead(372, 90, 1, 0)
    box(382, 90, "SPS", 20, 10)
    # The vessel's outlet, down and to another sheet
    x = 276.0
    pipe(x, 82, 150, [valve("GATE_VALVE", x, 98, True), label(f'3"-P-{number()}', x, 120, True)], vertical=True)
    arrow(x, 140, True)
    p.line(x, 150, 300, 150, 0.5)
    p.symbol("OFF_PAGE_CONNECTOR", 306.5, 150, TEXT="4A")
    p.text(306.5, 157, "Kondensat", 2.5)
    # Instruments on signal lines: an electric one and a capillary to the control system, a second interlock
    p.symbol("INSTRUMENT", 40, 165, FUNCTION="PDT", NUMBER=number())
    marked(45, 165, 150, 165, "zigzag", 12.0)
    arrowhead(150, 165, 1, 0)
    box(161, 165, "SPS", 22, 10)
    p.line(172, 165, 188, 165, 0.25, dashed=True)
    arrowhead(188, 165, 1, 0)
    interlock(195, 165, "I", "B")
    p.symbol("INSTRUMENT", 40, 185, FUNCTION="PT", NUMBER=number())
    marked(45, 185, 150, 185, "cross", 6.0)
    p.symbol("INSTRUMENT_DCS", 155, 185, FUNCTION="PIC", NUMBER=number())
    p.poly([(160, 185), (195, 185), (195, 173)], 0.25, dashed=True)
    # A second line with all the kinds of valve between flanges, to another sheet
    y = 215.0
    spec_break(y, 262, "B2C", "A1A")
    arrow(y, 325)
    (left, _), (right, _) = sorted(p.symbol("OFF_PAGE_CONNECTOR", 393.5, y, TEXT="5B").ports)
    pipe(y, 150, left, [valve("BALL_VALVE", y, 170), label(f'4"-P-{number()}', y, 200), inline("FT", y, 235),
                        valve("GATE_VALVE", y, 285), reducer('4"x3"', y, 305), valve("BALL_VALVE", y, 345)])
    legend(25, 205)
    return d


def slashed(d: Drawing) -> Drawing:
    """The drawing lettered with slashed noughts, as many CAD fonts and plotters letter them: a stroke across each 0
    of its texts, its symbols' attributes and its dimension values. For the image only: the reference keeps its texts,
    in which a slashed nought is a 0."""
    from PIL import ImageFont

    from app.plan2cad.render import CAP, FONT

    px = 200
    font = ImageFont.truetype(str(FONT), px)
    nl, _, nr, _ = font.getbbox("0", anchor="ls")
    texts = d.texts + [t for s in d.symbols for t in s.attributes.values()] + [m.text for m in d.dimensions if m.text]
    strokes = []
    for t in texts:
        f = t.height / CAP / px  # mm per font unit
        left, _, right, _ = font.getbbox(t.text, anchor="ls")
        a = math.radians(t.angle)
        ux, uy, vx, vy = math.cos(a), -math.sin(a), math.sin(a), math.cos(a)  # along the text and down its letters
        for i in (i for i, c in enumerate(t.text) if c == "0"):
            end = font.getbbox(t.text[:i + 1], anchor="ls")[2]
            u = (end - (nr - nl) / 2 - (left + right) / 2) * f  # the nought's middle from the text's middle
            du, dv = 0.3 * (nr - nl) * f, 0.36 * t.height  # from low left to high right inside it
            strokes.append(Line(t.x + (u - du) * ux + dv * vx, t.y + (u - du) * uy + dv * vy,
                                t.x + (u + du) * ux - dv * vx, t.y + (u + du) * uy - dv * vy, 0.11 * t.height))
    return replace(d, lines=d.lines + strokes)


# --------------------------------------------------------------------------- circuit diagram

def schematic(seed: int) -> Drawing:
    """Circuit diagram on A3 after IEC 60617: current paths between two supply rails, each with a fuse, a circuit
    breaker or a motor protection switch, contacts and push buttons in series and in parallel branches (junction dots
    where they branch off), a relay coil or a lamp, and in every other path a terminal; device tags, path numbers,
    wire numbers and the contacts of each relay under it. The rail M is earthed. Two cross connections and the dashed
    linkage of a push button run across other paths without joining them."""
    rng = np.random.default_rng(seed)
    d = Drawing(*PAPER["A3"])
    p = Pen(d)
    frame(p, [[("Muster GmbH Elektrotechnik", 3.0)], [("Steuerung Förderband 2", 3.0)], [("Stromlaufplan", 5.0)],
              [("Blatt 3/12", 2.5), ("E-%03d" % (seed % 1000), 2.5), ("24 V DC", 2.5)]])
    top, bottom, x0, step = 40.0, 205.0, 55.0, 30.0
    n = int(rng.integers(8, 11))
    xs = [x0 + k * step for k in range(n)]
    p.line(x0 - 20, top, xs[-1] + 15, top, 0.5)
    p.line(x0 - 20, bottom, xs[-1] + 15, bottom, 0.5)
    p.text(x0 - 26, top, "L+", 2.5)
    p.text(x0 - 26, bottom, "M", 2.5)
    wire, relays = 1, []

    def device(name, x, y, tag):
        """Places the device and returns where the wire ends above and below it (its ports)."""
        return sorted(py for _, py in p.symbol(name, x, y, TAG=tag).ports)

    for k, x in enumerate(xs):
        p.text(x, top - 8, str(k + 1), 2.5)
        p.circle(x, top, 0.6, 0.6)  # junction dot on the rail
        p.circle(x, bottom, 0.6, 0.6)
        ys = [top]
        guard = str(rng.choice(["FUSE", "FUSE", "CIRCUIT_BREAKER", "MOTOR_PROTECTION"]))
        ys += device(guard, x, top + 14, f"-{'Q' if guard == 'MOTOR_PROTECTION' else 'F'}{k + 1}")
        y = top + 40
        for j in range(int(rng.integers(1, 3))):
            name = str(rng.choice(["NO_CONTACT", "NC_CONTACT"] + (["PUSH_BUTTON"] if j == 0 else [])))
            ys += device(name, x, y, f"-{'S' if j == 0 else 'K'}{int(rng.integers(1, 9))}")
            y += 25
        if k % 3 == 1:  # a contact in parallel: a branch with junction dots where it leaves and joins
            bx, ya, yb = x + 12, top + 32, y - 22
            above, below = sorted(py for _, py in LIBRARY["NO_CONTACT"].ports)
            p.poly([(x, ya), (bx, ya), (bx, (ya + yb) / 2 + above)], 0.35)
            device("NO_CONTACT", bx, (ya + yb) / 2, f"-K{k + 1}")
            p.poly([(bx, (ya + yb) / 2 + below), (bx, yb), (x, yb)], 0.35)
            p.circle(x, ya, 0.6, 0.6)
            p.circle(x, yb, 0.6, 0.6)
        load_ = "COIL" if k % 4 != 3 else "LAMP"
        yc = bottom - 22
        tag = f"-K{k + 1}" if load_ == "COIL" else f"-H{k + 1}"
        ys += device(load_, x, yc, tag)
        if k % 2 == 0:  # a terminal on the way back to the rail
            ys += device("TERMINAL", x, bottom - 9, f"-X{k + 1}")
        ys += [bottom]
        for a, b in zip(ys[::2], ys[1::2]):
            p.line(x, a, x, b, 0.35)
            if b - a > 12:
                p.text(x + 3.5, (a + b) / 2, str(wire), 2.0, angle=90)
                wire += 1
        if load_ == "COIL":
            relays.append((x, tag))
            p.text(x, bottom + 6, f"{int(rng.integers(1, 9))}.{int(rng.integers(1, 9))}", 2.0)
    # The rail M earthed at its end
    xe = xs[-1] + 12
    p.circle(xe, bottom, 0.6, 0.6)
    p.line(xe, bottom, xe, bottom + 12, 0.35)
    p.symbol("PROTECTIVE_EARTH", xe, bottom + 15, TAG="PE")
    # A cross connection from one path to another that runs across the paths between without joining them
    a, b = sorted(rng.choice(range(1, n - 1), 2, replace=False))
    if b - a < 2:
        b = min(n - 1, a + 2)
    yx = top + 30
    p.line(xs[a], yx, xs[b], yx, 0.35)
    p.circle(xs[a], yx, 0.6, 0.6)
    p.circle(xs[b], yx, 0.6, 0.6)
    # A second one, lower down, across most paths
    yx = top + 77
    p.line(xs[1], yx, xs[n - 2], yx, 0.35)
    p.circle(xs[1], yx, 0.6, 0.6)
    p.circle(xs[n - 2], yx, 0.6, 0.6)
    # The linkage of a push button to its contact, dashed across the next path
    k = int(rng.integers(0, n - 2))
    p.line(xs[k] - 1.1, top + 40, xs[k + 2] - 1.1, top + 40, 0.25, dashed=True)
    return d


# --------------------------------------------------------------------------- building section

def section(seed: int, scale: int = 50, terminator: str = "tick") -> Drawing:
    """Section through a two-storey house on A3: cut walls and slabs with their material hatching (constructed
    hatching, part of the drawing), the ground with its earth hatching, a pitched roof, windows, levels with their
    marks (±0,00, +2,75, ...), a dimension chain, the axes of the walls (dash-dot) with their circles at the top, room
    names and the ground as it was (dotted). The dimension chain ends its dimensions with oblique ticks, or with filled
    arrowheads ("arrow"; outside where an interval is too short for two) or dots ("dot"). The axis circles are symbols
    of the library; the level marks are symbols of none, listed in the reference's ``unclassified``."""
    rng = np.random.default_rng(seed)
    cm = 10 / scale
    d = Drawing(*PAPER["A3"], scale=scale)
    p = Pen(d)
    frame(p, [[("Muster GmbH Architekten", 3.0)], [("Neubau Einfamilienhaus", 3.0)], [("Schnitt A-A", 5.0)],
              [(f"Maßstab 1:{scale}", 2.5), ("Plan-Nr. A-2%02d" % (seed % 100), 2.5), ("16.10.2026", 2.5)]])
    width = int(rng.integers(800, 1100))  # cm
    storey, slab, wall = 275, 20, 36
    x0, y0 = 90.0, 195.0  # the outer face of the left wall, level ±0,00 (top of the ground floor slab)

    def at(cx, cy):  # cm right of the left wall and above ±0,00 to sheet mm
        return x0 + cx * cm, y0 - cy * cm

    def hatch(xa, ya, xb, yb, gap, angle=45):  # constructed hatching of a rectangle given in cm
        (ax, ay), (bx, by) = at(xa, ya), at(xb, yb)
        lx, hx, ly, hy = min(ax, bx), max(ax, bx), min(ay, by), max(ay, by)
        p.rect(lx, ly, hx - lx, hy - ly, 0.5)
        t = math.tan(math.radians(angle))
        for c in np.arange(lx - (hy - ly) / t + gap, hx, gap):  # lines y = ly + (x - c) * t, clipped to the box
            xa_, xb_ = max(lx, c), min(hx, c + (hy - ly) / t)
            if xb_ - xa_ > 0.3:
                p.line(xa_, hy - (xa_ - c) * t, xb_, hy - (xb_ - c) * t, 0.18)

    gl = -int(rng.integers(30, 90))  # ground level, cm
    p.line(x0 - 40, at(0, gl)[1], x0 + width * cm + 40, at(0, gl)[1], 0.5)
    for gx in np.arange(x0 - 38, x0 + width * cm + 38, 4.0):  # earth: short strokes under the ground line
        gy = at(0, gl)[1]
        p.line(gx, gy + 0.8, gx + 2.0, gy + 2.8, 0.18)
    hatch(-10, -slab - 40, width + 10, -slab, 1.5)  # foundation slab
    for k in range(2):  # slabs over the ground floor and the first floor
        top_ = (k + 1) * storey
        hatch(0, top_ - slab, width, top_, 1.5)
    hatch(0, -slab, width, 0, 1.5)
    for xa in (0, width - wall):  # outer walls, masonry hatching
        hatch(xa, 0, xa + wall, 2 * storey - slab, 2.5)
    ridge = 2 * storey + int(rng.integers(180, 320))
    p.poly([at(-40, 2 * storey - 30), at(width / 2, ridge), at(width + 40, 2 * storey - 30)], 0.5)
    p.poly([at(-40, 2 * storey - 52), at(width / 2, ridge - 22), at(width + 40, 2 * storey - 52)], 0.5)
    for k in range(2):  # a window in each wall and storey: its frame and the glass line, cut
        for xa in (0, width - wall):
            (wx, wy), _ = at(xa, k * storey + 90), None
            p.rect(wx, wy - 135 * cm, wall * cm, 135 * cm, 0.35)
            p.line(wx + wall * cm / 2, wy - 135 * cm, wx + wall * cm / 2, wy, 0.18)
    levels = [gl, 0, storey, 2 * storey]
    for lv in levels:  # level marks at the left: a triangle on a short line, the level beside it
        lx, ly = at(-120, lv)
        p.line(lx - 8, ly, lx + 6, ly, 0.25)
        p.poly([(lx, ly), (lx - 1.6, ly - 2.4), (lx + 1.6, ly - 2.4)], 0.25, closed=True)
        # a symbol, but one of no library: where the output marks a symbol not classified, it is right (the box of the
        # triangle, as its line is the level's own, at the ground the ground line itself)
        d.unclassified.append(Unclassified((lx - 1.6, ly - 2.4, lx + 1.6, ly), "level mark"))
        p.text(lx - 10, ly - 3.5, "±0,00" if lv == 0 else ("+" if lv > 0 else "-") + de(abs(lv) / 100), 2.5,
               align="left")
    dx = at(width + 110, 0)[0]  # dimension chain at the right
    marks = [gl, 0, storey - slab, storey, 2 * storey - slab, 2 * storey]
    p.line(dx, at(0, marks[0])[1], dx, at(0, marks[-1])[1], 0.18)
    for k, m in enumerate(marks):
        y = at(0, m)[1]
        if terminator == "tick":
            p.line(dx - 1.0, y + 1.0, dx + 1.0, y - 1.0, 0.35)
        elif terminator == "dot":
            p.fill([(dx + 0.6 * math.cos(a), y + 0.6 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16, False)])
        else:  # an arrowhead into each dimension that ends here, or from outside where it is too short for two
            for other in marks[k - 1:k] + marks[k + 1:k + 2]:
                side = 1 if at(0, other)[1] > y else -1
                if abs(at(0, other)[1] - y) < 6:
                    side = -side
                p.fill([(dx, y), (dx - 0.4, y + 2.5 * side), (dx + 0.4, y + 2.5 * side)])
        p.line(dx - 3, y, dx + 3, y, 0.18)
    for a, b in zip(marks, marks[1:]):
        if b - a >= 40:
            p.text(dx - 2.5, (at(0, a)[1] + at(0, b)[1]) / 2, de((b - a) / 100), 2.5, angle=90)
        ya, yb = at(0, a)[1], at(0, b)[1]  # the lowest extension line lies in the ground line
        p.dimension((dx, ya), (dx, yb), de((b - a) / 100) if b - a >= 40 else "", (dx - 3, ya), (dx - 3, yb), 3.0)
    for i, xa in enumerate((wall / 2, width - wall / 2)):  # axes with their circles above the ridge
        ax_, ay_ = at(xa, ridge + 60)
        p.line(ax_, ay_ + 5, ax_, at(0, gl)[1] + 6, 0.18, linetype="dashdot")
        p.symbol("AXIS", ax_, ay_, h=3.5, AXIS="AB"[i])
    for k, name in enumerate(rng.permutation(["Wohnen", "Essen", "Schlafen", "Kind"])[:2]):
        x, y = at(width / 2, k * storey + 130)
        p.text(x, y, str(name), 3.5)
    gx, gy = at(width, gl + 15)  # the ground as it was, dotted, beside the house up to the dimension chain
    p.line(gx + 3, gy, gx + 17, gy, 0.25, linetype="dotted")
    return d


# --------------------------------------------------------------------------- site plan

def site_plan(seed: int) -> Drawing:
    """Site plan on A3 at 1:500: the plot's boundary (dash-dot) with its parcel number, the neighbouring plots, the
    house turned to the street with its hatching, the curved street with its name along it, contour lines with their
    heights along them, spot heights, trees, a north arrow (from the symbol library) and the distances of the house to
    the boundary."""
    rng = np.random.default_rng(seed)
    d = Drawing(*PAPER["A3"], scale=500)
    p = Pen(d)
    frame(p, [[("Muster GmbH Vermessung", 3.0)], [("Bauvorhaben Wohnhaus", 3.0)], [("Lageplan", 5.0)],
              [("Maßstab 1:500", 2.5), ("Flur 4", 2.5), ("16.10.2026", 2.5)]])
    turn = float(rng.uniform(8, 30))  # the plot and the house are turned to the street by this, in degrees
    c, s_ = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    cx, cy = 190.0, 125.0

    def rot(x, y):  # mm about the plot's centre, turned
        return cx + x * c - y * s_, cy + x * s_ + y * c

    pw, ph = float(rng.uniform(70, 95)), float(rng.uniform(55, 70))
    plot_ = [rot(-pw / 2, -ph / 2), rot(pw / 2, -ph / 2), rot(pw / 2, ph / 2), rot(-pw / 2, ph / 2)]
    p.poly(plot_, 0.5, closed=True, linetype="dashdot")
    for side in (-1, 1):  # neighbouring plots
        p.poly([rot(side * pw / 2, -ph / 2), rot(side * (pw / 2 + 60), -ph / 2), rot(side * (pw / 2 + 60), ph / 2),
                rot(side * pw / 2, ph / 2)], 0.25, dashed=True)
        p.text(*rot(side * (pw / 2 + 30), 0), f"{int(rng.integers(100, 400))}/{int(rng.integers(1, 9))}", 3.0,
               angle=-turn)
    p.text(*rot(0, -ph / 2 + 8), f"{int(rng.integers(100, 400))}/{int(rng.integers(1, 9))}", 3.5, angle=-turn)
    hw, hh = float(rng.uniform(22, 30)), float(rng.uniform(16, 22))
    hx, hy = float(rng.uniform(-8, 8)), float(rng.uniform(-4, 8))
    house = [rot(hx - hw / 2, hy - hh / 2), rot(hx + hw / 2, hy - hh / 2), rot(hx + hw / 2, hy + hh / 2),
             rot(hx - hw / 2, hy + hh / 2)]
    p.poly(house, 0.7, closed=True)
    for k in np.arange(-hh + 2.0, hw, 2.0):  # hatching of the house: lines at 45 degrees to its sides, clipped
        a0, b0 = max(0.0, k), max(0.0, -k)
        a1 = min(hw, k + hh)
        b1 = a1 - k
        if a1 - a0 > 0.5:
            p.line(*rot(hx - hw / 2 + a0, hy - hh / 2 + b0), *rot(hx - hw / 2 + a1, hy - hh / 2 + b1), 0.18)
    p.text(*rot(hx, hy + hh / 2 + 5), "Wohnhaus", 3.0, angle=-turn)
    # The street along the lower side of the plot, curving: two arcs and its name along it
    r0 = float(rng.uniform(260, 420))
    sx, sy = rot(0, ph / 2 + 12 + r0)
    a = math.degrees(math.atan2(cy + 0 - sy, cx - sx))
    for r in (r0, r0 + 14):
        p.arc(sx, sy, r, a - 22, a + 22, 0.5)
    name = str(rng.choice(["Lindenstraße", "Am Mühlbach", "Gartenweg", "Hauptstraße"]))
    tx, ty = sx + (r0 + 7) * math.cos(math.radians(a - 6)), sy + (r0 + 7) * math.sin(math.radians(a - 6))
    p.text(tx, ty, name, 3.5, angle=-(a - 6 + 90))
    # Contour lines: smooth curves with their height along them
    base = float(rng.integers(120, 140))
    for k in range(3):
        y_ = 40 + 30 * k + float(rng.uniform(-4, 4))
        xs_ = np.linspace(30, 395, 60)
        ys_ = y_ + 6 * np.sin(xs_ / 40 + k + rng.uniform(0, 3))
        for a_, b_ in zip(zip(xs_, ys_), list(zip(xs_, ys_))[1:]):
            p.line(*a_, *b_, 0.18)
        i = int(rng.integers(8, 50))
        ang = -math.degrees(math.atan2(ys_[i + 1] - ys_[i], xs_[i + 1] - xs_[i]))
        p.text(xs_[i], ys_[i] - 2.5, de(base + 0.5 * k) + "", 2.0, angle=ang)
    for _ in range(4):  # spot heights: a small cross and its height
        x, y = rot(float(rng.uniform(-pw / 2 + 5, pw / 2 - 5)), float(rng.uniform(-ph / 2 + 5, ph / 2 - 5)))
        if any(abs(x - q[0]) < 14 and abs(y - q[1]) < 10 for q in house):
            continue
        p.line(x - 1, y - 1, x + 1, y + 1, 0.25)
        p.line(x - 1, y + 1, x + 1, y - 1, 0.25)
        p.text(x + 1.5, y - 2.5, "+" + de(base + float(rng.uniform(0, 1.5))), 2.0, align="left")
    for _ in range(int(rng.integers(4, 8))):  # trees
        x, y = float(rng.uniform(40, 360)), float(rng.uniform(200, 250))
        p.circle(x, y, float(rng.uniform(2.0, 3.5)), 0.25)
        p.circle(x, y, 0.3, 0.3)
    p.symbol("NORTH_ARROW", 385.0, 225.0, h=3.5, LETTER="N")  # clear of the contours, trees and title block
    # Distance of the house to the left boundary, with ticks and its length
    a_, b_ = rot(-pw / 2, hy), rot(hx - hw / 2, hy)
    p.line(*a_, *b_, 0.18)
    for q in (a_, b_):
        p.line(q[0] - 1.0, q[1] + 1.0, q[0] + 1.0, q[1] - 1.0, 0.35)
    p.text((a_[0] + b_[0]) / 2 - 2.0 * s_, (a_[1] + b_[1]) / 2 - 2.0 * c,
           de((hx - hw / 2 + pw / 2) * 0.5), 2.0, angle=-turn)
    p.dimension(a_, b_, de((hx - hw / 2 + pw / 2) * 0.5), a_, b_, 0.0)  # it ends on the boundary and the house
    return d


# --------------------------------------------------------------------------- markups by hand

INKS = {"black": None, "pencil": (125, 125, 125), "red": (205, 40, 40), "blue": (40, 70, 200)}
MARKUP_PARTS = ("note", "cloud", "hatching", "leader", "strike")  # the parts of mark_up, each on its layer


def mark_up(d: Drawing, seed: int, ink: str = "black") -> Drawing:
    """The drawing with what someone adds by hand on a print, in ``ink`` (INKS): a handwritten note, a revision
    cloud, freehand hatching, a leader with its arrow to the note and a text struck out. The markups are polylines of
    the kind MARKUP in the ink's colour, drawn with a varying pen (a polyline per stretch of equal pressure), on a layer
    that names their part (MARKUP_PARTS), so that the benchmark can measure each part; the simulations draw them over
    the print in that colour."""
    rng = np.random.default_rng(seed + 104729)
    colour, out = INKS[ink], []
    W, H = d.width, d.height

    def stroke(points, part, base=0.45):  # split into stretches of different pressure, 3 to 12 mm each
        pts = [(float(x), float(y)) for x, y in points]
        k = 0
        while k < len(pts) - 1:
            n = int(rng.integers(15, 60))
            seg = pts[k:k + n + 1]
            w = float(np.clip(base * rng.uniform(0.8, 1.25), 0.25, 0.8))
            out.append(Polyline(seg, [0.0] * (len(seg) - 1), False, w, False, MARKUP, colour,
                                f"P2C-MARKUP-{part.upper()}"))
            k += n

    def wobble(points, amount=0.25, turn=0.4):
        """A pen's path through ``points``: every 0.2 mm, its turns rounded over ``turn`` mm and wandering off by
        about ``amount`` mm, slowly (over a millimetre or two), as a hand does; not a polygon with jittered corners."""
        p = np.array(list(points), float)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        t = np.append(np.arange(0, s[-1], 0.2), s[-1])
        q = np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])

        def smooth(a, sigma):  # Gaussian along the path, sigma in samples, the ends held
            k = np.exp(-0.5 * (np.arange(-3 * sigma, 3 * sigma + 1) / sigma) ** 2)
            pad = np.pad(a, ((len(k) // 2, len(k) // 2), (0, 0)), mode="edge")
            return np.column_stack([np.convolve(pad[:, i], k / k.sum(), "valid") for i in range(2)])

        if turn > 0:
            q = smooth(q, max(1, round(turn / 0.2)))
        drift = smooth(rng.normal(0, 1, q.shape), 8)
        return (q + drift / max(float(drift.std()), 1e-9) * amount).tolist()

    def spot(w, h):  # somewhere inside the frame, clear of the title block
        return float(rng.uniform(30, W - 20 - w)), float(rng.uniform(20, H - 60 - h))

    # A handwritten note: two words of joined, slanted letters of different heights (small letters, ascenders)
    nx, ny = spot(45, 10)
    x, h = nx, float(rng.uniform(2.5, 4.0))
    for _ in range(2):
        xs, ys = [], []
        for _ in range(int(rng.integers(4, 8))):
            t = np.linspace(0, 2 * math.pi, 24, endpoint=False)
            up, wide = h * float(rng.choice([0.6, 0.6, 1.0, 1.8])), float(rng.uniform(2.0, 3.4))
            lx = x + t / (2 * math.pi) * wide - float(rng.uniform(0.5, 1.0)) * np.sin(t)  # back over itself: a loop
            ly = ny + h - up * (0.5 - 0.5 * np.cos(t))
            xs += (lx + 0.25 * (ny + h - ly)).tolist()
            ys += ly.tolist()
            x += wide
        stroke(wobble(zip(xs, ys), 0.08, 0.2), "note")
        x += float(rng.uniform(3.5, 6))
    # A revision cloud: bumps around an area
    cx0, cy0 = spot(70, 45)
    cw, ch = float(rng.uniform(35, 70)), float(rng.uniform(22, 45))
    ring = []
    for (ax, ay), (bx, by) in zip([(cx0, cy0), (cx0 + cw, cy0), (cx0 + cw, cy0 + ch), (cx0, cy0 + ch)],
                                  [(cx0 + cw, cy0), (cx0 + cw, cy0 + ch), (cx0, cy0 + ch), (cx0, cy0)]):
        side = math.hypot(bx - ax, by - ay)
        n = max(2, int(side / float(rng.uniform(4.5, 7))))
        ux, uy = (bx - ax) / side, (by - ay) / side
        for i in range(n):  # a bump outwards (to the left of the walking direction, clockwise on screen)
            r = side / n / 2
            mx, my = ax + ux * (i + 0.5) * side / n, ay + uy * (i + 0.5) * side / n
            base = math.atan2(-uy, -ux)
            ring += [(mx + r * math.cos(base - f), my + r * math.sin(base - f)) for f in np.linspace(0, math.pi, 10)]
    stroke(wobble(ring + ring[:1], 0.15, 0.1), "cloud", 0.5)
    # Freehand hatching of an area: strokes back and forth
    hx, hy = spot(25, 18)
    zig = []
    for i in range(int(rng.integers(6, 11))):
        zig.append((hx + i * 2.6 + float(rng.uniform(-0.4, 0.4)),
                    hy + (0 if i % 2 else 16) + float(rng.uniform(-1, 1))))
    stroke(wobble(zig, 0.3), "hatching", 0.4)
    # A leader from the note to something in the drawing, with its arrow
    tx, ty = spot(5, 5)
    mid = ((nx + tx) / 2 + float(rng.uniform(-15, 15)), (ny + ty) / 2 + float(rng.uniform(-15, 15)))
    curve = [((1 - u) ** 2 * nx + 2 * (1 - u) * u * mid[0] + u * u * tx,
              (1 - u) ** 2 * (ny + 6) + 2 * (1 - u) * u * mid[1] + u * u * ty) for u in np.linspace(0, 1, 30)]
    stroke(wobble(curve, 0.12), "leader")
    dxl, dyl = curve[-1][0] - curve[-4][0], curve[-1][1] - curve[-4][1]
    ang = math.atan2(dyl, dxl)
    for side in (-1, 1):
        tip = (tx - 3 * math.cos(ang + side * 0.45), ty - 3 * math.sin(ang + side * 0.45))
        stroke(wobble([(tx, ty), tip], 0.05), "leader")
    # A text struck out
    if d.texts:
        t = d.texts[int(rng.integers(len(d.texts)))]
        half = text_length(t) / 2 + 1.5
        strike = [(t.x - half, t.y + float(rng.uniform(-0.6, 0.6))), (t.x + half, t.y + float(rng.uniform(-0.6, 0.6)))]
        if t.angle == 90:
            strike = [(t.x + float(rng.uniform(-0.6, 0.6)), t.y + half), (t.x + float(rng.uniform(-0.6, 0.6)), t.y - half)]
        stroke(wobble(np.linspace(*strike, 12).tolist(), 0.15), "strike")
    return replace(d, polylines=d.polylines + out)
