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
from app.plan2cad.model import PAPER, Arc, Circle, Drawing, Line, Polygon, Polyline, Text
from app.plan2cad.render import text_length
from app.plan2cad.simulate import MARKUP, photo, plot, scan  # noqa: F401 (part of the benchmark's API)

LIBRARY = load() | load(BUILT_IN / "iec-60617")  # P&ID and circuit diagram symbols, at their printed size
FIXTURES = load(FLOOR_PLAN)  # real size in mm

ROOMS = ["Wohnen", "Küche", "Bad", "Schlafen", "Kind", "Flur", "Büro", "Essen", "Gäste-WC", "Abstellraum", "Hobby"]


class Pen:
    """Small helper to add primitives to a drawing."""

    def __init__(self, d: Drawing):
        self.d = d

    def line(self, x1, y1, x2, y2, w=0.25, dashed=False):
        self.d.lines.append(Line(float(x1), float(y1), float(x2), float(y2), w, dashed))

    def poly(self, pts, w=0.25, closed=False, dashed=False):
        pts = list(pts) + ([pts[0]] if closed else [])
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            self.line(x1, y1, x2, y2, w, dashed)

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

    def symbol(self, name, x, y, angle=0.0, scale=1.0, h=2.0, **attributes):
        """A symbol of the library; its attribute texts stand where the library puts them."""
        s = place(LIBRARY[name], float(x), float(y), angle, False, scale)
        f = transform(x, y, angle, False, scale)
        s.attributes = {tag: Text(*map(float, f(*LIBRARY[name].tags[tag])), h, v) for tag, v in attributes.items()}
        self.d.symbols.append(s)


def frame(p: Pen, rows: list[list[tuple[str, float]]], tb_w: float = 170, tb_h: float = 40):
    """Sheet frame with a 20 mm filing margin and a title block in the bottom right corner."""
    W, H = p.d.width, p.d.height
    p.rect(20, 10, W - 30, H - 20, 0.7)
    x0, y0 = W - 10 - tb_w, H - 10 - tb_h
    rh = tb_h / len(rows)
    for i, row in enumerate(rows):
        if i:
            p.line(x0, y0 + i * rh, x0 + tb_w, y0 + i * rh, 0.35)
        cw = tb_w / len(row)
        for j, (s, h) in enumerate(row):
            if j:
                p.line(x0 + j * cw, y0 + i * rh, x0 + j * cw, y0 + (i + 1) * rh, 0.35)
            p.text(x0 + j * cw + 3, y0 + (i + 0.5) * rh, s, h, align="left")


def de(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


def printed_on(d: Drawing, paper: str) -> Drawing:
    """The drawing printed smaller on ``paper``, as a large P&ID is printed on A3 or A4 to take along: everything,
    pens and texts included, shrinks by the same factor."""
    f = min(PAPER[paper][0] / d.width, PAPER[paper][1] / d.height)
    return replace(d.scaled(f), printed=f)


# --------------------------------------------------------------------------- building plan

def building_plan(seed: int, fixtures: bool = False, scale: int = 50) -> Drawing:
    """Ground floor plan of a house on A3 at 1:50 (or 1:100) with walls, doors, windows, stairs and dimensions. With
    ``fixtures``, kitchen, bathroom and toilet get their fixtures from the floor plan library (the plan is otherwise
    the same)."""
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

    def hchain(xs, y_dim):
        for x in xs:
            p.line(*mm(x, -10), *mm(x, y_dim - 8), 0.18)
            x_, y_ = mm(x, y_dim)
            p.line(x_ - 1.0, y_ + 1.0, x_ + 1.0, y_ - 1.0, 0.35)
        p.line(*mm(xs[0], y_dim), *mm(xs[-1], y_dim), 0.18)
        for a, b in zip(xs, xs[1:]):
            if b - a > 50:
                x_, y_ = mm((a + b) / 2, y_dim)
                p.text(x_, y_ - 2.3, de((b - a) / 100), 2.5)

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

    hchain(sorted({0, bw} | {v for w in top_windows if w for v in w}), -60)
    hchain([0, bw], -100)
    vchain([0, yh - 12, yh + 12, bh], -60)
    vchain([0, bh], -100)
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
    control valve station with block and bypass valves, cooler, reducer, off-page connectors and instruments. Symbols
    come from the symbol library, valves of a random type and size."""
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
    p.line(fx + 10 + h, dy_, hx - rh, dy_, 0.5)
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


# --------------------------------------------------------------------------- circuit diagram

def schematic(seed: int) -> Drawing:
    """Circuit diagram on A3 after IEC 60617: current paths between two supply rails, each with a fuse, contacts in
    series and in parallel branches (junction dots where they branch off), and a relay coil or a lamp; device tags,
    path numbers, wire numbers and the contacts of each relay under it. Two cross connections and the dashed linkage
    of a push button run across other paths without joining them."""
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
        p.symbol(name, x, y, TAG=tag)

    for k, x in enumerate(xs):
        p.text(x, top - 8, str(k + 1), 2.5)
        p.circle(x, top, 0.6, 0.6)  # junction dot on the rail
        p.circle(x, bottom, 0.6, 0.6)
        ys = [top]
        device("FUSE", x, top + 14, f"-F{k + 1}")
        ys += [top + 9, top + 19]
        y = top + 40
        for j in range(int(rng.integers(1, 3))):
            name = str(rng.choice(["NO_CONTACT", "NC_CONTACT"]))
            device(name, x, y, f"-{'S' if j == 0 else 'K'}{int(rng.integers(1, 9))}")
            ys += [y - 5, y + 5]
            y += 25
        if k % 3 == 1:  # a contact in parallel: a branch with junction dots where it leaves and joins
            bx, ya, yb = x + 12, top + 32, y - 22
            p.poly([(x, ya), (bx, ya), (bx, (ya + yb) / 2 - 5)], 0.35)
            device("NO_CONTACT", bx, (ya + yb) / 2, f"-K{k + 1}")
            p.poly([(bx, (ya + yb) / 2 + 5), (bx, yb), (x, yb)], 0.35)
            p.circle(x, ya, 0.6, 0.6)
            p.circle(x, yb, 0.6, 0.6)
        load_ = "COIL" if k % 4 != 3 else "LAMP"
        yc = bottom - 22
        tag = f"-K{k + 1}" if load_ == "COIL" else f"-H{k + 1}"
        device(load_, x, yc, tag)
        ys += [yc - 5, yc + 5, bottom]
        for a, b in zip(ys[::2], ys[1::2]):
            p.line(x, a, x, b, 0.35)
            if b - a > 12:
                p.text(x + 3.5, (a + b) / 2, str(wire), 2.0, angle=90)
                wire += 1
        if load_ == "COIL":
            relays.append((x, tag))
            p.text(x, bottom + 6, f"{int(rng.integers(1, 9))}.{int(rng.integers(1, 9))}", 2.0)
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

def section(seed: int, scale: int = 50) -> Drawing:
    """Section through a two-storey house on A3: cut walls and slabs with their material hatching (constructed
    hatching, part of the drawing), the ground with its earth hatching, a pitched roof, windows, levels with their
    marks (±0,00, +2,75, ...), a dimension chain, the axes of the walls with their circles at the top and room
    names."""
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
        p.text(lx - 10, ly - 3.5, "±0,00" if lv == 0 else ("+" if lv > 0 else "-") + de(abs(lv) / 100), 2.5,
               align="left")
    dx = at(width + 110, 0)[0]  # dimension chain at the right
    marks = [gl, 0, storey - slab, storey, 2 * storey - slab, 2 * storey]
    p.line(dx, at(0, marks[0])[1], dx, at(0, marks[-1])[1], 0.18)
    for m in marks:
        y = at(0, m)[1]
        p.line(dx - 1.0, y + 1.0, dx + 1.0, y - 1.0, 0.35)
        p.line(dx - 3, y, dx + 3, y, 0.18)
    for a, b in zip(marks, marks[1:]):
        if b - a >= 40:
            p.text(dx - 2.5, (at(0, a)[1] + at(0, b)[1]) / 2, de((b - a) / 100), 2.5, angle=90)
    for i, xa in enumerate((wall / 2, width - wall / 2)):  # axes with their circles above the ridge
        ax_, ay_ = at(xa, ridge + 60)
        p.line(ax_, ay_ + 5, ax_, at(0, gl)[1] + 6, 0.18, dashed=True)
        p.circle(ax_, ay_, 5, 0.25)
        p.text(ax_, ay_, "AB"[i], 3.5)
    for k, name in enumerate(rng.permutation(["Wohnen", "Essen", "Schlafen", "Kind"])[:2]):
        x, y = at(width / 2, k * storey + 130)
        p.text(x, y, str(name), 3.5)
    return d


# --------------------------------------------------------------------------- site plan

def site_plan(seed: int) -> Drawing:
    """Site plan on A3 at 1:500: the plot's boundary (dash-dot) with its parcel number, the neighbouring plots, the
    house turned to the street with its hatching, the curved street with its name along it, contour lines with their
    heights along them, spot heights, trees, a north arrow and the distances of the house to the boundary."""
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
    p.poly(plot_, 0.5, closed=True, dashed=True)
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
    # North arrow: a filled half and an open half, N above
    nx, ny = 370.0, 45.0
    p.fill([(nx, ny - 9), (nx + 3, ny + 6), (nx, ny + 3)])
    p.poly([(nx, ny - 9), (nx - 3, ny + 6), (nx, ny + 3)], 0.25, closed=True)
    p.text(nx, ny - 13, "N", 3.5)
    # Distance of the house to the left boundary, with ticks and its length
    a_, b_ = rot(-pw / 2, hy), rot(hx - hw / 2, hy)
    p.line(*a_, *b_, 0.18)
    for q in (a_, b_):
        p.line(q[0] - 1.0, q[1] + 1.0, q[0] + 1.0, q[1] - 1.0, 0.35)
    p.text((a_[0] + b_[0]) / 2 - 2.0 * s_, (a_[1] + b_[1]) / 2 - 2.0 * c,
           de((hx - hw / 2 + pw / 2) * 0.5), 2.0, angle=-turn)
    return d


# --------------------------------------------------------------------------- markups by hand

INKS = {"black": None, "pencil": (125, 125, 125), "red": (205, 40, 40), "blue": (40, 70, 200)}


def mark_up(d: Drawing, seed: int, ink: str = "black") -> Drawing:
    """The drawing with what someone adds by hand on a print, in ``ink`` (INKS): a handwritten note, a revision
    cloud, freehand hatching, a leader with its arrow to the note and a text struck out. The markups are polylines of
    the kind MARKUP in the ink's colour, drawn with a varying pen (a polyline per stretch of equal pressure); the
    simulations draw them over the print in that colour."""
    rng = np.random.default_rng(seed + 104729)
    colour, out = INKS[ink], []
    W, H = d.width, d.height

    def stroke(points, base=0.45):  # split into stretches of different pressure, 3 to 12 mm each
        pts = [(float(x), float(y)) for x, y in points]
        k = 0
        while k < len(pts) - 1:
            n = int(rng.integers(15, 60))
            seg = pts[k:k + n + 1]
            w = float(np.clip(base * rng.uniform(0.8, 1.25), 0.25, 0.8))
            out.append(Polyline(seg, [0.0] * (len(seg) - 1), False, w, False, MARKUP, colour, "P2C-MARKUP"))
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
        stroke(wobble(zip(xs, ys), 0.08, 0.2))
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
    stroke(wobble(ring + ring[:1], 0.15, 0.1), 0.5)
    # Freehand hatching of an area: strokes back and forth
    hx, hy = spot(25, 18)
    zig = []
    for i in range(int(rng.integers(6, 11))):
        zig.append((hx + i * 2.6 + float(rng.uniform(-0.4, 0.4)),
                    hy + (0 if i % 2 else 16) + float(rng.uniform(-1, 1))))
    stroke(wobble(zig, 0.3), 0.4)
    # A leader from the note to something in the drawing, with its arrow
    tx, ty = spot(5, 5)
    mid = ((nx + tx) / 2 + float(rng.uniform(-15, 15)), (ny + ty) / 2 + float(rng.uniform(-15, 15)))
    curve = [((1 - u) ** 2 * nx + 2 * (1 - u) * u * mid[0] + u * u * tx,
              (1 - u) ** 2 * (ny + 6) + 2 * (1 - u) * u * mid[1] + u * u * ty) for u in np.linspace(0, 1, 30)]
    stroke(wobble(curve, 0.12))
    dxl, dyl = curve[-1][0] - curve[-4][0], curve[-1][1] - curve[-4][1]
    ang = math.atan2(dyl, dxl)
    for side in (-1, 1):
        tip = (tx - 3 * math.cos(ang + side * 0.45), ty - 3 * math.sin(ang + side * 0.45))
        stroke(wobble([(tx, ty), tip], 0.05))
    # A text struck out
    if d.texts:
        t = d.texts[int(rng.integers(len(d.texts)))]
        half = text_length(t) / 2 + 1.5
        strike = [(t.x - half, t.y + float(rng.uniform(-0.6, 0.6))), (t.x + half, t.y + float(rng.uniform(-0.6, 0.6)))]
        if t.angle == 90:
            strike = [(t.x + float(rng.uniform(-0.6, 0.6)), t.y + half), (t.x + float(rng.uniform(-0.6, 0.6)), t.y - half)]
        stroke(wobble(np.linspace(*strike, 12).tolist(), 0.15))
    return replace(d, polylines=d.polylines + out)
