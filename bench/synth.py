"""Synthetic drawings with ground truth; `scan` and `photo` (app.plan2cad.simulate) turn them into simulated scans
and phone photos. The drawings carry German labels (room names, title block), like the real plans the pipeline was
first built for; OCR reads them with `deu`.

All drawings are generated as vector primitives first; the raster images are rendered from them,
so every test image comes with an exact reference solution.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.plan2cad.library import load, place, transform
from app.plan2cad.model import PAPER, Arc, Circle, Drawing, Line, Polygon, Text
from app.plan2cad.render import text_length
from app.plan2cad.simulate import photo, scan  # noqa: F401 (part of the benchmark's API)

LIBRARY = load()

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


# --------------------------------------------------------------------------- building plan

CM = 0.2  # paper mm per real cm at 1:50


def building_plan(seed: int) -> Drawing:
    """Ground floor plan of a house on A3 at 1:50 with walls, doors, windows, stairs and dimensions."""
    rng = np.random.default_rng(seed)
    d = Drawing(*PAPER["A3"], scale=50)
    p = Pen(d)
    frame(p, [[("Muster GmbH Architekten", 3.0)], [("Neubau Einfamilienhaus", 3.0)],
              [("Grundriss Erdgeschoss", 5.0)],
              [("Maßstab 1:50", 2.5), ("Plan-Nr. A-1%02d" % (seed % 100), 2.5), ("16.10.2026", 2.5)]])
    bw, bh, to = int(rng.integers(1000, 1300)), int(rng.integers(780, 980)), 36  # cm
    ox, oy = 62.0, 42.0

    def mm(cx, cy):
        return ox + cx * CM, oy + cy * CM

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
        p.line(*mm(xf, yhg), *mm(xf + side * w, yhg), 0.35)
        start = {(1, True): 0, (1, False): 270, (-1, True): 90, (-1, False): 180}[(side, top)]
        p.arc(*mm(xf, yhg), w * CM, start, start + 90, 0.18)

    def door_h(y, half, xa, xb, side, w=88):  # door in a horizontal wall, swinging down (+1) or up (-1)
        x0 = int(rng.integers(xa + 25, xb - w - 25))
        grid[y - half:y + half, x0:x0 + w] = 0
        left = bool(rng.integers(2))
        yf, xhg = y + half * side, (x0 if left else x0 + w)
        p.line(*mm(xhg, yf), *mm(xhg, yf + side * w), 0.35)
        start = {(1, True): 0, (1, False): 90, (-1, True): 270, (-1, False): 180}[(side, left)]
        p.arc(*mm(xhg, yf), w * CM, start, start + 90, 0.18)
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
    p.rect(*mm(sx, sy), 26 * n * CM, 100 * CM, 0.35)
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
