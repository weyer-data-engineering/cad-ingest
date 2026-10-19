"""`cad-ingest library check` and `cad-ingest library build`: tools for authors of symbol libraries (see
docs/symbol-libraries.md).

    cad-ingest library check acme-pid/                                  # format, notes and a self-test
    cad-ingest library build blocks.dwg --out acme-pid                  # one file with many blocks
    cad-ingest library build symbols/ --out acme-pid --file-per-symbol  # one drawing per symbol
"""
import argparse
import json
import math
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import ezdxf
from ezdxf import units, xref

from .config import Settings
from .drawing import _load_dxf
from .errors import ConversionError
from .plan2cad import library, simulate
from .plan2cad.export import overlay
from .plan2cad.library import ID, Library, LibraryError, Template
from .plan2cad.model import PAPER, Circle, Drawing, Line, Text
from .plan2cad.pipeline import convert
from .plan2cad.symbols import SMALL

PIPE = 10.0  # mm of pipe drawn at each port in the self-test
POSES = ((0.0, False), (90.0, True))  # each symbol as drawn, and turned and mirrored
SHEET, MARGIN = "A3", 20.0  # self-test sheets
NEAR = 2.0  # mm: a symbol found this close to where one was drawn (or a quarter of its size) stands for it
SAMPLE = "B12"  # the text written at every attribute position in the self-test
RUNS = (("scan", "200 dpi scan", lambda d, seed: simulate.scan(d, seed, dpi=200)),
        ("photo", "12 MP phone photo", lambda d, seed: simulate.photo(d, seed)))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="cad-ingest library", description="Tools for symbol libraries: check one, or "
                                 "build one from existing CAD blocks (see docs/symbol-libraries.md).")
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="check the format and test how well the symbols are found")
    c.add_argument("library", help="installed library id, library folder, ZIP or DXF")
    c.add_argument("--no-self-test", action="store_true", help="only check the format (takes a second)")
    c.add_argument("--out", type=Path, help="folder for the review overlays of the self-test sheets")
    b = sub.add_parser("build", help="make a library from DXF or DWG files")
    b.add_argument("sources", nargs="+", help="DXF or DWG files, or folders of them")
    b.add_argument("--out", type=Path, required=True, help="new folder for library.toml and symbols.dxf")
    b.add_argument("--file-per-symbol", action="store_true",
                   help="each file is one symbol, named after the file, inserted at its base point ($INSBASE); "
                        "without this, the blocks of the files are the symbols")
    b.add_argument("--id", help="library id: lower-case letters, digits and hyphens (default: from --out)")
    b.add_argument("--name", help="display name (default: the id)")
    b.add_argument("--version", default="0.1.0")
    b.add_argument("--licence", help="SPDX licence identifier, e.g. CC-BY-4.0 or LicenseRef-ACME-internal")
    args = ap.parse_args(argv)
    try:
        if args.command == "check":
            lib = library.resolve(args.library, Settings.from_env().symbol_libraries)
            print("\n".join(describe(lib)))
            if not args.no_self_test:
                print("\n" + "\n".join(self_test(lib, args.out)))
            return 0
        return build(args)
    except LibraryError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# --------------------------------------------------------------------------- check

def describe(lib: Library) -> list[str]:
    """The library, its symbols and notes on what may keep a symbol from being found."""
    lines = [f"{lib.id} {lib.version or '(no version)'}: {lib.name}",
             f"Licence: {lib.licence or 'not stated'}. Drawing types: {', '.join(lib.drawing)}. "
             f"One drawing unit is {lib.unit:g} mm.", "", f"{_count(len(lib.symbols), 'symbol')}:"]
    width = max(len(name) for name in lib.symbols)
    for t in lib.symbols.values():
        w, h = _size(t)
        tags = f", attributes {', '.join(t.tags)}" if t.tags else ""
        lines.append(f"  {t.name:<{width}}  {w:5.1f} x {h:5.1f} mm, {len(t.ports)} ports{tags}  {t.description}")
    notes = lib.warnings + [note for t in lib.symbols.values() for note in advice(t)]
    if notes:
        lines += ["", "Notes:"] + [f"  - {note}" for note in notes]
    return lines


def advice(t: Template) -> list[str]:
    """What the matcher needs that the symbol lacks (see symbols.py for how candidates are found)."""
    w, h = _size(t)
    out = []
    if not 2 <= max(w, h) <= 100:
        out.append(f"{t.name}: {w:.1f} x {h:.1f} mm; blocks are drawn at their printed size, check the units "
                   "($INSUNITS) or the scale of the block")
    if any(math.hypot(*p) < 0.5 for p in t.ports):
        out.append(f"{t.name}: the insertion point lies on a port; put it in the middle of the symbol, because a "
                   "port is recognised by its direction from there")
    if not t.ports and not any(isinstance(q, Circle) for q in t.parts):
        out.append(f"{t.name}: no PORT points and no circle, " + (
            f"and larger than {SMALL:g} mm: it cannot be found" if math.hypot(w, h) > SMALL else
            "so it is only found where its strokes stand on their own, not on a pipe") + "; add PORT points where "
            "pipes connect")
    return out


def self_test(lib: Library, out: Path | None = None) -> list[str]:
    """Draws every symbol in two poses on synthetic sheets with pipes at its ports and text at its attributes,
    simulates a scan and a phone photo of each sheet and recognises them with this library alone."""
    placements = [(t, angle, mirror) for t in lib.symbols.values() for angle, mirror in POSES]
    cell = max(30.0, max(max(_size(t)) for t in lib.symbols.values()) + 2 * PIPE + 8)
    W, H = PAPER[SHEET]
    cols, rows = int((W - 2 * MARGIN) // cell), int((H - 2 * MARGIN) // cell)
    if not cols or not rows:
        return ["Self-test skipped: the largest symbol does not fit on an A3 sheet twice."]
    sheets = [placements[i:i + cols * rows] for i in range(0, len(placements), cols * rows)]
    found: dict[str, dict[str, list]] = {kind: {t.name: [] for t in lib.symbols.values()} for kind, _, _ in RUNS}
    extra, seconds, symbol_seconds = Counter(), Counter(), Counter()
    for n, batch in enumerate(sheets, 1):
        d, drawn = _sheet(batch, cols, cell, [f"Library {lib.id}", f"Self-test sheet {n}", "cad-ingest library check"])
        for kind, _, image in RUNS:
            started = time.perf_counter()
            r = convert(image(d, n), paper=SHEET, drawing="pid", templates=lib.symbols)
            seconds[kind] += time.perf_counter() - started
            symbol_seconds[kind] += r.info["timings_s"].get("symbols", 0)
            names = _match(drawn, r.drawing.symbols)
            for s, name in zip(drawn, names):
                found[kind][s.name].append(name)
            extra[kind] += len(r.drawing.symbols) - sum(name is not None for name in names)
            if out:
                out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / f"sheet-{n}-{kind}.png"), overlay(r.image, r.drawing, r.ppm))
    total = len(placements)
    width = max(len(name) for name in lib.symbols)
    lines = [f"Self-test: {_count(len(lib.symbols), 'symbol')}, each as drawn and turned and mirrored, on "
             f"{_count(len(sheets), SHEET + ' sheet')}, " + " and ".join(f"as a {label}" for _, label, _ in RUNS)
             + ":", "",
             f"  {'symbol':<{width}}  scan  photo  mistaken for"]
    for t in lib.symbols.values():
        wrong = Counter(n for kind, _, _ in RUNS for n in found[kind][t.name] if n not in (None, t.name))
        lines.append(f"  {t.name:<{width}}  " + "  ".join(
            f"{found[kind][t.name].count(t.name)}/{len(POSES)}".ljust(len(kind)) for kind, _, _ in RUNS)
            + "  " + ", ".join(f"{n} ({k}x)" for n, k in wrong.most_common()))
    kinds = [kind for kind, _, _ in RUNS]
    rate = {kind: sum(found[kind][t.name].count(t.name) for t in lib.symbols.values()) for kind in kinds}
    per_sheet = {kind: (seconds[kind] / len(sheets), symbol_seconds[kind] / len(sheets)) for kind in kinds}
    lines += ["", "Found: " + ", ".join(f"{k} {rate[k] / total:.0%} ({rate[k]}/{total})" for k in kinds)
              + ". Found where no symbol was drawn: " + ", ".join(f"{k} {extra[k]}" for k in kinds) + ".",
              "Time per sheet: " + ", ".join(f"{k} {per_sheet[k][0]:.1f} s" for k in kinds) + ", of which symbol "
              "recognition " + " and ".join(f"{per_sheet[k][1]:.1f} s" for k in kinds) + "."]
    if out:
        lines.append(f"Review overlays (symbols in teal): {out}/sheet-*.png")
    return lines


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _size(t: Template) -> tuple[float, float]:
    lo, hi = _bounds(t)
    return hi[0] - lo[0], hi[1] - lo[1]


def _bounds(t: Template) -> tuple[tuple[float, float], tuple[float, float]]:
    xs, ys = [], []
    for q in t.parts:
        if isinstance(q, Line):
            xs += [q.x1, q.x2]
            ys += [q.y1, q.y2]
        else:  # circles and arcs: their whole circle, which is close enough here
            xs += [q.cx - q.r, q.cx + q.r]
            ys += [q.cy - q.r, q.cy + q.r]
    return (min(xs), min(ys)), (max(xs), max(ys))


def _sheet(batch: list[tuple[Template, float, bool]], cols: int, cell: float,
           labels: list[str]) -> tuple[Drawing, list]:
    """A sheet with the symbols in a grid, each centred in its cell, with pipes at its ports, its through lines drawn
    on and a short text at each attribute position. The labels in the margin let OCR tell which way is up."""
    d = Drawing(*PAPER[SHEET])
    d.texts = [Text(MARGIN + 40 + 120 * i, MARGIN / 2, 3.5, label) for i, label in enumerate(labels)]
    for k, (t, angle, mirror) in enumerate(batch):
        (x0, y0), (x1, y1) = _bounds(t)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = library.transform(0, 0, angle, mirror)(mx, my)
        x, y = MARGIN + (k % cols + 0.5) * cell - dx, MARGIN + (k // cols + 0.5) * cell - dy
        f = library.transform(x, y, angle, mirror)
        s = library.place(t, x, y, angle, mirror)
        s.attributes = {tag: Text(*f(*at), 2.0, SAMPLE) for tag, at in t.tags.items()}
        d.symbols.append(s)
        for px, py in t.ports:  # pipes run away from the insertion point, as the matcher expects
            ux, uy = (px, py) if math.hypot(px, py) >= 0.5 else (px - mx, py - my)
            norm = math.hypot(ux, uy) or 1.0
            d.lines.append(Line(*f(px, py), *f(px + PIPE * ux / norm, py + PIPE * uy / norm), 0.5))
        for ln in t.through:
            ux, uy = ln.x2 - ln.x1, ln.y2 - ln.y1
            norm = math.hypot(ux, uy) or 1.0
            ux, uy = PIPE * ux / norm, PIPE * uy / norm
            d.lines.append(Line(*f(ln.x1 - ux, ln.y1 - uy), *f(ln.x2 + ux, ln.y2 + uy), 0.5))
    return d, list(d.symbols)


def _match(drawn: list, found: list) -> list[str | None]:
    """For each drawn symbol, the name of the found symbol standing in its place (None if there is none)."""
    def centre(s):
        xs = [v for q in s.parts for v in ((q.x1, q.x2) if isinstance(q, Line) else (q.cx,))]
        ys = [v for q in s.parts for v in ((q.y1, q.y2) if isinstance(q, Line) else (q.cy,))]
        return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

    free, out = {i: centre(s) for i, s in enumerate(found)}, []
    for s in drawn:
        (cx, cy), size = centre(s), max(_size(s.template)) * s.scale
        near = sorted((math.hypot(fx - cx, fy - cy), i) for i, (fx, fy) in free.items())
        if near and near[0][0] <= max(NEAR, size / 4):
            out.append(found[near[0][1]].name)
            del free[near[0][1]]
        else:
            out.append(None)
    return out


# --------------------------------------------------------------------------- build

def build(args) -> int:
    """Collects the blocks (or, with --file-per-symbol, the drawings) of DXF and DWG files into a new library."""
    out: Path = args.out
    lib_id = args.id or library.slug(out.resolve().name)
    if not re.fullmatch(ID, lib_id):
        raise LibraryError(f"--id must be lower-case letters, digits and hyphens: '{lib_id}'")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise LibraryError(f"--out must be a new or empty folder: {out}")
    files = [p for src in map(Path, args.sources)
             for p in (sorted(src.rglob("*")) if src.is_dir() else [src]) if p.suffix.lower() in (".dxf", ".dwg")]
    if not files:
        raise LibraryError("no DXF or DWG files found in the sources")
    target = ezdxf.new("R2018", units=units.InsertUnits.Millimeters)
    for layer in (library.PORT_LAYER, library.THROUGH_LAYER):  # ready for the author's ports and through lines
        target.layers.add(layer, color=1).dxf.plot = 0  # helpers for recognition, not part of the drawing
    notes = []
    with tempfile.TemporaryDirectory(prefix="cad-ingest-") as work:
        for path in files:
            warnings: list[str] = []
            try:
                doc = _load_dxf(path.suffix[1:].lower(), str(path), work, warnings)
            except ConversionError as e:
                notes.append(f"{path.name}: skipped, {e}")
                continue
            notes += [f"{path.name}: {w}" for w in warnings]
            added = _add(doc, path, target, args.file_per_symbol, notes)
            factor = units.conversion_factor(doc.units, units.InsertUnits.Millimeters) if doc.units else 1.0
            for name in added if factor != 1 else []:  # blocks of an inch drawing become millimetres
                block = target.blocks.get(name)
                block.block.dxf.base_point = block.block.dxf.base_point * factor
                for e in block:
                    e.scale_uniform(factor)
    out.mkdir(parents=True, exist_ok=True)
    target.saveas(out / "symbols.dxf")
    fields = {"id": lib_id, "name": args.name or lib_id, "version": args.version, "format": library.FORMAT,
              "licence": args.licence, "drawing": list(library.DRAWINGS), "file": "symbols.dxf"}
    (out / library.MANIFEST).write_text("# Written by `cad-ingest library build`; see docs/symbol-libraries.md.\n"
                                        + "".join(f"{k} = {json.dumps(v, ensure_ascii=False)}\n"
                                                  for k, v in fields.items() if v is not None), encoding="utf-8")
    lib = library.read(out)
    print("\n".join(describe(lib)))
    if notes:
        print("\nWhile building:\n" + "\n".join(f"  - {note}" for note in notes))
    print(f"\nNext: open {out / 'symbols.dxf'} in your CAD program, add POINTs on the layer PORT where pipes connect "
          f"and ATTDEFs where tags stand, then run `cad-ingest library check {out}`.")
    return 0



def _add(doc, path: Path, target, file_per_symbol: bool, notes: list[str]) -> list[str]:
    """Copies the symbols of one source file into ``target``; the first symbol of a name wins."""
    loader, added = xref.Loader(doc, target), []
    if file_per_symbol:
        name = "".join("_" if c in '<>/\\":;?*|=,`' else c for c in path.stem)
        if not len(doc.modelspace()):
            notes.append(f"{path.name}: skipped, the drawing is empty")
        elif name in target.blocks:
            notes.append(f"{path.name}: skipped, there is a symbol named {name} already")
        else:
            loader.load_modelspace(target_layout=target.blocks.new(name, base_point=doc.header.get("$INSBASE", (0, 0))))
            added.append(name)
    else:
        for block in doc.blocks:
            if block.name.startswith(("*", "_")) or not len(block):
                continue
            if block.name in target.blocks or block.name in added:
                notes.append(f"{path.name}: block {block.name} skipped, there is a symbol of that name already")
                continue
            loader.load_block_layout(block)
            added.append(block.name)
        if not added:
            notes.append(f"{path.name}: no blocks (use --file-per-symbol if each file is one symbol)")
    loader.execute()
    return added
