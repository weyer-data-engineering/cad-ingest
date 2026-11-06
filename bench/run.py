"""Benchmark: synthetic drawings with ground truth, converted from simulated scans and phone photos.

    python bench/run.py --out bench/out [--quick] [--seeds N] [--only NAME] [--on FEATURE,...] [--off FEATURE,...]
        [--real photo.heic:A4 ...]
    python bench/run.py --out bench/out --reevaluate

The cases cover floor plans, P&IDs, circuit diagrams, building sections and site plans, as clean plots, scans (good,
poor, skewed) and phone photos, with and without markups added by hand. --only runs the cases whose name contains one
of the given words; --on and --off switch steps of the recognition on and off (app/plan2cad/features.py), to
measure what each brings; --seeds N runs N variants of each case (1 with --quick, otherwise the case's own seeds), for
steadier numbers. Writes results.json, report.md, side-by-side panels and the resolution sweep chart to --out, and
each conversion to --out/pred: the output drawing with its info and its reference, pickled, in place of those of the
run before. --reevaluate measures the saved conversions again with the current evaluate.py and rewrites results.json
and report.md without converting anything, so a new or corrected measure applies to an earlier run. Load only pickles
of your own runs: loading one runs code.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import MISSING, fields, is_dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent)]  # bench and the app

from evaluate import evaluate  # noqa: E402
from synth import (building_plan, mark_up, photo, pid, pid_details, plot, printed_on, scan, schematic,  # noqa: E402
                   section, site_plan, slashed)

from app.plan2cad.export import overlay, to_figure  # noqa: E402
from app.plan2cad.imageio import load_image  # noqa: E402
from app.plan2cad.features import Features  # noqa: E402
from app.plan2cad.library import STARTER, read  # noqa: E402
from app.plan2cad.model import Drawing  # noqa: E402
from app.plan2cad.pipeline import convert  # noqa: E402

PRED = "pred"  # folder of --out with the conversions of the last run, for --reevaluate
MP48 = (8064, 6048)
GREY_LID = 210.0  # a scanner lid about as grey as the paper of a poor scan (222): colour does not tell them apart
PLAN, PID, CIRCUIT, SECTION, SITE = "floor plan", "P&ID", "circuit diagram", "section", "site plan"
# The P&ID starter library without two of the symbols the corpus draws: what is found of those is not classified
LACKING = [replace(read(STARTER), id="din-10628-lacking", symbols={
    k: t for k, t in read(STARTER).symbols.items() if k not in ("CHECK_VALVE", "REDUCER")})]
CASES = [  # name, kind of drawing, drawing, paper, image, seeds[, options of convert]
    ("Floor plan A3, scan 300 dpi", PLAN, building_plan, "A3", lambda d, s: scan(d, s, dpi=300), [1, 2, 3]),
    ("Floor plan A3, phone photo 12 MP", PLAN, building_plan, "A3", lambda d, s: photo(d, s), [1, 2, 3]),
    ("Floor plan A3, phone photo 12 MP, hard", PLAN, building_plan, "A3", lambda d, s: photo(d, s, hard=True),
     [1, 2, 3]),
    ("Floor plan A3 with fixtures, scan 300 dpi", PLAN, lambda s: building_plan(s, fixtures=True), "A3",
     lambda d, s: scan(d, s, dpi=300), [1, 2, 3]),
    ("Floor plan A3 with fixtures, phone photo 12 MP", PLAN, lambda s: building_plan(s, fixtures=True), "A3",
     lambda d, s: photo(d, s), [1, 2, 3]),
    ("Floor plan A3 1:100 with fixtures, scan 300 dpi", PLAN, lambda s: building_plan(s, fixtures=True, scale=100),
     "A3", lambda d, s: scan(d, s, dpi=300), [1, 2, 3]),
    ("Floor plan A3 1:100 with fixtures, phone photo 12 MP", PLAN,
     lambda s: building_plan(s, fixtures=True, scale=100), "A3", lambda d, s: photo(d, s), [1, 2, 3]),
    ("Floor plan A3 with red markups, scan 300 dpi", PLAN, lambda s: mark_up(building_plan(s), s, "red"), "A3",
     lambda d, s: scan(d, s, dpi=300), [1, 2]),
    ("Floor plan A3 with pencil markups, scan 300 dpi", PLAN, lambda s: mark_up(building_plan(s), s, "pencil"), "A3",
     lambda d, s: scan(d, s, dpi=300), [1, 2]),
    ("P&ID A1, scan 200 dpi", PID, lambda s: pid(s, "A1"), "A1", lambda d, s: scan(d, s, dpi=200), [1, 2]),
    ("P&ID A1, phone photo 12 MP", PID, lambda s: pid(s, "A1"), "A1", lambda d, s: photo(d, s), [1, 2]),
    ("P&ID A1 printed on A3, scan 300 dpi", PID, lambda s: printed_on(pid(s, "A1"), "A3"), "A3",
     lambda d, s: scan(d, s, dpi=300), [1, 2]),
    ("P&ID A1 printed on A4, phone photo 12 MP", PID, lambda s: printed_on(pid(s, "A1"), "A4"), "A4",
     lambda d, s: photo(d, s), [1, 2]),
    ("P&ID A1 printed on A3 with black markups, scan 300 dpi", PID,
     lambda s: mark_up(printed_on(pid(s, "A1"), "A3"), s, "black"), "A3", lambda d, s: scan(d, s, dpi=300), [1, 2]),
    ("P&ID A2 with a library lacking two symbols, scan 200 dpi", PID, lambda s: pid(s, "A2"), "A2",
     lambda d, s: scan(d, s, dpi=200), [1, 2], {"libraries": LACKING}),
    # ISA style with flanges, line labels in frames, marked signal lines and slashed noughts, as printed P&IDs show
    ("P&ID A3 with flanges and line labels, scan 300 dpi", PID, pid_details, "A3",
     lambda d, s: scan(slashed(d), s, dpi=300), [1, 2]),
    ("P&ID A3 with flanges and line labels, phone photo 12 MP", PID, pid_details, "A3",
     lambda d, s: photo(slashed(d), s), [1, 2]),
    # ... printed smaller, valves at about half the library size, with the marks of signal lines in a pen
    ("P&ID A3 with flanges printed on A4, marks in pen, scan 300 dpi", PID,
     lambda s: printed_on(pid_details(s, pen_marks=True), "A4"), "A4", lambda d, s: scan(slashed(d), s, dpi=300),
     [1, 2]),
    ("P&ID A3 with flanges against small bubbles, scan 300 dpi", PID, lambda s: pid_details(s, flush=True), "A3",
     lambda d, s: scan(slashed(d), s, dpi=300), [1, 2]),
    ("P&ID A0, scan 200 dpi", PID, lambda s: pid(s, "A0"), "A0", lambda d, s: scan(d, s, dpi=200), [1]),
    ("P&ID A0, phone photo 12 MP", PID, lambda s: pid(s, "A0"), "A0", lambda d, s: photo(d, s), [1]),
    ("P&ID A0, phone photo 48 MP", PID, lambda s: pid(s, "A0"), "A0", lambda d, s: photo(d, s, camera=MP48), [1]),
    ("Circuit diagram A3, clean plot 200 dpi", CIRCUIT, schematic, "A3", lambda d, s: plot(d, 200), [1, 2],
     {"dpi": 200}),
    ("Circuit diagram A3, scan 300 dpi", CIRCUIT, schematic, "A3", lambda d, s: scan(d, s, dpi=300), [1, 2]),
    ("Circuit diagram A3, poor scan 200 dpi skewed", CIRCUIT, schematic, "A3",
     lambda d, s: scan(d, s, dpi=200, poor=True, skew=4.0), [1, 2]),
    ("Circuit diagram A3 with black markups, scan 300 dpi", CIRCUIT, lambda s: mark_up(schematic(s), s, "black"),
     "A3", lambda d, s: scan(d, s, dpi=300), [1, 2]),
    ("Circuit diagram A3, poor scan 200 dpi on a grey lid", CIRCUIT, schematic, "A3",
     lambda d, s: scan(d, s, dpi=200, poor=True, lid=GREY_LID), [1, 2]),
    ("Section A3 1:50, scan 300 dpi", SECTION, section, "A3", lambda d, s: scan(d, s, dpi=300), [1, 2]),
    ("Section A3 1:50, phone photo 12 MP", SECTION, section, "A3", lambda d, s: photo(d, s), [1, 2]),
    ("Section A3 1:50 with blue markups, scan 300 dpi", SECTION, lambda s: mark_up(section(s), s, "blue"), "A3",
     lambda d, s: scan(d, s, dpi=300), [1, 2]),
    ("Section A3 1:50 with arrowheads, scan 300 dpi", SECTION, lambda s: section(s, terminator="arrow"), "A3",
     lambda d, s: scan(d, s, dpi=300), [1, 2]),
    ("Section A3 1:50 with dots, phone photo 12 MP", SECTION, lambda s: section(s, terminator="dot"), "A3",
     lambda d, s: photo(d, s), [1, 2]),
    ("Site plan A3 1:500, scan 300 dpi", SITE, site_plan, "A3", lambda d, s: scan(d, s, dpi=300), [1, 2]),
    ("Site plan A3 1:500, poor scan 200 dpi skewed", SITE, site_plan, "A3",
     lambda d, s: scan(d, s, dpi=200, poor=True, skew=4.0), [1, 2]),
    ("Site plan A3 1:500, poor scan 200 dpi on a grey lid", SITE, site_plan, "A3",
     lambda d, s: scan(d, s, dpi=200, poor=True, lid=GREY_LID), [1, 2]),
    ("Site plan A3 1:500 with pencil markups, phone photo 12 MP", SITE, lambda s: mark_up(site_plan(s), s, "pencil"),
     "A3", lambda d, s: photo(d, s), [1, 2]),
]
SWEEP = [3, 4, 5, 6, 8, 10]  # px/mm on the A0 P&ID
KEYS = ["recall_1mm", "precision_1mm", "recall_05mm", "precision_05mm", "deviation_mm", "dashed_recall_1mm",
        "circle_recall", "circle_precision", "text_exact", "text_cer", "text_false_positives", "text_review",
        "text_unflagged", "text_exact_codes", "text_exact_numbers", "text_exact_words",
        "pieces_per_object", "objects_whole", "entities", "reference_objects", "short_pieces",
        "breaks_per_m", "curve_pieces", "curves_as_arcs", "curves_as_splines",
        "symbol_recall", "symbol_precision", "symbol_attributes", "symbol_review", "symbol_unflagged",
        "connection_recall", "connection_precision", "gap_ends", "crumbs",
        "token_exact", "special_tokens_exact", "uncertain_share", "deviation_areas", "symbol_false",
        "markup_recall", "markup_precision", "markup_leaked", "markup_phantom_texts", "markup_fills",
        "markup_false_mm", "markup_found_note", "markup_found_cloud", "markup_found_hatching", "markup_found_leader",
        "markup_found_strike",
        "crossings_kept", "junctions_kept", "openings_kept", "pens_correct", "linetype_correct", "linetype_precision",
        "dimension_recall", "dimension_precision", "dimension_review", "dimension_false", "dimension_true",
        "dimension_on_object",
        "symbol_phantom_texts", "unclassified_on_symbols", "unclassified_elsewhere", "unclassified_found",
        "short_strokes_kept", "title_cells_found", "title_cells_exact", "printed_ok",
        "scale_ok", "seconds", "source_ppm"]


def run_one(d, paper, img, features=None, **options):
    started = time.perf_counter()
    r = convert(img, paper=paper, features=features, **options)
    m = evaluate(d, r.drawing) | {"seconds": time.perf_counter() - started, "source_ppm": r.info["source_ppm"]}
    return r, m


def mean(rows: list[dict]) -> dict:
    out = {}
    for k in KEYS:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    return out


def summary(name: str, kind: str, paper: str, rows: list[dict]) -> dict:
    """A case in results.json: the means of the measures over its seeds."""
    return {"name": name, "kind": kind, "paper": paper, "seeds": len(rows)} | mean(rows)


def variants(seeds: list[int], n: int | None) -> list[int]:
    """The seeds a case runs with: its own, or the first n of them, continued with the next numbers if it has fewer."""
    if n is None:
        return seeds
    return seeds[:n] + list(range(max(seeds) + 1, max(seeds) + 1 + n - len(seeds)))


def slug(name: str) -> str:
    """A case name as part of a file name."""
    return name.replace(" ", "_").replace(",", "").replace("&", "")


class Saved:
    """The conversions of a run, each pickled to a file of ``folder`` with the reference it is measured against and
    numbered in the order of the run, so that --reevaluate can measure them again. A new run replaces them, as it
    replaces results.json."""

    def __init__(self, folder: Path, on: str, off: str):
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.pkl"):
            old.unlink()
        self.folder, self.on, self.off, self.count = folder, on, off, 0

    def add(self, file: str, part: str, reference: Drawing | None, drawing: Drawing, info: dict, seconds: float,
            **about) -> None:
        """Saves one conversion: ``part`` is "case", "sweep" or "real" (a real photo, without a reference)."""
        record = {"part": part, "order": self.count, "on": self.on, "off": self.off, "reference": _plain(reference),
                  "drawing": _plain(drawing), "info": info, "seconds": seconds} | about
        (self.folder / f"{file}.pkl").write_bytes(pickle.dumps(record, pickle.HIGHEST_PROTOCOL))
        self.count += 1


def _plain(d: Drawing | None) -> Drawing | None:
    """The drawing with its symbols cut loose from the library they were placed from: no measure uses the library, and
    pickled with it, every file would hold all of it."""
    return None if d is None else replace(d, symbols=[replace(s, template=None) for s in d.symbols])


def _complete(q) -> None:
    """Gives the dataclasses in ``q`` the fields added to the model since they were saved, with their defaults, so that
    the current evaluate.py can measure the run of an older version."""
    if is_dataclass(q):
        for f in fields(q):
            if not hasattr(q, f.name):  # a new field without a default fails here, as it should
                setattr(q, f.name, f.default if f.default is not MISSING else f.default_factory())
            _complete(getattr(q, f.name))
    elif isinstance(q, (list, tuple)):
        for v in q:
            _complete(v)
    elif isinstance(q, dict):
        for v in q.values():
            _complete(v)


def load(folder: Path) -> list[dict]:
    """The conversions saved in ``folder``, in the order they were run."""
    records = []
    for path in folder.glob("*.pkl"):
        try:
            record = pickle.loads(path.read_bytes())
            _complete(record)
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, TypeError, ValueError) as e:
            raise SystemExit(f"{path}: cannot be read with the current code ({e})") from None
        records.append(record)
    if not records:
        raise SystemExit(f"{folder}: no saved conversions; run the benchmark first")
    return sorted(records, key=lambda r: r["order"])


def reevaluate(folder: Path) -> dict:
    """The results of the run saved in ``folder``, measured anew with the current evaluate.py; the runtimes are those
    of the run."""
    records = load(folder)
    results = {"cases": [], "sweep": [], "real": [], "on": records[0]["on"], "off": records[0]["off"]}
    cases: dict[tuple[str, str, str], list[dict]] = {}
    for r in records:
        if r["part"] == "real":
            results["real"].append(r["info"])
            continue
        m = evaluate(r["reference"], r["drawing"]) | {"seconds": r["seconds"], "source_ppm": r["info"]["source_ppm"]}
        if r["part"] == "sweep":
            results["sweep"].append({"ppm": r["ppm"]} | {k: m.get(k) for k in KEYS})
        else:
            cases.setdefault((r["case"], r["kind"], r["paper"]), []).append(m)
    results["cases"] = [summary(*case, rows) for case, rows in cases.items()]
    print(f"measured {len(records)} saved conversions again", flush=True)
    return results


def panel(img: np.ndarray, r, out: Path, name: str) -> None:
    """Input | review overlay | clean output, side by side, plus the full-size overlay."""
    ov = overlay(r.image, r.drawing, r.ppm, max_side=3000)
    cv2.imwrite(str(out / f"{name}_overlay.png"), ov)
    to_figure(r.drawing, out / f"{name}_output.png", "png", dpi=3000 / (max(r.drawing.width, r.drawing.height) / 25.4))
    rendered = cv2.imread(str(out / f"{name}_output.png"))
    h = 900
    parts = [cv2.resize(p, (round(p.shape[1] * h / p.shape[0]), h), interpolation=cv2.INTER_AREA)
             for p in (img, ov, rendered)]
    gap = np.full((h, 24, 3), 255, np.uint8)
    cv2.imwrite(str(out / f"{name}_panel.jpg"), np.hstack([parts[0], gap, parts[1], gap, parts[2]]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


def pct(v, digits=0):
    return "–" if v is None else f"{100 * v:.{digits}f} %"


def num(v, digits=2):
    return "–" if v is None else f"{v:.{digits}f}"


def sweep_chart(rows: list[dict], path: Path) -> None:
    fig = Figure(figsize=(7.2, 4.2), dpi=200, facecolor="#fcfcfb")
    ax = fig.add_axes((0.09, 0.14, 0.62, 0.74), facecolor="#fcfcfb")
    x = [r["ppm"] for r in rows]
    series = [("recall_1mm", "Lines found (≤ 1 mm)", "#2a78d6"),
              ("circle_recall", "Circles found", "#eb6834"),
              ("text_exact", "Texts read exactly", "#1baf7a")]
    ends = []
    for key, label, color in series:
        y = [100 * r[key] for r in rows]
        ax.plot(x, y, color=color, lw=2, marker="o", ms=5, label=label)
        ends.append([y[-1], label])
    ends.sort()
    for i in range(1, len(ends)):  # keep the direct labels at least 6 points apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 6)
    for y, label in ends:
        ax.annotate(label, (x[-1], y), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=8, color="#0b0b0b", annotation_clip=False)
    for v, label in ((2.9, "A0 as 12 MP photo"), (5.7, "A0 as 48 MP photo"), (7.9, "A0 scan 200 dpi")):
        ax.axvline(v, color="#a8a7a1", lw=1, ls=(0, (2, 3)))
        ax.text(v, 103, label, fontsize=7, color="#52514e", ha="center", va="bottom")
    ax.set_xlim(2.5, 10.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Resolution on paper (pixels per mm)", fontsize=8, color="#52514e")
    ax.set_ylabel("%", fontsize=8, color="#52514e")
    ax.tick_params(labelsize=8, colors="#52514e")
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#a8a7a1")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.text(0.09, 0.95, "P&ID A0: recognition versus image resolution", fontsize=10, color="#0b0b0b")
    fig.savefig(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/out")
    ap.add_argument("--quick", action="store_true", help="one seed per case, no A0 and no sweep")
    ap.add_argument("--seeds", type=int, help="variants per case (default: 1 with --quick, else the case's seeds)")
    ap.add_argument("--real", nargs="*", default=[], help="real photos as path:paper (no ground truth)")
    ap.add_argument("--report-only", action="store_true", help="rebuild report and chart from results.json")
    ap.add_argument("--reevaluate", action="store_true",
                    help=f"measure the conversions saved in OUT/{PRED} again with the current evaluate.py and rewrite "
                         "results.json, report and chart (no conversion)")
    ap.add_argument("--only", default="", help="comma-separated words: only the cases whose name contains one")
    ap.add_argument("--on", default="", help="comma-separated features to switch on (app/plan2cad/features.py)")
    ap.add_argument("--off", default="", help="comma-separated features to switch off")
    args = ap.parse_args()
    if args.seeds is not None and args.seeds < 1:
        ap.error("--seeds must be at least 1")
    if args.reevaluate and (args.quick or args.seeds or args.only or args.on or args.off or args.real):
        ap.error("--reevaluate measures the saved run as it was: no --quick, --seeds, --only, --on, --off or --real")
    features = Features.of([f for f in args.on.split(",") if f], [f for f in args.off.split(",") if f])
    only = [w.lower() for w in args.only.split(",") if w]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.reevaluate:
        (out / "results.json").write_text(json.dumps(reevaluate(out / PRED), indent=1, ensure_ascii=False))
    if args.report_only or args.reevaluate:
        results = json.loads((out / "results.json").read_text())
        if results["sweep"]:
            sweep_chart(results["sweep"], out / "sweep.png")
        (out / "report.md").write_text(report(results))
        return
    results = {"cases": [], "sweep": [], "real": [], "on": args.on, "off": args.off}
    saved = Saved(out / PRED, args.on, args.off)

    for name, kind, make, paper, image, seeds, *options in CASES:
        if args.quick and (paper == "A0") or only and not any(w in name.lower() for w in only):
            continue
        rows = []
        for seed in variants(seeds, args.seeds or (1 if args.quick else None)):
            d = make(seed)
            img = image(d, seed)
            r, m = run_one(d, paper, img, features, **(options[0] if options else {}))
            saved.add(f"{slug(name)}_{seed}", "case", d, r.drawing, r.info, m["seconds"], case=name, kind=kind,
                      paper=paper, seed=seed)
            rows.append(m)
            shown = ("recall_1mm", "precision_1mm", "text_exact", "symbol_recall", "symbol_precision", "seconds")
            print(f"{name} seed {seed}: " + ", ".join(f"{k}={m[k]:.3f}" for k in shown if m.get(k) is not None), flush=True)
            if seed == seeds[0]:
                panel(img, r, out, slug(name))
        results["cases"].append(summary(name, kind, paper, rows))

    if not args.quick and not only:
        d = pid(1, "A0")
        for ppm in SWEEP:
            r, m = run_one(d, "A0", scan(d, 1, dpi=ppm * 25.4, blur=0.8), features)
            saved.add(f"sweep_{ppm}", "sweep", d, r.drawing, r.info, m["seconds"], ppm=ppm)
            results["sweep"].append({"ppm": ppm} | {k: m.get(k) for k in KEYS})
            print(f"sweep {ppm} px/mm: recall={m['recall_1mm']:.3f} text={m['text_exact']:.3f}", flush=True)
        sweep_chart(results["sweep"], out / "sweep.png")

    for spec in args.real:
        path, paper = spec.rsplit(":", 1)
        img, dpi = load_image(Path(path).read_bytes())
        started = time.perf_counter()
        r = convert(img, paper=paper, dpi=dpi, features=features)
        info = r.info | {"seconds": time.perf_counter() - started, "file": Path(path).name}
        saved.add("real_" + Path(path).stem, "real", None, r.drawing, info, info["seconds"])
        results["real"].append(info)
        panel(cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), r, out, "real_" + Path(path).stem)
        print(f"real {path}: {info['counts']} in {info['seconds']:.1f} s", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    (out / "report.md").write_text(report(results))


def report(results: dict) -> str:
    lines = ["# cad-ingest photo and scan benchmark", "",
             "Synthetic drawings with ground truth; deviations measured on paper."
             + (f" Switched on: {results['on']}." if results.get("on") else "")
             + (f" Switched off: {results['off']}." if results.get("off") else ""), "",
             "| Case | px/mm | Lines found ≤ 1 mm | Lines correct ≤ 1 mm | found/correct ≤ 0.5 mm "
             "| Mean deviation | Dashed lines | Circles found/correct | Texts exact | Character error rate | Scale | Runtime |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in results["cases"]:
        lines.append(
            f"| {c['name']} | {num(c['source_ppm'], 1)} | {pct(c['recall_1mm'], 1)} | {pct(c['precision_1mm'], 1)} "
            f"| {pct(c['recall_05mm'])} / {pct(c['precision_05mm'])} | {num(c['deviation_mm'])} mm "
            f"| {pct(c['dashed_recall_1mm'])} | {pct(c['circle_recall'])} / {pct(c['circle_precision'])} "
            f"| {pct(c['text_exact'])} | {pct(c['text_cer'], 1)} | {pct(c['scale_ok'])} | {num(c['seconds'], 1)} s |")
    lines += ["", "## Whole objects instead of fragments", "",
              "How many DXF entities make up one object of the reference (1 = in one piece, as drawn in CAD).", "",
              "| Case | Pieces per object | Objects in one piece | DXF entities | Reference objects "
              "| Pieces under 1.5 mm | Texts to check |",
              "|---|---|---|---|---|---|---|"]
    for c in results["cases"]:
        lines.append(f"| {c['name']} | {num(c['pieces_per_object'])} | {pct(c['objects_whole'])} | {num(c['entities'], 0)} "
                     f"| {num(c['reference_objects'], 0)} | {num(c['short_pieces'], 0)} | {num(c['text_review'], 0)} |")
    lines += ["", "## Continuous lines and true curves", "",
              "Breaks: places per metre of linework where the output stops although the line runs on (a gap or a "
              "joint that has to be closed by hand). Curve pieces: output entities per arc of the reference (1: in one "
              "piece). Curves drawn as arcs: share of the reference arcs drawn as arcs, not as short straight lines; "
              "as splines: drawn as free curves instead. Line types right: share of the dashed, dash-dot and dotted "
              "lines of the reference found with their line type. Dash-dot and dotted on such lines: share of the "
              "output's dash-dot and dotted length that lies on reference lines of that line type.",
              "",
              "| Case | Breaks per metre | Pieces per curve | Curves drawn as arcs | As splines | Line types right "
              "| Dash-dot and dotted on such lines |",
              "|---|---|---|---|---|---|---|"]
    for c in results["cases"]:
        lines.append(f"| {c['name']} | {num(c.get('breaks_per_m'), 1)} | {num(c.get('curve_pieces'))} "
                     f"| {pct(c.get('curves_as_arcs'))} | {pct(c.get('curves_as_splines'))} "
                     f"| {pct(c.get('linetype_correct'))} | {pct(c.get('linetype_precision'))} |")
    lines += ["", "## Connected linework", "",
              "What has to be joined or cleaned up by hand in CAD. Ends stopping short: ends of lines and polylines "
              "that stop up to 2 mm short of other linework without touching it (a pipe that does not reach its "
              "junction, valve or vessel). Crumbs: pieces under 2 mm that touch nothing. The references have none.", "",
              "| Case | Ends stopping short | Crumbs |", "|---|---|---|"]
    for c in results["cases"]:
        lines.append(f"| {c['name']} | {num(c.get('gap_ends'), 0)} | {num(c.get('crumbs'), 0)} |")
    lines += ["", "## Texts by kind and errors left unmarked", "",
              "Exact texts among codes (letters and digits: tags, line numbers), numbers and words. Unmarked errors: "
              "reference texts read wrong without a text marked for review in their place, so nobody is pointed to "
              "them.", "",
              "| Case | Codes exact | Numbers exact | Words exact | Unmarked errors |", "|---|---|---|---|---|"]
    for c in results["cases"]:
        lines.append(f"| {c['name']} | {pct(c.get('text_exact_codes'))} | {pct(c.get('text_exact_numbers'))} "
                     f"| {pct(c.get('text_exact_words'))} | {pct(c.get('text_unflagged'), 1)} |")
    lines += ["", "## Symbols as CAD blocks", "",
              "Reference symbols placed as a block of the right kind at the right place (≤ 1.5 mm). Connections: "
              "pairs of symbols joined by linework, found / correct. To check: symbols marked as uncertain. Unmarked "
              "errors: symbols placed wrong and not marked, as a share of the reference symbols.", "",
              "| Case | Symbols found | Symbols correct | Attributes read | Connections found / correct | To check "
              "| Unmarked errors |",
              "|---|---|---|---|---|---|---|"]
    for c in results["cases"]:
        if c["symbol_recall"] is not None:
            lines.append(f"| {c['name']} | {pct(c['symbol_recall'], 1)} | {pct(c['symbol_precision'], 1)} "
                         f"| {pct(c['symbol_attributes'])} "
                         f"| {pct(c.get('connection_recall'))} / {pct(c.get('connection_precision'))} "
                         f"| {num(c.get('symbol_review'), 0)} | {pct(c.get('symbol_unflagged'), 1)} |")
    lines += ["", "## By kind of drawing", "",
              "Means over the cases of each kind. Tokens: words and numbers of the reference texts read exactly. "
              "Marked uncertain: texts and symbols marked to check, as a share of all. Deviation areas: patches of at "
              "least 2 mm² where the recognised linework and the reference differ by more than 1 mm. Crossings kept: "
              "lines that cross without joining still do; junctions kept: lines that join still do; openings kept: a "
              "wall broken off for a door or a window still is, and is there on both sides. Pens: the share "
              "of the linework drawn with the pen (ISO line width) of the reference. Dimensions: see "
              "[Dimensions](#dimensions).", "",
              "| Kind | Cases | Lines found / correct ≤ 1 mm | Texts exact | Tokens exact | Symbols found / correct "
              "| Marked uncertain | Deviation areas | Ends stopping short | Crossings / junctions / openings kept "
              "| Pens right | Dimensions true to their value / on the object |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    kinds: dict[str, list[dict]] = {}
    for c in results["cases"]:
        kinds.setdefault(c.get("kind", "?"), []).append(c)
    for kind, cs in kinds.items():
        m = mean(cs)
        lines.append(f"| {kind} | {len(cs)} | {pct(m['recall_1mm'], 1)} / {pct(m['precision_1mm'], 1)} "
                     f"| {pct(m['text_exact'])} | {pct(m['token_exact'])} "
                     f"| {pct(m['symbol_recall'])} / {pct(m['symbol_precision'])} | {pct(m['uncertain_share'])} "
                     f"| {num(m['deviation_areas'], 0)} | {num(m['gap_ends'], 0)} "
                     f"| {pct(m['crossings_kept'])} / {pct(m['junctions_kept'])} / {pct(m['openings_kept'])} "
                     f"| {pct(m['pens_correct'])} | {pct(m['dimension_true'])} / {pct(m['dimension_on_object'])} |")
    lines += ["", "## New measures per case", "",
              "Symbols on a drawing without symbols: blocks placed where the reference has none. Markups (the cases "
              "with markups added by hand): the share of the markup strokes put on a markup layer (found), the share of "
              "what is on a markup layer that is markup (correct), the share of the markup taken for the drawing "
              "(leaked), texts of the drawing read on a markup (phantoms) and fills made there. Special characters: "
              "the share of the reference tokens with one of ± Ø ° ² ³ ‰ × ′ ″ read exactly.", "",
              "| Case | Tokens exact | Special characters exact | Marked uncertain | Deviation areas "
              "| Symbols where there are none | Markup found / correct | Markup leaked | Phantom texts "
              "| Fills from markups | Crossings / junctions / openings kept | Pens right |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in results["cases"]:
        lines.append(f"| {c['name']} | {pct(c.get('token_exact'))} | {pct(c.get('special_tokens_exact'))} "
                     f"| {pct(c.get('uncertain_share'))} "
                     f"| {num(c.get('deviation_areas'), 0)} | {num(c.get('symbol_false'), 0)} "
                     f"| {pct(c.get('markup_recall'))} / {pct(c.get('markup_precision'))} "
                     f"| {pct(c.get('markup_leaked'))} | {num(c.get('markup_phantom_texts'), 0)} "
                     f"| {num(c.get('markup_fills'), 0)} "
                     f"| {pct(c.get('crossings_kept'))} / {pct(c.get('junctions_kept'))} "
                     f"/ {pct(c.get('openings_kept'))} "
                     f"| {pct(c.get('pens_correct'))} |")
    lines += ["", "## Dimensions", "",
              "Reference dimensions that came out as CAD dimension objects with both ends of the dimension line within "
              "1 mm and the value read exactly (none where none is written): found, and the share of the dimension "
              "objects that are right (correct). To check: dimensions marked as uncertain. Where there are none: "
              "dimension objects on a drawing without dimensions. True to their value: dimensions with a value whose "
              "measured points lie as far apart as the value says, within 1 mm in reality (what measuring the DXF "
              "gives). On the object: measured points of dimensions found that lie within 0.1 mm of the linework the "
              "reference dimension measures (where its extension line meets any).", "",
              "| Case | Dimensions found | Dimensions correct | To check | Where there are none | True to their value "
              "| On the object |",
              "|---|---|---|---|---|---|---|"]
    for c in results["cases"]:
        lines.append(f"| {c['name']} | {pct(c.get('dimension_recall'))} | {pct(c.get('dimension_precision'))} "
                     f"| {num(c.get('dimension_review'), 0)} | {num(c.get('dimension_false'), 0)} "
                     f"| {pct(c.get('dimension_true'))} | {pct(c.get('dimension_on_object'))} |")
    lines += ["", "## Around the symbols and small strokes", "",
              "Phantom texts: texts on a symbol of the reference where it has none, its strokes read as letters. Not "
              "classified: clusters of linework marked as a symbol no library has, on a reference symbol not placed "
              "right or one of no library (a level mark) / elsewhere, and the share of those symbols so marked. Short "
              "strokes kept: solid strokes of the reference from 0.5 to 3 mm (ticks, arrowheads, bars of symbols) "
              "that the output has.", "",
              "| Case | Phantom texts on symbols | Not classified: on symbols / elsewhere | Missed symbols marked "
              "| Short strokes kept | Crumbs |", "|---|---|---|---|---|---|"]
    for c in results["cases"]:
        lines.append(f"| {c['name']} | {num(c.get('symbol_phantom_texts'), 1)} "
                     f"| {num(c.get('unclassified_on_symbols'), 1)} / {num(c.get('unclassified_elsewhere'), 1)} "
                     f"| {pct(c.get('unclassified_found'))} | {pct(c.get('short_strokes_kept'), 1)} "
                     f"| {num(c.get('crumbs'), 0)} |")
    lines += ["", "## Title block and print size", "",
              "Title block cells: the cells of the reference title block that the output's table has (boxes "
              "overlapping by half), found / read exactly. Print size right: the share of its drawn size the sheet "
              "shows the drawing at, as measured on its lettering, within 15 % of the reference (1 for a drawing "
              "printed at its size).", "",
              "| Case | Title block cells found / exact | Print size right |", "|---|---|---|"]
    for c in results["cases"]:
        lines.append(f"| {c['name']} | {pct(c.get('title_cells_found'))} / {pct(c.get('title_cells_exact'))} "
                     f"| {pct(c.get('printed_ok'))} |")
    lines += ["", "## Markups by part", "",
              "Of each part of the markups added by hand, the share put on a markup layer: the handwritten note, the "
              "revision cloud, the freehand hatching, the leader with its arrowhead and the stroke through a text. Not "
              "markup: the length on a markup layer where there is no markup (also on drawings without markups).", "",
              "| Case | Note | Cloud | Hatching | Leader | Strike-out | Not markup |", "|---|---|---|---|---|---|---|"]
    for c in results["cases"]:
        if c.get("markup_found_note") is not None or c.get("markup_false_mm"):
            lines.append(f"| {c['name']} | {pct(c.get('markup_found_note'))} | {pct(c.get('markup_found_cloud'))} "
                         f"| {pct(c.get('markup_found_hatching'))} | {pct(c.get('markup_found_leader'))} "
                         f"| {pct(c.get('markup_found_strike'))} | {num(c.get('markup_false_mm'), 0)} mm |")
    if results["sweep"]:
        lines += ["", "## Resolution (P&ID A0, scan-like)", "",
                  "| px/mm | Lines found | Lines correct | Circles | Texts exact | Character error rate |",
                  "|---|---|---|---|---|---|"]
        for s in results["sweep"]:
            lines.append(f"| {s['ppm']} | {pct(s['recall_1mm'], 1)} | {pct(s['precision_1mm'], 1)} "
                         f"| {pct(s['circle_recall'])} | {pct(s['text_exact'])} | {pct(s['text_cer'], 1)} |")
    for r in results["real"]:
        lines += ["", f"Real photo {r['file']}: sheet found by {r['sheet']}, rotation {r['rotation_deg']}°, {r['counts']}, "
                      f"{num(r['seconds'], 1)} s"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
