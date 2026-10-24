"""Benchmark: synthetic drawings with ground truth, converted from simulated scans and phone photos.

    python bench/run.py --out bench/out [--quick] [--real photo.heic:A4 ...]

Writes results.json, report.md, side-by-side panels and the resolution sweep chart to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent)]  # bench and the app

from evaluate import evaluate  # noqa: E402
from synth import building_plan, photo, pid, scan  # noqa: E402

from app.plan2cad.export import overlay, to_figure  # noqa: E402
from app.plan2cad.imageio import load_image  # noqa: E402
from app.plan2cad.pipeline import convert  # noqa: E402

MP48 = (8064, 6048)
CASES = [  # name, drawing, paper, image, seeds
    ("Floor plan A3, scan 300 dpi", building_plan, "A3", lambda d, s: scan(d, s, dpi=300), [1, 2, 3]),
    ("Floor plan A3, phone photo 12 MP", building_plan, "A3", lambda d, s: photo(d, s), [1, 2, 3]),
    ("Floor plan A3, phone photo 12 MP, hard", building_plan, "A3", lambda d, s: photo(d, s, hard=True), [1, 2, 3]),
    ("Floor plan A3 with fixtures, scan 300 dpi", lambda s: building_plan(s, fixtures=True), "A3",
     lambda d, s: scan(d, s, dpi=300), [1, 2, 3]),
    ("Floor plan A3 with fixtures, phone photo 12 MP", lambda s: building_plan(s, fixtures=True), "A3",
     lambda d, s: photo(d, s), [1, 2, 3]),
    ("Floor plan A3 1:100 with fixtures, scan 300 dpi", lambda s: building_plan(s, fixtures=True, scale=100), "A3",
     lambda d, s: scan(d, s, dpi=300), [1, 2, 3]),
    ("Floor plan A3 1:100 with fixtures, phone photo 12 MP", lambda s: building_plan(s, fixtures=True, scale=100),
     "A3", lambda d, s: photo(d, s), [1, 2, 3]),
    ("P&ID A1, scan 200 dpi", lambda s: pid(s, "A1"), "A1", lambda d, s: scan(d, s, dpi=200), [1, 2]),
    ("P&ID A1, phone photo 12 MP", lambda s: pid(s, "A1"), "A1", lambda d, s: photo(d, s), [1, 2]),
    ("P&ID A0, scan 200 dpi", lambda s: pid(s, "A0"), "A0", lambda d, s: scan(d, s, dpi=200), [1]),
    ("P&ID A0, phone photo 12 MP", lambda s: pid(s, "A0"), "A0", lambda d, s: photo(d, s), [1]),
    ("P&ID A0, phone photo 48 MP", lambda s: pid(s, "A0"), "A0", lambda d, s: photo(d, s, camera=MP48), [1]),
]
SWEEP = [3, 4, 5, 6, 8, 10]  # px/mm on the A0 P&ID
KEYS = ["recall_1mm", "precision_1mm", "recall_05mm", "precision_05mm", "deviation_mm", "dashed_recall_1mm",
        "circle_recall", "circle_precision", "text_exact", "text_cer", "text_false_positives", "text_review",
        "text_unflagged", "text_exact_codes", "text_exact_numbers", "text_exact_words",
        "pieces_per_object", "objects_whole", "entities", "reference_objects", "short_pieces",
        "breaks_per_m", "curve_pieces", "curves_as_arcs",
        "symbol_recall", "symbol_precision", "symbol_attributes", "symbol_review", "symbol_unflagged",
        "connection_recall", "connection_precision",
        "scale_ok", "seconds", "source_ppm"]


def run_one(d, paper, img):
    started = time.perf_counter()
    r = convert(img, paper=paper)
    m = evaluate(d, r.drawing) | {"seconds": time.perf_counter() - started, "source_ppm": r.info["source_ppm"]}
    return r, m


def mean(rows: list[dict]) -> dict:
    out = {}
    for k in KEYS:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    return out


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
    ap.add_argument("--real", nargs="*", default=[], help="real photos as path:paper (no ground truth)")
    ap.add_argument("--report-only", action="store_true", help="rebuild report and chart from results.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        results = json.loads((out / "results.json").read_text())
        if results["sweep"]:
            sweep_chart(results["sweep"], out / "sweep.png")
        (out / "report.md").write_text(report(results))
        return
    results = {"cases": [], "sweep": [], "real": []}

    for name, make, paper, image, seeds in CASES:
        if args.quick and (paper == "A0"):
            continue
        rows = []
        for seed in seeds[:1] if args.quick else seeds:
            d = make(seed)
            img = image(d, seed)
            r, m = run_one(d, paper, img)
            rows.append(m)
            shown = ("recall_1mm", "precision_1mm", "text_exact", "symbol_recall", "symbol_precision", "seconds")
            print(f"{name} seed {seed}: " + ", ".join(f"{k}={m[k]:.3f}" for k in shown if m.get(k) is not None), flush=True)
            if seed == seeds[0]:
                panel(img, r, out, name.replace(" ", "_").replace(",", "").replace("&", ""))
        results["cases"].append({"name": name, "paper": paper, "seeds": len(rows)} | mean(rows))

    if not args.quick:
        d = pid(1, "A0")
        for ppm in SWEEP:
            r, m = run_one(d, "A0", scan(d, 1, dpi=ppm * 25.4, blur=0.8))
            results["sweep"].append({"ppm": ppm} | {k: m.get(k) for k in KEYS})
            print(f"sweep {ppm} px/mm: recall={m['recall_1mm']:.3f} text={m['text_exact']:.3f}", flush=True)
        sweep_chart(results["sweep"], out / "sweep.png")

    for spec in args.real:
        path, paper = spec.rsplit(":", 1)
        img, dpi = load_image(Path(path).read_bytes())
        started = time.perf_counter()
        r = convert(img, paper=paper, dpi=dpi)
        info = r.info | {"seconds": time.perf_counter() - started, "file": Path(path).name}
        results["real"].append(info)
        panel(cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), r, out, "real_" + Path(path).stem)
        print(f"real {path}: {info['counts']} in {info['seconds']:.1f} s", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    (out / "report.md").write_text(report(results))


def report(results: dict) -> str:
    lines = ["# cad-ingest photo and scan benchmark", "",
             "Synthetic drawings with ground truth; deviations measured on paper.", "",
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
              "piece). Curves drawn as arcs: share of the reference arcs drawn as arcs, not as short straight lines.",
              "",
              "| Case | Breaks per metre | Pieces per curve | Curves drawn as arcs |", "|---|---|---|---|"]
    for c in results["cases"]:
        lines.append(f"| {c['name']} | {num(c.get('breaks_per_m'), 1)} | {num(c.get('curve_pieces'))} "
                     f"| {pct(c.get('curves_as_arcs'))} |")
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
