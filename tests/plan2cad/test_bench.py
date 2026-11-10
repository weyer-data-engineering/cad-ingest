"""The benchmark keeps each conversion of a run and measures it again on request (bench/run.py)."""

import json
import pickle
import sys
from dataclasses import replace

import numpy as np
import pytest

import run  # bench/run.py; bench is on the path (conftest)
from app.plan2cad.model import Circle, Drawing, Line, Text
from app.plan2cad.pipeline import Result


def sheet(seed: int, paper: str = "A4") -> Drawing:
    """A small reference, a little different per seed."""
    return Drawing(297, 210, lines=[Line(10, 10, 287, 10), Line(10, 10, 10, 200 - seed)],
                   circles=[Circle(150, 150, 5)], texts=[Text(100, 100, 2.5, f"Plan {seed}")])


def stand_in(img, paper: str, **options) -> Result:
    """The pipeline's part in a run: the image is the reference itself, and comes back 0.7 mm off."""
    d = replace(img, lines=[replace(q, y1=q.y1 + 0.7, y2=q.y2 + 0.7) for q in img.lines])
    return Result(d, np.zeros((1, 1), np.uint8), 4.0, {"source_ppm": 4.0})


@pytest.fixture
def bench(monkeypatch, tmp_path):
    """bench/run.py writing to tmp_path, with one small case, a sweep of one resolution and a stand-in pipeline."""
    monkeypatch.setattr(run, "CASES", [("Test sheet, clean plot", "test", sheet, "A4", lambda d, s: d, [1, 2])])
    monkeypatch.setattr(run, "SWEEP", [4])
    monkeypatch.setattr(run, "pid", sheet)
    monkeypatch.setattr(run, "scan", lambda d, seed, dpi, blur: d)
    monkeypatch.setattr(run, "convert", stand_in)
    monkeypatch.setattr(run, "panel", lambda *args: None)

    def main(*argv: str) -> dict:
        monkeypatch.setattr(sys, "argv", ["run.py", "--out", str(tmp_path), *argv])
        run.main()
        return json.loads((tmp_path / "results.json").read_text())
    return main


def test_reevaluate_reproduces_the_run(bench, tmp_path):
    first = bench()
    report = (tmp_path / "report.md").read_text()
    assert first["cases"][0]["seeds"] == 2 and first["cases"][0]["recall_05mm"] < 1 and len(first["sweep"]) == 1
    assert sorted(p.name for p in (tmp_path / "pred").iterdir()) == \
        ["Test_sheet_clean_plot_1.pkl", "Test_sheet_clean_plot_2.pkl", "sweep_4.pkl"]
    assert bench("--reevaluate") == first
    assert (tmp_path / "report.md").read_text() == report


def test_reevaluate_applies_the_current_measures(bench, monkeypatch):
    first = bench("--only", "test")
    measure = run.evaluate
    monkeypatch.setattr(run, "evaluate", lambda gt, pred: measure(gt, pred) | {"crumbs": 7})
    again = bench("--reevaluate")
    assert again["cases"][0]["crumbs"] == 7 and first["cases"][0]["crumbs"] == 0
    assert again["cases"][0]["seconds"] == first["cases"][0]["seconds"]  # the runtime is the run's


def test_reevaluate_measures_a_run_saved_before_the_model_grew(bench, tmp_path):
    first = bench("--only", "test")
    path = tmp_path / "pred" / "Test_sheet_clean_plot_1.pkl"
    record = pickle.loads(path.read_bytes())
    del record["drawing"].__dict__["markups"], record["reference"].lines[0].__dict__["dashed"]  # not in the model yet
    path.write_bytes(pickle.dumps(record))
    assert bench("--reevaluate") == first


def test_seeds():
    assert run.variants([1, 2, 3], None) == [1, 2, 3]  # without --seeds as before
    assert run.variants([1, 2, 3], 1) == [1]
    assert run.variants([1, 2], 4) == [1, 2, 3, 4]


def test_seeds_per_case_and_a_new_run_replaces_the_saved_one(bench, tmp_path):
    assert bench("--only", "test", "--seeds", "3")["cases"][0]["seeds"] == 3
    assert bench("--quick")["cases"][0]["seeds"] == 1
    assert [p.name for p in (tmp_path / "pred").iterdir()] == ["Test_sheet_clean_plot_1.pkl"]


def test_reevaluate_without_a_saved_run(bench, tmp_path):
    with pytest.raises(SystemExit, match="no saved conversions"):
        bench("--reevaluate")
    (tmp_path / "pred").mkdir()
    (tmp_path / "pred" / "broken.pkl").write_bytes(b"not a pickle")
    with pytest.raises(SystemExit, match="broken.pkl: cannot be read"):
        bench("--reevaluate")
    with pytest.raises(SystemExit):  # the saved run is measured as it was
        bench("--reevaluate", "--only", "test")
