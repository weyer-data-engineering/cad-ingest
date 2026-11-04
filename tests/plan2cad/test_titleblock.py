"""The title block as a table (titleblock.py), in the report, the markdown and the DXF, and its benchmark measure."""
import math
from dataclasses import replace

import pytest
from evaluate import title_cells

from app import scan
from app.plan2cad import objects, titleblock, unclassified
from app.plan2cad.export import to_dxf
from app.plan2cad.features import Features
from app.plan2cad.model import Drawing, Line, Polyline, Text
from app.plan2cad.pipeline import convert


def closed_block() -> Drawing:
    """An A4 sheet with a frame and a title block with an outline at its bottom right: on the left a cell over the
    whole height of the block, on the right three rows."""
    d = Drawing(297, 210, polylines=[Polyline([(20, 10), (287, 10), (287, 200), (20, 200)], [0.0] * 4, closed=True,
                                              width=0.7, kind="frame")])
    d.lines += [Line(187, 170, 187, 200), Line(187, 170, 287, 170), Line(227, 170, 227, 200), Line(227, 180, 287, 180),
                Line(227, 190, 287, 190), Line(60, 60, 120, 60)]  # the last one is the drawing
    d.texts += [Text(207, 185, 3.5, "A-101"), Text(257, 175, 2.5, "Pump station"), Text(250, 185, 2.5, "Scale"),
                Text(268, 185, 2.5, "1:50"), Text(257, 195, 2.5, "Sheet 1/1"), Text(90, 50, 2.5, "Pump")]
    return d


def test_the_title_block_of_a_drawing_is_its_table_of_cells(plan):
    found = titleblock.find(plan)  # the reference of the benchmark: a block open at its top and left side
    assert found.box == pytest.approx(plan.title_block.box)
    assert [(c.row, c.col, c.rows, c.cols, c.text) for c in found.cells] == \
        [(c.row, c.col, c.rows, c.cols, c.text) for c in plan.title_block.cells]
    assert all(c.box == pytest.approx(r.box) for c, r in zip(found.cells, plan.title_block.cells))


def test_cells_that_no_ruling_divides_are_one_spanning_several():
    d = closed_block()
    found = titleblock.find(d)
    assert found.box == (187, 170, 287, 200)
    assert [(c.row, c.col, c.rows, c.cols, c.text) for c in found.cells] == [
        (0, 0, 3, 1, "A-101"), (0, 1, 1, 1, "Pump station"), (1, 1, 1, 1, "Scale 1:50"), (2, 1, 1, 1, "Sheet 1/1")]


def test_its_lines_go_on_their_own_layer_but_the_frame_stays():
    d = closed_block()
    titleblock.lay(d, titleblock.find(d))
    assert [ln.layer for ln in d.lines] == [titleblock.LAYER] * 5 + [None]
    assert d.polylines[0].layer is None
    layers = [e.dxf.layer for e in to_dxf(d).modelspace().query("LINE")]
    assert layers.count(titleblock.LAYER) == 5


@pytest.mark.parametrize("curve", ["arc", "wave"])
def test_a_curve_traced_as_chords_in_the_block_cuts_no_cells_and_goes_on_its_layer(curve):
    d = closed_block()
    if curve == "arc":  # eight chords of an arc of 6 mm radius in the left cell, flat at its top
        points = [(207 + 6 * math.cos(math.radians(a)), 192 + 6 * math.sin(math.radians(a)))
                  for a in (200 + 17.5 * k for k in range(9))]
    else:  # a wave of 3 mm over 30 mm as chords of 1.5 mm, which becomes a spline
        points = [(x, 186 + 3 * math.sin(math.pi * (x - 192) / 15)) for x in (192 + 1.5 * k for k in range(21))]
    d.polylines.append(Polyline(points, [0.0] * (len(points) - 1)))
    cells = [(c.row, c.col, c.rows, c.cols, c.text) for c in titleblock.find(closed_block()).cells]
    assert [(c.row, c.col, c.rows, c.cols, c.text) for c in titleblock.find(d).cells] != cells  # chords as rulings
    objects.curves(d)  # as the pipeline draws them before it reads the title block
    block = titleblock.find(d)
    titleblock.lay(d, block)
    assert [(c.row, c.col, c.rows, c.cols, c.text) for c in block.cells] == cells
    assert [q.layer for q in d.arcs + d.polylines[1:]] == [titleblock.LAYER]


@pytest.mark.parametrize("change", ["no rulings", "no frame", "one text"])
def test_no_table_without_a_ruled_block_with_texts_at_the_frame(change):
    d = closed_block()
    if change == "no rulings":
        d.lines = []
    elif change == "no frame":
        d.polylines = []
    else:
        d.texts = d.texts[:1]
    assert titleblock.find(d) is None


def test_the_report_and_the_markdown_have_the_table():
    d = closed_block()
    d.title_block = titleblock.find(d)
    d.title_block.cells[3].text, d.title_block.cells[3].uncertain = "Sheet 1|1", True
    cells = scan._report(d)["title_block"]["cells"]
    assert scan._report(d)["title_block"] | {"cells": None} == {"box_mm": [187, 170, 287, 200], "rows": 3, "cols": 2,
                                                                 "cells": None}
    assert cells[0] == {"row": 0, "col": 0, "row_span": 3, "col_span": 1, "box_mm": [187, 170, 227, 200],
                        "text": "A-101", "uncertain": False}
    info = {"drawing_type": None, "paper": "A4", "sheet_size_mm": [297, 210], "sheet": "paper", "scale": 50,
            "counts": {"lines": 6}}
    markdown = "\n".join(scan._page_markdown(info, d, "##"))
    assert "## Title block\n\n| | |\n|---|---|\n| A-101 | Pump station |\n|  | Scale 1:50 |\n" \
           "|  | Sheet 1\\|1 (to check) |" in markdown
    assert "title_block" not in scan._report(closed_block())  # none found, none reported


def test_title_cells_are_measured_by_their_box_and_text(plan):
    assert title_cells(plan, plan) == {"title_cells_found": 1.0, "title_cells_exact": 1.0}
    wrong = replace(plan.title_block, cells=[replace(c) for c in plan.title_block.cells])
    wrong.cells[0].text = "Muster GmbH Archltekten"
    assert title_cells(plan, replace(plan, title_block=wrong)) == {"title_cells_found": 1.0,
                                                                    "title_cells_exact": pytest.approx(5 / 6)}
    assert title_cells(plan, replace(plan, title_block=None)) == {"title_cells_found": 0.0, "title_cells_exact": 0.0}
    assert title_cells(Drawing(10, 10), plan) == {}


@pytest.mark.parametrize("measured, right", [(0.46, True), (0.3, False), (1.0, False)])
def test_the_print_size_is_right_within_15_percent(measured, right):
    from evaluate import evaluate

    def drawing(printed: float) -> Drawing:
        return Drawing(100, 100, lines=[Line(10, 10, 90, 10)], printed=printed)

    assert evaluate(drawing(0.5), drawing(measured))["printed_ok"] is right


def test_a_scan_gets_its_title_block_and_without_the_step_stays_as_it_was(plan, plan_scan, monkeypatch):
    laid, find = [], unclassified.find  # the lines on the title block's layer when shapes not classified are looked for

    def after_the_title_block(d, *args):
        laid.append(sum(q.layer == titleblock.LAYER for q in d.lines + d.polylines))
        find(d, *args)

    monkeypatch.setattr(unclassified, "find", after_the_title_block)
    on = convert(plan_scan, paper="A3").drawing
    off = convert(plan_scan, paper="A3", features=Features.of(off=["title_block"])).drawing
    assert laid[0] >= 5 and laid[1] == 0  # so none of the block's shapes is listed, then moved onto its layer
    assert title_cells(plan, on)["title_cells_exact"] >= 5 / 6
    assert sum(q.layer == titleblock.LAYER for q in on.lines + on.polylines) >= 5
    assert off.title_block is None and not any(q.layer for q in off.lines + off.polylines)
    assert [(pl.points, pl.kind) for pl in on.polylines] == [(pl.points, pl.kind) for pl in off.polylines]
    assert [t.text for t in on.texts] == [t.text for t in off.texts]


ROWS = [[("Muster GmbH", 3.0)], [("Projekt", 3.0)], [("Titel", 5.0)], [("1:50", 2.5), ("A-1", 2.5), ("Blatt 1", 2.5)]]


@pytest.mark.parametrize("width, height, block", [(297, 210, 170), (297, 420, 170), (210, 297, 180)])
def test_a_block_wider_than_half_the_sheet_keeps_its_bottom_rows(width, height, block):
    from synth import Pen, frame

    d = Drawing(width, height)
    frame(Pen(d), ROWS, tb_w=block)  # A4 landscape, A3 portrait, A4 portrait with a block across the whole frame
    x0, y0, x1, y1 = d.title_block.box
    d.lines += [Line(x0, y0, x1, y0, 0.35), Line(x0, y0, x0, y1, 0.35)]  # closed, as ISO 7200 draws it
    found = titleblock.find(d)
    assert found.box == pytest.approx(d.title_block.box)
    assert [(c.row, c.col, c.rows, c.cols, c.text) for c in found.cells] == \
        [(c.row, c.col, c.rows, c.cols, c.text) for c in d.title_block.cells]


def test_a_line_across_the_frame_is_no_title_block(plan):
    d = replace(plan, lines=plan.lines + [Line(20, plan.title_block.box[1] - 5, 410, plan.title_block.box[1] - 5)])
    assert titleblock.find(d).box == pytest.approx(plan.title_block.box)  # not the strip under the line


def test_a_text_just_above_an_open_block_stays_out_of_it(plan):
    x0, y0, _, _ = plan.title_block.box
    d = replace(plan, texts=plan.texts + [Text(x0 + 100, y0 - 3, 2.5, "Raum 1.04")])
    found = titleblock.find(d)
    assert found.box == pytest.approx(plan.title_block.box)  # the top row as high as the others
    assert [c.text for c in found.cells] == [c.text for c in plan.title_block.cells]
