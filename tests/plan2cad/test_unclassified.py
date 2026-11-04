import io

import ezdxf
import numpy as np
from synth import plot  # bench/synth.py; bench is on the path (conftest)

from app import scan
from app.plan2cad import titleblock, unclassified
from app.plan2cad.export import to_dxf
from app.plan2cad.features import Features
from app.plan2cad.library import load, place
from app.plan2cad.model import Circle, Dimension, Drawing, Line, Polyline, Text
from app.plan2cad.pipeline import convert

from .test_titleblock import closed_block


def bowtie(x: float, y: float) -> list[Line]:
    """A valve the libraries do not have: two triangles meeting in the middle, 5 x 3.2 mm."""
    return [Line(x - 2.5, y - 1.6, x - 2.5, y + 1.6), Line(x - 2.5, y + 1.6, x + 2.5, y - 1.6),
            Line(x + 2.5, y - 1.6, x + 2.5, y + 1.6), Line(x + 2.5, y + 1.6, x - 2.5, y - 1.6)]


def test_closed_clusters_of_symbol_size_between_pipes_are_marked_one_by_one():
    pipes = [Line(10, 50, 47.5, 50, 0.5), Line(52.5, 50, 57.5, 50, 0.5), Line(62.5, 50, 100, 50, 0.5)]
    d = Drawing(200, 100, lines=pipes + bowtie(50, 50) + bowtie(60, 50) + [Line(150, 50, 152, 50)],
                texts=[Text(120, 20, 2.5, "P-101")])
    unclassified.find(d)
    assert [tuple(round(v, 1) for v in u.box) for u in d.unclassified] == [(47.5, 48.4, 52.5, 51.6),
                                                                            (57.5, 48.4, 62.5, 51.6)]
    assert "4 strokes with 2 closed shapes" in d.unclassified[0].why  # its two triangles
    assert [q.layer for q in d.lines] == [None] * 3 + [unclassified.LAYER] * 8 + [None]  # pipes and crumb stay


def test_rings_letters_texts_and_symbols_found_are_not_marked():
    valve = place(load()["GATE_VALVE"], 150, 50)
    d = Drawing(200, 100, lines=[Line(10, 10, 12, 10), Line(12, 10, 12, 12.5), Line(12, 12.5, 10, 12.5),
                                 Line(10, 12.5, 10, 10)]  # a letter-sized box
                + bowtie(80, 50) + valve.parts,  # under a text, and under a symbol found
                circles=[Circle(30, 50, 5), Circle(30, 50, 0.6)],
                texts=[Text(80, 50, 3.5, "XYZ"), Text(30, 50, 2.5, "TI")], symbols=[valve])
    unclassified.find(d)
    assert not d.unclassified and not any(q.layer for q in d.lines)


def test_a_sheet_without_linework_has_none():
    for d in (Drawing(200, 100), Drawing(200, 100, circles=[Circle(30, 50, 5)])):
        unclassified.find(d)
        assert not d.unclassified


def test_unclassified_symbols_are_listed_in_the_report_markdown_and_on_their_layer():
    d = Drawing(200, 100, lines=bowtie(50, 50))
    unclassified.find(d)
    assert scan._report(d)["unclassified"] == [{"box_mm": [47.5, 48.4, 52.5, 51.6], "why": d.unclassified[0].why}]
    info = {"drawing_type": "pid", "paper": "A3", "sheet_size_mm": [200, 100], "sheet": "paper", "scale": None,
            "counts": d.counts()}
    assert "## Symbols not classified (no library symbol recognised)" in scan._page_markdown(info, d, "##")
    buf = io.StringIO()
    to_dxf(d).write(buf)
    doc = ezdxf.read(io.StringIO(buf.getvalue()))
    assert {e.dxf.layer for e in doc.modelspace()} == {unclassified.LAYER}
    assert d.scaled(2).unclassified[0].box == (95.0, 96.8, 105.0, 103.2)
    assert d.scaled(0.5).unclassified[0].why == d.unclassified[0].why and " mm" not in d.unclassified[0].why  # the box
    # gives the size, in sheet mm once a drawing printed small is brought to the sheet's size


def outline(*points) -> list[Line]:
    return [Line(*p, *q) for p, q in zip(points, points[1:] + points[:1])]


def test_a_frame_around_a_text_a_wall_and_equipment_are_not_marked_but_a_lone_shape_is():
    texts = [Text(x, 90, 2.5, "M") for x in range(10, 200, 10)]  # the sheet's usual letters: 2.5 mm
    box = outline((10, 10), (22, 10), (22, 18), (10, 18))  # around two lines of text
    triangle = outline((40, 18), (46, 18), (43, 12.8))  # a revision mark
    connector = outline((60, 10), (72, 10), (76, 14), (72, 18), (60, 18))  # an off-page connector holds a text too
    column = Polyline([(100, 10), (106, 10), (106, 16), (100, 16), (100, 10), (106, 16)], [0.0] * 5, kind="wall")
    d = Drawing(200, 100, lines=box + triangle + connector, polylines=[column],
                texts=texts + [Text(16, 12.5, 2.5, "DN50"), Text(16, 15.5, 2.5, "PN16"), Text(43, 16.5, 2.5, "3"),
                               Text(66, 14, 2.5, "1A")])
    unclassified.find(d)
    assert [u.box for u in d.unclassified] == [(60.0, 10.0, 76.0, 18.0)]  # five strokes: the connector only
    lone = Drawing(200, 100, polylines=[Polyline(column.points, column.bulges)], texts=texts)
    unclassified.find(lone)
    assert [u.box for u in lone.unclassified] == [(100.0, 10.0, 106.0, 16.0)]  # one item alone is marked too


def test_a_box_around_the_value_of_a_dimension_is_its_frame_as_around_a_text():
    def box() -> list[Line]:
        return outline((97, 48), (103, 48), (103, 52), (97, 52))  # a closed box of four strokes around "2,50"

    value = Text(100, 50, 2.5, "2,50")
    as_text = Drawing(200, 100, lines=box(), texts=[value])
    as_value = Drawing(200, 100, lines=box(), dimensions=[Dimension(80, 55, 120, 55, "2,50", (80, 60), (120, 60),
                                                                    text=value)])
    bare = Drawing(200, 100, lines=box())
    for d in (as_text, as_value, bare):
        unclassified.find(d)
    assert not as_text.unclassified and not as_value.unclassified  # the values of the dimensions are texts too
    assert len(bare.unclassified) == 1  # with nothing inside, the box alone looks like a symbol


def test_a_shape_in_the_title_block_is_the_blocks_not_a_symbol_not_classified():
    def sheet() -> Drawing:  # in the left cell of the title block, as a logo: a triangle in a ring, three rings
        d = closed_block()
        d.lines += outline((192, 177), (198, 177), (195, 172))
        d.circles += [Circle(195, 175.3, 3.6), Circle(208, 176, 2), Circle(212, 176, 2), Circle(210, 179.46, 2)]
        return d

    d = sheet()
    d.title_block = titleblock.find(d)  # as convert does it: the title block first, with its linework on its layer
    titleblock.lay(d, d.title_block)
    unclassified.find(d)
    assert not d.unclassified and [q.layer for q in d.lines[-3:] + d.circles] == [titleblock.LAYER] * 7
    alone = sheet()  # where no title block is found, both look like symbols
    unclassified.find(alone)
    assert [u.why.split(",")[0] for u in alone.unclassified] == ["4 strokes with 4 closed shapes",
                                                                  "3 strokes with 3 closed shapes"]


LETTERS = [Text(x, 90, 2.5, "M") for x in range(10, 200, 10)]  # the sheet's usual letters: 2.5 mm


def boxes(*xs: float, high: float) -> list[Line]:
    """Two boxes side by side, 3 mm wide each and ``high`` mm high, at each x of ``xs``: a shape of two openings."""
    return [q for x in xs for q in outline((x, 20), (x + 6, 20), (x + 6, 20 + high), (x, 20 + high))
            + [Line(x + 3, 20, x + 3, 20 + high)]]


def test_a_row_of_letters_not_read_is_lower_than_a_letter_however_long_and_is_not_marked():
    d = Drawing(200, 100, lines=boxes(20, high=2.2) + boxes(60, high=3.0), texts=LETTERS)  # an "88" not read: 2.2 mm
    unclassified.find(d)
    assert [u.box for u in d.unclassified] == [(60.0, 20.0, 66.0, 23.0)]  # higher than the letters: a symbol
    assert {q.layer for q in d.lines[:5]} == {None}
    bare = Drawing(200, 100, lines=boxes(20, high=2.2))  # a sheet without texts tells no letter height
    unclassified.find(bare)
    assert [u.box for u in bare.unclassified] == [(20.0, 20.0, 26.0, 22.2)]


def hatched_corner(x: float, y: float) -> Polyline | list[Line]:
    """The corner of an outline, 12 mm each way, as one polyline, and its hatching at 45 degrees, 1 mm apart."""
    side = Polyline([(x + 12, y), (x, y), (x, y + 12)], [0.0, 0.0])
    return [side] + [Line(x, y + k, x + k, y) for k in np.arange(1.4, 12, 1.4)]


def test_a_corner_of_hatching_encloses_only_narrow_openings_and_is_not_marked():
    corner = hatched_corner(100, 20)
    barred = outline((20, 20), (32, 20), (32, 32), (20, 32)) + [Line(20, 32, 32, 20)]  # same size: a box, a bar
    d = Drawing(200, 100, lines=corner[1:] + barred, polylines=corner[:1], texts=LETTERS)
    unclassified.find(d)
    assert [u.box for u in d.unclassified] == [(20.0, 20.0, 32.0, 32.0)]
    assert unclassified._closed(corner, (100, 20, 112, 32))[0] >= 5  # the slivers between the hatch lines are closed,
    assert unclassified._closed(corner, (100, 20, 112, 32))[1] < 1.0  # but under 1 mm wide: under a fifth of 12 mm
    width = unclassified._closed(outline((0, 0), (10, 0), (10, 10), (0, 10)), (0, 0, 10, 10))[1]
    assert abs(width - 9.5) < 0.3  # a 10 mm box drawn with strokes 0.5 mm wide holds a circle of 9.5 mm


def test_the_step_runs_only_when_switched_on(monkeypatch):
    calls = []
    monkeypatch.setattr(unclassified, "find", lambda d, largest=None: calls.append(largest))
    sheet = plot(Drawing(150, 100, lines=[Line(20, 50, 130, 50, 0.5)]), 300)
    convert(sheet, paper="A4", dpi=300, features=Features.of(off=["unclassified"]))
    assert not calls  # off: the linework stays on its usual layers, as before
    convert(sheet, paper="A4", dpi=300)
    assert len(calls) == 1
