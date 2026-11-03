import math

import pypdfium2
import pytest

from app.plan2cad.export import to_dxf, to_figure
from app.plan2cad.model import Arc, Drawing, Line, Polyline, Text


@pytest.fixture
def drawing():
    return Drawing(420, 297, lines=[Line(10, 20, 110, 20, 0.5)], arcs=[Arc(50, 50, 10, 0, 90)],
                   texts=[Text(100, 100, 2.5, "Flur")], scale=50)


def test_dxf_is_real_world_millimetres_with_y_up(drawing):
    doc = to_dxf(drawing)
    msp = doc.modelspace()
    line = msp.query("LINE")[0]
    assert tuple(line.dxf.start)[:2] == pytest.approx((500, 277 * 50))
    assert line.dxf.lineweight == 50
    arc = msp.query("ARC")[0]  # clockwise on screen from right to down = counter-clockwise from 270 to 0
    assert (arc.dxf.radius, arc.dxf.start_angle, arc.dxf.end_angle) == pytest.approx((500, 270, 0))
    assert msp.query("TEXT")[0].dxf.height == pytest.approx(125)
    assert not doc.audit().has_errors


def test_pdf_has_the_paper_size(drawing, tmp_path):
    to_figure(drawing, tmp_path / "plan.pdf", "pdf")
    w, h = pypdfium2.PdfDocument(tmp_path / "plan.pdf")[0].get_size()
    assert (w * 25.4 / 72, h * 25.4 / 72) == pytest.approx((420, 297), abs=0.5)


def test_polyline_arcs_keep_their_side_in_dxf():
    door = Polyline([(0, 0), (10, 0), (0, 10)], [0.0, math.tan(math.pi / 8)], width=0.35, kind="door")  # around (0, 0)
    pl = to_dxf(Drawing(100, 100, polylines=[door])).modelspace().query("LWPOLYLINE")[0]
    assert pl.dxf.layer == "P2C-DOOR"
    arc = [e for e in pl.virtual_entities() if e.dxftype() == "ARC"][0]
    assert tuple(arc.dxf.center)[:2] == pytest.approx((0, 100), abs=1e-3) and arc.dxf.radius == pytest.approx(10)


def test_a_text_is_fitted_to_its_length_as_printed():
    text = Text(100, 100, 2.5, "P-101", length=12.0)
    e = to_dxf(Drawing(420, 297, texts=[text])).modelspace().query("TEXT")[0]
    assert e.dxf.halign == 5  # fit
    assert e.dxf.insert[0] == pytest.approx(94) and e.dxf.align_point[0] == pytest.approx(106)


def test_a_drawing_printed_small_is_written_at_its_drawn_size():
    d = Drawing(420, 297, lines=[Line(10, 20, 110, 20, 0.25)], texts=[Text(100, 100, 1.25, "P-101")], printed=0.5)
    msp = to_dxf(d).modelspace()
    line = msp.query("LINE")[0]
    assert tuple(line.dxf.end)[:2] == pytest.approx((220, 554)) and line.dxf.lineweight == 50
    assert msp.query("TEXT")[0].dxf.height == pytest.approx(2.5)
