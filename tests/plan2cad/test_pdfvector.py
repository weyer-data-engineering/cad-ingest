import ctypes
import io
import math

import numpy as np
import pypdfium2 as pdfium
import pytest
from evaluate import evaluate
from PIL import Image
from pypdfium2 import raw
from synth import pid

from app.plan2cad import pdfvector
from app.plan2cad.export import to_dxf, to_figure

K = 0.5523  # handle length of a Bézier quarter circle, per radius


def cad_page(rotate: int = 0) -> bytes:
    """An A4 page as a CAD program exports it: orange pipes on layer "Rohrleitungen", a valve circle of four
    Béziers on layer "Armatüren" (the name in PDFDocEncoding), a blue fill and two texts, one of them vertical."""
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(595, 842)

    def put(obj, layer: bytes | None = None):
        if layer:
            mark = raw.FPDFPageObj_AddMark(obj, b"OC")
            raw.FPDFPageObjMark_SetStringParam(pdf.raw, obj, mark, b"Name", layer)
        raw.FPDFPage_InsertObject(page.raw, obj)

    for i in range(20):
        p = raw.FPDFPageObj_CreateNewPath(100 + 10 * i, 100)
        raw.FPDFPath_LineTo(p, 100 + 10 * i, 300)
        raw.FPDFPath_SetDrawMode(p, 0, True)
        raw.FPDFPageObj_SetStrokeColor(p, 255, 128, 0, 255)
        raw.FPDFPageObj_SetStrokeWidth(p, ctypes.c_float(1.0))
        put(p, b"Rohrleitungen")
    c = raw.FPDFPageObj_CreateNewPath(400, 500)  # radius 20 around (380, 500)
    for x1, y1, x2, y2, x3, y3 in [(400, 500 + 20 * K, 380 + 20 * K, 520, 380, 520),
                                   (380 - 20 * K, 520, 360, 500 + 20 * K, 360, 500),
                                   (360, 500 - 20 * K, 380 - 20 * K, 480, 380, 480),
                                   (380 + 20 * K, 480, 400, 500 - 20 * K, 400, 500)]:
        raw.FPDFPath_BezierTo(c, x1, y1, x2, y2, x3, y3)
    raw.FPDFPath_Close(c)
    raw.FPDFPath_SetDrawMode(c, 0, True)
    raw.FPDFPageObj_SetStrokeWidth(c, ctypes.c_float(0.5))
    put(c, b"Armat\xfcren")
    fill = raw.FPDFPageObj_CreateNewRect(100, 600, 50, 30)
    raw.FPDFPath_SetDrawMode(fill, raw.FPDF_FILLMODE_WINDING, False)
    raw.FPDFPageObj_SetFillColor(fill, 0, 0, 255, 255)
    put(fill)
    for s, angle, x, y in (("DN 50", 0, 100, 700), ("PI 101", 90, 300, 650)):
        t = raw.FPDFPageObj_NewTextObj(pdf.raw, b"Helvetica", ctypes.c_float(10))
        text = ctypes.create_string_buffer((s + "\0").encode("utf-16-le"))
        raw.FPDFText_SetText(t, ctypes.cast(text, raw.FPDF_WIDESTRING))
        a = math.radians(angle)
        raw.FPDFPageObj_Transform(t, math.cos(a), math.sin(a), -math.sin(a), math.cos(a), x, y)
        put(t)
    raw.FPDFPage_GenerateContent(page.raw)
    page.set_rotation(rotate)
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def read(data: bytes):
    pdf = pdfium.PdfDocument(data)
    page = pdf[0]
    return pdfvector.read(pdf, page), page


def test_cad_export_keeps_layers_colours_and_curves():
    d, page = read(cad_page())
    assert pdfvector.is_vector(page)
    assert (d.width, d.height) == pytest.approx((210, 297), abs=0.2)
    assert len(d.lines) == 20 and not d.polylines
    assert {(ln.color, ln.layer, round(ln.width, 3)) for ln in d.lines} == {((255, 128, 0), "Rohrleitungen", 0.353)}
    (c,) = d.circles
    assert (c.cx, c.cy, c.r, c.layer) == (pytest.approx(134.06, abs=0.01), pytest.approx(120.65, abs=0.01),
                                         pytest.approx(7.056, abs=0.01), "Armatüren")
    (g,) = d.polygons
    assert g.filled and g.color == (0, 0, 255)
    cap = 0.718 * 10 * pdfvector.PT  # Helvetica's capitals at 10 pt
    assert sorted((t.text, t.angle) for t in d.texts) == [("DN 50", 0.0), ("PI 101", 90.0)]
    assert all(t.height == pytest.approx(cap, rel=0.03) for t in d.texts)


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
def test_turned_page_is_read_as_shown(rotate):
    d, page = read(cad_page(rotate))
    ppm = 4 * 72 / 25.4
    shown = np.asarray(page.render(scale=4).to_pil().convert("L"))
    assert shown.shape == pytest.approx((d.height * ppm, d.width * ppm), abs=2)
    for e in d.lines:
        assert shown[int((e.y1 + e.y2) / 2 * ppm), int((e.x1 + e.x2) / 2 * ppm)] < 200
    (c,) = d.circles
    assert shown[int(c.cy * ppm), int((c.cx + c.r) * ppm)] < 200
    upright = {t.text: t.angle for t in d.texts}
    assert upright == {"DN 50": (-rotate) % 360, "PI 101": (90 - rotate) % 360}


def test_dxf_of_a_cad_export_keeps_its_layers_colours_and_pens():
    d, _ = read(cad_page())
    msp = to_dxf(d).modelspace()
    lines = msp.query("LINE")
    assert {(e.dxf.layer, e.rgb, e.dxf.lineweight) for e in lines} == {("Rohrleitungen", (255, 128, 0), 35)}
    assert msp.query("CIRCLE")[0].dxf.layer == "Armatüren"
    assert msp.query("CIRCLE")[0].dxf.lineweight == 18  # the nearest DXF pen to 0.176 mm, not a rounded P2C pen
    assert all(t.dxf.halign == 5 for t in msp.query("TEXT"))  # FIT between the ends of the baseline


def test_matplotlib_pdf_reads_back_exactly(tmp_path):
    gt = pid(3, "A3")
    to_figure(gt, tmp_path / "pid.pdf")
    d, _ = read((tmp_path / "pid.pdf").read_bytes())
    m = evaluate(gt, d)
    assert m["recall_05mm"] > 0.98 and m["precision_05mm"] > 0.99
    assert m["circle_recall"] == 1 and m["circle_precision"] == 1
    assert m["text_exact"] > 0.95
    assert not d.polygons  # the white sheet behind it all is paper, not a fill


def test_scan_in_a_pdf_is_not_a_vector_page():
    buf = io.BytesIO()
    Image.new("L", (1240, 1754), 255).save(buf, "PDF", resolution=150)
    assert not pdfvector.is_vector(pdfium.PdfDocument(buf.getvalue())[0])
