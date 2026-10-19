import io
import sys
import zipfile
from pathlib import Path

import cv2
import ezdxf
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "bench"))

from synth import building_plan, scan  # noqa: E402

from app.plan2cad import library  # noqa: E402
from app.plan2cad.model import Drawing, Line, Text  # noqa: E402
from app.plan2cad.render import render  # noqa: E402


@pytest.fixture(scope="session")
def plan():
    return building_plan(7)


@pytest.fixture(scope="session")
def plan_scan(plan):
    return scan(plan, 7, dpi=200)


def write_library(folder: Path, lib_id: str = "acme-pid", units: int = ezdxf.units.MM, scale: float = 1.0,
                  base: tuple = (0, 0)) -> Path:
    """A small company library: MY_VALVE, a box with a bar between two ports and a TAG below it, drawn in ``units``
    (``scale`` drawing units per millimetre) around the block's base point ``base``. Unlike any starter symbol."""
    folder.mkdir(parents=True)
    (folder / "library.toml").write_text(f'id = "{lib_id}"\nname = "ACME symbols"\nversion = "2.1.0"\n'
                                         'licence = "LicenseRef-ACME-internal"\n')
    doc = ezdxf.new("R2018", units=units)
    doc.layers.add("PORT")

    def at(x, y):
        return base[0] + x * scale, base[1] + y * scale

    block = doc.blocks.new("MY_VALVE", base_point=base, dxfattribs={"description": "Box valve"})
    block.add_lwpolyline([at(-3, -1.5), at(3, -1.5), at(3, 1.5), at(-3, 1.5)], close=True)
    block.add_line(at(0, -1.5), at(0, 1.5))
    for x in (-3, 3):
        block.add_point(at(x, 0), dxfattribs={"layer": "PORT"})
    block.add_attdef("TAG", at(0, -6), dxfattribs={"height": 2.0 * scale})
    doc.saveas(folder / "symbols.dxf")
    return folder


def zipped(folder: Path, prefix: str = "") -> bytes:
    """The files of a library folder as a ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for path in folder.iterdir():
            z.write(path, prefix + path.name)
    return buf.getvalue()


def valve_scan(lib: library.Library) -> bytes:
    """A 300 dpi scan (PNG) of an A4 sheet with a pipe through the MY_VALVE of ``lib``, tagged V-101."""
    d = Drawing(297, 210, texts=[Text(80, 30, 3.5, "Library test sheet"), Text(210, 30, 3.5, "Pipe with valve"),
                                 Text(80, 180, 3.5, "Muster GmbH")])
    valve = library.place(lib.symbols["MY_VALVE"], 150, 100)
    valve.attributes = {"TAG": Text(152, 106, 3.0, "V-101")}
    d.symbols.append(valve)
    d.lines += [Line(90, 100, 147, 100, 0.5), Line(153, 100, 210, 100, 0.5)]
    return cv2.imencode(".png", render(d, 300 / 25.4))[1].tobytes()
