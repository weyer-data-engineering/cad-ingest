"""The command line with symbol libraries: --library, `cad-ingest library check` and `cad-ingest library build`."""
import json
import re
import zipfile

import ezdxf
import pytest

from app import convert, library_tool
from app.plan2cad import library

from .conftest import valve_scan, write_library


def test_a_photo_is_read_with_the_given_library(tmp_path, capsys):
    folder = write_library(tmp_path / "acme")
    (tmp_path / "valve.png").write_bytes(valve_scan(library.read(folder)))
    args = [str(tmp_path / "valve.png"), str(tmp_path / "out"), "--paper", "A4", "--dpi", "300", "--drawing", "pid"]

    assert convert.main(args + ["--library", str(folder), "--library", "din-10628"]) == 0
    report = json.loads(zipfile.ZipFile(tmp_path / "out" / "valve.zip").read("report.json"))
    assert [lib["id"] for lib in report["libraries"]] == ["acme-pid", "din-10628"]
    assert [(s["name"], s["library"]) for s in report["symbols"]] == [("MY_VALVE", "acme-pid")]
    with pytest.raises(SystemExit):
        convert.main(args + ["--library", "nope"])
    assert "'nope' is neither an installed library" in capsys.readouterr().err


def test_check_lists_the_symbols_and_tests_them_on_a_scan(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(library_tool, "RUNS", library_tool.RUNS[:1])  # the scan only, to keep the test short
    assert convert.main(["library", "check", str(write_library(tmp_path / "acme"))]) == 0
    out = capsys.readouterr().out
    assert out.startswith("acme-pid 2.1.0: ACME symbols\nLicence: LicenseRef-ACME-internal.")
    assert re.search(r"MY_VALVE +6\.0 x +3\.0 mm, 2 ports, attributes TAG +Box valve", out)
    assert re.search(r"MY_VALVE +2/2", out) and "Found: scan 100% (2/2). Found where no symbol was drawn: scan 0." in out


def test_check_notes_what_keeps_symbols_from_being_found(tmp_path, capsys):
    doc = ezdxf.new("R2018", units=ezdxf.units.MM)
    doc.layers.add("PORT")
    at_port = doc.blocks.new("VALVE_AT_PORT")  # inserted at its left connection
    at_port.add_lwpolyline([(0, -1.6), (0, 1.6), (5, -1.6), (5, 1.6)], close=True)
    for x in (0, 5):
        at_port.add_point((x, 0), dxfattribs={"layer": "PORT"})
    doc.blocks.new("TANK").add_lwpolyline([(-20, -15), (20, -15), (20, 15), (-20, 15)], close=True)
    doc.blocks.new("DOT").add_line((0, 0), (0.5, 0))
    doc.saveas(tmp_path / "notes.dxf")

    assert convert.main(["library", "check", str(tmp_path / "notes.dxf"), "--no-self-test"]) == 0
    out = capsys.readouterr().out
    assert "VALVE_AT_PORT: the insertion point lies on a port" in out
    assert "TANK: no PORT points and no circle, and larger than 25 mm: it cannot be found" in out
    assert "DOT: 0.5 x 0.0 mm; blocks are drawn at their printed size" in out


def test_check_of_a_broken_library_fails(tmp_path, capsys):
    (tmp_path / "broken.dxf").write_text("not a drawing")
    assert convert.main(["library", "check", str(tmp_path / "broken.dxf")]) == 1
    assert capsys.readouterr().err.startswith("Error: broken.dxf: not a readable DXF file")


def test_build_makes_a_library_from_one_drawing_per_symbol(tmp_path, capsys, files):
    sources = tmp_path / "symbols"
    sources.mkdir()
    doc = ezdxf.new("R2018", units=ezdxf.units.IN)  # a gate valve in inches around the base point (2, 1)
    doc.header["$INSBASE"] = (2, 1, 0)
    doc.modelspace().add_lwpolyline([(1.9, 0.937), (1.9, 1.063), (2.1, 0.937), (2.1, 1.063)], close=True)
    doc.saveas(sources / "MY_GATE.dxf")
    ezdxf.new().saveas(sources / "EMPTY.dxf")
    if "dwg" in files:  # DWG sources are converted with LibreDWG (in the image)
        (sources / "BRACKET.dwg").write_bytes(files["dwg"].read_bytes())

    out = tmp_path / "acme"
    assert convert.main(["library", "build", str(sources), "--out", str(out), "--file-per-symbol",
                         "--licence", "CC0-1.0"]) == 0
    lib = library.read(out)
    assert (lib.id, lib.version, lib.licence) == ("acme", "0.1.0", "CC0-1.0")
    assert sorted(lib.symbols) == (["BRACKET"] if "dwg" in files else []) + ["MY_GATE"]
    gate = lib.symbols["MY_GATE"].parts[0]
    assert (gate.x1, gate.y1) == pytest.approx((-2.54, 1.6), abs=0.01)  # millimetres around the base point
    assert "EMPTY.dxf: skipped, the drawing is empty" in capsys.readouterr().out
    assert convert.main(["library", "build", str(sources), "--out", str(out)]) == 1  # the folder is not empty


def test_build_collects_the_blocks_of_a_drawing(tmp_path, capsys):
    out = tmp_path / "copy"
    assert convert.main(["library", "build", str(library.STARTER / "symbols.dxf"), "--out", str(out),
                         "--id", "copy", "--name", "Starter copy"]) == 0
    lib = library.read(out)
    assert (lib.id, lib.name, len(lib.symbols)) == ("copy", "Starter copy", 18)
    assert lib.symbols["PUMP"].ports == library.load()["PUMP"].ports
    assert "Next: open" in capsys.readouterr().out
