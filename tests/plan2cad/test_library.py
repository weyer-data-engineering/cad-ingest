"""Symbol libraries: the format, reading folders, ZIPs and bare DXFs, units, choosing and combining libraries."""
import io
import shutil
import zipfile

import ezdxf
import pytest

from app.plan2cad import library
from app.plan2cad.export import to_dxf
from app.plan2cad.library import LibraryError
from app.plan2cad.model import Drawing

from .conftest import write_library, zipped


def test_starter_library_is_built_in_and_the_default():
    starter = library.installed()["din-10628"]
    assert starter.version == "1.0.0" and starter.licence == "Apache-2.0" and starter.drawing == ["pid"]
    assert library.select() == [starter] and len(starter.symbols) == 18 and not starter.warnings
    assert {"name": "PUMP", "description": "Centrifugal pump"} in starter.summary()["symbols"]


def test_a_folder_its_zip_and_its_bare_dxf_hold_the_same_symbols(tmp_path):
    folder = write_library(tmp_path / "acme")
    (tmp_path / "acme.zip").write_bytes(zipped(folder, "acme/"))
    shutil.copy(folder / "symbols.dxf", tmp_path / "ACME Symbols.dxf")

    for source in ("acme", "acme.zip"):
        lib = library.read(tmp_path / source)
        assert (lib.id, lib.name, lib.version, list(lib.symbols)) == ("acme-pid", "ACME symbols", "2.1.0", ["MY_VALVE"])
    bare = library.read(tmp_path / "ACME Symbols.dxf")  # no library.toml: named after the file, P&IDs, no version
    assert (bare.id, bare.version, bare.drawing, list(bare.symbols)) == ("acme-symbols", None, ["pid"], ["MY_VALVE"])
    valve = bare.symbols["MY_VALVE"]
    assert valve.description == "Box valve" and valve.ports == [(-3, 0), (3, 0)] and valve.tags == {"TAG": (0, 6)}


def test_inch_blocks_around_a_base_point_become_millimetres_and_go_into_the_dxf_unchanged(tmp_path):
    lib = library.read(write_library(tmp_path / "inch", units=ezdxf.units.IN, scale=1 / 25.4, base=(10, 20)))
    valve = lib.symbols["MY_VALVE"]
    assert lib.unit == pytest.approx(25.4) and [v for p in valve.ports for v in p] == pytest.approx([-3, 0, 3, 0])

    doc = to_dxf(Drawing(200, 100, symbols=[library.place(valve, 100, 50)], scale=10))
    ref = doc.modelspace().query("INSERT")[0]
    assert ref.dxf.name == "MY_VALVE" and ref.dxf.xscale == pytest.approx(254)  # inch block at 1:10 in millimetres
    assert doc.blocks.get("MY_VALVE").block.dxf.base_point.isclose((10, 20, 0))


def test_the_earlier_library_wins_on_equal_block_names(tmp_path):
    acme = library.read(write_library(tmp_path / "acme"))
    starter = library.installed()["din-10628"]
    other = library.read(write_library(tmp_path / "other", lib_id="other"))

    assert library.combine([other, acme, starter])["MY_VALVE"].library is other
    assert library.combine([acme, starter])["PUMP"].library is starter
    assert [lib.id for lib in library.select(["din-10628"], [acme])] == ["acme-pid", "din-10628"]
    assert [lib.id for lib in library.select(own=[acme])] == ["acme-pid"]  # no defaults when a job brings its own


def test_installed_libraries_come_from_folders_and_must_have_distinct_ids(tmp_path):
    write_library(tmp_path / "libs" / "acme")
    (tmp_path / "libs" / "lost+found").mkdir()  # no library.toml: not a library
    folders = str(tmp_path / "libs")
    assert list(library.installed(folders)) == ["din-10628", "acme-pid"]
    assert library.select(defaults="acme-pid, din-10628", folders=folders)[0].id == "acme-pid"
    with pytest.raises(LibraryError, match="'nope' is not installed"):
        library.select(["nope"], folders=folders)

    write_library(tmp_path / "libs" / "copy")
    with pytest.raises(LibraryError, match="Two symbol libraries have the id 'acme-pid'"):
        library.installed(folders + ":")  # another cache key: the folder is read again


@pytest.mark.parametrize("manifest, reason", [
    ('id = "ACME"', "'id' must be lower-case"),
    ("format = 2", "library format 2 needs a newer cad-ingest"),
    ('drawing = ["building"]', "symbols are recognised on P&IDs so far"),
    ('file = "../symbols.dxf"', "'file' must be a DXF file inside the library"),
    ("version = 2.1", "'version' must be a string"),
    ("id = ", "library.toml: "),  # not TOML
])
def test_a_wrong_manifest_is_rejected_with_the_reason(tmp_path, manifest, reason):
    folder = write_library(tmp_path / "acme")
    (folder / "library.toml").write_text(manifest)
    with pytest.raises(LibraryError, match=reason) as e:
        library.read(folder)
    assert str(e.value).startswith("acme: ")


def test_unusable_files_are_rejected_with_the_reason(tmp_path, monkeypatch):
    folder = write_library(tmp_path / "acme")
    for name, data, reason in [("broken.zip", b"PK\x03\x04 broken", "broken.zip: not a readable ZIP file"),
                               ("notes.dxf", b"not a drawing", "notes.dxf: not a readable DXF file"),
                               ("symbols.dwg", b"AC1032", "convert others with `cad-ingest library build`")]:
        with pytest.raises(LibraryError, match=reason):
            library.from_bytes(data, name)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("acme/library.toml", 'id = "acme"')
    with pytest.raises(LibraryError, match="symbols.dxf is missing"):
        library.from_bytes(buf.getvalue(), "acme.zip")

    monkeypatch.setattr(library, "MAX_BYTES", 1000)  # a ZIP bomb is not unpacked beyond the limit
    with pytest.raises(LibraryError, match="larger than"):
        library.from_bytes(zipped(folder), "acme.zip")

    doc = ezdxf.new()
    doc.blocks.new("TEXT_ONLY").add_text("no outline")
    buf = io.StringIO()
    doc.write(buf)
    with pytest.raises(LibraryError, match="no block with lines, arcs, circles or polylines"):
        library.from_bytes(buf.getvalue().encode(), "texts.dxf")


def test_splines_and_nested_blocks_count_and_hatches_are_noted(tmp_path):
    doc = ezdxf.new("R2018", units=ezdxf.units.MM)
    part = doc.blocks.new("BODY")
    part.add_circle((0, 0), 4)
    tank = doc.blocks.new("TANK")
    tank.add_blockref("BODY", (0, 0))
    tank.add_spline([(-4, 4), (0, 6), (4, 4)])
    tank.add_hatch().paths.add_polyline_path([(-1, -1), (1, -1), (1, 1)])
    doc.blocks.new("_ArchTick").add_line((0, 0), (1, 1))  # an arrowhead of a dimension style, not a symbol
    doc.saveas(tmp_path / "tank.dxf")

    lib = library.read(tmp_path / "tank.dxf")
    assert list(lib.symbols) == ["BODY", "TANK"]
    kinds = {type(q).__name__ for q in lib.symbols["TANK"].parts}
    assert kinds == {"Circle", "Line"} and len(set(lib.symbols["TANK"].groups)) == 2  # the circle and the spline
    assert lib.warnings == ["TANK: not used for matching: 1 HATCH"]
