"""The conversions themselves, called directly: content of the markdown and the previews for every format."""
import io
import re

import pytest
import trimesh

from app import jobs
from app.config import Settings
from app.errors import JobError
from app.formats import FORMATS, PREVIEW

# extension -> (kind, text that must appear in the markdown)
EXPECTED = {
    "dxf": ("DRAWING_2D", "DWG-4711"),
    "dwg": ("DRAWING_2D", "DWG-4711 Bracket"),
    "plt": ("DRAWING_2D", "Hello plotter"),
    "step": ("MODEL_3D", "2 x Box part: size 10 x 20 x 30 mm, volume 6000"),
    "iges": ("MODEL_3D", "volume 6000"),
    "brep": ("MODEL_3D", "volume 6000"),
    "stl": ("MODEL_3D", "watertight, volume 6000"),
    "obj": ("MODEL_3D", "watertight, volume 6000"),
    "ply": ("MODEL_3D", "watertight, volume 6000"),
    "off": ("MODEL_3D", "watertight, volume 6000"),
    "3mf": ("MODEL_3D", "watertight, volume 6000"),
    "dae": ("MODEL_3D", "12 faces"),
    "gltf": ("MODEL_3D", "nodes: Chassis"),
    "glb": ("MODEL_3D", "12 faces"),
    "ifc": ("MODEL_3D", "Pset_WallCommon.FireRating = REI60"),
}


def pages(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page(?!s)", pdf))


def convert(path, tmp_path, operations=("extract", "preview"), settings=None):
    """Returns (result, preview bytes or None) of the conversion of the file `path`."""
    ext = path.suffix.lstrip(".").lower()
    preview = tmp_path / ("preview." + PREVIEW[FORMATS[ext][0]][0])
    result = jobs._convert(list(operations), ext, str(path), str(tmp_path), path.name, str(preview),
                           settings or Settings())
    return result, preview.read_bytes() if "preview" in operations else None


def pytest_generate_tests(metafunc):
    if "ext" in metafunc.fixturenames:
        metafunc.parametrize("ext", list(EXPECTED))


@pytest.fixture
def source(ext, files):
    if ext not in files:
        pytest.skip(f"no {ext} fixture (LibreDWG dwgadd missing)")
    return files[ext]


def test_extract_and_preview(ext, source, tmp_path):
    result, preview = convert(source, tmp_path)
    kind, text = EXPECTED[ext]
    assert FORMATS[ext][0] == kind
    assert text in result["markdown"]
    assert isinstance(result["warnings"], list)
    if kind == "DRAWING_2D":
        assert preview.startswith(b"%PDF") and pages(preview) >= 1
    else:
        assert preview.startswith(b"glTF")
        assert len(trimesh.load(io.BytesIO(preview), file_type="glb", force="scene").geometry) >= 1


def test_operations_are_independent(files, tmp_path):
    assert "markdown" not in convert(files["stl"], tmp_path, ["preview"])[0]
    assert convert(files["stl"], tmp_path, ["extract"])[1] is None


def test_preview_is_y_up(files, tmp_path):
    """The 30 high box (Z-up) must be 30 high along +Y; STEP is converted from mm to metres, STL passes through."""
    for ext, height in (("step", 0.03), ("stl", 30)):
        glb = convert(files[ext], tmp_path, ["preview"])[1]
        extents = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene").extents
        assert extents[1] == pytest.approx(height)


def test_pdf_has_one_page_per_non_empty_layout(files, tmp_path):
    """The fixture has model space and 'A3 Sheet' with content, plus an empty 'Layout1'."""
    assert pages(convert(files["dxf"], tmp_path, ["preview"])[1]) == 2
    assert pages(convert(files["plt"], tmp_path, ["preview"])[1]) == 1


def test_dxf_content(files, tmp_path):
    markdown = convert(files["dxf"], tmp_path, ["extract"])[0]["markdown"]
    for expected in ("Drawing units: mm", "Layout: A3 Sheet", "TITLE: Bracket; DRAWN_BY: J. Doe", "Bracket Ø12 plate",
                     "Material: S235JR Thickness 5 mm", "linear 100", "Sheet note"):
        assert expected in markdown


@pytest.mark.parametrize("name", ["broken.step", "broken.dxf", "broken.stl", "broken.ifc", "broken.plt"])
def test_garbage_is_unreadable(name, tmp_path):
    broken = tmp_path / name
    broken.write_bytes(b"this is not a CAD file\n" * 20)
    for operations in (["extract"], ["preview"]):
        with pytest.raises(JobError) as error:
            convert(broken, tmp_path, operations)
        assert (error.value.code, error.value.retryable) == ("unreadable", False) and error.value.message


def test_markdown_limit(files, tmp_path):
    result, _ = convert(files["dxf"], tmp_path, ["extract"], Settings(max_markdown_chars=200))
    assert "[Truncated" in result["markdown"] and result["warnings"]


def test_oversized_preview_is_simplified(tmp_path, monkeypatch):
    from app import mesh

    monkeypatch.setattr(mesh, "PREVIEW_SOFT_LIMIT_BYTES", 300_000)
    out = tmp_path / "sphere.glb"
    warnings = mesh.write_glb(trimesh.Scene(trimesh.creation.icosphere(subdivisions=5)), str(out), z_up=False)
    assert out.stat().st_size < 300_000 and warnings
