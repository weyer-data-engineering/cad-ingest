"""The command line converts one local file without Redis or a worker."""
import pytest

from app import convert


def test_drawing_gives_markdown_and_pdf(files, tmp_path, capsys):
    assert convert.main([str(files["dxf"]), str(tmp_path)]) == 0

    assert "Bracket" in (tmp_path / "drawing.md").read_text()
    assert (tmp_path / "drawing.pdf").read_bytes().startswith(b"%PDF")
    assert capsys.readouterr().out.split() == [str(tmp_path / "drawing.md"), str(tmp_path / "drawing.pdf")]


def test_model_without_preview_gives_only_markdown(files, tmp_path):
    assert convert.main([str(files["stl"]), str(tmp_path / "out"), "--no-preview"]) == 0

    assert [p.name for p in (tmp_path / "out").iterdir()] == ["box.md"]


def test_broken_file_fails_with_the_error_code(tmp_path, capsys):
    broken = tmp_path / "part.step"
    broken.write_text("not a STEP file")

    assert convert.main([str(broken), str(tmp_path / "out")]) == 1
    assert capsys.readouterr().err.startswith("part.step: unreadable:")


def test_unsupported_extension_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        convert.main([str(tmp_path / "notes.txt"), str(tmp_path)])


def test_pages_are_given_as_a_list_with_ranges(tmp_path, capsys):
    assert convert._pages("1-3,5") == [1, 2, 3, 5]
    with pytest.raises(SystemExit):
        convert.main(["plan.pdf", str(tmp_path), "--pages", "0"])
    assert "--pages: invalid _pages value: '0'" in capsys.readouterr().err
