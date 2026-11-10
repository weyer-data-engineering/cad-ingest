"""Photos and scans through the job API: the operation "vectorize" with a real Redis and worker (see tests/conftest.py),
and the symbol libraries a job brings along or has checked ("check_library")."""
import io
import json
import re
import time
import zipfile
from types import SimpleNamespace

import cv2
import ezdxf
import numpy as np
import pypdfium2
import pytest
from PIL import Image
from rq.timeouts import JobTimeoutException, UnixSignalDeathPenalty

from app import jobs, library_tool, scan
from app.config import Settings
from app.errors import ConversionError, JobError
from app.plan2cad import library

from ..conftest import submit, until, wait_for_status
from .conftest import valve_scan, write_library, zipped


def test_scan_becomes_dxf_with_markdown_and_preview(api, origin, worker, plan_scan):
    worker()
    origin.files["plan.png"] = cv2.imencode(".png", plan_scan)[1].tobytes()
    response = submit(api, origin, "plan.png", operations=["vectorize", "extract", "preview"],
                      result_url=origin.url("/preview/plan.zip"), options={"paper": "A3"})
    assert response.status_code == 202
    job = wait_for_status(api, response.json()["id"], "succeeded", "failed", seconds=120)
    assert job["status"] == "succeeded" and job["kind"] == "SCAN_2D", job
    assert job["result_content_type"] == "application/zip" and job["preview_content_type"] == "application/pdf"
    assert "- Scale: 1:50" in job["markdown"] and "- Maßstab 1:50" in job["markdown"]
    data, content_type = origin.previews["plan.zip"]
    files = zipfile.ZipFile(io.BytesIO(data))
    assert content_type == "application/zip"
    assert set(files.namelist()) == {"plan.dxf", "plan.pdf", "overlay.png", "underlay.png", "report.json"}
    report = json.loads(files.read("report.json"))
    assert report["scale"] == 50 and isinstance(report["review"], list)
    dxf = ezdxf.read(io.StringIO(files.read("plan.dxf").decode()))
    assert not dxf.audit().has_errors
    assert report["dimensions"] and len(dxf.modelspace().query("DIMENSION")) == len(report["dimensions"])
    assert "## Dimensions" in job["markdown"]
    assert origin.previews["plan.png"][0].startswith(b"%PDF")


def test_vectorize_is_only_for_photos_and_scans(api, origin):
    result = {"result_url": origin.url("/preview/result")}
    assert submit(api, origin, "plan.jpg").status_code == 415  # a photo is only read when asked for
    assert submit(api, origin, "drawing.dxf", operations=["vectorize"], **result).status_code == 415
    assert submit(api, origin, "plan.jpg", operations=["vectorize"]).status_code == 400  # no result_url
    assert submit(api, origin, "plan.jpg", operations=["vectorize"], options={"paper": "B5"}, **result).status_code == 400
    assert submit(api, origin, "plan.jpg", operations=["vectorize"], options={"scale": 50}, **result).status_code == 202
    assert submit(api, origin, "plans.pdf", operations=["vectorize"], options={"pages": [0]}, **result).status_code == 400
    assert submit(api, origin, "plans.pdf", operations=["vectorize"], options={"pages": [1, 3]}, **result).status_code == 202


def test_the_reviewer_is_only_for_vectorize_and_its_key_is_never_shown(api, origin):
    reviewer = {"base_url": "https://llm.example.com/v1", "api_key": "sk-secret", "model": "vision"}
    assert submit(api, origin, "drawing.dxf", reviewer=reviewer).status_code == 400
    response = submit(api, origin, "plan.jpg", operations=["vectorize"], result_url=origin.url("/preview/r"),
                      reviewer=reviewer)
    assert response.status_code == 202
    assert "sk-secret" not in api.get(f"/jobs/{response.json()['id']}").text


@pytest.mark.parametrize("name, content", [("blank.png", cv2.imencode(".png", np.full((900, 1200), 255, np.uint8))[1]),
                                           ("broken.jpg", b"\xff\xd8 not a photo")])
def test_photo_without_a_drawing_is_unreadable(name, content, tmp_path):
    path = tmp_path / name
    path.write_bytes(bytes(content))
    with pytest.raises(JobError) as e:
        jobs._convert(["vectorize"], path.suffix[1:], str(path), str(tmp_path), name, str(tmp_path / "preview.pdf"),
                      Settings(), {}, str(tmp_path / "result.zip"))
    assert e.value.code == "unreadable"


def pages(name: str, *images: np.ndarray) -> bytes:
    """The images as the grayscale pages of a PDF or TIFF (by the extension of ``name``) at 200 dpi."""
    first, *rest = [Image.fromarray(img).convert("L") for img in images]
    buf = io.BytesIO()
    if name.endswith(".pdf"):
        first.save(buf, "PDF", save_all=True, append_images=rest, resolution=200)
    else:
        first.save(buf, "TIFF", save_all=True, append_images=rest, dpi=(200, 200), compression="tiff_lzw")
    return buf.getvalue()


def test_every_page_of_a_pdf_scan_becomes_a_drawing(plan_scan, tmp_path):
    cover = np.full(plan_scan.shape[:2], 255, np.uint8)  # a page without a drawing, such as a cover letter
    path = tmp_path / "plans.pdf"
    path.write_bytes(pages(path.name, plan_scan, cover, plan_scan))
    result = jobs._convert(["vectorize", "extract", "preview"], "pdf", str(path), str(tmp_path), path.name,
                           str(tmp_path / "preview.pdf"), Settings(), {"paper": "A3"}, str(tmp_path / "result.zip"))
    files = zipfile.ZipFile(tmp_path / "result.zip")
    assert {"plan.dxf", "report.json", "page-3/plan.dxf", "page-3/underlay.png"} <= set(files.namelist())
    reports = [json.loads(files.read(name)) for name in ("report.json", "page-3/report.json")]
    assert [(r["page"], r["pages"], r["scale"]) for r in reports] == [(1, [1, 3], 50), (3, [1, 3], 50)]
    assert len(pypdfium2.PdfDocument(tmp_path / "preview.pdf")) == 2  # the preview has every drawing
    markdown = result["markdown"]
    assert "## Page 1 of 3" in markdown and "## Page 3 of 3" in markdown and "### Texts" in markdown
    check = "1 dimensions are to check; they are on the layer P2C-DIMENSION-CHECK"  # an interval without a value
    assert result["warnings"] == [f"Page 1: {check}", "Page 2: no drawing was recognised", f"Page 3: {check}"]


def test_pages_of_a_tiff_scan_stop_before_the_time_limit(plan_scan, tmp_path):
    path = tmp_path / "plans.tif"
    path.write_bytes(pages(path.name, plan_scan, plan_scan))
    result = jobs._convert(["vectorize"], "tif", str(path), str(tmp_path), path.name, str(tmp_path / "preview.pdf"),
                           Settings(), {"paper": "A3"}, str(tmp_path / "result.zip"), deadline=1.0)  # long past
    assert "page-2/plan.dxf" not in zipfile.ZipFile(tmp_path / "result.zip").namelist()
    assert result["warnings"] == ["Page 1: 1 dimensions are to check; they are on the layer P2C-DIMENSION-CHECK",
                                  "Pages left out to stay within the time limit: 2"]


def test_pages_asked_for_beyond_the_file_or_the_limit_are_left_out():
    assert scan._pages(None, 3, 20) == ([1, 2, 3], [])
    assert scan._pages([3, 1, 3, 7], 3, 20) == ([1, 3], ["Pages left out that the file does not have (it has 3): 7"])
    numbers, warnings = scan._pages(None, 30, 20)
    assert numbers == list(range(1, 21))
    assert warnings == ["Pages left out over the limit of 20 per job (MAX_SCAN_PAGES): 21, 22, 23, 24, 25, 26, 27, 28, "
                        "29, 30"]
    with pytest.raises(ConversionError, match="it has 3"):
        scan._pages([5], 3, 20)


def test_job_brings_its_own_symbol_library(api, origin, worker, tmp_path):
    worker()
    folder = write_library(tmp_path / "acme")
    origin.files["acme.zip"] = zipped(folder, "acme/")
    origin.files["valve.png"] = valve_scan(library.read(folder))
    response = submit(api, origin, "valve.png", operations=["vectorize", "extract"],
                      result_url=origin.url("/preview/valve.zip"), library_urls=[origin.url("/src/acme.zip")],
                      options={"paper": "A4", "dpi": 300, "drawing": "pid"})
    job = wait_for_status(api, response.json()["id"], "succeeded", "failed", seconds=120)

    assert job["status"] == "succeeded", job
    assert "- Box valve V-101" in job["markdown"]
    files = zipfile.ZipFile(io.BytesIO(origin.previews["valve.zip"][0]))
    report = json.loads(files.read("report.json"))
    assert report["libraries"] == [{"id": "acme-pid", "version": "2.1.0"}]  # only its own: no defaults
    assert [(s["name"], s["library"], s["attributes"]) for s in report["symbols"]] == [
        ("MY_VALVE", "acme-pid", {"TAG": "V-101"})]
    doc = ezdxf.read(io.StringIO(files.read("plan.dxf").decode()))
    assert "MY_VALVE" in doc.blocks and "GATE_VALVE" not in doc.blocks


def test_installed_libraries_are_listed_and_checked(api, origin, worker, tmp_path):
    assert api.get("/libraries").status_code == 503  # no worker has started yet
    write_library(tmp_path / "libs" / "acme")
    worker(SYMBOL_LIBRARIES=tmp_path / "libs")

    catalogue = api.get("/libraries").json()
    assert [lib["id"] for lib in catalogue["libraries"]] == ["building-symbols", "din-10628", "floor-plan",
                                                             "iec-60617", "acme-pid"]
    assert catalogue["default"] == ["din-10628", "floor-plan", "iec-60617", "building-symbols"]
    assert catalogue["libraries"][4]["symbols"] == [{"name": "MY_VALVE", "description": "Box valve"}]
    vectorize = {"operations": ["vectorize"], "result_url": origin.url("/preview/result")}
    origin.source_status["plan.jpg"].append(404)  # the accepted job ends at once
    assert submit(api, origin, "plan.jpg", options={"libraries": ["acme-pid", "din-10628"]}, **vectorize).status_code == 202
    unknown = submit(api, origin, "plan.jpg", "unknown", options={"libraries": ["nope"]}, **vectorize)
    assert unknown.status_code == 400 and unknown.json()["detail"].startswith("Unknown symbol library: nope")
    assert submit(api, origin, "plan.jpg", "path", options={"libraries": ["../x"]}, **vectorize).status_code == 400
    assert submit(api, origin, "plan.jpg", "step", options={"off": ["nonsense"]}, **vectorize).status_code == 400
    assert submit(api, origin, "drawing.dxf", library_urls=[origin.url("/src/acme.zip")]).status_code == 400


def test_an_unusable_library_fails_the_job(api, origin, worker, plan_scan):
    worker()
    origin.files["plan.png"] = cv2.imencode(".png", plan_scan)[1].tobytes()
    origin.files["notes.dxf"] = b"not a drawing"
    origin.source_status["gone.zip"].append(404)
    for name, code, retryable in [("notes.dxf", "invalid_library", False), ("gone.zip", "library_unavailable", True)]:
        response = submit(api, origin, "plan.png", code, operations=["vectorize"], result_url=origin.url("/preview/r"),
                          library_urls=[origin.url(f"/src/{name}")])
        job = wait_for_status(api, response.json()["id"], "succeeded", "failed")
        assert (job["status"], job["error"]["code"], job["error"]["retryable"], job["attempts"]) == \
            ("failed", code, retryable, 1), job


def test_a_library_the_worker_does_not_have_fails_the_job(tmp_path):
    path = tmp_path / "plan.png"
    path.write_bytes(b"")  # the libraries are chosen before the image is read
    with pytest.raises(JobError) as e:
        jobs._convert(["vectorize"], "png", str(path), str(tmp_path), "plan.png", str(tmp_path / "preview.pdf"),
                      Settings(), {"libraries": ["nope"]}, str(tmp_path / "result.zip"))
    assert e.value.code == "invalid_library" and "'nope' is not installed" in e.value.message


def _bare_library() -> bytes:
    """A library as a bare DXF: a valve inserted at one of its ports, which `cad-ingest library check` notes."""
    doc = ezdxf.new("R2018", units=ezdxf.units.MM)
    doc.layers.add("PORT")
    valve = doc.blocks.new("VALVE_AT_PORT", dxfattribs={"description": "Gate valve, inserted at a port"})
    valve.add_lwpolyline([(0, -1.6), (0, 1.6), (5, -1.6), (5, 1.6)], close=True)
    for x in (0, 5):
        valve.add_point((x, 0), dxfattribs={"layer": "PORT"})
    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode()


def _deflated(files: dict[str, str]) -> bytes:
    """A ZIP of the texts ``files`` (name -> text), compressed as a hostile library would be."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files.items():
            z.writestr(name, text)
    return buf.getvalue()


def test_a_library_is_checked_before_it_is_used(api, origin, worker, tmp_path):
    worker()
    origin.files["acme.zip"] = zipped(write_library(tmp_path / "acme"), "acme/")
    origin.files["valves.dxf"] = _bare_library()
    checked = {}
    for name in ("acme.zip", "valves.dxf"):
        response = submit(api, origin, name, operations=["check_library"], preview_url=None)
        assert response.status_code == 202
        checked[name] = job = wait_for_status(api, response.json()["id"], "succeeded", "failed")
        assert (job["status"], job["operations"], job["format"], job["warnings"]) == \
            ("succeeded", ["check_library"], name[-3:], []), job
        assert "kind" not in job and "markdown" not in job  # a library is no drawing
        until(lambda: origin.callbacks_of(name), 10)
        assert origin.callbacks_of(name) == [(job, 200)]  # the callback carries what GET returns

    assert checked["acme.zip"]["library"] == {
        "id": "acme-pid", "name": "ACME symbols", "version": "2.1.0", "licence": "LicenseRef-ACME-internal",
        "drawing": ["pid"], "size": "printed", "symbols": [{"name": "MY_VALVE", "description": "Box valve"}],
        "warnings": [], "self_test": None}
    bare = checked["valves.dxf"]["library"]  # no library.toml: named after the file
    assert (bare["id"], bare["version"], bare["symbols"]) == \
        ("valves", None, [{"name": "VALVE_AT_PORT", "description": "Gate valve, inserted at a port"}])
    assert bare["warnings"] == ["VALVE_AT_PORT: the insertion point lies on a port; put it in the middle of the "
                                "symbol, because a port is recognised by its direction from there"]
    status = api.post("/jobs/status", json={"ids": [checked["acme.zip"]["id"]]}).json()["jobs"]
    assert status[0]["status"] == "succeeded" and "library" not in status[0]  # like the markdown: GET has it


def test_an_unusable_library_fails_its_check(api, origin, worker, tmp_path):
    worker(MAX_LIBRARY_SIZE_BYTES=2000)
    origin.files.update({"broken.dxf": b"not a drawing", "broken.zip": b"not a zip either",
                         "large.zip": zipped(write_library(tmp_path / "acme"), "acme/"),
                         "inflated.zip": _deflated({"library.toml": f'name = "{"N" * 70_000}"'})})  # 220 bytes
    origin.source_status["gone.zip"].append(404)
    for name, code, message in [
            ("broken.dxf", "invalid_library", "broken.dxf: not a readable DXF file (DXFStructureError)"),
            ("broken.zip", "invalid_library", "broken.zip: not a readable ZIP file"),
            ("large.zip", "invalid_library", "A symbol library is larger than 2000 bytes"),
            ("inflated.zip", "invalid_library", "inflated.zip: library.toml is larger than 64 KB"),
            ("gone.zip", "library_unavailable", "The symbol library download answered HTTP 404 (127.0.0.1)")]:
        response = submit(api, origin, name, operations=["check_library"], preview_url=None)
        job = wait_for_status(api, response.json()["id"], "succeeded", "failed")
        assert (job["status"], job["error"]["code"], job["error"]["message"], job["attempts"]) == \
            ("failed", code, message, 1), job
        assert job["error"]["retryable"] == (code == "library_unavailable") and "library" not in job


def test_the_check_of_a_large_library_is_cut_to_size(api, origin, worker):
    """The texts of a library go into the job's document, so they are cut, and so are the lists of symbols and notes."""
    worker()
    doc = ezdxf.new("R2018", units=ezdxf.units.MM)
    for i in range(1005):  # neither ports, circle nor rectangle: a note on each
        doc.blocks.new(f"B{i}", dxfattribs={"description": "D" * 1000}).add_line((0, 0), (5, 0))
    dxf = io.StringIO()
    doc.write(dxf)
    origin.files["many.zip"] = _deflated({"library.toml": f'name = "{"N" * 1000}"\nnote = 1\n',
                                          "symbols.dxf": dxf.getvalue()})
    response = submit(api, origin, "many.zip", operations=["check_library"], preview_url=None)
    job = wait_for_status(api, response.json()["id"], "succeeded", "failed")

    assert job["warnings"] == ["Only the first 1000 of 1005 symbols are listed",
                               "Only the first 100 of 1006 notes are listed"], job
    checked = job["library"]
    assert (checked["id"], checked["name"], len(checked["warnings"])) == ("many", "N" * 300, 100)
    assert checked["symbols"] == [{"name": f"B{i}", "description": "D" * 300} for i in range(1000)]
    assert checked["warnings"][:2] == ["library.toml: unknown key 'note' is ignored",
                                       "B0: no PORT points, circle or rectangle, so it is only found where its strokes "
                                       "stand apart from other linework; add PORT points where pipes connect"]
    assert len(json.dumps(job)) < library_tool.CHECK_BYTES


def test_a_dxf_without_symbols_is_no_library(tmp_path):
    doc = ezdxf.new("R2018")
    doc.modelspace().add_line((0, 0), (10, 0))  # a drawing, but no block
    doc.saveas(tmp_path / "drawing.dxf")
    with pytest.raises(JobError) as e:
        jobs._check_library(str(tmp_path / "drawing.dxf"), "drawing.dxf", {}, time.monotonic() + 60)
    assert (e.value.code, e.value.message) == \
        ("invalid_library", "drawing.dxf: no block with lines, arcs, circles or polylines")


def test_a_check_still_too_large_after_cutting_fails(tmp_path, monkeypatch):
    """Cutting bounds the characters; characters that JSON escapes take up to twelve bytes each, so the bytes count."""
    monkeypatch.setattr(library_tool, "CHECK_BYTES", 100)
    (tmp_path / "acme.zip").write_bytes(zipped(write_library(tmp_path / "acme"), "acme/"))
    with pytest.raises(JobError) as e:
        jobs._check_library(str(tmp_path / "acme.zip"), "acme.zip", {}, time.monotonic() + 60)
    assert e.value.code == "invalid_library" and re.fullmatch(
        r"acme-pid: the names and descriptions of its symbols are too long \(\d+ bytes as JSON, at most 100\)",
        e.value.message)


def test_check_library_stands_alone(api, origin):
    check = {"operations": ["check_library"], "preview_url": None}
    assert submit(api, origin, "acme.zip", **check).status_code == 202
    assert submit(api, origin, "acme.dxf", **check, options={"self_test": True}).status_code == 202
    combined = submit(api, origin, "acme.zip", operations=["check_library", "extract"], preview_url=None)
    assert combined.status_code == 400 and "'check_library' cannot be combined" in combined.json()["detail"]
    assert submit(api, origin, "acme.zip", operations=["check_library", "vectorize"],
                  result_url=origin.url("/preview/r")).status_code == 400
    assert submit(api, origin, "acme.zip", **check, library_urls=[origin.url("/src/other.zip")]).status_code == 400
    assert submit(api, origin, "acme.toml", **check).status_code == 415  # a ZIP or a DXF
    assert submit(api, origin, "plan.png", **check).status_code == 415
    assert submit(api, origin, "acme.zip", operations=["extract"], preview_url=None).status_code == 415


def _two_sheets(tmp_path) -> str:
    """A library ZIP whose self-test takes two sheets: MY_VALVE twice on the first, a large TANK twice on the second."""
    folder = write_library(tmp_path / "acme")
    doc = ezdxf.readfile(folder / "symbols.dxf")  # a tank too large for more than two cells on a sheet
    doc.blocks.new("TANK").add_lwpolyline([(-80, -50), (80, -50), (80, 50), (-80, 50)], close=True)
    doc.saveas(folder / "symbols.dxf")
    (tmp_path / "acme.zip").write_bytes(zipped(folder, "acme/"))
    return str(tmp_path / "acme.zip")


def test_the_check_tests_the_library_if_asked_and_stops_in_time(tmp_path, monkeypatch):
    """The self-test as data; sheets it would not finish before the deadline are left out."""
    monkeypatch.setattr(library_tool, "RUNS", library_tool.RUNS[:1])  # the scan only, to keep the test short
    result = jobs._check_library(_two_sheets(tmp_path), "acme.zip", {"self_test": True}, 1.0)  # long past
    tested = result["library"]["self_test"]
    assert result["warnings"] == ["Self-test: 1 of 2 sheets left out to stay within the time limit"]
    assert (tested["sheets"], tested["scale"], tested["drawn"], tested["found"], tested["extra"]) == \
        (1, None, 2, {"scan": 2}, {"scan": 0})
    assert tested["seconds_per_sheet"]["scan"] > 0
    assert tested["symbols"] == [{"name": "MY_VALVE", "drawn": 2, "found": {"scan": 2}, "mistaken_for": {}},
                                 {"name": "TANK", "drawn": 0, "found": {"scan": 0}, "mistaken_for": {}}]


def test_a_sheet_that_outlasts_the_job_ends_the_self_test(tmp_path, monkeypatch):
    """Each sheet is read with the whole library, so a sheet of a large library may outlast the job: when the job's
    time limit comes, the sheets read before count and the format check is kept."""
    monkeypatch.setattr(library_tool, "RUNS", (("scan", "scan", lambda d, seed: d),))  # the sheet as it is drawn
    path = _two_sheets(tmp_path)

    def check(slow: int) -> dict:  # the ``slow``-th sheet takes longer than the job may
        sheets = []

        def convert(sheet, **_):  # finds every symbol where it was drawn
            sheets.append(sheet)
            if len(sheets) == slow:
                time.sleep(5)
            return SimpleNamespace(drawing=SimpleNamespace(symbols=sheet.symbols), info={"timings_s": {}})

        monkeypatch.setattr(library_tool, "convert", convert)
        with UnixSignalDeathPenalty(1, JobTimeoutException):  # as RQ runs a job, here with a time limit of 1 s
            return jobs._check_library(path, "acme.zip", {"self_test": True}, time.monotonic() + 600)

    result = check(slow=2)
    assert result["warnings"] == ["Self-test: 1 of 2 sheets left out to stay within the time limit"]
    assert [(s["name"], s["drawn"], s["found"]) for s in result["library"]["self_test"]["symbols"]] == \
        [("MY_VALVE", 2, {"scan": 2}), ("TANK", 0, {"scan": 0})]
    result = check(slow=1)
    assert result["warnings"] == ["Self-test: 2 of 2 sheets left out to stay within the time limit"]
    assert result["library"]["self_test"] is None and [s["name"] for s in result["library"]["symbols"]] == \
        ["MY_VALVE", "TANK"]
