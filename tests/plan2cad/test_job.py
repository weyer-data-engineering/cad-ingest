"""Photos and scans through the job API: the operation "vectorize" with a real Redis and worker (see tests/conftest.py)."""
import io
import json
import zipfile

import cv2
import ezdxf
import numpy as np
import pytest

from app import jobs
from app.config import Settings
from app.errors import JobError
from app.plan2cad import library

from ..conftest import submit, wait_for_status
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
    assert json.loads(files.read("report.json"))["scale"] == 50
    assert not ezdxf.read(io.StringIO(files.read("plan.dxf").decode())).audit().has_errors
    assert origin.previews["plan.png"][0].startswith(b"%PDF")


def test_vectorize_is_only_for_photos_and_scans(api, origin):
    result = {"result_url": origin.url("/preview/result")}
    assert submit(api, origin, "plan.jpg").status_code == 415  # a photo is only read when asked for
    assert submit(api, origin, "drawing.dxf", operations=["vectorize"], **result).status_code == 415
    assert submit(api, origin, "plan.jpg", operations=["vectorize"]).status_code == 400  # no result_url
    assert submit(api, origin, "plan.jpg", operations=["vectorize"], options={"paper": "B5"}, **result).status_code == 400
    assert submit(api, origin, "plan.jpg", operations=["vectorize"], options={"scale": 50}, **result).status_code == 202


@pytest.mark.parametrize("name, content", [("blank.png", cv2.imencode(".png", np.full((900, 1200), 255, np.uint8))[1]),
                                           ("broken.jpg", b"\xff\xd8 not a photo")])
def test_photo_without_a_drawing_is_unreadable(name, content, tmp_path):
    path = tmp_path / name
    path.write_bytes(bytes(content))
    with pytest.raises(JobError) as e:
        jobs._convert(["vectorize"], path.suffix[1:], str(path), str(tmp_path), name, str(tmp_path / "preview.pdf"),
                      Settings(), {}, str(tmp_path / "result.zip"))
    assert e.value.code == "unreadable"


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
    assert [lib["id"] for lib in catalogue["libraries"]] == ["din-10628", "acme-pid"]
    assert catalogue["default"] == ["din-10628"]
    assert catalogue["libraries"][1]["symbols"] == [{"name": "MY_VALVE", "description": "Box valve"}]
    vectorize = {"operations": ["vectorize"], "result_url": origin.url("/preview/result")}
    origin.source_status["plan.jpg"].append(404)  # the accepted job ends at once
    assert submit(api, origin, "plan.jpg", options={"libraries": ["acme-pid", "din-10628"]}, **vectorize).status_code == 202
    unknown = submit(api, origin, "plan.jpg", "unknown", options={"libraries": ["nope"]}, **vectorize)
    assert unknown.status_code == 400 and unknown.json()["detail"].startswith("Unknown symbol library: nope")
    assert submit(api, origin, "plan.jpg", "path", options={"libraries": ["../x"]}, **vectorize).status_code == 400
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
