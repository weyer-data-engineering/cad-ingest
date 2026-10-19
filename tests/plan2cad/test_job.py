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

from ..conftest import submit, wait_for_status


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
