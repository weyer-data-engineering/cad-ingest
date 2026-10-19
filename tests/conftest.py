"""Tiny fixture files are generated here, so no binary files are committed."""
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import ezdxf
import ifcopenshell
import ifcopenshell.api
import pytest
import redis
import trimesh
from fastapi.testclient import TestClient
from trimesh.exchange import gltf

from app import store
from app.main import app


def _dxf(path):
    doc = ezdxf.new("R2018", setup=True, units=ezdxf.units.MM)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 50), (0, 50), (0, 0)])
    msp.add_text("Bracket %%c12 plate", height=5).set_placement((5, 60))
    msp.add_mtext("{\\fArial|b1;Material:} S235JR\\PThickness 5 mm", dxfattribs={"char_height": 4}).set_location((120, 40))
    msp.add_linear_dim(base=(0, -15), p1=(0, 0), p2=(100, 0)).render()
    block = doc.blocks.new("TITLEBLOCK")
    block.add_lwpolyline([(0, 0), (180, 0), (180, 40), (0, 40), (0, 0)])
    for tag in ("TITLE", "DRAWN_BY", "DRAWING_NO"):
        block.add_attdef(tag, (5, 10), dxfattribs={"height": 4})
    msp.add_blockref("TITLEBLOCK", (0, -100)).add_auto_attribs(
        {"TITLE": "Bracket", "DRAWN_BY": "J. Doe", "DRAWING_NO": "DWG-4711"})
    sheet = doc.layouts.new("A3 Sheet")
    sheet.page_setup(size=(420, 297), margins=(10, 10, 10, 10), units="mm")
    sheet.add_text("Sheet note: scale 1:2", dxfattribs={"height": 5}).set_placement((20, 20))
    doc.saveas(path)


def _step_assembly(path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_ColorSurf, XCAFDoc_DocumentTool
    from OCP.gp import gp_Trsf, gp_Vec

    doc = TDocStd_Document(TCollection_ExtendedString("BinXCAF"))
    XCAFApp_Application.GetApplication_s().InitDocument(doc)
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    part = shapes.AddShape(BRepPrimAPI_MakeBox(10, 20, 30).Shape(), False)
    TDataStd_Name.Set_s(part, TCollection_ExtendedString("Box part"))
    XCAFDoc_DocumentTool.ColorTool_s(doc.Main()).SetColor(part, Quantity_Color(1, 0, 0, Quantity_TOC_RGB), XCAFDoc_ColorSurf)
    assembly = shapes.NewShape()
    TDataStd_Name.Set_s(assembly, TCollection_ExtendedString("Test assembly"))
    for i in range(2):
        move = gp_Trsf()
        move.SetTranslation(gp_Vec(50 * i, 0, 0))
        shapes.AddComponent(assembly, part, TopLoc_Location(move))
    shapes.UpdateAssemblies()
    writer = STEPCAFControl_Writer()
    writer.Transfer(doc)
    writer.Write(str(path))


def _box_files(directory) -> dict:
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools
    from OCP.IGESControl import IGESControl_Writer

    box = BRepPrimAPI_MakeBox(10, 20, 30).Shape()
    iges = IGESControl_Writer()
    iges.AddShape(box)
    iges.ComputeModel()
    iges.Write(str(directory / "box.iges"))
    BRepTools.Write_s(box, str(directory / "box.brep"))
    return {"iges": directory / "box.iges", "brep": directory / "box.brep"}


def _ifc(path):
    model = ifcopenshell.api.run("project.create_file", version="IFC4")
    project = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcProject", name="Demo Project")
    ifcopenshell.api.run("unit.assign_unit", model)
    context = ifcopenshell.api.run("context.add_context", model, context_type="Model")
    body = ifcopenshell.api.run("context.add_context", model, context_type="Model", context_identifier="Body",
                                target_view="MODEL_VIEW", parent=context)
    chain = [("IfcSite", "Site A"), ("IfcBuilding", "Building 1"), ("IfcBuildingStorey", "Ground floor")]
    parent = project
    for ifc_class, name in chain:
        child = ifcopenshell.api.run("root.create_entity", model, ifc_class=ifc_class, name=name)
        ifcopenshell.api.run("aggregate.assign_object", model, products=[child], relating_object=parent)
        parent = child
    wall = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcWall", name="Wall North")
    ifcopenshell.api.run("spatial.assign_container", model, products=[wall], relating_structure=parent)
    ifcopenshell.api.run("geometry.assign_representation", model, product=wall,
                         representation=ifcopenshell.api.run("geometry.add_wall_representation", model, context=body,
                                                             length=5, height=3, thickness=0.2))
    pset = ifcopenshell.api.run("pset.add_pset", model, product=wall, name="Pset_WallCommon")
    ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties={"FireRating": "REI60"})
    ifcopenshell.api.run("material.assign_material", model, products=[wall],
                         material=ifcopenshell.api.run("material.add_material", model, name="Concrete"))
    model.write(str(path))


@pytest.fixture(scope="session")
def files(tmp_path_factory) -> dict:
    """extension -> path of a tiny valid file."""
    d = tmp_path_factory.mktemp("cad")
    _dxf(d / "drawing.dxf")
    (d / "plot.plt").write_bytes(b"IN;SP1;PU0,0;PD1000,0,1000,1000,0,1000,0,0;PU;PA200,200;LBHello plotter\x03;")
    _step_assembly(d / "assembly.step")
    result = {"dxf": d / "drawing.dxf", "plt": d / "plot.plt", "step": d / "assembly.step", **_box_files(d)}
    _ifc(d / "wall.ifc")
    result["ifc"] = d / "wall.ifc"

    box = trimesh.creation.box(extents=(10, 20, 30))
    for ext in ("stl", "obj", "ply", "off", "3mf", "glb", "dae"):
        box.export(d / f"box.{ext}")
        result[ext] = d / f"box.{ext}"
    scene = trimesh.Scene()
    scene.add_geometry(box, node_name="Chassis", geom_name="BoxGeometry")
    (d / "box.gltf").write_bytes(gltf.export_gltf(scene, embed_buffers=True)["model.gltf"])
    result["gltf"] = d / "box.gltf"

    if shutil.which("dwgadd"):  # LibreDWG can write DWG from a simple script; its DXF -> DWG converter is unreliable
        (d / "drawing.add").write_text('line (0.0 0.0 0.0) (100.0 50.0 0.0)\ncircle (50.0 25.0 0.0) 10.0\n'
                                       'text "DWG-4711 Bracket" (5.0 60.0 0.0) 5\n')
        subprocess.run(["dwgadd", "-o", str(d / "drawing.dwg"), str(d / "drawing.add")], capture_output=True)
        if (d / "drawing.dwg").exists():
            result["dwg"] = d / "drawing.dwg"
    return result




# ---- infrastructure: a real Redis, an HTTP server that plays the client and storage, and worker processes ----

REPO = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def until(condition, seconds=60.0, message="timed out"):
    deadline = time.time() + seconds
    while not (result := condition()):
        assert time.time() < deadline, message
        time.sleep(0.1)
    return result


@pytest.fixture(scope="session")
def redis_url(tmp_path_factory):
    """A real Redis server for the whole session (redis-server must be on the PATH)."""
    port = free_port()
    proc = subprocess.Popen(["redis-server", "--port", str(port), "--save", "", "--appendonly", "no",
                             "--dir", str(tmp_path_factory.mktemp("redis"))], stdout=subprocess.DEVNULL)
    url = f"redis://127.0.0.1:{port}/0"
    until(lambda: _ping(url), 10, "redis-server did not start")
    yield url
    proc.terminate()


def _ping(url) -> bool:
    try:
        return redis.Redis.from_url(url).ping()
    except redis.RedisError:
        return False


@pytest.fixture(autouse=True)
def clean_redis(redis_url, monkeypatch):
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("RETRY_DELAY_SECONDS", "1")
    redis.Redis.from_url(redis_url).flushall()
    yield
    redis.Redis.from_url(redis_url).flushall()


@pytest.fixture
def api():
    return TestClient(app)


class Origin:
    """Plays the storage (source download, preview PUT) and the client (callback). Tests script its answers."""

    def __init__(self):
        self.files: dict[str, bytes] = {}  # name -> bytes served at /src/<name>
        self.chunked: set[str] = set()  # names served without Content-Length
        self.source_status: dict[str, list[int]] = defaultdict(list)  # answers to give before serving the file
        self.gate = threading.Event()  # a source download waits until it is set
        self.gate.set()
        self.hits: dict[str, int] = defaultdict(int)
        self.previews: dict[str, tuple[bytes, str]] = {}  # name -> (bytes, content type)
        self.preview_status: dict[str, list[int]] = defaultdict(list)
        self.callbacks: list[tuple[str, dict, int]] = []  # (name, JSON, status answered)
        self.callback_status: dict[str, list[int]] = defaultdict(list)
        origin = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status, body=b"", headers=None):
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                if "Content-Length" not in (headers or {}) and "Transfer-Encoding" not in (headers or {}):
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _scripted(self, queue: list[int]):
                return queue.pop(0) if queue else 200

            def do_GET(self):
                name = self.path.removeprefix("/src/")
                origin.hits[name] += 1
                origin.gate.wait(60)
                if (status := self._scripted(origin.source_status[name])) != 200:
                    return self._reply(status)
                body = origin.files[name]
                if name in origin.chunked:
                    self.send_response(200)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for i in range(0, len(body), 64):
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(body[i:i + 64]), body[i:i + 64]))
                    self.wfile.write(b"0\r\n\r\n")
                    return
                self._reply(200, body)

            def do_PUT(self):
                name = self.path.removeprefix("/preview/")
                data = self.rfile.read(int(self.headers["Content-Length"]))
                if (status := self._scripted(origin.preview_status[name])) == 200:
                    origin.previews[name] = (data, self.headers["Content-Type"])
                self._reply(status)

            def do_POST(self):
                name = self.path.removeprefix("/callback/")
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                status = self._scripted(origin.callback_status[name])
                origin.callbacks.append((name, body, status))
                self._reply(status)

        self.server = ThreadingHTTPServer(("127.0.0.1", free_port()), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server.server_port}{path}"

    def callbacks_of(self, name: str) -> list[tuple[dict, int]]:
        return [(body, status) for n, body, status in self.callbacks if n == name]


@pytest.fixture
def origin(files):
    server = Origin()
    server.files = {path.name: path.read_bytes() for path in files.values()}
    yield server
    server.gate.set()
    server.server.shutdown()


@pytest.fixture
def worker():
    """Starts `python -m app.worker` as a process group of its own; extra environment per call."""
    started = []

    def start(**env) -> subprocess.Popen:
        env = {**os.environ, "MAX_CONCURRENT_CONVERSIONS": "2", **{k: str(v) for k, v in env.items()}}
        proc = subprocess.Popen([sys.executable, "-m", "app.worker"], cwd=REPO, env=env, start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started.append(proc)
        until(lambda: _registered(env), 30, "worker did not register")
        return proc

    yield start
    for proc in started:
        kill_tree(proc)


def _registered(env) -> bool:
    return subprocess.run([sys.executable, "-m", "app.worker", "check"], cwd=REPO, env=env,
                          capture_output=True).returncode == 0


def descendants(pid: int) -> list[int]:
    """All child processes (recursively) of `pid`, from /proc."""
    children = defaultdict(list)
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                stat = (entry / "stat").read_text()
            except OSError:
                continue
            children[int(stat.rsplit(")", 1)[1].split()[1])].append(int(entry.name))
    found, todo = [], [pid]
    while todo:
        for child in children[todo.pop()]:
            found.append(child)
            todo.append(child)
    return found


def horses(proc: subprocess.Popen) -> list[int]:
    """The work-horses of a worker pool: its descendants that lead a process group of their own."""
    return [p for p in descendants(proc.pid) if _pgid(p) == p]


def _pgid(pid: int) -> int:
    try:
        return os.getpgid(pid)
    except ProcessLookupError:
        return 0


def kill_worker(proc: subprocess.Popen) -> None:
    """Kills the pool and its workers; work-horses lead groups of their own and survive, like orphans."""
    with suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    # RQ's scheduler lock (it moves jobs that wait for a re-run) runs out after about a minute; do not wait for that
    connection = store.connection()
    if keys := connection.keys("rq:scheduler-lock:*"):
        connection.delete(*keys)


def kill_tree(proc: subprocess.Popen) -> None:
    for pid in horses(proc):
        with suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)
    kill_worker(proc)


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"  # a zombie is as good as gone


def submit(api, origin, name: str, external_id: str | None = None, **body):
    """Submits a job for the origin's file `name`; returns the response."""
    external_id = external_id or name
    payload = {"external_id": external_id, "file_name": name, "source_url": origin.url(f"/src/{name}"),
               "operations": ["extract", "preview"], "preview_url": origin.url(f"/preview/{external_id}"),
               "callback_url": origin.url(f"/callback/{external_id}"), **body}
    return api.post("/jobs", json=payload)


def wait_for_status(api, job_id: str, *statuses: str, seconds=60.0) -> dict:
    return until(lambda: (job := api.get(f"/jobs/{job_id}").json())["status"] in statuses and job, seconds,
                 f"job {job_id} did not reach {statuses}")
