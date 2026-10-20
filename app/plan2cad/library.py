"""Symbol libraries: the CAD blocks recognised symbols become (valves, pumps, instruments, ...).

A library is a folder with a `library.toml` and a DXF, the same folder as a ZIP, or a bare DXF (see
docs/symbol-libraries.md). Each block of the DXF is one symbol around its insertion point, at its printed size (P&ID
symbols) or at its real size (fixtures on floor plans, shrunk by the drawing scale before matching). Its
lines, arcs, circles, polylines, splines and ellipses are the outline the matcher looks for; POINTs on the layer PORT
are where pipes connect, LINEs on the layer THROUGH a line that runs through the symbol (the pipe under a flow arrow:
required for a match, not part of the symbol), ATTDEFs its attributes, placed where their text stands. Everything
else (hatches, texts, ...) is not matched but stays in the block that goes into the result."""

from __future__ import annotations

import io
import math
import os
import re
import tomllib
import zipfile
import zlib
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import pairwise
from pathlib import Path, PurePosixPath

import ezdxf
from ezdxf import recover, units

from .model import Arc, Circle, Line, Symbol

FORMAT = 1  # the library format this version reads
DRAWINGS = ("pid", "building")  # the drawing types symbols are recognised on: P&IDs and floor plans
SIZES = ("printed", "real")  # blocks drawn at the size they have on the sheet, or at the size of the real object
ASSUMED_SCALES = (50, 100)  # drawing scales tried for real-size symbols while the scale of a drawing is unknown
DEFAULTS = "din-10628,floor-plan"  # the built-in libraries used when nothing else is chosen
ID = r"[a-z0-9][a-z0-9-]{0,63}"
MANIFEST = "library.toml"
BUILT_IN = Path(__file__).resolve().parents[2] / "libraries"  # shipped with cad-ingest, always installed
STARTER = BUILT_IN / "din-10628"
FLOOR_PLAN = BUILT_IN / "floor-plan"
PORT_LAYER, THROUGH_LAYER = "PORT", "THROUGH"
OUTLINE = {"LINE", "ARC", "CIRCLE", "LWPOLYLINE", "POLYLINE", "SPLINE", "ELLIPSE"}
FLATTEN = 0.05  # mm: splines and ellipses become straight pieces this close to the curve
NESTING = 8  # levels of blocks inside blocks that are followed
MAX_BYTES = 50 * 2**20  # largest DXF taken from a ZIP (after decompression)


class LibraryError(ValueError):
    """The library cannot be used; the message says why."""


@dataclass(eq=False)
class Library:
    id: str
    name: str
    version: str | None
    licence: str | None
    drawing: list[str]
    unit: float  # paper millimetres per drawing unit of the DXF ($INSUNITS)
    doc: ezdxf.document.Drawing = field(repr=False)
    symbols: dict[str, Template] = field(default_factory=dict, repr=False)  # by block name
    warnings: list[str] = field(default_factory=list, repr=False)  # what was ignored, for `cad-ingest library check`
    path: Path | None = None
    size: str = "printed"  # one of SIZES

    def summary(self) -> dict:
        """What GET /libraries tells about the library."""
        return {"id": self.id, "name": self.name, "version": self.version, "licence": self.licence,
                "drawing": self.drawing, "size": self.size,
                "symbols": [{"name": t.name, "description": t.description} for t in self.symbols.values()]}


@dataclass
class Template:
    """A library symbol in its own coordinates: millimetres, y pointing down like the sheet, origin = insertion point."""

    name: str
    description: str
    parts: list[Line | Arc | Circle]
    groups: list[int]  # the drawing entity (polyline, circle, ...) each part comes from
    ports: list[tuple[float, float]]
    tags: dict[str, tuple[float, float]]  # attribute tag -> where its text stands
    through: list[Line]  # lines that run through the symbol in the drawing
    unit: float = 1.0  # millimetres of these coordinates per drawing unit of the block in its library
    library: Library | None = field(default=None, repr=False, compare=False)


@lru_cache(maxsize=64)
def read(path: Path | str) -> Library:
    """The library in the folder, ZIP or DXF at ``path``."""
    path = Path(path)
    if not path.is_dir():
        try:
            data = path.read_bytes()
        except OSError as e:
            raise LibraryError(f"{path.name}: {e.strerror or e}") from None
        lib = from_bytes(data, path.name)
    else:
        try:
            if not (path / MANIFEST).is_file():
                raise LibraryError(f"the folder has no {MANIFEST}")
            manifest = _manifest((path / MANIFEST).read_bytes(), path.name)
            lib = _library(manifest, (path / manifest["file"]).read_bytes())
        except OSError as e:
            raise LibraryError(f"{path.name}: {e.strerror or e}") from None
        except LibraryError as e:
            raise LibraryError(f"{path.name}: {e}") from None
    lib.path = path
    return lib


def from_bytes(data: bytes, name: str) -> Library:
    """A library from the bytes of a ZIP or a DXF (a download); ``name`` is its file name, for the id and messages."""
    try:
        if data.startswith(b"PK\x03\x04"):
            return _library(*_unzip(data, name))
        if Path(name).suffix.lower() not in ("", ".dxf"):
            raise LibraryError("a library is a folder, a ZIP or a DXF file (convert others with "
                               "`cad-ingest library build`)")
        return _library(_manifest(None, Path(name).stem), data, bare=True)
    except LibraryError as e:
        raise LibraryError(f"{name}: {e}") from None


def load(path: Path | str = STARTER) -> dict[str, Template]:
    """The symbols of the library at ``path`` by block name (the starter library by default)."""
    return read(path).symbols


@lru_cache(maxsize=8)
def installed(folders: str = "") -> dict[str, Library]:
    """The built-in libraries plus those in ``folders`` (separated by ':', like SYMBOL_LIBRARIES), by id. A folder
    holds library folders, ZIPs and DXF files; sub-folders without a library.toml (lost+found) are skipped."""
    out: dict[str, Library] = {}
    for folder in [BUILT_IN] + [Path(f) for f in folders.split(os.pathsep) if f]:
        if not folder.is_dir():
            continue
        for entry in sorted(folder.iterdir()):
            if entry.name.startswith(".") or not ((entry / MANIFEST).is_file() if entry.is_dir()
                                                  else entry.suffix.lower() in (".zip", ".dxf")):
                continue
            lib = read(entry)
            if lib.id in out:
                raise LibraryError(f"Two symbol libraries have the id '{lib.id}': {out[lib.id].path} and {entry}")
            out[lib.id] = lib
    return out


def select(ids: list[str] | None = None, own: list[Library] | tuple = (), folders: str = "",
           defaults: str = DEFAULTS) -> list[Library]:
    """The libraries for one conversion, in order of precedence: ``own`` (the libraries a job brings along, or the
    CLI's), then the installed libraries ``ids``. Without either, the comma-separated ``defaults``."""
    available = installed(folders)
    chosen = list(own)
    for i in ids or ([] if chosen else [d.strip() for d in defaults.split(",") if d.strip()]):
        if i not in available:
            raise LibraryError(f"The symbol library '{i}' is not installed (installed: {', '.join(available)})")
        chosen.append(available[i])
    return chosen


def resolve(value: str, folders: str = "") -> Library:
    """An installed library by id, else the library folder, ZIP or DXF at the path ``value`` (the command line)."""
    available = installed(folders)
    if value in available and not os.path.exists(value):
        return available[value]
    if not os.path.exists(value):
        raise LibraryError(f"'{value}' is neither an installed library ({', '.join(available)}) nor a library "
                           "folder, ZIP or DXF file")
    return read(value)


def combine(libraries: list[Library], drawing: str = "pid", scale: int | None = None) -> dict[str, Template]:
    """The symbols of the libraries for one drawing type, at their size on the sheet. Real-size symbols are shrunk by
    the drawing ``scale`` (the denominator); while it is unknown, by each of ASSUMED_SCALES. Where two libraries
    define the same block name the earlier one wins, so a company library can replace a symbol of a starter library
    on purpose."""
    out: dict[str, Template] = {}
    for lib in libraries:
        if drawing not in lib.drawing:
            continue
        for name, t in lib.symbols.items():
            if any(other.name == name for other in out.values()):
                continue
            if lib.size == "real":
                out.update({f"{name} 1:{s}": scaled(t, 1 / s) for s in ([scale] if scale else ASSUMED_SCALES)})
            else:
                out[name] = t
    return out


def scaled(t: Template, f: float) -> Template:
    """The symbol ``f`` times as large: a real-size symbol at the drawing scale 1:(1/f)."""
    def part(q):
        if isinstance(q, Line):
            return replace(q, x1=q.x1 * f, y1=q.y1 * f, x2=q.x2 * f, y2=q.y2 * f)
        return replace(q, cx=q.cx * f, cy=q.cy * f, r=q.r * f)

    return replace(t, parts=[part(q) for q in t.parts], ports=[(x * f, y * f) for x, y in t.ports],
                   tags={tag: (x * f, y * f) for tag, (x, y) in t.tags.items()},
                   through=[part(q) for q in t.through], unit=t.unit * f)


# --------------------------------------------------------------------------- reading

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:64].strip("-") or "library"


def _manifest(raw: bytes | None, default_id: str) -> dict:
    """The fields of library.toml (``raw``, None for a bare DXF) with their defaults, checked."""
    try:
        m = tomllib.loads(raw.decode("utf-8")) if raw is not None else {}
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise LibraryError(f"{MANIFEST}: {e}") from None

    def get(key: str, kind: type, default):
        value = m.get(key, default)
        if value is not None and not isinstance(value, kind):
            raise LibraryError(f"{MANIFEST}: '{key}' must be {'a string' if kind is str else 'a ' + kind.__name__}")
        return value

    out = {"id": get("id", str, slug(default_id)), "format": get("format", int, FORMAT),
           "version": get("version", str, None), "licence": get("licence", str, m.get("license")),
           "drawing": get("drawing", list, ["pid"]), "size": get("size", str, "printed"),
           "file": get("file", str, "symbols.dxf")}
    out["name"] = get("name", str, out["id"])
    if not re.fullmatch(ID, out["id"]):
        raise LibraryError(f"{MANIFEST}: 'id' must be lower-case letters, digits and hyphens: '{out['id']}'")
    if not 1 <= out["format"] <= FORMAT:
        raise LibraryError(f"{MANIFEST}: library format {out['format']} needs a newer cad-ingest "
                           f"(this one reads format {FORMAT})")
    if not out["drawing"] or not all(d in DRAWINGS for d in out["drawing"]):
        raise LibraryError(f"{MANIFEST}: 'drawing' must list drawing types out of {', '.join(DRAWINGS)} "
                           "(P&IDs and floor plans)")
    if out["size"] not in SIZES:
        raise LibraryError(f"{MANIFEST}: 'size' must be one of {', '.join(SIZES)}: '{out['size']}'")
    file = PurePosixPath(out["file"])
    if file.is_absolute() or ".." in file.parts or file.suffix.lower() != ".dxf":
        raise LibraryError(f"{MANIFEST}: 'file' must be a DXF file inside the library: '{out['file']}'")
    out["unknown"] = sorted(set(m) - set(out) - {"license"})
    return out


def _unzip(data: bytes, name: str) -> tuple[dict, bytes]:
    """The manifest and the DXF of a zipped library folder. Nothing is written to disk, and no more than MAX_BYTES
    is decompressed, so a hostile ZIP can neither escape nor fill the disk."""
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise LibraryError("not a readable ZIP file") from None
    manifests = [i for i in z.infolist() if PurePosixPath(i.filename).name == MANIFEST
                 and len(PurePosixPath(i.filename).parts) <= 2]
    if len(manifests) != 1:
        raise LibraryError(f"a library ZIP holds one {MANIFEST}, at the top or in one folder")
    root = PurePosixPath(manifests[0].filename).parent
    manifest = _manifest(_member(z, manifests[0]), root.name or Path(name).stem)
    try:
        dxf = z.getinfo(str(root / manifest["file"]))
    except KeyError:
        raise LibraryError(f"{manifest['file']} is missing") from None
    return manifest, _member(z, dxf)


def _member(z: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    try:
        with z.open(info) as f:
            data = f.read(MAX_BYTES + 1)  # the size in the directory may lie
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError, EOFError, zlib.error) as e:
        raise LibraryError(f"{info.filename} cannot be unpacked ({e})") from None  # damaged, encrypted, ...
    if len(data) > MAX_BYTES:
        raise LibraryError(f"{info.filename} is larger than {MAX_BYTES // 2**20} MB")
    return data


def _library(manifest: dict, data: bytes, bare: bool = False) -> Library:
    where = "" if bare else f"{manifest['file']}: "  # the caller names a bare DXF
    try:
        doc, _ = recover.read(io.BytesIO(data))
    except Exception as e:  # ezdxf raises various types for broken input
        raise LibraryError(f"{where}not a readable DXF file ({type(e).__name__})") from None
    unit = units.conversion_factor(doc.units, units.InsertUnits.Millimeters) if doc.units else 1.0
    lib = Library(manifest["id"], manifest["name"], manifest["version"], manifest["licence"], manifest["drawing"],
                  unit, doc, warnings=[f"{MANIFEST}: unknown key '{k}' is ignored" for k in manifest["unknown"]],
                  size=manifest["size"])
    for block in doc.blocks:
        if block.name.startswith(("*", "_")):  # anonymous blocks (dimensions, ...) and arrowheads
            continue
        t, ignored = _template(block, unit)
        if ignored:
            lib.warnings.append(f"{block.name}: not used for matching: "
                                + ", ".join(f"{n} {kind}" for kind, n in sorted(ignored.items())))
        if t is None:
            if len(block):
                lib.warnings.append(f"{block.name}: skipped, it has no lines, arcs, circles or polylines")
        else:
            t.library = lib
            lib.symbols[block.name] = t
    if not lib.symbols:
        raise LibraryError(f"{where}no block with lines, arcs, circles or polylines")
    return lib


def _template(block, unit: float) -> tuple[Template | None, Counter]:
    parts, groups, ports, tags, through, ignored = [], [], [], {}, [], Counter()
    bx, by = block.block.dxf.base_point.x, block.block.dxf.base_point.y  # the insertion point of the block

    def down(p) -> tuple[float, float]:  # to paper millimetres around the insertion point, y down like the sheet
        return unit * (float(p[0]) - bx), 0.0 - unit * (float(p[1]) - by)  # 0.0 - avoids -0.0

    def visit(entities, depth: int) -> None:
        for e in entities:
            kind, layer = e.dxftype(), e.dxf.get("layer", "0").upper()
            if kind == "POINT" and layer == PORT_LAYER:
                ports.append(down(e.dxf.location))
            elif kind == "LINE" and layer == THROUGH_LAYER:
                through.append(_part(e, e, down, unit))
            elif kind == "ATTDEF":
                aligned = e.dxf.halign or e.dxf.valign
                tags[e.dxf.tag] = down(e.dxf.align_point if aligned else e.dxf.insert)
            elif kind in OUTLINE:
                pieces = _pieces(e, down, unit)
                parts.extend(pieces)
                groups.extend([len(set(groups))] * len(pieces))
            elif kind == "INSERT" and depth < NESTING:  # a block inside the block: its outline counts
                visit(e.virtual_entities(), depth + 1)
            else:
                ignored[kind] += 1

    visit(block, 0)
    if not parts:
        return None, ignored
    description = block.block.dxf.get("description", "") or block.name
    return Template(block.name, description, parts, groups, ports, tags, through, unit), ignored


def _pieces(e, down, unit: float) -> list[Line | Arc | Circle]:
    kind = e.dxftype()
    if kind in ("SPLINE", "ELLIPSE"):
        width = _width(e)
        return [Line(*down(a), *down(b), width) for a, b in pairwise(e.flattening(FLATTEN / unit))]
    if kind.endswith("POLYLINE"):
        return [_part(v, e, down, unit) for v in e.virtual_entities() if v.dxftype() in ("LINE", "ARC")]
    return [_part(e, e, down, unit)]


def _width(owner) -> float:
    lw = owner.dxf.get("lineweight", -1)
    return lw / 100 if lw > 0 else 0.35


def _part(e, owner, down, unit: float) -> Line | Arc | Circle:
    """A DXF line, arc or circle in sheet orientation (y down: counter-clockwise DXF angles become clockwise ones)."""
    width, kind = _width(owner), e.dxftype()
    if kind == "LINE":
        return Line(*down(e.dxf.start), *down(e.dxf.end), width)
    if kind == "CIRCLE":
        return Circle(*down(e.dxf.center), unit * e.dxf.radius, width)
    return Arc(*down(e.dxf.center), unit * e.dxf.radius, (-e.dxf.end_angle) % 360, (-e.dxf.start_angle) % 360, width)


# --------------------------------------------------------------------------- placing

def transform(x: float, y: float, angle: float = 0.0, mirror: bool = False, scale: float = 1.0):
    """Maps symbol coordinates to the sheet: mirror across the symbol's vertical axis, scale, turn clockwise on screen
    by ``angle`` degrees, move the origin to (x, y)."""
    c, s, m = math.cos(math.radians(angle)) * scale, math.sin(math.radians(angle)) * scale, -1 if mirror else 1
    return lambda u, v: (x + c * m * u - s * v, y + s * m * u + c * v)


def place(t: Template, x: float, y: float, angle: float = 0.0, mirror: bool = False, scale: float = 1.0) -> Symbol:
    """The symbol ``t`` placed on the sheet."""
    f = transform(x, y, angle, mirror, scale)

    def moved(q):
        if isinstance(q, Line):
            return Line(*f(q.x1, q.y1), *f(q.x2, q.y2), q.width)
        if isinstance(q, Circle):
            return Circle(*f(q.cx, q.cy), q.r * scale, q.width)
        start, end = (180 - q.end, 180 - q.start) if mirror else (q.start, q.end)  # mirroring reverses the direction
        return Arc(*f(q.cx, q.cy), q.r * scale, (start + angle) % 360, (end + angle) % 360, q.width)

    return Symbol(t.name, x, y, angle % 360, mirror, scale, [moved(q) for q in t.parts], [f(*p) for p in t.ports],
                  template=t)
