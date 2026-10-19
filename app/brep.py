"""Boundary-representation CAD (STEP, IGES, BREP) via OpenCascade (OCP). Names, colours and the assembly structure
survive because the files are read into an XCAF document."""
import mmap
import os
import re
from collections import Counter

from OCP.BRep import BRep_Builder
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.Bnd import Bnd_Box
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.IGESCAFControl import IGESCAFControl_Reader
from OCP.Message import Message_ProgressRange
from OCP.Quantity import Quantity_Color
from OCP.RWGltf import RWGltf_CafWriter
from OCP.RWMesh import RWMesh_CoordinateSystem
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
from OCP.TColStd import TColStd_IndexedDataMapOfStringString
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopoDS import TopoDS_Shape
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_ColorCurv, XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_DocumentTool

from .config import PREVIEW_SOFT_LIMIT_BYTES
from .errors import ConversionError

MAX_LINES = 500
MAX_PART_DETAILS = 300  # volume/area integration is slow on huge assemblies
LINEAR_DEFLECTION = 0.001  # relative to the bounding-box diagonal; the retry uses a 4x coarser mesh
ANGULAR_DEFLECTION = 0.5
STEP_EXTS = {"step", "stp"}


def extract(ext: str, path: str, workdir: str, name: str) -> tuple[str, list[str]]:
    lines = [f"# CAD model: {name}", ""]
    lines += _file_header(ext, path)
    doc = _read(ext, path)
    shapes, colors = _tools(doc)
    free = TDF_LabelSequence()
    shapes.GetFreeShapes(free)
    box = Bnd_Box()
    parts: dict[int, str] = {}  # shape label tag -> markdown line, computed once per unique part
    tree: list[str] = []
    budget = [MAX_PART_DETAILS]
    for i in range(1, free.Length() + 1):
        BRepBndLib.Add_s(shapes.GetShape_s(free.Value(i)), box, False)
        _walk(shapes, colors, free.Value(i), 1, tree, parts, budget)
    if box.IsVoid():
        raise ConversionError("The file contains no geometry")
    lines += ["- Overall size: " + _size(box) + " mm", f"- Unique parts: {len(parts)}", "", "## Structure", ""]
    lines += tree[:MAX_LINES] + ([f"- ... {len(tree) - MAX_LINES} more"] if len(tree) > MAX_LINES else [])
    return "\n".join(lines) + "\n", []


def preview(ext: str, path: str, out: str) -> list[str]:
    for factor in (1, 4):
        doc = _read(ext, path)
        shapes, _ = _tools(doc)
        free = TDF_LabelSequence()
        shapes.GetFreeShapes(free)
        box = Bnd_Box()
        for i in range(1, free.Length() + 1):
            BRepBndLib.Add_s(shapes.GetShape_s(free.Value(i)), box, False)
        if box.IsVoid():
            raise ConversionError("The file contains no geometry")
        diagonal = box.CornerMin().Distance(box.CornerMax())
        for i in range(1, free.Length() + 1):
            BRepMesh_IncrementalMesh(shapes.GetShape_s(free.Value(i)), diagonal * LINEAR_DEFLECTION * factor,
                                     False, ANGULAR_DEFLECTION * factor, True)
        writer = RWGltf_CafWriter(TCollection_AsciiString(out), True)
        converter = writer.ChangeCoordinateSystemConverter()
        converter.SetInputLengthUnit(0.001)  # OpenCascade works in mm after import, glTF in metres
        converter.SetInputCoordinateSystem(RWMesh_CoordinateSystem.RWMesh_CoordinateSystem_Zup)  # glTF output is Y-up
        if not writer.Perform(doc, TColStd_IndexedDataMapOfStringString(), Message_ProgressRange()) \
                or not os.path.exists(out) or os.path.getsize(out) == 0:
            raise ConversionError("The model contains no triangulable geometry")
        if os.path.getsize(out) <= PREVIEW_SOFT_LIMIT_BYTES:
            return []
    return [f"The preview is larger than {PREVIEW_SOFT_LIMIT_BYTES // 2**20} MB even with a coarse mesh"]


def _tools(doc: TDocStd_Document):
    return XCAFDoc_DocumentTool.ShapeTool_s(doc.Main()), XCAFDoc_DocumentTool.ColorTool_s(doc.Main())


def _read(ext: str, path: str) -> TDocStd_Document:
    doc = TDocStd_Document(TCollection_ExtendedString("BinXCAF"))
    XCAFApp_Application.GetApplication_s().InitDocument(doc)
    if ext in ("brep", "brp"):
        shape = TopoDS_Shape()
        if not BRepTools.Read_s(shape, path, BRep_Builder(), Message_ProgressRange()) or shape.IsNull():
            raise ConversionError("Not a readable BREP file")
        XCAFDoc_DocumentTool.ShapeTool_s(doc.Main()).AddShape(shape, False)
        return doc
    reader = STEPCAFControl_Reader() if ext in STEP_EXTS else IGESCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise ConversionError(f"Not a readable {'STEP' if ext in STEP_EXTS else 'IGES'} file")
    if not reader.Transfer(doc):
        raise ConversionError("The file contains no transferable geometry")
    return doc


def _name(label: TDF_Label) -> str:
    attribute = TDataStd_Name()
    return attribute.Get().ToExtString().strip() if label.FindAttribute(TDataStd_Name.GetID_s(), attribute) else ""


def _size(box: Bnd_Box) -> str:
    lo, hi = box.CornerMin(), box.CornerMax()
    return f"{hi.X() - lo.X():.6g} x {hi.Y() - lo.Y():.6g} x {hi.Z() - lo.Z():.6g}"


def _color(colors, label: TDF_Label) -> str:
    color = Quantity_Color()
    for kind in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen, XCAFDoc_ColorCurv):
        if colors.GetColor_s(label, kind, color):
            return "#%02x%02x%02x" % tuple(round(c * 255) for c in (color.Red(), color.Green(), color.Blue()))
    return ""


def _part_line(shapes, colors, label: TDF_Label, budget: list[int]) -> str:
    """Bounding box, volume, surface area and colour of one part."""
    shape = shapes.GetShape_s(label)
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box, False)
    details = [] if box.IsVoid() else [f"size {_size(box)} mm"]
    if budget[0] > 0 and not box.IsVoid():
        budget[0] -= 1
        volume, area = GProp_GProps(), GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, volume)
        BRepGProp.SurfaceProperties_s(shape, area)
        if volume.Mass() > 1e-9:
            details.append(f"volume {volume.Mass():.6g} mm3")
        details.append(f"surface {area.Mass():.6g} mm2")
    if color := _color(colors, label):
        details.append(f"colour {color}")
    return ", ".join(details)


def _walk(shapes, colors, label: TDF_Label, count: int, out: list[str], parts: dict, budget: list[int], depth: int = 0) -> None:
    """Appends the assembly tree below `label` as a nested list; equal components are merged to "N x"."""
    name = _name(label) or "(unnamed)"
    prefix = "  " * depth + "- " + (f"{count} x " if count > 1 else "")
    if not shapes.IsAssembly_s(label) or depth > 30:
        if label.Tag() not in parts:
            parts[label.Tag()] = _part_line(shapes, colors, label, budget)
        out.append(f"{prefix}{name}: {parts[label.Tag()]}")
        return
    out.append(f"{prefix}{name} (assembly)")
    components = TDF_LabelSequence()
    shapes.GetComponents_s(label, components)
    referred: Counter = Counter()
    labels: dict[int, TDF_Label] = {}
    for i in range(1, components.Length() + 1):
        ref = TDF_Label()
        if shapes.GetReferredShape_s(components.Value(i), ref):
            referred[ref.Tag()] += 1
            labels[ref.Tag()] = ref
    for tag, n in referred.items():
        _walk(shapes, colors, labels[tag], n, out, parts, budget, depth + 1)


def _file_header(ext: str, path: str) -> list[str]:
    """Author, originating system, timestamp and units straight from the file; XCAF does not expose them."""
    if ext in ("brep", "brp"):
        return ["- Format: OpenCascade BREP (no metadata, units unspecified)"]
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if ext in STEP_EXTS:
            return _step_header(data)
        return _iges_header(data)


def _strings(text: str) -> list[str]:
    return [s.replace("''", "'") for s in re.findall(r"'((?:[^']|'')*)'", text) if s.strip()]


def _step_header(data: mmap.mmap) -> list[str]:
    head = data[:200_000].decode("latin-1")
    lines = ["- Format: STEP"]
    if m := re.search(r"FILE_DESCRIPTION\s*\((.*?)\)\s*;", head, re.S):
        lines.append("- Description: " + "; ".join(_strings(m.group(1))))
    if m := re.search(r"FILE_NAME\s*\((.*?)\)\s*;", head, re.S):
        # FILE_NAME: name, timestamp, author(s), organisation(s), preprocessor, originating system, authorisation
        fields = _step_fields(m.group(1))
        for label, index in (("File name", 0), ("Timestamp", 1), ("Author", 2), ("Organization", 3),
                             ("Preprocessor", 4), ("Originating system", 5)):
            if index < len(fields) and fields[index]:
                lines.append(f"- {label}: {fields[index]}")
    if m := re.search(r"FILE_SCHEMA\s*\((.*?)\)\s*;", head, re.S):
        lines.append("- Schema: " + "; ".join(_strings(m.group(1))))
    if m := re.search(rb"CONVERSION_BASED_UNIT\('(\w+)'", data):
        lines.append("- Length unit in file: " + m.group(1).decode("latin-1").lower())
    elif m := re.search(rb"SI_UNIT\((\.\w+\.)?,\.METRE\.\)", data):
        lines.append("- Length unit in file: " + (m.group(1) or b"").decode("latin-1").strip(".").lower() + "metre")
    return lines


def _step_fields(args: str) -> list[str]:
    """Top-level fields of a STEP argument list; parenthesised tuples are joined with ', '."""
    fields, depth, current = [], 0, ""
    for part in re.findall(r"'(?:[^']|'')*'|[(),]|[^'(),]+", args):
        if part == "(":
            depth += 1
        elif part == ")":
            depth -= 1
        elif part == "," and depth == 0:
            fields.append(current)
            current = ""
        elif part.startswith("'"):
            current += (", " if current else "") + part[1:-1].replace("''", "'")
    return fields + [current]


def _iges_header(data: mmap.mmap) -> list[str]:
    rows = data[:100_000].decode("latin-1").splitlines()
    start = " ".join(r[:72].strip() for r in rows if r[72:73] == "S")
    global_text = "".join(r[:72].rstrip() for r in rows if r[72:73] == "G")
    # Fields are comma separated; strings are Hollerith encoded (nHtext)
    values, pos = [], 0
    while pos < len(global_text) and len(values) < 30:
        if m := re.match(r"(\d+)H", global_text[pos:]):
            end = pos + len(m.group(0)) + int(m.group(1))
            values.append(global_text[pos + len(m.group(0)):end])
            pos = end + 1
        else:
            end = global_text.find(",", pos)
            end = len(global_text) if end < 0 else end
            values.append(global_text[pos:end].strip())
            pos = end + 1
    lines = ["- Format: IGES"]
    if start:
        lines.append(f"- Description: {start}")
    for label, index in (("File name", 3), ("Originating system", 4), ("Length unit in file", 14), ("Author", 20),
                         ("Organization", 21), ("Created", 24)):
        if index < len(values) and values[index].strip():
            lines.append(f"- {label}: {values[index].strip()}")
    return lines
