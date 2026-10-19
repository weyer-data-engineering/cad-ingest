"""IFC building models via IfcOpenShell."""
import os
from collections import Counter

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as element_util
import ifcopenshell.util.unit as unit_util
import numpy as np
import trimesh

from .errors import ConversionError
from .mesh import write_glb

MAX_LINES = 300
MAX_PSET_ELEMENTS = 20_000  # property sets are read per element, which is slow on huge models


def extract(ext: str, path: str, workdir: str, name: str) -> tuple[str, list[str]]:
    model = _open(path)
    project = (model.by_type("IfcProject") or [None])[0]
    unit = unit_util.get_project_unit(model, "LENGTHUNIT")
    lines = [f"# BIM model: {name}", "", f"- Format: IFC, schema {model.schema}"]
    if unit:
        prefix = getattr(unit, "Prefix", None) or ""
        lines.append(f"- Length unit: {(prefix + str(unit.Name)).lower()}")
    if project:
        lines += ["", "## Hierarchy", ""]
        lines += _hierarchy(project, 0)[:MAX_LINES]

    elements = model.by_type("IfcElement")
    counts = Counter(e.is_a() for e in elements)
    lines += ["", "## Elements per class", ""] + [f"- {cls}: {n}" for cls, n in counts.most_common()][:MAX_LINES]

    named = Counter(f"{e.is_a()} {e.Name}" for e in elements if e.Name)
    if named:
        lines += ["", "## Named elements", ""]
        lines += [f"- {text}" + (f" (x{n})" if n > 1 else "") for text, n in list(named.items())[:MAX_LINES]]
        if len(named) > MAX_LINES:
            lines.append(f"- ... {len(named) - MAX_LINES} more")

    materials = sorted({m.Name for m in model.by_type("IfcMaterial") if m.Name})
    if materials:
        lines += ["", "## Materials", ""] + [f"- {m}" for m in materials[:MAX_LINES]]

    properties = Counter()
    for e in elements[:MAX_PSET_ELEMENTS]:
        for pset, values in element_util.get_psets(e, psets_only=True).items():
            if pset.startswith("Pset_") and pset.endswith("Common"):
                for prop, value in values.items():
                    if prop != "id" and isinstance(value, (str, bool, int, float)) and value != "":
                        properties[f"{pset}.{prop} = {value}"] += 1
    if properties:
        lines += ["", "## Common property sets", ""]
        lines += [f"- {text} (x{n})" for text, n in properties.most_common(MAX_LINES)]
    return "\n".join(lines) + "\n", []


def preview(ext: str, path: str, out: str) -> list[str]:
    model = _open(path)
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)  # element placement baked in; coordinates in metres
    iterator = ifcopenshell.geom.iterator(settings, model, min(4, os.cpu_count() or 1))
    scene = trimesh.Scene()
    if iterator.initialize():
        while True:
            shape = iterator.get()
            geometry = shape.geometry
            if len(geometry.faces):
                scene.add_geometry(_mesh(geometry), node_name=f"{shape.name or shape.type} {shape.guid}",
                                   geom_name=f"{shape.type} {shape.id}")
            if not iterator.next():
                break
    if not scene.geometry:
        raise ConversionError("The IFC file contains no geometry")
    return write_glb(scene, out, z_up=True)


def _open(path: str) -> ifcopenshell.file:
    try:
        return ifcopenshell.open(path)
    except Exception as e:
        raise ConversionError(f"Not a readable IFC file: {e}") from None


def _mesh(geometry) -> trimesh.Trimesh:
    """One mesh per product, coloured with its dominant IFC surface style (per-face colours would triple the vertices)."""
    mesh = trimesh.Trimesh(np.array(geometry.verts).reshape(-1, 3), np.array(geometry.faces).reshape(-1, 3), process=False)
    if geometry.materials:
        style = geometry.materials[int(np.bincount(np.clip(geometry.material_ids, 0, None)).argmax())]
        alpha = 1 - style.transparency if style.has_transparency and style.transparency == style.transparency else 1
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[*style.diffuse.components, alpha], metallicFactor=0.0,
            alphaMode="BLEND" if alpha < 1 else "OPAQUE"))
    return mesh


def _hierarchy(obj, depth: int) -> list[str]:
    """Project > site > building > storey, with the number of directly contained elements."""
    label = f"{obj.is_a()} {obj.Name or ''}".strip()
    if obj.is_a("IfcBuildingStorey") and obj.Elevation is not None:
        label += f" (elevation {obj.Elevation:g})"
    contained = len(element_util.get_contained(obj)) if hasattr(obj, "ContainsElements") else 0
    lines = ["  " * depth + f"- {label}" + (f": {contained} elements" if contained else "")]
    for part in element_util.get_parts(obj):
        if part.is_a("IfcSpatialStructureElement") or part.is_a("IfcSite"):
            lines += _hierarchy(part, depth + 1)
    return lines
