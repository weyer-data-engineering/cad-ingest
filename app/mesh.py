"""Triangle-mesh and scene formats (STL, OBJ, PLY, OFF, 3MF, DAE, glTF, GLB) via trimesh."""
import json
import math
from collections import defaultdict

import trimesh

from .config import PREVIEW_SOFT_LIMIT_BYTES
from .errors import ConversionError

MAX_LINES = 300
# Formats that are Z-up by convention (CAD, 3D printing); glTF/DAE/OBJ writers normally produce Y-up already.
Z_UP = {"stl", "ply", "off", "3mf"}
IGNORED_METADATA = {"file_path", "file_name", "file_type", "file_size"}


def extract(ext: str, path: str, workdir: str, name: str) -> tuple[str, list[str]]:
    scene = load(ext, path)
    nodes = defaultdict(list)  # geometry name -> names of the scene nodes that use it
    for node in scene.graph.nodes_geometry:
        nodes[scene.graph[node][1]].append(node)
    lines = [f"# 3D model: {name}", "", f"- Format: {ext.upper()}", f"- Objects: {len(scene.geometry)}",
             f"- Overall size: {_extents(scene.extents)} (file units)"]
    lines += _metadata(scene.metadata)
    lines += ["", "## Objects", ""]
    for i, (geometry_name, geometry) in enumerate(scene.geometry.items()):
        if i == MAX_LINES:
            lines.append(f"- ... {len(scene.geometry) - MAX_LINES} more")
            break
        lines.append(_geometry_line(geometry_name, geometry, nodes[geometry_name]))
    return "\n".join(lines) + "\n", []


def preview(ext: str, path: str, out: str) -> list[str]:
    return write_glb(load(ext, path), out, z_up=ext in Z_UP)


def load(ext: str, path: str) -> trimesh.Scene:
    scene = trimesh.load(path, file_type=ext, force="scene")
    scene.geometry = {n: g for n, g in scene.geometry.items() if len(getattr(g, "vertices", [])) > 0}
    if not scene.geometry:
        raise ConversionError("The file contains no geometry")
    return scene


def write_glb(scene: trimesh.Scene, out: str, z_up: bool) -> list[str]:
    """Exports `scene` as GLB with +Y up; oversized scenes are decimated once."""
    if z_up:
        scene.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
    data = scene.export(file_type="glb")
    warnings = []
    if len(data) > PREVIEW_SOFT_LIMIT_BYTES:
        keep = PREVIEW_SOFT_LIMIT_BYTES / len(data) * 0.8
        try:
            for name, geometry in scene.geometry.items():
                if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 1000:
                    scene.geometry[name] = geometry.simplify_quadric_decimation(face_count=int(len(geometry.faces) * keep))
            data = scene.export(file_type="glb")
            warnings.append("The preview was simplified to keep it small")
        except Exception as e:
            warnings.append(f"The preview is large and could not be simplified: {e}")
    with open(out, "wb") as f:
        f.write(data)
    return warnings


def _extents(values) -> str:
    return " x ".join(f"{v:.6g}" for v in values)


def _geometry_line(name: str, geometry, nodes: list[str]) -> str:
    details = [f"{len(geometry.vertices)} vertices"]
    if isinstance(geometry, trimesh.Trimesh):
        details.append(f"{len(geometry.faces)} faces")
        details.append(f"size {_extents(geometry.extents)}")
        if geometry.is_watertight:
            details.append(f"watertight, volume {abs(geometry.volume):.6g}")
        else:
            details.append("not watertight")
    line = f"- {name}: " + ", ".join(details)
    if len(nodes) > 1:
        line += f", {len(nodes)} instances"
    shown = [n for n in nodes if n != name][:5]
    return line + (f" (nodes: {', '.join(shown)})" if shown else "")


def _metadata(metadata: dict) -> list[str]:
    """Embedded metadata such as the glTF asset/extras or the 3MF title and designer."""
    lines = []
    for key, value in metadata.items():
        if key in IGNORED_METADATA or value in (None, "", {}, []):
            continue
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        lines.append(f"- Metadata {key}: {text[:500]}")
    return lines[:50]
