"""Supported formats: extension -> (kind, module that handles it). The extension decides, not the client MIME type."""

DRAWING_2D = "DRAWING_2D"
MODEL_3D = "MODEL_3D"
SCAN_2D = "SCAN_2D"  # photos and scans of drawings: only with the operation "vectorize", which needs one of these

# Preview file extension and content type per kind; the preview of a scan shows the recognised drawing.
PREVIEW = {DRAWING_2D: ("pdf", "application/pdf"), MODEL_3D: ("glb", "model/gltf-binary"),
           SCAN_2D: ("pdf", "application/pdf")}
# What "vectorize" uploads to result_url: a ZIP with the DXF, a PDF, the review overlay and a report.
RESULT_CONTENT_TYPE = "application/zip"

_TABLE = {
    DRAWING_2D: {"drawing": "dxf dwg plt hpgl hpg hp2"},
    MODEL_3D: {
        "brep": "step stp iges igs brep brp",
        "mesh": "stl obj ply off 3mf dae gltf glb",
        "ifc": "ifc",
    },
    SCAN_2D: {"scan": "jpg jpeg png tif tiff heic heif webp bmp pdf"},
}

FORMATS = {
    ext: (kind, module)
    for kind, modules in _TABLE.items()
    for module, exts in modules.items()
    for ext in exts.split()
}


def formats_by_kind() -> dict[str, list[str]]:
    return {kind: [e for e, (k, _) in FORMATS.items() if k == kind] for kind in _TABLE}
