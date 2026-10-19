"""Decodes uploads: photos (JPEG, PNG, TIFF, HEIC from iPhones) and the first page of a PDF scan."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pypdfium2
from PIL import Image, ImageOps
from pi_heif import register_heif_opener

register_heif_opener()
Image.MAX_IMAGE_PIXELS = 300_000_000  # A0 scans at 300 dpi have about 140 megapixels

PDF_DPI = 300
PDF_MAX_SIDE = 14000  # pixels; keeps A0 pages at about 300 dpi


def load_image(data: bytes) -> tuple[np.ndarray, float | None]:
    """Returns the image as BGR and its resolution in dpi if known (scans, PDF pages).

    Raises ValueError for anything that is not a supported image or PDF."""
    if data[:5] == b"%PDF-":
        page = pypdfium2.PdfDocument(data)[0]
        w, h = page.get_size()  # points
        dpi = min(PDF_DPI, PDF_MAX_SIDE * 72 / max(w, h))
        pil = page.render(scale=dpi / 72).to_pil().convert("RGB")
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), dpi
    try:
        pil = Image.open(io.BytesIO(data))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except Exception as e:  # PIL raises many different error types for broken or unknown files
        raise ValueError("Unsupported or damaged image file") from e
    dpi = pil.info.get("dpi", (0, 0))[0]
    # Phones store a meaningless 72 dpi; only scanner resolutions are useful.
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), float(dpi) if dpi and dpi >= 100 else None
