"""Decodes uploads: photos (JPEG, PNG, TIFF, HEIC from iPhones) and the pages of PDF and TIFF scans."""

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


def page_count(data: bytes) -> int:
    """The pages of a PDF or a multi-page TIFF; 1 for other images (a HEIC may hold more, but they are not pages).

    Raises ValueError for anything that is not a supported image or PDF."""
    try:
        if data[:5] == b"%PDF-":
            return len(pypdfium2.PdfDocument(data))
        pil = Image.open(io.BytesIO(data))
        return pil.n_frames if pil.format == "TIFF" else 1
    except Exception as e:  # PIL and PDFium raise many different error types for broken or unknown files
        raise ValueError("Unsupported or damaged image file") from e


def load_image(data: bytes, page: int = 0) -> tuple[np.ndarray, float | None]:
    """Returns the image (of a PDF or TIFF: the page, counted from 0) as BGR and its resolution in dpi if known
    (scans, PDF pages).

    Raises ValueError for anything that is not a supported image or PDF."""
    if data[:5] == b"%PDF-":
        sheet = pypdfium2.PdfDocument(data)[page]
        w, h = sheet.get_size()  # points
        dpi = min(PDF_DPI, PDF_MAX_SIDE * 72 / max(w, h))
        pil = sheet.render(scale=dpi / 72).to_pil().convert("RGB")
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), dpi
    try:
        pil = Image.open(io.BytesIO(data))
        pil.seek(page)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except Exception as e:  # PIL raises many different error types for broken or unknown files
        raise ValueError("Unsupported or damaged image file") from e
    dpi = pil.info.get("dpi", (0, 0))[0]
    # Phones store a meaningless 72 dpi; only scanner resolutions are useful.
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), float(dpi) if dpi and dpi >= 100 else None
