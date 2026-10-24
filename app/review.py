"""Asks a vision model about what the recognition of a photo or scan is unsure of: the texts it could not read
reliably and the symbols that may be the wrong ones (see plan2cad.symbols). The model sits behind an OpenAI-compatible
endpoint, such as a LiteLLM proxy, that the job names (``reviewer``). It sees only a small crop around each of these
elements, with the reading or, for a symbol, the library's drawing of it, and answers from a closed set. An answer is
taken only where it fits what was recognised; everything else stays to check. Without a reviewer nothing leaves the
service."""

import logging
import math
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np

from .plan2cad.library import place
from .plan2cad.model import Drawing, Symbol, Text
from .plan2cad.pipeline import Result
from .plan2cad.render import render

log = logging.getLogger(__name__)
MAX_ITEMS = 40  # elements asked about per page, least confident first
WORKERS = 4  # questions at a time
TIMEOUT = httpx.Timeout(10.0, read=60.0)
MARGIN = 3.0  # mm of the drawing around a symbol in its crop
SIDE = 256  # px: crops are enlarged to at least this on their shorter side, so the model sees the strokes
LENGTH = (0.5, 2.0)  # a reading replaces one at least half and at most twice as long: the box it was found in


def check(r: Result, reviewer: dict) -> tuple[dict, list[str]]:
    """Asks the reviewer about the elements to check on the page and takes the answers that fit: a text is confirmed
    or read anew, a symbol confirmed. Returns the counts for the report and warnings."""
    d = r.drawing
    items = sorted([t for t in d.texts if t.uncertain] + [s for s in d.symbols if s.doubt],
                   key=lambda e: e.confidence)[:MAX_ITEMS]
    if not items:
        return {}, []
    with ThreadPoolExecutor(WORKERS) as pool:
        answers = list(pool.map(lambda e: _ask(e, r, reviewer), items))
    counts = dict.fromkeys(("confirmed", "corrected", "kept", "unanswered"), 0)
    for e, answer in zip(items, answers):
        counts[_take(e, answer)] += 1
    host = urlsplit(reviewer["base_url"]).hostname
    warnings = [f"The reviewer model at {host} did not answer about {counts['unanswered']} elements; they are left "
                "to check"] if counts["unanswered"] else []
    return {"model": reviewer["model"], "asked": len(items)} | counts, warnings


def _ask(e: Text | Symbol, r: Result, reviewer: dict) -> str | None:
    """The reviewer's answer about one element, or None if it gave none."""
    if isinstance(e, Text):
        images = [_text_crop(e, r)]
        prompt = (f'The image is a crop of a scanned technical drawing with one line of text in its middle. A text '
                  f'recognition read it as "{e.text}". Reply with that text exactly as printed, and nothing else, or '
                  f'with UNREADABLE if you cannot read it.')
    else:
        x0, y0, x1, y1 = _extent(e)
        images = [_crop(r, x0 - MARGIN, y0 - MARGIN, x1 + MARGIN, y1 + MARGIN), _reference(e, r.ppm)]
        prompt = (f'The first image is a crop of a scanned technical drawing. The second image is the symbol '
                  f'"{e.template.description}" of a symbol library. Is the symbol in the middle of the first image '
                  f'this symbol? It may be drawn with other line widths. Reply with one word: YES, NO or UNSURE.')
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64encode(png).decode()}}
        for png in images]
    headers = (reviewer.get("headers") or {}) | {"Authorization": f"Bearer {reviewer['api_key']}"}
    try:
        response = httpx.post(reviewer["base_url"].rstrip("/") + "/chat/completions", timeout=TIMEOUT, headers=headers,
                              json={"model": reviewer["model"], "temperature": 0, "max_tokens": 100,
                                    "messages": [{"role": "user", "content": content}]})
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, ValueError, LookupError, TypeError, AttributeError) as error:
        log.warning("Reviewer at %s: %s", urlsplit(reviewer["base_url"]).hostname, type(error).__name__)
        return None


def _take(e: Text | Symbol, answer: str | None) -> str:
    """Takes the answer where it fits; returns what came of it."""
    if answer is None:
        return "unanswered"
    if isinstance(e, Symbol):
        if answer.upper().startswith("YES"):
            e.doubt = None
            return "confirmed"
        if answer.upper().startswith("NO"):
            e.doubt = f"the reviewer model sees no {e.template.description.lower()} here"
        return "kept"
    reading = answer.strip('"')
    if "\n" in reading or reading.upper() == "UNREADABLE" or \
            not LENGTH[0] <= len(reading) / max(1, len(e.text)) <= LENGTH[1]:
        return "kept"
    e.uncertain = False
    if reading == e.text:
        return "confirmed"
    e.text = reading
    return "corrected"


def _text_crop(t: Text, r: Result) -> bytes:
    """The text with some room around it, turned to read from left to right."""
    turns = round(t.angle / 90) % 4
    along = (t.length or 0.8 * t.height * len(t.text)) / 2 + 2 * t.height
    across = 1.5 * t.height
    w, h = (across, along) if turns % 2 else (along, across)
    return _crop(r, t.x - w, t.y - h, t.x + w, t.y + h, turns)


def _crop(r: Result, x0: float, y0: float, x1: float, y1: float, turns: int = 0) -> bytes:
    """A part of the sheet (mm) as PNG, turned back by quarter turns (counter-clockwise on screen)."""
    h, w = r.image.shape[:2]
    a = r.image[max(0, math.floor(y0 * r.ppm)):min(h, math.ceil(y1 * r.ppm)),
                max(0, math.floor(x0 * r.ppm)):min(w, math.ceil(x1 * r.ppm))]
    return _png(np.rot90(a, -turns))


def _reference(s: Symbol, ppm: float) -> bytes:
    """The library's drawing of the symbol, turned and sized as it was placed."""
    q = place(s.template, 0, 0, s.angle, s.mirror, s.scale)
    x0, y0, x1, y1 = _extent(q)
    q = place(s.template, MARGIN - x0, MARGIN - y0, s.angle, s.mirror, s.scale)
    return _png(render(Drawing(x1 - x0 + 2 * MARGIN, y1 - y0 + 2 * MARGIN, symbols=[q]), ppm))


def _extent(s: Symbol) -> tuple[float, float, float, float]:
    f = Drawing(0, 0, symbols=[s]).flat()
    xs = [v for q in f.lines for v in (q.x1, q.x2)] + [q.cx + k * q.r for q in f.arcs + f.circles for k in (-1, 1)]
    ys = [v for q in f.lines for v in (q.y1, q.y2)] + [q.cy + k * q.r for q in f.arcs + f.circles for k in (-1, 1)]
    return min(xs), min(ys), max(xs), max(ys)


def _png(a: np.ndarray) -> bytes:
    f = max(1.0, SIDE / max(1, min(a.shape[:2])))
    a = cv2.resize(np.ascontiguousarray(a), None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return cv2.imencode(".png", a)[1].tobytes()
