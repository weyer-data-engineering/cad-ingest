"""Asks a vision model about what the recognition of a photo or scan is unsure of: the texts it could not read
reliably and the symbols that may be the wrong ones (see plan2cad.symbols). The model sits behind an OpenAI-compatible
endpoint, such as a LiteLLM proxy, that the job names (``reviewer``). It sees only a small crop around each of these
elements, with the reading or, for a symbol, the library's drawing of it, and answers from a closed set. An answer is
taken only where it fits what was recognised; everything else stays to check. With the step review_handwriting it
also reads the handwriting of each note added by hand, from the note's image: always a reading to check, never print.
Without a reviewer nothing leaves the service."""

import logging
import math
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np

from .plan2cad.library import place
from .plan2cad.markup import READING_LAYER, WORD
from .plan2cad.model import Drawing, Markup, Symbol, Text
from .plan2cad.pipeline import Result
from .plan2cad.render import render

log = logging.getLogger(__name__)
MAX_ITEMS = 40  # elements asked about per page, least confident first
WORKERS = 4  # questions at a time
TIMEOUT = httpx.Timeout(10.0, read=60.0)
MARGIN = 3.0  # mm of the drawing around a symbol in its crop
SIDE = 256  # px: crops are enlarged to at least this on their shorter side, so the model sees the strokes
LENGTH = (0.5, 2.0)  # a reading replaces one at least half and at most twice as long: the box it was found in
MAX_NOTES = 20  # handwritten notes read per page
NOTE_LENGTH = 200  # characters: a longer reading of a note is no reading of it
NOTE_SIDE = 1600  # px: a note's image is shrunk to at most this on its longer side, as a note can cover a large area
HANDWRITING = ("The image shows part of a printed technical drawing, faded, with a note added on it by hand, in colour "
               "or in black or grey. Transcribe the handwriting of the note exactly as written, on one line, and "
               "nothing else; do not transcribe the printed text. Reply with UNREADABLE if there is no handwriting "
               "or you cannot read it.")


def check(r: Result, reviewer: dict, handwriting: bool = False) -> tuple[dict, list[str]]:
    """Asks the reviewer about the elements to check on the page and takes the answers that fit: a text is confirmed
    or read anew, a symbol confirmed; with ``handwriting``, the handwriting of each note is read (_note). Returns the
    counts for the report and warnings."""
    d = r.drawing
    items = sorted([t for t in d.texts if t.uncertain] + [s for s in d.symbols if s.doubt],
                   key=lambda e: e.confidence)[:MAX_ITEMS]
    notes = [(m, c) for m, c in zip(d.markups, r.crops) if m.kind == "note"][:MAX_NOTES] if handwriting else []
    if not items and not notes:
        return {}, []
    with ThreadPoolExecutor(WORKERS) as pool:
        answers = list(pool.map(lambda e: _ask(e, r, reviewer), items))
        readings = list(pool.map(lambda n: _post(HANDWRITING, [_png(n[1], NOTE_SIDE)], reviewer), notes))
    counts = dict.fromkeys(("confirmed", "corrected", "kept", "unanswered") + (("read",) if notes else ()), 0)
    for e, answer in zip(items, answers):
        counts[_take(e, answer)] += 1
    for (m, _), answer in zip(notes, readings):
        counts[_note(m, answer, d)] += 1
    host = urlsplit(reviewer["base_url"]).hostname
    warnings = [f"The reviewer model at {host} did not answer about {counts['unanswered']} elements; they are left "
                "to check"] if counts["unanswered"] else []
    return {"model": reviewer["model"], "asked": len(items) + len(notes)} | counts, warnings


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
    return _post(prompt, images, reviewer)


def _post(prompt: str, images: list[bytes], reviewer: dict) -> str | None:
    """The reviewer's answer to a question about images (PNG), or None if it gave none."""
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


def _note(m: Markup, answer: str | None, d: Drawing) -> str:
    """Takes the reviewer's reading of the handwriting of a note where it is one line: on the note, and as a text on
    the layer of readings in the note's colour, marked to check, as it is a reading of handwriting, never print.
    Returns what came of it."""
    if answer is None:
        return "unanswered"
    reading = answer.strip().strip('"“”').strip()
    if not reading or "\n" in reading or reading.upper().startswith("UNREADABLE") or len(reading) > NOTE_LENGTH:
        return "kept"
    m.reading, m.read_by = reading, "reviewer"
    x0, y0, x1, y1 = m.box
    d.texts.append(Text((x0 + x1) / 2, (y0 + y1) / 2, max(1.0, min(WORD, y1 - y0) / 2), reading, uncertain=True,
                        color=m.color, layer=READING_LAYER))
    return "read"


def _text_crop(t: Text, r: Result) -> bytes:
    """The text with some room around it, turned to read from left to right."""
    along = (t.length or 0.8 * t.height * len(t.text)) / 2 + 2 * t.height
    across = 1.5 * t.height
    if t.angle % 90:  # an oblique text: cut out along it, as a box cut by quarter turns would clip it
        m = cv2.getRotationMatrix2D((t.x * r.ppm, t.y * r.ppm), -t.angle, 1.0)  # turned clockwise by its angle
        m[:, 2] += (along - t.x) * r.ppm, (across - t.y) * r.ppm  # its centre to the middle of the crop
        size = (max(1, round(2 * along * r.ppm)), max(1, round(2 * across * r.ppm)))
        return _png(cv2.warpAffine(r.image, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    turns = round(t.angle / 90) % 4
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


def _png(a: np.ndarray, longest: int | None = None) -> bytes:
    """A crop as PNG, enlarged to SIDE on its shorter side, and shrunk to ``longest`` on its longer side if given."""
    f = max(1.0, SIDE / max(1, min(a.shape[:2])))
    if longest:
        f = min(f, longest / max(a.shape[:2]))
    a = cv2.resize(np.ascontiguousarray(a), None, fx=f, fy=f,
                   interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    return cv2.imencode(".png", a)[1].tobytes()
