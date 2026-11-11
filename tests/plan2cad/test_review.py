"""The reviewer model: asked about the texts and symbols to check, its answers taken where they fit."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from app import review, scan
from app.plan2cad import markup
from app.plan2cad.library import load, place
from app.plan2cad.model import Drawing, Line, Markup, Text
from app.plan2cad.pipeline import Result
from app.plan2cad.render import render

from ..conftest import free_port

LIBRARY = load()


@pytest.fixture
def model():
    """An OpenAI-compatible endpoint that answers from a script and keeps the requests."""
    state = {"answers": [], "status": 200, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, self.headers["Authorization"], body))
            state["customer"] = self.headers.get("x-litellm-customer-id")
            prompt = body["messages"][0]["content"][0]["text"]
            answer = next(a for key, a in state["answers"] if key in prompt)
            data = json.dumps({"choices": [{"message": {"role": "assistant", "content": answer}}]}).encode()
            self.send_response(state["status"])
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", free_port()), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["reviewer"] = {"base_url": f"http://127.0.0.1:{server.server_port}/v1", "api_key": "sk-test", "model": "m",
                         "headers": {"x-litellm-customer-id": "tenant:someone"}}
    yield state
    server.shutdown()


def page() -> Result:
    """A sheet with a gate valve and a pump on a pipe and a tag; the valve and the tag are to check."""
    valve, pump = place(LIBRARY["GATE_VALVE"], 60, 50), place(LIBRARY["PUMP"], 120, 50)
    tag = Text(60, 40, 3, "V-1O1", confidence=40.0, uncertain=True)
    sheet = Drawing(200, 100, lines=[Line(20, 50, 57.5, 50, 0.5), Line(62.5, 50, 113.5, 50, 0.5)],
                    texts=[Text(60, 40, 3, "V-101")], symbols=[valve, pump])
    valve.confidence, valve.doubt = 85.0, "only 85 % of its outline found"
    return Result(Drawing(200, 100, symbols=[valve, pump], texts=[tag]), render(sheet, 8), 8)


def test_the_reviewer_reads_the_text_and_confirms_the_symbol(model):
    model["answers"] = [("text recognition", "V-101"), ("Gate valve", "YES")]
    r = page()
    counts, warnings = review.check(r, model["reviewer"])
    assert counts == {"model": "m", "asked": 2, "confirmed": 1, "corrected": 1, "kept": 0, "unanswered": 0}
    assert not warnings
    assert [(t.text, t.uncertain) for t in r.drawing.texts] == [("V-101", False)]
    assert [s.doubt for s in r.drawing.symbols] == [None, None]
    assert {(path, auth) for path, auth, _ in model["requests"]} == {("/v1/chat/completions", "Bearer sk-test")}
    assert model["customer"] == "tenant:someone"
    images = sorted(sum(c["type"] == "image_url" for c in body["messages"][0]["content"])
                    for _, _, body in model["requests"])
    assert images == [1, 2]  # the text's crop; the symbol's crop and its drawing in the library, nothing else


def test_the_user_of_the_reviewer_goes_with_every_call_and_is_left_out_without_one(model):
    model["answers"] = [("text recognition", "V-101"), ("Gate valve", "YES")]
    review.check(page(), model["reviewer"])
    assert all("user" not in body for _, _, body in model["requests"])
    model["requests"].clear()
    review.check(page(), model["reviewer"] | {"user": "tenant-1"})
    assert [body["user"] for _, _, body in model["requests"]] == ["tenant-1", "tenant-1"]


def test_answers_that_do_not_fit_leave_the_elements_to_check(model):
    model["answers"] = [("text recognition", "The text in the middle of the image reads V-101"), ("Gate valve", "NO")]
    r = page()
    counts, _ = review.check(r, model["reviewer"])
    assert counts["kept"] == 2
    assert [(t.text, t.uncertain) for t in r.drawing.texts] == [("V-1O1", True)]  # too long for its place
    assert r.drawing.symbols[0].doubt == "the reviewer model sees no gate valve here"


def test_a_reviewer_that_fails_leaves_everything_to_check_and_names_only_its_host(model):
    model["answers"], model["status"] = [("", "")], 500
    r = page()
    counts, warnings = review.check(r, model["reviewer"])
    assert counts["unanswered"] == 2 and r.drawing.texts[0].uncertain and r.drawing.symbols[0].doubt
    assert warnings == ["The reviewer model at 127.0.0.1 did not answer about 2 elements; they are left to check"]


def test_nothing_is_asked_when_nothing_is_to_check(model):
    r = page()
    r.drawing.texts[0].uncertain, r.drawing.symbols[0].doubt = False, None
    assert review.check(r, model["reviewer"]) == ({}, []) and not model["requests"]


def test_an_oblique_text_to_check_is_cut_out_along_it():
    import cv2
    import numpy as np

    street = Text(60, 40, 3.0, "Lindenstraße", 45, confidence=60.0, uncertain=True)
    sheet = render(Drawing(120, 80, texts=[street]), 8)
    crop = cv2.imdecode(np.frombuffer(review._text_crop(street, Result(Drawing(120, 80), sheet, 8)), np.uint8),
                        cv2.IMREAD_GRAYSCALE)
    dark = np.argwhere(crop < 128)
    assert crop.shape[1] > 3 * crop.shape[0]  # as long as the text and a little more, as high as a line
    assert dark[:, 1].min() > 0.05 * crop.shape[1] and dark[:, 1].max() < 0.95 * crop.shape[1]  # whole, not clipped


def noted() -> Result:
    """A sheet with a note added by hand in red and its image, and a printed text that is read reliably."""
    note = Markup("note", (40.0, 20.0, 80.0, 30.0), (205, 40, 40))
    crop = np.full((120, 400, 3), 230, np.uint8)
    cv2.putText(crop, "removed", (20, 80), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (40, 40, 205), 3)
    d = Drawing(200, 100, texts=[Text(60, 60, 3, "P-101")], markups=[note])
    return Result(d, np.full((800, 1600), 255, np.uint8), 8, crops=[crop])


def test_the_reviewer_reads_a_handwritten_note_as_a_reading_to_check(model):
    model["answers"] = [("handwriting", "removed")]
    r = noted()
    counts, warnings = review.check(r, model["reviewer"], handwriting=True)
    assert counts == {"model": "m", "asked": 1, "confirmed": 0, "corrected": 0, "kept": 0, "unanswered": 0, "read": 1}
    note = r.drawing.markups[0]
    assert (note.reading, note.read_by) == ("removed", "reviewer") and not warnings
    reading = next(t for t in r.drawing.texts if t.text == "removed")
    assert reading.uncertain and reading.layer == markup.READING_LAYER and reading.color == note.color
    assert "read by the reviewer model, to check" in scan._markup_line(note)
    assert scan._report(r.drawing)["markups"][0]["read_by"] == "reviewer"


def test_without_the_step_handwriting_is_not_sent_to_the_reviewer(model):
    r = noted()
    assert review.check(r, model["reviewer"]) == ({}, []) and not model["requests"]
    assert r.drawing.markups[0].reading is None


@pytest.mark.parametrize("answer", ["UNREADABLE", "Unreadable.", "UNREADABLE - the strokes are too faint"])
def test_a_note_the_reviewer_cannot_read_keeps_no_reading(model, answer):
    model["answers"] = [("handwriting", answer)]
    r = noted()
    counts, _ = review.check(r, model["reviewer"], handwriting=True)
    assert counts["kept"] == 1 and r.drawing.markups[0].reading is None and len(r.drawing.texts) == 1


def test_a_large_note_is_sent_shrunk_and_a_small_one_enlarged():
    large, small = (cv2.imdecode(np.frombuffer(review._png(np.zeros(shape, np.uint8), review.NOTE_SIDE), np.uint8),
                                 cv2.IMREAD_UNCHANGED).shape[:2] for shape in ((3000, 6000, 3), (60, 200, 3)))
    assert max(large) == review.NOTE_SIDE and min(small) == review.SIDE
