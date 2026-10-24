"""The reviewer model: asked about the texts and symbols to check, its answers taken where they fit."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app import review
from app.plan2cad.library import load, place
from app.plan2cad.model import Drawing, Line, Text
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
