"""Profiles: what depends on the kind of drawing, a standard or a customer, kept as data out of the code. A profile is a
TOML file (docs/profiles.md) naming a kind of drawing, the evidence its texts give of it and the vocabulary its texts
are written in. The kind of drawing chooses the symbol libraries (library.Library.drawing); the vocabulary helps to
read a token whose letters and digits the text recognition confused ("-KI" for "-K1" where device tags are "-K1"); its
[steps] switch off steps that do not suit the kind, such as dimensions on a P&ID, which has none. Built-in profiles
are in profiles/; more can be added with the setting PROFILES (folders)."""

from __future__ import annotations

import itertools
import math
import os
import re
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .model import Circle, Text

FORMAT = 1
BUILT_IN = Path(__file__).resolve().parents[2] / "profiles"
TITLE, SCALE = 5, 3  # evidence a title counts, and a drawing scale typical of the kind; each other match counts once
LEAST = 1  # evidence a kind needs at least
CONFUSED = {"O": "0", "0": "O", "I": "1", "l": "1", "1": "I", "S": "5", "5": "S", "B": "8", "8": "B", "Z": "2",
            "2": "Z", "G": "6", "6": "G"}  # what the text recognition reads for what
SWAPS = 4  # at most this many confused characters in a token are tried
FRAMED = {"any": None, "letter": str.isalpha, "digit": str.isdigit}  # what a single character in a frame is read as
# the steps (features.py) that run only once the kind of drawing is known, which its profile can switch off; the others
# run, at least in part, before the texts that tell the kind are read
BY_KIND = ("dimensions", "vocabulary", "framed_letters", "dedupe", "square_lines", "title_block", "unclassified",
           "width_calibration", "dimension_fit", "pens", "read_handwriting")


class ProfileError(ValueError):
    """The profile cannot be used; the message says why."""


@dataclass(frozen=True)
class Profile:
    id: str
    name: str
    drawing: str
    titles: tuple[re.Pattern, ...]
    patterns: tuple[re.Pattern, ...]
    scales: tuple[int, ...]
    bubbles: bool
    vocabulary: tuple[re.Pattern, ...]  # whole tokens
    framed: str = "any"  # a single character alone in a frame is a "letter", a "digit" or "any" of them (FRAMED)
    steps_off: tuple[str, ...] = ()  # steps (features.py) switched off on the kind unless a job switches them on

    def evidence(self, texts: list[Text], circles: list[Circle], scale: int | None) -> int:
        """How much the texts, circles and scale of a sheet speak for this kind of drawing."""
        read = [t.text for t in texts if not t.uncertain]
        joined = " ".join(read)
        score = TITLE * sum(bool(p.search(joined)) for p in self.titles)
        score += sum(len(p.findall(s)) for p in self.patterns for s in read)
        score += SCALE if scale in self.scales else 0
        if self.bubbles:
            score += sum(any(math.hypot(t.x - c.cx, t.y - c.cy) < c.r for t in texts) for c in circles if 3 <= c.r <= 8)
        return score

    def read(self, text: str) -> str:
        """The text with each token that matches no word of the vocabulary as it was read, but does with some of its
        confused characters swapped (the fewest), so swapped."""
        if not self.vocabulary:
            return text
        return " ".join(self._token(w) for w in text.split(" "))

    def words(self, text: str) -> int:
        """How many tokens of the text are words of the vocabulary, as read or with confused characters swapped."""
        return sum(any(p.fullmatch(self._token(w)) for p in self.vocabulary) for w in text.split())

    def word(self, token: str) -> bool:
        """Whether the token is a word of the vocabulary as it stands."""
        return any(p.fullmatch(token) for p in self.vocabulary)

    def alone(self, text: str) -> str:
        """A single character read alone in a frame, such as an interlock's "I" or an axis's "B": as the letter or the
        digit it may be where the kind frames only letters or only digits (``framed``), else as read."""
        kind = FRAMED[self.framed]
        return CONFUSED[text] if kind and len(text) == 1 and text in CONFUSED and not kind(text) else text

    def _token(self, word: str) -> str:
        if not word or any(p.fullmatch(word) for p in self.vocabulary):
            return word
        spots = [i for i, ch in enumerate(word) if ch in CONFUSED][:SWAPS]
        for n in range(1, len(spots) + 1):
            for chosen in itertools.combinations(spots, n):
                chars = list(word)
                for i in chosen:
                    chars[i] = CONFUSED[chars[i]]
                candidate = "".join(chars)
                if any(p.fullmatch(candidate) for p in self.vocabulary):
                    return candidate
        return word


def load(path: Path) -> Profile:
    """A profile from its TOML file; ProfileError if it is not one."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise ProfileError(f"{path.name}: {e}") from None
    if data.get("format") != FORMAT:
        raise ProfileError(f"{path.name}: format must be {FORMAT}")
    evidence, vocabulary = data.get("evidence", {}), data.get("vocabulary", {})
    reading = data.get("reading", {})
    framed = reading.get("framed", "any") if isinstance(reading, dict) else None
    if not isinstance(framed, str) or framed not in FRAMED:
        raise ProfileError(f"{path.name}: reading.framed must be one of {', '.join(FRAMED)}")
    steps = data.get("steps", {})
    off = steps.get("off", []) if isinstance(steps, dict) else None
    if not isinstance(off, list) or any(n not in BY_KIND for n in off):
        raise ProfileError(f"{path.name}: steps.off can list only {', '.join(BY_KIND)}")
    try:
        return Profile(str(data["id"]), str(data.get("name", data["id"])), str(data["drawing"]),
                       _compiled(evidence.get("titles", [])), _compiled(evidence.get("patterns", [])),
                       tuple(int(s) for s in evidence.get("scales", [])), bool(evidence.get("bubbles", False)),
                       _compiled(p for words in vocabulary.values() for p in words), framed, tuple(off))
    except (KeyError, TypeError, ValueError, re.error) as e:
        raise ProfileError(f"{path.name}: {e}") from None


def _compiled(patterns) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(p) for p in patterns)


@lru_cache(maxsize=8)
def installed(folders: str = "") -> tuple[Profile, ...]:
    """The built-in profiles and those in the os.pathsep-separated ``folders``; a later one replaces one of its id."""
    found: dict[str, Profile] = {}
    for folder in [BUILT_IN] + [Path(f) for f in folders.split(os.pathsep) if f]:
        for path in sorted(folder.glob("*.toml")):
            p = load(path)
            found[p.id] = p
    return tuple(found.values())


def recognise(texts: list[Text], circles: list[Circle], scale: int | None, profiles: tuple[Profile, ...]) -> str:
    """The kind of drawing the evidence speaks for most (LEAST at least; the earlier profile on a tie), "" if none."""
    best, kind = LEAST - 1, ""
    for p in profiles:
        if (score := p.evidence(texts, circles, scale)) > best:
            best, kind = score, p.drawing
    return kind
