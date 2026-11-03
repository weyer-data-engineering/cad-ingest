"""The steps of the recognition that can be switched on and off one by one: to measure what each brings, on every kind
of drawing, and to turn one off where it does not suit a kind of drawing or a customer's sheets. Job API:
``options.on`` and ``options.off``; command line and benchmark: ``--on`` and ``--off``."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class Features:
    freehand: bool = True  # black and grey marks by hand (notes, clouds, freehand hatching) kept apart: freehand.py
    read_handwriting: bool = False  # read handwritten notes in colour with Tesseract, which misreads most of them
    vocabulary: bool = True  # read tokens in the vocabulary of the kind of drawing's profile: profiles.py
    dedupe: bool = True  # one text where the same was read twice at one place
    pens: bool = True  # a few pen widths, as many as the drawing has, instead of each line's own: objects.pens

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def of(cls, on: list[str] | tuple[str, ...] = (), off: list[str] | tuple[str, ...] = ()) -> Features:
        """The default features with ``on`` switched on and ``off`` switched off; ValueError for a name that is not a
        feature or is in both."""
        unknown = sorted((set(on) | set(off)) - set(cls.names()))
        if unknown:
            raise ValueError(f"unknown feature(s): {', '.join(unknown)} (known: {', '.join(cls.names())})")
        if both := sorted(set(on) & set(off)):
            raise ValueError(f"feature(s) both on and off: {', '.join(both)}")
        return replace(cls(), **{n: True for n in on}, **{n: False for n in off})
