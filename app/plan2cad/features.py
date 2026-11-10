"""The steps of the recognition that can be switched on and off one by one: to measure what each brings, on every kind
of drawing, and to turn one off where it does not suit a kind of drawing or a collection of drawings. Job API:
``options.on`` and ``options.off``; command line and benchmark: ``--on`` and ``--off``. A kind of drawing's profile
can switch steps off for its kind ([steps] off, see for_kind); a step a job names in ``on`` stays on."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace


@dataclass(frozen=True)
class Features:
    freehand: bool = True  # black and grey marks by hand (notes, clouds, freehand hatching) kept apart: freehand.py
    # print taken for pencil no more: a stroke is light against print of its width on the sheet (a thin printed line
    # is grey in a scan), and specks of print a mark leaves are no notes (freehand._light, _pencil)
    pencil_guard: bool = True
    # marks drawn into the linework judged stroke by stroke (freehand._strokes); off, as on phone photos it takes
    # printed hatching and fixtures for clouds and marks
    freehand_strokes: bool = False
    # leaders from a mark and strokes through printed text are marks too (freehand._leaders); off, as no gain is
    # measured yet, and a printed line between two rows of text looks like a strike-out
    leaders: bool = False
    colour_guard: bool = True  # colour a camera adds (noise, lens fringes, the paper's shadow) is no pen: markup.py
    read_handwriting: bool = False  # read handwritten notes in colour with Tesseract, which misreads most of them
    review_handwriting: bool = False  # the job's reviewer model reads each handwritten note, as a reading to check
    vocabulary: bool = True  # read tokens in the vocabulary of the kind of drawing's profile: profiles.py
    # look-alike characters (0 and 8, B and 8, D and 0) decided per token where the token tells: the vote of look-alike
    # glyphs passes over a character both readers read alike, a token that is a word of the kind's vocabulary in one
    # of its readings only takes that one, and a vote without a clear majority marks the text instead of choosing
    # (ocr._voted, pipeline._settled)
    lookalike_tokens: bool = True
    dedupe: bool = True  # one text where the same was read twice at one place
    hatching: bool = True  # hatching read as a text line of "IIII" or "////" is traced as lines again: hatching.py
    # nearly parallel pieces of lines are merged where they are drawn, measured at the angle of the longest and each
    # merged line drawn at the angle of its own pieces; measured each at its own, they were merged millimetres off,
    # and dense hatching moved across the line it ends on (vectorize._where_drawn)
    merge_where_drawn: bool = True
    # a few pen widths, as many as the drawing has, instead of each line's own (objects.pens); off, as the widths
    # measured on the binary image of a scan are a pen too thick, and this merges pens rather than putting them right
    # (with width_calibration, to be measured again)
    pens: bool = False
    oblique_text: bool = True  # text lines at any other angle read along them, both ways round: ocr._oblique, spotting
    # vertical and oblique texts read both ways round, the reading in the kind's vocabulary taken (pipeline._settled);
    # short vertical lines, such as wire numbers, told by their characters (spotting._stands)
    text_direction: bool = True
    # where a pipe runs along a text line's box from beyond its end, that end is the bars of a flange on it, not letters
    # ("HH" before a line label); a single character with a pipe through it and on past it is such bars ("H"); a text
    # a pipe runs into at both ends may be a symbol read as letters ("Hoeat"): spotting.off_pipes, pipeline._bars
    text_on_pipes: bool = True
    # a single character alone in a frame read as the letter or digit it may be, as the kind's profile says ([reading]
    # framed); off, as the benchmark has no such characters to measure it on
    framed_letters: bool = False
    dimensions: bool = True  # dimension lines with their terminators, extension lines and value as dimension objects
    # the measured points of the dimensions put on the objects they dimension (the gap kept as the offset), and the
    # objects moved along each direction so that the dimensions measure their values exactly; a value that does not
    # fit its length or does not add up with the others along it moves nothing and is marked: fit.py
    dimension_fit: bool = True
    # the tag of a symbol inside its ring (an instrument bubble) only from a text inside it, never from one that runs
    # out of it, such as its number read with the bars of the flanges beside it ("T40201T"): symbols._attributes; and a
    # spotted line inside a drawn ring that runs a little out of it, onto the bars of a flange against it, is cut at the
    # ring and read again, since the template's ring may be larger than the one drawn: spotting.off_rings
    attributes_inside: bool = True
    # symbols matched on the ink before the texts are taken as read, and their areas left out of the text recognition
    # but for their attributes (pipeline._symbols_first); off, as on the benchmark it found a few more symbols on a
    # scan but read fewer texts and attributes on a photo, and it searches the symbols twice
    symbols_first: bool = False
    # linework that looks like a symbol but was recognised as none of the libraries', kept on a layer of its own and
    # listed: unclassified.py
    unclassified: bool = True
    # what a symbol needs around it: a thin one across its line (an orifice plate) the line ending at both its ports
    # and none running through it, which the two bars of a flange have; one drawn on a line that goes on (a flow arrow)
    # strokes of its own, as thick as the sheet's, not part of a larger figure (a valve, a cross on a capillary line)
    # (symbols._in_gap, symbols._own)
    symbol_context: bool = True
    # each symbol at the size the sheet draws it at: a symbol found often enough held against the size of its own
    # kind, not of all (many sheets draw their valves smaller than their bubbles), and a group of strokes that is no
    # symbol at its own size tried at the size a symbol was found at three times elsewhere (a valve between the bars of
    # two flanges makes one group with them, larger than it), where it explains the whole group but for such bars;
    # symbols placed so count towards no size (symbols._one_size, _at_sizes, _whole). Off, as kinds found often at one
    # size keep each other there: on a dense sheet with many small marks (the crosses of a capillary line, the zigzags
    # of a signal line drawn in the normal pen), false arrows at their own small size confirm that size, and the guard
    # on symbols placed at a size found elsewhere does not stop that; a switch for sheets that draw their valves much
    # smaller than their bubbles
    sheet_sizes: bool = False
    # short strokes across a line kept: the two pieces of a tick that taking out its line cut in two are one line again
    # (vectorize.bridge), and a stroke across other linework is no crumb (topology.join); off, as on the benchmark it
    # kept far more ticks but cost a few junctions, curves and lines within 0.5 mm on floor plans and sections
    short_strokes: bool = False
    title_block: bool = True  # the title block as a table of its ruled cells, its linework on a layer: titleblock.py
    paper_edge: bool = True  # the paper of a scan on a lid as grey as it, by its straight edges: sheet.on_lid
    letter_rows: bool = True  # the print size measured on shapes standing in rows of characters, not on specks of dirt
    # dashed lines told apart as dashed, dash-dot and dotted by the pattern of their dashes, dots and gaps; long dashes
    # with dots between them and rows of dots found as such lines (vectorize.line_types, dash_dots, dotted)
    linetypes: bool = True
    # pen widths measured on the grey image (ink across the stroke over the ink's darkness) instead of on the binary
    # one, which blur and the threshold widen (vectorize.calibrate_widths)
    width_calibration: bool = True
    # arcs of large radius traced as arcs, and smooth runs of short pieces drawn as one arc or a spline (objects.curves)
    splines: bool = True
    # straight segments that run nearly along a direction of the sheet (the axes, a turned building, a hatching) turned
    # to exactly it, their meeting points placed where the turned lines meet, so that lines drawn parallel or at right
    # angles are exactly so and stay connected (square.py)
    square_lines: bool = True
    asked: frozenset[str] = field(default=frozenset(), compare=False)  # the steps the job switched on by name

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls) if f.type == "bool"]

    @classmethod
    def of(cls, on: list[str] | tuple[str, ...] = (), off: list[str] | tuple[str, ...] = ()) -> Features:
        """The default features with ``on`` switched on and ``off`` switched off; ValueError for a name that is not a
        feature or is in both."""
        unknown = sorted((set(on) | set(off)) - set(cls.names()))
        if unknown:
            raise ValueError(f"unknown feature(s): {', '.join(unknown)} (known: {', '.join(cls.names())})")
        if both := sorted(set(on) & set(off)):
            raise ValueError(f"feature(s) both on and off: {', '.join(both)}")
        return replace(cls(), **{n: True for n in on}, **{n: False for n in off}, asked=frozenset(on))

    def for_kind(self, off: tuple[str, ...]) -> tuple[Features, list[str]]:
        """The features with the steps the profile of a kind of drawing switches ``off`` switched off, but for those
        the job switched on by name; and the steps so switched off, which were on."""
        taken = sorted({n for n in off if getattr(self, n) and n not in self.asked})
        return replace(self, **dict.fromkeys(taken, False)), taken
