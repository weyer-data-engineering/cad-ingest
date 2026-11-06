# Changelog

All notable changes to cad-ingest are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/): the job API stays compatible within a major version.

## [Unreleased]

### Added

- Texts on photos and scans are read at any angle, not only horizontally and vertically (step `oblique_text`). A line of
  at least three characters on a straight line, such as a street name, a parcel number or a height along a contour, is
  read along its line, both ways round. It counts as turned only where it runs further off the axes than its letters can
  tell (3 degrees for a long line, about 8 for a short word), so a short word leaning on a phone photo stays level, and
  a level text in a box that leans for another reason, such as a level mark beside it, keeps the reading it was found
  with unless the reading along the box is clearly surer. It keeps its angle in `report.json` (any value from -180 to
  180; before only 0 or 90) and in the DXF. The reviewer model sees an oblique text to check cut out along its angle.
- Vertical and oblique texts are read both ways round (step `text_direction`). Once the kind of drawing is known, the
  reading made of words of its vocabulary is taken ("0,60", not "09'0"), and a near-level line is read upside down only
  so, not on confidence alone. A vertical text the text spotter was unsure of, which read again only reads the same more
  surely, stays marked to check unless each of its words is in the vocabulary ("3.94" for "3,94" on a building drawing).
  Short vertical lines, such as wire numbers, are told by their characters standing one above the other, and a
  horizontal tag at the end of a vertical label stays a text of its own. The direction classifier of the text spotter,
  which turned a short horizontal line upside down now and then ("L+" read as "+7"), is no longer asked.
- Profiles can say what a single character alone in a frame holds: `[reading] framed` is `letter`, `digit` or `any` (the
  default). With the step `framed_letters` (off by default) such a character is read as the letter or the digit it may
  be (O or 0, I or 1, S or 5, B or 8, Z or 2, G or 6). The built-in profiles all say `any`, as their drawings frame
  both.
- Dimension chains of floor plans and sections on photos and scans become linear DXF dimensions (step `dimensions`). A
  number along a thin line, between two lines at right angles that meet it, becomes a dimension on the layer
  `P2C-DIMENSION` with its value as read as its text. Each end needs a terminator on the image: an oblique tick (a slim
  stroke, so the dot where wires join is none), an arrowhead (filled or open, slim, also from outside) or a dot. A wall
  between the number and its dimension line does not hide it. The dimensions of a chain share their extension lines,
  their strokes and values leave the loose lines and texts, and a line that ends at a dimension line or an extension
  line is joined to it as CAD draws it. The strokes still count for the pens of the drawing.
  - A dimension whose value is read unreliably, is missing, does not fit its length at the drawing's scale or is read
    with another decimal separator than the other numbers of the sheet is marked to check: it is on
    `P2C-DIMENSION-CHECK`, magenta in `overlay.png`, in `review` with the kind `dimension` and counted in a job warning.
  - `report.json` has `dimensions` (`value`, `ends_mm`, `measured_mm`, `length_mm`, `angle`, `terminator`, `uncertain`,
    `doubt`) and `counts.dimensions`; the values stay in `texts`. The markdown has the sections `Dimensions` and
    `Dimensions to check (uncertain recognition)`.
- The dimensions of a converted plan dimension its objects and are right (step `dimension_fit`, on by default). Each
  measured point lies on the object its extension line points at (a wall face, a corner, the jamb of an opening, or the
  inner wall a chain dimensions through the outer wall), and the gap to the start of the extension line is the
  dimension's offset (DIMEXO), so the DXF looks as printed. The objects are moved along each direction so that every
  dimension measures its value exactly at the drawing's scale; lines that end on them follow, so the drawing stays
  connected. A value that does not fit its length moves nothing, and one that does not add up with the others along it,
  such as an overall dimension that is not the sum of its chain, is marked to check. `report.json` has `offset_mm` for
  each dimension. The benchmark measures the dimensions true to their value and on the object.
- A profile can switch steps of the recognition off for its kind of drawing (`[steps] off`, `docs/profiles.md`). The
  P&ID and circuit diagram profiles switch `dimensions` off: they carry no dimension chains, and a line number between
  two flanges was taken for a dimension. A job that names the step in `options.on` keeps it on. `report.json` lists the
  steps the kind switched off in `steps_off_by_kind`, and the markdown says so.
- The title block of a photo or scan becomes a table of its ruled cells, found in the corner where the drawing frame's
  right and bottom lines meet, without assuming a layout (step `title_block`). `report.json` has `title_block` (its box,
  the rows and columns of its grid, and each cell with its row, column, spans, box, text and whether it is to check),
  the markdown has a section `Title block`, and the DXF has the block's lines on the new layer `P2C-TITLEBLOCK`.
- Linework that looks like a symbol but was recognised as none of the libraries' stays linework and goes on the new
  layer `P2C-SYMBOL-UNCLASSIFIED` (step `unclassified`): a compact cluster of symbol size, at least as high as the
  sheet's usual text, with three or more strokes and a closed shape at least a fifth of its width across. Texts and
  their frames, walls, doors, equipment, the frame, signal lines, the title block and the symbols placed are none, and
  neither are rows of letters that were not read, the corners of hatching or freehand markups. `report.json` lists each
  under `unclassified` (`box_mm`, `why`), the markdown has a section
  `Symbols not classified (no library symbol recognised)`, and a warning gives their number. Such a place is a symbol
  the libraries lack, or one that recognition missed.
- The step `symbols_first` (off by default) matches the symbols of the libraries on the ink before the texts are taken
  as read, and leaves their areas, but for their attributes, out of the rest of the text recognition. On the benchmark
  it found a few more symbols on a scan but read fewer texts and attributes on a photo, and it searches the symbols
  twice.
- A symbol found three times or more can be held against the size of its own kind instead of that of all the symbols of
  the sheet, and groups of strokes tried at the sizes symbols were found at, such as a valve standing between two
  flanges (step `sheet_sizes`, off by default). Only a size found three times at the symbol's own size is tried, the
  symbol must explain at least 80 % of the strokes joined to it but for bars across its line (flanges), and symbols
  placed so count towards no size: a reducer on one triangle of a valve or a flow arrow on half the cross of a capillary
  line is no symbol. The step is off, as kinds found often at one size keep each other there: on a dense sheet with many
  small marks (the crosses of a capillary line, the zigzags of a signal line drawn in the normal pen), false arrows at
  their own small size confirm that size. Switch it on for sheets that draw their valves much smaller than their
  bubbles.
- The step `short_strokes` (off by default) keeps a short stroke across a line, such as a dimension tick or a flange
  mark, instead of dropping it as a crumb, and joins its two pieces where taking out the line cut it in two. On the
  benchmark it cost a few junctions, curves drawn as arcs and lines within 0.5 mm on floor plans and sections.
- Lines that are not continuous are told apart as dashed, dash-dot and dotted by the dashes, dots and gaps measured
  along them, with the length of one repeat (step `linetypes`). Long dashes with a dot between them, such as axes,
  become one dash-dot line, and rows of small dots one dotted line; fields and grids of dots, such as hatches, and rows
  of uneven specks are no lines. Before, such dots were lost and axes fell apart into loose dashes. In the DXF these
  lines have the linetypes `DASHED`, `DASHDOT` and `DOT`, each scaled to the pattern measured on it, on `P2C-DASH` or
  the layer of their kind; with the step off they keep `P2C_DASH`, as before. The preview draws all three.
- Curves come out as curves (step `splines`). Arcs of up to 1.5 m radius on paper, such as streets on site plans, are
  traced as arcs. A polyline that turns smoothly, by 8 degrees or more in all, becomes one arc where it lies on one
  circle, passing through the ends at which it is joined to other linework where a circle through them fits as well. A
  free curve traced as short straight pieces, such as a contour line, becomes a spline (DXF `SPLINE` through fit points
  on `P2C-CURVE`). True arcs, long straight pieces and a straight line traced with a jog where another line crosses it
  stay as they are.
- Lines drawn parallel or at right angles come out exactly so in the DXF of a photo or scan (step `square_lines`). A
  straight segment of a line, a polyline or a dimension whose direction lies within what the noise of its ends allows
  (0.5 degrees for a long segment to 5 for a short one) of the axes or of a direction that several segments of the
  sheet share, with its right angle (the sides of a building turned on a site plan, a hatching), is turned to exactly
  it. Where segments meet, the meeting point is placed where the turned lines meet, so corners stay closed and Ts stay
  on their line; a symbol moves onto the pipes at its ports, without turning. Pieces of one line a fraction of a
  millimetre apart get one offset. No point moves more than 0.5 mm. A single oblique line, the chords of a curve,
  texts, arcs and symbols are not turned. A profile can switch the step off for its kind (`[steps] off`), a job with
  `options.off`. On the benchmark (two or three cases of each kind), the straight segments along the axes come out
  exactly so in 97-100 % instead of 70-96 %, and pairs of lines parallel or at right angles in 90-100 % instead of
  8-93 % (off the axes, on sections and site plans, 90-100 % instead of 2-8 %); the lines found and their deviation,
  junctions, connections and symbols stay as they were.
- Marks by hand drawn into the drawing's own linework (a cloud across a wall, a note on a pipe) can be found stroke by
  stroke (step `freehand_strokes`, off by default: on phone photos it still takes printed hatching and fixtures for
  marks). The network is traced into strokes between its junctions, each stroke is judged by the rules of a mark, more
  strictly, and only the hand strokes go to `P2C-MARKUP`, at their width; the print they cross stays.
- Leaders and strike-outs by hand can go to the markup layer (step `leaders`, off by default until the benchmark shows a
  gain). A leader is a stroke from a mark found whole that ends in an arrowhead. A strike-out is a thin, nearly straight
  stroke through the middle of a printed text line, on a sheet with other marks; its letters stay to be read. Underlined
  texts, and printed lines that run on past a text, are left alone.
- A job's reviewer model can read the handwriting of each note from its image (step `review_handwriting`, off by
  default). The reading is a text to check on `P2C-MARKUP-TEXT`, never print. Each markup in `report.json` has `read_by`
  (`tesseract`, `reviewer` or `null`), `reviewer.read` counts the notes read, and the markdown says that the reviewer
  model read the note. An answer that starts with UNREADABLE is no reading.
- A starter library `building-symbols` with the marks of floor plans, sections and site plans: north arrow, axis (with
  its letter or number) and section marker (with its letter), drawn at their printed size. `DEFAULT_SYMBOL_LIBRARIES` is
  now `din-10628,floor-plan,iec-60617,building-symbols`.
- `iec-60617` 1.1.0 has a terminal, a circuit breaker, a motor protection switch, a push button and protective earth,
  each with ports and a `TAG`.
- `library.toml` may give `text_height`, the lettering that a library's printed-size symbols go with (default 2.0 mm).
  The sheet's lettering is compared with it to tell a sheet printed smaller or larger; `building-symbols` has 2.5 mm, as
  building drawings are lettered larger than P&IDs.
- The benchmark saves each conversion to `bench/out/pred` (output drawing, info, reference and runtime), and
  `python bench/run.py --reevaluate` measures a saved run again with the current `bench/evaluate.py` without converting
  anything, so a new or corrected measure applies to an earlier run. `--seeds N` runs N variants of each case, for
  steadier numbers.
- New benchmark cases: a P&ID A2 whose library lacks the check valve and the reducer; sections whose dimensions end in
  arrowheads (scan) and in dots (photo); a circuit diagram and a site plan scanned poorly on a scanner lid as grey as
  the paper; and a P&ID A3 in ISA style with the details that printed P&IDs carry, lettered with slashed noughts:
  flanges at valves and inline instrument bubbles, valves standing between flanges, line labels in rounded frames,
  electric and capillary signal lines with their marks, interlock diamonds, spec breaks and a legend under a double
  rule. It is scanned at 300 dpi and photographed, scanned printed on A4 (its valves at about half the library size, its
  signal lines and their marks in a pen rather than a hairline, small arrowheads where signal lines end), and scanned
  with its bubbles drawn a little smaller than the library's and their flanges against the ring.
- The drawings of the benchmark carry the new symbols (orifice plates on P&IDs; circuit breakers, motor protection
  switches, push buttons, terminals and protective earth on circuit diagrams; north arrows, section markers and axes on
  building drawings and site plans) and line types (dash-dot axes and plot boundaries, the ground as it was dotted).
  Their references hold their dimensions, the cells of their title blocks and, on sections, the level marks as symbols
  of no library.
- New benchmark measures: special characters exact (`special_tokens_exact`); markups by part (note, cloud, hatching,
  leader, strike-out) and the length on a markup layer that is no markup, also on drawings without markups; line types
  right (`linetype_correct`, `linetype_precision`); curves drawn as splines (`curves_as_splines`); dimensions found,
  correct, marked and false (`dimension_recall`, `dimension_precision`, `dimension_review`, `dimension_false`); phantom
  texts on symbols; marks of symbols not classified on missed symbols and elsewhere, and the share of missed symbols
  marked; short strokes kept; the cells of the title block found and read exactly; the print size (`printed_ok`);
  openings in walls kept (`openings_kept`); and lines along the axes exactly so (`axis_exact`) and pairs of lines
  parallel or at right angles exactly so, of all and of those off the axes (`square_pairs`, `oblique_pairs`).
- Marks by hand in black or pencil (notes, revision clouds, freehand hatching) are kept apart from the drawing, on
  `P2C-MARKUP` and in `markups`, before lines and texts are read: told apart by their shape (neither straight nor an arc
  nor an ellipse between their corners, or turning too often for anything else; chains of arcs for clouds) and their
  grey (pencil), with rules that hold on any kind of drawing (step `freehand`, `docs/photos-and-scans.md#markups`).
- Steps of the recognition can be switched on and off by name: job API `options.on` and `options.off` (`400` for an
  unknown name), command line and benchmark `--on` and `--off`. Each entry here names its step;
  `docs/photos-and-scans.md#steps-that-can-be-switched` lists them all, with their defaults.
- Profiles of kinds of drawing as data (`profiles/`, `docs/profiles.md`): the evidence that recognises the kind and the
  vocabulary its texts are written in, for P&IDs, building drawings, site plans and circuit diagrams. New kinds
  `electrical` and `site` for `options.drawing`; new setting `PROFILES` for further profile folders. A token read with
  letters for digits ("-KI") is read in the vocabulary where that makes it a word of it ("-K1"; step `vocabulary`).
- A starter library of circuit diagram symbols after IEC 60617 (`iec-60617`), used on circuit diagrams and one of the
  default libraries (`DEFAULT_SYMBOL_LIBRARIES`).
- A text read twice at one place is kept once (step `dedupe`). The step `pens` (off by default) gives the linework a few
  pens, as many as the drawing has.
- The benchmark has circuit diagrams, building sections and site plans, as clean plots, scans, poor and skewed scans
  and photos, and cases with markups by hand in black, pencil and colour. New measures: tokens exact, the share marked
  to check, areas where output and reference differ, symbols placed where there are none, markups found, correct and
  taken for the drawing, phantom texts and fills on markups, and crossings and junctions kept; and a table by kind of
  drawing.
- Symbol libraries as add-ons (see `docs/symbol-libraries.md`). A library is a folder with a `library.toml` and a
  DXF, the same folder as a ZIP, or a bare DXF; units come from `$INSUNITS`, blocks are read around their base point,
  and splines, ellipses and nested blocks count towards a symbol's outline. The result DXF holds the library's
  original blocks.
  - Workers read the built-in starter libraries and those in `SYMBOL_LIBRARIES` when they start and stop on a broken
    one; `DEFAULT_SYMBOL_LIBRARIES` sets the libraries a job gets when it names none.
  - Job API: `options.libraries` chooses installed libraries (`400` for unknown ones), `library_urls` brings libraries
    along with a job, `GET /libraries` lists the installed ones. New error codes `invalid_library` and
    `library_unavailable`; new setting `MAX_LIBRARY_SIZE_BYTES`.
  - `report.json` lists the libraries used and, for every symbol, its library.
  - Command line: `--library` (an installed id or a path, repeatable), `cad-ingest library check` (format, notes and
    a self-test on simulated scans and photos) and `cad-ingest library build` (from DWG and DXF files).
- Symbols on floor plans: fixtures such as toilets, washbasins, bathtubs, shower trays, kitchen sinks, hobs and
  washing machines become blocks too, with their back on the wall.
  - Libraries say which drawings they are for (`drawing = ["building"]`) and whether their blocks are drawn at real
    size (`size = "real"`, like architectural CAD blocks); real-size blocks are shrunk by the drawing scale, and 1:50
    and 1:100 are tried while the scale is unknown. The DXF holds them at real size.
  - A second built-in library, `floor-plan`, with seven fixtures and one of the default libraries
    (`DEFAULT_SYMBOL_LIBRARIES`); every library is used on the kind of drawing it is for.
  - Fixtures are found from their rectangular outline, also where a side is the face of the wall or the edge of a
    counter, and from two opposite sides where OCR read a small fixture at 1:100 as letters ("IOl"). A symbol never
    takes a piece out of a wall.
  - The benchmark has furnished floor plans at 1:50 and 1:100 (scan and phone photo), and `cad-ingest library check
    --scale` tests real-size libraries at a drawing scale.
- Every page of a PDF or multi-page TIFF scan is converted, not only the first one. The first page with a drawing stays
  at the root of the ZIP, every further page n goes to `page-n/` with the same files; each `report.json` names its
  `page` and all `pages` converted, the preview holds every page and the markdown has a section per page. Pages
  without a drawing are skipped with a warning.
  - Job API: `options.pages` chooses pages; new setting `MAX_SCAN_PAGES` (default 20). A job leaves out the pages
    it would not finish within `CONVERSION_TIMEOUT_SECONDS` and says so in a warning, rather than failing.
  - Command line: `--pages 1-3,5`.
- Vector PDFs, such as CAD exports, are read as they are instead of being recognised on an image: lines, arcs,
  circles, fills and texts keep their exact geometry, colour, pen width and CAD layer (from the PDF's optional
  content), and texts their string, angle, slant, height and length. Nothing is classified or replaced by symbols,
  so the DXF shows what the PDF shows. A PDF page that is a scan still takes the image route.
  - `report.json`: `sheet` is `page` for such a page, and `paper` is `null` where the page is no standard size.
  - The markdown lists the CAD layers with the number of elements on each.
- Markups on scanned prints: pen strokes and highlighter in colour on a black and white print are kept apart from the
  drawing instead of turning into stray lines and symbols and garbling the print under them (see
  `docs/photos-and-scans.md#markups`).
  - The DXF has pen strokes as polylines in their colour on `P2C-MARKUP`, highlighter as transparent fills on
    `P2C-HIGHLIGHT` and what Tesseract reads of the handwriting, as a guess, on `P2C-MARKUP-TEXT`.
  - `report.json` lists them under `markups` (kind, box, colour, reading and the printed texts and symbol tags they
    cover), the ZIP holds an image `markup-k.png` of each one, and the markdown has a section on them.
- Text spotting on photos and scans: a learned text detector and recogniser (PP-OCRv6 small, Apache-2.0 models run by
  RapidOCR on ONNX Runtime, on the CPU) reads the sheet after the character grouping and replaces the texts whose
  letters lie in the lines it finds. It finds letters that touch lines and bubble rings, and reads plotter fonts, the
  slashed zero and touching letters much better than Tesseract. New text needs a score of 0.8 (0.9 for one or two
  characters, which must also be no taller than the usual text), less sure replacements are marked for review, and
  characters both readers confuse go to the vote of their look-alikes. Lines cut by a tile edge are read again whole,
  and wide spaces the recogniser dropped are put back. The image fetches and checks the models when it is built. New
  dependencies: `rapidocr`, `onnxruntime`.
- Pipes and symbols on photos and scans are connected more reliably: a symbol beside the pipe that runs through two of
  its ports (a flow arrow) is moved onto it; flow arrows are tried where strokes stand beside a straight line and need
  their pipe to go on past both ends; strokes left wholly inside a symbol with ports are dropped, so they no longer
  join the pipes on either side; and a gate valve with a ball, filled or filled in by blur, is a ball valve.
- What to check on photos and scans: symbols that may be the wrong ones (less than 90 % of their outline found, or a
  symbol of another kind fits almost as well) go to the new layer `P2C-SYMBOL-CHECK` and appear in magenta in
  `overlay.png`; `report.json` has the new list `review` of the texts and symbols to check, least confident first, and
  each symbol its `confidence` and `doubt`; the markdown has a section `Symbols to check`. The benchmark counts the
  symbols marked and the wrong ones left unmarked.
- Reviewer model for `vectorize`: a job may name a vision model behind an OpenAI-compatible endpoint (`reviewer`:
  `base_url`, `api_key`, `model`, optional `headers`), such as a LiteLLM proxy. It is asked about the texts and symbols
  to check, sees only a crop around each, and its answers are taken where they fit; `report.json` counts them under
  `reviewer`. The key and headers are dropped when the job ends and never returned or logged.
- Sheet grammar for texts on photos and scans: inch marks read as foot marks or degree signs after pipe sizes, and the
  equals sign of key=value labels read as a colon, are put right where the sheet shows enough of these notations.
- Benchmark: exact texts by kind (codes such as tags and line numbers, numbers, words), errors left unmarked (texts
  read wrong without a review mark in their place), and connections between symbols (pairs joined by linework, found
  and correct).
- Large drawings printed small (an A1 P&ID on A3 or A4) are recognised at their drawn size, since the tolerances of
  recognition are made for drawings at their drawn size; the share they are printed at is measured on the height of
  the character-sized shapes of ink. On an image with fewer than 5 pixels per drawn millimetre (a phone photo of a
  small print) lines and texts are still found on the sheet. `report.json` has the new field `printed`. The benchmark
  has two such cases (A1 on A3, scanned; A1 on A4, photographed).
- Connected linework on photos and scans: once the symbols are placed, line ends that stop up to 2 mm short of other
  linework are joined to it (a branch to its pipe, a signal line to its bubble, a pipe to its valve or vessel), two
  loose ends whose lines meet nearby form a corner, a symbol port that faces a pipe or the next symbol's port within
  6 mm is piped to it, a line drawn broken where it crosses another becomes one line across it (without a junction),
  pieces that then meet end to end become one polyline, and loose pieces under 2 mm are dropped. The benchmark counts
  the ends that stop short and the loose crumbs.
- Fewer wrong symbols: a symbol needs 85 % of its outline instead of 80 % (a symbol not placed keeps its strokes as
  lines, which is better than a wrong block); a symbol whose strokes end freely in the library, such as the barbs of a
  flow arrow, is not placed where those strokes run on (the edges of a line number's frame, the cross in a box,
  letters); letters inside a symbol that holds none, such as a reducer, count against it; and what is left of letters
  that could not be read in an instrument bubble is dropped, the bubble marked to check ("its text could not be read").

### Changed

- Pen widths are measured on the grey image, as the ink across a stroke divided by the darkness of the ink nearby,
  instead of on the binary image, which blur and the threshold widen (step `width_calibration`). It changes widths only,
  never geometry.
- Look-alike characters (0 and 8, B and 8, D and 0) are decided per token where the token tells (step
  `lookalike_tokens`). The vote of look-alike glyphs passes over a character both readers read alike, changes a
  character only with a clear majority (12 of the 16 glyphs most like it) and marks the text for review where it has
  none; a token that is a word of the kind's vocabulary in only one of its readings takes that one. Before, a class of
  glyphs read half one way and half the other went one way or the other as a whole, and switching a step elsewhere on
  the sheet could flip it in dozens of tokens. The P&ID profile's vocabulary has line numbers ("DN80-P-1011") and
  equipment tags ("P-101", "B-501"; not with I or O) besides nominal sizes and pressures.
- Starter libraries: the orifice plate of `din-10628` 1.1.0 is a thin plate across the pipe and is now found; drawn as
  two bars, it was not, and orifice plates drawn as two bars are still not recognised. The make and break contacts of
  `iec-60617` 1.1.0 have their ports where their wires stop, not at the ends of leads along the wire, so they are found
  from the ends of their wires. `cad-ingest library check` no longer asks for ports on a symbol that is found along its
  `THROUGH` line.
- Benchmark measures: junctions kept no longer counts the bends of a curve drawn as short straight lines, and crossings
  kept counts output lines taken for dashed as running through. "Symbols where there are none" is measured on every
  drawing, and breaks per metre no longer counts a wire that ends at the port of a symbol. In the measures of objects in
  one piece, breaks and ends stopping short, a chain of dimensions counts as one entity. A mark on the level mark of a
  section counts as one on a symbol of no library, not as a false mark.
- Handwriting in colour markups is no longer read by default: Tesseract made up most of what it read. `reading` is
  `null`; the step `read_handwriting` reads it as before. The markdown section is now `Markups (added by hand)`.
- A pen stroke in colour no longer leaves a faint grey line in the drawing: the grey image lifts it and its fringe to
  the paper.
- The DXF of a drawing printed small is at its drawn size, so its texts and symbols have the sizes they were drawn
  with; positions in `report.json` stay in sheet millimetres.
- Texts in the DXF, the tags in symbols included, are lettered as with a stencil: texts of about one height get one
  height and one width per character, so the same label looks the same everywhere, and are fitted to that length in
  any font. A text printed clearly narrower keeps its printed length, so a tag stays inside its bubble and a line
  number inside its frame; one spaced wider keeps where it starts, so notes stay flush left. The preview squeezes a
  text along its line only, as CAD does, and a text whose letters were found only in part keeps its size.
- A spotted text line through which a stroke runs from above to below its letters (the end of a line number's frame,
  the ring of the bubble beside it, the line between two labels) is cut there into two texts, where the letters on
  either side are a letter's height apart or read with a space between them.
- A thick round ring, such as the mark of a note, is a circle with a wide pen instead of a filled disc, and the letter
  in it is read.
- The starter library moved from `app/plan2cad/symbols.dxf` to `libraries/din-10628`.
- A symmetric symbol (one that looks the same turned or mirrored) takes the pose in which most of its attribute
  positions have a text, so a tag below a symmetric valve is no longer missed, and then the pose with its back on the
  wall (a hob with its insertion point on the wall, not on its front edge). Poses whose outlines lie within 0.75 mm
  of each other count as the same.
- Text that is only strokes of a found symbol read as letters ("00" on a hob) is dropped, and text ink counts towards
  the circles and small closed shapes of a symbol.
- The simulated scans and photos of the benchmark moved to `app/plan2cad/simulate.py`.
- Lines in strong colour count as ink as dark as black ones (there the darkest colour channel is the grey value), so
  yellow and orange lines of a CAD plot are no longer lost. Photos keep their plain brightness.
- Text on scans is read with Tesseract's most accurate models (`tessdata_best` 4.1.0, German and English) instead of
  the fast ones Ubuntu ships. Characters are grouped into lines by similar height first, so two text lines beside a
  slanted stroke stay apart, and a character read unlike the glyphs most like it on the sheet is corrected by their
  vote (the slashed zero of a plotter font misread as 8, 9 or B). An O or I next to a digit in a code is a 0 or 1,
  and after "Maßstab" a scale read with letters ("1 SO") counts as 1:50. On the benchmark's P&ID scan, 93 % instead
  of 87 % of the texts are read exactly. The best models are slower: that scan takes about a tenth longer, the hard
  phone photo up to two thirds.
- Symbols are not taken from letters or from the frame around a text: a symbol may find at most 30 % of its outline
  on text that reads as words, and only a symbol that holds text (an instrument bubble) may be the box of a line
  number, an interlock diamond or the end of a label.
- Symbols drawn at the printed size keep to the size of most symbols on the sheet (at most 1.4 times larger or
  smaller), so the x marks of a capillary line are no longer taken for flow arrows, nor note markers for instruments.
- Symbols drawn at the printed size are looked for at the size of the sheet's text: on a sheet whose text is 3 mm
  high instead of about 2 mm, at one and a half times the library's size.
- Symbol recognition is faster on dense sheets. A symbol with a single connected port is only looked for at pipe
  ends (straight lines from 4 mm), not at the ends of the short strokes of letters, which also gives fewer false
  symbols on text; and candidates are checked with one search for nearby ink instead of two.
- Lines on scans and photos come out in one piece more often, and curves as true arcs, so that a conversion needs less
  rework (see `docs/photos-and-scans.md#how-it-works`):
  - A traced stroke that runs straight and then bends is cut into the fewest straight lines and arcs that follow it:
    a rounded corner, a vessel head or the round end of a line-number box is one arc or a few tangent ones, not a
    chain of short lines (a stroke that is round as a whole stays one arc, as before). Arcs that lie on one circle
    and were cut apart by a crossing or touching line become one arc again, also within a polyline.
  - A gap of up to 5 mm in a line is closed where the grey image shows a stroke along it (a faint stretch darker
    than the paper beside it, a crossing that cut it), and a branch from 4 mm that stops short of the pipe it points
    at is extended onto it. Gaps of bare paper, such as dashes, and the inside of a small symbol stay open; short
    pieces are joined only into a run of 8 mm or more (a dotted line), not the two sides of a valve.
  - Faint stretches of horizontal and vertical lines (a pale pen, a line that came out dotted) count as ink.
  - Of parallel pipes that end beside one symbol port, only the one pointing at it is snapped onto it, so the others
    are no longer turned onto the port.
  - Two or three dashes that run from one stroke to another, such as a short signal line between an instrument bubble
    and the symbol it acts on, become one dashed line; a short dash on its own is no longer dropped as a speck, and a
    dashed line is never dropped as a crumb.
- Text on scans and photos is read more completely:
  - A single character is read where it stands alone in a frame, such as the loop letter of an instrument bubble or
    an interlock diamond.
  - The text in circles that hold text is read a second time without the ring and the line across a shared display,
    so letters that touch them are no longer lost.
  - A text line read with low confidence stays a marked text also where the scan merged touching letters, as long as
    as many characters were read as its length holds.
  - A bar read at either end of a text line ("| LT"), the end of a leader beside it, stays linework.
- Symbols on scans and photos:
  - Nothing is placed in a strip of notes and tables below a line across the whole frame (revision table, issue
    record, title block).
  - A symbol drawn around a circle is centred on the circle found in the scan, and a symbol a little beside the pipes
    that end at its ports is moved onto them. A pipe end is snapped onto a port only where the port lies on its course,
    so that no pipe is turned askew.
  - A symbol drawn without curves, such as an off-page connector or a flow arrow, is not placed on the round end of a
    line-number box.
- Benchmark: new metrics for what makes rework tedious, `breaks_per_m` (places per metre of reference linework where
  the output stops although the line runs on), `curve_pieces` and `curves_as_arcs`, in a new report section.

### Fixed

- The two bars of a flange across a pipe were read as a sure letter "H", alone or joined to a line label beside them
  ("HH2"-P-30207"), and a valve between two pipes read as letters ("Hoeat") kept its strokes from the symbols. The end
  of a spotted line that a pipe runs along is now cut off (a line along it throughout, such as the crosses of a
  capillary line read as "xxxx", is linework with what else was read in it), a single character with a pipe through it
  and on past it is linework, and a text a pipe runs into at both ends is no word to the symbol matching (step
  `text_on_pipes`).
- An instrument bubble took as its number a text that runs out of its ring, such as the number read with the bars of the
  flanges on either side ("T40201T"). Its attributes now come only from texts inside the ring. Where the bars touch the
  drawn ring from outside, they are strokes of the ring rather than letters, and the text seemed to end inside the ring:
  a spotted line inside a drawn ring is now cut at the ring where it runs out of it by a character or so, and the part
  inside is read again (step `attributes_inside`).
- Flanges were taken for orifice plates, and the marks of signal lines and the triangles of valves for flow arrows. A
  plate across its line, such as an orifice plate, now needs the line to end at both of its ports and none to run on
  through it, and a symbol drawn on a line, such as a flow arrow, strokes of its own: not a hairline and not part of a
  larger figure (step `symbol_context`).
- Print was taken for pencil marks (step `pencil_guard`). A stroke is pencil now only where it is lighter than the print
  of its width on the same sheet (a thin printed line is grey in a scan; it stays in the drawing), and what a mark
  through print leaves of the print, specks smaller than a mark (the edges of digits fused with an instrument bubble,
  the thin marks of a signal line), is no note; the specks stay out of the drawing as before.
- On phone photos, colour that the camera adds was taken for pen strokes: white balance noise, lens fringes along black
  print and the paper's coloured shadow (step `colour_guard`). A pen stroke now keeps one hue around each of its pixels,
  and its note is at least 2 mm long. Two pens of different colours that cross both stay, and so does a pen traced over
  a thin printed line.
- Hatching that the text recognition read as a text line ("IIII", "////"), such as the hatching of a cut slab in a
  section, was left as a text to check, and its lines were missing. It is traced as lines with the outline it ends on
  (step `hatching`).
- Nearly parallel lines were merged millimetres off far from the corner of the sheet, and dense hatching moved across
  the line it ends on: pieces within a degree of one another were each measured at their own angle. They are now
  measured at the angle of the longest piece, and each merged line is drawn at the angle of its own pieces, so a long
  line keeps its angle beside a short piece of the same direction (step `merge_where_drawn`).
- On a scanner lid about as grey as the paper, the whole image was taken as the sheet, and the scale was off by the
  lid's border. The paper is now found by its straight edges, checked against the aspect ratio of its size and, where
  known, the scan's dpi (step `paper_edge`).
- Specks of dirt counted towards the print size and made a poor scan look like a print at a fifth of its size. Only
  character heights at which most shapes stand in rows count now (step `letter_rows`).
- A P&ID with more vertical pipe labels than horizontal text was turned by 90 degrees. The rotation check now scores
  each rotation by horizontal text the right way up together with vertical text reading bottom to top.
- The round ends of a line-number box and rounded corners between two lines were read as brackets, so they were
  missing from the drawing and garbled the text next to them. They are now kept from text recognition.
- A circle in a box (a shared display instrument) was missed when OCR read its quarters as brackets, and the round
  end of a line-number box could be taken for a circle.
- Parallel lines close together, such as a pipe running into the box of its line number, could be pulled onto one
  averaged line when short strokes between them (letters touching the lines) chained them, so lines were shifted and
  box sides went missing. A piece now joins the line it continues at that line's own offset; pieces running beside
  one another less than 0.7 mm apart are still one stroke.
- A letter touching the round end of a line-number box was kept from text recognition with it.
- The frame of a long line-number box holding several texts was not found, so symbols could be placed on it.
- Letters in a small box, such as a line size, were read together with the box: the reading was garbled and the box
  went missing. A box or ring around letters is no longer taken for a character.
- A symbol could be placed on a small closed box whose text was not read, such as an instrument bubble on its sides
  or a valve on its letters. A symbol without a rectangle of its own no longer lies mostly on such a box.
- A P&ID listing areas in m² was taken for a floor plan. The drawing type now weighs the
  room areas and the building scale against nominal sizes, instrument bubbles and a title such as "R&I" or "R+I".

## [1.1.0] - unreleased

First public release under the name cad-ingest.

### Added

- Photos and scans of drawings to DXF (job operation `vectorize`): floor plans and P&IDs up to A0 from phone photos
  (JPEG, PNG, HEIC, WebP, TIFF, BMP) or scans (including PDF).
  - The sheet is found and straightened. Light is evened out, and linework, arcs, circles, dashed lines and filled
    areas are vectorised.
  - Texts are read with Tesseract OCR, and the scale is taken from the title block.
  - P&ID symbols are placed as blocks of a DXF symbol library, with their tags as attributes.
  - The result is a ZIP with the DXF in real-world millimetres, a PDF, a review overlay and a JSON report.
- `cad-ingest` command (`python -m app.convert`) that converts a single local file without Redis or a worker.
- Benchmark with synthetic drawings and ground truth (`bench/`).
- Multi-arch images (`linux/amd64`, `linux/arm64`) on GitHub Container Registry, with SBOM and build provenance; the
  test suite runs natively on both architectures.
- Open source project files: Apache-2.0 licence, contributing guide, code of conduct, security policy and
  `docker-compose.yml`.

## [1.0.0] - 2026-10-15

### Added

- Job service for CAD files: an HTTP API plus RQ workers on Redis, with queues, priorities, retries with backoff,
  callbacks, cancellation and restart safety.
- `extract`: searchable markdown from DXF, DWG, HPGL, STEP, IGES, BREP, IFC and mesh formats (STL, OBJ, PLY, OFF,
  3MF, DAE, glTF, GLB). It covers title blocks, texts, dimensions, layers, blocks, assembly trees, sizes and IFC
  property sets.
- `preview`: PDF for 2D drawings, binary glTF (GLB) for 3D models.
- Isolation of every job in its own process with a time and memory limit.

[1.1.0]: https://github.com/weyer-data-engineering/cad-ingest/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/weyer-data-engineering/cad-ingest/releases/tag/v1.0.0
