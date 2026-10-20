# Photos and scans to DXF (`vectorize`)

Engineering teams often receive plans only as phone photos or scans. The job operation `vectorize` (and the
`cad-ingest` command for photos and scans) turns them into an editable CAD drawing. The code is in `app/plan2cad`, the
job handler in `app/scan.py`.

![A simulated phone photo of a synthetic P&ID next to the DXF made from it](images/pid-photo-to-dxf.jpg)

*Synthetic test drawing from the benchmark, not a real document.*

## How it works

1. **Sheet**: finds the paper in the photo (GrabCut, edges refined at full resolution) and removes the perspective. If
   the paper runs out of the picture, it uses the printed drawing frame instead (aspect ratio from the perspective,
   Zhang & He). Scans without visible paper edges are deskewed.
2. **Light**: on a colour plot, takes the darkest colour channel as grey value where the colour is strong (so yellow
   and orange lines are ink as dark as black ones; colour casts of photos are too weak for this). On a black and white
   print, strong colour is a markup added by hand and is taken out of the drawing, see [Markups](#markups). Then it
   divides by the paper brightness (a closing that follows soft shadow edges) and binarises against the darkest ink
   nearby, so glare and light toner keep their lines.
3. **Vectors**: filled areas (thicker than 1.5 mm), horizontal and vertical lines (morphology), circles (Hough, verified
   on all ink; text found on a ring is dropped as misread symbol arcs), everything else by skeleton tracing with arc
   fitting. Collinear pieces are joined, regular pieces become dashed lines, corners are closed.
4. **Objects**: pieces that meet end to end become one polyline (a wall outline, a pipe run through its bends, a vessel
   with its heads); slivers and loose specks from the tracing are dropped. Floor plans get door swings (leaf and quarter
   arc) and walls, P&IDs equipment (large closed outlines), signal lines (dashed or ending at an instrument bubble) and
   pipes.
5. **Text**: characters of similar height are grouped into lines first, and small marks such as dots, hyphens and
   quotes join the line they sit in, so two text lines beside a slanted stroke stay apart. The lines are read by
   Tesseract in batches with its most accurate models (`tessdata_best`). A drawing is written in one font, so a
   character read unlike the glyphs that look most like it on the same sheet is corrected by their vote: the slashed
   zero of a plotter font reads as 8, 9 or B now and then, but as 0 mostly. In a code, an O or I next to a digit is a
   0 or 1 (`PN16`, `C=01XY2Z`). The sheet is turned if its text
   reads better turned: each rotation counts horizontal text the right way up together with vertical text reading
   bottom to top, so a P&ID full of vertical pipe labels is not taken for a sideways sheet. The scale is read from
   the title block. The drawing type weighs room areas (m²) and a building scale against nominal sizes (DN),
   instrument bubbles and a title such as "R&I". Text that cannot be read reliably stays a text (a
   guess on its own layer to check) instead of turning into a heap of short strokes.
6. **Symbols**: valves, pumps and instruments on P&IDs (and on drawings of unknown kind), toilets, washbasins, bathtubs
   and other fixtures on floor plans are looked up in symbol libraries and placed as their CAD blocks instead of loose
   strokes, see below.
7. **Markups**: pen strokes and highlighter in colour on a black and white print become their own layers, see
   [Markups](#markups).

## Output

The ZIP holds `plan.dxf` (real-world millimetres, one layer per kind of object: `P2C-WALL`, `P2C-DOOR`, `P2C-PIPE`,
`P2C-EQUIPMENT`, `P2C-SIGNAL`, `P2C-FRAME`, `P2C-LINE`, `P2C-DASH`, `P2C-CURVE`, `P2C-FILL`, `P2C-TEXT`,
`P2C-TEXT-CHECK`, and for markups `P2C-MARKUP`, `P2C-HIGHLIGHT` and `P2C-MARKUP-TEXT`; ISO line weights, the
straightened photo `underlay.png` as raster underlay on `P2C-UNDERLAY`),
`plan.pdf` (vector, on the paper size), `overlay.png` (the recognised vectors in colour on the faded photo, symbols in
teal, for checking), `markup-k.png` for each markup and `report.json` (sheet, resolution, drawing type, scale, counts, timings, all texts, the symbols
with their attributes and ports, the lines joining two symbol ports and the markups). `extract` lists the recognised
texts, symbols (with their tags) and markups as markdown, so photographed plans become searchable too.

Scans often arrive as PDFs or TIFFs with several pages. Every page is converted on its own (one at a time, so memory
stays that of one page): the first page with a drawing at the root of the ZIP, every further page n in `page-n/` with
the same files. Pages without a drawing, such as a cover letter, are skipped with a warning. `options.pages` (command
line `--pages 1-3,5`) picks pages; a job converts at most `MAX_SCAN_PAGES` (20) and leaves out the pages it would not
finish within its time limit. Scanned PDF pages are rendered at 300 dpi; pages drawn in vectors are read directly,
see [Vector PDFs](#vector-pdfs).

![Review overlay: recognised lines, arcs, symbols and texts in colour on the faded photo](images/review-overlay.jpg)

*Detail of `overlay.png` for the synthetic P&ID above: solid lines in red, dashed lines in blue, arcs and circles in
green, library symbols in teal, texts in orange and texts to check in magenta. "DN550-P-101C" (really "DN50-P-1010")
was not read reliably, so it went to the review layer `P2C-TEXT-CHECK`.*

To try it on a single file without the job API (writes `plan.md`, `plan.pdf` and `plan.zip` to `out/`):

```bash
docker run --rm --user "$(id -u)" -v "$PWD:/data" -w /data ghcr.io/weyer-data-engineering/cad-ingest \
  cad-ingest plan.heic out --paper A3
```

There is no DWG output: the ODA File Converter may be used by non-members for non-commercial applications only, and
LibreDWG writes only old DWG versions, experimentally. AutoCAD and other CAD programs open the DXF directly.

## Markups

Plans come back from site and review with notes on them: a cloud around a valve with "removed" next to it, a pipe
struck out, a line marked with a highlighter. On a black and white print, every stroke in strong colour is such a
markup. Read as part of the drawing, it would turn into stray lines and symbols and garble the printed text under
it. So the converter takes markups out before it reads the drawing and keeps them on their own layers:

- **Pen strokes** become polylines in their colour on `P2C-MARKUP`. Strokes close to one another make one note.
- **Highlighter** (light colour at least 1 mm wide) becomes a transparent fill in its colour on `P2C-HIGHLIGHT`. The
  print under it shows through and is read as usual.
- **Handwriting** in a note is read by Tesseract where it can. The reading goes on `P2C-MARKUP-TEXT` as a guess to
  check: Tesseract is made for print, so it reads clear block letters and misses much of joined-up writing.
- **What a markup covers**: the printed texts and the tags of the symbols inside a pen loop, under a pen stroke or
  under highlighter, such as the instruments in a revision cloud.

`report.json` lists them under `markups`, each with its `kind` (`note` or `highlight`), `box_mm` (x0, y0, x1, y1 on
the sheet), `color` (RGB), the handwriting `reading` (or `null`), the printed `texts` and symbol tags it covers and its `image`: the
ZIP holds `markup-k.png` for each markup, the markup in colour on the faded print around it. These images are for a
person or a vision model to read what Tesseract could not. The markdown has a section `Markups (added by hand in
colour)`, for example `- Note in red: “removed” (handwriting, read automatically); covers: FV 101, FT 101`.

On a colour plot, where coloured lines are a large share of the ink, all colour is print and no markups are taken out.
Markups in black or grey cannot be told from the print.

## Vector PDFs

A PDF exported from CAD holds the drawing itself, not a picture of it. Such a page (at least 20 paths and no image
covering 30 % of it) is read as it is, without image recognition (`app/plan2cad/pdfvector.py`):

- Lines, arcs, circles, polylines and fills keep their exact geometry, colour and pen width (the DXF gets the nearest
  of all DXF line weights). CAD programs write arcs and circles as Bézier curves; curves that follow a circle within
  0.03 mm become arcs and circles again.
- Every element keeps the CAD layer it was drawn on where the PDF names it (optional content), and the DXF gets the
  same layers. The markdown lists the layers with the number of elements on each.
- Texts keep their string, angle and colour. Height, length and slant are measured on the text as it is drawn, and
  the DXF text is fitted between the ends of its baseline, so it covers the same place in any font.
- The page is the sheet: `paper` and `dpi` do not apply. `report.json` has `sheet` `page`, the page size, and `paper`
  `null` where that is no standard size. A page turned with /Rotate is read the way it is shown.
- Nothing is classified or replaced by symbols: walls, pipes and valves stay the strokes they were drawn as, on their
  layers, so the DXF shows exactly what the PDF shows. The drawing type and scale are still read from the texts.

A PDF page that is a scan, or a drawing flattened to an image, takes the image route described above.

## Symbol library

Recognised symbols become references (`INSERT`) to the blocks of a symbol library on the layer `P2C-SYMBOL`, with
their texts as attributes (the tag of a pump, the letters and number of an instrument), so the DXF holds a valve as a
valve, not as six strokes, and a washbasin as a washbasin. Pipe ends are snapped onto the ports of the symbols they
lead to.

The symbols come from symbol libraries: DXF files with one block per symbol, with points where pipes connect and the
places of their attributes. Each library is for P&IDs or for floor plans, and only used on its kind of drawing. Two
starter libraries are built in, written by `bench/make_library.py`:

- `libraries/din-10628`: 18 simplified P&ID symbols after DIN EN ISO 10628-2: gate, ball, butterfly, check, angle,
  three-way, control and safety valve, centrifugal pump, compressor, heat exchanger, local and control-room instrument,
  reducer, filter, orifice plate, off-page connector and flow arrow.
- `libraries/floor-plan`: seven fixtures at real size: toilet, washbasin, bathtub, shower tray, kitchen sink with two
  bowls, hob and washing machine. They are shrunk by the scale of the drawing (from the title block or the job's
  `scale`; 1:50 and 1:100 are tried while it is unknown), and the DXF holds them at real size.

A company adds its own blocks as a library of its own, installed on the server or brought along by a job; [Symbol
libraries](symbol-libraries.md) explains the format and the tools that check and build one.

How a symbol is found: candidates come from two line ends that two ports of a symbol fit (a valve between two pipe ends;
this fixes position, turn and size), from rectangles whose proportions fit a rectangle of the symbol (the box of a
washbasin, whose back may be the face of the wall it hangs on), from circles (pumps, instruments, hotplates) and, where
nothing was found yet, from single pipe ends (straight lines from 4 mm; shorter strokes are mostly letters) and from
groups of small strokes. Each candidate is matched against the
image itself (the centre lines of the ink, not the traced vectors, which round off arrowheads and springs): a point of
the symbol counts as found where ink runs close by in the same direction. A symbol needs 80 % of its outline and half
of each of its parts (a ball valve needs its ball), a line leaving at least one port and, if it has one, a straight
line through it. Ink inside the symbol that it does not explain counts against it, except for text at its attribute
positions. Of overlapping candidates the one explaining the most ink wins. A symmetric symbol fits as well turned half
way or mirrored; it takes the pose in which most of its attribute positions have a text, and then the one with its
back on the wall: a fixture's insertion point is in the middle of its back, and the linework left around it continues
that side. Nothing is placed in the title block in the bottom right corner of the frame, whose small text would
otherwise look like symbols.

OCR often reads small fixtures at 1:100 as letters (a washbasin as "IOl"). So two opposite sides of a rectangle are
enough as a candidate, text may stand in for the outer outline and for small closed shapes such as a bowl or a
hotplate, and a text made of the strokes of a symbol is dropped once the symbol is found. The other way round, letters
must not make a symbol: on a dense P&ID, a V or an X in a pipe label fits a valve. A symbol may find at most 30 % of
its outline on text that reads as words or a code (anything but the circles and bars a symbol reads as, such as "oo"
or "101"). And the frame around a text (the box of a line number, an interlock diamond, the end of a label) is only
taken as a symbol that holds text inside its outline, such as an instrument bubble.

Libraries whose symbols are drawn at the printed size (P&ID symbols, unlike fixtures) are drawn larger or smaller on
some sheets, for example a large drawing printed on a smaller sheet. The converter takes the size from the text: where
the sheet's median text height is more than a quarter larger or a fifth smaller than 2 mm, the symbols are looked
for at that ratio (between half and twice their size).

A sheet draws its symbols at one size. Once at least 12 symbols drawn at the printed size are found, their median size
is the sheet's, and candidates more than 1.4 times larger or smaller are left out: at the smallest size the library
allows, the x marks of a capillary line fit a flow arrow and a filled note marker fits an instrument bubble.
Fixtures are left alone, as their size is that of the real object.

## Benchmark

`bench/` generates drawings with an exact reference solution (floor plan A3, also with fixtures at 1:50 and 1:100;
P&ID A1/A0), simulates scans and phone photos (perspective, waviness, folds, shadow, glare, defocus, hand shake, noise,
phone sharpening, JPEG) and measures the result against the reference:

```bash
python bench/run.py --out bench/out [--quick] [--real photo.heic:A4]
```

Metrics: share of reference linework found within 1 mm / 0.5 mm on paper (completeness) and share of found linework
that is right (correctness), mean deviation, dashed lines, circles, exact texts, character error rate, scale, objects in
one piece, symbols placed as the right block within 1.5 mm, attributes read, runtime. Mean over 1-3 seeds per case:

| Case | px/mm | Lines found / correct (<= 1 mm) | Mean deviation | Circles found | Texts exact | Symbols found / correct | Attributes read | Runtime |
|---|---|---|---|---|---|---|---|---|
| Floor plan A3, scan 300 dpi | 11.8 | 100 % / 100 % | 0.08 mm | - | 100 % | - | - | 7.8 s |
| Floor plan A3, phone photo 12 MP | 7.8 | 99.9 % / 100 % | 0.21 mm | - | 100 % | - | - | 5.3 s |
| Floor plan A3, phone photo 12 MP, hard | 6.7 | 77 % / 83 % | 0.44 mm | - | 30 % | - | - | 17.2 s |
| Floor plan A3 1:50 with fixtures, scan 300 dpi | 11.8 | 100 % / 100 % | 0.09 mm | 100 % | 100 % | 100 % / 100 % | - | 10.0 s |
| Floor plan A3 1:50 with fixtures, phone photo 12 MP | 7.8 | 99.9 % / 100 % | 0.22 mm | 78 % | 100 % | 100 % / 100 % | - | 6.6 s |
| Floor plan A3 1:100 with fixtures, scan 300 dpi | 11.8 | 99.9 % / 100 % | 0.10 mm | 100 % | 96 % | 96 % / 96 % | - | 9.7 s |
| Floor plan A3 1:100 with fixtures, phone photo 12 MP | 7.8 | 99.7 % / 99.9 % | 0.22 mm | 67 % | 90 % | 93 % / 90 % | - | 7.6 s |
| P&ID A1, scan 200 dpi | 7.9 | 99.9 % / 99.4 % | 0.12 mm | 100 % | 87 % | 94 % / 93 % | 87 % | 31.0 s |
| P&ID A1, phone photo 12 MP | 3.9 | 99.2 % / 96.7 % | 0.23 mm | 89 % | 18 % | 78 % / 77 % | 10 % | 32.4 s |
| P&ID A0, scan 200 dpi | 7.9 | 100 % / 98.9 % | 0.12 mm | 100 % | 84 % | 91 % / 91 % | 81 % | 53.3 s |
| P&ID A0, phone photo 12 MP | 2.8 | 90.6 % / 94.9 % | 0.33 mm | 82 % | 1 % | 62 % / 61 % | 0 % | 62.1 s |
| P&ID A0, phone photo 48 MP | 5.6 | 99.1 % / 98.3 % | 0.23 mm | 85 % | 34 % | 82 % / 85 % | 23 % | 45.4 s |

Linework and circles hold up down to 3 px per mm of paper, text does not: on the A0 P&ID, exact texts (2 mm high, many
of them inside instrument bubbles) rise from 14 % at 3 px/mm to 58 % at 6 and about 82 % from 8 px/mm. 81-99 % of the
objects of the reference drawing come out in one piece (1.13-1.88 DXF elements per object). Symbols that are missed
stay linework. On floor plans, all fixtures are found at 1:50; at 1:100, where OCR reads many of them as letters,
93-96 %.

## Limits

- Handwriting is read only in part (see [Markups](#markups)); the markup images in the ZIP show what it says.
- Small text needs about 8 px per mm of paper. An A0 sheet as one 12 MP photo has 2.8 px/mm, so its text is not
  readable; use 200-300 dpi scans for A0/A1. Unread text is kept as a guess on `P2C-TEXT-CHECK`.
- Thin lines (0.18 mm, e.g. dimension lines) need about 6 px per mm of paper. On a 12 MP photo of an A1 floor plan
  (4 px/mm) about two thirds of them are lost, and without a readable title block the scale stays unknown; pass
  `scale` and `drawing` in `options` when they are known.
- Measured line widths are too close to tell main pipes from thinner lines, so P&ID lines are classified by geometry,
  not by pen width.
- Non-flat paper (waves, folds) shifts lines by up to a millimetre; there is no dewarping yet.
- Only symbols in the selected libraries are recognised; anything else stays linework. Fine details need resolution:
  below about 6 px per mm of paper a ball valve's ball is lost and it comes out as a gate valve. The orifice plate of
  the starter library (two short bars across a pipe) is not found yet.
- Fixtures at 1:100 (a washbasin is 6 x 4.5 mm on the sheet) are mostly read as letters; about one in ten is placed
  as the wrong fixture or the wrong way round, for example a fixture in a corner, which fits both walls.
- Fixtures are only found at the drawing scale; while it is unknown, 1:50 and 1:100 are tried. Pass `scale` in
  `options` for other scales.
- Symbol recognition adds about 20 s to an A1 sheet and 35 s to an A0 sheet with the starter library; larger
  libraries take longer.
- Vector PDFs: symbols are not recognised on them yet, and a text is a single-line TEXT in the DXF's standard font,
  stretched to the length it has in the PDF. White fills count as paper and are left out, so a white mask over lines
  (such as a text background) does not hide them in the DXF.
