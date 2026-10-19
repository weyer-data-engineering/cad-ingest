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
2. **Light**: divides by the paper brightness (a closing that follows soft shadow edges) and binarises against the
   darkest ink nearby, so glare and light toner keep their lines.
3. **Vectors**: filled areas (thicker than 1.5 mm), horizontal and vertical lines (morphology), circles (Hough, verified
   on all ink; text found on a ring is dropped as misread symbol arcs), everything else by skeleton tracing with arc
   fitting. Collinear pieces are joined, regular pieces become dashed lines, corners are closed.
4. **Objects**: pieces that meet end to end become one polyline (a wall outline, a pipe run through its bends, a vessel
   with its heads); slivers and loose specks from the tracing are dropped. Floor plans get door swings (leaf and quarter
   arc) and walls, P&IDs equipment (large closed outlines), signal lines (dashed or ending at an instrument bubble) and
   pipes.
5. **Text**: characters are grouped into lines and read by Tesseract in batches; the sheet is turned if most text is
   upside down or sideways. The scale is read from the title block. Text that cannot be read reliably stays a text (a
   guess on its own layer to check) instead of turning into a heap of short strokes.
6. **Symbols** (P&IDs and drawings of unknown kind): valves, pumps, instruments and the like are looked up in a symbol
   library and placed as its CAD blocks instead of loose strokes, see below.

## Output

The ZIP holds `plan.dxf` (real-world millimetres, one layer per kind of object: `P2C-WALL`, `P2C-DOOR`, `P2C-PIPE`,
`P2C-EQUIPMENT`, `P2C-SIGNAL`, `P2C-FRAME`, `P2C-LINE`, `P2C-DASH`, `P2C-CURVE`, `P2C-FILL`, `P2C-TEXT`,
`P2C-TEXT-CHECK`, ISO line weights, the straightened photo `underlay.png` as raster underlay on `P2C-UNDERLAY`),
`plan.pdf` (vector, on the paper size), `overlay.png` (the recognised vectors in colour on the faded photo, symbols in
teal, for checking) and `report.json` (sheet, resolution, drawing type, scale, counts, timings, all texts, the symbols
with their attributes and ports, and the lines joining two symbol ports). `extract` lists the recognised texts and
symbols (with their tags) as markdown, so photographed plans become searchable too.

Scans often arrive as PDFs or TIFFs with several pages. Every page is converted on its own (one at a time, so memory
stays that of one page): the first page with a drawing at the root of the ZIP, every further page n in `page-n/` with
the same files. Pages without a drawing, such as a cover letter, are skipped with a warning. `options.pages` (command
line `--pages 1-3,5`) picks pages; a job converts at most `MAX_SCAN_PAGES` (20) and leaves out the pages it would not
finish within its time limit. PDF pages are rendered at 300 dpi.

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

## Symbol library

Recognised symbols become references (`INSERT`) to the blocks of a symbol library on the layer `P2C-SYMBOL`, with
their texts as attributes (the tag of a pump, the letters and number of an instrument), so the DXF holds a valve as a
valve, not as six strokes. Pipe ends are snapped onto the ports of the symbols they lead to.

The symbols come from symbol libraries: DXF files with one block per symbol, with points where pipes connect and the
places of their attributes. The built-in starter library is in `libraries/din-10628` (written by
`bench/make_library.py`). A company adds its own blocks as a library of its own, installed on the server or brought
along by a job; [Symbol libraries](symbol-libraries.md) explains the format and the tools that check and build one.

The starter library holds 18 simplified symbols after DIN EN ISO 10628-2: gate, ball, butterfly, check, angle,
three-way, control and safety valve, centrifugal pump, compressor, heat exchanger, local and control-room instrument,
reducer, filter, orifice plate, off-page connector and flow arrow.

How a symbol is found: candidates come from two line ends that two ports of a symbol fit (a valve between two pipe ends;
this fixes position, turn and size), from circles (pumps, instruments) and, where nothing was found yet, from single
line ends and from groups of small strokes. Each candidate is matched against the image itself (the centre lines of
the ink, not the traced vectors, which round off arrowheads and springs): a point of the symbol counts as found where
ink runs close by in the same direction. A symbol needs 80 % of its outline and half of each of its parts (a ball valve
needs its ball), a line leaving at least one port and, if it has one, a straight line through it. Ink inside the
symbol that it does not explain counts against it, except for text at its attribute positions. Of overlapping
candidates the one explaining the most ink wins. A symmetric symbol fits as well turned half way or mirrored; it takes
the pose in which most of its attribute positions have a text. Nothing is placed in the title block in the bottom right
corner of the frame, whose small text would otherwise look like symbols.

## Benchmark

`bench/` generates drawings with an exact reference solution (floor plan A3 1:50, P&ID A1/A0), simulates scans and
phone photos (perspective, waviness, folds, shadow, glare, defocus, hand shake, noise, phone sharpening, JPEG) and
measures the result against the reference:

```bash
python bench/run.py --out bench/out [--quick] [--real photo.heic:A4]
```

Metrics: share of reference linework found within 1 mm / 0.5 mm on paper (completeness) and share of found linework
that is right (correctness), mean deviation, dashed lines, circles, exact texts, character error rate, scale, objects in
one piece, symbols placed as the right block within 1.5 mm, attributes read, runtime. Mean over 1-3 seeds per case:

| Case | px/mm | Lines found / correct (<= 1 mm) | Mean deviation | Circles found | Texts exact | Symbols found / correct | Attributes read | Runtime |
|---|---|---|---|---|---|---|---|---|
| Floor plan A3, scan 300 dpi | 11.8 | 100 % / 100 % | 0.08 mm | - | 100 % | - | - | 7.5 s |
| Floor plan A3, phone photo 12 MP | 7.8 | 99.9 % / 100 % | 0.21 mm | - | 100 % | - | - | 5.2 s |
| Floor plan A3, phone photo 12 MP, hard | 6.7 | 77 % / 83 % | 0.44 mm | - | 30 % | - | - | 16.6 s |
| P&ID A1, scan 200 dpi | 7.9 | 99.9 % / 99.4 % | 0.12 mm | 100 % | 87 % | 94 % / 93 % | 87 % | 28.8 s |
| P&ID A1, phone photo 12 MP | 3.9 | 99.2 % / 96.7 % | 0.23 mm | 89 % | 18 % | 78 % / 77 % | 10 % | 32.0 s |
| P&ID A0, scan 200 dpi | 7.9 | 100 % / 98.9 % | 0.12 mm | 100 % | 84 % | 91 % / 91 % | 81 % | 56.0 s |
| P&ID A0, phone photo 12 MP | 2.8 | 90.6 % / 94.9 % | 0.33 mm | 82 % | 1 % | 62 % / 61 % | 0 % | 65.2 s |
| P&ID A0, phone photo 48 MP | 5.6 | 99.1 % / 98.3 % | 0.23 mm | 85 % | 34 % | 81 % / 84 % | 23 % | 45.2 s |

Linework and circles hold up down to 3 px per mm of paper, text does not: on the A0 P&ID, exact texts (2 mm high, many
of them inside instrument bubbles) rise from 14 % at 3 px/mm to 58 % at 6 and about 82 % from 8 px/mm. 82-99 % of the
objects of the reference drawing come out in one piece (1.13-1.87 DXF elements per object). Symbols that are missed
stay linework.

## Limits

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
- Symbol recognition adds about 20 s to an A1 sheet and 35 s to an A0 sheet with the starter library; larger
  libraries take longer.
