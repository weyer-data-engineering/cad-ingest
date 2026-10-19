# Symbol libraries

When cad-ingest turns a photo or scan of a P&ID or a floor plan into DXF (`vectorize`), it replaces recognised
symbols with blocks of a symbol library: a valve becomes a valve block with its tag as an attribute, a washbasin a
washbasin block, not loose strokes. cad-ingest ships two starter libraries: `din-10628` with 18 P&ID symbols after
DIN EN ISO 10628-2, and `floor-plan` with seven fixtures of floor plans (toilet, washbasin, bathtub, shower tray,
kitchen sink, hob, washing machine). You can add your own company libraries next to them, so that the DXF contains
your blocks, with your layers, colours and attribute names.

A library is data, not code: nothing in a library is ever run. A library that a job brings along is read like the
job's own file, in its work-horse under the same time and memory limits.

## What a library looks like

A library is a folder with a short description and a DXF file:

```
acme-pid/
  library.toml   what the library is
  symbols.dxf    one block per symbol
  LICENSE        optional, the terms under which others may use it
```

The same folder as a ZIP works too, and so does a bare DXF file (its file name is then its id).

`library.toml`:

```toml
id = "acme-pid"                          # unique: lower-case letters, digits and hyphens
name = "ACME P&ID symbols"
version = "2.1.0"
format = 1                               # the library format; this version of cad-ingest reads format 1
licence = "LicenseRef-ACME-internal"     # SPDX identifier, e.g. CC-BY-4.0
drawing = ["pid"]                        # the drawings it is for: "pid" (P&IDs), "building" (floor plans) or both
size = "printed"                         # blocks drawn at their size on the sheet ("printed") or of the object ("real")
file = "symbols.dxf"
```

Every field is optional. Without one, `id` comes from the folder or file name, `name` is the id, `format` is 1,
`drawing` is `["pid"]`, `size` is `printed` and `file` is `symbols.dxf`. Unknown fields are ignored, and
`cad-ingest library check` points them out.

**P&ID symbols and floor plan fixtures.** A library is used on the kind of drawing it is for: a P&ID gets the
libraries with `"pid"`, a floor plan those with `"building"`. P&ID symbols are drawn at the size they have on the
sheet (`size = "printed"`, a valve about 5 mm long). Fixtures and furniture in architectural CAD libraries are drawn at
the size of the real object (`size = "real"`, a bathtub 1700 mm long). cad-ingest shrinks real-size blocks by the
drawing scale, which it reads from the title block or takes from the job's `scale` option; while the scale is
unknown, it tries 1:50 and 1:100. The DXF holds the blocks at real size either way.

## The blocks

Each block of the DXF is one symbol around its insertion point (the block's base point), at its printed or its real
size (see `size` above):

| Element in the block | Meaning |
|---|---|
| lines, arcs, circles, polylines, splines, ellipses | the outline that is looked for on the sheet |
| `POINT` on the layer `PORT` | where a pipe connects |
| `LINE` on the layer `THROUGH` | a line that runs through the symbol on the sheet, like the pipe under a flow arrow; required for a match, not part of the symbol |
| `ATTDEF` | an attribute, placed where its text stands (a tag below a valve, the letters in an instrument bubble) |
| blocks inside the block | their outline counts as part of the symbol |
| block description | the name of the symbol in the markdown |

Everything else, such as hatches, solid fills and fixed texts, is not used to find the symbol, but it stays in the
block that goes into the DXF. The DXF always gets your original block.

The units come from the DXF header (`$INSUNITS`): a library drawn in inches is converted. Without units, millimetres
are assumed. Blocks whose names start with `*` or `_` (anonymous blocks and dimension arrowheads) are not symbols.

What makes a symbol easy to find:

- **Ports.** Most P&ID symbols are found where pipe ends meet their ports. Put the insertion point of a P&ID symbol
  in the middle, not on a port: a port is recognised by its direction from the insertion point.
- **Rectangles.** Fixtures and furniture are found from a rectangle of their outline: the box of a washbasin, the
  cistern of a toilet. A side may be part of a longer line, such as the wall face behind a washbasin or the front edge
  of a counter, and two opposite sides are enough where OCR read the others as letters. Circles (pumps, instruments,
  hotplates) are anchors too. A symbol with neither ports, rectangle nor circle is only found where its strokes stand
  apart from other linework.
- **No lines close along a side.** At 1:100, a line a few centimetres inside the edge of a fixture looks like the
  second face of a wall: a washing machine with a line for its control panel may be found where a washbasin hangs on
  a thin wall. Leave such lines out of the outline, or draw them well inside.
- **Fixtures with their back on the wall.** Put the insertion point of a fixture in the middle of its back, the side
  against the wall, as the `floor-plan` library does. When a fixture looks the same turned or mirrored (a hob, a
  shower tray), cad-ingest places it so that its insertion point lies on the wall.
- **Distinct shapes.** Two symbols that differ only by a fill or a letter inside cannot be told apart yet. The
  self-test of `cad-ingest library check` shows such mix-ups.
- **Attributes where the text stands.** A text within about 4 mm of an attribute position becomes that attribute.
  For a symmetric symbol, such as a valve that looks the same turned half way, the attributes decide which way round
  it is placed.

## Making a library from existing blocks

`cad-ingest library build` collects existing CAD blocks into a new library folder:

```bash
# one DXF or DWG file with many blocks
cad-ingest library build acme-blocks.dwg --out acme-pid --name "ACME P&ID symbols" --licence LicenseRef-ACME-internal
# a folder with one drawing per symbol, as AutoCAD block libraries often are: each file becomes a symbol named
# after the file, inserted at the drawing's base point ($INSBASE)
cad-ingest library build symbols/ --out acme-pid --file-per-symbol
```

Inch drawings are converted to millimetres. Where two blocks have the same name, the first one is kept. The command
lists the symbols it collected and what is still missing. In most cases you then open `symbols.dxf` in your CAD
program, add `POINT`s on the layer `PORT` where pipes connect and `ATTDEF`s where tags stand, and save it as DXF.

In the image, run the commands with your files mounted:

```bash
docker run --rm --user "$(id -u)" -v "$PWD:/data" -w /data ghcr.io/weyer-data-engineering/cad-ingest \
  cad-ingest library build acme-blocks.dwg --out acme-pid
```

## Checking a library

`cad-ingest library check` reads a library, lists its symbols and notes what may keep a symbol from being found:

```
$ cad-ingest library check acme-pid --no-self-test
acme-pid 2.1.0: ACME P&ID symbols
Licence: LicenseRef-ACME-internal. Drawing types: pid. Blocks drawn at printed size; one drawing unit is 1 mm.

2 symbols:
  MY_VALVE          6.0 x    3.0 mm, 2 ports, attributes TAG  Box valve
  VALVE_AT_PORT     5.0 x    3.2 mm, 2 ports  Gate valve, inserted at a port

Notes:
  - VALVE_AT_PORT: the insertion point lies on a port; put it in the middle of the symbol, because a port is recognised by its direction from there
```

Then it tests the library. Every symbol is drawn twice, as in the block and turned and mirrored, with pipes at its
ports and a short text at each attribute, on synthetic A3 sheets. Each sheet is turned into a simulated 200 dpi scan
and a simulated 12 MP phone photo and read with this library alone:

```
Self-test: 18 symbols, each as drawn and turned and mirrored, on 1 A3 sheet, as a 200 dpi scan and as a 12 MP phone photo:

  symbol              scan  photo  mistaken for
  GATE_VALVE          2/2   2/2
  ...
  ORIFICE             0/2   0/2

Found: scan 94% (34/36), photo 94% (34/36). Found where no symbol was drawn: scan 0, photo 0.
Time per sheet: scan 9.5 s, photo 18.7 s, of which symbol recognition 2.9 s and 2.7 s.
```

A library for floor plans is tested on floor plan sheets, a real-size library at 1:50 (`--scale 100` for 1:100).
`--out folder` writes the review overlay of every sheet, with the symbols that were found in teal. `--no-self-test`
only checks the format, which takes a second; the self-test takes about half a minute per sheet. The command exits
with 1 if the library cannot be read.

The output above is the starter library. It shows a known weakness: the orifice plate, two short bars across a pipe,
is not found yet.

## Using libraries

**Command line.** `--library` names a library, either an installed one by id or a folder, ZIP or DXF. Repeat it for
several. Where two libraries have a block of the same name, the earlier one wins:

```bash
cad-ingest plan.jpg out/ --paper A1 --library ./acme-pid --library din-10628
```

**Service, installed libraries.** Workers read the libraries in `SYMBOL_LIBRARIES` (default `/libraries`) when they
start, in addition to the built-in starter libraries. Mount a folder there or build an image of your own (see
[Operations](operations.md#symbol-libraries)). `DEFAULT_SYMBOL_LIBRARIES` (default `din-10628,floor-plan`) names the
libraries a job gets when it names none. A job chooses others with `"options": {"libraries": ["acme-pid",
"din-10628"]}`, and `GET /libraries` lists what is installed. Each library is used on the kind of drawing it is for.

**Service, libraries a job brings along.** `"library_urls": ["https://storage/acme-pid.zip?sig=..."]` downloads one or
more libraries (a ZIP or a DXF) with the job, for example a library that each tenant of a platform keeps in its own
storage. They come first, then the installed ones the job names. The default libraries are only used when the job
names no library at all, so add `"libraries": ["din-10628", "floor-plan"]` to keep the starter symbols. A library a
job brings along is read in its work-horse under the same limits as the file, and never stored.

The `report.json` of every result lists the libraries that were used for the kind of drawing, with their versions,
and for every symbol the library it came from.

## Sharing a library

A library is a folder, so it can live in a Git repository or be passed on as a ZIP. Give it a licence if others may
use it. Company libraries can stay private: mounted on your own servers or passed with the job, they are never
published.
