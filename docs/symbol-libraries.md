# Symbol libraries

When cad-ingest turns a photo or scan of a P&ID into DXF (`vectorize`), it replaces recognised symbols with blocks of
a symbol library: a valve becomes a valve block with its tag as an attribute, not six loose strokes. cad-ingest ships
a starter library with 18 symbols after DIN EN ISO 10628-2. You can add your own company library next to it, so that
the DXF contains your blocks, with your layers, colours and attribute names.

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
drawing = ["pid"]                        # the drawing types the library is for; symbols are found on P&IDs so far
file = "symbols.dxf"
```

Every field is optional. Without one, `id` comes from the folder or file name, `name` is the id, `format` is 1,
`drawing` is `["pid"]` and `file` is `symbols.dxf`. Unknown fields are ignored, and `cad-ingest library check`
points them out.

## The blocks

Each block of the DXF is one symbol, drawn at the size it has on the printed sheet, around its insertion point (the
block's base point):

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

- **Ports.** Most symbols are found where pipe ends meet their ports. A symbol without ports and without a circle
  is only found where its strokes stand on their own, and not at all if it is larger than 25 mm.
- **Insertion point in the middle.** A port is recognised by its direction from the insertion point. Put the
  insertion point in the middle of the symbol, not on a port.
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
Licence: LicenseRef-ACME-internal. Drawing types: pid. One drawing unit is 1 mm.

2 symbols:
  MY_VALVE         6.0 x   3.0 mm, 2 ports, attributes TAG  Box valve
  VALVE_AT_PORT    5.0 x   3.2 mm, 2 ports  Gate valve, inserted at a port

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
Time per sheet: scan 9.2 s, photo 19.4 s, of which symbol recognition 2.4 s and 3.0 s.
```

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
start, in addition to the built-in starter library. Mount a folder there or build an image of your own (see
[Operations](operations.md#symbol-libraries)). `DEFAULT_SYMBOL_LIBRARIES` (default `din-10628`) names the libraries a
job gets when it names none. A job chooses others with `"options": {"libraries": ["acme-pid", "din-10628"]}`, and
`GET /libraries` lists what is installed.

**Service, libraries a job brings along.** `"library_urls": ["https://storage/acme-pid.zip?sig=..."]` downloads one or
more libraries (a ZIP or a DXF) with the job, for example a library that each tenant of a platform keeps in its own
storage. They come first, then the installed ones the job names. The default libraries are only used when the job
names no library at all, so add `"libraries": ["din-10628"]` to keep the starter symbols. A library a job brings
along is read in its work-horse under the same limits as the file, and never stored.

The `report.json` of every result lists the libraries that were used, with their versions, and for every symbol the
library it came from.

## Sharing a library

A library is a folder, so it can live in a Git repository or be passed on as a ZIP. Give it a licence if others may
use it. Company libraries can stay private: mounted on your own servers or passed with the job, they are never
published.
