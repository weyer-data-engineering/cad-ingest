# Profiles

What depends on the kind of drawing, a standard or a customer is kept out of the code, as data: symbol libraries
([Symbol libraries](symbol-libraries.md)) and profiles. A profile names a kind of drawing, the evidence its texts give
of it and the vocabulary its texts are written in. The recognition stays generic; a profile only says what the words of
one kind of drawing look like.

cad-ingest has four built-in profiles in `profiles/`:

| File | Kind (`drawing`) | Evidence | Vocabulary | A single character in a frame |
|---|---|---|---|---|
| `building.toml` | `building` | titles such as "Grundriss", "Schnitt"; room areas ("12,40 m²"); scales 1:20 to 1:200 | levels ("±0,00", "+2,75"), areas | any (the circles of the axes hold letters and digits) |
| `electrical.toml` | `electrical` | titles such as "Stromlaufplan"; device tags ("-K1", IEC 81346); "L+"; supply ("24 V DC") | device tags | any (current paths are numbered in boxes) |
| `pid.toml` | `pid` | titles "R&I", "P&ID"; nominal sizes ("DN 50"); circles with text in them (instrument bubbles) | nominal sizes and pressures ("DN50", "PN16"), line numbers ("DN80-P-1011"), equipment ("P-101", "B-501"; not with I or O) | any (letters such as an actuator's M or an interlock's I, digits in note hexagons and revision triangles) |
| `site.toml` | `site` | titles such as "Lageplan", "Flurstück"; scales 1:250 to 1:1000 | parcel numbers ("123/4"), heights | any |

## What a profile does

- **The kind of drawing.** Where a job does not name it (`options.drawing`), each profile weighs the evidence on the
  sheet: a title counts 5, a typical scale 3, every other match 1. The kind with the most evidence (at least 1) wins;
  on a tie, the profile whose file name comes first. The kind chooses the symbol libraries: a library is used on the
  kinds it names in its `drawing` list.
- **Reading tokens in the vocabulary.** The text recognition confuses letters and digits that look alike (O and 0, I
  and 1, S and 5, B and 8, Z and 2, G and 6). A token (a word of a text) that matches no word of the vocabulary as it
  was read, but does with up to four such characters swapped, is read so: "-KI" as "-K1" on a circuit diagram,
  "DNSO" as "DN50" on a P&ID. A token that matches nothing either way stays as it was read. This is the step
  `vocabulary` ([Steps that can be switched](photos-and-scans.md#steps-that-can-be-switched)).
- **Look-alike characters the vote left open.** Characters of look-alike glyphs (0 and 8, B and 8, D and 0) are put
  to the vote of the glyphs most like them on the sheet. Where the vote had a say in a token, the token decides
  where it can: a token that is a word of the vocabulary in only one of its readings, as read or as voted, takes that
  one ("DN80-P-1011", not "0N80-P-1011"; "B-501", not "8-501"). A token it does not decide stays as voted, and the
  text is marked for review where the vote had no clear majority. This is the step `lookalike_tokens`.
- **The way round a text reads.** A vertical or oblique text is read both ways round. Where both readings are
  plausible, the one with more tokens that are words of the vocabulary is taken: "0,60" rather than "09'0" (the same
  text upside down) on a building drawing. Where that does not tell, the surer reading stays. A vertical text the
  text spotter was unsure of, which read again only reads the same more surely, stays marked to check unless each of
  its tokens is a word of the vocabulary: "3.94" on a building drawing, whose numbers have a decimal comma. This is
  the step `text_direction`.
- **A single character in a frame.** The text recognition reads a letter for a digit that looks like it now and then
  (O and 0, I and 1, S and 5, B and 8, Z and 2, G and 6). Where a kind of drawing frames single letters only, or single
  digits only, `[reading] framed` says so, and a single character alone in a closed frame (a circle, a diamond, a box)
  is read as the letter or the digit it may be. This is the step `framed_letters`, off by default. The built-in
  profiles all say `any`: the axis circles of a building drawing hold letters and digits, and a P&ID frames letters
  (an actuator's M, an interlock's I) as well as digits (note hexagons, revision triangles). Set it in your own profile
  where your drawings frame only one of them.
- **Steps switched off for the kind.** `[steps] off` names steps of the recognition ([Steps that can be
  switched](photos-and-scans.md#steps-that-can-be-switched)) that do not suit the kind. They are off on a sheet of the
  kind, recognised or named in `options.drawing`, unless the job names them in `options.on`; `report.json` lists those
  switched off in `steps_off_by_kind`, and the markdown says so. Only the steps that run once the kind is known can be
  switched off so: `dimensions`, `vocabulary`, `framed_letters`, `dedupe`, `square_lines`, `title_block`,
  `unclassified`, `width_calibration`, `pens` and `read_handwriting`; the others run, at least in part, before the
  texts are read, and a profile that names one cannot be read. The
  built-in `pid.toml` and `electrical.toml` switch `dimensions` off: P&IDs and circuit diagrams carry no dimension
  chains, and a line number between two flanges, or a value between two terminals, would be taken for a dimension.

## Format

A profile is a TOML file. Patterns are Python regular expressions; those under `[evidence]` are searched in each text,
those under `[vocabulary]` must match a whole token. The names of the vocabulary lists are free and only for people.

```toml
format = 1                       # the profile format, 1
id = "acme-pid"                  # unique; a profile of the id of a built-in one replaces it
name = "P&ID of Acme"            # for people
drawing = "pid"                  # the kind of drawing; symbol libraries name the kinds they are for

[evidence]
titles = ['R&I', 'P&ID']         # each found anywhere on the sheet counts 5
patterns = ['\bDN ?\d+']         # each match counts 1
scales = []                      # drawing scales typical of the kind; the sheet's scale counts 3
bubbles = true                   # each circle 3 to 8 mm in radius with text in it counts 1

[vocabulary]
tags = ['[A-Z]{2,4}-\d{3,5}[A-Z]?']
lines = ['\d{1,2}"-[A-Z]{2}\d{2}']

[reading]
framed = "letter"                # a single character alone in a frame: "letter", "digit" or "any" (the default)

[steps]
off = ["dimensions"]             # steps switched off on the kind unless a job switches them on (default: none)
```

Keep vocabulary patterns specific: a pattern that matches almost anything ("\w+") changes nothing, but one that
matches a word only with a digit in it turns letters of real words into digits. A profile is data only; no code of a
profile is ever run.

## Adding profiles

Profiles (`*.toml`) in the folders of the setting `PROFILES` (separated by `:`) are read after the built-in ones; one
with the id of a built-in profile replaces it. A profile that cannot be read stops the conversion with its file name and
the reason. The kinds `building`, `electrical`, `pid` and `site` can be named in `options.drawing`; a profile of a new
kind is used when the sheet's evidence speaks for it.
