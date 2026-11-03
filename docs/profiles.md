# Profiles

What depends on the kind of drawing, a standard or a customer is kept out of the code, as data: symbol libraries
([Symbol libraries](symbol-libraries.md)) and profiles. A profile names a kind of drawing, the evidence its texts give
of it and the vocabulary its texts are written in. The recognition stays generic; a profile only says what the words of
one kind of drawing look like.

cad-ingest has four built-in profiles in `profiles/`:

| File | Kind (`drawing`) | Evidence | Vocabulary |
|---|---|---|---|
| `building.toml` | `building` | titles such as "Grundriss", "Schnitt"; room areas ("12,40 m²"); scales 1:20 to 1:200 | levels ("±0,00", "+2,75"), areas |
| `electrical.toml` | `electrical` | titles such as "Stromlaufplan"; device tags ("-K1", IEC 81346); "L+"; supply ("24 V DC") | device tags |
| `pid.toml` | `pid` | titles "R&I", "P&ID"; nominal sizes ("DN 50"); circles with text in them (instrument bubbles) | nominal sizes and pressures ("DN50", "PN16") |
| `site.toml` | `site` | titles such as "Lageplan", "Flurstück"; scales 1:250 to 1:1000 | parcel numbers ("123/4"), heights |

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
```

Keep vocabulary patterns specific: a pattern that matches almost anything ("\w+") changes nothing, but one that
matches a word only with a digit in it turns letters of real words into digits. A profile is data only; no code of a
profile is ever run.

## Adding profiles

Profiles (`*.toml`) in the folders of the setting `PROFILES` (separated by `:`) are read after the built-in ones; one
with the id of a built-in profile replaces it. A profile that cannot be read stops the conversion with its file name and
the reason. The kinds `building`, `electrical`, `pid` and `site` can be named in `options.drawing`; a profile of a new
kind is used when the sheet's evidence speaks for it.
