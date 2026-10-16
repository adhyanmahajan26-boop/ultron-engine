# B-roll fetching

`scripts/fetch_assets.py` pulls stills from Openverse and Wikimedia Commons.
No API key. Every download records source URL, creator and licence in
`assets.json` beside the files.

## Hard-won rules

**Openverse rejects `page_size > 20` with HTTP 401.** Not 400, not a clamp — a
401 that reads like an auth failure. Exceeding it silently empties every
result set and the pipeline falls through to Wikimedia page scans. Always
`min(n, 20)`.

**Ask for concrete, photographable nouns.** Stock search matches objects, not
ideas. `write_script.py` instructs Ornith accordingly, and it is the single
biggest factor in relevance.

- Good: `server rack data center room`, `cpu processor chip closeup`
- Bad: `person talking to camera`, `abstract network connections`,
  `privacy and freedom`

Abstract queries returned a woman lying on a bed for "person talking to
camera". Never request people speaking — the presenter is already on screen.

**Commons indexes PDFs, DjVu, SVG and maps as "images".** They render as
colourful landscape pages that pass pixel heuristics. Reject by title
extension and keyword before downloading.

**Filter tiers must degrade.** `size=large` + `aspect_ratio=wide` +
`license_type=commercial` gives the best-looking results but returns nothing
for many queries. Try strict first, then relax. `license_type=commercial` also
drops NC/ND results that would otherwise need manual vetting.

**Rank before downloading.** Score candidates by query words appearing in the
title, and prefer Openverse over Wikimedia — Wikimedia skews to documents and
diagrams.

**Dedupe across scenes.** The same shot twice reads as a mistake.

## `looks_like_photo()` thresholds

Calibrated against known-good photos vs known-bad page scans:

| Check | Threshold | Why |
|---|---|---|
| portrait (`h > w`) | reject | every bad scan was portrait; cleanest signal |
| width | `< 500` reject | too small to fill 1280x720 |
| near-white pixels | `> 45%` reject | paper |
| saturation | `< 3` reject | flat greyscale text plate |

**Do not raise the saturation threshold.** A legitimate dark server-room photo
scored 5.4 while a colour document scored 27.6 — saturation does not separate
them. Orientation does.

## Known ceiling

Openverse and Wikimedia are free CC corpora, not stock libraries. Expect
"Wikipedia documentary" rather than polished b-roll: real but plain, and
occasionally a miss. This is a corpus limitation, not a code one.

Better options, if the user ever wants them:
- **Pexels / Unsplash APIs** — free key, proper b-roll aesthetics. The real fix.
- **Local image generation** — fully offline and always on-topic, at the cost
  of another model and render time.
