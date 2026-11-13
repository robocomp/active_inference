# Object colour, measured — design (2026-11-10, revised 2026-11-11)

Status: APPROVED by the owner 2026-11-11 — fixed palette first, tracked objects only, brightness included.

## Why

Live chat test chat_e2e2 (2026-11-10). Asked "go to the yellow chair", the robot looked through its camera and said
"I see a yellow chair in front of me". The two chairs in view were magenta. Asked again, it called them "black" and
"grey". The vision model followed the question, not the pixels, and nothing linked a colour in the picture to
`chair_34` or `chair_35`. A colour said to the person must be a **measurement attached to the object**, not a model's
reading of a picture.

## What already exists (found 2026-11-11; the first version of this design missed it)

- **retina** measures every mask's colour: `mask_color_summary` (`retina/src/graph_publisher.cpp`) publishes, per
  slice, the median CHROMATICITY (R,G,B)/(R+G+B) over the mask's interior grid cells (`mask_color_rgb`), its
  between-cell variance (`mask_color_var`) and the number of contributing cells (`mask_color_neff`).
- **common/mask_ingestor** carries them into each `MaskSlice` (`color_chroma`, `color_var`, `color_neff`).
- **common/appearance_belief** fuses them per instance (conjugate Gaussian, per-frame information capped as a
  common-mode, drift with time). chair_concept and table_concept keep one per instance.
- Its only output is `mesh_color_rgb`, the 3D viewer's tint, faded toward grey by confidence. Nothing names the
  colour, LTOM stores none, and language_agent reads none.

Chromaticity is invariant to the illumination gain by construction, so it separates hues (magenta, pink, yellow) but
cannot tell black from white from grey, and brown falls on orange. That needs **brightness**.

## Goals

- Every tracked object of a class whose agent keeps an appearance belief (chair first) carries a colour NAME with
  its probability; `find_objects`, `where_is` and the object lines say it ("chair_34, magenta p 0.93").
- LTOM keeps it, so a remembered object has its colour too.
- When no colour has been measured, the chat says it does not know, never a model's guess.

## Non-goals / anti-goals

- No classifier, no GPU; the concept agents do arithmetic only.
- No hard colour boundaries: the name is the most probable entry of a posterior, its probability reported.
- The appearance belief stays out of every geometric fit, association gate and existence belief
  (appearance_belief.h's containment rule). Naming only reads it.
- Untracked classes: not measured beyond what retina already does (owner 2026-11-11).
- Person-given colour names: later, on top of the palette (owner 2026-11-11).

## How

1. **retina** adds brightness to the same per-cell loop: per interior cell the median of (R+G+B)/(3·255), and per
   slice the median over cells and the between-cell variance, with the same 8-bit quantisation floor.
   Published as `mask_color_lum`: 3 floats per slice (lum, var, n). n counts its own cells: a pure-black cell has
   brightness but no chromaticity, and a black chair must be nameable (the naming then drops the hue dimension).
2. **mask_ingestor** reads it into `MaskSlice::color_lum`, `color_lum_var` (absent attribute ⇒ zero information).
3. **appearance_belief** grows a fourth channel, brightness, fused exactly as the chromaticity channels (same
   per-frame cap, same drift). Brightness is NOT illumination-invariant: its drift and the palette's width carry
   that (see 4); the Webots light is fixed, the real robot's is not (named risk).
4. **Naming** (`common/appearance_belief/colour_name.h`): a fixed palette of 12 basic terms — red, orange, yellow,
   green, blue, purple, magenta, pink, brown, white, grey, black — each a prototype (r, g, L) from an sRGB value.
   Likelihood of each term: a Gaussian whose covariance is the belief's variance PLUS the term's own spread (how
   wide the word is), equal priors; the reported name is the argmax, its posterior probability goes with it. A dark
   surface has a wide chromaticity variance from retina's floor, so near-black is decided by brightness alone,
   without any rule saying so.
5. **chair_concept** publishes `color_name` (string) and `color_prob` (float) on the chair node next to
   `mesh_color_rgb`, only once its appearance belief is initialised.
6. **LTOM** reads them per sighting, keeps the latest per object, persists them in the repertory and publishes them
   on the memory graph.
7. **language_agent**: live and remembered object lines add the colour ("magenta, p 0.93"); `find_objects` and
   `where_is` accept a colour word; a question about colour with none measured answers "I have not measured its
   colour". `look` stays for open questions about appearance; its colour words never pick an object.

## Constraints

- New attributes (`mask_color_lum`, `color_name`, `color_prob`) need `REGISTER_TYPE` lines in cortex's
  `dsr_attr_name.h`, and the owner reinstalls cortex. Code using them does not compile before that.
- Rates before features: the brightness median adds one sort per interior cell to a loop that already does three;
  retina's rate is measured before and after.

## Calls made by the builder (owner to check)

- Palette prototypes and term width (σ_chroma 0.04, σ_L 0.15, the same for every word) — chosen from sRGB values, to
  be checked against the measured spread of one chair's colour over a tour. A width growing as 1/L for dark words was
  tried and dropped: it spread "black" so thin that a narrower "brown" won on a near-black object; darkness already
  widens the hue through retina's quantisation floor.
- Brightness packed as 3 floats per slice in one attribute rather than three attributes.
- `find_objects`/`where_is` with a colour keep objects measured that colour AND objects not measured (they may be it,
  and their line says "colour not measured"); a different measured colour excludes. Every object line now ends with
  its colour or "colour not measured", so the model never fills the gap.
- retina's `mask_color_summary` moved to `retina/src/mask_color_summary.h` (unchanged apart from brightness) so it has a
  test; the palette's names live in `common/appearance_belief/colour_terms.h` (no Eigen) for the language agent.
- LTOM keeps the LATEST measured name, not a fused one: the agent's belief is already the fusion.
- **Foreign-mask frames (added 2026-11-11 after the live traces).** Single frames of a neutral surface (wall, table:
  chromaticity 0.346/0.334/0.320, hundreds of cells, tiny spread) credited to chair_35 renamed it "grey"/"white". The
  first proposal, an extra between-frame variance per frame, did NOT remove those flips when the traces were replayed.
  What did: `AppearanceBelief::observe()` explains each frame as either this object (its predictive density) or
  something else (`p_foreign_mask` = 0.05, uniform density) and counts it with its probability of being the object.
  Replayed on three traces (2 700 frames): no flips, and the result is the same for any prior from 0.005 to 0.2. A lasting change
  is still followed through the drift (about 3 s); a wrong FIRST frame is corrected the same way. The trace gains a
  `w` column (that probability).
