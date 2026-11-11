# Object colour, measured — short design (2026-11-10)

Status: PROPOSAL for the owner. Nothing built.

## Why

Live chat test chat_e2e2 (2026-11-10). Asked "go to the yellow chair", the robot looked through its camera and said
"I see a yellow chair in front of me". The two chairs in view were magenta and pink. Asked again, it called them
"black" and "grey". The vision model followed the question, not the pixels, and nothing linked a colour in the
picture to `chair_34` or `chair_35`. A colour said to the person must be a **measurement attached to the object**,
not a model's reading of a picture.

## Goals

- Every object a concept agent tracks carries its measured colour; `find_objects`, `where_is` and the object lines
  say it ("chair_34 (magenta)"), and "go to the yellow chair" becomes a lookup over the objects' colours.
- The long-term memory (LTOM) keeps it, so a remembered object has its colour too.
- When no colour has been measured, the chat says it does not know (never a model's guess).

## Non-goals / anti-goals

- No classifier, no GPU, nothing in the concept agents but arithmetic (owner's rule: perception that needs pixels is
  computed in retina).
- No hard colour boundaries (`if hue > k`): the name is a posterior, its probability reported.
- Not a material or texture model; one dominant colour per object.

## How

1. **Retina measures, per mask, from the mask's own pixels** (the same loop that collects the mask's 3D points in
   `graph_publisher.cpp`; the box is NOT used: it holds floor and wall too). It converts the pixels to CIELab and
   publishes, per slice, the mean Lab and its covariance (6 floats + count) next to the existing per-slice arrays.
   The covariance carries what the colour is worth: a small, uniform mask is a tight measurement; a mixed or dark one
   is wide.
2. **The shared mask ingestor fuses it** (`common/mask_ingestor`, used by every concept agent): each associated
   slice updates the instance's colour by a precision-weighted (Gaussian) update; lighting and viewpoint variation
   enter as a common-mode covariance added to each measurement, so one view cannot collapse it (the house pattern).
3. **The object node gets two attributes**: `color_lab` (mean, 3 floats) and `color_name` with its probability.
   Naming is a posterior over a small fixed palette (about 11 basic colour terms, prototypes in Lab, each a Gaussian
   likelihood); the reported name is the most probable, and its probability goes with it. A flat posterior reads
   "unclear colour", which is a statement of the model, not a threshold.
4. **LTOM stores** `color_lab` and the name per object, and the memory graph publishes them; `load_remembered`
   reads them.
5. **language_agent**: `object_line` and `remembered_line` add the colour ("chair_34 (magenta, p 0.9)");
   `find_objects` gains a colour filter; an answer about colour with no measured colour says "I have not measured
   its colour". `look` stays for open questions about appearance, but its colour words are never used to pick an
   object.

## Constraints

- New attributes need `REGISTER_TYPE` lines in cortex's `dsr_attr_name.h`, and the owner reinstalls cortex
  (CLAUDE.md). Type-attributed access only.
- Rates before features: the Lab conversion is per mask pixel on the CPU; measure retina's rate before and after,
  and the change must not drop it.
- Lighting: Webots light is fixed; on the real robot the light changes, which the common-mode covariance absorbs in
  part. Colour constancy (white balance from the scene) is left out of this design and named as a risk.

## Left to the builder

The palette and its Lab prototypes; the size of the common-mode covariance (from data: the spread of one object's
colour over a tour); the wire layout of the per-slice floats; how the tool phrases the probability.

## Open questions for the owner

1. Is a fixed basic-colour palette enough, or should colour names come from the person ("this is the yellow chair")
   as instance names do?
2. Should retina also publish the colour for masks of classes no concept agent tracks (so `look`-free answers cover
   them)?
