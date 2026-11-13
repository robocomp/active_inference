# An unrecognised doorway is a hypothesis, not an extension of the room

Status: approved and built 2026-11-09 (owner: "do it"), uncommitted, as `room_concept/src/doorway_necks.h`. See §7 for how the build differs.
Related: `2026-11-05-structure-from-evidence-design.md` §3.2 (door birth by comparison, the typical-door prior),
`2026-11-05-exploration-tour-design.md`, commit 79fc14b1 (refiner: no edit through a *known* doorway).

## 1. The principle (user, 2026-11-06)

Missing a door is something that will happen. It is not an estimation failure. It is a state of knowledge the
epistemic planner can act on. The estimator must keep the alternative open instead of deciding it silently.

## 2. Where the estimator decides it silently today

The post-promotion `LayoutRefiner` (layout_refiner.h) has one explanation for swept-free cells outside the outline:
the room is bigger (`extend`, scored by two-part MDL). The per-beam door filter keeps the next room's floor out of
the evidence only for doors door_concept has found. Through an opening it has not found, the next room's floor is
evidence the room is bigger, and the extend is adopted. Two harms follow:
- the room's outline (and its LTSM memory) is wrong, and
- the next room, when entered, has that block excluded from its free-space map (FreeSpacePolygon::set_exclusion).

⚠ Honest status of the motivating case: tour16 looked like this (room_2 grew 11 m² into room_3 next to the
never-found DOOR_2). It turned out to be a frame bug: plain SLAM was not stepped while a crossing dead-reckoned,
fixed in 40c92029. So there is NO live instance yet. The design stands on the principle. Its first test is
synthetic (§5), and live cases must be collected as they occur (watch: an extend adjacent to a gap in a measured
wall, while door_concept lists no door there).

## 3. The model

For each `extend` candidate B (a box outside the current outline V) the refiner compares two explanations of the
same evidence, in the same units (nats, the MDL it already uses):

- **H_room**: today's term. Outline V ∪ B, data NLL under it, plus κ·Δ|V| plus ln n (the edit's address).
- **H_door**: V is unchanged. B is seen through an opening O in V's boundary. O is the part of the shared boundary
  ∂V ∩ ∂B where beams PASSED (free cells on both sides, no surface cells). The wall on either side of O is measured
  (surface cells). Data term: B's free cells are another room's floor and cost V nothing. The opening's own cells are
  no surface, so they cost V the same outlier term today's model charges a gap in a modelled wall. Model term:
  − log P(door in this wall) − log p(width(O) | door) + (address ln n). The prior is the one §3.2 of
  structure-from-evidence fixes: width 0.85 ± 0.12 m, the per-wall base rate (first door .4, further .01, to confirm).

The decision is the posterior odds, not a threshold:
- ΔF = F(H_room) − F(H_door).
- ΔF ≪ 0: adopt the extend (a real bay of the room seen through a wide mouth).
- ΔF ≫ 0: do not extend. Publish a **door candidate** at O.
- In between: neither. The outline is not edited across O, and O is published as a door candidate with
  P(door) = σ(ΔF). The planner sees the uncertainty.

Why this is not a threshold in disguise: a 2.5 m wide mouth pays a large −log p(width | door) (several σ out), so
H_room wins there. A 0.9 m gap with measured wall on both sides pays almost nothing, so H_door wins. Neither is a
cutoff. "Measured wall on both sides" enters through the surface likelihood H_room must pay (the flanking wall
cells end up inside V ∪ B when B extends through the wall line) rather than as a test.

## 4. What changes downstream

1. **Exclusion follows the belief.** Only the settled outline V is excluded from the next room's map, never a
   contested block. Room_3's lost corner in tour16 would have been kept even had the extend been real.
2. **Door candidate → epistemic value.** The candidate is published (one small node, or an attribute on the room,
   owner room_concept; to decide). door_concept treats it as a prior location for a door look. Under §3.2 that look
   is a comparison F(exists) vs F(not) seeded at O. The explorer gains a viewpoint whose expected information is
   the entropy of P(door at O).
3. **Exploration drive.** A door candidate counts as *found but unconfirmed*. Before leaving a promoted room the
   drive may spend a look on it (bounded by MaxRoomS as everything else). A confirmed door joins the normal list.
   A rejected one closes the opening, and the extend may come back on later evidence.

## 5. Tests (written first)

- Synthetic: rectangle room, 0.9 m gap in one wall, free cells of a second room seen through it. Expect no extend
  and a door candidate at the gap with P(door) > ½.
- Synthetic: a 2.4 m wide alcove (bay) seen through its full-width mouth. Expect the extend adopted and no door
  candidate.
- Synthetic: the same 0.9 m gap with NO wall measured on one side (unseen). Expect P(door) near the prior. No edit
  either way.
- Replay: tour15's 57.9 s refiner input with DOOR_0 removed from the known doors. The 0.3 m spike becomes a door
  candidate at DOOR_0, not an extend (it is the case that motivated 79fc14b1 from the other side).
- Exclusion: a contested block is not excluded from the next room.
- Live: tours as usual. Count door candidates, how many become doors, and how many rooms grow through a gap.

## 6. Open points for the user

- Who owns the door candidate: room_concept (the refiner produced it) or door_concept (doors are its concept)? My
  default: room_concept writes the hypothesis, door_concept owns any door born from it (Lifecycle: birth stays
  tracker-mediated).
- Should promotion (manhattan_layout + description-length choice) use the same comparison, or only the refiner? A
  leak can also happen before promotion. Default: refiner first, promotion once the refiner version is measured.

## 7. As built (2026-11-09)

The live instance: clean26, start room_2, took in 4.8 m2 of room_3's floor at promotion, seen through DOOR_2 before it was found. room_3 then lost about 12 m2 and the robot stalled.

- **Where it runs.** On the outline at promotion only (§6: the leak was there). Cutting after a refiner edit was tried and removed: in clean30 two post-promotion cuts landed next to a door in use, and its leaf estimate collapsed seconds later (P(open) 0.00, robot held at the mouth); the link was not proved. `explain_openings` tries each neck: a chord between two outline vertices along the room's axes.
- **The score.** `DL(outline) - DL(outline without B) + ln P(door) - (w-w0)^2 / 2(s0^2+s_w^2)`, on the refiner's own description length and on the cells outside B. Door wins when positive. Width noise is the LiDAR raster's 5 cm cell, not the camera mask's 0.20 m (at 0.20 m a real 2.7 m neck of room_1 is cut; chosen after seeing that, on the same data).
- **A neck needs floor beyond it.** Without free cells past the wall's band there is no door hypothesis (a kink in a wall has none).
- **Ownership (§6).** room_concept writes `room_doorways` (metaconcept: parent = room, delimiting_polygon_x/y = opening end points). door_concept turns each into a pending candidate, so the look of `2026-11-09-undecided-candidate-look-design.md` resolves it, and birth stays tracker-mediated.
- **The refiner may not grow through a hypothesised opening** (`grows_through_doorway`, as for a known door).

Flagged: the width prior (1.00 +/- 0.06 m) is this simulated world's; at +/- 0.12 a 1.74 m bay of room_1 would be cut again. P(door) is the first-door base rate on every wall. A neck is a vertex-to-vertex chord, so a neck ending mid-edge is not tried. Not built: a rejected opening does not let the outline grow back. Hypotheses reach DoorCore outside the input tape, so a replay does not see them.

Offline, on the refiner's recorded inputs: clean26's outline 29.9 -> 25.1 m2 (true 25.1). Passing tours keep every room, except a 0.4 m2, 0.92 m wide nook of room_1 (clean20, clean22, and one of that size in clean26), which is not at a door. Cost 40-310 ms, once per promotion or adopted edit.
