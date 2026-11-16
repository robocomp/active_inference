# Why real countable objects get absence evidence — offline carve replay (2026-11-15)

Script: `2026-11-15-carve-replay.py RUN_DIR [gt]` (scratch quality; kept as evidence, the builder rewrites it).
Data: `room_concept/estimation_scripts/crossing/objects_live6` (main checkout). No Webots needed.

## Method

- Cycles: every other cycle on which a real object's existence L fell while LiDAR absence was non-zero ("drop cycles").
- Robot pose: Webots ground truth (`gt_pose.tsv`, ~1 Hz, interpolated). Helios at 1.1075 m, 0.155 m behind the base,
  32 rings uniform over −15.5..+54.5° (the inverted H32F70), 0.4° azimuth.
- Scene: the true Webots shapes as boxes (Bed.proto mattress / footboard / headboard, the nightstand body on four legs,
  Toilet.proto bowl / pedestal / tank, the walls), first hit per beam.
- Room → world: bedroom from the bed and both nightstands (rotation 0, the fitted boxes land 2–6 cm from GT);
  bathroom pinned by the toilet itself (so its run is only the "centred on GT" case).
- Scored three ways, with the agent's sensor model (Pd .85, Pc .05, σ 3 cm + footprint σ):
  1. `carve_box` as shipped (2-D slab, one z test at mid-chord);
  2. a 3-D slab test against the fitted cuboid [0, H];
  3. the same 3-D test against the true composed parts.
  In 2 and 3 the through term is discounted by the surface weight (see finding 2).

The simulation reproduces the log where both are comparable: bed occ 285 sim vs 281 logged, nightstand_6 occ 486 vs
475, through 529 vs 674. nightstand_7 (155 vs 48) and the toilet agree less well.

## Findings

1. **The cause is the 2-D surface test, not open volumes.** `carve_box` calls a return "surface" when it lands near
   the side face the beam entered through in x–y. The Helios looks DOWN on furniture below 1.1 m, so most returns land
   on the TOP face, metres (bed) or decimetres (nightstand) behind the near side face, and are scored "interior" —
   "a solid object would have blocked this" — i.e. evidence AGAINST. Bed: 1640 of 1744 against-units are returns from
   the top of the real mattress. Centring the box exactly on GT changes nothing, so it is not registration.
2. **The through term leaks on thin parts.** For a part thinner than the surface σ (a 10 cm headboard), a real hit
   on the front face still carries Φ((1 − t_far)/σ_t) of "came out the back". Discounting the through term by the
   surface weight removes it (bed composed through 4082 → 284).
3. **The open volumes are not reached.** At the drop-cycle ranges (0.7–2.1 m) the lowest Helios ring is 0.53–0.9 m
   above the floor: it cannot see under the nightstand body (0.05–0.41) or the mattress (< 0.25). My earlier
   explanation (beams through the gaps) was wrong.
4. **The composed shape still matters for the toilet**: with a 3-D cuboid it stays net against (−6 per cycle) because
   the fitted 0.84 m box is mostly air above the bowl; with bowl + pedestal + tank it is +169.

| object | `carve_box` (shipped) | 3-D, fitted cuboid | 3-D, true composed parts |
|---|---|---|---|
| bed_10 | −151 | +405 | +891 |
| nightstand_6 | +3 | +25 | +21 |
| nightstand_7 | +1 | +10 | +14 |
| toilet_6 (centred on GT) | −35 | −6 | +169 |

Raw ΔL per drop cycle before saturation; box at its fitted place unless stated.

## Not checked

- The silhouette channel (nightstand_6 camera free_eff 2170 on its drop cycles). Masks are not logged per pixel, so
  this needs a live diagnostic. A projected solid cuboid includes the pixels between the legs; projecting the composed
  parts would address that too, unverified.
- bpearl: occupancy only, so it cannot produce absence; not replayed.
- The composed carve sums parts (a crude union); overlapping toilet parts double-count some interior (719).
