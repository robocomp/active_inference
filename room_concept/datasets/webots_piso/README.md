# Webots `piso` — ground truth for a closed-loop layout-calibration run

`truth.txt` is the apartment outline in the `WS_LAYOUTS` format (`name;x,y x,y …`, metres, CCW),
derived from `webots-piso/worlds/piso_layout.svg`.

## Scale: 60 px = 1 m. The evidence, not an assumption

- **Every** path coordinate is a multiple of 6 px, i.e. the plan was drawn on a 10 cm grid at 60 px/m.
- The outline then measures 8.10 m x 9.10 m, a plausible flat.
- Independently, the two pillars in `webots-shadow/worlds/piso/piso.wbt` sit at x = -8.310 and
  -0.311, i.e. **7.999 m apart**, against an outline span of 8.10 m — consistent once the pillars are
  inset from the walls.

The SVG y axis points down and is flipped here; winding is normalised to CCW.

## ⚠ What still has to be settled by the run, not by me

1. **Frame alignment.** The estimator's map frame is start-relative, so the estimate and this polygon
   are not in a common frame. Comparing them requires a rigid alignment — and aligning before
   measuring REMOVES the global pose error, so it measures SHAPE error only. That is arguably the
   right quantity for corner-sigma calibration, but it must be stated in the paper, not glossed.
2. **Whether the estimator is even compared against this.** In `given` mode the layout comes from the
   DSR graph, not from a file. If that layout is itself derived from this same SVG, the map error is
   ~0 by construction: the run is then a clean CONTROL (does the propagation hold up in a real loop
   with no map error?) rather than a test of the systematic term. Useful, but a different claim —
   check which layout the graph carries before quoting the result as either.
3. The walls in `piso.wbt` are **meshes**, not parametrised boxes, so this SVG is the only cheap
   source of truth; the 43 `Box` nodes in the world are furniture, shelving and pillars.

## Running it

The live agent now writes `tmp/layout_trace.csv` (one row per frame: published polygon, per-corner
sigma, per-edge sigma_d, closed/publishable) — same columns and semantics as the offline harness's
`RunConfig::poly_csv`, so `paper/icra/mkfigs.py` and the Hungarian matcher read both.

Start Webots with `webots-shadow/worlds/piso/piso.wbt` and the usual agent set, drive a tour, then
the analysis is the same one used for the 594 rooms.
