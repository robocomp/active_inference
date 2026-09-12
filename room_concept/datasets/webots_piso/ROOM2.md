# Room 2 of `tworooms-piso.wbt` — a known rectangle, and why that matters

Ground truth taken verbatim from the world's own generator,
`webots-shadow/worlds/piso/make_floor_tworooms.py`:

    ROOM2 = [(-7.88, 9.20), (-1.66, 9.20), (-1.66, 13.59), (-7.88, 13.59)]

That array is the **air polygon** (the generator's floor slab is "exactly the air polygon"), so these
are the INNER wall faces — exactly what the estimator should recover. **6.22 m x 4.39 m, 27.3 m²**,
consistent with the docstring's 27.233 m² after the difference with the apartment.

## Why this room is worth more than another tour of the flat

1. **The ground truth is frame-invariant.** Edge lengths and corner angles do not depend on where the
   estimator put its map frame, so they can be checked with **no rigid alignment**. Every other
   layout comparison in this work has to align first, and alignment absorbs the global pose error;
   here it does not arise.
2. **A 4-corner answer is the CORRECT answer**, so it separates two hypotheses that the piso run
   confounds: if the estimator recovers 6.22 x 4.39 with honest sigma, the machinery is sound and the
   20-vertex flat's collapse to a rectangle is about exploration (who drives). If it does not, the
   fault is upstream of driving.
3. **Standing still is the point, not a compromise.** We measured that at rest the POSE covariance
   collapses 3x while the innovation grows (see `TODO_PER_WALL_OFFSET.md`). Whether the published
   LAYOUT sigma inherits that has never been checked, and a stationary run in a known rectangle
   checks it directly.

## Two caveats at the scale we are measuring (sigma is ~4.7 cm here)

- **The south edge carries a 3 cm ambiguity.** The generator extends room 2 "south to y = 9.20 so it
  overlaps the apartment polygon by ~3 cm"; the real inner face is the apartment's north wall at
  y = 9.231..9.256. So the true height is **4.36-4.39 m**, not a single number.
- **The south wall is about a degree off-axis** (y = 9.256 at its west end, 9.231 at its east), so a
  perfectly Manhattan estimate is not the right answer and a ~1 deg tilt there is truth, not error.

Width, 6.22 m, has neither problem and is the cleanest single number to check.

## Pre-registered predictions, before the run

- The polygon closes on 4 corners.
- Recovered width within ~2 sigma of 6.22 m.
- **If the parked-covariance defect extends to the layout channel, the declared corner sigma will
  fall BELOW the realised edge error while the robot is stationary**, and keep falling the longer it
  sits. That is the interesting outcome and the one to watch; a sigma that stops falling is evidence
  the layout channel does not inherit the defect.
