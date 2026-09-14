# Room 2 of `tworooms-piso.wbt` — a known rectangle

## Ground truth: 6.000 x 4.000 m interior

Taken from the **wall solids** in the world, not from the floor slab:

| wall | centre | thickness | inner face |
|---|---|---|---|
| WEST  | x = -7.825 | 0.110 | x = -7.770 |
| EAST  | x = -1.715 | 0.110 | x = -1.770 |
| SOUTH | y =  9.425 | 0.110 | y =  9.480 |
| NORTH | y = 13.535 | 0.110 | y = 13.480 |

**Interior 6.000 m x 4.000 m.** The south wall is split (`ROOM2_WALL_SOUTH_WEST/EAST`) leaving a
**1.06 m doorway** between x = -5.300 and x = -4.240.

⚠ **An earlier version of this file said 6.22 x 4.39 and was WRONG.** That came from the `ROOM2`
array in `make_floor_tworooms.py`, which is the FLOOR SLAB — it extends under the walls. The name of
a variable is not a statement about which surface it describes; check the walls.

## Why this room is worth more than another tour of the flat

1. **The ground truth is frame-invariant.** Edge lengths and corner angles do not depend on where the
   estimator put its map frame, so they can be checked with **no rigid alignment**. Every other
   layout comparison in this work has to align first, and alignment absorbs the global pose error;
   here it does not arise.
2. **A 4-corner answer is the CORRECT answer**, so it separates two hypotheses that the piso run
   confounds: if the estimator recovers 6.000 x 4.000 with honest sigma, the machinery is sound and the
   20-vertex flat's collapse to a rectangle is about exploration (who drives). If it does not, the
   fault is upstream of driving.
3. **Standing still is the point, not a compromise.** We measured that at rest the POSE covariance
   collapses 3x while the innovation grows (see `TODO_PER_WALL_OFFSET.md`). Whether the published
   LAYOUT sigma inherits that has never been checked, and a stationary run in a known rectangle
   checks it directly.

## Two caveats at the scale we are measuring (sigma is ~4.7 cm here)

- ⚠ The two caveats below were written for the RETRACTED 6.22 x 4.39 floor-slab figure and are kept
  only to explain why that number existed. They do NOT apply to the wall-solid truth at the top of
  this file, which is exact: the slab extends under the walls and its south edge overlaps the
  apartment polygon by ~3 cm, which is where "4.36-4.39 m" came from.
- **The south wall is about a degree off-axis** (y = 9.256 at its west end, 9.231 at its east), so a
  perfectly Manhattan estimate is not the right answer and a ~1 deg tilt there is truth, not error.

Width, **6.000 m** from the wall solids, is the cleanest single number to check.

## Pre-registered predictions, before the run

- The polygon closes on 4 corners.
- Recovered width within ~2 sigma of 6.000 m.
- **If the parked-covariance defect extends to the layout channel, the declared corner sigma will
  fall BELOW the realised edge error while the robot is stationary**, and keep falling the longer it
  sits. That is the interesting outcome and the one to watch; a sigma that stops falling is evidence
  the layout channel does not inherit the defect.


## RESULT (2026-09-12, Webots, closed loop)

| | measured | truth | error | declared sigma | error/sigma |
|---|---|---|---|---|---|
| width | 5.994 m | 6.000 m | **-0.006 m** | 0.0283 m | 0.21 |
| height | 4.003 m | 4.000 m | **+0.003 m** | 0.0283 m | 0.11 |

**And the shape was right before the uncertainty was.** Parked, the estimate was already
5.994 x 4.013 — but corner sigma sat at 0.9313 m and edge sigma_d at exactly its 0.5000 prior for
**657 seconds**, through five associated segments per frame, moving by one part in ten thousand.
One rotation in place dropped corner sigma **33x** to 0.0283 and sigma_d to 0.0096, while the
geometry moved by millimetres.

⚠ **THE TAIL OF THIS LOG IS NOT UNIFORMLY SAMPLED.** dt is ~50 ms up to frame 13477 and then
**29 917 ms** to 13478 and **43 227 ms** to 13479. The 0.0283 / 33x figures come from those last
frames, i.e. from the far side of a 30-second hole, so they are the converged value and NOT the
effect of the turn alone. Measured where the log is actually sampled, the turn is **319 deg over
24.5 s** and takes corner sigma 0.9885 -> **0.0480** (21x) at frame 13477, which is also the FIRST
frame the layout is publishable. The paper quotes 13477 for exactly this reason. Anything computed
across 13477->13478 is a lower bound on unsampled motion, not a measurement.

So the estimator had the room correct and **refused to claim precision it had not earned from
motion**. That is the epistemic claim demonstrated in a closed loop rather than argued: repeated
looks from a fixed pose are correlated evidence and buy nothing, and the covariance says so.
Contrast [[pose-covariance-collapses-when-parked]], where the POSE covariance does the opposite and
shrinks on exactly that evidence — the same run shows both the right behaviour and the wrong one in
two different channels.
