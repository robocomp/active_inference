# Closed-loop runs, Webots `piso`

One file per run, copied out of `tmp/layout_trace.csv` before the next run truncates it.

- `joystick_piso_*.csv` — **arm (b), the ablation**: human drives with the joystick, EIG planner not
  choosing targets (no `[planner]` line appeared in the agent's output for the whole run). 1360
  frames / 97 s. The polygon **stayed a 4-vertex rectangle for the entire run** against a 20-vertex
  flat, while the declared sigma converged to 0.047 m and the layout was marked publishable from 23 s
  on. Confidently wrong shape, honestly-small sigma on each of the four walls it did find.
  ⚠ The pose column reads exactly `0,0,0` for the last third — that is the `has_state() == false`
  branch of the writer, not a parked robot. Unexplained; check whether it recurs.

Planned:
- `planner_piso_*.csv` — arm (a), same everything, EIG planner choosing viewpoints.
- `parked_rect_*.csv` — the 6 x 4.2 m room behind the passage door, robot stationary. Ground truth
  is a KNOWN RECTANGLE, so edge lengths and angles can be checked **without any frame alignment**,
  which is the one caveat every other layout comparison here carries.

- `room2_parked_seedbug_*.csv` — robot parked in room 2 (6.000 x 4.000 m interior; an earlier note said 6.22 x 4.39, which was the FLOOR SLAB and is retracted in ROOM2.md), 4740 frames / 242 s.
  **Diagnostic run that found the seed-box defect.** The polygon locks at 8 vertices and never
  resolves: four of them are the real room, four are the {+-10, +-10} m fallback seed box, one still
  carrying its untouched prior sigma of 2.665 after four minutes. The room ITSELF is recovered well —
  6.18 x 3.98 m against 6.000 x 4.000, with sigma 0.021-0.022 m on the two well-observed corners.
  ⚠ The original note called that "a 4 cm width error at about 1.8 sigma" — that was computed against
  the RETRACTED 6.22 truth. Against the real 6.000 it is an **18 cm** width error at about 8.6 sigma,
  i.e. the estimate was over-confident here, not comfortably inside its own bound. The shape was
  polluted AND the width was wrong; only the height (3.98 vs 4.000) was good.
  Superseded by the fix; keep as the "before" arm.
