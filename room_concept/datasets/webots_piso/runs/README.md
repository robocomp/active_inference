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
