# Corner-channel analysis — how to produce a number we can defend

## Why this exists

An earlier §V-E quoted `0.0952 m / 0.0113 m / 8.1x` from an ad-hoc heredoc that was never committed,
against a run that was later overwritten. `src/room_concept.h` opens the probe with
`std::ios::trunc`, so **every run destroys the previous one**. Re-running the analysis on surviving
data gave `0.0403 / 0.0063 / 5.6x`, and nothing could adjudicate between them because neither the
script nor the data still existed. That is the failure this directory prevents.

## The two files, and why you need both

| file | carries | needed for |
|---|---|---|
| `tmp/corner_probe.csv` | one row per corner candidate at the association gate: innovation, each covariance term, `d2`, `over_gate` | NIS, the repeat/scatter split |
| `tmp/layout_trace.csv` | one row per frame: pose, published polygon, sigmas | rotation bands, stationary detection |

The probe file carries **no pose**, so without the trace there are no bands and no stationary
stretches — i.e. no headline number. Save both or the run is wasted.

## Procedure (identical for Webots and the real robot)

1. Run the tour. Include **deliberate stops of >= 30 s** — the repeat/scatter split is measured only
   over stationary stretches, so a tour without stops produces nothing.
2. **Immediately, before anything else starts the agent again:**
   ```
   D=datasets/<place>/<run>; mkdir -p $D
   cp tmp/corner_probe.csv  $D/corner_probe.csv
   cp tmp/layout_trace.csv  $D/layout_trace.csv
   ```
3. Analyse:
   ```
   python3 analysis/corner_channel/split_and_nis.py $D/corner_probe.csv $D/layout_trace.csv
   ```
4. Paste the output into the run's notes. Anything quoted in the paper comes from that output.

## Two traps the script makes visible rather than hiding

- **NIS is censored by its own gate.** `d2` is the statistic the association gate tests, so rejected
  rows are absent and the surviving distribution is truncated. On the 2026-09-13 run, 28.6% of
  candidates were rejected. An aggregate over accepted rows is "calibrated given accepted", not
  "calibrated"; the script prints the rejection fraction next to every NIS so this cannot be read
  past.
- **A truncated final line.** A log copied while the agent is still writing ends mid-row. The reader
  drops any row whose field count is wrong rather than parsing a short row into silent nonsense.
