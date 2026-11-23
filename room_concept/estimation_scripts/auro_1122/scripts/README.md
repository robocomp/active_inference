# AuRo 11-22/23 grading and series scripts (archived as run)

These are the scripts behind the reports in `auro_1122/*.md`. They are copied verbatim from the session scratchpad in which they ran, so some still set `S=`/`SCR=` to that `/tmp/claude-1000/...` path. To rerun one, set those variables to this folder.

- `review/`: promised vs delivered over all claims (`t3s1.py`) and the T1–T5 log reviews.
- `review2/`: layout error and calibration stops (`layout_err.py`, `stops.py`, `runinfo.tsv`).
- `review3/`: miss-counter counterfactual and recognition swaps (`m1*.py`, `m3.py`).
- `review5/`: per-visit claim windows (`cw.py`, `an.py`, `st.py`).
- Series:
  - λ sweep: `lambda_sweep_1122.sh`.
  - C4 legs: `c4_leg.sh`.
  - C3a tours: `c3a_run.sh`.
  - Step 1: `step1_series*.sh`, `step1f_series.sh`.
  - Chair fix: `step1g_series.sh`.
  - Leg-1 door validation: `leg1_series.sh`.
  - Controller CPU-affinity test: `affinity_toggle.sh`.
  - Item 14: `c14_removal_1122.sh`.
  - C3b: `c3b_series_1122.sh`.
- `hold.py`: GT vs estimated motion in time windows (leg-1 diagnosis).
