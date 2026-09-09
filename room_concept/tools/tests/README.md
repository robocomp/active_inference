# `test_vertex_offset_nuisance.cpp`

Checks the per-vertex offset marginalisation (`mount_lidar_pair.h`) on SYNTHETIC data whose truth is
known: yaw is exactly 0, and each of 23 corners carries its own constant u-offset drawn from
N(0, 5.3 px) — the spread measured live.

★ It reproduces the LIVE DEFECT first, which is what makes it a test rather than a demonstration:
with the nuisance off the solve returns yaw = 0.783 ± 0.0065 px, a 120-sigma confident wrong answer.
With it on, 0.352 ± 0.742, consistent with the truth.

Asserted by eye against the pre-registration in `VALIDATION_THREE_DEVICE_CORNERS.md` §2 stage 1:

| | expected | measured |
|---|---|---|
| yaw sigma inflation | ≈ design effect, sqrt(n/clusters) = 130x | 114x |
| pitch / height inflation | strictly SMALLER than yaw's | 2.7x / 4.8x |
| chi2/dof | falls toward 1 (the offsets WERE the missing variance) | 4.12 → 1.00 |
| eff_params | → 2 per cluster once the data pays for them | 46.0 = 2 x 23 |
| S -> 0 | reproduces the old solve bit for bit | YES |

Build (it lifts the structs out of the header, so it needs no DSR and no Qt):

```
python3 tools/tests/lift.py && g++ -std=c++23 -O2 -I/usr/include/eigen3 -Itools/tests \
    tools/tests/test_vertex_offset_nuisance.cpp -o /tmp/t && /tmp/t
```

⚠ A standalone harness has no Qt, so it stays in the "C" locale and any locale parsing bug vanishes
(CLAUDE.md). Nothing here parses a file, so that does not bite — but do not add file reading to it
without `std::setlocale(LC_ALL, "")`.
