# human_concept — C++ core (active-inference body-pose estimator)

Eigen-only C++ port of the Python prototype's algorithmic core
(`../python/.../body tracking/python/`). No torch / DSR / Qt — this is the reusable
math layer that will later drop into the full RoboComp/DSR `human_concept` agent as the
`HumanModel` / `HumanFitter`.

## Layout

- `core/body18.h` — BODY_18 keypoint indices, names, skeleton edges.
- `core/human_kinematic_model.{h,cpp}` — 11-DOF forward kinematics → 18×3 keypoints,
  segment-length derivation, joint limits. (port of `human_kinematic_model.py`)
- `core/vfe_inference.{h,cpp}` — Kabsch SVD alignment + precision-weighted
  Gauss–Newton / Laplace belief update. Jacobian via **central finite differences**
  through the FK+Kabsch chain; prior gradients are closed-form. (port of `vfe_inference.py`)
- `harness/replay_main.cpp` — read keypoints from CSV, run the estimator, print the same
  summary / per-joint table as the Python `main.py` (no ZED, no OpenCV).
- `test/sanity_checks.cpp` — FK rest-pose + Kabsch unit checks (self-contained).
- `test/dump_reference.py` — runs the **original autograd Python** to emit an
  authoritative parity reference (`inputs.csv`, `reference.csv`).
- `test/parity_check.cpp` — replays `inputs.csv` through the C++ estimator and compares
  to `reference.csv` within tolerance.

## Build & verify

```bash
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && make -C build -j

./build/sanity_checks                                   # FK + Kabsch unit checks
python3 test/dump_reference.py --frames 40              # needs torch+numpy; writes test/*.csv
./build/parity_check test/inputs.csv test/reference.csv # C++ vs Python parity
./build/replay_main --input test/inputs.csv             # console replay
```

Current parity on the synthetic sequence: `max|dmu| ≈ 8e-5` rad/m, `mean_l2`/`rmse`
identical to 1e-6, `uncertainty_trace` relative diff ≈ 0 — i.e. the finite-difference
C++ reproduces the autograd Python.

## CSV format (replay / parity input)

`frame, kp0_x,kp0_y,kp0_z, …, kp17_x,kp17_y,kp17_z [, c0, …, c17]` — 1+54 columns
(no confidence) or 1+54+18 (with per-joint confidence in [0,100]). Missing keypoints use
`nan`. Lines starting with `#` are ignored.

## Notes / build constraints

Keep flags clean: **no `-march=native`, no `-DEIGEN_MAX_ALIGN_BYTES=0`** so the core
links cleanly into the libdsr-based agent later (see `../../CLAUDE.md`).

## Not yet done (future, see plan)

- Wrap the core into the full DSR `human_concept` agent (config/instance/model/fitter/
  scene_graph + presence; `person` DSR nodes), mirroring `bottle_concept`/`table_concept`.
- Live input source (ZED SDK vs a media-plane skeleton frame vs DSR).
- Analytic FK Jacobian swap-in; Pinocchio free-flyer reformulation.
- OpenCV / Qt visualization.
