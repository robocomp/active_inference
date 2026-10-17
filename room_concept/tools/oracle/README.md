# Refactor oracle

Recorded output of every room_concept selftest / bench mode that runs without the simulator, written by
`tools/refactor_oracle.sh --record` and checked by `tools/refactor_oracle.sh` after each refactor step
(plan: `docs/refactor_proposal_2026-10-13.md`, step 0). One file per case; the last line is the exit code.

It records what the tree DOES, not what it should do. A refactor step must reproduce these outputs,
including the failures below, which predate the refactor:

| case | pre-existing failure |
|---|---|
| `pose_field_bias_selftest` | 2 checks: "xy diagonal == sigma_b^2", "heading follows the scan's own lever" |
| `wss_default` | 2 checks: apartment layout Hausdorff 0.367 m (limit 0.20), right pillar mis-explained 0.65 |

Determinism: every case was run twice on the recorded tree; outputs are identical once wall-clock
timings (`N ms`) and date-time stamps are masked, which `tools/oracle_compare.py` does. The CMake
targets are single-threaded by construction (`wall_slam_selftest` re-execs itself with
`OMP_NUM_THREADS=1`).

Exact comparison is the default. The torch -> Eigen port changes float reduction order, so it is
checked with `--rtol` (record the tolerance used in the commit message).

Not covered here (needs Webots): live distributions — a `round_trips.sh` run recorded under
`estimation_scripts/crossing/` is the live baseline for each phase.
