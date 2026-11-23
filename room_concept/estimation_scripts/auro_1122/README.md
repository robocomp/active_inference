# AuRo evidence, 2026-11-22

These are tracked copies of the reports. The run folders stay in the git-ignored `crossing/`, so the files here name runs by their folder.

| file | item |
|---|---|
| nees_summary.md | item 1: per-agent calibration (NEES), with the C3a regrade (gauge frames + fridge reference point) |
| ../explorer_rerun_1122/SUMMARY.md | item 2: explorer rerun on the 594 MatterportLayout plans |
| c3b5_summary.md, c3b5_report.csv | item 13: out-of-view moved chair, 5 × 2 runs (negative) |
| c14_summary.md, c14_report.csv | item 14: removed objects, 3 runs (no removal ever detected) |
| c3a_w2_summary.md | C3a object tour repeats in the owner's new world (closet, 2 more chairs) |

**Scripts** (in `../scripts/`):
- `nees_agents.py`: item 1.
- `c3a_objects.py`: C3a.
- `c3b_report.py`, `run_frames.py`: item 13.
- `c14_report.py`: item 14.
- `concept_agents.sh harvest`: copies each agent's log from the checkout it ran from.
