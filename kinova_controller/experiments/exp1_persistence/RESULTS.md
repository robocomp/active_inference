# EXP1 — Persistence / resume test (2026-06-10)

Goal: show the learned model precision Π_m is durable across process restarts —
i.e. a second round, loading the saved Π_m from disk, starts SKILLED rather than novice.

Protocol: round1 from Π_m=0 (fresh novice); round2 launched WITHOUT resetting the
confidence file (resumes the persisted Π_m). Fixed pick (0.1,-0.25)/place (0.15,-0.25),
round_cycles=10, learn_pick_place=true, precision_reweighting=true, sensory_precision=2.0.
Persisted file: /tmp/kinova_confidence.txt (stores Π_m). Per-rep metrics: *_metrics.csv.

## Result
RESUME PROVEN: round1 first grasp commit at c=0.00 (cold novice); round2 first commit at
c=0.61 (skilled from rep 0), loaded from Π_m=3.10 on disk. The learned precision is durable.

CAVEAT (and the motivation for the next fix): cold-starting skilled (collapsed standoff +
short settle from rep 0) over-aggressed and CLIPPED the bottle on the reach → "bottle not
held" misses → calibration correctly deflated Π_m 3.10→0.90 (c→0.31). This exposed the
need for a bottle-as-obstacle term during the retrieval/approach so the skilled approach
rounds to the standoff without tipping the bottle. See round2.log.

Files: round1.log/round1_metrics.csv, round2.log/round2_metrics.csv.
