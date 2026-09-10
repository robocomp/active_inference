#!/usr/bin/env python3
"""Complete skill-learning experiment figure: Round 1 (novice->skilled) and
Round 2 (retained skill), kinova_controller precision re-weighting.

Round 1 starts at confidence 0 (novice): the controller fuses bottle MODEL with
observation, weight w=1-confidence. As confirmed grasps raise confidence it samples
the bottle less (observations collapse) and moves on the model = open-loop.
Round 2 starts at the persisted confidence (=1.0): the skill is retained; surprises
transiently re-engage feedback (confidence decays, observations rise), then re-settle.
"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))

def load(p):
    d = np.genfromtxt(p, delimiter=",", names=True)
    return (d["rep"], d["success"].astype(int), d["confidence"],
            d["cycle_time_s"], d["observations"])

rounds = [("Round 1 — novice → skilled", "round1_novice_metrics.csv"),
          ("Round 2 — retained skill",   "round2_skilled_metrics.csv")]

fig, ax = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
fig.suptitle("kinova_controller: closed-loop → model-based open-loop via precision re-weighting",
             fontsize=13, fontweight="bold")

for c, (title, fname) in enumerate(rounds):
    rep, succ, conf, ct, obs = load(os.path.join(here, fname))
    ok = succ == 1
    # row 0: observations (log) — the closed-loop -> open-loop signal
    a = ax[0, c]
    a.plot(rep, obs, "-", color="0.6", zorder=1)
    a.scatter(rep[ok], obs[ok], c="tab:green", zorder=3, label="grasp confirmed")
    a.scatter(rep[~ok], obs[~ok], c="tab:red", marker="x", s=70, zorder=3, label="failed")
    a.set_yscale("log"); a.set_title(title, fontsize=11)
    a.grid(True, which="both", alpha=0.3)
    if c == 0: a.set_ylabel("perception look-ups / rep")
    a.legend(fontsize=8, loc="upper right")
    # row 1: confidence (model weight)
    a = ax[1, c]
    a.plot(rep, conf, "-o", color="tab:blue")
    a.scatter(rep[~ok], conf[~ok], c="tab:red", marker="x", s=70, zorder=3)
    a.set_ylim(-0.02, 1.05); a.grid(True, alpha=0.3)
    if c == 0: a.set_ylabel("confidence  Π_m/(Π_m+Π_s)")
    # row 2: cycle time
    a = ax[2, c]
    a.plot(rep, ct, "-", color="0.6", zorder=1)
    a.scatter(rep[ok], ct[ok], c="tab:green", zorder=3)
    a.scatter(rep[~ok], ct[~ok], c="tab:red", marker="x", s=70, zorder=3)
    a.set_xlabel("rep"); a.grid(True, alpha=0.3)
    if c == 0: a.set_ylabel("cycle time (s)")

fig.tight_layout(rect=(0, 0, 1, 0.96))
out = os.path.join(here, "experiment_skill_learning.png")
fig.savefig(out, dpi=130); print("wrote", out)

# console summary
for title, fname in rounds:
    rep, succ, conf, ct, obs = load(os.path.join(here, fname))
    nov = slice(0, 3); skl = (conf >= 0.95)
    print(f"\n{title}: {succ.sum()}/{len(rep)} success")
    print(f"  first 3 reps : obs={obs[nov].mean():.0f}  time={ct[nov].mean():.1f}s")
    if skl.any():
        print(f"  conf>=0.95   : obs={obs[skl].mean():.1f}  time={ct[skl].mean():.1f}s  ({skl.sum()} reps)")
