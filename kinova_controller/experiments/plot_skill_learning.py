#!/usr/bin/env python3
"""Plot the novice->skilled learning curve from kinova_controller skill metrics.

Usage:  python3 plot_skill_learning.py [metrics.csv]
Default reads the newest experiments/skill_metrics_*.csv.

The headline signal is `observations` (perception look-ups per rep): as the
precision-reweighting confidence rises, the controller fuses the bottle MODEL
in with weight Pi_m/(Pi_m+Pi_s), samples the bottle less often, and moves
faster -- i.e. it switches from closed-loop reaching to model-based open-loop.
"""
import sys, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    cands = sorted(glob.glob(os.path.join(here, "skill_metrics_*.csv")))
    path = cands[-1] if cands else "/tmp/kinova_skill_metrics.csv"

rep, success, conf, ctime, obs = [], [], [], [], []
with open(path) as f:
    next(f)  # header: rep,success,confidence,cycle_time_s,observations
    for line in f:
        line = line.strip()
        if not line:
            continue
        r, s, c, t, o = line.split(",")
        rep.append(int(r)); success.append(int(s)); conf.append(float(c))
        ctime.append(float(t)); obs.append(float(o))

rep = np.array(rep); success = np.array(success); conf = np.array(conf)
ctime = np.array(ctime); obs = np.array(obs)
ok = success == 1

fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
fig.suptitle("kinova_controller: novice -> skilled (precision re-weighting)\n"
             f"{os.path.basename(path)}  ({len(rep)} reps)", fontsize=12)

# Panel 1: observations per rep (the closed-loop -> open-loop signal)
ax[0].plot(rep, obs, "-", color="0.6", zorder=1)
ax[0].scatter(rep[ok], obs[ok], c="tab:green", label="grasp confirmed", zorder=3)
ax[0].scatter(rep[~ok], obs[~ok], c="tab:red", marker="x", s=70, label="failed", zorder=3)
ax[0].set_ylabel("perception look-ups / rep")
ax[0].set_yscale("log")
ax[0].set_title("Observations collapse: closed-loop -> model-based open-loop")
ax[0].grid(True, which="both", alpha=0.3); ax[0].legend()

# Panel 2: confidence trajectory (Pi_m / (Pi_m + Pi_s))
ax[1].plot(rep, conf, "-o", color="tab:blue")
ax[1].scatter(rep[~ok], conf[~ok], c="tab:red", marker="x", s=70, zorder=3,
              label="failure -> confidence decays")
ax[1].set_ylabel("confidence  (model weight)")
ax[1].set_ylim(0, 1.02)
ax[1].set_title("Learned skill: rises with confirmed grasps, decays on failures")
ax[1].grid(True, alpha=0.3); ax[1].legend()

# Panel 3: cycle time per rep
ax[2].plot(rep, ctime, "-", color="0.6", zorder=1)
ax[2].scatter(rep[ok], ctime[ok], c="tab:green", zorder=3)
ax[2].scatter(rep[~ok], ctime[~ok], c="tab:red", marker="x", s=70, zorder=3)
ax[2].set_ylabel("cycle time (s)")
ax[2].set_xlabel("rep")
ax[2].set_title("Cycle time per rep")
ax[2].grid(True, alpha=0.3)

fig.tight_layout(rect=(0, 0, 1, 0.96))
out = os.path.splitext(path)[0] + ".png"
fig.savefig(out, dpi=130)
print("wrote", out)

# Console summary: novice (first 3) vs skilled (confidence >= 0.5 & success)
nov = slice(0, 3)
skl = ok & (conf >= 0.5)
print(f"\nnovice (reps 0-2):   obs mean={obs[nov].mean():.0f}  "
      f"time mean={ctime[nov].mean():.1f}s")
if skl.any():
    print(f"skilled (conf>=0.5): obs mean={obs[skl].mean():.1f}  "
          f"time mean={ctime[skl].mean():.1f}s  ({skl.sum()} reps)")
    print(f"observation reduction: {obs[nov].mean()/obs[skl].mean():.1f}x fewer look-ups")
