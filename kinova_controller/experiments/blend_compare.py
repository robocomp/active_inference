#!/usr/bin/env python3
"""Look-ahead-blend A/B: baseline (blend_radius=0) vs blend (blend_radius=0.04), clean round
(no probe), natural learning, same deterministic pick sweep. The blend is inert at novice
(skill_c·radius), so we compare the SKILLED reps (confidence>=0.7) — where it is active —
for fluidity (SPARC, stops, cycle time) AND a per-FSM-phase submovement breakdown, the
diagnostic that showed the old scalar carry-through ADDED stops in the legs it touched.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
FS = 50.0
PHASES = ["Track", "Insert", "Close", "Lift", "PlaceMv", "PlaceLo", "Release", "Retreat"]

def smooth(s, win=5):
    s = np.asarray(s, float)
    return s if len(s) < win else np.convolve(s, np.ones(win) / win, mode="same")
def sparc(s, fs=FS, fc=10.0, amp_th=0.05, padlevel=4):
    s = np.asarray(s, float)
    if len(s) < 8 or s.max() <= 0: return np.nan
    nfft = int(2 ** (np.ceil(np.log2(len(s))) + padlevel)); f = np.arange(0, fs, fs / nfft)
    M = np.abs(np.fft.fft(s, nfft)); M /= M.max()
    sel = f <= fc; f, M = f[sel], M[sel]
    ix = np.where(M >= amp_th)[0]
    if len(ix) < 2: return np.nan
    f, M = f[ix[0]:ix[-1] + 1], M[ix[0]:ix[-1] + 1]
    return float(-np.sum(np.sqrt((np.diff(f) / (f[-1] - f[0])) ** 2 + np.diff(M) ** 2)))
def stops(s, v_move=0.05, v_stop=0.02):
    return len(find_peaks(np.asarray(s, float), height=v_move, prominence=v_move - v_stop)[0])

here = os.path.dirname(os.path.abspath(__file__))
def load(tag, conf_min=0.7):
    d = np.genfromtxt(os.path.join(here, f"fluid_{tag}.csv"), delimiter=",", names=True)
    md = np.genfromtxt(os.path.join(here, f"fluid_{tag}_metrics.csv"), delimiter=",", names=True)
    succ = {int(r): int(s) for r, s in zip(md["rep"], md["success"])}
    tim  = {int(r): float(t) for r, t in zip(md["rep"], md["cycle_time_s"])}
    nsucc, ntot = int(np.sum(md["success"])), len(md["rep"])
    rows, ph_sub, n = [], np.zeros(len(PHASES)), 0
    for r in np.unique(d["rep"]).astype(int):
        sel = d["rep"] == r
        if float(np.median(d["confidence"][sel])) < conf_min or succ.get(int(r), 0) != 1:
            continue
        sp = smooth(d["ee_speed"][sel]); ph = d["phase"][sel].astype(int)
        rows.append((sparc(sp), stops(sp), tim.get(int(r), len(sp) / FS)))
        for p in find_peaks(sp, height=0.05, prominence=0.03)[0]:
            ph_sub[ph[p]] += 1
        n += 1
    arr = np.array([x for x in rows if not np.isnan(x[0])], float)
    return arr, nsucc, ntot, ph_sub / max(n, 1), n

base, bs, bt, bph, bn = load("baseB")
blend, cs, ct, cph, cn = load("blendB")
print(f"grasp success:  baseline {bs}/{bt}   blend {cs}/{ct}")
print(f"SKILLED reps (conf>=0.7):  baseline n={bn}   blend n={cn}\n")
print(f"{'metric':<20}{'baseline':>14}{'blend':>14}{'Δ':>10}")
for i, lab in enumerate(["SPARC (→0)", "stops", "cycle time s"]):
    a, b = base[:, i], blend[:, i]
    print(f"{lab:<20}{a.mean():>8.2f}±{a.std():<5.2f}{b.mean():>8.2f}±{b.std():<5.2f}{b.mean()-a.mean():>+10.2f}")

print(f"\nper-phase submovements / rep (the corner-stop diagnostic):")
print(f"{'phase':<9}{'baseline':>11}{'blend':>9}{'Δ':>9}")
for i, name in enumerate(PHASES):
    print(f"{name:<9}{bph[i]:>11.2f}{cph[i]:>9.2f}{cph[i]-bph[i]:>+9.2f}")
print(f"{'TOTAL':<9}{bph.sum():>11.2f}{cph.sum():>9.2f}{cph.sum()-bph.sum():>+9.2f}")

fig, ax = plt.subplots(1, 4, figsize=(15, 4))
fig.suptitle("Look-ahead blend (radius=0.04): skilled-rep fluidity — baseline vs blend",
             fontweight="bold")
for i, lab in enumerate(["SPARC (→0 smoother)", "stops/submoves", "cycle time (s)"]):
    ax[i].boxplot([base[:, i], blend[:, i]], tick_labels=["base", "blend"], showmeans=True)
    ax[i].set_title(lab, fontsize=9); ax[i].grid(alpha=0.3)
x = np.arange(len(PHASES))
ax[3].bar(x - 0.2, bph, 0.4, label="base"); ax[3].bar(x + 0.2, cph, 0.4, label="blend")
ax[3].set_xticks(x); ax[3].set_xticklabels(PHASES, rotation=45, fontsize=7)
ax[3].set_title("submoves/rep by phase", fontsize=9); ax[3].legend(fontsize=7); ax[3].grid(alpha=0.3)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(here, "blend_ab.png"), dpi=130)
print("\nwrote blend_ab.png")
