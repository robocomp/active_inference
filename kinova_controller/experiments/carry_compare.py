#!/usr/bin/env python3
"""Carry-through (A) A/B: baseline (carry=0) vs mechanism-A (carry=0.12), fixed pick,
natural learning. Carry-through is inert at novice (skill_c·carry), so we compare the
SKILLED reps (confidence>=0.7) — where it is active — for both fluidity and grasp success.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
FS = 50.0

def smooth(s, win=5):
    s = np.asarray(s, float)
    return s if len(s) < win else np.convolve(s, np.ones(win)/win, mode="same")
def sparc(s, fs=FS, fc=10.0, amp_th=0.05, padlevel=4):
    s = np.asarray(s, float)
    if len(s) < 8 or s.max() <= 0: return np.nan
    nfft = int(2**(np.ceil(np.log2(len(s)))+padlevel)); f = np.arange(0, fs, fs/nfft)
    M = np.abs(np.fft.fft(s, nfft)); M /= M.max()
    sel = f <= fc; f, M = f[sel], M[sel]
    ix = np.where(M >= amp_th)[0]
    if len(ix) < 2: return np.nan
    f, M = f[ix[0]:ix[-1]+1], M[ix[0]:ix[-1]+1]
    return float(-np.sum(np.sqrt((np.diff(f)/(f[-1]-f[0]))**2 + np.diff(M)**2)))
def stops(s, v_move=0.05, v_stop=0.02):
    return len(find_peaks(np.asarray(s, float), height=v_move, prominence=v_move-v_stop)[0])

here = os.path.dirname(os.path.abspath(__file__))
def load(tag, conf_min=0.7):
    d = np.genfromtxt(os.path.join(here, f"fluid_{tag}.csv"), delimiter=",", names=True)
    md = np.genfromtxt(os.path.join(here, f"fluid_{tag}_metrics.csv"), delimiter=",", names=True)
    succ = {int(r): int(s) for r, s in zip(md["rep"], md["success"])}
    tim  = {int(r): float(t) for r, t in zip(md["rep"], md["cycle_time_s"])}
    nsucc = int(np.sum(md["success"])); ntot = len(md["rep"])
    rows = []
    for r in np.unique(d["rep"]).astype(int):
        sel = d["rep"] == r
        if float(np.median(d["confidence"][sel])) < conf_min or succ.get(int(r), 0) != 1:
            continue
        sp = smooth(d["ee_speed"][sel])
        rows.append((sparc(sp), stops(sp), tim.get(int(r), len(sp)/FS)))
    return np.array([x for x in rows if not np.isnan(x[0])], float), nsucc, ntot

base, bs, bt = load("baseA")
carry, cs, ct = load("carryA")
print(f"grasp success:  baseline {bs}/{bt}   carry-A {cs}/{ct}")
print(f"SKILLED reps (conf>=0.7):  baseline n={len(base)}   carry-A n={len(carry)}\n")
print(f"{'metric':<20}{'baseline':>14}{'carry-A':>14}{'Δ':>10}")
for i, lab in enumerate(["SPARC (→0)", "stops", "cycle time s"]):
    a, b = base[:, i], carry[:, i]
    print(f"{lab:<20}{a.mean():>8.2f}±{a.std():<5.2f}{b.mean():>8.2f}±{b.std():<5.2f}{b.mean()-a.mean():>+10.2f}")

fig, ax = plt.subplots(1, 3, figsize=(11, 4))
fig.suptitle("Carry-through (A): skilled-rep fluidity — baseline vs carry=0.12", fontweight="bold")
for i, lab in enumerate(["SPARC (→0 smoother)", "stops/submoves", "cycle time (s)"]):
    ax[i].boxplot([base[:, i], carry[:, i]], tick_labels=["base", "carry-A"], showmeans=True)
    ax[i].set_title(lab, fontsize=9); ax[i].grid(alpha=0.3)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(here, "carry_ab.png"), dpi=130)
print("\nwrote carry_ab.png")
