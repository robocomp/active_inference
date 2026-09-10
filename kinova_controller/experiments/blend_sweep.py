#!/usr/bin/env python3
"""blend_radius sweep: {0.0, 0.02, 0.04, 0.06}. Clean rounds, natural learning, same
deterministic pick sweep. Per skilled rep (confidence>=0.7, where the blend is active)
report SPARC, stops, cycle time, and a combined fluidity-time objective
    J(θ) = α·cycle_time + β·(−SPARC)        (lower = better: faster AND smoother)
This is the first 1-D slice of the J(θ) surface for the CMA-ES/SPSA loop.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
FS = 50.0
ALPHA, BETA = 1.0, 1.0   # J = α·time + β·(−SPARC); tune the trade-off here

SWEEP = [(0.00, "baseB"), (0.02, "r02B"), (0.04, "blendB"), (0.06, "r06B")]

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
    rows = []
    for r in np.unique(d["rep"]).astype(int):
        sel = d["rep"] == r
        if float(np.median(d["confidence"][sel])) < conf_min or succ.get(int(r), 0) != 1:
            continue
        sp = smooth(d["ee_speed"][sel])
        rows.append((sparc(sp), stops(sp), tim.get(int(r), len(sp) / FS)))
    return np.array([x for x in rows if not np.isnan(x[0])], float), nsucc, ntot

print(f"{'radius':>7}{'success':>9}{'n_skl':>6}{'SPARC':>14}{'stops':>13}{'cycle_s':>14}{'J':>9}")
res = []
for rad, tag in SWEEP:
    arr, nsucc, ntot = load(tag)
    sp, st, ct = arr[:, 0], arr[:, 1], arr[:, 2]
    J = ALPHA * ct.mean() + BETA * (-sp.mean())
    res.append((rad, sp.mean(), sp.std(), st.mean(), st.std(), ct.mean(), ct.std(), J, len(arr)))
    print(f"{rad:>7.2f}{nsucc:>6}/{ntot:<2}{len(arr):>6}"
          f"{sp.mean():>9.2f}±{sp.std():<4.2f}{st.mean():>8.1f}±{st.std():<4.1f}"
          f"{ct.mean():>9.2f}±{ct.std():<4.2f}{J:>9.2f}")

R = np.array([r[0] for r in res])
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("blend_radius sweep — skilled-rep fluidity & combined objective J(θ)", fontweight="bold")
for axi, (col, lab, yl) in zip(ax, [(1, "SPARC (→0 smoother)", "SPARC"),
                                    (3, "stops / submovements", "stops"),
                                    (5, "cycle time (s)", "s"),
                                    (7, "J = time + (−SPARC)  (lower=better)", "J")]):
    y = np.array([r[col] for r in res]); e = np.array([r[col + 1] for r in res]) if col + 1 in (2, 4, 6) else None
    axi.errorbar(R, y, yerr=e, marker="o", capsize=3)
    axi.set_xlabel("blend_radius (m)"); axi.set_title(lab, fontsize=9); axi.set_ylabel(yl); axi.grid(alpha=0.3)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(here, "blend_sweep.png"), dpi=130)
print("\nwrote blend_sweep.png")
