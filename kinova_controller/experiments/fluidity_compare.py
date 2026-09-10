#!/usr/bin/env python3
"""Clean novice-vs-skilled fluidity comparison from the two PINNED-confidence segments
(fixed pick, no probe, conf_gain=0 → confidence held at 0 and 1.0). The only difference
between the two datasets is the confidence level, so this isolates its effect on fluidity.

Reads experiments/fluid_novice.csv and experiments/fluid_skilled.csv (+ *_metrics.csv).
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

FS = 50.0  # Hz (Period.Compute = 20 ms)

def smooth(speed, win=5):
    speed = np.asarray(speed, float)
    if len(speed) < win: return speed
    return np.convolve(speed, np.ones(win) / win, mode="same")

def sparc(speed, fs=FS, padlevel=4, fc=10.0, amp_th=0.05):
    speed = np.asarray(speed, float)
    if len(speed) < 8 or np.max(speed) <= 0: return np.nan
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(speed, nfft)); Mf = Mf / np.max(Mf)
    sel = f <= fc; f_sel, Mf_sel = f[sel], Mf[sel]
    inx = np.where(Mf_sel >= amp_th)[0]
    if len(inx) < 2: return np.nan
    f_sel = f_sel[inx[0]:inx[-1] + 1]; Mf_sel = Mf_sel[inx[0]:inx[-1] + 1]
    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0]); dM = np.diff(Mf_sel)
    return float(-np.sum(np.sqrt(df ** 2 + dM ** 2)))

def stops(speed, v_move=0.05, v_stop=0.02):
    pk, _ = find_peaks(np.asarray(speed, float), height=v_move, prominence=v_move - v_stop)
    return len(pk)

here = os.path.dirname(os.path.abspath(__file__))

def per_rep(seg):
    d = np.genfromtxt(os.path.join(here, f"fluid_{seg}.csv"), delimiter=",", names=True)
    mp = os.path.join(here, f"fluid_{seg}_metrics.csv")
    m = {}
    if os.path.exists(mp):
        md = np.genfromtxt(mp, delimiter=",", names=True)
        for r, s, c, t, o in zip(md["rep"], md["success"], md["confidence"],
                                 md["cycle_time_s"], md["observations"]):
            m[int(r)] = (int(s), float(t), float(o))
    out = []
    for r in np.unique(d["rep"]).astype(int):
        sp_prof = smooth(d["ee_speed"][d["rep"] == r])
        s, t, o = m.get(int(r), (1, len(sp_prof) / FS, np.nan))
        if s == 1:                      # only successful reps (failures aren't a "skill" sample)
            out.append((sparc(sp_prof), stops(sp_prof), len(sp_prof) / FS, t, o))
    return np.array([r for r in out if not np.isnan(r[0])], float)

nov, skl = per_rep("novice"), per_rep("skilled")
labels = ["SPARC (→0 smoother)", "stops/submoves", "ActiveEFE dur (s)", "cycle time (s)", "look-ups"]
print(f"{'metric':<20}{'novice':>16}{'skilled':>16}{'Δ':>10}")
for i, lab in enumerate(labels):
    a, b = nov[:, i], skl[:, i]
    print(f"{lab:<20}{a.mean():>8.2f}±{a.std():<6.2f}{b.mean():>8.2f}±{b.std():<6.2f}{b.mean()-a.mean():>+10.2f}")
print(f"\nn: novice={len(nov)}  skilled={len(skl)}")

# box plots
fig, ax = plt.subplots(1, 5, figsize=(15, 4))
fig.suptitle("Novice (conf=0) vs Skilled (conf=1.0) — fixed pick, no probe", fontweight="bold")
for i, lab in enumerate(labels):
    ax[i].boxplot([nov[:, i], skl[:, i]], labels=["nov", "skl"], showmeans=True)
    ax[i].set_title(lab, fontsize=9); ax[i].grid(alpha=0.3)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = os.path.join(here, "fluidity_novice_vs_skilled.png")
fig.savefig(out, dpi=130); print("wrote", out)
