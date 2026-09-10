#!/usr/bin/env python3
"""Characterise pick-and-place FLUIDITY vs confidence from the per-cycle EE-speed log.

Per rep we compute, over the whole pick-place EE speed profile:
  - SPARC  : spectral arc length (Balasubramanian et al. 2015), the standard movement-
             smoothness metric. Negative; CLOSER TO 0 = smoother. A clean single reach
             ≈ -1.5; a multi-stop staccato motion is much more negative.
  - stops  : number of submovements = speed peaks above v_move separated by dips below
             v_stop. Fewer = more fluid (ideally ~1 per segment, not one per FSM gate).
  - dur    : ActiveEFE duration (s).
joined with the per-rep metrics (confidence, success, cycle_time, observations).

Usage: python3 fluidity_analysis.py [fluidity.csv] [metrics.csv]
"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

FS = 50.0  # Hz (Period.Compute = 20 ms)

def sparc(speed, fs=FS, padlevel=4, fc=10.0, amp_th=0.05):
    """Spectral arc length (canonical implementation). speed: 1-D speed profile."""
    speed = np.asarray(speed, float)
    if len(speed) < 8 or np.max(speed) <= 0:
        return np.nan
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(speed, nfft))
    Mf = Mf / np.max(Mf)
    sel = f <= fc
    f_sel, Mf_sel = f[sel], Mf[sel]
    inx = np.where(Mf_sel >= amp_th)[0]
    if len(inx) < 2:
        return np.nan
    f_sel = f_sel[inx[0]:inx[-1] + 1]
    Mf_sel = Mf_sel[inx[0]:inx[-1] + 1]
    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0])
    dM = np.diff(Mf_sel)
    return float(-np.sum(np.sqrt(df ** 2 + dM ** 2)))

def smooth(speed, win=5):
    """Light moving-average low-pass (≈100 ms at 50 Hz) to remove the finite-difference
    FK speed noise that inflates the raw peak count, while preserving real submovements
    (which are ≥200 ms apart)."""
    speed = np.asarray(speed, float)
    if len(speed) < win:
        return speed
    k = np.ones(win) / win
    return np.convolve(speed, k, mode="same")

def stops(speed, v_move=0.05, v_stop=0.02):
    """Number of submovements = prominence-separated peaks above v_move (input already
    smoothed), so each is a genuine re-acceleration rather than sensor jitter."""
    pk, _ = find_peaks(np.asarray(speed, float), height=v_move, prominence=v_move - v_stop)
    return len(pk)

here = os.path.dirname(os.path.abspath(__file__))
fpath = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kinova_fluidity.csv"
mpath = sys.argv[2] if len(sys.argv) > 2 else "/tmp/kinova_skill_metrics.csv"

# fluidity log: rep,confidence,phase,ee_speed
d = np.genfromtxt(fpath, delimiter=",", names=True)
reps = np.unique(d["rep"]).astype(int)

# metrics: rep,success,confidence,cycle_time_s,observations
m = {}
if os.path.exists(mpath):
    md = np.genfromtxt(mpath, delimiter=",", names=True)
    for r, s, t, o in zip(md["rep"], md["success"], md["cycle_time_s"], md["observations"]):
        m[int(r)] = (int(s), float(t), float(o))

rows = []
for r in reps:
    sel = d["rep"] == r
    speed = smooth(d["ee_speed"][sel])     # de-noise once for both metrics
    conf = float(np.median(d["confidence"][sel]))
    succ, ctime, obs = m.get(int(r), (None, len(speed) / FS, None))
    rows.append((int(r), conf, sparc(speed), stops(speed), len(speed) / FS, succ, ctime, obs))

print(f"{'rep':>3} {'conf':>5} {'SPARC':>7} {'stops':>5} {'dur_s':>6} {'ok':>2} {'time_s':>6} {'obs':>4}")
for r, c, sp, st, du, ok, t, o in rows:
    print(f"{r:>3} {c:>5.2f} {sp:>7.2f} {st:>5d} {du:>6.1f} {str(ok):>2} {t:>6.1f} {o if o is None else int(o):>4}")

# novice (conf<0.3) vs skilled (conf>=0.7) summary
arr = np.array([(c, sp, st, du, t) for _, c, sp, st, du, _, t, _ in rows
                if not np.isnan(sp)], float)
nov = arr[arr[:, 0] < 0.3]; skl = arr[arr[:, 0] >= 0.7]
def smry(a, name):
    if len(a):
        print(f"{name:8} n={len(a)}  SPARC={a[:,1].mean():+.2f}  stops={a[:,2].mean():.1f}  "
              f"dur={a[:,3].mean():.1f}s  cycle={a[:,4].mean():.1f}s")
print("\n--- novice vs skilled ---")
smry(nov, "novice"); smry(skl, "skilled")

# plot vs confidence
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
fig.suptitle("Pick-and-place fluidity vs confidence (current controller)", fontweight="bold")
A = np.array([(c, sp, st, du, t) for _, c, sp, st, du, _, t, _ in rows], float)
for a, col, lab, yl in [(ax[0,0],1,"SPARC (→0 = smoother)","SPARC"),
                        (ax[0,1],2,"stops / submovements","stops"),
                        (ax[1,0],4,"cycle time (s)","s"),
                        (ax[1,1],3,"ActiveEFE duration (s)","s")]:
    a.scatter(A[:,0], A[:,col], c="tab:blue")
    a.set_xlabel("confidence"); a.set_title(lab); a.set_ylabel(yl); a.grid(alpha=0.3)
fig.tight_layout(rect=(0,0,1,0.96))
out = os.path.join(here, "fluidity_vs_confidence.png")
fig.savefig(out, dpi=130); print("\nwrote", out)
