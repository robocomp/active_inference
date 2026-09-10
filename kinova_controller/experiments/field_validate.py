#!/usr/bin/env python3
"""Validate the preference-field reformulation (option c, steps 1-2) against the hand-coded
controllers it should subsume — same clean rounds, natural learning, deterministic pick sweep:

  step 1 (PARITY):    field NOVICE reps (conf<0.3, prec_current→prec_stop=high) should match
                      the discrete-stop baseline (fluid_baseB) novice reps.
  step 2 (EMERGENCE): field SKILLED reps (conf>=0.7, prec_current→prec_pass=low) should match
                      the committed look-ahead blend (fluid_blendB) skilled reps.
"""
import os, numpy as np
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
def load(tag, lo, hi):   # reps with median confidence in [lo,hi), successful only
    d = np.genfromtxt(os.path.join(here, f"fluid_{tag}.csv"), delimiter=",", names=True)
    md = np.genfromtxt(os.path.join(here, f"fluid_{tag}_metrics.csv"), delimiter=",", names=True)
    succ = {int(r): int(s) for r, s in zip(md["rep"], md["success"])}
    tim  = {int(r): float(t) for r, t in zip(md["rep"], md["cycle_time_s"])}
    rows = []
    for r in np.unique(d["rep"]).astype(int):
        sel = d["rep"] == r; c = float(np.median(d["confidence"][sel]))
        if not (lo <= c < hi) or succ.get(int(r), 0) != 1: continue
        sp = smooth(d["ee_speed"][sel])
        rows.append((sparc(sp), stops(sp), tim.get(int(r), len(sp) / FS)))
    return np.array([x for x in rows if not np.isnan(x[0])], float)
def succ_count(tag):
    md = np.genfromtxt(os.path.join(here, f"fluid_{tag}_metrics.csv"), delimiter=",", names=True)
    return int(np.sum(md["success"])), len(md["rep"])

def row(name, a):
    if len(a) == 0: print(f"  {name:<22} (no reps)"); return
    print(f"  {name:<22} n={len(a):<2} SPARC={a[:,0].mean():+.2f}±{a[:,0].std():.2f}  "
          f"stops={a[:,1].mean():.1f}±{a[:,1].std():.1f}  cycle={a[:,2].mean():.1f}s")

print("grasp success:  base %d/%d   blend %d/%d   field %d/%d\n"
      % (*succ_count("baseB"), *succ_count("blendB"), *succ_count("fieldB")))
print("STEP 1 — PARITY (novice, conf<0.3): field should match the discrete-stop baseline")
row("baseB  (discrete stop)", load("baseB",  -1, 0.3))
row("fieldB (Π_c high)",      load("fieldB", -1, 0.3))
print("\nSTEP 2 — EMERGENCE (skilled, conf>=0.7): field should match the look-ahead blend")
row("blendB (hand blend)",    load("blendB", 0.7, 2))
row("fieldB (Π_c low)",       load("fieldB", 0.7, 2))
