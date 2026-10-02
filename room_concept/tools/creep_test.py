#!/usr/bin/env python3
"""creep_test.py — the registered slow-approach test of thesis §ex-rest (Chapter 9).

Drives the SIMULATED base directly through the Webots bridge's OmniRobot interface with a fixed
schedule of creeping speeds, so that the rest-on-prediction mixture (PreintZuptOnPrediction) can be
graded where it can bite: real motion slower than the odometry noise distinguishes.

    python3 tools/creep_test.py                 # run the sweep (~11 min); Ctrl-C stops the base
    python3 tools/creep_test.py --dry-run       # print the schedule and its duration only
    python3 tools/creep_test.py --report A.csv B.csv   # grade two heading logs (off, on) by speed band

Protocol (as registered in §ex-rest):
  * the controller agent must NOT be commanding the base (stop it, or leave it with no mission);
  * room_concept running, robot parked with >= 0.7 m clear in front AND behind, and room to turn;
  * run once with PreintZuptOnPrediction = true and once with = false (restart room_concept between,
    the flag is read at start-up), same session, same start pose. Each run writes its own
    tmp/heading/heading_<start>.csv; pass those two to --report.

Every leg goes forward then back by the same distance, so the robot ends where it started.
"""
import argparse, glob, os, sys, time

SPEEDS_MS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]   # m/s, forward creep
TURNS_RS  = [0.01, 0.02, 0.05]                      # rad/s, turn on the spot
LEG_MAX_S, LEG_MAX_M = 15.0, 0.6                    # a leg is at most 15 s and 0.6 m
STOP_S, SETTLE_S, FINAL_S = 5.0, 30.0, 60.0
REPEATS = 2
RATE_HZ = 10.0                                      # commands are re-sent; the bridge may time out a stale one

HERE = os.path.dirname(os.path.abspath(__file__))
BRIDGE_GEN = os.path.join(HERE, "..", "..", "..", "webots-bridge", "generated")


def schedule():
    """List of (label, side_m_s, adv_m_s, rot_rad_s, seconds)."""
    s = [("settle", 0.0, 0.0, 0.0, SETTLE_S)]
    for rep in range(REPEATS):
        for v in SPEEDS_MS:
            T = min(LEG_MAX_S, LEG_MAX_M / v)
            s += [(f"fwd {100*v:.1f}cm/s r{rep}", 0, v, 0, T), ("stop", 0, 0, 0, STOP_S),
                  (f"back {100*v:.1f}cm/s r{rep}", 0, -v, 0, T), ("stop", 0, 0, 0, STOP_S)]
        for w in TURNS_RS:
            s += [(f"ccw {w:.2f}rad/s r{rep}", 0, 0, w, LEG_MAX_S), ("stop", 0, 0, 0, STOP_S),
                  (f"cw {w:.2f}rad/s r{rep}", 0, 0, -w, LEG_MAX_S), ("stop", 0, 0, 0, STOP_S)]
    s.append(("final park", 0, 0, 0, FINAL_S))
    return s


def run(endpoint):
    import Ice
    Ice.loadSlice(f"-I{BRIDGE_GEN} --all {os.path.join(BRIDGE_GEN, 'OmniRobot.ice')}")
    import RoboCompOmniRobot
    with Ice.initialize(sys.argv) as ic:
        base = RoboCompOmniRobot.OmniRobotPrx.checkedCast(ic.stringToProxy(f"omnirobot:{endpoint}"))
        if base is None:
            sys.exit(f"no OmniRobot at {endpoint} -- is the Webots bridge running?")
        os.makedirs(os.path.join(HERE, "..", "tmp", "heading"), exist_ok=True)
        logp = os.path.join(HERE, "..", "tmp", "heading", time.strftime("creep_%Y-%m-%d_%H-%M-%S.csv"))
        with open(logp, "w") as log:
            log.write("wall_ms,label,side_m_s,adv_m_s,rot_rad_s,seconds\n")
            try:
                for label, side, adv, rot, T in schedule():
                    log.write(f"{int(time.time()*1000)},{label},{side},{adv},{rot},{T:.2f}\n"); log.flush()
                    print(f"{time.strftime('%H:%M:%S')}  {label:24s} {T:5.1f} s")
                    t_end = time.time() + T
                    while time.time() < t_end:
                        # setSpeedBase(side mm/s, forward mm/s, rot rad/s) -- the controller's convention
                        base.setSpeedBase(side * 1000.0, adv * 1000.0, rot)
                        time.sleep(1.0 / RATE_HZ)
            finally:
                base.stopBase()
                print("base stopped; schedule log:", os.path.normpath(logp))


def report(paths):
    """Grade heading logs by TRUE speed band: signed and |error| of predicted vs true path, 2-s windows."""
    import numpy as np
    W = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    def load(f):
        L = [l for l in open(f) if not l.startswith("#")]; h = L[0].strip().split(",")
        rows = [l.strip().split(",") for l in L[1:]]; rows = [r for r in rows if len(r) == len(h)]
        a = np.array([[float(x) if x not in ("", "nan", "-nan") else np.nan for x in r] for r in rows])
        return {k: a[:, i] for i, k in enumerate(h)}
    edges = [0.0, 0.01, 0.02, 0.035, 0.065, 0.10, 0.20, 9.9]
    names = ["<1", "1-2", "2-3.5", "3.5-6.5", "6.5-10", "10-20", ">20"]
    res = {}
    for f in paths:
        d = load(f); t = (d["ts_ms"] - d["ts_ms"][0]) / 1000.0
        ex, ey, gx, gy = d["est_x"], d["est_y"], d["gt_x"], d["gt_y"]
        pd = np.hypot(d["pred_x"][1:] - ex[:-1], d["pred_y"][1:] - ey[:-1]); gd = np.hypot(np.diff(gx), np.diff(gy))
        on = int(np.nanmax(d["rest_on"])) if "rest_on" in d else 0
        w2 = (t[1:] // 2).astype(int); band = {n: [] for n in names}
        for b in np.unique(w2):
            m = w2 == b
            if m.sum() < 10: continue
            T = t[1:][m][-1] - t[1:][m][0] + 1e-9; path = gd[m].sum()
            if path < 0.004: continue                       # under 2 mm/s: not motion
            k = np.searchsorted(edges, path / T, side="right") - 1
            band[names[min(k, len(names) - 1)]].append((pd[m].sum() - path) / path)
        res[f] = (on, band)
        print(f"\n{os.path.basename(f)}  rest mixture {'ON' if on else 'OFF'}")
        print("  band (cm/s)   n   signed median   |err| median")
        for n in names:
            e = np.array(band[n])
            if len(e): print(f"  {n:9s} {len(e):5d}   {100*np.median(e):+8.2f} %     {100*np.median(np.abs(e)):7.2f} %")
    ons = [f for f in paths if res[f][0] == 1]; offs = [f for f in paths if res[f][0] == 0]
    if len(ons) == 1 and len(offs) == 1:
        a, b = res[offs[0]][1], res[ons[0]][1]; ok_all = True; graded = 0
        print("\nREGISTERED CRITERION (bands below 6.5 cm/s with >= 20 windows per arm):")
        for n in names[:4]:
            if len(a[n]) < 20 or len(b[n]) < 20:
                print(f"  {n:9s} not graded (n off={len(a[n])}, on={len(b[n])})"); continue
            graded += 1
            sm = 100 * np.median(b[n]); dab = 100 * (np.median(np.abs(b[n])) - np.median(np.abs(a[n])))
            ok = abs(sm) <= 5 and dab <= 5; ok_all &= ok
            print(f"  {n:9s} signed {sm:+6.2f} % (|.|<=5)   |err| on-off {dab:+6.2f} pts (<=5)   {'PASS' if ok else 'FAIL'}")
        print("  VERDICT:", "INCONCLUSIVE (no band graded)" if graded == 0 else ("PASS" if ok_all else "FAIL"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="tcp -h localhost -p 10004")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", nargs="+")
    a = ap.parse_args()
    if a.report:
        report(a.report)
    elif a.dry_run:
        tot = sum(x[4] for x in schedule())
        for x in schedule(): print(f"{x[0]:24s} side {x[1]:+.3f} adv {x[2]:+.3f} rot {x[3]:+.3f}  {x[4]:5.1f} s")
        print(f"total {tot/60:.1f} min")
    else:
        run(a.endpoint)
