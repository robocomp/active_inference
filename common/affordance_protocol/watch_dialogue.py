#!/usr/bin/env python3
"""Follow the room_concept <-> controller conversation live, and NAME the failure when it starts.

Built 2026-08-19 after a day in which every diagnosis was reconstructed by hand from two logs on one
time axis. Each of the failures below took ~20 minutes of robot time to identify; all of them are
visible in seconds if you put the two sides next to each other and know what to look for.

  ROOM  = room_concept (producer): chooses a standpoint, arms the shared afford_room node
  CTRL  = controller  (consumer): selects it, plans, drives, reports

usage:
  watch_dialogue.py                 one snapshot of the last 45 s + verdict
  watch_dialogue.py --follow        keep watching, print each new event, alert on a pattern
  watch_dialogue.py --window 120    seconds of history to consider
"""
import sys, json, math, glob, time, os

ROOM_GLOB = "room_concept/tmp/sdf_localizer/log_*.csv"
TRACK_CSV = "controller/tracker_diag.csv"   # the PLAIN tracker's per-cycle log (was mppi_diag)
CTRL_JSON = "controller/affordance_select.jsonl"

def read_room(path, since=0.0):
    out = []
    for line in open(path):
        p = line.split(",")
        if len(p) < 104:
            continue
        try:
            t = float(p[1])
            if t <= since: continue
            out.append({"t": t, "rx": float(p[51]), "ry": float(p[52]), "out": p[96].strip(),
                        "compl": int(float(p[97])), "tx": float(p[100]), "ty": float(p[101]),
                        "ok": p[102].strip(), "adv": float(p[4]), "rot": float(p[5])})
        except (ValueError, IndexError):
            continue
    return out

def read_ctrl(path, since=0.0):
    out = []
    if not os.path.exists(path): return out
    for line in open(path):
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except json.JSONDecodeError: continue      # torn tail line: detected, never misread
        if r.get("t_ms", 0) > since: out.append(r)
    return out

def dist(a, b, c, d): return math.hypot(a - c, b - d)

# ── THE FAILURE PATTERNS, each one measured on the real robot ────────────────────────────────────
def read_path_tracking(path, n_tail=4000):
    """Cross-track error and arc-length progress. ★path_kappa is a -999 SENTINEL, not a curvature —
    do not use it; it is the fifth column today whose name promises what it does not carry."""
    import csv as _csv
    if not os.path.exists(path): return []
    rows = [l for l in open(path) if not l.lstrip().startswith("#")]
    rd = list(_csv.reader(rows))
    if len(rd) < 2: return []
    hdr = rd[0]; I = {c: i for i, c in enumerate(hdr)}
    if "pd_cross_err_m" not in I: return []
    out = []
    for r in rd[1:][-n_tail:]:
        if len(r) != len(hdr): continue
        try:
            out.append((abs(float(r[I["pd_cross_err_m"]])), float(r[I["track_s"]]),
                        float(r[I["cmd_adv"]]), float(r[I["cmd_rot"]])))
        except (ValueError, KeyError):
            continue
    return out

def diagnose_path(track):
    """★THE ROBOT MUST STAY ON ITS PATH — every safety layer is computed against it. Leaving it is a
    safety event, not a performance one, so it is reported separately from the protocol patterns."""
    v = []
    if len(track) < 50: return v
    THR = 0.35
    ce = [t[0] for t in track]
    eps, i = [], 0
    while i < len(ce):
        if ce[i] > THR:
            j = i
            while j < len(ce) - 1 and ce[j + 1] > THR: j += 1
            seg = range(i, j + 1)
            adv = track[j][1] - track[i][1]
            rots = [track[k][3] for k in seg]
            flips = sum(1 for a, b in zip(rots, rots[1:]) if a * b < 0 and abs(a) > 0.4 and abs(b) > 0.4)
            eps.append(((j - i + 1) * 0.05, max(ce[k] for k in seg), adv, flips))
            i = j + 1
        else:
            i += 1
    p50 = sorted(ce)[len(ce) // 2]
    if eps:
        worst = max(eps)
        off = sum(e[0] for e in eps)
        kind = ("CHATTER (two authorities alternating at the rotation cap)" if worst[3] > 3
                else "DRIFT (one controller, no arc-length progress)" if abs(worst[2]) < 0.3
                else "wide turn")
        v.append(f"✗ OFF PATH: {len(eps)} excursion(s) >{THR} m, worst {worst[0]:.1f}s peak "
                 f"{worst[1]:.2f} m, {off:.0f}s total ({100*off/(len(ce)*0.05):.0f}% of window). "
                 f"Shape: {kind}. Every safety layer is computed against the path.")
    elif p50 < 0.05:
        v.append(f"✓ on path: cross-track p50 {p50*1000:.0f} mm, no excursion >{THR} m")
    return v

def diagnose(room, ctrl, window_s):
    if len(room) < 20:
        return ["not enough data"]
    v = []
    span = (room[-1]["t"] - room[0]["t"]) / 1000.0 or 1.0
    moved = 0.0
    for a, b in zip(room, room[1:]):
        d = dist(a["rx"], a["ry"], b["rx"], b["ry"])
        if d < 0.5: moved += d
    dcompl = room[-1]["compl"] - room[0]["compl"]
    rate = dcompl / (span / 60.0)
    last = room[-1]
    d_tgt = dist(last["rx"], last["ry"], last["tx"], last["ty"])
    cmd = sum(abs(r["adv"]) for r in room) / len(room), sum(abs(r["rot"]) for r in room) / len(room)

    # 1. PHANTOM COMPLETIONS — the producer completing its own offers
    far = [r for r in room if not math.isnan(r["tx"]) and dist(r["rx"], r["ry"], r["tx"], r["ty"]) > 1.0]
    if dcompl >= 3 and moved < 0.5 and len(far) > 0.8 * len(room):
        v.append(f"✗ PHANTOM COMPLETIONS: {dcompl} completions in {span:.0f}s while the robot moved "
                 f"{moved:.2f} m and stayed >1 m from the offered cell. The producer is completing its "
                 f"own offers — a not-pending reading believed as an edge (affordance_manager.cpp:302).")

    # 2. STARVATION — the consumer rejecting everything
    if ctrl:
        rej = [c for c in ctrl if c.get("reject")]
        notgt = [c for c in ctrl if not c.get("has_target")]
        if len(rej) > 0.5 * len(ctrl):
            kinds = {}
            for c in rej: kinds[c["reject"]] = kinds.get(c["reject"], 0) + 1
            v.append(f"✗ SELECTION STARVATION: consumer rejected on {100*len(rej)//len(ctrl)}% of cycles "
                     f"{kinds}. If 'just-completed' dominates, the suppression is matching cells it "
                     f"should not (it is keyed on the STANDPOINT and must fail OPEN).")
        elif len(notgt) > 0.5 * len(ctrl) and moved < 0.5:
            v.append(f"✗ NO TARGET: consumer held no target on {100*len(notgt)//len(ctrl)}% of cycles "
                     f"and the robot moved {moved:.2f} m. Producer is offering but nothing is taken.")

    # 3. STALE CLAIM — node Executing, consumer has nothing
    if ctrl:
        stale = [c for c in ctrl
                 if not c.get("has_target") and any(x.get("state") == "Executing" for x in c.get("candidates", []))]
        if len(stale) > 0.15 * len(ctrl):
            v.append(f"✗ STALE CLAIM: {100*len(stale)//len(ctrl)}% of cycles read Executing while the "
                     f"consumer held no target — it dropped the target without releasing the claim. "
                     f"The producer needs its execution LEASE to reclaim.")

    # 4. REFUSE LOOP — busy wire, no progress
    refus = [r for r in room if r["out"] == "3"]
    if rate > 25 and moved < 2.0:
        v.append(f"✗ REFUSE LOOP: {rate:.0f} completions/min with {moved:.2f} m travelled. A completion "
                 f"path that costs no physical time is cycling at software speed.")
    elif len(refus) > 0.4 * len(room):
        v.append(f"⚠ high refusal share: {100*len(refus)//len(room)}% of samples")

    # 5. FROZEN WITH WORK TO DO — the consumer has everything and issues nothing
    if ctrl:
        ready = [c for c in ctrl if c.get("has_target") and c.get("has_plan")]
        if moved < 0.3 and len(ready) > 0.5 * len(ctrl) and cmd[0] < 0.02 and cmd[1] < 0.05:
            v.append(f"✗ FROZEN WITH A PLAN: target and plan held on {100*len(ready)//len(ctrl)}% of "
                     f"cycles, commands {cmd[0]:.4f}/{cmd[1]:.4f}, moved {moved:.2f} m. Not a protocol "
                     f"fault — the consumer cannot produce a command from where it is.")

    # 6. TARGET CHURN
    # ★NaN targets (before the producer has chosen anything) are not cells. Each NaN is a distinct
    # object in a set, so counting them inflates the churn figure and fires a false alarm — which this
    # detector did on its very first run. A detector that cries wolf is the same defect as one that
    # stays silent; both stop being read.
    cells = {(round(r["tx"], 2), round(r["ty"], 2)) for r in room
             if not (math.isnan(r["tx"]) or math.isnan(r["ty"]))}
    real = [r for r in room if not (math.isnan(r["tx"]) or math.isnan(r["ty"]))]
    span_real = ((real[-1]["t"] - real[0]["t"]) / 1000.0) if len(real) > 1 else span
    if len(cells) > span_real / 2 and len(cells) > 4:
        v.append(f"⚠ TARGET CHURN: {len(cells)} distinct cells in {span_real:.0f}s — no target survives long "
                 f"enough to drive to.")

    v += diagnose_path(read_path_tracking(TRACK_CSV))
    if not any(x.startswith("✗") for x in v):
        v.append(f"✓ healthy: {dcompl} completions in {span:.0f}s ({rate:.1f}/min), moved {moved:.1f} m "
                 f"({moved/span:.3f} m/s), d_target now {d_tgt:.2f} m")
    return v

def snapshot(window_s, quiet=False):
    rf = sorted(glob.glob(ROOM_GLOB))
    if not rf: return ["no room log"], []
    room_all = read_room(rf[-1])
    ctrl_all = read_ctrl(CTRL_JSON)
    if not room_all: return ["no room rows"], []
    tend = room_all[-1]["t"]
    room = [r for r in room_all if 0 <= tend - r["t"] <= window_s * 1000]
    ctrl = [c for c in ctrl_all if 0 <= tend - c.get("t_ms", 0) <= window_s * 1000]
    ev, prev = [], None
    for r in room:
        k = (round(r["tx"], 2), round(r["ty"], 2), r["out"], r["compl"])
        if k != prev:
            ev.append((r["t"], "ROOM", f"offers ({r['tx']:+.2f},{r['ty']:+.2f}) outcome={r['out']} "
                                      f"compl={r['compl']} robot=({r['rx']:+.2f},{r['ry']:+.2f}) "
                                      f"d={dist(r['rx'],r['ry'],r['tx'],r['ty']):.2f}"))
            prev = k
    prevc = None
    for c in ctrl:
        cd = c.get("candidates", [{}])[0] if c.get("candidates") else {}
        k = (c.get("has_target"), c.get("has_plan"), c.get("reject"), cd.get("state"))
        if k != prevc:
            ev.append((c["t_ms"], "CTRL", f"target={c.get('has_target')} plan={c.get('has_plan')} "
                                         f"reject='{c.get('reject')}' sees={cd.get('state','-')}"))
            prevc = k
    ev.sort()
    return diagnose(room, ctrl, window_s), ev

def main():
    follow = "--follow" in sys.argv
    window = 45
    if "--window" in sys.argv: window = int(sys.argv[sys.argv.index("--window") + 1])
    if not follow:
        verdict, ev = snapshot(window)
        for t, w, s in ev[-30:]:
            print(f"  {w}  {s}")
        print()
        for v in verdict: print(v)
        return
    print(f"following the dialogue (window {window}s) — ctrl-c to stop\n")
    last_verdict = None
    while True:
        verdict, _ = snapshot(window)
        key = "|".join(verdict)
        if key != last_verdict:
            print(f"[{time.strftime('%H:%M:%S')}]")
            for v in verdict: print(f"  {v}")
            print()
            last_verdict = key
        time.sleep(5)

if __name__ == "__main__":
    main()
