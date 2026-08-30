#!/usr/bin/env python3
"""
path_departure_monitor — watch how far the robot is from the route it is following, live, and record
every episode where it leaves.

WHY A MONITOR AND NOT A NEW LOG. The controller already writes tracker_diag.csv at the control rate
with everything this needs: pd_cross_err_m (the tracker's own signed cross-track e_y), carrot_bear
(heading error), track_s (arc length), min_esdf (clearance to the nearest obstacle), pose_x/y/th,
cmd_adv, meas_speed, gate_scale and path_gen. So this attaches to a RUNNING controller and needs no
restart, no code change and no extra work on the control thread.

THE CRITERION IS DERIVED, NOT PICKED. "Off the path" is reported against the three length scales that
already mean something to this robot, and every episode records which one it crossed:

  drift  > 1 planner cell (CellSize, 0.06 m)   the resolution at which the route is even KNOWN.
                                               Below this, "off the path" is not a measurable claim.
  off    > the body's INSCRIBED radius          the body is a full radius from where the planner
                                               tested it, so it is standing somewhere nothing certified.
  unsafe   min_esdf < inscribed radius          the nearest obstacle is inside the body. This is not a
                                               tracking statement at all, it is a collision one, and it
                                               is reported even when cross-track is small.

★A ROUTE RE-AUTHORING IS NOT A DEPARTURE, and conflating them is the easy mistake here. When the band
deforms the curve or a repair splices it, the PATH moves under the robot and e_y steps without the
robot doing anything. tracker_diag carries path_gen for exactly this; an episode that begins within
one cycle of a path_gen change is tagged `route-moved` and kept out of the departure tally rather than
silently inflating it.

Usage:
    tools/path_departure_monitor.py [--file tracker_diag.csv] [--inscribed 0.230] [--cell 0.06]
                                    [--episodes path_departures.csv] [--quiet-s 5]
"""
import argparse, csv, math, os, sys, time

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="tracker_diag.csv")
    # Defaults are THIS robot's, and they are printed at startup so any number below is attributable.
    # inscribed: the controller's own [body] block reports it (P3Bot, from its mesh, 0.230 m).
    # cell: Planner.CellSize in etc/config.toml.
    p.add_argument("--inscribed", type=float, default=0.230)
    p.add_argument("--cell", type=float, default=0.06)
    p.add_argument("--episodes", default="path_departures.csv")
    p.add_argument("--quiet-s", type=float, default=5.0, help="seconds between live status lines")
    p.add_argument("--once", action="store_true", help="analyse what is already in the file, then exit")
    return p.parse_args()

def follow(path, once):
    """Yield rows, tolerating a controller restart that truncates or replaces the file."""
    f = None; ino = None; hdr = None
    while True:
        try:
            st = os.stat(path)
        except FileNotFoundError:
            if once: return
            time.sleep(0.5); continue
        if f is None or st.st_ino != ino or f.tell() > st.st_size:
            if f: f.close()
            f = open(path, "r", errors="replace"); ino = st.st_ino; hdr = None
            yield ("__restart__", None)
        line = f.readline()
        if not line:
            if once: return
            time.sleep(0.1); continue
        line = line.rstrip("\n")
        if not line or line.startswith("#"): continue
        parts = line.split(",")
        if hdr is None or parts[0] == "t_ms":
            if parts[0] == "t_ms":
                hdr = {k: i for i, k in enumerate(parts)}
            continue
        if len(parts) != len(hdr): continue
        yield (parts, hdr)

def num(row, hdr, key, default=float("nan")):
    try: return float(row[hdr[key]])
    except Exception: return default

class Episode:
    def __init__(self, t_ms, tier, row, hdr, route_moved):
        self.t0 = t_ms; self.t1 = t_ms; self.tier = tier
        self.peak = abs(num(row, hdr, "pd_cross_err_m"))
        self.peak_at = (num(row, hdr, "pose_x"), num(row, hdr, "pose_y"))
        self.s0 = num(row, hdr, "track_s"); self.s1 = self.s0
        self.min_clear = num(row, hdr, "min_esdf")
        self.max_head = abs(num(row, hdr, "carrot_bear"))
        self.max_speed = num(row, hdr, "meas_speed")
        self.min_gate = num(row, hdr, "gate_scale")
        self.route_moved = route_moved
        self.n = 1
    def update(self, t_ms, tier, row, hdr):
        self.t1 = t_ms; self.n += 1
        if tier > self.tier: self.tier = tier
        c = abs(num(row, hdr, "pd_cross_err_m"))
        if c > self.peak:
            self.peak = c; self.peak_at = (num(row, hdr, "pose_x"), num(row, hdr, "pose_y"))
        self.s1 = num(row, hdr, "track_s")
        self.min_clear = min(self.min_clear, num(row, hdr, "min_esdf"))
        self.max_head = max(self.max_head, abs(num(row, hdr, "carrot_bear")))
        self.max_speed = max(self.max_speed, num(row, hdr, "meas_speed"))
        self.min_gate = min(self.min_gate, num(row, hdr, "gate_scale"))

TIER_NAME = {1: "drift", 2: "off", 3: "unsafe"}

def main():
    a = parse_args()
    print(f"[monitor] watching {a.file}")
    print(f"[monitor] scales: 1 cell = {a.cell:.3f} m (route resolution) | inscribed = {a.inscribed:.3f} m "
          f"(body radius). drift > cell, off > inscribed, unsafe = clearance below inscribed.")
    print( "[monitor] an episode starting within one cycle of a path_gen change is tagged route-moved "
           "and excluded from the departure tally.")
    ep_f = open(a.episodes, "w", newline="")
    ep_w = csv.writer(ep_f)
    ep_w.writerow(["t_start_ms","dur_s","tier","peak_cross_m","peak_x","peak_y","s_from_m","s_to_m",
                   "min_clearance_m","max_heading_err_rad","max_speed_mps","min_gate_scale",
                   "cycles","route_moved"])
    ep_f.flush()

    cur = None; last_gen = None; last_gen_t = None
    n = 0; worst = 0.0; sum_sq = 0.0; last_report = 0.0
    counts = {1: 0, 2: 0, 3: 0}; moved = 0

    def close(e):
        nonlocal moved
        dur = (e.t1 - e.t0) / 1000.0
        ep_w.writerow([int(e.t0), f"{dur:.2f}", TIER_NAME[e.tier], f"{e.peak:.4f}",
                       f"{e.peak_at[0]:.3f}", f"{e.peak_at[1]:.3f}", f"{e.s0:.2f}", f"{e.s1:.2f}",
                       f"{e.min_clear:.3f}", f"{e.max_head:.3f}", f"{e.max_speed:.3f}",
                       f"{e.min_gate:.3f}", e.n, int(e.route_moved)])
        ep_f.flush()
        if e.route_moved: moved += 1
        else: counts[e.tier] += 1
        tag = "  [route-moved: the CURVE shifted, not the robot]" if e.route_moved else ""
        print(f"[{TIER_NAME[e.tier]:6s}] {dur:5.2f} s  peak |e_y| {e.peak:.3f} m at ({e.peak_at[0]:+.2f},"
              f"{e.peak_at[1]:+.2f})  s {e.s0:.1f}->{e.s1:.1f} m  clear {e.min_clear:.3f} m  "
              f"e_psi {e.max_head:.2f} rad  v {e.max_speed:.2f} m/s{tag}", flush=True)

    for row, hdr in follow(a.file, a.once):
        if row == "__restart__":
            if cur: close(cur); cur = None
            last_gen = None
            print("[monitor] --- log restarted (controller relaunched or file rotated) ---", flush=True)
            continue
        t = num(row, hdr, "t_ms")
        cross = num(row, hdr, "pd_cross_err_m")
        clear = num(row, hdr, "min_esdf")
        gen = num(row, hdr, "path_gen")
        if math.isnan(t) or math.isnan(cross): continue
        if last_gen is not None and gen != last_gen: last_gen_t = t
        last_gen = gen

        n += 1; sum_sq += cross * cross; worst = max(worst, abs(cross))

        tier = 0
        if abs(cross) > a.cell: tier = 1
        if abs(cross) > a.inscribed: tier = 2
        if not math.isnan(clear) and clear < a.inscribed: tier = 3

        if tier:
            if cur is None:
                # Within one control cycle (~50 ms; be generous at 250 ms) of the route changing, this
                # is the curve moving, not the robot.
                rm = last_gen_t is not None and (t - last_gen_t) <= 250.0
                cur = Episode(t, tier, row, hdr, rm)
            else:
                cur.update(t, tier, row, hdr)
        elif cur is not None:
            close(cur); cur = None

        now = time.time()
        if now - last_report >= a.quiet_s:
            last_report = now
            rms = math.sqrt(sum_sq / n) if n else 0.0
            print(f"[monitor] {n:7d} cycles | cross rms {rms*1000:6.1f} mm  worst {worst*1000:6.1f} mm | "
                  f"episodes drift {counts[1]} off {counts[2]} unsafe {counts[3]} "
                  f"(+{moved} route-moved) | now |e_y| {abs(cross)*1000:5.1f} mm clear {clear:.3f} m",
                  flush=True)

    if cur: close(cur)
    rms = math.sqrt(sum_sq / n) if n else 0.0
    print(f"\n[monitor] TOTAL {n} cycles | cross rms {rms*1000:.1f} mm | worst {worst*1000:.1f} mm")
    print(f"[monitor] departures: drift {counts[1]}, off {counts[2]}, unsafe {counts[3]}; "
          f"route-moved (excluded) {moved}")
    print(f"[monitor] episodes -> {a.episodes}")

if __name__ == "__main__":
    main()
