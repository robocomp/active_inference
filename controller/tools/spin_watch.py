#!/usr/bin/env python3
"""spin_watch — report, live, every episode where the robot stops advancing because its CARROT
collapsed onto it, and separate those from episodes where something is genuinely in the way.

WHY THIS AND NOT path_departure_monitor. That monitor answers "is the robot ON its route"; it is blind
to a robot that sits exactly on the route and turns in place, because cross-track stays ~0 the whole
time. The stop the user reports in narrow passages is exactly that shape, so it needs its own channel.

THE DISCRIMINATION IS THE POINT, and it is made from columns the controller already writes:
    carrot_dist  < carrot-eps  ★IN PLAIN MODE THIS COLUMN IS NOT A CARROT DISTANCE. plain_tracker.cpp
                               overwrites it with v_cmd * plain_T_lag, so "carrot_dist ~ 0" MEANS
                               "commanded forward speed is zero" and nothing else. It is used here as
                               exactly that and no causal claim is attached to it. (carrot_bear is
                               likewise overwritten with e_psi, the heading error.)
    min_esdf     vs inscribed  what the map thinks is there. A stop with min_esdf well above the
                               inscribed radius is NOT a blocked robot, whatever it looks like.
    gate_hard_stop  the controller's own "I refused for safety" flag. (n_collisions was
                    dropped from the CSV with the MPPI sampler, 2026-09-09.)
Every episode prints which of those held, so "it stopped" is never reported without saying why.
"""
import argparse, csv, math, os, time

def rows_from(path, start_at_end):
    f = open(path)
    hdr = None
    for line in f:
        if line.startswith('#'): continue
        hdr = next(csv.reader([line])); break
    if start_at_end: f.seek(0, os.SEEK_END)
    buf = ''
    while True:
        chunk = f.readline()
        if not chunk:
            time.sleep(0.1); continue
        buf += chunk
        if not buf.endswith('\n'): continue
        line, buf = buf, ''
        if line.startswith('#') or line.startswith('t_ms'): continue
        try: vals = next(csv.reader([line]))
        except Exception: continue
        if len(vals) != len(hdr): continue
        yield dict(zip(hdr, vals))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="tracker_diag.csv")
    p.add_argument("--carrot-eps", type=float, default=0.02, help="m; carrot this short is 'on the robot'")
    p.add_argument("--inscribed", type=float, default=0.230, help="m; body inscribed radius")
    p.add_argument("--min-report-s", type=float, default=0.5)
    p.add_argument("--status-s", type=float, default=30.0)
    p.add_argument("--from-start", action="store_true")
    a = p.parse_args()
    F = lambda r, k: float(r[k])

    ep = None; n = 0; n_ep = 0; spin_s = 0.0; t_status = time.time(); t_first = None; t_last = None
    print(f"[spin_watch] v_cmd*T_lag<{a.carrot_eps} = zero commanded advance | inscribed {a.inscribed} m", flush=True)
    for r in rows_from(a.file, not a.from_start):
        try:
            t = int(r['t_ms']) / 1000.0; cd = F(r, 'carrot_dist')
        except Exception:
            continue
        n += 1
        t_first = t_first or t; t_last = t
        if cd < a.carrot_eps:
            if ep is None: ep = {'t0': t, 'th0': F(r, 'pose_th'), 'rows': []}
            ep['rows'].append(r)
        elif ep is not None:
            dur = ep['rows'][-1] and (float(ep['rows'][-1]['t_ms']) / 1000.0 - ep['t0'])
            if dur >= a.min_report_s:
                rs = ep['rows']; n_ep += 1; spin_s += dur
                th = [F(x, 'pose_th') for x in rs]
                swept = sum(abs((th[i+1] - th[i] + math.pi) % (2*math.pi) - math.pi) for i in range(len(th)-1))
                esdf = min(F(x, 'min_esdf') for x in rs)
                dx = F(rs[-1], 'pose_x') - F(rs[0], 'pose_x'); dy = F(rs[-1], 'pose_y') - F(rs[0], 'pose_y')
                moved = math.hypot(dx, dy)
                hard = any(F(x, 'gate_hard_stop') > 0 for x in rs)
                ds = F(rs[-1], 'track_s') - F(rs[0], 'track_s')
                gen = F(rs[-1], 'path_gen') != F(rs[0], 'path_gen')
                if hard:              why = "BLOCKED (controller refused: gate hard-stop)"
                elif esdf < a.inscribed: why = f"BLOCKED (map says {esdf:.2f} m < inscribed {a.inscribed} m)"
                else:                 why = f"NOT BLOCKED - {esdf:.2f} m free; zero advance held while turning in place"
                print(f"[spin  ] {dur:5.2f} s at ({F(rs[0],'pose_x'):+.2f},{F(rs[0],'pose_y'):+.2f}) "
                      f"turned {math.degrees(swept):6.1f} deg, advanced {moved*100:5.1f} cm, "
                      f"route s {ds:+.2f} m{' [route rebuilt]' if gen else ''} | {why}", flush=True)
            ep = None
        if time.time() - t_status > a.status_s:
            span = max(1e-6, (t_last - t_first))
            print(f"[status] {n} cycles / {span:5.0f} s | spinning-in-place {spin_s:6.1f} s "
                  f"({100*spin_s/span:4.1f}% of the time) in {n_ep} episodes | now carrot {cd:5.2f} m "
                  f"adv {F(r,'cmd_adv'):5.2f} clear {F(r,'min_esdf'):.2f} m", flush=True)
            t_status = time.time()

if __name__ == "__main__":
    main()
