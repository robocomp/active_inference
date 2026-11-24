#!/usr/bin/env python3
"""The parked test (spec 2026-11-23 shared-engine, Tests item 3; plan rev 2 Task P1) checked on a live run.

    parked_report.py <run>          <run> = a run folder (parked_test.sh), or its name under ../crossing/

For each object of aba_segments.tsv (A1 -> B -> A2, DWELL_S each) and each theta component (cx cy H w h yaw; h = depth), with
the predictions REGISTERED in docs/superpowers/results/2026-11-25-p1-predictions.md (the model: aba_model.py):
  (i)  flat:   sigma over the second half of the A1 dwell does not move: |sigma_end/sigma_mid - 1| <= 10 % (two reported
               sigmas) + the permanence prior's growth sqrt(1 + n q0/sigma_mid^2) - 1 over that half (q0 per predict step);
       floor:  sigma_A1 >= the shape-latent floor x (1 - 5 %): PC Delta/sqrt(12 n_c), S Delta/sqrt(12) (with the class prior);
       level:  sigma_A1 in [sigma_lo (1 - 5 %), sigma_hi (1 + 5 %)], the two view sets of aba_model.py;
  (ii) drop:   sigma_B / sigma_A1 in [min(r_lo, r_hi) (1 - 10 %), max(r_lo, r_hi) (1 + 10 %)], r = the model's ratio with the
               logged rho; and sigma_B in its level band.
Each check is evaluated under both shape-latent hypotheses (PC = one latent per channel, the engine's; S = one shared). A DOF
with Delta = 0 has no floor (reported n/a, not passed). The run's GROUND-TRUTH error at the end of the last dwell gives a
second, independent reading: sum over objects and DOFs with Delta > 0 of ln N(e; 0, s_S^2) - ln N(e; 0, s_PC^2), with
s_X = the logged sigma x (model sigma under X / model sigma under PC) (> 0 favours S). n = 3 objects: weak evidence, said so.

A run with no aba_segments.tsv (e.g. objects_replay1) is parsed and its parked stretches (consecutive updates with every logged
rho exactly 1) are listed with their sigma trace — NO pass/fail can come from it: there is no scripted A/B/A to predict.
"""
import math, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # run with python3 -I: the siblings below
import aba_model as Mdl
import aba_runlog as R
from aba_geometry import Frames, read_gauge, read_gt_error, wrap

DOFS = Mdl.DOFS


def verdict(ok): return 'PASS' if ok else 'FAIL'


def flat_check(ups, t0, t1, prm):
    """(sigma_mid, sigma_end, rel change, allowance, ok) per DOF over the second half of [t0, t1]."""
    mid_t = t0 + (t1 - t0) / 2
    second = [d for d in ups if mid_t <= d['t'] <= t1]
    if len(second) < 2: return None
    n_pred = (t1 - mid_t) / 1000.0 / prm['predict_dt_s']
    out = []
    for k in range(6):
        s0, s1 = second[0]['sig'][k], second[-1]['sig'][k]
        grow = math.sqrt(1 + n_pred * prm['q0'][k] / (s0 * s0)) - 1 if s0 > 0 else math.inf
        rel = s1 / s0 - 1 if s0 > 0 else math.nan
        out.append((s0, s1, rel, Mdl.TOL_RATIO + grow, abs(rel) <= Mdl.TOL_RATIO + grow))
    return out


def no_aba(run, cls_list):
    print(f'{run.name}: no aba_segments.tsv — parsing and parked-stretch extraction only; NO PASS/FAIL can come from this run '
          '(no scripted A/B/A, so there is nothing registered to predict).')
    # this checkout's manifests: an older run's checkout may predate [shape] / [prior.permanence] (objects_replay1 ran at 0c515eaf)
    root = R.checkout_root(Path('/nonexistent'))
    print(f'  (manifests/configs for the descriptive allowance: this checkout, {root}; the run ran {R.checkout_root(run)})')
    for cls in cls_list:
        rows = R.ai2_rows(run, cls)
        if not rows: print(f'  {cls}: no {cls}_ai2_log.csv'); continue
        prm = Mdl.class_params(cls, root)
        nodes = sorted({d['node'] for d in rows})
        print(f'  {cls}: {len(rows)} rows, nodes {nodes}, manifest/config from {root}')
        for node in nodes:
            ups = R.updates(rows, node)
            runs = R.parked_runs(ups)
            print(f'    {node}: {len(ups)} updates (cam {sum(d["cam"] for d in ups)}, LiDAR {sum(d["L"] for d in ups)}), '
                  f'{len(runs)} parked stretches')
            for r in runs[:3]:
                t0, t1 = r[0]['t'], r[-1]['t']
                fc = flat_check(ups, t0, t1, prm)
                print(f'      stretch {(t1 - t0) / 1000:.1f} s, {len(r)} updates: sigma start/end ' +
                      ' '.join(f'{DOFS[k]} {r[0]["sig"][k]:.4f}/{r[-1]["sig"][k]:.4f}' for k in range(6)))
                if fc:
                    print('        second-half change (descriptive only): ' +
                          ' '.join(f'{DOFS[k]} {fc[k][2]:+.3f}' for k in range(6)))


def main(arg):
    run = R.run_path(arg)
    segs = R.segments(run)
    if not segs and (run / 'aba_segments.tsv').exists():
        print(f'{run.name}: aba_segments.tsv has no well-formed row: NO VERDICT -> FAIL'); return 1
    if not segs:
        no_aba(run, ('nightstand', 'bed', 'toilet')); return 0
    root = R.checkout_root(run)
    print(f'{run.name}: parked test, predictions registered in docs/superpowers/results/2026-11-25-p1-predictions.md; '
          f'manifest/config from {root}')
    score = {h: [0, 0] for h in Mdl.HYPS}   # passes, checks
    llr, llr_n = 0.0, 0
    tol_m, tol_src = R.controller_goal_tolerance()
    print(f'visit validity: arrival within {tol_m:.3f} m of the scripted standpoint ({tol_src})')
    for defn, S in segs.items():
        why = R.visit_validity(S, tol_m)
        if why:
            # the visit did not happen as scripted: nothing it logged may pass a check
            print(f'== {defn}: NO VERDICT -> FAIL (' + '; '.join(why) + ')')
            for h in Mdl.HYPS: score[h][1] += 1
            continue
        plan = R.plan_of(run, defn)
        if plan is None:
            print(f'== {defn}: no aba_plan_{defn}.json: NO VERDICT -> FAIL')
            for h in Mdl.HYPS: score[h][1] += 1
            continue
        T = plan['truth']; cls = T['cls']; prm = Mdl.class_params(cls, root)
        rows = R.ai2_rows(run, cls)
        a1 = S['A1']
        m = R.match_node_to_truth(run, rows, S, T)
        node = m['node']
        print(f'== {defn} ({cls}): {R.describe_match(m)}')
        if node is None:
            print('   no node is the object: NO VERDICT -> FAIL')
            for h in Mdl.HYPS: score[h][1] += 1
            continue
        ups = R.updates(rows, node)
        order = list(R.SEGS)   # all three present (visit_validity)
        dwells = [(S[s]['t_dwell0_ms'], S[s]['t_dwell1_ms']) for s in order]
        logged, pred, nch = {}, {}, {}
        for i, s in enumerate(order):
            d = R.last_before(ups, dwells[i][1])
            logged[s] = d['sig'] if d else [math.nan] * 6
            vhi = R.views_hi(ups, dwells[:i + 1]); vlo = R.views_lo(ups, dwells[i][1])
            pred[s] = dict(hi=R.predicted_sigmas(vhi, prm), lo=R.predicted_sigmas(vlo, prm))
            nch[s] = int(any(v['cam'] for v in vlo)) + int(any(v['L'] for v in vlo))
            rc = [v for v in vhi][-1] if vhi else None
            if rc and i > 0:
                print(f'   {s}: logged rho into the dwell (product since the previous dwell): cam {rc["rho_cam"]:.3e}, '
                      f'L {rc["rho_L"]:.3e}, chain {rc["rho_chain"]:.3e}; channels in the dwell: '
                      f'{"cam " if rc["cam"] else ""}{"L" if rc["L"] else ""}')
        fc = flat_check(ups, *dwells[0], prm)
        print(f'   {"dof":4s} {"s_A1mid":>8s} {"s_A1end":>8s} {"flat":>12s} | {"floor PC":>8s} {"floor S":>8s} | '
              f'{"PC lo..hi":>17s} {"S lo..hi":>17s} | {"s_B":>7s} {"B/A1":>6s} {"PC r":>13s} {"S r":>13s}')
        for k, dof in enumerate(DOFS):
            sA = logged['A1'][k]
            flat = fc[k] if fc else None
            line = f'   {dof:4s} {flat[0] if flat else math.nan:8.4f} {sA:8.4f} '
            line += (f'{flat[2]:+.3f}<={flat[3]:.3f} ' + ('ok' if flat[4] else 'NO')) if flat else '     no data '
            fl = {h: Mdl.floor(k, prm, h, nch['A1']) for h in Mdl.HYPS}
            line += f'| {fl["PC"]:8.4f} {fl["S"]:8.4f} |'
            for h in Mdl.HYPS:
                lo, hi = pred['A1']['lo'][h][k], pred['A1']['hi'][h][k]
                line += f' {lo:7.4f}..{hi:7.4f}'
            res = {h: [] for h in Mdl.HYPS}
            for h in Mdl.HYPS:
                if flat: res[h].append(flat[4])
                else: res[h].append(False)
                if prm['delta'][k] > 0: res[h].append(sA >= fl[h] * (1 - Mdl.TOL_SINGLE))
                lo, hi = pred['A1']['lo'][h][k], pred['A1']['hi'][h][k]
                res[h].append(lo * (1 - Mdl.TOL_SINGLE) <= sA <= hi * (1 + Mdl.TOL_SINGLE))
            if 'B' in logged:
                sB = logged['B'][k]; ratio = sB / sA if sA > 0 else math.nan
                line += f' | {sB:7.4f} {ratio:6.3f}'
                for h in Mdl.HYPS:
                    rl = pred['B']['lo'][h][k] / pred['A1']['lo'][h][k]
                    rh = pred['B']['hi'][h][k] / pred['A1']['hi'][h][k]
                    a, b = min(rl, rh) * (1 - Mdl.TOL_RATIO), max(rl, rh) * (1 + Mdl.TOL_RATIO)
                    line += f' {a:5.3f}..{b:5.3f}{"+" if a <= ratio <= b else "-"}'
                    res[h].append(a <= ratio <= b)
                    lo, hi = pred['B']['lo'][h][k], pred['B']['hi'][h][k]
                    res[h].append(lo * (1 - Mdl.TOL_SINGLE) <= sB <= hi * (1 + Mdl.TOL_SINGLE))
            print(line + '  ' + ' '.join(f'{h}:{sum(res[h])}/{len(res[h])}' for h in Mdl.HYPS))
            for h in Mdl.HYPS: score[h][0] += sum(res[h]); score[h][1] += len(res[h])
        # ground truth at the end of the last dwell, in the published room frame of this visit
        last = order[-1]; t_end = S[last]['t_dwell1_ms']
        d_end = R.last_before(ups, t_end)
        fr = m['frames']   # the same visit fit the match used
        px, py, pyaw = fr.world_to_pub(T['cx'], T['cy'], T['yaw_theta_world'], t_end)
        truth = [px, py, T['H'], T['w'], T['h'], pyaw]
        print(f'   ground truth (published frame, fit n={fr.n} rms {fr.rms:.3f} m): ' +
              ' '.join(f'{DOFS[k]} {truth[k] if k != 4 else T["h"][0]:.3f}' for k in range(6)) +
              (f' (h {T["h"][0]:.3f}..{T["h"][1]:.3f})' if T['h'][0] != T['h'][1] else ''))
        errs = []
        for k in range(6):
            x = d_end['est'][k]
            if k == 4: e = 0.0 if T['h'][0] <= x <= T['h'][1] else min(x - T['h'][0], x - T['h'][1], key=abs)
            elif k == 5: e = wrap(x - truth[5])
            else: e = x - truth[k]
            errs.append(e)
        print('   error at the end of ' + last + ': ' + ' '.join(f'{DOFS[k]} {errs[k]:+.4f} ({errs[k] / d_end["sig"][k]:+.2f} sigma)'
                                                         for k in range(6)))
        for k in range(6):
            if prm['delta'][k] == 0: continue
            sp = d_end['sig'][k]
            ss = sp * pred[last]['hi']['S'][k] / pred[last]['hi']['PC'][k]
            llr += (-math.log(ss) - 0.5 * (errs[k] / ss) ** 2) - (-math.log(sp) - 0.5 * (errs[k] / sp) ** 2); llr_n += 1
    print('== summary (registered checks; a DOF with Delta = 0 has no floor check)')
    for h in Mdl.HYPS:
        p, n = score[h]
        print(f'   {h}: {p}/{n} checks pass -> {verdict(p == n and n > 0)}')
    fav = [h for h in Mdl.HYPS if score[h][0] == score[h][1] and score[h][1] > 0]
    print(f'   sigma trace favours: {fav[0] if len(fav) == 1 else ("both" if fav else "neither")} '
          f'(the engine implements PC, so the sigma trace tests the implementation against its own model)')
    if llr_n:
        print(f'   ground-truth error: ln L(S) - ln L(PC) = {llr:+.2f} nats over {llr_n} (object, DOF) pairs with Delta > 0 -> '
              f'favours {"S" if llr > 0 else "PC"} (weak: one error per DOF per object)')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2: print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1]))
