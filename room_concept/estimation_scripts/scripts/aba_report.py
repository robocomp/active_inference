#!/usr/bin/env python3
"""The scripted A/B/A visit (AuRo test b, amended; shared-engine plan rev 2, Task P1): promise vs delivery.

    aba_report.py <run>          <run> = a run folder (parked_test.sh), or its name under ../crossing/

Per object of aba_segments.tsv and per segment (A1, B, A2):
  promise   = the engine's offer gain from <class>_gain_query.csv (EpistemicPlanner::gain_at, the very computation of the
              agent's offers: at the planner's own standpoint it equals promised_nats bit for bit, gain_query_test), for the matched
              node, with the update serial it was evaluated on. Two per segment: the ARRIVAL promise (arr_query_id: asked on
              arrival, before the dwell, for the pose reached — the TESTED one) and the CLAIM promise (query_id: asked before
              the robot moves, for the scripted standpoint — context);
  offer     = context only: the planner's own last offer at the claim (aba_promises.tsv / offers.csv) and its distance from the
              scripted standpoint;
  net       = the visit's NET entropy change from the logged sigmas: 1/2 sum_k ln(sigma_k^2 at the claim / sigma_k^2 at the end
              of the dwell) — the quantity the registered band predicts (a sum of MARGINAL drops);
  delivered = the sum of the node's delivered_nats over the same window (delivered.csv: one row per accepted update, the joint
              theta entropy drop from the Sigma that update STARTED from). Printed beside the net change, not tested: Sigma also
              grows between updates (predict / ageing steps outside update()), which the sum never subtracts, so on a visit
              with many updates it exceeds the net change (on objects_replay1's bed, 20 s windows sum to ~10 nats).
Registered band (docs/superpowers/results/2026-11-25-p1-predictions.md, model aba_model.py): the predicted delivery of a
segment is 1/2 sum_k ln(sigma_k^2 before / sigma_k^2 after) with the two view sets ("lo": every update a view, "hi": the dwells
only) and the LOGGED rho; band = [min, max] -+ 6 ln(1.1) (each DOF's sigma ratio within the stop rule's 10 %). The prediction
sums MARGINAL drops; delivered.csv is the JOINT (log det) drop — they differ by the change in the components' mutual
information, which the band does not cover (stated, not absorbed).
Checks (addenda of 2026-11-25 to the predictions doc; the second supersedes the first's promise checks):
  at A1, B and A2: |arrival promise - dwell net| <= 6 ln(1.1) (the stop-rule tolerance). The arrival promise is the gain query
                   asked on arrival, before the dwell, from the arrival Sigma, for the pose reached; the dwell net is
                   1/2 sum_k ln(sigma_k^2(t_dwell0) / sigma_k^2(t_dwell1)). Parked at rho = 1 the dwell is ONE look, which is
                   what one hypothetical update predicts. The promise is a JOINT theta entropy drop, the net a MARGINAL sum:
                   that difference is not covered (stated);
  at B and A2:     the visit net (claim -> dwell end) in the registered band;
  the claim-time promise is printed as context only;
  net_A2 <= net_B (for an AR(1) shared error a third view adds no more than the second: (1 - rho)^2 >= 0).
A visit that did not happen as scripted (aba_runlog.visit_validity: a missing segment, a goto / face / dwell not completed, a
gain query not answered, an arrival farther than the controller's goal tolerance) gets NO VERDICT and FAILS.
"""
import math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # run with python3 -I: the siblings below
import aba_model as Mdl
import aba_runlog as R


def read_g5(path, value):
    import csv
    out = {}
    if not path.exists(): return out
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try: out.setdefault(r['node'], []).append((int(r['t_ms']), float(r[value]), r))
            except (KeyError, ValueError): continue
    for v in out.values(): v.sort(key=lambda q: q[0])
    return out


def main(arg):
    run = R.run_path(arg)
    segs = R.segments(run)
    if not segs and (run / 'aba_segments.tsv').exists():
        print(f'{run.name}: aba_segments.tsv has no well-formed row: NO VERDICT -> FAIL'); return 1
    if not segs:
        print(f'{run.name}: no aba_segments.tsv — no scripted A/B/A in this run, so no promise/delivery check can be made.')
        return 0
    root = R.checkout_root(run)
    prom_live = R.read_tsv(run / 'aba_promises.tsv')
    tol_nats = 6 * math.log(1 + Mdl.TOL_RATIO)
    score = {h: [0, 0] for h in Mdl.HYPS}
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
        cls = plan['truth']['cls']; prm = Mdl.class_params(cls, root)
        rows = R.ai2_rows(run, cls)
        node, n = R.match_node(rows, S['A1']['t_dwell0_ms'], S['A1']['t_dwell1_ms'])
        print(f'== {defn} ({cls}): node {node}')
        if node is None:
            print('   no update in the A1 dwell: NO VERDICT -> FAIL')
            for h in Mdl.HYPS: score[h][1] += 1
            continue
        ups = R.updates(rows, node)
        deliv = read_g5(run / f'{cls}_delivered.csv', 'delivered_nats').get(node, [])
        offers = read_g5(run / f'{cls}_offers.csv', 'promised_nats').get(node, [])
        order = list(R.SEGS)   # all three present (visit_validity)
        dwells = [(S[s]['t_dwell0_ms'], S[s]['t_dwell1_ms']) for s in order]
        sig = {}
        for i, s in enumerate(order):
            sig[s] = dict(hi=R.predicted_sigmas(R.views_hi(ups, dwells[:i + 1]), prm),
                          lo=R.predicted_sigmas(R.views_lo(ups, dwells[i][1]), prm))
        res = {h: [] for h in Mdl.HYPS}
        dl = {}
        gq = [r for r in R.read_csv(run / f'{cls}_gain_query.csv') if r.get('node') == node]
        for i, s in enumerate(order):
            seg = S[s]
            t0, t1 = seg['t_claim_ms'], seg['t_dwell1_ms']
            dsum = sum(v for t, v, _ in deliv if t0 < t <= t1 and math.isfinite(v))
            u0, u1 = R.last_before(ups, t0), R.last_before(ups, t1)
            dl[s] = (sum(math.log(u0['sig'][k] / u1['sig'][k]) for k in range(6)) if u0 and u1 else math.nan)
            # the TESTED promise: asked on arrival, before the dwell, from the arrival Sigma; tested against the DWELL's net change
            qa = [r for r in gq if r['request_id'] == seg.get('arr_query_id', '')]
            pa = float(qa[-1]['gain_nats']) if qa else None
            w0, w1 = R.last_before(ups, seg['t_dwell0_ms']), R.last_before(ups, t1)
            dwell_net = sum(math.log(w0['sig'][k] / w1['sig'][k]) for k in range(6)) if w0 and w1 else math.nan
            line = f'   {s:2s}: dwell net {dwell_net:7.3f} nats'
            if pa is not None:
                ok = abs(pa - dwell_net) <= tol_nats
                line += f' | arrival promise {pa:7.3f} nats (serial {qa[-1]["update_serial"]}) -> |promise - dwell net| ' \
                        f'{abs(pa - dwell_net):.3f} {"<=" if ok else ">"} {tol_nats:.3f}'
            else:
                ok = False
                line += ' | NO answered arrival gain query'
            for h in Mdl.HYPS: res[h].append(ok)
            # context only: the claim-time promise of the scripted standpoint, the visit's net change, the sum of delivered.csv
            qc = [r for r in gq if r['request_id'] == seg.get('query_id', '')]
            line += f'\n        (context: claim promise {float(qc[-1]["gain_nats"]):.3f} nats; ' if qc else '\n        (context: no claim promise; '
            line += f'visit net {dl[s]:.3f} nats; sum of delivered.csv {dsum:.3f})'
            # context only: the planner's own offer
            live = [r for r in prom_live if r['def'] == defn and r['seg'] == s and r['node'] == node]
            if live: p = (float(live[-1]['promised_nats']), float(live[-1]['standpoint_x']), float(live[-1]['standpoint_y']))
            else:
                before = [o for o in offers if o[0] <= t0]
                p = (before[-1][1], float(before[-1][2]['standpoint_x']), float(before[-1][2]['standpoint_y'])) if before else None
            if p:
                sx, sy = float(seg.get('pub_x', 'nan')), float(seg.get('pub_y', 'nan'))
                line += f'\n        (context: planner offer {p[0]:.3f} nats at {math.hypot(p[1] - sx, p[2] - sy):.2f} m from the scripted standpoint)'
            if i > 0:
                prev = order[i - 1]
                for h in Mdl.HYPS:
                    e = [0.5 * sum(2 * math.log(sig[prev][v][h][k] / sig[s][v][h][k]) for k in range(6)) for v in ('lo', 'hi')]
                    a, b = min(e) - tol_nats, max(e) + tol_nats
                    ok_d = a <= dl[s] <= b
                    res[h].append(ok_d)
                    line += f'\n        {h}: band {a:6.3f}..{b:6.3f} -> visit net {"in" if ok_d else "OUT"}'
            print(line)
        if 'B' in dl and 'A2' in dl:
            ok = dl['A2'] <= dl['B']
            print(f'   net A2 <= net B: {dl["A2"]:.3f} <= {dl["B"]:.3f} -> {"PASS" if ok else "FAIL"}')
            for h in Mdl.HYPS: res[h].append(ok)
        for h in Mdl.HYPS: score[h][0] += sum(res[h]); score[h][1] += len(res[h])
    print('== summary')
    for h in Mdl.HYPS:
        p, n = score[h]
        print(f'   {h}: {p}/{n} checks pass -> {"PASS" if p == n and n > 0 else "FAIL"}')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2: print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1]))
