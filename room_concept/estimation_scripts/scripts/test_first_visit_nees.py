#!/usr/bin/env python3
"""first_visit_nees.py on synthetic cases whose answer is known.   python3 -I test_first_visit_nees.py
No check uses a bound fitted to a result: the statistical ones compare the sample mean with 1 in units of the sample's own
standard error (4 of them), and the algebraic ones are exact to rounding."""
import math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import first_visit_nees as F
import aba_runlog as R

FAILS = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  {detail}' if detail else ''))
    if not ok: FAILS.append(name)


def rand_cov(rng, n=6):
    A = rng.normal(size=(n, n)); S = np.diag(10.0 ** rng.uniform(-3, -1, n))
    return S @ (A @ A.T + n * np.eye(n)) @ S


def integration():
    import csv, tempfile
    from pathlib import Path
    import aba_geometry as G
    wbt = F.DEFAULT_WBT
    polys = F.polygons(wbt)
    t0 = 1_000_000

    def write(path, header, rows):
        with open(path, 'w', newline='') as fh:
            w = csv.writer(fh, delimiter='\t' if str(path).endswith('.tsv') else ','); w.writerow(header); w.writerows(rows)

    def make_run(tmp, cls, nodes, legacy=False, dumps=()):
        """A run folder: the robot sits at each of `nodes`' truth for 10 s; nodes = [(name, est, sig, ups)], ups = update stamps."""
        run = Path(tmp); (run / 'replay').mkdir(parents=True, exist_ok=True)
        pos = nodes[0][4]
        write(run / 'gt_pose.tsv', ['t_ms', 'x', 'y', 'yaw'], [[t0 + 1000 * i, pos[0], pos[1], 0.0] for i in range(11)])
        write(run / 'gt_error.csv', ['ts_ms', 'est_x', 'est_y', 'est_theta', 'gt_x_world', 'gt_y_world', 'gt_theta'],
              [[t0 + 1000 * i, pos[0], pos[1], 0.3, pos[0], pos[1], 0.3 + math.pi / 2] for i in range(11)])
        sig_k, est_k = R.SIG, R.EST
        if legacy:
            hdr = ['cycle', 'node', 'pkt_ts', 'npts', 'gated', *est_k, *sig_k]
            rows = [[c, n, t, npts, g, *e, *sg] for n, e, sg, ups, _ in nodes for c, (t, npts, g) in enumerate(ups, 1)]
        else:
            hdr = ['cycle', 'node', 'pkt_ts', 'npts', 'gated', 'lidar_only', 'beam_qsum', 'rho_cam', 'rho_L', 'rho_chain', *est_k, *sig_k,
                   'spose_x', 'spose_y', 'spose_psi', 'lever_x', 'lever_y']
            rows = [[c, n, t, npts, g, 0, 0.0, 1, 1, 1, *e, *sg, 0, 0, 0, 0, 0] for n, e, sg, ups, _ in nodes for c, (t, npts, g) in enumerate(ups, 1)]
        write(run / f'{cls}_ai2_log.csv', hdr, rows)
        for d in dumps: (run / 'replay' / d).write_bytes(b'')
        return run

    def stub(rows_for):
        def runner(dump, binp, cache): return rows_for(Path(dump).name)
        return runner

    def replay_rows(n, theta, sd, cyc0=1, dth=0.0, dP=0.0):
        P = np.diag([s_ ** 2 for s_ in sd]); out = []
        for i in range(n):
            r = dict(upd=i, cycle=cyc0 + i, gated=0, lidar_only=0, dth_live=dth, dP_live=dP)
            r.update({c: v for c, v in zip(F.DOFS, theta)})
            r.update({f'S_{a}_{b}': P[a, b] for a in range(6) for b in range(a, 6)})
            out.append(r)
        return out, 0, ''

    # ---- a nightstand: node on NIGHTSTAND_R, replay theta 2 cm off in cx with sigma 2 cm everywhere -> e^T P^-1 e / 6 = 1/6
    T = G.object_truth(wbt, 'NIGHTSTAND_R'); sd = [0.02] * 6
    est = [T['cx'], T['cy'], T['H'], T['w'], T['h'][0], T['yaw_theta_world']]
    ups = [(t0 + 2000 + 1000 * i, 100, 0) for i in range(3)]
    node = ('nightstand_1', est, sd, ups, (T['cx'], T['cy']))
    theta = [est[0] + 0.02] + est[1:]
    with tempfile.TemporaryDirectory() as tmp:
        run = make_run(tmp, 'nightstand', [node], dumps=['nightstand_1_1_1.bin'])
        def go(rows_for, smoke=False):
            return {r['key']: r for r in F.score_tour(run, wbt, 'replay', tmp + '/c', smoke, {'nightstand'}, runner=stub(rows_for))}
        r = go(lambda n: replay_rows(3, theta, sd, 1))['NIGHTSTAND_R']
        check('integration: matched node, full P, NEES/DOF = 1/6', r['status'] == 'ok' and abs(r['nees'] - 1 / 6) < 1e-6 and r['label'] == '', f"{r['status']} {r.get('nees')} {r['why']}")
        r = go(lambda n: replay_rows(3, theta, sd, 1, dP=1e-3))['NIGHTSTAND_R']
        check('integration: replay not bit-identical -> no full P', r['status'] == 'no full P' and 'bit-identical' in r['why'], r['why'])
        r = go(lambda n: replay_rows(3, theta, sd, 1, dP=1e-3), smoke=True)['NIGHTSTAND_R']
        check('integration: --smoke waives the identity check', r['status'] == 'ok')
        rc_ = {x['key']: x for x in F.score_tour(run, wbt, 'replay', tmp + '/c', False, {'nightstand'}, runner=stub(lambda n: replay_rows(3, theta, sd, 1, dP=1e-3)), counterfactual=True)}['NIGHTSTAND_R']
        check('integration: --counterfactual skips only the identity gate', rc_['status'] == 'ok' and abs(rc_['nees'] - 1 / 6) < 1e-6, f"{rc_['status']} {rc_['why']}")
        rc_ = {x['key']: x for x in F.score_tour(run, wbt, 'replay', tmp + '/c', False, {'nightstand'}, runner=stub(lambda n: (replay_rows(3, theta, sd, 1, dP=1e-3)[0], 1, 'boom')), counterfactual=True)}['NIGHTSTAND_R']
        check('integration: --counterfactual keeps the exit-code gate', rc_['status'] == 'no full P' and 'exit 1' in rc_['why'], rc_['why'])
        rc_ = {x['key']: x for x in F.score_tour(run, wbt, 'replay', tmp + '/c', False, {'nightstand'}, runner=stub(lambda n: replay_rows(2, theta, sd, 1, dP=1e-3)), counterfactual=True)}['NIGHTSTAND_R']
        check('integration: --counterfactual keeps the alignment gate', rc_['status'] == 'no full P' and 'updates' in rc_['why'], rc_['why'])
        # --yaw-var=reported: the yaw diagonal of P becomes std_yaw^2, the rest of P unchanged; yaw 2 cm-equivalent error, std_yaw 4x
        def yaw_rows(n):
            rows_, rc0, err0 = replay_rows(3, [est[0]] + est[1:5] + [est[5] + 0.02], sd, 1)
            for r_ in rows_: r_['std_yaw'] = 0.08
            return rows_, rc0, err0
        r_w = {x['key']: x for x in F.score_tour(run, wbt, 'replay', tmp + '/c', False, {'nightstand'}, runner=stub(yaw_rows))}['NIGHTSTAND_R']
        r_r = {x['key']: x for x in F.score_tour(run, wbt, 'replay', tmp + '/c', False, {'nightstand'}, runner=stub(yaw_rows), yaw_var='reported')}['NIGHTSTAND_R']
        check('integration: default yaw variance = the within-mode S(5,5)', r_w['status'] == 'ok' and abs(r_w['nees'] - 1 / 6) < 1e-6, f"{r_w.get('nees')}")
        check('integration: --yaw-var=reported grades e_yaw against std_yaw^2', r_r['status'] == 'ok' and abs(r_r['nees'] - (0.02 / 0.08) ** 2 / 6) < 1e-6 and abs(r_r['yaw_var_within'] - 0.02 ** 2) < 1e-15, f"{r_r.get('nees')}")
        lines = []
        F.labelled(lines.append, True, 'reported')('x')
        F.labelled(lines.append, False, 'within')('y')
        check('labels: opt-in statistics prefix every line, the default none', lines == ['[' + F.COUNTERFACTUAL_LABEL + '; ' + F.REPORTED_YAW_LABEL + '] x', 'y'], str(lines))
        r = go(lambda n: replay_rows(2, theta, sd, 1))['NIGHTSTAND_R']
        check('integration: alignment gate, row count', r['status'] == 'no full P' and 'updates' in r['why'], r['why'])
        r = go(lambda n: replay_rows(3, theta, sd, 5))['NIGHTSTAND_R']
        check('integration: alignment gate, cycle', r['status'] == 'no full P' and 'differ' in r['why'], r['why'])
        rows = replay_rows(3, theta, sd, 1)[0]; rows[2]['upd'] = 7
        r = go(lambda n: (rows, 0, ''))['NIGHTSTAND_R']
        check('integration: alignment gate, upd index', r['status'] == 'no full P' and 'differ' in r['why'], r['why'])
        r = go(lambda n: (replay_rows(3, theta, sd, 1)[0], 1, 'boom'))['NIGHTSTAND_R']
        check('integration: replay tool exit != 0 -> no full P', r['status'] == 'no full P' and 'exit 1' in r['why'], r['why'])
        recs = F.score_tour(run, wbt, 'replay', tmp + '/c', False, {'nightstand'}, runner=stub(lambda n: replay_rows(3, theta, sd, 1)))
        check('integration: the other nightstand is unmatched (reported)', {x['key']: x['status'] for x in recs}['NIGHTSTAND_L'] == 'unmatched')
        check('integration: unmatched fails the class', not F.summarise(recs, 'nightstand')['passed'])
    # ---- one node matched to both nightstands -> duplicate on both
    TL = G.object_truth(wbt, 'NIGHTSTAND_L')
    mid = [(est[0] + TL['cx']) / 2, (est[1] + TL['cy']) / 2] + est[2:]
    with tempfile.TemporaryDirectory() as tmp:
        run = make_run(tmp, 'nightstand', [('nightstand_1', mid, [3.0] * 6, ups, (T['cx'], T['cy']))], dumps=['nightstand_1_1_1.bin'])
        recs = F.score_tour(run, wbt, 'replay', tmp + '/c', True, {'nightstand'}, runner=stub(lambda n: replay_rows(3, mid, [3.0] * 6, 1)))
        st = {x['key']: x['status'] for x in recs}
        check('integration: one node matched to two instances -> duplicate on both', st == {'NIGHTSTAND_L': 'duplicate', 'NIGHTSTAND_R': 'duplicate'}, str(st))
    # ---- the legacy table: its own log header, read; no dump -> diagonal and labelled; a dump -> full P; a bad dump -> no full P
    TT = G.object_truth(wbt, 'Table1')
    test = [TT['cx'], TT['cy'], TT['H'], TT['w'], TT['h'][0], TT['yaw_theta_world']]
    tnode = ('table_1', test, [0.02] * 6, ups, (TT['cx'], TT['cy']))
    with tempfile.TemporaryDirectory() as tmp:
        run = make_run(tmp, 'table', [tnode], legacy=True)
        rows, legacy = F.class_rows(run, 'table')
        check('legacy table log is read (aba_runlog.ai2_rows reads none of it)', legacy and len(rows) == 3 and R.ai2_rows(run, 'table') == [], f'{len(rows)} rows')
        r = {x['key']: x for x in F.score_tour(run, wbt, 'replay', tmp + '/c', False, {'table'})}['Table1']
        check('legacy table: matched, diagonal, labelled old engine', r['status'] == 'ok' and r['label'] == 'old engine, diagonal' and r['legacy_log'], f"{r['status']} {r['why']}")
        check('legacy table: diagonal NEES of a perfect node is 0', abs(r['nees']) < 1e-9, str(r.get('nees')))
    with tempfile.TemporaryDirectory() as tmp:
        run = make_run(tmp, 'table', [tnode], legacy=True, dumps=['table_1_1_1.bin'])
        tth = [test[0] + 0.02] + test[1:]
        r = {x['key']: x for x in F.score_tour(run, wbt, 'replay', tmp + '/c', False, {'table'}, runner=stub(lambda n: replay_rows(3, tth, [0.02] * 6, 1)))}['Table1']
        check('table with a valid dump: full P, no old-engine label', r['status'] == 'ok' and r['label'] == '' and abs(r['nees'] - 1 / 6) < 1e-6, f"{r['status']} {r.get('label')} {r.get('nees')}")
        r = {x['key']: x for x in F.score_tour(run, wbt, 'replay', tmp + '/c', False, {'table'}, runner=stub(lambda n: replay_rows(3, tth, [0.02] * 6, 1, dP=1.0)))}['Table1']
        check('table whose dump fails validation: no full P, never diagonal', r['status'] == 'no full P' and not r.get('label'), f"{r['status']} {r.get('label')}")
    # ---- c2v: the image is chosen by the FULL-P NEES
    Tt = dict(cls='table', symmetry='c2v', H=0.75, w=1.2, h=(0.8, 0.8))
    est_c = (1.03, 1.97, 0.76, 0.85, 1.22, 0.33 + math.pi / 2)      # the swapped description of a (1.2, 0.8, 0.3) table
    P_c = rand_cov(np.random.default_rng(5))
    e_c, Pc = F.graded_error_full(est_c, [math.sqrt(P_c[i, i]) for i in range(6)], P_c, Tt, (1.0, 2.0, 0.3))
    allq = []
    for e_, s_, perm in R.c2v_images(est_c, [1.0] * 6):
        ee = R.graded_errors(e_, s_, dict(Tt, symmetry='none'), (1.0, 2.0, 0.3))[0]
        allq.append(F.nees_full(ee, P_c[np.ix_(list(perm), list(perm))])[0])
    check('c2v: image with the smallest full-P NEES', abs(F.nees_full(e_c, Pc)[0] - min(allq)) < 1e-12, f'{F.nees_full(e_c, Pc)[0]:.4f} vs {min(allq):.4f}')


def main():
    rng = np.random.default_rng(1)
    # 1. e ~ N(0, P): NEES/DOF has mean 1 (chi2_6 / 6); each DOF's e^2/P_kk has mean 1
    P = rand_cov(rng)
    E = rng.multivariate_normal(np.zeros(6), P, size=4000)
    q = np.array([F.nees_full(e, P)[0] for e in E]); pd = np.mean([F.nees_full(e, P)[1] for e in E], 0)
    se = q.std(ddof=1) / math.sqrt(len(q))
    check('e ~ N(0,P): mean NEES/DOF = 1', abs(q.mean() - 1) < 4 * se, f'mean {q.mean():.4f} se {se:.4f}')
    check('e ~ N(0,P): per-DOF e^2/P_kk = 1', all(abs(pd[k] - 1) < 4 * math.sqrt(2.0 / len(E)) for k in range(6)), f'{np.round(pd, 3)}')
    # 2. a known value: e = (sqrt(2) s) on a correlated pair gives exactly e^T P^-1 e / 6
    P2 = np.eye(6); P2[0, 1] = P2[1, 0] = 0.9; e2 = np.array([1.0, -1.0, 0, 0, 0, 0])
    exact = float(e2 @ np.linalg.inv(P2) @ e2) / 6
    q2, pd2 = F.nees_full(e2, P2)
    check('full P is not diagonal NEES', abs(q2 - exact) < 1e-12 and abs(q2 - F.nees_diag(e2, np.ones(6))[0]) > 0.1, f'{q2:.6f} vs diag {F.nees_diag(e2, np.ones(6))[0]:.6f}')
    # 3. the statistic is invariant to a common rotation of e and P (a property of e^T P^-1 e; the scorer itself never rotates P:
    #    the truth is taken into the published frame, so e and P already share it)
    a = 0.7; Rm = np.eye(6); Rm[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    Pa = P.copy(); Pa[:2, :2] = np.diag([1e-4, 4e-2]); Pa[:2, 2:] = 0; Pa[2:, :2] = 0
    ea = np.array([0.01, 0.15, 0.01, 0.0, 0.0, 0.0])
    q_a = F.nees_full(ea, Pa)[0]; q_r = F.nees_full(Rm @ ea, Rm @ Pa @ Rm.T)[0]
    check('NEES invariant to a common rotation of e and P', abs(q_a - q_r) < 1e-12 * max(1, q_a), f'{q_a:.6f} {q_r:.6f}')
    # and the frames themselves: a truth taken into a rotated published frame lands where Frames.world_to_pub says
    import aba_geometry as G
    ph, ts = 0.4, (0.3, -0.2)
    gt = [(1000 + i, ts[0] + math.cos(ph) * x - math.sin(ph) * y, ts[1] + math.sin(ph) * x + math.cos(ph) * y, ph, x, y, math.pi / 2)
          for i, (x, y) in enumerate([(0, 0), (1, 0), (0, 1), (2, 1), (1, 2), (-1, 1)])]
    fr = G.Frames(gt, [(0, 0.5, 0.0, 0.3)], 1000, 1010)
    px, py, pyaw = fr.world_to_pub(1.0, 2.0, 0.0, 1005)
    ix, iy = ts[0] + math.cos(ph) * 1 - math.sin(ph) * 2, ts[1] + math.sin(ph) * 1 + math.cos(ph) * 2
    ex, ey = 0.5 + math.cos(0.3) * ix - math.sin(0.3) * iy, 0.0 + math.sin(0.3) * ix + math.cos(0.3) * iy
    check('truth into the published frame (fit + gauge)', math.hypot(px - ex, py - ey) < 1e-9, f'({px:.5f}, {py:.5f}) vs ({ex:.5f}, {ey:.5f})')
    # 4. c2v (the table): the same table described by its swapped image has the same e, and P permuted with the image gives the same NEES
    T = dict(cls='table', symmetry='c2v', H=0.75, w=1.2, h=(0.8, 0.8), cx=0, cy=0, yaw_theta_world=0.3)
    truth_pub = (1.0, 2.0, 0.3)
    est = (1.01, 1.98, 0.76, 1.19, 0.81, 0.31); P6 = rand_cov(rng)
    sig = tuple(math.sqrt(P6[i, i]) for i in range(6))
    e1, Pg1 = F.graded_error_full(est, sig, P6, T, truth_pub)
    sw = [0, 1, 2, 4, 3, 5]
    est_sw = (est[0], est[1], est[2], est[4], est[3], F.R._wrap(est[5] + math.pi / 2))      # the same table, w and h swapped, yaw + pi/2
    P_sw = P6[np.ix_(sw, sw)]; sig_sw = tuple(math.sqrt(P_sw[i, i]) for i in range(6))
    e2_, Pg2 = F.graded_error_full(est_sw, sig_sw, P_sw, T, truth_pub)
    q_1, q_2 = F.nees_full(e1, Pg1)[0], F.nees_full(e2_, Pg2)[0]
    check('c2v: the swapped description scores the same', abs(q_1 - q_2) < 1e-9 * max(1, q_1) and np.allclose(e1, e2_), f'{q_1:.6f} {q_2:.6f}')
    q_np = F.nees_full(e2_, P_sw)[0]
    check('c2v: P left unpermuted would be wrong', abs(q_np - q_1) > 1e-6 * q_1, f'{q_np:.6f} vs {q_1:.6f}')
    # 5. the bed's h is an interval: no error inside, the distance to the nearest end outside
    Tb = dict(cls='bed', symmetry='none', H=0.72, w=1.766, h=(2.09, 2.12))
    ein = F.graded_error_full((0, 0, 0.72, 1.766, 2.10, 0), (1,) * 6, np.eye(6), Tb, (0, 0, 0))[0]
    eout = F.graded_error_full((0, 0, 0.72, 1.766, 2.20, 0), (1,) * 6, np.eye(6), Tb, (0, 0, 0))[0]
    check('bed h interval', ein[4] == 0.0 and abs(eout[4] - 0.08) < 1e-12, f'{ein[4]} {eout[4]:.4f}')
    # 6. first stay: from the first sample inside to the last inside before the first sample in another room; doorway samples neutral
    ser = [(0, 'a'), (1, 'a'), (2, None), (3, 'a'), (4, None), (5, 'b'), (6, 'a')]
    check('first stay (doorway samples neutral)', F.first_stay(ser, 'a') == (0, 3, False), str(F.first_stay(ser, 'a')))
    check('first stay of the other room', F.first_stay(ser, 'b') == (5, 5, False), str(F.first_stay(ser, 'b')))
    check('first stay open at the end of the log', F.first_stay([(0, 'b'), (1, 'a'), (2, 'a')], 'a') == (1, 2, True))
    check('never inside', F.first_stay([(0, 'b'), (1, None)], 'a') is None)
    # 7. the bootstrap CI of a constant is that constant; of two points it lies between them
    check('bootstrap CI', F.bootstrap_ci([2.0] * 5) == (2.0, 2.0) and 1.0 <= F.bootstrap_ci([1.0, 3.0])[0] <= F.bootstrap_ci([1.0, 3.0])[1] <= 3.0)
    # 8. the verdict: honest draws pass at n = 10; one unmatched visit fails the class whatever the mean
    recs = [dict(run='r', key=f'k{i}', name='k', cls='bed', room='room_1', status='ok', nees=1.0, perdof=[1.0] * 6, why='') for i in range(10)]
    check('PASS: 10 draws at 1', F.summarise(recs, 'bed')['passed'])
    check('FAIL: 9 draws', not F.summarise(recs[:9], 'bed')['passed'])
    check('FAIL: one unmatched visit', not F.summarise(recs + [dict(run='r', key='u', name='u', cls='bed', room='room_1', status='unmatched', why='x')], 'bed')['passed'])
    check('FAIL: one visit with no full P', not F.summarise(recs + [dict(run='r', key='u', name='u', cls='bed', room='room_1', status='no full P', why='x')], 'bed')['passed'])
    check('FAIL: mean 2.5', not F.summarise([dict(r, nees=2.5) for r in recs], 'bed')['passed'])
    integration()
    print('\n' + ('ALL PASS' if not FAILS else f'FAILED: {FAILS}'))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
