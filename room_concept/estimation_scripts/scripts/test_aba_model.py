#!/usr/bin/env python3
"""The scalar model behind the parked test's registered predictions (aba_model.py), against closed forms, and the run-log view
building (aba_runlog.py).   python3 -I test_aba_model.py"""
import math, os, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aba_model as M
import aba_runlog as R

REL = 1e-9


def close(a, b, rel=REL): return abs(a - b) <= rel * max(abs(a), abs(b))


def prm(v_cam, v_L, delta, P0):
    return dict(delta=[delta] * 6, d=[delta * delta / 12] * 6, P0=[P0] * 6, v={'cam': [v_cam] * 6, 'L': [v_L] * 6},
                q0=[0.0] * 6, predict_dt_s=0.1)


def view(cam=True, L=False, rho=1.0, rho_chain=1.0, chain=0.0):
    return dict(cam=cam, L=L, rho_cam=rho, rho_L=rho, rho_chain=rho_chain, chain=[chain] * 6)


def main():
    # 1. parked_synthetic_test's closed forms (one camera channel, no shape latent, no chain):
    #    one view   sigma^2 = (1/P0 + 1/v)^-1;   two views at rho   sigma^2 = (1/P0 + 2/(v (1 + rho)))^-1
    p = prm(0.03 ** 2, 0.0, 0.0, 0.5 ** 2)
    for h in M.HYPS:
        assert close(M.theta_var([view()], 0, p, h), 1 / (1 / 0.25 + 1 / 0.0009))
        for rho in (0.0, 0.3, 0.9):
            assert close(M.theta_var([view(), view(rho=rho)], 0, p, h), 1 / (1 / 0.25 + 2 / (0.0009 * (1 + rho))))
    # 2. the shape latent cannot be averaged away: many independent views of both channels approach the floor from above,
    #    Delta^2/24 for per-channel latents, Delta^2/12 for one shared latent (no prior)
    p = prm(0.02 ** 2, 0.02 ** 2, 0.2, math.inf)
    vs = [view(L=True, rho=0.0 if j else 1.0) for j in range(400)]
    for h, n in (('PC', 2), ('S', 1)):
        s = math.sqrt(M.theta_var(vs, 3, p, h)); f = M.floor(3, p, h, 2)
        assert close(f, 0.2 / math.sqrt(12 * n)), (h, f)
        assert f <= s <= f * 1.01, (h, s, f)
    # 3. a chain error shared by both channels of a view adds to both hypotheses alike; parked (rho_chain = 1) it is a floor too
    p = prm(0.02 ** 2, 0.02 ** 2, 0.0, math.inf)
    s1 = M.theta_var([view(L=True, chain=0.05 ** 2)] + [view(L=True, rho=0.0, chain=0.05 ** 2) for _ in range(300)], 0, p, 'PC')
    assert s1 >= 0.05 ** 2, s1
    # 4. rho as viewpoint.h: an arc at constant range has dphi = the angle and dz -> 0 (the path is summed in 200 chords, as
    #    accumulate_path sums odometry steps: each chord's radial part is O(z dtheta^2), 200 x 2 x 2 (pi/800)^2 ~ 0.006 m)
    A, B, c = (0.0, -2.0), (math.sqrt(2.0), -math.sqrt(2.0)), (0.0, 0.0)
    pr = M.planned_rho(A, B, c, 1.6, 0.0, (0.0, 1.0))
    q = pr['arc']
    assert q['dz'] < 0.01 and close(q['dphi'], math.pi / 4, 1e-3), q
    assert close(q['rho_cam'], math.exp(-q['dphi'] / math.atan(q['w_perp'] / 4.0) - q['dz'] / 2.0), 1e-9)
    assert pr['chord']['dz'] > 0 and pr['chord']['rho_cam'] < pr['arc']['rho_cam']
    # 4b. the Kalman form (theta_var) and the GLS form (theta_var_gls) of the same model agree, on mixed channels, rho, chain
    import random
    rnd = random.Random(7)
    for trial in range(20):
        p = prm(rnd.uniform(1e-4, 1e-3), rnd.uniform(1e-4, 1e-3), rnd.uniform(0.0, 0.2), rnd.choice([math.inf, 0.01]))
        vs = [dict(cam=rnd.random() < 0.7, L=rnd.random() < 0.8, rho_cam=rnd.random(), rho_L=rnd.random(),
                   rho_chain=rnd.random(), chain=[rnd.uniform(0.0, 0.003)] * 6) for _ in range(rnd.randint(1, 12))]
        vs[0]['cam'] = True
        for h in M.HYPS:
            a, b = M.theta_var(vs, 0, p, h), M.theta_var_gls(vs, 0, p, h)
            assert abs(a - b) <= 1e-6 * b, (trial, h, a, b)
    # 5. the run-log views: updates with every rho exactly 1 are one view; a rho below 1 starts a new one
    ups = [dict(t=t, cam=True, L=True, rho_cam=r, rho_L=r, rho_chain=1.0, chain=[0.0] * 6) for t, r in
           ((0, 0.5), (1, 1.0), (2, 1.0), (3, 0.2), (4, 1.0))]
    v = R.views_lo(ups, 10)
    assert len(v) == 2 and close(v[0]['rho_cam'], 0.5) and close(v[1]['rho_cam'], 0.2)
    vh = R.views_hi(ups, [(0, 1), (3, 4)])
    assert len(vh) == 2 and close(vh[1]['rho_cam'], 0.2)   # the product since the first dwell ended (t = 1): 1 * 0.2 * 1
    assert [len(r) for r in R.parked_runs(ups)] == [3, 2]
    visits()
    node_match()
    kitchen_plans()
    chair_plan()
    c2v_errors()
    table_plan()
    print('test_aba_model: OK')


HEADER = ('def seg stand_x stand_y stand_yaw room pub_x pub_y pub_yaw fit t_claim_ms t_arrive_ms t_dwell0_ms t_dwell1_ms goto '
          'turn_deg face gt_x gt_y gt_yaw dwell query_id query_answered arr_query_id arr_query_answered').split()


def seg_row(defn, seg, t, **kw):
    r = dict(zip(HEADER, [defn, seg, '1.0', '2.0', '0', 'room_2', '0', '0', '0', 'fit', str(t), str(t + 10), str(t + 20),
                           str(t + 80), 'completed', '5.0', 'completed', '1.05', '2.0', '0', 'completed', str(t), 'yes',
                           str(t + 15), 'yes']))
    r.update(kw)
    return '\t'.join(r[h] for h in HEADER)


def visits():
    # 6. a visit counts only if it happened as scripted (review I1): every segment, every step completed, both queries
    #    answered, the arrival within the controller's goal tolerance; a malformed row is skipped and leaves its segment missing
    tol, src = R.controller_goal_tolerance()
    assert tol > 0 and 'goal' in src.lower(), (tol, src)
    rows = [seg_row('OK', 'A1', 1000), seg_row('OK', 'B', 2000), seg_row('OK', 'A2', 3000),
            seg_row('NOB', 'A1', 1000), seg_row('NOB', 'A2', 3000),
            seg_row('TIMEOUT', 'A1', 1000), seg_row('TIMEOUT', 'B', 2000, goto='TIMEOUT'), seg_row('TIMEOUT', 'A2', 3000),
            seg_row('NOCLAIM', 'A1', 1000, query_answered='no'), seg_row('NOCLAIM', 'B', 2000), seg_row('NOCLAIM', 'A2', 3000),
            seg_row('RETRY', 'A1', 1000), seg_row('RETRY', 'A2', 3000),
            seg_row('RETRY', 'B', 2000, face='completed[a1:165.5deg:failed_timeout;a2:132.0deg:completed;a3:1.2deg:within]'),
            seg_row('NOFACE', 'A1', 1000), seg_row('NOFACE', 'A2', 3000),
            seg_row('NOFACE', 'B', 2000, face='failed[a1:165.5deg:failed;a2:150deg:failed;a3:140deg:failed;a4:139deg:out_of_attempts]'),
            seg_row('OFF', 'A1', 1000), seg_row('OFF', 'B', 2000, gt_x=str(1.0 + tol + 0.01)), seg_row('OFF', 'A2', 3000),
            seg_row('FRAME', 'A1', 1000), seg_row('FRAME', 'B', 2000, goto='frame_failed', gt_x='nan', gt_y='nan',
                                                  arr_query_answered='no', dwell='none'),
            seg_row('FRAME', 'A2', 3000),
            'BAD\tB\t1.0']                                                     # malformed: too few columns
    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        (run / 'aba_segments.tsv').write_text('\t'.join(HEADER) + '\n' + '\n'.join(rows) + '\n')
        said = []
        S = R.segments(run, report=said.append)
        assert len(said) == 1 and 'malformed' in said[0], said
        assert R.visit_validity(S['OK'], tol) == []
        assert any('B: segment missing' in w for w in R.visit_validity(S['NOB'], tol))
        assert any('B: goto TIMEOUT' in w for w in R.visit_validity(S['TIMEOUT'], tol))
        assert any('B: arrival' in w for w in R.visit_validity(S['OFF'], tol))
        assert R.visit_validity(S['NOCLAIM'], tol) == []                     # unanswered claim query: context only
        assert R.visit_validity(S['RETRY'], tol) == []                       # a face that needed retries but got there
        assert any('B: face failed[' in w for w in R.visit_validity(S['NOFACE'], tol))
        assert any('B: goto frame_failed' in w for w in R.visit_validity(S['FRAME'], tol))
        assert 'BAD' not in S
        # both reports refuse every broken visit (NO VERDICT) and never PASS overall
        for rep in ('parked_report.py', 'aba_report.py'):
            out = subprocess.run([sys.executable, '-I', os.path.join(os.path.dirname(os.path.abspath(__file__)), rep), str(run)],
                                 capture_output=True, text=True).stdout
            for bad in ('NOB', 'TIMEOUT', 'OFF', 'FRAME', 'NOFACE'):
                assert f'== {bad}: NO VERDICT -> FAIL' in out, (rep, bad, out)
            assert 'PASS' not in out.split('== summary')[-1], (rep, out)


AI2_COLS = ('node pkt_ts gated npts lidar_only beam_qsum rho_cam rho_L rho_chain std_cx std_cy std_H std_w std_h std_yaw_within '
            'cx cy H w h yaw spose_x spose_y spose_psi lever_x lever_y').split()


def node_match():
    # 7. the node is matched to the ground truth by its own posterior (live run p1_nightstand2: the most-updated node was the
    #    OTHER nightstand). The decoy has 5x the updates; the right node is chosen; a run where no node is near is NO VERDICT.
    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        # gt_error: internal frame == world (est = gt, est_theta = gt_theta - pi/2), no gauge file: published == world
        with open(run / 'gt_error.csv', 'w') as f:
            f.write('ts_ms,est_x,est_y,est_theta,gt_x_world,gt_y_world,gt_theta\n')
            for t in range(0, 4000, 100): f.write(f'{t},{1.0 + t * 1e-4},2.0,0.0,{1.0 + t * 1e-4},2.0,{math.pi / 2}\n')
        def row(node, t, cx, cy):
            v = dict(node=node, pkt_ts=t, gated=0, npts=100, lidar_only=0, beam_qsum=1, rho_cam=1, rho_L=1, rho_chain=1,
                     std_cx=0.03, std_cy=0.03, std_H=0.02, std_w=0.02, std_h=0.02, std_yaw_within=0.02, cx=cx, cy=cy, H=0.6,
                     w=0.5, h=0.5, yaw=0, spose_x=0.01, spose_y=0.01, spose_psi=0.01, lever_x=0, lever_y=1)
            return ','.join(str(v[c]) for c in AI2_COLS)
        lines = [row('nightstand_6', t, -1.36, 3.88) for t in range(1000, 2000, 10)]          # the decoy: 100 updates
        lines += [row('nightstand_5', t, 0.88, 3.86) for t in range(1000, 2000, 50)]          # the object: 20 updates
        (run / 'nightstand_ai2_log.csv').write_text(','.join(AI2_COLS) + '\n' + '\n'.join(lines) + '\n')
        rows = R.ai2_rows(run, 'nightstand')
        S = {'A1': dict(t_claim_ms=500, t_dwell0_ms=1000, t_dwell1_ms=2000), 'A2': dict(t_dwell1_ms=3900)}
        assert R.match_node(rows, 1000, 2000)[0] == 'nightstand_6'                          # what update counting picked
        m = R.match_node_to_truth(run, rows, S, dict(cx=0.867, cy=3.878))
        assert m['node'] == 'nightstand_5' and m['runner'] == 'nightstand_6', m
        assert m['d2'] <= R.CHI2_2_99 < m['runner_d2'], m
        m = R.match_node_to_truth(run, rows, S, dict(cx=5.0, cy=-3.0))                      # nothing there
        assert m['node'] is None and 'chi2' in m['why'], m


def kitchen_plans():
    # 8. K1 live validation: the fridge and the hood (room_1, the kitchen) planned from the world file. The truth is the mesh layer
    #    (Fridge body; the hood's canopy frustum, which hangs: H = its TOP above the floor, z0 = its underside, as the engine
    #    publishes a hanging class, countable_vertical_span.h); the back is the object's own axis facing the wall; the standpoint
    #    rule is unchanged (the hood's nominal A, 1.5 m in front, stands in Worktop3: the rule moves it in, and says so).
    import aba_geometry as G
    wbt = os.environ.get('WBT_PATH', '/home/pbustos/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt')
    if not os.path.exists(wbt):
        print('test_aba_model: SKIP kitchen plans (no world file)'); return
    f = G.plan(wbt, 'FRIDGE_0'); T = f['truth']
    assert T['cls'] == 'refrigerator' and T['room_poly'] == 'room_1', T
    assert abs(T['back'][0] + 1.0) < 1e-4 and abs(T['back'][1]) < 1e-4, T['back']   # the west wall (the node's yaw is pi - 7e-6)
    assert abs(T['w'] - 0.6792) < 1e-3 and abs(T['h'][0] - 0.5389) < 1e-3 and abs(T['H'] - 1.9888) < 1e-3 and abs(T['z0']) < 1e-3, T
    assert f['feasible']
    c = (T['cx'], T['cy'])
    for p in (f['A'], f['B']): assert abs(math.hypot(p[0] - c[0], p[1] - c[1]) - f['z']) < 1e-6   # same range at A and B
    assert f['A'][0] > T['cx'] and abs(f['A'][1] - T['cy']) < 1e-4                               # A straight in front
    h = G.plan(wbt, 'HOOD_0'); T = h['truth']
    assert T['cls'] == 'hood' and abs(T['H'] - 1.9) < 1e-6 and abs(T['z0'] - 1.6) < 1e-6, T       # hangs: top 1.9, underside 1.6
    assert abs(T['w'] - 1.296) < 1e-6 and abs(T['h'][0] - 0.5) < 1e-6, T
    assert h['feasible'] and h['moved_m'] < 0.0, h                                                # moved IN by the worktop
    assert 'Worktop3' in h['obstacles']


def chair_plan():
    # 9. K2b: the visit chair CHAIR_5 (aba_geometry.choose_chair; the rule is in its docstring). Truth = the Chair.proto's overall
    #    visual box (w 0.50, depth 0.534, top 0.999 above the node's z), the back on the BACKREST side (derived from the vertices
    #    above the seat, not from the walls), the seat layer kept as a second field. CHAIR_1 is sunk 0.3 m below the floor.
    import aba_geometry as G
    wbt = os.environ.get('WBT_PATH', '/home/pbustos/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt')
    if not os.path.exists(wbt):
        print('test_aba_model: SKIP chair plan (no world file)'); return
    box, back_local, seat = G.chair_geometry()
    assert back_local == [-1.0, 0.0], back_local                                  # the backrest is on the proto's local -x
    assert abs(box[0] - 0.534) < 1e-3 and abs(box[1] - 0.50) < 1e-3 and abs(box[3] - 0.999) < 1e-3, box
    p = G.plan(wbt, 'CHAIR_5'); T = p['truth']
    assert T['cls'] == 'chair' and T['room_poly'] == 'room_2', T                 # wbt_rooms' room_2 = the bathroom
    assert abs(T['back'][0]) < 1e-4 and abs(T['back'][1] + 1.0) < 1e-4, T['back']   # backrest to the south (rotation -(-pi/2))
    assert abs(T['w'] - 0.50) < 1e-3 and abs(T['h'][0] - 0.534) < 1e-3 and abs(T['H'] - 1.039) < 1e-3, T
    assert 'seat' in T and abs(T['seat']['top'] - 0.543) < 1e-3, T['seat']
    assert p['feasible'], p
    c = (T['cx'], T['cy'])
    for q in (p['A'], p['B']): assert abs(math.hypot(q[0] - c[0], q[1] - c[1]) - p['z']) < 1e-6
    assert p['A'][1] > T['cy']                                                    # A in FRONT of the seat (north), not behind the backrest
    t1 = G.object_truth(wbt, 'CHAIR_1')
    assert abs(t1['origin'][3] + 0.3) < 1e-9 and abs(t1['H'] - 0.699) < 1e-3, t1  # sunk: the choice rule excludes it


def c2v_errors():
    # 10. K2c: the symmetry-aware ground-truth error. A c2v truth and each of its four group images score ZERO; a non-c2v class
    #     keeps its plain error (an image is NOT equivalent there); a swapped image reports the truth's labels and sigmas.
    T = dict(cls='table', symmetry='c2v', H=0.73, w=1.8, h=[0.8, 0.8])
    truth = (-1.84, 4.43, 0.3)
    sig = (0.02, 0.03, 0.01, 0.05, 0.04, 0.02)
    est = (-1.84, 4.43, 0.73, 1.8, 0.8, 0.3)
    for e_, s_, perm in R.c2v_images(est, sig):
        err, es, pm = R.graded_errors(e_, s_, T, truth)
        assert max(abs(x) for x in err) < 1e-12, (e_, err)
        assert es[3] == 0.05 and es[4] == 0.04, es                  # back in the truth's labels: sigma_w, sigma_h
    err, _, pm = R.graded_errors((-1.84, 4.43, 0.73, 0.8, 1.8, 0.3 + math.pi / 2), sig, T, truth)
    assert max(abs(x) for x in err) < 1e-12 and pm == (0, 1, 2, 4, 3, 5), (err, pm)
    Tn = dict(T, symmetry='none', cls='nightstand')
    err, _, _ = R.graded_errors((-1.84, 4.43, 0.73, 1.8, 0.8, 0.3 + math.pi), sig, Tn, truth)
    assert abs(abs(err[5]) - math.pi) < 1e-9, err                    # a nightstand turned round is wrong by pi


def table_plan():
    # 11. K2c: the visit table TABLE_1 (aba_geometry.choose_table; the rule is in its docstring). Truth = the top slab of
    #     table1.dae (1.8 x 0.8, top 0.730 above the floor), c2v from the manifest; it stands on the floor (z0 = 0).
    import aba_geometry as G
    wbt = os.environ.get('WBT_PATH', '/home/pbustos/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt')
    if not os.path.exists(wbt):
        print('test_aba_model: SKIP table plan (no world file)'); return
    p = G.plan(wbt, 'TABLE_1'); T = p['truth']
    assert T['cls'] == 'table' and T['symmetry'] == 'c2v' and T['room_poly'] == 'room_1', T
    assert {round(T['w'], 3), round(T['h'][0], 3)} == {1.8, 0.8} and abs(T['H'] - 0.7298) < 1e-3 and T['z0'] == 0.0, T
    assert abs(T['top']['top_under'] - 0.824) < 1e-3, T['top']
    assert p['feasible'], p
    c = (T['cx'], T['cy'])
    for q in (p['A'], p['B']): assert abs(math.hypot(q[0] - c[0], q[1] - c[1]) - p['z']) < 1e-6
    names = [n for _, n, _, _, _, _ in G.world_tables(wbt)]
    assert 'RoundTable' in names and 'Desk' in names, names              # every table node is listed by the choice rule


if __name__ == '__main__': main()
