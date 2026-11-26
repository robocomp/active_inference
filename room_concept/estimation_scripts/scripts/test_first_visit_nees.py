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
    # 3. rotated frame: (cx, cy) block and P rotated together leave the NEES unchanged; P alone rotated does not
    a = 0.7; Rm = np.eye(6); Rm[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    Pa = P.copy(); Pa[:2, :2] = np.diag([1e-4, 4e-2]) * 1.0; Pa[:2, 2:] = 0; Pa[2:, :2] = 0
    ea = np.array([0.01, 0.15, 0.01, 0.0, 0.0, 0.0])
    q_a = F.nees_full(ea, Pa)[0]
    q_r = F.nees_full(Rm @ ea, Rm @ Pa @ Rm.T)[0]
    q_wrong = F.nees_full(Rm @ ea, Pa)[0]
    check('rotated frame: e and P rotated together', abs(q_a - q_r) < 1e-12 * max(1, q_a), f'{q_a:.6f} {q_r:.6f}')
    check('rotated frame: P left behind is a different (wrong) number', abs(q_a - q_wrong) > 1e-3 * q_a, f'{q_a:.4f} vs {q_wrong:.4f}')
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
    print('\n' + ('ALL PASS' if not FAILS else f'FAILED: {FAILS}'))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
