#!/usr/bin/env python3
"""The scalar model behind the parked test's registered predictions (aba_model.py), against closed forms, and the run-log view
building (aba_runlog.py).   python3 -I test_aba_model.py"""
import math, os, sys

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
    # 5. the run-log views: updates with every rho exactly 1 are one view; a rho below 1 starts a new one
    ups = [dict(t=t, cam=True, L=True, rho_cam=r, rho_L=r, rho_chain=1.0, chain=[0.0] * 6) for t, r in
           ((0, 0.5), (1, 1.0), (2, 1.0), (3, 0.2), (4, 1.0))]
    v = R.views_lo(ups, 10)
    assert len(v) == 2 and close(v[0]['rho_cam'], 0.5) and close(v[1]['rho_cam'], 0.2)
    vh = R.views_hi(ups, [(0, 1), (3, 4)])
    assert len(vh) == 2 and close(vh[1]['rho_cam'], 0.2)   # the product since the first dwell ended (t = 1): 1 * 0.2 * 1
    assert [len(r) for r in R.parked_runs(ups)] == [3, 2]
    print('test_aba_model: OK')


if __name__ == '__main__': main()
