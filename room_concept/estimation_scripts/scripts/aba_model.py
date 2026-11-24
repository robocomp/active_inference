#!/usr/bin/env python3
"""The registered predictions of the parked test and the A/B/A visit (shared-engine plan rev 2, Task P1), as code.

    aba_model.py registered [<worktree>]   print the per-class numbers registered in docs/superpowers/results/2026-11-25-p1-predictions.md

One scalar model per theta component k (cx cy H w h yaw), used by parked_report.py and aba_report.py with the LOGGED inputs:

  every measured "view" j of channel c in {cam, L} sees   y_jc = theta_k + b_jc + e_j + s
    b_jc  the channel's shared error: marginal v_c, AR(1) across views with the LOGGED rho_c (ai2 log rho_cam / rho_L);
          v_cam = kZedYoloBoundaryStdM^2 on cx cy H w h (countable_latent_inputs.h, measured), AI2CommonModeYawStd^2 on yaw;
          v_L = AI2CommonMode{Pos,Pos,Size,Size,Size,Yaw}Std^2 (the class config) — the engine's own sig2_cam / sig2_L;
    e_j   the pose chain, shared by both channels of a view: marginal from the LOGGED spose_* and lever_* (chain_B rows of
          common/ai_belief/viewpoint.h: cx = spose_x^2 + lever_y^2 spose_psi^2, cy = spose_y^2 + lever_x^2 spose_psi^2,
          yaw = spose_psi^2; the x-psi cross-covariance is not logged and is left out), AR(1) with the LOGGED rho_chain;
    s     the shape latent, rho = 1, marginal d = Delta_k^2/12 (Delta = the manifest's [shape] layer_spread_m):
          hypothesis "PC" (per-channel, what the engine implements: kSc, kSL): s_cam and s_L independent;
          hypothesis "S"  (one shared latent): the same s for both channels.
  A parked view pins y exactly (its frames are dense and their fresh share is ~0 parked), so
      sigma_k^2 = 1 / (1/P0_k + 1' C^-1 1),   C = the covariance of the y_jc above,
  P0_k = the class prior (manifest: [prior.height] std on H, [prior.footprint] std on w and h, [prior.alignment] std_deg on
  yaw; none on cx, cy — the birth position covariance is not logged, so the predicted cx, cy sigma is biased UP).
  Not in the model (stated, not absorbed into a tolerance): cross-DOF coupling, the once-entered wall terms (stand-off on the
  wall-normal position and depth, room-axis yaw), the fresh ego-motion share of updates taken while driving.

  Two view sets bracket the truth:
    "hi" = the DWELLS only (A1, B, A2: one view each, rho between them = the product of the logged rho over every update in
           between): drops any view taken while driving, so sigma is an UPPER bound;
    "lo" = every measured update its own view (a new view whenever a logged rho is below 1): counts each driving update as a
           fully informative parked view, so sigma is a LOWER bound.
  Floor (test i): with the shape latent at rho = 1, no number of views takes sigma_k below
      PC: (1/P0 + n_c 12/Delta^2)^-1/2  (n_c channels that measured; Delta/sqrt(24) for two, no prior)
      S:  (1/P0 + 12/Delta^2)^-1/2      (Delta/sqrt(12))
  Tolerances, from the solver's stop rule (common/ai_belief/tests/toy_model.h, gn_tol_sigma = 0.1 in countable_belief.h):
  one reported sigma vs an analytic value 5 %; a ratio of two reported sigmas 10 %; in nats per DOF ln(1.1).
"""
import math, os, re, sys
import numpy as np

try:
    import tomllib
except ImportError:   # python < 3.11
    tomllib = None

DOFS = ('cx', 'cy', 'H', 'w', 'h', 'yaw')
KZED_BOUNDARY_STD_M = 0.0289       # countable_concept/src/countable_latent_inputs.h kZedYoloBoundaryStdM (measured, F1.6)
GN_TOL_SIGMA = 0.1                 # countable_belief.h gn_tol_sigma (the stop rule)
TOL_SINGLE = GN_TOL_SIGMA / 2      # toy_model.h sigma_tol_single
TOL_RATIO = GN_TOL_SIGMA           # toy_model.h sigma_tol_ratio
HELIOS_RING_GAP_RAD = 0.0392       # the agents log it at start: "[lidar-measures] Helios SensorModel: max ring gap 0.0392 rad"
HYPS = ('PC', 'S')


# ------------------------------------------------------------------------------------- rho, as common/ai_belief/viewpoint.h
def accumulate_path(p, sensor, step, obj):
    """p = [dphi, dz]; adds |step·t|/z and |step·r| (viewpoint.h accumulate_path)."""
    r = np.asarray(sensor, float) - np.asarray(obj, float); z = float(np.hypot(*r))
    if z < 1e-3: return                       # degenerate geometry (sensor at the object centre), as viewpoint.h
    rh = r / z; th = np.array([-rh[1], rh[0]]); s = np.asarray(step, float)
    p[0] += abs(float(s @ th)) / z; p[1] += abs(float(s @ rh))


def rho_camera(dphi, dz, z, w_perp):
    half = math.atan(w_perp / (2 * z))
    return math.exp(-(0.0 if dphi == 0 else abs(dphi) / half) - abs(dz) / z)


def rho_lidar_lateral(dphi, g):
    return 1.0 if g <= 0 else math.exp(-abs(dphi) / g)


def w_perp(w, h, back, sensor, obj):
    """countable_fitter.cpp: w|ux·t| + h|uy·t|, ux along the wall, uy = back, t ⊥ the line of sight."""
    r = np.asarray(sensor, float) - np.asarray(obj, float); r /= np.hypot(*r)
    t = np.array([-r[1], r[0]]); uy = np.asarray(back, float); ux = np.array([uy[1], -uy[0]])
    return w * abs(float(ux @ t)) + h * abs(float(uy @ t))


def planned_rho(A, B, c, w, h, back, g=HELIOS_RING_GAP_RAD, steps=200):
    """rho of the A -> B move along the straight chord and along the arc round c (the controller's path lies between)."""
    A, B, c = (np.asarray(v, float) for v in (A, B, c))
    out = {}
    za, zb = float(np.hypot(*(A - c))), float(np.hypot(*(B - c)))
    a0, a1 = math.atan2(*(A - c)[::-1]), math.atan2(*(B - c)[::-1])
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    paths = {'chord': [A + (B - A) * k / steps for k in range(steps + 1)],
             'arc': [c + (za + (zb - za) * k / steps) * np.array([math.cos(a0 + da * k / steps), math.sin(a0 + da * k / steps)])
                     for k in range(steps + 1)]}
    for name, pts in paths.items():
        p = [0.0, 0.0]
        for q0, q1 in zip(pts[:-1], pts[1:]): accumulate_path(p, q0, q1 - q0, c)
        wp = w_perp(w, h, back, B, c)
        out[name] = dict(dphi=p[0], dz=p[1], w_perp=wp, rho_cam=rho_camera(p[0], p[1], zb, wp),
                         rho_L=rho_lidar_lateral(p[0], g))
    return out


# ------------------------------------------------------------------------------------- class parameters (manifest + config)
def _toml(path):
    if tomllib is None: raise RuntimeError('python >= 3.11 (tomllib) needed to read the manifest')
    with open(path, 'rb') as fh: return tomllib.load(fh)   # locale-independent by construction


def _config_float(path, key):
    for line in open(path):
        m = re.match(rf'\s*{key}\s*=\s*([-+0-9.eE]+)', line)
        if m: return float(m.group(1))                        # python float(): locale-independent
    raise KeyError(f'{key} not in {path}')


def class_params(cls, root):
    """Every model input that is not logged: Delta, d = Delta^2/12, P0, v_cam, v_L, q0 (permanence, per predict step)."""
    man = _toml(os.path.join(root, 'common', 'concept_manifest', f'{cls}.concept.toml'))
    cfg = os.path.join(root, f'{cls}_concept', 'etc', 'config.toml')
    delta = [float(x) for x in man['shape']['layer_spread_m']]
    fp, ht = man['prior']['footprint'], man['prior']['height']
    yaw_std = math.radians(float(man['prior']['alignment']['std_deg']))
    P0 = [math.inf, math.inf, ht['std_m'] ** 2, fp['std_m'] ** 2, fp['std_m'] ** 2, yaw_std ** 2]
    pos, size, yaw = (_config_float(cfg, f'AI2CommonMode{k}Std') for k in ('Pos', 'Size', 'Yaw'))
    b2 = KZED_BOUNDARY_STD_M ** 2
    perm = man['prior']['permanence']; dt = _config_float(cfg, 'AI2AgeNominalDtS')
    rate = perm['move_rate_per_day'] / 86400.0 * dt
    q0 = [rate * perm['move_scale_m'] ** 2] * 2 + [rate * fp['std_m'] ** 2] * 3 + [rate * perm['move_scale_rad'] ** 2]
    return dict(cls=cls, delta=delta, d=[x * x / 12 for x in delta], P0=P0,
                v={'cam': [b2] * 5 + [yaw ** 2], 'L': [pos ** 2, pos ** 2, size ** 2, size ** 2, size ** 2, yaw ** 2]},
                q0=q0, predict_dt_s=dt, layer_source=man['shape'].get('layer_spread_source', ''))


# ------------------------------------------------------------------------------------- the scalar model
def theta_var_gls(views, k, prm, hyp):
    """sigma_k^2 after the views. views: [dict(cam, L (bools), rho_cam, rho_L, rho_chain (the rho INTO this view from the
    previous one), chain (list of 6 chain variances))]."""
    idx = [(j, c) for j, v in enumerate(views) for c in ('cam', 'L') if v[c]]
    if not idx: return prm['P0'][k]
    # cumulative log rho per channel: corr(view j, view j') = exp(L[max] - L[min])
    lr = {c: np.cumsum([0.0] + [math.log(max(v['rho_' + c], 1e-300)) for v in views[1:]]) for c in ('cam', 'L', 'chain')}
    J = np.array([j for j, _ in idx]); ch = np.array([c for _, c in idx])
    lo, hi = np.minimum.outer(J, J), np.maximum.outer(J, J)
    sd = np.sqrt(np.array([views[j]['chain'][k] for j in J]))
    C = np.outer(sd, sd) * np.exp(lr['chain'][hi] - lr['chain'][lo])
    same = ch[:, None] == ch[None, :]
    for c in ('cam', 'L'):
        m = same & (ch[:, None] == c)
        C[m] += (prm['v'][c][k] * np.exp(lr[c][hi] - lr[c][lo]))[m]
    d = prm['d'][k]
    C += d if hyp == 'S' else d * same
    n = len(idx)
    one = np.ones(n)
    # lstsq: near-duplicate views (rho within roundoff of 1) make C numerically singular; the minimum-norm solve is the
    # pseudo-inverse's answer (numerical rank, not a modelling cut-off).
    x = np.linalg.lstsq(C, one, rcond=None)[0]
    info = float(one @ x) + (0.0 if math.isinf(prm['P0'][k]) else 1.0 / prm['P0'][k])
    return 1.0 / info if info > 0 else math.inf


def theta_var(views, k, prm, hyp):
    """sigma_k^2 after the views: the same model as theta_var_gls (the GLS 1/(1/P0 + 1'C^-1 1)), computed by a Kalman filter over
    the latent state [theta, b_cam, b_L, e, s_cam, s_L] (PC) or [theta, b_cam, b_L, e, s] (S) — O(views) instead of the GLS's
    O(views^3), which took minutes on a live run's ~1700 updates (test_aba_model checks the two agree). Each view measures
    theta + b_c + e + s_c exactly (R = 0); a view repeating what is already known has a singular innovation, which the
    pseudo-inverse handles (numerical rank, not a modelling cut-off). Between views: b_c <- rho_c b_c + w, e <- rho a e + w with
    a = sd_j / sd_{j-1} (the marginal follows the logged chain), s unchanged. NO PRIOR (cx, cy): the first exact measurement
    defines theta = y - (the latents it carries), so theta's covariance is that of those latents — exact, no diffuse stand-in."""
    if not any(vw['cam'] or vw['L'] for vw in views): return prm['P0'][k]
    pc = hyp == 'PC'
    n = 6 if pc else 5
    TH, BC, BL, E = 0, 1, 2, 3
    SC, SL = (4, 5) if pc else (4, 4)
    P0 = prm['P0'][k]
    v, d = prm['v'], prm['d'][k]

    def h_of(c):
        h = np.zeros(n); h[TH] = 1.0; h[BC if c == 'cam' else BL] = 1.0; h[E] = 1.0; h[SC if c == 'cam' else SL] = 1.0
        return h

    def update(P, chans):
        H = np.array([h_of(c) for c in chans])
        Sx = H @ P @ H.T
        K = P @ H.T @ np.linalg.pinv(Sx, rcond=1e-12, hermitian=True)
        P = P - K @ Sx @ K.T
        return 0.5 * (P + P.T)

    P = np.zeros((n, n))
    P[BC, BC], P[BL, BL] = v['cam'][k], v['L'][k]
    P[SC, SC] = d; P[SL, SL] = d
    sd_prev = None
    started = False
    for j, vw in enumerate(views):
        sd = math.sqrt(vw['chain'][k])
        if not started:
            P[E, E] = sd * sd
        else:
            F = np.eye(n); Q = np.zeros((n, n))
            for c, i in (('cam', BC), ('L', BL)):
                r = vw['rho_' + c]; F[i, i] = r; Q[i, i] = v[c][k] * (1.0 - r * r)
            r = vw['rho_chain']
            if sd_prev and sd_prev > 0.0: F[E, E] = r * sd / sd_prev; Q[E, E] = sd * sd * (1.0 - r * r)
            else: F[E, E] = 0.0; Q[E, E] = sd * sd
            P = F @ P @ F.T + Q
        sd_prev = sd
        chans = [c for c in ('cam', 'L') if vw[c]]
        if not chans: continue
        if not started and math.isinf(P0):
            # theta := y1 - (latents of the first channel): theta's row/column are minus that combination of the latents
            T = np.eye(n); T[TH] = -h_of(chans[0]); T[TH, TH] = 0.0
            P = T @ P @ T.T
            started = True
            if len(chans) > 1: P = update(P, chans[1:])
            continue
        if not started: P[TH, TH] = P0; started = True
        P = update(P, chans)
    return float(P[TH, TH])


def floor(k, prm, hyp, n_channels):
    d = prm['delta'][k]
    if d == 0.0: return 0.0
    info = (n_channels if hyp == 'PC' else 1) * 12.0 / (d * d) + (0.0 if math.isinf(prm['P0'][k]) else 1.0 / prm['P0'][k])
    return 1.0 / math.sqrt(info)


def nominal_views(n, rho_between, channels=('cam', 'L'), chain=None):
    """n dwells, independent channels measuring each, the same rho between consecutive dwells (registered numbers)."""
    ch = chain or [0.0] * 6
    return [dict(cam='cam' in channels, L='L' in channels, rho_cam=rho_between['cam'], rho_L=rho_between['L'],
                 rho_chain=rho_between['chain'], chain=ch) for _ in range(n)]


def registered(root):
    """The numbers of the registered-predictions document: floors, planned rho, and the nominal A/B/A sigmas and deliveries
    (both channels measuring, chain left out — the report puts the logged chain in)."""
    import json, subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    wbt = os.environ.get('WBT_PATH', '/home/pbustos/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt')
    rows = []
    for defn, cls in (('NIGHTSTAND_R', 'nightstand'), ('BED_1', 'bed'), ('TOILET_1', 'toilet')):
        prm = class_params(cls, root)
        plan = json.loads(subprocess.run([sys.executable, os.path.join(here, 'aba_geometry.py'), 'plan', wbt, defn],
                                         capture_output=True, text=True, check=True).stdout)
        print(f'== {defn} ({cls})')
        if not plan['feasible']: print('  NO FEASIBLE STANDPOINTS:', plan['why']); continue
        T = plan['truth']
        print(f"  truth: centre ({T['cx']:.3f}, {T['cy']:.3f}) back ({T['back'][0]:+.3f}, {T['back'][1]:+.3f}) w {T['w']:.4f} "
              f"h {T['h'][0]:.4f}..{T['h'][1]:.4f} H {T['H']:.4f}  [{T['source']}]")
        print(f"  standpoints: z0 {plan['z0']:.3f} m -> z {plan['z']:.3f} m (moved {plan['moved_m']:+.2f}), side {plan['side']:+d}, "
              f"r_clear {plan['r_clear']:.4f}")
        print(f"    A ({plan['A'][0]:.3f}, {plan['A'][1]:.3f}) yaw {plan['A'][2]:+.3f}   B ({plan['B'][0]:.3f}, {plan['B'][1]:.3f}) "
              f"yaw {plan['B'][2]:+.3f}")
        for u in plan['footprints_failed']:
            print(f"    footprint not read: {u['name']} at ({u['origin'][0]:.2f}, {u['origin'][1]:.2f}), "
                  f"{u['dist_to_standpoint_m']:.2f} m from the nearer standpoint ({u['why'][:60]})")
        pr = plan['planned_rho']
        for p in ('chord', 'arc'):
            q = pr[p]
            print(f"    {p:5s}: dphi {q['dphi']:.4f} rad, dz {q['dz']:.4f} m, w_perp {q['w_perp']:.3f} -> rho_cam {q['rho_cam']:.3e}, "
                  f"rho_L {q['rho_L']:.3e}")
        rho_ab = dict(cam=max(pr['chord']['rho_cam'], pr['arc']['rho_cam']), L=max(pr['chord']['rho_L'], pr['arc']['rho_L']),
                      chain=1.0)
        print(f"  {'dof':4s} {'Delta':>7s} {'D/sqrt12':>9s} {'D/sqrt24':>9s} | {'floor PC':>9s} {'floor S':>9s} | "
              f"{'sA PC':>8s} {'sB PC':>8s} {'sA2 PC':>8s} | {'sA S':>8s} {'sB S':>8s} {'sA2 S':>8s}")
        dH = {h: [0.0, 0.0] for h in HYPS}
        for k, dof in enumerate(DOFS):
            D = prm['delta'][k]
            s = {h: [math.sqrt(theta_var(nominal_views(n, rho_ab), k, prm, h)) for n in (1, 2, 3)] for h in HYPS}
            for h in HYPS:
                dH[h][0] += math.log(s[h][0] / s[h][1]); dH[h][1] += math.log(s[h][1] / s[h][2])
            print(f"  {dof:4s} {D:7.3f} {D / math.sqrt(12):9.4f} {D / math.sqrt(24):9.4f} | {floor(k, prm, 'PC', 2):9.4f} "
                  f"{floor(k, prm, 'S', 2):9.4f} | " + ' '.join(f'{x:8.4f}' for x in s['PC']) + ' | ' +
                  ' '.join(f'{x:8.4f}' for x in s['S']))
            rows.append((defn, dof, D, s))
        for h in HYPS:
            print(f"  delivered (sum of marginal ln sigma drops), {h}: B {dH[h][0]:.3f} nats, A2 {dH[h][1]:.3f} nats "
                  f"(band +- {6 * math.log(1 + TOL_RATIO):.3f})")
    return rows


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'registered':
        root = sys.argv[2] if len(sys.argv) > 2 else os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        registered(root)
    else:
        print(__doc__); sys.exit(2)
