# Offline replica of rc::mount::Accum::solve (mount_lidar_pair.h): A = H + I, x = A^-1 b,
# p = -x in units of prior sigma, sigma_i = sqrt(C_ii) * sqrt(max(1, chi2/dof)).
# ★ std::from_chars discipline does not apply: python's float() is locale-independent.
import sys, numpy as np

PSIG = np.array([0.0035, 0.010, 0.0035, 1.0])   # pitch rad, height m, yaw rad, dt
NAME = ["pitch", "height", "yaw", "dt"]

def load(path):
    H = np.zeros((4,4)); b = np.zeros(4); rTr = 0.0; n = 0; cam = robot = "?"
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith('#'): continue
        f = ln.split(',')
        if f[0] == 'camera': cam = f[1]
        elif f[0] == 'robot': robot = f[1]
        elif f[0] == 'n': n = int(f[1])
        elif f[0] == 'rTr': rTr = float(f[1])
        elif f[0] == 'H': H[int(f[1]), int(f[2])] = H[int(f[2]), int(f[1])] = float(f[3])
        elif f[0] == 'b': b[int(f[1])] = float(f[2])
    return robot, cam, H, b, rTr, n

def solve(H, b, rTr, n):
    A = H + np.eye(4); C = np.linalg.inv(A); x = C @ b
    chi2 = max(0.0, rTr - x @ b)
    dof = max(1.0, 2.0*n - 4.0)
    chi2_dof = chi2 / dof
    infl = np.sqrt(max(1.0, chi2_dof))
    sig = np.sqrt(np.maximum(0.0, np.diag(C))) * infl
    return -x, sig, chi2_dof, C

for path in sys.argv[1:]:
    robot, cam, H, b, rTr, n = load(path)
    p, sig, chi2_dof, C = solve(H, b, rTr, n)
    # condition number over the 3 estimated parameters (dt carries no information at all)
    ev = np.linalg.eigvalsh((H + np.eye(4))[:3,:3])
    print(f"\n=== {robot}/{cam}   n={n} pairs   chi2/dof={chi2_dof:.2f}   cond={ev.max()/ev.min():.1f}")
    for i in range(3):
        phys = p[i]*PSIG[i]*(180/np.pi if i in (0,2) else 1.0)
        psig = sig[i]*PSIG[i]*(180/np.pi if i in (0,2) else 1.0)
        unit = "deg" if i in (0,2) else "m"
        inf  = "INFORMED" if sig[i] < 0.9 else "prior-dominated"
        print(f"  {NAME[i]:7s} {phys:+9.4f} {unit}  +/- {psig:.4f}   (sigma_units {sig[i]:.3f}, {inf})")
    # correlations between the estimated parameters
    D = np.sqrt(np.diag(C)[:3])
    R = C[:3,:3] / np.outer(D, D)
    print(f"  rho: pitch/height {R[0,1]:+.3f}  pitch/yaw {R[0,2]:+.3f}  height/yaw {R[1,2]:+.3f}")
