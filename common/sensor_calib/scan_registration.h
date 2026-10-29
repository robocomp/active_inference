// scan_registration -- 2-D point-to-line registration of two scans (the measurement behind the Helios mount
// estimate). Gauss-Newton on r_j = n_i . (R p_j + t - q_i), q_i the nearest reference point, n_i the PCA normal
// of its neighbourhood. No distance gate: every pair is weighted by the EM responsibility of an inlier Gaussian
// (sigma_m) against a uniform over outlier_window_m (the window is the uniform's support, not a cut-off).
// The inlier sigma is annealed from half the window down to sigma_m (a fixed 1 cm Gaussian has an empty basin for a
// scan displaced by tens of cm). Convergence: |step| < 1e-4 (NN flips leave a ~2e-5 limit cycle, so 1e-6 never fires).
// cov propagates the range noise of BOTH scans (ref noise is common to ~normal_k residuals); see the .cpp.
// normal_k: keep it >= ~24 at 0.25 deg / 1 cm; 8 neighbours span ~7 cm and their tilted normals fake along-wall info.
// Pure Eigen + header-only nanoflann.
#pragma once
#include <Eigen/Dense>
#include <vector>

namespace sc::calib
{
struct ScanRegConfig {
    int    max_iters   = 30;
    int    normal_k    = 8;        // neighbours for the PCA normal of each reference point
    double sigma_m     = 0.01;     // inlier sigma of a point-to-line residual [m]
    double outlier_window_m = 0.5; // the uniform the outlier component spreads over (search window, not a gate)
    double pi_inlier0  = 0.8;      // prior inlier share, re-estimated by EM
};
struct ScanRegResult {
    bool ok = false;
    Eigen::Vector3d xyt = Eigen::Vector3d::Zero();    // pose of CUR in REF: p_ref = R(t) p_cur + (x, y)
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();// sigma^2 (H^-1 + H^-1 M_ref H^-1), both scans' noise
    int n_used = 0; double inlier_share = 0.0, rms = 0.0; int iters = 0;
};
[[nodiscard]] ScanRegResult register_scans_2d(const std::vector<Eigen::Vector2f>& ref,
                                              const std::vector<Eigen::Vector2f>& cur,
                                              const Eigen::Vector3d& init_xyt,
                                              const ScanRegConfig& cfg = {});
}  // namespace sc::calib
