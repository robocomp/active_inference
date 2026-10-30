// scan_registration -- 2-D point-to-line registration of two scans (the measurement behind the Helios mount
// estimate). Gauss-Newton on r_j = n_i . (R p_j + t - q_i), q_i the nearest reference point, n_i the PCA normal
// of its neighbourhood. No distance gate: every pair is weighted by the EM responsibility of an inlier Gaussian
// (sigma_m) against a uniform over outlier_window_m (the window is the uniform's support, not a cut-off).
// The inlier sigma is annealed from half the window down to sigma_m (a fixed 1 cm Gaussian has an empty basin for a
// scan displaced by tens of cm). Each step is a MAP step under the weak pose prior (scan extent, 1 rad) centred on the
// initial guess; convergence = step below a tenth of a posterior standard deviation (Mahalanobis, so units mix correctly).
// Residual = distance to the line fitted through the normal_k neighbours of the nearest reference point.
// cov = sigma^2 (H^-1 + H^-1 M_ref H^-1): both scans' range noise, the reference part being a correlation (a ref
// point sits in ~k fitted lines). H is first corrected for the information the noisy normals manufacture (errors
// in variables); axes not observed beyond that are reported as uninformed (variance = scan extent^2).
// Pure Eigen + header-only nanoflann.
#pragma once
#include <Eigen/Dense>
#include <vector>

namespace sc::calib
{
struct ScanRegConfig {
    int    max_iters   = 30;
    int    normal_k    = 24;       // neighbours for the PCA normal of each reference point
    double sigma_m     = 0.01;     // inlier sigma of a point-to-line residual [m]
    double outlier_window_m = 0.5; // the uniform the outlier component spreads over (search window, not a gate)
    double pi_inlier0  = 0.8;      // prior inlier share, re-estimated by EM
};
struct ScanRegResult {
    bool ok = false;
    Eigen::Vector3d xyt = Eigen::Vector3d::Zero();    // pose of CUR in REF: p_ref = R(t) p_cur + (x, y)
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();// see above
    int n_used = 0; double inlier_share = 0.0, rms = 0.0; int iters = 0;
};
[[nodiscard]] ScanRegResult register_scans_2d(const std::vector<Eigen::Vector2f>& ref,
                                              const std::vector<Eigen::Vector2f>& cur,
                                              const Eigen::Vector3d& init_xyt,
                                              const ScanRegConfig& cfg = {});
}  // namespace sc::calib
