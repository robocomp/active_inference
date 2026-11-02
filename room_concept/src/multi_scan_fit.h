// multi_scan_fit.h -- adoption after a recognition swap (spec 2026-11-01-relocaliser-redesign sec 6).
// The last M keyframes are fitted JOINTLY against the remembered room K, around ltsm's T(K<-P) prior with its covariance.
// No likelihood is re-implemented: the scans are placed in the latest keyframe's frame and handed to rc::reloc::polish
// (the relocaliser's own robust per-point mixture, EM-annealed sigma, door cells, common-mode Laplace covariance); the
// prior then enters as a Gaussian factor (one information-form fusion, exact when l is quadratic around the mode).
// Pure function: deep-copied inputs, no DSR, no Qt -- safe on a worker thread.
#pragma once
#include "reloc_search.h"
#include <utility>
#include <vector>

namespace rc::rivals
{
/// `scans[i].first` = keyframe i's pose RELATIVE to the latest keyframe (x, y, theta; latest = identity), `.second` its
/// points in the keyframe's own robot frame. `prior`/`prior_cov` = the pose of the LATEST keyframe in K. Returns the
/// latest keyframe's pose in K: Mode::pose, Mode::cov = Laplace covariance with the prior folded in, Mode::log_lik and
/// Mode::median_abs on the joint point set. No scans / no polygon: pose = prior, cov = prior_cov, log_lik = NaN.
rc::reloc::Mode multi_scan_fit(const rc::reloc::LikelihoodRaster& K, const std::vector<Eigen::Vector2f>& polygon,
                               const std::vector<std::pair<Eigen::Vector3f, std::vector<Eigen::Vector2f>>>& scans,
                               const Eigen::Vector3f& prior, const Eigen::Matrix3f& prior_cov, const rc::reloc::Params& p);
}
