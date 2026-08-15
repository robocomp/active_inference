/*
 * common/motion_corruption/motion_corruption.h
 *
 * Shared, header-only "ego-motion mask corruption" model. Producer-side physics for the
 * saccadic-suppression / VOR idea: while the camera moves, a YOLO mask is corrupted in a way
 * the consumer cannot reconstruct without the capture-time ego-motion, exposure and timing.
 * This file is the SINGLE place that owns that math (mirrors the YOLO-score→covariance map):
 *
 *   producer (retina) — has exposure, accurate capture stamp, camera twist at capture, and the
 *                          raw image-plane mask geometry → computes the corruption DECOMPOSITION.
 *   common/ (here)       — the interaction-matrix → (dot_d, bias, variance) math, stateless/pure.
 *   consumer (concept)   — decides gate-vs-downweight (per-concept b_max), per-DOF split, and folds
 *                          the variance into its innovation cov. NOT here.
 *
 * Derivation: the image Jacobian (interaction matrix, Corke RVC §15 / Chaumette) gives the image-plane
 * velocity of a static world point under camera twist. Metric mask displacement ≈ Z·‖ṡ‖·Δt. Rotation
 * carries NO 1/Z (depth-independent, range-amplified, edge-compounded via the (1+x²) terms); translation
 * does (depth-attenuated). The window Δt splits into exposure blur + timing jitter (symmetric → variance)
 * and a timing OFFSET (biased → the consumer should gate on it). See EFE/perception notes.
 *
 * Convention: the interaction matrix is in the STANDARD optical frame (x right, y DOWN, z FORWARD along
 * the optical axis). The retina's zed frame is (x right, y forward/depth, z up); convert the twist
 * with vcam_to_optical() before calling mask_corruption().
 */

#pragma once

#include <cmath>

#include <Eigen/Dense>

namespace rc::motion
{

// Camera body twist: velocity of the camera w.r.t. a STATIC scene, expressed in the camera's own
// frame. Linear in m/s, angular in rad/s.
struct Twist
{
    Eigen::Vector3f v = Eigen::Vector3f::Zero();
    Eigen::Vector3f w = Eigen::Vector3f::Zero();
};

// Per-mask corruption decomposition (all consumer-agnostic — the consumer decides what to do with them).
struct MaskCorruption
{
    float dot_d  = 0.0f;   // metric position-corruption speed Z·‖ṡ‖ (m/s) — diagnostic / per-DOF input
    float bias_m = 0.0f;   // systematic mask displacement from the timing OFFSET (m) — gate on this
    float var_m  = 0.0f;   // measurement-noise variance to ADD to R (m²) — blur + timing jitter, ×peripheral
};

struct CorruptionParams
{
    float exposure_s      = 0.005f;   // T_exp: rolling/global exposure → motion blur smear window
    float timing_jitter_s = 0.010f;   // σ_t: stdev of the mask-vs-pose timestamp mismatch (zero-mean part)
    float timing_offset_s = 0.0f;     // δt: KNOWN systematic mask-vs-pose lag (biased part); 0 if unknown
};

// Finite-difference the camera body twist from two consecutive camera poses (world←camera), expressed
// in the camera frame at the previous step (≈ the body twist over the interval). Zero twist if dt<=0.
inline Twist body_twist_from_poses(const Eigen::Matrix3f& R_prev, const Eigen::Vector3f& t_prev,
                                   const Eigen::Matrix3f& R_curr, const Eigen::Vector3f& t_curr,
                                   float dt_s)
{
    Twist tw;
    if (not (dt_s > 0.0f))
        return tw;
    const Eigen::Matrix3f R_rel = R_prev.transpose() * R_curr;
    const Eigen::Vector3f t_rel = R_prev.transpose() * (t_curr - t_prev);
    const Eigen::AngleAxisf aa(R_rel);
    tw.w = (aa.angle() * aa.axis()) / dt_s;
    tw.v = t_rel / dt_s;
    return tw;
}

// Map a twist from the retina zed frame (x right, y forward/depth, z up) into the standard optical
// frame (x right, y down, z forward): x_o=x, y_o=-z, z_o=y. Same map applies to linear and angular.
inline Twist vcam_to_optical(const Twist& t)
{
    Twist o;
    o.v = Eigen::Vector3f(t.v.x(), -t.v.z(), t.v.y());
    o.w = Eigen::Vector3f(t.w.x(), -t.w.z(), t.w.y());
    return o;
}

// Interaction matrix (image Jacobian) for a point at normalized optical coords (xn,yn) and optical-axis
// depth Z: [ẋn; ẏn] = L·[v; w]. Standard form (Chaumette/Corke). Z clamped away from 0.
inline Eigen::Matrix<float, 2, 6> interaction_matrix(float xn, float yn, float Z)
{
    const float z = (std::abs(Z) < 1e-3f) ? (Z < 0 ? -1e-3f : 1e-3f) : Z;
    Eigen::Matrix<float, 2, 6> L;
    //        vx        vy        vz          wx            wy           wz
    L << -1.0f / z,   0.0f,    xn / z,     xn * yn,   -(1.0f + xn * xn),   yn,
          0.0f,    -1.0f / z,  yn / z,  (1.0f + yn * yn),  -xn * yn,      -xn;
    return L;
}

// Core model. twist_optical MUST already be in the optical frame (use vcam_to_optical). (xn,yn) are
// normalized optical image coords of the mask centroid; Z is its optical-axis depth (m).
inline MaskCorruption mask_corruption(const Twist& twist_optical,
                                      float xn, float yn, float Z,
                                      const CorruptionParams& p)
{
    MaskCorruption out;
    if (not std::isfinite(Z) or Z <= 0.0f)
        return out;

    Eigen::Matrix<float, 6, 1> xi;
    xi << twist_optical.v, twist_optical.w;
    const Eigen::Vector2f s_dot = interaction_matrix(xn, yn, Z) * xi;   // normalized image flow (1/s)

    // Metric position-corruption speed: a normalized-image displacement ṡ deprojects to Z·ṡ in metres.
    out.dot_d = Z * s_dot.norm();

    // Peripheral penalty (lens distortion / grazing angle worst at the rim) — reuse the matrix's own
    // (1+r²) structure. r = normalized radius from the principal point.
    const float r2  = xn * xn + yn * yn;
    const float rho = 1.0f + r2;

    // Bias: a KNOWN timing offset shifts the whole mask by dot_d·δt (systematic; consumer gates on it).
    out.bias_m = out.dot_d * std::abs(p.timing_offset_s);

    // Variance: exposure blur (uniform over T_exp → var = (dot_d·T_exp)²/12) + zero-mean timing jitter
    // (dot_d·σ_t)², scaled by the peripheral penalty. This is the term to ADD to the consumer's R.
    const float blur_var   = (out.dot_d * p.exposure_s) * (out.dot_d * p.exposure_s) / 12.0f;
    const float jitter_var = (out.dot_d * p.timing_jitter_s) * (out.dot_d * p.timing_jitter_s);
    out.var_m = rho * (blur_var + jitter_var);

    return out;
}

}   // namespace rc::motion
