// human_kinematic_model.h
// BODY_18 kinematic generator using fixed segment lengths + main joint angles only.
// Infers angles (not bone lengths) for shoulders, elbows, and a rigid lower body.
// Outputs canonical keypoints (18,3), pelvis/root near origin, Y-up.
//
// Ported from human_kinematic_model.py. Pure Eigen (float32), no torch/DSR.
#pragma once

#include <Eigen/Dense>
#include <array>
#include <optional>

#include "body18.h"

namespace rc::human
{
// Parameter vector layout (11 DOF), matching the Python ordering exactly:
//   [0..2] sh_L  (yaw, pitch, roll)
//   [3..5] sh_R  (yaw, pitch, roll)
//   [6]    el_L  (flex)
//   [7]    el_R  (flex)
//   [8]    lb_x  (lower-body translation x)
//   [9]    lb_z  (lower-body translation z)
//   [10]   lb_roll (lower-body rotation about Y)
using Vec11   = Eigen::Matrix<float, 11, 1>;
using KpArray = Eigen::Matrix<float, NUM_KP, 3>;  // 18x3 keypoints

inline constexpr int DOF = 11;

// Named accessors into the 11-vector (mirrors _angles_from_vector).
struct Angles
{
    Eigen::Vector3f sh_L;   // yaw, pitch, roll
    Eigen::Vector3f sh_R;   // yaw, pitch, roll
    float el_L;
    float el_R;
    float lb_x;
    float lb_z;
    float lb_roll;

    static Angles from_vector(const Vec11 &x)
    {
        Angles a;
        a.sh_L    = x.segment<3>(0);
        a.sh_R    = x.segment<3>(3);
        a.el_L    = x(6);
        a.el_R    = x(7);
        a.lb_x    = x(8);
        a.lb_z    = x(9);
        a.lb_roll = x(10);
        return a;
    }
};

// ---------------- rotations ----------------
Eigen::Matrix3f Rx(float a);
Eigen::Matrix3f Ry(float a);
Eigen::Matrix3f Rz(float a);
// yaw about Y, pitch about X, roll about Z  ->  Ry(yaw) * Rx(pitch) * Rz(roll)
Eigen::Matrix3f euler_yaw_pitch_roll(float yaw, float pitch, float roll);

// ---------------- segment lengths (fixed) ----------------
struct SegmentLengths
{
    float torso;
    float shoulder_offset_x;
    float hip_offset_x;
    float upper_arm;
    float lower_arm;
    float thigh;
    float calf;
    // Face offsets relative to NECK (kept fixed, not optimized)
    Eigen::Vector3f nose_off;
    Eigen::Vector3f reye_off;
    Eigen::Vector3f leye_off;
    Eigen::Vector3f rear_off;
    Eigen::Vector3f lear_off;
};

// Derive fixed segment lengths and face offsets from a template skeleton (18x3).
SegmentLengths lengths_from_standard(const KpArray &standard_kp);

// Canonical BODY_18 rest-pose template used to derive lengths (from main.py).
KpArray standard_template();

// ---------------- kinematic model ----------------
class HumanKinematicModel
{
public:
    explicit HumanKinematicModel(const SegmentLengths &lengths) : L_(lengths) {}
    HumanKinematicModel() : L_(lengths_from_standard(standard_template())) {}   // default = standard template

    // 11-DOF angles -> 18x3 canonical keypoints.
    KpArray forward(const Angles &angles) const;
    KpArray forward(const Vec11 &x) const { return forward(Angles::from_vector(x)); }

    const SegmentLengths &lengths() const { return L_; }
    void set_lengths(const SegmentLengths &l) { L_ = l; }   // for per-person online calibration

private:
    SegmentLengths L_;
};

// Online per-person bone-length calibration: EMA the scalar segment lengths in `L` toward the limb
// distances measured from observed keypoints `kp` (room frame), weight `alpha` on the new measurement.
// A segment is only updated when BOTH its endpoints are finite AND (if `conf` is given) above
// `min_conf` [0,100] — so a low-confidence keypoint doesn't corrupt the calibrated lengths. Face
// offsets are left untouched. Sanity-bounds each measurement to [5cm, 1.2m].
void calibrate_lengths(SegmentLengths &L, const KpArray &kp,
                       const std::optional<std::array<float, NUM_KP>> &conf,
                       float min_conf, float alpha);

// ---------------- joint limits ----------------
// Per-group [min, max] in radians (positions for lb_x/lb_z are in meters).
struct JointLimits
{
    Eigen::Vector3f sh_min, sh_max;   // shoulder yaw/pitch/roll
    float el_min, el_max;             // elbow flex
    float lb_x_min, lb_x_max;
    float lb_z_min, lb_z_max;
    float lb_roll_min, lb_roll_max;
};

JointLimits default_joint_limits_radians();

}  // namespace rc::human
