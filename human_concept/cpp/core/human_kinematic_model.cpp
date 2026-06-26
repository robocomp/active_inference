// human_kinematic_model.cpp  — port of human_kinematic_model.py
#include "human_kinematic_model.h"

#include <cmath>
#include <limits>

namespace rc::human
{
// ---------------- rotations ----------------
Eigen::Matrix3f Rx(float a)
{
    const float c = std::cos(a), s = std::sin(a);
    Eigen::Matrix3f R;
    R << 1.f, 0.f, 0.f,
         0.f,   c,  -s,
         0.f,   s,   c;
    return R;
}

Eigen::Matrix3f Ry(float a)
{
    const float c = std::cos(a), s = std::sin(a);
    Eigen::Matrix3f R;
    R <<   c, 0.f,   s,
         0.f, 1.f, 0.f,
          -s, 0.f,   c;
    return R;
}

Eigen::Matrix3f Rz(float a)
{
    const float c = std::cos(a), s = std::sin(a);
    Eigen::Matrix3f R;
    R <<   c,  -s, 0.f,
           s,   c, 0.f,
         0.f, 0.f, 1.f;
    return R;
}

Eigen::Matrix3f euler_yaw_pitch_roll(float yaw, float pitch, float roll)
{
    // yaw about Y, pitch about X, roll about Z
    return Ry(yaw) * Rx(pitch) * Rz(roll);
}

// ---------------- rest-pose template (from main.py standard_template_np) ----------------
KpArray standard_template()
{
    KpArray kp = KpArray::Zero();

    kp.row(KP::NOSE)  << 0.0f, 1.65f, 0.0f;
    kp.row(KP::R_EYE) << 0.08f, 1.68f, 0.05f;
    kp.row(KP::L_EYE) << -0.08f, 1.68f, 0.05f;
    kp.row(KP::R_EAR) << 0.12f, 1.65f, 0.0f;
    kp.row(KP::L_EAR) << -0.12f, 1.65f, 0.0f;

    kp.row(KP::NECK) << 0.0f, 1.50f, 0.0f;

    kp.row(KP::R_SHOULDER) << 0.20f, 1.50f, 0.0f;
    kp.row(KP::L_SHOULDER) << -0.20f, 1.50f, 0.0f;

    kp.row(KP::R_ELBOW) << 0.35f, 1.20f, 0.0f;
    kp.row(KP::R_WRIST) << 0.50f, 0.95f, 0.0f;
    kp.row(KP::L_ELBOW) << -0.35f, 1.20f, 0.0f;
    kp.row(KP::L_WRIST) << -0.50f, 0.95f, 0.0f;

    kp.row(KP::R_HIP) << 0.15f, 1.00f, 0.0f;
    kp.row(KP::L_HIP) << -0.15f, 1.00f, 0.0f;

    kp.row(KP::R_KNEE)  << 0.15f, 0.60f, 0.0f;
    kp.row(KP::R_ANKLE) << 0.15f, 0.10f, 0.0f;
    kp.row(KP::L_KNEE)  << -0.15f, 0.60f, 0.0f;
    kp.row(KP::L_ANKLE) << -0.15f, 0.10f, 0.0f;

    return kp;
}

// ---------------- segment lengths ----------------
SegmentLengths lengths_from_standard(const KpArray &kp)
{
    auto row = [&](int i) -> Eigen::Vector3f { return kp.row(i).transpose(); };
    auto d   = [&](int i, int j) -> float { return (row(i) - row(j)).norm(); };

    const Eigen::Vector3f pelvis = 0.5f * (row(KP::L_HIP) + row(KP::R_HIP));

    SegmentLengths L;
    L.torso             = (row(KP::NECK) - pelvis).norm();
    L.shoulder_offset_x = std::abs(kp(KP::R_SHOULDER, 0) - kp(KP::NECK, 0));
    L.hip_offset_x      = std::abs(kp(KP::R_HIP, 0) - pelvis(0));
    L.upper_arm         = d(KP::R_SHOULDER, KP::R_ELBOW);
    L.lower_arm         = d(KP::R_ELBOW, KP::R_WRIST);
    L.thigh             = d(KP::R_HIP, KP::R_KNEE);
    L.calf              = d(KP::R_KNEE, KP::R_ANKLE);

    const Eigen::Vector3f neck = row(KP::NECK);
    L.nose_off = row(KP::NOSE) - neck;
    L.reye_off = row(KP::R_EYE) - neck;
    L.leye_off = row(KP::L_EYE) - neck;
    L.rear_off = row(KP::R_EAR) - neck;
    L.lear_off = row(KP::L_EAR) - neck;

    return L;
}

// ---------------- forward kinematics ----------------
KpArray HumanKinematicModel::forward(const Angles &a) const
{
    const SegmentLengths &L = L_;
    KpArray kp = KpArray::Constant(std::numeric_limits<float>::quiet_NaN());

    // Lower body as a rigid part on the ground plane: x,z translation + rotation about Y.
    const Eigen::Vector3f pelvis(a.lb_x, 0.0f, a.lb_z);
    const Eigen::Matrix3f R_lb = Ry(a.lb_roll);

    const Eigen::Vector3f neck = pelvis + Eigen::Vector3f(0.0f, L.torso, 0.0f);

    const Eigen::Vector3f l_sh = neck + Eigen::Vector3f(-L.shoulder_offset_x, 0.0f, 0.0f);
    const Eigen::Vector3f r_sh = neck + Eigen::Vector3f(L.shoulder_offset_x, 0.0f, 0.0f);

    const Eigen::Vector3f l_hip = pelvis + R_lb * Eigen::Vector3f(-L.hip_offset_x, 0.0f, 0.0f);
    const Eigen::Vector3f r_hip = pelvis + R_lb * Eigen::Vector3f(L.hip_offset_x, 0.0f, 0.0f);

    const Eigen::Matrix3f R_sh_L = euler_yaw_pitch_roll(a.sh_L(0), a.sh_L(1), a.sh_L(2));
    const Eigen::Matrix3f R_sh_R = euler_yaw_pitch_roll(a.sh_R(0), a.sh_R(1), a.sh_R(2));

    // Elbow hinges (flexion) about local X.
    const Eigen::Matrix3f R_el_L = Rx(a.el_L);
    const Eigen::Matrix3f R_el_R = Rx(a.el_R);

    const Eigen::Vector3f v_upper_arm(0.0f, -L.upper_arm, 0.0f);
    const Eigen::Vector3f v_lower_arm(0.0f, -L.lower_arm, 0.0f);

    const Eigen::Vector3f l_el = l_sh + R_sh_L * v_upper_arm;
    const Eigen::Vector3f r_el = r_sh + R_sh_R * v_upper_arm;

    const Eigen::Vector3f l_wr = l_el + R_sh_L * (R_el_L * v_lower_arm);
    const Eigen::Vector3f r_wr = r_el + R_sh_R * (R_el_R * v_lower_arm);

    const Eigen::Vector3f v_thigh_rot = R_lb * Eigen::Vector3f(0.0f, -L.thigh, 0.0f);
    const Eigen::Vector3f v_calf_rot  = R_lb * Eigen::Vector3f(0.0f, -L.calf, 0.0f);

    const Eigen::Vector3f l_kn = l_hip + v_thigh_rot;
    const Eigen::Vector3f r_kn = r_hip + v_thigh_rot;
    const Eigen::Vector3f l_an = l_kn + v_calf_rot;
    const Eigen::Vector3f r_an = r_kn + v_calf_rot;

    // Face keypoints (fixed offsets from neck).
    const Eigen::Vector3f nose  = neck + L.nose_off;
    const Eigen::Vector3f r_eye = neck + L.reye_off;
    const Eigen::Vector3f l_eye = neck + L.leye_off;
    const Eigen::Vector3f r_ear = neck + L.rear_off;
    const Eigen::Vector3f l_ear = neck + L.lear_off;

    kp.row(KP::NOSE)       = nose.transpose();
    kp.row(KP::NECK)       = neck.transpose();
    kp.row(KP::R_SHOULDER) = r_sh.transpose();
    kp.row(KP::R_ELBOW)    = r_el.transpose();
    kp.row(KP::R_WRIST)    = r_wr.transpose();
    kp.row(KP::L_SHOULDER) = l_sh.transpose();
    kp.row(KP::L_ELBOW)    = l_el.transpose();
    kp.row(KP::L_WRIST)    = l_wr.transpose();
    kp.row(KP::R_HIP)      = r_hip.transpose();
    kp.row(KP::R_KNEE)     = r_kn.transpose();
    kp.row(KP::R_ANKLE)    = r_an.transpose();
    kp.row(KP::L_HIP)      = l_hip.transpose();
    kp.row(KP::L_KNEE)     = l_kn.transpose();
    kp.row(KP::L_ANKLE)    = l_an.transpose();
    kp.row(KP::R_EYE)      = r_eye.transpose();
    kp.row(KP::L_EYE)      = l_eye.transpose();
    kp.row(KP::R_EAR)      = r_ear.transpose();
    kp.row(KP::L_EAR)      = l_ear.transpose();

    return kp;
}

// ---------------- joint limits ----------------
JointLimits default_joint_limits_radians()
{
    const float deg = static_cast<float>(M_PI) / 180.0f;
    JointLimits lim;
    lim.sh_min = Eigen::Vector3f(-90.f, -90.f, -90.f) * deg;
    lim.sh_max = Eigen::Vector3f(90.f, 90.f, 90.f) * deg;
    lim.el_min = 0.0f;
    lim.el_max = 150.0f * deg;
    lim.lb_x_min = -1.0f;  lim.lb_x_max = 1.0f;
    lim.lb_z_min = -1.0f;  lim.lb_z_max = 1.0f;
    lim.lb_roll_min = -45.0f * deg;  lim.lb_roll_max = 45.0f * deg;
    return lim;
}

}  // namespace rc::human
