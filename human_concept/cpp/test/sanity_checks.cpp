// sanity_checks.cpp
// Self-contained unit checks for the kinematic model and Kabsch alignment.
// Exits non-zero on any failure.
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <random>

#include "../core/body18.h"
#include "../core/human_kinematic_model.h"
#include "../core/vfe_inference.h"

using namespace rc::human;

namespace
{
int g_failures = 0;

void check(bool ok, const char *what)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

bool close(float a, float b, float tol = 1e-4f) { return std::fabs(a - b) <= tol; }
}  // namespace

int main()
{
    const SegmentLengths L = lengths_from_standard(standard_template());
    const HumanKinematicModel model(L);

    // ---- FK at zero angles: structural rest-pose facts ----
    const KpArray kp = model.forward(Vec11::Zero());

    auto row = [&](int i) { return Eigen::Vector3f(kp.row(i).transpose()); };

    const Eigen::Vector3f pelvis = 0.5f * (row(KP::L_HIP) + row(KP::R_HIP));
    check(pelvis.norm() < 1e-5f, "pelvis at origin at zero angles");

    check((row(KP::NECK) - Eigen::Vector3f(0, L.torso, 0)).norm() < 1e-5f,
          "neck = (0, torso, 0)");

    check(close((row(KP::R_SHOULDER) - row(KP::R_ELBOW)).norm(), L.upper_arm),
          "right upper-arm length preserved");
    check(close((row(KP::R_ELBOW) - row(KP::R_WRIST)).norm(), L.lower_arm),
          "right lower-arm length preserved");
    check(close((row(KP::L_HIP) - row(KP::L_KNEE)).norm(), L.thigh),
          "left thigh length preserved");
    check(close((row(KP::L_KNEE) - row(KP::L_ANKLE)).norm(), L.calf),
          "left calf length preserved");

    // Arms hang straight down at zero shoulder/elbow angle.
    check(close(row(KP::R_WRIST).x(), row(KP::R_SHOULDER).x()) &&
          close(row(KP::R_WRIST).y(), L.torso - (L.upper_arm + L.lower_arm)),
          "right wrist hangs straight below shoulder");

    // Face offsets reproduced from neck.
    check((row(KP::NOSE) - (row(KP::NECK) + L.nose_off)).norm() < 1e-5f,
          "nose = neck + nose_off");

    // ---- Kabsch: identity ----
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> uni(-1.f, 1.f);
        Eigen::MatrixXf P(8, 3);
        for (int i = 0; i < 8; ++i)
            P.row(i) = Eigen::RowVector3f(uni(rng), uni(rng), uni(rng));
        const KabschResult kb = kabsch(P, P);
        check((kb.R - Eigen::Matrix3f::Identity()).norm() < 1e-4f, "Kabsch identity R ~ I");
        check(kb.t.norm() < 1e-4f, "Kabsch identity t ~ 0");
    }

    // ---- Kabsch: recover a known rotation + translation ----
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> uni(-1.f, 1.f);
        Eigen::MatrixXf src(10, 3);
        for (int i = 0; i < 10; ++i)
            src.row(i) = Eigen::RowVector3f(uni(rng), uni(rng), uni(rng));

        const Eigen::Matrix3f R0 = euler_yaw_pitch_roll(0.5f, -0.3f, 0.2f);
        const Eigen::Vector3f t0(0.4f, -0.2f, 0.7f);
        Eigen::MatrixXf dst(10, 3);
        for (int i = 0; i < 10; ++i)
            dst.row(i) = (R0 * src.row(i).transpose() + t0).transpose();

        const KabschResult kb = kabsch(src, dst);
        check((kb.R - R0).norm() < 1e-3f, "Kabsch recovers known rotation");
        check((kb.t - t0).norm() < 1e-3f, "Kabsch recovers known translation");
    }

    // ---- Estimator runs and converges on a synthetic observation ----
    {
        // Build a synthetic 'live' from a known pose, perfectly observable.
        Vec11 truth = Vec11::Zero();
        truth(0) = 0.3f;  truth(1) = -0.2f;          // sh_L yaw/pitch
        truth(3) = -0.3f; truth(4) = -0.2f;          // sh_R yaw/pitch (mirrored)
        truth(6) = 0.6f;  truth(7) = 0.6f;           // elbows
        const KpArray live = model.forward(truth);

        InferenceConfig cfg;
        cfg.gn_steps = 5;
        AInfLaplacePoseEstimator est(model, cfg);
        InferenceResult res;
        for (int it = 0; it < 20; ++it) res = est.infer(live);  // let belief settle

        check(std::isfinite(res.mean_l2) && res.mean_l2 < 0.05f,
              "estimator fits a noise-free synthetic pose (mean_L2 < 5cm)");
        check(std::isfinite(res.uncertainty_trace),
              "uncertainty_trace is finite after convergence");
    }

    std::printf("\n%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
