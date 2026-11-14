// class_evidence_test — the shape log-evidence ratio and the birth height density (pure math).
#include <clocale>
#include <cstdio>
#include "countable_class_evidence.h"

using namespace rc::countable;
static int bad = 0;
static void check(const char* name, bool c) { std::printf("%s %s\n", c ? "PASS" : "FAIL", name); if (not c) ++bad; }

static SizeGaussian data(float H, float w, float h, float sd = 0.02f)
{
    SizeGaussian g; g.mean = Eigen::Vector3f(H, w, h); g.info = Eigen::Matrix3f::Identity() / (sd * sd); return g;
}

int main()
{
    std::setlocale(LC_ALL, "");
    const ClassShapePrior bed{0.65f, 0.10f, 1.75f, 0.20f, 2.10f, 0.20f, 1.25f, 0.8f};
    const ClassShapePrior ns {0.62f, 0.10f, 0.51f, 0.06f, 0.51f, 0.06f, 0.62f, 0.8f};
    const ClassShapePrior wc {0.90f, 0.15f, 0.58f, 0.08f, 0.73f, 0.08f, 0.90f, 0.8f};
    const float alt = 1.0f, sz = 0.05f, eps = 0.05f;
    auto P = [](const ClassShapePrior& c, float lam) { return class_probability(logit(c.p_label_true) + lam); };

    const float l6 = class_log_evidence(data(1.86f, 1.98f, 0.28f), bed, alt);
    const float l7 = class_log_evidence(data(0.65f, 1.75f, 2.10f), bed, alt);
    const float lt = class_log_evidence(data(0.88f, 0.34f, 0.86f), wc, alt);
    std::printf("Lambda bed6=%.3f bed7=%.3f toilet=%.3f  P: %.4f %.4f %.4f\n", l6, l7, lt, P(bed, l6), P(bed, l7), P(wc, lt));
    check("bed6_rejected", P(bed, l6) < 0.05f);
    check("bed7_accepted", P(bed, l7) > 0.9f);
    check("nightstand_accepted", P(ns, class_log_evidence(data(0.62f, 0.51f, 0.51f), ns, alt)) > 0.9f);
    check("toilet_accepted", P(wc, lt) > 0.5f);
    SizeGaussian none; none.mean = Eigen::Vector3f(0.65f, 1.75f, 2.10f);
    check("no_data_is_zero", class_log_evidence(none, bed, alt) == 0.0f);
    SizeGaussian wide = data(0.65f, 1.75f, 2.10f, 5.0f);
    check("wide_data_tends_to_zero", std::fabs(class_log_evidence(wide, bed, alt)) < 0.5f);
    check("swap_invariant", std::fabs(class_log_evidence(data(0.65f, 2.10f, 1.75f), bed, alt) - l7) < 1e-4f);

    auto B = [&](const ClassShapePrior& c, float x) { return birth_height_log_evidence(x, c, sz, eps, alt); };
    std::printf("birth: mattress %.3f headboard %.3f blackboard %.3f\n", P(bed, B(bed, 0.65f)), P(bed, B(bed, 1.25f)), P(bed, B(bed, 1.90f)));
    check("birth_mattress_ok", P(bed, B(bed, 0.65f)) > 0.9f);
    check("birth_headboard_ok", P(bed, B(bed, 1.25f)) > 0.9f);
    check("birth_blackboard_low", P(bed, B(bed, 1.90f)) < 0.3f);
    check("birth_bleed_is_slowed_not_stopped", P(bed, B(bed, 1.90f)) > 0.1f);
    check("birth_nightstand_tall_low", B(ns, 1.90f) < B(ns, 0.62f));

    std::vector<Eigen::Vector3f> pts;
    for (int i = 0; i < 1000; ++i) pts.emplace_back(0.f, 0.f, float(i) / 999.0f);
    for (int i = 0; i < 5; ++i) pts.emplace_back(0.f, 0.f, 3.0f);
    check("robust_extent", std::fabs(robust_z_extent(pts) - 0.94f) < 0.02f);
    return bad ? 1 : 0;
}
