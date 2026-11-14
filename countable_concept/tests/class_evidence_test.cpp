// class_evidence_test — the shape log-evidence ratio and the birth height density (pure math).
#include <clocale>
#include <cstdio>
#include <random>
#include "countable_belief.h"

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

    // per-class box-fit error (H, w, h) in metres, owner option A 11-14
    ClassShapePrior bedF = bed, wcF = wc;
    bedF.H_fit_std = 0.10f; bedF.w_fit_std = 0.05f; bedF.h_fit_std = 0.05f;
    wcF.H_fit_std = 0.05f;  wcF.w_fit_std = 0.26f;  wcF.h_fit_std = 0.15f;
    const float l6 = class_log_evidence(data(1.86f, 1.98f, 0.28f), bed, alt);
    const float l7 = class_log_evidence(data(0.65f, 1.75f, 2.10f), bed, alt);
    const float lt = class_log_evidence(data(0.88f, 0.34f, 0.86f), wc, alt);
    std::printf("Lambda bed6=%.3f bed7=%.3f toilet=%.3f  P: %.4f %.4f %.4f\n", l6, l7, lt, P(bed, l6), P(bed, l7), P(wc, lt));
    check("bed6_rejected", P(bed, l6) < 0.05f);
    check("bed7_accepted", P(bed, l7) > 0.9f);
    check("nightstand_accepted", P(ns, class_log_evidence(data(0.62f, 0.51f, 0.51f), ns, alt)) > 0.9f);
    check("toilet_accepted", P(wc, lt) > 0.5f);
    {
        const float a = class_log_evidence(data(0.87f, 0.30f, 0.89f), wcF, alt), b = class_log_evidence(data(0.88f, 0.25f, 0.89f), wcF, alt);
        const float a0 = class_log_evidence(data(0.87f, 0.30f, 0.89f), wc, alt);
        std::printf("toilet misfit: P(0.87,0.30,0.89)=%.4f (no fit std %.4f)  P(0.88,0.25,0.89)=%.4f\n", P(wcF, a), P(wc, a0), P(wcF, b));
        check("toilet_misfit_kept", P(wcF, a) > 0.8f and P(wcF, b) > 0.8f);
        const float b6 = class_log_evidence(data(1.86f, 1.98f, 0.28f), bedF, alt);
        std::printf("bed6 with fit std: P=%.5f\n", P(bedF, b6));
        check("bed6_still_rejected", P(bedF, b6) < 0.05f);
        // all fit stds 0 == the old class side (explicit zeros, same numbers as l6/l7/lt)
        ClassShapePrior z = bed; z.H_fit_std = z.w_fit_std = z.h_fit_std = 0.0f;
        check("fit_std_zero_is_old", std::fabs(class_log_evidence(data(0.65f, 1.75f, 2.10f), z, alt) - l7) < 1e-5f
              and std::fabs(class_log_evidence(data(1.86f, 1.98f, 0.28f), z, alt) - l6) < 1e-5f);
        check("swap_invariant_with_fit_std", std::fabs(class_log_evidence(data(0.65f, 2.10f, 1.75f), bedF, alt)
                                                     - class_log_evidence(data(0.65f, 1.75f, 2.10f), bedF, alt)) < 1e-4f);
        check("fit_std_swap_invariant_toilet", std::fabs(class_log_evidence(data(0.87f, 0.89f, 0.30f), wcF, alt) - a) < 1e-4f);
    }
    SizeGaussian none; none.mean = Eigen::Vector3f(0.65f, 1.75f, 2.10f);
    check("no_data_is_zero", class_log_evidence(none, bed, alt) == 0.0f);
    SizeGaussian wide = data(0.65f, 1.75f, 2.10f, 5.0f);
    check("wide_data_tends_to_zero", std::fabs(class_log_evidence(wide, bed, alt)) < 0.5f);
    check("swap_invariant", std::fabs(class_log_evidence(data(0.65f, 2.10f, 1.75f), bed, alt) - l7) < 1e-4f);

    auto B = [&](const ClassShapePrior& c, float x) { return birth_height_log_evidence(x, c, sz, eps, alt); };
    std::printf("birth: mattress %.3f headboard %.3f blackboard %.3f\n", P(bed, B(bed, 0.65f)), P(bed, B(bed, 1.25f)), P(bed, B(bed, 1.90f)));
    check("birth_mattress_ok", P(bed, B(bed, 0.65f)) > 0.85f);
    check("birth_headboard_ok", P(bed, B(bed, 1.25f)) > 0.85f);
    check("birth_blackboard_low", P(bed, B(bed, 1.90f)) < 0.3f);
    check("birth_bleed_is_slowed_not_stopped", P(bed, B(bed, 1.90f)) > 0.1f);
    check("birth_nightstand_tall_low", B(ns, 1.90f) < B(ns, 0.62f));

    std::vector<Eigen::Vector3f> pts;
    for (int i = 0; i < 1000; ++i) pts.emplace_back(0.f, 0.f, float(i) / 999.0f);
    for (int i = 0; i < 5; ++i) pts.emplace_back(0.f, 0.f, 3.0f);
    check("robust_extent", std::fabs(robust_z_extent(pts) - 0.94f) < 0.02f);
    // candidate_plausibility: the birth score IS the class probability (bed parameters).
    rc::CountableBeliefParams bp;
    bp.prior_height_m = 0.65f; bp.prior_height_std = 0.10f;
    bp.prior_footprint_m = 1.75f; bp.prior_footprint_std = 0.20f; bp.prior_depth_m = 2.10f;
    bp.headboard_h_m = 1.25f;
    bp.class_p_label_true = 0.8f; bp.class_alt_std_m = 1.0f; bp.birth_height_noise_m = 0.05f; bp.birth_height_outlier_frac = 0.05f;
    std::mt19937 rng(7); std::uniform_real_distribution<float> U(0, 1);
    auto cloud = [&](int n, float zmax) { std::vector<Eigen::Vector3f> v; for (int i = 0; i < n; ++i) v.emplace_back(U(rng), U(rng), zmax * U(rng)); return v; };
    check("candidate_sparse_is_prior", rc::CountableBelief::candidate_plausibility(cloud(8, 1.9f), bp) == 0.8f);
    const float pb = rc::CountableBelief::candidate_plausibility(cloud(400, 1.9f), bp);
    // A mattress seen from above-front: its top face (blanket, z = 0.72 per Bed.proto) is a large share of the points, its side runs floor..top, with floor-contact points (robust extent = 0.72).
    auto mattress = [&]() { auto v = cloud(240, 0.72f); for (int i = 0; i < 160; ++i) v.emplace_back(U(rng), U(rng), 0.72f); for (int i = 0; i < 30; ++i) v.emplace_back(U(rng), U(rng), 0.0f); return v; };
    const float pm = rc::CountableBelief::candidate_plausibility(mattress(), bp);
    const float pu = rc::CountableBelief::candidate_plausibility(cloud(400, 0.65f), bp);   // side face only: q97-q03 = 0.94 x 0.65
    std::printf("candidate: blackboard(1.9 m) %.4f  mattress(extent 0.72) %.4f  mattress(side only, extent 0.61) %.4f\n", pb, pm, pu);
    // a partly seen mattress: floor-contact points absent, robust extent ~0.61
    auto partial = [&]() { std::vector<Eigen::Vector3f> v; for (int i = 0; i < 400; ++i) v.emplace_back(U(rng), U(rng), 0.61f * U(rng)); v.emplace_back(0, 0, 0.0f); v.emplace_back(0, 0, 0.61f); return v; };
    const float pp = rc::CountableBelief::candidate_plausibility(partial(), bp);
    std::printf("candidate: partial mattress (extent %.3f) %.4f\n", rc::countable::robust_z_extent(partial()), pp);
    check("candidate_partial_mattress", pp > 0.8f);
    // NB the exact-0.65 extent sits ON the edge of the class height range: P = 0.891 (see birth_mattress_ok), not > 0.9.
    check("candidate_blackboard_cloud", pb < 0.3f and pm > 0.9f);
    const auto sp = bp.class_shape_prior();
    check("shape_prior_top_is_headboard", sp.H_top == 1.25f and sp.H_mean == 0.65f and sp.w_mean == 1.75f and sp.h_mean == 2.10f);
    bp.headboard_h_m = -1.0f;
    check("shape_prior_top_without_headboard", bp.class_shape_prior().H_top == 0.65f);
    return bad ? 1 : 0;
}
