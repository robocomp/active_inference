/*
 * place_map_test.cpp — the algebra and the file format of the panoramic place map. Standalone:
 *
 *     g++ -std=c++23 -O1 -I/usr/include/eigen3 place_map_test.cpp -o place_map_test && ./place_map_test
 *
 * ★ THIS IS THE FIRST TEST UNDER common/ THAT PARSES A FILE, so it must call setlocale(LC_ALL, "")
 * itself. A harness has no Qt, so it never leaves the "C" locale, and a save/load round trip would
 * pass here while failing in the agent -- which runs under es_ES.UTF-8, where the C library reads
 * "0.5" as 0 and stops. Testing the round trip in "C" answers a different question than the one that
 * matters. (run_tests.sh carries the same warning.)
 *
 * What is actually load-bearing here, and invisible in a passing run:
 *   - the SE(2) covariance transport, because the RT edge stores the INVERTED covariance and getting
 *     this wrong is silent until a calibration test months later;
 *   - the circular-shift SIGN, because the wrong one is a reflection, not a rotation;
 *   - the angular wrap in the mixture, because a linear residual near +-pi reads as a 2pi error;
 *   - and the loud refusal on a header mismatch, because the failure it prevents looks like "the
 *     model got worse" rather than "you loaded the wrong map".
 */

#include "place_map.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>

using namespace rc::place;

static int g_fail = 0;
static void check(bool ok, const char* what)
{
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
    if (not ok) ++g_fail;
}
static bool close_f(float a, float b, float tol = 1e-4f) { return std::fabs(a - b) <= tol; }

// ── helpers ─────────────────────────────────────────────────────────────────────────────────────
static Header make_header(std::uint32_t dim = 8, std::uint32_t S = 16)
{
    Header h; h.dim = dim; h.n_sectors = S;
    h.model_name = "test-model"; h.room_name = "room"; h.azimuth_tune_deg = 0.f;
    return h;
}

/// A synthetic keyframe descriptor: each sector gets a distinct, L2-normalised random vector, so a
/// cyclic permutation is unambiguously recoverable. CLS is the mean direction.
static std::vector<float> make_desc(const Header& h, std::mt19937& rng)
{
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> d(h.stride(), 0.f);
    for (std::uint32_t s = 0; s < h.n_sectors; ++s)
    {
        float* v = d.data() + std::size_t(1 + s) * h.dim;
        float n = 0.f;
        for (std::uint32_t c = 0; c < h.dim; ++c) { v[c] = g(rng); n += v[c] * v[c]; }
        n = std::sqrt(n);
        for (std::uint32_t c = 0; c < h.dim; ++c) v[c] /= n;
    }
    float n = 0.f;
    for (std::uint32_t s = 0; s < h.n_sectors; ++s)
        for (std::uint32_t c = 0; c < h.dim; ++c) d[c] += d[std::size_t(1 + s) * h.dim + c];
    for (std::uint32_t c = 0; c < h.dim; ++c) n += d[c] * d[c];
    n = std::sqrt(std::max(n, 1e-12f));
    for (std::uint32_t c = 0; c < h.dim; ++c) d[c] /= n;
    return d;
}

/// Rotate a descriptor's sector array by k, mirroring what a yaw change does to the panorama.
static std::vector<float> roll_desc(const std::vector<float>& d, const Header& h, int k)
{
    std::vector<float> o(d.size());
    std::copy_n(d.begin(), h.dim, o.begin());                       // CLS is yaw-invariant
    const int S = int(h.n_sectors);
    for (int s = 0; s < S; ++s)
        std::copy_n(d.begin() + std::size_t(1 + s) * h.dim, h.dim,
                    o.begin() + std::size_t(1 + (s + k) % S) * h.dim);
    return o;
}

// ── 1. SE(2) covariance transport ───────────────────────────────────────────────────────────────
static void test_cov_transport()
{
    std::puts("SE(2) covariance transport (the RT edge stores the INVERTED covariance)");
    Eigen::Vector3f p(1.3f, -0.7f, 0.9f);
    Eigen::Matrix3f C;
    C <<  4e-4f, 1e-5f, 2e-6f,
          1e-5f, 9e-4f, 3e-6f,
          2e-6f, 3e-6f, 1e-3f;

    // Transport there, then back at the INVERSE pose: J(p)^-1 == J(inv(p)) because inversion is an
    // involution. If someone "simplifies" this to J(p) twice, this test is what stops them.
    const Eigen::Matrix3f fwd  = invert_se2_cov(C, p);
    const Eigen::Matrix3f back = invert_se2_cov(fwd, inv_se2(p));
    check((back - C).cwiseAbs().maxCoeff() < 1e-9f, "round trip J(inv(p)) . J(p) == I");
    check((fwd - C).cwiseAbs().maxCoeff() > 1e-6f,  "transport is NOT a no-op (yaw leaks into xy)");
    check((fwd - fwd.transpose()).cwiseAbs().maxCoeff() < 1e-12f, "stays symmetric");

    // Applying J(p) twice must NOT be the identity -- that is the mistake the round trip guards.
    const Eigen::Matrix3f wrong = invert_se2_cov(fwd, p);
    check((wrong - C).cwiseAbs().maxCoeff() > 1e-9f, "J(p) applied twice is NOT the identity");

    // At the origin with zero yaw the map is a pure reflection, so variances survive unchanged.
    const Eigen::Matrix3f at0 = invert_se2_cov(C, Eigen::Vector3f(0.f, 0.f, 0.f));
    check(close_f(at0(0,0), C(0,0), 1e-9f) and close_f(at0(2,2), C(2,2), 1e-9f),
          "identity pose leaves the diagonal untouched");
}

// ── 2. circular shift recovery ──────────────────────────────────────────────────────────────────
static void test_circular_shift()
{
    std::puts("circular cross-correlation recovers the sector shift");
    const Header h = make_header();
    std::mt19937 rng(7);
    PlaceMap m(h);
    Keyframe k; k.id = 1; k.robot_pose = { 2.f, 3.f, 0.2f }; k.ricoh_pose = k.robot_pose;
    const auto base = make_desc(h, rng);
    m.add(k, base);
    // A few distractors so the prefilter has something to reject.
    for (int i = 0; i < 20; ++i)
    {
        Keyframe d; d.id = std::uint32_t(100 + i);
        d.robot_pose = { float(i), -float(i), 0.f }; d.ricoh_pose = d.robot_pose;
        m.add(d, make_desc(h, rng));
    }

    int nok = 0;
    for (int s = 0; s < int(h.n_sectors); ++s)
    {
        const auto q = roll_desc(base, h, s);
        const auto r = m.match(q, 1);
        if (r.size() == 1 and r[0].kf == 0 and r[0].shift == s) ++nok;
    }
    check(nok == int(h.n_sectors), "all 16 shifts recovered exactly, correct keyframe");

    // Sub-sector interpolation must stay inside the sector it refines.
    const auto r = m.match(roll_desc(base, h, 5), 1);
    check(not r.empty() and std::fabs(r[0].shift_interp - 5.f) <= 0.5f,
          "parabolic refinement stays within +-0.5 sector");
    check(not r.empty() and r[0].margin > 0.f, "peak margin is positive and reported");
}

// ── 3. the mixture ──────────────────────────────────────────────────────────────────────────────
static void test_mixture()
{
    std::puts("pose mixture: normalisation, angular wrap, bimodal spread");
    const Header h = make_header();
    std::mt19937 rng(11);
    PlaceMap m(h);
    // Two keyframes at OPPOSITE ends of a room with near-identical appearance: the aliasing case
    // this whole design exists to answer honestly.
    const auto d = make_desc(h, rng);
    Keyframe a; a.id = 1; a.robot_pose = { -3.f, 0.f, 0.f }; a.ricoh_pose = a.robot_pose;
    a.cov = Eigen::Vector3f(1e-4f, 1e-4f, 1e-4f).asDiagonal();
    Keyframe b = a; b.id = 2; b.robot_pose = { 3.f, 0.f, 0.f }; b.ricoh_pose = b.robot_pose;
    m.add(a, d); m.add(b, d);

    MixtureParams p;
    const auto ms = m.match(d, 2);
    check(ms.size() == 2, "both aliased keyframes retrieved");
    const auto mix = m.to_mixture(ms, p);
    check(mix.size() == 2, "mixture has both components");
    float wsum = 0.f; for (float w : mix.w) wsum += w;
    check(close_f(wsum, 1.f), "weights sum to 1");

    // ★ The honest answer to an ambiguous place is a WIDE mixture, not its midpoint.
    const Eigen::Matrix3f S = mix.moment_match();
    check(S(0, 0) > 4.0f, "bimodal aliasing yields LARGE x variance (not a confident midpoint)");
    check(mix.logpdf(a.robot_pose) > mix.logpdf(Eigen::Vector3f(0.f, 0.f, 0.f)),
          "a true mode scores higher than the midpoint between modes");

    // Angular wrap: a keyframe at +179 deg and a query at -179 deg are 2 deg apart, not 358.
    PlaceMap w(h);
    Keyframe c; c.id = 3; c.robot_pose = { 0.f, 0.f, 3.1241f }; c.ricoh_pose = c.robot_pose;
    c.cov = Eigen::Vector3f(1e-4f, 1e-4f, 1e-4f).asDiagonal();
    w.add(c, d);
    MixtureParams p2 = p; p2.yaw_interp_resid_rad = 0.02f;
    const auto wm = w.to_mixture(w.match(d, 1), p2);
    check(wm.logpdf(Eigen::Vector3f(0.f, 0.f, -3.1241f))
        > wm.logpdf(Eigen::Vector3f(0.f, 0.f,  0.0f)),
          "residual across the +-pi seam is angular, not linear");

    const Eigen::Matrix3f Sw = wm.moment_match();
    check(Sw(2, 2) <= float(M_PI) * float(M_PI) / 3.0f + 1e-6f,
          "yaw variance capped at the uniform-circle variance");

    std::mt19937 srng(3);
    const auto smp = mix.sample(srng, 200);
    check(smp.size() == 200, "sample() returns the requested count");
    int left = 0; for (const auto& s : smp) if (s.x() < 0.f) ++left;
    check(left > 40 and left < 160, "samples are spread across BOTH modes, not just the argmax");
}

// ── 4. rho(): similarity -> spread, continuous and bounded ──────────────────────────────────────
static void test_rho()
{
    std::puts("rho(): the covariance-not-a-switch mapping");
    MixtureParams p;
    const float hi = p.rho(p.decay_a + p.decay_b);      // at the fitted peak
    const float mid = p.rho(p.decay_b + 0.5f * p.decay_a);
    const float lo = p.rho(p.decay_b - 0.1f);           // at/below the floor: no information
    check(hi < mid and mid < lo, "spread grows monotonically as similarity falls");
    check(close_f(lo, p.max_pos_sigma_m), "at the noise floor it saturates at max_pos_sigma_m");
    check(hi > 0.f and std::isfinite(hi), "peak similarity gives a finite, positive spread");
}

// ── 5. insertion policy ─────────────────────────────────────────────────────────────────────────
static void test_insertion()
{
    std::puts("insertion policy: spatial density, and deliberately yaw-blind");
    const Header h = make_header();
    std::mt19937 rng(5);
    PlaceMap m(h);
    Keyframe k; k.id = 1; k.ricoh_pose = { 0.f, 0.f, 0.f }; k.robot_pose = k.ricoh_pose;
    m.add(k, make_desc(h, rng));
    check(not m.should_insert({ 0.2f, 0.f, 0.f }, 0.5f), "a pose within d_m is redundant");
    check(m.should_insert({ 0.8f, 0.f, 0.f }, 0.5f), "a pose beyond d_m is inserted");
    // ★ Same position, opposite heading: NOT a new keyframe. One keyframe covers every yaw, because
    // a yaw-displaced view is a cyclic permutation of descriptors already stored. A yaw clause here
    // would put several components at one place and over-weight it in the softmax.
    check(not m.should_insert({ 0.f, 0.f, 3.14f }, 0.5f), "same place, opposite yaw is NOT inserted");
}

// ── 6. persistence, under the AGENT's locale ────────────────────────────────────────────────────
static void test_persistence()
{
    std::puts("save/load round trip (locale-hostile on purpose)");
    const char* loc = std::setlocale(LC_ALL, "es_ES.UTF-8");
    if (not loc) loc = std::setlocale(LC_ALL, "");
    std::printf("  [locale] LC_NUMERIC = %s  (decimal comma is the hazard)\n",
                std::setlocale(LC_NUMERIC, nullptr));

    const auto dir = std::filesystem::temp_directory_path() / "place_map_test";
    std::filesystem::create_directories(dir);
    const auto idx = (dir / "map.csv").string(), blob = (dir / "map.bin").string();

    Header h = make_header(8, 16);
    h.pool_p = 3.0f; h.band_lo = 6; h.band_hi = 12; h.azimuth_tune_deg = 1.25f;
    std::mt19937 rng(13);
    PlaceMap m(h);
    for (int i = 0; i < 5; ++i)
    {
        Keyframe k; k.id = std::uint32_t(i); k.stamp_ms = 1700000000000ull + std::uint64_t(i);
        k.ricoh_pose = { 0.260417f * float(i), -9.2e-3f * float(i), 0.5f };
        k.robot_pose = { 0.260417f * float(i) + 0.01f, -9.2e-3f * float(i), 0.5f };
        k.cov << 4e-4f, 1e-5f, 2e-6f,  1e-5f, 9e-4f, 3e-6f,  2e-6f, 3e-6f, 1e-3f;
        m.add(k, make_desc(h, rng));
    }
    check(m.save(idx, blob), "save() succeeds");

    PlaceMap r;
    std::string why;
    check(r.load(idx, blob, &h, &why), ("load() succeeds" + (why.empty() ? "" : " [" + why + "]")).c_str());
    check(r.size() == m.size(), "keyframe count round-trips");
    bool poses_ok = true, cov_ok = true, desc_ok = true;
    for (std::size_t i = 0; i < r.size(); ++i)
    {
        // ★ 0.260417 is the exact value that became 0 in the 2026-08-03 depth-dataset corruption.
        poses_ok &= (r.keyframes()[i].ricoh_pose - m.keyframes()[i].ricoh_pose).cwiseAbs().maxCoeff() == 0.f
                 and (r.keyframes()[i].robot_pose - m.keyframes()[i].robot_pose).cwiseAbs().maxCoeff() == 0.f;
        cov_ok   &= (r.keyframes()[i].cov - m.keyframes()[i].cov).cwiseAbs().maxCoeff() < 1e-9f;
        const auto a = r.descriptor(i), b = m.descriptor(i);
        for (std::size_t c = 0; c < a.size(); ++c) desc_ok &= (a[c] == b[c]);
    }
    check(poses_ok, "poses round-trip EXACTLY (comma locale + 6-sig-digit default)");
    check(cov_ok, "covariance survives, and stays symmetric");
    check(desc_ok, "descriptors are bit-identical (binary blob, no decimal round-trip)");
    check(r.header().band_lo == 6 and r.header().band_hi == 12, "elevation band round-trips");
    check(close_f(r.header().azimuth_tune_deg, 1.25f), "azimuth_tune_deg round-trips");
    check(r.header().model_name == "test-model", "model name round-trips");

    // Retrieval must behave identically through the file.
    const auto q = roll_desc(std::vector<float>(m.descriptor(2).begin(), m.descriptor(2).end()), h, 4);
    const auto rr = r.match(q, 1);
    check(rr.size() == 1 and rr[0].kf == 2 and rr[0].shift == 4,
          "a loaded map retrieves exactly as the in-memory one did");

    // ★ Loud, specific refusal -- the whole point of storing the header.
    Header bad = h; bad.n_sectors = 8;
    PlaceMap r2; std::string why2;
    check(not r2.load(idx, blob, &bad, &why2) and why2.find("n_sectors") != std::string::npos,
          "sector-count mismatch refuses and NAMES the field");
    bad = h; bad.band_lo = 0; why2.clear();
    check(not r2.load(idx, blob, &bad, &why2) and why2.find("band") != std::string::npos,
          "elevation-band mismatch refuses and names the field");
    bad = h; bad.azimuth_tune_deg = 0.f; why2.clear();
    check(not r2.load(idx, blob, &bad, &why2) and why2.find("azimuth") != std::string::npos,
          "ricoh azimuth-calibration mismatch refuses (a silently ROTATED map)");
    bad = h; bad.room_name = "room_2"; why2.clear();
    check(not r2.load(idx, blob, &bad, &why2) and why2.find("room") != std::string::npos,
          "room-identity mismatch refuses (poses in an obsolete frame)");

    std::filesystem::remove_all(dir);
}

int main()
{
    test_cov_transport();
    test_circular_shift();
    test_mixture();
    test_rho();
    test_insertion();
    test_persistence();
    std::printf("\nplace_map_test: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
