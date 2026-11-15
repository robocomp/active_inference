// lidar_golden_test — a 40-frame GOLDEN fit trajectory, recorded at the lidar-measures plan's BASE commit before any edit
// to lidar_ray_factor.h / lidar_select.h. Its purpose is to prove that adding per-ray weights (empty ⇒ 1) leaves the
// numbers BIT-IDENTICAL. Re-run by every later task of that plan; it must keep passing unchanged.
// A nightstand-like belief seeded off-truth by (+0.05, −0.04, H +0.03) runs 40 deterministic frames (closed-form point
// grids + closed-form Helios-like rays, no <random>). Frame k: k%5==4 ⇒ predict(); else k%3==0 ⇒ update with rays;
// else update with points only. After each frame: state vec (6), covariance diagonal (6), log_volatility (6).
// BIT-EXACT FOR THIS COMPILER AT -O2 ONLY (the tables are hex floats): a different compiler/flags may legitimately
// differ in the last ulp; re-record then, at a commit where the change under test is not yet applied.
// Run: ./lidar_golden_test          Re-record: ./lidar_golden_test --print > rows.inc
#include <cmath>
#include <cstdio>
#include <cstring>
#include "countable_belief.h"

using namespace rc;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static constexpr int kFrames = 40, kCols = 18;

static void run(float out[kFrames][kCols])
{
    CountableBeliefParams p;
    p.prior_footprint_m = 0.51f; p.prior_footprint_std = 0.06f; p.prior_height_m = 0.62f; p.prior_height_std = 0.10f;
    p.plaus_height_min = 0.40f; p.plaus_height_soft = 0.08f;
    p.wall_flush_prior = 0.90f;
    p.room_axis_std    = 3.0f * 0.017453293f;
    p.volatility_infer = true;                                   // exercise log_volatility as well
    // Truth: 0.51 x 0.51 x 0.62, centre (0.30, −0.255), back on the wall y = 0, yaw 0.
    const float tx = 0.30f, ty = -0.255f, tH = 0.62f, tw = 0.51f;
    CountableBeliefState s0{tx + 0.05f, ty - 0.04f, tH + 0.03f, tw, tw, 0.0f};
    CountableBelief b(s0, p);
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));

    CountableFrame pts;                                           // 200 points on the two visible faces
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 10; ++j)     // front y = −0.51
        pts.points.emplace_back(tx - 0.5f * tw + tw * (i + 0.5f) / 12.0f, ty - 0.5f * tw, tH * (j + 0.5f) / 10.0f);
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 10; ++j)      // left x = tx − tw/2
        pts.points.emplace_back(tx - 0.5f * tw, ty - 0.5f * tw + tw * (i + 0.5f) / 8.0f, tH * (j + 0.5f) / 10.0f);
    pts.R.assign(pts.points.size(), p.sigma_base_m * p.sigma_base_m);
    pts.z_top_obs = tH;
    pts.wall.ok = true; pts.wall.p = Eigen::Vector2f(0, 0); pts.wall.n = Eigen::Vector2f(0, -1);
    pts.wall.sigma_m = 0.02f; pts.wall.length_m = 4.0f;
    pts.room_axis_ok = true; pts.room_axis = 0.0f;

    rc::ai::LidarRays rays;                                       // 120 first hits on the truth front face
    rays.origin = Eigen::Vector3f(tx, ty - 0.5f * tw - 1.5f, 1.1075f);
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 10; ++j)
        rays.endpoints.emplace_back(tx - 0.5f * tw + tw * (i + 0.5f) / 12.0f, ty - 0.5f * tw, tH * (j + 0.5f) / 10.0f);
    rays.precision = 2500.0f; rays.robust_c_m = 0.05f;

    for (int k = 0; k < kFrames; ++k)
    {
        if (k % 5 == 4) b.predict();
        else { CountableFrame f = pts; if (k % 3 == 0) f.lidar = rays; b.update(f); }
        const auto v = b.state().vec(); const auto d = b.covariance().diagonal(); const auto w = b.log_volatility();
        for (int i = 0; i < 6; ++i) { out[k][i] = v(i); out[k][6 + i] = d(i); out[k][12 + i] = w(i); }
    }
}

#include "lidar_golden_rows.inc"

int main(int argc, char** argv)
{
    static float got[kFrames][kCols];
    run(got);
    if (argc > 1 and std::strcmp(argv[1], "--print") == 0)
    {
        std::printf("static const float kGolden[kFrames][kCols] = {\n");
        for (int k = 0; k < kFrames; ++k)
        {
            std::printf("  {");
            for (int c = 0; c < kCols; ++c) std::printf("%a%s", static_cast<double>(got[k][c]), c + 1 < kCols ? "f, " : "f");
            std::printf("},\n");
        }
        std::printf("};\n");
        return 0;
    }
    int nbad = 0;
    for (int k = 0; k < kFrames; ++k)
        if (std::memcmp(got[k], kGolden[k], sizeof(float) * kCols) != 0)
        {
            if (++nbad <= 3)
                for (int c = 0; c < kCols; ++c)
                    if (std::memcmp(&got[k][c], &kGolden[k][c], sizeof(float)) != 0)
                    { std::printf("frame %d col %d: got %a want %a\n", k, c, double(got[k][c]), double(kGolden[k][c])); break; }
        }
    CHECK(nbad == 0);
    std::printf("lidar_golden_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
