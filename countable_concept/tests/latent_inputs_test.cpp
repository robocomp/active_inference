// latent_inputs_test — the flag-on fitter's persistent latent inputs (rc::persistent_latent_inputs, the computation
// CountableFitter's latent_inputs() calls; shared-engine Task 5 ruling). At two ranges × two obliquities, with a pose-chain
// variance and ego-motion present:
//   - sig2_cam = the measured ZED YOLO-seg boundary std² on cx, cy, H, w, h and the configured yaw² (F1.6); sig2_L = the
//     common-mode stds squared (cx, cy: pos²; H, w, h: size²; yaw: yaw²); both exactly, at every view: no range growth, no
//     obliquity term;
//   - the chain variance does NOT enter them: its share is the latent e (Σ_pose, staged by stage_persistent), and adding it
//     here too would count the robot's pose error twice;
//   - fresh = the ego-motion variances, per DOF.
// Run: ./latent_inputs_test
#include <cstdio>
#include "countable_latent_inputs.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

int main()
{
    const rc::CommonModeStds cm{0.03f, 0.02f, 0.03f};   // the bed / nightstand / toilet AI2CommonMode* values
    const float p2 = cm.pos * cm.pos, s2 = cm.size * cm.size, y2 = cm.yaw * cm.yaw;
    for (const float range : {1.0f, 6.0f})
        for (const float oblq : {1.0f, 0.1f})
        {
            rc::LatentView v;
            v.range_m = range; v.obliquity_cos = oblq;
            v.chain_var_xx = 4e-4f; v.chain_var_yy = 9e-4f;
            v.mot_pos_var = 1e-4f; v.mot_size_var = 2e-4f; v.mot_yaw_var = 3e-4f;
            const rc::LatentInputs li = rc::persistent_latent_inputs(cm, v);
            const float b2 = rc::kZedYoloBoundaryStdM * rc::kZedYoloBoundaryStdM;
            const std::array<float, 6> want_cam = {b2, b2, b2, b2, b2, y2}, want_L = {p2, p2, s2, s2, s2, y2};
            CHECK(li.sig2_cam == want_cam);
            CHECK(li.sig2_L == want_L);
            const std::array<float, 6> fresh = {1e-4f, 1e-4f, 2e-4f, 2e-4f, 2e-4f, 3e-4f};
            CHECK(li.fresh == fresh);
            std::printf("range %.0f m, |cos| %.1f: sig2_cam = (%.6g, %.6g, %.6g, %.6g, %.6g, %.6g)\n", range, oblq,
                        li.sig2_cam[0], li.sig2_cam[1], li.sig2_cam[2], li.sig2_cam[3], li.sig2_cam[4], li.sig2_cam[5]);
        }
    std::printf(bad ? "latent_inputs_test FAILED (%d)\n" : "latent_inputs_test OK\n", bad);
    return bad ? 1 : 0;
}
