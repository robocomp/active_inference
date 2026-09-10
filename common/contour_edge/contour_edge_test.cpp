/*
 * contour_edge_test.cpp — the classifier-free existence channel's four cases, locked in.
 *
 * The properties under test are the ones the channel's WHOLE VALUE rests on, and three of the four were
 * wrong in a first draft or in the live agent:
 *
 *   1. it can CONFIRM   — a surface where the belief predicts one, with space behind its edge
 *   2. it can REFUTE    — nothing at the predicted depth, i.e. we are looking straight through it.
 *                         A control-relative depth statistic scored this 0.000: with the object gone the
 *                         believed contour and every control alike find nothing, and the difference of
 *                         two silences is silence. A channel that can only confirm is a ratchet.
 *   3. it ABSTAINS on a phantom painted on a wall AT the predicted distance — perfect depth agreement,
 *                         no space behind the boundary. This is the case the RGB half cannot tell from a
 *                         real object at all, and it is why the depth half exists.
 *   4. NOT MEASURED ≠ REFUTED — no depth reads, or no surviving control, must leave the belief alone.
 *                         Conflating the two deleted a live door on 2026-09-09.
 *
 * Single translation unit (the runner builds one .cpp per test), so the implementations are #included.
 */

#include "contour_edge_project.cpp"
#include "contour_depth_check.cpp"
#include "../existence_belief/existence_belief.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <limits>

static int failures = 0;

static void check(bool ok, const char* what)
{
    if (not ok) { std::printf("  FAIL: %s\n", what); ++failures; }
}

using namespace rc::edges;

// A pinhole in this codebase's ZED convention: x right, y FORWARD (the quantity the depth plane stores),
// z up. Camera at the world origin looking down +y.
static PointProjector make_proj()
{
    return [](const Eigen::Vector3d& Pw) -> std::optional<ProjectedVertex>
    {
        if (Pw.y() <= 0.20) return std::nullopt;
        ProjectedVertex v;
        v.px = cv::Point2f(static_cast<float>(640.0 + Pw.x() * 600.0 / Pw.y()),
                           static_cast<float>(360.0 - Pw.z() * 600.0 / Pw.y()));
        v.depth_m = static_cast<float>(Pw.y());
        return v;
    };
}

int main()
{
    std::printf("contour_edge_test\n");

    BoxFootprint box;
    box.cx = 0.0f; box.cy = 4.0f; box.yaw = 0.0f;
    box.w = 1.0f; box.d = 0.6f; box.z_min = 0.0f; box.z_max = 2.0f;
    const auto set = project_box_face(box, Eigen::Vector3d(0, 0, 1.0), make_proj());

    check(set.face.valid(),            "a box in front of the camera projects a valid face");
    check(set.face.px.size() == 4,     "the face is a quad");
    check(set.controls.size() == 4,    "four controls survive in an unobstructed view");
    check(set.n_faces_visible == 1,    "exactly one face of an axis-aligned box faces the camera head-on");
    // Every vertex carries the near face's depth, and it is the FORWARD coordinate, not the norm: the
    // corners are off-axis, so a norm would read strictly greater than 3.70 on all four.
    const float near_face = box.cy - 0.5f * box.d;
    for (const float d : set.face.depth_m)
        check(std::abs(d - near_face) < 1e-3f, "vertex depth is the forward coordinate, not the range");

    const std::vector<std::vector<cv::Point>> poly{set.face.px};

    // 1. PRESENT: the object's face at its true depth, background well beyond it.
    cv::Mat present(720, 1280, CV_32F, cv::Scalar(6.0f));
    cv::fillPoly(present, poly, cv::Scalar(near_face));
    const auto d_present = contour_depth_support(present, set.face, set.controls);
    check(d_present.n_samples > 0,   "a contour in frame is measured");
    check(d_present.verdict > 0.9f,  "a surface where predicted, with space behind, CONFIRMS");
    check(std::abs(d_present.mean_bias_m) < 0.01f, "a correct belief shows no depth bias");

    // 2. ABSENT: we see the background straight through where the object should be.
    cv::Mat absent(720, 1280, CV_32F, cv::Scalar(6.0f));
    const auto d_absent = contour_depth_support(absent, set.face, set.controls);
    check(d_absent.n_samples > 0,    "an absent object is still MEASURED (that is the point)");
    check(d_absent.verdict < -0.9f,  "nothing at the predicted depth REFUTES");
    check(d_absent.mean_bias_m > 2.0f, "and the bias says how far past it we are seeing");

    // 3. ON A WALL at exactly the predicted depth: agreement is perfect, recession is absent.
    cv::Mat wall(720, 1280, CV_32F, cv::Scalar(near_face));
    const auto d_wall = contour_depth_support(wall, set.face, set.controls);
    check(std::abs(d_wall.verdict) < 0.05f, "a phantom painted on a wall at the right depth ABSTAINS");

    // 4a. NOT MEASURED: an empty depth image yields no samples, and no evidence.
    const auto d_none = contour_depth_support(cv::Mat(), set.face, set.controls);
    check(d_none.n_samples == 0,     "no depth plane ⇒ nothing measured");
    check(d_none.verdict == 0.0f,    "and nothing measured is NOT a refutation");

    // 4b. No return BEYOND the boundary (dropout, or an open doorway) must not be read as recession —
    // a channel that scores its own blind spot as confirmation is a shield, not a test.
    cv::Mat holes(720, 1280, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    cv::fillPoly(holes, poly, cv::Scalar(near_face));
    const auto d_holes = contour_depth_support(holes, set.face, set.controls);
    check(d_holes.n_no_return > 0,          "the no-return samples are counted");
    check(std::abs(d_holes.verdict) < 0.05f, "no reading beyond the edge says NOTHING, not yes");

    // The evidence conversion: symmetric, bounded by ONE confident observation, and silent when unmeasured.
    const rc::exist::SensorModel sm;
    const float llr = std::log(sm.detection_prob / sm.clutter_prob);
    const auto ev_yes  = rc::exist::contour_evidence(d_present.verdict, d_present.n_samples, sm);
    const auto ev_no   = rc::exist::contour_evidence(d_absent.verdict,  d_absent.n_samples,  sm);
    const auto ev_hold = rc::exist::contour_evidence(1.0f, 0, sm);
    check(ev_yes.log_odds_delta > 0.0f and ev_yes.e_occ > 0.0f,  "confirmation raises L");
    check(ev_no.log_odds_delta  < 0.0f and ev_no.e_free > 0.0f,  "refutation lowers L");
    check(std::abs(ev_yes.log_odds_delta + ev_no.log_odds_delta) < 0.05f,
          "the two are SYMMETRIC — a test, not a shield");
    check(std::abs(ev_yes.log_odds_delta) < llr,
          "one cycle is bounded by one confident observation's worth");
    check(ev_hold.n_reached == 0 and ev_hold.log_odds_delta == 0.0f,
          "unmeasured HOLDs, however strong the verdict handed in");
    // ★A HUGE verdict must not buy more than one observation: the tanh has to bound, not just squash.
    const auto ev_big = rc::exist::contour_evidence(50.0f, 100, sm);
    check(std::abs(ev_big.log_odds_delta) <= llr + 1e-5f, "a large verdict still cannot exceed the cap");

    // A box BEHIND the camera projects nothing at all — not a small contour, none.
    BoxFootprint behind = box; behind.cy = -4.0f;
    const auto set_behind = project_box_face(behind, Eigen::Vector3d(0, 0, 1.0), make_proj());
    check(not set_behind.face.valid(), "a box behind the camera yields no contour");

    if (failures == 0) std::printf("  contour_edge: all checks passed\n");
    else               std::printf("  contour_edge: %d CHECK(S) FAILED\n", failures);
    return failures > 0;
}
