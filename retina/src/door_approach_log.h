/*
 * door_approach_log.h — watch ONE class's evidence evolve as the robot drives toward it.
 *
 * WHY THIS EXISTS, and why it is not another column on detect_drops.csv.
 *
 * The question is not "did the detector fire on this frame" — detect_drops.csv already answers that,
 * per frame, in isolation. The question is what the evidence DOES along an approach: a door 8 m down a
 * corridor is a handful of pixels losing narrowly to `wall`, and the same door at 2 m is unmistakable.
 * Somewhere in between the argmax flips. The whole point of the graded export is that the posterior
 * should have been climbing smoothly for metres before that flip, and nothing in this fleet records the
 * two side by side against RANGE, which is the variable the flip is actually a function of.
 *
 * Three properties this file has that the existing logs do not:
 *
 *  1. ★A ROW IS WRITTEN ON FRAMES WHERE NOTHING WAS FOUND. A log that only records detections cannot
 *     show a detection appearing — it starts at the moment of success and the whole approach is
 *     missing. Worse, it is the same defect this codebase has already been bitten by twice: a latched
 *     `det_alive` flag that was 1 on 9432 of 9432 rows and made a fittable dataset look degenerate, and
 *     a counter that could not distinguish "refused" from "never asked". Every semantic frame emits a
 *     `peak` row whether or not anything survived, so absence of evidence is recorded AS a row.
 *
 *  2. ★THE INDEPENDENT VARIABLE DOES NOT DEPEND ON THE DETECTION SUCCEEDING. `anchor_range_m` is the
 *     distance from the camera pose to a ROOM-frame point the door was deprojected to while it WAS
 *     detected — so it keeps counting down through the frames where nothing is found, which are the
 *     only frames the question is about. The first version used the peak pixel's own depth, and that
 *     was wrong in a way worth remembering: on a frame with no door the peak is the max of a flat
 *     field and lands anywhere (it sat on the robot's own body at 0.27 m for 95 rows), so the axis
 *     existed only where the answer was already yes. `range_m` is kept beside it as the raw peak
 *     depth, useful only on rows where the class actually won.
 *     It is not taken from a door node's pose (which does not exist during the half of the approach
 *     that matters) nor from any belief, which would make the instrument a function of what it audits.
 *     Viewpoint (x, y, z, yaw) rides on every row: misses from one pose are not independent samples.
 *
 *  4. ★THE COVARIATES ARE MEASURED, NOT ARGUED. The first run showed the detection is stable while
 *     parked (0 dropouts in 672 frames) and flickers at ~2 Hz while moving, under motion said to be
 *     pure forward translation — where a door near the focus of expansion has almost no image-plane
 *     motion to blur. `ego_v`/`ego_w` turn "there is no rotation" from a premise into a column, and
 *     `blur` (variance of Laplacian) and `luma` test the two mechanisms that could still vary
 *     frame-to-frame under that motion: actual image sharpness, and auto-exposure hunting.
 *
 *  3. ★THE ARGMAX AND THE POSTERIOR ARE ON THE SAME ROW. `argmax_px` is what the live pipeline
 *     believes; `p_class` and `margin` are what the model actually computed. Reading them together is
 *     the entire experiment: the frame where argmax_px goes 0 → nonzero, against the range at which
 *     `margin` crossed zero, is the cost of collapsing a posterior to a label.
 *
 * ⚠SCOPE, stated so the file is not over-read: the `peak` row tracks the single strongest pixel in the
 * frame. With two doors in view it follows whichever is winning, and it can hop between them — the
 * viewpoint and pixel columns are there so that hop is visible rather than silent. It is an instrument
 * for one approach down one corridor, which is what it was asked for.
 *
 * Diagnostic only. Nothing here is published, read by any agent, or fed back into any belief.
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <Eigen/Dense>
#include <dsr/api/dsr_eigen_defs.h>

#include "rgbd_data.h"

struct SegDetection;
namespace rc::semantic { struct SemanticMap; }

namespace rc::diag
{

class DoorApproachLog
{
public:
    // `label` is the ADE20K class name to follow ("door"). Disabled until configure() is called with
    // enabled=true; a disabled instance touches no file and costs one bool test per frame.
    void configure(std::string path, std::string label, bool enabled);

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    // One call per SEMANTIC-FRESH frame, from the main thread. Writes one `peak` row always, plus one
    // `mask` row per published detection carrying the followed label.
    //   room_T_sensor — pinned to this frame's stamp by the worker; gives the viewpoint columns.
    //   masks         — what actually reached the graph this frame (post min_area, post YOLO priority),
    //                   so a class the model found and this component then dropped shows up as a peak
    //                   row with no mask row beside it.
    //   class_names   — ADE20K-150 table, to resolve the competitor's id to a name.
    // `spec_*` carry the door second opinion for this frame, THREE-VALUED on purpose (see
    // perception_stage.h): not eligible / eligible-but-skipped / ran. A denial (ran with no boxes) is
    // evidence and must be distinguishable in the file from never having been asked.
    void log(std::uint64_t stamp_ms, bool is_360,
             const Mat::RTMat& room_T_sensor,
             const RGBDData& rgbd,
             const rc::semantic::SemanticMap& map,
             const std::vector<SegDetection>& masks,
             const std::vector<std::string>& class_names,
             int spec_eligible = -1, int spec_ran = -1,
             int spec_n = -1, float spec_best_conf = -1.0f,
             const std::string& spec_best_label = {});

private:
    bool          enabled_ = false;
    std::string   path_;
    std::string   label_;
    int           class_id_ = -1;      // resolved from label_ on the first frame that carries names
    std::ofstream csv_;

    // ── ANCHORED RANGE ──────────────────────────────────────────────────────────────────────────
    // ★The peak pixel's own depth is NOT a range to the door: on a frame where nothing door-like is
    // found, the peak is the max of a nearly flat field and lands anywhere — in the first live run it
    // repeatedly landed on the robot's own body at 0.27 m, and those rows then binned as though they
    // measured a door at 27 cm. An axis that only exists when the detection succeeds cannot be used to
    // study detection failing against it.
    // So: the door is deprojected to a ROOM-frame point on frames where a mask of this class exists
    // with valid depth, and range is computed from the camera pose on EVERY frame afterwards,
    // including the silent ones. The door is static; the robot moves; the room frame is where that is
    // true. `anchor_jump_m` reports how far each refresh moved the anchor, so a re-association onto a
    // different door is visible in the file instead of quietly redefining the axis mid-approach.
    bool            anchored_ = false;
    Eigen::Vector3f anchor_room_ = Eigen::Vector3f::Zero();
    std::uint64_t   anchor_stamp_ms_ = 0;
    float           anchor_jump_m_ = 0.f;

    // Previous camera pose, to finite-difference the ego-motion covariates. ★Logged rather than
    // assumed: the motion is said to be pure translation, and `ego_w` is what turns that from a
    // premise into a measurement.
    //
    // ★PER SOURCE, and the first version was not. zed and ricoh rows interleave one-for-one, so a
    // single shared previous-pose differenced a ZED pose against a RICOH pose — two sensors 0.33 m
    // apart in z, over the few ms between them. It produced ego_v up to 12.7 m/s on a robot that does
    // 0.27, which is the only reason it was caught: the number was impossible rather than merely
    // wrong. A plausible-looking corruption here would have been read as data.
    struct PrevPose
    {
        bool            have = false;
        Eigen::Vector3f pos = Eigen::Vector3f::Zero();
        float           yaw = 0.f;
        std::uint64_t   stamp_ms = 0;
    };
    PrevPose prev_[2];   // [0] = zed, [1] = ricoh
};

}   // namespace rc::diag
