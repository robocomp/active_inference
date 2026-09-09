#include "door_approach_log.h"

#include "yolo_semantic.h"        // rc::semantic::SemanticMap
#include "yolo_seg_detector.h"    // SegDetection
#include "../../common/diag_log/rotating_csv.h"

#include <algorithm>
#include <array>
#include <format>
#include <cmath>
#include <limits>
#include <print>

namespace rc::diag
{

namespace
{

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// Depth at a frame pixel, in metres, or NaN where the sensor had nothing. ★0 is NOT a distance: the
// ZED writes 0 where stereo failed, and a 0 in a range column would read as "the door is at the
// camera" and quietly drag any fit through the origin.
float depth_at(const RGBDData& rgbd, int u, int v)
{
    if (rgbd.depth.empty() or rgbd.width <= 0 or rgbd.height <= 0)
        return kNaN;
    if (u < 0 or v < 0 or u >= rgbd.width or v >= rgbd.height)
        return kNaN;
    const float d = rgbd.depth[static_cast<std::size_t>(v) * static_cast<std::size_t>(rgbd.width)
                               + static_cast<std::size_t>(u)];
    return (std::isfinite(d) and d > 0.f) ? d : kNaN;
}

// ★ONE declaration of the schema, used to write the header AND to place every value. The alternative
// — emitting fields positionally with hand-counted commas — is a bug this codebase has already paid
// for: cabinet's ai2 header drifted out of step with its body and "every column past cy parsed under
// the wrong name". I had that bug in the first draft of this file too (the mask row was one column
// short and silently wrote its area into the margin column). A row cannot be misaligned here without
// failing to compile.
enum Col { STAMP, SOURCE, KIND, LABEL, CAM_X, CAM_Y, CAM_Z, CAM_YAW,
           U, V, RANGE_M, P_CLASS, P_TOP, TOP_LABEL, MARGIN,
           AREA_PX, AREA_FRAC, MEAN_P, CONF, ARGMAX_PX,
           ANCHOR_RANGE_M, ANCHOR_BEARING, ANCHOR_AGE_S, ANCHOR_JUMP_M,
           EGO_V, EGO_W, BLUR, LUMA,
           SPEC_ELIGIBLE, SPEC_RAN, SPEC_N, SPEC_CONF, SPEC_LABEL, N_COLS };

constexpr std::array<const char*, N_COLS> kHeader = {
    "stamp_ms", "source", "kind", "label", "cam_x", "cam_y", "cam_z", "cam_yaw",
    "u", "v", "range_m", "p_class", "p_top", "top_label", "margin",
    "area_px", "area_frac", "mean_p", "conf", "argmax_px",
    "anchor_range_m", "anchor_bearing", "anchor_age_s", "anchor_jump_m",
    "ego_v", "ego_w", "blur", "luma",
    "spec_eligible", "spec_ran", "spec_n", "spec_conf", "spec_label" };

// A row starts entirely EMPTY. An unset cell stays empty rather than becoming a 0 or a -1: a sentinel
// survives into a plot as data, and "not measured" is not a small number.
struct Row
{
    std::array<std::string, N_COLS> cell;

    void set(Col c, std::string v)        { cell[c] = std::move(v); }
    void set(Col c, std::uint64_t v)      { cell[c] = std::to_string(v); }
    void set(Col c, int v)                { cell[c] = std::to_string(v); }
    // Non-finite is left EMPTY, which is how NaN reaches the file — never as the text "nan", and
    // never as a number.
    void set(Col c, float v)
    {
        if (std::isfinite(v))
            cell[c] = std::format("{:.6g}", v);
    }

    void emit(std::ofstream& f) const
    {
        for (int i = 0; i < N_COLS; ++i)
            f << cell[static_cast<std::size_t>(i)] << (i + 1 < N_COLS ? ',' : '\n');
    }
};

std::string header_line()
{
    std::string h;
    for (int i = 0; i < N_COLS; ++i)
    {
        h += kHeader[static_cast<std::size_t>(i)];
        h += (i + 1 < N_COLS ? ',' : '\n');
    }
    return h;
}

// Sharpness and brightness of the frame, both cheap and both testing a specific mechanism.
//   blur — variance of the Laplacian on a downscaled grey image. The standard focus measure: HIGH is
//          sharp, low is smooth/blurred. Downscaled first so it measures the image rather than sensor
//          noise, and so it costs microseconds.
//   luma — mean grey level, which is what moves when auto-exposure hunts.
// ★Neither is a threshold or a gate; they are columns. The point is to let the data say which
// mechanism (if either) tracks the dropouts, instead of arguing it from first principles.
void frame_stats(const cv::Mat& bgr, float& blur, float& luma)
{
    blur = luma = kNaN;
    if (bgr.empty() or bgr.type() != CV_8UC3)
        return;
    cv::Mat small, grey, lap;
    const int target_w = 320;
    const double sc = static_cast<double>(target_w) / std::max(1, bgr.cols);
    if (sc < 1.0)
        cv::resize(bgr, small, cv::Size(), sc, sc, cv::INTER_AREA);
    else
        small = bgr;
    cv::cvtColor(small, grey, cv::COLOR_BGR2GRAY);
    luma = static_cast<float>(cv::mean(grey)[0]);
    cv::Laplacian(grey, lap, CV_32F);
    cv::Scalar mu, sigma;
    cv::meanStdDev(lap, mu, sigma);
    blur = static_cast<float>(sigma[0] * sigma[0]);
}

}   // namespace

void DoorApproachLog::configure(std::string path, std::string label, bool enabled)
{
    path_    = std::move(path);
    label_   = std::move(label);
    enabled_ = enabled and not path_.empty() and not label_.empty();
}

void DoorApproachLog::log(std::uint64_t stamp_ms, bool is_360,
                          const Mat::RTMat& room_T_sensor,
                          const RGBDData& rgbd,
                          const rc::semantic::SemanticMap& map,
                          const std::vector<SegDetection>& masks,
                          const std::vector<std::string>& class_names,
                          int spec_eligible, int spec_ran, int spec_n, float spec_best_conf,
                          const std::string& spec_best_label)
{
    if (not enabled_)
        return;

    // Resolve the followed label once, from the SAME name table the model was loaded with. A config
    // key holding a bare class id would be a second source of truth for something the model already
    // declares, and it is exactly the kind of pair that drifts apart silently.
    if (class_id_ < 0)
    {
        for (std::size_t i = 0; i < class_names.size(); ++i)
            if (class_names[i] == label_)
            {
                class_id_ = static_cast<int>(i);
                break;
            }
        if (class_id_ < 0)
        {
            std::println("[DoorApproachLog] disabled — '{}' is not in the model's class table", label_);
            enabled_ = false;
            return;
        }
    }

    if (not csv_.is_open())
        if (not rc::diag::open_rotating(csv_, path_, header_line()))
        {
            enabled_ = false;
            return;
        }

    const char* src = is_360 ? "ricoh" : "zed";

    // Viewpoint. ★Not decoration: a miss repeated at 20 Hz from one pose is ONE look, and the only way
    // to see that in the file afterwards is to have written down where the camera was each time.
    const Eigen::Vector3f t = room_T_sensor.translation().cast<float>();
    const float yaw = static_cast<float>(std::atan2(room_T_sensor.linear()(1, 0),
                                                   room_T_sensor.linear()(0, 0)));

    // Ego-motion, finite-differenced against the previous logged frame. ★`ego_w` exists so that "the
    // motion is pure translation" is a measured column and not an assumption the analysis inherits.
    float ego_v = kNaN, ego_w = kNaN;
    PrevPose& prev = prev_[is_360 ? 1 : 0];   // never difference one sensor's pose against the other's
    if (prev.have and stamp_ms > prev.stamp_ms)
    {
        const float dt = static_cast<float>(stamp_ms - prev.stamp_ms) * 1e-3f;
        if (dt > 0.f)
        {
            ego_v = (t - prev.pos).norm() / dt;
            float dyaw = yaw - prev.yaw;
            while (dyaw >  static_cast<float>(M_PI)) dyaw -= 2.f * static_cast<float>(M_PI);
            while (dyaw < -static_cast<float>(M_PI)) dyaw += 2.f * static_cast<float>(M_PI);
            ego_w = dyaw / dt;
        }
    }
    prev.have = true;
    prev.pos = t;
    prev.yaw = yaw;
    prev.stamp_ms = stamp_ms;

    float blur = kNaN, luma = kNaN;
    frame_stats(rgbd.bgr, blur, luma);

    // Range to the ANCHOR — the door's room-frame position from when it was last actually seen. This
    // is the column that keeps counting down through the silent frames.
    // ★Deliberately computed BEFORE this frame's own detection refreshes the anchor, so a row's range
    // never depends on the outcome recorded on that same row. The alternative reads like a measurement
    // and is circular: it was the `ex_p` / `ex_p_prior` mistake in the detect_probe work, where the
    // logged prior was one cycle behind its own row and the offset — not the value — was what lied.
    float anchor_range = kNaN, anchor_bearing = kNaN, anchor_age = kNaN;
    if (anchored_)
    {
        const Eigen::Vector3f d_room = anchor_room_ - t;
        anchor_range = d_room.norm();
        // Bearing in the CAMERA frame: +x is right, +y is depth (see the deprojection below), so a
        // door dead ahead reads 0 and one leaving the field of view walks toward +-pi/2.
        const Eigen::Vector3f d_cam = room_T_sensor.linear().cast<float>().transpose() * d_room;
        anchor_bearing = std::atan2(d_cam.x(), d_cam.y());
        // SIGNED. The anchor is deliberately shared between the two sources — the door is the door,
        // and a ricoh frame gets a range from wherever the ZED last saw it. But the two carry
        // different stamps, so a ricoh frame can be older than the zed refresh that set the anchor.
        // That is clock skew between sources, not a missing value, and NaN-ing it (as the first
        // version did) silently emptied the column on 42% of ricoh rows.
        anchor_age = (static_cast<double>(stamp_ms) - static_cast<double>(anchor_stamp_ms_)) * 1e-3;
    }

    const auto new_row = [&](const char* kind)
    {
        Row r;
        r.set(STAMP, stamp_ms);
        r.set(SOURCE, std::string{src});
        r.set(KIND, std::string{kind});
        r.set(LABEL, label_);
        r.set(CAM_X, t.x());
        r.set(CAM_Y, t.y());
        r.set(CAM_Z, t.z());
        r.set(CAM_YAW, yaw);
        r.set(ANCHOR_RANGE_M, anchor_range);
        r.set(ANCHOR_BEARING, anchor_bearing);
        r.set(ANCHOR_AGE_S, anchor_age);
        r.set(ANCHOR_JUMP_M, anchor_jump_m_);
        r.set(EGO_V, ego_v);
        r.set(EGO_W, ego_w);
        r.set(BLUR, blur);
        r.set(LUMA, luma);
        // -1 in these means "this build/frame carried no specialist information at all" — which is a
        // third thing again, distinct from "not eligible" (0) and "ran" (1).
        if (spec_eligible >= 0) r.set(SPEC_ELIGIBLE, spec_eligible);
        if (spec_ran >= 0)      r.set(SPEC_RAN, spec_ran);
        if (spec_n >= 0)        r.set(SPEC_N, spec_n);
        if (spec_best_conf >= 0.0f) r.set(SPEC_CONF, spec_best_conf);
        if (not spec_best_label.empty()) r.set(SPEC_LABEL, spec_best_label);
        return r;
    };

    // ── the PEAK row: written every frame, including frames with no evidence at all ──────────────
    //
    // Searched at the classifier's NATIVE resolution rather than on an upsampled plane: the peak of a
    // bilinearly-enlarged field is an artefact of the interpolation, and the model never had an opinion
    // between its own cells. NaN cells are "not looked at" (a 360 strip the scheduler skipped) and are
    // skipped rather than treated as zero.
    int   best_nx = -1, best_ny = -1;
    float best_p  = kNaN;
    const int ch = map.prob_channel(class_id_);
    if (ch >= 0 and map.graded() and not map.probs[static_cast<std::size_t>(ch)].empty())
    {
        const cv::Mat& plane = map.probs[static_cast<std::size_t>(ch)];
        for (int r = 0; r < plane.rows; ++r)
        {
            const float* pr = plane.ptr<float>(r);
            for (int c = 0; c < plane.cols; ++c)
                if (std::isfinite(pr[c]) and (not std::isfinite(best_p) or pr[c] > best_p))
                {
                    best_p = pr[c]; best_nx = c; best_ny = r;
                }
        }
    }

    // How many pixels the LIVE pipeline calls this class — the argmax's own verdict, on the same row
    // as the posterior that produced it. This is the column that dates the flip.
    int argmax_px = 0;
    if (not map.labels.empty() and class_id_ >= 0 and class_id_ < 256)
        argmax_px = cv::countNonZero(map.labels == static_cast<unsigned char>(class_id_));

    Row peak = new_row("peak");
    peak.set(ARGMAX_PX, argmax_px);
    if (best_nx >= 0 and map.probs_src_size.width > 0 and map.probs_src_size.height > 0)
    {
        const cv::Mat& plane = map.probs[static_cast<std::size_t>(ch)];
        // Native cell → frame pixel, cell centre, the inverse of prob_at's mapping.
        const int u = std::clamp(static_cast<int>((best_nx + 0.5f) * map.probs_src_size.width
                                                  / static_cast<float>(plane.cols)),
                                 0, map.probs_src_size.width - 1);
        const int v = std::clamp(static_cast<int>((best_ny + 0.5f) * map.probs_src_size.height
                                                  / static_cast<float>(plane.rows)),
                                 0, map.probs_src_size.height - 1);

        // The competitor AT THE SAME PIXEL. ★Paired, deliberately: an earlier version of this analysis
        // compared max P(door) over a region with max P(wall) over the same region — different pixels —
        // and invented a margin no pixel actually had (it read a landslide where the truth was a coin
        // flip). The margin is only meaningful read off one pixel.
        //
        // ★Read at the NATIVE CELL, not through probs_at(u, v). probs_at samples bilinearly at a frame
        // pixel, and the round trip cell → frame pixel → bilinear sample does not return the cell's own
        // value: it lands between centres and smooths the peak down. The first live rows showed it —
        // p_class 0.991 (native) beside p_top 0.972 for the SAME class at the SAME pixel, an 0.019
        // disagreement inside one row. Two sampling conventions in one row is an instrument that
        // contradicts itself, and the disagreement is largest exactly at a peak, which is every row
        // here. Cells are what the model computed; the interpolation is ours.
        std::vector<std::pair<int, float>> ranked;
        ranked.reserve(map.prob_class_ids.size());
        for (std::size_t k = 0; k < map.prob_class_ids.size(); ++k)
            if (const cv::Mat& pl = map.probs[k];
                not pl.empty() and best_ny < pl.rows and best_nx < pl.cols)
                ranked.emplace_back(map.prob_class_ids[k], pl.at<float>(best_ny, best_nx));
        std::ranges::sort(ranked, [](const auto& a, const auto& b)
        {
            if (std::isnan(a.second)) return false;
            if (std::isnan(b.second)) return true;
            return a.second > b.second;
        });
        float p_top = kNaN, margin = kNaN;
        std::string top_label;
        if (not ranked.empty())
        {
            p_top = ranked.front().second;
            const int top_id = ranked.front().first;
            top_label = (top_id >= 0 and top_id < static_cast<int>(class_names.size()))
                            ? class_names[static_cast<std::size_t>(top_id)] : std::to_string(top_id);
            // Signed: positive ⇒ this class WON here, negative ⇒ it lost by this much. One column that
            // says both which side of the flip we are on and how far from it.
            const float runner_up = (ranked.size() >= 2) ? ranked[1].second : kNaN;
            margin = (top_id == class_id_) ? (p_top - runner_up) : (best_p - p_top);
        }

        peak.set(U, u);
        peak.set(V, v);
        peak.set(RANGE_M, depth_at(rgbd, u, v));
        peak.set(P_CLASS, best_p);
        peak.set(P_TOP, p_top);
        peak.set(TOP_LABEL, top_label);
        peak.set(MARGIN, margin);
    }
    // else: ungraded model, or every plane "not looked at". The row is still written, with the
    // posterior columns EMPTY — it records that the frame happened and that nothing was measured on
    // it, which is a different fact from a low probability.
    peak.emit(csv_);

    // ── one row per PUBLISHED mask of this class ────────────────────────────────────────────────
    //
    // A peak row with a high posterior and no mask row beside it is the interesting case: the model
    // found it and this component dropped it (min_area / YOLO priority), which from a concept agent's
    // side is indistinguishable from the door not being there. detect_drops.csv says which stage did
    // it; this file says how far away it was when it started happening.
    const float frame_px = static_cast<float>(std::max(1, map.labels.cols * map.labels.rows));
    for (const auto& d : masks)
    {
        if (d.label != label_)
            continue;

        const int cu = d.bbox.x + d.bbox.width / 2;
        const int cv_ = d.bbox.y + d.bbox.height / 2;

        // Area and mean posterior over the MASK itself, not the bbox: a door is a rectangle only when
        // seen face-on, and at the far end of a corridor it is a sliver inside a much larger box.
        int   area = 0;
        double p_sum = 0.0;
        int   p_n = 0;
        if (not d.mask.empty() and d.mask.type() == CV_8UC1)
            for (int r = 0; r < d.mask.rows; ++r)
            {
                const unsigned char* mr = d.mask.ptr<unsigned char>(r);
                for (int c = 0; c < d.mask.cols; ++c)
                    if (mr[c])
                    {
                        ++area;
                        if (const float p = map.prob_at(class_id_, c, r, kNaN); std::isfinite(p))
                        {
                            p_sum += p; ++p_n;
                        }
                    }
            }

        // ★REFRESH THE ANCHOR from this detection: deproject the mask centroid to the room frame, so
        // the range axis survives the frames where the detection does not. Camera convention is the
        // one the rest of this component uses (graph_publisher.cpp:985): x right, y DEPTH, z up.
        // Guarded on a valid depth — a centroid over a stereo hole would anchor the door onto noise.
        if (const float dz = depth_at(rgbd, cu, cv_);
            std::isfinite(dz) and rgbd.focal_x > 0.f and rgbd.focal_y > 0.f)
        {
            const float cx = static_cast<float>(rgbd.width) * 0.5f;
            const float cy = static_cast<float>(rgbd.height) * 0.5f;
            const Eigen::Vector3f p_cam((static_cast<float>(cu) - cx) * dz / rgbd.focal_x,
                                        dz,
                                        (cy - static_cast<float>(cv_)) * dz / rgbd.focal_y);
            const Eigen::Vector3f p_room = (room_T_sensor.cast<float>() * p_cam.cast<float>()).eval();
            // How far this refresh moved the anchor. A few centimetres is the same door seen again;
            // metres means the association jumped, and the axis changed meaning mid-approach. Recorded
            // rather than suppressed — a filter here would hide exactly the event worth seeing.
            anchor_jump_m_ = anchored_ ? (p_room - anchor_room_).norm() : 0.f;
            anchor_room_ = p_room;
            anchor_stamp_ms_ = stamp_ms;
            anchored_ = true;
        }

        Row row = new_row("mask");
        row.set(U, cu);
        row.set(V, cv_);
        row.set(RANGE_M, depth_at(rgbd, cu, cv_));
        row.set(AREA_PX, area);
        row.set(AREA_FRAC, static_cast<float>(area) / frame_px);
        row.set(MEAN_P, p_n > 0 ? static_cast<float>(p_sum / p_n) : kNaN);
        row.set(CONF, d.confidence);
        row.set(ARGMAX_PX, argmax_px);
        row.emit(csv_);
    }

    csv_.flush();
}

}   // namespace rc::diag
