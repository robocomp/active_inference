/*
 * graph_publisher.cpp — DSR semantic_grid exports (masks).
 */

// These pull in oneTBB (tbb/profiling.h). Include them BEFORE any header that activates Qt's `emit`
// keyword macro (dsr_api.h via graph_publisher.h, and <QtGlobal>), otherwise the empty `emit` macro
// mangles TBB's profiling.h and it fails to compile.
#include "yolo_processor.h"   // SegDetection
#include "yolo_human.h"       // rc::human_pose::PoseDetection, BODY18_FROM_COCO

#include "graph_publisher.h"

#include "../../common/motion_corruption/motion_corruption.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <print>
#include <sstream>
#include <unordered_map>
#include <utility>

#include <QtGlobal>
#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::stamp_creation

GraphPublisher::GraphPublisher(std::shared_ptr<DSR::DSRGraph> graph,
                               const RetinaParams& params,
                               std::function<void()> relayout)
    : G_(std::move(graph)), params_(params), relayout_(std::move(relayout))
{}

void GraphPublisher::publish(const RGBDData& rgbd, const Mat::RTMat& room_T_zed,
                             const std::vector<SegDetection>& detections, std::uint64_t frame_ts_ms,
                             const std::vector<BearingDetection>& bearing_detections)
{
    if (ensure_node("masks", "Plum", masks_ready_, /*relayout=*/true))
        upload_masks(rgbd, room_T_zed, detections, frame_ts_ms, bearing_detections);
}

bool GraphPublisher::ensure_node(const char* name, const char* color, bool& ready, bool relayout)
{
    if (ready)
        return true;
    // NO ROOM GATE. These nodes are parented under `zed` and now carry camera-frame payloads, so a room
    // is irrelevant to creating them — gating on one is what stopped YOLO running at all without
    // localization. `robot` is still required: it roots the RT chain the `zed` node hangs from.
    if (G_->get_nodes_by_type("robot").empty())
        return false;
    if (G_->get_node(name).has_value())
    {
        ready = true;
        return true;
    }

    auto zed_node = G_->get_node("zed");
    if (!zed_node.has_value())
    {
        qWarning() << "[GraphPublisher] WARNING: 'zed' node not found — cannot create" << name;
        return false;
    }

    auto node = DSR::Node::create<semantic_grid_node_type>(name);
    G_->add_or_modify_attrib_local<color_att>(node, std::string{color});
    G_->add_or_modify_attrib_local<level_att>(node, 4);
    G_->add_or_modify_attrib_local<parent_att>(node, zed_node.value().id());
    G_->add_or_modify_attrib_local<pos_x_att>(node, 105.849792f);
    G_->add_or_modify_attrib_local<pos_y_att>(node, 291.904266f);

    rc::provenance::stamp_creation(*G_, node);   // birth stamp: epoch ms + local ISO-8601
    const auto id = G_->insert_node(node);
    if (!id.has_value())
    {
        qWarning() << "[GraphPublisher] ERROR: failed to insert" << name << "node";
        return false;
    }

    ready = true;
    qInfo() << "[GraphPublisher] Created" << name << "node id=" << *id << "(semantic_grid) under 'zed'";
    G_->get_rt_api()->insert_or_assign_edge_RT(zed_node.value(), *id, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f});
    if (relayout and relayout_)
        relayout_();
    return true;
}

void GraphPublisher::publish_semantic(const cv::Mat& labels, std::uint64_t stamp)
{
    if (labels.empty() or labels.type() != CV_8UC1)
        return;
    if (!ensure_node("semantic", "Teal", semantic_ready_, /*relayout=*/true))
        return;
    auto node_opt = G_->get_node("semantic");
    if (!node_opt.has_value())
        return;
    auto& node = node_opt.value();

    // Flatten to a contiguous byte buffer (a cv::Mat row may be padded). labels is already at the ZED
    // image resolution, so semantic_labels[v*width + u] is the class id of original-image pixel (u,v).
    const int w = labels.cols, h = labels.rows;
    std::vector<std::uint8_t> buf;
    buf.reserve(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    for (int r = 0; r < h; ++r)
    {
        const std::uint8_t* p = labels.ptr<std::uint8_t>(r);
        buf.insert(buf.end(), p, p + w);
    }

    G_->add_or_modify_attrib_local<semantic_labels_att>(node, buf);
    G_->add_or_modify_attrib_local<semantic_width_att>(node, w);
    G_->add_or_modify_attrib_local<semantic_height_att>(node, h);
    G_->add_or_modify_attrib_local<semantic_timestamp_ms_att>(node, stamp);
    G_->add_or_modify_attrib_local<semantic_frame_id_att>(node, ++semantic_seq_);
    G_->update_node(std::move(node));
}

namespace
{

// Per-slice APPEARANCE summary (MaskColor.*): what the concept agents fold into their per-instance
// appearance belief to tint the display mesh. Three modelling decisions live here, all deliberate:
//
//  1. CHROMATICITY, not raw RGB. Observed = albedo x irradiance + specular. The viewer applies its own
//     ambient+diffuse shading, so publishing observed RGB would bake the room's lighting into the asset
//     and then shade it a second time. Normalising by (R+G+B) is invariant to the per-frame illumination
//     gain by construction — the dominant nuisance removed without inferring a light model.
//  2. GRID CELLS, not pixels. Neighbouring pixels on one surface are massively correlated, so treating
//     each as an independent sample would shrink the variance like 1/sqrt(N) and make this channel
//     absurdly overconfident. One sample per cell, and the BETWEEN-CELL scatter is the reported variance.
//  3. INTERIOR cells only. A cell contributes only if it is fully foreground AND its four neighbours are
//     too — silhouette erosion at cell granularity, which keeps background bleed at the mask edge out of
//     the estimate. Per-cell MEDIAN (not mean) additionally rejects specular highlights inside a cell.
//
// Note the variance floor is not a magic number: it is first-order propagation of 8-bit QUANTISATION
// through the normalisation, so it grows for dark pixels — which is correct, since a dark surface really
// does have poorly determined chromaticity. It also keeps a single-cell slice from claiming zero variance.
struct MaskColorSummary
{
    Eigen::Vector3f chroma = Eigen::Vector3f::Zero();   // median (r,g,b), each in [0,1], summing to 1
    Eigen::Vector3f var    = Eigen::Vector3f::Zero();   // robust between-cell variance, per channel
    float           n_eff  = 0.0f;                      // contributing interior cells (0 ⇒ no information)
};

// bgr: CV_8UC3 in BGR order (RGBDData::rgb is BGR despite the field name — media_plane_source converts
// the RGB8 media-plane view on arrival). mask_bin: full-frame 0/255 silhouette. Bounds are the detection
// bbox already clamped to the image.
MaskColorSummary mask_color_summary(const cv::Mat& bgr, const cv::Mat& mask_bin,
                                    int min_x, int min_y, int max_x, int max_y, int cell_px)
{
    MaskColorSummary out;
    if (bgr.empty() or bgr.type() != CV_8UC3 or cell_px <= 0)
        return out;

    const int nx = (max_x - min_x) / cell_px;
    const int ny = (max_y - min_y) / cell_px;
    if (nx <= 0 or ny <= 0)
        return out;

    // Pass 1: a cell is FULL iff every one of its pixels is foreground. That is a definition, not a tuned
    // cutoff — it is what makes the neighbour test below a clean erosion.
    std::vector<char> full(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny), 0);
    for (int cy = 0; cy < ny; ++cy)
        for (int cx = 0; cx < nx; ++cx)
        {
            const int x0 = min_x + cx * cell_px, y0 = min_y + cy * cell_px;
            bool all_fg = true;
            for (int r = y0; r < y0 + cell_px and all_fg; ++r)
                for (int c = x0; c < x0 + cell_px; ++c)
                    if (mask_bin.at<std::uint8_t>(r, c) == 0) { all_fg = false; break; }
            full[static_cast<std::size_t>(cy) * nx + cx] = all_fg ? 1 : 0;
        }

    // Pass 2: one median-chromaticity sample per INTERIOR cell (full, with all four neighbours full).
    std::vector<Eigen::Vector3f> cell_chroma;
    cell_chroma.reserve(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
    double sum_intensity = 0.0;   // mean (R+G+B)/255 over contributing cells, for the quantisation floor
    std::array<std::vector<float>, 3> ch;
    for (int cy = 1; cy < ny - 1; ++cy)
        for (int cx = 1; cx < nx - 1; ++cx)
        {
            const std::size_t idx = static_cast<std::size_t>(cy) * nx + cx;
            if (!full[idx] or !full[idx - 1] or !full[idx + 1]
                or !full[idx - nx] or !full[idx + nx])
                continue;

            const int x0 = min_x + cx * cell_px, y0 = min_y + cy * cell_px;
            for (auto& v : ch) { v.clear(); v.reserve(static_cast<std::size_t>(cell_px) * cell_px); }
            double cell_intensity = 0.0;
            for (int r = y0; r < y0 + cell_px; ++r)
                for (int c = x0; c < x0 + cell_px; ++c)
                {
                    const cv::Vec3b& p = bgr.at<cv::Vec3b>(r, c);
                    const float b = p[0], g = p[1], rr = p[2];   // BGR order
                    const float s = rr + g + b;
                    if (s <= 0.0f)          // a pure-black pixel has no defined chromaticity at all
                        continue;
                    ch[0].push_back(rr / s);
                    ch[1].push_back(g  / s);
                    ch[2].push_back(b  / s);
                    cell_intensity += s / 255.0;
                }
            if (ch[0].empty())
                continue;

            const std::size_t mid = ch[0].size() / 2;
            Eigen::Vector3f med;
            for (int k = 0; k < 3; ++k)
            {
                std::nth_element(ch[k].begin(), ch[k].begin() + static_cast<std::ptrdiff_t>(mid), ch[k].end());
                med[k] = ch[k][mid];
            }
            cell_chroma.push_back(med);
            sum_intensity += cell_intensity / static_cast<double>(ch[0].size());
        }

    if (cell_chroma.empty())
        return out;

    // Robust centre + scale across cells: median and (1.4826*MAD)^2, which is the variance consistent with
    // a median centre. A handful of cells landing on a highlight or a thin occluder cannot drag either.
    std::vector<float> comp(cell_chroma.size());
    Eigen::Vector3f mad = Eigen::Vector3f::Zero();
    const std::size_t mid = cell_chroma.size() / 2;
    for (int k = 0; k < 3; ++k)
    {
        for (std::size_t i = 0; i < cell_chroma.size(); ++i) comp[i] = cell_chroma[i][k];
        std::nth_element(comp.begin(), comp.begin() + static_cast<std::ptrdiff_t>(mid), comp.end());
        out.chroma[k] = comp[mid];
        for (std::size_t i = 0; i < cell_chroma.size(); ++i) comp[i] = std::abs(cell_chroma[i][k] - out.chroma[k]);
        std::nth_element(comp.begin(), comp.begin() + static_cast<std::ptrdiff_t>(mid), comp.end());
        mad[k] = comp[mid];
    }

    // Quantisation floor: one LSB is 1/255 of a channel; through r = R/S its first-order effect is
    // (1/255)/S, with S the mean channel sum in [0,3]. Uniform over an LSB ⇒ divide the square by 12.
    const double mean_intensity = std::max(1e-3, sum_intensity / static_cast<double>(cell_chroma.size()));
    const float  quant_var = static_cast<float>((1.0 / (255.0 * mean_intensity)) * (1.0 / (255.0 * mean_intensity)) / 12.0);

    for (int k = 0; k < 3; ++k)
    {
        const float robust_sd = 1.4826f * mad[k];
        out.var[k] = robust_sd * robust_sd + quant_var;
    }
    out.n_eff = static_cast<float>(cell_chroma.size());
    return out;
}

}   // namespace

// FRAME CONTRACT — the 'masks' node is a SHARED contract read by every concept agent through ONE
// component, common/mask_ingestor. Mask support points are DUAL-PUBLISHED, 1-to-1:
//   * mask_support_points_cam  → ZED/CAMERA frame. THE ONLY 3-D array published. Raw deprojection,
//                                before any pose is applied. Centroids/bboxes accompany it in the SAME
//                                frame; common/mask_ingestor recomputes both in the consumer's target
//                                frame after transforming, so no room-frame variant is needed.
//
// ★THIS PRODUCER NO LONGER APPLIES room_T_zed TO MASKS. It used to, and that made a RAW-PERCEPTION
// component depend on a LOCALIZATION one: room_concept was a required peer, and when the RT chain
// failed to resolve, room_T_sensor defaulted to Identity and masks went out in camera coordinates
// LABELLED as room frame — undetectably. It also HID the localization covariance, which only the
// component that applies the transform can account for (J·Sigma_chain·J^T). The transform now lives in
// common/mask_ingestor, capture-stamp pinned and forward-extrapolated over the RT lag.
// The legacy room-frame "mask_support_points" array is GONE — do not reintroduce it.
//
// EGO-MOTION CORRUPTION ANNOTATION (MaskMotion.enabled, default on; common/motion_corruption):
// while the camera moves, a mask is corrupted in a way the consumer can't reconstruct without the
// capture-time twist/exposure/timing. The producer attaches the DECOMPOSITION; the consumer decides
// gate-vs-downweight (per-concept b_max), per-DOF split, and folds the variance into its innovation cov.
//   per-mask (1-to-1 with mask_label_ids):
//     mask_motion_dotd     — metric position-corruption speed Z·‖ṡ‖ (m/s), diagnostic
//     mask_motion_bias     — systematic displacement from a KNOWN timing offset (m) — GATE on this
//     mask_motion_var      — variance to ADD to R (m²): exposure blur + timing jitter, ×peripheral
//     mask_trunc_frac      — fraction of silhouette pixels on the image border (truncation → bias)
//     mask_centroid_radius — normalized centroid radius from the principal point (periphery penalty)
//   per-frame (global): mask_cam_twist [vx,vy,vz,wx,wy,wz] (OPTICAL frame) + mask_frame_dt_s.
//
// APPEARANCE ANNOTATION (MaskColor.enabled, default on; see mask_color_summary above):
//   mask_color_rgb / mask_color_var (3 each per mask) + mask_color_neff (1 per mask) — the slice's median
//   chromaticity, its between-cell variance, and the number of contributing interior cells. Consumed by
//   common/appearance_belief to tint an agent's display mesh. DISPLAY-ONLY: nothing in any geometric fit
//   reads it, which is what keeps a channel with this many correlated samples from being able to do harm.
//   Ricoh slices publish n_eff==0 (detected on the panorama, not on this zed frame ⇒ no pixels here).
void GraphPublisher::upload_masks(const RGBDData& rgbd, const Mat::RTMat& room_T_zed,
                                  const std::vector<SegDetection>& detections, std::uint64_t frame_ts_ms,
                                  const std::vector<BearingDetection>& bearing_detections)
{
    // Keep masks stream progressing even when voxel grid processing is paused.
    const std::uint64_t sensing_frame = ++masks_publish_seq_;
    if (sensing_frame == last_masks_uploaded_frame_)
        return;
    last_masks_uploaded_frame_ = sensing_frame;

    auto masks_node_opt = G_->get_node("masks");
    if (!masks_node_opt.has_value())
        return;
    auto& masks_node = masks_node_opt.value();

    if (rgbd.depth.empty() || rgbd.width <= 0 || rgbd.height <= 0 || rgbd.focal_x <= 0.f || rgbd.focal_y <= 0.f)
    {
        G_->add_or_modify_attrib_local<mask_frame_id_att>(masks_node, static_cast<int>(sensing_frame));
        G_->add_or_modify_attrib_local<mask_timestamp_ms_att>(masks_node, frame_ts_ms);
        G_->add_or_modify_attrib_local<mask_count_att>(masks_node, 0);
        G_->add_or_modify_attrib_local<mask_labels_att>(masks_node, std::string{});
        G_->add_or_modify_attrib_local<mask_label_ids_att>(masks_node, std::vector<float>{});
        G_->add_or_modify_attrib_local<mask_confidences_att>(masks_node, std::vector<float>{});
        G_->add_or_modify_attrib_local<mask_support_offsets_att>(masks_node, std::vector<float>{0.0f});
        G_->add_or_modify_attrib_local<mask_support_points_cam_att>(masks_node, std::vector<float>{});
        G_->add_or_modify_attrib_local<mask_centroids_xyz_att>(masks_node, std::vector<float>{});
        G_->add_or_modify_attrib_local<mask_bbox_min_xyz_att>(masks_node, std::vector<float>{});
        G_->add_or_modify_attrib_local<mask_bbox_max_xyz_att>(masks_node, std::vector<float>{});
        G_->add_or_modify_attrib_local<mask_pixels_xy_att>(masks_node, std::vector<float>{});
        G_->add_or_modify_attrib_local<mask_pixel_offsets_att>(masks_node, std::vector<float>{0.0f});
        G_->add_or_modify_attrib_local<mask_depth_var_att>(masks_node, std::vector<float>{});
        if (params_.MASK_COLOR_ENABLED)
        {
            G_->add_or_modify_attrib_local<mask_color_rgb_att>(masks_node, std::vector<float>{});
            G_->add_or_modify_attrib_local<mask_color_var_att>(masks_node, std::vector<float>{});
            G_->add_or_modify_attrib_local<mask_color_neff_att>(masks_node, std::vector<float>{});
        }
        if (params_.MASK_MOTION_ENABLED)
        {
            G_->add_or_modify_attrib_local<mask_motion_dotd_att>(masks_node, std::vector<float>{});
            G_->add_or_modify_attrib_local<mask_motion_bias_att>(masks_node, std::vector<float>{});
            G_->add_or_modify_attrib_local<mask_motion_var_att>(masks_node, std::vector<float>{});
            G_->add_or_modify_attrib_local<mask_trunc_frac_att>(masks_node, std::vector<float>{});
            G_->add_or_modify_attrib_local<mask_centroid_radius_att>(masks_node, std::vector<float>{});
            G_->add_or_modify_attrib_local<mask_range_att>(masks_node, std::vector<float>{});
            G_->add_or_modify_attrib_local<mask_cam_twist_att>(masks_node, std::vector<float>{0,0,0,0,0,0});
            G_->add_or_modify_attrib_local<mask_frame_dt_s_att>(masks_node, 0.0f);
            G_->add_or_modify_attrib_local<mask_rt_lag_s_att>(masks_node, -1.0f);
            G_->add_or_modify_attrib_local<mask_rt_gap_s_att>(masks_node, -1.0f);
        }
        G_->update_node(std::move(masks_node));
        return;
    }

    const Eigen::Matrix3f room_rotation = room_T_zed.linear().cast<float>();
    const Eigen::Vector3f room_translation = room_T_zed.translation().cast<float>();
    const float fx = rgbd.focal_x;
    const float fy = rgbd.focal_y;
    const float cx = static_cast<float>(rgbd.width) * 0.5f;
    const float cy = static_cast<float>(rgbd.height) * 0.5f;

    // Ego-motion camera twist (optical frame) for the per-mask corruption annotation. Finite-difference
    // the zed pose against the previous published frame, pinned to capture stamps; zero on the first
    // frame or when no stamp is available (corruption then evaluates to 0 → masks pass through unweighted).
    rc::motion::Twist cam_twist_optical;
    float frame_dt_s = 0.0f;
    if (params_.MASK_MOTION_ENABLED and have_last_zed_ and frame_ts_ms > 0 and last_zed_ts_ms_ > 0
        and frame_ts_ms > last_zed_ts_ms_)
    {
        frame_dt_s = static_cast<float>(frame_ts_ms - last_zed_ts_ms_) * 1e-3f;
        const auto tw = rc::motion::body_twist_from_poses(last_zed_R_, last_zed_t_,
                                                          room_rotation, room_translation, frame_dt_s);
        cam_twist_optical = rc::motion::vcam_to_optical(tw);
    }
    const rc::motion::CorruptionParams corruption_params{
        .exposure_s      = params_.MASK_MOTION_EXPOSURE_S,
        .timing_jitter_s = params_.MASK_MOTION_TIMING_JITTER_S,
        .timing_offset_s = params_.MASK_MOTION_TIMING_OFFSET_S};

    // Lazily open the diagnostic CSV (once). |v|,|w| are the frame twist magnitudes; per-mask columns
    // let you confirm dot_d≈0 when static and that it spikes under panning (rotation dominates).
    if (params_.MASK_MOTION_ENABLED and params_.MASK_MOTION_CSV_LOG and not motion_csv_open_attempted_)
    {
        motion_csv_open_attempted_ = true;
        motion_csv_.open("etc/mask_motion_log.csv", std::ios::out | std::ios::trunc);
        if (motion_csv_.is_open())
            motion_csv_ << "frame,ts_ms,dt_s,rt_lag_s,rt_gap_s,v_norm,w_norm,label,Z,xn,yn,radius,trunc,dotd,bias,var\n";
    }
    const float v_norm = cam_twist_optical.v.norm();
    const float w_norm = cam_twist_optical.w.norm();

    // Localization timing lag: capture stamp minus the NEWEST room→robot RT block stamp. This is the δt
    // the no-extrapolation clamp leaves when the robot-pose stream lags the camera — i.e. the residual
    // that drives the corruption BIAS. Measuring it here sizes MaskMotion.timing_offset_s and quantifies
    // the effect of raising room_concept's RT publish rate. -1 = couldn't read (edge/attr/stamp missing).
    float rt_lag_s = -1.0f;
    float rt_gap_s = -1.0f;   // spacing between the two newest RT blocks in this replica (block density)
    if (params_.MASK_MOTION_ENABLED and frame_ts_ms > 0)
    {
        const auto room_nodes  = G_->get_nodes_by_type("room");
        const auto robot_nodes = G_->get_nodes_by_type("robot");
        if (not room_nodes.empty() and not robot_nodes.empty())
        {
            // The dynamic localization RT edge is robot→room (robot-rooted bootstrap; room_concept
            // dsr_update_pose writes parent=robot, child=room). Try that first, then the reverse.
            auto edge = G_->get_rt_api()->get_edge_RT(robot_nodes.front(), room_nodes.front().id());
            if (not edge.has_value())
                edge = G_->get_rt_api()->get_edge_RT(room_nodes.front(), robot_nodes.front().id());
            if (edge.has_value())
            {
                if (auto ts = G_->get_attrib_by_name<rt_timestamps_att>(edge.value()); ts.has_value())
                {
                    // Two newest stamps in the consumer's ring: newest → rt_lag; (newest - second) →
                    // rt_gap, the consumer-side block spacing. rt_gap ~16ms = the 60Hz blocks actually
                    // reach this replica (dense ring → good interpolation); ~200ms = only ~5Hz arrive.
                    std::uint64_t newest = 0, second = 0;
                    for (const auto t : ts->get())   // ring buffer; 0 = empty slot
                    {
                        if (t > newest) { second = newest; newest = t; }
                        else if (t > second and t < newest) { second = t; }
                    }
                    if (newest > 0)
                        rt_lag_s = static_cast<float>(static_cast<std::int64_t>(frame_ts_ms)
                                                      - static_cast<std::int64_t>(newest)) * 1e-3f;
                    if (newest > 0 and second > 0)
                        rt_gap_s = static_cast<float>(newest - second) * 1e-3f;
                }
            }
        }
    }

    std::vector<float> label_ids;
    std::vector<float> confidences;
    std::vector<float> support_offsets;
    // Dual-publish: the SAME support points in CAMERA (zed) frame (raw deprojected, no room transform),
    // 1-to-1 with support_points/support_offsets. Lets a consumer transform through the probabilistic
    // chain (zed→…→target) itself — pinned to the capture stamp + carrying localization covariance —
    // instead of inheriting the room transform baked here. Room-frame array kept for legacy consumers.
    std::vector<float> support_points_cam;
    std::vector<float> centroids_xyz;
    std::vector<float> bbox_min_xyz;
    std::vector<float> bbox_max_xyz;
    // Raw 2D mask silhouette (foreground pixels, BEFORE the depth gate) so downstream agents have
    // the RGB mask as an evidence channel independent of depth — flat (col,row) pairs per mask,
    // delimited by mask_pixel_offsets exactly like support_points/support_offsets.
    std::vector<float> mask_pixels_xy;
    std::vector<float> mask_pixel_offsets;
    // Per-mask ego-motion corruption decomposition (1-to-1 with label_ids). See common/motion_corruption.
    std::vector<float> motion_dotd;       // metric position-corruption speed (m/s)
    std::vector<float> motion_bias;       // systematic timing-offset displacement (m) — consumer gates on this
    std::vector<float> motion_var;        // variance to ADD to R (m²) — blur + jitter, ×peripheral
    std::vector<float> trunc_frac;        // fraction of silhouette pixels on the image border (truncation)
    std::vector<float> centroid_radius;   // normalized centroid radius from principal point (periphery)
    std::vector<float> mask_range;         // mean camera→mask depth Z (m): static range — consumer scales R + pose common-mode (a far view can't resolve orientation)
    std::vector<float> has_depth_flags;    // 1.0 = 3D slice (zed, or ricoh with lidar depth); 0.0 = ricoh bearing-only
    std::vector<float> mask_source;        // sensor SOURCE per mask: 0.0 = zed (front RGB-D), 1.0 = ricoh (360). Unlike
                                           //   mask_has_depth this is unambiguous (a lidar-depth ricoh mask is still 1.0).
    std::vector<float> azimuths;           // room-frame bearing (rad); meaningful only for has_depth==0 slices
    std::vector<float> depth_var;          // σ_range² (m²) to ADD to R along the mask ray (common/depth_projection).
                                           // 0 for dense-depth zed masks; the scored range_var for lidar-depth ricoh masks.
    // Per-mask appearance summary (MaskColor.*, see mask_color_summary above). DISPLAY-ONLY evidence: no
    // geometric fit reads it. 3 floats each per mask for rgb/var, 1 for n_eff; n_eff==0 ⇒ no information.
    std::vector<float> color_rgb;
    std::vector<float> color_var;
    std::vector<float> color_neff;
    std::ostringstream labels_joined;
    std::size_t total_support_points = 0;
    std::size_t fg_gate_dropped = 0, fg_gate_in = 0;   // foreground depth-gate telemetry (this frame)
    int         fg_gate_masks   = 0;
    support_offsets.push_back(0.0f);
    mask_pixel_offsets.push_back(0.0f);

    const std::size_t mask_stride = std::max<std::size_t>(1, params_.VOXEL_DECIMATION_FACTOR);

    for (std::size_t det_idx = 0; det_idx < detections.size(); ++det_idx)
    {
        const auto& det = detections[det_idx];
        if (det.mask.empty())
            continue;

        cv::Mat mask_bin;
        cv::threshold(det.mask, mask_bin, 127, 255, cv::THRESH_BINARY);
        const int min_x = std::max(0, det.bbox.x);
        const int min_y = std::max(0, det.bbox.y);
        const int max_x = std::min(rgbd.width, det.bbox.x + det.bbox.width);
        const int max_y = std::min(rgbd.height, det.bbox.y + det.bbox.height);

        // Pass 1: gather ALL masked room points with valid depth. No depth/range gate — every
        // masked pixel that deprojects is kept (radius outlier removal below still trims the
        // sparse silhouette-edge tail).
        std::vector<Eigen::Vector3f> gated;   // CAMERA (zed) frame — the only frame this path knows
        gated.reserve(static_cast<std::size_t>(det.bbox.area() / std::max<int>(1, static_cast<int>(mask_stride * mask_stride))));
        std::vector<float> det_pixels;   // raw 2D foreground (col,row), depth-independent

        for (int row = min_y; row < max_y; row += static_cast<int>(mask_stride))
        {
            for (int col = min_x; col < max_x; col += static_cast<int>(mask_stride))
            {
                if (mask_bin.at<std::uint8_t>(row, col) == 0)
                    continue;

                det_pixels.push_back(static_cast<float>(col));
                det_pixels.push_back(static_cast<float>(row));

                const std::size_t depth_idx = static_cast<std::size_t>(row * rgbd.width + col);
                if (depth_idx >= rgbd.depth.size())
                    continue;

                const float depth = rgbd.depth[depth_idx];
                if (!std::isfinite(depth) || depth <= 0.0f)
                    continue;

                const float px = (static_cast<float>(col) - cx) * depth / fx;
                const float py = depth;
                const float pz = (cy - static_cast<float>(row)) * depth / fy;
                gated.emplace_back(px, py, pz);
            }
        }

        if (gated.empty())
            continue;

        // Radius outlier removal: keep points that have enough neighbours nearby. The dense
        // object body survives; the sparse silhouette-edge tail is trimmed. Bucketed into a
        // uniform spatial hash grid (cell = MASK_OUTLIER_RADIUS_M) instead of all-pairs distance
        // checks: an isolated silhouette-edge point — exactly the case this filter is meant to
        // discard — used to scan the ENTIRE point set before concluding it has too few
        // neighbours (worst-case O(n²) on the very points it removes); the grid only visits the
        // 3x3x3 neighbouring cells, so cost tracks local density instead of total point count.
        // Survivors of the outlier + foreground gates, CAMERA frame. Both gates are rigid-transform
        // invariant — the outlier test is a distance, the foreground test is camera depth — so running
        // them here rather than on room-transformed copies selects exactly the same points.
        std::vector<Eigen::Vector3f> mask_points;
        if (params_.MASK_OUTLIER_MIN_NEIGHBORS > 0 and params_.MASK_OUTLIER_RADIUS_M > 0.0f
            and gated.size() > static_cast<std::size_t>(params_.MASK_OUTLIER_MIN_NEIGHBORS))
        {
            const float radius = params_.MASK_OUTLIER_RADIUS_M;
            const float r2 = radius * radius;
            const int min_neighbours = params_.MASK_OUTLIER_MIN_NEIGHBORS;

            const auto cell_of = [radius](const Eigen::Vector3f& p) -> std::array<int, 3>
            {
                return {static_cast<int>(std::floor(p.x() / radius)),
                        static_cast<int>(std::floor(p.y() / radius)),
                        static_cast<int>(std::floor(p.z() / radius))};
            };
            struct CellHash
            {
                std::size_t operator()(const std::array<int, 3>& k) const noexcept
                {
                    std::size_t h = static_cast<std::size_t>(k[0]);
                    h = h * 1000003u ^ static_cast<std::size_t>(k[1]);
                    h = h * 1000003u ^ static_cast<std::size_t>(k[2]);
                    return h;
                }
            };

            std::unordered_map<std::array<int, 3>, std::vector<std::size_t>, CellHash> grid;
            grid.reserve(gated.size());
            for (std::size_t i = 0; i < gated.size(); ++i)
                grid[cell_of(gated[i])].push_back(i);

            mask_points.reserve(gated.size());
            for (std::size_t i = 0; i < gated.size(); ++i)
            {
                const auto base = cell_of(gated[i]);
                int neighbours = 0;
                for (int dx = -1; dx <= 1 and neighbours < min_neighbours; ++dx)
                    for (int dy = -1; dy <= 1 and neighbours < min_neighbours; ++dy)
                        for (int dz = -1; dz <= 1 and neighbours < min_neighbours; ++dz)
                        {
                            const auto it = grid.find({base[0] + dx, base[1] + dy, base[2] + dz});
                            if (it == grid.end())
                                continue;
                            for (const std::size_t j : it->second)
                            {
                                if (j == i or (gated[i] - gated[j]).squaredNorm() > r2)
                                    continue;
                                if (++neighbours >= min_neighbours)
                                    break;
                            }
                        }
                if (neighbours >= min_neighbours)
                    mask_points.push_back(gated[i]);
            }
        }
        else
            mask_points = std::move(gated);

        if (mask_points.empty())
            continue;

        // FOREGROUND depth gate — kills the WALL-BEHIND points that bleed into the slice when the robot
        // moves: the RGB mask and depth map skew, so silhouette-edge pixels sample the background depth and
        // deproject onto the far wall. That's a COHERENT cluster the radius filter above can't remove. Anchor
        // on the near surface (low percentile of camera-frame depth py; zed frame y = depth) and drop points
        // more than MASK_DEPTH_GATE_BAND_M behind it — the object occludes the background, so those far
        // returns can't be the object.
        // ⚠ TEMPORARY HARD THRESHOLD — a cliff that silently amputates any object deeper than the band (a 2 m
        // table along its length loses its far edge). Band widened to 3.0 m as a stopgap. The real fix is
        // model-conditioned association: let the concept fitter accept/reject points by its own surface
        // hypothesis (SAM2 already tightens the edge upstream). REMOVE this whole block once that is tested
        // live — do NOT keep growing the number.
        if (params_.MASK_DEPTH_GATE_BAND_M > 0.0f and mask_points.size() >= 8)
        {
            std::vector<float> depths;
            depths.reserve(mask_points.size());
            for (const auto& pc : mask_points) depths.push_back(pc.y());   // zed camera y = depth along axis
            const std::size_t knear = depths.size() / 10;                   // ~10th-pct = near surface
            std::nth_element(depths.begin(), depths.begin() + knear, depths.end());
            const float d_max = depths[knear] + params_.MASK_DEPTH_GATE_BAND_M;
            std::vector<Eigen::Vector3f> fg;
            fg.reserve(mask_points.size());
            for (const auto& p : mask_points)
                if (p.y() <= d_max)
                    fg.push_back(p);
            if (const std::size_t dropped = mask_points.size() - fg.size(); dropped > 0)
            {
                fg_gate_dropped += dropped;
                fg_gate_in      += mask_points.size();
                ++fg_gate_masks;
            }
            mask_points = std::move(fg);
            if (mask_points.empty())
                continue;
        }

        Eigen::Vector3f min_pt = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
        Eigen::Vector3f max_pt = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
        Eigen::Vector3f sum_pt = Eigen::Vector3f::Zero();
        for (const auto& p : mask_points)
        {
            sum_pt += p;
            min_pt = min_pt.cwiseMin(p);
            max_pt = max_pt.cwiseMax(p);
        }

        if (!labels_joined.str().empty())
            labels_joined << '|';
        labels_joined << det.label;

        label_ids.push_back(static_cast<float>(det.class_id));
        confidences.push_back(det.confidence);
        has_depth_flags.push_back(1.0f);   // zed slice: carries 3D support points
        mask_source.push_back(0.0f);       // source = zed
        azimuths.push_back(0.0f);
        depth_var.push_back(0.0f);          // zed dense depth ⇒ no extra range variance beyond R_base
        // Appearance summary from the FULL-RES silhouette (mask_bin), not the decimated det_pixels: the
        // interior-cell test needs true occupancy to erode the boundary correctly. Disabled or too-small
        // masks yield n_eff==0, which simply carries no information downstream.
        {
            const auto cs = params_.MASK_COLOR_ENABLED
                          ? mask_color_summary(rgbd.bgr, mask_bin, min_x, min_y, max_x, max_y,
                                               params_.MASK_COLOR_CELL_PX)
                          : MaskColorSummary{};
            color_rgb.insert(color_rgb.end(), {cs.chroma.x(), cs.chroma.y(), cs.chroma.z()});
            color_var.insert(color_var.end(), {cs.var.x(), cs.var.y(), cs.var.z()});
            color_neff.push_back(cs.n_eff);
        }
        support_offsets.push_back(static_cast<float>(support_offsets.back() + static_cast<float>(mask_points.size())));
        mask_pixels_xy.insert(mask_pixels_xy.end(), det_pixels.begin(), det_pixels.end());
        mask_pixel_offsets.push_back(static_cast<float>(mask_pixel_offsets.back() + static_cast<float>(det_pixels.size() / 2)));

        const Eigen::Vector3f centroid = sum_pt / static_cast<float>(mask_points.size());
        centroids_xyz.push_back(centroid.x());
        centroids_xyz.push_back(centroid.y());
        centroids_xyz.push_back(centroid.z());

        bbox_min_xyz.push_back(min_pt.x());
        bbox_min_xyz.push_back(min_pt.y());
        bbox_min_xyz.push_back(min_pt.z());
        bbox_max_xyz.push_back(max_pt.x());
        bbox_max_xyz.push_back(max_pt.y());
        bbox_max_xyz.push_back(max_pt.z());

        for (const auto& point : mask_points)
        {
            support_points_cam.push_back(point.x());
            support_points_cam.push_back(point.y());
            support_points_cam.push_back(point.z());
        }

        if (params_.MASK_MOTION_ENABLED)
        {
            // Pixel centroid + truncation from the raw 2D foreground (det_pixels = col,row, depth-free).
            float sum_col = 0.0f, sum_row = 0.0f;
            int on_border = 0;
            const std::size_t n_px = det_pixels.size() / 2;
            for (std::size_t i = 0; i + 1 < det_pixels.size(); i += 2)
            {
                const float c = det_pixels[i], r = det_pixels[i + 1];
                sum_col += c;
                sum_row += r;
                if (c <= 0.0f or r <= 0.0f
                    or c >= static_cast<float>(rgbd.width - 1) or r >= static_cast<float>(rgbd.height - 1))
                    ++on_border;
            }
            const float tau = (n_px > 0) ? static_cast<float>(on_border) / static_cast<float>(n_px) : 0.0f;
            // Mean optical-axis depth from the camera-frame survivors (.y() is depth in the zed frame).
            float sum_depth = 0.0f;
            for (const auto& p : mask_points)
                sum_depth += p.y();
            const float Z = mask_points.empty() ? 0.0f : sum_depth / static_cast<float>(mask_points.size());
            const float col_c = (n_px > 0) ? sum_col / static_cast<float>(n_px) : cx;
            const float row_c = (n_px > 0) ? sum_row / static_cast<float>(n_px) : cy;
            const float xn = (col_c - cx) / fx;
            const float yn = (row_c - cy) / fy;
            const auto corr = rc::motion::mask_corruption(cam_twist_optical, xn, yn, Z, corruption_params);
            motion_dotd.push_back(corr.dot_d);
            motion_bias.push_back(corr.bias_m);
            motion_var.push_back(corr.var_m);
            trunc_frac.push_back(tau);
            const float radius = std::sqrt(xn * xn + yn * yn);
            centroid_radius.push_back(radius);
            mask_range.push_back(Z);   // static range: distance carries info loss even at zero motion

            if (motion_csv_.is_open())
                motion_csv_ << sensing_frame << ',' << frame_ts_ms << ',' << frame_dt_s << ','
                            << rt_lag_s << ',' << rt_gap_s << ',' << v_norm << ',' << w_norm << ','
                            << det.label << ',' << Z << ',' << xn << ',' << yn << ',' << radius << ','
                            << tau << ',' << corr.dot_d << ',' << corr.bias_m << ',' << corr.var_m << '\n';
        }

        total_support_points += mask_points.size();
    }

    // Foreground-gate telemetry (throttled ≈1s): far wall-behind points removed this frame + masks affected.
    // Only prints when the gate actually dropped something, so a clean scene stays quiet. Tune the band from
    // these numbers: steady large drops on a static scene ⇒ band too tight (clipping the object).
    {
        static std::size_t fg_call_no = 0, fg_last_print = 0;
        ++fg_call_no;
        if (params_.VERBOSE_DEBUG and fg_gate_dropped > 0 and (fg_call_no - fg_last_print) >= 20)
        {
            fg_last_print = fg_call_no;
            std::println("[mask-fg-gate] dropped {}/{} pts across {} zed mask(s) (band {:.2f}m, frame={})",
                         fg_gate_dropped, fg_gate_in, fg_gate_masks, params_.MASK_DEPTH_GATE_BAND_M, sensing_frame);
        }
    }

    // Append the RGB-360 (ricoh) bearing-only detections as NO-DEPTH slices in the SAME node (Part B, see
    // RICOH_360_PERIPHERAL_DETECTION.md). They carry a room-frame bearing instead of 3D points:
    // support_begin==support_end (no points), no raw pixels, NaN centroid/bbox. Consumers skip !has_depth
    // slices until the bearing confirm/birth path (Part C) exists — inert-but-available today.
    // (Only emitted alongside a valid zed depth frame; the empty-depth early-return above drops them for
    //  that rare/transient frame, which is fine for peripheral evidence.)
    // Zed→room inverse, to also express any lidar-depth ricoh support points in the ZED frame so they
    // flow through the consumer's existing zed→target chain (no ingestor frame change / no camera tag).
    const Eigen::Matrix3f zed_R = room_T_zed.linear().cast<float>();
    const Eigen::Vector3f zed_t = room_T_zed.translation().cast<float>();
    const Eigen::Matrix3f zed_R_inv = zed_R.transpose();               // room→zed rotation
    const auto room_to_zed = [&](const Eigen::Vector3f& p_room) { return zed_R_inv * (p_room - zed_t); };

    for (const auto& b : bearing_detections)
    {
        if (!labels_joined.str().empty())
            labels_joined << '|';
        labels_joined << b.label;
        label_ids.push_back(b.class_id);
        confidences.push_back(b.confidence);
        mask_source.push_back(1.0f);       // source = ricoh (both bearing-only and lidar-depth ricoh slices)
        azimuths.push_back(b.azimuth_room_rad);
        mask_pixel_offsets.push_back(mask_pixel_offsets.back());    // no raw silhouette pixels either way
        // A ricoh slice is detected on the 360 panorama, not on this zed frame, so there are no pixels here
        // to summarise. n_eff==0 says exactly that — no colour information — keeping the arrays 1-to-1.
        color_rgb.insert(color_rgb.end(), {0.0f, 0.0f, 0.0f});
        color_var.insert(color_var.end(), {0.0f, 0.0f, 0.0f});
        color_neff.push_back(0.0f);

        if (b.has_lidar_depth and not b.support_room.empty())
        {
            // FULL 3D slice from the reprojected lidar. The hits arrive room-framed (they are built on the
            // main thread from the lidar cloud), so they are converted to ZED here — the published frame —
            // and the centroid/bbox are accumulated in that same frame, never in the room.
            has_depth_flags.push_back(1.0f);
            depth_var.push_back(b.range_var);
            support_offsets.push_back(static_cast<float>(support_offsets.back() + static_cast<float>(b.support_room.size())));
            Eigen::Vector3f mn  = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
            Eigen::Vector3f mx  = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            for (const auto& pr : b.support_room)
            {
                const Eigen::Vector3f pz = room_to_zed(pr);
                support_points_cam.push_back(pz.x()); support_points_cam.push_back(pz.y()); support_points_cam.push_back(pz.z());
                mn = mn.cwiseMin(pz); mx = mx.cwiseMax(pz); sum += pz;
            }
            const Eigen::Vector3f c_zed = sum / static_cast<float>(b.support_room.size());
            centroids_xyz.insert(centroids_xyz.end(), {c_zed.x(), c_zed.y(), c_zed.z()});
            bbox_min_xyz.insert(bbox_min_xyz.end(),   {mn.x(), mn.y(), mn.z()});
            bbox_max_xyz.insert(bbox_max_xyz.end(),   {mx.x(), mx.y(), mx.z()});
            if (params_.MASK_MOTION_ENABLED)
            {
                // No zed-image interaction matrix for a ricoh mask ⇒ neutral motion terms; range_var
                // (depth_var) carries the ricoh-specific uncertainty. mask_range = the depth centroid range.
                motion_dotd.push_back(0.0f);  motion_bias.push_back(0.0f);  motion_var.push_back(0.0f);
                trunc_frac.push_back(0.0f);   centroid_radius.push_back(0.0f);
                mask_range.push_back(b.centroid_room.norm());
            }
        }
        else
        {
            // Bearing-only slice (no lidar depth): unchanged Part-B behaviour.
            has_depth_flags.push_back(0.0f);
            depth_var.push_back(0.0f);
            support_offsets.push_back(support_offsets.back());     // no 3D points → begin == end
            const float nan = std::numeric_limits<float>::quiet_NaN();
            centroids_xyz.insert(centroids_xyz.end(), {nan, nan, nan});
            bbox_min_xyz.insert(bbox_min_xyz.end(),   {nan, nan, nan});
            bbox_max_xyz.insert(bbox_max_xyz.end(),   {nan, nan, nan});
            if (params_.MASK_MOTION_ENABLED)
            {
                motion_dotd.push_back(0.0f);  motion_bias.push_back(0.0f);      motion_var.push_back(0.0f);
                trunc_frac.push_back(0.0f);   centroid_radius.push_back(0.0f);  mask_range.push_back(0.0f);
            }
        }
    }

    const int mask_count = static_cast<int>(label_ids.size());
    G_->add_or_modify_attrib_local<mask_frame_id_att>(masks_node, static_cast<int>(sensing_frame));
    G_->add_or_modify_attrib_local<mask_timestamp_ms_att>(masks_node, frame_ts_ms);
    G_->add_or_modify_attrib_local<mask_count_att>(masks_node, mask_count);
    G_->add_or_modify_attrib_local<mask_labels_att>(masks_node, labels_joined.str());
    G_->add_or_modify_attrib_local<mask_label_ids_att>(masks_node, label_ids);
    G_->add_or_modify_attrib_local<mask_confidences_att>(masks_node, confidences);
    G_->add_or_modify_attrib_local<mask_support_offsets_att>(masks_node, support_offsets);
    G_->add_or_modify_attrib_local<mask_support_points_cam_att>(masks_node, support_points_cam);
    G_->add_or_modify_attrib_local<mask_centroids_xyz_att>(masks_node, centroids_xyz);
    G_->add_or_modify_attrib_local<mask_bbox_min_xyz_att>(masks_node, bbox_min_xyz);
    G_->add_or_modify_attrib_local<mask_bbox_max_xyz_att>(masks_node, bbox_max_xyz);
    G_->add_or_modify_attrib_local<mask_pixels_xy_att>(masks_node, mask_pixels_xy);
    G_->add_or_modify_attrib_local<mask_pixel_offsets_att>(masks_node, mask_pixel_offsets);
    G_->add_or_modify_attrib_local<mask_has_depth_att>(masks_node, has_depth_flags);
    G_->add_or_modify_attrib_local<mask_source_att>(masks_node, mask_source);
    G_->add_or_modify_attrib_local<mask_azimuth_att>(masks_node, azimuths);
    G_->add_or_modify_attrib_local<mask_depth_var_att>(masks_node, depth_var);
    if (params_.MASK_COLOR_ENABLED)
    {
        G_->add_or_modify_attrib_local<mask_color_rgb_att>(masks_node, color_rgb);
        G_->add_or_modify_attrib_local<mask_color_var_att>(masks_node, color_var);
        G_->add_or_modify_attrib_local<mask_color_neff_att>(masks_node, color_neff);
    }
    if (params_.MASK_MOTION_ENABLED)
    {
        G_->add_or_modify_attrib_local<mask_motion_dotd_att>(masks_node, motion_dotd);
        G_->add_or_modify_attrib_local<mask_motion_bias_att>(masks_node, motion_bias);
        G_->add_or_modify_attrib_local<mask_motion_var_att>(masks_node, motion_var);
        G_->add_or_modify_attrib_local<mask_trunc_frac_att>(masks_node, trunc_frac);
        G_->add_or_modify_attrib_local<mask_centroid_radius_att>(masks_node, centroid_radius);
        G_->add_or_modify_attrib_local<mask_range_att>(masks_node, mask_range);
        const std::vector<float> twist_flat{
            cam_twist_optical.v.x(), cam_twist_optical.v.y(), cam_twist_optical.v.z(),
            cam_twist_optical.w.x(), cam_twist_optical.w.y(), cam_twist_optical.w.z()};
        G_->add_or_modify_attrib_local<mask_cam_twist_att>(masks_node, twist_flat);
        G_->add_or_modify_attrib_local<mask_frame_dt_s_att>(masks_node, frame_dt_s);
        G_->add_or_modify_attrib_local<mask_rt_lag_s_att>(masks_node, rt_lag_s);
        G_->add_or_modify_attrib_local<mask_rt_gap_s_att>(masks_node, rt_gap_s);
    }
    G_->update_node(std::move(masks_node));

    // Remember this zed pose for the next frame's twist finite-difference (only with a valid capture stamp).
    if (params_.MASK_MOTION_ENABLED and frame_ts_ms > 0)
    {
        last_zed_R_     = room_rotation;
        last_zed_t_     = room_translation;
        last_zed_ts_ms_ = frame_ts_ms;
        have_last_zed_  = true;
    }
    if (motion_csv_.is_open())
        motion_csv_.flush();

    if (params_.VERBOSE_DEBUG)
        qInfo() << "[Masks] Uploaded" << mask_count << "masks and" << total_support_points << "support points to DSR (frame=" << sensing_frame << ")";
}

void GraphPublisher::publish_skeletons(const RGBDData& rgbd,
                                       const std::vector<rc::human_pose::PoseDetection>& poses,
                                       std::uint64_t frame_ts_ms)
{
    if (ensure_node("skeleton", "MediumAquamarine", skeleton_ready_, /*relayout=*/true))
        upload_skeletons(rgbd, poses, frame_ts_ms);
}

// Deproject YOLO-pose keypoints to BODY_18 3D in the CAMERA (zed) frame and publish on the
// 'skeleton' node. Layout is fixed-stride (18 joints/body) so consumers slice by body index —
// no offsets array. Missing/low-confidence joints are written as NaN with conf 0. Coordinates are
// in metres, camera frame (x right, y forward/depth, z up) — the SAME convention upload_masks uses
// before its room transform, just stopped one step earlier. Re-parenting into room/robot frame is
// the consumer's job (human_concept owns the person node + its RT edge).
void GraphPublisher::upload_skeletons(const RGBDData& rgbd,
                                      const std::vector<rc::human_pose::PoseDetection>& poses,
                                      std::uint64_t frame_ts_ms)
{
    namespace hp = rc::human_pose;

    const std::uint64_t sensing_frame = ++skeleton_publish_seq_;
    if (sensing_frame == last_skeleton_uploaded_frame_)
        return;
    last_skeleton_uploaded_frame_ = sensing_frame;

    auto node_opt = G_->get_node("skeleton");
    if (!node_opt.has_value())
        return;
    auto& node = node_opt.value();

    const float fx = rgbd.focal_x;
    const float fy = rgbd.focal_y;
    const float cx = static_cast<float>(rgbd.width) * 0.5f;
    const float cy = static_cast<float>(rgbd.height) * 0.5f;
    const float NaN = std::numeric_limits<float>::quiet_NaN();
    const float kp_conf_min = params_.SKELETON_KP_CONF_MIN;
    const int   patch = std::max(0, params_.SKELETON_DEPTH_PATCH);

    // Median depth over a (2*patch+1) window around (col,row); 0 if no valid sample. A single-pixel
    // depth on a thin limb is often missing/noisy, so a small patch stabilises the joint range.
    auto sample_depth = [&](int col, int row) -> float
    {
        std::vector<float> vals;
        vals.reserve(static_cast<std::size_t>((2 * patch + 1) * (2 * patch + 1)));
        for (int r = row - patch; r <= row + patch; ++r)
            for (int c = col - patch; c <= col + patch; ++c)
            {
                if (r < 0 || c < 0 || r >= rgbd.height || c >= rgbd.width)
                    continue;
                const std::size_t idx = static_cast<std::size_t>(r) * rgbd.width + static_cast<std::size_t>(c);
                if (idx >= rgbd.depth.size())
                    continue;
                const float d = rgbd.depth[idx];
                if (std::isfinite(d) && d > 0.0f)
                    vals.push_back(d);
            }
        if (vals.empty())
            return 0.0f;
        std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
        return vals[vals.size() / 2];
    };

    // Deproject one COCO keypoint to camera-frame metres; conf gates validity. Returns false if no
    // usable depth or below the confidence floor (caller writes NaN/0).
    auto deproject_coco = [&](const hp::PoseKp& kp, Eigen::Vector3f& out) -> bool
    {
        if (kp.conf < kp_conf_min)
            return false;
        const int col = static_cast<int>(std::lround(kp.x));
        const int row = static_cast<int>(std::lround(kp.y));
        if (col < 0 || row < 0 || col >= rgbd.width || row >= rgbd.height)
            return false;
        const float depth = sample_depth(col, row);
        if (depth <= 0.0f)
            return false;
        out = Eigen::Vector3f((static_cast<float>(col) - cx) * depth / fx,
                              depth,
                              (cy - static_cast<float>(row)) * depth / fy);
        return true;
    };

    std::vector<float> ids;
    std::vector<float> kp_xyz;    // count * 18 * 3, camera frame metres (NaN = missing)
    std::vector<float> kp_conf;   // count * 18, [0,1]
    ids.reserve(poses.size());
    kp_xyz.reserve(poses.size() * hp::NUM_BODY18 * 3);
    kp_conf.reserve(poses.size() * hp::NUM_BODY18);

    int written = 0;
    for (std::size_t p = 0; p < poses.size(); ++p)
    {
        const auto& pose = poses[p];

        // Deproject the 17 COCO joints first; BODY_18 then reorders + synthesizes the neck in 3D.
        std::array<Eigen::Vector3f, hp::NUM_COCO_KP> coco_xyz;
        std::array<bool,  hp::NUM_COCO_KP> coco_ok{};
        for (int k = 0; k < hp::NUM_COCO_KP; ++k)
            coco_ok[static_cast<std::size_t>(k)] = deproject_coco(pose.kp[static_cast<std::size_t>(k)],
                                                                  coco_xyz[static_cast<std::size_t>(k)]);

        ids.push_back(pose.track_id >= 0 ? static_cast<float>(pose.track_id) : static_cast<float>(p));

        for (int b = 0; b < hp::NUM_BODY18; ++b)
        {
            const int src = hp::BODY18_FROM_COCO[static_cast<std::size_t>(b)];
            if (src < 0)  // neck = 3D midpoint of shoulders (valid only if both shoulders are)
            {
                const int ls = hp::COCO_L_SHOULDER, rs = hp::COCO_R_SHOULDER;
                if (coco_ok[ls] && coco_ok[rs])
                {
                    const Eigen::Vector3f neck = 0.5f * (coco_xyz[ls] + coco_xyz[rs]);
                    kp_xyz.insert(kp_xyz.end(), {neck.x(), neck.y(), neck.z()});
                    kp_conf.push_back(std::min(pose.kp[ls].conf, pose.kp[rs].conf));
                }
                else
                {
                    kp_xyz.insert(kp_xyz.end(), {NaN, NaN, NaN});
                    kp_conf.push_back(0.0f);
                }
                continue;
            }
            if (coco_ok[static_cast<std::size_t>(src)])
            {
                const Eigen::Vector3f& v = coco_xyz[static_cast<std::size_t>(src)];
                kp_xyz.insert(kp_xyz.end(), {v.x(), v.y(), v.z()});
                kp_conf.push_back(pose.kp[static_cast<std::size_t>(src)].conf);
            }
            else
            {
                kp_xyz.insert(kp_xyz.end(), {NaN, NaN, NaN});
                kp_conf.push_back(0.0f);
            }
        }
        ++written;
    }

    G_->add_or_modify_attrib_local<skeleton_frame_id_att>(node, static_cast<int>(sensing_frame));
    G_->add_or_modify_attrib_local<skeleton_timestamp_ms_att>(node, frame_ts_ms);
    G_->add_or_modify_attrib_local<skeleton_count_att>(node, written);
    G_->add_or_modify_attrib_local<skeleton_ids_att>(node, ids);
    G_->add_or_modify_attrib_local<skeleton_kp_xyz_att>(node, kp_xyz);
    G_->add_or_modify_attrib_local<skeleton_kp_conf_att>(node, kp_conf);
    G_->update_node(std::move(node));

    // Detection signal: log when the people count changes (so you SEE 0→1 when someone steps in),
    // and always under verbose. valid_joints = keypoints that got a real depth (the rest are NaN).
    const long valid_joints = std::count_if(kp_conf.begin(), kp_conf.end(), [](float c){ return c > 0.0f; });
    // Log on a count change (0↔N transitions), plus a periodic heartbeat (~every 30 frames ≈ 3s @10Hz)
    // while anyone is present so the stream stays visible in steady state. std::cout (not qInfo) so it
    // shows on the same channel as the detector banners — RoboComp typically filters Qt info messages.
    const bool changed   = (written != last_logged_skeleton_count_);
    const bool heartbeat = (written > 0 and sensing_frame % 30 == 0);
    if (changed or heartbeat or params_.VERBOSE_DEBUG)
    {
        std::println("[Skeleton] {} people, {} of {} joints with depth (camera frame, frame={})",
                     written, valid_joints, written * 18, sensing_frame);
        last_logged_skeleton_count_ = written;
    }
}

void GraphPublisher::cleanup_semantic_grid_nodes()
{
    for (const auto& node : G_->get_nodes_by_type("semantic_grid"))
    {
        qInfo() << "[GraphPublisher] Removing stale '" << node.name().c_str() << "' node (id=" << node.id() << ") from DSR graph";
        G_->delete_node(node.id());
    }
    masks_ready_    = false;
    skeleton_ready_ = false;
}
