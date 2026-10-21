// input_snapshot.cpp — see input_snapshot.h. Every read here mirrors a row of the enumeration in
// .superpowers/sdd/2026-10-21-door-core-replay/task-2-report.md §1a; the row's call site is named where it helps.
#include "input_snapshot.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <dsr/api/dsr_camera_api.h>
#include "../../common/mask_ingestor/mask_ingestor.h"
#include "../../common/lidar_ingestor/concept_lidar_ingestor.h"
#include "../../common/room_resolve/room_resolve.h"
#include "../../common/rt_twist/rt_twist.h"
#include "../../common/interaction_anchor/interaction_anchor.h"
#include "door_semantic_field.h"

namespace rc::door
{
namespace
{
using core::Mat4d;
using core::Mat6d;

std::int64_t steady_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::optional<Mat4d> tf(DSR::InnerEigenAPI* inner, const std::string& dst, const std::string& src, std::uint64_t ts = 0,
                        DSR::RT_API::TimeQuery q = DSR::RT_API::TimeQuery::Nearest)
{
    if (inner == nullptr or dst.empty() or src.empty()) return std::nullopt;
    const auto T = ts == 0 ? inner->get_transformation_matrix(dst, src, 0)
                           : inner->get_transformation_matrix(dst, src, ts, "RT", q);
    if (not T.has_value()) return std::nullopt;   // ALWAYS check the optional (CLAUDE.md)
    return T->matrix();
}

std::optional<core::RtEdge> rt_edge(DSR::DSRGraph& G, std::uint64_t from, std::uint64_t to)
{
    const auto e = G.get_edge(from, to, "RT");
    if (not e.has_value()) return std::nullopt;
    const auto tr  = G.get_attrib_by_name<rt_translation_att>(e.value());
    const auto rot = G.get_attrib_by_name<rt_rotation_euler_xyz_att>(e.value());
    core::RtEdge r;
    r.parent = from;
    if (tr.has_value() and tr->get().size() >= 3) r.translation = {tr->get()[0], tr->get()[1], tr->get()[2]};
    if (rot.has_value() and rot->get().size() >= 3) r.rotation_euler_xyz = {rot->get()[0], rot->get()[1], rot->get()[2]};
    return r;
}

core::ObjectInfo object_info(DSR::DSRGraph& G, DSR::InnerEigenAPI* inner, const DSR::Node& n, const std::string& room_frame)
{
    core::ObjectInfo o;
    o.id = n.id();
    o.name = n.name();
    o.type = n.type();
    if (const auto p = G.get_attrib_by_name<parent_att>(n); p.has_value()) o.parent = p.value();
    if (const auto v = G.get_attrib_by_name<width_m_att>(n); v.has_value())  o.width_m = v.value();
    if (const auto v = G.get_attrib_by_name<depth_m_att>(n); v.has_value())  o.depth_m = v.value();
    if (const auto v = G.get_attrib_by_name<height_m_att>(n); v.has_value()) o.height_m = v.value();
    o.room_T_node = tf(inner, room_frame, n.name());
    o.is_anchor = rc::interaction::is_anchor(G, n);
    if (const auto v = G.get_attrib_by_name<epistemic_pending_att>(n); v.has_value()) o.epistemic_pending = v.value();
    if (const auto v = G.get_attrib_by_name<door_open_prob_att>(n); v.has_value())   o.door_open_prob = v.value();
    if (const auto v = G.get_attrib_by_name<door_phi_rad_att>(n); v.has_value())     o.door_phi_rad = v.value();
    return o;
}

core::MaskSlice to_core(const rc::MaskIngestor::MaskSlice& s)
{
    core::MaskSlice o;
    o.label = s.label; o.class_id = s.class_id; o.confidence = s.confidence;
    o.centroid = s.centroid; o.bbox_min = s.bbox_min; o.bbox_max = s.bbox_max;
    o.support_begin = s.support_begin; o.support_end = s.support_end;
    o.pixel_begin = s.pixel_begin; o.pixel_end = s.pixel_end;
    o.motion_var = s.motion_var; o.motion_bias = s.motion_bias; o.motion_dotd = s.motion_dotd;
    o.trunc_frac = s.trunc_frac; o.centroid_radius = s.centroid_radius; o.range = s.range;
    o.has_depth = s.has_depth; o.azimuth_room_rad = s.azimuth_room_rad;
    o.source = static_cast<std::uint8_t>(s.source);
    o.depth_var = s.depth_var; o.color_chroma = s.color_chroma; o.color_var = s.color_var; o.color_neff = s.color_neff;
    return o;
}

/// Node chain from `n` up to the root, by the `parent` attribute.
std::vector<DSR::Node> ancestors(DSR::DSRGraph& G, std::optional<DSR::Node> n)
{
    std::vector<DSR::Node> out;
    for (int guard = 0; n.has_value() and guard < 64; ++guard)
    {
        out.push_back(n.value());
        n = G.get_parent_node(n.value());
    }
    return out;
}

/// room ← zed as the RT steps InnerGaussianAPI composes (door_fitter's chain covariance, Part B), each with its
/// transform and covariance at `ts` — room side first (traversed child→parent: inverted), then down to the zed.
std::vector<core::RtChainStep> room_zed_chain(DSR::DSRGraph& G, DSR::RT_API* rt, const std::string& room_frame,
                                              std::uint64_t ts)
{
    std::vector<core::RtChainStep> steps;
    if (rt == nullptr or room_frame.empty()) return steps;
    const auto up_room = ancestors(G, G.get_node(room_frame));
    const auto up_zed  = ancestors(G, G.get_node("zed"));
    if (up_room.empty() or up_zed.empty()) return steps;
    std::size_t lz = up_zed.size(), lr = up_room.size();
    for (std::size_t i = 0; i < up_zed.size() and lz == up_zed.size(); ++i)
        for (std::size_t j = 0; j < up_room.size(); ++j)
            if (up_zed[i].id() == up_room[j].id()) { lz = i; lr = j; break; }
    if (lz == up_zed.size()) return steps;   // no common ancestor: nothing to record
    auto step = [&](const DSR::Node& parent, const DSR::Node& child, bool inverted) -> bool
    {
        const auto e = G.get_edge(parent.id(), child.id(), "RT");
        if (not e.has_value()) return false;
        const auto T = rt->get_edge_RT_as_rtmat(e.value(), ts);
        if (not T.has_value()) return false;
        core::RtChainStep s;
        s.parent = parent.name(); s.child = child.name(); s.inverted = inverted;
        s.T = T->matrix();
        if (const auto C = rt->get_edge_RT_covariance(e.value(), ts); C.has_value()) s.cov = C.value();
        steps.push_back(std::move(s));
        return true;
    };
    for (std::size_t j = 0; j < lr; ++j)
        if (not step(up_room[j + 1], up_room[j], true)) return {};
    for (std::size_t i = lz; i-- > 0;)
        if (not step(up_zed[i + 1], up_zed[i], false)) return {};
    return steps;
}

std::string slurp(const std::string& path, bool& present)
{
    std::ifstream f(path, std::ios::binary);
    present = f.is_open();
    if (not present) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
}  // namespace

// ═══ events ════════════════════════════════════════════════════════════════════════════════════════════════════════

void InputSnapshot::on_node_deleted(std::uint64_t id)
{
    if (armed_ and not recorder_.stopped())
        latch_.node_deleted(id, steady_ns());
}

void InputSnapshot::on_door_upserted(DSR::DSRGraph& G, DSR::InnerEigenAPI* inner, const DSR::Node& node)
{
    if (not armed_ or recorder_.stopped() or not is_foreign(node.id(), agent_id_)) return;
    core::DoorNodeUpserted u;
    u.node = object_info(G, inner, node, rc::room::current_room_frame(G));
    if (u.node.parent.has_value()) u.rt_parent = rt_edge(G, u.node.parent.value(), node.id());
    latch_.door_upserted(std::move(u), steady_ns());
}

void InputSnapshot::on_ui_request(bool open, const std::string& provider_id)
{
    if (armed_ and not recorder_.stopped())
        latch_.ui_request({open, provider_id}, steady_ns());
}

void InputSnapshot::on_owned_swept(const std::string& why, std::vector<std::uint64_t> removed)
{
    if (armed_ and not recorder_.stopped())
        latch_.owned_swept({why, std::move(removed)}, steady_ns());
}

void InputSnapshot::on_identity_load(const std::string& identity_path, const std::string& registration_path,
                                     std::optional<std::uint64_t> initial_room, const std::string& initial_room_name)
{
    if (not armed_) return;
    core::IdentityLoad l;
    l.csv_text = slurp(identity_path, l.file_present);
    l.registration_csv_text = slurp(registration_path, l.registration_present);
    l.initial_room = initial_room;
    l.initial_room_name = initial_room_name;
    latch_.identity_load(std::move(l));
}

// ═══ the cycle ═════════════════════════════════════════════════════════════════════════════════════════════════════

void InputSnapshot::begin_cycle(DSR::DSRGraph& G, DSR::InnerEigenAPI* inner, DSR::RT_API* rt, std::uint64_t cycle,
                                std::int64_t t_steady_ns, std::int64_t t_wall_ms, const rc::MaskIngestor* masks)
{
    if (not (core_active_ or recording())) return;
    cur_ = {};
    cur_.cycle = cycle;
    cur_.t_steady_ns = t_steady_ns;
    cur_.t_wall_ms = t_wall_ms;
    latch_.begin_cycle(cur_);
    in_cycle_ = true;
    semantic_touched_ = false;
    rgb_ = depth_ = nullptr;
    rois_rgb_.clear();
    rois_depth_.clear();

    // ── rooms (resolve_current_room / rc::room::current_room / latch_recall_names / refresh_room_geometry) ──────────
    auto& R = cur_.rooms;
    R.graph_current_room = rc::room::current_room(G);
    const auto ce = rc::room::current_edges(G);
    R.n_current_edges = ce.count;
    if (ce.count == 1) R.current_edge_room = ce.room;
    const std::string room_frame = R.graph_current_room.has_value() ? rc::room::name_of(G, *R.graph_current_room)
                                                                    : std::string{};
    for (const auto& n : G.get_nodes_by_type("metaconcept"))
        if (n.name().starts_with("recall_")
            and G.get_attrib_by_name<object_subtype_att>(n).value_or(std::string{}) == "room_recall")
            R.recall_names.push_back(n.name().substr(7));
    std::string root_name;
    if (const auto roots = G.get_nodes_by_type("root"); not roots.empty()) root_name = roots.front().name();
    for (const auto& n : G.get_nodes_by_type("room"))
    {
        core::RoomInfo r;
        r.id = n.id();
        r.name = n.name();
        r.is_proto = rc::room::is_proto(G, n.id());
        r.root_T_room = tf(inner, root_name, n.name());
        const auto px = G.get_attrib_by_name<delimiting_polygon_x_att>(n);
        const auto py = G.get_attrib_by_name<delimiting_polygon_y_att>(n);
        if (px.has_value() and py.has_value())
            for (std::size_t i = 0; i < std::min(px->get().size(), py->get().size()); ++i)
                r.polygon.emplace_back(px->get()[i], py->get()[i]);
        if (const auto lv = G.get_attrib_by_name<level_att>(n); lv.has_value()) r.level = lv.value();
        R.rooms.push_back(std::move(r));
    }
    keep_foreign(R.rooms, agent_id_);

    // ── walls (DoorSceneGraph::resolve_wall) ────────────────────────────────────────────────────────────────────────
    for (const auto& wn : G.get_nodes_by_type("wall"))
    {
        core::WallInfo w;
        w.id = wn.id();
        if (const auto p = G.get_attrib_by_name<parent_att>(wn); p.has_value())
        {
            w.parent = p.value();
            if (const auto pn = G.get_node(p.value()); pn.has_value())
                if (const auto pp = G.get_attrib_by_name<parent_att>(pn.value()); pp.has_value())
                    w.parent_parent = pp.value();
            w.rt = rt_edge(G, p.value(), wn.id());
        }
        if (w.parent_parent.has_value()) w.rt_from_room = rt_edge(G, w.parent_parent.value(), wn.id());
        if (const auto v = G.get_attrib_by_name<width_m_att>(wn); v.has_value()) w.width_m = v.value();
        cur_.walls.push_back(std::move(w));
    }
    keep_foreign(cur_.walls, agent_id_);

    // ── objects (foreign claims, NBV obstacles, the name allocator, mirrors' sources) — FOREIGN ONLY ───────────────
    // ── + the controller's flags on OUR door nodes → affordances[] ─────────────────────────────────────────────────
    for (const char* type : {"object", "box"})
        for (const auto& n : G.get_nodes_by_type(type))
        {
            if (is_foreign(n.id(), agent_id_))
                cur_.objects.push_back(object_info(G, inner, n, room_frame));
            else if (const auto v = G.get_attrib_by_name<epistemic_pending_att>(n); v.has_value())
            {
                core::AffordanceFlags f;
                f.node_id = n.id(); f.exists = true; f.name = n.name(); f.epistemic_pending = v.value();
                cur_.affordances.push_back(std::move(f));
            }
        }

    // ── our affordance nodes: the CONTROLLER's flags + our target (PragmaticAffordance::write_proposal's epoch bump) ─
    for (const auto& n : G.get_nodes_by_type("affordance"))
    {
        if (is_foreign(n.id(), agent_id_)) continue;
        core::AffordanceFlags f;
        f.node_id = n.id(); f.exists = true; f.name = n.name();
        if (const auto v = G.get_attrib_by_name<active_att>(n); v.has_value())                   f.active = v.value();
        if (const auto v = G.get_attrib_by_name<epistemic_pending_att>(n); v.has_value())        f.epistemic_pending = v.value();
        if (const auto v = G.get_attrib_by_name<epistemic_target_epoch_att>(n); v.has_value())   f.target_epoch = v.value();
        if (const auto v = G.get_attrib_by_name<epistemic_target_x_m_att>(n); v.has_value())     f.target_x = v.value();
        if (const auto v = G.get_attrib_by_name<epistemic_target_y_m_att>(n); v.has_value())     f.target_y = v.value();
        if (const auto v = G.get_attrib_by_name<epistemic_target_yaw_rad_att>(n); v.has_value()) f.target_yaw = v.value();
        cur_.affordances.push_back(std::move(f));
    }
    for (const auto& e : G.get_edges_by_type("executing"))
    {
        core::ExecutingEdge x;
        x.from = e.from(); x.to = e.to();
        if (const auto v = G.get_attrib_by_name<executing_epoch_att>(e); v.has_value())          x.epoch = v.value();
        if (const auto v = G.get_attrib_by_name<executing_target_x_m_att>(e); v.has_value())     x.x = v.value();
        if (const auto v = G.get_attrib_by_name<executing_target_y_m_att>(e); v.has_value())     x.y = v.value();
        if (const auto v = G.get_attrib_by_name<executing_target_yaw_rad_att>(e); v.has_value()) x.yaw = v.value();
        cur_.executing.push_back(x);
    }

    // ── robot (specificworker_anchor.cpp robot_node_name / P0 covariance / twist; door_fitter room_T_zed_matrix) ───
    auto& B = cur_.robot;
    if (const auto robots = G.get_nodes_by_type("robot"); not robots.empty())
    {
        const auto& rn = robots.front();
        B.present = true;
        B.name = rn.name();
        B.id = rn.id();
        B.room_T_robot = tf(inner, room_frame, rn.name());
        B.room_T_body  = tf(inner, room_frame, "body");
        B.room_T_zed   = tf(inner, room_frame, "zed");
        B.robot_T_zed  = tf(inner, rn.name(), "zed");
        B.body_T_zed   = tf(inner, "body", "zed");
        B.root_T_body  = tf(inner, root_name, "body");
        if (rt != nullptr and R.graph_current_room.has_value())
        {
            if (const auto e = G.get_edge(rn.id(), *R.graph_current_room, "RT"); e.has_value())
            {
                B.rt_cov_edge_robot_to_room = true;
                if (const auto C = rt->get_edge_RT_covariance(e.value()); C.has_value()) B.rt_cov = C.value();
            }
            else if (const auto e2 = G.get_edge(*R.graph_current_room, rn.id(), "RT"); e2.has_value())
            {
                B.rt_cov_edge_robot_to_room = false;
                if (const auto C = rt->get_edge_RT_covariance(e2.value()); C.has_value()) B.rt_cov = C.value();
            }
        }
        if (const auto tw = rc::rt::robot_body_twist(G, rn); tw.has_value())
            B.twist = core::Twist{tw->linear, tw->angular, tw->stamp_ms};
        if (const auto st = rc::rt::robot_twist_sim_stamps(G, rn); st.has_value())
            B.sim_stamps = core::SimStamps{st->first, st->second};
        if (const auto v = G.get_attrib_by_name<robot_current_speed_variance_att>(rn); v.has_value())
            B.speed_variance = v->get();
        const auto gx = G.get_attrib_by_name<robot_gt_x_att>(rn);
        const auto gy = G.get_attrib_by_name<robot_gt_y_att>(rn);
        const auto ga = G.get_attrib_by_name<robot_gt_angle_att>(rn);
        if (gx.has_value() and gy.has_value() and ga.has_value()) B.gt = core::Vec3f(gx.value(), gy.value(), ga.value());
    }

    // ── camera (door_fitter CameraAPI; anchor cam_rgb_width / focalx / fov) ────────────────────────────────────────
    if (const auto zed = G.get_node("zed"); zed.has_value())
    {
        auto& C = cur_.camera;
        if (const auto api = G.get_camera_api(zed.value()); api)
        {
            C.present = true;
            C.width = api->get_width(); C.height = api->get_height();
            C.focal_x = api->get_focal_x(); C.focal_y = api->get_focal_y();
            C.projection_model = static_cast<std::uint8_t>(api->get_projection_model());
        }
        if (const auto v = G.get_attrib_by_name<cam_rgb_width_att>(zed.value()); v.has_value())  C.cam_rgb_width = v.value();
        if (const auto v = G.get_attrib_by_name<cam_rgb_focalx_att>(zed.value()); v.has_value()) C.cam_rgb_focalx = v.value();
        if (const auto v = G.get_attrib_by_name<cam_fov_att>(zed.value()); v.has_value())        C.cam_fov = v.value();
    }

    // A refresh BETWEEN cycles (the presence pump) may have changed the packet the top of compute() reads.
    if (masks != nullptr) capture_masks(G, inner, rt, *masks);
}

void InputSnapshot::capture_masks(DSR::DSRGraph& G, DSR::InnerEigenAPI* inner, DSR::RT_API* rt, const rc::MaskIngestor& m)
{
    if (not in_cycle_) return;
    const auto& p = m.packet();
    cur_.mask_stream_ms_since_last_frame = m.ms_since_last_frame();
    const MaskKey key{p.valid, p.frame_id, p.timestamp_ms, p.slices.size(), p.support_points.size(), p.mask_pixels.size()};
    if (last_mask_.has_value() and last_mask_.value() == key) return;   // unchanged: nullopt = "keep the last one"
    last_mask_ = key;

    core::MaskPacket o;
    o.valid = p.valid;
    o.frame_id = p.frame_id;
    o.timestamp_ms = p.timestamp_ms;
    o.slices.reserve(p.slices.size());
    for (const auto& s : p.slices) o.slices.push_back(to_core(s));
    o.support_points = p.support_points;
    o.support_points_cam = p.support_points_cam;
    o.mask_pixels = p.mask_pixels;
    o.cam_twist = p.cam_twist;
    o.frame_dt_s = p.frame_dt_s;
    o.ms_since_last_frame = cur_.mask_stream_ms_since_last_frame;
    if (p.valid)
    {
        o.room_T_zed_capture = m.applied_transform();
        const std::string room_frame = rc::room::current_room_frame(G);
        std::string robot;
        if (const auto robots = G.get_nodes_by_type("robot"); not robots.empty()) robot = robots.front().name();
        // check_anchor_mask_frames: Interpolated at the stamp, ts==0 when the producer published none
        o.room_T_robot_capture = tf(inner, room_frame, robot, p.timestamp_ms, DSR::RT_API::TimeQuery::Interpolated);
        o.room_zed_chain = room_zed_chain(G, rt, room_frame, p.timestamp_ms);
    }
    cur_.masks = std::move(o);
}

void InputSnapshot::capture_bpearl(DSR::DSRGraph& G, DSR::InnerEigenAPI* inner, const rc::ConceptLidarIngestor& l)
{
    if (not in_cycle_) return;
    auto sweep = make_bpearl_sweep(l.bpearl_fresh(), l.sweep_bpearl_room(), l.origin_bpearl_room(),
                                   l.bpearl_robot_fresh(), l.sweep_bpearl_robot(), l.origin_bpearl_robot(),
                                   l.bpearl_robot_stamp_ms());
    if (not sweep.has_value()) return;
    auto& b = sweep.value();
    if (b.robot_fresh)
    {
        std::string robot;
        if (const auto robots = G.get_nodes_by_type("robot"); not robots.empty()) robot = robots.front().name();
        if (b.robot_stamp_ms > 0)   // check_anchor_ray_frames: room←robot at the capture stamp, Interpolated
            b.room_T_robot_capture = tf(inner, rc::room::current_room_frame(G), robot,
                                        static_cast<std::uint64_t>(b.robot_stamp_ms), DSR::RT_API::TimeQuery::Interpolated);
    }
    cur_.bpearl = std::move(sweep);
}

void InputSnapshot::capture_semantic(const rc::SemanticProbField& f, bool refreshed_ok)
{
    if (not in_cycle_) return;
    semantic_touched_ = true;
    if (not (refreshed_ok and f.valid()))
    {
        cur_.semantic_state = core::SemanticState::Invalid;
        cur_.semantic.reset();
        last_semantic_valid_ = false;
        last_semantic_stamp_.reset();
        return;
    }
    // Unchanged only when we are SURE: the last one was valid and carried the same non-zero stamp. Otherwise Fresh
    // (task-2-report.md §5.5: a wrong "unchanged" would silently feed stale evidence).
    const std::uint64_t st = f.stamp_ms();
    if (last_semantic_valid_ and st != 0 and last_semantic_stamp_ == st)
    {
        cur_.semantic_state = core::SemanticState::Unchanged;
        cur_.semantic.reset();
        return;
    }
    core::SemanticPlane s;
    s.pw = f.pw(); s.ph = f.ph(); s.frame_w = f.frame_w(); s.frame_h = f.frame_h();
    if (st != 0) s.stamp_ms = st;
    s.plane = f.plane();
    cur_.semantic_state = core::SemanticState::Fresh;
    cur_.semantic = std::move(s);
    last_semantic_valid_ = true;
    last_semantic_stamp_ = st;
}

void InputSnapshot::note_images(const cv::Mat* bgr, std::uint64_t rgb_stamp_ms, float frame_ref,
                                const cv::Mat* depth, std::uint64_t depth_stamp_ms, int rgb_cols, int rgb_rows)
{
    if (not in_cycle_) return;
    rgb_cols_ = rgb_cols;
    rgb_rows_ = rgb_rows;
    rgb_ = (bgr != nullptr and not bgr->empty()) ? bgr : nullptr;
    depth_ = (depth != nullptr and not depth->empty()) ? depth : nullptr;
    rgb_stamp_ = rgb_stamp_ms;
    depth_stamp_ = depth_stamp_ms;
    frame_ref_ = frame_ref;
}

void InputSnapshot::note_roi(const std::vector<cv::Point>& poly, int extra_px, bool rgb, bool depth)
{
    if (not in_cycle_ or poly.empty() or not (rgb or depth)) return;
    const int m = kRoiMarginPx + std::max(0, extra_px);
    for (std::size_t i = 0; i < poly.size(); ++i)
    {
        const cv::Point a = poly[i], b = poly[(i + 1) % poly.size()];
        const double len = std::hypot(double(b.x - a.x), double(b.y - a.y));
        const int n = std::max(1, static_cast<int>(std::ceil(len / kEdgeChunkPx)));
        for (int k = 0; k < n; ++k)
        {
            const double t0 = double(k) / n, t1 = double(k + 1) / n;
            const double xa = a.x + t0 * (b.x - a.x), ya = a.y + t0 * (b.y - a.y);
            const double xb = a.x + t1 * (b.x - a.x), yb = a.y + t1 * (b.y - a.y);
            add({static_cast<int>(std::floor(std::min(xa, xb))) - m, static_cast<int>(std::floor(std::min(ya, yb))) - m,
                 static_cast<int>(std::ceil(std::max(xa, xb))) + 1 + m, static_cast<int>(std::ceil(std::max(ya, yb))) + 1 + m},
                rgb, depth);
        }
    }
}

void InputSnapshot::note_roi(const cv::Point2f& p, int extra_px, bool rgb, bool depth)
{
    if (not in_cycle_ or not (rgb or depth) or not (std::isfinite(p.x) and std::isfinite(p.y))) return;
    const int m = kRoiMarginPx + std::max(0, extra_px);
    const int x = static_cast<int>(std::floor(p.x)), y = static_cast<int>(std::floor(p.y));
    add({x - m, y - m, x + 2 + m, y + 2 + m}, rgb, depth);
}

void InputSnapshot::add_rect(std::vector<Rect>& rois, Rect r)
{
    // Merge with a box it overlaps ONLY when the union costs no pixel the two did not already cover (union area ≤ sum
    // of the two areas): collinear chunks along a jamb fuse into one strip, a diagonal edge does not balloon into its
    // bounding box. Repeated until nothing merges; boxes that stay apart may overlap (a few duplicated pixels).
    auto area = [](const Rect& q) { return std::int64_t(q.x1 - q.x0) * (q.y1 - q.y0); };
    for (const auto& q : rois)
        if (q.x0 <= r.x0 and q.y0 <= r.y0 and r.x1 <= q.x1 and r.y1 <= q.y1) return;   // already covered
    for (bool merged = true; merged;)
    {
        merged = false;
        for (auto it = rois.begin(); it != rois.end(); ++it)
        {
            const Rect u{std::min(r.x0, it->x0), std::min(r.y0, it->y0), std::max(r.x1, it->x1), std::max(r.y1, it->y1)};
            if (area(u) <= area(r) + area(*it))
            {
                r = u;
                rois.erase(it);
                merged = true;
                break;
            }
        }
    }
    rois.push_back(r);
}

void InputSnapshot::end_cycle()
{
    if (not in_cycle_) return;
    if (not semantic_touched_)   // not refreshed this cycle: the field the code holds is whatever the last refresh left
        cur_.semantic_state = last_semantic_valid_ ? core::SemanticState::Unchanged : core::SemanticState::Invalid;

    // ── images: only if a check sampled them this cycle, cropped to what it sampled — and only for the TAPE ────────────
    const bool rec = recording();
    if (rec and not rois_rgb_.empty() and rgb_ != nullptr)
    {
        core::ImageFrame g;
        g.format = core::PixelFormat::Gray8;
        g.width = rgb_->cols; g.height = rgb_->rows;
        g.stamp_ms = rgb_stamp_;
        g.frame_ref = frame_ref_;
        for (const auto& r : rois_rgb_)
        {
            const cv::Rect cr = cv::Rect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0) & cv::Rect(0, 0, rgb_->cols, rgb_->rows);
            if (cr.area() <= 0) continue;
            cv::Mat gray;
            if (rgb_->channels() == 3) cv::cvtColor((*rgb_)(cr), gray, cv::COLOR_BGR2GRAY);   // = prepare_frame's step 1
            else gray = (*rgb_)(cr).clone();
            if (gray.type() != CV_8UC1) continue;
            if (not gray.isContinuous()) gray = gray.clone();
            core::ImagePatch p;
            p.x0 = cr.x; p.y0 = cr.y; p.w = cr.width; p.h = cr.height;
            p.bytes.assign(gray.data, gray.data + gray.total());
            g.patches.push_back(std::move(p));
        }
        if (not g.patches.empty()) cur_.rgb_gray = std::move(g);
    }
    if (rec and not rois_depth_.empty() and depth_ != nullptr and depth_->type() == CV_32FC1)
    {
        core::ImageFrame d;
        d.format = core::PixelFormat::DepthF32;   // DepthIngestor's decoded frame (metres, NaN = no return)
        d.width = depth_->cols; d.height = depth_->rows;
        d.stamp_ms = depth_stamp_;
        // ROIs are in RGB pixels; the depth reads are scaled by depth/rgb (contour_depth_support's depth_scale_*).
        // ★The SAME scale the live checks use (depth_scale_* = depth / RGB frame size, 1 when the RGB frame is empty),
        //  from the RGB frame's size passed by note_images even on a cycle whose RGB half did not run.
        const double sx = rgb_cols_ > 0 ? double(depth_->cols) / rgb_cols_ : 1.0;
        const double sy = rgb_rows_ > 0 ? double(depth_->rows) / rgb_rows_ : 1.0;
        for (const auto& r : rois_depth_)
        {
            const int x0 = int(std::floor(r.x0 * sx)) - kRoiMarginPx, y0 = int(std::floor(r.y0 * sy)) - kRoiMarginPx;
            const int x1 = int(std::ceil(r.x1 * sx)) + kRoiMarginPx,  y1 = int(std::ceil(r.y1 * sy)) + kRoiMarginPx;
            const cv::Rect cr = cv::Rect(x0, y0, x1 - x0, y1 - y0) & cv::Rect(0, 0, depth_->cols, depth_->rows);
            if (cr.area() <= 0) continue;
            const cv::Mat crop = (*depth_)(cr).clone();
            core::ImagePatch p;
            p.x0 = cr.x; p.y0 = cr.y; p.w = cr.width; p.h = cr.height;
            const auto* b = reinterpret_cast<const std::uint8_t*>(crop.data);
            p.bytes.assign(b, b + crop.total() * crop.elemSize());
            d.patches.push_back(std::move(p));
        }
        if (not d.patches.empty()) cur_.depth = std::move(d);
    }

    if (rec)
        recorder_.record(std::move(cur_));   // after the cycle (actuator transcript complete); MOVED to the writer thread
    latch_.end_cycle();
    in_cycle_ = false;
    rgb_ = depth_ = nullptr;
    rois_rgb_.clear();
    rois_depth_.clear();
    cur_ = {};
}

}  // namespace rc::door
