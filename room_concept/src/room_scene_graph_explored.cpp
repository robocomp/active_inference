/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_scene_graph.h.
 */
// P(room explored) on the room node — the belief a mission's `until explored(room)` reads (Task 5, 2026-10-21).
// The model is in room_explored.h (Luce stop-vs-look on the explorer's own price, noisy-OR with memory); this file is
// the agent's side of it: writing the attribute, restoring memory on a re-entry, and the coverage plan of a partly
// remembered room. tools/room_explored_selftest.cpp runs the same explorer calls on a synthetic room.
#include "room_scene_graph.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>

#include <QDebug>

namespace rc
{

void RoomSceneGraph::publish_explored(std::optional<float> p)
{
    if (not p.has_value() or G_ == nullptr or dsr_room_id_ == 0) return;   // not assessed: the node says nothing
    if (explored_room_id_ == dsr_room_id_ and *p == explored_p_) return;   // unchanged: no graph churn
    auto rn = G_->get_node(dsr_room_id_);
    if (not rn.has_value()) return;
    G_->add_or_modify_attrib_local<room_explored_p_att>(rn.value(), *p);
    if (not G_->update_node(rn.value())) return;
    qInfo().noquote() << QString("[room][explored] %1: P(explored) = %2 (memory %3)")
                             .arg(QString::fromStdString(rn->name())).arg(*p, 0, 'f', 3).arg(explored_.p_mem(), 0, 'f', 2);
    explored_room_id_ = dsr_room_id_;
    explored_p_ = *p;
}

std::string RoomSceneGraph::room_node_name() const
{
    if (G_ == nullptr or dsr_room_id_ == 0) return {};
    const auto rn = G_->get_node(dsr_room_id_);
    return rn.has_value() ? rn->name() : std::string{};
}

void RoomSceneGraph::note_recall_memory(const Recall& recall)
{
    resume_camera_cover(recall.name);   // before the settled publish below: an unfinished camera phase is not "explored"
    const bool had_record = explored_.memory().recall(recall.name).has_value();
    explored_.enter_recalled(recall.name, recall.explored_p);
    // The explorer's standpoints and its countable outline are the REMEMBERED outline (published frame): a re-entry
    // never ran the start-up ring that fills fs_published_ for a surveyed room.
    if (fs_published_.size() < 3) fs_published_ = published_polygon();
    qInfo().noquote() << QString("[room][explored] %1 recalled: memory restores P(explored) = %2 (%3)")
                             .arg(QString::fromStdString(recall.name)).arg(explored_.p_mem(), 0, 'f', 3)
                             .arg(recall.explored_p ? "carried by the recall"
                                  : had_record ? "this agent's record of the room"
                                  : "outline only: the promoted layout is complete");
    // A complete memory is explored on entry; a partial one is published on the explorer's first plan.
    if (not explored_.recalled_partial() and not camera_cover_open()) publish_explored(explored_.settled(recall.name));
}

float RoomSceneGraph::footprint_area() const
{
    return static_cast<float>(M_PI) * 0.25f
         * (params_->ROBOT_WIDTH * params_->ROBOT_WIDTH + params_->ROBOT_LENGTH * params_->ROBOT_LENGTH);
}

rc::layout::Plan RoomSceneGraph::plan_recalled_rest(const rc::layout::FeasibleGrid& grid, const Eigen::Vector2f& robot,
                                                    const rc::layout::ExplorerParams& ep,
                                                    const rc::layout::Admissible& standable,
                                                    const rc::layout::Admissible& deferred)
{
    // The survey's coverage gain (unseen cells inside or bordering the outline, the free-space record restarted at the
    // crossing), counted only for what memory may lack. Same price as the survey: the robot's footprint.
    const auto countable = free_space_->countable_mask(fs_published_, 0.40f);
    const float p_mem = explored_.p_mem();
    return rc::layout::plan_coverage(grid, robot, ep, standable,
        [this, &countable, p_mem](const Eigen::Vector2f& v)
        { return rc::explored::recalled_gain(p_mem, floor_nats(free_space_->visible_unknown_area(T_pub_int_ * v, 8.f, 180, &countable))); },
        floor_nats(footprint_area()), deferred);   // item A: m^2 -> nats (occupancy prior), gain and price alike
}

// ── CAMERA COVERAGE OF THE BOUNDARY (camera_coverage.h; Task 12, 2026-10-29) ─────────────────────────────────────────
void RoomSceneGraph::resolve_zed_mount()
{
    if (zed_mount_resolved_ or not G_ or not rt_api_ or dsr_robot_id_ == 0) return;
    // Same walk as resolve_lidar_origin: zed → … → robot node, read once (the mount does not move). ROBOT_GEOMETRY.md:
    // the zed frame is x-right, y-DEPTH, z-up, so its optical axis is the frame's +y column.
    auto node = G_->get_node("zed");
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    bool ok = node.has_value();
    for (int depth = 0; ok and depth < 8 and node->id() != dsr_robot_id_; ++depth)
    {
        const auto rt = rt_api_->get_RT_pose_from_parent(node.value());
        const auto parent = G_->get_attrib_by_name<parent_att>(node.value());
        if (not rt.has_value() or not parent.has_value()) { ok = false; break; }
        T = Eigen::Affine3d(rt->matrix()) * T;
        node = G_->get_node(parent.value());
        ok = node.has_value();
    }
    ok = ok and node->id() == dsr_robot_id_;
    if (ok)
    {
        zed_xy_ = T.translation().head<2>().cast<float>();
        const Eigen::Vector2f f = T.linear().col(1).head<2>().cast<float>();
        if (f.norm() > 1e-6f) zed_fwd_ = f.normalized();
        cam_sensor_.height_m = static_cast<float>(T.translation().z());
        zed_mount_resolved_ = true;
        qInfo() << "[room][camera-cover] zed mount: (" << zed_xy_.x() << "," << zed_xy_.y() << ") m, axis ("
                << zed_fwd_.x() << "," << zed_fwd_.y() << "), height" << cam_sensor_.height_m << "m in the robot frame";
    }
    else if (++zed_mount_tries_ == 200 and not camera_model_warned_)
    {
        camera_model_warned_ = true;
        qWarning() << "[room][camera-cover] could not walk zed -> robot in the RT tree: NO camera coverage of the room's"
                   << "boundary — rooms will be called explored on the LiDAR alone (doors on unseen walls may be missed)";
    }
}

void RoomSceneGraph::observe_camera_cover(const rc::RoomConcept::UpdateResult& res)
{
    if (not G_ or not startup_proto_ or camera_finished_) return;   // this agent's room, while its camera question is open
    resolve_zed_mount();
    // intrinsics per call until complete: robot_concept publishes them only once frames flow (graph_obstacles.h)
    if (not cam_sensor_.complete())
        if (const auto zed = G_->get_node("zed"); zed.has_value())
            if (auto cam = G_->get_camera_api(zed.value()); cam)
            {
                const float fx = cam->get_focal_x(), fy = cam->get_focal_y();
                const float W = static_cast<float>(cam->get_width()), H = static_cast<float>(cam->get_height());
                if (fx > 0.f and W > 0.f and fy > 0.f and H > 0.f and not cam_sensor_.has_vertical())   // once
                {
                    cam_sensor_.hfov_rad = 2.f * std::atan(0.5f * W / fx);
                    cam_sensor_.vfov_rad = 2.f * std::atan(0.5f * H / fy);
                    qInfo() << "[room][camera-cover] zed intrinsics: hfov" << cam_sensor_.hfov_rad * 57.29578f
                            << "deg, vfov" << cam_sensor_.vfov_rad * 57.29578f << "deg";
                }
            }
    if (startup_promoted_ and not cam_sensor_.complete() and not camera_model_warned_)
    {
        camera_model_warned_ = true;
        qWarning() << "[room][camera-cover] camera model INCOMPLETE at promotion (hfov" << cam_sensor_.hfov_rad * 57.29578f
                   << "vfov" << cam_sensor_.vfov_rad * 57.29578f << "deg, height" << cam_sensor_.height_m << "m) — until it"
                   << "completes the room's boundary camera coverage cannot be assessed and afford_room goes to the pose"
                   << "planner on the LiDAR survey alone";
    }
    if (not zed_mount_resolved_ or not cam_sensor_.complete()) return;
    // The boundary in the PUBLISHED frame: the promoted room's polygon, or the survey's ring before promotion (its views
    // are carried onto the promoted outline by Coverage::set_outline).
    auto outline = startup_promoted_ ? published_polygon() : fs_published_;
    if (outline.size() < 3) return;
    cam_cover_.set_outline(outline);
    const Eigen::Affine2f pose_pub = T_pub_int_ * res.robot_pose;
    const Eigen::Vector2f cam = pose_pub * zed_xy_;
    const Eigen::Vector2f fwd = pose_pub.linear() * zed_fwd_;
    cam_cover_.observe(cam, std::atan2(fwd.y(), fwd.x()), cam_sensor_);
}

rc::layout::Plan RoomSceneGraph::plan_camera_cover(const rc::layout::FeasibleGrid& grid, const Eigen::Vector2f& robot,
                                                   const rc::layout::ExplorerParams& ep,
                                                   const rc::layout::Admissible& standable,
                                                   const rc::layout::Admissible& deferred)
{
    // A standpoint is in the internal frame; the coverage is in the published one. The look is scored at its BEST heading
    // (camera_coverage.h best_look) — and that heading is the yaw published with it (plan_layout), which the controller
    // turns to on arrival. (It was the ARRIVAL heading: the L's short leg left the robot facing its end wall, user 2026-11-01.) The zed sits a few cm from the robot origin — ignored here, as the LiDAR
    // gain ignores its own mount; observe() uses the real zed pose. An unreachable best target is excluded and the plan
    // re-run (plan_coverage_reachable): it never ends the phase by itself.
    const Eigen::Vector2f r_pub = T_pub_int_ * robot;
    return rc::layout::plan_coverage_reachable(grid, robot, ep, standable,
        [this, r_pub](const Eigen::Vector2f& v)
        {
            const Eigen::Vector2f v_pub = T_pub_int_ * v;
            return (v_pub - r_pub).squaredNorm() > 1e-9f ? wall_nats(cam_cover_.best_look(v_pub, cam_sensor_).first) : 0.f;
        },
        wall_nats(cam_cover_.price(cam_sensor_)), deferred);   // item A: m of wall -> nats (door prior), gain and price alike
}

void RoomSceneGraph::latch_camera_phase()
{
    // The camera phase's OWN verdict: plan_coverage's `finished` — no REACHABLE look newly shows one door's worth of wall
    // (unreachable targets were excluded and re-planned; recently refused ones keep it "deferred"). layout_finished_ is
    // the LiDAR survey's and is left alone.
    if (camera_visits0_ < 0) camera_visits0_ = layout_visits_;   // the phase's own visits, not the survey's
    if (not layout_last_plan_.finished or camera_finished_) return;
    camera_finished_ = true;
    learn_coverage_priors();   // item A: the finished room teaches the occupancy and door priors (once per room)
    float total = 0.f;
    for (const auto& c : cam_cover_.cells()) total += c.len;
    const float price = cam_cover_.price(cam_sensor_), residual = cam_cover_.residual_length(cam_sensor_);
    qInfo().noquote() << QString("[room][camera-cover] boundary camera coverage FINISHED: %1 of %2 m of wall could still "
                                 "be shown to the detector, no reachable look worth one door (%3 m) — afford_room goes "
                                 "to the pose planner")
                             .arg(residual, 0, 'f', 2).arg(total, 0, 'f', 2).arg(price, 0, 'f', 2);
    // ★ LOUD while the detector envelope is uncalibrated: a phase that finishes without a single camera-phase visit while
    //   more than a door's worth of wall was never shown is almost always a room too narrow for the envelope's band
    //   (~2.4-3.1 m for a door with the default envelope), i.e. the model, not the room, ended it.
    if (layout_visits_ == camera_visits0_ and residual > price and not camera_narrow_warned_)
    {
        camera_narrow_warned_ = true;
        qWarning().noquote() << QString("[room][camera-cover] WARNING: camera phase FINISHED WITHOUT A VISIT with %1 m of "
                                        "wall never shown to the door detector (attainable P %2) — the room is called "
                                        "explored but a door there may be undiscovered; the detector envelope is "
                                        "UNCALIBRATED (rc::detect defaults)")
                                    .arg(residual, 0, 'f', 2).arg(cam_cover_.attainable_p(cam_sensor_), 0, 'f', 2);
    }
}

void RoomSceneGraph::remember_camera_cover()
{
    if (const std::string name = room_node_name(); not name.empty() and startup_proto_ and cam_cover_.ready())
        camera_memory_[name] = CameraMemo{cam_cover_, camera_finished_};
}

void RoomSceneGraph::resume_camera_cover(const std::string& room)
{
    // A re-entry: the cameras' record of this room (published frame = the room's own, the one the recall restores).
    // No record = a room this run never surveyed with the cameras (remembered from before): no camera question is
    // reopened — a remembered door there is door_concept's look (door_look.h), not this coverage.
    if (const auto it = camera_memory_.find(room); it != camera_memory_.end())
    {
        cam_cover_ = it->second.cover;
        camera_finished_ = it->second.finished;
        camera_visits0_ = -1;
        if (not camera_finished_)
            qInfo().noquote() << QString("[room][camera-cover] %1 re-entered with its boundary camera coverage UNFINISHED "
                                         "(%2 m still to show the detector): the camera phase resumes")
                                     .arg(QString::fromStdString(room)).arg(cam_cover_.residual_length(cam_sensor_), 0, 'f', 2);
    }
    else
        camera_finished_ = true;
}

}   // namespace rc

namespace rc
{
// ── AuRo item A (2026-11-23): the room-building offer in NATS (coverage_info.h) ─────────────────────────────────────
float RoomSceneGraph::floor_nats(float area_m2) const
{
    auto& self = const_cast<RoomSceneGraph&>(*this);
    if (not coverage_priors_set_)
    {
        self.occ_prior_  = {params_->STARTUP_PROTO_OCC_PRIOR_A, params_->STARTUP_PROTO_OCC_PRIOR_B};
        self.door_prior_ = {params_->STARTUP_PROTO_DOOR_PRIOR_A, params_->STARTUP_PROTO_DOOR_PRIOR_B};
        self.ell_prior_  = {std::log(std::max(1e-3f, params_->STARTUP_PROTO_ELL_PRIOR_M)), std::max(1e-3f, params_->STARTUP_PROTO_ELL_PRIOR_N)};
        self.coverage_priors_set_ = true;
    }
    const float cell = free_space_ ? free_space_->params().cell : params_->STARTUP_PROTO_CELL;
    return rc::coverage::area_to_nats(area_m2, cell, occ_prior_, ell_prior_);
}

float RoomSceneGraph::wall_nats(float length_m) const
{
    (void)floor_nats(0.f);   // sets the priors on first use
    return rc::coverage::wall_to_nats(length_m, cam_cover_.params().w_m, door_prior_);
}

void RoomSceneGraph::learn_coverage_priors()
{
    if (coverage_priors_learnt_ or not free_space_ or fs_published_.size() < 3 or not G_) return;
    (void)floor_nats(0.f);
    coverage_priors_learnt_ = true;
    // occupancy: the finished room's own raster inside its outline (the same countable band the explorer counts in)
    const auto mask = free_space_->countable_mask(fs_published_, 0.40f);
    // the correlation length first: it also says how many INDEPENDENT observations the cell counts are worth
    if (const auto ell = free_space_->occupancy_correlation_length(mask); ell.has_value()) ell_prior_.add(*ell);
    const double cell = free_space_->params().cell, l = std::max(cell, ell_prior_.value()), per = cell * cell / (l * l);
    const auto [n_free, n_occ] = free_space_->count_states(mask);
    occ_prior_.add(static_cast<double>(n_occ) * per, static_cast<double>(n_free) * per);
    // doors: door nodes hanging from this room vs the door-width segments of its boundary
    double perimeter = 0.0;
    for (std::size_t i = 0; i < fs_published_.size(); ++i)
        perimeter += (fs_published_[(i + 1) % fs_published_.size()] - fs_published_[i]).norm();
    const double w = std::max(1e-3f, cam_cover_.params().w_m);
    long n_doors = 0;
    for (const auto& n : G_->get_nodes())
        if (n.name().starts_with("door") and G_->get_attrib_by_name<parent_att>(n).value_or(0) == dsr_room_id_) ++n_doors;
    const double segments = std::max(static_cast<double>(n_doors), perimeter / w);
    door_prior_.add(static_cast<double>(n_doors), segments - static_cast<double>(n_doors));
    qInfo().noquote() << QString("[room][coverage] priors learnt from this room: occupancy %1 occupied / %2 free cells -> "
                                 "P(occ) %3, %4 nats/region, correlation length %9 m; doors %5 of %6 segments -> P(door) %7, %8 nats/segment")
                             .arg(n_occ).arg(n_free).arg(occ_prior_.mean(), 0, 'f', 3).arg(occ_prior_.nats_per_unit(), 0, 'f', 3)
                             .arg(n_doors).arg(segments, 0, 'f', 1).arg(door_prior_.mean(), 0, 'f', 3)
                             .arg(door_prior_.nats_per_unit(), 0, 'f', 3).arg(ell_prior_.value(), 0, 'f', 3);
}

void RoomSceneGraph::log_room_offer(std::string_view source, const Eigen::Vector2f& target_pub, float pose_info, float neglect,
                                    float change_info, float coverage_info, float total)
{
    if (params_->ROOM_OFFER_LOG.empty()) return;
    if (not room_offer_log_.is_open())
    {
        room_offer_log_.open(params_->ROOM_OFFER_LOG, std::ios::out | std::ios::trunc);
        room_offer_log_.imbue(std::locale::classic());
        room_offer_log_ << "wall_ms,source,tx,ty,pose_info,neglect,change_info,coverage_info,total,p_occ,occ_a,occ_b,p_door,door_a,door_b,ell_m,hazard,change_tests\n";
    }
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const auto& cb = epistemic_->epistemic_planner().change_belief();
    room_offer_log_ << now << ',' << source << ',' << target_pub.x() << ',' << target_pub.y() << ',' << pose_info << ','
                    << neglect << ',' << change_info << ',' << coverage_info << ',' << total << ',' << occ_prior_.mean() << ',' << occ_prior_.a
                    << ',' << occ_prior_.b << ',' << door_prior_.mean() << ',' << door_prior_.a << ',' << door_prior_.b << ',' << ell_prior_.value()
                    << ',' << cb.hazard() << ',' << cb.tests() << '\n';
    room_offer_log_.flush();
}

}   // namespace rc

namespace rc
{
// ── AuRo item B (2026-11-23): the room's change belief — rooms, record, test log ─────────────────────────────────────
void RoomSceneGraph::change_room_step(rc::EpistemicPlanner& planner)
{
    if (not planner.params.use_change_belief) return;
    const std::string name = room_node_name();
    if (name.empty() or planner.room_corners().size() < 3) return;
    if (not change_memory_loaded_) change_memory_load(planner);
    // the room's bounds in its own published frame (the frame its record lives in, restored by a recall)
    Eigen::Vector2f lo(std::numeric_limits<float>::max(), std::numeric_limits<float>::max()), hi = -lo;
    for (const auto& c : planner.room_corners())
    {
        const Eigen::Vector2f q = T_pub_int_ * c;
        lo = lo.cwiseMin(q); hi = hi.cwiseMax(q);
    }
    planner.change_enter_room(name, lo, hi, T_pub_int_);
    (void)floor_nats(0.f);
    planner.set_change_unknown_nats(occ_prior_.nats_per_unit());
    auto& cb = planner.change_belief();
    if (not cb.on_test and not params_->CHANGE_TESTS_LOG.empty())
    {
        change_tests_log_.open(params_->CHANGE_TESTS_LOG, std::ios::out | std::ios::trunc);
        change_tests_log_.imbue(std::locale::classic());
        change_tests_log_ << "wall_s,room,cell,cx_pub,cy_pub,first,age_s,p_cell,flip,promised,delivered,hazard,detect,false_alarm\n";
        cb.on_test = [this, &cb](const rc::change::Test& t)
        {
            const auto c = cb.centre(t.cell);
            change_tests_log_ << std::fixed << t.t << ',' << cb.room() << ',' << t.cell << ',' << c.x() << ',' << c.y() << ','
                              << int(t.first) << ',' << t.age << ',' << t.p_cell << ',' << int(t.flip) << ',' << t.promised << ',' << t.delivered << ','
                              << cb.hazard() << ',' << cb.detect() << ',' << cb.false_alarm() << '\n';
            change_tests_log_.flush();
        };
    }
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (now_ms - change_memory_saved_ms_ > 30000)
    {
        change_memory_save(planner); change_memory_saved_ms_ = now_ms;
        long with_matter = 0;
        for (int k = 0; k < cb.size(); ++k) with_matter += cb.cell(k).matter == 1 ? 1 : 0;
        rc::status::print("[room][change] matter stream: {} batches stamped ({} too old), {} points in the grid; "
                          "{} sweeps, {} with a scan; {} of {} cells hold matter\n",
                          low_batches_stamped_, low_batches_too_old_, planner.matter_points_in_grid_,
                          planner.change_sweeps_, planner.change_sweeps_scanned_, with_matter, cb.size());
    }
}

// Format (text, one record per line, '.' decimals): "posterior <n> <logw...>" then "cell <room> <cx> <cy> <t_obs> <matter>".
void RoomSceneGraph::change_memory_load(rc::EpistemicPlanner& planner)
{
    change_memory_loaded_ = true;
    if (params_->CHANGE_MEMORY.empty()) return;
    std::ifstream f(params_->CHANGE_MEMORY);
    if (not f) return;
    std::map<std::string, std::vector<rc::change::ChangeBelief::Record>> recs;
    std::vector<double> post;
    std::string line;
    const auto num = [](std::string_view s, double& v) { return std::from_chars(s.data(), s.data() + s.size(), v).ec == std::errc{}; };
    while (std::getline(f, line))
    {
        std::vector<std::string_view> tok;
        for (std::size_t i = 0; i < line.size();)
        {
            const auto j = line.find(' ', i);
            tok.emplace_back(line.data() + i, (j == std::string::npos ? line.size() : j) - i);
            i = j == std::string::npos ? line.size() : j + 1;
        }
        if (tok.empty()) continue;
        if (tok[0] == "posterior")
            for (std::size_t i = 2; i < tok.size(); ++i) { double v; if (num(tok[i], v)) post.push_back(v); }
        else if (tok[0] == "cell" and tok.size() == 6)
        {
            double x, y, t, m;
            if (num(tok[2], x) and num(tok[3], y) and num(tok[4], t) and num(tok[5], m))
                recs[std::string(tok[1])].push_back({Eigen::Vector2f(float(x), float(y)), t, static_cast<std::int8_t>(m)});
        }
    }
    auto& cb = planner.change_belief();
    const bool ok = cb.import_posterior(post);
    std::size_t n = 0;
    for (const auto& [_, v] : recs) n += v.size();
    cb.restore_records(std::move(recs));
    qInfo().noquote() << QString("[room][change] memory %1: posterior %2 (λ %3 /m²/s, d %4, f %5), %6 cell records")
                             .arg(QString::fromStdString(params_->CHANGE_MEMORY)).arg(ok ? "restored" : "prior")
                             .arg(cb.hazard(), 0, 'e', 2).arg(cb.detect(), 0, 'f', 3).arg(cb.false_alarm(), 0, 'f', 3).arg(n);
}

void RoomSceneGraph::change_memory_save(const rc::EpistemicPlanner& planner)
{
    if (params_->CHANGE_MEMORY.empty()) return;
    const auto& cb = planner.change_belief();
    const std::string tmp = params_->CHANGE_MEMORY + ".tmp";
    {
        std::ofstream f(tmp, std::ios::out | std::ios::trunc);
        if (not f) return;
        f.imbue(std::locale::classic());
        f.precision(17);
        const auto post = cb.export_posterior();
        f << "posterior " << post.size();
        for (const double v : post) f << ' ' << v;
        f << '\n';
        for (const auto& [room, rs] : cb.all_records())
            for (const auto& r : rs)
                f << "cell " << room << ' ' << r.centre.x() << ' ' << r.centre.y() << ' ' << r.t_obs << ' ' << int(r.matter) << '\n';
    }
    std::rename(tmp.c_str(), params_->CHANGE_MEMORY.c_str());
}
}   // namespace rc
