/*
 *    Copyright (C) 2026 by RoboComp CORTEX Team
 *    This file is part of RoboComp — GNU GPL v3.
 */

/**
 * SpecificWorker — residual_concept: lifecycle + presence + orchestration.
 */

#include "specificworker.h"

#include <cstdlib>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <print>

#include <QCoreApplication>
#include <QTimer>

#include <dsr/api/dsr_api.h>

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check)
    : GenericWorker(configLoader, tprx)
{
    this->startup_check_flag = startup_check;
    if (this->startup_check_flag)
    {
        this->startup_check();
        return;
    }

    cfg_ = rc::load_residual_config(configLoader);

    // Isolated unit tests run once at startup so a broken belief / clusterer is caught before the live loop.
    rc::ResidualBelief::self_test();
    rc::ResidualClusterer::self_test();
    rc::zed_boost_self_test();
    rc::OccupancyGrid::self_test();   // PHASE-0 REBUILD: safety-layer grid (completeness / occluded / carve / z-aware)

#ifdef HIBERNATION_ENABLED
    hibernationChecker.start(500);
#endif

    const int period = configLoader.get<int>("Period.Compute");

    states["Waiting"] = std::make_unique<GRAFCETStep>("Waiting", period,
        std::bind(&SpecificWorker::waiting_loop, this), std::bind(&SpecificWorker::waiting_enter, this));
    states["Operating"] = std::make_unique<GRAFCETStep>("Operating", period,
        std::bind(&SpecificWorker::operating_loop, this), std::bind(&SpecificWorker::operating_enter, this));
    states["Degraded"] = std::make_unique<GRAFCETStep>("Degraded", period,
        std::bind(&SpecificWorker::degraded_loop, this), std::bind(&SpecificWorker::degraded_enter, this));

    states["Compute"]->addTransition(states["Compute"].get(), SIGNAL(entered()), states["Waiting"].get());
    states["Waiting"]->addTransition(this, SIGNAL(presenceReady()), states["Operating"].get());
    states["Operating"]->addTransition(this, SIGNAL(presenceLost()), states["Degraded"].get());
    states["Degraded"]->addTransition(states["Degraded"].get(), SIGNAL(entered()), states["Waiting"].get());

    statemachine.addState(states["Waiting"].get());
    statemachine.addState(states["Operating"].get());
    statemachine.addState(states["Degraded"].get());

    statemachine.setChildMode(QState::ExclusiveStates);
    statemachine.start();

    auto error = statemachine.errorString();
    if (error.length() > 0) { qWarning() << error; throw error; }
}

SpecificWorker::~SpecificWorker()
{
    request_shutdown();
    std::print("residual_concept: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;
    save_window_settings();
    if (G)
        disconnect(G.get(), nullptr, this, nullptr);
    zed_ingestor_.reset();
    lidar_ingestor_.reset();
    inner_eigen_.reset();
    cleanup_owned_nodes();
}

void SpecificWorker::terminal_shutdown()
{
    static std::atomic<bool> terminating{false};
    if (terminating.exchange(true))
        return;
    request_shutdown();
    if (G)
    {
        try { G->reset(); }
        catch (...) {}
    }
    std::cout.flush();
    std::cerr.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    std::_Exit(EXIT_SUCCESS);
}

void SpecificWorker::initialize()
{
    std::print("residual_concept: initialize()\n");
    GenericWorker::initialize();

    if (not G)
    {
        qWarning() << "residual_concept: DSR graph not available in initialize()";
        return;
    }

    // Agent-presence protocol wiring (mirrors bottle_concept — Degraded DEBOUNCE is the CLAUDE.md rule).
    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    presence_coordinator_.set_transition_hooks({
        .request_presence_ready = [this]() { emit presenceReady(); },
        .request_presence_lost  = [this]() { emit presenceLost(); },
    });
    presence_coordinator_.set_peer_hooks({
        .on_peer_restarted = [](std::uint32_t id) { qInfo() << "[Presence] peer" << id << "restarted"; },
        .on_optional_peer_lost  = [this](const std::string& name, std::uint32_t id) { on_optional_peer_lost(name, id); },
        .on_optional_peer_ready = [this](const std::string& name, std::uint32_t id) { on_optional_peer_ready(name, id); },
    });
    presence_coordinator_.set_lifecycle_hooks({
        .on_waiting_enter = [this]()
        {
            const auto missing = presence_coordinator_.missing_required_names();
            if (missing.empty()) qInfo("[SM] -> Waiting");
            else { QString m; for (const auto& l : missing) m += " " + QString::fromStdString(l);
                   qInfo() << "[SM] -> Waiting (missing:" << m.trimmed() << ")"; }
        },
        .on_operating_enter = [this]()
        {
            qInfo("[SM] -> Operating: all required peers present");
            // One-time leftover sweep AFTER the graph is fully joined — catches residual_* nodes left by a
            // crashed/uncleaned prior run that were still syncing in when initialize() ran. Guarded so a
            // Degraded→Operating re-entry (transient peer flap) does NOT wipe THIS run's live residuals.
            if (not startup_sweep_done_)
            {
                startup_sweep_done_ = true;
                remove_owned_residual_nodes();
            }
        },
        .on_operating_loop  = [this]() { compute(); },
        .on_degraded_enter  = [this]()
        {
            if (shutting_down_) return;
            qInfo("[SM] -> Degraded: required peer lost — %d ms grace before shutdown", REQUIRED_LOSS_GRACE_MS);
            QTimer::singleShot(REQUIRED_LOSS_GRACE_MS, this, [this]()
            {
                if (shutting_down_) return;
                if (presence_coordinator_.all_required_ready())
                { qInfo("[SM] required peers recovered during grace — staying alive"); return; }
                qWarning("[SM] required peer still missing after grace — shutting down cleanly");
                terminal_shutdown();
            });
        },
    });
    presence_coordinator_.start();

    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::terminal_shutdown, Qt::UniqueConnection);

    rt_api_       = G->get_rt_api();
    inner_eigen_  = G->get_inner_eigen_api();
    gaussian_api_ = std::make_unique<DSR::InnerGaussianAPI>(G.get());

    scene_graph_ = std::make_unique<rc::ResidualSceneGraph>(G, rt_api_.get(), inner_eigen_.get(), cfg_,
                                                            [this] { trigger_graph_layout_twopi(); });
    scene_graph_->set_chain_cov_source(gaussian_api_.get(), cfg_.lidar_frame_node, cfg_.rt_cov_add_chain);

    fitter_        = std::make_unique<rc::ResidualFitter>(G, inner_eigen_.get(), cfg_);
    lidar_ingestor_ = std::make_unique<rc::ResidualLidarIngestor>(G, inner_eigen_.get(), cfg_);
    if (cfg_.zed_boost_enabled)
        zed_ingestor_ = std::make_unique<rc::ResidualZedIngestor>(G, cfg_);
    clusterer_.set_params(cfg_.cluster);

    connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

    remove_owned_residual_nodes();

    const auto rooms = G->get_nodes_by_type("room");
    if (not rooms.empty()) room_node_id_ = rooms.front().id();
    else                   qWarning() << "residual_concept: no room node found at startup";
}

std::vector<rc::SpecialistSdf> SpecificWorker::build_specialist_sdfs() const
{
    // Every modelled object's published box geometry becomes an SDF explainer; the clusterer subtracts the
    // points each explains, so what survives is the residual. Box centre is lifted by height/2 (RT pose ≈
    // the object base) so an object resting ON a table (above its slab) is NOT swallowed. NOTE: the RT-pose-
    // is-base assumption + solid-box (vs slab+legs) approximation is the first live-tuning point.
    std::vector<rc::SpecialistSdf> out;
    if (not inner_eigen_)
        return out;
    for (const char* type : {"table", "chair", "cylinder"})
        for (const auto& n : G->get_nodes_by_type(type))
        {
            const float w = G->get_attrib_by_name<width_m_att> (n).value_or(0.0f);
            const float d = G->get_attrib_by_name<depth_m_att> (n).value_or(0.0f);
            const float h = G->get_attrib_by_name<height_m_att>(n).value_or(0.0f);
            if (w <= 0.0f or d <= 0.0f or h <= 0.0f)
                continue;
            const auto T = inner_eigen_->get_transformation_matrix("room", n.name(), 0);
            if (not T.has_value())
                continue;
            Eigen::Matrix4d M;
            { const auto& s = T.value().matrix(); for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) M(i, j) = s(i, j); }
            const Eigen::Vector3f centre(static_cast<float>(M(0, 3)), static_cast<float>(M(1, 3)),
                                         static_cast<float>(M(2, 3)) + 0.5f * h);
            const float yaw = std::atan2(static_cast<float>(M(1, 0)), static_cast<float>(M(0, 0)));
            const float hx = 0.5f * w, hy = 0.5f * d, hz = 0.5f * h;
            out.push_back([centre, yaw, hx, hy, hz](const Eigen::Vector3f& p)
                          { return rc::ResidualClusterer::box_sdf_3d(p, centre, yaw, hx, hy, hz); });
        }
    return out;
}

// Collect the modelled horizontal SUPPORT surfaces (tabletops) — top-z + oriented footprint — so an obstacle
// resting on one anchors its box bottom to the TABLE, not the floor. Rebuilt each cycle from the graph.
void SpecificWorker::build_support_surfaces()
{
    support_surfaces_.clear();
    if (not inner_eigen_ or not G)
        return;
    for (const auto& n : G->get_nodes_by_type("table"))
    {
        const float w = G->get_attrib_by_name<width_m_att> (n).value_or(0.0f);
        const float d = G->get_attrib_by_name<depth_m_att> (n).value_or(0.0f);
        const float h = G->get_attrib_by_name<height_m_att>(n).value_or(0.0f);
        if (w <= 0.0f or d <= 0.0f or h <= 0.0f)
            continue;
        const auto T = inner_eigen_->get_transformation_matrix("room", n.name(), 0);
        if (not T.has_value())
            continue;
        const auto& M = T.value().matrix();
        SupportSurface s;
        s.cx = static_cast<float>(M(0, 3)); s.cy = static_cast<float>(M(1, 3));
        s.top_z = static_cast<float>(M(2, 3)) + h;                    // base at RT pose z, top a height above
        s.yaw = std::atan2(static_cast<float>(M(1, 0)), static_cast<float>(M(0, 0)));
        s.hx = 0.5f * w; s.hy = 0.5f * d;
        support_surfaces_.push_back(s);
    }
}

// The support an obstacle rests on = the HIGHEST modelled surface whose top is ≤ z_min (the obstacle's lowest
// return) and whose footprint contains (cx,cy). Floor (0) is the default. No threshold: a table-standing
// object's lowest return is above the tabletop, so top_z ≤ z_min selects the table; a floor object's z_min is
// below any tabletop, so only the floor qualifies.
float SpecificWorker::support_z_for(float cx, float cy, float z_min) const
{
    float support = 0.0f;   // floor
    for (const auto& s : support_surfaces_)
    {
        if (s.top_z <= support or s.top_z > z_min)                   // not higher than the current best, or above the object
            continue;
        const float dx = cx - s.cx, dy = cy - s.cy;
        const float c = std::cos(s.yaw), si = std::sin(s.yaw);
        const float lx = c * dx + si * dy, ly = -si * dx + c * dy;   // into the tabletop's local frame
        if (std::abs(lx) <= s.hx and std::abs(ly) <= s.hy)           // (cx,cy) over the tabletop
            support = s.top_z;
    }
    return support;
}

std::vector<rc::SpecialistExplainer> SpecificWorker::build_explainers(
    const Eigen::Vector2f& robot_xy, const std::vector<rc::SpecialistSdf>& objects) const
{
    std::vector<rc::SpecialistExplainer> ex;
    const auto& C = cfg_.cluster;

    // 0: ROOM FLOOR — the room's floor plane (z=0). A grazing floor return lands higher with range, so the
    //    floor model's tolerance GROWS with range (grazing angle × distance + datum offset) — not a flat gate.
    const float fz0 = C.floor_z0, fsl = C.floor_slope;
    ex.push_back([fz0, fsl, robot_xy](const Eigen::Vector3f& p)
                 { return p.z() < fz0 + fsl * (p.head<2>() - robot_xy).norm(); });
    // 1: ROOM CEILING — overhead structure above the navigation band.
    const float cz = C.ceil_z;
    ex.push_back([cz](const Eigen::Vector3f& p) { return p.z() > cz; });
    // 2: ROBOT — its own body returns.
    const float rr2 = C.robot_radius_m * C.robot_radius_m;
    ex.push_back([rr2, robot_xy](const Eigen::Vector3f& p) { return (p.head<2>() - robot_xy).squaredNorm() < rr2; });
    // 3: ROOM WALLS — the room_concept delimiting polygon. A return outside the room, or within wall_margin of
    //    a wall edge, is explained by the room's wall model.
    const auto poly = read_room_polygon();
    const float wm = C.wall_margin_m;
    if (poly.size() >= 3)
        ex.push_back([poly, wm](const Eigen::Vector3f& p)
                     { const Eigen::Vector2f xy = p.head<2>();
                       return not rc::ResidualClusterer::point_in_polygon(poly, xy)
                              or rc::ResidualClusterer::dist_to_polygon_boundary(poly, xy) < wm; });
    else
        ex.push_back([](const Eigen::Vector3f&) { return false; });   // no room polygon yet → no wall model
    // 4+: MODELLED OBJECTS — table/chair/bottle box SDFs. A point is explained if it is INSIDE the object's
    //     volume OR within `em` of its surface (signed sdf < em, NOT |sdf| < em). Claiming the interior — not
    //     just a surface shell — keeps the filter CONSISTENT with the dissolve test (which uses signed sdf at
    //     the centroid): otherwise a cluster sitting deep inside a modelled box leaks through the surface
    //     filter, births, and is dissolved the same cycle → an endless birth↔dissolve loop. On-table survival
    //     is unaffected: an object resting ABOVE the box has sdf > em (outside) and still survives.
    const float em = C.explain_margin_m;
    for (const auto& sdf : objects)
        ex.push_back([sdf, em](const Eigen::Vector3f& p) { return sdf(p) < em; });
    return ex;
}

std::vector<Eigen::Vector2f> SpecificWorker::read_room_polygon() const
{
    static bool logged = false;
    const auto once = [&](const char* why) { if (not logged) { logged = true; std::print("[residual] room polygon EMPTY: {}\n", why); } };
    std::vector<Eigen::Vector2f> poly;
    if (room_node_id_ == 0) { once("room_node_id==0"); return poly; }
    const auto room = G->get_node(room_node_id_);
    if (not room.has_value()) { once("room node not found"); return poly; }
    const auto px = G->get_attrib_by_name<delimiting_polygon_x_att>(room.value());
    const auto py = G->get_attrib_by_name<delimiting_polygon_y_att>(room.value());
    if (not px.has_value() or not py.has_value()) { once("delimiting_polygon_x/y attr absent on room node"); return poly; }
    const auto& xs = px->get(); const auto& ys = py->get();
    const std::size_t n = std::min(xs.size(), ys.size());
    poly.reserve(n);
    for (std::size_t i = 0; i < n; ++i) poly.emplace_back(xs[i], ys[i]);
    // One-time dump of the actual vertices so the polygon units/extent/alignment vs the wall clusters is
    // readable off disk (compare against residual_diag.csv cluster cx,cy).
    static bool dumped = false;
    if (not dumped)
    {
        dumped = true;
        std::ofstream f("etc/residual_room_poly.csv", std::ios::out | std::ios::trunc);
        if (f.is_open()) { f << "i,x,y\n"; for (std::size_t i = 0; i < n; ++i) f << i << ',' << xs[i] << ',' << ys[i] << '\n'; }
    }
    return poly;
}

void SpecificWorker::compute()
{
    if (not G or not rt_api_)
        return;
    if (room_node_id_ == 0)
    {
        const auto rooms = G->get_nodes_by_type("room");
        if (rooms.empty()) return;
        room_node_id_ = rooms.front().id();
    }

    // Fresh LiDAR sweep is the only trigger — no sweep, no re-cluster this cycle (instances persist).
    if (not lidar_ingestor_ or not lidar_ingestor_->pump())
        return;
    current_ts_ = lidar_ingestor_->last_stamp_ms();

    const Eigen::Vector2f robot_xy = lidar_ingestor_->origin_room().head<2>();
    robot_xy_ = robot_xy;                                                // for the negative-info death gate

    // ── PHASE-0 REBUILD: occupancy-grid safety layer, running LIVE as a diagnostic (publish still via the old
    //    path until it's verified stable). Init once from the room polygon bounds; then integrate every sweep. ──
    {
        if (not grid_ready_)
        {
            const auto poly = read_room_polygon();
            float xmn = -6, ymn = -6, xmx = 6, ymx = 6;
            if (poly.size() >= 3)
            {
                xmn = ymn = 1e9f; xmx = ymx = -1e9f;
                for (const auto& p : poly) { xmn = std::min(xmn, p.x()); xmx = std::max(xmx, p.x()); ymn = std::min(ymn, p.y()); ymx = std::max(ymx, p.y()); }
                xmn -= 0.5f; ymn -= 0.5f; xmx += 0.5f; ymx += 0.5f;      // margin around the room
            }
            rc::OccGridParams gp;
            gp.floor_z0 = cfg_.cluster.floor_z0; gp.floor_slope = cfg_.cluster.floor_slope; gp.ceil_z = cfg_.cluster.ceil_z;
            grid_.reset(xmn, ymn, xmx, ymx, gp);
            grid_ready_ = grid_.valid();
            std::println("[grid] init bounds=({:.1f},{:.1f})..({:.1f},{:.1f}) cells={}x{}",
                         xmn, ymn, xmx, ymx, grid_.width(), grid_.height());
        }
        if (grid_ready_)   // integrate now (needs no explainers); read-out + publish happen after explainers below
            grid_.integrate_sweep(lidar_ingestor_->origin_room(), lidar_ingestor_->sweep_room());
    }

    // Robot yaw rate (rad/s) from the room<-robot RT edge, for the ego-motion point-reliability term.
    float rot_rate = 0.0f;
    if (rt_api_)
        if (const auto robots = G->get_nodes_by_type("robot"); not robots.empty())
            if (auto e = rt_api_->get_edge_RT(robots.front(), room_node_id_); e.has_value())
                if (auto rv = G->get_attrib_by_name<rt_rotation_euler_xyz_velocity_att>(e.value());
                    rv.has_value() and rv->get().size() >= 3)
                    rot_rate = std::abs(rv->get()[2]);
    fitter_->set_sensor_context(lidar_ingestor_->origin_room(), rot_rate);
    scene_graph_->set_sensor_origin(lidar_ingestor_->origin_room());   // directional inflation (grow away from sensor)
    const auto specialists = build_specialist_sdfs();                    // object SDFs (for the dissolve test)
    const auto explainers  = build_explainers(robot_xy, specialists);   // floor+ceiling+walls+robot+objects
    build_support_surfaces();                                            // tabletops → obstacle base anchoring

    // ── PHASE-1 REBUILD: read the RESIDUAL out of the grid (occupied ∧ ¬explained) and publish it. A cell is
    //    "explained" if any known-model explainer accounts for its representative point → walls + specialists
    //    drop out (floor/ceiling already excluded by the nav band), leaving object-only components. Evidence is
    //    never deleted, only masked at read-out — a specialist mis-fit cannot erase a real obstacle. Also
    //    publishes the residual cells on a `grid` node under room for the voxelizer 3-D display. ──
    if (grid_ready_)
    {
        const rc::OccupancyGrid::CellExplained cell_explained =
            [&explainers](float x, float y, float zlo, float zhi) -> bool
            {
                const Eigen::Vector3f p(x, y, 0.5f * (zlo + zhi));
                for (const auto& ex : explainers) if (ex(p)) return true;
                return false;
            };
        const auto comps = grid_.occupied_components(2, cell_explained);
        static int gc = 0;
        if ((gc++ % 20) == 0)
        {
            std::size_t maxc = 0; for (const auto& c : comps) maxc = std::max<std::size_t>(maxc, c.n_cells);
            std::println("[grid] residual components={} max_cells={} (walls/specialists subtracted at read-out)",
                         comps.size(), maxc);
        }
        publish_grid_display(cell_explained);   // `grid` node under room: residual cell centres for the 3-D viewer
    }

    // ZED dense-depth: pump the depth + cache the camera→room transform (at the depth stamp), then backproject
    // the WHOLE FoV into a dense room-frame cloud. It is fed into clustering ALONGSIDE the LiDAR sweep so
    // forward objects (the ZED sees ~110°/5 m densely) get plenty of points → detected earlier and sized
    // better than sparse LiDAR alone. Floor/walls/specialists are subtracted by the same explainers.
    zed_ready_ = false;
    std::vector<Eigen::Vector3f> zed_scene;
    // A rigid transform is sane iff finite, its rotation columns are ~unit, and its translation is inside the
    // room envelope. The RT interpolator can EXTRAPOLATE to garbage (non-orthonormal → ~1e31) on a stale/bad
    // depth timestamp — that scatters the ZED cloud (and, projected, throws the viewer's boxes wildly off).
    const auto sane_rt = [](const Eigen::Matrix4f& M)
    {
        if (not M.allFinite()) return false;
        for (int c = 0; c < 3; ++c) { const float n = M.block<3, 1>(0, c).norm(); if (n < 0.5f or n > 2.0f) return false; }
        return std::abs(M(0, 3)) < 50.0f and std::abs(M(1, 3)) < 50.0f and std::abs(M(2, 3)) < 10.0f;
    };
    if (zed_ingestor_)
    {
        zed_ingestor_->pump();
        if (zed_ingestor_->has_depth() and inner_eigen_)
        {
            bool got = false;
            // Try the depth stamp first (pose at capture time); fall back to latest (0) if it extrapolated to garbage.
            for (const std::uint64_t stamp : {zed_ingestor_->stamp_ms(), std::uint64_t{0}})
            {
                const auto T = inner_eigen_->get_transformation_matrix("room", "zed", stamp);
                if (not T.has_value()) continue;
                Eigen::Matrix4f m;
                { const auto& s = T.value().matrix(); for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) m(i, j) = static_cast<float>(s(i, j)); }
                if (sane_rt(m)) { room_T_cam_ = m; got = true; if (stamp != 0) break; else { static int fb = 0; if ((fb++ % 30) == 0) std::println("[ResidualZed] room_T_zed at depth-stamp was GARBAGE → using latest RT (timestamp extrapolation)"); } break; }
            }
            zed_ready_ = got;
            if (not got) { static int w = 0; if ((w++ % 30) == 0) std::println("[ResidualZed] room_T_zed GARBAGE at BOTH depth-stamp and latest — ZED skipped (RT chain bad, not just the stamp)"); }
            if (zed_ready_ and cfg_.zed_detection_enabled)
            {
                zed_scene = rc::backproject_fov(zed_ingestor_->depth(), zed_ingestor_->intrinsics(),
                                                room_T_cam_, cfg_.zed_boost);
                // ROBUST infrastructure subtraction on the dense ZED cloud BEFORE it joins detection: remove
                // floor/ceiling/wall with a band that grows as the ZED's own depth noise (σ0 + q·range²), so a
                // small calibration offset or far-range degradation can't leak a sheet that bridges objects.
                const Eigen::Vector3f zed_origin = room_T_cam_.block<3, 1>(0, 3);   // ZED position in room
                zed_scene = rc::ResidualClusterer::subtract_infrastructure(
                    zed_scene, zed_origin, read_room_polygon(), cfg_.zed_infra);
            }
        }
    }

    // One-shot ZED explainer breakdown (runs even with the detection feed OFF): shows why the dense ZED cloud
    // over-connects — how many ZED points each explainer removes vs keeps, plus the z-range and the MEAN z of
    // kept LOW points. If floor ZED points survive with mean z ≈ 0.1–0.2, the ZED→room z is offset (floor
    // lifted above the band) → that is the bridging cause. Read etc/residual_zed_filter_stats.csv off disk.
    {
        static bool zed_once = false;
        if (zed_ready_ and not zed_once)
        {
            zed_once = true;
            const auto zc = rc::backproject_fov(zed_ingestor_->depth(), zed_ingestor_->intrinsics(),
                                                room_T_cam_, cfg_.zed_boost);
            std::vector<int> cnt(explainers.size(), 0);
            int kept = 0, low = 0, low_kept = 0; double low_kept_z = 0.0;
            float zmn = 1e9f, zmx = -1e9f;
            for (const auto& p : zc)
            {
                zmn = std::min(zmn, p.z()); zmx = std::max(zmx, p.z());
                int hit = -1;
                for (int i = 0; i < static_cast<int>(explainers.size()); ++i) if (explainers[i](p)) { hit = i; break; }
                const bool is_low = p.z() < 0.30f;
                if (is_low) ++low;
                if (hit < 0) { ++kept; if (is_low) { ++low_kept; low_kept_z += p.z(); } } else ++cnt[hit];
            }
            int objs = 0; for (std::size_t i = 4; i < cnt.size(); ++i) objs += cnt[i];
            std::ofstream f("etc/residual_zed_filter_stats.csv", std::ios::out | std::ios::trunc);
            if (f.is_open())
                f << "total,floor,ceiling,robot,wall,objects,kept,zmin,zmax,low_below0.30,low_kept,low_kept_meanz\n"
                  << zc.size() << ',' << (cnt.size() > 0 ? cnt[0] : 0) << ',' << (cnt.size() > 1 ? cnt[1] : 0) << ','
                  << (cnt.size() > 2 ? cnt[2] : 0) << ',' << (cnt.size() > 3 ? cnt[3] : 0) << ',' << objs << ','
                  << kept << ',' << zmn << ',' << zmx << ',' << low << ',' << low_kept << ','
                  << (low_kept ? low_kept_z / low_kept : 0.0) << '\n';
        }
    }

    // Combined detection cloud = LiDAR sweep (360°, sparse) + dense ZED FoV points (forward, dense, already
    // infrastructure-subtracted → only object residual survives, no floor/wall sheet to bridge).
    if (cfg_.zed_detection_enabled)   // how many ZED points survived robust subtraction (should be ~object mass)
    {
        static int zc = 0;
        if ((zc++ % 20) == 0) std::println("[ResidualZed] residual ZED points after infra-subtract = {}", zed_scene.size());
    }
    std::vector<Eigen::Vector3f> cloud = lidar_ingestor_->sweep_room();
    cloud.insert(cloud.end(), zed_scene.begin(), zed_scene.end());
    clusters_ = clusterer_.extract(cloud, explainers);
    std::size_t max_pre = 0; for (const auto& c : clusters_) max_pre = std::max(max_pre, c.points.size());
    // Navigation fragment-merge: coalesce clusters closer than the bridge gap into ONE obstacle (a LiDAR-
    // shattered tabletop's ring-arcs). Clearance for "can't fit between" belongs to the planner (C-space).
    clusters_ = rc::ResidualClusterer::merge_close_clusters(std::move(clusters_), cfg_.cluster.merge_gap_m);
    {   // BLOB DIAGNOSTIC: is a room-sized cluster forming, and does the MERGE create it (post≫pre) or DBSCAN?
        std::size_t max_post = 0; for (const auto& c : clusters_) max_post = std::max(max_post, c.points.size());
        static int mc = 0;
        if ((mc++ % 20) == 0)
            std::println("[cluster] n={} max_npts pre-merge={} post-merge={}", clusters_.size(), max_pre, max_post);
    }

    // One-shot LiDAR-vs-room-model OFFSET measurement. The room polygon (inner wall surface) is TRUSTED
    // ground truth — the LiDAR should hit the walls ON it. If wall-height returns land systematically INSIDE
    // it (mean signed distance < 0), the raw LiDAR is PULLED IN toward the sensor by that amount — which is
    // exactly the residual-box bias the user sees (floor/walls/table project fine because they're model-based;
    // only residual uses raw LiDAR). A near-zero mean ⇒ LiDAR is faithful and the pull-in is elsewhere.
    {
        static bool wall_once = false;
        if (not wall_once)
        {
            wall_once = true;
            const auto poly = read_room_polygon();
            if (poly.size() >= 3)
            {
                const auto& sw = lidar_ingestor_->sweep_room();
                int n = 0, inside = 0; double sum_signed = 0.0;
                for (const auto& p : sw)
                {
                    if (p.z() < 0.20f or p.z() > 1.60f) continue;                 // wall-height band (skip floor/ceiling)
                    const Eigen::Vector2f xy = p.head<2>();
                    const float d = rc::ResidualClusterer::dist_to_polygon_boundary(poly, xy);
                    if (d > 0.40f) continue;                                      // only returns NEAR a wall
                    const bool in = rc::ResidualClusterer::point_in_polygon(poly, xy);
                    sum_signed += in ? -d : d; if (in) ++inside; ++n;             // inside = negative = pulled in
                }
                std::ofstream f("etc/residual_wall_offset.csv", std::ios::out | std::ios::trunc);
                if (f.is_open())
                    f << "n_near_wall,mean_signed_dist_inside_neg,inside,outside,robot_x,robot_y\n"
                      << n << ',' << (n ? sum_signed / n : 0.0) << ',' << inside << ',' << (n - inside) << ','
                      << robot_xy.x() << ',' << robot_xy.y() << '\n';
            }
        }
    }

    // Diagnostic CSV (read directly off disk — no terminal paste). One row per cluster with the sweep/
    // specialists/room_poly counts (room_poly=0 ⇒ walls flood; specialists=0 ⇒ nothing subtracted) plus
    // each cluster's footprint + z-band + point count, so wall/subtraction issues are diagnosable by file.
    if (not cfg_.diag_csv_path.empty())
    {
        if (not diag_csv_.is_open())
        {
            diag_csv_.open(cfg_.diag_csv_path, std::ios::out | std::ios::trunc);
            if (diag_csv_.is_open())
                diag_csv_ << "cycle,ts,sweep,specialists,explainers,clusters,instances,idx,cx,cy,w,d,zmin,zmax,npts,ox,oy\n";
        }
        if (diag_csv_.is_open())
        {
            ++diag_cycle_;
            const std::size_t sweep_n = lidar_ingestor_->sweep_room().size();
            const std::size_t inst_n  = fitter_->instances().size();
            const auto pre = [&](int idx) {
                diag_csv_ << diag_cycle_ << ',' << current_ts_ << ',' << sweep_n << ',' << specialists.size()
                          << ',' << explainers.size() << ',' << clusters_.size() << ',' << inst_n << ',' << idx << ',';
            };
            const float ox = robot_xy_.x(), oy = robot_xy_.y();   // sensor origin (room) — registration check
            if (clusters_.empty()) { pre(-1); diag_csv_ << ",,,,,,," << ox << ',' << oy << '\n'; }
            for (int i = 0; i < static_cast<int>(clusters_.size()); ++i)
            {
                const auto& c = clusters_[i];
                pre(i);
                diag_csv_ << c.centroid.x() << ',' << c.centroid.y() << ',' << c.w_seed << ',' << c.d_seed
                          << ',' << c.z_min << ',' << c.z_max << ',' << c.points.size()
                          << ',' << ox << ',' << oy << '\n';
            }
            diag_csv_.flush();
        }
    }

    // One-time per-explainer breakdown (which model accounts for how many returns): floor / ceiling / robot /
    // wall / objects / kept. All unified — no geometric-threshold masks. Read off disk, no terminal paste.
    {
        static bool once = false;
        if (not once)
        {
            once = true;
            const auto& sw = lidar_ingestor_->sweep_room();
            std::vector<int> cnt(explainers.size(), 0);
            int kept = 0;
            for (const auto& p : sw)
            {
                int hit = -1;
                for (int i = 0; i < static_cast<int>(explainers.size()); ++i)
                    if (explainers[i](p)) { hit = i; break; }
                if (hit < 0) ++kept; else ++cnt[hit];
            }
            const auto at = [&](std::size_t i) { return i < cnt.size() ? cnt[i] : 0; };
            int objs = 0; for (std::size_t i = 4; i < cnt.size(); ++i) objs += cnt[i];
            std::ofstream f("etc/residual_filter_stats.csv", std::ios::out | std::ios::trunc);
            if (f.is_open())
            {
                f << "total,floor,ceiling,robot,wall,objects,kept,robot_x,robot_y\n";
                f << sw.size() << ',' << at(0) << ',' << at(1) << ',' << at(2) << ',' << at(3) << ','
                  << objs << ',' << kept << ',' << robot_xy.x() << ',' << robot_xy.y() << '\n';
            }
        }
    }

    cluster_centroids_.clear();
    for (const auto& c : clusters_) cluster_centroids_.push_back(c.centroid);
    fitter_->set_clusters(&clusters_);

    // ZED pipeline diagnostic (throttled): pinpoints which gate blocks the ZED feed when it stays empty.
    {
        static int zc = 0;
        if ((zc++ % 20) == 0)
        {
            const bool has_ing = static_cast<bool>(zed_ingestor_);
            const bool has_dep = has_ing and zed_ingestor_->has_depth();
            std::println("[ResidualZed] ingestor={} has_depth={} stamp={} zed_node={} tf_ok(zed_ready)={} "
                         "instances={}",
                         has_ing, has_dep, has_ing ? zed_ingestor_->stamp_ms() : 0,
                         G->get_node("zed").has_value(), zed_ready_, fitter_->instances().size());
        }
    }

    run_instance_tracker(specialists);

    for (const auto& node : G->get_nodes_by_type("obstacle"))
        if (node.name().rfind("residual_", 0) == 0)
            process_residual_node(node);

    remove_by_occupancy_evidence();   // compelling free-space evidence, not a miss counter
    dissolve_explained_instances(specialists);

    // Keep the DSR graph tidy: if any obstacle node was created or deleted this cycle, re-run the radial
    // (twopi) layout ONCE — coalesced here rather than per-op so a multi-birth/death cycle triggers a single
    // relayout instead of a burst. No-op when headless (no graph viewer).
    if (graph_dirty_)
    {
        trigger_graph_layout_twopi();
        graph_dirty_ = false;
    }
}

namespace
{
// Overlap of two ORIENTED footprint boxes = the fraction of the SMALLER box's area that lies inside the
// LARGER one, sampled on a grid in the smaller box's local frame. Unlike an inscribed-circle test this is
// correct for ELONGATED boxes (a long merged box whose end overlaps a small box) — two physical bodies can't
// occupy the same footprint, so a high ratio means they are the same body / a duplicate. 1 = fully contained.
float footprint_overlap_ratio(const rc::ResidualState& A, const rc::ResidualState& B)
{
    const bool a_small = (A.w * A.d) <= (B.w * B.d);
    const rc::ResidualState& S = a_small ? A : B;      // smaller
    const rc::ResidualState& L = a_small ? B : A;      // larger
    const float cs = std::cos(S.yaw), ss = std::sin(S.yaw);
    const float cl = std::cos(L.yaw), sl = std::sin(L.yaw);
    constexpr int N = 6;
    int inside = 0, total = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            const float lx = (-0.5f + (i + 0.5f) / N) * S.w, ly = (-0.5f + (j + 0.5f) / N) * S.d;
            const float rx = S.cx + cs * lx - ss * ly, ry = S.cy + ss * lx + cs * ly;   // sample → room
            const float dx = rx - L.cx, dy = ry - L.cy;
            const float Lx = cl * dx + sl * dy, Ly = -sl * dx + cl * dy;                 // room → larger's local
            ++total;
            if (std::abs(Lx) <= 0.5f * L.w and std::abs(Ly) <= 0.5f * L.d) ++inside;
        }
    return total > 0 ? static_cast<float>(inside) / static_cast<float>(total) : 0.0f;
}
}  // namespace

void SpecificWorker::merge_overlapping_instances()
{
    if (cfg_.tracker_merge_overlap <= 0.0f) return;
    auto& insts = fitter_->instances();
    if (insts.size() < 2) return;

    std::vector<std::uint64_t> ids; ids.reserve(insts.size());
    for (auto& [id, _] : insts) ids.push_back(id);

    std::vector<std::uint64_t> removed;
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        if (std::count(removed.begin(), removed.end(), ids[i])) continue;
        for (std::size_t j = i + 1; j < ids.size(); ++j)
        {
            if (std::count(removed.begin(), removed.end(), ids[j])) continue;
            const auto ia = insts.find(ids[i]), ib = insts.find(ids[j]);
            if (ia == insts.end() or ib == insts.end()) continue;
            if (footprint_overlap_ratio(ia->second.model.state(), ib->second.model.state()) < cfg_.tracker_merge_overlap)
                continue;
            const bool keep_i = ia->second.matched_frames >= ib->second.matched_frames;
            const std::uint64_t drop = keep_i ? ids[j] : ids[i];
            std::print("residual_concept: [tracker] MERGE id={}\n", drop);
            fitter_->forget_node(drop);
            G->delete_node(drop);
            graph_dirty_ = true;                          // node deleted → relayout at end of cycle
            removed.push_back(drop);
            if (drop == ids[i]) break;
        }
    }
}

void SpecificWorker::retire_diverged_instances()
{
    if (cfg_.diverged_retire_frames <= 0) return;
    std::vector<std::uint64_t> doomed;
    for (auto& [id, inst] : fitter_->instances())
        if (inst.frames_diverged >= cfg_.diverged_retire_frames)
            doomed.push_back(id);
    for (const std::uint64_t id : doomed)
    {
        std::print("residual_concept: [tracker] RETIRE-DIVERGED id={}\n", id);
        fitter_->forget_node(id);
        G->delete_node(id);
        graph_dirty_ = true;                              // node deleted → relayout at end of cycle
    }
}

void SpecificWorker::run_instance_tracker(const std::vector<rc::SpecialistSdf>& specialists)
{
    merge_overlapping_instances();
    retire_diverged_instances();

    rc::TrackerParams tp;
    tp.gate_mahalanobis  = cfg_.tracker_gate_mahalanobis;
    tp.gate_fallback_m   = cfg_.tracker_gate_fallback_m;
    tp.birth_frames      = cfg_.tracker_birth_frames;
    tp.death_frames      = cfg_.tracker_death_frames;
    tp.birth_min_sep_m   = cfg_.tracker_birth_min_sep_m;
    tp.detection_noise_m = cfg_.tracker_detection_noise_m;
    tp.nll_cost          = cfg_.tracker_nll_cost;
    tracker_.set_params(tp);

    std::vector<rc::TrackView> tracks;
    tracks.reserve(fitter_->instances().size());
    for (auto& [id, inst] : fitter_->instances())
    {
        rc::TrackView t;
        t.id = id;
        const auto& s = inst.model.state();
        t.xy = {s.cx, s.cy};
        if (inst.belief_initialized)
        {
            t.cov = inst.belief.covariance().block<2, 2>(0, 0);
            t.has_cov = true;
        }
        // Negative-information death gate: only accrue a "miss" (toward retirement) when this obstacle
        // SHOULD be seen — its centre is within LiDAR range of the robot. Out of range → miss HELD → the
        // obstacle PERSISTS in the map when the robot drives away (no die+rebirth churn on revisit).
        const float range = cfg_.tracker_expected_visible_range_m;
        t.expected_visible = (range <= 0.0f) or (t.xy - robot_xy_).norm() < range;
        inst.expected_visible = t.expected_visible;
        tracks.push_back(t);
        inst.assigned_cluster_idx = -1;
    }

    std::vector<rc::DetectionView> dets;
    dets.reserve(clusters_.size());
    for (int i = 0; i < static_cast<int>(clusters_.size()); ++i)
        dets.push_back({clusters_[i].centroid, i});

    const auto res = tracker_.update(tracks, dets);

    // DEATH (only when death is enabled — residuals are transient so it's ON by default).
    if (cfg_.tracker_death_enabled)
        for (const std::uint64_t id : res.deaths)
        {
            std::print("residual_concept: [tracker] DEATH id={}\n", id);
            fitter_->forget_node(id);
            G->delete_node(id);
            graph_dirty_ = true;                          // node deleted → relayout at end of cycle
        }

    // ASSOCIATE: route each matched detection's cluster to its instance (read in observe()).
    for (int d = 0; d < static_cast<int>(dets.size()); ++d)
        if (res.assignment[d] >= 0)
        {
            const std::uint64_t id = tracks[res.assignment[d]].id;
            if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
                it->second.assigned_cluster_idx = dets[d].slice_index;
        }

    // BIRTH: spawn a residual obstacle from each promoted cluster, seeded with its PCA footprint + z-band.
    const float dis_margin = cfg_.cluster.explain_margin_m;
    for (const int d : res.births)
    {
        const auto& cl = clusters_[dets[d].slice_index];
        const Eigen::Vector3f centroid(cl.centroid.x(), cl.centroid.y(), 0.5f * (cl.z_min + cl.z_max));
        // Don't birth a cluster whose centre lies inside a modelled object — it would be dissolved the same
        // cycle (birth↔dissolve loop). Same signed-SDF test as dissolve_explained_instances → they agree.
        bool inside_specialist = false;
        for (const auto& sdf : specialists)
            if (sdf(centroid) < dis_margin) { inside_specialist = true; break; }
        if (inside_specialist)
            continue;
        const auto new_id = scene_graph_->create_instance_from_detection(
            centroid, cl.yaw_seed, cl.w_seed, cl.d_seed, std::max(0.05f, cl.z_max - cl.z_min),
            room_node_id_, current_ts_);
        if (new_id != 0)
        {
            fitter_->note_birth(new_id, cl.centroid);
            graph_dirty_ = true;                          // node created → relayout at end of cycle
        }
    }
}

void SpecificWorker::publish_grid_display(const rc::OccupancyGrid::CellExplained& explained)
{
    static int t = 0;
    if ((t++ % 5) != 0) return;                         // ~2 Hz display update (attr-only; no node churn)
    if (not G or room_node_id_ == 0 or not rt_api_) return;
    auto room = G->get_node(room_node_id_);
    if (not room.has_value()) return;

    // Create the `grid` node once (owned; cleaned up on exit via the "grid" [Owns] entry).
    auto gopt = G->get_node("grid");
    if (not gopt.has_value())
    {
        DSR::Node gn = DSR::Node::create<grid_node_type>("grid");
        const float rpx = G->get_attrib_by_name<pos_x_att>(room.value()).value_or(200.f);
        const float rpy = G->get_attrib_by_name<pos_y_att>(room.value()).value_or(200.f);
        G->add_or_modify_attrib_local<pos_x_att>(gn, rpx);
        G->add_or_modify_attrib_local<pos_y_att>(gn, rpy + 100.f);
        const auto idopt = G->insert_node(gn);
        if (not idopt.has_value()) return;
        rt_api_->insert_or_assign_edge_RT(room.value(), idopt.value(), {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f});
        graph_dirty_ = true;
        std::println("[grid] published 'grid' node id={} under room", idopt.value());
        gopt = G->get_node(idopt.value());
        if (not gopt.has_value()) return;
    }

    // Residual cell centres (occupied ∧ ¬explained), flat [x0,y0,x1,y1,…] room-frame, + the cell size, so the
    // voxelizer can draw a square per cell. The z-band could follow later; a flat display height is enough now.
    auto gn = gopt.value();
    std::vector<float> cells = grid_.residual_cell_centres(explained);
    G->runtime_checked_add_or_modify_attrib_local(gn, "grid_cells_xy", cells);
    G->runtime_checked_add_or_modify_attrib_local(gn, "grid_cell_size", static_cast<float>(grid_.params().cell_size_m));
    G->update_node(gn);
}

void SpecificWorker::process_residual_node(const DSR::Node& node)
{
    fitter_->ensure_instance(node, room_node_id_);
    auto& inst = fitter_->instances().at(node.id());
    ++inst.processed_cycles;
    inst.last_obs_timestamp_ms = current_ts_;

    // NOTE: ZED depth now enters via the DETECTION cloud (backproject_fov, fed to clustering), so the dense
    // ZED points are already part of this instance's cluster and the box-region backproject is redundant
    // (keeping both would double-count the forward points). inst.zed_points stays empty.
    inst.zed_points.clear();

    const auto obs = fitter_->observe(inst);
    if (not obs.has_fresh_data and inst.matched_frames < 3)
        return;

    // Infer the support surface this obstacle rests on (floor or a tabletop) → the fit anchors its box bottom
    // there, so a floor object rises from the floor and an on-table object rises from the table.
    if (obs.has_fresh_data and obs.cluster != nullptr)
        inst.support_z = support_z_for(obs.cluster->centroid.x(), obs.cluster->centroid.y(), obs.cluster->z_min);

    const float fe = fitter_->run_inference(inst, obs);
    if (auto n = G->get_node(node.id()); n.has_value())
        scene_graph_->step_write_model(inst, n.value(), fe, room_node_id_, current_ts_);
    inst.prev_free_energy = fe;
}

void SpecificWorker::dissolve_explained_instances(const std::vector<rc::SpecialistSdf>& specialists)
{
    if (specialists.empty())
        return;
    const float margin = cfg_.cluster.explain_margin_m;
    std::vector<std::uint64_t> doomed;
    for (auto& [id, inst] : fitter_->instances())
    {
        const auto& s = inst.model.state();
        const Eigen::Vector3f centre(s.cx, s.cy, s.cz);
        // The instance's centre now lies inside (or on) a modelled object → a specialist has claimed its
        // core. Invalidate it; the next cycle's residual re-cluster rebuilds the survivors (the fission).
        for (const auto& sdf : specialists)
            if (sdf(centre) < margin) { doomed.push_back(id); break; }
    }
    for (const std::uint64_t id : doomed)
    {
        std::print("residual_concept: [dissolve] id={} — a specialist claimed its interior\n", id);
        fitter_->forget_node(id);
        G->delete_node(id);
        graph_dirty_ = true;                              // node deleted → relayout at end of cycle
    }
}

void SpecificWorker::remove_by_occupancy_evidence()
{
    if (not lidar_ingestor_)
        return;
    const auto& sweep = lidar_ingestor_->sweep_room();
    if (sweep.empty())
        return;
    const Eigen::Vector3f origin = lidar_ingestor_->origin_room();
    // Decision boundary in log-odds: remove when P(occupied) < removal_prob.
    const float p = std::clamp(cfg_.removal_prob, 1e-3f, 0.5f);
    const float L_remove = std::log(p / (1.0f - p));
    const float L_max = cfg_.occupancy_logodds_max;

    // Sensor-model log-likelihood-ratios (physical rates). See-through (free) evidence is only admitted when
    // the obstacle is UNOBSERVED this cycle: a box is a SOLID abstraction of a possibly-HOLLOW obstacle (a
    // table is empty under its top), so beams passing through the interior are NOT absence evidence while we
    // are actively seeing the object. Occupancy evidence always counts.
    const float pd = std::clamp(cfg_.carve.detection_prob, 1e-3f, 1.0f - 1e-3f);
    const float pc = std::clamp(cfg_.carve.clutter_prob,   1e-3f, 1.0f - 1e-3f);
    const float llr_occ  = std::log(pd / pc);
    const float llr_free = std::log((1.0f - pd) / (1.0f - pc));

    std::vector<std::uint64_t> doomed;
    for (auto& [id, inst] : fitter_->instances())
    {
        if (not inst.belief_initialized)
            continue;
        const auto& s = inst.model.state();
        // Surface localisation uncertainty for the soft carve = the belief's footprint position σ.
        const auto& S = inst.belief.covariance();
        const float surf_sigma = std::sqrt(std::max(0.0f, 0.5f * (S(0, 0) + S(1, 1))));
        const rc::ResidualClusterer::CarveEvidence ev = rc::ResidualClusterer::carve_box(
            origin, sweep, s.cx, s.cy, s.yaw, s.w, s.d, s.cz - 0.5f * s.height, s.cz + 0.5f * s.height,
            surf_sigma, cfg_.carve);
        if (ev.n_reached == 0)                                  // not looked at this cycle → HOLD (no update)
            continue;
        // Observed = a fresh residual cluster associated to this instance this cycle (we are seeing it).
        const bool observed = inst.assigned_cluster_idx >= 0;
        const float raw = ev.e_occ * llr_occ + (observed ? 0.0f : ev.e_free * llr_free);
        const float delta = (llr_occ > 1e-6f) ? llr_occ * std::tanh(raw / llr_occ) : raw;
        inst.occupancy_logodds = std::clamp(inst.occupancy_logodds + delta, -L_max, L_max);
        if (inst.occupancy_logodds < L_remove)
            doomed.push_back(id);
    }
    for (const std::uint64_t id : doomed)
    {
        std::print("residual_concept: [carve] id={} — free-space evidence: the space is empty (logodds<{:.2f})\n",
                   id, L_remove);
        fitter_->forget_node(id);
        G->delete_node(id);
        graph_dirty_ = true;
    }
}

void SpecificWorker::del_node_slot(std::uint64_t id)
{
    if (fitter_)
        fitter_->forget_node(id);
    trigger_graph_layout_twopi();
}

void SpecificWorker::emergency()   { std::print("residual_concept: emergency()\n"); }
void SpecificWorker::restore()     { std::print("residual_concept: restore()\n"); }

int SpecificWorker::startup_check()
{
    std::print("residual_concept: startup_check()\n");
    QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
    return 0;
}
