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
#include <fstream>
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
    rc::semantic_self_test();         // RGB-semantic floor down-weighting (height gate / freshness / class / safety)
    rc::floor_plane_self_test();      // data-driven floor-plane fit (offset+tilt recovery / flat-z0 no-op / hold)

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
    soft_objects_.clear();
    if (not inner_eigen_)
        return out;
    const float sensor_var = cfg_.cluster.explain_sensor_sigma_m * cfg_.cluster.explain_sensor_sigma_m;
    for (const char* type : {"table", "chair", "cylinder"})
        for (const auto& n : G->get_nodes_by_type(type))
        {
            const float w = G->get_attrib_by_name<width_m_att> (n).value_or(0.0f);
            const float d = G->get_attrib_by_name<depth_m_att> (n).value_or(0.0f);
            const float h = G->get_attrib_by_name<height_m_att>(n).value_or(0.0f);
            static int sk = 0;
            const bool log_skip = (sk++ % 200) == 0;
            if (w <= 0.0f or d <= 0.0f or h <= 0.0f)
            {
                if (log_skip) std::println("[collapse-skip] '{}' w={:.2f} d={:.2f} h={:.2f} (needs all >0)", n.name(), w, d, h);
                continue;
            }
            const auto T = inner_eigen_->get_transformation_matrix("room", n.name(), 0);
            if (not T.has_value())
            {
                if (log_skip) std::println("[collapse-skip] '{}' no room→object transform", n.name());
                continue;
            }
            Eigen::Matrix4d M;
            { const auto& s = T.value().matrix(); for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) M(i, j) = s(i, j); }
            const Eigen::Vector3f centre(static_cast<float>(M(0, 3)), static_cast<float>(M(1, 3)),
                                         static_cast<float>(M(2, 3)) + 0.5f * h);
            const float yaw = std::atan2(static_cast<float>(M(1, 0)), static_cast<float>(M(0, 0)));
            const float hx = 0.5f * w, hy = 0.5f * d, hz = 0.5f * h;
            out.push_back([centre, yaw, hx, hy, hz](const Eigen::Vector3f& p)
                          { return rc::ResidualClusterer::box_sdf_3d(p, centre, yaw, hx, hy, hz); });
            // Soft-collapse descriptor: 2D footprint + top height + σ's from the object's OWN published position
            // covariance (rt_covariance, which already folds in the localisation chain → range-aware) ⊕ sensor.
            float var_xy = 0.0f, var_z = 0.0f;
            if (const auto e = G->get_edge(room_node_id_, n.id(), "RT"); e.has_value())
                if (const auto cov = G->get_attrib_by_name<rt_covariance_att>(e.value());
                    cov.has_value() and cov->get().size() >= 15)
                {
                    var_xy = 0.5f * (cov->get()[0] + cov->get()[7]);   // ½(σ²x+σ²y)
                    var_z  = cov->get()[14];                            // σ²z (6×6 diagonal index 2)
                }
            soft_objects_.push_back({centre.x(), centre.y(), yaw, hx, hy,
                                     static_cast<float>(M(2, 3)) + h,          // z_top (box top face)
                                     std::sqrt(std::max(0.0f, var_xy) + sensor_var),
                                     std::sqrt(std::max(0.0f, var_z)  + sensor_var)});
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
    // NOTE: modelled OBJECTS are handled separately as a SOFT collapse (explained_probability, see compute()),
    // not as hard boolean explainers — so their marginal uncertainty attenuates the residual continuously. This
    // list is only the hard infrastructure (floor/ceiling/robot/walls), which are genuine 0/1 explanations.
    (void) objects;
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
        {
            ego_reliability_ = compute_ego_reliability();   // <1 while moving → down-weight the whole sweep
            // DATA-DRIVEN FLOOR PLANE: fit the floor from the RAW LiDAR (best floor coverage) and reference the
            // grid's obstacle band to it, so a new scenario's offset/tilted floor never latches as phantoms. The
            // estimator always runs+logs (so the offset is visible); the grid uses it only when the flag is on.
            floor_plane_ = rc::estimate_floor_plane(lidar_ingestor_->sweep_room(), lidar_ingestor_->origin_room(),
                                                    cfg_.cluster.floor_z0, cfg_.cluster.floor_slope,
                                                    cfg_.floor_plane, floor_plane_);
            if (cfg_.floor_plane.enabled and floor_plane_.valid)
                grid_.set_floor_plane(floor_plane_.a, floor_plane_.b, floor_plane_.c);
            else
                grid_.set_floor_plane(0.0f, 0.0f, 0.0f);
            static int fpc = 0;
            if ((fpc++ % 40) == 0 and floor_plane_.valid)
                std::println("[floor-plane] z = {:.3f}·x + {:.3f}·y + {:.3f}  (offset {:.1f} cm, tilt {:.1f}°) {}",
                             floor_plane_.a, floor_plane_.b, floor_plane_.c, floor_plane_.c * 100.0f,
                             std::atan(std::hypot(floor_plane_.a, floor_plane_.b)) * 57.2958f,
                             cfg_.floor_plane.enabled ? "[APPLIED]" : "[log-only, flag off]");
            const auto& lidar_sweep = filtered_lidar_sweep();   // bpearl floor grazing removed (per-device band)
            grid_.integrate_sweep(lidar_ingestor_->origin_room(), lidar_sweep,
                                  /*begin_cycle=*/true, ego_reliability_);            // accumulate LiDAR
            integrate_zed_into_grid();   // accumulate dense ZED FoV as a second sensor (fills grazed tabletops)
            grid_.commit_cycle();        // ONE log-odds update per cell (hit precedence) — the stability fix
            log_grid_diag();
        }
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

    // Diagnostic: each object's SOFT-collapse σ (from its OWN published position covariance ⊕ sensor noise). A
    // tiny σ on a distant object ⇒ overconfident upstream covariance (chain-cov off / room localiser too sure),
    // which is why it wouldn't collapse — the honest signal, surfaced here rather than papered over by a floor.
    static int cs = 0;
    if ((cs++ % 40) == 0)
    {
        const int n_tab = static_cast<int>(G->get_nodes_by_type("table").size());
        const int n_chr = static_cast<int>(G->get_nodes_by_type("chair").size());
        const int n_cyl = static_cast<int>(G->get_nodes_by_type("cylinder").size());
        std::string s;
        for (const auto& o : soft_objects_) s += std::format("[σxy={:.3f} σz={:.3f} ztop={:.2f} hx={:.2f} hy={:.2f}] ", o.sigma_xy, o.sigma_z, o.z_top, o.hx, o.hy);
        std::println("[collapse] nodes table={} chair={} cylinder={} → {} objects; {}",
                     n_tab, n_chr, n_cyl, soft_objects_.size(), s.empty() ? "(none)" : s);
    }

    // ── PHASE-1 REBUILD: read the RESIDUAL out of the grid (occupied ∧ ¬explained) and publish it. A cell is
    //    "explained" if any known-model explainer accounts for its representative point → walls + specialists
    //    drop out (floor/ceiling already excluded by the nav band), leaving object-only components. Evidence is
    //    never deleted, only masked at read-out — a specialist mis-fit cannot erase a real obstacle. Also
    //    publishes the residual cells on a `grid` node under room for the voxelizer 3-D display. ──
    if (grid_ready_)
    {
        // p_explained(cell) = max( hard infrastructure (0/1),  soft objects  Φ(−sdf2D/σ_xy) ). The object term
        // marginalises the 2D FOOTPRINT over the object's own published covariance (range-aware via the
        // localisation chain), plus the fit margin. NO z-gate: this is a FLOOR-navigation costmap — the robot
        // must avoid the whole object footprint regardless of what sits on it, and a per-cm z-ceiling on the
        // box top spuriously kept the entire tabletop whenever the fitted height was a few cm short.
        const float fitm = cfg_.cluster.explain_fit_margin_m;       // under-fit slack (< planner clearance)
        const rc::OccupancyGrid::CellExplained cell_explained =
            [&explainers, this, fitm](float x, float y, float zlo, float zhi) -> float
            {
                const Eigen::Vector3f p(x, y, 0.5f * (zlo + zhi));
                for (const auto& ex : explainers) if (ex(p)) return 1.0f;          // hard: floor/ceiling/robot/walls
                float pe = 0.0f;
                for (const auto& o : soft_objects_)
                {
                    // 2D oriented-box signed distance of (x,y) to the object footprint (negative inside); the
                    // fit margin shifts the 0.5-crossing OUTWARD by `fitm` to absorb a known under-fit / legs.
                    const float c = std::cos(o.yaw), s = std::sin(o.yaw), dx = x - o.cx, dy = y - o.cy;
                    const float lx = c * dx + s * dy, ly = -s * dx + c * dy;
                    const float qx = std::abs(lx) - o.hx, qy = std::abs(ly) - o.hy;
                    const float sdf2d = std::hypot(std::max(qx, 0.0f), std::max(qy, 0.0f)) + std::min(std::max(qx, qy), 0.0f) - fitm;
                    pe = std::max(pe, 0.5f * std::erfc(sdf2d / (std::max(1e-4f, o.sigma_xy) * 1.41421356f)));   // Φ(−sdf2D/σxy)
                }
                return pe;
            };
        const auto comps = grid_.occupied_components(2, cell_explained, grid_.params().inflate_radius_m);
        static int gc = 0;
        if ((gc++ % 20) == 0)
        {
            std::size_t maxc = 0; for (const auto& c : comps) maxc = std::max<std::size_t>(maxc, c.n_cells);
            std::println("[grid] residual components={} max_cells={} (walls/specialists subtracted at read-out)",
                         comps.size(), maxc);
        }
        // `grid` node under room: residual cells + inflated border (display) + inflated component hulls
        // (`grid_polygons`, encoded) which the controller reads and plans against alongside the known objects.
        // Residual obstacle NODES are no longer uploaded.
        publish_grid_display(cell_explained, comps);
    }

    // Keep the DSR graph tidy: relayout once if a node was created/deleted this cycle (no-op when headless).
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


void SpecificWorker::integrate_zed_into_grid()
{
    // Second sensor for the occupancy grid: the dense ZED depth FoV. LiDAR rings only GRAZE horizontal surfaces
    // (a tabletop) so they never fill it; the ZED covers it densely, and the extra evidence per cell is what
    // makes the costmap stable. Integrated with the CAMERA as ray origin so the z-aware ray-carve stays correct.
    if (not cfg_.grid_zed_enabled or not zed_ingestor_ or not inner_eigen_) return;
    zed_ingestor_->pump();                                   // main-thread drain of the newest depth frame
    if (not zed_ingestor_->has_depth()) return;

    // room←zed at the sweep stamp (Nearest — the camera pose moves with the robot).
    const auto rt = inner_eigen_->get_transformation_matrix("room", "zed", current_ts_, "RT",
                                                            DSR::RT_API::TimeQuery::Nearest);
    if (not rt.has_value()) return;
    const Eigen::Matrix4f room_T_cam = rt->matrix().cast<float>();
    if (not room_T_cam.allFinite()) return;

    auto pts = rc::backproject_fov(zed_ingestor_->depth(), zed_ingestor_->intrinsics(),
                                   room_T_cam, cfg_.zed_boost);
    if (pts.empty()) return;
    const Eigen::Vector3f cam_origin = room_T_cam.col(3).head<3>();
    // ROBUST infrastructure rejection BEFORE the grid: ZED stereo depth noise grows with range² (σ0+q·r²), so
    // the grid's fixed nav-band lets noisy floor points read as obstacles → phantom floor obstacles everywhere.
    // subtract_infrastructure removes floor/ceiling/walls with a k·σ(r) band (the same filter the old ZED
    // detection path used); what remains is real ZED residual (tabletops, objects). LiDAR is precise and needs
    // no such filter (its own nav-band suffices).
    const std::size_t before = pts.size();
    pts = rc::ResidualClusterer::subtract_infrastructure(pts, cam_origin, read_room_polygon(), cfg_.zed_infra);
    static int zc = 0;
    if ((zc++ % 40) == 0)
        std::println("[zed-grid] fov={} → {} after infra-subtract (floor/ceiling/wall removed)", before, pts.size());
    if (pts.empty()) return;

    // RGB-SEMANTIC floor down-weighting (a second, uncorrelated cue vs the ZED floor phantoms that survive the
    // geometric infra-subtract above): sample the dense YOLO-sem label of each surviving ZED point (reproject it
    // back into its own frame → exact pixel) and down-weight the HIT of a floor-class near-floor return. Height-
    // gated + freshness-decayed + never-discard (see residual_semantic.h). No-op when Semantic.DownweightFloor=off.
    std::vector<float> hit_scale;
    const std::vector<float>* scale_ptr = nullptr;
    if (cfg_.semantic_floor.enabled and refresh_semantic_map())
    {
        const Eigen::Matrix4f cam_T_room = room_T_cam.inverse();
        const float age_s = current_ts_ > semantic_map_.stamp_ms
                          ? static_cast<float>(current_ts_ - semantic_map_.stamp_ms) * 1e-3f : 0.0f;
        hit_scale = rc::semantic_obstacle_weights(pts, cam_T_room, zed_ingestor_->intrinsics(),
                                                  zed_ingestor_->depth().width, zed_ingestor_->depth().height,
                                                  semantic_map_, cam_origin, age_s, cfg_.semantic_floor);
        scale_ptr = &hit_scale;
        static int sc = 0;
        if ((sc++ % 40) == 0)
        {
            long dw = 0; float mn = 1.0f;
            for (float w : hit_scale) if (w < 0.999f) { ++dw; mn = std::min(mn, w); }
            std::println("[zed-sem] {}/{} floor pts down-weighted (min w={:.2f}, map age={:.2f}s)",
                         dw, hit_scale.size(), mn, age_s);
        }
    }
    grid_.integrate_sweep(cam_origin, pts, /*begin_cycle=*/false, ego_reliability_, scale_ptr);   // same cycle+ego
}

bool SpecificWorker::refresh_semantic_map()
{
    // Read the voxelizer's `semantic` node (dense YOLO-sem label map under `zed`). The blob is large and published
    // at ~2 Hz, so re-copy it only when the frame_id advanced; otherwise reuse the cache. Main-thread graph read.
    const auto node = G->get_node("semantic");
    if (not node.has_value()) return semantic_map_.valid();   // node not up yet → keep whatever we had (or invalid)

    const auto fid = G->get_attrib_by_name<semantic_frame_id_att>(node.value());
    const int  frame_id = fid.has_value() ? fid.value() : -1;
    if (frame_id == semantic_map_.frame_id and semantic_map_.valid())
        return true;                                          // unchanged → keep the cached copy

    const auto w  = G->get_attrib_by_name<semantic_width_att>(node.value());
    const auto h  = G->get_attrib_by_name<semantic_height_att>(node.value());
    const auto ts = G->get_attrib_by_name<semantic_timestamp_ms_att>(node.value());
    const auto lb = G->get_attrib_by_name<semantic_labels_att>(node.value());
    if (not (w.has_value() and h.has_value() and lb.has_value())) return semantic_map_.valid();

    semantic_map_.width    = w.value();
    semantic_map_.height   = h.value();
    semantic_map_.stamp_ms = ts.has_value() ? ts.value() : 0;
    semantic_map_.frame_id = frame_id;
    semantic_map_.labels   = lb.value().get();                // copy the flattened class-id buffer (ref_wrapper → get)
    return semantic_map_.valid();
}

const std::vector<Eigen::Vector3f>& SpecificWorker::filtered_lidar_sweep()
{
    // Drop the LOW bpearl lidar's FLOOR-GRAZING returns (device-specific higher floor band) while keeping helios
    // and bpearl's real (>band) low-obstacle returns — so bpearl still catches short obstacles helios misses,
    // without ringing the robot with phantom floor cells. plane_id: helios=0, bpearl=1 (empty ⇒ fused → keep all).
    const auto& pts = lidar_ingestor_->sweep_room();
    const auto& pid = lidar_ingestor_->plane_id();
    if (pid.size() != pts.size()) return pts;                    // no per-device tag (fused lidar3D) → pass through
    const Eigen::Vector3f o = lidar_ingestor_->origin_room();
    const float bz0 = cfg_.cluster.bpearl_floor_z0, slope = cfg_.cluster.floor_slope;
    lidar_filtered_.clear(); lidar_filtered_.reserve(pts.size());
    long dropped = 0;
    for (std::size_t i = 0; i < pts.size(); ++i)
    {
        if (pid[i] == 1)                                         // bpearl → apply its higher floor band
        {
            const float range = std::hypot(pts[i].x() - o.x(), pts[i].y() - o.y());
            if (pts[i].z() < bz0 + slope * range) { ++dropped; continue; }
        }
        lidar_filtered_.push_back(pts[i]);
    }
    static int bc = 0;
    if ((bc++ % 40) == 0)
        std::println("[bpearl-floor] dropped {} grazing floor pts (band z0={:.2f})", dropped, bz0);
    return lidar_filtered_;
}

float SpecificWorker::compute_ego_reliability() const
{
    // EGO-MOTION precision: the room<-robot RT edge carries the robot's body twist. Fast translation/rotation
    // means more pose jitter + motion blur this sweep → trust it less: 1/(1 + |v|/vel0 + |ω|/omega0). Still → 1.
    float v = 0.0f, w = 0.0f;
    if (rt_api_)
        if (const auto robots = G->get_nodes_by_type("robot"); not robots.empty())
            if (auto e = rt_api_->get_edge_RT(robots.front(), room_node_id_); e.has_value())
            {
                if (auto tv = G->get_attrib_by_name<rt_translation_velocity_att>(e.value());
                    tv.has_value() and tv->get().size() >= 2)
                    v = std::hypot(tv->get()[0], tv->get()[1]);
                if (auto rv = G->get_attrib_by_name<rt_rotation_euler_xyz_velocity_att>(e.value());
                    rv.has_value() and rv->get().size() >= 3)
                    w = std::abs(rv->get()[2]);
            }
    const float v0 = std::max(1e-3f, cfg_.motion_vel0_mps), w0 = std::max(1e-3f, cfg_.motion_omega0_rps);
    return 1.0f / (1.0f + v / v0 + w / w0);
}

void SpecificWorker::log_grid_diag()
{
    if (not grid_ready_) return;
    const auto& d = grid_.last_sweep_diag();
    static long cyc = 0;
    static std::ofstream f;
    if (not f.is_open())
    {
        f.open("etc/grid_diag.csv", std::ios::out | std::ios::trunc);
        f << "cycle,occupied,hits,misses,miss_blocked_zaware,latched,released,hit_then_cleared\n";
    }
    f << cyc << ',' << grid_.occupied_count() << ',' << d.hits << ',' << d.misses << ','
      << d.miss_blocked_zaware << ',' << d.cells_latched << ',' << d.cells_released << ','
      << d.hit_then_cleared << '\n';
    if ((cyc % 20) == 0)
    {
        f.flush();
        std::println("[grid-diag] occ={} hits={} miss={} zaware_block={} latch={} release={} hit_then_cleared={}",
                     grid_.occupied_count(), d.hits, d.misses, d.miss_blocked_zaware,
                     d.cells_latched, d.cells_released, d.hit_then_cleared);
    }
    ++cyc;
}

void SpecificWorker::publish_grid_display(const rc::OccupancyGrid::CellExplained& explained,
                                          const std::vector<rc::OccComponent>& comps)
{
    static int t = 0;
    if ((t++ % 5) != 0) return;                         // ~2 Hz display update (attr-only; no node churn)
    if (not G or room_node_id_ == 0 or not rt_api_) return;
    auto room = G->get_node(room_node_id_);
    if (not room.has_value()) return;

    // Create the `residual` node once (owned; cleaned up on exit via the "residual" [Owns] entry). Node NAME is
    // "residual"; node TYPE stays grid_node_type ("grid") — consumers still key attrs off the grid_* type.
    auto gopt = G->get_node("residual");
    if (not gopt.has_value())
    {
        DSR::Node gn = DSR::Node::create<grid_node_type>("residual");
        const float rpx = G->get_attrib_by_name<pos_x_att>(room.value()).value_or(200.f);
        const float rpy = G->get_attrib_by_name<pos_y_att>(room.value()).value_or(200.f);
        G->add_or_modify_attrib_local<pos_x_att>(gn, rpx);
        G->add_or_modify_attrib_local<pos_y_att>(gn, rpy + 100.f);
        const auto idopt = G->insert_node(gn);
        if (not idopt.has_value()) return;
        rt_api_->insert_or_assign_edge_RT(room.value(), idopt.value(), {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f});
        graph_dirty_ = true;
        std::println("[grid] published 'residual' node id={} under room", idopt.value());
        gopt = G->get_node(idopt.value());
        if (not gopt.has_value()) return;
    }

    // Residual cell centres (occupied ∧ ¬explained), published as the REGISTERED `residual_pts` attribute
    // (flat x,y,z triples, room frame; z = a small display height). runtime_checked rejects unregistered
    // names, so we reuse residual_pts (a vector<float>) rather than a bespoke grid attr. Voxelizer reads it
    // off the `grid` node and draws a cell per point.
    auto gn = gopt.value();
    const auto to_xyz = [](const std::vector<float>& xy) {
        std::vector<float> xyz; xyz.reserve(xy.size() / 2 * 3);
        for (std::size_t i = 0; i + 1 < xy.size(); i += 2) { xyz.push_back(xy[i]); xyz.push_back(xy[i + 1]); xyz.push_back(0.02f); }
        return xyz;
    };
    // Two display layers (dedicated, type-checked grid attrs): grid_occupied_cells = raw OCCUPIED cells
    // (colour A) with z = the cell's REAL top height (hit z-band max), so a 3-D consumer can raise a
    // surface to the true obstacle height; grid_border_cells = the INFLATED clearance ring (colour B),
    // kept at a flat display z (clearance is a footprint, not an obstacle).
    G->add_or_modify_attrib_local<grid_occupied_cells_att>(gn, grid_.residual_cell_centres_xyz(explained));
    G->add_or_modify_attrib_local<grid_border_cells_att>  (gn, to_xyz(grid_.inflated_border_centres(explained, grid_.params().inflate_radius_m)));
    G->add_or_modify_attrib_local<grid_cell_size_att>     (gn, static_cast<float>(grid_.params().cell_size_m));

    // Inflated component HULLS for the CONTROLLER's planner, encoded in grid_obstacle_hulls:
    //   [ P, (V, x0,y0, …, x_{V-1},y_{V-1}) × P ]   — P polygons, each V vertices (room-frame footprint,
    // already half-robot-width inflated). The controller decodes this and plans around each hull together
    // with the known object boxes.
    // Prefer each component's CONCAVE outline loops (C/U/L concavities stay navigable for the planner);
    // fall back to the convex hull only when the trace was degenerate (empty). Count total polygons first.
    std::size_t n_poly = 0;
    for (const auto& c : comps) n_poly += c.outline.empty() ? 1 : c.outline.size();
    std::vector<float> poly; poly.push_back(static_cast<float>(n_poly));
    const auto push_ring = [&](const std::vector<Eigen::Vector2f>& ring) {
        poly.push_back(static_cast<float>(ring.size()));
        for (const auto& v : ring) { poly.push_back(v.x()); poly.push_back(v.y()); }
    };
    for (const auto& c : comps)
    {
        if (c.outline.empty()) push_ring(c.hull);
        else for (const auto& ring : c.outline) push_ring(ring);
    }
    G->add_or_modify_attrib_local<grid_obstacle_hulls_att>(gn, std::move(poly));

    // ── BELIEF FIELD for the planner (plan over belief, not geometry): dense row-major P (collision RISK) and
    //    Var[P] (EPISTEMIC), collapsed to zero wherever a modelled object explains the cell. grid_field_meta =
    //    [xmin, ymin, cell, w, h] lets the consumer index the arrays without any other assumptions. ──
    std::vector<float> field_prob, field_var;
    grid_.occupancy_fields(field_prob, field_var, explained);
    // ASYMMETRIC temporal low-pass on the published field: risk rises at ema_up (instant → never lag a new
    // obstacle, safe), falls at ema_down (slow → a flickering edge cell holds its risk, stable). Occupancy latch
    // is untouched. Re-seed on first publish / extent change.
    if (pub_prob_ema_.size() != field_prob.size())
    { pub_prob_ema_ = field_prob; pub_var_ema_ = field_var; }
    else
    {
        const float au = cfg_.grid_field_ema_up, ad = cfg_.grid_field_ema_down;
        for (std::size_t i = 0; i < field_prob.size(); ++i)
        {
            const float a = (field_prob[i] >= pub_prob_ema_[i]) ? au : ad;   // rise fast, fall slow
            pub_prob_ema_[i] += a * (field_prob[i] - pub_prob_ema_[i]);
            pub_var_ema_[i]  += ad * (field_var[i]  - pub_var_ema_[i]);       // variance: light symmetric smoothing
        }
    }
    std::vector<float> meta{grid_.xmin(), grid_.ymin(), grid_.cell_size(),
                            static_cast<float>(grid_.width()), static_cast<float>(grid_.height())};
    G->add_or_modify_attrib_local<grid_occupancy_prob_att>(gn, pub_prob_ema_);
    G->add_or_modify_attrib_local<grid_occupancy_var_att> (gn, pub_var_ema_);
    G->add_or_modify_attrib_local<grid_field_meta_att>    (gn, std::move(meta));
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
