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
#include <charconv>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <locale>
#include <print>
#include <string_view>

#include <QCoreApplication>
#include <QTimer>

#include <dsr/api/dsr_api.h>
#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::stamp_creation

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

    // Shadow-mode birth/death record (CONCEPT_AGENT_LIFECYCLE.md §4.2). Recording only — see
    // log_phantom_event(). Truncating: one file per run.
    phantom_log_.open("etc/residual_phantom_events.csv");
    GenericWorker::initialize();

    if (not G)
    {
        qWarning() << "residual_concept: DSR graph not available in initialize()";
        return;
    }

    // Agent-presence protocol wiring (mirrors bottle_concept — Degraded DEBOUNCE is the CLAUDE.md rule).
    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    // Colour this agent's node in the graph view by its live health: the coordinator already
    // publishes the presence lifecycle; this adds the external FSM axis (Initialize/Compute/
    // Emergency/Restore). Generic discovery via objectName(), so genericworker regeneration
    // cannot break it.
    presence_coordinator_.attach_state_machine(&statemachine);
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
    // Post schema-migration every concept publishes its instance as a generic `object` node
    // (class in the object_subtype string attr), so the whole modelled-furniture set is exactly
    // get_nodes_by_type("object") — no per-class type list needed here (any object with a valid
    // box becomes an explainer).
    for (const auto& n : G->get_nodes_by_type("object"))
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
            bool have_cov = false;
            if (const auto e = G->get_edge(room_node_id_, n.id(), "RT"); e.has_value())
                if (const auto cov = G->get_attrib_by_name<rt_covariance_att>(e.value());
                    cov.has_value() and cov->get().size() >= 15)
                {
                    var_xy = 0.5f * (cov->get()[0] + cov->get()[7]);   // ½(σ²x+σ²y)
                    var_z  = cov->get()[14];                            // σ²z (6×6 diagonal index 2)
                    have_cov = true;
                }
            // A MISSING covariance is not benign and used to be completely silent. var_xy stays 0, so σ collapses
            // to the bare sensor term (3 cm) — the sharpest boundary this object can possibly have — and the
            // object therefore explains LESS area than it should, leaving a rim of unexplained cells that ships as
            // residual. The failure direction is straight into phantom production, so say so. Warn once per node.
            if (not have_cov and cov_warned_.insert(n.id()).second)
                std::println("residual_concept: [collapse] '{}' has NO rt_covariance on its room→object RT edge — "
                             "σ falls back to the sensor floor ({:.2f} m), so it under-claims its own footprint "
                             "and the rim will ship as residual", n.name(), std::sqrt(sensor_var));
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
    // Only tables are support surfaces. Every concept is now a generic `object` node, so filter
    // the object set down to tables. Tables carry object_subtype "round"/"square" (the shape model,
    // NOT the literal "table"), so the reliable class discriminator is the "table_" name prefix.
    for (const auto& n : G->get_nodes_by_type("object"))
    {
        if (not n.name().starts_with("table"))
            continue;
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

    // 0: ROOM FLOOR — the room's floor plane AS ACTUALLY FITTED (z=a·x+b·y+c), not the z=0 datum. The grid's hit
    //    band already follows the fit (FloorPlane.Enabled); scoring the read-out against z=0 instead means one
    //    surface with two datums, and the gap between them (4 cm offset + 0.6° tilt in this apartment, so up to
    //    ~8 cm across the room) is exactly the band in which a cell built from floor returns can never be
    //    explained as floor again. A grazing floor return lands higher with range, so the tolerance still GROWS
    //    with range (grazing angle × distance) — not a flat gate.
    const float fz0 = C.floor_z0, fsl = C.floor_slope;
    const bool  fp_on = cfg_.floor_plane.enabled and floor_plane_.valid;
    const float fpa = fp_on ? floor_plane_.a : 0.0f, fpb = fp_on ? floor_plane_.b : 0.0f,
                fpc = fp_on ? floor_plane_.c : 0.0f;
    ex.push_back([fz0, fsl, fpa, fpb, fpc, robot_xy](const Eigen::Vector3f& p)
                 { const float floor_z = fpa * p.x() + fpb * p.y() + fpc;
                   return p.z() < floor_z + fz0 + fsl * (p.head<2>() - robot_xy).norm(); });
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

// SHADOW-MODE birth/death recorder — CONCEPT_AGENT_LIFECYCLE.md §4.2, theory in MODEL_HISTORY.md §4.
// RECORDS ONLY; it can never alter a birth or a removal. NOTE: residual retires on divergence, not on sensor absence — p_detect does not apply.
void SpecificWorker::log_phantom_event(std::string_view event, std::uint64_t id, std::string_view name,
                                       float x, float y, const rc::ResidualInstance* inst, std::string_view note)
{
    if (not phantom_log_.is_open())
        return;
    rc::history::PhantomEvent e;
    e.event = event; e.id = id; e.name = name; e.x = x; e.y = y; e.note = note;
    // Observer pose → view bearing: the classifier failure is VIEWPOINT-dependent, so the eventual p_FA field
    // is keyed on (world cell × bearing), never place alone.
    if (inner_eigen_)
        if (const auto rtb = inner_eigen_->get_transformation_matrix("room", "body", 0); rtb.has_value())
        {
            const auto& Tm = rtb.value();
            e.robot_x = static_cast<float>(Tm(0, 3));
            e.robot_y = static_cast<float>(Tm(1, 3));
            e.robot_yaw = std::atan2(static_cast<float>(Tm(1, 0)), static_cast<float>(Tm(0, 0)));
            e.view_bearing = std::atan2(e.robot_y - y, e.robot_x - x);
            e.range_m = std::hypot(e.robot_x - x, e.robot_y - y);
        }
    if (inst)
    {
        e.age_cycles    = inst->processed_cycles;
        e.p_detect      = 0.0f;
        e.central_frac  = 0.0f;
        e.in_fov_frac   = (0 > 0) ? 1.0f : 0.0f;
        e.exist_logodds = 0.0f;
    }
    phantom_log_.write(e);
}

// One [perf] line per second: the loop's ACTUAL rate and what it costs.
//   period mean/worst — measured call-to-call, so it reports the rate the loop really runs at, not the
//     configured timer period. The worst gap in the window is what a stall feels like; the mean hides it.
//   busy — mean time spent INSIDE compute(). This loop is SWEEP-DRIVEN: with no fresh LiDAR sweep the cycle
//     returns immediately, so busy ≪ period is the healthy signature here, and busy → period means the
//     clustering/fit/grid work has saturated the cycle.
//   cpu — FPSCounter::get_cpu_use() is `times()`-based: user+sys ticks of the WHOLE PROCESS over the window's
//     wall time. It therefore counts the LiDAR/ZED ingestor and DDS reader threads too, and >100% is normal
//     on this multi-threaded agent (100% = one core saturated). Call it exactly once per window — it is
//     stateful (each call consumes the interval since the previous one).
namespace
{
// ── /proc/self/status field reader (kB) ───────────────────────────────────────────────────────────
// Returns -1 if the field is absent/unparseable. std::from_chars, NEVER atoi/strtol/`>>`: this is a Qt
// program and Qt calls setlocale(LC_ALL, "") at startup, which on these es_ES.UTF-8 machines makes every
// locale-sensitive parse in the process suspect (CLAUDE.md, "Parsing numbers from files"). from_chars is
// locale-independent by construction and reports failure instead of guessing.
long read_proc_status_kb(std::string_view key)
{
    std::ifstream st("/proc/self/status");
    if (not st.is_open())
        return -1;
    std::string line;
    while (std::getline(st, line))
    {
        if (not std::string_view{line}.starts_with(key))
            continue;
        const char* b = line.data() + key.size();
        const char* const e = line.data() + line.size();
        while (b < e and (*b == ':' or *b == ' ' or *b == '\t')) ++b;
        long v = -1;
        return std::from_chars(b, e, v).ec == std::errc{} ? v : -1;
    }
    return -1;
}

// ── diagnostic-CSV rotation ───────────────────────────────────────────────────────────────────────
// Open `path` for writing, ARCHIVING any existing file instead of truncating it.
//
// WHY: every diagnostic writer here used to open with ios::trunc, so the FIRST restart after an incident
// destroyed the evidence for that incident. Exactly that happened on 2026-08-08: residual_concept was
// OOM-killed holding 33.8 GB, and restarting it to investigate overwrote the floor_diag.csv rows for
// cycle 4080 — the last cycle it ever completed, and the one that would have said whether the read-out
// handed the publish path a pathological component set. The previous file now moves to
// etc/archive/<stem>-<when>.csv. Only the newest KEEP_RUNS archives per stem are kept, so a long-lived
// working copy cannot fill the disk with dead runs.
//
// The stamp is taken at ROTATION time (which is startup), so it reads as "archived when this run began",
// and the fixed-width zero-padded fields sort chronologically as plain text. Built with std::format and
// explicit integer fields rather than strftime, for the same locale reason as creation_stamp.h.
constexpr std::size_t KEEP_RUNS = 20;

void open_diag_csv(std::ofstream& f, const std::string& path)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path p{path};

    if (const auto sz = fs::file_size(p, ec); not ec and sz > 0)
    {
        const fs::path dir = p.parent_path() / "archive";
        fs::create_directories(dir, ec);

        const std::time_t secs = static_cast<std::time_t>(rc::provenance::now_ms() / 1000);
        std::tm tmv{};
        const std::string stamp = ::localtime_r(&secs, &tmv) != nullptr
            ? std::format("{:04}{:02}{:02}-{:02}{:02}{:02}", tmv.tm_year + 1900, tmv.tm_mon + 1,
                          tmv.tm_mday, tmv.tm_hour, tmv.tm_min, tmv.tm_sec)
            : std::format("epoch{}", secs);
        const std::string stem = p.stem().string();
        const fs::path dst = dir / std::format("{}-{}{}", stem, stamp, p.extension().string());

        fs::rename(p, dst, ec);
        if (ec)   // different filesystem (or a racing reader): fall back to a copy, keep the evidence
        {
            ec.clear();
            fs::copy_file(p, dst, fs::copy_options::overwrite_existing, ec);
        }
        if (ec)
            std::println("[diag] ⚠ could not archive '{}' ({}) — it will be TRUNCATED", path, ec.message());

        std::vector<fs::path> archived;
        const std::string prefix = stem + "-";
        for (const auto& entry : fs::directory_iterator(dir, ec))
            if (entry.is_regular_file(ec) and entry.path().filename().string().starts_with(prefix))
                archived.push_back(entry.path());
        if (archived.size() > KEEP_RUNS)
        {
            std::ranges::sort(archived);   // "<stem>-<stamp>" sorts chronologically
            for (std::size_t i = 0; i + KEEP_RUNS < archived.size(); ++i)
                fs::remove(archived[i], ec);
        }
    }

    f.open(path, std::ios::out | std::ios::trunc);
    // imbue the classic locale so a float can never be written with a COMMA separator if the C++ global
    // locale is ever changed — the write side of the same asymmetry that corrupted the depth dataset
    // (CLAUDE.md; phantom_log.h does this for the same reason).
    if (f.is_open())
        f.imbue(std::locale::classic());
}
}  // namespace

void SpecificWorker::log_compute_perf(double busy_ms)
{
    const auto now = std::chrono::steady_clock::now();
    if (perf_window_begin_.time_since_epoch().count() == 0)
        perf_window_begin_ = now;
    if (perf_last_call_.time_since_epoch().count() != 0)
        perf_worst_ms_ = std::max(perf_worst_ms_,
                                  std::chrono::duration<double, std::milli>(now - perf_last_call_).count());
    perf_last_call_ = now;
    ++perf_calls_;
    perf_busy_ms_ += busy_ms;

    const double elapsed_ms = std::chrono::duration<double, std::milli>(now - perf_window_begin_).count();
    if (elapsed_ms < 1000.0)
        return;

    const double period_ms = elapsed_ms / std::max(1u, perf_calls_);
    const double cpu_pct   = std::max(0.0f, fps.get_cpu_use());

    // ── LEAK WATCH ────────────────────────────────────────────────────────────────────────────────
    // RSS every window, next to the timings, on DISK. Read straight from /proc/self/status rather than
    // FPSCounter::get_mem_use() (which divides kB by 1000 and returns an int, so a 3 MB/min growth is
    // invisible in it) and reported as a RATE, because the absolute number says nothing: what matters is
    // whether it is monotonic. VmHWM comes along because peak==current is precisely what distinguishes a
    // steady leak from an allocation spike glibc never returned to the OS.
    //
    // This pairs with grid_diag.csv via `grid_cycle`: the 2026-08-08 OOM run's telemetry stopped dead at
    // grid cycle 4080 (25 min in) and then grew for 68 more minutes with no rows at all, so the one thing
    // the logs could not answer was whether the loop stalled BEFORE or AFTER the memory ran away. A row
    // per second carrying both the cycle number and RSS answers that on the next occurrence.
    const long rss_kb = read_proc_status_kb("VmRSS");
    const long hwm_kb = read_proc_status_kb("VmHWM");
    if (perf_rss0_kb_ < 0 and rss_kb > 0)
    {
        perf_rss0_kb_ = rss_kb;
        perf_t0_      = now;
    }
    const double since_start_s = perf_t0_.time_since_epoch().count() == 0
        ? 0.0 : std::chrono::duration<double>(now - perf_t0_).count();
    const long   grown_kb      = (rss_kb > 0 and perf_rss0_kb_ > 0) ? rss_kb - perf_rss0_kb_ : 0;
    const double rate_avg_kb_min = since_start_s > 1.0 ? grown_kb / (since_start_s / 60.0) : 0.0;

    // WINDOWED rate: RSS change across the trailing PERF_RATE_WINDOW_S seconds — the number to actually
    // watch (see the header for why the lifetime average cannot detect onset). 0 until the window has
    // spanned PERF_RATE_MIN_SPAN_S, because a rate over a two-second span is page-granularity noise.
    double rate_win_kb_min = 0.0;
    if (rss_kb > 0)
    {
        perf_rss_hist_.emplace_back(since_start_s, rss_kb);
        while (perf_rss_hist_.size() > 1
               and since_start_s - perf_rss_hist_.front().first > PERF_RATE_WINDOW_S)
            perf_rss_hist_.pop_front();
        const auto [t_old, rss_old] = perf_rss_hist_.front();
        if (const double span = since_start_s - t_old; span >= PERF_RATE_MIN_SPAN_S)
            rate_win_kb_min = static_cast<double>(rss_kb - rss_old) / (span / 60.0);
    }

    std::println("[perf] fps={:.1f} period mean {:.1f} ms WORST {:.1f} ms busy {:.1f} ms cpu={:.0f}% "
                 "rss={:.1f} MB ({:+.1f} since start) growth {:+.2f} MB/min [{:.0f}s win] avg {:+.2f} "
                 "hwm={:.1f} MB",
                 1000.0 / std::max(1e-3, period_ms), period_ms, perf_worst_ms_,
                 perf_busy_ms_ / std::max(1u, perf_calls_), cpu_pct,
                 rss_kb / 1024.0, grown_kb / 1024.0, rate_win_kb_min / 1024.0, PERF_RATE_WINDOW_S,
                 rate_avg_kb_min / 1024.0, hwm_kb / 1024.0);

    // One-shot open: on failure do NOT retry, or every window would archive the file again and churn the
    // whole archive directory through the KEEP_RUNS prune once per second.
    static bool perf_csv_tried = false;
    if (not perf_csv_tried)
    {
        perf_csv_tried = true;
        open_diag_csv(perf_csv_, "etc/perf_diag.csv");
        if (perf_csv_.is_open())
            perf_csv_ << "t_s,iso_time,grid_cycle,calls,fps,period_mean_ms,period_worst_ms,busy_mean_ms,"
                         "cpu_pct,rss_kb,hwm_kb,rss_delta_kb,rate_kb_per_min_win,rate_kb_per_min_avg\n";
        else
            std::println("[perf] ⚠ could not open etc/perf_diag.csv — leak watch is stdout-only this run");
    }
    if (perf_csv_.is_open())
    {
        const auto ms = rc::provenance::now_ms();
        perf_csv_ << std::format("{:.1f},{},{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.0f},{},{},{},{:.1f},{:.1f}\n",
                                 since_start_s, rc::provenance::iso8601_local(ms), grid_diag_cycle_,
                                 perf_calls_, 1000.0 / std::max(1e-3, period_ms), period_ms,
                                 perf_worst_ms_, perf_busy_ms_ / std::max(1u, perf_calls_), cpu_pct,
                                 rss_kb, hwm_kb, grown_kb, rate_win_kb_min, rate_avg_kb_min);
        perf_csv_.flush();   // one small row per second: flush so a SIGKILLed run still has its last rows
    }

    perf_window_begin_ = now;
    perf_calls_ = 0;
    perf_busy_ms_ = 0.0;
    perf_worst_ms_ = 0.0;
}

void SpecificWorker::compute()
{
    // Rate/cost meter: RAII so EVERY exit path is timed — most cycles leave through one of the early returns
    // below (no graph, no room, no fresh sweep), and a meter placed at the end would only ever see the
    // expensive cycles and would report a period several times too long.
    const auto perf_t0 = std::chrono::steady_clock::now();
    struct PerfScope
    {
        SpecificWorker* self; const std::chrono::steady_clock::time_point& t0;
        ~PerfScope() { self->log_compute_perf(std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - t0).count()); }
    } perf_scope{this, perf_t0};

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
            // Forgetting + self-body: give evidence a finite lifetime and put the robot's own body in the sensor
            // model. Without these the grid accumulates monotonically (occluded cells immortal, z-band only ever
            // widens, self-returns latched permanently along the driven path).
            gp.forget_half_life_s  = cfg_.grid_forget_half_life_s;
            gp.forget_occupied_only = cfg_.grid_forget_occupied_only;
            gp.forget_visible_only  = cfg_.grid_forget_visible_only;
            gp.forget_range_weighted = cfg_.grid_forget_range_weighted;
            gp.forget_can_unlatch   = cfg_.grid_forget_can_unlatch;
            gp.bin_span_m           = cfg_.grid_bin_span_m;
            gp.clear_stop_max_m     = cfg_.grid_clear_stop_max_m;
            gp.collision_band_top_m = cfg_.grid_collision_band_top_m;
            gp.lidar_clearance_m    = cfg_.grid_lidar_clearance_m;
            gp.self_body_sigma_m   = cfg_.grid_self_body_sigma_m;
            // NO C-SPACE INFLATION. The controller now collides its ACTUAL footprint polygon against the grid
            // (common/robot_footprint + controller/src/grid_planner), so inflating here is double-counted: a
            // 0.25 m dilation plus the robot's own 0.28 m inscribed radius demands ~0.53 m of clearance from
            // every occupied cell, which merges sparse residual into blobs that fill the room and makes every
            // target report "not footprint-feasible". Publish the TRUE occupied extent and let the one consumer
            // that knows the robot's shape apply it. Set >0 only if a consumer without a footprint model needs
            // pre-inflated hulls again.
            gp.inflate_radius_m    = cfg_.grid_inflate_radius_m;
            // Tightened occupied condition: the floor competes for every return in a mixture instead of being a
            // hard step, and a floor return finally carries its free-space meaning for its own cell.
            gp.floor_responsibility = cfg_.grid_floor_responsibility;
            gp.floor_return_clears  = cfg_.grid_floor_return_clears;
            gp.floor_sigma_min_m    = cfg_.grid_floor_sigma_min_m;
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
            // BPEARL ONLY — see floor_datum_sweep(). helios cannot be a floor datum, only a reference.
            floor_plane_ = rc::estimate_floor_plane(floor_datum_sweep(), lidar_ingestor_->origin_room(),
                                                    cfg_.cluster.floor_z0, cfg_.cluster.floor_slope,
                                                    cfg_.floor_plane, floor_plane_);
            // The fit's own residual RMS goes with it: it is the σ of the floor component in the per-return
            // obstacle-vs-floor mixture, so the model's tolerance for a near-floor return is set by MEASURED floor
            // quality (≈7 cm here — larger than floor_z0) instead of by a constant band.
            if (cfg_.floor_plane.enabled and floor_plane_.valid)
                grid_.set_floor_plane(floor_plane_.a, floor_plane_.b, floor_plane_.c, floor_plane_.rms);
            else
                grid_.set_floor_plane(0.0f, 0.0f, 0.0f, floor_plane_.valid ? floor_plane_.rms : 0.0f);
            static int fpc = 0;
            if ((fpc++ % 40) == 0 and floor_plane_.valid)
                std::println("[floor-plane] z = {:.3f}·x + {:.3f}·y + {:.3f}  (offset {:.1f} cm, tilt {:.1f}°) {}",
                             floor_plane_.a, floor_plane_.b, floor_plane_.c, floor_plane_.c * 100.0f,
                             std::atan(std::hypot(floor_plane_.a, floor_plane_.b)) * 57.2958f,
                             cfg_.floor_plane.enabled ? "[APPLIED]" : "[log-only, flag off]");
            // SELF-BODY envelope for this cycle: returns off our own body must not write occupancy into a LATCHED
            // map (they would re-emerge as residual the moment the robot drives away — a trail of phantoms).
            // Placed at the sensor origin's xy, which is the robot axis, exactly as the read-out robot explainer
            // does. Radius matches the lidar driver's [Footprint] disc so both agree on ONE body model.
            grid_.set_self_body(lidar_ingestor_->origin_room().x(), lidar_ingestor_->origin_room().y(),
                                cfg_.grid_self_body_radius_m);
            // PER-DEVICE integration: each LiDAR carries its own floor sigma, taken from its own floor fit.
            // See integrate_lidar_per_device(). Falls back to one merged sweep when the per-device tag is
            // unavailable — in which case the single datum sigma applies to everything,
            // which is the behaviour that let the floor latch, so the fallback warns.
            if (not integrate_lidar_per_device())
            {
                static int wc = 0;
                if ((wc++ % 200) == 0)
                    std::println("[floor-sigma] no per-device tag — ONE sigma for all sensors; helios's grazing "
                                 "floor returns may latch as obstacles");
                const auto& lidar_sweep = filtered_lidar_sweep();   // bpearl floor grazing removed (per-device band)
                grid_.integrate_sweep(lidar_ingestor_->origin_room(), lidar_sweep,
                                      /*begin_cycle=*/true, ego_reliability_);        // accumulate LiDAR
            }
            integrate_zed_into_grid();   // accumulate dense ZED FoV as a second sensor (fills grazed tabletops)
            // Elapsed time since the previous commit drives the FORGETTING term (unobserved cells relax toward
            // the prior). Measured, not assumed, so the half-life is in seconds regardless of the cycle rate.
            const auto now_commit = std::chrono::steady_clock::now();
            const float commit_dt_s = last_commit_.time_since_epoch().count() == 0
                ? 0.0f
                : std::chrono::duration<float>(now_commit - last_commit_).count();
            last_commit_ = now_commit;
            grid_.commit_cycle(commit_dt_s);   // ONE log-odds update per cell (hit precedence) + forgetting
            log_grid_diag();
            // log_floor_diag() now runs in the READ-OUT block below — it needs the explainer predicate to report
            // the RESIDUAL set (the cells this agent actually publishes) rather than every latched cell.
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
        // All concepts are generic `object` nodes now; split the per-class counts by object_subtype
        // (tables report "round"/"square"), falling back to the name prefix.
        int n_tab = 0, n_chr = 0, n_cyl = 0;
        for (const auto& n : G->get_nodes_by_type("object"))
        {
            const auto st = G->get_attrib_by_name<object_subtype_att>(n);
            const std::string sub = st.has_value() ? st.value() : std::string{};
            if      (n.name().starts_with("table")  or sub == "table" or sub == "round" or sub == "square") ++n_tab;
            else if (n.name().starts_with("chair")  or sub == "chair")  ++n_chr;
            else if (n.name().starts_with("bottle") or sub == "bottle") ++n_cyl;
        }
        std::string s;
        for (const auto& o : soft_objects_) s += std::format("[σxy={:.3f} σz={:.3f} ztop={:.2f} hx={:.2f} hy={:.2f}] ", o.sigma_xy, o.sigma_z, o.z_top, o.hx, o.hy);
        std::println("[collapse] nodes table={} chair={} bottle={} → {} objects; {}",
                     n_tab, n_chr, n_cyl, soft_objects_.size(), s.empty() ? "(none)" : s);
    }

    // ── PHASE-1 REBUILD: read the RESIDUAL out of the grid (occupied ∧ ¬explained) and publish it. A cell is
    //    "explained" if any known-model explainer accounts for its representative point → walls + specialists
    //    drop out (floor/ceiling already excluded by the nav band), leaving object-only components. Evidence is
    //    never deleted, only masked at read-out — a specialist mis-fit cannot erase a real obstacle. Also
    //    publishes the residual cells on a `grid` node under room for the retina 3-D display. ──
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
                // NOTE: the floor competes for evidence ONCE, at integration time (the per-return mixture
                // responsibility in OccGridParams::floor_responsibility), and the log-odds then accumulate over
                // time. Do NOT also apply that responsibility here. A cell that crossed occ_set in spite of the
                // discount has earned it with repeated evidence; re-applying the discount at read-out is a
                // second, evidence-BLIND height test, and it would permanently mask a real 20 cm obstacle no
                // matter how many frames confirmed it. The read-out only subtracts what a model geometrically
                // OWNS (floor below the band, ceiling, robot, walls, object footprints).
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
        // ★NO C-SPACE INFLATION (2026-08-20). inflate_radius_m (0.25 m, half a robot width) existed to
        // pre-clear the published component HULLS for the controller's planner. The controller now takes
        // the occupancy grid directly and applies the robot's true footprint itself, so inflating here
        // is a second copy of the same allowance: measured, one 5 cm residual cell became a no-go zone
        // close to a metre across, and nothing downstream could tell the two inflations apart.
        // Components are still computed — the logs, the height profile and the specialist subtraction
        // all read them — just at their true extent.
        const auto comps = grid_.occupied_components(2, cell_explained, 0.f);
        log_floor_diag(cell_explained, comps);   // plane fit + the RESIDUAL set's height profile, jointly
        log_releases();                          // ...and WHY each cell removed this cycle was removed
        dump_residual_cells(cell_explained);     // ...and WHERE they are, which a histogram cannot say
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
        // Shadow-mode death record (§4.2). residual retires on DIVERGENCE, not on sensor absence, so
        // p_detect does not apply — recorded as unattributable so it is never read as a phantom.
        if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
            log_phantom_event("DEATH", id, it->second.node_name,
                              it->second.model.state().cx, it->second.model.state().cy, &it->second,
                              "retire-diverged (no existence channel)");
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
            // Shadow-mode birth record (CONCEPT_AGENT_LIFECYCLE.md §4.2): place + viewpoint that
            // produced it, so a phantom that dies young is attributable to both.
            log_phantom_event("BIRTH", new_id, "", cl.centroid.x(), cl.centroid.y(), nullptr, "");
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
    auto infra = cfg_.zed_infra;                                  // reference the ZED floor band to the FITTED floor
    if (cfg_.floor_plane.enabled and floor_plane_.valid)
    { infra.floor_a = floor_plane_.a; infra.floor_b = floor_plane_.b; infra.floor_c = floor_plane_.c; }
    // MARK vs CLEAR. Removing the infrastructure points outright also removed their RAYS, and a ZED floor or wall
    // return is the longest ray the camera casts — precisely the one carrying the most free-space evidence. With
    // ZedInfraClears the classification becomes a mark_mask instead: the point is still raytraced, only its
    // endpoint is forbidden to assert occupancy. Same classifier, same marking, strictly more clearing.
    std::vector<std::uint8_t> mark_mask;
    const std::vector<std::uint8_t>* mask_ptr = nullptr;
    if (cfg_.grid_zed_infra_clears)
    {
        mark_mask = rc::ResidualClusterer::infrastructure_mask(pts, cam_origin, read_room_polygon(), infra);
        mask_ptr = &mark_mask;
    }
    else
        pts = rc::ResidualClusterer::subtract_infrastructure(pts, cam_origin, read_room_polygon(), infra);
    static int zc = 0;
    if ((zc++ % 40) == 0)
    {
        const std::size_t marking = mask_ptr
            ? static_cast<std::size_t>(std::ranges::count(mark_mask, std::uint8_t{1})) : pts.size();
        std::println("[zed-grid] fov={} → {} may mark, {} raytraced ({})", before, marking, pts.size(),
                     mask_ptr ? "infra CLEARS (mask)" : "infra removed (old: rays lost)");
    }
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
    grid_.integrate_sweep(cam_origin, pts, /*begin_cycle=*/false, ego_reliability_, scale_ptr, mask_ptr);
}

bool SpecificWorker::refresh_semantic_map()
{
    // Read the retina's `semantic` node (dense YOLO-sem label map under `zed`). The blob is large and published
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

const std::vector<Eigen::Vector3f>& SpecificWorker::device_sweep(std::uint8_t plane, bool keep_floor)
{
    // One device's returns, with bpearl's own floor-grazing cut re-applied inline. Deliberately NOT built by
    // indexing filtered_lidar_sweep(): that function drops points, so plane_id (indexed on the RAW sweep) no
    // longer aligns with its output, and any index correspondence would break silently the next time that
    // filter changes. Re-deriving the one cut it applies is cheaper and cannot desynchronise.
    const auto& pid = lidar_ingestor_->plane_id();
    device_pts_.clear();
    if (pid.size() != lidar_ingestor_->sweep_room().size()) return device_pts_;   // no tag → caller falls back
    const auto& raw = lidar_ingestor_->sweep_room();
    const Eigen::Vector3f o = lidar_ingestor_->origin_room();
    const float bz0 = cfg_.cluster.bpearl_floor_z0, slope = cfg_.cluster.floor_slope;
    const float hz0 = cfg_.cluster_helios_floor_z0;
    // Referenced to the FITTED floor plane, so the cut follows an offset/tilted floor instead of assuming z=0.
    const bool use_fit = cfg_.floor_plane.enabled and floor_plane_.valid;
    device_pts_.reserve(raw.size() / 2);
    long dropped = 0;
    for (std::size_t i = 0; i < raw.size(); ++i)
    {
        if (pid[i] != plane) continue;
        // PER-DEVICE floor band. bpearl keeps its existing cut; helios gets its own, larger one, because it
        // only ever grazes the floor and reads it 13-17 cm high. See Clusterer.HeliosFloorZ0 for why the two
        // model-level alternatives (per-device sigma, per-device reference plane) cannot handle a
        // range-dependent BIAS. Without this, helios's own floor latches as obstacle across the whole room.
        const float z0 = (pid[i] == 1) ? bz0 : hz0;
        const float range = std::hypot(raw[i].x() - o.x(), raw[i].y() - o.y());
        const float floor_z = use_fit ? floor_plane_.z_at(raw[i].x(), raw[i].y()) : 0.0f;
        // keep_floor: hand the near-floor returns on instead of deleting them. They are not obstacle evidence —
        // the grid's own band (set_device_floor_z0, the SAME z0 used here) still refuses to mark them — but they
        // ARE clearing evidence, and the longest rays in the sweep at that. Deleting a return does not only lose
        // its mark, it loses the whole ray and every free cell along it.
        if (raw[i].z() < floor_z + z0 + slope * range) { ++dropped; if (not keep_floor) continue; }
        device_pts_.push_back(raw[i]);
    }
    static int dsc = 0;
    if (plane == 0 and (dsc++ % 40) == 0)
        std::println("[helios-floor] {} {} near-floor pts (band z0={:.2f}); {} handed to the grid",
                     keep_floor ? "kept" : "dropped", dropped, hz0, device_pts_.size());
    last_device_floor_dropped_ = dropped;
    return device_pts_;
}

bool SpecificWorker::integrate_lidar_per_device()
{
    // ── PER-DEVICE FLOOR SIGMA ──
    // The floor component's sigma was a single number (the datum fit's RMS) applied to every return. That cannot
    // be right for two sensors that see the floor completely differently: bpearl is a downward dome measuring it
    // HEAD-ON (~1 cm scatter), helios is an upright lidar at ~1.1 m that only ever GRAZES it, landing 13-17 cm
    // high. Feeding helios's returns a 1 cm sigma tells the mixture that a 15 cm-high helios floor return is a
    // 15-sigma outlier — i.e. certainly an obstacle — and the whole floor latches. That is exactly what happened
    // the moment the datum was corrected: the previously CONTAMINATED datum sat 5.9 cm high and had been
    // accidentally holding the nav band above helios's bias. Fixing the datum removed the accident, so the real
    // defect underneath — one sigma for two very different sensors — finally showed.
    //
    // So integrate device by device, and take each device's sigma from ITS OWN floor fit. No tuned constant:
    // bpearl's sigma is the bpearl fit's RMS, helios's is the helios fit's RMS, both measured every cycle and
    // both self-calibrating if the scenario changes. The PLANE — the datum — always stays bpearl's; helios's fit
    // contributes its scatter and never moves the floor.
    //
    // The grid accumulates per-cell flags across sensors within one cycle (begin_cycle=false on later sweeps),
    // which is the same mechanism the ZED pass already uses, so this needs no new machinery.
    const auto& pid = lidar_ingestor_->plane_id();
    if (pid.size() != lidar_ingestor_->sweep_room().size() or pid.empty())
        return false;                                            // fused plane, no per-device tag → caller falls back

    // ── MARKING FILTER vs CLEARING FILTER (Stage 1, 2026-08-19) ──
    // With grid_floor_band_in_grid the near-floor returns are no longer deleted here; the grid is told each
    // device's band instead (set_device_floor_z0) and applies it inside the sensor model. Marking is unchanged by
    // construction — the grid tests the SAME `z < floor_z + z0 + slope·r` that device_sweep tested — but the
    // returns now also carry their rays, and a below-band return clears its own cell via
    // mark_floor_endpoint_flag. That path had never executed once in the live pipeline (floor_clears == 0 on
    // 9381/9381 cycles) because its input was being filtered away upstream.
    const bool keep = cfg_.grid_floor_band_in_grid;

    const auto& bp = device_sweep(1, keep);
    std::vector<Eigen::Vector3f> bpearl_pts(bp.begin(), bp.end());   // device_sweep reuses one buffer

    // The helios floor fit keeps its ORIGINAL (filtered) input, deliberately. Its rms feeds sig_he, which scales
    // every helios hit through floor_obstacle_responsibility — so changing what it sees would change marking, and
    // this stage is meant to be marking-neutral. Fit first, then re-derive the cloud we actually integrate.
    const auto& he_fit = device_sweep(0, /*keep_floor=*/false);
    if (bpearl_pts.empty() and he_fit.empty()) return false;
    floor_plane_helios_ = rc::estimate_floor_plane(he_fit, lidar_ingestor_->origin_room(),
                                                   cfg_.cluster.floor_z0, cfg_.cluster.floor_slope,
                                                   cfg_.floor_plane, floor_plane_helios_);
    const float a = (cfg_.floor_plane.enabled and floor_plane_.valid) ? floor_plane_.a : 0.0f;
    const float b = (cfg_.floor_plane.enabled and floor_plane_.valid) ? floor_plane_.b : 0.0f;
    const float c = (cfg_.floor_plane.enabled and floor_plane_.valid) ? floor_plane_.c : 0.0f;
    // ★★★AN UNFITTED FLOOR IS AN UNKNOWN FLOOR, NOT A PERFECT ONE.
    // When no fit converges these both used to fall back to sigma = 0, which tells the marking model that the
    // floor's height is known EXACTLY — the opposite of the truth. Measured live 2026-08-23 on this robot:
    // `bpearl 0 / 11948 pts feed the floor fit ... rms 0.000, valid=false` — bpearl contributes NO points, so
    // there is no floor datum at all, the plane stays at z=0 and sigma at 0. A helios grazing return landing
    // 25 cm high at 4 m then scored 0.43 obstacle-responsibility and latched in a few frames: floor phantoms
    // born at range, refuted only once the robot drove close enough to see the floor properly. That is exactly
    // the "noise on the floor that disappears when the robot comes close" — and it must not be CREATED.
    // With no datum the honest sigma is the scale of the disagreement we cannot rule out: how far the floor
    // could plausibly sit from the assumed z=0. FloorPlane.MaxOffsetM is precisely that bound, already declared.
    const float sig_unknown = cfg_.floor_plane.max_offset_m;
    const float sig_bp = floor_plane_.valid ? floor_plane_.rms : sig_unknown;
    // If the helios fit has not converged either, fall back to the datum's sigma rather than to zero — zero
    // would reinstate exactly the over-confidence this function exists to remove.
    const float sig_he = floor_plane_helios_.valid ? std::max(floor_plane_helios_.rms, sig_bp) : sig_bp;

    grid_.set_floor_plane(a, b, c, sig_bp);
    grid_.set_device_floor_z0(keep ? cfg_.cluster.bpearl_floor_z0 : -1.0f);
    grid_.set_sensor_min_range(cfg_.bpearl_min_range_m);   // its dead shell: no free evidence inside it
    grid_.set_sensor_noise(0.0f, 0.0f);                    // lidar-grade: full clearing authority
    grid_.set_sensor_id(1);
    grid_.integrate_sweep(lidar_ingestor_->origin_room(), bpearl_pts, /*begin_cycle=*/true, ego_reliability_);

    const auto& he = device_sweep(0, keep);          // re-derive: unfiltered when the flag is on
    grid_.set_floor_plane(a, b, c, sig_he);
    grid_.set_device_floor_z0(keep ? cfg_.cluster_helios_floor_z0 : -1.0f);
    grid_.set_sensor_min_range(cfg_.helios_min_range_m);
    grid_.set_sensor_noise(0.0f, 0.0f);
    grid_.set_sensor_id(2);
    grid_.integrate_sweep(lidar_ingestor_->origin_room(), he, /*begin_cycle=*/false, ego_reliability_);
    grid_.set_floor_plane(a, b, c, sig_bp);   // leave the datum's sigma in place for the ZED pass that follows
    grid_.set_device_floor_z0(-1.0f);         // ...and the DEFAULT band: ZED is not one of these two devices
    grid_.set_sensor_min_range(cfg_.zed_min_range_m);
    // The ZED declares its OWN depth noise, sigma(r) = Sigma0M + SigmaQuad·r². It raytraces ~8000 rays a cycle
    // while only ~250 may mark, so it is almost purely a clearing sensor — and measured 2026-08-23 it was doing
    // 92% of all removals, its hotspots landing exactly on the two tables. Its clearing now carries the
    // precision it actually has: full weight up close, a few percent at 4 m, where its endpoint is uncertain by
    // 13 cm and an overestimated depth would send the ray through the tabletop it really hit.
    grid_.set_sensor_noise(cfg_.zed_infra.sigma0_m, cfg_.zed_infra.sigma_quad);
    grid_.set_sensor_id(3);

    static int dc = 0;
    if ((dc++ % 40) == 0)
        std::println("[floor-sigma] bpearl {} pts σ={:.3f} m | helios {} pts σ={:.3f} m (own fit, rms {:.3f}, "
                     "valid={}) | band-in-grid={}", bpearl_pts.size(), sig_bp, he.size(), sig_he,
                     floor_plane_helios_.rms, floor_plane_helios_.valid, keep);
    return true;
}

const std::vector<Eigen::Vector3f>& SpecificWorker::floor_datum_sweep()
{
    // BPEARL ONLY. The floor fit used to take the MERGED sweep, which quietly made helios a co-author of the
    // floor datum — and helios physically cannot be one. It is an upright 360 lidar at ~1.1 m, so every floor
    // return it produces is GRAZING and at range, reading several cm high by construction. ROBOT_GEOMETRY.md
    // says so ("use for walls, not the floor datum", bias ~+17 cm) and room_concept's own startup check says so
    // ("reference only, NOT the datum"); this fit was the one place that ignored both.
    //
    // The cost was not a constant offset — it was an unstable one. Because the bias grows with range, it moves
    // with the scene: helios's floor reading was measured at 170 mm in one spot and 130 mm in another, purely
    // from where the robot happened to stand. Fitted through a plane that also has a tilt term, that range
    // correlation shows up as TILT. Measured on the same floor at the same moment:
    //     merged (helios+bpearl):  +5.9 cm, tilt 1.0°   — and 2.88 / 11.2 / 5.9 cm across three runs
    //     bpearl only:             −1.0 cm              — stable to the bin across three runs
    // That datum is handed to set_floor_plane() and therefore sets the obstacle band, so a wandering 6 cm
    // offset with a 1° tilt raises the latch threshold across the room AND varies it with position: genuinely
    // low obstacles stop latching, in a way that changes as the robot drives.
    //
    // bpearl is the head-on downward dome — dense, near, and the sensor the geometry check already trusts.
    // Fall back to the merged sweep when the per-device tag is absent, because an empty
    // point set would silently HOLD the previous estimate and look like stability rather than a missing input.
    const auto& pts = lidar_ingestor_->sweep_room();
    const auto& pid = lidar_ingestor_->plane_id();
    if (pid.size() != pts.size())
    {
        static int warn = 0;
        if ((warn++ % 200) == 0)
            std::println("[floor-datum] no per-device tag ({} pts, {} ids) — falling back to the MERGED sweep; "
                         "the floor datum will carry helios's grazing bias", pts.size(), pid.size());
        return pts;
    }
    floor_datum_pts_.clear();
    floor_datum_pts_.reserve(pts.size() / 2);
    for (std::size_t i = 0; i < pts.size(); ++i)
        if (pid[i] == 1) floor_datum_pts_.push_back(pts[i]);   // 1 = bpearl
    static int fc = 0;
    if ((fc++ % 40) == 0)
        std::println("[floor-datum] bpearl {} / {} pts feed the floor fit (helios excluded: grazing, not a datum)",
                     floor_datum_pts_.size(), pts.size());
    // If bpearl produced nothing this cycle, prefer the merged sweep over an empty fit for the same reason.
    return floor_datum_pts_.empty() ? pts : floor_datum_pts_;
}

const std::vector<Eigen::Vector3f>& SpecificWorker::filtered_lidar_sweep()
{
    // Drop the LOW bpearl lidar's FLOOR-GRAZING returns (device-specific higher floor band) while keeping helios
    // and bpearl's real (>band) low-obstacle returns — so bpearl still catches short obstacles helios misses,
    // without ringing the robot with phantom floor cells. plane_id: helios=0, bpearl=1 (empty ⇒ keep all).
    const auto& pts = lidar_ingestor_->sweep_room();
    const auto& pid = lidar_ingestor_->plane_id();
    if (pid.size() != pts.size()) return pts;                    // no per-device tag → pass through
    const Eigen::Vector3f o = lidar_ingestor_->origin_room();
    const float bz0 = cfg_.cluster.bpearl_floor_z0, slope = cfg_.cluster.floor_slope;
    // Referenced to the FITTED floor plane, like the grid's own band. bpearl is the LOW dome sensor that grazes the
    // floor around the robot, so its band is the one most sensitive to the datum: with the apartment floor sitting
    // up to 8 cm above z=0, a z=0-referenced 12 cm band is only ~4 cm of real clearance and leaks a phantom ring.
    const bool fp_on = cfg_.floor_plane.enabled and floor_plane_.valid;
    const auto floor_z = [&](const Eigen::Vector3f& q)
    { return fp_on ? floor_plane_.a * q.x() + floor_plane_.b * q.y() + floor_plane_.c : 0.0f; };
    lidar_filtered_.clear(); lidar_filtered_.reserve(pts.size());
    long dropped = 0;
    for (std::size_t i = 0; i < pts.size(); ++i)
    {
        if (pid[i] == 1)                                         // bpearl → apply its higher floor band
        {
            const float range = std::hypot(pts[i].x() - o.x(), pts[i].y() - o.y());
            if (pts[i].z() < floor_z(pts[i]) + bz0 + slope * range) { ++dropped; continue; }
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

void SpecificWorker::dump_residual_cells(const rc::OccupancyGrid::CellExplained& explained)
{
    // WHERE the residual cells are, not just how tall they are. The height histogram says 13% of the published
    // mass sits above 1.20 m, which "must be walls" in an apartment — but a histogram cannot tell a wall from a
    // wardrobe, and acting on that guess is how the floor_clears defect survived ten days. So dump the geometry
    // and let it answer: dist_wall_m small ⇒ the room polygon is failing to claim its own wall; obj_sdf_m small
    // ⇒ a specialist under-claims its footprint; both large ⇒ a genuinely free-standing object nobody models.
    if (cfg_.grid_cell_dump_every_n <= 0 or not grid_ready_) return;
    if ((grid_diag_cycle_ % cfg_.grid_cell_dump_every_n) != 0) return;

    static std::ofstream f;
    if (not f.is_open())
    {
        open_diag_csv(f, "etc/residual_cells.csv");
        if (not f.is_open()) return;
        f << "cycle,x,y,z,inside_poly,dist_wall_m,obj_sdf_m\n";
    }
    const auto xyz = grid_.residual_cell_centres_xyz(explained);
    const auto poly = read_room_polygon();
    const bool have_poly = poly.size() >= 3;
    const float fitm = cfg_.cluster.explain_fit_margin_m;
    for (std::size_t i = 0; i + 2 < xyz.size(); i += 3)
    {
        const float x = xyz[i], y = xyz[i + 1], z = xyz[i + 2];
        const Eigen::Vector2f p2(x, y);
        const int inside = have_poly ? (rc::ResidualClusterer::point_in_polygon(poly, p2) ? 1 : 0) : -1;
        const float dw = have_poly ? rc::ResidualClusterer::dist_to_polygon_boundary(poly, p2) : -1.0f;
        // Signed 2-D distance to the NEAREST modelled object footprint (negative inside), same construction the
        // read-out explainer uses — so a small positive value means "just outside a specialist's claim".
        float best = 1e9f;
        for (const auto& o : soft_objects_)
        {
            const float c = std::cos(o.yaw), s = std::sin(o.yaw), dx = x - o.cx, dy = y - o.cy;
            const float lx = c * dx + s * dy, ly = -s * dx + c * dy;
            const float qx = std::abs(lx) - o.hx, qy = std::abs(ly) - o.hy;
            best = std::min(best, std::hypot(std::max(qx, 0.0f), std::max(qy, 0.0f))
                                  + std::min(std::max(qx, qy), 0.0f) - fitm);
        }
        f << grid_diag_cycle_ << ',' << x << ',' << y << ',' << z << ',' << inside << ',' << dw << ','
          << (soft_objects_.empty() ? -1.0f : best) << '\n';   // -1 ⇒ NO object nodes exist to claim anything
    }
    f.flush();
    std::println("[cell-dump] cycle {}: {} residual cells written (poly {} verts, {} objects)",
                 grid_diag_cycle_, xyz.size() / 3, poly.size(), soft_objects_.size());
}

void SpecificWorker::log_grid_diag()
{
    if (not grid_ready_) return;
    const auto& d = grid_.last_sweep_diag();
    long& cyc = grid_diag_cycle_;   // member, not a static: log_compute_perf logs it as the join key
    static std::ofstream f;
    if (not f.is_open())
    {
        open_diag_csv(f, "etc/grid_diag.csv");
        // forgotten = cells released by the TIME-DECAY rather than by a see-through: space that no clearing beam
        // could ever have reached (occluded / behind us). self_damped = returns whose hit weight the self-body
        // sensor term attenuated. Both are 0 in the old never-forget build, so a run with zeros here means the
        // new terms are not engaging — check Grid.ForgetHalfLifeS / Grid.SelfBodyRadiusM before believing a null
        // result. The headline health metric stays miss_blocked_zaware vs misses: it ratcheted 85%→96% of
        // see-throughs discarded on the old build, and should now stay well under `misses`.
        // floor_damped = in-band returns whose weight the floor RESPONSIBILITY cut (the near-floor returns that
        // used to latch a cell outright); floor_clears = below-band returns that freed THEIR OWN cell, the
        // evidence the old inverse sensor model discarded entirely. Zeros in either column mean the tightened
        // occupied condition is not engaging — check Grid.FloorResponsibility / Grid.FloorReturnClears and the
        // fit's rms in floor_diag.csv before believing a null result.
        // ── Stage 1 columns (2026-08-19). floor_clears sat at 0 for 9381 consecutive cycles and the log could not
        // say why, because one counter meant both "the gate refused it" and "nothing was ever offered". Now:
        //   floor_rets     = below-band returns the grid was OFFERED. 0 ⇒ they are being deleted UPSTREAM
        //                    (device_sweep) — check Grid.FloorBandInGrid. This is the defect that hid for weeks.
        //   floor_blocked  = offered but refused by the SUPPORT gate (cell's lowest evidence is floating, so the
        //                    beam passed underneath and refutes nothing). Healthy under a tabletop.
        //   floor_clears   = offered, accepted, cell freed.  floor_rets = floor_clears + floor_blocked.
        //   marks_suppr    = returns raytraced with marking suppressed (ZED infrastructure). 0 while
        //                    ZedBoost.FeedGrid and Grid.ZedInfraClears are both on ⇒ the mask is not plumbed.
        //   floor_dropped  = near-floor returns device_sweep saw last cycle; with FloorBandInGrid on they are
        //                    handed to the grid rather than deleted, so this should track floor_rets.
        // miss_blocked_zaware now counts ONLY the traverse gate (a beam passing at another height), never the
        // support gate — the two were summed together and neither could be read.
        f << "cycle,occupied,hits,misses,miss_blocked_zaware,latched,released,hit_then_cleared,"
             "forgotten,self_damped,floor_damped,floor_clears,"
             "floor_rets,floor_blocked,marks_suppr,floor_dropped,decayed,zheld,held,unseen,decay_w,"
             "clear_damped,clear_surf,clear_blind,clear_p,bad_pts,repaired,bins_conf,bins_refut,unsupported\n";
    }
    f << cyc << ',' << grid_.occupied_count() << ',' << d.hits << ',' << d.misses << ','
      << d.miss_blocked_zaware << ',' << d.cells_latched << ',' << d.cells_released << ','
      << d.hit_then_cleared << ',' << d.cells_forgotten << ',' << d.self_hits_damped << ','
      << d.floor_damped_hits << ',' << d.floor_endpoint_clears << ','
      << d.floor_endpoint_returns << ',' << d.floor_endpoint_blocked << ','
      << d.marks_suppressed << ',' << last_device_floor_dropped_ << ','
      << d.cells_decayed << ',' << d.cells_zheld << ',' << d.cells_held << ',' << d.cells_unseen << ','
      << (d.cells_decayed > 0 ? d.decay_weight_sum / d.cells_decayed : 0.0) << ','
      << d.clear_damped << ',' << d.clear_surface_damped << ',' << d.clear_blind << ','
      << d.clear_stopped << ',' << d.clear_blind_shell << ','
      << d.bad_points << ',' << d.cells_repaired << ','
      << d.bins_confirmed << ',' << d.bins_refuted << '\n';
    if ((cyc % 20) == 0)
    {
        f.flush();
        const double blocked_frac = (d.misses + d.miss_blocked_zaware) > 0
            ? 100.0 * static_cast<double>(d.miss_blocked_zaware) / (d.misses + d.miss_blocked_zaware) : 0.0;
        std::println("[grid-diag] occ={} hits={} miss={} zaware_block={} ({:.0f}% of clearing) latch={} "
                     "release={} forgotten={} self_damped={} floor_damped={} floor_clears={} hit_then_cleared={}",
                     grid_.occupied_count(), d.hits, d.misses, d.miss_blocked_zaware, blocked_frac,
                     d.cells_latched, d.cells_released, d.cells_forgotten, d.self_hits_damped,
                     d.floor_damped_hits, d.floor_endpoint_clears, d.hit_then_cleared);
    }
    ++cyc;
}

// Floor diagnostic: the estimated floor plane (offset/tilt/fit quality) next to the HEIGHT PROFILE of the cells
// this agent actually publishes as obstacles. This is the evidence that decides the phantom question, and it lands
// on disk (the [floor-plane] line only ever went to stdout, which nobody captures).
//
// READ IT AS: `resid` (occupied ∧ ¬explained) is the number that matters — `occupied` counts every latched cell and
// in an apartment the WALLS are ~80% of them, which is why the previous version of this log looked healthy (a mass
// of tall wall cells in h_gt120) while the robot could not navigate. If the low `r_le*` bins carry the residual
// mass, the published obstacles ARE floor-height returns; `rms_m` then says how much scatter the floor model is
// working with, and `off_cm`/`tilt_deg` whether the plane itself is the mismatch. Residual mass in the 0.25–0.70 m
// bins instead means real furniture the object agents are failing to claim — an explainer problem, not a floor one.
// The height used for binning is each cell's CURRENT top (dispz_), not the never-contracting running max.
// ── WHY WAS THIS RESIDUAL REMOVED? ───────────────────────────────────────────────────────────────────────────
// grid_diag.csv counts releases; it cannot say why any PARTICULAR obstacle went, and "the table in the middle of
// the room disappeared" is a question about one particular obstacle. One row per released cell, with the evidence
// that finished it. Read it like this:
//   age_cycles LARGE + cause=see-through  -> a structure that stood for minutes was deleted by seconds of beams.
//   clear_z just OUTSIDE [zmn, zmx]       -> a grazing beam, the one that used to erase flat surfaces.
//   range_m small                         -> the robot was on top of it, where the lidar can see least.
// A table standing in the room must produce NO rows at all inside its footprint: it does not fly, and with no
// table_concept running nothing may explain it away.
void SpecificWorker::log_releases()
{
    if (cfg_.release_csv_path.empty()) return;
    const auto& ev = grid_.last_releases();
    if (ev.empty()) return;
    static bool first = true;
    std::ofstream f(cfg_.release_csv_path, first ? std::ios::trunc : std::ios::app);
    if (not f) return;
    f.imbue(std::locale::classic());       // decimal POINT regardless of LANG — see CLAUDE.md
    if (first) { f << "cycle,x,y,lo_before,lo_after,zmn,zmx,top_z,clear_z,clear_w,range_m,age_cycles,"
                     "cause,sensor\n";
                 first = false; }
    for (const auto& e : ev)
        f << grid_diag_cycle_ << ',' << e.x << ',' << e.y << ',' << e.lo_before << ',' << e.lo_after << ','
          << e.zmn << ',' << e.zmx << ',' << e.last_z << ',' << e.clear_z << ',' << e.clear_w << ','
          << e.range_m << ',' << e.age_cycles << ','
          << (e.cause == 0 ? "see-through" : e.cause == 1 ? "decay" : "no-support") << ','
          // WHICH SENSOR finished the cell. helios, bpearl and the ZED fail in completely different ways, and
          // "the table is being eroded" cannot be acted on until this column says which one is doing it.
          << (e.src == 1 ? "bpearl" : e.src == 2 ? "helios" : e.src == 3 ? "zed" : "?") << '\n';
}

void SpecificWorker::log_floor_diag(const rc::OccupancyGrid::CellExplained& explained,
                                    const std::vector<rc::OccComponent>& comps)
{
    if (not grid_ready_) return;
    static const std::vector<float> edges{0.10f, 0.15f, 0.25f, 0.40f, 0.70f, 1.20f};
    static long cyc = 0;
    static std::ofstream f;
    if (not f.is_open())
    {
        open_diag_csv(f, "etc/floor_diag.csv");
        f << "cycle,applied,a,b,c,off_cm,tilt_deg,n_cand,rms_m,occupied,resid,ncomp,max_cells,"
             "r_le10,r_le15,r_le25,r_le40,r_le70,r_le120,r_gt120\n";
    }
    const auto bins = grid_.residual_height_hist(edges, explained);
    std::size_t maxc = 0; for (const auto& c : comps) maxc = std::max<std::size_t>(maxc, c.n_cells);
    f << cyc << ',' << (cfg_.floor_plane.enabled ? 1 : 0) << ','
      << floor_plane_.a << ',' << floor_plane_.b << ',' << floor_plane_.c << ','
      << floor_plane_.c * 100.0f << ','
      << std::atan(std::hypot(floor_plane_.a, floor_plane_.b)) * 57.2958f << ','
      << floor_plane_.n_candidates << ',' << floor_plane_.rms << ',' << grid_.occupied_count() << ','
      << grid_.residual_count(explained) << ',' << comps.size() << ',' << maxc;
    for (const long b : bins) f << ',' << b;
    f << '\n';
    if ((cyc % 20) == 0) f.flush();
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
        rc::provenance::stamp_creation(*G, gn);   // birth stamp: epoch ms + local ISO-8601
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
    // names, so we reuse residual_pts (a vector<float>) rather than a bespoke grid attr. Retina reads it
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

    // ── NO MORE COMPONENT HULLS ──────────────────────────────────────────────────────────────────
    // ★★★grid_obstacle_hulls is GONE (2026-08-20). It published each connected component as an exact
    // rectangular cover, C-space inflated by 0.25 m, which the controller decoded into polygons and
    // rasterised back into a grid: grid → components → inflate → cover → decode → grid. The cover was
    // exact by area but not by identity — a tiling of a blob is unstable frame to frame, and the
    // planner's world churned at 20 Hz beneath a stationary robot. The inflation was also counted
    // twice, here and again when the planner applied the robot's own footprint.
    // ★The controller now reads `grid_occupied_cells` — which this agent already publishes, and which
    // the 3-D viewers already use — and marks it directly. One representation, one inflation, no trip.
    // The `[grid-publish]` area-ratio guard went with it; its job (published area must track occupied
    // area at ~1.00x) is now true by construction, because there is no second representation to drift.

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
