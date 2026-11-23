/*
 *    Copyright (C) 2026 by RoboComp CORTEX Team
 *    This file is part of RoboComp — GNU GPL v3.
 */

/**
 * SpecificWorker — residual_concept: lifecycle + presence + orchestration.
 */

#include "specificworker.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)
#include "../../common/room_resolve/room_resolve.h"   // rc::room::current_room (proto-aware, deterministic)

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

    // Isolated unit tests run once at startup so a broken polygon helper / grid is caught before the live loop.
    rc::ResidualClusterer::self_test();
    rc::OccupancyGrid::self_test();

#ifdef HIBERNATION_ENABLED
    hibernationChecker.start(500);
#endif

    // Registered rather than read raw, so the period appears in the startup table like every
    // other key. req() keeps the throw-if-missing behaviour: a period is not something to
    // default silently.
    int period = 0;
    rc::cfg::Reader(configLoader, "residual_concept").req("Period.Compute", period,
            "GRAFCET step period (ms) for every state of this agent's state machine");

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

    lidar_ingestor_ = std::make_unique<rc::ResidualLidarIngestor>(G, inner_eigen_.get(), cfg_);

    connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

    remove_owned_residual_nodes();

    if (const auto room = rc::room::current_room(*G); room.has_value()) room_node_id_ = *room;
    else                   qWarning() << "residual_concept: no room node found at startup";

    // ── WHAT THIS AGENT IS ACTUALLY RUNNING ────────────────────────────────────────────────────
    //
    // ★PUBLISHED AT THE END OF initialize(), NOT WHERE THE CONFIG IS PARSED. This agent's own keys
    // are settled much earlier, but the SHARED presence unit reads its sixteen [Presence.*]/[Owns.*]
    // keys when the coordinator is configured, further down this same function. Publishing before
    // that armed the unread sweep on a registry those sixteen had not reached yet, and named every
    // one of them "in the file, read by nothing" - confident false positives from the one check whose
    // whole value is that it does not cry wolf. The rule is general: publish when the LAST reader has
    // run, which is the end of startup, not the end of parsing.
    //
    // Prints only the DELTAS - values differing from the code default, plus every A/B arm even at its
    // default - and writes the full table to etc/config_effective.csv, this run's own record of which
    // arm it was. Config is read once at startup, so a file's mtime never says which run used it.
    //
    // declare_complete() CLAIMS that every config key this agent reads goes through a Reader, and it
    // ARMS the unread sweep; check_registry_complete.sh residual_concept is the grep that keeps the claim
    // honest. Re-run it whenever a config read is added.
    rc::cfg::exempt_generated_prefixes();
    rc::cfg::registry().declare_complete("residual_concept");
    rc::cfg::Reader(configLoader, "residual_concept").publish("etc/config_effective.csv");
}

void SpecificWorker::build_soft_objects()
{
    // Every modelled object's published box geometry becomes a SOFT explainer of the grid cells under its
    // footprint (see SoftObject). RT pose ≈ the object base, so the box top is pose z + height.
    soft_objects_.clear();
    if (not inner_eigen_)
        return;
    const float sensor_var = cfg_.object_sensor_sigma_m * cfg_.object_sensor_sigma_m;
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
            const auto T = inner_eigen_->get_transformation_matrix(rc::room::current_room_frame(*G), n.name(), 0);
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
            const float hx = 0.5f * w, hy = 0.5f * d;
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
}

std::vector<rc::SpecialistExplainer> SpecificWorker::build_explainers(const Eigen::Vector2f& robot_xy) const
{
    std::vector<rc::SpecialistExplainer> ex;

    // 0: ROOM FLOOR — a flat band over z = 0. nvblox already judged the floor at integration, per sweep, against
    //    the OBSERVING sensor's range (FloorPush); a range-grown band here, measured from where the robot is NOW,
    //    re-labels an obstacle seen up close as floor once the robot drives away (a 15 cm box vanished past ~2.9 m).
    const float fz0 = cfg_.floor_z0;
    ex.push_back([fz0](const Eigen::Vector3f& p) { return p.z() < fz0; });
    // 1: ROOM CEILING — overhead structure above the navigation band.
    const float cz = cfg_.ceil_z;
    ex.push_back([cz](const Eigen::Vector3f& p) { return p.z() > cz; });
    // 2: ROBOT — its own body returns.
    const float rr2 = cfg_.robot_radius_m * cfg_.robot_radius_m;
    ex.push_back([rr2, robot_xy](const Eigen::Vector3f& p) { return (p.head<2>() - robot_xy).squaredNorm() < rr2; });
    // 3: ROOM WALLS — the room_concept delimiting polygon. A return outside the room, or within wall_margin of
    //    a wall edge, is explained by the room's wall model.
    const auto poly = read_room_polygon();
    const float wm = cfg_.wall_margin_m;
    if (poly.size() >= 3)
        ex.push_back([poly, wm](const Eigen::Vector3f& p)
                     { const Eigen::Vector2f xy = p.head<2>();
                       return not rc::ResidualClusterer::point_in_polygon(poly, xy)
                              or rc::ResidualClusterer::dist_to_polygon_boundary(poly, xy) < wm; });
    else
        ex.push_back([](const Eigen::Vector3f&) { return false; });   // no room polygon yet → no wall model
    // Modelled OBJECTS are a SOFT collapse (soft_objects_, see compute()), not hard explainers here.
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
    if (not dumped and cfg_.csv_logging)
    {
        dumped = true;
        std::ofstream f("etc/residual_room_poly.csv", std::ios::out | std::ios::trunc);
        if (f.is_open()) { f << "i,x,y\n"; for (std::size_t i = 0; i < n; ++i) f << i << ',' << xs[i] << ',' << ys[i] << '\n'; }
    }
    return poly;
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
    if (not perf_csv_tried and cfg_.csv_logging)
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
        const auto room = rc::room::current_room(*G);
        if (not room.has_value()) return;
        room_node_id_ = *room;
    }

    // Fresh LiDAR sweep is the only trigger — no sweep, no re-cluster this cycle (instances persist).
    if (not lidar_ingestor_ or not lidar_ingestor_->pump())
        return;

    const Eigen::Vector2f robot_xy = lidar_ingestor_->origin_room().head<2>();

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
            // Only what the read-out of the nvblox columns uses (OccupancyGrid::ingest_columns and the read-outs).
            gp.floor_z0 = cfg_.floor_z0; gp.ceil_z = cfg_.ceil_z;
            gp.bin_span_m           = cfg_.grid_bin_span_m;
            gp.lidar_clearance_m    = cfg_.grid_lidar_clearance_m;
            gp.speckle_min_neighbours = cfg_.grid_speckle_min_neighbours;
            gp.speckle_grace_cycles   = cfg_.grid_speckle_grace_cycles;
            gp.speckle_min_component_cells = cfg_.grid_speckle_min_component_cells;
            // NO C-SPACE INFLATION: the controller collides its real footprint against the grid, so inflating here
            // would count it twice.
            gp.inflate_radius_m    = 0.0f;
            grid_.reset(xmn, ymn, xmx, ymx, gp);
            grid_ready_ = grid_.valid();
            std::println("[grid] init bounds=({:.1f},{:.1f})..({:.1f},{:.1f}) cells={}x{}",
                         xmn, ymn, xmx, ymx, grid_.width(), grid_.height());
            if (grid_ready_)
                nvb_ = std::make_unique<rc::NvbloxMap>(cfg_, grid_);
        }
        if (grid_ready_)   // integrate now (needs no explainers); read-out + publish happen after explainers below
        {
            // Observer for this cycle: the robot axis (range of each cell, and the LidarClearanceM hold).
            grid_.set_self_body(lidar_ingestor_->origin_room().x(), lidar_ingestor_->origin_room().y(), 0.0f);
            const auto t0 = std::chrono::steady_clock::now();
            const std::string room = rc::room::current_room_frame(*G);
            nvb_->integrate(*lidar_ingestor_, [this, &room](const std::string& sensor, std::int64_t stamp)
                -> std::optional<Eigen::Matrix4d>
            {
                const auto T = inner_eigen_->get_transformation_matrix(room, sensor, static_cast<std::uint64_t>(stamp),
                                                                       "RT", DSR::RT_API::TimeQuery::Interpolated);
                if (not T.has_value()) return std::nullopt;
                Eigen::Matrix4d M;   // element-wise copy: see the Eigen ABI note in src/CMakeLists.txt
                const auto& m = T.value().matrix();
                for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) M(i, j) = m(i, j);
                return M;
            }, grid_.occupied_count());
            const auto ti = std::chrono::steady_clock::now();
            grid_.ingest_columns(nvb_->columns(), cfg_.nvblox_k_obs, cfg_.nvblox_read_min_z, cfg_.nvblox_occ_clear_lo,
                                 cfg_.nvblox_release_cycles);
            const auto t1 = std::chrono::steady_clock::now();
            nvb_->log_map_time(std::chrono::duration<double, std::milli>(t1 - t0).count(),
                               std::chrono::duration<double, std::milli>(t1 - ti).count(), grid_.occupied_count());
            log_grid_diag();
        }
    }

    build_soft_objects();                                    // modelled objects → soft read-out collapse
    const auto explainers = build_explainers(robot_xy);     // floor + ceiling + robot + walls (hard)

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
        const float fitm = cfg_.object_fit_margin_m;       // under-fit slack (< planner clearance)
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
        log_births();                            // ...and WHAT CREATED each cell that appeared
        dump_residual_cells(cell_explained);     // ...and WHERE they are, which a histogram cannot say
        static int gc = 0;
        if ((gc++ % 20) == 0)
        {
            std::size_t maxc = 0; for (const auto& c : comps) maxc = std::max<std::size_t>(maxc, c.n_cells);
            // `speckle` = lone residual cells the read-out withheld (Grid.SpeckleMinNeighbours). Read it against
            // the components: if it is the same order as what ships, the map is mostly lone cells and the filter
            // is carrying it — which is a reason to look at why they are being born, not to raise the filter.
            std::println("[grid] residual components={} max_cells={} speckle_dropped={} "
                         "(walls/specialists subtracted at read-out)",
                         comps.size(), maxc, grid_.speckle_dropped());
        }
        // `grid` node under room: residual cells + inflated border (display) + inflated component hulls
        // (`grid_polygons`, encoded) which the controller reads and plans against alongside the known objects.
        // Residual obstacle NODES are no longer uploaded.
        publish_grid_display(cell_explained, comps);
        nvb_->show(grid_, cell_explained, lidar_ingestor_->origin_room());
    }

    // Keep the DSR graph tidy: relayout once if a node was created/deleted this cycle (no-op when headless).
    if (graph_dirty_)
    {
        trigger_graph_layout_twopi();
        graph_dirty_ = false;
    }
}

void SpecificWorker::dump_residual_cells(const rc::OccupancyGrid::CellExplained& explained)
{
    // WHERE the residual cells are, not just how tall they are. The height histogram says 13% of the published
    // mass sits above 1.20 m, which "must be walls" in an apartment — but a histogram cannot tell a wall from a
    // wardrobe, and acting on that guess is how the floor_clears defect survived ten days. So dump the geometry
    // and let it answer: dist_wall_m small ⇒ the room polygon is failing to claim its own wall; obj_sdf_m small
    // ⇒ a specialist under-claims its footprint; both large ⇒ a genuinely free-standing object nobody models.
    if (not cfg_.csv_logging or cfg_.cell_dump_every_n <= 0 or not grid_ready_) return;
    if ((grid_diag_cycle_ % cfg_.cell_dump_every_n) != 0) return;

    // Every LATCHED cell, with what keeps it latched: the column's max voxel (lo), how many voxels are still
    // positive and at what heights, how long since a return struck it, and the refuted-material weight since.
    static std::ofstream od;
    if (not od.is_open())
    {
        open_diag_csv(od, "etc/residual_occ_debug.csv");
        if (od.is_open())
            od << "cycle,x,y,lo,n_pos_vox,z_pos_min,z_pos_max,sup_max,age_hit_cycles,refute_w,age_latched_cycles,"
                  "seen_frac,range_m\n";
    }
    if (od.is_open()) { grid_.dump_occupied(od, grid_diag_cycle_); od.flush(); }

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
    const float fitm = cfg_.object_fit_margin_m;
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
    if (cfg_.csv_logging and not f.is_open())
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
             "pose_damped,pose_sig_pos,pose_sig_th,speckle,"
        // ★ This header MUST match the write below field for field. It drifted once (two columns were added to
        //   the row and not to the header) and every column after decay_w was then read as its neighbour —
        //   including by me, while diagnosing. Count them if you touch either.
             "clear_damped,clear_surf,clear_blind,clear_stopped,blind_shell,bad_pts,repaired,"
             "bins_conf,bins_refut,clear_lateral,clear_lat_exempt\n";
    }
    if (f.is_open())
    f << cyc << ',' << grid_.occupied_count() << ',' << d.hits << ',' << d.misses << ','
      << d.miss_blocked_zaware << ',' << d.cells_latched << ',' << d.cells_released << ','
      << d.hit_then_cleared << ',' << d.cells_forgotten << ',' << d.self_hits_damped << ','
      << d.floor_damped_hits << ',' << d.floor_endpoint_clears << ','
      << d.floor_endpoint_returns << ',' << d.floor_endpoint_blocked << ','
      << d.marks_suppressed << ',' << last_device_floor_dropped_ << ','
      << d.cells_decayed << ',' << d.cells_zheld << ',' << d.cells_held << ',' << d.cells_unseen << ','
      << (d.cells_decayed > 0 ? d.decay_weight_sum / d.cells_decayed : 0.0) << ','
      << d.pose_damped << ',' << grid_.pose_sigma_pos() << ',' << grid_.pose_sigma_theta() << ','
      << grid_.speckle_dropped() << ','
      << d.clear_damped << ',' << d.clear_surface_damped << ',' << d.clear_blind << ','
      << d.clear_stopped << ',' << d.clear_blind_shell << ','
      << d.bad_points << ',' << d.cells_repaired << ','
      << d.bins_confirmed << ',' << d.bins_refuted << ','
      << d.clear_lateral << ',' << d.clear_lateral_exempt << '\n';
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

// Floor diagnostic: the HEIGHT PROFILE of the cells this agent publishes as obstacles (etc/floor_diag.csv).
// READ IT AS: `resid` (occupied ∧ ¬explained) is the number that matters — `occupied` counts every latched cell and
// in an apartment the WALLS are ~80% of them. Residual mass in the low `r_le*` bins = floor-height returns; in the
// 0.25–0.70 m bins = real furniture the object agents are failing to claim (an explainer problem, not a floor one).
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
// ── WHAT CREATED THIS CELL? ──────────────────────────────────────────────────────────────────────────────────
// One row per cell the moment it latches. In an EMPTY room every row is a phantom, so this file is the whole
// diagnosis: `sensor` names the culprit, `z` says what height it thought it saw, `range_m` how far away, and
// `bins` whether the column is a one-voxel sliver (noise) or a real vertical extent (a thing).
void SpecificWorker::log_births()
{
    if (not cfg_.csv_logging or cfg_.birth_csv_path.empty()) return;
    const auto& ev = grid_.last_latches();
    if (ev.empty()) return;
    static bool first = true;
    std::ofstream f(cfg_.birth_csv_path, first ? std::ios::trunc : std::ios::app);
    if (not f) return;
    f.imbue(std::locale::classic());       // decimal POINT regardless of LANG — see CLAUDE.md
    if (first) { f << "cycle,x,y,z,lo,w,range_m,bins,sensor,robot_x,robot_y\n"; first = false; }
    for (const auto& e : ev)
        f << grid_diag_cycle_ << ',' << e.x << ',' << e.y << ',' << e.z << ',' << e.lo << ',' << e.w << ','
          << e.range_m << ',' << e.bins << ','
          << (e.src == 1 ? "bpearl" : e.src == 2 ? "helios" : e.src == 3 ? "zed" : "?") << ','
          << e.robot_x << ',' << e.robot_y << '\n';
}

void SpecificWorker::log_releases()
{
    if (not cfg_.csv_logging or cfg_.release_csv_path.empty()) return;
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
          << (e.cause == 0 ? "see-through" : e.cause == 1 ? "decay" : e.cause == 3 ? "stale" : "no-support") << ','
          // WHICH SENSOR finished the cell. helios, bpearl and the ZED fail in completely different ways, and
          // "the table is being eroded" cannot be acted on until this column says which one is doing it.
          << (e.src == 1 ? "bpearl" : e.src == 2 ? "helios" : e.src == 3 ? "zed" : "?") << '\n';
}

void SpecificWorker::log_floor_diag(const rc::OccupancyGrid::CellExplained& explained,
                                    const std::vector<rc::OccComponent>& comps)
{
    if (not grid_ready_ or not cfg_.csv_logging) return;
    static const std::vector<float> edges{0.10f, 0.15f, 0.25f, 0.40f, 0.70f, 1.20f};
    static long cyc = 0;
    static std::ofstream f;
    if (not f.is_open())
    {
        open_diag_csv(f, "etc/floor_diag.csv");
        f << "cycle,occupied,resid,ncomp,max_cells,"
             "r_le10,r_le15,r_le25,r_le40,r_le70,r_le120,r_gt120,"
             // ...and the same profile over the OCCUPIED set, before any explainer subtracts. Until now only
             // the RESIDUAL was histogrammed — what SHIPS — so a phantom population the explainers happen to
             // mask could fill the map without leaving a trace in any log. o_le* is what is actually THERE.
             "o_le10,o_le15,o_le25,o_le40,o_le70,o_le120,o_gt120\n";
    }
    const auto bins = grid_.residual_height_hist(edges, explained);
    std::size_t maxc = 0; for (const auto& c : comps) maxc = std::max<std::size_t>(maxc, c.n_cells);
    f << cyc << ',' << grid_.occupied_count() << ','
      << grid_.residual_count(explained) << ',' << comps.size() << ',' << maxc;
    for (const long b : bins) f << ',' << b;
    for (const long b : grid_.occupied_height_hist(edges)) f << ',' << b;
    f << '\n';
    if ((cyc % 20) == 0) f.flush();
    ++cyc;
}

void SpecificWorker::publish_grid_display(const rc::OccupancyGrid::CellExplained& explained,
                                          const std::vector<rc::OccComponent>& comps)
{
    static int t = 0;
    // ★WAS A HARDCODED `% 5` (2 Hz), CALLED A "display update" — which it was, until the controller
    // started planning against this grid. See ResidualConfig::grid_publish_every_n for why it moved
    // to 10 Hz and what it costs. Attr-only either way: no node churn.
    const int every_n = std::max(1, cfg_.grid_publish_every_n);
    if ((t++ % every_n) != 0) return;
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

void SpecificWorker::del_node_slot(std::uint64_t)
{
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
