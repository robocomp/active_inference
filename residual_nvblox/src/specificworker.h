/*
 *    Copyright (C) 2026 by RoboComp CORTEX Team
 *
 *    This file is part of RoboComp — GNU GPL v3 (see the project license).
 */

/**
 * SpecificWorker — residual_nvblox: the null/catch-all concept at the bottom of the hierarchy.
 * RoboComp lifecycle + presence protocol + orchestration. compute() wires the collaborators:
 *   lidar_ingestor_ (room-frame sweep) → nvb_ (nvblox occupancy, per sensor frame) → grid_ (room-frame cells
 *   read out of the voxel columns) → explainers (floor/ceiling/robot/walls + soft object footprints) → the
 *   `residual` node: every occupied cell no model explains, which the controller's planner avoids.
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <genericworker.h>
#include <Eigen/Dense>

#include <dsr/api/dsr_rt_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "residual_config.h"
#include "residual_clusterer.h"
#include "residual_occupancy_grid.h"   // PHASE-0 REBUILD: occupancy-grid safety layer
#include "residual_nvblox_map.h"
#include "residual_lidar_ingestor.h"

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
    ~SpecificWorker();

public slots:
    void initialize();
    void compute();
    void emergency();
    void restore();
    int  startup_check();

    void modify_node_slot(std::uint64_t, const std::string&) {}
    void modify_node_attrs_slot(std::uint64_t, const std::vector<std::string>&) {}
    void modify_edge_slot(std::uint64_t, std::uint64_t, const std::string&) {}
    void modify_edge_attrs_slot(std::uint64_t, std::uint64_t, const std::string&, const std::vector<std::string>&) {}
    void del_edge_slot(std::uint64_t, std::uint64_t, const std::string&) {}
    void del_node_slot(std::uint64_t from);

private:
    // ── Presence protocol (copied verbatim from bottle_concept) ──
    void waiting_enter();  void waiting_loop();
    void operating_enter(); void operating_loop();
    void degraded_enter(); void degraded_loop();
    void cleanup_owned_nodes();
    void request_shutdown();
    void terminal_shutdown();
    static constexpr int REQUIRED_LOSS_GRACE_MS = 3000;
    void on_optional_peer_lost(const std::string& name, std::uint32_t id);
    void on_optional_peer_ready(const std::string& name, std::uint32_t id);
    // Delete every "residual_*" obstacle node this agent owns (startup sweep + teardown).
    void remove_owned_residual_nodes();

    // PHASE-1 REBUILD: publish the residual occupancy as a `grid` node under room (cell centres + cell size)
    // for the retina 3-D display. Throttled; the node is created once and its cell attribute is updated.
    void  publish_grid_display(const rc::OccupancyGrid::CellExplained& explained,
                               const std::vector<rc::OccComponent>& comps);
    // DIAGNOSTIC: append per-sweep grid dynamics to etc/grid_diag.csv (hits/misses/latched/released + the
    // "hit_then_cleared" smoking-gun for grazing beams erasing a horizontal surface they graze). Optionally
    // probes a rectangular region [GridProbe*] (e.g. a tabletop) reporting occupied/hit cells inside it.
    void  log_grid_diag();
    // Per-cell geometry of the PUBLISHED residual set → etc/residual_cells.csv, every Grid.CellDumpEveryN
    // cycles (0 = off). The height histogram in floor_diag says how TALL the blocking mass is; it cannot say
    // whether a 1.4 m cell is an unclaimed wall or an unmodelled wardrobe, and those need opposite fixes.
    void  dump_residual_cells(const rc::OccupancyGrid::CellExplained& explained);
    void  log_births();                      // per-latched-cell trace → residual_births.csv (WHAT created it)
    void  log_releases();                    // per-released-cell trace → residual_releases.csv (WHY it went)
    // The height profile of the cells that actually LEAVE this agent → etc/floor_diag.csv.
    // Takes the read-out predicate because `occupied` alone cannot answer the phantom question: in an apartment
    // the walls are ~80% of the latched cells, so a histogram of every latched cell reports a healthy-looking
    // mass of tall wall cells and says nothing about the handful of floor phantoms that block navigation.
    void  log_floor_diag(const rc::OccupancyGrid::CellExplained& explained,
                         const std::vector<rc::OccComponent>& comps);

    // ── compute() rate + cost meter ──
    // Accumulates one reporting window (1 s) of loop timings and prints a single [perf] line. `busy_ms` is the
    // time spent INSIDE this compute() call; the period is measured call-to-call, so both the loop's real rate
    // and the fraction of it this agent actually consumes are visible. See the .cpp for how CPU% is read.
    void log_compute_perf(double busy_ms);
    std::chrono::steady_clock::time_point perf_window_begin_{};   // start of the current 1 s reporting window
    std::chrono::steady_clock::time_point perf_last_call_{};      // previous compute() entry (call-to-call period)
    unsigned int perf_calls_ = 0;        // compute() calls in this window
    double       perf_busy_ms_ = 0.0;    // summed time inside compute() in this window
    double       perf_worst_ms_ = 0.0;   // worst call-to-call period in this window
    // LEAK WATCH (see log_compute_perf). One row per reporting window carrying the timings AND VmRSS, so a
    // slow leak is visible as a RATE and can be joined against grid_diag.csv on `grid_cycle`. The baseline
    // is the FIRST window rather than construction time, so it excludes startup/graph-sync warm-up.
    std::ofstream perf_csv_;                                     // etc/perf_diag.csv (rotated, see open_diag_csv)
    long          perf_rss0_kb_ = -1;                            // VmRSS at the first window (growth baseline)
    std::chrono::steady_clock::time_point perf_t0_{};            // when that baseline was taken
    long          grid_diag_cycle_ = 0;                          // grid_diag.csv row counter (the join key)
    // Trailing (t_s, VmRSS kB) samples for the WINDOWED growth rate.
    //
    // The lifetime average (rss - baseline)/elapsed is the WRONG statistic for leak detection and was
    // actively misleading here: measured on 2026-08-08 it reported 7.84 → 1.93 MB/min across a stretch
    // where RSS was CONSTANT, purely because the numerator froze while the denominator kept growing. The
    // failure it cannot see is the one that matters — a leak that STARTS after warm-up gets diluted across
    // the whole run, so the burst that killed this agent would have shown a small, reassuring average for
    // many minutes. A trailing window reports what is happening NOW and surfaces onset within one window.
    static constexpr double PERF_RATE_WINDOW_S   = 60.0;   // width of the trailing window
    static constexpr double PERF_RATE_MIN_SPAN_S = 15.0;   // below this the span is too short to mean anything
    std::deque<std::pair<double, long>> perf_rss_hist_;

    // Refresh soft_objects_ from the graph's modelled `object` nodes (published box + RT covariance).
    void build_soft_objects();
    // The HARD explainers: room floor/ceiling/walls (from room_concept) + the robot. Objects are soft_objects_.
    std::vector<rc::SpecialistExplainer> build_explainers(const Eigen::Vector2f& robot_xy) const;
    // Room delimiting polygon (room frame) for the clusterer's wall/outside reject; empty if unavailable.
    std::vector<Eigen::Vector2f> read_room_polygon() const;

    // ── Members ──
    bool startup_check_flag = false;
    bool owned_nodes_cleaned_ = false;
    bool startup_sweep_done_ = false;   // one-time leftover-residual sweep at first Operating (graph joined)
    std::atomic<bool> shutting_down_{false};
    AgentPresenceCoordinator presence_coordinator_;

    rc::ResidualConfig cfg_;

    std::unique_ptr<DSR::RT_API>           rt_api_;
    std::unique_ptr<DSR::InnerEigenAPI>    inner_eigen_;
    std::uint64_t                          room_node_id_ = 0;

    // Soft-collapse descriptor per modelled object (table/chair/cylinder): 2D footprint + top height + the σ's
    // from the object's OWN published position covariance (⊕ sensor noise). A grid cell is explained by an
    // object with probability Φ(−sdf2D/σ_xy)·Φ((z_top+noise−z_hi)/σ_z): a horizontal-surface hit (tabletop,
    // z_hi≈z_top) collapses, while a return sitting ABOVE the surface (an on-table obstacle) does NOT. Filled
    // each cycle by build_soft_objects.
    struct SoftObject { float cx, cy, yaw, hx, hy, z_top, sigma_xy, sigma_z; };
    std::vector<SoftObject> soft_objects_;
    // Nodes already warned about a missing rt_covariance — the warning is a one-per-node fact about that agent's
    // publishing, not a per-cycle event, and at 10 Hz an unthrottled println would bury the log.
    mutable std::set<std::uint64_t> cov_warned_;

    // Temporally-smoothed PUBLISHED belief field (asymmetric EMA — see grid_field_ema_*). Persists across
    // publishes; re-initialised if the grid extent changes. Display/planner-facing only; not the safety latch.
    std::vector<float> pub_prob_ema_, pub_var_ema_;
    long last_device_floor_dropped_ = 0;            // grid_diag column kept for the stock CSV layout (always 0)

    // PHASE-0 REBUILD: the occupancy-grid safety layer (runs live as a diagnostic first, then becomes the
    // single source of obstacle truth once verified stable).
    rc::OccupancyGrid grid_;
    bool              grid_ready_ = false;

    std::unique_ptr<rc::NvbloxMap> nvb_;   // built with the grid (its sampling follows the grid's cells)

    // Collaborators (constructed in initialize()). Declared in dependency order.
    std::unique_ptr<rc::ResidualLidarIngestor> lidar_ingestor_;
    bool                             graph_dirty_ = false; // a node was created/deleted this cycle → relayout (twopi)

signals:
    void presenceReady();
    void presenceLost();
};

#endif // SPECIFICWORKER_H
