/*
 *    Copyright (C) 2026 by RoboComp CORTEX Team
 *
 *    This file is part of RoboComp — GNU GPL v3 (see the project license).
 */

/**
 * SpecificWorker — residual_concept agent: the null/catch-all concept at the bottom of the hierarchy.
 * RoboComp lifecycle + presence protocol + orchestration only. compute() wires the collaborators:
 *   lidar_ingestor_ (room-frame sweep) → clusterer_ (residual filter subtracting specialist geometry →
 *   3D DBSCAN) → tracker_ (birth/associate/death of "obstacle" nodes named residual_*) → fitter_
 *   (per-instance box-footprint free-energy fit + write-back). Any LiDAR return no specialist agent
 *   (table/chair/bottle/human) claims becomes a residual obstacle the controller's planner avoids.
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <genericworker.h>
#include <Eigen/Dense>

#include <dsr/api/dsr_rt_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_inner_gaussian_api.h>

#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "../../common/instance_tracker/instance_tracker.h"
#include "residual_config.h"
#include "residual_clusterer.h"
#include "residual_occupancy_grid.h"   // PHASE-0 REBUILD: occupancy-grid safety layer
#include "residual_scene_graph.h"
#include "residual_fitter.h"
#include "../../common/phantom_log/phantom_log.h"   // rc::history::PhantomLog (shadow-mode birth/death record)
#include "residual_lidar_ingestor.h"
#include "residual_zed_ingestor.h"

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
    ~SpecificWorker();

public slots:
    void initialize();
    void compute();
    // SHADOW-MODE birth/death recorder (CONCEPT_AGENT_LIFECYCLE.md §4.2). Records ONLY — it can never
    // alter a birth or a removal. Attribution fields captured at death separate a genuine classifier
    // phantom from one of our own removal defects.
    void log_phantom_event(std::string_view event, std::uint64_t id, std::string_view name,
                           float x, float y, const rc::ResidualInstance* inst, std::string_view note);

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

    // A modelled horizontal support surface (a tabletop): its top-z and oriented footprint. An obstacle
    // whose observed bottom is at/above this surface, over its footprint, is assumed to REST on it.
    struct SupportSurface { float top_z = 0.0f, cx = 0.0f, cy = 0.0f, yaw = 0.0f, hx = 0.0f, hy = 0.0f; };
    // Refresh the support surfaces (table tops) this cycle; and infer which one an obstacle rests on: the
    // HIGHEST surface whose top is ≤ z_min under (cx,cy). Returns the floor (0) when none applies.
    void  build_support_surfaces();
    float support_z_for(float cx, float cy, float z_min) const;

    // PHASE-1 REBUILD: publish the residual occupancy as a `grid` node under room (cell centres + cell size)
    // for the retina 3-D display. Throttled; the node is created once and its cell attribute is updated.
    void  publish_grid_display(const rc::OccupancyGrid::CellExplained& explained,
                               const std::vector<rc::OccComponent>& comps);
    // Integrate the dense ZED depth FoV into the occupancy grid as a SECOND sensor (fills LiDAR-grazed
    // tabletops, stabilises the costmap). Camera ray origin → z-aware carve stays correct. Gated by grid_zed_enabled.
    void  integrate_zed_into_grid();
    // Refresh semantic_map_ from the DSR `semantic` node (retina's dense YOLO-sem label map under `zed`), but
    // only re-copy the large blob when its frame_id changed. Returns true if a valid map is cached. Gated by the
    // Semantic.DownweightFloor flag (no-op / false when off). Main-thread graph read.
    bool  refresh_semantic_map();
    // Ego-motion evidence trust (0..1): 1 when still, <1 while the robot moves (pose jitter + blur). Scales the
    // whole sweep's grid evidence so the stable accumulated field dominates during motion (motion-stability).
    float compute_ego_reliability() const;
    // LiDAR sweep with the LOW bpearl lidar's floor-grazing returns removed (device-specific floor band), so
    // bpearl keeps its low-obstacle value without ringing the robot with phantom floor. Buffered in lidar_filtered_.
    // BPEARL-ONLY sweep for the floor-plane fit. helios is an upright lidar whose floor returns are all
    // grazing, so it reads several cm high and its bias moves with the scene — it is a wall sensor, not a
    // floor datum. Falls back to the merged sweep if the per-device tag is unavailable. See the .cpp.
    const std::vector<Eigen::Vector3f>& floor_datum_sweep();
    // Per-device sweeps for the two-pass grid integration. Each LiDAR is integrated with ITS OWN floor sigma,
    // because a single sigma cannot describe both: bpearl measures the floor head-on (~1 cm scatter) while
    // helios only ever grazes it (~13 cm). See integrate_lidar_per_device().
    const std::vector<Eigen::Vector3f>& device_sweep(std::uint8_t plane);
    // Integrate the LiDAR sweep device-by-device, setting the floor component's sigma per device from that
    // device's OWN floor fit. The PLANE (the datum) always stays bpearl's; helios's fit contributes only its
    // scatter. Returns false if no per-device tag is available (caller falls back to a single merged sweep).
    bool integrate_lidar_per_device();
    rc::FloorPlane floor_plane_helios_;   // helios-only fit — used ONLY for its rms (never as the datum)
    const std::vector<Eigen::Vector3f>& filtered_lidar_sweep();
    // DIAGNOSTIC: append per-sweep grid dynamics to etc/grid_diag.csv (hits/misses/latched/released + the
    // "hit_then_cleared" smoking-gun for grazing beams erasing a horizontal surface they graze). Optionally
    // probes a rectangular region [GridProbe*] (e.g. a tabletop) reporting occupied/hit cells inside it.
    void  log_grid_diag();
    // Floor-plane fit + the height profile of the cells that actually LEAVE this agent → etc/floor_diag.csv.
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

    // ── Per-cycle orchestration ──
    void run_instance_tracker(const std::vector<rc::SpecialistSdf>& specialists);
    void merge_overlapping_instances();      // collapse two boxes fitted to the same object (footprint IoU)
    void retire_diverged_instances();        // drop diverged models (energy==0 for N frames)
    void process_residual_node(const DSR::Node& node);
    // Build the SpecialistSdf list from the graph's table/chair/cylinder nodes (their published box geometry).
    // Object SDFs are used for the DISSOLVE test; also wrapped into explainers below.
    std::vector<rc::SpecialistSdf> build_specialist_sdfs() const;
    // Build the unified EXPLAINER list = room floor/ceiling/walls (from room_concept) + robot + the object
    // SDFs. The residual is every LiDAR return NO explainer accounts for — no geometric-threshold masks.
    std::vector<rc::SpecialistExplainer> build_explainers(const Eigen::Vector2f& robot_xy,
                                                          const std::vector<rc::SpecialistSdf>& objects) const;
    // Room delimiting polygon (room frame) for the clusterer's wall/outside reject; empty if unavailable.
    std::vector<Eigen::Vector2f> read_room_polygon() const;
    // DISSOLVE-on-explained-interior (the fission policy): retire an instance whose footprint centre a
    // specialist now explains (e.g. a table modelled inside a table+chairs blob) so the next cycle's
    // residual re-cluster rebuilds the survivors fresh.
    void dissolve_explained_instances(const std::vector<rc::SpecialistSdf>& specialists);

    // Occupancy-evidence removal: carve the LiDAR sweep against each in-range instance box, integrate the
    // per-cycle occupancy log-odds, and remove an obstacle only once the accumulated evidence says the space
    // is empty (beams repeatedly pass THROUGH where it should be). Occlusion / out-of-range → HELD.
    void remove_by_occupancy_evidence();

    // ── Members ──
    bool startup_check_flag = false;
    bool owned_nodes_cleaned_ = false;
    bool startup_sweep_done_ = false;   // one-time leftover-residual sweep at first Operating (graph joined)
    std::atomic<bool> shutting_down_{false};
    AgentPresenceCoordinator presence_coordinator_;

    rc::ResidualConfig cfg_;

    std::unique_ptr<DSR::RT_API>           rt_api_;
    std::unique_ptr<DSR::InnerEigenAPI>    inner_eigen_;
    std::unique_ptr<DSR::InnerGaussianAPI> gaussian_api_;
    std::uint64_t                          room_node_id_ = 0;

    // Soft-collapse descriptor per modelled object (table/chair/cylinder): 2D footprint + top height + the σ's
    // from the object's OWN published position covariance (⊕ sensor noise). A grid cell is explained by an
    // object with probability Φ(−sdf2D/σ_xy)·Φ((z_top+noise−z_hi)/σ_z): a horizontal-surface hit (tabletop,
    // z_hi≈z_top) collapses, while a return sitting ABOVE the surface (an on-table obstacle) does NOT. Filled
    // each cycle by build_specialist_sdfs (mutable so it stays const).
    struct SoftObject { float cx, cy, yaw, hx, hy, z_top, sigma_xy, sigma_z; };
    mutable std::vector<SoftObject> soft_objects_;

    // Temporally-smoothed PUBLISHED belief field (asymmetric EMA — see grid_field_ema_*). Persists across
    // publishes; re-initialised if the grid extent changes. Display/planner-facing only; not the safety latch.
    std::vector<float> pub_prob_ema_, pub_var_ema_;
    float ego_reliability_ = 1.0f;   // this cycle's ego-motion evidence trust (compute_ego_reliability())
    // Timestamp of the previous grid commit. The elapsed time drives the grid's FORGETTING term, so the evidence
    // half-life stays expressed in seconds and does not silently change with the cycle rate.
    std::chrono::steady_clock::time_point last_commit_{};
    std::vector<Eigen::Vector3f> device_pts_;       // reusable buffer: single-device sweep (device_sweep)
    std::vector<Eigen::Vector3f> floor_datum_pts_;  // reusable buffer: bpearl-only points for the floor fit
    std::vector<Eigen::Vector3f> lidar_filtered_;   // reusable buffer: sweep with bpearl floor grazing removed

    // Cached ZED semantic label map (refreshed from the `semantic` node only when its frame_id changes — the map
    // is a large blob published at ~2 Hz). Feeds the RGB-semantic floor down-weighting of ZED grid hits.
    rc::SemanticMap semantic_map_;

    // Data-driven floor plane (z=a·x+b·y+c), estimated each cycle from the LiDAR sweep and EMA-smoothed. The grid
    // references its obstacle band to this instead of z=0, so an offset/tilted floor (new scenario) never latches.
    rc::FloorPlane floor_plane_;

    // PHASE-0 REBUILD: the occupancy-grid safety layer (runs live as a diagnostic first, then becomes the
    // single source of obstacle truth once verified stable).
    rc::OccupancyGrid grid_;
    bool              grid_ready_ = false;

    // Collaborators (constructed in initialize()). Declared in dependency order.
    std::unique_ptr<rc::ResidualSceneGraph>    scene_graph_;
    std::unique_ptr<rc::ResidualFitter>        fitter_;
    rc::history::PhantomLog                             phantom_log_;   // shadow-mode birth/death record
    std::unique_ptr<rc::ResidualLidarIngestor> lidar_ingestor_;
    std::unique_ptr<rc::ResidualZedIngestor>   zed_ingestor_;   // ZED dense-depth boost
    rc::ResidualClusterer                      clusterer_;
    Eigen::Matrix4f                            room_T_cam_ = Eigen::Matrix4f::Identity();   // camera→room this cycle
    bool                                       zed_ready_  = false;

    rc::InstanceTracker tracker_;
    std::vector<rc::ResidualCluster> clusters_;   // this cycle's detections (fitter reads by index)
    std::vector<Eigen::Vector2f>     cluster_centroids_;   // parallel to clusters_ (for birth seeding)
    std::uint64_t                    current_ts_ = 0;      // capture stamp of this cycle's sweep
    Eigen::Vector2f                  robot_xy_ = Eigen::Vector2f::Zero();   // robot centre (room), this cycle
    bool                             graph_dirty_ = false; // a node was created/deleted this cycle → relayout (twopi)
    std::vector<SupportSurface>      support_surfaces_;    // modelled tabletops this cycle (obstacle base anchoring)
    std::ofstream                    diag_csv_;            // per-cluster diagnostic log
    int                              diag_cycle_ = 0;

signals:
    void presenceReady();
    void presenceLost();
};

#endif // SPECIFICWORKER_H
