/*
 * self_calibration — online calibration of the robot's head-sensor mounts (Helios/Bpearl LiDAR, ZED, Ricoh)
 * from raw media-plane streams, outside room_concept's localiser loop.
 *
 * THIS FILE IS THE SKELETON ONLY: the agent runs the presence protocol (mirrors bottle_concept), owns its own
 * agent node, and cleans it up on a graceful stop. Task 2.2 adds the input layer (sensor_inputs.h);
 * no estimation yet.
 */
#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <memory>
#include <cstdint>
#include <string>

#include <genericworker.h>

#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "../../common/concept_presence/concept_presence.h"   // rc::presence::ConceptProtocol (SHARED)
#include "../../common/sensor_calib/trajectory_harvester.h"
#include "dashboard_model.h"
class DashboardView;
#include "helios_mount_stage.h"                                    // Helios planar mount from registered keyframe pairs (Phase 4.4)
#include "lidar_mount_stage.h"                                     // floor + vertical LiDAR mount factors (Phase 3)
#include "sensor_inputs.h"                                         // the input layer (plain-data outputs)
#include "zed_floor_stage.h"                                       // ZED floor plane from the depth image (Phase 4.1)

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

    // Generated slots, intentionally empty: this agent connects NO DSR signal (it polls). If one is ever
    // connected it must be Queued, never Qt::DirectConnection (CLAUDE.md).
    void modify_node_slot(std::uint64_t, const std::string&){};
    void modify_node_attrs_slot(std::uint64_t, const std::vector<std::string>&){};
    void modify_edge_slot(std::uint64_t, std::uint64_t, const std::string&){};
    void modify_edge_attrs_slot(std::uint64_t, std::uint64_t, const std::string&, const std::vector<std::string>&){};
    void del_edge_slot(std::uint64_t, std::uint64_t, const std::string&){};
    void del_node_slot(std::uint64_t){};

private:
    // ── Presence protocol (hooks delegate to the coordinator) ───────────────────
    void waiting_enter();
    void waiting_loop();
    void operating_enter();
    void operating_loop();
    void degraded_enter();
    void degraded_loop();

    void remove_stale_nodes();        // sweep leftovers of a crashed previous run (initialize + first Operating)
    void cleanup_owned_nodes();
    void request_shutdown();
    void terminal_shutdown();         // request_shutdown() then a crash-free _Exit (SIGTERM / confirmed Degraded)

    bool startup_check_flag = false;
    bool owned_nodes_cleaned_ = false;
    std::atomic<bool> shutting_down_{false};

    sc::InputsConfig inputs_cfg_;                  // [Inputs] section of the config
    sc::calib::HarvesterConfig harvester_cfg_;     // [Harvester] section
    std::unique_ptr<sc::calib::TrajectoryHarvester> harvester_;
    long npairs_logged_ = 0;
    std::int64_t harvester_log_ms_ = 0;            // wall clock of the last summary log
    std::unique_ptr<sc::SensorInputs> inputs_;     // created lazily on the first Operating tick (main thread)

    // ── LiDAR mount factors (src/lidar_mount_stage.*), [MountFactors] section ──
    bool mount_factors_enabled_ = true;
    int  mount_log_period_ms_ = 10000;
    sc::LidarMountStageConfig mount_cfg_;
    std::unique_ptr<sc::LidarMountStage> mount_stage_;
    std::int64_t mount_log_ms_ = 0;
    void feed_mount_factors(const sc::LidarIn& l);

    // ── ZED floor plane (src/zed_floor_stage.*), [ZedFloor] section ──
    bool zed_floor_enabled_ = true;
    int  zed_floor_log_period_ms_ = 10000;
    sc::ZedFloorStage::Config zed_floor_cfg_;
    std::unique_ptr<sc::ZedFloorStage> zed_floor_stage_;
    std::int64_t zed_floor_log_ms_ = 0;
    void feed_zed_floor(const sc::ImageIn& im);

    // ── Helios planar mount (src/helios_mount_stage.*), [HeliosMount] section ──
    bool helios_enabled_ = true;
    int  helios_log_period_ms_ = 10000;
    sc::HeliosMountConfig helios_cfg_;
    std::unique_ptr<sc::HeliosMountStage> helios_stage_;
    std::int64_t helios_log_ms_ = 0;
    std::deque<std::int64_t> odo_stamps_;          // recent odometry source stamps (clock check only)
    bool helios_clock_logged_ = false;
    void note_helios_clock(std::int64_t keyframe_stamp_ms);

    // ── Dashboard (specificworker_dashboard.cpp). DISPLAY ONLY; all null/no-op when [Dashboard] show = false ──
    void build_dashboard();
    void feed_dashboard_keyframe(bool captured, bool stalled);
    void feed_dashboard_pairs(const std::vector<sc::calib::Pair>& pairs);
    void refresh_dashboard();                 // ≤ 5 Hz, from compute()
    void save_dashboard_geometry() const;
    bool dashboard_show_ = true;
    std::unique_ptr<sc::dash::DashboardModel> dash_model_;
    QWidget* dashboard_window_ = nullptr;     // owns the view (parentless top-level, deleted in the destructor)
    DashboardView* dash_view_ = nullptr;
    std::chrono::steady_clock::time_point dash_last_tp_{};
    sc::dash::StreamRates dash_rates_;
    long dash_resets_logged_ = 0;

    rc::presence::ConceptProtocol presence_protocol_;
    AgentPresenceCoordinator      presence_coordinator_;

signals:
    void presenceReady();
    void presenceLost();
};

#endif
