/*
 * sensor_calibration — online calibration of the robot's head-sensor mounts (Helios/Bpearl LiDAR, ZED, Ricoh)
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
#include "sensor_inputs.h"                                         // the input layer (plain-data outputs)

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
    std::chrono::steady_clock::time_point dash_last_tp_{}, dash_rate_tp_{};
    std::map<std::string, std::uint64_t> dash_rate_totals_;
    std::map<std::string, double> dash_rates_;

    rc::presence::ConceptProtocol presence_protocol_;
    AgentPresenceCoordinator      presence_coordinator_;

signals:
    void presenceReady();
    void presenceLost();
};

#endif
