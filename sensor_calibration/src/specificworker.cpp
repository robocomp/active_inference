#include "specificworker.h"

#include "../../common/agent_exit/terminal_exit.h"        // rc::agent::terminal_exit (SHARED)
#include "../../common/config_report/config_read.h"       // rc::cfg::Reader (SHARED)

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

    // Agent-presence state machine: Waiting -> Operating -> Degraded (as bottle_concept).
    int period = 0;
    rc::cfg::Reader(configLoader, "sensor_calibration").req("Period.Compute", period,
            "GRAFCET step period (ms) for every state of this agent's state machine");

    {
        const rc::cfg::Reader cfg(configLoader, "sensor_calibration");
        inputs_cfg_.camera_rate_hz   = cfg.d("Inputs.camera_rate_hz", inputs_cfg_.camera_rate_hz,
                "zed rgb/depth and ricoh hand-over rate (Hz): a COMPUTE budget, frames beyond it are drained unconverted; <=0 disables cameras");
        inputs_cfg_.stall_lidar_ms   = cfg.i("Inputs.stall_lidar_ms", inputs_cfg_.stall_lidar_ms, "LiDAR stream declared stalled after this silence (ms)");
        inputs_cfg_.stall_camera_ms  = cfg.i("Inputs.stall_camera_ms", inputs_cfg_.stall_camera_ms, "camera stream declared stalled after this silence (ms)");
        inputs_cfg_.stall_imu_ms     = cfg.i("Inputs.stall_imu_ms", inputs_cfg_.stall_imu_ms, "IMU stream declared stalled after this silence (ms)");
        inputs_cfg_.log_period_ms    = cfg.i("Inputs.log_period_ms", inputs_cfg_.log_period_ms, "stream-rate log period (ms)");
    }

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
    if (error.length() > 0)
    {
        qWarning() << error;
        throw error;
    }
}

SpecificWorker::~SpecificWorker()
{
    request_shutdown();
    std::print("sensor_calibration: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;
    inputs_.reset();   // subscribers + InnerEigenAPI go down first, from the main thread, while G is alive
    if (G)
        disconnect(G.get(), nullptr, this, nullptr);   // sever graph callbacks before any teardown
    cleanup_owned_nodes();
}

void SpecificWorker::terminal_shutdown()
{
    static std::atomic<bool> terminating{false};
    if (terminating.exchange(true))
        return;
    rc::agent::terminal_exit([this] { request_shutdown(); }, [this] { if (G) G->reset(); });
}

void SpecificWorker::initialize()
{
    std::print("sensor_calibration: initialize()\n");
    GenericWorker::initialize();   // creates the DSR graph and, with it, this agent's own node "<name> <id>"

    if (not G)
    {
        qWarning() << "sensor_calibration: DSR graph not available in initialize()";
        return;
    }

    // Nothing here reads pixels out of the graph (media plane only, later tasks).
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att, laser_X_att, laser_Y_att, laser_Z_att>();

    // The whole presence protocol is SHARED (common/concept_presence): debounced Degraded, one-shot post-sync
    // sweep on first Operating. This agent has NO primary input stream yet, so the input is "always live".
    presence_protocol_.wire(
        presence_coordinator_, &statemachine, this, configLoader, G, static_cast<std::uint32_t>(agent_id),
        {
            .shutting_down      = [this] { return shutting_down_.load(); },
            .pump_primary_input = {},
            .primary_age_ms     = [] { return std::int64_t{-1}; },
            .primary_live       = [] { return true; },
            .primary_stalled    = [](std::int64_t*) { return false; },
            .emit_ready         = [this] { emit presenceReady(); },
            .emit_lost          = [this] { emit presenceLost(); },
            .compute            = [this] { compute(); },
            .terminal_shutdown  = [this] { terminal_shutdown(); },
            .on_first_operating = [this] { remove_stale_nodes(); },
            .on_optional_peer_lost  = [](const std::string& name, std::uint32_t)
                                      { qInfo() << "[Presence] optional peer lost:" << QString::fromStdString(name); },
            .on_optional_peer_ready = [](const std::string& name, std::uint32_t)
                                      { qInfo() << "[Presence] optional peer ready:" << QString::fromStdString(name); },
        });

    // SIGTERM / Ctrl+C (sigwatch -> quit -> aboutToQuit) goes through the crash-free terminal shutdown.
    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::terminal_shutdown, Qt::UniqueConnection);

    remove_stale_nodes();
    std::print("sensor_calibration: initialize() done, agent node '{} {}' present\n", agent_name, agent_id);
}

void SpecificWorker::compute()
{
    if (shutting_down_.load() or not G)
        return;
    // Created here (an Operating tick, main thread), never at t=0 from initialize(): the graph is loaded and the
    // sensor nodes have synced by now (CLAUDE.md, media plane).
    if (not inputs_)
        inputs_ = std::make_unique<sc::SensorInputs>(G, inputs_cfg_);
    inputs_->poll();
    // Task 2.3 (trajectory harvester) takes inputs_->take_*() here; the queues are bounded until then.
    fps.print("sensor_calibration compute", 3000);
}

void SpecificWorker::emergency()
{
    fps.print("Emergency worker", 3000);
}

void SpecificWorker::restore()
{
    std::print("sensor_calibration: restore()\n");
}

int SpecificWorker::startup_check()
{
    std::print("sensor_calibration: startup_check()\n");
    QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
    return 0;
}

// ─── Presence state-machine step hooks (delegated to the coordinator) ──────────
void SpecificWorker::waiting_enter()   { presence_coordinator_.waiting_enter(); }
void SpecificWorker::waiting_loop()    { presence_coordinator_.waiting_loop(); }
void SpecificWorker::operating_enter() { presence_coordinator_.operating_enter(); }
void SpecificWorker::operating_loop()  { presence_coordinator_.operating_loop(); }
void SpecificWorker::degraded_enter()  { presence_coordinator_.degraded_enter(); }
void SpecificWorker::degraded_loop()   { presence_coordinator_.degraded_loop(); }

// The only node this agent owns today is its own agent node ("sensor_calibration 27"), which the DSR graph
// constructor (re)creates and cleanup_owned_nodes() deletes through [Owns]. This sweep reaps anything ELSE
// carrying the agent's name left by a crashed run (future proposal nodes). The live agent node is excluded.
void SpecificWorker::remove_stale_nodes()
{
    if (not G)
        return;
    std::vector<std::uint64_t> stale;
    for (const auto& node : G->get_nodes_by_type("metaconcept"))
        if (node.name().starts_with("sensor_calibration"))
            stale.push_back(node.id());
    for (const auto id : stale)
        G->delete_node(id);
    if (not stale.empty())
        std::print("sensor_calibration: swept {} stale node(s)\n", stale.size());
}

void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;
    remove_stale_nodes();                          // while the monitor is still up, before [Owns] deletion
    presence_coordinator_.cleanup_owned_nodes();   // stops the monitor FIRST, then deletes the [Owns] nodes
}
