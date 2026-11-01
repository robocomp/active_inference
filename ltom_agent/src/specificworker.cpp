/** SpecificWorker -- ltom_agent. See specificworker.h. */

#include "specificworker.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)
#include "graph_sightings.h"
#include "repertory_io.h"

#include <QCoreApplication>
#include <QTimer>

#include <chrono>
#include <cstdio>
#include <cstdlib>   // std::_Exit -- crash-free terminal shutdown
#include <iostream>
#include <print>
#include <thread>

#include <dsr/api/dsr_api.h>

// --- Constructor / Destructor ---

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check)
    : GenericWorker(configLoader, tprx), startup_check_flag(startup_check)
{
    if (startup_check_flag)
    {
        this->startup_check();
        return;
    }

    int period = 0;
    rc::cfg::Reader(configLoader, "ltom_agent").req("Period.Compute", period,
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

    const auto err = statemachine.errorString();
    if (err.length() > 0)
    {
        qWarning() << err;
        throw err;
    }
}

SpecificWorker::~SpecificWorker()
{
    request_shutdown();
    std::print("ltom_agent: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;
    if (repertory_ and repertory_->dirty()) save_now("exit");   // persist BEFORE the node cleanup / graph reset (nothing changed: keep .prev a real older generation)
    cleanup_owned_nodes();     // our presence agent node on the live graph, through the shared coordinator ([Owns])
}

void SpecificWorker::terminal_shutdown()
{
    static std::atomic<bool> terminating{false};
    if (terminating.exchange(true))
        return;
    // Crash-free terminal exit (matches the fleet): cleanup, clean DDS teardown of every graph, then _Exit instead of
    // returning into the Ice communicator teardown + static destruction, whose cross-TU order aborts.
    request_shutdown();
    for (const auto& [name, g] : Graphs)
        if (g)
        {
            try { g->reset(); }
            catch (...) {}
        }
    std::cout.flush();
    std::cerr.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));   // let the del-deltas reach peers
    std::_Exit(EXIT_SUCCESS);
}

// --- Initialisation ---

void SpecificWorker::initialize()
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::print("ltom_agent: initialize()\n");
    GenericWorker::initialize();

    if (not G)
        qFatal("ltom_agent needs [Agent.dsr] (the live graph, domain 0)");
    if (Graphs.size() != 1)
        qFatal("ltom_agent joins ONE graph ([Agent.dsr]); the config declares %zu", Graphs.size());

    // Heavy payloads are never read here.
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att, laser_X_att, laser_Y_att, laser_Z_att>();

    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    presence_coordinator_.attach_state_machine(&statemachine);
    presence_coordinator_.set_transition_hooks({
        .request_presence_ready = [this]() { emit presenceReady(); },
        .request_presence_lost  = [this]() { emit presenceLost(); },
    });
    presence_coordinator_.set_peer_hooks({
        .on_peer_restarted = [](std::uint32_t id) { qInfo() << "[Presence] peer" << id << "restarted"; },
    });
    presence_coordinator_.set_lifecycle_hooks({
        .on_waiting_enter   = []() { qInfo("[SM] -> Waiting"); },
        .on_operating_enter = [this]() { operating_ = true; qInfo("[SM] -> Operating"); },
        .on_operating_loop  = [this]() { compute(); },
        .on_degraded_enter  = [this]()
        {
            operating_ = false;
            if (shutting_down_) return;
            // Debounce (CLAUDE.md): with no required peer this should never fire, but if [Presence] ever gains one,
            // a transient flap must not tear the agent down.
            qInfo("[SM] -> Degraded: required peer lost -- %d ms grace before shutdown", REQUIRED_LOSS_GRACE_MS);
            QTimer::singleShot(REQUIRED_LOSS_GRACE_MS, this, [this]()
            {
                if (shutting_down_) return;
                if (presence_coordinator_.all_required_ready())
                { qInfo("[SM] required peers recovered during grace -- staying alive"); return; }
                qWarning("[SM] required peer still missing after grace -- shutting down cleanly");
                terminal_shutdown();
            });
        },
    });
    presence_coordinator_.start();

    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::terminal_shutdown, Qt::UniqueConnection);

    // -- [LTOM] config. The poll period is Period.Compute (compute() runs once per Operating step) --
    const rc::cfg::Reader r(configLoader, "ltom_agent");
    params_.owner_grace_ms = r.i("LTOM.owner_grace_ms", 3000,
            "owner absence shorter than this is a flap; = presence heartbeat_timeout_ms");
    params_.save_period_s  = r.i("LTOM.save_period_s", 30, "repertory save period (s)");
    params_.repertory_path = r.s("LTOM.repertory_path", "etc/ltom_repertory.json", "repertory store of record");
    params_.csv_path       = r.s("LTOM.csv_path", "etc/ltom_repertory.csv", "repertory CSV for inspection");
    params_.classes        = r.v<std::string>("LTOM.classes", {"chair", "table", "bottle", "cabinet", "refrigerator"},
            "object classes in scope (name prefix <class>_ and an object_uid)");
    std::println("[ltom] config: owner grace {} ms, save {} s, {} class(es), repertory {}",
                 params_.owner_grace_ms, params_.save_period_s, params_.classes.size(),
                 params_.repertory_path);

    repertory_ = std::make_unique<ltom::Repertory>(params_.owner_grace_ms);
    const auto res = ltom::load_repertory(params_.repertory_path, *repertory_);
    repertory_->close_open_lives_as_gap();   // our own downtime is a gap, never a death
    save_blocked_ = res == ltom::LoadResult::blocked;
    if (save_blocked_)
        std::println(stderr, "⚠⚠ [ltom] a bad repertory file could not be moved aside: NOT SAVING until it is removed and the agent restarted");
    const char* how = res == ltom::LoadResult::loaded ? "loaded" : res == ltom::LoadResult::loaded_prev ? "loaded from .prev"
                    : res == ltom::LoadResult::missing ? "no file, starting empty"
                    : res == ltom::LoadResult::blocked ? "BAD FILE NOT MOVED ASIDE, saving disabled" : "UNREADABLE, starting empty";
    std::println("[ltom] repertory {}: {} ({} entries)", params_.repertory_path, how, repertory_->entries().size());
    inner_eigen_ = G->get_inner_eigen_api();   // once, on the main thread, after the graph is up

    rc::cfg::exempt_generated_prefixes();
    rc::cfg::registry().declare_complete("ltom_agent");
    rc::cfg::Reader(configLoader, "ltom_agent").publish("etc/config_effective.csv");
}

void SpecificWorker::save_now(const char* why)
{
    if (not repertory_ or save_blocked_) return;
    const bool ok = ltom::save_repertory(params_.repertory_path, *repertory_) and ltom::write_csv(params_.csv_path, *repertory_);
    if (ok) repertory_->clear_dirty();
    last_save_ = std::chrono::steady_clock::now();
    std::println("[ltom] saved ({}): {} entries{}", why, repertory_->entries().size(), ok ? "" : " -- WRITE FAILED");
}

void SpecificWorker::compute()
{
    if (not operating_ or shutting_down_ or not repertory_ or not inner_eigen_) return;

    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();   // same clock as timestamp_creation
    const auto view = ltom::read_graph(*G, *inner_eigen_, params_.classes);
    const auto ch = repertory_->observe(view.alive, view.owners_present, view.present, now_ms);

    const auto log = [this](const char* what, const std::vector<std::uint64_t>& uids)
    {
        for (const auto uid : uids)
        {
            const auto& e = repertory_->entries().at(uid);
            const auto room = e.keyframes.empty() ? std::int64_t{-1} : e.keyframes.back().room_key;
            std::println("[ltom] {} {} uid={} room {}", what, e.name, uid, room);
        }
    };
    log("born", ch.born);
    log("reopened", ch.reopened);
    log("died", ch.died);
    log("owner_exit", ch.owner_exited);

    if (repertory_->dirty() and std::chrono::steady_clock::now() - last_save_ >= std::chrono::seconds(params_.save_period_s))
        save_now("period");
}

void SpecificWorker::emergency() {}

void SpecificWorker::restore() {}

int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
    return 0;
}
