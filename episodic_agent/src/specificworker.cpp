/*
 *    Copyright (C) 2026 by RoboComp CORTEX Team
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/** SpecificWorker — episodic_agent. See specificworker.h. */

#include "specificworker.h"
#include "episode_text.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)
#include "../../common/episodic/em_journal.h"
#include "../../common/episodic/em_query_dsr.h"
#include "../../common/window_geometry/window_geometry_saver.h"   // rc::ui::GeometrySaver (SHARED)

#include <QCoreApplication>
#include <QFontDatabase>
#include <QSettings>
#include <QVBoxLayout>

#include <dsr/gui/viewers/graph_viewer/graph_viewer.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>   // std::_Exit — crash-free terminal shutdown
#include <iostream>
#include <iterator>
#include <print>
#include <thread>

#include <dsr/api/dsr_api.h>

// ─── Constructor / Destructor ─────────────────────────────────────────────────

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check)
    : GenericWorker(configLoader, tprx), startup_check_flag(startup_check)
{
    if (startup_check_flag)
    {
        this->startup_check();
        return;
    }

    int period = 0;
    rc::cfg::Reader(configLoader, "episodic_agent").req("Period.Compute", period,
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
    std::print("episodic_agent: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;
    poll_timer_.stop();
    leave_ltsm();
    // EM's own stop: whatever is open closes `interrupted_by_em_shutdown` and goes to the archive AND to domain 4
    // (ruling R22) — both callers (destructor, terminal_shutdown) run this before any graph is torn down.
    if (builder_ and recorder_)
    {
        const auto closed = builder_->shutdown(now_ms());
        write(closed);
        std::println("[episodic] shutdown: {} open record(s) closed as interrupted_by_em_shutdown", closed.size());
    }
    save_window_settings();
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

// ─── Initialisation ──────────────────────────────────────────────────────────

void SpecificWorker::initialize()
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // the [episodic] lines reach a redirected log as they happen
    std::print("episodic_agent: initialize()\n");
    GenericWorker::initialize();
    // Window geometry: GenericWorker restores it (QSettings "RoboComp"/episodic_agent, queued singleShot) and saves it
    // on aboutToQuit; this also saves after every resize/move, so a run that does not end cleanly keeps it too.
    for (const auto& [name, window] : windows)
        if (window != nullptr) new rc::ui::GeometrySaver(window.get(), [this] { save_window_settings(); });
    QTimer::singleShot(0, this, [this]()     // queued AFTER GenericWorker's restore: reports what it restored
    {
        for (const auto& [name, window] : windows)
            if (window != nullptr)
            {
                const QRect g = window->geometry();
                std::println("[episodic] window '{}' geometry {}x{}+{}+{}", name, g.width(), g.height(), g.x(), g.y());
            }
    });

    wire_view_data();

    // Graphs by NAME, failing legibly (the flat single-instance config files its one graph under "").
    const auto pick = [this](const std::string& key) -> std::shared_ptr<DSR::DSRGraph>
    {
        const auto it = Graphs.find(key);
        return it == Graphs.end() ? nullptr : it->second;
    };
    G_em_ = pick("em");
    if (not G or G.get() != pick("dsr").get() or not G_em_)
    {
        std::string found;
        for (const auto& [name, _] : Graphs) found += "\"" + name + "\" ";
        qFatal("episodic_agent needs [Agent.dsr] (alphabetically first, so G aliases it) and [Agent.em]; config "
               "declares: %s", found.c_str());
    }
    if (not G_em_->get_node("root").has_value())
        qFatal("the episodic-memory graph has no root node -- is em_root.json reachable from the CWD?");
    if (pick("ltsm"))
        qFatal("episodic_agent: remove [Agent.ltsm] from the config -- ltsm memory is joined lazily ([EM] ltsm_domain)");

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
        .on_optional_peer_lost  = [this](const std::string& n, std::uint32_t i) { on_optional_peer_lost(n, i); },
        .on_optional_peer_ready = [this](const std::string& n, std::uint32_t i) { on_optional_peer_ready(n, i); },
    });
    presence_coordinator_.set_lifecycle_hooks({
        .on_waiting_enter = [this]()
        {
            const auto missing = presence_coordinator_.missing_required_names();
            if (missing.empty()) qInfo("[SM] -> Waiting");
            else
            {
                QString m;
                for (const auto& label : missing) m += " " + QString::fromStdString(label);
                qInfo() << "[SM] -> Waiting (missing:" << m.trimmed() << ")";
            }
        },
        .on_operating_enter = [this]() { qInfo("[SM] -> Operating"); },
        .on_operating_loop  = [this]() { compute(); },
        .on_degraded_enter  = [this]()
        {
            if (shutting_down_) return;
            // Debounce (CLAUDE.md): with no required peer this should never fire, but if [Presence] ever gains one,
            // a transient flap must not tear the agent down.
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

    // ── Episodic memory ────────────────────────────────────────────────────────────────────────────────
    rc::cfg::Reader r(configLoader, "episodic_agent");
    const std::string archive_dir = r.s("EM.archive_dir", "etc/episodes",
            "append-only archive, one JSONL file per local day (relative to the CWD: run from the component root)");
    const int poll_ms = r.i("EM.poll_ms", 250, "poll period of the live graph + the controller's journal (main thread)");
    ltsm_domain_ = r.i("EM.ltsm_domain", 2, "DDS domain of ltsm_agent's memory graph; joined read-only while ltsm_agent is present");

    facts_ = std::make_unique<rc::em::DsrFactSource>(G, nullptr);   // memory graph set by sync_ltsm
    em_graph_ = std::make_unique<rc::em::EmGraph>(G_em_);
    sink_.g = em_graph_.get();
    archive_ = std::make_unique<rc::em::Archive>(archive_dir);
    recorder_ = std::make_unique<rc::em::Recorder>(*archive_, sink_, rc::em::day_of(now_ms()));
    if (const int lost = recorder_->start(now_ms()); lost > 0)
        std::println("[episodic] {} record(s) left open by a previous run closed as lost (R21)", lost);
    builder_ = std::make_unique<rc::em::Builder>(*facts_, [this](std::uint64_t t0) { return recorder_->next_id(t0); });
    metrics_ = std::make_unique<rc::em::Metrics>(rc::em::Metrics::path_for(archive_dir, now_ms()), now_ms());
    if (not metrics_->ok()) qWarning() << "[episodic] could not open the metrics CSV under" << QString::fromStdString(archive_dir);
    std::println("[episodic] archive {}: {} past day(s) folded into {} stats key(s); today {} reseeded with {} record(s)",
                 archive_dir, std::ranges::count_if(archive_->days(), [&](const std::string& d) { return d < recorder_->today(); }),
                 recorder_->stats().size(), recorder_->today(), archive_->load_day(recorder_->today()).size());
    const rc::em::Store seeded = rc::em::load_store(*G_em_);     // read back what domain 4 now holds
    std::println("[episodic] domain 4 holds {} record node(s) and {} stats node(s)", seeded.today.size(), seeded.stats.size());

    // NO update_node/update_edge signal connects (CLAUDE.md): a main-thread poll.
    connect(&poll_timer_, &QTimer::timeout, this, &SpecificWorker::poll);
    poll_timer_.start(poll_ms);

    // Every config key this agent reads goes through rc::cfg::Reader (check_registry_complete.sh episodic_agent).
    rc::cfg::exempt_generated_prefixes();
    rc::cfg::registry().declare_complete("episodic_agent");
    rc::cfg::Reader(configLoader, "episodic_agent").publish("etc/config_effective.csv");
}

// ─── The poll ────────────────────────────────────────────────────────────────

std::uint64_t SpecificWorker::now_ms()
{
    using namespace std::chrono;
    return std::uint64_t(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
}

void SpecificWorker::write(const std::vector<rc::em::Record>& out)
{
    if (out.empty()) return;
    recorder_->write(out);
    if (metrics_) metrics_->on_written(out);
}

void SpecificWorker::poll()
{
    if (shutting_down_ or not builder_) return;
    const auto t_start = std::chrono::steady_clock::now();
    const std::uint64_t now = now_ms();
    std::vector<rc::em::Record> out;
    const auto take = [&](std::vector<rc::em::Record> v) { std::ranges::move(v, std::back_inserter(out)); };

    sync_ltsm();   // ltsm_agent gone ⇒ drop its memory graph before any fact is read from it; back ⇒ join (~1 s)

    // Structural events are kept only if the journal node had been seen BEFORE this poll: until the builder has
    // primed on a journal nothing is open to carry them, and the initial graph sync — which can land in the same poll
    // as the journal node — must not be reported as a burst of births. Diffing every poll keeps the snapshot current.
    const bool controller_seen_before = controller_seen_;
    std::vector<rc::em::Event> events = diff_.poll(*G);

    if (const auto wire = rc::em::read_journal(*G); wire.has_value())
    {
        if (not controller_seen_) std::println("[episodic] controller journal node '{}' found", rc::em::kJournalNodeName);
        controller_seen_ = controller_present_ = true;
        if (*wire != last_wire_)
        {
            last_wire_ = *wire;
            if (metrics_) metrics_->on_wire(last_wire_.size());
            if (const auto w = rc::em::journal_from_wire(last_wire_); w.has_value())
            {
                take(builder_->on_journal(*w, now));
                bad_wire_logged_ = false;
            }
            else if (not bad_wire_logged_)
            {
                qWarning() << "[episodic] controller em_journal does not parse — ignored until it changes";
                bad_wire_logged_ = true;
            }
        }
    }
    else if (controller_present_)
    {
        controller_present_ = false;
        last_wire_.clear();
        auto lost = builder_->owner_lost(now);
        std::println("[episodic] controller journal node gone — {} open record(s) closed as lost", lost.size());
        take(std::move(lost));
    }
    else if (not controller_seen_ and not absent_logged_)
    {
        absent_logged_ = true;
        std::println("[episodic] controller absent — idling until its journal appears");
    }

    if (not controller_seen_before) events.clear();
    take(builder_->on_events(events, now));     // also re-checks the self mission (room changes)
    write(out);
    if (metrics_)
    {
        metrics_->on_poll(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count());
        metrics_->maybe_row(now, builder_->open());
    }
}

// ─── "View data" popup ───────────────────────────────────────────────────────

void SpecificWorker::wire_view_data()
{
    // cortex's GraphViewer emits view_data_signal(id, type) for "View data" on a node with no built-in widget — every
    // domain-4 node (robot_concept wires the same signal for its stream viewers). The em window only (Agent.em
    // graph = true); the live graph has no viewer here.
    const auto it = graph_viewers.find("em");
    if (it == graph_viewers.end() or not it->second) return;
    auto* gv = qobject_cast<DSR::GraphViewer*>(it->second->get_widget(DSR::DSRViewer::view::graph));
    if (not gv) return;
    QObject::connect(gv, &DSR::GraphViewer::view_data_signal, this,
                     [this](std::uint64_t id, const std::string&) { show_node_text(id); }, Qt::QueuedConnection);
}

void SpecificWorker::show_node_text(std::uint64_t id)
{
    if (shutting_down_ or not G_em_) return;
    const auto n = G_em_->get_node(id);
    if (not n) return;
    const std::string text = rc::em::node_text(rc::em::load_store(*G_em_), n->name());
    if (not text_popup_)
    {
        const auto w = windows.find("em");
        text_popup_ = new QDialog(w != windows.end() ? w->second.get() : nullptr);
        text_popup_->setAttribute(Qt::WA_DeleteOnClose);   // QPointer resets; the next click builds a fresh one
        auto* layout = new QVBoxLayout(text_popup_);
        text_view_ = new QPlainTextEdit(text_popup_);
        text_view_->setReadOnly(true);
        text_view_->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
        layout->addWidget(text_view_);
        QSettings settings(QStringLiteral("RoboComp"), QString::fromStdString(agent_name));
        if (not text_popup_->restoreGeometry(settings.value("episode_popup/geometry").toByteArray()))
            text_popup_->resize(760, 420);
        new rc::ui::GeometrySaver(text_popup_, [this]
        {
            if (not text_popup_) return;
            QSettings s(QStringLiteral("RoboComp"), QString::fromStdString(agent_name));
            s.setValue("episode_popup/geometry", text_popup_->saveGeometry());
        });
    }
    text_popup_->setWindowTitle(QString::fromStdString(n->name()));
    text_view_->setPlainText(QString::fromStdString(text));
    text_popup_->show();
    text_popup_->raise();
    text_popup_->activateWindow();
}

void SpecificWorker::GraphSink::upsert(const rc::em::Record& r)
{
    if (g and not g->upsert(r)) qWarning() << "[episodic] domain-4 upsert failed for" << QString::fromStdString(r.id);
}

void SpecificWorker::GraphSink::upsert(const rc::em::Stats& s)
{
    if (g and not g->upsert(s))
        qWarning() << "[episodic] domain-4 upsert failed for" << QString::fromStdString(rc::em::EmGraph::stats_name(s));
}

void SpecificWorker::GraphSink::remove(const std::string& id)
{
    if (g) g->remove(id);
}

void SpecificWorker::compute()
{
    // Everything runs on poll_timer_ (main thread); the GRAFCET loop only keeps the presence protocol.
}

void SpecificWorker::emergency() { std::print("episodic_agent: emergency()\n"); }
void SpecificWorker::restore()   { std::print("episodic_agent: restore()\n"); }
int SpecificWorker::startup_check()
{
    std::print("episodic_agent: startup_check()\n");
    return 0;
}
