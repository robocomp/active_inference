/*
 *    Copyright (C) 2026 by YOUR NAME HERE
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
#include "specificworker.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)

#include <QAction>
#include <QSettings>
#include <QWheelEvent>
#include <dsr/gui/viewers/graph_viewer/graph_viewer.h>

namespace
{
DSR::GraphViewer *graph_view_of(const std::shared_ptr<DSR::DSRViewer> &viewer)
{
    return viewer ? qobject_cast<DSR::GraphViewer *>(viewer->get_widget(DSR::DSRViewer::view::graph))
                  : nullptr;
}
}   // namespace

#include <cmath>
#include <format>

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check) : GenericWorker(configLoader, tprx)
{
	this->startup_check_flag = startup_check;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
		#ifdef HIBERNATION_ENABLED
			hibernationChecker.start(500);
		#endif
		
		// Example statemachine:
		/***
		//Your definition for the statesmachine (if you dont want use a execute function, use nullptr)
		states["CustomState"] = std::make_unique<GRAFCETStep>("CustomState", period, 
															std::bind(&SpecificWorker::customLoop, this),  // Cyclic function
															std::bind(&SpecificWorker::customEnter, this), // On-enter function
															std::bind(&SpecificWorker::customExit, this)); // On-exit function

		//Add your definition of transitions (addTransition(originOfSignal, signal, dstState))
		states["CustomState"]->addTransition(states["CustomState"].get(), SIGNAL(entered()), states["OtherState"].get());
		states["Compute"]->addTransition(this, SIGNAL(customSignal()), states["CustomState"].get()); //Define your signal in the .h file under the "Signals" section.

		//Add your custom state
		statemachine.addState(states["CustomState"].get());
		***/

		statemachine.setChildMode(QState::ExclusiveStates);
		statemachine.start();

		auto error = statemachine.errorString();
		if (error.length() > 0){
			qWarning() << error;
			throw error;
		}
	}
}

SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;
	// Graceful exit (SIGTERM/SIGINT, never kill -9): let go of the live passages this agent owns.
	if (live_passage)
		if (const int n = live_passage->remove_owned(); n > 0)
			std::cout << "[passage] removed " << n << " owned live passage(s) on shutdown" << std::endl;
	save_graph_zoom();   // the views still exist here; GenericWorker's destructor saves the geometry
	// Last chance to write: a graceful stop (SIGTERM/SIGINT, never kill -9) runs this. A SIGKILL
	// does not, which is why the fence save above is the primary one and this is only a backstop.
	if (store) store->save("graceful shutdown");
	/*
	for (auto const& [name, g] : Graphs) {
	    g->write_to_json_file("./"+agent_name+"_"+name+".json");
	}
	*/
}


void SpecificWorker::initialize()
{
    // MUST come first: GenericWorker::initialize() is what builds one QMainWindow + DSRViewer per
    // entry of Graphs, so removing it leaves the agent running with no UI at all.
    GenericWorker::initialize();

    // Queued AFTER GenericWorker's own singleShot(0) restore of the window geometry, so the view is
    // re-zoomed at its final size.
    QTimer::singleShot(0, this, [this]() { restore_graph_zoom(); });
    if (auto *app = QCoreApplication::instance(); app != nullptr)
        connect(app, &QCoreApplication::aboutToQuit, this, [this]() { save_graph_zoom(); });

    // Bind the graphs BY NAME, and fail legibly if a name is missing. Graphs is keyed by the
    // [Agent.<name>] sub-table names, EXCEPT in the single-instance form (no sub-tables at all),
    // where genericworker.cpp takes the surNames.empty() branch and files the one graph under "".
    // Graphs.at("dsr") on such a config throws std::out_of_range from inside a Qt event handler,
    // which surfaces as an unreadable `error: unordered_map::at` and then a terminate. So look the
    // keys up non-throwingly and say what was actually found.
    const auto pick = [this](const std::string &key) -> std::shared_ptr<DSR::DSRGraph>
    {
        const auto it = Graphs.find(key);
        return it == Graphs.end() ? nullptr : it->second;
    };
    G_ltsm = pick("ltsm");
    G_dsr  = pick("dsr");

    if (not G_ltsm)
    {
        std::string found;
        for (const auto &[name, _] : Graphs) found += "\"" + name + "\" ";
        qFatal("no graph named \"ltsm\" -- config declares graphs: %s. This agent needs an "
               "[Agent.ltsm] sub-table; the flat single-instance form files its graph under \"\".",
               found.c_str());
    }

    // G_dsr absent is a legitimate MEMORY-ONLY run (etc/config_solo.toml): work on the memory graph
    // with no fleet up and no way to disturb one. Everything touching the live graph is then off.
    if (G_dsr)
    {
        // G already points at Graphs.at("dsr") because getSurNames() sorts (ConfigLoader.cpp:277)
        // and "dsr" < "ltsm" -- but that is a subtlety nobody should have to re-derive, so check it.
        if (G.get() != G_dsr.get())
            qFatal("G does not alias the \"dsr\" graph -- did the [Agent.*] sub-tables get renamed "
                   "so that \"dsr\" is no longer alphabetically first?");
    }
    else
        qInfo() << "[ltsm_agent] MEMORY-ONLY run: no \"dsr\" graph declared, the live working graph "
                   "is not attached.";

    // The memory graph is seeded from ltsm_root.json, so it must already hold its root. A DSR graph
    // without a node named "root" breaks get_node_root(), and with it the whole RT API.
    if (not G_ltsm->get_node("root").has_value())
        qFatal("LTSM graph has no \"root\" node -- check ltsm_root.json and that the agent was "
               "started from the component root (the path is resolved against the CWD).");

    // Acceptance evidence for this step: two graphs, two domains, one process.
    // Read through the registry rather than inline in the log line, so the two domains appear in the
    // startup table like every other key. They are the acceptance evidence for "two graphs, two
    // domains, one process" - exactly the kind of fact worth having on the record of a run.
    rc::cfg::Reader cfgr(configLoader, "ltsm_agent");
    int dsr_domain = 0, ltsm_domain = 0;
    cfgr.opt("Agent.dsr.domain", dsr_domain,
             "DDS domain of the SHARED live graph this agent reads rooms from");
    cfgr.opt("Agent.ltsm.domain", ltsm_domain,
             "DDS domain of this agent's OWN long-term memory graph - separate on purpose");
    if (G_dsr)
        qInfo() << "[ltsm_agent] dsr  graph: domain" << dsr_domain
                << "nodes" << static_cast<int>(G_dsr->size());
    qInfo() << "[ltsm_agent] ltsm graph: domain" << ltsm_domain
            << "nodes" << static_cast<int>(G_ltsm->size())
            << "root id" << G_ltsm->get_node("root")->id();

    probe_enabled = cfgr.b("SelfTest.probe", false,
            "write a node per cycle into the MEMORY graph and check both directions for leaks",
            rc::cfg::diagnostic);
    if (probe_enabled)
        qInfo() << "[ltsm_agent] SelfTest.probe ON -- writing a node per cycle into the MEMORY "
                   "graph and checking both directions for leaks.";

    // ── Eviction ────────────────────────────────────────────────────────────────────────────
    // Absent key ⇒ OFF. Moving a room into memory also moves the `current` edge, which is what
    // releases every other agent from the old room, so this must never arm itself by default.
    // ABSENT ⇒ OFF is this component's rule, and the Reader keeps it: the default passed here IS
    // false, so the table shows `default false` rather than leaving the reader to infer that an
    // absent key means off. That inference is what "absent is not off, but it reads as off" warns
    // about, and here it happens to be right - which is exactly why it should be stated, not assumed.
    const auto flag = [&](const char *key, std::string_view what)
    { return cfgr.b(key, false, what); };
    eviction_enabled     = flag("Eviction.enabled",
            "move a room out of the live graph into memory - also moves the `current` edge, which "
            "releases every other agent from the old room, so it must never arm itself by default");
    eviction_stage_check = flag("Eviction.stage_check",
            "verify each eviction stage before advancing to the next");
    eviction_backstop    = flag("Eviction.backstop_sweep",
            "sweep for rooms a missed eviction left behind");

    // Persistence. Absent key ⇒ OFF, like everything else here. The path resolves against the CWD,
    // as every path in this component's config does.
    {
        const std::string path = cfgr.s("Memory.persist_path", "etc/ltsm_memory.json",
                "where the long-term memory graph is written; resolved against the CWD");
        store = std::make_unique<ltsm::MemoryStore>(G_ltsm, path,
                flag("Memory.persist", "persist the memory graph across runs"));
        if (store->enabled())
        {
            qInfo() << "[ltsm_agent] PERSISTENCE ARMED --" << path.c_str()
                    << "(saved at the eviction fence and on a graceful exit)";
            store->load();
        }
        else
            qInfo() << "[ltsm_agent] persistence OFF -- memory is re-seeded from the .json every "
                       "launch and everything accumulated is lost on exit. [Memory] persist = true "
                       "arms it.";
    }
    // The memory view re-lays itself out (twopi) on every node birth/death, from here on -- so the seed
    // below is laid out too.
    setup_memory_layout();

    if (G_dsr)
        evictor = std::make_unique<ltsm::RoomEviction>(G_dsr, G_ltsm);

    promotion_enabled = G_dsr and flag("Promotion.enabled",
            "promote a proto-room to a real room once the evidence supports it");
    promotion_stage_check = flag("Promotion.stage_check",
            "verify each promotion stage before advancing");
    cfgr.opt("Promotion.decision_prob", promotion_decision_prob,
             "posterior probability at which a promotion is decided");
    if (promotion_enabled and promotion_stage_check)
    {
        // ★ FIXTURE MANIPULATION, domain 3 only: the pose covariance a JSON-seeded edge cannot carry
        // (see tools/make_proto_stage_fixture.py). σ = 5 cm in x/y, 0.02 rad in yaw, robot (PARENT) frame.
        const auto robot = G_dsr->get_node("Shadow");
        const auto proto = G_dsr->get_node("room_2");
        Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();
        cov(0, 0) = cov(1, 1) = 0.05 * 0.05;
        cov(5, 5) = 0.02 * 0.02;
        if (not robot.has_value() or not proto.has_value()
            or not G_dsr->get_rt_api()->insert_or_assign_edge_RT_covariance(
                   robot.value(), proto->id(), DSR::RT_API::CovarianceKind::Pose, cov))
            qFatal("[promotion][stage] could not write the fixture's pose covariance on Shadow->room_2");
    }
    if (promotion_enabled)
        qInfo() << "[ltsm_agent] PROMOTION ARMED -- a proto-room the robot is past the doorway of (P >="
                << promotion_decision_prob << ") becomes current; the old room goes to memory and is REMOVED"
                   " from the working memory";

    // ── Live passage ────────────────────────────────────────────────────────────────────────
    // Constructed with the LIVE graph only: it can never see (or sweep) memory's own passage_N.
    if (G_dsr and flag("LivePassage.enabled",
            "track door crossings on the LIVE graph only, so it can never sweep memory's own "
            "passage_N nodes"))
    {
        live_passage = std::make_unique<ltsm::PassageLive>(G_dsr);
        live_passage_stage_check = flag("LivePassage.stage_check",
                "verify each live-passage stage before advancing");
        const int stale = live_passage->sweep_stale();
        qInfo() << "[ltsm_agent] LIVE PASSAGE ARMED -- proto-room entry doors are matched to the current"
                   " room's door; swept" << stale << "stale live passage(s) from a previous run";
    }

    // ── Startup seed ────────────────────────────────────────────────────────────────────────
    // Every room in the working memory that long-term memory does not hold yet gets a room node in
    // memory, with its floor, walls and the doors that exist now. The live graph is fully synced by
    // the time we get here (the DSRGraph constructor blocks until the server has sent it), so "what
    // exists now" is the fleet's current belief, not a half-received graph. Read-only on the live
    // graph; it moves no `current` edge and releases nothing, so it is not gated behind eviction.
    if (evictor)
    {
        const auto so = evictor->seed_from_working_memory();
        std::string listing;
        for (const auto &[live, mem] : so.rooms) listing += std::format(" {}→{}", live, mem);
        qInfo().noquote() << QString::fromStdString(std::format(
            "[seed] working memory holds {} room(s): {} already remembered, {} written to memory "
            "({} nodes, {} door(s)).{}",
            so.rooms_seen, so.rooms_known, so.rooms_created, so.nodes, so.doors, listing));
        if (so.rooms_seen == 0)
            qInfo() << "[seed] no room in the working memory yet -- nothing to remember. (room_concept"
                       " not up, or its layout has not converged.)";
        if (so.rooms_created > 0 and store)
        {
            store->mark_dirty();
            store->save("startup seed");
        }
        if (eviction_stage_check) check_seed(so);
    }

    if (eviction_enabled and G_dsr and G_ltsm)
    {
        passages = std::make_unique<ltsm::PassageHarvest>(
            G_dsr, G_ltsm,
            cfgr.s("Passages.csv_path", "etc/passages.csv",
                   "where harvested door-crossing passages are logged", rc::cfg::diagnostic));
        qInfo() << "[ltsm_agent] EVICTION ARMED -- stage_check=" << eviction_stage_check
                << "backstop_sweep=" << eviction_backstop
                << "\n            (see EVICTION.md; the `current` edge this writes is what tells the"
                   " other agents to let the old room go)";
    }
    else if (eviction_enabled)
        qWarning() << "[ltsm_agent] Eviction.enabled is set but there are not two graphs -- ignored.";

    // No DSR signals are connected on purpose. CLAUDE.md: if you do not need a signal, do not
    // connect it -- and never with Qt::DirectConnection, which would run the slot on a FastDDS
    // reader thread and corrupt the heap.

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
    // ARMS the unread sweep; check_registry_complete.sh ltsm_agent is the grep that keeps the claim
    // honest. Re-run it whenever a config read is added.
    rc::cfg::exempt_generated_prefixes();
    rc::cfg::registry().declare_complete("ltsm_agent");
    rc::cfg::Reader(configLoader, "ltsm_agent").publish("etc/config_effective.csv");
}


void SpecificWorker::compute()
{
    // The room the robot has left is moved into memory here, once both rooms are hung from the
    // robot. Everything else in this function is the step-1 isolation probe.
    //
    // The passage poll comes FIRST and runs unconditionally: a judgement episode is recorded when
    // door_concept closes one, which mostly happens with no room change at all (the apartment is one
    // fitted room, so crossing its doors fires no eviction). Harvesting only at the fence would miss
    // nearly every crossing the robot makes.
    if (passages) passages->poll();
    if (evictor and eviction_enabled) step_eviction();

    if (live_passage)
    {
        // The initialize() sweep can run before a crashed run's leftovers finish syncing; sweep once
        // more on the first cycle (CLAUDE.md, startup stale-sweep).
        if (not std::exchange(live_passage_swept_, true))
            live_passage->sweep_stale();
        const auto st = live_passage->step();
        ++live_cycles_;
        if (live_passage_stage_check) check_live_passage(st);
    }

    // AFTER the live passage step: the promotion reads the door pair that step matched.
    if (promotion_enabled and evictor)
    {
        const bool promoted = step_promotion();
        ++promotion_cycles_;
        if (promotion_stage_check) check_promotion(promoted);
    }

    if (not probe_enabled)
        return;

    // Isolation self-test. Write into the MEMORY graph only, then assert neither graph can see
    // anything of the other's. If the two DSRGraphs ever shared a domain -- the exact bug in
    // robocomp-shadow/agents/agent_dual_dsr, where the ctor's 5th arg is all_same_host and NOT
    // domain_id, so both its graphs land on domain 0 -- these flags would go true.
    const std::string name = "ltsm_probe_" + std::to_string(++probe_count);
    auto node = DSR::Node::create<object_node_type>(name);
    G_ltsm->add_or_modify_attrib_local<pos_x_att>(node, 50.f * static_cast<float>(probe_count));
    G_ltsm->add_or_modify_attrib_local<pos_y_att>(node, 0.f);
    G_ltsm->insert_node(node);

    // Cross-checks need the live graph; on a memory-only run there is nothing to leak into.
    const bool leaked_to_dsr  = G_dsr and G_dsr->get_node(name).has_value();        // must stay false
    const bool leaked_to_ltsm = G_ltsm->get_node("Shadow").has_value();             // must stay false
    qInfo() << "[ltsm_agent probe]" << name.c_str()
            << " ltsm.size=" << static_cast<int>(G_ltsm->size())
            << " dsr.size="  << (G_dsr ? static_cast<int>(G_dsr->size()) : -1)
            << " leak(ltsm->dsr)=" << leaked_to_dsr
            << " leak(dsr->ltsm)=" << leaked_to_ltsm;
    if (leaked_to_dsr or leaked_to_ltsm)
        qFatal("DOMAIN ISOLATION BROKEN -- the two graphs can see each other.");
}


//////////////////////////////////////////////////////////////////////////////////////////////////
// Eviction — one attempt per cycle. See EVICTION.md for the transaction and its ordering.
//////////////////////////////////////////////////////////////////////////////////////////////////
bool SpecificWorker::step_eviction()
{
    // BOOTSTRAP. The let-go rule is "a `current` edge exists AND points at another room", so a
    // graph with no edge at all releases nothing. With a single room, place the first one.
    evictor->ensure_current_edge();

    const auto cand = evictor->detect();
    if (not cand.has_value())
        return false;

    qInfo() << "[eviction] two rooms under" << QString::fromStdString(cand->robot_name)
            << "-- leaving" << QString::fromStdString(cand->old_room_name)
            << "(born" << static_cast<qulonglong>(cand->old_birth_ms) << ") for"
            << QString::fromStdString(cand->new_room_name)
            << "(born" << static_cast<qulonglong>(cand->new_birth_ms) << ")";

    const auto out = evictor->evict(cand.value());
    if (not out.ok)
    {
        qWarning() << "[eviction] NOT DONE:" << QString::fromStdString(out.reason)
                   << "-- nothing was released; the fleet keeps working in"
                   << QString::fromStdString(cand->old_room_name);
        return false;
    }
    finish_eviction(out, cand.value());
    if (eviction_stage_check) check_eviction(out, cand.value());
    return true;
}

void SpecificWorker::finish_eviction(const ltsm::Outcome &out, const ltsm::Candidate &cand_ref)
{
    const auto *cand = &cand_ref;
    ++evictions_done;
    const Mat::Vector3d t = out.seam.translation();
    const double yaw = std::atan2(out.seam.rotation()(1, 0), out.seam.rotation()(0, 0));
    qInfo().noquote() << QString::fromStdString(std::format(
        "[eviction] DONE #{}: {} archived as memory node {}; seam T({}<-{}) = "
        "(x={:+.4f} y={:+.4f} yaw={:+.4f}); {} nodes / {} RT edges copied; door '{}' at "
        "({:+.4f},{:+.4f}) in {} and ({:+.4f},{:+.4f}) in {}; `current` now on {}",
        evictions_done, cand->old_room_name, out.mem_old_room_id, cand->new_room_name,
        cand->old_room_name, t.x(), t.y(), yaw, out.nodes_copied, out.edges_copied,
        out.door_name.empty() ? "<none>" : out.door_name,
        out.door_in_old.x(), out.door_in_old.y(), cand->old_room_name,
        out.door_in_new.x(), out.door_in_new.y(), cand->new_room_name,
        cand->new_room_name));

    // Fold whatever this door has accumulated into the passage node now that one exists to carry it.
    // Before this moment the episodes live only in the CSV, which is the durable record anyway.
    if (passages and not out.door_name.empty() and not out.mem_doorway.empty())
        if (const auto dw = G_ltsm->get_node(out.mem_doorway); dw.has_value())
            passages->fold_into(dw->id(), out.door_name);

    // THE FENCE IS THE SAVE POINT. The copy is verified and `current` has moved, so this is the one
    // instant the memory graph is provably consistent -- and it coincides with the only thing that
    // durably changes it. See MemoryStore for why this is not a change counter.
    if (store)
    {
        store->mark_dirty();
        store->save("eviction #" + std::to_string(evictions_done));
    }

    // The owners let their own nodes go now that their room is no longer current. The sweep is the
    // backstop for owners that are not running to do it -- never the normal path.
    if (eviction_backstop and not cand->remove_old)
        if (const int reaped = evictor->sweep_orphans(cand->old_room_id); reaped > 0)
            qWarning() << "[eviction] backstop reaped" << reaped
                       << "node(s) nobody was alive to let go";
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Promotion
//////////////////////////////////////////////////////////////////////////////////////////////////
std::optional<double> SpecificWorker::body_radius() const
{
    if (const auto body = G_dsr->get_node("body"); body.has_value())
    {
        const auto w = G_dsr->get_attrib_by_name<width_m_att>(body.value());
        const auto d = G_dsr->get_attrib_by_name<depth_m_att>(body.value());
        if (w.has_value() and d.has_value()) return 0.5 * std::max(w.value(), d.value());
    }
    return std::nullopt;
}

bool SpecificWorker::step_promotion()
{
    const auto r_body = body_radius();
    if (not r_body.has_value())
    {
        static bool warned = false;
        if (not std::exchange(warned, true))
            qWarning() << "[promotion] no `body` node with width_m/depth_m in the live graph -- cannot tell "
                          "whether the robot's footprint is past the doorway, so nothing is promoted";
        return false;
    }

    for (const auto &r : evictor->promotion_evidence(r_body.value()))
    {
        if (not r.valid)
        {
            qInfo().noquote() << QString::fromStdString(std::format(
                "[promotion] {}: no evidence -- {}", r.proto_name, r.why_not));
            continue;
        }
        qInfo().noquote() << QString::fromStdString(std::format(
            "[promotion] {}: s={:+.2f} m past the aperture (r_body={:.2f}, σ={:.3f}) → P(inside)={:.3f}"
            "  (promote at ≥ {:.2f})", r.proto_name, r.s, r.body_radius, r.sigma_s, r.p_inside,
            promotion_decision_prob));

        // ⚠ DECISION LEVEL (Promotion.decision_prob) -- the one threshold on this path, and flagged as such.
        // Promotion is an ACT on a belief: it deletes a room from the working memory. The belief itself is
        // continuous (Φ under the pose covariance, no gate); acting on it needs a level, exactly like
        // room_concept's ProtoRoom.BirthProb. Repeated cycles are NOT fused: they are the same pose belief
        // re-read, and adding them up would manufacture confidence from one measurement.
        if (r.p_inside < promotion_decision_prob) continue;

        auto cand = evictor->promotion_candidate(r);
        if (not cand.has_value())
        {
            qWarning() << "[promotion]" << QString::fromStdString(r.proto_name)
                       << "is ready but there is no `current` room to promote it from";
            continue;
        }
        if (live_passage)
            if (const auto pair = live_passage->pair_for(r.proto_id); pair.has_value())
                std::tie(cand->old_door_id, cand->new_door_id) = pair.value();

        // Measured on the LIVE graph before anything moves: the old door as the new room sees it. After the
        // transaction the old room is gone, so this is the only independent reference the stage has.
        if (cand->old_door_id != 0)
            if (const auto d = G_dsr->get_node(cand->old_door_id); d.has_value())
                if (const auto t = G_dsr->get_inner_eigen_api()->get_transformation_matrix(cand->new_room_name, d->name());
                    t.has_value())
                    door_in_new_before_ = t->translation();

        qInfo().noquote() << QString::fromStdString(std::format(
            "[promotion] PROMOTING {} (P(inside)={:.3f}) over {}; crossed door {} ↔ {}",
            cand->new_room_name, r.p_inside, cand->old_room_name,
            cand->old_door_id ? G_dsr->get_node(cand->old_door_id).transform([](const auto &n){ return n.name(); }).value_or("?") : "<nearest>",
            cand->new_door_id ? G_dsr->get_node(cand->new_door_id).transform([](const auto &n){ return n.name(); }).value_or("?") : "<none>"));

        const auto out = evictor->evict(cand.value());
        if (not out.ok)
        {
            qWarning() << "[promotion] NOT DONE:" << QString::fromStdString(out.reason);
            continue;
        }
        ++promotions_done;
        finish_eviction(out, cand.value());
        qInfo().noquote() << QString::fromStdString(std::format(
            "[promotion] DONE #{}: {} is now current; {} copied to memory and removed from the working memory "
            "({} live nodes); memory passage '{}' matches {} ↔ {}",
            promotions_done, cand->new_room_name, cand->old_room_name, out.live_removed,
            out.mem_doorway.empty() ? "<none>" : out.mem_doorway,
            out.mem_door_old.empty() ? "<none>" : out.mem_door_old,
            out.mem_door_new.empty() ? "<none>" : out.mem_door_new));
        last_promotion_ = out;
        last_promotion_cand_ = cand.value();
        return true;                                   // one per cycle
    }
    return false;
}

void SpecificWorker::check_eviction(const ltsm::Outcome &out, const ltsm::Candidate &cand)
{
    // NO HARD-CODED GEOMETRY. Both assertions compare two INDEPENDENT computations of the same
    // quantity, so they cannot agree by construction:
    //   1. the door's pose carried through the seam vs. the pose read straight off the live chain;
    //   2. MEMORY's own RT tree vs. the live graph -- this is the one that would catch a bad
    //      Euler packing in rt_attrs(), since only the memory side goes through it.
    const auto fail = [](const std::string &what) { qFatal("[eviction][stage] %s", what.c_str()); };

    if (not out.seam_ok) fail("the seam was never measured");

    if (not out.door_name.empty())
    {
        const Mat::Vector3d through_seam = out.seam * out.door_in_old;
        const double e = (through_seam - out.door_in_new).norm();
        qInfo().noquote() << QString::fromStdString(std::format(
            "[eviction][stage] door through the seam ({:+.6f},{:+.6f}) vs measured "
            "({:+.6f},{:+.6f})  err={:.2e} m",
            through_seam.x(), through_seam.y(), out.door_in_new.x(), out.door_in_new.y(), e));
        if (e > 1e-4) fail(std::format("the door does not round-trip through the seam: {:.4f} m", e));

        // ★ THE SAME DOOR, COMPOSED INSIDE MEMORY, THROUGH THE SEAM. Memory stores the door ONCE
        // (the room-0 face) and the stub for room 1 with the seam on its RT edge; asking memory
        // "where is that face in room 1's frame?" walks r<new>_room <- r<old>_room <- ... <- face
        // and must reproduce what the LIVE graph measured through an entirely different chain
        // (door <- wall <- floor <- room_0 <- robot -> room_1). Nothing is stored twice, so this
        // compares two computations rather than a value against its own copy -- and it is what
        // would catch a bad Euler packing in rt_attrs(), which only the memory side goes through.
        auto ie = G_ltsm->get_inner_eigen_api();
        const auto mem_room = G_ltsm->get_node(out.mem_new_room_id);
        if (not mem_room.has_value()) fail("memory lost the new room's stub");
        const auto in_mem = ie->get_transformation_matrix(mem_room->name(), out.mem_door_old);
        if (not in_mem.has_value())
            fail(std::format("memory has no RT chain {} <- {}", mem_room->name(), out.mem_door_old));
        const double me = (in_mem->translation() - out.door_in_new).norm();
        qInfo().noquote() << QString::fromStdString(std::format(
            "[eviction][stage] memory composes {} <- {} through the seam as ({:+.6f},{:+.6f})  err={:.2e} m",
            mem_room->name(), out.mem_door_old, in_mem->translation().x(),
            in_mem->translation().y(), me));
        if (me > 1e-4) fail(std::format("memory's own RT chain disagrees by {:.4f} m", me));

        // The abstract passage: frameless, one `match` edge to the face we have seen, one `exit`
        // edge to the room reached through it. The SECOND match edge arrives only when room 1 is
        // itself evicted carrying door_concept's own door node -- an independent measurement of the
        // same hole, which is the point of not fabricating it here.
        if (out.mem_doorway.empty()) fail("no passage node was created for the crossed door");
        const auto dw = G_ltsm->get_node(out.mem_doorway);
        if (not dw.has_value()) fail("the passage node is not in memory");
        if (dw->type() != "metaconcept") fail("the passage node has the wrong type");
        if (not DSR::DSRGraph::get_node_edges_by_type(dw.value(), "RT").empty())
            fail("the passage node has an RT edge -- it must stay OUT of the metric tree");
        const auto matches = DSR::DSRGraph::get_node_edges_by_type(dw.value(), "match");
        const auto exits   = DSR::DSRGraph::get_node_edges_by_type(dw.value(), "exit");
        if (matches.size() != 1) fail(std::format("passage has {} match edges, expected 1 (only one "
                                                  "side has been seen)", matches.size()));
        if (exits.size() != 1) fail("passage must have exactly one `exit` edge to the room reached");
        qInfo().noquote() << QString::fromStdString(std::format(
            "[eviction][stage] passage '{}' is frameless, matches '{}', exits to '{}'",
            out.mem_doorway, out.mem_door_old, mem_room->name()));
    }

    // The fence: `current` moved, and NOTHING was deleted -- the old room is still there, waiting
    // for its owners to let it go.
    const auto robot = G_dsr->get_node(cand.robot_id);
    if (not robot.has_value()) fail("the robot node vanished");
    bool on_new = false, on_old = false;
    for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(robot.value(), "current"))
    {
        on_new = on_new or e.to() == cand.new_room_id;
        on_old = on_old or e.to() == cand.old_room_id;
    }
    if (not on_new) fail("the `current` edge did not move to the new room");
    if (on_old)     fail("the old room is still marked current");
    if (not G_dsr->get_node(cand.old_room_id).has_value())
        fail("the old room was DELETED -- eviction must leave it for its owner to let go");

    // ── The passage chain, end to end ──────────────────────────────────────────────────────────
    // The fixture's door carries one judgement episode. By the folding rules it is an UNCLAIMED
    // crossing: logged, and its `passage_llr` folds, but the crossing itself must NOT -- the robot
    // only walks through doorways that are already passable, so folding those would make alpha climb
    // while beta never moves.
    if (passages and not out.mem_doorway.empty())
    {
        const auto dw = G_ltsm->get_node(out.mem_doorway);
        if (not dw.has_value()) fail("the passage node vanished before the fold");

        // llr = ln(10) from Beta(1,1) has an analytic answer; anything else means the moment-matched
        // mixture is wrong, and a wrong update here is invisible in the MEAN (which matches soft
        // counts exactly) -- only the variance would differ, which is why it is asserted numerically.
        const float pa = G_ltsm->get_attrib_by_name<passage_pass_alpha_att>(dw.value()).value_or(-1.f);
        const float pb = G_ltsm->get_attrib_by_name<passage_pass_beta_att>(dw.value()).value_or(-1.f);
        qInfo().noquote() << QString::fromStdString(std::format(
            "[eviction][stage] passage Beta(pass) = ({:.6f}, {:.6f})  expected (1.638298, 0.936170)",
            pa, pb));
        if (std::fabs(pa - 1.638298f) > 1e-3f or std::fabs(pb - 0.936170f) > 1e-3f)
            fail(std::format("the llr fold is wrong: Beta({:.6f},{:.6f})", pa, pb));

        // The unclaimed crossing must not have been folded as a claim: exactly one llr update means
        // the total mass is below 3 (Beta(1,1) plus one bounded observation), not 3 (a full count).
        if (pa + pb >= 3.f) fail("an unclaimed crossing was folded as a claimed positive");
        if (G_ltsm->get_attrib_by_name<passage_crossings_att>(dw.value()).value_or(0) != 1)
            fail("the crossing was not counted");

        // ★ I MUST NOT ERASE door_concept's DATA. update_node is a whole-node replace, so an agent
        // that harvests by rebuilding a node copy would silently wipe the producer's attributes.
        // This asserts MY half of that contract; the producer's re-fetch is theirs to test.
        const auto live_door = G_dsr->get_node(out.door_name);
        if (not live_door.has_value()) fail("the live door node vanished");
        if (not G_dsr->get_attrib_by_name<passage_seq_att>(live_door.value()).has_value()
            or not G_dsr->get_attrib_by_name<passage_llr_att>(live_door.value()).has_value())
            fail("harvesting ERASED the producer's passage_* attributes");

        const auto verdict = passages->audit(out.door_name);
        qInfo().noquote() << QString::fromStdString(verdict.text);
        if (verdict.result != ltsm::AuditVerdict::Result::Ok)
            fail("the inversion audit should read OK on a crossed row with a strong positive llr");

        // ── THE ASSISTED BRANCH, injected ──────────────────────────────────────────────────────
        // ★ FIXTURE MANIPULATION, AND ONLY HERE. On domain 3 there is no door_concept to write a
        // second episode, so the test writes one itself onto the synthetic door. Never do this
        // against a real graph: writing a producer's attributes is exactly the boundary the rest of
        // this agent keeps. It is worth the exception because this is the branch I got wrong first
        // time — crediting `pass` for a crossing that only happened because the door was opened.
        auto door = G_dsr->get_node(out.door_name);
        if (not door.has_value()) fail("the live door vanished before the assisted-branch check");
        G_dsr->add_or_modify_attrib_local<passage_seq_att>(door.value(), 2);
        G_dsr->add_or_modify_attrib_local<passage_crossed_att>(door.value(), true);
        G_dsr->add_or_modify_attrib_local<passage_outcome_att>(door.value(), std::string("Satisfied"));
        G_dsr->add_or_modify_attrib_local<passage_open_requested_att>(door.value(), true);
        G_dsr->add_or_modify_attrib_local<passage_open_answer_att>(door.value(), std::string("Delivered"));
        G_dsr->add_or_modify_attrib_local<passage_llr_att>(door.value(), 0.f);   // says nothing on its own
        G_dsr->update_node(door.value());

        passages->poll();
        passages->fold_into(dw->id(), out.door_name);

        const auto dw2 = G_ltsm->get_node(out.mem_doorway);
        if (not dw2.has_value()) fail("the passage node vanished after the assisted fold");
        const float oa = G_ltsm->get_attrib_by_name<passage_open_alpha_att>(dw2.value()).value_or(-1.f);
        const float ob = G_ltsm->get_attrib_by_name<passage_open_beta_att>(dw2.value()).value_or(-1.f);
        const float pa2 = G_ltsm->get_attrib_by_name<passage_pass_alpha_att>(dw2.value()).value_or(-1.f);
        const float pb2 = G_ltsm->get_attrib_by_name<passage_pass_beta_att>(dw2.value()).value_or(-1.f);
        qInfo().noquote() << QString::fromStdString(std::format(
            "[eviction][stage] after an ASSISTED Satisfied: Beta(open) = ({:.4f},{:.4f}), "
            "Beta(pass) = ({:.6f},{:.6f}) -- pass must be UNCHANGED", oa, ob, pa2, pb2));
        if (std::fabs(oa - 2.f) > 1e-3f or std::fabs(ob - 1.f) > 1e-3f)
            fail("an accepted open request did not credit the `open` Beta");
        if (std::fabs(pa2 - 1.638298f) > 1e-3f or std::fabs(pb2 - 0.936170f) > 1e-3f)
            fail("a crossing that needed an opening moved the `pass` Beta -- it must not: "
                 "'it must have been shut, we asked' rests on a prior-dominated estimate");
        if (G_ltsm->get_attrib_by_name<passage_crossings_att>(dw2.value()).value_or(0) != 2)
            fail("the second crossing was not counted");

        // ── -1 IS NOT 0, THROUGH THE WHOLE ROUND TRIP ──────────────────────────────────────────
        // door_concept's sentinels are NEGATIVE because 0 is a real reading: `open_prob == 0` is a
        // door believed shut, which is evidence, while "not measured" is an absence. The CSV is an
        // append-only record read back months later, so the distinction has to survive the write
        // AND the parse — a reader that turned -1 into 0 would silently convert every unobservable
        // door into a confidently-shut one.
        G_dsr->add_or_modify_attrib_local<passage_seq_att>(door.value(), 3);
        G_dsr->add_or_modify_attrib_local<passage_open_prob_att>(door.value(), -1.f);
        G_dsr->add_or_modify_attrib_local<passage_duration_s_att>(door.value(), -1.f);
        G_dsr->add_or_modify_attrib_local<passage_outcome_att>(door.value(), std::string(""));
        G_dsr->add_or_modify_attrib_local<passage_crossed_att>(door.value(), false);
        G_dsr->update_node(door.value());
        passages->poll();

        const auto sentinels = passages->last_row_for(out.door_name);
        if (not sentinels.has_value()) fail("the sentinel episode was not recorded");
        qInfo().noquote() << QString::fromStdString(std::format(
            "[eviction][stage] sentinel round trip: open_prob={:+.1f} duration_s={:+.1f} "
            "(seq {}) -- both must be -1, never 0",
            sentinels->open_prob, sentinels->duration_s, sentinels->seq));
        if (sentinels->seq != 3) fail("the sentinel episode did not round-trip its sequence");
        if (sentinels->open_prob >= 0.f or sentinels->duration_s >= 0.f)
            fail("a NOT-MEASURED sentinel came back as a reading: -1 collapsed to 0 somewhere "
                 "between the graph, the CSV and the parser");
    }

    qInfo() << "[eviction][stage] PASS -- memory holds" << static_cast<int>(G_ltsm->size())
            << "nodes, the live graph still holds the old room, `current` is on"
            << QString::fromStdString(cand.new_room_name);
}

void SpecificWorker::check_seed(const ltsm::SeedOutcome &so)
{
    // Two independent computations again: the live RT chain (door <- wall <- floor <- room) against
    // MEMORY's chain over the copied edges. Only memory's side goes through the copy, so a dropped
    // edge or a re-parenting slip shows up as a disagreement, not as a value checked against itself.
    const auto fail = [](const std::string &what) { qFatal("[seed][stage] %s", what.c_str()); };

    if (so.rooms_seen + so.rooms_proto != static_cast<int>(G_dsr->get_nodes_by_type("room").size()))
        fail("the seed did not see every live room");
    if (so.rooms_created != so.rooms_seen) fail("the stage's memory starts empty, so every room must be written");

    // The live room a node belongs to: climb `parent` until a room. Through the robot every room
    // reaches every door, so reachability alone cannot say which room owns a door.
    const auto owning_room = [this](DSR::Node n) -> std::string
    {
        for (int hops = 0; hops < 32; ++hops)
        {
            if (n.type() == "room") return n.name();
            const auto p = G_dsr->get_attrib_by_name<parent_att>(n);
            if (not p.has_value()) return {};
            const auto up = G_dsr->get_node(p.value());
            if (not up.has_value()) return {};
            n = up.value();
        }
        return {};
    };

    auto live_ie = G_dsr->get_inner_eigen_api();
    auto mem_ie  = G_ltsm->get_inner_eigen_api();
    int doors_checked = 0;
    for (const auto &[live_room, mem_room] : so.rooms)
    {
        if (not G_ltsm->get_node(mem_room).has_value())
            fail(std::format("{} is not in memory as {}", live_room, mem_room));
        const std::string prefix = mem_room.substr(0, mem_room.size() - live_room.size());

        for (const auto &obj : G_dsr->get_nodes_by_type("object"))
        {
            if (not obj.name().starts_with("door") or owning_room(obj) != live_room) continue;
            const auto live_t = live_ie->get_transformation_matrix(live_room, obj.name());
            const auto mem_t  = mem_ie->get_transformation_matrix(mem_room, prefix + obj.name());
            if (not live_t.has_value()) fail(std::format("live graph has no chain {} <- {}", live_room, obj.name()));
            if (not mem_t.has_value())
                fail(std::format("door {} was not seeded into {} (or has no RT chain there)", obj.name(), mem_room));
            const double e = (mem_t->translation() - live_t->translation()).norm();
            qInfo().noquote() << QString::fromStdString(std::format(
                "[seed][stage] {}{} in {}: memory ({:+.6f},{:+.6f}) vs live ({:+.6f},{:+.6f})  err={:.2e} m",
                prefix, obj.name(), mem_room, mem_t->translation().x(), mem_t->translation().y(),
                live_t->translation().x(), live_t->translation().y(), e));
            if (e > 1e-4) fail(std::format("seeded door pose disagrees by {:.4f} m", e));
            ++doors_checked;
        }
    }
    if (doors_checked == 0) fail("the fixture has a door and none was checked");

    // EVERY WALL comes over, and none arrives folded: doors hang from walls, so a missing or collapsed
    // wall draws the door detached from its room in memory's view (seen live 2026-09-14).
    for (const auto &[live_room, mem_room] : so.rooms)
    {
        const std::string prefix = mem_room.substr(0, mem_room.size() - live_room.size());
        for (const auto &w : G_dsr->get_nodes_by_type("wall"))
            if (owning_room(w) == live_room and not G_ltsm->get_node(prefix + w.name()).has_value())
                fail(std::format("wall {} of {} is missing from memory", w.name(), live_room));
    }
    for (const auto &n : G_ltsm->get_nodes())
        if (G_ltsm->get_attrib_by_name<collapsed_att>(n).has_value())
            fail(std::format("{} arrived in memory with the viewer's `collapsed` flag", n.name()));
    if (doors_checked != so.doors) fail(std::format("{} door(s) seeded but {} checked", so.doors, doors_checked));

    // Structure only: every `object` that came over must be a door -- no furniture.
    for (const auto &n : G_ltsm->get_nodes_by_type("object"))
        if (n.name().find("door") == std::string::npos)
            fail(std::format("the seed copied furniture ({}) -- it must copy structure only", n.name()));

    // Two rooms ⇒ the later one hangs off the earlier, its pose measured through the live tree.
    if (so.rooms.size() >= 2)
    {
        const auto &[a_live, a_mem] = so.rooms[0];
        const auto &[b_live, b_mem] = so.rooms[1];
        const auto live_t = live_ie->get_transformation_matrix(a_live, b_live);
        const auto mem_t  = mem_ie->get_transformation_matrix(a_mem, b_mem);
        if (not live_t.has_value() or not mem_t.has_value()) fail("no chain between the two seeded rooms");
        const double e = (mem_t->translation() - live_t->translation()).norm();
        qInfo().noquote() << QString::fromStdString(std::format(
            "[seed][stage] {} in {}: memory vs live err={:.2e} m", b_mem, a_mem, e));
        if (e > 1e-4) fail(std::format("the two seeded rooms are {:.4f} m off their live relation", e));
    }
    qInfo() << "[seed][stage] PASS --" << so.rooms_created << "room(s)," << doors_checked << "door(s)";
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Live passage stage check (etc/config_stage_proto.toml, fixture etc/stage_proto_room.json)
//////////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::check_live_passage(const ltsm::PassageLive::Step &st)
{
    const auto fail = [](const std::string &what) { qFatal("[passage][stage] %s", what.c_str()); };

    // Cycle 1 creates; cycle 2 must be a no-op (idempotent); cycle 3 removes the mirror and checks the
    // passage goes with it. The fixture is the only graph this manipulation is allowed on (domain 3).
    const auto passages = live_passage->owned();
    const auto current_targets = [this]()
    {
        std::vector<std::string> out;
        for (const auto &e : G_dsr->get_edges_by_type("current"))
            if (const auto n = G_dsr->get_node(e.to()); n.has_value()) out.push_back(n->name());
        return out;
    };

    if (live_cycles_ <= 2)
    {
        if (passages.size() != 1) fail(std::format("cycle {}: {} live passages, expected 1", live_cycles_, passages.size()));
        const auto matches = DSR::DSRGraph::get_node_edges_by_type(passages.front(), "match");
        if (matches.size() != 2) fail(std::format("the passage has {} match edges, expected 2", matches.size()));
        std::vector<std::string> ends;
        for (const auto &e : matches)
            if (const auto n = G_dsr->get_node(e.to()); n.has_value()) ends.push_back(n->name());
        std::ranges::sort(ends);
        // door_1 is the apartment door at the crossing; door_3 is the far distractor the argmin must reject.
        if (ends != std::vector<std::string>{"door_1", "door_2"})
            fail(std::format("the passage matches {} ↔ {}, expected door_1 ↔ door_2",
                             ends.size() > 0 ? ends[0] : "?", ends.size() > 1 ? ends[1] : "?"));
        if (not DSR::DSRGraph::get_node_edges_by_type(passages.front(), "RT").empty())
            fail("the live passage has an RT edge -- it must stay out of the metric tree");
        if (live_cycles_ == 2 and (st.created != 0 or st.rematched != 0 or st.deleted != 0))
            fail("the second cycle changed the passage -- the step is not idempotent");

        if (current_targets() != std::vector<std::string>{"room"})
            fail("`current` moved off the apartment while the new room is proto");
        if (evictions_done != 0) fail("a proto-room triggered an eviction");
        for (const auto &n : G_ltsm->get_nodes())
            if (n.name().find("room_2") != std::string::npos)
                fail(std::format("the proto-room reached long-term memory as {}", n.name()));
        if (not G_ltsm->get_node("r0_room").has_value()) fail("the apartment was not seeded into memory");
        qInfo() << "[passage][stage] cycle" << live_cycles_ << "OK -- one passage, door_1 <-> door_2,"
                << "`current` on the apartment, no eviction";

        if (live_cycles_ == 2)
        {
            // ★ FIXTURE MANIPULATION, domain 3 only: the mirror disappears (door_concept retired it).
            const auto mirror = G_dsr->get_node("door_2");
            if (not mirror.has_value() or not G_dsr->delete_node(mirror->id()))
                fail("could not remove the mirror door from the fixture");
        }
        return;
    }
    if (live_cycles_ == 3)
    {
        if (st.deleted != 1 or not passages.empty())
            fail(std::format("the mirror is gone but {} passage(s) remain (deleted={})", passages.size(), st.deleted));
        qInfo() << "[passage][stage] PASS -- the passage died with its door; memory holds"
                << static_cast<int>(G_ltsm->size()) << "nodes and no proto-room";
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Promotion stage check (etc/config_stage_promote.toml, fixture etc/stage_promote.json)
//////////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::check_promotion(bool promoted_this_cycle)
{
    const auto fail = [](const std::string &what) { qFatal("[promotion][stage] %s", what.c_str()); };

    if (promotion_cycles_ == 1)
    {
        if (not promoted_this_cycle or promotions_done != 1)
            fail("the robot is 1.2 m past the doorway with σ 5 cm and nothing was promoted");
        const auto &out = last_promotion_;

        // ── LIVE ─────────────────────────────────────────────────────────────────────────────────
        const auto r2 = G_dsr->get_node("room_2");
        if (not r2.has_value()) fail("the promoted room vanished");
        if (G_dsr->get_edge(r2->id(), r2->id(), "proto").has_value()) fail("room_2 is still proto");
        const auto cur = G_dsr->get_edges_by_type("current");
        if (cur.size() != 1 or cur.front().to() != r2->id()) fail("`current` is not (only) on room_2");
        for (const char *gone : {"room", "floor", "wall_0", "wall_1", "wall_2", "wall_3", "door_1", "door_3"})
            if (G_dsr->get_node(gone).has_value())
                fail(std::format("{} is still in the working memory", gone));
        if (not G_dsr->get_node("Shadow").has_value() or not G_dsr->get_node("body").has_value())
            fail("removing the old room took the robot with it");

        // ── MEMORY ───────────────────────────────────────────────────────────────────────────────
        const auto mr = G_ltsm->get_node("r0_room");
        const auto mn = G_ltsm->get_node("r2_room_2");
        if (not mr.has_value() or not mn.has_value()) fail("memory lacks r0_room or r2_room_2");
        if (not G_ltsm->get_edge(mr->id(), mn->id(), "RT").has_value()) fail("r2_room_2 does not hang from r0_room");
        if (not G_ltsm->get_node("r2_door_2").has_value()) fail("the new room's door (entry mirror) is not in memory");

        // Every wall hangs from the floor -- wall_2 hangs from the ROOM in the fixture, on purpose.
        const auto mfloor = G_ltsm->get_node("r0_floor");
        if (not mfloor.has_value()) fail("r0_floor missing");
        for (int i = 0; i < 4; ++i)
        {
            const auto w = G_ltsm->get_node(std::format("r0_wall_{}", i));
            if (not w.has_value()) fail(std::format("r0_wall_{} missing", i));
            if (G_ltsm->get_attrib_by_name<parent_att>(w.value()).value_or(0) != mfloor->id()
                or not G_ltsm->get_edge(mfloor->id(), w->id(), "RT").has_value()
                or G_ltsm->get_edge(mr->id(), w->id(), "RT").has_value())
                fail(std::format("r0_wall_{} does not hang (only) from r0_floor", i));
        }

        // The passage: two faces of one hole, frameless, exiting to the new room.
        const auto pass = G_ltsm->get_node(out.mem_doorway);
        if (not pass.has_value()) fail("no memory passage");
        std::vector<std::string> faces;
        for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(pass.value(), "match"))
            if (const auto n = G_ltsm->get_node(e.to()); n.has_value()) faces.push_back(n->name());
        std::ranges::sort(faces);
        if (faces != std::vector<std::string>{"r0_door_1", "r2_door_2"})
            fail(std::format("memory passage matches {} faces, expected r0_door_1 and r2_door_2", faces.size()));
        if (not DSR::DSRGraph::get_node_edges_by_type(pass.value(), "RT").empty()) fail("memory passage has an RT edge");

        // GEOMETRY, two independent computations each:
        //  (a) r0_door_1 composed in memory into r2_room_2's frame vs the LIVE reading taken before promotion;
        //  (b) the mirror r2_door_2 composed into r0_room's frame lands 4 cm from door_1 -- the fixture's offset.
        auto ie = G_ltsm->get_inner_eigen_api();
        const auto a = ie->get_transformation_matrix("r2_room_2", "r0_door_1");
        const auto b = ie->get_transformation_matrix("r0_room", "r2_door_2");
        const auto d1 = ie->get_transformation_matrix("r0_room", "r0_door_1");
        if (not a.has_value() or not b.has_value() or not d1.has_value()) fail("memory cannot compose across the seam");
        const double ea = (a->translation() - door_in_new_before_).norm();
        const double eb = std::fabs((b->translation() - d1->translation()).head<2>().norm() - 0.04);
        qInfo().noquote() << QString::fromStdString(std::format(
            "[promotion][stage] r0_door_1 in r2_room_2: memory vs live err={:.2e} m; r2_door_2 vs r0_door_1 "
            "offset {:.4f} m (fixture 0.04)", ea, (b->translation() - d1->translation()).head<2>().norm()));
        if (ea > 1e-4) fail(std::format("memory's seam disagrees with the live graph by {:.4f} m", ea));
        if (eb > 1e-3) fail("the two faces of the door do not coincide through the seam");
        // (c) a wall that hung from the ROOM live keeps its pose after re-hanging under the floor.
        const auto w2 = ie->get_transformation_matrix("r0_room", "r0_wall_2");
        if (not w2.has_value() or (w2->translation() - Mat::Vector3d(0.0, 2.5, 0.0)).norm() > 1e-4)
            fail("r0_wall_2 moved when it was re-hung under the floor");
        qInfo() << "[promotion][stage] cycle 1 OK";
        return;
    }
    if (promotion_cycles_ == 2)
    {
        if (promoted_this_cycle or promotions_done != 1) fail("a second promotion happened");
        if (live_passage and not live_passage->owned().empty())
            fail("the live passage outlived the old room's door");
        qInfo() << "[promotion][stage] PASS -- room_2 is current, the apartment is in memory only, memory holds"
                << static_cast<int>(G_ltsm->size()) << "nodes";
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Memory-view relayout
//////////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::setup_memory_layout()
{
    const auto it = graph_viewers.find("ltsm");
    if (it == graph_viewers.end()) return;
    auto *gv = graph_view_of(it->second);
    if (gv == nullptr) return;

    for (const auto &n : G_ltsm->get_nodes()) memory_known_ids_.insert(n.id());

    // UI debounce, not a model threshold: a burst of inserts (the seed, an eviction) lays out once.
    static constexpr int kLayoutCoalesceMs = 100;
    memory_layout_timer_.setSingleShot(true);
    connect(&memory_layout_timer_, &QTimer::timeout, gv, [gv]() { gv->compute_layout("twopi"); });

    // QUEUED (CLAUDE.md): never run a slot on a DDS reader thread. update_node_signal also fires for
    // plain updates -- including the pos_x/pos_y that compute_layout itself writes back -- so only an
    // id not seen before counts as an ADDITION; that is also what stops the layout re-triggering itself.
    connect(G_ltsm.get(), &DSR::DSRGraph::update_node_signal, this,
            [this](std::uint64_t id, const std::string &, DSR::SignalInfo)
            {
                if (memory_known_ids_.insert(id).second) memory_layout_timer_.start(kLayoutCoalesceMs);
            },
            Qt::QueuedConnection);
    connect(G_ltsm.get(), &DSR::DSRGraph::del_node_signal, this,
            [this](std::uint64_t id, DSR::SignalInfo)
            {
                if (memory_known_ids_.erase(id) > 0) memory_layout_timer_.start(kLayoutCoalesceMs);
            },
            Qt::QueuedConnection);
    memory_layout_timer_.start(kLayoutCoalesceMs);   // the seeded root / a restored generation
}

void SpecificWorker::restore_graph_zoom()
{
    QSettings settings(QStringLiteral("RoboComp"), QString::fromStdString(agent_name));
    for (const auto &[name, viewer] : graph_viewers)
    {
        auto *gv = graph_view_of(viewer);
        if (gv == nullptr) continue;

        // Track whether the zoom on screen is the USER's. Only that one is worth restoring: the view
        // refits itself to the whole graph until the user zooms or pans, and saving an automatic fit
        // would freeze next run's view at a scale nobody chose. Wheel / left-drag hands the view to
        // the user (the same two gestures GraphViewer itself counts); its "Fit graph to view" menu
        // action hands it back.
        graph_user_framed_[name] = false;
        viewport_graph_[gv->viewport()] = name;
        gv->viewport()->installEventFilter(this);
        for (auto *action : gv->findChildren<QAction *>())
            if (action->text().contains(QStringLiteral("Fit graph")))
                connect(action, &QAction::triggered, this, [this, name]() { graph_user_framed_[name] = false; });

        settings.beginGroup(settings_group_name(name, agent_id));
        bool ok = false;
        const double zoom = settings.value(QStringLiteral("graph_zoom")).toDouble(&ok);
        settings.endGroup();
        if (not ok or zoom <= 0.) continue;      // never saved (or saved while auto-fitting)

        // ★ WHY A SYNTHETIC WHEEL EVENT. GraphViewer refits the viewport on every graph change unless
        // its PRIVATE `user_framed_` is set, and only a wheel event or a drag sets it
        // (cortex gui/viewers/graph_viewer/graph_viewer.cpp:97-122). A bare setTransform() would be
        // thrown away by the next refit, ~150 ms later. One wheel notch marks the view as user-framed
        // (and zooms by 10%, which the setTransform below overwrites). The clean fix is a public
        // "set user view" in cortex's GraphViewer; this avoids needing a cortex reinstall for it.
        const QPointF centre(gv->viewport()->rect().center());
        QWheelEvent notch(centre, gv->viewport()->mapToGlobal(centre), QPoint(), QPoint(0, 120),
                          Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
        QCoreApplication::sendEvent(gv->viewport(), &notch);

        // The layout is not the one the zoom was saved on (memory is rebuilt every launch), so centre
        // on the graph as it is now rather than on a stale scene point.
        const auto apply = [gv, zoom]()
        {
            gv->setTransform(QTransform::fromScale(zoom, zoom));
            gv->centerOn(gv->scene.itemsBoundingRect().center());
        };
        apply();
        // ★ AND ONCE MORE, after the refit that may already be IN FLIGHT. The latch stops new refits
        // from being scheduled, but a refit timer armed BEFORE it (the startup seed inserts nodes just
        // before this runs) still fires, and its timeout lambda refits without checking the latch
        // (cortex graph_viewer.cpp:52-56; only schedule_refit() checks it). Measured: 2.5 restored,
        // 1.033 on screen 500 ms later. The timer is single-shot at 150 ms (graph_viewer.cpp:92), so
        // re-applying after twice that lands after it. Fixing that lambda in cortex makes this redundant.
        static constexpr int kCortexRefitCoalesceMs = 150;
        QTimer::singleShot(2 * kCortexRefitCoalesceMs, gv, apply);
        graph_user_framed_[name] = true;
        qInfo() << "[ltsm_agent] restored graph zoom" << zoom << "on" << QString::fromStdString(name);
    }
}

void SpecificWorker::save_graph_zoom() const
{
    QSettings settings(QStringLiteral("RoboComp"), QString::fromStdString(agent_name));
    for (const auto &[name, viewer] : graph_viewers)
    {
        const auto *gv = graph_view_of(viewer);
        if (gv == nullptr) continue;
        settings.beginGroup(settings_group_name(name, agent_id));
        const auto framed = graph_user_framed_.find(name);
        if (framed != graph_user_framed_.end() and framed->second)
            settings.setValue(QStringLiteral("graph_zoom"), gv->transform().m11());
        else
            settings.remove(QStringLiteral("graph_zoom"));   // auto-fit: next run auto-fits too
        settings.endGroup();
    }
    settings.sync();
}

bool SpecificWorker::eventFilter(QObject *watched, QEvent *event)
{
    if (const auto it = viewport_graph_.find(watched); it != viewport_graph_.end())
    {
        const bool wheel = event->type() == QEvent::Wheel;
        const bool drag  = event->type() == QEvent::MouseMove
                           and (static_cast<QMouseEvent *>(event)->buttons() & Qt::LeftButton);
        if (wheel or drag) graph_user_framed_[it->second] = true;
    }
    return GenericWorker::eventFilter(watched, event);   // observe only, never consume
}

void SpecificWorker::emergency()
{
    fps.print("Emergency worker", 3000);
    //emergencyCODE
    //
    //if (SUCCESSFUL) //The componet is safe for continue
    //  emmit goToRestore()
}


//Execute one when exiting to emergencyState
void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
    //restoreCODE
    //Restore emergency component

}


int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
	return 0;
}



