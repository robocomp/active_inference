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
    if (G_dsr)
        qInfo() << "[ltsm_agent] dsr  graph: domain" << configLoader.get<int>("Agent.dsr.domain")
                << "nodes" << static_cast<int>(G_dsr->size());
    qInfo() << "[ltsm_agent] ltsm graph: domain" << configLoader.get<int>("Agent.ltsm.domain")
            << "nodes" << static_cast<int>(G_ltsm->size())
            << "root id" << G_ltsm->get_node("root")->id();

    probe_enabled = configLoader.exists("SelfTest.probe")
                    and configLoader.get<bool>("SelfTest.probe");
    if (probe_enabled)
        qInfo() << "[ltsm_agent] SelfTest.probe ON -- writing a node per cycle into the MEMORY "
                   "graph and checking both directions for leaks.";

    // ── Eviction ────────────────────────────────────────────────────────────────────────────
    // Absent key ⇒ OFF. Moving a room into memory also moves the `current` edge, which is what
    // releases every other agent from the old room, so this must never arm itself by default.
    const auto flag = [this](const char *key)
    { return configLoader.exists(key) and configLoader.get<bool>(key); };
    eviction_enabled     = flag("Eviction.enabled");
    eviction_stage_check = flag("Eviction.stage_check");
    eviction_backstop    = flag("Eviction.backstop_sweep");

    // Persistence. Absent key ⇒ OFF, like everything else here. The path resolves against the CWD,
    // as every path in this component's config does.
    {
        const std::string path = configLoader.exists("Memory.persist_path")
                               ? configLoader.get<std::string>("Memory.persist_path")
                               : std::string("etc/ltsm_memory.json");
        store = std::make_unique<ltsm::MemoryStore>(G_ltsm, path, flag("Memory.persist"));
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
    if (eviction_enabled and G_dsr and G_ltsm)
    {
        evictor = std::make_unique<ltsm::RoomEviction>(G_dsr, G_ltsm);
        passages = std::make_unique<ltsm::PassageHarvest>(
            G_dsr, G_ltsm,
            configLoader.exists("Passages.csv_path") ? configLoader.get<std::string>("Passages.csv_path")
                                                     : std::string("etc/passages.csv"));
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
    if (evictor) step_eviction();

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

    if (eviction_stage_check) check_eviction(out, cand.value());

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
    if (eviction_backstop)
        if (const int reaped = evictor->sweep_orphans(cand->old_room_id); reaped > 0)
            qWarning() << "[eviction] backstop reaped" << reaped
                       << "node(s) nobody was alive to let go";
    return true;
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



