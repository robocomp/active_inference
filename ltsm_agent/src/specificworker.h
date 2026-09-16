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

/**
	\brief
	@author authorname
*/

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

// If you want to reduce the period automatically due to lack of use, you must uncomment the following line
//#define HIBERNATION_ENABLED

#include <genericworker.h>

#include "room_eviction.h"
#include "memory_store.h"
#include "passage_harvest.h"
#include "passage_live.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>

/**
 * \brief Class SpecificWorker implements the core functionality of the component.
 */
class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    /**
     * \brief Constructor for SpecificWorker.
     * \param configLoader Configuration loader for the component.
     * \param tprx Tuple of proxies required for the component.
     * \param startup_check Indicates whether to perform startup checks.
     */
	SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);

	/**
     * \brief Destructor for SpecificWorker.
     */
	~SpecificWorker();


public slots:

	/**
	 * \brief Initializes the worker one time.
	 */
	void initialize();

	/**
	 * \brief Main compute loop of the worker.
	 */
	void compute();

	/**
	 * \brief Handles the emergency state loop.
	 */
	void emergency();

	/**
	 * \brief Restores the component from an emergency state.
	 */
	void restore();

    /**
     * \brief Performs startup checks for the component.
     * \return An integer representing the result of the checks.
     */
	int startup_check();

	void modify_node_slot(std::uint64_t, const std::string &type){};
	void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
	void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type){};
	void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names){};
	void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
	void del_node_slot(std::uint64_t from){};     
private:

	/**
     * \brief Flag indicating whether startup checks are enabled.
     */
	bool startup_check_flag;

	// The two graphs this agent straddles, named so no later reader has to re-derive which one
	// G aliases. Both are entries of GenericWorker::Graphs, built from the [Agent.dsr] and
	// [Agent.ltsm] sub-tables of etc/config.toml, each on its own DDS domain.
	//   G_dsr  == G == Graphs.at("dsr")   domain 0, the live shared working graph (joined)
	//   G_ltsm ==      Graphs.at("ltsm")  domain 2, long-term spatial memory (seeded and owned)
	std::shared_ptr<DSR::DSRGraph> G_dsr;
	std::shared_ptr<DSR::DSRGraph> G_ltsm;

	// [SelfTest] probe -- off unless the config says otherwise. Writes a node per cycle into the
	// memory graph and asserts, both ways, that neither graph can see the other's contents.
	bool probe_enabled = false;
	int  probe_count   = 0;

	// ── Eviction (../EVICTION.md) ────────────────────────────────────────────────────────────
	// Moving a departed room into memory. DISARMED by default: it is armed on the offline stage
	// (etc/config_stage.toml, a synthetic live graph on domain 3) and stays off against the real
	// fleet until room_concept can hand over two rooms.
	// The object itself exists whenever the live graph does: the STARTUP SEED (rooms of the working
	// memory that memory does not hold yet) always runs through it, because seed and eviction must
	// share its room numbering. Only step_eviction() is gated on eviction_enabled.
	std::unique_ptr<ltsm::RoomEviction> evictor;
	bool eviction_enabled  = false;
	bool eviction_stage_check = false;   // assert the outcome instead of only printing it
	bool eviction_backstop = false;      // reap nodes whose owner was not running to let them go
	int  evictions_done    = 0;

	// ── Persistence (../EVICTION.md) ─────────────────────────────────────────────────────────
	// Long-term memory that restarts from zero is not memory. Written, wired and DEFAULT OFF:
	// arming it is [Memory] persist = true. Saves at the eviction fence and on a graceful exit.
	std::unique_ptr<ltsm::MemoryStore> store;

	// ── Passage statistic (../EVICTION.md) ───────────────────────────────────────────────────
	// Polls the live doors for new judgement episodes every cycle -- NOT only at the fence, because
	// most crossings change no room and would fire no eviction to harvest at. Rows always go to the
	// CSV; the Betas fold into the passage node when one exists to carry them.
	std::unique_ptr<ltsm::PassageHarvest> passages;

	/// One eviction attempt, once per compute cycle. Returns false when there was nothing to do.
	bool step_eviction();
	/// Everything after a successful evict(): log, passage fold, persistence save, backstop.
	void finish_eviction(const ltsm::Outcome &out, const ltsm::Candidate &cand);

	// ── Promotion (proto-room → current room) ─────────────────────────────────────────────────
	// ltsm decides room IDENTITY: a proto-room the robot's whole footprint is confidently past the crossed
	// aperture of becomes THE room -- proto edge removed, `current` moved, both rooms copied into memory
	// with the two doors matched through a memory passage, the old room deleted from the working memory.
	// [Promotion] enabled (absent ⇒ OFF), decision_prob = the one flagged decision level.
	bool   promotion_enabled = false;
	bool   promotion_stage_check = false;
	double promotion_decision_prob = 0.95;
	int    promotions_done = 0;
	int    promotion_cycles_ = 0;
	ltsm::Outcome   last_promotion_;          ///< for the stage assertions
	ltsm::Candidate last_promotion_cand_;
	Mat::Vector3d   door_in_new_before_{0, 0, 0};   ///< live: old door in the new room's frame, pre-promotion
	bool step_promotion();
	/// r_body = half the larger footprint side of the `body` node (width_m/depth_m) -- the same definition
	/// room_concept uses for its crossing test. nullopt when the graph does not say.
	std::optional<double> body_radius() const;
	void check_promotion(bool promoted_this_cycle);
	/// The stage assertions: the door round-trips through the seam, and MEMORY's own RT chain
	/// reproduces the pose measured on the live graph. qFatal on failure -- a broken transaction
	/// must not pass quietly.
	void check_eviction(const ltsm::Outcome &out, const ltsm::Candidate &cand);
	/// Stage assertion for the startup seed: every live room is in memory, and every door's pose in its
	/// room's frame composed through MEMORY's RT tree equals the one read off the live graph.
	void check_seed(const ltsm::SeedOutcome &so);

	// ── Graph-view ZOOM, persisted beside the generated geometry/state ─────────────────────────
	// The generated save/restore_window_settings() keep each window's size and dock layout; the
	// zoom of its graph view is not part of either blob, so it is stored here as `graph_zoom` in the
	// same QSettings group. Only a zoom the USER chose is kept -- see restore_graph_zoom().
	void restore_graph_zoom();
	void save_graph_zoom() const;
	bool eventFilter(QObject *watched, QEvent *event) override;
	std::unordered_map<std::string, bool> graph_user_framed_;      ///< graph name → zoom is the user's
	std::unordered_map<const QObject *, std::string> viewport_graph_;   ///< viewport → graph name

	// ── Memory-view relayout ──────────────────────────────────────────────────────────────────
	// twopi over the memory graph whenever a node is ADDED or DELETED (not on attribute churn).
	// Coalesced, so a seed or an eviction writing a dozen nodes lays out once.
	QTimer memory_layout_timer_;
	std::unordered_set<std::uint64_t> memory_known_ids_;
	void setup_memory_layout();

	// ── Live passage (src/passage_live.h) ─────────────────────────────────────────────────────
	// The proto-room's entry door matched to the current room's door by one live `passage_<n>`.
	// [LivePassage] enabled; absent ⇒ OFF, like everything that writes the shared graph here.
	std::unique_ptr<ltsm::PassageLive> live_passage;
	bool live_passage_stage_check = false;
	bool live_passage_swept_ = false;       ///< the post-sync stale sweep ran (first compute cycle)
	int  live_cycles_ = 0;
	/// Stage assertions for etc/config_stage_proto.toml. qFatal on failure.
	void check_live_passage(const ltsm::PassageLive::Step &st);

signals:
	//void customSignal();
};

#endif
