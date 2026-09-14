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

#include <memory>

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
	/// The stage assertions: the door round-trips through the seam, and MEMORY's own RT chain
	/// reproduces the pose measured on the live graph. qFatal on failure -- a broken transaction
	/// must not pass quietly.
	void check_eviction(const ltsm::Outcome &out, const ltsm::Candidate &cand);

signals:
	//void customSignal();
};

#endif
