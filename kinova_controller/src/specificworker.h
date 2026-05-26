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

#include "kinematics.h"
#include "efe_gradient.h"

#include <Eigen/Dense>
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

	std::unique_ptr<Kinematics> kinematics_;

	// EFE-reach test target — arm-base frame, metres.
	// The arm base is mounted on the desk top, so arm-base-z = 0 IS the
	// table surface. (0.4, 0.0, 0.0) is 40 cm in front of the arm base,
	// centred laterally, at table level — i.e. tool_frame touches the
	// desk's top surface. In Webots world coords this maps to roughly
	// (-0.303 + 0.4, -0.023, 0.71) ≈ (+0.10, -0.02, +0.71).
	Eigen::Vector3d reach_target_{0.4, 0.0, 0.1};
	bool proxy_unreachable_warned_ = false;
	bool joint_dump_pending_ = true;     // dump first received TJoints, once

	// Controller lifecycle: home to a known rest pose before enabling EFE.
	// Same angles as the bridge's pre-homing — but the agent commands them
	// itself so behaviour does not depend on whether the bridge pre-homed.
	enum class Phase { SendingRestPose, Homing, ActiveEFE };
	Phase phase_ = Phase::SendingRestPose;
	std::array<double, Kinematics::N_ARM_JOINTS> rest_pose_angles_{
		0.0, 0.3, 0.0, 1.5, 0.0, 1.4, 0.0
	};
	int homing_settled_ticks_ = 0;
	static constexpr double HOMING_TOLERANCE_RAD = 0.05;  // ≈ 2.9°
	static constexpr int    HOMING_SETTLE_TICKS  = 5;     // consecutive cycles within tolerance

signals:
	//void customSignal();
};

#endif
