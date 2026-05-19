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
#ifndef GENERICWORKER_H
#define GENERICWORKER_H

#include <stdint.h>
#include <grafcetStep/GRAFCETStep.h>
#include <ConfigLoader/ConfigLoader.h>
#include <QStateMachine>
#include <QEvent>
#include <QString>
#include <functional>
#include <atomic>
#include <QtCore>
#include <variant>
#include <unordered_map>

#include "dsr/api/dsr_api.h"
#include "dsr/gui/dsr_gui.h"

#include <memory>

#include <FullPoseEstimation.h>
#include <FullPoseEstimationPub.h>
#include <GenericBase.h>
#include <JoystickAdapter.h>
#include <Lidar3D.h>
#include <OmniRobot.h>

#define BASIC_PERIOD 100

using TuplePrx = std::tuple<RoboCompLidar3D::Lidar3DPrxPtr,RoboCompOmniRobot::OmniRobotPrxPtr>;


class GenericWorker : public QObject
{
Q_OBJECT
public:
	GenericWorker(const ConfigLoader& configLoader, TuplePrx tprx);
	virtual ~GenericWorker();
	virtual void killYourSelf();

	void setPeriod(const std::string& state, int period);
	int getPeriod(const std::string& state);

	QStateMachine statemachine;
	QTimer hibernationChecker;
	std::atomic_bool hibernation = false;


	RoboCompLidar3D::Lidar3DPrxPtr lidar3d_proxy;
	RoboCompOmniRobot::OmniRobotPrxPtr omnirobot_proxy;

	virtual void FullPoseEstimationPub_newFullPose (RoboCompFullPoseEstimation::FullPoseEuler pose) = 0;

	virtual void JoystickAdapter_sendData (RoboCompJoystickAdapter::TData data) = 0;


protected:
	std::unordered_map<std::string, std::unique_ptr<GRAFCETStep>> states;
	ConfigLoader configLoader;
	//DSR params
	std::string agent_name;
	int agent_id;

	// DSR graph
	std::unordered_map<std::string, std::shared_ptr<DSR::DSRGraph>> Graphs;
	std::shared_ptr<DSR::DSRGraph> G;
	// DSR graph viewer
	std::unordered_map<std::string, std::shared_ptr<DSR::DSRViewer>> graph_viewers;
	std::unordered_map<std::string, std::unique_ptr<QMainWindow>> windows;
	std::shared_ptr<DSR::DSRViewer> setupViewer(std::shared_ptr<DSR::DSRGraph> graph, const std::string& prefix, QMainWindow* parent);
	void restore_window_settings();
	void save_window_settings() const;




private:

public slots:
	virtual void initialize() = 0;
	virtual void compute() = 0;
	virtual void emergency() = 0;
	virtual void restore() = 0;
	void hibernationCheck();
	void hibernationTick();
	
signals:
	void kill();
	void goToEmergency();
	void goToRestore();
};

#endif
