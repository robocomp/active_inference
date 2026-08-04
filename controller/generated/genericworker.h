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
#include <unordered_set>
#include <iostream>
#include <string>

#include "dsr/api/dsr_api.h"
#include "dsr/gui/dsr_gui.h"
#include <doublebuffer/DoubleBuffer.h>
#include <memory>

#include <GenericBase.h>
#include <OmniRobot.h>

#define BASIC_PERIOD 100

using TuplePrx = std::tuple<RoboCompOmniRobot::OmniRobotPrxPtr>;


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


	RoboCompOmniRobot::OmniRobotPrxPtr omnirobot_proxy;


protected:
	std::unordered_map<std::string, std::unique_ptr<GRAFCETStep>> states;
	ConfigLoader configLoader;
	// ── WHY THE `catch (...)` BELOW IS NOT SILENT ANY MORE ────────────────────────────────────────
	// An optional key that fails to load is indistinguishable, at runtime, from a key nobody wrote —
	// both leave the built-in default in place and print nothing. That is correct for the intended
	// case and catastrophic for the two unintended ones, which have cost real debugging time:
	//   • SHADOWED  — the key IS in the file but under a different prefix, because a dotted key
	//     written below a [Section] header is namespaced under it. `Transforms.foo` placed inside
	//     [Controller] becomes `Controller.Transforms.foo`; the lookup misses and the default wins.
	//     (2026-08-04: seven controller keys, incl. one being actively toggled for an experiment,
	//     had NEVER been read. A whole lap was run and analysed before the flag was found inert.)
	//   • TYPE MISMATCH — the key is there and readable, but as the wrong variant arm, so `get`
	//     throws. (2026-08-03: `VelocityOutputPeriodMs = 25` is an int; read as double it threw and
	//     the agent silently ran the 50 ms default.)
	// Both are now reported. A key that is simply ABSENT stays quiet — that is the supported way to
	// say "use the default", and warning on it would bury the two real faults in noise.
	void report_config_miss(const std::string& key, const char* what) const
	{
		if (configLoader.exists(key))
		{
			std::cerr << "[config] ★ key '" << key << "' EXISTS but could not be read as the requested "
			          << "type — the built-in default is in force. " << what << '\n';
			return;
		}
		for (const auto candidate : configLoader.getKeys())
		{
			const std::string full{candidate};
			if (full.size() > key.size() + 1 and full.compare(full.size() - key.size(), key.size(), key) == 0
			    and full[full.size() - key.size() - 1] == '.')
			{
				std::cerr << "[config] ★ key '" << key << "' NOT FOUND, but the file defines '" << full
				          << "' — a dotted key written below a [Section] header is namespaced under it. "
				          << "The built-in default is in force; move it above the first [Section] or give "
				          << "it its own header.\n";
				return;
			}
		}
		// Absent and unshadowed: deliberate default, nothing to report.
	}

	template <typename ValueType>
	void load_optional(const std::string& key, ValueType& value) const
	{
		try
		{
			value = configLoader.get<ValueType>(key);
		}
		catch (const std::exception& e) { report_config_miss(key, e.what()); }
		catch (...) { report_config_miss(key, "(unknown exception)"); }
	}

	template <typename ConfigType, typename ValueType>
	void load_optional_cast(const std::string& key, ValueType& value) const
	{
		try
		{
			value = static_cast<ValueType>(configLoader.get<ConfigType>(key));
		}
		catch (const std::exception& e) { report_config_miss(key, e.what()); }
		catch (...) { report_config_miss(key, "(unknown exception)"); }
	}

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
	void trigger_graph_layout_twopi();
	void restore_window_settings();
	void save_window_settings() const;




private:
	static constexpr int kWindowStateVersion = 1;
	static QString settings_group_name(const std::string& graph_name, int agent_id);
	std::unordered_set<std::string> participant_layout_done_graphs;

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
