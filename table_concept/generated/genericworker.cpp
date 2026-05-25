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
#include "genericworker.h"
/**
* \brief Default constructor
*/
GenericWorker::GenericWorker(const ConfigLoader& configLoader, TuplePrx tprx) : QObject()
{

	this->configLoader = configLoader;
    if (!this->configLoader.get<bool>("Component.Debug.Verbose")) {
        qInstallMessageHandler([](QtMsgType, const QMessageLogContext&, const QString&) {});
    }

	states["Initialize"] = std::make_unique<GRAFCETStep>("Initialize", BASIC_PERIOD, nullptr, std::bind(&GenericWorker::initialize, this));
	states["Compute"] = std::make_unique<GRAFCETStep>("Compute", configLoader.get<int>("Period.Compute"), std::bind(&GenericWorker::compute, this));
	states["Emergency"] = std::make_unique<GRAFCETStep>("Emergency", configLoader.get<int>("Period.Emergency"), std::bind(&GenericWorker::emergency, this));
	states["Restore"] = std::make_unique<GRAFCETStep>("Restore", BASIC_PERIOD, nullptr, std::bind(&GenericWorker::restore, this));

	states["Initialize"]->addTransition(states["Initialize"].get(), SIGNAL(entered()), states["Compute"].get());
	states["Compute"]->addTransition(this, SIGNAL(goToEmergency()), states["Emergency"].get());
	states["Emergency"]->addTransition(this, SIGNAL(goToRestore()), states["Restore"].get());
	states["Restore"]->addTransition(states["Restore"].get(), SIGNAL(entered()), states["Compute"].get());

	statemachine.addState(states["Initialize"].get());
	statemachine.addState(states["Compute"].get());
	statemachine.addState(states["Emergency"].get());
	statemachine.addState(states["Restore"].get());

	statemachine.setInitialState(states["Initialize"].get());

	connect(&hibernationChecker, SIGNAL(timeout()), this, SLOT(hibernationCheck()));


    agent_name = this->configLoader.get<std::string>("Agent.name");
    agent_id = this->configLoader.get<int>("Agent.id");

    // Create graph
    auto surNames = configLoader.getSurNames("Agent");
    if (surNames.empty()) {
        int domain = this->configLoader.exists("Agent.domain") ? this->configLoader.get<int>("Agent.domain") : 0;
        auto [it, inserted] = Graphs.emplace("", std::make_shared<DSR::DSRGraph>(0, agent_name, agent_id, 
                                        this->configLoader.get<std::string>("Agent.configFile"), 
                                        true, domain));
        std::cout << "Graph loaded" << std::endl;
        G = it->second;
    } 
    else {
        std::cout << "Multiple graphs found: " << surNames.size() << std::endl;
        for (std::string_view surName : surNames) {
            std::string name{surName};
            std::string prefix = "Agent." + name;

            Graphs.emplace(name, std::make_shared<DSR::DSRGraph>(0, agent_name, agent_id, 
                                            configLoader.get<std::string>(prefix + ".configFile"), 
                                            true, 
                                            configLoader.get<int>(prefix + ".domain")));
            std::cout << "Graph " << name << " loaded" << std::endl;
        }
        G = Graphs.at(std::string(surNames.front()));
    }
}

/**
* \brief Default destructor
*/
GenericWorker::~GenericWorker()
{
    for (auto& [name, graphPtr] : Graphs) {
        if (!graphPtr) continue;
        auto grid_nodes = graphPtr->get_nodes_by_type("grid");
        for (auto grid : grid_nodes) {
            graphPtr->delete_node(grid);
        }
    }
}
void GenericWorker::killYourSelf()
{
	qDebug("Killing myself");
	emit kill();
}

/**
* \brief Change compute period of state
* @param state name of state
* @param period Period in ms
*/
void GenericWorker::setPeriod(const std::string& state, int period)
{
    auto it = states.find(state); 
    if (it != states.end() && it->second != nullptr)
    {
		it->second->setPeriod(period);
		std::cout << "Period for state " << state << " changed to " << period << "ms" << std::endl << std::flush;
	}
    else
        std::cerr << "No change in the period, the state is not valid or not configured."<< std::endl;
}

int GenericWorker::getPeriod(const std::string& state)
{
    auto it = states.find(state);

    if (it == states.end() || it->second == nullptr)
    {
        std::cerr << "Invalid or unconfigured state: " << state << std::endl;
        return -1; 
	}
    return it->second->getPeriod();
}

void GenericWorker::hibernationCheck()
{
	//Time between activity to activate hibernation
    static const int HIBERNATION_TIMEOUT = 5000;

    static std::chrono::high_resolution_clock::time_point lastWakeTime = std::chrono::high_resolution_clock::now();
	static int originalPeriod;
    static bool isInHibernation = false;

	// Update lastWakeTime by calling a function
    if (hibernation)
    {
        hibernation = false;
        lastWakeTime = std::chrono::high_resolution_clock::now();

		// Restore period
        if (isInHibernation)
        {
            this->setPeriod("Compute", originalPeriod);
            isInHibernation = false;
        }
    }

    auto now = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastWakeTime);

	//HIBERNATION_TIMEOUT exceeded, change period
    if (elapsedTime.count() > HIBERNATION_TIMEOUT && !isInHibernation)
    {
        isInHibernation = true;
		originalPeriod = this->getPeriod("Compute");
        this->setPeriod("Compute", 500);
    }
}

void GenericWorker::hibernationTick(){
	hibernation = true;
}

std::shared_ptr<DSR::DSRViewer> GenericWorker::setupViewer(std::shared_ptr<DSR::DSRGraph> graph, const std::string& prefix, QMainWindow* parent)
{
    int current_opts = 0;
    DSR::DSRViewer::view main = DSR::DSRViewer::view::none;
    using opts = DSR::DSRViewer::view;

    // Estructura de datos para iterar las opciones (más limpio que muchos IFs)
    const std::vector<std::pair<std::string, opts>> options = {
        {"tree", opts::tree}, {"graph", opts::graph}, 
        {"2d", opts::scene},  {"3d", opts::osg}
    };

    for (const auto& [suffix, flag] : options) {
        if (this->configLoader.get<bool>(prefix + "." + suffix)) {
            current_opts |= flag;
            if (suffix == "graph") main = opts::graph;
        }
    }
    if (current_opts!=0)
    	return std::make_shared<DSR::DSRViewer>(parent, graph, current_opts, main);
	else
		return nullptr;
};

void GenericWorker::initialize(){
    for (const auto& [name, Graph] : Graphs) {
        std::unique_ptr<QMainWindow> window = std::make_unique<QMainWindow>();
        window->setWindowTitle(QString("%1-%2|%3").arg(QString::fromStdString(agent_name)).arg(agent_id).arg(QString::fromStdString(name)));

        std::string prefix = "Agent";
        if (Graphs.size()>1)
            prefix += "." +name;

        std::shared_ptr<DSR::DSRViewer> viewer = setupViewer(Graph, prefix, window.get());
        if (viewer){
            graph_viewers.emplace(name, std::move(viewer));
            windows.emplace(name, std::move(window));
        }
    }
};