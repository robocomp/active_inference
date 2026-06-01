#include "specificworker.h"

void SpecificWorker::waiting_enter()
{
    presence_coordinator_.waiting_enter();
}

void SpecificWorker::waiting_loop()
{
    presence_coordinator_.waiting_loop();
}

void SpecificWorker::operating_enter()
{
    presence_coordinator_.operating_enter();
}

void SpecificWorker::operating_loop()
{
    presence_coordinator_.operating_loop();
    if (auto it = graph_viewers.find(""); it != graph_viewers.end() && it->second)
        it->second->set_external_fps(states.at("Operating")->getActualFps());
}

void SpecificWorker::degraded_enter()
{
    presence_coordinator_.degraded_enter();
}

void SpecificWorker::degraded_loop()
{
    presence_coordinator_.degraded_loop();
}

void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;

    obstacle_tracker_.clear_published_obstacles();
    presence_coordinator_.cleanup_owned_nodes();
}

void SpecificWorker::on_optional_peer_lost(const std::string &name, std::uint32_t /*id*/)
{
    std::cout << "[Presence] optional peer lost: " << name << std::endl;
}

void SpecificWorker::on_optional_peer_ready(const std::string &name, std::uint32_t /*id*/)
{
    std::cout << "[Presence] optional peer ready: " << name << std::endl;
}

void SpecificWorker::emergency()
{
    std::cout << "Emergency worker" << std::endl;
}

void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
}

int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
	return 0;
}