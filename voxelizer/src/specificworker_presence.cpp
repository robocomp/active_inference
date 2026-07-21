#include "specificworker.h"
#include "graph_publisher.h"
#include <print>
#include <chrono>

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

    if (graph_publisher_)
        graph_publisher_->cleanup_semantic_grid_nodes();
    presence_coordinator_.cleanup_owned_nodes();
}

void SpecificWorker::on_optional_peer_lost(const std::string &name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer lost:" << QString::fromStdString(name);
}

void SpecificWorker::on_optional_peer_ready(const std::string &name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer ready:" << QString::fromStdString(name);
}

bool SpecificWorker::room_node_present() const
{
    // get_nodes_by_type is serialized by DSRGraph's shared_mutex → safe from the main (SM) thread.
    return G and not G->get_nodes_by_type("room").empty();
}

// One structured JSON line per FSM transition on stdout (not routed through qInfo, so it prints regardless
// of the Verbose message-handler). Stable schema — a monitor greps `[SM][event]` and parses the JSON; the
// `reason` field is a fixed vocabulary: constraints_satisfied | required_peer_lost | room_unstable.
void SpecificWorker::log_sm_event(const std::string& from, const std::string& to,
                                  const std::string& reason, const std::string& detail)
{
    const auto ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now().time_since_epoch()).count();
    std::println(R"([SM][event] {{"event":"transition","agent":"{}","from":"{}","to":"{}","reason":"{}","detail":"{}","ts_ms":{}}})",
                 agent_name, from, to, reason, detail, ts_ms);
}
