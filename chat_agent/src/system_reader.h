#pragma once
/*
 * system_reader.h — fills system_facts.h from the live graph and the cameras, for the system tools (agents, sensors,
 * graph_summary, inspect) and the STATE's agents line. Read-only. Main thread (the session's timers), like the
 * CameraGrabber it reads rates from. 2026-10-17.
 */

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <dsr/api/dsr_api.h>

#include "camera_grabber.h"
#include "system_facts.h"

namespace rc::chat
{

class SystemReader
{
public:
    SystemReader(std::shared_ptr<DSR::DSRGraph> G, const CameraGrabber& cameras) : G_(std::move(G)), cameras_(cameras) {}

    [[nodiscard]] std::vector<AgentFact> agents() const;
    /// Every node carrying a media descriptor; the cameras' measured rate from the CameraGrabber.
    [[nodiscard]] std::vector<SensorFact> sensors() const;
    [[nodiscard]] GraphSummary graph(const std::string& current_room) const;
    /// A node by name, or by id if `name_or_id` is a number.
    [[nodiscard]] std::optional<NodeView> inspect(const std::string& name_or_id) const;

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    const CameraGrabber& cameras_;
};

}   // namespace rc::chat
