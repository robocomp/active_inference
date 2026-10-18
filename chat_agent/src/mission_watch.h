#pragma once
/*
 * mission_watch.h — chat_agent's read side of the controller's `controller_mission` node (controller/src/mission_node.h),
 * 2026-10-17. Read-only: the controller owns the node and every attribute on it.
 *
 * poll() returns only what changed since the last call: a new mission IR (decoded, plus the JSON it came from) and/or
 * a new progress snapshot. The plan window (plan_view.h) is fed from it on the session's status tick.
 */

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <dsr/api/dsr_api.h>

#include "../../common/mission/mission_ir.h"
#include "../../common/mission/mission_progress.h"

namespace rc::chat
{

class MissionWatch
{
public:
    static constexpr const char* kName = "controller_mission";

    struct Update
    {
        std::optional<std::pair<rc::mission::Mission, std::string>> mission;   ///< the IR and its JSON
        std::optional<rc::mission::Progress> progress;
    };

    explicit MissionWatch(std::shared_ptr<DSR::DSRGraph> G) : G_(std::move(G)) {}

    Update poll();

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    std::string last_ir_, last_progress_;
    int last_serial_ = -1;
};

}   // namespace rc::chat
