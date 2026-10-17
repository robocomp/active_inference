#pragma once
/*
 * mission_node.h — the controller's own report of the mission it runs (2026-10-17).
 *
 * One node, `controller_mission` (type intention, Shadow --has--> it), owned and written ONLY by the controller,
 * whatever the mission's source — chat, remote_cmd.txt or the panel — so the chat's plan window reads one place, not
 * the chat-only has_intention edge. (The episodic memory reads the controller's journal, em_journal, instead.)
 *
 *   current_intention     the mission IR, JSON (mission_json.h)
 *   mission_serial        Interpreter::serial(): +1 per mission started
 *   mission_source        chat · remote · panel
 *   state                 Interpreter::status()        (in_progress · completed · failed|<step>|<outcome> · aborted|…)
 *   bt_state              Interpreter::progress()      (the current step line)
 *   timestamp_agent       the mission's first tick (ms since the epoch)
 *   mission_progress      encode_progress(): every step's state (mission_progress.h)
 *
 * Every attribute is written only when one of them changed, in one update_node. The node is deleted on a graceful
 * shutdown (cleanup_owned_nodes) and a previous run's is replaced on first publish.
 */

#include <cstdint>
#include <string>

#include <dsr/api/dsr_api.h>

#include "../../common/mission/mission_interpreter.h"

namespace rc
{

class MissionNode
{
public:
    static constexpr const char* kName = "controller_mission";
    static constexpr const char* kRobotNode = "Shadow";

    /// Write what changed since the last call. Creates the node (and its edge) the first time.
    void publish(DSR::DSRGraph& G, const rc::mission::Interpreter& in, const std::string& source, const std::string& ir_json);
    /// Graceful shutdown: delete our node.
    void remove(DSR::DSRGraph& G);

private:
    bool ensure(DSR::DSRGraph& G);

    std::uint64_t id_ = 0;
    bool edge_done_ = false;
    std::string last_;   ///< everything last written, concatenated: the "did anything change" key
};

}   // namespace rc
