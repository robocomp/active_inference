#pragma once
/*
 * chat_inbox.h — the controller's end of chat_agent's command channel (2026-10-12, robot chat phase 1).
 *
 * chat_agent owns ONE node, `chat_intention` (type `intention`, has_intention edge from the robot node "Shadow").
 * It writes the command text into `current_intention` and stamps `timestamp_agent` (epoch ms); a new stamp is a new
 * command. The controller reads it here and writes back how that command is going — ON THE has_intention EDGE:
 *
 *   state            in_progress · completed · aborted · failed|<reason>
 *   bt_state         "<step>|<last [door-passage] event>|driving=on|off"
 *   timestamp_agent  the stamp of the command this status is about (chat_agent ignores a report on an older one)
 *
 * ★ ONE WRITER PER OBJECT, BECAUSE OF THE LWW ENGINE. cortex's update_node re-versions EVERY attribute of the local
 *   copy at the writer's clock (dsr_lww_sync_engine.cpp update_node_local: to_attr_state_map(attrs, version)), even
 *   though it only SENDS the changed ones. A status write onto the node in the few ms before chat_agent's new command
 *   arrives would give the controller's stale `current_intention` a NEWER version, and the incoming command would lose
 *   the merge and never be seen. So chat_agent writes only the node, and the controller writes only the edge.
 *
 * ★ POLLED, NEVER SIGNALLED (CLAUDE.md): no update_node_signal connect. poll() and publish() are called from the
 *   control thread's loop, which owns the session and the door passage; graph reads/writes are thread-safe.
 *
 * ★ A STALE COMMAND IS NEVER EXECUTED. The first time a node is seen (controller restart, chat_agent restart) its
 *   command is executed only if it is younger than kFirstSightMaxAgeMs; otherwise its stamp is merely adopted. A
 *   protocol rule, not a model threshold: a command typed minutes ago into a session that has since restarted is
 *   not a request anyone is still making.
 *
 * The status mapping (status_of) is a pure function of what was asked and what the passage reports, so it can be
 * read without a graph.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <dsr/api/dsr_api.h>

#include "controller_door_passage.h"
#include "controller_relative_motion.h"

namespace rc
{

class ChatInbox
{
public:
    static constexpr const char* kNodeName = "chat_intention";
    static constexpr std::uint64_t kFirstSightMaxAgeMs = 3000;

    struct Command
    {
        std::string text;
        std::uint64_t stamp_ms = 0;
    };

    /// What the last chat command asked for, kept by the caller so each cycle can mirror its progress.
    struct Tracked
    {
        enum class Kind { None, Cross, Stop, Uncross, Rejected, Motion } kind = Kind::None;
        std::uint64_t stamp = 0;   ///< the command's timestamp_agent, echoed on the edge
        std::string door;     ///< Cross: the door the passage was started through
        std::string reason;   ///< Rejected: why
        std::uint64_t motion_id = 0;   ///< Motion: RelativeMotion::View::id of the run this command started
    };

    /// A new command since the last call, or nullopt. Never blocks; a missing node is "no command".
    std::optional<Command> poll(DSR::DSRGraph& G, std::uint64_t now_ms);

    /// Mirror (stamp, state, bt_state) onto the has_intention edge, writing only when something differs from the
    /// edge's CURRENT values (an unchanged status costs no DDS traffic). No edge yet (no robot node) = no report.
    void publish(DSR::DSRGraph& G, std::uint64_t stamp, const std::string& state, const std::string& bt_state);

    /// (state, bt_state) for a tracked command given the passage's view and whether the base is armed.
    [[nodiscard]] static std::pair<std::string, std::string> status_of(const Tracked& t, const DoorPassage::View& v,
                                                                      bool driving);
    /// The same for a `move` / `turn` command: in_progress while it runs, completed / failed|<reason> / aborted.
    /// bt_state = "<turning|moving|done|failed|cancelled>|<progress or reason>|driving=on|off".
    [[nodiscard]] static std::pair<std::string, std::string> status_of_motion(const Tracked& t,
                                                                             const RelativeMotion::View& v, bool driving);

private:
    std::uint64_t node_id_ = 0;      ///< the node whose stamps we follow; a different id = a new channel
    std::uint64_t last_stamp_ = 0;
};

}   // namespace rc
