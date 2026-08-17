#pragma once
/*
 * common/agent_exit/terminal_exit.h — the crash-free terminal exit every concept agent ends with.
 *
 * WHY AN AGENT HARD-EXITS INSTEAD OF RETURNING. Returning runs the Ice communicator teardown and C++ static
 * destruction, which have undefined cross-TU order: a global/DDS holder copies a graph Node after the
 * node-type registry static is gone, Node::type() throws, and the process aborts on the way out. bottle hits
 * a second one — it is the only agent with an active OUTGOING Ice client proxy, so it races an Ice worker
 * thread against IceUtil::Mutex destruction (ThreadSyscallException EINVAL). By this point the state is
 * persisted and graph presence is cleanly removed, so there is nothing left for destructors to do that the
 * OS will not do better.
 *
 * ★THE STEP A BARE _Exit SKIPS, AND WHY IT IS THE IMPORTANT ONE. G->reset() removes THIS agent's DDS
 * participant and entities from the shared graph — the clean "Publisher unmatched" path — WITHOUT touching
 * the Ice communicator. Skip it and peers keep seeing a half-deleted node (node present, its room->obj RT
 * edge already gone) and SEGV walking the RT tree; a fast restart then hits "agent id N already connected".
 * The 300 ms is not a guess at a race, it is transmit time: the del-deltas have to physically reach peers
 * before the process vanishes.
 *
 * ⚠This does not return. Anything that must happen on the way out belongs in `before_exit` (each agent's
 * request_shutdown), which is called first and is expected to be idempotent.
 */

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace rc::agent
{

// `graph_reset` should call G->reset() if G is alive, and do nothing otherwise. Passed as a callable so this
// header needs no DSR — the agents own their graph handle's lifetime and this must not second-guess it.
template <class BeforeExit, class GraphReset>
[[noreturn]] inline void terminal_exit(BeforeExit&& before_exit, GraphReset&& graph_reset)
{
    // 1) Sever graph callbacks, delete owned DSR nodes (publishes the del-deltas) and notify peers.
    //    Idempotent by contract (each agent guards with shutting_down_).
    before_exit();

    // 2) Leave the DDS graph cleanly — see the header note; this is what a bare _Exit skips.
    try { graph_reset(); }
    catch (...) { /* best-effort: we are exiting regardless */ }

    // 3) Let the writers actually transmit the removals + the participant departure.
    std::cout.flush();
    std::cerr.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // 4) Hard-exit, skipping static destruction and the Ice communicator teardown.
    std::_Exit(EXIT_SUCCESS);
}

}  // namespace rc::agent
