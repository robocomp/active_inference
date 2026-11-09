#pragma once
/*
 * recognition_cycle.h -- one recognition cycle as compute() runs it: the recogniser's step, then RoomSwap executing its
 * request (a Swap POSTS a proposal, room_proposal.h) and acknowledging back. Kept here (not inline in specificworker.cpp) so
 * the WIRING is testable without the agent.
 *
 * ORDER inside the cycle:
 *   1. RoomRecognizer::step() -- a request (Swap/Undo) and, separately, `released`: an executed swap's episode ended.
 *   2. released => RoomSwap::on_room_left(): the window closes BEFORE the request is executed, because that request may
 *      already belong to the NEXT episode (a newer provisional room began in the same step). Pending => withdrawn; adopted =>
 *      hold released, proposal withdrawn.
 *   3. the request, executed and acknowledged (a request is not an execution).
 *   4. RoomSwap::step(): room_concept's verdict. Rejected, or withdrawn (P gone / the robot left P before a verdict) =>
 *      RoomRecognizer::on_proposal_rejected(): back to judging, the evidence REBASED at the rejection (spec: rejection is
 *      per proposal; ltsm may propose again with new evidence).
 */
#include "room_recognizer.h"
#include "room_swap.h"

#include <QDebug>
#include <QString>

#include <format>

namespace ltsm
{
inline RoomRecognizer::Step recognition_cycle(RoomRecognizer &rec, RoomSwap *swap)
{
    const auto rq = rec.step();
    if (rq.act != RoomRecognizer::Step::Act::None)
        qInfo().noquote() << QString::fromStdString(std::format("[recog] {} requested for room_id {} (executed by RoomSwap)",
                rq.act == RoomRecognizer::Step::Act::Swap ? "SWAP" : "UNDO", rq.room_id));
    if (swap == nullptr) return rq;
    if (rq.released and swap->state() != RoomSwap::State::Entering)
    {
        qInfo() << "[recog] the held swap's episode ended: closing RoomSwap's window";
        swap->on_room_left();
    }
    if (rq.act == RoomRecognizer::Step::Act::Swap)
    {
        if (swap->swap(rq.room_id, rq.pose_in_k, rec.provisional_id(), rq.withdraw_crossing))
        {
            rec.on_swap_executed(rq.room_id);
            if (not rec.swapped())
                qWarning() << "[swap] the recogniser ignored the acknowledgement of the swap to room_id" << static_cast<qulonglong>(rq.room_id)
                           << "(episode or commit no longer matches); the undo window is driven by RoomSwap";
        }
        else rec.on_swap_failed();
    }
    else if (rq.act == RoomRecognizer::Step::Act::Undo)
    {
        if (swap->undo()) rec.on_undo_executed();
        else qWarning() << "[swap] UNDO requested for room_id" << static_cast<qulonglong>(rq.room_id) << "but no swap stands";
    }
    // The window is otherwise closed by the eviction that leaves the recognised room (RoomSwap::on_evict/on_evicted).
    if (const auto ev = swap->step(); ev == RoomSwap::Event::Rejected or ev == RoomSwap::Event::Withdrawn)
        rec.on_proposal_rejected();
    return rq;
}
}   // namespace ltsm
