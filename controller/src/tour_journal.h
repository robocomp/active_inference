#pragma once
/*
 * tour_journal.h — panel tours in the episodic journal (final review 2026-10-17, finding 8).
 *
 * A tour (MissionRunner, MissionOnly — started only by the panel's Run button, specificworker.cpp) is an operator
 * command like any other, so it is journaled as a HUMAN mission, source "panel", key "p<n>": MissionStart when a run
 * is first seen running, MissionEnd when it is seen stopped, outcome "completed" for a run that ended by itself
 * (stop reason "completed" / "exhausted"), else "aborted|<stop reason>".
 * MISSION LEVEL ONLY: MissionRunner has no waypoint legs any more (continuous characterisation, controller_mission.h),
 * so there is nothing cheap and honest to journal as steps; claims under it still nest (Builder: innermost mission).
 *
 * Pure (em_journal.h + Qt Core). Fed once per control pass by service_journal — the same choke point as the drive
 * mode, so no MissionRunner call site has to remember it.
 */

#include "../../common/episodic/em_journal.h"

#include <cstdint>
#include <format>
#include <string>
#include <vector>

#include <QJsonDocument>
#include <QJsonObject>

namespace rc
{

class TourJournal
{
public:
    struct View
    {
        bool running = false;
        std::string name;           ///< the running tour's name (when running)
        int loops = 0;              ///< laps asked for (when running)
        std::string mode;           ///< to_string(DriveMode)
        std::string stop_reason;    ///< MissionRunner::stop_reason() (when stopped)
    };

    /// The entries this observation implies (t_ms / serial left to the journal).
    std::vector<rc::em::JournalEntry> observe(const View& v)
    {
        using K = rc::em::JournalEntry::Kind;
        std::vector<rc::em::JournalEntry> out;
        if (v.running and open_.empty())
        {
            open_ = std::format("p{}", ++n_);
            const QJsonObject text{{"title", QString::fromStdString("tour " + v.name)},
                                   {"tour", QString::fromStdString(v.name)},
                                   {"loops", v.loops},
                                   {"mode", QString::fromStdString(v.mode)}};
            out.push_back({.kind = K::MissionStart, .mission = open_, .op = "tour", .source = "panel",
                           .text = QJsonDocument(text).toJson(QJsonDocument::Compact).toStdString()});
        }
        else if (not v.running and not open_.empty())
        {
            const bool by_itself = v.stop_reason == "completed" or v.stop_reason == "exhausted";
            out.push_back({.kind = K::MissionEnd, .mission = open_,
                           .outcome = by_itself ? std::string("completed")
                                    : v.stop_reason.empty() ? std::string("aborted")
                                                            : "aborted|" + v.stop_reason});
            open_.clear();
        }
        return out;
    }

private:
    std::string open_;          ///< the tour open in the journal ("p3"); "" = none
    std::uint64_t n_ = 0;
};

}   // namespace rc
