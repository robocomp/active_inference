/*
 * phantom_log.h — SHARED shadow-mode recorder for instance birth/death events.
 *
 * Stage 4.2 of CONCEPT_AGENT_LIFECYCLE.md ("every agent must record phantom events"). Header-only, no DSR,
 * no DDS — just an append-only CSV with a schema that is IDENTICAL across agents, so one analysis script
 * works on the whole fleet.
 *
 * WHY. Classifiers make CONFIDENT mistakes under partial occlusion — part of a radiator reads as a chair with
 * a high score, births a phantom, and it survives until a good run of masks from a proper pose kills it.
 * Score cannot filter that. But the RECURRENCE is learnable: the same place, seen from the same bearing, lies
 * the same way every tour. This file records the raw material; MODEL_HISTORY.md §4 is what will eventually
 * consume it as a per-class false-alarm field p_FA(world cell × view bearing) that lowers
 * Detection::birth_evidence.
 *
 * ★SHADOW MODE — RECORDING ONLY. Nothing here feeds back into any belief, birth or removal decision, and it
 * must stay that way until the removal path is validated. A birth→death event is evidence of EITHER a
 * detector false positive OR a REMOVAL false positive, and July 2026 produced three independent removal
 * defects (ZedClearLosFloor deleting chairs the ZED could not see; silhouettes voting removal through walls;
 * an inert p_detect stuck at the saturation ceiling) that all manufacture exactly this signature. Learning
 * from them would bake removal bugs into a persistent spatial prior — and because a suppressed birth produces
 * no contradicting evidence, the error would be self-sealing and unfalsifiable.
 *
 * Hence the attribution fields (p_detect / in_fov / central / fixated at death): a death only counts as a
 * PHANTOM event if the disconfirmation was CONFIDENT. Deaths clustering where p_detect was LOW mean the log
 * is recording our own removal bugs, not the classifier's errors. That check gates arming §4.2.
 */

#pragma once

#include "../diag_log/rotating_csv.h"   // rc::diag::open_rotating — keep the previous run

#include <cstdint>
#include <fstream>
#include <locale>     // std::locale::classic — see the imbue in open()
#include <string>
#include <string_view>

namespace rc::history
{

// One birth or death, with everything needed to (a) locate it in the world, (b) locate the VIEWPOINT it was
// seen from, and (c) decide whether a death was a confident disconfirmation. Defaults are the "unknown /
// not applicable" values so a birth only has to fill the first few fields.
struct PhantomEvent
{
    std::string_view event;          // "BIRTH" | "DEATH" | "MERGE" | "SUPPRESS" — free-form, uppercase
    std::uint64_t    id = 0;         // instance/node id
    std::string_view name;           // node name (e.g. "chair_3"), for cross-referencing ai2_log.csv

    float x = 0.0f, y = 0.0f;        // instance centre, ROOM frame (m) — the world cell key

    // Observer pose at the event, ROOM frame. Together with (x,y) this gives the view bearing that the
    // eventual p_FA field is indexed on — the failure is viewpoint-dependent, so place alone is not enough.
    float robot_x = 0.0f, robot_y = 0.0f, robot_yaw = 0.0f;
    float view_bearing = 0.0f;       // room-frame bearing instance → camera (rad); atan2(ry-y, rx-x)
    float range_m = 0.0f;            // observer → instance distance (m)

    int   age_cycles = 0;            // cycles the instance survived (0 at birth) — a phantom dies young

    // ── Attribution: was the DISCONFIRMATION confident? (death only; 0 at birth) ──────────────────────
    float p_detect    = 0.0f;        // P(detect | present) at death — THE field that validates the log
    // ★A FRACTION, NOT A FLAG: detectable silhouette samples / attempted (SilhouetteExistence::in_fov_frac()).
    // The attribution question is "how much of the object could the sensor actually have SEEN from here",
    // and a 0-or-1 probed/not-probed stand-in cannot answer it — worse, written into a column documented as a
    // fraction it reads as data while carrying none. Five agents wrote the flag; table wrote the fraction and
    // said why. The note is here, on the field, so the next agent reads it before choosing what to store.
    float in_fov_frac = 0.0f;
    float central_frac = 0.0f;
    int   fixated     = 0;           // was the killing observation an admissible (close/centred/still) view?
    float exist_logodds = 0.0f;      // existence L at the moment of the event

    std::string_view note;           // free-form ("logodds -3.04", "outside room", …)
};

// Append-only CSV writer. One instance per agent; open() once and write() per event. Silently inert if the
// path is empty or cannot be opened — recording must never be able to take an agent down.
//
// ★IT ROTATES, IT DOES NOT TRUNCATE (2026-08-16). It used to open with std::ios::trunc "like the sibling
// *_events.csv writers", which is the exact behaviour common/diag_log/rotating_csv.h was written to end: the
// recorder erased the tape whenever the thing being recorded restarted, and the restart is often provoked BY
// the fault, so the interesting run was reliably the one that got lost. A PHANTOM log is the worst possible
// file to lose that way — a phantom is intermittent by definition, and the evidence that it was born, lived
// and died is precisely what a restart threw away. One fix here reaches all seven agents.
class PhantomLog
{
public:
    void open(const std::string& path)
    {
        if (path.empty()) return;
        rc::diag::open_rotating(f_, path);
        if (not f_.is_open()) return;
        // ★MANDATORY on these machines (LANG=es_ES.UTF-8): Qt calls setlocale(LC_ALL, "") at startup, and if
        // the C++ global locale is ever imbued from it, operator<< starts inserting THOUSANDS separators and a
        // COMMA decimal point — silently corrupting a comma-separated file. Every other CSV writer in the
        // fleet does this (table_fitter.cpp ai2_csv_, birth_surprise_csv_, …). See CLAUDE.md "Parsing numbers
        // from files". Readers of this file must use std::from_chars for the same reason.
        f_.imbue(std::locale::classic());
        f_ << "seq,event,id,name,x,y,robot_x,robot_y,robot_yaw,view_bearing,range_m,age_cycles,"
              "p_detect,in_fov_frac,central_frac,fixated,exist_logodds,note\n";
        f_.flush();
    }

    bool is_open() const { return f_.is_open(); }

    void write(const PhantomEvent& e)
    {
        if (not f_.is_open()) return;
        f_ << seq_++ << ',' << e.event << ',' << e.id << ',' << e.name << ','
           << e.x << ',' << e.y << ','
           << e.robot_x << ',' << e.robot_y << ',' << e.robot_yaw << ','
           << e.view_bearing << ',' << e.range_m << ',' << e.age_cycles << ','
           << e.p_detect << ',' << e.in_fov_frac << ',' << e.central_frac << ','
           << e.fixated << ',' << e.exist_logodds << ',' << e.note << '\n';
        f_.flush();   // a crash mid-tour must not lose the tail; these are low-rate events
    }

private:
    std::ofstream f_;
    long          seq_ = 0;
};

}  // namespace rc::history
