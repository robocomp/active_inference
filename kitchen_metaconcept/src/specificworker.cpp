/*
 *    Copyright (C) 2026 by RoboComp CORTEX Team
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * SpecificWorker — kitchen_metaconcept, Milestone 2 (READ-ONLY). See specificworker.h for the model,
 * the motivating defect, and why no graph-update signal is connected.
 */

#include "specificworker.h"
#include "../../common/diag_log/rotating_csv.h"   // keep the previous run instead of wiping it

#include <QTimer>
#include <QCoreApplication>
#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <cmath>
#include <cstdlib>   // std::_Exit — crash-free terminal shutdown
#include <iostream>  // std::cout/cerr flush
#include <locale>
#include <numbers>
#include <print>
#include <ranges>
#include <string_view>
#include <thread>    // brief DDS flush before _Exit

#include <dsr/api/dsr_api.h>

namespace {

const char* tier_name(rc::KitchenTier t)
{ switch (t) { case rc::KitchenTier::Tall: return "tall";
               case rc::KitchenTier::Wall: return "wall";
               default:                    return "base"; } }

constexpr float kRad2Deg = 57.29577951f;

// Geometry fallback for a class that does not determine its own tier (a plain "cabinet"). Both
// boundaries sit in WIDE physical gaps, so any value inside the gap gives identical answers — they
// separate cases rather than tune a response, which is why they are not doing a threshold's work:
//   · MOUNTING: a floor-standing carcass has z0 ≈ 0; a wall unit's is ≈ 1.45. Gap: 0.0 .. 1.45 m.
//   · TOP:      a base unit's top is the worktop ≈ 0.90; a tall unit's is ≈ 1.85. Gap: 0.9 .. 1.85 m.
// A class that DOES determine its tier (refrigerator → tall, hood → wall) never reaches these.
constexpr float kFloorStandingMaxM = 0.35f;   // z0 above this ⇒ wall-mounted
constexpr float kWorktopTopMaxM    = 1.25f;   // floor-standing with a top above this ⇒ tall, not base

}   // namespace

// ─── Constructor / Destructor ─────────────────────────────────────────────────

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader,
                               TuplePrx tprx,
                               bool startup_check)
    : GenericWorker(configLoader, tprx),
      startup_check_flag(startup_check)
{
    if (startup_check_flag)
    {
        this->startup_check();
        return;
    }

    load_config(configLoader);

    const int period = configLoader.get<int>("Period.Compute");

    states["Waiting"] = std::make_unique<GRAFCETStep>("Waiting", period,
        std::bind(&SpecificWorker::waiting_loop, this),
        std::bind(&SpecificWorker::waiting_enter, this));
    states["Operating"] = std::make_unique<GRAFCETStep>("Operating", period,
        std::bind(&SpecificWorker::operating_loop, this),
        std::bind(&SpecificWorker::operating_enter, this));
    states["Degraded"] = std::make_unique<GRAFCETStep>("Degraded", period,
        std::bind(&SpecificWorker::degraded_loop, this),
        std::bind(&SpecificWorker::degraded_enter, this));

    states["Compute"]->addTransition(states["Compute"].get(), SIGNAL(entered()), states["Waiting"].get());
    states["Waiting"]->addTransition(this, SIGNAL(presenceReady()), states["Operating"].get());
    states["Operating"]->addTransition(this, SIGNAL(presenceLost()), states["Degraded"].get());
    states["Degraded"]->addTransition(states["Degraded"].get(), SIGNAL(entered()), states["Waiting"].get());

    statemachine.addState(states["Waiting"].get());
    statemachine.addState(states["Operating"].get());
    statemachine.addState(states["Degraded"].get());

    statemachine.setChildMode(QState::ExclusiveStates);
    statemachine.start();

    const auto err = statemachine.errorString();
    if (err.length() > 0)
    {
        qWarning() << err;
        throw err;
    }
}

SpecificWorker::~SpecificWorker()
{
    request_shutdown();
    std::print("kitchen_metaconcept: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;

    save_window_settings();
    cleanup_owned_nodes();
}

void SpecificWorker::terminal_shutdown()
{
    static std::atomic<bool> terminating{false};
    if (terminating.exchange(true))
        return;   // _Exit is coming; never run this twice

    // Crash-free terminal exit (matches the level-1 agents). After our cleanup, hard-exit instead of
    // returning into the Ice communicator teardown + C++ static destruction, which run with
    // undefined cross-TU order and abort.
    request_shutdown();
    if (G)
    {
        try { G->reset(); }   // clean DDS participant/entity removal without touching the Ice communicator
        catch (...) { /* best-effort: we are exiting regardless */ }
    }
    std::cout.flush();
    std::cerr.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));   // let the del-deltas reach peers
    std::_Exit(EXIT_SUCCESS);
}

void SpecificWorker::load_config(const ConfigLoader& cfg)
{
    cfg_ = rc::load_kitchen_config(cfg);
}

// ─── Initialisation ──────────────────────────────────────────────────────────

void SpecificWorker::initialize()
{
    std::print("kitchen_metaconcept: initialize()\n");
    GenericWorker::initialize();

    if (not G)
    {
        qWarning() << "kitchen_metaconcept: DSR graph not available in initialize()";
        return;
    }

    // Ignore heavy payload attributes in local graph updates (we only read poses/geometry).
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att, laser_X_att, laser_Y_att, laser_Z_att>();

    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    presence_coordinator_.attach_state_machine(&statemachine);
    presence_coordinator_.set_transition_hooks({
        .request_presence_ready = [this]() { emit presenceReady(); },
        .request_presence_lost  = [this]() { emit presenceLost(); },
    });
    presence_coordinator_.set_peer_hooks({
        .on_peer_restarted = [](std::uint32_t id) { qInfo() << "[Presence] peer" << id << "restarted"; },
        .on_optional_peer_lost  = [this](const std::string& n, std::uint32_t i) { on_optional_peer_lost(n, i); },
        .on_optional_peer_ready = [this](const std::string& n, std::uint32_t i) { on_optional_peer_ready(n, i); },
    });
    presence_coordinator_.set_lifecycle_hooks({
        .on_waiting_enter = [this]()
        {
            const auto missing = presence_coordinator_.missing_required_names();
            if (missing.empty())
                qInfo("[SM] -> Waiting");
            else
            {
                QString m;
                for (const auto& label : missing) m += " " + QString::fromStdString(label);
                qInfo() << "[SM] -> Waiting (missing:" << m.trimmed() << ")";
            }
        },
        .on_operating_enter = [this]() { qInfo("[SM] -> Operating: all required peers present"); },
        .on_operating_loop  = [this]() { compute(); },
        .on_degraded_enter  = [this]()
        {
            if (shutting_down_)
                return;
            // Debounce: a transient required-peer flap (startup handshake, brief node churn) fires
            // presenceLost momentarily and then recovers; tearing down here would kill the agent on a
            // blip. Wait a grace period and only shut down if a required peer is STILL missing.
            qInfo("[SM] -> Degraded: required peer lost — %d ms grace before shutdown", REQUIRED_LOSS_GRACE_MS);
            QTimer::singleShot(REQUIRED_LOSS_GRACE_MS, this, [this]()
            {
                if (shutting_down_)
                    return;
                if (presence_coordinator_.all_required_ready())
                { qInfo("[SM] required peers recovered during grace — staying alive"); return; }
                qWarning("[SM] required peer still missing after grace — shutting down cleanly");
                terminal_shutdown();
            });
        },
    });
    presence_coordinator_.start();

    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::terminal_shutdown, Qt::UniqueConnection);

    // ── §6 Delta 1: NO update_node/update_edge signal connects. This agent's input IS the graph, so
    // it is the most exposed to the DirectConnection trap; it polls on the MAIN thread instead.

    // RT_API resolves the room→member edge pose + covariance out of the edge's timestamp ring buffer
    // (rt_translation/rt_covariance are ring-buffered attrs — reading them raw is wrong).
    rt_api_ = G->get_rt_api();

    // Even at M2 (which owns nothing) sweep the prefix: a FUTURE milestone's node left behind by a
    // crashed run would otherwise linger with no process alive to own it, and the sweep is the only
    // thing that reaps it.
    remove_owned_kitchen_nodes();

    // ★Write both CSVs through the classic locale. std::ofstream formats through the C++ global
    // locale, which stays "C" today — but pinning it here means these files can never acquire a
    // comma separator if that ever changes, and every reader of them uses std::from_chars anyway.
    if (not cfg_.csv_path.empty())
    {
        rc::diag::open_rotating(csv_, cfg_.csv_path);
        if (csv_.is_open())
        {
            csv_ << "cycle,node,class,class_logodds,yaw_deg,std_yaw_deg,pinned,worktop,depth,length,"
                    "tier,std_x,std_y,has_cov,has_size_cov,std_worktop,std_depth,membership\n";
        }
        else qWarning() << "kitchen_metaconcept: cannot open CSV" << QString::fromStdString(cfg_.csv_path);
    }
    if (not cfg_.fit_csv_path.empty())
    {
        rc::diag::open_rotating(fit_csv_, cfg_.fit_csv_path);
        if (fit_csv_.is_open())
        {
            fit_csv_ << "cycle,n_members,n_axis,n_worktop,n_tall,n_wall,log_odds,p_frame,f_couple,"
                        "f_couple_if_adopted,d_f_adopt,"
                        "axis_deg,axis_std_deg,worktop,worktop_std,depth,depth_std,depth_tall,depth_wall,"
                        "node,class,membership,pinned,member_yaw_deg,axis_resid_deg,prior_yaw_deg,prior_kappa,"
                        "member_worktop,prior_worktop,member_depth,prior_depth\n";
        }
        else qWarning() << "kitchen_metaconcept: cannot open fit CSV"
                        << QString::fromStdString(cfg_.fit_csv_path);
    }

    if (not cfg_.outline_csv_path.empty())
    {
        if (rc::diag::open_rotating(outline_csv_, cfg_.outline_csv_path,
                "cycle,tier,run_a,run_b,vertex_x,vertex_y,gap_a,gap_b,end_a_high,end_b_high,cross\n"))
            outline_csv_.imbue(std::locale::classic());
        else qWarning() << "kitchen_metaconcept: cannot open outline CSV"
                        << QString::fromStdString(cfg_.outline_csv_path);
    }

    rc::KitchenBeliefParams bp;
    bp.axis_model_std     = cfg_.axis_model_std_deg / kRad2Deg;
    bp.worktop_model_std  = cfg_.worktop_model_std_m;
    bp.depth_model_std    = cfg_.depth_model_std_m;
    bp.clutter_frac       = cfg_.clutter_frac;
    bp.evidence_ema_alpha = cfg_.evidence_ema_alpha;
    kitchen_belief_.set_params(bp);

    // The belief is pure Eigen — verify it in isolation at startup (recipe §3.10).
    rc::KitchenBelief::self_test();

    const auto rooms = G->get_nodes_by_type("room");
    if (not rooms.empty())
        room_node_id_ = rooms.front().id();
    else
        qWarning() << "kitchen_metaconcept: no room node found at startup";

    std::print("kitchen_metaconcept: READ-ONLY milestone — fits and logs, writes NOTHING to the graph.\n");
}

// ─── Main compute loop ───────────────────────────────────────────────────────

void SpecificWorker::compute()
{
    if (not G)
        return;

    if (room_node_id_ == 0)
    {
        const auto rooms = G->get_nodes_by_type("room");
        if (rooms.empty()) return;
        room_node_id_ = rooms.front().id();
    }

    ++cycle_;

    // §6 Delta 1 — graph-reader front end: poll the members on the MAIN thread.
    const auto snap = poll_members();
    log_members_csv(snap);

    // M2 — fit the frame to this poll. READ-ONLY: no node birthed, no edge written, no attribute
    // pushed to any peer. It logs the down-prior it WOULD publish so the premise can be checked
    // against live data before anything is allowed to steer a cabinet.
    step_kitchen_belief(snap);
    log_fit_csv(snap);

    // The continuous-shape view: where the runs SHOULD meet, versus where they actually end.
    step_outline(snap);
    log_outline_csv();

    if (cfg_.log_period_frames > 0 and (cycle_ % static_cast<std::uint64_t>(cfg_.log_period_frames)) == 0)
    {
        // Report the REJECTIONS alongside the counts: "0 members" only means there is no kitchen if
        // we also know how many objects were there and why each was passed over.
        std::string others;
        for (const auto& n : snap.unclassified)
            others += (others.empty() ? "" : " ") + n;

        if (kitchen_belief_.seeded())
        {
            // Report the WEAKEST member alongside the aggregate: the evidence is a sum, so a single
            // member sitting on the clutter column is the usual reason a good-looking fit still
            // scores badly, and the summary line must not hide which one.
            float worst = 2.0f; std::string worst_name = "-";
            for (const auto& m : snap.members)
                if (const float mp = kitchen_belief_.membership_prob(m); mp < worst)
                { worst = mp; worst_name = m.name; }
            // F_couple = the coupling's contribution to the fleet's free energy, and what it would
            // become if every element adopted the frame's prior. NEGATIVE F means the coupled model
            // beats independent units; dF < 0 means readjustment actually pays. Those two numbers are
            // the whole claim of this layer, so they belong in the one line that gets read.
            const float F  = kitchen_belief_.free_energy(snap.members);
            const float Fa = kitchen_belief_.free_energy_if_adopted(snap.members);
            std::print("[kitchen_metaconcept] members={} (axis={} pinned, base={} tall={} wall={}) | "
                       "axis={:.2f}°±{:.2f}° worktop={:.3f} depth={:.3f}/{:.3f}/{:.3f} | "
                       "F={:+.2f} dF_adopt={:+.2f} p={:.2f} | weakest {}={:.2f}\n",
                       snap.members.size(), kitchen_belief_.n_axis_members(),
                       kitchen_belief_.n_worktop_members(), kitchen_belief_.n_tall_members(),
                       kitchen_belief_.n_upper_members(),
                       kRad2Deg * kitchen_belief_.state().axis,
                       kRad2Deg * std::sqrt(std::max(0.0f, kitchen_belief_.sigma().axis)),
                       kitchen_belief_.state().worktop, kitchen_belief_.state().depth,
                       kitchen_belief_.state().depth_tall, kitchen_belief_.state().depth_upper,
                       F, Fa - F, kitchen_belief_.p_frame(), worst_name, worst);
            if (not outline_floor_.joints().empty() or not outline_wall_.joints().empty())
                std::print("[kitchen_metaconcept]   outline: {} floor joints, {} wall joints | "
                           "worst gap {:+.3f} m ({})\n",
                           outline_floor_.joints().size(), outline_wall_.joints().size(),
                           std::abs(outline_floor_.worst_gap()) > std::abs(outline_wall_.worst_gap())
                               ? outline_floor_.worst_gap() : outline_wall_.worst_gap(),
                           (outline_floor_.worst_gap() > 0.0f) ? "hole" : "overlap");

            // WHO is in the frame this cycle. The aggregate line cannot answer "is the right set of
            // objects participating?", which is the first question asked of any grouping — and a
            // member silently joining or dropping out is exactly how a shared value drifts.
            // Per member: name(tier, membership, [pin]) — the pin marks it as AXIS evidence.
            std::string who;
            std::size_t n_sizecov = 0;
            for (const auto& m : snap.members)
            {
                if (m.has_size_cov) ++n_sizecov;
                who += std::format(" {}({},{:.2f}{}{})", m.name, tier_name(m.tier),
                                   kitchen_belief_.membership_prob(m),
                                   m.pinned ? ",pin" : "", m.has_size_cov ? "" : ",NOSIZECOV");
            }
            std::print("[kitchen_metaconcept]   participants:{}\n", who.empty() ? " <none>" : who);
            // ★A silent fallback to the length-scaled stand-in looks exactly like success in every
            // other number, so say it out loud: with the stand-in the frame cannot tell a badly-fitted
            // long run from a well-fitted one, and the shared value gets dragged.
            if (n_sizecov < snap.members.size())
                std::print("[kitchen_metaconcept]   ⚠ {}/{} members publish no object_size_variance — "
                           "weighting those by run LENGTH (a poor proxy for fit quality)\n",
                           snap.members.size() - n_sizecov, snap.members.size());
        }
        else
            std::print("[kitchen_metaconcept] members=0 | objects={} (other-class={} [{}], no-RT-pose={}) room_id={}\n",
                       snap.objects_total, snap.skipped_class, others, snap.skipped_no_rt, room_node_id_);
    }

    fps_counter_.print("[kitchen_metaconcept Compute]");
}

// ─── Graph-reader front end (§6 Delta 1) ─────────────────────────────────────

// Class of a generic "object" node. Every concept agent now publishes type()=="object" with the
// class in object_subtype (name prefixes cabinet_*/… unchanged). Accept the attribute only when it
// names a class we know and otherwise fall through to the name prefix — for some producers
// object_subtype carries a SHAPE rather than a class (a table's round/square).
std::string SpecificWorker::object_class_of(const DSR::Node& node) const
{
    std::string cls;
    if (const auto s = G->get_attrib_by_name<object_subtype_att>(node); s.has_value())
        if (const std::string& sv = s.value();
            std::ranges::find(cfg_.member_classes, sv) != cfg_.member_classes.end())
            cls = sv;

    if (cls.empty())
        for (const auto& c : cfg_.member_classes)
            if (node.name().rfind(c, 0) == 0)   // NAME-prefix fallback
            { cls = c; break; }

    // ★NO island special-casing (2026-08-10, by inspection of the apartment). What cabinet_concept
    // names `cabinet_island` here is NOT a free-standing island: it is an elongated cabinet attached
    // to a wall by its NARROW side — a peninsula. Its chart axis is the wall NORMAL, which is the same
    // 90° grid as the wall runs by construction, so it needs no separate class and no separate
    // treatment in this schema. Genuine free-standing islands are deferred.
    //
    // Promoting it to a distinct class was also actively harmful: a class prior of +2.5 raises the
    // member's mixture weight to 0.98, and a badly-fitting member with a STRONG class prior is much
    // more damaging than one with a weak prior (its clutter floor drops from −1.77 to −3.91 nats).
    // With 3 wall runs + this unit that alone reaches −9.2 and pins the frame evidence at the −8 clamp
    // — which is what the first live run showed.
    return cls;
}

// Which TIER a member belongs to. The CLASS decides where it can (a refrigerator is a tall unit, a
// hood is wall-mounted); otherwise geometry does, using the two wide physical gaps documented above.
//
// This is the most valuable thing the subtype channel provides: not "is this kitchen-ish" but "WHICH
// shared quantities apply to it". A tall unit shares the axis and a tall depth and has no worktop —
// scoring its 1.85 m top against the worktop plane is what evicted the live refrigerator.
rc::KitchenTier SpecificWorker::resolve_tier(const std::string& cls, float z0, float top) const
{
    if (const auto it = std::ranges::find(cfg_.member_classes, cls); it != cfg_.member_classes.end())
    {
        const auto idx = static_cast<std::size_t>(std::distance(cfg_.member_classes.begin(), it));
        if (idx < cfg_.member_class_tiers.size())
        {
            const std::string& t = cfg_.member_class_tiers[idx];
            if (t == "base") return rc::KitchenTier::Base;
            if (t == "tall") return rc::KitchenTier::Tall;
            if (t == "wall") return rc::KitchenTier::Wall;
            // "auto" (or anything unrecognised) falls through to geometry.
        }
    }
    if (z0 > kFloorStandingMaxM)      return rc::KitchenTier::Wall;   // mounted off the floor
    if (top > kWorktopTopMaxM)        return rc::KitchenTier::Tall;   // floor-standing but no worktop
    return rc::KitchenTier::Base;
}

// Class-evidence lookup: log[P(class | kitchen unit) / P(class | not)]. An unknown class contributes
// 0 (uninformative) rather than being rejected — class re-weights membership, it never gates it.
float SpecificWorker::class_logodds_of(const std::string& cls) const
{
    const auto it = std::ranges::find(cfg_.member_classes, cls);
    if (it == cfg_.member_classes.end())
        return 0.0f;
    const auto idx = static_cast<std::size_t>(std::distance(cfg_.member_classes.begin(), it));
    return idx < cfg_.member_class_logodds.size() ? cfg_.member_class_logodds[idx] : 0.0f;
}

// One member's room-frame pose + the covariance ITS OWN agent published. That published Σ is the
// measurement noise of this level-2 belief (§6 Delta 2) — a run the network is unsure of contributes
// weakly to the frame fit by construction, so there is no confidence gate to tune.
std::optional<rc::KitchenMember> SpecificWorker::read_member(const DSR::Node& node, const std::string& cls)
{
    if (not rt_api_ or room_node_id_ == 0)
        return std::nullopt;

    const auto edge = G->get_edge(room_node_id_, node.id(), "RT");
    if (not edge.has_value())
        return std::nullopt;   // not room-parented (yet) — nothing to measure

    // ★Always check the optional: the RT chain bails to {} at any missing node/edge/rtmat, and
    // dereferencing that nullopt is the crash mode (CLAUDE.md).
    const auto rtmat = rt_api_->get_edge_RT_as_rtmat(edge.value());
    if (not rtmat.has_value())
        return std::nullopt;

    float var_z0 = 0.0f;   // published variance of the carcass bottom (RT translation z)

    rc::KitchenMember m;
    m.id   = node.id();
    m.name = node.name();
    m.cls  = cls;
    m.class_logodds = class_logodds_of(cls);

    const auto& mat = rtmat->matrix();
    const float z0  = static_cast<float>(mat(2, 3));
    m.xy  = Eigen::Vector2f(static_cast<float>(mat(0, 3)), static_cast<float>(mat(1, 3)));
    m.yaw = std::atan2(static_cast<float>(mat(1, 0)), static_cast<float>(mat(0, 0)));

    // Geometry: cabinet_concept publishes width=along-run L, depth=carcass depth, height=z1−z0, and
    // parks the RT translation at the carcass BOTTOM — so the worktop plane is z0 + height.
    const float width  = G->get_attrib_by_name<width_m_att>(node).value_or(0.0f);
    const float depth  = G->get_attrib_by_name<depth_m_att>(node).value_or(0.0f);
    const float height = G->get_attrib_by_name<height_m_att>(node).value_or(0.0f);
    m.length  = width;
    m.depth   = depth;
    m.worktop = z0 + height;
    m.tier    = resolve_tier(cls, z0, m.worktop);

    // Published 6×6 SE3 covariance, row-major [x,y,z,rx,ry,rz]: xx at 0, yy at 7, YAW at 35. This is
    // the channel cabinet_concept's kitchen path was missing entirely until write_kitchen_rt_covariance
    // landed — without it every run below falls to the wide no-covariance fallback and the frame fit
    // cannot tell a 2.2 m polygon-pinned run from a 0.4 m self-derived stub.
    if (const auto cov = G->get_attrib_by_name<rt_covariance_att>(edge.value());
        cov.has_value() and cov.value().get().size() >= 36)
    {
        const auto& c = cov.value().get();
        m.var_yaw = c[35];
        m.has_cov = true;
        m.var_x   = c[0];
        m.var_y   = c[7];
        var_z0    = c[14];
    }
    else
    {
        // Stay honest: a WIDE fallback so an uncalibrated member barely influences the frame, rather
        // than a tight default that would let a newborn peer dominate the level-2 belief.
        m.var_yaw = rc::KitchenMember::kNoCovVarYaw;
        m.has_cov = false;
        m.var_x = m.var_y = 1.0f;
    }

    // ESTIMATION variance of this member's own size — the producer's OWN posterior when it publishes
    // one (cortex `object_size_variance` = [var_width, var_depth, var_height], m²), else a
    // length-scaled stand-in.
    //
    // ★This is the term that decides whether a badly-fitted member can drag the shared value. With the
    // stand-in the weight comes from run LENGTH, and length is a poor proxy for fit quality — a 2.25 m
    // run that fitted its top 14 cm low still got a small variance and pulled the consensus worktop
    // ~5 cm below the majority. With the real marginal, that run down-weights itself.
    //
    // worktop = z0 + height, and for a floor-standing unit z0 is pinned to the floor with a variance
    // far smaller than the height's, so var(worktop) ≈ var(height); the RT z variance is added for a
    // wall-mounted unit, where the mounting height is genuinely uncertain.
    if (const auto sv = G->get_attrib_by_name<object_size_variance_att>(node);
        sv.has_value() and sv.value().get().size() >= 3)
    {
        const auto& v = sv.value().get();
        m.var_depth   = std::max(0.0f, v[1]);
        m.var_worktop = std::max(0.0f, v[2]) + (m.tier == rc::KitchenTier::Wall ? std::max(0.0f, var_z0) : 0.0f);
        m.has_size_cov = true;
    }
    else
    {
        const float len_ref = std::max(0.25f, m.length);
        m.var_worktop = cfg_.worktop_meas_std_m * cfg_.worktop_meas_std_m / len_ref;
        m.var_depth   = cfg_.depth_meas_std_m   * cfg_.depth_meas_std_m   / len_ref;
        m.has_size_cov = false;
    }

    // POLYGON-PINNED? A wall-anchored run inherits its yaw from the room polygon and publishes the
    // merge-tolerance variance for it; a self-derived island publishes its measured (much looser)
    // misalignment. That difference is what makes the structural cavity possible — see the note on
    // compute_cavity_priors.
    m.pinned = m.has_cov and
               (kRad2Deg * std::sqrt(std::max(0.0f, m.var_yaw)) <= cfg_.pinned_yaw_std_max_deg);
    return m;
}

// Poll the members: ONE get_nodes_by_type("object") filtered by class. Read-only, main-thread, no
// graph-update signals connected.
SpecificWorker::MemberSnapshot SpecificWorker::poll_members()
{
    MemberSnapshot s;
    if (not G)
        return s;

    for (const auto& node : G->get_nodes_by_type("object"))
    {
        ++s.objects_total;
        const std::string cls = object_class_of(node);
        if (cls.empty())
        {
            ++s.skipped_class;
            if (s.unclassified.size() < 4)
                s.unclassified.push_back(node.name());
            continue;
        }
        auto m = read_member(node, cls);
        if (not m.has_value()) { ++s.skipped_no_rt; continue; }
        if (not m->has_cov)    ++s.no_cov;
        if (m->pinned)         s.pinned.push_back(*m);
        s.members.push_back(std::move(*m));
    }
    return s;
}

// ─── Frame belief (M2: fits and logs, writes nothing) ────────────────────────

void SpecificWorker::step_kitchen_belief(const MemberSnapshot& s)
{
    if (s.members.empty())
        return;

    // The AXIS is fitted from polygon-pinned members only when configured (the structural cavity).
    // If none are pinned yet, fall back to all members — a frame from self-derived axes is weaker,
    // and it says so through its own posterior spread rather than through a gate.
    const auto& axis_src = (cfg_.axis_from_pinned_only and not s.pinned.empty()) ? s.pinned : s.members;
    if (not kitchen_belief_.update(s.members, axis_src))
        return;

    kitchen_belief_.update_log_odds(s.members);
    compute_cavity_priors(s);
}

// The message sent DOWN to member i must come from a frame fitted WITHOUT i. See the header for why
// this schema also gets a STRUCTURAL down-date the ring could not have.
void SpecificWorker::compute_cavity_priors(const MemberSnapshot& s)
{
    cavity_.clear();
    for (const auto& m : s.members)
    {
        // Refit both the axis source and the scalar DOFs with this member removed. For a pinned
        // member the axis part is a genuine leave-one-out; for an unpinned one (the island) it is
        // usually a no-op on the axis — it was never in that evidence set — but it still matters for
        // the shared worktop and depth, which every member does contribute to.
        std::vector<rc::KitchenMember> others, others_axis;
        others.reserve(s.members.size());
        for (const auto& o : s.members)
            if (o.id != m.id) others.push_back(o);
        const auto& axis_pool = (cfg_.axis_from_pinned_only and not s.pinned.empty()) ? s.pinned : s.members;
        for (const auto& o : axis_pool)
            if (o.id != m.id) others_axis.push_back(o);

        if (others.empty() or others_axis.empty())
            continue;   // a frame of one cannot tell its only member anything it did not already say

        rc::KitchenBelief cav(kitchen_belief_.params());
        cav.set_log_odds(kitchen_belief_.log_odds());   // the existence evidence is not per-member
        if (not cav.update(others, others_axis))
            continue;
        cavity_[m.id] = kitchen_belief_.prior_for(m, cav);
    }
}

// ─── The OUTLINE (step 1: measures joints, corrects nothing) ─────────────────

// Centroid of the room polygon — used ONLY to decide which side of a run its front face is on.
// Get it wrong for a wall run and the face line lands on the wall side, moving every joint by a full
// carcass depth, so it is worth taking from the actual polygon rather than assuming a winding.
std::optional<Eigen::Vector2f> SpecificWorker::room_interior() const
{
    if (not G or room_node_id_ == 0) return std::nullopt;
    const auto room = G->get_node(room_node_id_);
    if (not room.has_value()) return std::nullopt;
    const auto px = G->get_attrib_by_name<delimiting_polygon_x_att>(room.value());
    const auto py = G->get_attrib_by_name<delimiting_polygon_y_att>(room.value());
    if (not px.has_value() or not py.has_value()) return std::nullopt;
    const auto& xs = px.value().get();
    const auto& ys = py.value().get();
    const std::size_t n = std::min(xs.size(), ys.size());
    if (n < 3) return std::nullopt;
    double cx = 0.0, cy = 0.0;
    for (std::size_t i = 0; i < n; ++i) { cx += xs[i]; cy += ys[i]; }
    return Eigen::Vector2f(static_cast<float>(cx / n), static_cast<float>(cy / n));
}

void SpecificWorker::step_outline(const MemberSnapshot& s)
{
    const auto interior = room_interior();
    if (not interior.has_value() or s.members.empty())
        return;

    // Two chains, because a floor unit and a wall unit are at different heights and joining them
    // would invent a corner where there is only a cabinet above another cabinet.
    std::vector<rc::OutlineSeg> floor_segs, wall_segs;
    for (const auto& m : s.members)
    {
        rc::OutlineSeg g;
        g.name   = m.name;
        g.centre = m.xy;
        g.u      = Eigen::Vector2f(std::cos(m.yaw), std::sin(m.yaw));
        g.n      = Eigen::Vector2f(-g.u.y(), g.u.x());
        // Point the normal INTO the room, so `centre + d/2·n` is the surface actually seen.
        if (g.n.dot(*interior - g.centre) < 0.0f) g.n = -g.n;
        g.length = m.length;
        g.depth  = m.depth;
        (m.tier == rc::KitchenTier::Wall ? wall_segs : floor_segs).push_back(g);
    }
    outline_floor_.set_segments(std::move(floor_segs));
    outline_wall_.set_segments(std::move(wall_segs));
}

// ─── Diagnostics ─────────────────────────────────────────────────────────────

// Per-joint record. A row is a pair of runs that ought to meet, the vertex they should share, and
// how far each one's end is from it: POSITIVE is a hole, NEGATIVE an overlap.
void SpecificWorker::log_outline_csv()
{
    if (not outline_csv_.is_open())
        return;
    // NB: not named `emit` — that is a Qt macro that expands to nothing, so the declaration would
    // silently collapse to `const auto = ...`.
    const auto write_tier = [&](const char* tier, const rc::KitchenOutline& o)
    {
        for (const auto& j : o.joints())
        {
            const auto& a = o.segments()[static_cast<std::size_t>(j.i)];
            const auto& b = o.segments()[static_cast<std::size_t>(j.j)];
            outline_csv_ << cycle_ << ',' << tier << ',' << a.name << ',' << b.name << ','
                         << j.vertex.x() << ',' << j.vertex.y() << ','
                         << j.gap_i << ',' << j.gap_j << ','
                         << (j.end_i_high ? 1 : 0) << ',' << (j.end_j_high ? 1 : 0) << ','
                         << j.cross << '\n';
        }
    };
    write_tier("floor", outline_floor_);
    write_tier("wall",  outline_wall_);
    outline_csv_.flush();
}

void SpecificWorker::log_members_csv(const MemberSnapshot& s)
{
    if (not csv_.is_open())
        return;
    for (const auto& m : s.members)
        csv_ << cycle_ << ',' << m.name << ',' << m.cls << ',' << m.class_logodds << ','
             << kRad2Deg * m.yaw << ',' << kRad2Deg * std::sqrt(std::max(0.0f, m.var_yaw)) << ','
             << (m.pinned ? 1 : 0) << ',' << m.worktop << ',' << m.depth << ',' << m.length << ','
             << tier_name(m.tier) << ',' << std::sqrt(std::max(0.0f, m.var_x)) << ','
             << std::sqrt(std::max(0.0f, m.var_y)) << ',' << (m.has_cov ? 1 : 0) << ','
             << (m.has_size_cov ? 1 : 0) << ',' << std::sqrt(std::max(0.0f, m.var_worktop)) << ','
             << std::sqrt(std::max(0.0f, m.var_depth)) << ','
             << kitchen_belief_.membership_prob(m) << '\n';
    csv_.flush();
}

void SpecificWorker::log_fit_csv(const MemberSnapshot& s)
{
    if (not fit_csv_.is_open() or not kitchen_belief_.seeded())
        return;

    const auto& st = kitchen_belief_.state();
    const auto& sg = kitchen_belief_.sigma();
    const float F  = kitchen_belief_.free_energy(s.members);
    const float Fa = kitchen_belief_.free_energy_if_adopted(s.members);
    for (const auto& m : s.members)
    {
        const auto it = cavity_.find(m.id);
        const rc::KitchenBelief::MemberPrior p = (it != cavity_.end()) ? it->second
                                                                       : rc::KitchenBelief::MemberPrior{};
        fit_csv_ << cycle_ << ',' << s.members.size() << ',' << kitchen_belief_.n_axis_members() << ','
                 << kitchen_belief_.n_worktop_members() << ',' << kitchen_belief_.n_tall_members() << ','
                 << kitchen_belief_.n_upper_members() << ',' << kitchen_belief_.log_odds() << ','
                 << kitchen_belief_.p_frame() << ',' << F << ',' << Fa << ',' << (Fa - F) << ','
                 << kRad2Deg * st.axis << ',' << kRad2Deg * std::sqrt(std::max(0.0f, sg.axis)) << ','
                 << st.worktop << ',' << std::sqrt(std::max(0.0f, sg.worktop)) << ','
                 << st.depth   << ',' << std::sqrt(std::max(0.0f, sg.depth))   << ','
                 << st.depth_tall << ',' << st.depth_upper << ','
                 << m.name << ',' << m.cls << ',' << kitchen_belief_.membership_prob(m) << ','
                 << (m.pinned ? 1 : 0) << ','
                 << kRad2Deg * m.yaw << ',' << kRad2Deg * p.axis_resid << ','
                 << kRad2Deg * p.yaw << ',' << p.kappa << ','
                 << m.worktop << ',' << p.worktop << ','
                 << m.depth   << ',' << p.depth   << '\n';
    }
    fit_csv_.flush();
}

// ─── GRAFCET / lifecycle leftovers ───────────────────────────────────────────

void SpecificWorker::emergency()
{
    std::print("kitchen_metaconcept: emergency()\n");
}

void SpecificWorker::restore()
{
    std::print("kitchen_metaconcept: restore()\n");
}

int SpecificWorker::startup_check()
{
    std::print("kitchen_metaconcept: startup_check()\n");
    return 0;
}
