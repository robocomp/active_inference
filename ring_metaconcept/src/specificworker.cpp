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
 * SpecificWorker — ring_metaconcept agent (CONCEPT_AGENT_RECIPE.md §6, Milestone 1 scaffold)
 *
 * Chassis identical to the level-1 concept agents (presence + GRAFCET + crash-free exit).
 * The one deliberate difference is the FRONT END: instead of a mask/media consumer this
 * agent is a GRAPH-READER. Each compute() it polls the constituent member nodes (anchor +
 * ring, i.e. table + chair) straight from the DSR graph, on the MAIN thread, and connects
 * NONE of the update_node/update_edge signals — those fire on the FastDDS reader threads
 * and a DirectConnection slot there corrupts the heap (see CLAUDE.md / §6 Delta 1).
 *
 * This milestone reaches Operating and reads each member's room-frame pose together with
 * the covariance that member's own agent published on its RT edge — the meta-concept's
 * MEASUREMENTS. The RingBelief, evidence-based birth of dining_set, and the leave-one-out
 * down-prior come in M2–M4.
 */

#include "specificworker.h"

#include <QTimer>
#include <QCoreApplication>
#include <algorithm>
#include <array>
#include <cmath>
#include <print>
#include <cstdlib>   // std::_Exit — crash-free terminal shutdown
#include <thread>    // brief DDS flush before _Exit
#include <chrono>
#include <iostream>  // std::cout/cerr flush
#include <limits>
#include <string_view>
#include <unordered_set>
#include <vector>

// DSR attribute name tags — generated from dsr_attr_name.h
#include <dsr/api/dsr_api.h>

namespace {
// Fallback uncertainty for a member whose agent has not published an rt_covariance yet. Deliberately
// WIDE (1 m, 180°): an uncalibrated member should barely move the arrangement fit. A tight default
// here would be an invisible way for a newborn peer to dominate the level-2 belief.
constexpr float kNoCovVarM2  = 1.0f;    // m²
constexpr float kNoCovVarYaw = 9.87f;   // rad² ≈ (π)²

// Member yaw CONVENTION offset — see RingBeliefParams::member_yaw_offset for the full note.
// ChairBelief puts the backrest on the −y seat edge, so a chair FACES its local +y: the inward
// direction equals chair_yaw + π/2. Without this the facing residual reads ~90° for a chair that is
// in fact pointing straight at the table. Moves into RingBeliefParams when the belief lands.
constexpr float kMemberYawOffset = -1.5707963f;

// Signed angle difference wrapped to (−π, π].
inline float wrap_pi(float a)
{
    while (a >  static_cast<float>(M_PI)) a -= 2.0f * static_cast<float>(M_PI);
    while (a <= -static_cast<float>(M_PI)) a += 2.0f * static_cast<float>(M_PI);
    return a;
}
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

    // Compute → Waiting on start
    states["Compute"]->addTransition(states["Compute"].get(), SIGNAL(entered()), states["Waiting"].get());
    // Waiting → Operating when all required peers are ready
    states["Waiting"]->addTransition(this, SIGNAL(presenceReady()), states["Operating"].get());
    // Operating → Degraded when a required peer is lost
    states["Operating"]->addTransition(this, SIGNAL(presenceLost()), states["Degraded"].get());
    // Degraded → Waiting immediately (self-kill scheduled inside degraded_enter)
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
    std::print("ring_metaconcept: SpecificWorker destroyed.\n");
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

    // Crash-free terminal exit (matches the level-1 agents). After our cleanup, hard-exit
    // instead of returning into the Ice communicator teardown + C++ static destruction,
    // which run with undefined cross-TU order and abort.
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

// ─── Initialisation ──────────────────────────────────────────────────────────

void SpecificWorker::initialize()
{
    std::print("ring_metaconcept: initialize()\n");
    GenericWorker::initialize();

    // Ignore heavy payload attributes in local graph updates (we only read poses/geometry).
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att, laser_X_att, laser_Y_att, laser_Z_att>();

    if (not G)
    {
        qWarning() << "ring_metaconcept: DSR graph not available in initialize()";
        return;
    }

    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    // Colour this agent's node in the graph view by its live health: the coordinator already
    // publishes the presence lifecycle; this adds the external FSM axis (Initialize/Compute/
    // Emergency/Restore). Generic discovery via objectName(), so genericworker regeneration
    // cannot break it.
    presence_coordinator_.attach_state_machine(&statemachine);
    presence_coordinator_.set_transition_hooks({
        .request_presence_ready = [this]() { emit presenceReady(); },
        .request_presence_lost  = [this]() { emit presenceLost(); },
    });
    presence_coordinator_.set_peer_hooks({
        .on_peer_restarted = [](std::uint32_t id)
        {
            qInfo() << "[Presence] peer" << id << "restarted";
        },
        .on_optional_peer_lost = [this](const std::string& name, std::uint32_t id)
        {
            on_optional_peer_lost(name, id);
        },
        .on_optional_peer_ready = [this](const std::string& name, std::uint32_t id)
        {
            on_optional_peer_ready(name, id);
        },
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
                for (const auto& label : missing)
                    m += " " + QString::fromStdString(label);
                qInfo() << "[SM] -> Waiting (missing:" << m.trimmed() << ")";
            }
        },
        .on_operating_enter = [this]()
        {
            qInfo("[SM] -> Operating: all required peers present");
            // ★Second stale-sweep, once. The initialize() pass can run BEFORE nodes leaked by a
            // crashed previous run have finished syncing from the persistent server, so this
            // post-sync pass is the one that actually reaps them (CLAUDE.md).
            if (not operating_sweep_done_)
            {
                operating_sweep_done_ = true;
                remove_owned_dining_set_nodes();
            }
        },
        .on_operating_loop = [this]()
        {
            compute();
        },
        .on_degraded_enter = [this]()
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
                {
                    qInfo("[SM] required peers recovered during grace — staying alive");
                    return;
                }
                qWarning("[SM] required peer still missing after grace — shutting down cleanly");
                terminal_shutdown();
            });
        },
    });
    presence_coordinator_.start();

    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::terminal_shutdown, Qt::UniqueConnection);

    // ── §6 Delta 1: NO update_node/update_edge signal connects. This agent's input is the
    // graph itself; it is polled on the MAIN thread from compute(). Connecting those signals
    // (which fire on the FastDDS reader threads) is the DirectConnection heap-corruption crash.

    // RT_API resolves the room→member edge pose + covariance out of the edge's timestamp ring
    // buffer (rt_translation/rt_covariance are ring-buffered attrs — reading them raw is wrong).
    rt_api_      = G->get_rt_api();
    inner_eigen_ = G->get_inner_eigen_api();
    scene_graph_ = std::make_unique<rc::RingSceneGraph>(G, rt_api_.get(), cfg_);

    // Remove any leftover dining_set nodes from a previous (crashed) run — clean slate.
    remove_owned_dining_set_nodes();

    // ★Both streams are imbued with the CLASSIC locale. Qt calls setlocale(LC_ALL, "") at startup and
    // these machines run LANG=es_ES.UTF-8, where the decimal separator is a COMMA. std::ofstream
    // formats through the C++ global locale, which is "C" today — but one std::locale::global call
    // anywhere would silently start writing "0,53" into a comma-separated file. Cheap insurance
    // (CLAUDE.md, "Parsing numbers from files").
    if (not cfg_.csv_path.empty())
    {
        csv_.open(cfg_.csv_path, std::ios::out | std::ios::trunc);
        if (csv_.is_open())
        {
            csv_.imbue(std::locale::classic());
            csv_ << "cycle,rig,cluster,node,class,x,y,yaw_deg,std_x,std_y,std_yaw_deg,has_cov,"
                    "radius,bearing_deg,facing_resid_deg\n";
        }
        else
            qWarning() << "ring_metaconcept: cannot open CSV" << QString::fromStdString(cfg_.csv_path);
    }
    if (not cfg_.fit_csv_path.empty())
    {
        ring_csv_.open(cfg_.fit_csv_path, std::ios::out | std::ios::trunc);
        if (ring_csv_.is_open())
        {
            ring_csv_.imbue(std::locale::classic());
            ring_csv_ << "cycle,rig,rig_node,n_ring,slots,log_odds,p_ring,cx,cy,radius,yaw_deg,"
                         "anchor,anchor_w,node,member_yaw_deg,prior_yaw_deg,prior_std_deg,"
                         "correction_deg,cavity_shift_deg\n";
        }
        else
            qWarning() << "ring_metaconcept: cannot open fit CSV"
                       << QString::fromStdString(cfg_.fit_csv_path);
    }

    // Belief parameters from config; the room polygon supplies the null's support (see room_area_m2).
    // Held as a member and copied into every hypothesis born from now on.
    rc::RingBeliefParams bp;
    bp.sigma_slot_m      = cfg_.sigma_slot_m;
    bp.clutter_frac      = cfg_.clutter_frac;
    bp.occupancy_q       = cfg_.occupancy_q;
    bp.member_yaw_offset   = cfg_.member_yaw_offset_deg / 57.29578f;
    bp.facing_model_std    = cfg_.facing_model_std_deg / 57.29578f;
    bp.evidence_ema_alpha  = cfg_.evidence_ema_alpha;
    bp.room_area_m2        = room_area_m2().value_or(cfg_.room_area_m2_fallback);
    belief_params_ = bp;
    std::print("ring_metaconcept: null support = {:.1f} m² ({})\n", bp.room_area_m2,
               room_area_m2().has_value() ? "room polygon" : "config fallback");

    // ★The partition MIRRORS the arrangement model — there is no partition-only tunable, so the two
    // can never disagree about what a ring is. A linking distance in RingConfig would be exactly the
    // magic threshold CLAUDE.md forbids.
    partition_params_ = {bp.prior_radius_m, bp.prior_radius_std, bp.sigma_slot_m, bp.room_area_m2};

    // Both are pure Eigen — verify them in isolation at startup (recipe §3.10).
    rc::RingBelief::self_test();
    rc::partition_self_test();

    // Resolve room node (the shared frame the members live under).
    const auto rooms = G->get_nodes_by_type("room");
    if (not rooms.empty())
        room_node_id_ = rooms.front().id();
    else
        qWarning() << "ring_metaconcept: no room node found at startup";
}

// ─── Main compute loop ───────────────────────────────────────────────────────

void SpecificWorker::compute()
{
    if (not G)
        return;

    // Refresh room node id if not yet found
    if (room_node_id_ == 0)
    {
        const auto rooms = G->get_nodes_by_type("room");
        if (rooms.empty()) return;
        room_node_id_ = rooms.front().id();
    }

    ++cycle_;

    // §6 Delta 1 — graph-reader front end: poll the constituent members on the MAIN thread.
    const auto members = poll_members();

    // WHICH members form a rig, and which anchor centres each — inferred, not assumed. Then match
    // those clusters onto the hypotheses that already exist BEFORE anything touches the graph.
    const auto clusters = partition_snapshot(members);
    associate_and_maintain(clusters);

    log_members_csv(members, clusters);
    step_ring_beliefs(members, clusters);
    log_ring_csv(members, clusters);

    // Throttled report so the scaffold is observable (later milestones replace this with the
    // ring belief / evidence / down-prior). Prints on stdout (qInfo is RoboComp-filtered).
    if (cfg_.log_period_frames > 0 and (cycle_ % static_cast<std::uint64_t>(cfg_.log_period_frames)) == 0)
    {
        // Report the REJECTIONS alongside the counts: "0 and 0" only means the scene has no dining
        // furniture if we also know how many objects were there and why each was passed over.
        std::string others;
        for (const auto& n : members.unclassified)
            others += (others.empty() ? "" : " ") + n;

        std::print("[ring_metaconcept] members: {}={} {}={} | objects={} (other-class={} [{}], no-RT-pose={}) room_id={} clusters={} rigs={}\n",
                   cfg_.anchor_class, members.anchors.size(),
                   cfg_.ring_class,   members.ring.size(),
                   members.objects_total, members.skipped_class, others,
                   members.skipped_no_rt, room_node_id_, clusters.size(), hypotheses_.size());

        // One line per cluster. THIS is what makes the partition visible: which anchor each group of
        // chairs actually selected, and how much authority that anchor was given.
        for (std::size_t c = 0; c < clusters.size(); ++c)
        {
            const auto& cl = clusters[c];
            const std::string anchor = cl.anchor_id == 0
                ? std::string("none")
                : members.anchors[cl.anchor_index].name;
            std::print("    cluster#{}: {}={} anchor={} w={:.4g} (lambda={:+.2f})\n",
                       c, cfg_.ring_class, cl.member_ids.size(), anchor,
                       cl.anchor_weight, cl.anchor_logodds);
        }
    }

    // Overall compute()-cycle rate heartbeat (throttled inside FPSCounter).
    fps_counter_.print("[ring_metaconcept Compute]");
}

// ─── Graph-reader front end (§6 Delta 1) ─────────────────────────────────────

// Class of a generic "object" node. Every concept agent now publishes type()=="object" with the
// class in object_subtype (name prefixes table_*/chair_* unchanged). For a TABLE, object_subtype
// carries the SHAPE (round/square) instead of a class, so accept the attribute only when it names
// a class we know and otherwise fall through to the name prefix. Mirrors the retina's
// scene_processor.cpp object_class_of so both agents classify a node identically.
std::string SpecificWorker::object_class_of(const DSR::Node& node) const
{
    const std::array<std::string_view, 2> classes{cfg_.anchor_class, cfg_.ring_class};

    if (const auto s = G->get_attrib_by_name<object_subtype_att>(node); s.has_value())
        if (const std::string& sv = s.value();
            std::ranges::find(classes, sv) != classes.end())
            return sv;

    for (const auto cls : classes)
        if (node.name().rfind(cls, 0) == 0)   // NAME-prefix fallback
            return std::string(cls);

    return {};
}

// One member's room-frame pose + the covariance ITS OWN agent published. That published Σ is the
// measurement noise of this level-2 belief (§6 Delta 2) — an uncertain chair contributes weakly to
// the arrangement fit by construction, so there is no confidence gate to tune.
std::optional<RingMember> SpecificWorker::read_member(const DSR::Node& node, const std::string& cls)
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

    RingMember m;
    m.id   = node.id();
    m.name = node.name();
    m.cls  = cls;

    const auto& mat = rtmat->matrix();
    m.xy  = Eigen::Vector2f(static_cast<float>(mat(0, 3)), static_cast<float>(mat(1, 3)));
    m.yaw = std::atan2(static_cast<float>(mat(1, 0)), static_cast<float>(mat(0, 0)));

    // Published 6×6 SE3 covariance, row-major [x,y,z,rx,ry,rz] — the layout table/chair WRITE
    // (see chair_scene_graph.cpp::write_rt_covariance): xx at 0, yy at 7, YAW at 35.
    if (const auto cov = G->get_attrib_by_name<rt_covariance_att>(edge.value());
        cov.has_value() and cov.value().get().size() >= 36)
    {
        const auto& c = cov.value().get();
        m.Sigma_xy << c[0], c[1], c[6], c[7];
        m.var_yaw  = c[35];
        m.has_cov  = true;
    }
    else
    {
        // No published Σ yet (a freshly-born member). Stay honest: a WIDE fallback so it barely
        // influences the arrangement, rather than a tight default that would let an uncalibrated
        // member dominate the fit.
        m.Sigma_xy = Eigen::Matrix2f::Identity() * kNoCovVarM2;
        m.var_yaw  = kNoCovVarYaw;
        m.has_cov  = false;
    }
    return m;
}

// Poll the constituent members: ONE get_nodes_by_type("object") filtered by class. Read-only,
// main-thread, no graph-update signals connected.
SpecificWorker::MemberSnapshot SpecificWorker::poll_members()
{
    MemberSnapshot s;
    if (not G)
        return s;

    const auto objects = G->get_nodes_by_type("object");
    s.objects_total = objects.size();

    for (const auto& node : objects)
    {
        const auto cls = object_class_of(node);
        if (cls.empty())
        {
            ++s.skipped_class;
            if (s.unclassified.size() < 6)
                s.unclassified.push_back(node.name());
            continue;
        }
        const auto m = read_member(node, cls);
        if (not m.has_value())
        {
            ++s.skipped_no_rt;
            continue;
        }
        if (cls == cfg_.anchor_class) s.anchors.push_back(*m);
        else                          s.ring.push_back(*m);
    }
    return s;
}

// ─── Arrangement belief (§6 Delta 2) — READ-ONLY this milestone ──────────────

// Area of the room polygon by the shoelace formula. The independent-objects null is a uniform over
// the room, so its density is 1/area and this number SCALES the ring-vs-null log-Bayes factor
// directly — a hard-coded constant here would silently set how eager the agent is to see rings.
std::optional<float> SpecificWorker::room_area_m2() const
{
    if (not G or room_node_id_ == 0)
        return std::nullopt;
    const auto room = G->get_node(room_node_id_);
    if (not room.has_value())
        return std::nullopt;

    const auto px = G->get_attrib_by_name<delimiting_polygon_x_att>(room.value());
    const auto py = G->get_attrib_by_name<delimiting_polygon_y_att>(room.value());
    if (not px.has_value() or not py.has_value())
        return std::nullopt;

    const auto& xs = px.value().get();
    const auto& ys = py.value().get();
    const std::size_t n = std::min(xs.size(), ys.size());
    if (n < 3)
        return std::nullopt;

    double a2 = 0.0;
    for (std::size_t i = 0, j = n - 1; i < n; j = i++)
        a2 += static_cast<double>(xs[j]) * ys[i] - static_cast<double>(xs[i]) * ys[j];
    const float area = static_cast<float>(std::abs(a2) * 0.5);
    return area > 1.0f ? std::optional<float>(area) : std::nullopt;
}

// One poll as evidence. Ring members are the "points"; each carries its OWN published variance as R
// (an uncertain chair contributes weakly, by construction). The anchor is NOT a point — it enters as
// the centre prior, so a table can never be mistaken for a chair occupying a seat.
rc::RingFrame SpecificWorker::build_ring_frame(const MemberSnapshot& members, const rc::Cluster& cluster,
                                               const rc::RingHypothesis& h) const
{
    rc::RingFrame f;
    f.points.reserve(cluster.member_index.size());
    f.R.reserve(cluster.member_index.size());

    // Only THIS cluster's members. Feeding every chair in the room to every rig is what made two
    // seating groups collapse into one bad fit.
    for (const std::size_t mi : cluster.member_index)
    {
        const auto& m = members.ring[mi];
        f.points.emplace_back(m.xy.x(), m.xy.y(), 0.0f);
        // The engine takes ONE scalar R per point while the peer publishes a 2×2. Use the mean
        // eigenvalue (trace/2) — exact for the isotropic case and a fair summary otherwise. The
        // residual is a scalar distance, so there is no direction for the anisotropy to act along.
        f.R.push_back(std::max(1e-6f, 0.5f * (m.Sigma_xy(0, 0) + m.Sigma_xy(1, 1))));
    }

    if (cluster.anchor_id != 0)
    {
        const auto& a = members.anchors[cluster.anchor_index];
        f.has_anchor  = true;
        f.anchor_xy   = a.xy;

        // ★The anchor asserts "the ring centre is HERE". That is a TENDENCY, not a law — the same
        // class of claim as facing_model_std — so the arrangement's own positional slack goes in in
        // quadrature, and the whole prior is SCALED by the probability that this anchor really is
        // this cluster's centre. A table the chairs did not select therefore contributes ~nothing,
        // continuously, with no gate anywhere.
        //
        // Before this the raw Σ⁻¹ (≈11 for σ=0.30 m) was applied unconditionally to whichever table
        // get_nodes_by_type("object") happened to return first. Measured live 2026-08-11: it welded
        // the centre onto a table 5 m from the chairs and held it there for 3000+ cycles, the radius
        // saturated against max_radius_m, and the rig was never born.
        Eigen::Matrix2f S = a.Sigma_xy;
        const float slack = belief_params_.sigma_slot_m * belief_params_.sigma_slot_m;
        S(0, 0) = std::max(1e-6f, S(0, 0) + slack);
        S(1, 1) = std::max(1e-6f, S(1, 1) + slack);
        f.anchor_info = cluster.anchor_weight * S.inverse();
    }

    f.slot_visibility = compute_slot_visibility(h.belief.state(), h.belief.slot_count());
    return f;
}

// Observability of each seat from where the robot is standing. This is what makes an EMPTY seat
// mean something: absence is evidence only where we looked. Soft edges on both FoV and range keep it
// continuous — a seat at the rim of the frame is partially observed, not a step from seen to unseen,
// so the slot-count evidence degrades smoothly as the robot turns away instead of snapping.
std::vector<float> SpecificWorker::compute_slot_visibility(const rc::RingBeliefState& s, int n_slots) const
{
    std::vector<float> vis;
    if (not cfg_.slot_visibility_enabled or not inner_eigen_)
        return vis;   // empty ⇒ the belief treats every slot as fully visible

    // ★Always check the optional: the chain bails to {} at any missing node/edge/rtmat.
    const auto room_T_zed = inner_eigen_->get_transformation_matrix("room", "zed", 0);
    if (not room_T_zed.has_value())
        return vis;

    const auto& m = room_T_zed->matrix();
    const Eigen::Vector2f cam(static_cast<float>(m(0, 3)), static_cast<float>(m(1, 3)));
    // The DSR `zed` frame looks along +y (see the zed camera frame convention), so the optical axis
    // in the room plane is the frame's second column.
    const Eigen::Vector2f axis(static_cast<float>(m(0, 1)), static_cast<float>(m(1, 1)));
    const float axis_yaw = std::atan2(axis.y(), axis.x());

    const float half_fov = 0.5f * cfg_.zed_hfov_deg / 57.29578f;
    const float fov_soft = std::max(1e-3f, cfg_.zed_fov_soft_deg / 57.29578f);
    const float rng_soft = std::max(1e-3f, cfg_.zed_range_soft_m);

    // Smooth 0→1 ramp, 0.5 at the boundary.
    const auto soft = [](float margin, float width)
    { return 0.5f * (1.0f + std::tanh(margin / width)); };

    vis.resize(static_cast<std::size_t>(n_slots));
    for (int k = 0; k < n_slots; ++k)
    {
        const float th = s.yaw + 2.0f * static_cast<float>(M_PI) * static_cast<float>(k)
                                 / static_cast<float>(n_slots);
        const Eigen::Vector2f slot(s.cx + s.radius * std::cos(th), s.cy + s.radius * std::sin(th));
        const Eigen::Vector2f d = slot - cam;
        const float range = d.norm();
        const float az    = wrap_pi(std::atan2(d.y(), d.x()) - axis_yaw);

        vis[static_cast<std::size_t>(k)] = soft(half_fov - std::abs(az), fov_soft)
                                         * soft(cfg_.zed_max_range_m - range, rng_soft);
    }
    return vis;
}

// GN on a circle has a narrow basin, so seed rather than starting from the prior mean: centre from
// the anchor (or the member centroid), radius from the mean member distance, phase so that slot 0
// lands on the first member.
rc::RingBeliefState SpecificWorker::seed_state_for(const MemberSnapshot& members,
                                                   const rc::Cluster& cluster,
                                                   const rc::RingFrame& frame) const
{
    Eigen::Vector2f centre = Eigen::Vector2f::Zero();
    if (frame.has_anchor)
        centre = frame.anchor_xy;
    else
    {
        for (const std::size_t mi : cluster.member_index) centre += members.ring[mi].xy;
        centre /= static_cast<float>(std::max<std::size_t>(1, cluster.member_index.size()));
    }

    float radius = 0.0f;
    for (const std::size_t mi : cluster.member_index) radius += (members.ring[mi].xy - centre).norm();
    radius /= static_cast<float>(std::max<std::size_t>(1, cluster.member_index.size()));

    const Eigen::Vector2f d0 = members.ring[cluster.member_index.front()].xy - centre;
    rc::RingBeliefState s;
    s.cx     = centre.x();
    s.cy     = centre.y();
    s.radius = std::clamp(radius, belief_params_.min_radius_m, belief_params_.max_radius_m);
    s.yaw    = std::atan2(d0.y(), d0.x());
    return s;
}

// ★A seed is a starting point, not a commitment. resolve_slot_count already refits from both the
// current state and a closed-form seed and keeps whichever explains the frame better; do the same for
// the whole state vector. The old one-shot `ring_seeded_` latch meant a seed taken on cycle 30 from
// the wrong anchor could never be revisited, and it wasn't — the centre was byte-identical 3000
// cycles later. Here a wrong seed is corrected by evidence instead of surviving forever.
void SpecificWorker::reseed_if_better(rc::RingHypothesis& h, const rc::RingFrame& frame,
                                      const MemberSnapshot& members, const rc::Cluster& cluster)
{
    const rc::RingBeliefState seed = seed_state_for(members, cluster, frame);

    if (not h.seeded)   // nothing to compare against yet
    {
        h.belief.set_state(seed);
        h.seeded = true;
        std::print("ring_metaconcept: rig={} seeded at ({:.2f},{:.2f}) r={:.2f} from {} member(s)\n",
                   h.key, seed.cx, seed.cy, seed.radius, cluster.member_index.size());
        return;
    }

    rc::RingBelief trial = h.belief;
    trial.set_state(seed);
    trial.update(frame);
    if (trial.log_evidence(frame, trial.state()) > h.belief.log_evidence(frame, h.belief.state()))
    {
        std::print("ring_metaconcept: rig={} RE-SEEDED ({:.2f},{:.2f}) r={:.2f} -> ({:.2f},{:.2f}) r={:.2f}\n",
                   h.key, h.belief.state().cx, h.belief.state().cy, h.belief.state().radius,
                   trial.state().cx, trial.state().cy, trial.state().radius);
        h.belief.set_state(trial.state());
    }
}

void SpecificWorker::step_ring_beliefs(const MemberSnapshot& members,
                                       const std::vector<rc::Cluster>& clusters)
{
    for (std::size_t c = 0; c < clusters.size(); ++c)
    {
        // associate_and_maintain has already guaranteed one hypothesis per cluster, in order.
        if (c < hypotheses_.size())
            step_hypothesis(hypotheses_[c], members, clusters[c]);
    }
}

void SpecificWorker::step_hypothesis(rc::RingHypothesis& h, const MemberSnapshot& members,
                                     const rc::Cluster& cluster)
{
    // Below two ring members the arrangement is under-determined. This is NOT an evidence gate — it
    // is the point at which the fit has fewer constraints than DOFs, so there is nothing to update.
    // (With 0-1 members the posterior would simply stay at the prior anyway; skipping saves the
    // process noise from random-walking the phase in the meantime.)
    if (cluster.member_index.size() < 2)
        return;

    const auto frame = build_ring_frame(members, cluster, h);

    // Re-seed on creation, and whenever this hypothesis's IDENTITY changed — a different attached
    // anchor or a different member set means the basin the belief converged into may no longer be
    // the right one.
    reseed_if_better(h, frame, members, cluster);

    h.belief.update(frame);
    // weight = 1.0: the ego-motion reliability term (a poll taken while the robot moves carries
    // smeared member poses) is not wired yet — see the CSV caveat.
    h.belief.resolve_slot_count(frame, 1.0f);
    // ★After the slot count, model-compare the radius — GN cannot be trusted with it (see
    // RingBelief::resolve_radius). Must run BEFORE update_log_odds so the existence evidence is
    // scored at the state we actually believe, not at the collapsed one.
    h.belief.resolve_radius(frame);
    h.belief.update_log_odds(frame, 1.0f);

    compute_cavity_priors(h, members, cluster, frame);
    publish_rig(h, members, cluster);
}

// ─── Partition + hypothesis lifecycle ────────────────────────────────────────

int SpecificWorker::hypothesis_hold_cycles() const
{
    // Derived, not tuned: do NOT churn faster than the evidence can change. The EMA rate is the
    // model's own statement of how long a revision takes, so retiring a hypothesis sooner than that
    // discards state the belief has not yet had the chance to revise.
    return std::max(1, static_cast<int>(std::ceil(1.0f / std::max(1e-3f, cfg_.evidence_ema_alpha))));
}

std::vector<rc::Cluster> SpecificWorker::partition_snapshot(const MemberSnapshot& members) const
{
    const auto to_partition = [](const std::vector<RingMember>& src)
    {
        std::vector<rc::PartitionMember> out;
        out.reserve(src.size());
        for (const auto& m : src)
            out.push_back({m.id, m.xy, std::max(1e-6f, 0.5f * (m.Sigma_xy(0, 0) + m.Sigma_xy(1, 1)))});
        return out;
    };
    return rc::partition_members(to_partition(members.ring), to_partition(members.anchors),
                                 partition_params_);
}

// Match clusters onto existing hypotheses, then reorder `hypotheses_` so that entry i belongs to
// cluster i. Nothing here touches the graph except the retirement of a hypothesis that has been
// missing for the whole hold period.
void SpecificWorker::associate_and_maintain(const std::vector<rc::Cluster>& clusters)
{
    std::vector<std::pair<std::uint64_t, std::vector<std::uint64_t>>> rigs;
    rigs.reserve(hypotheses_.size());
    for (const auto& h : hypotheses_)
        rigs.emplace_back(h.key, h.member_ids);

    const auto assoc = rc::associate_clusters(clusters, rigs);

    // Resolve every key to an INDEX up front, so nothing below ever inspects a moved-from element.
    const auto index_of = [this](std::uint64_t key) -> std::size_t
    {
        for (std::size_t i = 0; i < hypotheses_.size(); ++i)
            if (hypotheses_[i].key == key)
                return i;
        return hypotheses_.size();
    };

    std::vector<rc::RingHypothesis> next;
    next.reserve(clusters.size() + assoc.unmatched_rigs.size());

    // ★Entry i of `next` belongs to cluster i — step_ring_beliefs relies on that alignment.
    for (std::size_t c = 0; c < clusters.size(); ++c)
    {
        const auto key = assoc.cluster_to_rig[c];
        const std::size_t idx = key < 0 ? hypotheses_.size() : index_of(static_cast<std::uint64_t>(key));

        rc::RingHypothesis h;
        if (idx < hypotheses_.size())
        {
            h      = std::move(hypotheses_[idx]);   // keeps its belief state, log-odds EMA and node
            h.miss = 0;
        }
        else
        {
            h.key = ++next_rig_key_;
            h.belief.set_params(belief_params_);
        }
        // The membership is simply refreshed. A changed member set or anchor needs no flag: the seed
        // is re-evaluated against the evidence EVERY cycle (reseed_if_better), so a basin that stops
        // being the best explanation is left behind on its own.
        h.member_ids    = clusters[c].member_ids;
        h.anchor_id     = clusters[c].anchor_id;
        h.anchor_weight = clusters[c].anchor_weight;
        next.push_back(std::move(h));
    }

    // Hypotheses with no cluster this cycle: HOLD them (keeping node, state and log-odds) and only
    // retire once the hold period elapses. A member that flickers for one cycle must not birth and
    // kill a rig node — that churn is the ★hazard SCHEMA_GENERALITY_TODO.md §4 Step 1 calls out.
    const int hold = hypothesis_hold_cycles();
    for (const auto key : assoc.unmatched_rigs)
    {
        const std::size_t idx = index_of(key);
        if (idx >= hypotheses_.size())
            continue;
        rc::RingHypothesis h = std::move(hypotheses_[idx]);
        if (++h.miss >= hold)
        {
            // Deliberately NOT gated on cfg_.publish: in observe-only nothing was ever born, so this
            // is a provable no-op (RingSceneGraph keys on rigs_, which only ensure_rig_node fills).
            // Gating it would be worse than useless — it is the one call that reaps a node whose
            // hypothesis has died, and skipping it could strand one.
            if (scene_graph_)
                scene_graph_->remove_rig_node(h.key);
            std::print("ring_metaconcept: rig={} retired after {} cycles without members\n", h.key, h.miss);
            continue;   // dropped: not carried into `next`
        }
        next.push_back(std::move(h));   // held
    }

    hypotheses_ = std::move(next);
}

// Birth / refresh the rig node and push each member's CAVITY prior down its group_member edge.
//
// The existence decision is the model comparison's own boundary: log_odds > 0 means the arrangement
// explains the members better than the independent-objects null, < 0 means it does not. That is not
// a tuned threshold — it is where the two hypotheses are equally likely. The EMA (see
// RingBeliefParams::evidence_ema_alpha) is what keeps it from chattering across the boundary: the
// estimate moves slowly, so a marginal scene drifts rather than flickers.
void SpecificWorker::publish_rig(rc::RingHypothesis& h, const MemberSnapshot& members,
                                 const rc::Cluster& cluster)
{
    // ★Observe-only mode writes NOTHING to the graph — not even a retirement. Placed ahead of every
    // other branch for exactly that reason, and mirroring kitchen_metaconcept::publish_priors.
    if (not cfg_.publish or not scene_graph_)
        return;

    if (h.belief.log_odds() <= 0.0f)
    {
        scene_graph_->remove_rig_node(h.key);   // stopped out-explaining the null → retire it
        return;
    }

    if (scene_graph_->ensure_rig_node(h.key, h.belief, room_node_id_) == 0)
        return;

    std::unordered_set<std::uint64_t> live;

    // The ANCHOR is a member of the set too, and the membership edge should say so — a consumer or a
    // viewer reading the group otherwise sees chairs and no table, which misrepresents it. It carries
    // κ = 0: structurally a member, but no yaw prior, because "face the centre" is meaningless for the
    // object AT the centre (and a round table has no meaningful heading at all). κ = 0 is already the
    // contract's "ignore me", so table_concept needs no change to coexist with this. slot_index = −1
    // marks it as the anchor rather than a seat. Phase 2's rig_shape_round_logodds rides this edge.
    // ★Only when the chairs actually chose this table (w > 0.5 — "more likely this cluster's centre
    // than not", the same equal-probability boundary as log_odds > 0). An anchor the partition
    // attached with a near-zero weight is not a member of the set and must not be edged as one.
    if (cluster.anchor_id != 0 and cluster.anchor_weight > 0.5f)
    {
        rc::MemberPrior a;
        a.member_id  = cluster.anchor_id;
        a.slot_index = -1;
        a.yaw_prior  = 0.0f;
        a.yaw_kappa  = 0.0f;
        scene_graph_->publish_member_prior(h.key, a);
        live.insert(a.member_id);
    }

    for (const std::size_t mi : cluster.member_index)
    {
        const auto& m  = members.ring[mi];
        const auto  it = h.cavity.find(m.id);
        if (it == h.cavity.end())
            continue;   // no honest cavity message for this member → send none at all

        rc::MemberPrior p;
        p.member_id = m.id;
        p.yaw_prior = it->second.yaw;
        // ★Precision = p(ring) × 1/Var[ψ]. The model-averaging factor is what makes a shaky
        // arrangement push softly: it is not a confidence knob multiplied in, it is the probability
        // that the arrangement hypothesis holds at all. facing_var already carries the intrinsic
        // "chairs face the table" spread, which BOUNDS how hard this can ever pull.
        const float var = std::max(1e-6f, it->second.std_dev * it->second.std_dev);
        p.yaw_kappa = h.belief.p_ring() / var;

        // Which seat this member sits in (nearest slot) — for the Phase-2 radial prior and so a
        // consumer can tell two members apart when the rig has more seats than occupants.
        int   best_k = -1;
        float best_d = std::numeric_limits<float>::max();
        for (int k = 0; k < h.belief.slot_count(); ++k)
        {
            const float d = (h.belief.slot_xy(k) - m.xy).norm();
            if (d < best_d) { best_d = d; best_k = k; }
        }
        p.slot_index = best_k;
        if (best_k >= 0)
            p.slot_xy = h.belief.slot_xy(best_k);
        // Radial precision stays 0 until Phase 2 — the contract is that 0 means INERT, so a consumer
        // reading these fields today correctly ignores them.
        p.info_xx = 0.0f;
        p.info_yy = 0.0f;

        scene_graph_->publish_member_prior(h.key, p);
        live.insert(m.id);
    }
    scene_graph_->retain_members(h.key, live);   // departed chairs: κ→0, then drop the edge
}

// Leave-one-out: member i's down-prior is computed from an arrangement refitted WITHOUT i. The fit
// is position-only, so the self-confirmation being removed is i's own position pulling the centre
// toward itself — which would then be handed back to i as "face this way". Cheap: N tiny refits from
// the converged state, and N is the number of chairs.
void SpecificWorker::compute_cavity_priors(rc::RingHypothesis& h, const MemberSnapshot& members,
                                           const rc::Cluster& cluster, const rc::RingFrame& frame)
{
    h.cavity.clear();

    for (std::size_t i = 0; i < cluster.member_index.size(); ++i)
    {
        const auto& m = members.ring[cluster.member_index[i]];

        // The frame minus this member. The anchor stays: the table is not member i, and it is what
        // keeps the cavity centre well-conditioned when only one other chair remains.
        rc::RingFrame cf = frame;
        cf.points.clear();
        cf.R.clear();
        for (std::size_t j = 0; j < frame.points.size(); ++j)
            if (j != i)
            {
                cf.points.push_back(frame.points[j]);
                cf.R.push_back(frame.R[j]);
            }

        // Nothing left to constrain a centre — no honest cavity message exists, so send none.
        if (cf.points.empty() and not cf.has_anchor)
            continue;

        rc::RingBelief trial = h.belief;   // copies state + Σ; refit from the converged point
        trial.update(cf);

        rc::CavityPrior cp;
        cp.yaw     = trial.facing_yaw_for(m.xy);
        cp.std_dev = std::sqrt(trial.facing_var_for(m.xy, m.Sigma_xy));
        // How much the full fit was talking to itself about this member.
        cp.shift   = wrap_pi(cp.yaw - h.belief.facing_yaw_for(m.xy));
        h.cavity[m.id] = cp;
    }
}

// ─── Diagnostics ─────────────────────────────────────────────────────────────

// Per-member snapshot to CSV — the channel that actually gets read (stdout is for the heartbeat).
void SpecificWorker::log_members_csv(const MemberSnapshot& members, const std::vector<rc::Cluster>& clusters)
{
    if (not csv_.is_open() or cfg_.log_period_frames <= 0)
        return;
    if ((cycle_ % static_cast<std::uint64_t>(cfg_.log_period_frames)) != 0)
        return;

    constexpr float kRad2Deg = 57.29578f;

    // Which cluster each node belongs to, and that cluster's own centre. ★The structural residuals
    // below are now measured against the anchor THIS member's cluster selected — measuring every
    // chair against one arbitrary table is what produced the nonsensical radius=5.6 rows.
    struct Assign { std::int64_t rig = -1; std::int64_t cluster = -1; bool have_centre = false; Eigen::Vector2f centre; };
    std::unordered_map<std::uint64_t, Assign> assign;

    for (std::size_t c = 0; c < clusters.size(); ++c)
    {
        const auto& cl = clusters[c];
        Assign a;
        a.cluster = static_cast<std::int64_t>(c);
        a.rig     = c < hypotheses_.size() ? static_cast<std::int64_t>(hypotheses_[c].key) : -1;
        if (cl.anchor_id != 0)
        {
            a.have_centre = true;
            a.centre      = members.anchors[cl.anchor_index].xy;
        }
        else   // anchorless cluster: the members' own centroid is the only centre we can name
        {
            Eigen::Vector2f mean = Eigen::Vector2f::Zero();
            for (const std::size_t mi : cl.member_index) mean += members.ring[mi].xy;
            a.have_centre = not cl.member_index.empty();
            a.centre      = mean / static_cast<float>(std::max<std::size_t>(1, cl.member_index.size()));
        }
        for (const std::size_t mi : cl.member_index) assign[members.ring[mi].id] = a;
        if (cl.anchor_id != 0)
        {
            Assign anchor_row = a;
            anchor_row.have_centre = false;   // the anchor IS the centre; it has no radius of its own
            assign[cl.anchor_id]   = anchor_row;
        }
    }

    const auto row = [&](const RingMember& m)
    {
        const auto it = assign.find(m.id);
        const std::int64_t rig     = it == assign.end() ? -1 : it->second.rig;
        const std::int64_t cluster = it == assign.end() ? -1 : it->second.cluster;

        float radius = 0.0f, bearing = 0.0f, facing_resid = 0.0f;
        if (it != assign.end() and it->second.have_centre)
        {
            const Eigen::Vector2f d = m.xy - it->second.centre;
            radius       = d.norm();
            bearing      = std::atan2(d.y(), d.x());
            // Inward direction is bearing+π; convert to the member's yaw convention before comparing.
            facing_resid = wrap_pi(m.yaw - (bearing + static_cast<float>(M_PI) + kMemberYawOffset));
        }
        csv_ << cycle_ << ',' << rig << ',' << cluster << ','
             << m.name << ',' << m.cls << ','
             << m.xy.x() << ',' << m.xy.y() << ',' << m.yaw * kRad2Deg << ','
             << std::sqrt(std::max(0.0f, m.Sigma_xy(0, 0))) << ','
             << std::sqrt(std::max(0.0f, m.Sigma_xy(1, 1))) << ','
             << std::sqrt(std::max(0.0f, m.var_yaw)) * kRad2Deg << ','
             << (m.has_cov ? 1 : 0) << ','
             << radius << ',' << bearing * kRad2Deg << ',' << facing_resid * kRad2Deg << '\n';
    };
    for (const auto& m : members.anchors) row(m);
    for (const auto& m : members.ring)    row(m);
    csv_.flush();   // block-buffered to a file otherwise — a killed agent would lose the tail
}

// The arrangement fit, and — the point of this milestone — the down-prior it WOULD publish for each
// member, next to that member's CURRENT heading. `correction_deg` is what the rig would ask the chair
// to change; `prior_std_deg` is how strongly it would ask (the down-prior precision is
// p_ring/facing_var, so a wide σ here means the rig would barely push).
void SpecificWorker::log_ring_csv(const MemberSnapshot& members, const std::vector<rc::Cluster>& clusters)
{
    if (not ring_csv_.is_open() or cfg_.log_period_frames <= 0)
        return;
    if ((cycle_ % static_cast<std::uint64_t>(cfg_.log_period_frames)) != 0)
        return;

    constexpr float kRad2Deg = 57.29578f;

    for (std::size_t c = 0; c < clusters.size() and c < hypotheses_.size(); ++c)
    {
        const auto& cl = clusters[c];
        auto&       h  = hypotheses_[c];
        if (not h.seeded)
            continue;

        const auto& s = h.belief.state();
        const std::string anchor_name = cl.anchor_id == 0 ? std::string("-")
                                                          : members.anchors[cl.anchor_index].name;
        const std::string rig_node = scene_graph_ ? scene_graph_->rig_node_name(h.key) : std::string("-");

        for (const std::size_t mi : cl.member_index)
        {
            const auto& m = members.ring[mi];
            // The published prior is the CAVITY one (fitted without this member). Fall back to the
            // full fit only when no cavity message exists, and say so via cavity_shift = 0.
            const auto it = h.cavity.find(m.id);
            const float prior_yaw = (it != h.cavity.end()) ? it->second.yaw
                                                           : h.belief.facing_yaw_for(m.xy);
            const float prior_std = (it != h.cavity.end()) ? it->second.std_dev
                                                           : std::sqrt(h.belief.facing_var_for(m.xy, m.Sigma_xy));
            const float shift      = (it != h.cavity.end()) ? it->second.shift : 0.0f;
            const float correction = wrap_pi(prior_yaw - m.yaw);

            ring_csv_ << cycle_ << ',' << h.key << ',' << rig_node << ','
                      << cl.member_index.size() << ',' << h.belief.slot_count() << ','
                      << h.belief.log_odds() << ',' << h.belief.p_ring() << ','
                      << s.cx << ',' << s.cy << ',' << s.radius << ',' << s.yaw * kRad2Deg << ','
                      << anchor_name << ',' << cl.anchor_weight << ','
                      << m.name << ',' << m.yaw * kRad2Deg << ',' << prior_yaw * kRad2Deg << ','
                      << prior_std * kRad2Deg << ',' << correction * kRad2Deg << ','
                      << shift * kRad2Deg << '\n';
        }
    }
    ring_csv_.flush();
}

// ─── Initialisation helpers ──────────────────────────────────────────────────

void SpecificWorker::load_config(const ConfigLoader& cfg)
{
    cfg_ = rc::load_ring_config(cfg);
}

// ─── Lifecycle stubs ─────────────────────────────────────────────────────────

void SpecificWorker::emergency()
{
    std::print("ring_metaconcept: emergency()\n");
}

void SpecificWorker::restore()
{
    std::print("ring_metaconcept: restore()\n");
}

int SpecificWorker::startup_check()
{
    std::print("ring_metaconcept: startup_check()\n");
    return 0;
}
