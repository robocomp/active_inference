#pragma once
// The relocaliser's Task R1/R1b runtime state, one member of RoomConcept (rt_) — kept here so the agent's header does
// not grow with it. Localiser thread only, except `pending_passage` (RoomConcept::passage_mutex_).
#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <vector>
#include "passage_prior.h"
#include "door_apertures.h"
#include "reloc_search.h"
#include <algorithm>

namespace rc::reloc
{
    struct Runtime
    {
        /// The last search HELD the pose (msprt_decide → Hold): nothing decisive, the held pose alive; rivals carry it.
        bool last_search_held = false;
        /// A held episode is open: its rivals accumulate what the search could not decide; re-searching would re-seed
        /// them and discard that. Ends when they resolve, are rejected (lower SPRT bound) or empty, or P(lost) < ½.
        bool hold_open = false;
        std::uint64_t searches = 0;                              ///< searches fired by the lost belief (diagnostic)
        std::optional<rc::passage::Passage> pending_passage;     ///< set by note_passage_handover, consumed with the reset
        std::optional<rc::passage::Passage> passage;             ///< the passage of the last hand-over
        std::vector<rc::passage::Anchor>    door_anchor;         ///< forward / reversed components, carried by odometry
        Eigen::Matrix3f held_spread = Eigen::Matrix3f::Zero();   ///< while a held episode is open: Σ w_k d_k d_kᵀ
        int   door_anchor_frames = 0;                            ///< frames since: the kidnap hazard's exposure
        Eigen::Vector3f anchor_prev_est = Eigen::Vector3f::Zero();   ///< the pose the next prediction is built on
        bool  anchor_prev_valid = false;
        float passage_term = 0.f;                                ///< this frame's doorway term (0 = none)
        float view_beyond  = 0.f;                                ///< this frame's through-a-doorway fraction (Task R2)
        /// Nats of tracking evidence earned since the hand-over, on views the room owes and that are new (Task R2): the
        /// anchor retires once this exceeds the anchor's own information about the pose (anchor_information).
        double anchor_track_nats = 0.0;
    };

    /// The room's doorways as the SCAN sees them, ROOM (internal) frame: the door list, the crossed door at the geometry
    /// the crossing measured (Task R1b: the 1 Hz list misplaces it for ~1 s after a hand-over), each seated on the
    /// room's own boundary (DoorApertures::seat_in_room). One source for the tracker's per-beam weights, the
    /// relocaliser's likelihood and the lost belief's view-through-the-door fraction (Task R2).
    /// `passage_only`: also add the crossed door when the door list does not carry it (the tracker's weights do). The
    /// relocaliser and the view fraction pass false: they score a doorway only on door_concept's LIVE open/leaf
    /// belief — the passage alone carries the p_open of the crossing instant, never updated, and a doorway asserted open
    /// is strong evidence (every return at its line is charged as "should have been beyond").
    inline std::vector<DoorAperture> seated_doors(std::vector<DoorAperture> doors,
                                                  const std::optional<rc::passage::Passage>& passage,
                                                  const std::vector<Eigen::Vector2f>& poly, bool passage_only = true)
    {
        if (passage.has_value())
        {
            const auto& pg = *passage;
            const Eigen::Vector2f a = pg.centre - 0.5f * pg.width * pg.tangent, b = pg.centre + 0.5f * pg.width * pg.tangent;
            bool found = false;
            for (auto& d : doors)
                if (pg.door_id != 0 and d.id == pg.door_id) { d.a = a; d.b = b; found = true; }
            if (not found and passage_only)
            {
                DoorAperture d;
                d.a = a; d.b = b; d.id = pg.door_id; d.name = "passage";
                d.p_open = pg.p_open;
                d.prior_closed = pg.p_open < 0.f ? 0.5f : 1.f - std::clamp(pg.p_open, 0.f, 1.f);
                doors.push_back(std::move(d));
            }
        }
        for (auto& d : doors) DoorApertures::seat_in_room(d, poly);
        return doors;
    }

    /// The same doorways in the relocaliser's form: the mouth, its outward normal, the per-beam leaf-free fraction
    /// (1 − prior_closed), and P(the door is in fact shut) = 1 − (1 − h)·p_open — door_concept's passability belief
    /// (½ when unmeasured), with the hazard h as the chance the door changed since it was measured: the localiser's
    /// own "the world changed under the map" rate, so even a door believed open with certainty can be wrong once.
    inline std::vector<Aperture> reloc_apertures(const std::vector<DoorAperture>& seated, float hazard)
    {
        std::vector<Aperture> out;
        const float h = std::clamp(hazard, 0.f, 1.f);
        for (const auto& d : seated)
        {
            const float open = 1.f - std::clamp(d.prior_closed, 0.f, 1.f);
            if (not (open > 0.f)) continue;   // a door whose leaf covers its whole span: every beam is a room return
            const float p_open = d.p_open < 0.f ? 0.5f : std::clamp(d.p_open, 0.f, 1.f);
            out.push_back({d.a, d.b, d.n_out, open, 1.f - (1.f - h) * p_open});
        }
        return out;
    }

    /// The fraction of the scan the room does NOT owe: the mean, over its returns, of the tracker's own posterior that a
    /// return came from beyond an open doorway (1 − DoorApertures::weight) under `pose`. Room frame, points in the robot
    /// frame. 0 with no doorway in view.
    inline float view_beyond_fraction(const std::vector<DoorAperture>& seated, const Eigen::Vector3f& pose,
                                      const std::vector<Eigen::Vector3f>& pts_robot, float sigma_obs)
    {
        if (seated.empty() or pts_robot.empty()) return 0.f;
        const float c = std::cos(pose.z()), s = std::sin(pose.z());
        const Eigen::Vector2f o = pose.head<2>();
        double acc = 0.0;
        for (const auto& p : pts_robot)
        {
            const Eigen::Vector2f q(c * p.x() - s * p.y() + o.x(), s * p.x() + c * p.y() + o.y());
            acc += 1.0 - static_cast<double>(DoorApertures::weight(seated, o, q, sigma_obs));
        }
        return static_cast<float>(acc / static_cast<double>(pts_robot.size()));
    }

    /// The anchor's information about the pose, in nats: KL(A ‖ uniform over the room × the circle), mixed over its
    /// components — log(area·2π) − ½ log((2πe)³ |Σ_c|). Zero or less ⇔ it says no more than "somewhere in the room".
    inline double anchor_information(const std::vector<rc::passage::Anchor>& anchor, double area)
    {
        double info = 0.0;
        for (const auto& a : anchor)
        {
            const double det = std::max(1e-30, static_cast<double>(a.cov.determinant()));
            const double h = 0.5 * (3.0 * std::log(2.0 * M_PI * M_E) + std::log(det));
            info += static_cast<double>(a.w) * (std::log(std::max(area, 1e-6) * 2.0 * M_PI) - h);
        }
        return info;
    }

    /// A lost search's posterior over its modes and the held pose (RoomConcept::mode_posterior).
    struct ModePosterior { std::vector<double> log_post; int top_idx = 0, keep_idx = -1; bool keep_travelled = false; };
}   // namespace rc::reloc
