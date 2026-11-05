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
        std::uint64_t searches = 0;                              ///< multi-scan late-recall fits applied (diagnostic)
        std::optional<rc::passage::Passage> pending_passage;     ///< set by note_passage_handover, consumed with the reset
        std::optional<rc::passage::Passage> passage;             ///< the passage of the last hand-over
        std::vector<rc::passage::Anchor>    door_anchor;         ///< forward / reversed components, carried by odometry
        Eigen::Matrix3f held_spread = Eigen::Matrix3f::Zero();   ///< while rivals carry mass: Σ w_k d_k d_kᵀ about the pose
        int   door_anchor_frames = 0;                            ///< frames since: the kidnap hazard's exposure
        Eigen::Vector3f anchor_prev_est = Eigen::Vector3f::Zero();   ///< the pose the next prediction is built on
        bool  anchor_prev_valid = false;
        float passage_term = 0.f;                                ///< this frame's doorway term (0 = none)
    };

    /// The room's doorways as the SCAN sees them, ROOM (internal) frame: the door list, the crossed door at the geometry
    /// the crossing measured (Task R1b: the 1 Hz list misplaces it for ~1 s after a hand-over), each seated on the
    /// room's own boundary (DoorApertures::seat_in_room). One source for the tracker's per-beam weights, the
    /// relocaliser's likelihood.
    /// `passage_only`: also add the crossed door when the door list does not carry it (the tracker's weights do). The
    /// relocaliser passes false: it scores a doorway only on door_concept's LIVE open/leaf
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
                if (pg.door_id != 0 and d.id == pg.door_id) { d.a = a; d.b = b; d.sigma_along = pg.sig_t; found = true; }
            if (not found and passage_only)
            {
                DoorAperture d;
                d.a = a; d.b = b; d.id = pg.door_id; d.name = "passage";
                d.p_open = pg.p_open;
                d.sigma_along = pg.sig_t;
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
            out.push_back({d.a, d.b, d.n_out, open, 1.f - (1.f - h) * p_open, d.sigma_along});
        }
        return out;
    }

    /// Pose information of a Gaussian in nats: KL(N(·, Σ) ‖ uniform over the room × the circle) = log(area·2π) − H(N),
    /// H = ½ log((2πe)³ |Σ|). Zero or less ⇔ it says no more than "somewhere in the room".
    inline double gaussian_information(const Eigen::Matrix3f& cov, double area)
    {
        const double det = std::max(1e-30, static_cast<double>(cov.determinant()));
        return std::log(std::max(area, 1e-6) * 2.0 * M_PI) - 0.5 * (3.0 * std::log(2.0 * M_PI * M_E) + std::log(det));
    }

    /// The anchor's pose information, KL(A ‖ uniform), EXACT for components with disjoint support (forward and reversed
    /// through the door differ by π in yaw, σ_θ ≲ 0.3 rad): H(A) = Σ w_c H(N_c) + H(w), so KL = Σ w_c KL(N_c) − H(w).
    inline double anchor_information(const std::vector<rc::passage::Anchor>& anchor, double area)
    {
        double info = 0.0, hw = 0.0;
        for (const auto& a : anchor)
        {
            const double w = std::clamp(static_cast<double>(a.w), 0.0, 1.0);
            info += w * gaussian_information(a.cov, area);
            if (w > 0.0) hw -= w * std::log(w);
        }
        return info - hw;
    }
}   // namespace rc::reloc
