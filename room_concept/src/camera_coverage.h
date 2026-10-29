#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
// CAMERA COVERAGE OF THE ROOM'S BOUNDARY — what the door detector has been given a chance to see (Task 12, 2026-10-29).
//
// ★ WHY. The layout explorer's coverage gain counts space the LIDAR has seen (FreeSpacePolygon::visible_unknown_area),
//   and a room's outline is LiDAR-complete after two or three standpoints. Doors are not found by the LiDAR: door_concept
//   births them from CAMERA detections (YOLO on the ZED), which need the wall in the frustum at a usable range and
//   incidence. Live recog_proto2 leg 4: room_3's outline was mapped, the explorer finished, the room was "explored" —
//   and DOOR_2, on a wall no camera had faced, was never detected, so the mission `cross new` could not resolve.
//
// ★ THE MODEL. The boundary is cut into door-sized windows (cells). Each holds `seen` = the best P(detect | a door
//   there) any camera pose so far has offered it — the shared detection-weighted NBV model (common/nbv/viewpoint_score.h:
//   predicted fill on both image axes, the detector envelope, frustum + wall occlusion), the same one door_concept's
//   own look affordance (door_look.h) prices a look at a remembered door with. The best, not a product over frames:
//   frames from one viewpoint are not independent looks (door_look.h's absence channel makes the same argument), so a
//   window seen at P = 0.4 is seen at 0.4 until a BETTER view comes, not after ten more frames of the same one.
//
//   gain(v, ψ) = Σ_i len_i · max(0, P_i(v, ψ) − seen_i)
//              = metres of boundary a look from v along ψ would newly put before the detector (probability-weighted).
//   ψ = the ARRIVAL heading (robot → v): the view the camera will actually have when the robot gets there, whether or
//       not the controller turns to the published yaw — so a plan's prediction is what the next frames observe, and a
//       visit is credited by what the camera ACTUALLY saw (observe() runs on the real pose every frame).
//   price = ONE DOOR at the detector's ATTAINABLE P: door width · max_d P(detect | door head-on at range d) — the most
//       one door-width of wall can ever add (rc::nbv::standoff_band's peak). A trip that cannot newly show one door's
//       worth is not worth taking (the camera analogue of the LiDAR survey's footprint-area price); plan_coverage's own
//       `finished`. ★ Pricing at the attainable P, not P = 1: with the uncalibrated envelope P never exceeds ~0.43 for a
//       door, so a 1.8 m stub (two windows) would never have paid a full door width — exactly where doors sit.
//
// ★ CONSTANTS. The door prior is common/nbv/door_prior.h (door_look.h's values): what the detector looks for, not a
//   tuning knob. The detector envelope is the Sensor's (rc::detect defaults = door_concept's DoorModel.Detect*
//   defaults), uncalibrated there and here alike.
//
// Pure (Eigen + the common NBV header). tools/camera_coverage_test.cpp.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Dense>

#include "../../common/nbv/viewpoint_score.h"
#include "../../common/nbv/door_prior.h"

namespace rc::camcov
{
using Params = rc::nbv::DoorPrior;

struct Cell
{
    Eigen::Vector2f c{0.f, 0.f};   ///< window centre, on the outline
    Eigen::Vector2f u{1.f, 0.f};   ///< along the wall (unit)
    float len  = 0.f;              ///< boundary metres this window stands for
    float seen = 0.f;              ///< best P(detect | door there) any camera pose has offered
};

inline float wrap(float a) { return std::atan2(std::sin(a), std::cos(a)); }

class Coverage
{
public:
    explicit Coverage(Params p = {}) : p_(p) {}

    [[nodiscard]] const Params& params() const { return p_; }
    /// The best P(detect) a door window can ever be offered: head-on, at the envelope's best range (no occluder).
    [[nodiscard]] float attainable_p(const rc::nbv::Sensor& s) const
    {
        if (not s.complete()) return 0.f;
        rc::nbv::Target t;
        t.w = p_.w_m; t.h = p_.t_m; t.z0 = 0.f; t.z1 = p_.h_m;
        return rc::nbv::standoff_band(t, s, Eigen::Vector2f::Zero(), Eigen::Vector2f(0.f, 1.f), 0.f).peak_p;
    }
    /// One door's worth at the attainable P: the price of a trip.
    [[nodiscard]] float price(const rc::nbv::Sensor& s) const { return p_.w_m * attainable_p(s); }
    [[nodiscard]] const std::vector<Cell>& cells() const { return cells_; }
    [[nodiscard]] bool ready() const { return not cells_.empty(); }
    void reset() { cells_.clear(); occl_.clear(); outline_.clear(); }

    /// Cut `outline` (closed polygon) into door-width windows. An unchanged outline is a no-op; a changed one keeps,
    /// for each new window, the best `seen` of the old windows within half a door width of it on a parallel wall
    /// (the same piece of wall, re-estimated) — the camera's history is not thrown away when the layout moves.
    void set_outline(const std::vector<Eigen::Vector2f>& outline)
    {
        if (outline.size() < 3 or outline == outline_) return;
        std::vector<Cell> fresh;
        for (std::size_t k = 0; k < outline.size(); ++k)
        {
            const Eigen::Vector2f a = outline[k], b = outline[(k + 1) % outline.size()];
            const float L = (b - a).norm();
            if (not (L > 1e-3f)) continue;
            const Eigen::Vector2f u = (b - a) / L;
            const int n = std::max(1, static_cast<int>(std::lround(L / p_.w_m)));
            for (int i = 0; i < n; ++i)
            {
                Cell c;
                c.c = a + u * (L * (static_cast<float>(i) + 0.5f) / static_cast<float>(n));
                c.u = u;
                c.len = L / static_cast<float>(n);
                for (const auto& o : cells_)
                    if ((o.c - c.c).norm() <= 0.5f * p_.w_m and std::abs(o.u.dot(c.u)) > std::cos(std::numbers::pi_v<float> / 4.f))
                        c.seen = std::max(c.seen, o.seen);   // ↑ parallel = within half a right angle (a Manhattan outline's two classes)
                fresh.push_back(c);
            }
        }
        cells_ = std::move(fresh);
        outline_ = outline;
        occl_.clear();
        occl_.reserve(cells_.size());
        for (const auto& c : cells_)   // the walls, with THIS window punched out (the leaf sits inside its own wall)
        {
            const std::vector<rc::nbv::WallGap> gap{{c.c, 0.5f * p_.w_m}};
            occl_.push_back(rc::nbv::wall_obstacles(outline_, p_.t_m, gap));
        }
    }

    /// P(detect | a door in window i) from a camera at `pos` looking along `yaw` (outline frame).
    [[nodiscard]] float p_detect(std::size_t i, const Eigen::Vector2f& pos, float yaw, const rc::nbv::Sensor& s) const
    {
        const auto tg = target(i);
        // exact pruning (no threshold): P = envelope · visibility ≤ envelope, so a window the envelope alone rules out
        // needs no ray test
        const auto [fmax, fmin] = rc::nbv::predicted_fill_axes(tg, pos, s, yaw);
        if (not std::isfinite(fmax) or not (rc::detect::p_detect(fmax, fmin, 1.f, s.env) > 0.f)) return 0.f;
        const auto sc = rc::nbv::score({pos, yaw, 1.f, -1}, tg, s, occl_[i], 0.f);
        return std::isfinite(sc.p_detect) ? sc.p_detect : 0.f;
    }

    /// One camera frame: every window keeps the best detection chance it has been offered.
    void observe(const Eigen::Vector2f& pos, float yaw, const rc::nbv::Sensor& s)
    {
        if (not s.complete()) return;
        for (std::size_t i = 0; i < cells_.size(); ++i)
            cells_[i].seen = std::max(cells_[i].seen, p_detect(i, pos, yaw, s));
    }

    /// What a look from `v` along `yaw` would newly put before the detector: boundary metres, probability-weighted.
    [[nodiscard]] float look_gain(const Eigen::Vector2f& v, float yaw, const rc::nbv::Sensor& s) const
    {
        if (not s.complete()) return 0.f;
        float g = 0.f;
        for (std::size_t i = 0; i < cells_.size(); ++i)
            g += cells_[i].len * std::max(0.f, p_detect(i, v, yaw, s) - cells_[i].seen);
        return g;
    }

    /// Boundary metres the detector could still be shown: Σ len·max(0, attainable P − seen).
    [[nodiscard]] float residual_length(const rc::nbv::Sensor& s) const
    {
        const float pm = attainable_p(s);
        float a = 0.f;
        for (const auto& c : cells_) a += c.len * std::max(0.f, pm - c.seen);
        return a;
    }

    /// Boundary metres weighted by what is still unseen: Σ len·(1 − seen) (diagnostics).
    [[nodiscard]] float unseen_length() const
    {
        float a = 0.f;
        for (const auto& c : cells_) a += c.len * (1.f - c.seen);
        return a;
    }

private:
    [[nodiscard]] rc::nbv::Target target(std::size_t i) const
    {
        const Cell& c = cells_[i];
        rc::nbv::Target t;
        t.cx = c.c.x(); t.cy = c.c.y(); t.yaw = std::atan2(c.u.y(), c.u.x());
        t.w = p_.w_m; t.h = p_.t_m; t.z0 = 0.f; t.z1 = p_.h_m;
        return t;
    }

    Params p_;
    std::vector<Cell> cells_;
    std::vector<std::vector<rc::nbv::Obstacle>> occl_;
    std::vector<Eigen::Vector2f> outline_;
};

// ── WHO DRIVES afford_room ────────────────────────────────────────────────────────────────────────────────────────
struct OwnerInputs
{
    bool proto = false;              ///< startup_proto_
    bool promoted = false;           ///< startup_promoted_
    bool contested = false;          ///< the refiner holds contested cells
    bool recalled_partial = false;   ///< a partly remembered room
    bool layout_finished = false;    ///< the LiDAR survey's / recalled rest's latched verdict
    bool camera_open = false;        ///< camera model complete, boundary camera coverage not finished (a re-entry
                                     ///< resumes it from the room's camera memory)
};
/// The layout explorer drives afford_room for: the proto survey; a promoted room the refiner contests; a partly remembered
/// room until finished; and (Task 12) a promoted, freshly surveyed room until the cameras have covered its boundary.
/// Otherwise the pose planner (EpistemicPlanner) does, as before.
[[nodiscard]] inline bool explorer_owns_room(const OwnerInputs& in)
{
    return (in.proto and not in.promoted) or (in.promoted and in.contested)
        or (in.recalled_partial and not in.layout_finished)
        or (in.promoted and not in.recalled_partial and in.camera_open);
}
}   // namespace rc::camcov
