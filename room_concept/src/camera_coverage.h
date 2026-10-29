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
//   gain(v)  = max over headings ψ of  Σ_i len_i · max(0, P_i(v, ψ) − seen_i)
//            = metres of boundary a look from v would newly put before the detector (probability-weighted).
//   price    = one door width: a trip that cannot newly show at least one door's width of wall is not worth taking
//              (the camera analogue of the LiDAR survey's footprint-area price). plan_coverage's own `finished`.
//
// ★ CONSTANTS. The door prior (0.90 x 2.00 m, 5 cm leaf) is door_concept's (DoorPriorW/H, door_look.h Params): it is
//   what the detector looks for, not a tuning knob. The detector envelope is the Sensor's (rc::detect defaults =
//   door_concept's DoorModel.Detect* defaults), uncalibrated there and here alike.
//
// Pure (Eigen + the common NBV header). tools/camera_coverage_test.cpp.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Dense>

#include "../../common/nbv/viewpoint_score.h"

namespace rc::camcov
{
struct Params
{
    float door_w = 0.90f;   ///< m — door_concept's door prior width (DoorPriorW): the window a detection needs
    float door_h = 2.00f;   ///< m — its height (DoorPriorW/H)
    float wall_t = 0.05f;   ///< m — leaf/wall thickness (door_look.h thickness_m): occluder depth only
};

struct Cell
{
    Eigen::Vector2f c{0.f, 0.f};   ///< window centre, on the outline
    Eigen::Vector2f u{1.f, 0.f};   ///< along the wall (unit)
    float len  = 0.f;              ///< boundary metres this window stands for
    float seen = 0.f;              ///< best P(detect | door there) any camera pose has offered
};

struct Look
{
    float gain = 0.f;   ///< metres of boundary newly put before the detector (probability-weighted)
    float yaw  = 0.f;   ///< the heading that buys it (outline frame)
};

inline float wrap(float a) { return std::atan2(std::sin(a), std::cos(a)); }

class Coverage
{
public:
    explicit Coverage(Params p = {}) : p_(p) {}

    [[nodiscard]] const Params& params() const { return p_; }
    [[nodiscard]] float price() const { return p_.door_w; }
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
            const int n = std::max(1, static_cast<int>(std::lround(L / p_.door_w)));
            for (int i = 0; i < n; ++i)
            {
                Cell c;
                c.c = a + u * (L * (static_cast<float>(i) + 0.5f) / static_cast<float>(n));
                c.u = u;
                c.len = L / static_cast<float>(n);
                for (const auto& o : cells_)
                    if ((o.c - c.c).norm() <= 0.5f * p_.door_w and std::abs(o.u.dot(c.u)) > std::cos(std::numbers::pi_v<float> / 4.f))
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
            const std::vector<rc::nbv::WallGap> gap{{c.c, 0.5f * p_.door_w}};
            occl_.push_back(rc::nbv::wall_obstacles(outline_, p_.wall_t, gap));
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

    /// The best look from standpoint `v`: the heading among the windows' own bearings (each window centred in the
    /// frame) whose frustum holds the most still-unseen boundary, then scored exactly at that heading.
    [[nodiscard]] Look best_look(const Eigen::Vector2f& v, const rc::nbv::Sensor& s) const
    {
        Look out;
        if (not s.complete() or cells_.empty()) return out;
        const std::size_t n = cells_.size();
        std::vector<float> bearing(n), resid(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const Eigen::Vector2f d = cells_[i].c - v;
            bearing[i] = std::atan2(d.y(), d.x());
            resid[i] = cells_[i].len * std::max(0.f, p_detect(i, v, bearing[i], s) - cells_[i].seen);
        }
        const float half = 0.5f * s.hfov_rad;
        float best = -1.f;
        std::size_t arg = 0;
        for (std::size_t k = 0; k < n; ++k)
        {
            if (not (resid[k] > 0.f)) continue;
            float g = 0.f;
            for (std::size_t j = 0; j < n; ++j)
                if (std::abs(wrap(bearing[j] - bearing[k])) <= half) g += resid[j];
            if (g > best) { best = g; arg = k; }
        }
        if (best <= 0.f) return out;
        out.yaw = bearing[arg];
        for (std::size_t i = 0; i < n; ++i)
            out.gain += cells_[i].len * std::max(0.f, p_detect(i, v, out.yaw, s) - cells_[i].seen);
        return out;
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
        t.w = p_.door_w; t.h = p_.wall_t; t.z0 = 0.f; t.z1 = p_.door_h;
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
    bool camera_open = false;        ///< camera model complete, room not recalled, boundary camera coverage not finished
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
