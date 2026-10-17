// NOTE: no genericworker.h here, deliberately. It drags in dsr_gui.h -> QMainWindow, and this
// file is also linked into the offline bench (wall_slam_selftest), which has no Qt Widgets.
// Only pose_publisher.cpp needs that include, for DSR's signal emitter.
#include "door_apertures.h"
#include "../../common/room_resolve/room_resolve.h"   // rc::room::is_proto_mirror

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <QDebug>
#include <QString>

#include <algorithm>
#include <cmath>
#include <limits>

namespace rc
{
    namespace
    {
        /// Do the segments p->p+r and q->q+s cross? Standard 2-D orientation test, written out rather
        /// than pulled from a helper so the degenerate cases are visible: parallel beams and
        /// zero-length apertures return false, which is the safe answer (no discount).
        bool segments_cross(const Eigen::Vector2f& p, const Eigen::Vector2f& r,
                            const Eigen::Vector2f& q, const Eigen::Vector2f& s, float* t_out = nullptr)
        {
            const float rxs = r.x() * s.y() - r.y() * s.x();
            if (std::abs(rxs) < 1e-9f) return false;          // parallel or degenerate
            const Eigen::Vector2f qp = q - p;
            const float t = (qp.x() * s.y() - qp.y() * s.x()) / rxs;
            const float u = (qp.x() * r.y() - qp.y() * r.x()) / rxs;
            // t in (0,1): the crossing is BEFORE the return, i.e. the beam really did pass through the
            // opening on its way. A return that lands exactly in the doorway plane (t == 1) is the door
            // frame itself and keeps its weight.
            if (t_out != nullptr) *t_out = t;
            return t > 1e-4f and t < 1.f - 1e-4f and u >= 0.f and u <= 1.f;
        }
    }   // namespace

    std::vector<DoorAperture> DoorApertures::refresh(DSR::DSRGraph& G, DSR::InnerEigenAPI& inner,
                                                     const std::string& room_frame)
    {
        std::vector<DoorAperture> out;
        std::unordered_map<std::uint64_t, std::pair<Eigen::Vector2f, Eigen::Vector2f>> kept;
        tally_ = Tally{};

        // Doors are generic `object` nodes named door_* carrying object_subtype == "door"
        // (door_scene_graph.cpp). Every get_nodes_by_type("object") here MUST keep that filter, or a
        // fridge becomes a doorway.
        for (const auto& n : G.get_nodes_by_type("object"))
        {
            ++tally_.objects;
            if (not n.name().starts_with("door")) continue;
            ++tally_.named_door;
            const auto sub = G.get_attrib_by_name<object_subtype_att>(n);
            if (not sub.has_value() or sub.value() != "door") continue;
            // An ENTRY MIRROR under a proto-room is the same physical doorway as a door already here;
            // discounting it too would weigh the same beams twice (w = (1 - p_open)^2).
            if (rc::room::is_proto_mirror(G, n)) continue;
            ++tally_.subtype_ok;

            // ── THE CHEAP HALF: the door's own belief, read every cycle ──────────────────────────
            // Unmeasured publishes a sentinel outside [0,1]. ★ It no longer means "treat as shut" (2026-10-12):
            // that switched the filter off for the whole of room_2 after a crossing, because the door adopted
            // there is unmeasured until door_concept re-fits it (live try14: ~865 s, a ~8 cm pose bias along the
            // door normal). Unmeasured is now an even prior, and each return's range decides (weight()).
            const auto p = G.get_attrib_by_name<door_open_prob_att>(n);
            const bool measured = p.has_value() and std::isfinite(p.value()) and p.value() >= 0.f and p.value() <= 1.f;
            const float p_open = measured ? p.value() : -1.f;
            if (p_open > 0.f) ++tally_.believed_open;

            // ── THE EXPENSIVE HALF: walked ONCE per door, not once per cycle ─────────────────────
            auto it = cache_.find(n.id());
            if (it == cache_.end())
            {
                const auto w = G.get_attrib_by_name<width_m_att>(n);
                if (not w.has_value() or not std::isfinite(w.value()) or w.value() <= 0.f)
                { ++tally_.no_width; continue; }

                // room <- door, latest (ts == 0). Returns nullopt at any missing link in the chain
                // rather than throwing, so a door whose wall has gone is simply not an aperture yet —
                // check the optional, never dereference it (CLAUDE.md, and still true).
                const auto T = inner.get_transformation_matrix(room_frame, n.name());
                if (not T.has_value()) { ++tally_.no_transform; continue; }
                const Eigen::Matrix4d M = T.value().matrix();
                if (not M.allFinite()) { ++tally_.no_transform; continue; }

                const Eigen::Vector2f c(static_cast<float>(M(0, 3)), static_cast<float>(M(1, 3)));
                // The door's own x axis is the wall tangent (phi == 0 yaw == wall tangent,
                // door_geometry.h), so the aperture spans width_m along it, centred on the node.
                const Eigen::Vector2f u(static_cast<float>(M(0, 0)), static_cast<float>(M(1, 0)));
                if (u.norm() < 1e-6f) continue;
                const Eigen::Vector2f dir = u.normalized() * (0.5f * w.value());
                ++resolves_;
                qInfo().noquote() << QString("[room][doors] aperture resolved for %1: (%2,%3)-(%4,%5), "
                                             "width %6 m — walked once, reused from here")
                                         .arg(QString::fromStdString(n.name()))
                                         .arg((c - dir).x(), 0, 'f', 2).arg((c - dir).y(), 0, 'f', 2)
                                         .arg((c + dir).x(), 0, 'f', 2).arg((c + dir).y(), 0, 'f', 2)
                                         .arg(w.value(), 0, 'f', 2);
                it = cache_.emplace(n.id(), std::make_pair(c - dir, c + dir)).first;
            }

            kept.insert(*it);
            DoorAperture ap;
            ap.a = it->second.first;
            ap.b = it->second.second;
            ap.p_open = p_open;
            // ★ π FROM THE LEAF ANGLE, NOT FROM PASSABILITY (2026-10-12). door_open_prob is P(clear span >= the
            //   ROBOT's passage width) — the right question for `cross`, the wrong one for a beam: a door ajar at
            //   40° is impassable (p_open ≈ 0 ⇒ π = 1, every return kept) yet leaves a quarter of the aperture
            //   open to the next room. What a beam needs is the fraction of the aperture span the leaf covers,
            //   and the leaf (hinged on one jamb) projects w·cos φ onto the wall line — max(0, cos φ), which is
            //   0 past 90° and 1 only when the leaf is flush. door_phi_rad is the posterior MEAN door_concept
            //   publishes (−1 = unmeasured); p_open stays the fallback, ½ when neither is known.
            const auto phi = G.get_attrib_by_name<door_phi_rad_att>(n);
            if (phi.has_value() and std::isfinite(phi.value()) and phi.value() >= 0.f and phi.value() <= float(M_PI))
                ap.prior_closed = std::max(0.f, std::cos(phi.value()));
            else
                ap.prior_closed = measured ? 1.f - p_open : 0.5f;
            ap.half_depth = 0.5f * G.get_attrib_by_name<depth_m_att>(n).value_or(0.04f);
            ap.id = n.id();
            ap.name = n.name();
            ++tally_.emitted;
            out.push_back(std::move(ap));
        }
        // Doors that left the graph leave the cache with them; nothing here expires on a timer.
        cache_.swap(kept);
        return out;
    }

    void DoorApertures::seat_in_room(DoorAperture& ap, const std::vector<Eigen::Vector2f>& polygon)
    {
        if (polygon.size() < 3) return;
        const Eigen::Vector2f c = 0.5f * (ap.a + ap.b);
        // The room's wall at this doorway = the polygon edge closest to the aperture's centre. If the layout drew the
        // doorway as a notch, that edge IS the aperture line and δ = 0: exactly the behaviour before this existed.
        float best = std::numeric_limits<float>::infinity();
        Eigen::Vector2f p0, p1;
        for (std::size_t i = 0; i < polygon.size(); ++i)
        {
            const Eigen::Vector2f& e0 = polygon[i];
            const Eigen::Vector2f& e1 = polygon[(i + 1) % polygon.size()];
            const Eigen::Vector2f e = e1 - e0;
            const float L2 = e.squaredNorm();
            if (L2 < 1e-12f) continue;
            const float t = std::clamp((c - e0).dot(e) / L2, 0.f, 1.f);
            if (const float dist = (c - (e0 + t * e)).norm(); dist < best) { best = dist; p0 = e0; p1 = e1; }
        }
        if (not std::isfinite(best)) return;
        const Eigen::Vector2f u = (p1 - p0).normalized();
        const auto onto = [&](const Eigen::Vector2f& q) -> Eigen::Vector2f { return p0 + (q - p0).dot(u) * u; };
        ap.recess = (c - onto(c)).norm();
        ap.a = onto(ap.a);
        ap.b = onto(ap.b);
    }

    float DoorApertures::weight(const std::vector<DoorAperture>& apertures,
                                const Eigen::Vector2f& beam_origin, const Eigen::Vector2f& hit, float sigma_obs)
    {
        if (apertures.empty()) return 1.f;
        const Eigen::Vector2f r = hit - beam_origin;
        const float range = r.norm();
        if (range < 1e-4f) return 1.f;
        float w = 1.f;
        for (const auto& ap : apertures)
        {
            float t = 0.f;
            if (not segments_cross(beam_origin, r, ap.a, ap.b - ap.a, &t)) continue;
            // d = how far past the room's boundary (the mouth) the return landed (the crossing is at range t·|r|).
            const float d   = (1.f - t) * range;
            const float var = sigma_obs * sigma_obs + ap.half_depth * ap.half_depth;
            const float sd  = std::sqrt(var);
            // ★ TWO QUESTIONS, NOT ONE (2026-10-12, see seat_in_room). (1) Which surface returned this beam: the leaf,
            //   which sits δ = recess past the mouth (d ~ N(δ, var) truncated to d > 0 by the crossing test), or the
            //   next room (uniform over kBeyondExtentM)? (2) Does the room's SDF model that surface? It models a
            //   surface AT THE MOUTH; the leaf is one only insofar as its line coincides with the mouth — the
            //   Bhattacharyya overlap of N(0, var) and N(δ, var), exp(−δ²/8var). δ = 0 reproduces the old weight
            //   exactly (half-normal, overlap 1). A shut door recessed into a thick wall no longer pins the room's
            //   boundary 20 cm outside its own face: its returns are real, but they are not THIS polygon's wall.
            const float overlap = std::exp(-ap.recess * ap.recess / (8.f * var));
            if (ap.prior_closed >= 1.f) { w *= overlap; continue; }   // measured shut: the leaf is certain
            const float z       = (d - ap.recess) / sd;
            const float trunc   = 0.5f * std::erfc(-ap.recess / (sd * float(M_SQRT2)));   // P(d > 0 | leaf)
            const float surface = ap.prior_closed * std::exp(-0.5f * z * z) / (std::sqrt(2.f * float(M_PI)) * sd * trunc);
            const float beyond  = (1.f - ap.prior_closed) / kBeyondExtentM;
            w *= overlap * surface / (surface + beyond);
        }
        return w;
    }
}   // namespace rc
