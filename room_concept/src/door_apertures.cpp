#include <genericworker.h>   // FIRST: DSR's signal emitter is not self-contained

#include "door_apertures.h"

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <QDebug>
#include <QString>

#include <cmath>

namespace rc
{
    namespace
    {
        /// Do the segments p->p+r and q->q+s cross? Standard 2-D orientation test, written out rather
        /// than pulled from a helper so the degenerate cases are visible: parallel beams and
        /// zero-length apertures return false, which is the safe answer (no discount).
        bool segments_cross(const Eigen::Vector2f& p, const Eigen::Vector2f& r,
                            const Eigen::Vector2f& q, const Eigen::Vector2f& s)
        {
            const float rxs = r.x() * s.y() - r.y() * s.x();
            if (std::abs(rxs) < 1e-9f) return false;          // parallel or degenerate
            const Eigen::Vector2f qp = q - p;
            const float t = (qp.x() * s.y() - qp.y() * s.x()) / rxs;
            const float u = (qp.x() * r.y() - qp.y() * r.x()) / rxs;
            // t in (0,1): the crossing is BEFORE the return, i.e. the beam really did pass through the
            // opening on its way. A return that lands exactly in the doorway plane (t == 1) is the door
            // frame itself and keeps its weight.
            return t > 1e-4f and t < 1.f - 1e-4f and u >= 0.f and u <= 1.f;
        }
    }   // namespace

    std::vector<DoorAperture> DoorApertures::read_from_graph(DSR::DSRGraph& G, DSR::InnerEigenAPI& inner,
                                                             const std::string& room_frame)
    {
        std::vector<DoorAperture> out;
        // Doors are generic `object` nodes named door_* carrying object_subtype == "door"
        // (door_scene_graph.cpp). Every get_nodes_by_type("object") here MUST keep that filter, or a
        // fridge becomes a doorway.
        for (const auto& n : G.get_nodes_by_type("object"))
        {
            if (not n.name().starts_with("door")) continue;
            const auto sub = G.get_attrib_by_name<object_subtype_att>(n);
            if (not sub.has_value() or sub.value() != "door") continue;

            const auto w = G.get_attrib_by_name<width_m_att>(n);
            if (not w.has_value() or not std::isfinite(w.value()) or w.value() <= 0.f) continue;

            // room <- door. ts == 0: main thread only, and it returns nullopt at any missing link in
            // the chain rather than throwing — a door parented to a wall that has gone is simply not
            // an aperture this cycle.
            const auto T = inner.get_transformation_matrix(room_frame, n.name());
            if (not T.has_value()) continue;
            const Eigen::Matrix4d M = T.value().matrix();
            if (not M.allFinite()) continue;

            const Eigen::Vector2f c(static_cast<float>(M(0, 3)), static_cast<float>(M(1, 3)));
            // The door's own x axis is the wall tangent (phi == 0 yaw == wall tangent, door_geometry.h),
            // so the aperture spans width_m along it, centred on the node.
            const Eigen::Vector2f u(static_cast<float>(M(0, 0)), static_cast<float>(M(1, 0)));
            if (u.norm() < 1e-6f) continue;
            const Eigen::Vector2f dir = u.normalized() * (0.5f * w.value());

            DoorAperture ap;
            ap.a = c - dir;
            ap.b = c + dir;
            ap.id = n.id();
            ap.name = n.name();
            // Unmeasured publishes a sentinel (a large negative), so anything outside [0,1] means
            // "no evidence" and weighs as closed. Clamping instead of rejecting would turn a sentinel
            // into a confident claim.
            const auto p = G.get_attrib_by_name<door_open_prob_att>(n);
            ap.p_open = (p.has_value() and std::isfinite(p.value()) and p.value() >= 0.f and p.value() <= 1.f)
                            ? p.value() : 0.f;
            out.push_back(std::move(ap));
        }
        return out;
    }

    float DoorApertures::weight(const std::vector<DoorAperture>& apertures,
                                const Eigen::Vector2f& beam_origin, const Eigen::Vector2f& hit)
    {
        if (apertures.empty()) return 1.f;
        const Eigen::Vector2f r = hit - beam_origin;
        if (r.squaredNorm() < 1e-8f) return 1.f;
        float w = 1.f;
        for (const auto& ap : apertures)
        {
            if (ap.p_open <= 0.f) continue;                  // closed, or unmeasured: nothing to discount
            if (segments_cross(beam_origin, r, ap.a, ap.b - ap.a))
                w *= (1.f - ap.p_open);
        }
        return w;
    }
}   // namespace rc
