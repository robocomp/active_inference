#include "object_anchor_source.h"

#include <algorithm>
#include <cmath>
#include <print>
#include <string_view>

namespace rc
{
    namespace
    {
        // Marginalise a 6×6 SE(3) covariance [x,y,z,rx,ry,rz] down to SE(2) [x,y,yaw] (rows/cols 0,1,5).
        Eigen::Matrix3f marginal_se2(const DSR::Cov6d& c6)
        {
            constexpr int idx[3] = {0, 1, 5};
            Eigen::Matrix3f m;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    m(i, j) = static_cast<float>(c6(idx[i], idx[j]));
            return m;
        }

        // ─── Built-in reader: a furniture node (table_concept, refrigerator_concept, …) → SE(2) landmark ──
        // Every furniture agent now publishes a generic type()=="object" node carrying its class in
        // object_subtype (and, by convention, as its name prefix). This reader is registered under
        // "object" and gather() calls it for EVERY object node, so it first has to filter by class.
        //
        // Nothing below this gate was ever table-specific: the map anchor comes from the parent→node
        // Gaussian, the observation from the shared object_anchor contract, the freshness counter from
        // an attribute the fridge already writes too. Only the gate was.
        bool read_object_anchor(const DSR::Node& node, DSR::DSRGraph& G, DSR::InnerGaussianAPI& gauss,
                                const ObjectAnchorSource::Config& cfg, ObjectAnchorObs& out)
        {
            // 0. Class gate: which classes are admitted is config, not code (Config::subtypes).
            std::string matched;
            for (const auto& cls : cfg.subtypes)
            {
                if (std::string_view(node.name()).starts_with(cls)) { matched = cls; break; }
                if (const auto s = G.get_attrib_by_name<object_subtype_att>(node);
                    s.has_value() and s.value() == cls) { matched = cls; break; }
            }
            if (matched.empty())
                return false;

            // 1. Map anchor p_o + Σ_o (ROOM frame) via the chain-propagated Gaussian transform.
            //    Query in the object's PARENT (room) frame: the room→table edge covariance already
            //    folds the localization chain (table_concept), so we must NOT recompose through
            //    room→robot here — that would double-count the robot-pose uncertainty.
            const auto parent = G.get_attrib_by_name<parent_att>(node);
            if (not parent.has_value())
                return false;
            const auto parent_node = G.get_node(parent.value());
            if (not parent_node.has_value())
                return false;

            const auto g = gauss.get_transformation_gaussian(parent_node->name(), node.name());
            if (not g.has_value())
                return false;

            // GaussianPose3D: mean [x,y,z,rx,ry,rz], cov 6×6.  Reduce to SE(2) [x,y,yaw].
            out.pose_world = {static_cast<float>(g->mean(0)),
                              static_cast<float>(g->mean(1)),
                              static_cast<float>(g->mean(5))};
            const Eigen::Matrix3f map_cov = marginal_se2(g->covariance);
            out.map_pos_sigma = std::sqrt(std::max({0.0f, map_cov(0, 0), map_cov(1, 1)}));   // confidence in p_o

            // 2. Per-frame robot-frame measurement z_o (+ optional R_o) authored by the producer.
            //    No measurement this frame ⇒ no relative constraint can be formed ⇒ skip.
            //    size 2 = [x,y] position-only landmark (single-view yaw is biased); size ≥3 adds yaw.
            const auto obs_opt = G.get_attrib_by_name<obj_obs_robot_att>(node);
            if (not obs_opt.has_value() or obs_opt->get().size() < 2)
                return false;
            const auto& obs = obs_opt->get();
            const bool has_yaw = obs.size() >= 3;
            out.obs_robot       = {obs[0], obs[1], has_yaw ? obs[2] : 0.0f};
            out.has_orientation = has_yaw;

            // Measurement noise R_o (robot frame). Position (2×2) may be ANISOTROPIC — the producer can
            // publish a full symmetric 2×2 as obj_obs_robot_cov=[Rxx,Ryy,Rxy] (ray-loose, see
            // ray_anisotropic_cov.h): disambiguated by size (position-only obs + cov size 3 ⇒ full 2×2).
            Eigen::Matrix2f R_pos = cfg.meas_sigma_xy * cfg.meas_sigma_xy * Eigen::Matrix2f::Identity();
            float meas_var_yaw = cfg.meas_sigma_yaw * cfg.meas_sigma_yaw;
            if (const auto cov_opt = G.get_attrib_by_name<obj_obs_robot_cov_att>(node);
                cov_opt.has_value() and cov_opt->get().size() >= 2)
            {
                const auto& c = cov_opt->get();
                if (not has_yaw and c.size() >= 3)          // [Rxx, Ryy, Rxy] full anisotropic 2×2
                    R_pos << c[0], c[2], c[2], c[1];
                else                                         // diagonal [σx², σy²] (+ σyaw² if present)
                {
                    R_pos(0, 0) = c[0]; R_pos(1, 1) = c[1]; R_pos(0, 1) = R_pos(1, 0) = 0.f;
                    if (has_yaw and c.size() >= 3) meas_var_yaw = c[2];
                }
            }

            // ── Freshness as precision ──────────────────────────────────────────────────────────────
            // obj_obs_robot is only rewritten when the producer actually sees the table; between sightings
            // it PERSISTS (stale). Read the producer's own age counter (table_frames_since_detection) and
            // inflate R_o by (1 + age/age_scale)² so a stale observation lands with Λ≈0 — it fades out
            // smoothly instead of pulling the pose to a phantom "in front" reading. Fresh (age 0) ⇒ ×1.
            if (cfg.freshness_enable)
            {
                // Type-attributed read — compile-checked against dsr_attr_name.h (CLAUDE.md). The string +
                // Attribute::dec() form this replaced bypassed the declared type entirely.
                int age = 0;
                if (const auto v = G.get_attrib_by_name<table_frames_since_detection_att>(node); v.has_value())
                    age = std::max(0, v.value());
                const float scale = std::max(1e-3f, cfg.freshness_age_scale);
                const float f2 = (1.0f + static_cast<float>(age) / scale);   // σ→ f·σ  ⇒  var → f²·var
                R_pos *= f2 * f2;
                meas_var_yaw *= f2 * f2;
            }

            // 3. Innovation S = Σ_o ⊕ R_o, then Λ = S^{-1}. Σ_o carries the map's cross-terms (from the
            //    Gaussian chain); R_o's 2×2 position block may be anisotropic. Position-only ⇒ invert just
            //    the 2×2 block (proper marginal), yaw row/col of Λ left zero so yaw never enters the loss.
            Eigen::Matrix3f info = Eigen::Matrix3f::Zero();
            if (has_yaw)
            {
                Eigen::Matrix3f s = map_cov;
                s.topLeftCorner<2, 2>() += R_pos;
                s(2, 2) += meas_var_yaw;
                s += 1e-8f * Eigen::Matrix3f::Identity();      // numerical floor, not a behavioural gate
                info = s.inverse();
            }
            else
            {
                Eigen::Matrix2f s2 = map_cov.topLeftCorner<2, 2>() + R_pos;
                s2 += 1e-8f * Eigen::Matrix2f::Identity();
                info.topLeftCorner<2, 2>() = s2.inverse();
            }
            out.information = info;
            out.type    = matched;
            out.node_id = node.id();
            return true;
        }
    } // namespace

    ObjectAnchorSource::ObjectAnchorSource()
    {
        // Furniture nodes are generic type()=="object" now; the reader self-filters by Config::subtypes.
        register_type("object", &read_object_anchor);
    }

    void ObjectAnchorSource::register_type(const std::string& node_type, Reader reader)
    {
        readers_[node_type] = std::move(reader);
    }

    std::vector<ObjectAnchorObs> ObjectAnchorSource::gather(DSR::DSRGraph& G, DSR::InnerGaussianAPI& gauss,
                                                            const Config& cfg) const
    {
        std::vector<ObjectAnchorObs> anchors;
        if (not cfg.enable)
            return anchors;

        for (const auto& [type, reader] : readers_)
            for (const auto& node : G.get_nodes_by_type(type))
            {
                ObjectAnchorObs obs;
                if (not reader(node, G, gauss, cfg, obs))
                    continue;

                // ── PIN: break the circularity ──────────────────────────────────────────────────────
                // The live p_o (obs.pose_world) is re-derived from the robot pose each frame, so it always
                // agrees with room ⇒ zero information. Instead snapshot the world pose ONCE the map is
                // confident (σ < validate_sigma) and reuse that FIXED value as p_o thereafter. Now z_o vs
                // the pinned p_o carries a real residual when the pose drifts.
                auto it = pinned_pose_.find(obs.node_id);
                if (it == pinned_pose_.end())
                {
                    if (obs.map_pos_sigma > cfg.validate_sigma)
                        continue;   // not confident yet — don't emit a circular (un-pinned) anchor
                    it = pinned_pose_.emplace(obs.node_id, obs.pose_world).first;
                    std::print("[room][anchors] PINNED {} #{} at world ({:.2f},{:.2f},{:.2f}) (map σ={:.0f}mm)\n",
                               obs.type, obs.node_id, obs.pose_world.x(), obs.pose_world.y(),
                               obs.pose_world.z(), obs.map_pos_sigma * 1000.f);
                }
                obs.pose_world = it->second;   // use the FIXED pinned landmark, not the live (circular) pose
                anchors.push_back(std::move(obs));
            }

        // Drop pins for tables whose node no longer exists (moved/deleted → stale landmark).
        for (auto it = pinned_pose_.begin(); it != pinned_pose_.end(); )
            it = G.get_node(it->first).has_value() ? std::next(it) : pinned_pose_.erase(it);

        return anchors;
    }
} // namespace rc
