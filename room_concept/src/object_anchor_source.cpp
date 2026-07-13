#include "object_anchor_source.h"

#include <algorithm>
#include <cmath>
#include <print>

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

        // ─── Built-in reader: DSR "table" node (table_concept) → SE(2) pose landmark ───────────────
        bool read_table_anchor(const DSR::Node& node, DSR::DSRGraph& G, DSR::InnerGaussianAPI& gauss,
                               const ObjectAnchorSource::Config& cfg, ObjectAnchorObs& out)
        {
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

            Eigen::Vector3f meas_var{cfg.meas_sigma_xy * cfg.meas_sigma_xy,
                                     cfg.meas_sigma_xy * cfg.meas_sigma_xy,
                                     cfg.meas_sigma_yaw * cfg.meas_sigma_yaw};
            if (const auto cov_opt = G.get_attrib_by_name<obj_obs_robot_cov_att>(node);
                cov_opt.has_value() and cov_opt->get().size() >= 2)
            {
                const auto& c = cov_opt->get();
                meas_var[0] = c[0];
                meas_var[1] = c[1];
                if (c.size() >= 3) meas_var[2] = c[2];
            }

            // 3. Innovation S = Σ_o ⊕ R_o, then Λ = S^{-1}. Σ_o carries the map's cross-terms (from the
            //    Gaussian chain); R_o is diagonal. Position-only ⇒ invert just the 2×2 block (proper
            //    marginal), yaw row/col of Λ left zero so the yaw residual never enters the loss.
            Eigen::Matrix3f info = Eigen::Matrix3f::Zero();
            if (has_yaw)
            {
                Eigen::Matrix3f s = map_cov;
                s.diagonal() += meas_var;
                s += 1e-8f * Eigen::Matrix3f::Identity();      // numerical floor, not a behavioural gate
                info = s.inverse();
            }
            else
            {
                Eigen::Matrix2f s2 = map_cov.topLeftCorner<2, 2>();
                s2.diagonal() += meas_var.head<2>();
                s2 += 1e-8f * Eigen::Matrix2f::Identity();
                info.topLeftCorner<2, 2>() = s2.inverse();
            }
            out.information = info;
            out.type    = "table";
            out.node_id = node.id();
            return true;
        }
    } // namespace

    ObjectAnchorSource::ObjectAnchorSource()
    {
        register_type("table", &read_table_anchor);
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
