/*
 * specificworker_consumers.cpp — the graph consumers' channels a countable class may publish beyond the common set
 * (shared-engine K2b). They are what the legacy chair_concept agent published, ported so that the chair keeps every
 * consumer when it runs on this binary; each is behind its Concept.* switch and defaults to the pre-K2b behaviour, so
 * bed, nightstand, toilet, refrigerator and hood are unchanged. The spike that lists who reads what is in
 * .superpowers/sdd/2026-11-23-shared-engine/task-K2b-report.md.
 *
 *   step_approach          aff_<name>_approach (common/pragmatic_affordance/object_approach.h): language_agent "approach",
 *                          the controller's pragmatic claim. The FRONT is the class's −y face (the countable convention:
 *                          R(yaw)·ŷ points from the front into the back), named only while the yaw is surely known.
 *   update_thumbnails      "<class>_thumbs": the best retina crop per viewing angle, read by ltom (graph_sightings.cpp).
 *   update_identity_links  changed object (spec 2026-11-15): a GHOST existence belief at each remembered spot and the
 *                          identity link (same_as_uid / same_as_prob / left_spot_prob) on objects born new.
 *
 * Main thread only (graph reads + writes; the transforms are ts == 0 on the instance the main thread owns, CLAUDE.md).
 */
#include "specificworker.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <print>

#include "../../common/recall_match/recall_candidates.h"
#include "../../common/recall_match/identity_link.h"     // rc::recall::link_posterior / ghost_evidence / left_spot_prob
#include "../../common/appearance_belief/colour_name.h"  // rc::appearance::name_colour (the link's colour term)
#include "../../common/thumbs/thumb_wire.h"              // rc::thumbs::decode / encode
#include "../../common/room_resolve/room_resolve.h"      // rc::room::current_room / current_room_frame

// The approach (common/pragmatic_affordance/object_approach.h): the shared standoff rule and sides, fed from the belief.
// The robot's σ is the pose-chain variance the fitter already computes for this instance; the stand-off floor is where the
// object is SEEN best (owner 2026-11-11, the legacy chair: it stopped too close to see it), or farther if the body needs
// the room to turn (object_approach_geometry.h).
void SpecificWorker::step_approach(rc::CountableInstance& inst)
{
    rc::pragmatic::ApproachInputs in;
    std::optional<Eigen::Vector2f> robot;
    if (inner_eigen_)
        if (const auto T = inner_eigen_->get_transformation_matrix(rc::room::current_room_frame(*G), "body"); T.has_value())
            robot = Eigen::Vector2f(static_cast<float>(T.value()(0, 3)), static_cast<float>(T.value()(1, 3)));
    in.known = inst.ai2_initialized and robot.has_value();
    if (in.known)
    {
        const auto& b = inst.ai2_belief.state();
        const auto cov = inst.ai2_belief.covariance_reported();
        const float yaw_sigma = std::sqrt(std::max(0.0f, inst.ai2_belief.yaw_marginal_var()));
        in.centre = Eigen::Vector2f(b.cx, b.cy);
        in.yaw = b.yaw;
        in.yaw_sigma = yaw_sigma;
        in.half_extent = Eigen::Vector2f(0.5f * b.w, 0.5f * b.h);
        // the front = the −y face: R(yaw)·(−ŷ) points at yaw − 90° (the legacy chair's +y seat side at its yaw + 90°)
        if (rc::pragmatic::approach::front_known(yaw_sigma)) in.front_yaw = b.yaw - std::numbers::pi_v<float> / 2;
        in.pos_cov = cov.block<2, 2>(0, 0);
        in.robot = *robot;
        in.robot_pos_sigma_m = std::sqrt(std::max({0.0f, inst.chain_cov_xx, inst.chain_cov_yy}));
        if (fitter_->has_room_polygon()) in.room_polygon = fitter_->room_polygon();
    }
    rc::pragmatic::ApproachCfg acfg;
    if (inst.ai2_initialized)
    {
        const auto& b = inst.ai2_belief.state();
        if (const auto dv = fitter_->best_view_distance(b.w, b.H); dv.has_value()) acfg.stand.min_standoff_m = *dv;
    }
    inst.approach.update(in, acfg);
}

// Retina cuts crops at a low rate on "thumbs_zed". Each instance keeps the best one per viewing angle; no image work here.
// The viewing angle is binned against the PUBLISHED yaw (the countable convention), the same yaw ltom stores with the pose.
void SpecificWorker::update_thumbnails()
{
    const auto node = G->get_node("thumbs_zed");
    const auto bytes = node ? G->get_attrib_by_name<compressed_data_att>(*node) : std::nullopt;
    auto frame = bytes ? rc::thumbs::decode(bytes->get()) : std::nullopt;
    const auto& pkt = mask_ingestor_->packet();
    if (frame and frame->frame != last_thumb_frame_ and frame->frame == static_cast<std::uint64_t>(pkt.frame_id))
    {
        last_thumb_frame_ = frame->frame;
        const auto camera = inner_eigen_ ? inner_eigen_->transform(rc::room::current_room_frame(*G), Eigen::Vector3d::Zero(), "zed")
                                         : std::nullopt;
        if (camera)
            for (auto& [id, inst] : fitter_->instances())
                for (const int idx : inst.assigned_mask_idxs)
                {
                    if (idx < 0 or idx >= static_cast<int>(pkt.slices.size())) continue;
                    const auto& slice = pkt.slices[idx];
                    if (not slice.is_zed()) continue;
                    const auto crop = std::ranges::find(frame->thumbs, static_cast<std::uint64_t>(slice.node_index), &rc::thumbs::Thumb::key);
                    if (crop == frame->thumbs.end()) continue;
                    const auto& s = inst.model.state();
                    const int bin = rc::thumbs::view_bin(static_cast<float>(camera->x()), static_cast<float>(camera->y()), s.cx, s.cy, s.yaw,
                                                         rc::thumbs::kViewBins);
                    const float visible = static_cast<float>(slice.pixel_end - slice.pixel_begin) * (1.f - slice.trunc_frac);
                    thumbs_dirty_ = inst.thumbs.offer(bin, visible, crop->jpeg) or thumbs_dirty_;
                }
    }
    static constexpr auto kPublishPeriod = std::chrono::seconds(1);   // machine guard: ltom polls at 2 Hz
    const auto now = std::chrono::steady_clock::now();
    if (not thumbs_dirty_ or now - last_thumb_publish_ < kPublishPeriod) return;
    rc::thumbs::ThumbFrame out;
    for (const auto& [id, inst] : fitter_->instances())
    {
        const auto obj = G->get_node(id);
        const auto uid = obj ? G->get_attrib_by_name<object_uid_att>(*obj) : std::nullopt;
        if (not uid) continue;
        for (const auto& [bin, view] : inst.thumbs.bins())
            out.thumbs.push_back({.key = *uid, .bin = bin, .score = view.score, .jpeg = view.jpeg});
    }
    const auto agent = G->get_node(std::format("{} {}", agent_name, agent_id));
    const std::string name = cfg_.concept_class + "_thumbs";   // ltom reads <class>_thumbs (graph_sightings.cpp)
    auto published = G->get_node(name);
    const bool fresh = not published.has_value();
    DSR::Node n = fresh ? DSR::Node::create<metaconcept_node_type>(name) : *published;
    G->add_or_modify_attrib_local<compressed_data_att>(n, rc::thumbs::encode(out));
    if (fresh)
    {
        if (not agent) return;
        G->add_or_modify_attrib_local<parent_att>(n, agent->id());
        G->add_or_modify_attrib_local<level_att>(n, G->get_attrib_by_name<level_att>(*agent).value_or(1) + 1);
        const auto new_id = G->insert_node(n);
        if (not new_id) return;
        G->insert_or_assign_edge(DSR::Edge::create<has_edge_type>(agent->id(), *new_id));
    }
    else
        G->update_node(n);
    thumbs_dirty_ = false;
    last_thumb_publish_ = now;
}

// ── Changed object (spec 2026-11-15-changed-object, owner's choice B; ported from the legacy chair) ─────────────────────────
// For each untaken recall in the current room, a GHOST existence belief at its old spot: absence is charged as for a live
// instance (P(a look resolves a prior-sized object there), the silhouette's frustum + occlusion), and HELD while an instance
// stands on the spot. For each instance born new this run (and each one ltom already links to), the identity posterior over
// {recall_k..., new}; the MAP is published on the node (same_as_uid / same_as_prob / left_spot_prob). The name never changes.
void SpecificWorker::update_identity_links()
{
    auto& insts = fitter_->instances();
    std::erase_if(born_new_ids_, [&](std::uint64_t id) { return not insts.contains(id); });
    const auto room = rc::recall_room(room_node_id_, rc::room::current_room(*G));
    // No room this cycle (a room change, a relocalisation glitch): no information, so HOLD — ghosts and links stay as they were.
    if (not room.has_value()) return;
    const std::string prefix = cfg_.concept_class + "_";
    std::vector<rc::RecallCand> cands;
    {
        std::set<std::string> live_names;
        for (const auto& n : G->get_nodes_by_type("object"))
            if (n.name().starts_with(prefix)) live_names.insert(n.name());
        cands = rc::recall_candidates(*G, *room, cfg_.concept_class, static_cast<std::uint32_t>(G->get_agent_id()),
                                      scene_graph_->identity_last(), live_names);
    }
    // This frame's detections of the class (room frame): the ZED slices the tracker would take.
    std::vector<Eigen::Vector2f> det_xy;
    if (const auto& pkt = mask_ingestor_->packet(); pkt.valid)
        for (const auto& sl : pkt.slices)
            if (sl.label == cfg_.concept_class and sl.may_fit_geometry() and sl.depth_var <= 0.0f)
                det_xy.emplace_back(sl.centroid.x(), sl.centroid.y());
    // The robot (the viewpoint of the look), for the ghost's viewpoint-novelty weighting.
    std::optional<Eigen::Vector2f> robot;
    if (inner_eigen_)
        if (const auto T = inner_eigen_->get_transformation_matrix(rc::room::current_room_frame(*G), "body"); T.has_value())
            robot = Eigen::Vector2f(static_cast<float>(T.value()(0, 3)), static_cast<float>(T.value()(1, 3)));

    // Ghosts: one per offered recall; a recall that leaves the list takes its ghost with it.
    std::set<std::uint64_t> offered;
    for (const auto& c : cands) offered.insert(c.cand.uid);
    std::erase_if(ghosts_, [&](const auto& kv) { return not offered.contains(kv.first); });
    const float depth = cfg_.ai2_prior_depth_m > 0.0f ? cfg_.ai2_prior_depth_m : cfg_.ai2_prior_footprint_m;
    const float half = 0.5f * std::max(cfg_.ai2_prior_footprint_m, depth);
    for (const auto& c : cands)
    {
        auto [it, fresh] = ghosts_.try_emplace(c.cand.uid);
        Ghost& g = it->second;
        if (fresh)
        {
            // Starts at the owner's last belief that the object existed (0 = not stated = even odds).
            const float p = std::clamp(std::exp(c.cand.log_prior), 1e-3f, 1.f - 1e-3f);   // keeps logit finite (degenerate guard)
            g.L0 = c.cand.log_prior == 0.f ? 0.f : std::log(p / (1.f - p));
            g.belief.set_max(cfg_.existence_logodds_max);
            g.belief.set(g.L0);
            g.L0 = g.belief.logodds();   // after the belief's own bound
        }
        // An object seen standing on the spot (a live instance, or this frame's detection) CONFIRMS "still there" (occupancy
        // confirms); nothing there removes; out of view holds. The footprint is the class prior's.
        const auto on_spot = [&](float x, float y)
        { return std::abs(x - c.cand.mu.x()) < half and std::abs(y - c.cand.mu.y()) < half; };
        const bool seen_there =
            std::ranges::any_of(insts, [&](const auto& kv) { return on_spot(kv.second.model.state().cx, kv.second.model.state().cy); })
            or std::ranges::any_of(det_xy, [&](const Eigen::Vector2f& d) { return on_spot(d.x(), d.y()); });
        const auto ev = rc::recall::ghost_evidence(fitter_->ghost_p_vis(c.cand.mu), seen_there, cfg_.ghost_evidence_gain);
        if (ev.p_vis <= 0.f) continue;   // not in view: HOLD
        const float before = rc::recall::left_spot_prob(g.belief.logodds(), g.L0);
        if (robot) g.belief.note_viewpoint(robot->x(), robot->y(), c.cand.mu.x(), c.cand.mu.y(), half);
        g.belief.integrate(ev.p_vis, ev.ratio);
        const float after = rc::recall::left_spot_prob(g.belief.logodds(), g.L0);
        if ((before > 0.5f) != (after > 0.5f))
            std::print("{}: [link] ghost {}{} left_spot p={:.2f}\n", log_name(), prefix, c.n, after);
    }

    // The link of each eligible live instance.
    const rc::recall::LinkParams lp{cfg_.link_pi_move, cfg_.link_pi_gone, cfg_.link_lambda_new};
    struct Pair { std::uint64_t node; std::size_t cand; float p; bool confirmed; };
    std::vector<Pair> pairs;
    for (auto& [id, inst] : insts)
    {
        const auto node = G->get_node(id);
        const auto uid = node ? G->get_attrib_by_name<object_uid_att>(*node) : std::nullopt;
        if (not uid.has_value()) continue;
        const bool linked_by_ltom = std::ranges::any_of(cands, [&](const auto& c) { return c.same_as_uid == *uid; });
        if (not born_new_ids_.contains(id) and not linked_by_ltom) continue;
        std::vector<rc::recall::LinkCand> lc;
        std::vector<std::size_t> idx;
        for (std::size_t k = 0; k < cands.size(); ++k)
        {
            const auto& c = cands[k];
            if (c.same_as_uid != 0 and c.same_as_uid != *uid) continue;   // linked to another live instance
            if (c.same_as_uid == *uid and c.same_as_prob >= 1.f)          // the owner confirmed it
            {
                pairs.push_back({id, k, 1.f, true});
                lc.clear();
                idx.clear();
                break;
            }
            const auto gi = ghosts_.find(c.cand.uid);
            const float llr = gi == ghosts_.end() ? 0.f : gi->second.belief.logodds() - gi->second.L0;
            lc.push_back({c.cand.uid, c.cand.log_prior, c.colour_name, c.colour_prob, llr});
            idx.push_back(k);
        }
        if (lc.empty()) continue;
        const auto cn = rc::appearance::name_colour(inst.appearance);
        const std::optional<std::array<float, 12>> colour =
            cn.name.empty() ? std::nullopt : std::optional<std::array<float, 12>>(cn.probs);
        const auto link = rc::recall::link_posterior(lc, colour, lp);
        if (link.index.has_value()) pairs.push_back({id, idx[*link.index], link.posterior[*link.index], false});
    }
    // Each memory is linked to at most one live instance: the most probable pairing wins.
    std::ranges::sort(pairs, std::greater{}, &Pair::p);
    std::set<std::size_t> cand_taken;
    std::set<std::uint64_t> node_linked;
    std::map<std::uint64_t, Pair> chosen;
    for (const Pair& pr : pairs)
        if (not cand_taken.contains(pr.cand) and not node_linked.contains(pr.node))
        {
            cand_taken.insert(pr.cand);
            node_linked.insert(pr.node);
            chosen.emplace(pr.node, pr);
        }
    for (auto& [id, inst] : insts)
    {
        const auto ch = chosen.find(id);
        const std::uint64_t was = inst.link_uid;
        if (ch == chosen.end()) { inst.link_uid = 0; inst.link_prob = 0.f; inst.link_left_spot_prob = 0.f; }
        else
        {
            const auto& c = cands[ch->second.cand];
            inst.link_uid = c.cand.uid;
            inst.link_prob = ch->second.p;
            const auto gi = ghosts_.find(c.cand.uid);
            inst.link_left_spot_prob = gi == ghosts_.end() ? 0.f : rc::recall::left_spot_prob(gi->second.belief.logodds(), gi->second.L0);
        }
        if (was != inst.link_uid)
        {
            const auto node = G->get_node(id);
            const std::string name = node ? node->name() : std::to_string(id);
            if (inst.link_uid == 0) std::print("{}: [link] {} -> none\n", log_name(), name);
            else std::print("{}: [link] {} -> {}{} p={:.2f} left={:.2f}{}\n", log_name(), name, prefix,
                            static_cast<int>(inst.link_uid & rc::kUidNumberMask), inst.link_prob, inst.link_left_spot_prob,
                            ch->second.confirmed ? " (confirmed)" : "");
        }
    }
}
