/*
 * object_approach.cpp — compute an object's approach options and keep them on its one `approach` node.
 * See the header for the protocol and the rules.
 */

#include "object_approach.h"

#include <print>

namespace rc::pragmatic
{

void ObjectApproach::init(std::shared_ptr<DSR::DSRGraph> G, std::uint64_t object_node_id, std::string object_name,
                          PragmaticAffordance::Policy policy, Sides sides)
{
    G_     = G;
    sides_ = sides;
    aff_.init(std::move(G), object_node_id, std::move(object_name), "approach", policy);
}

std::vector<ObjectApproach::Option> ObjectApproach::compute_options(const ApproachInputs& in, const ApproachCfg& cfg)
{
    using namespace approach;
    std::vector<Dir> dirs;
    std::size_t first = 0;   // the default's index in dirs
    if (sides_ == Sides::DefaultOnly)
    {
        const Vec2 to_robot = in.robot - in.centre;
        const Vec2 d = in.front_yaw.has_value() ? Vec2{std::cos(*in.front_yaw), std::sin(*in.front_yaw)}
                       : to_robot.norm() > 1e-6f ? Vec2(to_robot.normalized()) : Vec2{1, 0};
        dirs.push_back({"default", d});
    }
    else if (sides_ == Sides::Fronted and in.front_yaw.has_value())
        dirs = fronted_sides(*in.front_yaw);   // "front" is first
    else
    {
        dirs = faces_.update(in.yaw, in.yaw_sigma);
        const Dir& f = facing(dirs, in.centre, in.robot);
        first = static_cast<std::size_t>(&f - dirs.data());
    }

    std::vector<Option> out;
    const auto add = [&](const Dir& d)
    {
        const Pose p = standpoint(in.centre, in.yaw, in.half_extent, d.dir, in.pos_cov, in.robot_pos_sigma_m, cfg.stand);
        if (fits_in_room(Vec2{p.x, p.y}, *in.room_polygon, cfg.turn_radius_m)) out.push_back({d.kind, p});
    };
    add(dirs[first]);
    for (std::size_t i = 0; i < dirs.size(); ++i)
        if (i != first) add(dirs[i]);
    // The default did not fit (against a wall): the options are still ordered, and the first that fits leads.
    return out;
}

void ObjectApproach::update(const ApproachInputs& in, const ApproachCfg& cfg)
{
    if (not G_) return;
    aff_.poll();
    read_executing_option();
    const bool executing = aff_.state() == PragmaticAffordance::State::executing;

    Offer offer;
    offer.known = in.known and in.room_polygon.has_value();
    if (offer.known)
    {
        if (not executing) options_ = compute_options(in, cfg);   // frozen under a claim
        offer.p        = options_.empty() ? 0.0f : 1.0f;
        offer.contract = rc::affordance::Contract::reach();
        if (options_.empty())
            offer.why = "no side of it fits in the room with room to turn";
        else
        {
            const auto& d = options_.front().pose;
            offer.x_m = d.x;
            offer.y_m = d.y;
            offer.yaw_rad = d.yaw;
            const float dist = (in.robot - Eigen::Vector2f{d.x, d.y}).norm();
            offer.value = approach::arrival_value(cfg.value, dist, cfg.arrival_m, in.robot_pos_sigma_m);
            offer.why = std::format("{} side(s) reachable, default {}", options_.size(), options_.front().name);
        }
    }
    aff_.update(offer);
    write_options();   // under a claim options_ is frozen, so this writes nothing new
}

void ObjectApproach::write_options()
{
    const auto id = aff_.node_id();
    if (id == 0 or options_.empty()) return;
    const bool same = id == written_node_ and written_.size() == options_.size()
                      and std::equal(written_.begin(), written_.end(), options_.begin(), [](const Option& a, const Option& b)
                                     { return a.name == b.name and std::abs(a.pose.x - b.pose.x) <= 1e-4f
                                              and std::abs(a.pose.y - b.pose.y) <= 1e-4f
                                              and std::abs(a.pose.yaw - b.pose.yaw) <= 1e-4f; });
    if (same) return;
    auto node = G_->get_node(id);
    if (not node.has_value()) return;
    std::string names;
    std::vector<float> xs, ys, ws;
    for (const auto& o : options_)
    {
        if (not names.empty()) names += ',';
        names += o.name;
        xs.push_back(o.pose.x);
        ys.push_back(o.pose.y);
        ws.push_back(o.pose.yaw);
    }
    G_->add_or_modify_attrib_local<aff_option_names_att>(*node, names);
    G_->add_or_modify_attrib_local<aff_option_x_m_att>(*node, xs);
    G_->add_or_modify_attrib_local<aff_option_y_m_att>(*node, ys);
    G_->add_or_modify_attrib_local<aff_option_yaw_rad_att>(*node, ws);
    G_->update_node(*node);
    written_      = options_;
    written_node_ = id;
}

void ObjectApproach::read_executing_option()
{
    executing_option_.reset();
    const auto id = aff_.node_id();
    if (id == 0) return;
    for (const auto& e : G_->get_edges_to_id(id))
    {
        if (e.type() != "executing") continue;
        executing_option_ = G_->get_attrib_by_name<executing_option_att>(e).value_or(std::string{});
        if (claimed_option_ != *executing_option_)
            std::print("[aff_{}] consumer claims option '{}'\n", aff_.kind(), *executing_option_);
        claimed_option_ = *executing_option_;
        return;
    }
}

std::optional<ObjectApproach::Completion> ObjectApproach::consume_completion()
{
    if (not aff_.consume_completion()) return std::nullopt;
    return Completion{.option = std::exchange(claimed_option_, std::string{}), .outcome = aff_.outcome()};
}

void ObjectApproach::remove()
{
    aff_.remove();
    written_.clear();
    written_node_ = 0;
    executing_option_.reset();
    claimed_option_.clear();
}

void ObjectApproach::on_node_deleted(std::uint64_t id)
{
    if (id != aff_.node_id()) return;
    aff_.on_node_deleted(id);
    written_.clear();
    written_node_ = 0;
    executing_option_.reset();
    claimed_option_.clear();
}

}   // namespace rc::pragmatic
