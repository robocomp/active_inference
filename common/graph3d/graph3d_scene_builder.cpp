#include "graph3d_scene_builder.h"

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "../agent_state_publisher/agent_status.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace rc::graph3d
{

namespace
{

constexpr Rgb kRobot    {0.92f, 0.94f, 0.98f};   // bottom of the ladder: bright, it anchors everything
constexpr Rgb kSlate    {0.42f, 0.47f, 0.54f};   // the room: the container, deliberately quiet
constexpr Rgb kAmber    {0.95f, 0.70f, 0.25f};   // affordances
constexpr Rgb kViolet   {0.62f, 0.48f, 0.96f};   // meta-concepts
constexpr Rgb kPurple   {0.55f, 0.45f, 0.78f};   // RT edges (matches dsr_gui edge_colors.h)
constexpr Rgb kGrey     {0.50f, 0.50f, 0.56f};   // has (mind→agent)
constexpr Rgb kSteel    {0.70f, 0.75f, 0.82f};

[[nodiscard]] Rgb mix(const Rgb &a, const Rgb &b, const float t) noexcept
{
	return {a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t};
}

[[nodiscard]] Rgb hsv(const float h, const float s, const float v) noexcept
{
	const float i = std::floor(h * 6.0f);
	const float f = h * 6.0f - i;
	const float p = v * (1.0f - s);
	const float q = v * (1.0f - f * s);
	const float t = v * (1.0f - (1.0f - f) * s);
	switch (static_cast<int>(i) % 6)
	{
		case 0:  return {v, t, p};
		case 1:  return {q, v, p};
		case 2:  return {p, v, t};
		case 3:  return {p, q, v};
		case 4:  return {t, p, v};
		default: return {v, p, q};
	}
}

// The four names rc::agent_status can produce, as RGB.
[[nodiscard]] Rgb health_color(const std::string_view name) noexcept
{
	if (name == "green")  return {0.28f, 0.80f, 0.38f};
	if (name == "orange") return {0.96f, 0.62f, 0.16f};
	if (name == "red")    return {0.92f, 0.28f, 0.28f};
	return {0.52f, 0.52f, 0.56f};   // gray / unknown
}

// "table_concept 7" → "table_"; "ring_metaconcept 23" → "ring_". The stem is what the agent's owned
// node names are prefixed with by convention, which is the fallback when CRDT provenance is unclear.
[[nodiscard]] std::string owned_prefix_of(const std::string &agent_node_name)
{
	std::string base = agent_node_name.substr(0, agent_node_name.find(' '));
	for (const std::string_view suffix : {"_metaconcept", "_concept"})
		if (base.size() > suffix.size() and base.ends_with(suffix))
		{
			base.resize(base.size() - suffix.size());
			break;
		}
	return base.empty() ? std::string{} : base + "_";
}

}   // namespace

SceneBuilder::SceneBuilder() = default;
SceneBuilder::~SceneBuilder() = default;

void SceneBuilder::set_graph(std::shared_ptr<DSR::DSRGraph> graph)
{
	g_ = std::move(graph);
	inner_.reset();
	prefix_owner_.clear();
}

std::optional<SceneBuilder::Classified> SceneBuilder::classify(const std::string &type,
                                                               const std::string &subtype,
                                                               const std::string &name) noexcept
{
	// ── Not drawn at all. These are the RT tree's PLUMBING, not things the system believes in:
	// `root` is the bare origin (dropping it makes the ROBOT the effective root of the drawn
	// hierarchy — the root→Shadow edge goes with it), and body/sensor mounts are the robot's own
	// internal structure. The robot stands for the whole assembly; showing its chassis and five
	// sensor frames adds seven nodes and says nothing about the scene. (`mind` is not listed here:
	// it is the anchor of the agent layer, so it stays classified and is hidden with that stratum.)
	if (type == "root" or type == "body" or type == "laser" or type == "rgbd" or type == "imu"
	    or type == "camera")
		return std::nullopt;

	if (type == "agent")
		return Classified{Kind::Agent, Glyph::Pin};
	if (type == "mind")
		return Classified{Kind::Agent, Glyph::Sphere};
	if (type == "affordance" or name.starts_with("aff_"))
		return Classified{Kind::Affordance, Glyph::Diamond};
	// A meta-concept is an `object` like any other; only its subtype says it is a concept OVER
	// concepts. ring_metaconcept writes object_subtype="dining_set" (ring_config.h).
	if (subtype == "dining_set" or subtype == "rig" or subtype == "ring" or name.starts_with("dining_set"))
		return Classified{Kind::Meta, Glyph::Ring};

	// The two rungs below the instances. `room` is the container; the ROBOT is the bottom of the
	// ladder, matching the graph's real parenting (root is not drawn, and room is a child of robot).
	if (type == "robot")
		return Classified{Kind::Robot, Glyph::Robot};
	if (type == "room")
		return Classified{Kind::Room, Glyph::Box};

	// Walls and the floor are concept instances like any other — room_concept fits and believes in
	// them exactly the way table_concept believes in a table.
	if (type == "wall" or type == "floor" or type == "plane")
		return Classified{Kind::Instance, Glyph::Box};

	if (subtype == "chair")                        return Classified{Kind::Instance, Glyph::Chair};
	if (subtype == "bottle" or type == "cylinder") return Classified{Kind::Instance, Glyph::Cylinder};
	if (subtype == "table" or subtype == "cabinet"
	    or subtype == "refrigerator" or subtype == "door"
	    or type == "box" or type == "obstacle")    return Classified{Kind::Instance, Glyph::Box};

	return Classified{Kind::Instance, Glyph::Sphere};
}

Rgb SceneBuilder::identity_color(const std::uint32_t agent_id) noexcept
{
	// Golden-angle hue walk: consecutive agent ids land far apart on the wheel, so neighbouring ids
	// (5,6,7,8…) never come out as near-identical colours. Stable across runs — the id is the seed.
	const float hue = std::fmod(static_cast<float>(agent_id) * 0.6180339887f, 1.0f);
	return hsv(hue, 0.62f, 0.96f);
}

Scene SceneBuilder::build()
{
	Scene scene;
	if (not g_)
		return scene;

	if (not inner_)
		inner_ = g_->get_inner_eigen_api();

	// ── Reference frame: everything metric is expressed in the room, falling back up the tree when
	// the room has not synced yet, so the view still draws during startup.
	std::string ref_frame;
	for (const std::string_view t : {"room", "robot", "root"})
		if (const auto v = g_->get_nodes_by_type(std::string{t}); not v.empty())
		{
			ref_frame = v.front().name();
			break;
		}

	const auto graph_nodes = g_->get_nodes();

	// The RT tree, read once: `rt_member` gates whether a node can be asked for a pose at all, and
	// `rt_parent` is what makes an instance's level follow its RT path (a part sits above its whole).
	std::unordered_set<std::uint64_t>               rt_member;
	std::unordered_map<std::uint64_t, std::uint64_t> rt_parent;   // child → parent
	for (const auto &e : g_->get_edges_by_type("RT"))
	{
		rt_member.insert(e.from());
		rt_member.insert(e.to());
		rt_parent[e.to()] = e.from();
	}

	// ── Agent registry: agent nodes carry their own agent_id, which is what CRDT provenance on every
	// other node has to be matched against.
	std::unordered_map<std::uint32_t, std::uint64_t> agent_by_id;
	prefix_owner_.clear();
	for (const auto &n : graph_nodes)
	{
		if (n.type() != "agent")
			continue;
		if (const auto aid = g_->get_attrib_by_name<agent_id_att>(n); aid.has_value())
			agent_by_id[aid.value()] = n.id();
		if (auto p = owned_prefix_of(n.name()); not p.empty())
			prefix_owner_[std::move(p)] = n.id();
	}

	// ── Pass 1: classify, place, size.
	std::unordered_map<std::uint64_t, std::size_t> index_of;
	std::unordered_map<std::string, std::uint64_t> id_of_name;
	scene.nodes.reserve(graph_nodes.size());

	for (const auto &n : graph_nodes)
	{
		const std::string subtype = g_->get_attrib_by_name<object_subtype_att>(n).value_or("");
		const auto        cls     = classify(n.type(), subtype, n.name());
		if (not cls.has_value())
			continue;   // RT plumbing: root, robot chassis, sensor mounts
		const auto [kind, glyph] = cls.value();

		Node3D node;
		node.id      = n.id();
		node.name    = n.name();
		node.type    = n.type();
		node.subtype = subtype;
		node.kind    = kind;
		node.glyph   = glyph;

		// Metric pose, asked for ONLY where it can exist. `mind` carries parent/level attributes
		// naming Shadow as its parent but has no RT edge to it (shadow.json), and agent nodes have
		// no RT edge either; inner_eigen walks UP via parent_att, fails to find the edge and qWarns
		// ("Cannot find RT edge between Parent (Shadow,200) and son (mind,260)") before returning
		// nullopt. Harmless but emitted on every call, so at 5 Hz it floods the log. Gating on actual
		// RT-tree membership removes the cause rather than muting the message.
		// ALWAYS check the optional regardless: a missing node/edge/rtmat anywhere in the chain makes
		// get_transformation_matrix return {} rather than throwing (CLAUDE.md).
		if (not ref_frame.empty() and rt_member.contains(n.id()))
			if (const auto T = inner_->get_transformation_matrix(ref_frame, n.name(), 0); T.has_value())
			{
				const auto &M = T.value();
				node.pos[0] = static_cast<float>(M.translation().x());
				node.pos[1] = static_cast<float>(M.translation().y());
				node.yaw    = static_cast<float>(std::atan2(M.linear()(1, 0), M.linear()(0, 0)));
				node.placed = true;
			}

		// z is assigned later, once the meta nesting depth is known and the ladder has a height.
		// (An earlier version added a per-RT-depth rise to spread the robot's sensor chain; that was
		// wrong for everything else, because `room` is itself a child of the robot — so walls, tables
		// and chairs were all silently lifted off their own plane.)

		// Glyph size hints at real extent without pretending to be the geometry — this is a graph
		// view, not a scene view.
		const float w = g_->get_attrib_by_name<width_m_att>(n).value_or(0.0f);
		const float d = g_->get_attrib_by_name<depth_m_att>(n).value_or(0.0f);
		node.radius = (w > 0.0f or d > 0.0f) ? std::clamp(0.35f * std::max(w, d), 0.10f, 0.32f)
		                                     : (kind == Kind::Agent ? 0.16f : 0.12f);

		node.owner_agent_id = n.agent_id();

		index_of[node.id]      = scene.nodes.size();
		id_of_name[node.name]  = node.id;
		scene.nodes.push_back(std::move(node));
	}

	// ── Pin the ROOM glyph directly above the ROBOT. Its true pose is the room frame's ORIGIN, which
	// sits at a corner of the footprint — drawing the node there turned the robot→room RT edge, and
	// one rung up every room→instance edge, into a long diagonal across the whole floor. Stacking it
	// makes the bottom of the ladder read as a spine with the instances fanning out of it. Only the
	// GLYPH moves: the room polygon stays at its true metric footprint, since that is what gives x/y
	// its meaning.
	{
		float rx = 0.0f, ry = 0.0f;
		bool  have_robot = false;
		for (const auto &n : scene.nodes)
			if (n.kind == Kind::Robot and n.placed)
			{
				rx         = n.pos[0];
				ry         = n.pos[1];
				have_robot = true;
				break;
			}
		if (have_robot)
			for (auto &n : scene.nodes)
				if (n.kind == Kind::Room)
				{
					n.pos[0] = rx;
					n.pos[1] = ry;
					n.placed = true;
				}
	}

	// ── Pass 1b: the LADDER. Both of its tails are open-ended, which is what makes the number of
	// planes a per-build quantity rather than an enum.
	std::unordered_map<std::uint64_t, Kind> kind_of;
	for (const auto &n : scene.nodes)
		kind_of[n.id] = n.kind;

	std::unordered_set<std::uint64_t> visiting;   // guard: a malformed graph can hold a cycle

	// (i) INSTANCES follow the RT PATH. An instance parented to another instance is a PART of it —
	// a door hanging off a wall, a bottle re-parented onto its supporting table
	// (bottle_scene_graph.cpp) — so it belongs one rung above its parent, not alongside it. Since it
	// also keeps its own metric x/y, and a part sits physically on its parent, this draws the door
	// vertically above its wall for free. Depth is measured in INSTANCE-to-instance hops: a child of
	// the room is depth 0, and a parent that is not a drawn instance (the robot, or an excluded
	// sensor mount) also anchors at 0.
	std::unordered_map<std::uint64_t, int> inst_depth;
	std::function<int(std::uint64_t)>      inst_depth_of = [&](const std::uint64_t id) -> int
	{
		if (const auto it = inst_depth.find(id); it != inst_depth.end())
			return it->second;
		if (not visiting.insert(id).second)
			return 0;
		int depth = 0;
		if (const auto p = rt_parent.find(id); p != rt_parent.end())
			if (const auto k = kind_of.find(p->second); k != kind_of.end() and k->second == Kind::Instance)
				depth = 1 + inst_depth_of(p->second);
		visiting.erase(id);
		inst_depth[id] = depth;
		return depth;
	};

	int max_inst = 0;
	for (const auto &n : scene.nodes)
		if (n.kind == Kind::Instance)
			max_inst = std::max(max_inst, inst_depth_of(n.id));

	// (ii) META levels stack on top of the DEEPEST instance plane: a rig grouping plain instances is
	// meta-1, a rig grouping THOSE rigs is meta-2, and so on down the `group_member` chain.
	std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> members;   // rig → its members
	for (const auto &e : g_->get_edges_by_type("group_member"))
		members[e.from()].push_back(e.to());

	std::unordered_map<std::uint64_t, int> meta_depth;
	std::function<int(std::uint64_t)>      meta_depth_of = [&](const std::uint64_t id) -> int
	{
		if (const auto it = meta_depth.find(id); it != meta_depth.end())
			return it->second;
		if (not visiting.insert(id).second)
			return 1;   // cycle — stop unwinding rather than recursing forever
		int deepest = 0;
		if (const auto it = members.find(id); it != members.end())
			for (const std::uint64_t m : it->second)
				if (const auto k = kind_of.find(m); k != kind_of.end() and k->second == Kind::Meta)
					deepest = std::max(deepest, meta_depth_of(m));
		visiting.erase(id);
		const int d = deepest + 1;   // groups only instances ⇒ 1
		meta_depth[id] = d;
		return d;
	};

	int max_meta = 0;
	for (const auto &n : scene.nodes)
		if (n.kind == Kind::Meta)
			max_meta = std::max(max_meta, meta_depth_of(n.id));

	const int deepest_inst = kLevelInstance + max_inst;

	// Optional layers stack ON TOP of everything: an affordance is a property of an instance and an
	// agent is an observer of the whole thing, so neither is a rung of the containment ladder —
	// parking them above keeps the ladder itself honest.
	const int top         = deepest_inst + max_meta;
	const int aff_level   = top + 1;
	const int agent_level = top + (cfg_.show_affordances ? 2 : 1);

	int level_count = top + 1;
	if (cfg_.show_affordances) level_count = std::max(level_count, aff_level + 1);
	if (cfg_.show_agents)      level_count = std::max(level_count, agent_level + 1);

	scene.level_z.resize(static_cast<std::size_t>(level_count));
	scene.level_names.resize(static_cast<std::size_t>(level_count));
	for (int l = 0; l < level_count; ++l)
	{
		scene.level_z[static_cast<std::size_t>(l)] = cfg_.stratum_gap * static_cast<float>(l);
		scene.level_names[static_cast<std::size_t>(l)] =
		    l == kLevelRobot ? "ROBOT"
		    : l == kLevelRoom ? "ROOM"
		    : (cfg_.show_agents and l == agent_level)    ? "AGENTS"
		    : (cfg_.show_affordances and l == aff_level) ? "AFFORDANCES"
		    : l == kLevelInstance                        ? "INSTANCES"
		    : l <= deepest_inst ? "PARTS-" + std::to_string(l - kLevelInstance)
		                        : "META-" + std::to_string(l - deepest_inst);
	}

	for (auto &node : scene.nodes)
	{
		switch (node.kind)
		{
			case Kind::Robot:      node.level = kLevelRobot;                          break;
			case Kind::Room:       node.level = kLevelRoom;                           break;
			case Kind::Instance:   node.level = kLevelInstance + inst_depth_of(node.id); break;
			case Kind::Meta:       node.level = deepest_inst + meta_depth_of(node.id); break;
			case Kind::Affordance: node.level = aff_level;                            break;
			case Kind::Agent:      node.level = agent_level;                          break;
		}
		node.pos[2] = scene.level_z[static_cast<std::size_t>(
		    std::clamp(node.level, 0, level_count - 1))];
	}

	// ── Pass 2: ownership. CRDT provenance (DSR::Node::agent_id — the writer) is authoritative when
	// it names a live agent. It is the LAST writer, so for the rare node several agents update it can
	// drift; the name-prefix convention and affordance-inherits-parent cover those.
	std::unordered_map<std::uint64_t, std::uint64_t> owner_node;   // node id → agent node id
	for (auto &node : scene.nodes)
	{
		if (node.type == "agent")
			continue;
		if (const auto it = agent_by_id.find(node.owner_agent_id); it != agent_by_id.end())
		{
			owner_node[node.id] = it->second;
			continue;
		}
		std::size_t best = 0;
		for (const auto &[prefix, agent_id] : prefix_owner_)
			if (node.name.starts_with(prefix) and prefix.size() > best)
			{
				best                = prefix.size();
				owner_node[node.id] = agent_id;
			}
	}
	// Affordances inherit their parent's owner: aff_table_1 belongs to whoever owns table_1.
	for (const auto &node : scene.nodes)
	{
		if (not node.name.starts_with("aff_") or owner_node.contains(node.id))
			continue;
		if (const auto it = id_of_name.find(node.name.substr(4)); it != id_of_name.end())
			if (const auto o = owner_node.find(it->second); o != owner_node.end())
				owner_node[node.id] = o->second;
	}

	// ── Pass 3: agent placement + colour. An agent has no pose of its own, so it floats above the
	// centroid of everything it wrote — which is exactly the reading "this agent is responsible for
	// that region of the room".
	const std::uint64_t now_ns          = get_unix_timestamp();
	const std::uint64_t stale_after_ns  = static_cast<std::uint64_t>(cfg_.stale_after_ms) * 1000000ULL;
	std::unordered_map<std::uint64_t, Rgb> agent_hue;

	for (auto &node : scene.nodes)
	{
		if (node.type != "agent")
			continue;

		float sx = 0.0f, sy = 0.0f;
		int   count = 0;
		for (const auto &other : scene.nodes)
			if (other.placed and owner_node.contains(other.id) and owner_node.at(other.id) == node.id)
			{
				sx += other.pos[0];
				sy += other.pos[1];
				++count;
			}
		if (count > 0)
		{
			node.pos[0] = sx / static_cast<float>(count);
			node.pos[1] = sy / static_cast<float>(count);
			node.placed = true;
		}

		const auto    n_opt = g_->get_node(node.id);
		std::string   fsm, presence;
		bool          stale = true;
		std::uint32_t aid   = static_cast<std::uint32_t>(node.id);
		if (n_opt.has_value())
		{
			const auto beat = g_->get_attrib_by_name<timestamp_agent_att>(n_opt.value());
			// No heartbeat at all has never been seen alive — treat as stale rather than inventing a
			// healthy colour (same rule as rc::AgentStatusOverlay).
			stale    = not beat.has_value() or now_ns < beat.value()
			           or (now_ns - beat.value()) > stale_after_ns;
			fsm      = g_->get_attrib_by_name<agent_fsm_state_att>(n_opt.value()).value_or("");
			presence = g_->get_attrib_by_name<agent_presence_state_att>(n_opt.value()).value_or("");
			aid      = g_->get_attrib_by_name<agent_id_att>(n_opt.value()).value_or(aid);
		}
		node.dimmed   = stale;
		node.sublabel = stale ? "stale" : (fsm.empty() ? "?" : fsm) + "/" + (presence.empty() ? "?" : presence);
		node.color    = health_color(stale ? agent_status::STALE_COLOR
		                                   : agent_status::color_for(agent_status::fsm_from_name(fsm),
		                                                             agent_status::presence_from_name(presence)));

		agent_hue[node.id] = identity_color(aid);
	}

	// Agents that own nothing (or share a footprint with a peer) would stack on one point. Spread
	// them apart with a few relaxation passes rather than a hard grid, so the ones that DO have a
	// meaningful centroid barely move.
	for (int pass = 0; pass < 12; ++pass)
	{
		bool moved = false;
		for (std::size_t i = 0; i < scene.nodes.size(); ++i)
		{
			if (scene.nodes[i].kind != Kind::Agent)
				continue;
			for (std::size_t j = i + 1; j < scene.nodes.size(); ++j)
			{
				if (scene.nodes[j].kind != Kind::Agent)
					continue;
				float dx = scene.nodes[j].pos[0] - scene.nodes[i].pos[0];
				float dy = scene.nodes[j].pos[1] - scene.nodes[i].pos[1];
				float d2 = dx * dx + dy * dy;
				constexpr float kMinSep = 0.55f;
				if (d2 >= kMinSep * kMinSep)
					continue;
				if (d2 < 1e-6f)   // exactly coincident: deterministic nudge, seeded by index
				{
					const float a = 2.399963f * static_cast<float>(i + 1);
					dx = std::cos(a);
					dy = std::sin(a);
					d2 = 1.0f;
				}
				const float d    = std::sqrt(d2);
				const float push = 0.5f * (kMinSep - d) / d;
				scene.nodes[i].pos[0] -= dx * push;
				scene.nodes[i].pos[1] -= dy * push;
				scene.nodes[j].pos[0] += dx * push;
				scene.nodes[j].pos[1] += dy * push;
				moved = true;
			}
		}
		if (not moved)
			break;
	}

	// ── Pass 4: colour every non-agent node by WHO WROTE IT. Health lives on the agent glyph; hue
	// carries identity, so ownership is legible at rest without turning on the ribbons.
	for (auto &node : scene.nodes)
	{
		if (node.type == "agent")
			continue;
		const auto it  = owner_node.find(node.id);
		const Rgb  hue = (it != owner_node.end() and agent_hue.contains(it->second))
		                     ? agent_hue.at(it->second)
		                     : kSteel;
		switch (node.kind)
		{
			case Kind::Robot:      node.color = kRobot;                  break;
			case Kind::Room:       node.color = mix(kSlate, hue, 0.25f); break;
			case Kind::Affordance: node.color = mix(kAmber, hue, 0.35f); break;
			case Kind::Meta:       node.color = mix(kViolet, hue, 0.35f); break;
			case Kind::Agent:      node.color = kGrey;                   break;   // mind
			case Kind::Instance:   node.color = mix(kSteel, hue, 0.72f); break;
		}
		if (not node.placed)
			node.color = mix(node.color, kGrey, 0.6f);   // no RT chain — say so instead of lying
	}

	// ── Edges, styled by MEANING. The same "RT" string carries both a real containment (room→table)
	// and a mere mount link (body→zed); both are drawn alike here because both ARE transforms.
	const auto add_edges = [&](const std::string &type, const EdgeKind kind, const Rgb &base,
	                           const float alpha)
	{
		for (const auto &e : g_->get_edges_by_type(type))
		{
			if (not index_of.contains(e.from()) or not index_of.contains(e.to()))
				continue;
			Edge3D out;
			out.from  = e.from();
			out.to    = e.to();
			out.kind  = kind;
			out.alpha = alpha;
			out.color = base;
			if (kind == EdgeKind::Member)
				if (const auto it = owner_node.find(e.from());
				    it != owner_node.end() and agent_hue.contains(it->second))
					out.color = agent_hue.at(it->second);
			scene.edges.push_back(out);
		}
	};

	add_edges("RT", EdgeKind::RT, kPurple, 0.75f);
	add_edges("has_intention", EdgeKind::Intention, kAmber, 0.90f);
	add_edges("group_member", EdgeKind::Member, kViolet, 0.60f);
	add_edges("has", EdgeKind::Has, kGrey, 0.45f);

	// Ownership is not a graph edge at all — it is CRDT provenance made visible. Emitted for every
	// node; the renderer shows only the selected agent's, because 12 agents at once is spaghetti.
	for (const auto &[node_id, agent_node_id] : owner_node)
	{
		if (node_id == agent_node_id or not index_of.contains(agent_node_id))
			continue;
		scene.edges.push_back({.from  = agent_node_id,
		                       .to    = node_id,
		                       .kind  = EdgeKind::Ownership,
		                       .color = agent_hue.contains(agent_node_id) ? agent_hue.at(agent_node_id) : kSteel,
		                       .alpha = 0.40f});
	}

	// ── Ground: the real room footprint, so the metric x/y has a visible frame of reference. The
	// plane itself is defined even when no polygon has synced yet — the renderer drops its plumb
	// lines onto it either way.
	scene.ground.z = scene.level_z[static_cast<std::size_t>(kLevelRoom)];
	if (const auto rooms = g_->get_nodes_by_type("room"); not rooms.empty())
	{
		const auto &r  = rooms.front();
		const auto  px = g_->get_attrib_by_name<delimiting_polygon_x_att>(r);
		const auto  py = g_->get_attrib_by_name<delimiting_polygon_y_att>(r);
		if (px.has_value() and py.has_value())
		{
			const auto &xs = px->get();
			const auto &ys = py->get();
			const std::size_t n = std::min(xs.size(), ys.size());
			if (n >= 3)
			{
				scene.ground.polygon.reserve(n);
				for (std::size_t i = 0; i < n; ++i)
					scene.ground.polygon.push_back({xs[i], ys[i]});
				scene.ground.height = g_->get_attrib_by_name<room_height_att>(r).value_or(2.5f);
				scene.ground.valid  = true;
			}
		}
	}

	// ── Optional-layer filter, applied LAST on purpose. The hidden kinds still had to be built:
	// agent nodes are what resolve CRDT provenance into the owner hue that tints every instance, and
	// an affordance is what a has_intention edge points at. Building then dropping keeps the visible
	// nodes coloured exactly as they would be with everything shown.
	const auto hidden = [&](const Kind k)
	{
		return (k == Kind::Affordance and not cfg_.show_affordances)
		    or (k == Kind::Agent and not cfg_.show_agents);
	};

	std::erase_if(scene.nodes, [&](const Node3D &n) { return hidden(n.kind); });

	std::unordered_set<std::uint64_t> kept;
	kept.reserve(scene.nodes.size());
	for (const auto &n : scene.nodes)
		kept.insert(n.id);
	// An edge with a dropped endpoint goes with it — including every ownership ribbon once the
	// agent stratum is hidden, with no special case needed.
	std::erase_if(scene.edges,
	              [&](const Edge3D &e) { return not kept.contains(e.from) or not kept.contains(e.to); });

	return scene;
}

}   // namespace rc::graph3d
