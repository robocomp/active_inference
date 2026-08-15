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
// The inside surface of a wall: a NEUTRAL light grey, deliberately carrying no hue at all.
// An earlier version blended 82 % toward white but kept 18 % of the wall's blue, which came out a
// pale CYAN — and the top face, a 50/50 blend of that with the hue, came out plainly blue-teal.
// Since the top is the largest slice of a wall from a camera above the room, the whole ring still
// read blue. Grey is grey: no amount of shading turns it back into the class colour.
constexpr Rgb kInteriorGrey {0.84f, 0.85f, 0.86f};

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
	// Field/blob nodes are DATA PRODUCTS hung on the tree as a carrier, not things the room
	// contains: the retina's `semantic_grid` lives under `zed`, and residual_concept's `grid`
	// ("residual") is an occupancy field. Neither has a footprint, so both drew as an anonymous
	// sphere on the instances plane.
	if (type == "semantic_grid" or type == "grid")
		return std::nullopt;

	if (type == "agent")
		return Classified{Kind::Agent, Glyph::Pin};
	if (type == "mind")
		return Classified{Kind::Agent, Glyph::Sphere};
	if (type == "affordance" or name.starts_with("aff_"))
		return Classified{Kind::Affordance, Glyph::Diamond};
	// A concept OVER concepts carries its own DSR type — that is precisely what keeps it OUT of every
	// consumer's get_nodes_by_type("object") sweep (the retina would draw its footprint as a solid,
	// residual_concept would carve it) while this view, the one place a rig belongs, draws it. The
	// subtype/name arm below is the legacy path for nodes born before the retype; ring_metaconcept
	// still writes object_subtype="dining_set" (ring_config.h) as the CLASS.
	if (type == "metaconcept" or subtype == "dining_set" or subtype == "rig" or subtype == "ring"
	    or name.starts_with("dining_set"))
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

std::string SceneBuilder::class_of(const std::string &type, const std::string &subtype,
                                   const std::string &name)
{
	// object_subtype is trusted ONLY when it names a class we know. For tables it historically
	// carried the SHAPE ("round"/"square") instead, which is exactly why the repo's own
	// object_class_of helpers fall back to the name prefix rather than believing it blindly.
	static constexpr std::string_view kClasses[] = {"table",  "chair", "cabinet",     "refrigerator",
	                                                "bottle", "door",  "person",      "wall",
	                                                "floor",  "obstacle", "dining_set"};
	for (const std::string_view c : kClasses)
		if (subtype == c)
			return std::string{c};
	for (const std::string_view c : kClasses)
		if (name.starts_with(c))
			return std::string{c};
	// The DSR type is the last resort — it is what walls, floors, obstacles and people carry, since
	// those are not `object` nodes and have no subtype at all.
	if (type == "wall" or type == "floor" or type == "obstacle" or type == "person")
		return type;
	return {};   // unclassified ⇒ folds into the neutral "other" slot
}

Rgb SceneBuilder::class_color(const std::string &cls) noexcept
{
	// Categorical palette, FIXED ORDER, never cycled. Machine-selected against this widget's dark
	// surface (#16181D) on the ALL-PAIRS pairlist — correct here because in a 3-D scene any class can
	// end up next to any other, unlike a bar chart where only neighbours compete.
	//
	// ★The search space is the CONCEPT classes only. Structure (wall, floor) and the reserved
	// singletons (robot, room, affordance, meta, "other") were held FIXED and fed in as obstacles the
	// concepts must also clear, which is what stops a repeat of the two collisions this replaced:
	// door was literally the wall's hue, and refrigerator sat next to meta-violet. Structure staying
	// quiet while the concepts carry the colour is also the right emphasis for the view.
	//
	// Verified all-pairs, concepts: worst normal-vision ΔE 15.4 (≥15 hard floor), worst CVD ΔE 8.0
	// deutan (meets the 8.0 target, up from 6.6 before). Across the whole scene the only sub-floor
	// pairs are room↔wall (10.0) and room↔floor (14.8) — structure vs structure, pre-existing, and
	// separated by plane and glyph rather than by hue; no concept pair is below the floor.
	// Two knowingly-accepted deviations, both because these are LIT 3-D SOLIDS, not flat chart fills:
	//   · three slots sit outside the dark-mode L band [0.48,0.67]. The band exists so flat fills read
	//     consistently on a flat surface, but the fragment shader already multiplies every colour by
	//     a shade term in [0.34,1.0] with face orientation, so a slot's on-screen lightness sweeps a
	//     ~3× range within one glyph. Pairwise separation is the property that survives that, and it
	//     is what was optimised — trading the band for it raised CVD from 5.5 to 8.0.
	//   · one slot is just under 3:1 against the surface. The documented relief for that is a visible
	//     label, and this view has always-on per-node labels plus a legend.
	if (cls == "table")        return {0.722f, 0.510f, 0.075f};   // #b88213 ochre — wood
	if (cls == "chair")        return {0.027f, 0.678f, 0.663f};   // #07ada9 teal
	if (cls == "cabinet")      return {0.043f, 0.894f, 0.580f};   // #0be494 mint
	if (cls == "refrigerator") return {1.000f, 0.435f, 0.478f};   // #ff6f7a coral
	if (cls == "bottle")       return {0.475f, 0.016f, 0.945f};   // #7904f1 violet
	if (cls == "person")       return {0.855f, 0.035f, 0.357f};   // #da095b crimson
	// A door is no longer given the wall's hue. It IS part of a wall, but that reasoning made the two
	// literally indistinguishable, and a doorway is one of the few things in the scene you actively
	// look FOR — so it gets the most conspicuous slot rather than the wall's.
	if (cls == "door")         return {0.867f, 0.027f, 0.753f};   // #dd07c0 magenta
	// ── Structure: deliberately quiet.
	// ★ A wall is GREY, not blue. It was #006d95 through five rounds of "the inner faces are still
	// blue" — each round I repainted one more surface (interior face, then the top, then per-wall
	// containment) while leaving the class hue on the others, so some blue always survived somewhere
	// on the slab and the report never changed. A wall has five visible faces and no camera shows
	// them all; "make the blue face grey" is unwinnable one face at a time. Grey everywhere ends it,
	// and it is the better design anyway: structure should be quiet so the concepts carry the colour
	// (the same reasoning that made wall/floor FIXED ANCHORS of the palette search rather than
	// members of it). Verified against the concept palette: worst pair #c6c9cc↔mint CVD ΔE 6.5.
	// The interior face and top are lighter still — see kInteriorGrey — so the slab keeps its form.
	if (cls == "wall")         return {0.52f,  0.535f, 0.55f};    // #858b8c — exterior/legend
	if (cls == "floor")        return {0.447f, 0.392f, 0.000f};   // #726400
	return kSteel;                                                // "other": obstacles, unknown
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
		node.cls     = class_of(n.type(), subtype, n.name());
		node.kind    = kind;
		node.glyph   = glyph;
		// The agent owns its own appearance (same contract the retina consumes); the viewer stays
		// type-agnostic and falls back to the synthetic glyph when no mesh is published.
		node.mesh_path = g_->get_attrib_by_name<mesh_path_att>(n).value_or("");
		// The robot predates that contract: shadow.json puts its mesh in `path`, relative to
		// robot_concept's own run dir rather than to the components root, so it cannot be resolved
		// the same way as everyone else's. Substitute the qualified path from config — the whole
		// point of drawing the real chassis is that its silhouette says "this is the robot" without
		// a legend, and a two-box glyph does not.
		if (kind == Kind::Robot and node.mesh_path.empty())
			node.mesh_path = cfg_.robot_mesh_path;

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
		const float h = g_->get_attrib_by_name<height_m_att>(n).value_or(0.0f);
		node.radius = (w > 0.0f or d > 0.0f) ? std::clamp(0.35f * std::max(w, d), 0.10f, 0.32f)
		                                     : (kind == Kind::Agent ? 0.16f : 0.12f);
		node.draw   = {2.0f * node.radius, 2.0f * node.radius, 2.0f * node.radius};

		// ── Published proportions beat the isotropic cube ────────────────────────────────────
		// Every display mesh in this repo is authored as a UNIT BOX (x,y ∈ [-0.5,0.5], z ∈ [0,1])
		// precisely so the consumer stretches it by the node's own width_m/depth_m/height_m. Drawn
		// into a 2r cube instead, the asset carries none of its own shape and the node's published
		// geometry is thrown away: roughly-cubic furniture survives that, but a DOOR (0.70 × 0.05 ×
		// 2.0 m) came out as a ~0.5 m box that looked nothing like the door it publishes, and a
		// bottle fares no better.
		//
		// So whenever all three dimensions are published, keep their RATIOS and pick one uniform
		// scale — the largest that fits both budgets this view imposes: the height must not punch
		// through the plane above, and the footprint must stay in the same size band as its
		// neighbours' glyphs. Uniform, so shape is preserved rather than three independent
		// distortions. Walls and the floor are excluded below: their footprint must stay TRUE
		// because tracing the real room outline is their entire job.
		if (w > 0.0f and d > 0.0f and h > 0.0f)
		{
			const float k = std::min(0.40f * cfg_.stratum_gap / h,
			                         2.0f * node.radius / std::max(w, d));
			node.draw = {w * k, d * k, h * k};
		}

		if (node.cls == "floor")
		{
			// The floor's OBJ is the room layout, centred on the polygon's bbox — so it is drawn at
			// TRUE footprint size (width_m/depth_m are that bbox) and flat. Its z span is degenerate,
			// which add_mesh() handles.
			node.draw = {std::max(w, 0.05f), std::max(d, 0.05f), 0.01f};
			// Room-sized and opaque, it is a lid: it hides the robot one rung below and every
			// drop-line crossing it, i.e. the two things that make the vertical stack readable.
			node.alpha = cfg_.floor_alpha;
		}
		else if (kind == Kind::Robot)
		{
			// Fit the real chassis into a box about half the inter-plane gap tall, PRESERVING its
			// proportions — a 1.4 m robot stretched into a 0.24 m cube would be unrecognisable, and
			// recognisability is the entire reason for using the mesh. Footprint follows from the
			// aspect, so only the height is chosen here.
			node.draw        = {2.0f * node.radius, 2.0f * node.radius, 0.55f * cfg_.stratum_gap};
			node.keep_aspect = true;
		}
		else if (node.cls == "wall")
		{
			// A wall panel is drawn at its TRUE length so the walls trace the real footprint and line
			// up with the room polygon a rung below. Its height is NOT true, though: a 2.5 m wall
			// would punch straight through the plane above it, so it is capped to a fraction of the
			// inter-plane gap. Thickness is nominal but NOT hairline: the renderer builds a wall as a
			// solid slab so its room-facing face can be told apart, and that face needs enough depth
			// separation from the outer one to read as a surface rather than an edge.
			node.draw = {std::max(w, 0.05f), 0.06f, 0.40f * cfg_.stratum_gap};
		}
		// A door deliberately does NOT get the wall treatment, even though it sits on a wall. Matching
		// the wall means inheriting the wall's vertical compression (true length, height squashed to
		// the plane budget) — and a wall is a long low ribbon, so a door built that way comes out
		// 0.70 m wide by 0.38 m tall: LANDSCAPE, the one shape a door never is. Being recognisable as
		// a door beats lining up with the wall underneath, so it falls through to the general
		// proportional rule above and stays portrait. It ends up narrower than its true opening;
		// that is the price, and the wall below still carries the true footprint.

		node.owner_agent_id = n.agent_id();

		index_of[node.id]      = scene.nodes.size();
		id_of_name[node.name]  = node.id;
		scene.nodes.push_back(std::move(node));
	}

	// The ROOM node is drawn at its TRUE pose, which — the reference frame being the room itself —
	// is the room frame's ORIGIN. So the robot→room RT edge terminates on that origin, and the
	// room→instance edges fan out of it, which is exactly what the instances are measured against:
	// every pose here comes from get_transformation_matrix(room, node), i.e. room coordinates.
	// (An earlier revision pinned this glyph above the robot to shorten those diagonals. That is
	// deliberately NOT done any more: now that the room plane carries the floor mesh and the wall
	// panels, where the origin sits inside the footprint is real information, not clutter.)

	// ── Ground: the real room footprint, so the metric x/y has a visible frame of reference. Read
	// HERE, before anything consumes it — the floor centring just below needs the polygon, and when
	// this lived at the end of build() that centring silently saw an empty vector every frame.
	// (`ground.z` still cannot be set until the ladder exists; pass 1b fills it in.)
	if (const auto rooms = g_->get_nodes_by_type("room"); not rooms.empty())
	{
		const auto &r  = rooms.front();
		const auto  px = g_->get_attrib_by_name<delimiting_polygon_x_att>(r);
		const auto  py = g_->get_attrib_by_name<delimiting_polygon_y_att>(r);
		if (px.has_value() and py.has_value())
		{
			const auto &xs = px->get();
			const auto &ys = py->get();
			const std::size_t np = std::min(xs.size(), ys.size());
			if (np >= 3)
			{
				scene.ground.polygon.reserve(np);
				for (std::size_t i = 0; i < np; ++i)
					scene.ground.polygon.push_back({xs[i], ys[i]});
				scene.ground.height = g_->get_attrib_by_name<room_height_att>(r).value_or(2.5f);
				scene.ground.valid  = true;
			}
		}
	}

	// The floor node's RT pose is the room ORIGIN — a corner of the layout, since it is only a
	// semantic parent for floor-attached objects. Its mesh is centred on the polygon's bbox, so the
	// glyph is moved to that centre; otherwise the floor is drawn a half-room off (and its label
	// sits in the corner). Same reasoning as pinning the room glyph above the robot.
	if (scene.ground.polygon.size() >= 3)
	{
		float minx = scene.ground.polygon[0][0], maxx = minx;
		float miny = scene.ground.polygon[0][1], maxy = miny;
		for (const auto &p : scene.ground.polygon)
		{
			minx = std::min(minx, p[0]); maxx = std::max(maxx, p[0]);
			miny = std::min(miny, p[1]); maxy = std::max(maxy, p[1]);
		}
		for (auto &n : scene.nodes)
			if (n.cls == "floor")
			{
				n.pos[0] = 0.5f * (minx + maxx);
				n.pos[1] = 0.5f * (miny + maxy);
				n.placed = true;
			}
	}

	// ── Pass 1b: the LADDER. Both of its tails are open-ended, which is what makes the number of
	// planes a per-build quantity rather than an enum.
	std::unordered_map<std::uint64_t, Kind>        kind_of;
	std::unordered_map<std::uint64_t, std::string> cls_of;
	for (const auto &n : scene.nodes)
	{
		kind_of[n.id] = n.kind;
		cls_of[n.id]  = n.cls;
	}

	std::unordered_set<std::uint64_t> visiting;   // guard: a malformed graph can hold a cycle

	// (i) INSTANCE LEVELS, resolved recursively so two rules fall out of one definition:
	//   · the ROOM SHELL — the floor and the walls — IS the room, so it sits ON the room's own
	//     plane: the floor mesh lies flat there and the wall panels rise vertically out of it,
	//     which is what makes that plane read as a room rather than as a scatter of glyphs.
	//     Everything the room merely CONTAINS is lifted one rung clear of it.
	//   · a PART sits one rung above its WHOLE — a door on its wall, a bottle on the table
	//     bottle_concept re-parented it onto — because it hangs off it in the RT tree.
	// The recursion composes them: level(door) = level(wall)+1 puts a door directly above the wall
	// it belongs to even though the wall lives a rung lower than ordinary furniture.
	const auto is_shell = [&](const std::uint64_t id)
	{
		const auto it = cls_of.find(id);
		return it != cls_of.end() and (it->second == "wall" or it->second == "floor");
	};

	std::unordered_map<std::uint64_t, int> inst_level;
	std::function<int(std::uint64_t)>      inst_level_of = [&](const std::uint64_t id) -> int
	{
		if (const auto it = inst_level.find(id); it != inst_level.end())
			return it->second;
		if (not visiting.insert(id).second)
			return kLevelInstance;
		int lvl = kLevelInstance;   // a child of the room (or of anything undrawn)
		if (is_shell(id))
			lvl = kLevelRoom;       // the shell IS the room, not something standing in it
		else if (const auto p = rt_parent.find(id); p != rt_parent.end())
			if (const auto k = kind_of.find(p->second); k != kind_of.end() and k->second == Kind::Instance)
				lvl = inst_level_of(p->second) + 1;
		visiting.erase(id);
		inst_level[id] = lvl;
		return lvl;
	};

	int deepest_inst = kLevelInstance;
	for (const auto &n : scene.nodes)
		if (n.kind == Kind::Instance)
			deepest_inst = std::max(deepest_inst, inst_level_of(n.id));

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
			case Kind::Instance:   node.level = inst_level_of(node.id);               break;
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

	// ── Pass 4: colour. Instances are coloured by their CLASS, not by their owning agent: "which of
	// these is a chair" is the question the eye actually asks of this view, and a class palette is
	// stable across runs where an agent-id hash is not. Ownership has not been lost — it still drives
	// the selection ribbons, where it is asked for explicitly.
	// Robot / room / meta keep RESERVED colours outside the categorical set: each is a singleton on
	// a plane of its own, so it never competes with the class palette for separation.
	for (auto &node : scene.nodes)
	{
		if (node.type == "agent")
			continue;
		switch (node.kind)
		{
			case Kind::Robot:      node.color = kRobot;                        break;
			case Kind::Room:       node.color = kSlate;                        break;
			case Kind::Affordance: node.color = kAmber;                        break;
			case Kind::Meta:       node.color = kViolet;                       break;
			case Kind::Agent:      node.color = kGrey;                         break;   // mind
			case Kind::Instance:   node.color = class_color(node.cls);         break;
		}
		if (not node.placed)
			node.color = mix(node.color, kGrey, 0.6f);   // no RT chain — say so instead of lying
		node.color_inside = node.color;   // single-sided by default; walls override just below
	}

	// ── Walls get a lit INSIDE face. A room is experienced from within, and a ring of uniformly
	// coloured panels gives no clue which side of one you are looking at — the enclosure reads as a
	// fence rather than a room. Painting the interior surface near-white is what real rooms look
	// like, and it makes "inside" legible from any camera angle without a single extra glyph.
	//
	// Which side is interior CANNOT come from the mesh: wall.obj is a zero-thickness quad whose two
	// faces are the same triangles, and its winding is an authoring detail that could flip. It has to
	// come from the layout.
	//
	// ★ AND IT CANNOT COME FROM THE POLYGON'S CENTROID EITHER. That was the first attempt, and it is
	// wrong for exactly the scenes we run: the apartment is one non-convex layout whose 30 wall nodes
	// bound SEVERAL spaces, while `ground.polygon` is a single room's outline. For any wall not facing
	// that room, "toward the centroid" is not "toward the space this wall encloses", so roughly half
	// the ring came out inverted — grey on the street side, blue facing the room. A centroid answers
	// "which way is the middle", and the question is "which side is indoors".
	//
	// So: test CONTAINMENT. Step a short distance off each face and ask which probe lands INSIDE the
	// polygon (even-odd ray cast, correct for non-convex outlines). Exactly one inside ⇒ that is the
	// interior. Both or neither ⇒ this wall does not bound this room (an interior partition, or a wall
	// of another space); say so by leaving it single-sided rather than guessing.
	if (scene.ground.valid and scene.ground.polygon.size() >= 3)
	{
		const auto &poly = scene.ground.polygon;
		const auto inside_polygon = [&poly](const float px, const float py)
		{
			bool in = false;
			for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
				if ((poly[i][1] > py) != (poly[j][1] > py) and
				    px < (poly[j][0] - poly[i][0]) * (py - poly[i][1]) / (poly[j][1] - poly[i][1]) + poly[i][0])
					in = not in;
			return in;
		};
		// Probe distance: clear of the slab's own thickness, well under a room's width.
		constexpr float kProbeM = 0.25f;

		for (auto &node : scene.nodes)
		{
			if (node.cls != "wall" or not node.placed)
				continue;
			// The panel's local +y in world coordinates — its inward/outward normal axis.
			const float ny = std::cos(node.yaw), nx = -std::sin(node.yaw);
			const bool plus_in  = inside_polygon(node.pos[0] + kProbeM * nx, node.pos[1] + kProbeM * ny);
			const bool minus_in = inside_polygon(node.pos[0] - kProbeM * nx, node.pos[1] - kProbeM * ny);
			if (plus_in == minus_in)
				continue;   // both sides indoors, or neither — not this room's boundary, stay one-sided
			node.inside_sign  = plus_in ? 1.0f : -1.0f;
			// Flat grey, NOT a blend with the wall hue — see kInteriorGrey. The wall's class colour
			// survives on its outward faces, which is where it is still needed to identify the wall.
			node.color_inside = kInteriorGrey;
		}
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
