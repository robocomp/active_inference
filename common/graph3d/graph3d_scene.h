#pragma once
/*
 * graph3d_scene.h — the POD vocabulary of a STRATIFIED SPATIAL view of the DSR graph.
 *
 * The planar dsr_gui GraphViewer draws every edge with the same weight, so the two relations that
 * actually structure this system are invisible in it: which AGENT owns a node, and which
 * META-CONCEPT groups a set of instances. Neither is a transform, so a spatial force layout has
 * nothing to lay them out by.
 *
 * This scene fixes that by splitting the two roles of the layout:
 *
 *     z  =  the CONTAINMENT LADDER   robot → room → instances → meta-1 → meta-2 → …
 *    x,y =  the node's TRUE METRIC POSE in the room frame
 *
 * so hierarchy becomes literally vertical while spatial meaning is preserved horizontally. The
 * ladder mirrors the graph's real parenting (`room` IS a child of the robot here — see
 * room_scene_graph.cpp), and the meta tail is OPEN-ENDED: a rig whose members are instances sits on
 * meta-1, a rig grouping those rigs sits on meta-2, and so on for as deep as `group_member` goes.
 *
 * `kind` says WHAT a node is; `level` says WHICH PLANE it landed on. They are deliberately separate
 * — the number of planes depends on how deep the meta nesting currently runs, so it is computed per
 * build and cannot be an enum.
 *
 * Deliberately dependency-free (no Qt, no DSR): rc::graph3d::SceneBuilder fills it from the graph
 * and rc::viewers::GLGraph3DViewer draws it, and neither needs to know about the other. Positions
 * are in the room frame, METRES, Z-UP — the renderer owns the mapping to its own GL convention.
 */

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rc::graph3d
{

using Vec3 = std::array<float, 3>;   // room frame, metres, z-up
using Rgb  = std::array<float, 3>;   // 0..1

// What a node IS. Note this is NOT the plane it sits on: every Meta node is Kind::Meta, but they
// spread across meta-1, meta-2, … according to how deep their grouping nests.
enum class Kind : int
{
	Robot,        // the one `robot` node — bottom of the ladder, everything hangs off it
	Room,         // the container the instances live in
	Instance,     // walls, floor, tables, chairs, cabinets, bottles, obstacles, people…
	Meta,         // a concept OVER concepts (dining_set / rig), level set by nesting depth
	Affordance,   // aff_* — a property of an instance, not a rung of the ladder (off by default)
	Agent,        // agent nodes + mind — who does the believing (off by default)
};

// Fixed bottom of the ladder. Meta levels start at kLevelInstance + 1 and grow from there.
inline constexpr int kLevelRobot    = 0;
inline constexpr int kLevelRoom     = 1;
inline constexpr int kLevelInstance = 2;
inline constexpr int kBaseLevels    = 3;

// Glyph shape carries the node's kind at a glance, so the plane it sits on is confirmed by its
// silhouette rather than by reading the label.
enum class Glyph : int
{
	Box,       // furniture-like instances (table, cabinet, refrigerator, obstacle) + walls
	Cylinder,  // bottles, cylindrical fits
	Chair,     // seat + backrest tell, so orientation is readable
	Sphere,    // generic / unclassified
	Diamond,   // affordances
	Ring,      // meta-concepts (a rig IS a ring of slots)
	Pin,       // agents — an inverted cone, pointing DOWN at what it owns
	Robot,     // the robot itself
};

// Edge kinds are rendered by MEANING, not by DSR type string, because the same "RT" carries both a
// real containment (room→table) and a mere mount link.
enum class EdgeKind : int
{
	RT,         // transform / spatial containment          — thin, purple, straight
	Intention,  // object --has_intention--> affordance      — amber, vertical
	Member,     // rig --group_member--> member              — translucent curved bundle (THE meta cue)
	Has,        // mind --has--> agent                       — dashed grey
	Ownership,  // agent ⇢ every node it wrote               — very translucent, agent's identity hue
};

struct Node3D
{
	std::uint64_t id = 0;
	std::string   name;
	std::string   type;        // DSR node type
	std::string   subtype;     // object_subtype, when present
	std::string   cls;         // canonical class ("table"/"chair"/…) — what the colour encodes
	// Display mesh the OWNING AGENT published (repo-relative, e.g. "chair_concept/meshes/chair.obj").
	// When present the renderer draws THAT instead of a synthetic glyph, so the silhouette and its
	// axis convention come from the agent rather than being re-derived — and re-derived wrongly.
	std::string   mesh_path;
	Vec3          pos{0, 0, 0};
	Rgb           color{0.7f, 0.75f, 0.8f};
	// Two-sided panels (walls). A wall asset is a ZERO-THICKNESS quad, so its two faces are
	// geometrically the same triangles and cannot be shaded apart — the renderer instead emits the
	// panel twice, displaced half a thickness either way along its local y, and paints the copy on
	// the room-interior side with `color_inside`.
	//   inside_sign  0 ⇒ single-sided, draw once in `color` (every node but a wall)
	//               ±1 ⇒ the sign of LOCAL Y that points into the room
	// Which sign that is depends on the wall's yaw and on where the room's interior actually is, so
	// it is resolved from the room polygon rather than from the mesh's winding — an asset authored
	// with the opposite face order must not silently turn every room inside out.
	Rgb           color_inside{0.7f, 0.75f, 0.8f};
	float         inside_sign = 0.0f;
	float         radius = 0.10f;
	// Metre size the display mesh is drawn at, per LOCAL axis (x,y = footprint, z = height). Mostly
	// isotropic and schematic (2·radius cubed), because this is a graph view rather than a scene
	// view — but a wall is a long thin panel whose whole point is to trace the real footprint, so it
	// needs its true length and a separately capped height. Glyphs ignore this and use `radius`.
	Vec3          draw{0.2f, 0.2f, 0.2f};
	// How `draw` is honoured. false (default) = STRETCH each local axis independently to fill the
	// box, which is what a schematic cube or a wall panel wants. true = scale UNIFORMLY to fit
	// inside it, preserving the asset's proportions — for a mesh whose silhouette is the point
	// (the robot), squashing it into a cube destroys exactly the thing it was included for.
	bool          keep_aspect = false;
	// Opacity of the solid geometry, 0 = invisible, 1 = opaque. Below 1 the node is drawn in a
	// second, blended pass with depth-writes off, so whatever is behind it stays visible. This is
	// what lets the floor read as a floor without hiding the robot standing a rung underneath it.
	float         alpha = 1.0f;
	Glyph         glyph = Glyph::Sphere;
	Kind          kind  = Kind::Instance;
	int           level = kLevelInstance;
	float         yaw   = 0.0f;         // room-frame heading (rad), for glyphs that show orientation
	std::uint32_t owner_agent_id = 0;   // CRDT writer id; 0 ⇒ unknown
	bool          placed = false;       // false ⇒ no valid RT chain; position is a fallback
	bool          dimmed = false;       // agent heartbeat stale
	std::string   sublabel;             // e.g. "Compute/Operating" or "stale"
};

struct Edge3D
{
	std::uint64_t from = 0;   // node ids (not indices — the renderer resolves them)
	std::uint64_t to   = 0;
	EdgeKind      kind  = EdgeKind::RT;
	Rgb           color{0.6f, 0.5f, 0.8f};
	float         alpha = 1.0f;
};

// The room footprint, drawn on the ROOM plane so the metric x/y has a visible reference and the
// instances above it read as being inside it.
struct Ground
{
	std::vector<std::array<float, 2>> polygon;   // room frame, metres
	float                             height = 0.0f;
	float                             z      = 0.0f;   // == level_z[kLevelRoom]
	bool                              valid  = false;
};

struct Scene
{
	std::vector<Node3D> nodes;
	std::vector<Edge3D> edges;
	Ground              ground;
	// One entry per plane, bottom-up. Size varies with meta nesting depth (and with the optional
	// affordance/agent layers), so both of these are sized per build.
	std::vector<float>       level_z;
	std::vector<std::string> level_names;

	[[nodiscard]] const Node3D *find(const std::uint64_t id) const noexcept
	{
		for (const auto &n : nodes)
			if (n.id == id)
				return &n;
		return nullptr;
	}
};

}   // namespace rc::graph3d
