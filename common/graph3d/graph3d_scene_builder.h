#pragma once
/*
 * graph3d_scene_builder.h — turns a live DSR graph into a rc::graph3d::Scene.
 *
 * Everything that needs to know about DSR lives here; the renderer
 * (common/viewers/gl_graph3d_viewer.h) sees only the POD Scene. Any agent can therefore build the
 * stratified view, not just robot_concept.
 *
 * DEPENDS ON dsr_api ONLY — never dsr_gui. The dsr_gui headers define an AbstractGraphicViewer that
 * collides with the component-local one, which is why they must stay confined to a single TU (see
 * robot_concept/src/CMakeLists.txt). Keeping this file gui-free is what lets it be included freely.
 *
 * THREADING: build() calls InnerEigenAPI::get_transformation_matrix() with timestamp==0, which
 * reads and writes an UNLOCKED cache. Per CLAUDE.md that path is safe only single-threaded per
 * instance, and the invalidation slots are queued onto the main thread — so build() is MAIN THREAD
 * ONLY. Every returned optional is checked; a missing node/edge/rtmat simply leaves the node
 * unplaced instead of throwing.
 */

#include "graph3d_scene.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace DSR
{
class DSRGraph;
class InnerEigenAPI;
}

namespace rc::graph3d
{

class SceneBuilder
{
public:
	struct Config
	{
		// Vertical spacing between strata, metres. Sized against a typical apartment room (~5-6 m
		// across) so the stack is legibly tall without dwarfing the floor plan.
		float stratum_gap = 1.20f;
		// Display-only staleness horizon for agent nodes, mirroring rc::AgentStatusOverlay. Greying
		// early costs nothing and is undone the moment the heartbeat resumes.
		int stale_after_ms = 3000;

		// Floor opacity. The floor mesh spans the whole room on the ROOM plane, so drawn opaque it
		// hides the robot on the plane below it and every drop-line that crosses it — the two things
		// that make the stack readable. Translucent, it still reads as a floor and still anchors the
		// footprint, without being a lid.
		float floor_alpha = 0.35f;

		// The robot node carries `path = "meshes/shadow.obj"` — relative to ROBOT_CONCEPT'S OWN run
		// dir, the older convention, whereas every mesh_path this viewer resolves is relative to the
		// components root. Rather than teach the renderer a special case, name the qualified path
		// here. Empty ⇒ fall back to the synthetic Robot glyph.
		std::string robot_mesh_path = "robot_concept/meshes/shadow.obj";

		// The containment ladder — robot → room → instances → meta-1 → meta-2 → … — is always drawn:
		// it IS the view. These two are a second, orthogonal story (what can be done with a thing,
		// and who is doing the believing) that doubles the node count, so they stay off by default.
		// Everything downstream — placement, colour, ownership, edges — already handles them; when
		// enabled they stack as extra planes ON TOP of the deepest meta level.
		bool show_affordances = false;
		bool show_agents      = false;
	};

	SceneBuilder();
	~SceneBuilder();

	SceneBuilder(const SceneBuilder &) = delete;
	SceneBuilder &operator=(const SceneBuilder &) = delete;

	void set_graph(std::shared_ptr<DSR::DSRGraph> graph);
	[[nodiscard]] Config &config() noexcept { return cfg_; }

	// Full rebuild from the current graph state. Cheap enough (~50 nodes) to run at a few Hz.
	// Returns an empty Scene if no graph is attached.
	[[nodiscard]] Scene build();

private:
	struct Classified
	{
		Kind  kind;
		Glyph glyph;
	};

	// nullopt ⇒ do not draw this node at all. Reserved for the RT tree's PLUMBING — the `root`
	// origin and the robot's own chassis/sensor mounts — which are frames, not things believed in.
	[[nodiscard]] static std::optional<Classified> classify(const std::string &type,
	                                                        const std::string &subtype,
	                                                        const std::string &name) noexcept;

	// Stable, well-separated identity colour per agent (golden-angle hue walk). Distinct from the
	// agent's HEALTH colour: health answers "is it working?", identity answers "who wrote this?",
	// and the view needs both at once. Used for the ownership ribbons, not for node fill.
	[[nodiscard]] static Rgb identity_color(std::uint32_t agent_id) noexcept;

	// The canonical class of an object node — "table", "chair", "wall", … — which is what its colour
	// encodes. Mirrors the convention already used by retina/src/scene_processor.cpp
	// (`object_class_of`) and ring_metaconcept: object_subtype wins ONLY when it names a known
	// class, because for tables it historically carried the SHAPE ("round"/"square"); otherwise the
	// node-name prefix decides, and the DSR type is the last resort.
	[[nodiscard]] static std::string class_of(const std::string &type, const std::string &subtype,
	                                          const std::string &name);

	// Fixed-order categorical palette, assigned per class and never cycled.
	[[nodiscard]] static Rgb class_color(const std::string &cls) noexcept;

	std::shared_ptr<DSR::DSRGraph>      g_;
	std::unique_ptr<DSR::InnerEigenAPI> inner_;   // one instance, main thread only (ts==0 cache)
	Config                              cfg_;

	// name-prefix → agent node id, derived from the live agent nodes each build (e.g. agent
	// "table_concept 7" claims names starting with "table_"). Fallback for nodes whose CRDT writer
	// id does not resolve.
	std::unordered_map<std::string, std::uint64_t> prefix_owner_;
};

}   // namespace rc::graph3d
