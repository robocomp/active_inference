/*
 * object_approach.h — every object's `approach` affordance: ONE node, the sides it can be approached from listed
 * INSIDE it as options. SHARED by the concept agents. Plan: docs/superpowers/plans/2026-10-18-object-approach-affordance.md.
 *
 * ★ONE ACTION, PARAMETERISED — NOT A NODE PER SIDE (user decision 2026-10-18). "Approach the table from the
 * right" is the same action as "approach the table" with a parameter. The node publishes the options it CAN
 * serve — only those, so a side against a wall is simply not on the list — each with its standpoint, in the
 * parallel aff_option_names / aff_option_x_m / _y_m / _yaw_rad attributes. names[0] is the default and its pose
 * is the node's own epistemic_target_*, so a consumer that knows nothing of options still drives somewhere valid.
 * The consumer picks one by writing executing_option on its `executing` edge. Still "the object proposes":
 * a consumer may only choose a name listed here.
 *
 * Which options (object_approach_geometry.h has the standoff rule and the side geometry):
 *   Fronted     — front, left, back, right from the object's front (its OWN left/right); front is the default.
 *                 Front not known yet (a square chair seat's yaw mode unsettled) → the box faces, as Frontless.
 *   Frontless   — the four box faces named by room axis (xp, yp, xn, yn); the face towards the robot first.
 *   DefaultOnly — one option, "default": from the front when known, else from the robot's side (a person).
 *
 * The precondition is "at least one option fits in the room" (1 or 0). No pose / no room polygon this cycle is
 * NOT a failed precondition: the offer holds (PragmaticAffordance rule 3). While the consumer EXECUTES, nothing is
 * rewritten — neither the target (PragmaticAffordance rule 1) nor the options: the claimed side's pose must not move
 * under the robot, and a σ that grows while the object is not looked at cannot push the goal away.
 *
 * MAIN-THREAD ONLY.
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "object_approach_geometry.h"
#include "pragmatic_affordance.h"

namespace rc::pragmatic
{

struct ApproachInputs
{
    bool known = false;                    ///< a fitted pose this cycle AND a robot pose
    Eigen::Vector2f centre{0, 0};          ///< object centre, room frame
    float yaw = 0.0f;                      ///< the footprint's orientation, room frame
    float yaw_sigma = 0.0f;                ///< its σ: keeps face names from flipping near 45°
    Eigen::Vector2f half_extent{0, 0};     ///< footprint half sizes along the object's own x, y
    std::optional<float> front_yaw{};      ///< the object's front, room frame; nullopt = none, or not known yet
    Eigen::Matrix2f pos_cov = Eigen::Matrix2f::Zero();   ///< object position Σ, room frame
    Eigen::Vector2f robot{0, 0};           ///< robot, room frame
    float robot_pos_sigma_m = 0.0f;
    std::optional<std::vector<Eigen::Vector2f>> room_polygon{};   ///< nullopt = unknown → the offer holds
};

struct ApproachCfg
{
    approach::StandCfg stand{};   ///< clearance (half the body diagonal; a person's social distance) and k
    float turn_radius_m = 0.398f; ///< an option's standpoint must be this far from every wall: half the body diagonal
    float value = 0.3f;           ///< what the consumer ranks on before the robot arrives
    float arrival_m = 0.25f;      ///< the controller's goal_threshold (trajectory_controller), where value halves
};

class ObjectApproach
{
public:
    enum class Sides : std::uint8_t { Fronted, Frontless, DefaultOnly };

    struct Option
    {
        std::string    name;
        approach::Pose pose;
    };
    struct Completion
    {
        std::string             option;    ///< the side the consumer had claimed ("" = it named none: the default)
        rc::affordance::Outcome outcome;
    };

    void init(std::shared_ptr<DSR::DSRGraph> G, std::uint64_t object_node_id, std::string object_name,
              PragmaticAffordance::Policy policy, Sides sides);

    /// One call per compute cycle, always (pass known = false when nothing is known).
    void update(const ApproachInputs& in, const ApproachCfg& cfg);

    /// The option the consumer is executing, from its `executing` edge; nullopt when nothing claims the node.
    [[nodiscard]] const std::optional<std::string>& executing_option() const { return executing_option_; }
    /// Exactly once per terminal transition: which option and how it ended.
    [[nodiscard]] std::optional<Completion> consume_completion();

    void remove();                              ///< teardown (the object is going away)
    void on_node_deleted(std::uint64_t id);     ///< from del_node_slot

    [[nodiscard]] const PragmaticAffordance& affordance() const { return aff_; }
    [[nodiscard]] const std::vector<Option>& options() const { return options_; }

    /// The options for these inputs, the default first; only those that fit in the room. Pure, for tests and logs.
    [[nodiscard]] std::vector<Option> compute_options(const ApproachInputs& in, const ApproachCfg& cfg);

private:
    void write_options();
    void read_executing_option();

    std::shared_ptr<DSR::DSRGraph> G_;
    PragmaticAffordance aff_;
    Sides sides_ = Sides::Frontless;
    approach::FaceFrame faces_;
    std::vector<Option> options_;          ///< as last computed while not executing
    std::vector<Option> written_;          ///< as last written to the node
    std::uint64_t written_node_ = 0;       ///< the node those were written to (a re-created node gets them again)
    std::optional<std::string> executing_option_;
    std::string claimed_option_;           ///< kept past the claim, for the completion
};

}   // namespace rc::pragmatic
