// ObjectApproach (object_approach.h): ONE `approach` node per object, the sides it can be approached from listed
// INSIDE it as options (aff_option_*), the default first. On a real DSR graph, private DDS domain.
// Build+run (cortex SOURCE headers first: the aff_option_* attributes are registered there before a reinstall):
//   C=~/robocomp/components/cortex; g++ -std=c++23 -O0 -g -fPIC -I$C/api/include -I$C/core/include \
//     -I/usr/include/eigen3 $(pkg-config --cflags Qt6Core) -DSHADOW_JSON='"'$PWD/../../../robot_concept/shadow.json'"' \
//     -o /tmp/toa test_object_approach.cpp ../object_approach.cpp ../pragmatic_affordance.cpp \
//     ../../affordance_manager/affordance_manager.cpp -ldsr_api -ldsr_core -lfastcdr -lfastdds \
//     $(pkg-config --libs Qt6Core) && /tmp/toa
#include "../object_approach.h"

#include <cmath>
#include <numbers>
#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok)
    {
        ++failures;
        std::println("FAIL line {}: {}", line, what);
    }
}
#define CHECK(x) check((x), __LINE__, #x)
bool near(double a, double b, double tol = 1e-3) { return std::abs(a - b) < tol; }

using rc::pragmatic::ObjectApproach;
using rc::pragmatic::ApproachInputs;
using rc::pragmatic::ApproachCfg;
using V = Eigen::Vector2f;
constexpr float pi = std::numbers::pi_v<float>;

struct World
{
    std::shared_ptr<DSR::DSRGraph> G;
    std::uint64_t object_id = 0;
    std::uint64_t consumer_id = 0;
};

World make_world(int k, const std::string& object)
{
    World w;
    w.G = std::make_shared<DSR::DSRGraph>("object_approach_test_" + std::to_string(k), 980 + k,
                                          std::string(SHADOW_JSON), true, static_cast<int8_t>(70 + k), DSR::Queue);
    DSR::Node room = DSR::Node::create<room_node_type>("room_1");
    w.G->add_or_modify_attrib_local<level_att>(room, 1);
    w.G->insert_node(room);
    DSR::Node obj = DSR::Node::create<object_node_type>(object);
    w.G->add_or_modify_attrib_local<level_att>(obj, 2);
    w.object_id = w.G->insert_node(obj).value();
    DSR::Node agent = DSR::Node::create<agent_node_type>("controller");
    w.consumer_id = w.G->insert_node(agent).value();
    return w;
}

/// A 10 × 10 m room, the object at (5, 5), the robot at (2, 5), everything known.
ApproachInputs inputs()
{
    ApproachInputs in;
    in.known        = true;
    in.centre       = V{5, 5};
    in.yaw          = 0.0f;
    in.yaw_sigma    = 0.02f;
    in.half_extent  = V{0.3f, 0.3f};
    in.pos_cov      = Eigen::Matrix2f::Zero();
    in.robot        = V{2, 5};
    in.robot_pos_sigma_m = 0.0f;
    in.room_polygon = std::vector<V>{{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    return in;
}

constexpr rc::pragmatic::PragmaticAffordance::Policy fast{.offer_prob = 0.6f, .withdraw_prob = 0.35f, .stable_cycles = 1};

std::optional<DSR::Node> node(World& w, const std::string& name) { return w.G->get_node(name); }

std::vector<std::string> names(World& w, const std::string& n)
{
    const auto nd = node(w, n);
    std::vector<std::string> out;
    if (not nd) return out;
    const std::string s = w.G->get_attrib_by_name<aff_option_names_att>(*nd).value_or("");
    std::size_t a = 0;
    while (a <= s.size() and not s.empty())
    {
        const auto b = s.find(',', a);
        out.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) break;
        a = b + 1;
    }
    return out;
}

std::vector<float> floats(World& w, const std::string& n, auto att)
{
    const auto nd = node(w, n);
    if (not nd) return {};
    const auto v = w.G->get_attrib_by_name<decltype(att)>(*nd);
    return v.has_value() ? std::vector<float>(v.value().get()) : std::vector<float>{};
}

bool has(const std::vector<std::string>& v, std::string_view s) { return std::ranges::find(v, s) != v.end(); }

/// The consumer claims the node (protocol flags) and writes its executing edge, naming an option.
void claim(World& w, const std::string& n, const std::string& option, float x, float y, float yaw)
{
    auto nd = node(w, n);
    w.G->add_or_modify_attrib_local<active_att>(*nd, true);
    w.G->update_node(*nd);
    auto e = DSR::Edge::create<executing_edge_type>(w.consumer_id, nd->id());
    w.G->add_or_modify_attrib_local<executing_epoch_att>(e, w.G->get_attrib_by_name<epistemic_target_epoch_att>(*nd).value_or(1));
    w.G->add_or_modify_attrib_local<executing_target_x_m_att>(e, x);
    w.G->add_or_modify_attrib_local<executing_target_y_m_att>(e, y);
    w.G->add_or_modify_attrib_local<executing_target_yaw_rad_att>(e, yaw);
    w.G->add_or_modify_attrib_local<executing_option_att>(e, option);
    w.G->insert_or_assign_edge(e);
}
void complete(World& w, const std::string& n, rc::affordance::Outcome o)
{
    auto nd = node(w, n);
    rc::affordance::write_outcome(*w.G, *nd, o);
    w.G->add_or_modify_attrib_local<active_att>(*nd, false);
    w.G->add_or_modify_attrib_local<epistemic_pending_att>(*nd, false);
    w.G->update_node(*nd);
    w.G->delete_edge(w.consumer_id, nd->id(), "executing");
}
}   // namespace

int main()
{
    const ApproachCfg cfg{};   // clearance 0.398, k 2, value 0.3, arrival 0.25, turn radius 0.398

    // ── a frontless object: ONE node, its four faces as options, the face towards the robot first ───────────────
    {
        auto w = make_world(0, "table_1");
        ObjectApproach a;
        a.init(w.G, w.object_id, "table_1", fast, ObjectApproach::Sides::Frontless);
        a.update(inputs(), cfg);
        CHECK(node(w, "aff_table_1_approach").has_value());
        CHECK(not node(w, "aff_table_1_approach_xn").has_value());     // no node per side
        const auto n = names(w, "aff_table_1_approach");
        CHECK(n.size() == 4 and n.front() == "xn");                    // the robot is on −x: that face is the default
        CHECK(has(n, "xp") and has(n, "yp") and has(n, "yn"));
        const auto xs = floats(w, "aff_table_1_approach", aff_option_x_m_att{});
        const auto ys = floats(w, "aff_table_1_approach", aff_option_y_m_att{});
        const auto ws = floats(w, "aff_table_1_approach", aff_option_yaw_rad_att{});
        CHECK(xs.size() == 4 and ys.size() == 4 and ws.size() == 4);
        CHECK(not xs.empty() and near(xs[0], 5.0 - 0.3 - 0.398) and near(ys[0], 5.0) and near(ws[0], 0.0));
        const auto nd = node(w, "aff_table_1_approach");               // the node's own target IS the default
        CHECK(nd and near(w.G->get_attrib_by_name<epistemic_target_x_m_att>(*nd).value_or(0), 5.0 - 0.3 - 0.398));
        CHECK(nd and w.G->get_attrib_by_name<aff_kind_att>(*nd).value_or("") == "approach");
        a.remove();
        CHECK(not node(w, "aff_table_1_approach").has_value());
    }

    // ── a side the room cannot hold is not offered: the object proposes only what is possible ───────────────────
    {
        auto w = make_world(1, "table_2");
        ObjectApproach a;
        a.init(w.G, w.object_id, "table_2", fast, ObjectApproach::Sides::Frontless);
        auto in = inputs();
        in.centre = V{9.3f, 5};   // against the +x wall: the xp standpoint would be outside
        in.robot  = V{6, 5};
        a.update(in, cfg);
        const auto n = names(w, "aff_table_2_approach");
        CHECK(n.size() == 3 and not has(n, "xp") and n.front() == "xn");
    }

    // ── a fronted object: front first (the default), its own left/right ─────────────────────────────────────────
    {
        auto w = make_world(2, "chair_1");
        ObjectApproach a;
        a.init(w.G, w.object_id, "chair_1", fast, ObjectApproach::Sides::Fronted);
        auto in = inputs();
        in.front_yaw = pi / 2;   // the seat faces +y; the robot is on −x
        a.update(in, cfg);
        const auto n = names(w, "aff_chair_1_approach");
        CHECK(n.size() == 4 and n.front() == "front" and has(n, "left") and has(n, "right") and has(n, "back"));
        const auto ys = floats(w, "aff_chair_1_approach", aff_option_y_m_att{});
        CHECK(not ys.empty() and near(ys[0], 5.0 + 0.3 + 0.398));      // in front of the seat, not on the robot's side

        // the front not known yet (a square seat's yaw mode unsettled): the box faces, the robot's side first
        auto w2 = make_world(3, "chair_2");
        ObjectApproach b;
        b.init(w2.G, w2.object_id, "chair_2", fast, ObjectApproach::Sides::Fronted);
        in.front_yaw.reset();
        b.update(in, cfg);
        const auto m = names(w2, "aff_chair_2_approach");
        CHECK(m.size() == 4 and m.front() == "xn" and not has(m, "front"));
    }

    // ── a person: one option, a social distance, no footprint ────────────────────────────────────────────────────
    {
        auto w = make_world(4, "person_1");
        ObjectApproach a;
        a.init(w.G, w.object_id, "person_1", fast, ObjectApproach::Sides::DefaultOnly);
        auto in = inputs();
        in.half_extent = V{0, 0};
        ApproachCfg pc = cfg;
        pc.stand.clearance_m = 1.2f;
        a.update(in, pc);
        const auto n = names(w, "aff_person_1_approach");
        const auto xs = floats(w, "aff_person_1_approach", aff_option_x_m_att{});
        CHECK(n.size() == 1 and n.front() == "default");
        CHECK(not xs.empty() and near(xs[0], 5.0 - 1.2));               // on the robot's side, 1.2 m off
    }

    // ── unknown is not false: no room polygon / no fit → nothing is offered, a standing offer holds ─────────────
    {
        auto w = make_world(5, "table_3");
        ObjectApproach a;
        a.init(w.G, w.object_id, "table_3", fast, ObjectApproach::Sides::Frontless);
        auto in = inputs();
        in.room_polygon.reset();
        a.update(in, cfg);
        CHECK(not node(w, "aff_table_3_approach").has_value());
        a.update(inputs(), cfg);
        CHECK(node(w, "aff_table_3_approach").has_value());
        in = inputs();
        in.known = false;
        a.update(in, cfg);
        CHECK(node(w, "aff_table_3_approach").has_value());            // held, not withdrawn
    }

    // ── a claim names an option; while it runs the options are FROZEN; the completion says which side ───────────
    {
        auto w = make_world(6, "table_4");
        ObjectApproach a;
        a.init(w.G, w.object_id, "table_4", fast, ObjectApproach::Sides::Frontless);
        a.update(inputs(), cfg);
        const auto xs0 = floats(w, "aff_table_4_approach", aff_option_x_m_att{});
        claim(w, "aff_table_4_approach", "yp", 5.0f, 5.698f, -pi / 2);
        a.update(inputs(), cfg);
        CHECK(a.executing_option() == std::optional<std::string>("yp"));
        auto moved = inputs();
        moved.centre = V{6, 6};          // the object refits a metre away while the robot drives to it
        moved.robot  = V{5, 2};          // and the robot is now on another side
        a.update(moved, cfg);
        CHECK(floats(w, "aff_table_4_approach", aff_option_x_m_att{}) == xs0);
        CHECK(a.options().size() == 4 and near(a.options().front().pose.x, xs0.front()));   // frozen in memory too
        CHECK(names(w, "aff_table_4_approach").front() == "xn");       // not re-ordered under the claim either
        complete(w, "aff_table_4_approach", rc::affordance::Outcome::Satisfied);
        a.update(moved, cfg);
        const auto c = a.consume_completion();
        CHECK(c.has_value() and c->option == "yp" and c->outcome == rc::affordance::Outcome::Satisfied);
        CHECK(not a.consume_completion().has_value());
        a.update(moved, cfg);            // re-offered at the new place, options refreshed
        CHECK(names(w, "aff_table_4_approach").front() == "yn");
    }

    // ── the value falls as the robot arrives at the default standpoint ───────────────────────────────────────────
    {
        auto w = make_world(7, "table_5");
        ObjectApproach a;
        a.init(w.G, w.object_id, "table_5", fast, ObjectApproach::Sides::Frontless);
        auto in = inputs();
        in.robot = V{5.0f - 0.3f - 0.398f, 5};   // standing on it
        in.robot_pos_sigma_m = 0.05f;
        a.update(in, cfg);
        const auto nd = node(w, "aff_table_5_approach");
        CHECK(nd and w.G->get_attrib_by_name<epistemic_gain_att>(*nd).value_or(1) < 0.01f);
    }

    // ── somebody else deletes the node: back to absent, and it comes back ────────────────────────────────────────
    {
        auto w = make_world(8, "table_6");
        ObjectApproach a;
        a.init(w.G, w.object_id, "table_6", fast, ObjectApproach::Sides::Frontless);
        a.update(inputs(), cfg);
        const auto id = node(w, "aff_table_6_approach")->id();
        w.G->delete_node(id);
        a.on_node_deleted(id);
        a.update(inputs(), cfg);
        CHECK(node(w, "aff_table_6_approach").has_value());
        CHECK(names(w, "aff_table_6_approach").size() == 4);
    }

    std::println("test_object_approach: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
