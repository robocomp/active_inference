// Task 6 (room-transition protocol, spec C2.7): room_concept adopts or rejects ltsm's recognition proposal.
// On a REAL DSR graph (private DDS domain 57): P = room_4 (provisional: `proto` self-edge) with its floor (identity),
// a wall on the floor, the robot, and a foreign node (door_concept's) under P; ltsm's proposal_room_2 under P with
// RT P→proposal = T(P←K). Accept renames P in place (same id) and re-expresses P's OWN children by T(K←P); reject only
// writes the verdict; a proposal on a non-provisional room, or whose name another node holds, is refused without a throw.
//   make -C build proposal_adoption_test && ../bin/proposal_adoption_test        (needs cortex with rename, 43bfe16)
#include "proposal_adoption.h"

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_rt_api.h>
#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <QCoreApplication>
#include <clocale>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>
#include <string>

static int failures = 0;
static void check(bool ok, const std::string& what)
{ if (not ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str()); }
static bool near(float a, float b, float tol = 1e-4f) { return std::abs(a - b) <= tol; }
static bool near_angle(float a, float b, float tol = 1e-4f)
{ return std::abs(std::remainder(a - b, 2.f * std::numbers::pi_v<float>)) <= tol; }

static std::shared_ptr<DSR::DSRGraph> G;
static std::uint64_t robot = 0, P = 0, floor_id = 0, wall = 0, door = 0, prop = 0;
static const Eigen::Vector3f kRobotInP{1.f, 2.f, 0.3f}, kWallInFloor{1.f, 0.f, 0.f}, kDoorInP{0.5f, -1.f, 0.2f};
static const Eigen::Vector3f kK_in_P{2.f, 1.f, std::numbers::pi_v<float> / 2};   // T(P←K)

static void put_rt(std::uint64_t from, std::uint64_t to, const Eigen::Vector3f& q, float z = 0.f)
{
    auto n = G->get_node(from).value();
    G->get_rt_api()->insert_or_assign_edge_RT(n, to, std::vector<float>{q.x(), q.y(), z}, std::vector<float>{0.f, 0.f, q.z()});
}

static const std::array<std::uint64_t, 3> kStamps{1000, 1100, 1200};
static Eigen::Vector3f robot_at(int i) { return kRobotInP + Eigen::Vector3f(0.1f * i, -0.05f * i, 0.02f * i); }

static std::optional<Eigen::Vector3f> rt_of(std::uint64_t from, std::uint64_t to, std::uint64_t ts = 0)
{
    const auto e = G->get_edge(from, to, "RT");
    if (not e.has_value()) return std::nullopt;
    const auto m = G->get_rt_api()->get_edge_RT_as_rtmat(e.value(), ts);
    if (not m.has_value()) return std::nullopt;
    return Eigen::Vector3f(m->translation().x(), m->translation().y(), std::atan2(m->rotation()(1, 0), m->rotation()(0, 0)));
}

static bool same_pose(const std::optional<Eigen::Vector3f>& a, const Eigen::Vector3f& b)
{ return a.has_value() and near(a->x(), b.x()) and near(a->y(), b.y()) and near_angle(a->z(), b.z()); }

static std::uint64_t add(const char* type, const std::string& name, std::uint64_t parent)
{
    DSR::Node n; n.type(type); n.name(name);
    if (parent != 0) G->add_or_modify_attrib_local<parent_att>(n, parent);
    return G->insert_node(n).value();
}

static void reset(bool proto = true, bool with_proposal = true)
{
    for (const char* t : {"robot", "room", "floor", "wall", "door", "metaconcept"})
        for (const auto& n : G->get_nodes_by_type(t)) G->delete_node(n.id());
    const auto root = G->get_node("root").value().id();
    P = add("room", "room_4", root);
    { auto n = G->get_node(P).value(); G->add_or_modify_attrib_local<room_id_att>(n, std::uint64_t{4}); G->update_node(n); }
    put_rt(root, P, {0.f, 0.f, 0.f});
    if (proto) G->insert_or_assign_edge(DSR::Edge::create<proto_edge_type>(P, P));
    floor_id = add("floor", "room_4_floor", P);
    G->get_rt_api()->insert_or_assign_edge_RT_identity(P, floor_id);
    wall = add("wall", "room_4_wall_0", floor_id);
    put_rt(floor_id, wall, kWallInFloor, 1.25f);
    robot = add("robot", "Shadow", P);
    {   // a timestamped HISTORY ring, with a pose covariance (σx 0.2, σy 0.1), as dsr_update_pose writes it
        std::vector<float> cov(36, 0.f);
        cov[0] = 0.04f; cov[7] = 0.01f; cov[35] = 0.001f;
        for (int i = 0; i < 3; ++i)
        {
            auto pn = G->get_node(P).value();
            const auto q = robot_at(i);
            G->get_rt_api()->insert_or_assign_edge_RT(pn, robot, std::vector<float>{q.x(), q.y(), 0.f},
                                                     std::vector<float>{0.f, 0.f, q.z()}, cov, kStamps[i]);
        }
    }
    door = add("door", "door_9", P);
    put_rt(P, door, kDoorInP);
    if (not with_proposal) return;
    DSR::Node n; n.type("metaconcept"); n.name("proposal_room_2");
    G->add_or_modify_attrib_local<object_subtype_att>(n, std::string("room_proposal"));
    G->add_or_modify_attrib_local<state_att>(n, std::string("pending"));
    G->add_or_modify_attrib_local<parent_att>(n, P);
    G->add_or_modify_attrib_local<room_id_att>(n, std::uint64_t{2});
    G->add_or_modify_attrib_local<delimiting_polygon_x_att>(n, std::vector<float>{0.f, 4.f, 4.f, 0.f});
    G->add_or_modify_attrib_local<delimiting_polygon_y_att>(n, std::vector<float>{0.f, 0.f, 3.f, 3.f});
    G->add_or_modify_attrib_local<recall_pose_prior_att>(n, std::vector<float>{1.f, 1.3f, 0.f, -1.27f, 0.01f, 0.f, 0.f, 0.04f, 0.f, 0.001f});
    prop = G->insert_node(n).value();
    put_rt(P, prop, kK_in_P);
}

static std::string state_of(std::uint64_t id)
{
    const auto n = G->get_node(id);
    return n.has_value() ? G->get_attrib_by_name<state_att>(n.value()).value_or(std::string{"?"}) : std::string{"<gone>"};
}

static std::optional<rc::proposal::Proposal> the_proposal()
{
    for (auto& p : rc::proposal::pending(*G)) if (p.node_id == prop) return p;
    return std::nullopt;
}

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");   // the agent runs under Qt's locale
    QCoreApplication app(argc, argv);
    G = std::make_shared<DSR::DSRGraph>("proposal_adoption_test", 197, std::string(ROOT_JSON), true, 57);
    using namespace rc::proposal;
    const Eigen::Affine2f T_P_K = Eigen::Translation2f(kK_in_P.head<2>()) * Eigen::Rotation2Df(kK_in_P.z());
    const Eigen::Affine2f T_K_P = T_P_K.inverse();

    std::printf("pure frame algebra\n");
    {
        const auto m = modes_in({1.f, 1.f, 0.f, 0.f, 0.01f, 0.f, 0.f, 0.04f, 0.f, 0.001f}, T_P_K);
        check(m.size() == 1 and near(m[0].mu.x(), 2.f) and near(m[0].mu.y(), 2.f) and near_angle(m[0].mu.z(), kK_in_P.z()),
              "a robot-in-K mode is composed into P's frame");
        check(not m.empty() and near(m[0].S(0, 0), 0.04f) and near(m[0].S(1, 1), 0.01f), "its covariance is rotated with it");
        check(modes_in(std::vector<float>(9, 1.f), T_P_K).empty(), "a packing that is not 10 per mode yields nothing");
    }

    std::printf("accept_renames_in_place\n");
    {
        reset();
        const auto p = the_proposal();
        check(p.has_value() and p->well_formed and p->live_name == "room_2" and p->parent == P, "the pending proposal is read");
        check(p.has_value() and admit(*G, *p, P) == Admission::Judge, "a proposal on this agent's provisional room is judged");
        const auto a = p.has_value() ? adopt_in_place(*G, *p, robot) : Adoption{};
        check(a.ok, "adopted (" + a.why + ")");
        check(answer(*G, prop, kAccepted) and state_of(prop) == "accepted", "state = accepted");
        const auto n = G->get_node(P);
        check(n.has_value() and n->name() == "room_2", "same id, renamed to K's live name");
        check(not G->get_node("room_4").has_value() and G->get_node("room_2").has_value() and G->get_node("room_2")->id() == P,
              "the old name is free, the new one resolves to P's id");
        check(n.has_value() and G->get_attrib_by_name<room_id_att>(n.value()).value_or(0) == 2, "room_id = K's");
        const auto px = n.has_value() ? G->get_attrib_by_name<delimiting_polygon_x_att>(n.value()) : std::nullopt;
        const auto py = n.has_value() ? G->get_attrib_by_name<delimiting_polygon_y_att>(n.value()) : std::nullopt;
        check(px.has_value() and py.has_value() and px->get() == std::vector<float>{0.f, 4.f, 4.f, 0.f}
              and py->get() == std::vector<float>{0.f, 0.f, 3.f, 3.f}, "delimiting polygon = K's outline");
        check(not G->get_edge(P, P, "proto").has_value(), "the proto self-edge is gone");
        bool ring_ok = true;
        for (int i = 0; i < 3; ++i) ring_ok = ring_ok and same_pose(rt_of(P, robot, kStamps[i]), transform_pose(T_K_P, robot_at(i)));
        check(ring_ok, "EVERY block of the robot's history ring is re-expressed by T(K<-P), at its own timestamp");
        const auto mid = rt_of(P, robot, 1050);   // between two blocks: both in K, so the interpolation is too
        check(mid.has_value() and (mid->head<2>() - transform_pose(T_K_P, robot_at(0)).head<2>()).norm() < 0.2f,
              "a query between two blocks interpolates inside K's frame");
        const auto e = G->get_edge(P, robot, "RT");
        const auto cv = e.has_value() ? G->get_attrib_by_name<rt_covariance_att>(e.value()) : std::nullopt;
        {
            const auto tr = e.has_value() ? G->get_attrib_by_name<rt_translation_att>(e.value()) : std::nullopt;
            const auto st = e.has_value() ? G->get_attrib_by_name<rt_timestamps_att>(e.value()) : std::nullopt;
            int unwritten = 0; bool untouched = true;
            if (tr.has_value() and st.has_value() and st->get().size() * 3 == tr->get().size())
                for (std::size_t b = 0; b < st->get().size(); ++b)
                    if (st->get()[b] == 0)
                    {
                        ++unwritten;
                        untouched = untouched and tr->get()[3 * b] == 0.f and tr->get()[3 * b + 1] == 0.f;
                    }
            check(unwritten > 0 and untouched, "never-written ring slots (stamp 0) are left as they were ("
                  + std::to_string(unwritten) + " slots)");
        }
        check(cv.has_value() and cv->get().size() >= 36 and near(cv->get()[0], 0.01f) and near(cv->get()[7], 0.04f),
              "the pose covariance is rotated with the frame (90 deg: sxx <-> syy)");
        check(same_pose(rt_of(floor_id, wall), transform_pose(T_K_P, kWallInFloor)), "this agent's wall is re-expressed");
        check(same_pose(rt_of(P, floor_id), Eigen::Vector3f::Zero()), "the floor stays the room's identity");
        check(same_pose(rt_of(P, door), kDoorInP), "another agent's child is left to its owner");
        check(a.reexpressed == 2, "exactly two own edges re-expressed (robot, wall)");
    }

    std::printf("reject_marks_only\n");
    {
        reset();
        check(answer(*G, prop, kRejected) and state_of(prop) == "rejected", "state = rejected");
        const auto n = G->get_node(P);
        check(n.has_value() and n->name() == "room_4" and G->get_attrib_by_name<room_id_att>(n.value()).value_or(0) == 4,
              "P keeps its name and room_id");
        check(G->get_edge(P, P, "proto").has_value(), "P is still provisional");
        check(same_pose(rt_of(P, robot), robot_at(2)) and same_pose(rt_of(floor_id, wall), kWallInFloor), "nothing moved");
        check(not the_proposal().has_value(), "a rejected proposal is no longer pending (never answered twice)");
    }

    std::printf("proposal_on_non_proto_rejected\n");
    {
        reset(false);
        const auto p = the_proposal();
        check(p.has_value() and admit(*G, *p, P) == Admission::NotProvisional, "P without a proto edge: not judged");
        reset(true);
        const auto q = the_proposal();
        check(q.has_value() and admit(*G, *q, 0) == Admission::NotProvisional, "this agent has no provisional room: not judged");
        check(q.has_value() and admit(*G, *q, P + 1) == Admission::NotProvisional, "a proposal about another room: not judged");
        reset(false);
        const auto r = the_proposal();
        const auto a = r.has_value() ? adopt_in_place(*G, *r, robot) : Adoption{.ok = true};
        check(not a.ok and G->get_node(P)->name() == "room_4", "adopt_in_place itself refuses a non-provisional P");
    }

    std::printf("rename_collision_rejected\n");
    {
        reset();
        const auto other = add("room", "room_2", G->get_node("root").value().id());
        const auto p = the_proposal();
        check(p.has_value() and admit(*G, *p, P) == Admission::NameTaken, "K's name held by another id: not judged");
        bool threw = false;
        Adoption a;
        try { a = adopt_in_place(*G, *p, robot); } catch (...) { threw = true; }
        check(not threw and not a.ok, "adopt_in_place on a held name returns false, no throw (" + a.why + ")");
        const auto n = G->get_node(P);
        check(n.has_value() and n->name() == "room_4" and G->get_node("room_2")->id() == other, "neither node renamed");
        check(G->get_edge(P, P, "proto").has_value() and same_pose(rt_of(P, robot), robot_at(2)), "nothing else written");
        // and the cortex guard itself, past this module's pre-check: a rename to a held name is refused by update_node
        auto raw = G->get_node(P).value();
        raw.name("room_2");
        bool ok = true, threw2 = false;
        try { ok = G->update_node(raw); } catch (...) { threw2 = true; }
        check(not threw2 and not ok, "cortex update_node: rename onto a held name returns false (no throw)");
    }

    std::printf(failures == 0 ? "all ok (0 failures)\n" : "FAILED (%d failures)\n", failures);
    G.reset();
    return failures == 0 ? 0 : 1;
}
