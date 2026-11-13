// room_swap: executing the recogniser's requests (ltsm). A swap PROPOSES (room_proposal.h); memory is committed on "accepted".   run from the component root:   ./bin/room_swap_test
// Memory: K = r3_room_3 (known, with a floor and a door face), L = r2_room_2 (the room being left, door face r2_door_3).
// Case A: the provisional room room_4 is a PROTO (no memory twin).  Case B: it was PROMOTED: memory holds its twin r4_room_4
// and passage_5 (r2_door_3 <-> r4_door_9), and the previous room is archived (no live L).
#include "../src/room_swap.h"
#include "../src/room_recall.h"
#include <QCoreApplication>
#include <clocale>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <sstream>
#include <string>

static int failures = 0;
static void check(bool ok, const std::string &what) { if (!ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str()); }

static std::shared_ptr<DSR::DSRGraph> live, mem;
static std::uint64_t robot_id = 0, live_L = 0, live_D = 0, live_P = 0, live_M = 0;

static void put_edge(DSR::DSRGraph &G, std::uint64_t from, std::uint64_t to, double x, double y, double th)
{
    auto n = G.get_node(from).value();
    G.get_rt_api()->insert_or_assign_edge_RT(n, to, std::vector<float>{(float)x, (float)y, 0.f}, std::vector<float>{0.f, 0.f, (float)th});
}

static void reset()
{
    for (auto *g : {live.get(), mem.get()})
        for (const char *t : {"room", "floor", "wall", "object", "metaconcept", "robot"})
            for (const auto &n : g->get_nodes_by_type(t)) g->delete_node(n.id());
}

static std::uint64_t mem_node(const char *type, const std::string &name, std::uint64_t parent, double x, double y, double th,
                              std::optional<std::uint64_t> rid = std::nullopt, std::vector<float> poly = {}, const char *subtype = nullptr)
{
    DSR::Node n; n.type(type); n.name(name);
    if (rid) mem->add_or_modify_attrib_local<room_id_att>(n, *rid);
    if (subtype) mem->add_or_modify_attrib_local<object_subtype_att>(n, std::string(subtype));
    if (not poly.empty())
    {
        mem->add_or_modify_attrib_local<delimiting_polygon_x_att>(n, std::vector<float>{poly[0], poly[1], poly[1], poly[0]});
        mem->add_or_modify_attrib_local<delimiting_polygon_y_att>(n, std::vector<float>{poly[2], poly[2], poly[3], poly[3]});
    }
    if (parent != 0) mem->add_or_modify_attrib_local<parent_att>(n, parent);
    const auto id = mem->insert_node(n).value();
    put_edge(*mem, parent != 0 ? parent : mem->get_node("root").value().id(), id, x, y, th);
    return id;
}

static std::uint64_t live_node(const char *type, const std::string &name, std::uint64_t parent, double x, double y, double th,
                               std::optional<std::uint64_t> rid = std::nullopt, const char *subtype = nullptr)
{
    DSR::Node n; n.type(type); n.name(name);
    if (rid) live->add_or_modify_attrib_local<room_id_att>(n, *rid);
    if (subtype) live->add_or_modify_attrib_local<object_subtype_att>(n, std::string(subtype));
    live->add_or_modify_attrib_local<parent_att>(n, parent);
    const auto id = live->insert_node(n).value();
    put_edge(*live, parent, id, x, y, th);
    return id;
}

static void set_current(std::uint64_t room)
{
    for (const auto &e : live->get_edges_by_type("current")) live->delete_edge(e.from(), e.to(), "current");
    live->insert_or_assign_edge(DSR::Edge::create<current_edge_type>(robot_id, room));
}

static void passage(const char *name, std::uint64_t a, std::uint64_t b)
{
    DSR::Node pw; pw.type("metaconcept"); pw.name(name);
    mem->add_or_modify_attrib_local<object_subtype_att>(pw, std::string("passage"));
    const auto pid = mem->insert_node(pw).value();
    mem->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, a));
    mem->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, b));
}

static void build_memory(bool promoted_twin)
{
    const std::vector<float> box{0.f, 4.f, 0.f, 4.f};
    const auto K = mem_node("room", "r3_room_3", 0, 0, 0, 0, 3, box);
    const auto fl = mem_node("floor", "r3_floor", K, 0, 0, 0);
    { auto fn = mem->get_node(fl).value(); mem->add_or_modify_attrib_local<width_m_att>(fn, 7.f); mem->update_node(fn); }
    mem_node("object", "r3_door_5", K, 2, 0, 0.5, std::nullopt, {}, "door");
    const auto L = mem_node("room", "r2_room_2", 0, 0, 0, 0, 2, box);
    const auto fL = mem_node("object", "r2_door_3", L, 4, 2, 0, std::nullopt, {}, "door");
    if (promoted_twin)
    {
        const auto P = mem_node("room", "r4_room_4", 0, 0, 0, 0, 4, box);
        const auto fP = mem_node("object", "r4_door_9", P, 1, 0, 1, std::nullopt, {}, "door");
        passage("passage_5", fL, fP);
        { auto pw = mem->get_node("passage_5").value(); mem->add_or_modify_attrib_local<passage_crossings_att>(pw, 7); mem->update_node(pw); }
    }
}

// live: robot, L (current) with the crossed door, the provisional room (a proto, or promoted+current for case B) with its mirror
static void build_live(bool promoted)
{
    DSR::Node r; r.type("robot"); r.name("Shadow");
    robot_id = live->insert_node(r).value();
    put_edge(*live, live->get_node("root").value().id(), robot_id, 0, 0, 0);
    live_L = live_D = 0;
    if (not promoted)
    {
        live_L = live_node("room", "room_2", robot_id, -3, -1, 0.2, 2);
        live_D = live_node("object", "door_3", live_L, 4, 2, 0, std::nullopt, "door");
        set_current(live_L);
    }
    live_P = live_node("room", "room_4", robot_id, 1, 2, -0.3, 4);
    live_M = live_node("object", "door_9", live_P, 1, 0, 1, std::nullopt, "door");
    if (not promoted) live->insert_or_assign_edge(DSR::Edge::create<proto_edge_type>(live_P, live_P));
    else set_current(live_P);
}

static rc::recog::PoseMix pose_k()
{
    rc::recog::PoseGauss g;
    g.mu = Eigen::Vector3f(1.f, 2.f, 0.3f);
    g.S = Eigen::Vector3f(0.04f, 0.04f, 0.01f).asDiagonal();
    g.w = 1.f;
    return {g};
}

// L's outline on the far side of the crossed door (the fixture's default box sits exactly ON the door's plane, where
// seam_through_passage's pi-orientation side test is a coin flip)
static void far_side_outline_for_L()
{
    auto L = mem->get_node("r2_room_2").value();
    mem->add_or_modify_attrib_local<delimiting_polygon_x_att>(L, std::vector<float>{3.f, 5.f, 5.f, 3.f});
    mem->add_or_modify_attrib_local<delimiting_polygon_y_att>(L, std::vector<float>{4.f, 4.f, 6.f, 6.f});
    mem->update_node(L);
}

// the relocalised pose with the accuracy a real fix has (2 cm, 0.01 rad)
static rc::recog::PoseMix pose_tight()
{
    auto m = pose_k();
    m[0].S = Eigen::Vector3f(0.0004f, 0.0004f, 0.0001f).asDiagonal();
    return m;
}

static std::set<std::string> mem_names()
{
    std::set<std::string> s;
    for (const char *t : {"room", "floor", "wall", "object", "metaconcept"})
        for (const auto &n : mem->get_nodes_by_type(t)) s.insert(n.name());
    return s;
}
static int mem_rooms() { return int(mem->get_nodes_by_type("room").size()); }

static std::set<std::string> snapshot_room(const std::string &room)
{
    std::set<std::string> s;
    for (const char *t : {"room", "floor", "wall", "object"})
        for (const auto &n : mem->get_nodes_by_type(t))
            if (ltsm::is_descendant(*mem, n.id(), room))
            {
                std::ostringstream o;
                o << n.name() << "|" << n.type();
                for (const auto &[k, v] : n.attrs()) o << "|" << k << "=" << v;
                for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(n, "RT"))
                {
                    o << "|RT>" << e.to();
                    for (const auto &[k, v] : e.attrs()) o << "|" << k << "=" << v;
                }
                s.insert(o.str());
            }
    return s;
}

static int recall_nodes()
{
    int n = 0;
    for (const auto &x : live->get_nodes_by_type("metaconcept")) if (x.name().starts_with("recall_")) ++n;
    return n;
}

struct Fix
{
    ltsm::RoomEviction ev;
    ltsm::RoomRecall recall;
    ltsm::RoomSwap sw;
    Fix() : ev(live, mem), recall(live, mem, ev), sw(live, mem, ev) {}
};

static int proposal_nodes()
{
    int n = 0;
    for (const auto &x : live->get_nodes_by_type("metaconcept"))
        if (live->get_attrib_by_name<object_subtype_att>(x).value_or("") == "room_proposal") ++n;
    return n;
}

/// room_concept accepts the proposal (Task 6 writes `state`), and the next cycle reads the verdict: the staging is committed.
static bool accept(Fix &f)
{
    auto n = live->get_node("proposal_room_3");
    if (not n) return false;
    live->add_or_modify_attrib_local<state_att>(n.value(), std::string("accepted"));
    live->update_node(n.value());
    return f.sw.step() == ltsm::RoomSwap::Event::Accepted;
}

/// room_concept's accepted verdict as Task 20 writes it: the accepted T(P <- K) = the prior on the edge, and the fit's covariance.
static void accept_with_fit(Fix &f)
{
    auto n = live->get_node("proposal_room_3");
    if (not n) return;
    const auto par = live->get_attrib_by_name<parent_att>(n.value()).value_or(0);
    const auto e = live->get_edge(par, n->id(), "RT");
    const auto m = e.has_value() ? live->get_rt_api()->get_edge_RT_as_rtmat(e.value()) : std::nullopt;
    if (not m.has_value()) return;
    std::vector<float> cov(36, 0.f);
    cov[0] = 0.04f; cov[7] = 0.04f; cov[35] = 0.01f;
    live->add_or_modify_attrib_local<rt_translation_att>(n.value(), std::vector<float>{(float)m->translation().x(), (float)m->translation().y(), 0.f});
    live->add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(n.value(), std::vector<float>{0.f, 0.f, (float)std::atan2(m->linear()(1, 0), m->linear()(0, 0))});
    live->add_or_modify_attrib_local<rt_covariance_att>(n.value(), std::move(cov));
    live->add_or_modify_attrib_local<state_att>(n.value(), std::string("accepted"));
    live->update_node(n.value());
    f.sw.step();
}

static Mat::RTMat se2(double x, double y, double th)
{
    Mat::RTMat T = Mat::RTMat::Identity();
    T.linear() = Eigen::AngleAxisd(th, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    T.translation() = Eigen::Vector3d(x, y, 0.0);
    return T;
}
static int passage_count()
{
    int n = 0;
    for (const auto &x : mem->get_nodes_by_type("metaconcept"))
        if (mem->get_attrib_by_name<object_subtype_att>(x).value_or("") == "passage") ++n;
    return n;
}
static std::optional<DSR::Node> only_passage()
{
    for (const auto &x : mem->get_nodes_by_type("metaconcept"))
        if (mem->get_attrib_by_name<object_subtype_att>(x).value_or("") == "passage") return x;
    return std::nullopt;
}
static bool any_name_contains(const std::string &s)
{
    for (const char *t : {"room", "floor", "wall", "object", "metaconcept"})
        for (const auto &n : mem->get_nodes_by_type(t)) if (n.name().find(s) != std::string::npos) return true;
    return false;
}

static std::vector<std::string> captured_log;
static void capture(QtMsgType, const QMessageLogContext &, const QString &msg) { captured_log.push_back(msg.toStdString()); std::fputs((msg.toStdString() + "\n").c_str(), stdout); }

int main(int argc, char **argv)
{
    std::setlocale(LC_ALL, "");
    setvbuf(stdout, nullptr, _IONBF, 0);
    QCoreApplication app(argc, argv);
    live = std::make_shared<DSR::DSRGraph>("swap_live", 196, "ltsm_root.json", true, 46);
    mem  = std::make_shared<DSR::DSRGraph>("swap_mem", 197, "ltsm_root.json", true, 47);

    std::printf("swap_proto_leaves_no_trace_in_memory\n");
    {
        reset(); build_memory(false); build_live(false);
        Fix f;
        const auto before = mem_names();
        const int rooms = mem_rooms();
        check(not f.sw.swap(99, pose_k(), live_P) and f.sw.state() == ltsm::RoomSwap::State::Entering, "an unknown room id is refused, nothing changes");
        check(f.sw.swap(3, pose_k(), live_P), "swap accepted");
        check(f.sw.state() == ltsm::RoomSwap::State::Staged, "state Staged");
        check(live->get_node("room_4").has_value() and live->get_node("door_9").has_value(), "rule 1: the live proto stays, with its mirror");
        check(live->get_node("room_2").has_value(), "the room being left is untouched");
        check(recall_nodes() == 0, "no late recall is posted");
        const auto rc = live->get_node("proposal_room_3");
        check(rc.has_value(), "proposal_room_3 posted");
        if (rc)
        {
            const auto pp = live->get_attrib_by_name<recall_pose_prior_att>(rc.value());
            check(pp.has_value() and pp->get().size() == 10 and pp->get()[0] == 1.f and pp->get()[1] == 1.f and pp->get()[2] == 2.f, "recall_pose_prior packs [w,x,y,th,S...]");
            const auto par = live->get_attrib_by_name<parent_att>(rc.value());
            check(par.has_value() and par.value() == live_P, "hangs under the provisional room");
            const auto px = live->get_attrib_by_name<delimiting_polygon_x_att>(rc.value());
            check(px.has_value() and px->get().size() == 4, "carries K's outline");
            const auto T = live->get_inner_eigen_api()->get_transformation_matrix("proposal_room_3", "Shadow");
            check(T.has_value() and std::abs(T->translation().x() - 1.0) < 1e-3 and std::abs(T->translation().y() - 2.0) < 1e-3,
                  "the robot is at the recognised pose in the proposed room's frame");
        }
        check(mem_names() == before and mem_rooms() == rooms, "memory: nothing written (same nodes, same room count)");
        check(f.sw.undo(), "undo accepted");
        check(proposal_nodes() == 0 and recall_nodes() == 0, "undo withdraws the proposal");
        check(mem_names() == before and mem_rooms() == rooms and f.sw.state() == ltsm::RoomSwap::State::Entering, "memory still untouched, state Entering");
        check(not f.sw.undo(), "undo with nothing staged is refused");
    }

    std::printf("swap_promoted_stages_twin_then_drops_it\n");
    {
        reset(); build_memory(true); build_live(true);
        Fix f;
        check(mem_rooms() == 3, "memory: K, L, provisional twin = 3 rooms");
        check(f.sw.swap(3, pose_k(), live_P), "swap accepted");
        const auto twin = mem->get_node("r4_room_4");
        check(twin.has_value() and mem->get_attrib_by_name<object_subtype_att>(twin.value()).value_or("") == "staged", "the provisional twin is staged");
        check(mem_rooms() == 3, "no duplicate room node while staged");
        check(live->get_node("room_4").has_value(), "rule 1: the live provisional room stays");
        check(live->get_node("proposal_room_3").has_value(), "proposal for K posted under it");
        // F3 (TLA ping-pong): while the proposal is pending, the provisional room's OWN door still leads where memory says —
        // its face in the staged twin identifies the hole, the room behind it (archived L) is not staged. recog_proto15 /
        // the model: the recall was retired here and a crossing back in this window birthed a NEW room.
        f.recall.step();
        {
            const auto rc2 = live->get_node("recall_room_2");
            check(rc2.has_value(), "F3: recall_room_2 is posted under the provisional room's door while its twin is staged");
            if (rc2.has_value())
                check(live->get_attrib_by_name<parent_att>(rc2.value()).value_or(0) == live_M, "F3: it hangs under door_9");
        }
        // an archived-room passage into the staged twin must not feed the recall of a live door
        const auto L2 = live_node("room", "room_2", robot_id, -3, -1, 0.2, 2);
        live_node("object", "door_3", L2, 4, 2, 0, std::nullopt, "door");
        f.recall.step();
        check(not live->get_node("recall_room_4").has_value(), "RoomRecall::step skips the staged twin");
        const auto Lr = live->get_node("room_2").value();
        live->delete_node(Lr.id());
        check(accept(f), "room_concept accepts");
        f.sw.on_room_left("room_3");
        check(not mem->get_node("r4_room_4").has_value() and not mem->get_node("r4_door_9").has_value(), "accepted: the staged twin and its faces are gone");
        {
            const auto pw = mem->get_node("passage_5");
            check(pw.has_value(), "the L<->twin passage survives");
            if (pw)
            {
                const auto m = DSR::DSRGraph::get_node_edges_by_type(pw.value(), "match");
                const auto fL = mem->get_node("r2_door_3");
                check(m.size() == 2 and fL.has_value() and std::ranges::any_of(m, [&](const auto &e) { return e.to() == fL->id(); }), "with L's face and (carry-over) K's new face, the twin's gone");
                check(mem->get_attrib_by_name<passage_crossings_att>(pw.value()).value_or(0) == 7, "and its crossing history intact");
            }
            check(mem->get_node("r2_door_3").has_value() and mem->get_node("r2_room_2").has_value(), "L's archive door is intact");
            for (const char *t : {"room", "floor", "wall", "object"})
                for (const auto &n : mem->get_nodes_by_type(t))
                    if (n.name().starts_with("r4_")) check(false, "no node of the twin remains: " + n.name());
        }
        check(mem_rooms() == 2, "memory: exactly the known rooms (K, L)");
        check(proposal_nodes() == 0 and f.sw.state() == ltsm::RoomSwap::State::Entering, "left: proposal withdrawn, state Entering");
        check(not f.ev.held("r3_room_3"), "the archive is released");
    }

    std::printf("withdrawn_crossing_removes_the_false_room\n");
    {   // clean16: L is the room P is adopted as. No new room stays in memory or in the graph, L intact, the false passage gone.
        reset(); build_memory(true); build_live(true);
        Fix f;
        const auto accept_L = [&] {
            auto n = live->get_node("proposal_room_2");
            if (not n) return false;
            const auto par = live->get_attrib_by_name<parent_att>(n.value()).value_or(0);
            const auto e = live->get_edge(par, n->id(), "RT");
            const auto m = e.has_value() ? live->get_rt_api()->get_edge_RT_as_rtmat(e.value()) : std::nullopt;
            if (not m.has_value()) return false;
            std::vector<float> cov(36, 0.f); cov[0] = 0.04f; cov[7] = 0.04f; cov[35] = 0.01f;
            live->add_or_modify_attrib_local<rt_translation_att>(n.value(), std::vector<float>{(float)m->translation().x(), (float)m->translation().y(), 0.f});
            live->add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(n.value(), std::vector<float>{0.f, 0.f, (float)std::atan2(m->linear()(1, 0), m->linear()(0, 0))});
            live->add_or_modify_attrib_local<rt_covariance_att>(n.value(), std::move(cov));
            live->add_or_modify_attrib_local<state_att>(n.value(), std::string("accepted"));
            live->update_node(n.value());
            return f.sw.step() == ltsm::RoomSwap::Event::Accepted; };
        check(f.sw.swap(2, pose_k(), live_P, true), "swap to L posted (the withdrawal is a proposal like any swap)");
        check(live->get_node("proposal_room_2").has_value(), "the proposal for L hangs under P");
        check(accept_L(), "room_concept accepts (its fit is the verdict)");
        check(mem_rooms() == 2, "no new room stays in memory (K and L only)");
        check(not mem->get_node("r4_room_4").has_value() and not mem->get_node("r4_door_9").has_value(), "P's twin and its face are gone");
        check(not mem->get_node("passage_5").has_value(), "the false passage L<->P is gone");
        check(not mem->get_node("r2_door_3").has_value(), "L's face of it is gone: no door in L stays behind the false crossing");
        check(mem->get_node("r2_room_2").has_value(), "L is intact, not archived away");
        check(not any_name_contains("r2_door_x") and passage_count() == 0, "nothing was carried: no passage written");
        check(f.sw.state() == ltsm::RoomSwap::State::Adopted and f.ev.held("r2_room_2"), "L is held until the robot leaves it, as any adopted room");
        f.sw.on_room_left("room_2");
        check(f.sw.state() == ltsm::RoomSwap::State::Entering and not f.ev.held("r2_room_2"), "left: window closed");
    }
    {   // a passage holding a face of a THIRD room is a real hole memory knew: only P's face of it goes
        reset(); build_memory(true); build_live(true);
        Fix f;
        // passage_5 joins L and P's twin; a known room K's face of the same hole makes it a real, known passage
        const auto k = mem->get_node("r3_room_3").value();
        const auto fK = mem_node("object", "r3_door_6", k.id(), 3, 0, 0, std::nullopt, {}, "door");
        mem->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(mem->get_node("passage_5")->id(), fK));
        check(f.sw.swap(2, pose_k(), live_P, true), "withdrawal of a known door");
        auto n = live->get_node("proposal_room_2").value();
        std::vector<float> cov(36, 0.f); cov[0] = 0.04f; cov[7] = 0.04f; cov[35] = 0.01f;
        live->add_or_modify_attrib_local<rt_translation_att>(n, std::vector<float>{0.f, 0.f, 0.f});
        live->add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(n, std::vector<float>{0.f, 0.f, 0.f});
        live->add_or_modify_attrib_local<rt_covariance_att>(n, std::move(cov));
        live->add_or_modify_attrib_local<state_att>(n, std::string("accepted"));
        live->update_node(n);
        check(f.sw.step() == ltsm::RoomSwap::Event::Accepted, "accepted");
        check(mem->get_node("passage_5").has_value() and mem->get_node("r2_door_3").has_value(), "a passage to a third room is a real hole: it and L's face stay");
        check(not mem->get_node("r4_door_9").has_value(), "P's face goes (with the twin)");
    }

    std::printf("wrong_commit_undone_without_corrupting_known_room\n");
    {
        reset(); build_memory(true); build_live(true);
        Fix f;
        const auto snapK = snapshot_room("r3_room_3");
        const auto snapP = snapshot_room("r4_room_4");
        const auto before = mem_names();
        check(f.sw.swap(3, pose_k(), live_P), "swap accepted");
        f.recall.step();                                   // evidence for `new` arrives: nothing may touch K
        check(f.sw.undo(), "undo accepted");
        check(snapshot_room("r3_room_3") == snapK, "the known room's subtree (nodes, attrs, RT edges) is bit-identical");
        check(snapshot_room("r4_room_4") == snapP, "the provisional twin is restored exactly (subtype, door face)");
        check(mem_names() == before, "no node added or lost in memory");
        // the provisional room itself is never recalled (it never left the live graph); the recall under ITS door to the archived
        // room L behind it is F3's, and is right (TLA ping-pong: retiring it here birthed a new room on the way back)
        check(live->get_node("room_4").has_value() and not live->get_node("recall_room_4").has_value(),
              "the provisional room never left the live graph: nothing to re-post");
        check(live->get_node("recall_room_2").has_value(), "F3: the way back to the archived room stays recalled through the staging");
        check(proposal_nodes() == 0, "K's proposal is gone");
        check(not f.ev.held("r3_room_3"), "the archive is released");
    }

    std::printf("known_room_archive_read_only_while_staged\n");
    {
        reset(); build_memory(false); build_live(false);
        Fix f;
        check(f.sw.swap(3, pose_k(), live_P) and accept(f), "swap accepted, proposal accepted");
        // the adopted room live under K's name (room_concept renames P in place, Task 6; stood in for here by a node named
        // room_3); the robot then leaves it into a new proto
        const auto K = live_node("room", "room_3", robot_id, 0, 0, 0, 3);
        live_node("object", "door_extra", K, 1, 1, 0);   // a door: ltsm archives structure only (furniture never)
        const auto N = live_node("room", "room_6", robot_id, 2, 2, 0, 6);
        ltsm::Candidate c;
        c.robot_id = robot_id; c.robot_name = "Shadow";
        c.old_room_id = K; c.old_room_name = "room_3"; c.new_room_id = N; c.new_room_name = "room_6";
        c.remove_old = false;
        const auto snapK = snapshot_room("r3_room_3");
        const auto out = f.ev.evict(c);
        check(not out.ok and out.reason.starts_with("deferred"), "evict of the recalled room is deferred while Staged");
        check(snapshot_room("r3_room_3") == snapK and not mem->get_node("r3_door_extra").has_value(), "nothing was written into the known twin");
        f.sw.on_evict(c);
        check(f.sw.state() == ltsm::RoomSwap::State::Adopted and not f.ev.held("r3_room_3"), "on_evict only releases the hold: still Adopted");
        f.sw.on_evicted(c, false);
        check(f.ev.held("r3_room_3") and f.sw.state() == ltsm::RoomSwap::State::Adopted, "a failed eviction restores the hold");
        f.sw.on_evict(c);
        const auto out2 = f.ev.evict(c);
        check(out2.ok and mem->get_node("r3_door_extra").has_value(), "after the release the eviction writes as usual");
        f.sw.on_evicted(c, out2.ok);
        check(f.sw.state() == ltsm::RoomSwap::State::Entering and proposal_nodes() == 0, "a successful eviction closes the window and withdraws the proposal");
    }

    std::printf("entering_the_staged_room_flips_current_without_writing_into_it\n");
    {
        reset(); build_memory(false); build_live(false);
        Fix f;
        check(f.sw.swap(3, pose_k(), live_P) and accept(f), "swap accepted, proposal accepted");
        const auto K = live_node("room", "room_3", robot_id, 0, 0, 0, 3);       // the adopted room (renamed P stand-in)
        live_node("object", "door_extra", K, 1, 1, 0);   // a door: ltsm archives structure only (furniture never)
        ltsm::Candidate c;      // promotion of the adopted K: old = L, new = K
        c.robot_id = robot_id; c.robot_name = "Shadow";
        c.old_room_id = live_L; c.old_room_name = "room_2"; c.new_room_id = K; c.new_room_name = "room_3";
        c.remove_old = false;
        const auto snapK = snapshot_room("r3_room_3");
        const auto out = f.ev.evict(c);
        check(out.ok and out.current_flipped, "the promotion into the recalled room goes through (current flips)");
        check(snapshot_room("r3_room_3") == snapK and not mem->get_node("r3_door_extra").has_value(), "K's twin is not copied into while staged");
        f.sw.on_room_left();
    }

    std::printf("turn_around_in_doorway\n");
    {
        reset(); build_memory(false); build_live(false);
        Fix f;
        check(f.sw.swap(3, pose_k(), live_P), "swap");
        check(f.sw.undo(), "undo");
        check(live->get_node(live_P).has_value(), "the proto never left: nothing to re-birth");
        check(f.sw.swap(3, pose_k(), live_P), "swap again");
        check(proposal_nodes() == 1 and recall_nodes() == 0, "exactly one proposal node, no recall_*");
        check(f.sw.undo() and proposal_nodes() == 0, "and undo leaves none");
    }

    std::printf("turn_around_promoted_case_leaves_no_orphan_proposal\n");
    {
        reset(); build_memory(true); build_live(true);
        Fix f;
        const auto snapP = snapshot_room("r4_room_4");
        check(f.sw.swap(3, pose_k(), live_P), "swap");
        check(proposal_nodes() == 1 and recall_nodes() == 0, "staged: exactly one proposal, no recall");
        check(f.sw.undo(), "undo");
        check(proposal_nodes() == 0 and recall_nodes() == 0 and snapshot_room("r4_room_4") == snapP, "nothing live remains, the twin is restored exactly");
        check(f.sw.swap(3, pose_k(), live_P), "swap again");
        check(proposal_nodes() == 1, "staged again: exactly one proposal");
        check(accept(f), "accepted");
        f.sw.on_room_left("room_3");
        check(proposal_nodes() == 0 and recall_nodes() == 0 and not mem->get_node("r4_room_4").has_value(), "after leaving: no proposal remains, the twin is gone");
    }

    std::printf("proposal_kept_while_pending_and_adopted_until_left\n");
    {
        reset(); build_memory(true); build_live(true);
        Fix h;
        check(h.sw.swap(3, pose_k(), live_P), "swap");
        check(h.sw.step() == ltsm::RoomSwap::Event::None and live->get_node("proposal_room_3").has_value(), "pending: the proposal stands");
        check(accept(h) and live->get_node("proposal_room_3").has_value(), "accepted: it stands (door_concept reads it)");
        h.sw.on_room_left("room_3");
        check(not live->get_node("proposal_room_3").has_value(), "withdrawn when the window closes");
    }

    std::printf("door2_becomes_passage_of_both_rooms\n");
    {
        reset(); build_memory(false); build_live(false);
        Fix f;
        check(passage_count() == 0, "(no passage yet: case A)");
        check(f.sw.swap(3, pose_k(), live_P) and accept(f), "swap accepted, proposal accepted");
        f.sw.on_room_left("room_3");
        check(passage_count() == 1, "exactly one passage");
        const auto pw = only_passage();
        if (pw)
        {
            const auto m = DSR::DSRGraph::get_node_edges_by_type(pw.value(), "match");
            check(m.size() == 2, "two match edges");
            bool in_r2 = false, in_r3 = false;
            for (const auto &e : m)
            {
                in_r2 = in_r2 or ltsm::is_descendant(*mem, e.to(), "r2_room_2");
                in_r3 = in_r3 or ltsm::is_descendant(*mem, e.to(), "r3_room_3");
            }
            check(in_r2 and in_r3, "one face in R2, one in R3");
            const auto S = ltsm::seam_through_passage(*mem, pw->name(), "r3_room_3", "r2_room_2");
            check(S.has_value(), "seam_through_passage(R3, R2) resolves");
            if (S)
            {
                const Mat::RTMat truth = se2(1.0, 2.0, 0.3) * se2(-3.0, -1.0, 0.2);       // T(K <- robot) T(robot <- L)
                const double err = (S->translation() - truth.translation()).head<2>().norm();
                const double dth = std::abs(std::remainder(std::atan2(S->linear()(1, 0), S->linear()(0, 0))
                                                           - std::atan2(truth.linear()(1, 0), truth.linear()(0, 0)), 2 * M_PI));
                check(err < 0.05 and dth < 0.05, "the seam is within 5 cm of truth (" + std::to_string(err) + " m, " + std::to_string(dth) + " rad)");
            }
        }
    }

    std::printf("door2_case_b_rehangs_the_surviving_passage\n");
    {
        reset(); build_memory(true); build_live(true);
        Fix f;
        check(f.sw.swap(3, pose_k(), live_P) and accept(f), "swap accepted, proposal accepted");
        f.sw.on_room_left("room_3");
        check(passage_count() == 1, "still one passage (re-hung, not duplicated)");
        const auto pw = mem->get_node("passage_5");
        check(pw.has_value(), "it is the original passage_5");
        if (pw)
        {
            check(mem->get_attrib_by_name<passage_crossings_att>(pw.value()).value_or(0) == 7, "crossing history intact");
            const auto m = DSR::DSRGraph::get_node_edges_by_type(pw.value(), "match");
            bool in_r2 = false, in_r3 = false;
            for (const auto &e : m)
            {
                in_r2 = in_r2 or ltsm::is_descendant(*mem, e.to(), "r2_room_2");
                in_r3 = in_r3 or ltsm::is_descendant(*mem, e.to(), "r3_room_3");
            }
            check(m.size() == 2 and in_r2 and in_r3, "faces in R2 and R3 (the twin's is gone)");
            const auto S = ltsm::seam_through_passage(*mem, "passage_5", "r3_room_3", "r2_room_2");
            check(S.has_value(), "seam_through_passage(R3, R2) resolves");
        }
    }

    std::printf("no_face_in_provisional_name\n");
    {
        for (const bool promoted : {false, true})
        {
            reset(); build_memory(promoted); build_live(promoted);
            Fix f;
            f.sw.swap(3, pose_k(), live_P);
            accept(f);
            f.sw.on_room_left("room_3");
            check(not any_name_contains("room_4") and not any_name_contains("r4_"),
                  std::string("no memory node carries the provisional room's name (") + (promoted ? "B" : "A") + ")");
        }
    }

    std::printf("carry_over_idempotent\n");
    {
        reset(); build_memory(false); build_live(false);
        Fix f;
        f.sw.swap(3, pose_k(), live_P);
        accept(f);
        f.sw.on_room_left("room_3");
        const auto before = mem_names();
        const int np = passage_count();
        check(f.sw.carry_over("r2_door_3", 3), "a second call reports success");
        check(mem_names() == before and passage_count() == np, "and changes nothing");
    }

    std::printf("second_crossing_recalls_directly\n");
    {
        reset(); build_memory(false); build_live(false);
        Fix f;
        f.sw.swap(3, pose_k(), live_P);
        accept(f);
        f.sw.on_room_left("room_3");
        live->delete_node(live->get_node("room_2").value().id());
        const auto L2 = live_node("room", "room_2", robot_id, -3, -1, 0.2, 2);
        live_node("object", "door_3", L2, 4, 2, 0, std::nullopt, "door");
        f.recall.step();
        const auto rc = live->get_node("recall_room_3");
        check(rc.has_value(), "recall_room_3 posted under the door");
        if (rc) check(not live->get_attrib_by_name<recall_late_att>(rc.value()).value_or(false), "and it is an ordinary one (no late flag)");
    }

    // ── association with K's own unmatched face of the crossed door (the real case) ──
    const auto seam_err = [&](const Mat::RTMat &truth) {
        const auto S = ltsm::seam_through_passage(*mem, only_passage() ? only_passage()->name() : "", "r3_room_3", "r2_room_2");
        if (not S) return std::make_pair(1e9, 1e9);
        const double e = (S->translation() - truth.translation()).head<2>().norm();
        const double d = std::abs(std::remainder(std::atan2(S->linear()(1, 0), S->linear()(0, 0)) - std::atan2(truth.linear()(1, 0), truth.linear()(0, 0)), 2 * M_PI));
        return std::make_pair(e, d);
    };
    const Mat::RTMat truthA = se2(1.0, 2.0, 0.3) * se2(-3.0, -1.0, 0.2);
    const Mat::RTMat face_truth = truthA * se2(4.0, 2.0, 0.0);

    std::printf("adopts_k_face_off_by_6cm_and_fuses\n");
    {
        reset(); build_memory(false); build_live(false); far_side_outline_for_L();
        const auto K = mem->get_node("r3_room_3").value().id();
        const double ex = face_truth.translation().x() + 0.04, ey = face_truth.translation().y() + 0.04;
        mem_node("object", "r3_door_old", K, ex, ey, std::atan2(face_truth.linear()(1, 0), face_truth.linear()(0, 0)) + 0.01, std::nullopt, {}, "door");
        Fix f;
        f.sw.swap(3, pose_tight(), live_P);
        accept(f);
        f.sw.on_room_left("room_3");
        check(not mem->get_node("r3_door_3").has_value() and passage_count() == 1, "no new face: the existing one is adopted");
        const auto d = mem->get_node("r3_door_old");
        const auto T = mem->get_inner_eigen_api()->get_transformation_matrix("r3_room_3", "r3_door_old");
        check(d.has_value() and T.has_value(), "adopted door resolves in K");
        if (T)
        {
            const double raw = std::hypot(0.04, 0.04);
            const double now = (T->translation() - face_truth.translation()).head<2>().norm();
            check(now < raw and now < 0.05, "its pose was PRECISION-FUSED toward the chain (" + std::to_string(now) + " m, was " + std::to_string(raw) + ")");
        }
        const auto [e, dth] = seam_err(truthA);
        check(e < 0.05 and dth < 0.05, "seam within 5 cm / 0.05 rad (" + std::to_string(e) + ", " + std::to_string(dth) + ")");
    }

    std::printf("adopts_the_same_face_turned_by_pi\n");
    {
        reset(); build_memory(false); build_live(false); far_side_outline_for_L();
        const auto K = mem->get_node("r3_room_3").value().id();
        mem_node("object", "r3_door_old", K, face_truth.translation().x() + 0.04, face_truth.translation().y() - 0.04,
                 std::atan2(face_truth.linear()(1, 0), face_truth.linear()(0, 0)) + M_PI, std::nullopt, {}, "door");
        Fix f;
        f.sw.swap(3, pose_tight(), live_P);
        accept(f);
        f.sw.on_room_left("room_3");
        check(not mem->get_node("r3_door_3").has_value(), "adopted although turned by pi");
        const auto T = mem->get_inner_eigen_api()->get_transformation_matrix("r3_room_3", "r3_door_old");
        if (T)
        {
            const double dy = std::abs(std::remainder(std::atan2(T->linear()(1, 0), T->linear()(0, 0)) - std::atan2(face_truth.linear()(1, 0), face_truth.linear()(0, 0)), 2 * M_PI));
            check(dy < 0.05, "orientation brought onto the consistent branch (" + std::to_string(dy) + " rad off)");
        }
        const auto [e, dth] = seam_err(truthA);
        check(e < 0.05 and dth < 0.05, "seam within 5 cm / 0.05 rad (" + std::to_string(e) + ", " + std::to_string(dth) + ")");
    }

    std::printf("distractor_door_2m_away_is_not_adopted\n");
    {
        reset(); build_memory(false); build_live(false); far_side_outline_for_L();
        const auto K = mem->get_node("r3_room_3").value().id();
        mem_node("object", "r3_door_far", K, face_truth.translation().x() + 2.0, face_truth.translation().y(), 0.0, std::nullopt, {}, "door");
        Fix f;
        f.sw.swap(3, pose_tight(), live_P);
        accept(f);
        f.sw.on_room_left("room_3");
        check(mem->get_node("r3_door_3").has_value(), "a new face was written");
        const auto pw = only_passage();
        bool far_owned = false;
        if (pw) for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(pw.value(), "match")) far_owned = far_owned or (mem->get_node(e.to()) and mem->get_node(e.to())->name() == "r3_door_far");
        check(not far_owned, "the distractor stays unowned");
        const auto [e, dth] = seam_err(truthA);
        check(e < 0.05 and dth < 0.05, "seam within 5 cm / 0.05 rad");
    }

    std::printf("live_door_oriented_from_the_other_room\n");
    {
        reset(); build_memory(false); build_live(false); far_side_outline_for_L();
        put_edge(*live, live_L, live_D, 4, 2, M_PI);
        put_edge(*mem, mem->get_node("r2_room_2").value().id(), mem->get_node("r2_door_3").value().id(), 4, 2, M_PI);
        Fix f;
        f.sw.swap(3, pose_tight(), live_P);
        accept(f);
        f.sw.on_room_left("room_3");
        const auto [e, dth] = seam_err(truthA);
        check(e < 0.05 and dth < 0.05, "theta = pi door: seam within 5 cm / 0.05 rad (" + std::to_string(e) + ", " + std::to_string(dth) + ")");
    }

    std::printf("case_b_seam_matches_truth\n");
    {
        reset(); build_memory(true); build_live(true);
        Fix f;
        f.sw.swap(3, pose_tight(), live_P);
        accept(f);
        f.sw.on_room_left("room_3");
        // truth: T(K<-robot) T(robot<-P) T(P<-L), T(P<-L) from the L<->twin passage faces
        const Mat::RTMat truthB = se2(1.0, 2.0, 0.3) * se2(1.0, 2.0, -0.3) * (se2(1.0, 0.0, 1.0) * se2(4.0, 2.0, 0.0).inverse());
        const auto S = ltsm::seam_through_passage(*mem, "passage_5", "r3_room_3", "r2_room_2");
        check(S.has_value(), "resolves");
        if (S)
        {
            const double e = (S->translation() - truthB.translation()).head<2>().norm();
            const double d = std::abs(std::remainder(std::atan2(S->linear()(1, 0), S->linear()(0, 0)) - std::atan2(truthB.linear()(1, 0), truthB.linear()(0, 0)), 2 * M_PI));
            check(e < 0.05 and d < 0.05, "within 5 cm / 0.05 rad of truth (" + std::to_string(e) + ", " + std::to_string(d) + ")");
        }
    }

    std::printf("loose_robot_pose_trusts_the_stored_door\n");
    {
        reset(); build_memory(false); build_live(false); far_side_outline_for_L();
        const auto K = mem->get_node("r3_room_3").value().id();
        mem_node("object", "r3_door_old", K, face_truth.translation().x() + 0.03, face_truth.translation().y() - 0.03,
                 std::atan2(face_truth.linear()(1, 0), face_truth.linear()(0, 0)), std::nullopt, {}, "door");
        auto loose = pose_k();
        loose[0].S = Eigen::Vector3f(0.04f, 0.04f, 0.01f).asDiagonal();      // 0.2 m / 0.1 rad
        Fix f;
        f.sw.swap(3, loose, live_P);
        accept(f);
        f.sw.on_room_left("room_3");
        check(not mem->get_node("r3_door_3").has_value(), "adopted");
        const auto T = mem->get_inner_eigen_api()->get_transformation_matrix("r3_room_3", "r3_door_old");
        const double now = T ? (T->translation() - face_truth.translation()).head<2>().norm() : 1e9;
        check(now < 0.05, "the accurate stored door stays within 5 cm of truth (" + std::to_string(now) + " m)");
    }

    std::printf("tight_chain_wrong_stored_door_follows_the_chain\n");
    {
        reset(); build_memory(false); build_live(false); far_side_outline_for_L();
        const auto K = mem->get_node("r3_room_3").value().id();
        mem_node("object", "r3_door_old", K, face_truth.translation().x() + 0.25, face_truth.translation().y(),
                 std::atan2(face_truth.linear()(1, 0), face_truth.linear()(0, 0)), std::nullopt, {}, "door");
        Fix f;
        f.sw.swap(3, pose_tight(), live_P);
        accept(f);
        f.sw.on_room_left("room_3");
        const auto T = mem->get_inner_eigen_api()->get_transformation_matrix("r3_room_3", "r3_door_old");
        const double now = T ? (T->translation() - face_truth.translation()).head<2>().norm() : 1e9;
        check(not mem->get_node("r3_door_3").has_value() and now < 0.6 * 0.25, "adopted and pulled to the chain (" + std::to_string(now) + " m of 0.25)");
    }

    std::printf("mouth_pose_of_the_provisional_room_is_the_seam\n");
    {
        reset(); build_memory(false); build_live(false); far_side_outline_for_L();
        // L_from_P consistent with the chain, shifted 0.3 m along L's x: the seam must follow the mouth
        const Mat::RTMat L_from_P = (truthA.inverse() * se2(1.0, 2.0, 0.3) * se2(1.0, 2.0, -0.3));
        auto P = live->get_node(live_P).value();
        const double x = L_from_P.translation().x() + 0.3, y = L_from_P.translation().y(), th = std::atan2(L_from_P.linear()(1, 0), L_from_P.linear()(0, 0));
        live->add_or_modify_attrib_local<rt_translation_att>(P, std::vector<float>{(float)x, (float)y, 0.f});
        live->add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(P, std::vector<float>{0.f, 0.f, (float)th});
        live->add_or_modify_attrib_local<interaction_node_id_att>(P, (std::uint64_t)live_L);
        std::vector<float> cov36(36, 0.f);
        cov36[0] = cov36[7] = 0.0004f; cov36[35] = 0.0001f;
        live->add_or_modify_attrib_local<rt_covariance_att>(P, cov36);
        live->update_node(P);
        Fix f;
        f.sw.swap(3, pose_tight(), live_P);
        accept(f);
        f.sw.on_room_left("room_3");
        const auto [e, dth] = seam_err(truthA);
        // K_from_L = K_from_P * mouth^-1 with the mouth moved by 0.3 m along L's x => the seam moves by ~0.3 m
        check(e > 0.2 and e < 0.4 and dth < 0.05, "the seam follows the published mouth (off the chain by " + std::to_string(e) + " m)");
    }

    std::printf("stale_staged_swept_at_start\n");
    {
        reset(); build_memory(true); build_live(true);
        // a crashed run: the twin is staged, a late recall is live
        const auto twin = mem->get_node("r4_room_4").value();
        auto t2 = twin;
        mem->add_or_modify_attrib_local<object_subtype_att>(t2, std::string("staged"));
        mem->update_node(t2);
        Fix f;
        f.recall.post_late(robot_id, "r3_room_3", Mat::RTMat::Identity(), {});     // an older build's late recall
        check(recall_nodes() == 1 and f.recall.late_count() == 1, "(a late recall is live)");
        ltsm::RoomProposal(live, mem).post(live_P, "r3_room_3", Mat::RTMat::Identity(), {});
        check(proposal_nodes() == 1, "(a proposal is live)");
        const int n = f.sw.sweep_stale();
        check(n >= 3 and proposal_nodes() == 0, "sweep reports what it removed (" + std::to_string(n) + "), the proposal too");
        check(not mem->get_node("r4_room_4").has_value() and not mem->get_node("r4_door_9").has_value(), "the staged twin is removed from memory");
        check(recall_nodes() == 0 and mem_rooms() == 2, "the late recall is removed; K and L remain");
        check(f.sw.sweep_stale() == 0, "a second sweep finds nothing");
    }

    // recog_try14 (P2): a FRESH->FRESH crossing leaves the room being left LIVE-ONLY (it is archived when the robot leaves the
    // recognised room), so at swap time its door face is not in memory. The '[swap] crossed door X <-> K face Y' line the tour
    // driver reads must still be printed -- from the live face -- and nothing may be written.
    std::printf("swap_time_log_with_live_only_previous_room\n");
    {
        reset(); build_memory(false); build_live(false);
        for (const char *nm : {"r2_door_3", "r2_room_2"}) if (const auto n = mem->get_node(nm); n.has_value()) mem->delete_node(n->id());
        check(not mem->get_node("r2_door_3").has_value() and live->get_node("door_3").has_value(), "(L is live-only: no face of it in memory)");
        Fix f;
        const auto before = mem_names();
        captured_log.clear();
        qInstallMessageHandler(capture);
        const bool ok = f.sw.swap(3, pose_k(), live_P);
        qInstallMessageHandler(nullptr);
        check(ok, "swap accepted");
        const bool printed = std::ranges::any_of(captured_log, [](const std::string &l) { return l.starts_with("[swap] crossed door r2_door_3 <-> K face "); });
        check(printed, "the swap-time association line is printed from the live face");
        check(passage_count() == 0 and not any_name_contains("r2_door_3"), "and nothing was written to memory");
        (void) before;
    }

    std::printf("swap_through_a_known_door_keeps_the_passage\n");
    {   // (c) L's crossed door is ALREADY a passage to K: the carry-over must recognise it (one passage, its two faces, no new face)
        reset(); build_memory(false); build_live(false);
        passage("passage_7", mem->get_node("r2_door_3")->id(), mem->get_node("r3_door_5")->id());
        Fix f;
        check(f.sw.swap(3, pose_k(), live_P), "swap accepted");
        accept_with_fit(f);
        check(f.sw.state() == ltsm::RoomSwap::State::Adopted, "accepted: Adopted");
        const auto m = DSR::DSRGraph::get_node_edges_by_type(only_passage().value(), "match");
        const auto has = [&](const char *n) { const auto d = mem->get_node(n); return d.has_value() and std::ranges::any_of(m, [&](const auto &e) { return e.to() == d->id(); }); };
        check(passage_count() == 1 and m.size() == 2 and has("r2_door_3") and has("r3_door_5"), "one passage with its original two faces");
        check(not any_name_contains("r3_door_3"), "no new face written for the known hole");
        f.sw.on_room_left("room_3");
        check(passage_count() == 1 and DSR::DSRGraph::get_node_edges_by_type(only_passage().value(), "match").size() == 2, "leaving K: still one passage, two faces");
    }
    {   // (c) a recognised room that is NOT the passage's far side: the known passage is not rewritten, and no third face is written
        reset(); build_memory(false); build_live(false);
        const auto r9 = mem_node("room", "r9_room_9", 0, 0, 0, 0, 9, {0.f, 4.f, 0.f, 4.f});
        const auto f9 = mem_node("object", "r9_door_1", r9, 1, 0, 0, std::nullopt, {}, "door");
        passage("passage_7", mem->get_node("r2_door_3")->id(), f9);
        Fix f;
        check(f.sw.swap(3, pose_k(), live_P), "swap accepted");
        accept_with_fit(f);
        const auto m = DSR::DSRGraph::get_node_edges_by_type(only_passage().value(), "match");
        const auto has = [&](const char *n) { const auto d = mem->get_node(n); return d.has_value() and std::ranges::any_of(m, [&](const auto &e) { return e.to() == d->id(); }); };
        check(passage_count() == 1 and m.size() == 2 and has("r2_door_3") and has("r9_door_1"), "the passage keeps its own two rooms (r2, r9), no face of K added");
        check(not any_name_contains("r3_door_3"), "no new face of K for that hole");
        f.sw.on_room_left("room_3");
    }

    std::printf("recall_goes_to_the_door_nearest_its_face\n");
    {   // (door_13, cmp_br2) two live doors at one face: one RT edge, one parent, the nearer door; and it follows the door when the other leaves
        reset(); build_memory(true); build_live(true);
        Fix f;
        const auto near_id = live_node("object", "door_5", live_P, 1.05, 0.0, 1, std::nullopt, "door");   // door_5: the restored door, nearest the face r4_door_9
        f.recall.step();
        auto rc2 = live->get_node("recall_room_2");
        check(rc2.has_value() and live->get_attrib_by_name<parent_att>(rc2.value()).value_or(0) == live_M, "door_9, exactly at the face, wins the recall over the farther door_5");
        check(not live->get_edge(near_id, rc2->id(), "RT").has_value(), "and door_5 has no RT edge to it");
        // door_9 (the mirror) moves away (door_concept re-seats it elsewhere): door_5 is now the only door at the face
        put_edge(*live, live_P, live_M, 3.0, 3.0, 1.0);
        f.recall.step();
        rc2 = live->get_node("recall_room_2");
        check(rc2.has_value() and live->get_attrib_by_name<parent_att>(rc2.value()).value_or(0) == near_id, "the recall MOVES to door_5");
        check(rc2.has_value() and live->get_edge(near_id, rc2->id(), "RT").has_value(), "with its RT edge from door_5");
        check(rc2.has_value() and not live->get_edge(live_M, rc2->id(), "RT").has_value(), "and the stale RT edge from door_9 is gone");
    }

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "all ok", failures);
    std::fflush(stdout);
    std::_Exit(failures ? 1 : 0);
}
