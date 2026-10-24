// room_swap: executing the recogniser's requests (ltsm).   run from the component root:   ./bin/room_swap_test
// Memory: K = r3_room_3 (known, with a floor and a door face), L = r2_room_2 (the room being left, door face r2_door_3).
// Case A: the provisional room room_4 is a PROTO (no memory twin).  Case B: it was PROMOTED: memory holds its twin r4_room_4
// and passage_5 (r2_door_3 <-> r4_door_9), and the previous room is archived (no live L).
#include "../src/room_swap.h"
#include <QCoreApplication>
#include <clocale>
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
    Fix() : ev(live, mem), recall(live, mem, ev), sw(live, mem, ev, recall) {}
};

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
        check(not live->get_node("room_4").has_value() and not live->get_node("door_9").has_value(), "the live proto is removed with its mirror");
        check(live->get_node("room_2").has_value(), "the room being left is untouched");
        const auto rc = live->get_node("recall_room_3");
        check(rc.has_value(), "recall_room_3 posted");
        if (rc)
        {
            check(live->get_attrib_by_name<recall_late_att>(rc.value()).value_or(false), "recall_late = true");
            const auto pp = live->get_attrib_by_name<recall_pose_prior_att>(rc.value());
            check(pp.has_value() and pp->get().size() == 10 and pp->get()[0] == 1.f and pp->get()[1] == 1.f and pp->get()[2] == 2.f, "recall_pose_prior packs [w,x,y,th,S...]");
            const auto par = live->get_attrib_by_name<parent_att>(rc.value());
            check(par.has_value() and par.value() == live_D, "hangs under the crossed door");
            const auto px = live->get_attrib_by_name<delimiting_polygon_x_att>(rc.value());
            check(px.has_value() and px->get().size() == 4, "carries K's outline");
            const auto T = live->get_inner_eigen_api()->get_transformation_matrix("recall_room_3", "Shadow");
            check(T.has_value() and std::abs(T->translation().x() - 1.0) < 1e-3 and std::abs(T->translation().y() - 2.0) < 1e-3,
                  "the robot is at the recognised pose in the recalled room's frame");
        }
        check(mem_names() == before and mem_rooms() == rooms, "memory: nothing written (same nodes, same room count)");
        check(f.sw.undo(), "undo accepted");
        check(not live->get_node("recall_room_3").has_value() and recall_nodes() == 0, "undo removes the recall");
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
        check(not live->get_node("room_4").has_value(), "the live provisional room is discarded");
        check(live->get_node("recall_room_3").has_value(), "late recall for K posted (under the robot: no door survives)");
        // an archived-room passage into the staged twin must not feed the recall of a live door
        const auto L2 = live_node("room", "room_2", robot_id, -3, -1, 0.2, 2);
        live_node("object", "door_3", L2, 4, 2, 0, std::nullopt, "door");
        f.recall.step();
        check(not live->get_node("recall_room_4").has_value(), "RoomRecall::step skips the staged twin");
        const auto Lr = live->get_node("room_2").value();
        live->delete_node(Lr.id());
        f.sw.on_room_left("room_3");
        check(not mem->get_node("r4_room_4").has_value() and not mem->get_node("r4_door_9").has_value(), "on_room_left: the staged twin and its faces are gone");
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
        check(not live->get_node("recall_room_3").has_value() and f.sw.state() == ltsm::RoomSwap::State::Entering, "late recall dropped, state Entering");
        check(not f.ev.held("r3_room_3"), "the archive is released");
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
        const auto rp = live->get_node("recall_room_4");
        check(rp.has_value(), "a recall for the provisional room is re-posted (it comes back through the same mechanism)");
        if (rp)
        {
            const auto px = live->get_attrib_by_name<delimiting_polygon_x_att>(rp.value());
            check(px.has_value() and px->get().size() == 4, "carrying the twin's outline");
        }
        check(not live->get_node("recall_room_3").has_value(), "K's late recall is gone");
        check(not f.ev.held("r3_room_3"), "the archive is released");
    }

    std::printf("known_room_archive_read_only_while_staged\n");
    {
        reset(); build_memory(false); build_live(false);
        Fix f;
        check(f.sw.swap(3, pose_k(), live_P), "swap accepted");
        // room_concept re-births K from the recall; the robot then leaves it into a new proto
        const auto K = live_node("room", "room_3", robot_id, 0, 0, 0, 3);
        live_node("object", "extra", K, 1, 1, 0);
        const auto N = live_node("room", "room_6", robot_id, 2, 2, 0, 6);
        ltsm::Candidate c;
        c.robot_id = robot_id; c.robot_name = "Shadow";
        c.old_room_id = K; c.old_room_name = "room_3"; c.new_room_id = N; c.new_room_name = "room_6";
        c.remove_old = false;
        const auto snapK = snapshot_room("r3_room_3");
        const auto out = f.ev.evict(c);
        check(not out.ok and out.reason.starts_with("deferred"), "evict of the recalled room is deferred while Staged");
        check(snapshot_room("r3_room_3") == snapK and not mem->get_node("r3_extra").has_value(), "nothing was written into the known twin");
        f.sw.on_evict(c);
        check(f.sw.state() == ltsm::RoomSwap::State::Staged and not f.ev.held("r3_room_3"), "on_evict only releases the hold: still Staged");
        f.sw.on_evicted(c, false);
        check(f.ev.held("r3_room_3") and f.sw.state() == ltsm::RoomSwap::State::Staged, "a failed eviction restores the hold");
        f.sw.on_evict(c);
        const auto out2 = f.ev.evict(c);
        check(out2.ok and mem->get_node("r3_extra").has_value(), "after the release the eviction writes as usual");
        f.sw.on_evicted(c, out2.ok);
        check(f.sw.state() == ltsm::RoomSwap::State::Entering and recall_nodes() == 0, "a successful eviction closes the window and drops the late recall");
    }

    std::printf("entering_the_staged_room_flips_current_without_writing_into_it\n");
    {
        reset(); build_memory(false); build_live(false);
        Fix f;
        check(f.sw.swap(3, pose_k(), live_P), "swap accepted");
        const auto K = live_node("room", "room_3", robot_id, 0, 0, 0, 3);
        live_node("object", "extra", K, 1, 1, 0);
        ltsm::Candidate c;      // promotion of the re-born K: old = L, new = K
        c.robot_id = robot_id; c.robot_name = "Shadow";
        c.old_room_id = live_L; c.old_room_name = "room_2"; c.new_room_id = K; c.new_room_name = "room_3";
        c.remove_old = false;
        const auto snapK = snapshot_room("r3_room_3");
        const auto out = f.ev.evict(c);
        check(out.ok and out.current_flipped, "the promotion into the recalled room goes through (current flips)");
        check(snapshot_room("r3_room_3") == snapK and not mem->get_node("r3_extra").has_value(), "K's twin is not copied into while staged");
        f.sw.on_room_left();
    }

    std::printf("turn_around_in_doorway\n");
    {
        reset(); build_memory(false); build_live(false);
        Fix f;
        check(f.sw.swap(3, pose_k(), live_P), "swap");
        check(f.sw.undo(), "undo");
        live_P = live_node("room", "room_4", robot_id, 1, 2, -0.3, 4);               // room_concept re-births the proto
        live_M = live_node("object", "door_9", live_P, 1, 0, 1, std::nullopt, "door");
        live->insert_or_assign_edge(DSR::Edge::create<proto_edge_type>(live_P, live_P));
        check(f.sw.swap(3, pose_k(), live_P), "swap again");
        check(f.recall.late_count() == 1 and recall_nodes() == 1, "exactly one late recall node, no orphan recall_*");
        check(f.sw.undo() and recall_nodes() == 0, "and undo leaves none");
    }

    std::printf("turn_around_promoted_case_leaves_no_orphan_recall\n");
    {
        reset(); build_memory(true); build_live(true);
        Fix f;
        check(f.sw.swap(3, pose_k(), live_P), "swap");
        check(recall_nodes() == 1, "staged: exactly one recall (K's)");
        check(f.sw.undo(), "undo");
        check(live->get_node("recall_room_4").has_value() and recall_nodes() == 1, "the twin's recall is posted (and tracked)");
        live_P = live_node("room", "room_4", robot_id, 1, 2, -0.3, 4);               // room_concept re-births P from it
        live_M = live_node("object", "door_9", live_P, 1, 0, 1, std::nullopt, "door");
        set_current(live_P);
        f.sw.step();
        check(recall_nodes() == 0, "step retires the twin's recall once its room is live again");
        check(f.sw.swap(3, pose_k(), live_P), "swap again");
        check(recall_nodes() == 1 and live->get_node("recall_room_3").has_value(), "staged again: exactly one recall");
        f.sw.on_room_left("room_3");
        check(recall_nodes() == 0 and not mem->get_node("r4_room_4").has_value(), "after leaving: no recall_* remains, the twin is gone");
        // undo, then the robot leaves with the twin's recall still outstanding
        reset(); build_memory(true); build_live(true);
        Fix g;
        check(g.sw.swap(3, pose_k(), live_P) and g.sw.undo() and recall_nodes() == 1, "swap, undo: the twin's recall outstanding");
        ltsm::Candidate c; c.old_room_name = "room_x";
        g.sw.on_evicted(c, true);
        check(recall_nodes() == 0, "the robot leaving retires it");
    }

    std::printf("k_recall_survives_k_room_appearing_while_staged\n");
    {
        reset(); build_memory(true); build_live(true);
        Fix h;
        check(h.sw.swap(3, pose_k(), live_P), "swap");
        live_node("room", "room_3", robot_id, 0, 0, 0, 3);
        h.sw.step();
        check(live->get_node("recall_room_3").has_value(), "K's room node exists: K's recall is NOT retired while staged");
        h.sw.on_room_left("room_3");
        check(not live->get_node("recall_room_3").has_value(), "retired when the window closes");
    }

    std::printf("door2_becomes_passage_of_both_rooms\n");
    {
        reset(); build_memory(false); build_live(false);
        Fix f;
        check(passage_count() == 0, "(no passage yet: case A)");
        check(f.sw.swap(3, pose_k(), live_P), "swap accepted");
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
        check(f.sw.swap(3, pose_k(), live_P), "swap accepted");
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
        f.sw.on_room_left("room_3");
        live->delete_node(live->get_node("room_2").value().id());
        const auto L2 = live_node("room", "room_2", robot_id, -3, -1, 0.2, 2);
        live_node("object", "door_3", L2, 4, 2, 0, std::nullopt, "door");
        f.recall.step();
        const auto rc = live->get_node("recall_room_3");
        check(rc.has_value(), "recall_room_3 posted under the door");
        if (rc) check(not live->get_attrib_by_name<recall_late_att>(rc.value()).value_or(false), "and it is an ordinary one (no late flag)");
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
        f.recall.post_late(robot_id, "r3_room_3", Mat::RTMat::Identity(), {});
        check(recall_nodes() == 1 and f.recall.late_count() == 1, "(a late recall is live)");
        const int n = f.sw.sweep_stale();
        check(n >= 2, "sweep reports what it removed (" + std::to_string(n) + ")");
        check(not mem->get_node("r4_room_4").has_value() and not mem->get_node("r4_door_9").has_value(), "the staged twin is removed from memory");
        check(recall_nodes() == 0 and mem_rooms() == 2, "the late recall is removed; K and L remain");
        check(f.sw.sweep_stale() == 0, "a second sweep finds nothing");
    }

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "all ok", failures);
    std::fflush(stdout);
    std::_Exit(failures ? 1 : 0);
}
