// room_proposal: recognition PROPOSES (room-transition protocol, spec C2.6/C2.9).   run from the component root:   ./bin/room_proposal_test
// Memory: K = r3_room_3 (known), L = r2_room_2 (the room being left, door face r2_door_3). Case B (the normal case since
// promotion follows room_concept's switch): the provisional room room_4 is promoted and current, memory holds its twin
// r4_room_4 and passage_5 (r2_door_3 <-> r4_door_9). Case A: room_4 is still a proto under the robot, room_2 live.
// The swap writes ONE proposal under P and changes nothing else live; memory staging is committed on "accepted"
// (carry-over, twin dropped) and undone on "rejected", on P gone, or when the robot leaves P before a verdict.
#include "../src/room_swap.h"
#include "../src/room_proposal.h"
#include "../src/room_recall.h"
#include "../../common/room_resolve/room_resolve.h"
#include <QCoreApplication>
#include <clocale>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <sstream>
#include <string>

static int failures = 0;
static void check(bool ok, const std::string &what) { if (not ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str()); }

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
    mem_node("object", "r3_door_5", K, 2, 0, 0.5, std::nullopt, {}, "door");
    const auto L = mem_node("room", "r2_room_2", 0, 0, 0, 0, 2, box);
    const auto fL = mem_node("object", "r2_door_3", L, 4, 2, 0, std::nullopt, {}, "door");
    if (promoted_twin)
    {
        const auto P = mem_node("room", "r4_room_4", 0, 0, 0, 0, 4, box);
        const auto fP = mem_node("object", "r4_door_9", P, 1, 0, 1, std::nullopt, {}, "door");
        passage("passage_5", fL, fP);
    }
}

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

static std::vector<DSR::Node> proposals()
{
    std::vector<DSR::Node> out;
    for (const auto &n : live->get_nodes_by_type("metaconcept"))
        if (live->get_attrib_by_name<object_subtype_att>(n).value_or("") == "room_proposal") out.push_back(n);
    return out;
}

static int k_faces_in(const std::string &passage_name)
{
    const auto pw = mem->get_node(passage_name);
    if (not pw) return -1;
    int n = 0;
    for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(pw.value(), "match"))
        if (ltsm::is_descendant(*mem, e.to(), "r3_room_3")) ++n;
    return n;
}

static int passages_with_k_face()
{
    int n = 0;
    for (const auto &x : mem->get_nodes_by_type("metaconcept"))
        if (mem->get_attrib_by_name<object_subtype_att>(x).value_or("") == "passage" and k_faces_in(x.name()) > 0) ++n;
    return n;
}

/// room_concept's answer, as it writes it (Task 6): `state` on the proposal node.
static void answer(const std::string &state)
{
    auto n = live->get_node("proposal_room_3");
    if (not n) return;
    live->add_or_modify_attrib_local<state_att>(n.value(), std::string(state));
    live->update_node(n.value());
}

struct Fix
{
    ltsm::RoomEviction ev;
    ltsm::RoomSwap sw;
    Fix() : ev(live, mem), sw(live, mem, ev) {}
};

static Mat::RTMat se2(double x, double y, double th)
{
    Mat::RTMat T = Mat::RTMat::Identity();
    T.linear() = Eigen::AngleAxisd(th, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    T.translation() = Eigen::Vector3d(x, y, 0.0);
    return T;
}

int main(int argc, char **argv)
{
    std::setlocale(LC_ALL, "");
    setvbuf(stdout, nullptr, _IONBF, 0);
    QCoreApplication app(argc, argv);
    live = std::make_shared<DSR::DSRGraph>("prop_live", 198, "ltsm_root.json", true, 48);
    mem  = std::make_shared<DSR::DSRGraph>("prop_mem", 199, "ltsm_root.json", true, 49);

    std::printf("post_writes_pending_child_of_P\n");
    for (const bool promoted : {false, true})
    {
        std::printf(" case %s\n", promoted ? "B (promoted)" : "A (proto)");
        reset(); build_memory(promoted); build_live(promoted);
        Fix f;
        const auto before = mem_names();
        check(f.sw.swap(3, pose_k(), live_P), "swap accepted");
        check(f.sw.state() == ltsm::RoomSwap::State::Staged, "state Staged (pending the verdict)");
        check(live->get_node(live_P).has_value() and live->get_node(live_M).has_value(), "rule 1: P and its mirror are untouched (nothing deleted live)");
        check(promoted or live->get_node("room_2").has_value(), "the room being left is untouched");
        int recalls = 0;
        for (const auto &x : live->get_nodes_by_type("metaconcept")) if (x.name().starts_with("recall_")) ++recalls;
        check(recalls == 0, "no late recall is posted");
        const auto ps = proposals();
        check(ps.size() == 1, "exactly one proposal node");
        const auto p = live->get_node("proposal_room_3");
        check(p.has_value() and p->type() == "metaconcept", "proposal_room_3, type metaconcept");
        if (p)
        {
            check(live->get_attrib_by_name<state_att>(p.value()).value_or("") == "pending", "state = pending");
            check(live->get_attrib_by_name<parent_att>(p.value()).value_or(0) == live_P, "parent = P");
            const auto Pl = live->get_node_level(live->get_node(live_P).value()).value_or(-1);
            check(live->get_attrib_by_name<level_att>(p.value()).value_or(-1) == Pl + 1, "level = P's + 1");
            check(live->get_edge(live_P, p->id(), "RT").has_value(), "RT edge P -> proposal");
            check(live->get_attrib_by_name<room_id_att>(p.value()).value_or(0) == 3, "room_id = K's");
            const auto px = live->get_attrib_by_name<delimiting_polygon_x_att>(p.value());
            check(px.has_value() and px->get().size() == 4, "carries K's outline");
            const auto pp = live->get_attrib_by_name<recall_pose_prior_att>(p.value());
            check(pp.has_value() and pp->get().size() == 10 and pp->get()[0] == 1.f and pp->get()[1] == 1.f and pp->get()[2] == 2.f,
                  "recall_pose_prior packs the robot in K [w,x,y,th,S...]");
            // the RT edge holds T(P <- K): through it the robot lands at its recognised pose in K's frame
            const auto T = live->get_inner_eigen_api()->get_transformation_matrix("proposal_room_3", "Shadow");
            check(T.has_value() and std::abs(T->translation().x() - 1.0) < 1e-3 and std::abs(T->translation().y() - 2.0) < 1e-3
                      and std::abs(std::atan2(T->linear()(1, 0), T->linear()(0, 0)) - 0.3) < 1e-3,
                  "T(P <- K) on the RT edge: the robot is at (1, 2, 0.3) in K's frame");
        }
        check(f.ev.held("r3_room_3"), "K's archive is held (read-only while staged)");
        if (promoted)
        {
            const auto t = mem->get_node("r4_room_4");
            check(t.has_value() and mem->get_attrib_by_name<object_subtype_att>(t.value()).value_or("") == "staged", "case B: the twin is staged");
        }
        check(mem_names() == before and k_faces_in("passage_5") <= 0 and passages_with_k_face() == 0,
              "memory: NO carry-over before the verdict (no node added, no K face in any passage)");
        check(f.sw.step() == ltsm::RoomSwap::Event::None and f.sw.state() == ltsm::RoomSwap::State::Staged, "no verdict yet: step() changes nothing");
    }

    std::printf("reproposal_replaces_pending\n");
    {
        reset(); build_memory(true); build_live(true);
        ltsm::RoomProposal rp(live, mem);
        const auto a = rp.post(live_P, "r3_room_3", se2(1, 2, 0.3), {});
        check(a != 0 and rp.verdict() == ltsm::RoomProposal::Verdict::Pending, "first proposal pending");
        const auto b = rp.post(live_P, "r3_room_3", se2(1.5, 2, 0.3), {});
        check(b != 0 and b != a and not live->get_node(a).has_value(), "re-proposal: a NEW node, the superseded one deleted by ltsm");
        check(proposals().size() == 1 and rp.id() == b and rp.verdict() == ltsm::RoomProposal::Verdict::Pending, "exactly one proposal stands, the newest, pending");
        const auto c = rp.post(live_P, "r2_room_2", se2(0, 0, 0), {});
        check(c != 0 and proposals().size() == 1 and live->get_node("proposal_room_2").has_value() and not live->get_node("proposal_room_3").has_value(),
              "a proposal for another room replaces it too (one per P)");
        // a proposal another instance left under P (e.g. a crashed run's) is superseded as well
        ltsm::RoomProposal other(live, mem);
        other.post(live_P, "r3_room_3", se2(1, 2, 0.3), {});
        check(proposals().size() == 1 and rp.verdict() == ltsm::RoomProposal::Verdict::Gone, "the older instance reads its proposal Gone");
        other.withdraw();
        check(proposals().empty() and other.verdict() == ltsm::RoomProposal::Verdict::Gone, "withdraw deletes it");
        // verdicts
        rp.post(live_P, "r3_room_3", se2(1, 2, 0.3), {});
        answer("accepted");
        check(rp.verdict() == ltsm::RoomProposal::Verdict::Accepted, "state accepted => Accepted");
        answer("rejected");
        check(rp.verdict() == ltsm::RoomProposal::Verdict::Rejected, "state rejected => Rejected");
        live->delete_node(live_P);
        check(rp.verdict() == ltsm::RoomProposal::Verdict::Gone, "P gone => Gone");
    }

    std::printf("accepted_commits_staging\n");
    for (const bool promoted : {false, true})
    {
        std::printf(" case %s\n", promoted ? "B (promoted)" : "A (proto)");
        reset(); build_memory(promoted); build_live(promoted);
        Fix f;
        check(f.sw.swap(3, pose_k(), live_P), "swap accepted");
        check(passages_with_k_face() == 0, "(no carry-over while pending)");
        answer("accepted");
        check(f.sw.step() == ltsm::RoomSwap::Event::Accepted, "step(): Accepted");
        check(f.sw.state() == ltsm::RoomSwap::State::Adopted, "state Adopted");
        check(passages_with_k_face() == 1, "carry-over written on the verdict: the crossed door is a passage of K");
        if (promoted)
        {
            check(not mem->get_node("r4_room_4").has_value() and not mem->get_node("r4_door_9").has_value(), "case B: the staged twin is dropped");
            check(mem->get_node("passage_5").has_value() and k_faces_in("passage_5") == 1, "passage_5 survives, re-hung onto K");
        }
        check(f.ev.held("r3_room_3"), "K stays held until the robot leaves it");
        check(live->get_node("proposal_room_3").has_value(), "the accepted proposal stays (door_concept reads it, Task 8)");
        check(f.sw.step() == ltsm::RoomSwap::Event::None, "the verdict is consumed once");
        f.sw.on_room_left("room_3");
        check(not f.ev.held("r3_room_3") and f.sw.state() == ltsm::RoomSwap::State::Entering, "leaving: hold released, Entering");
        check(proposals().empty(), "leaving: the proposal is withdrawn");
        check(passages_with_k_face() == 1, "the carry-over stays");
    }

    std::printf("rejected_unstages_and_rebases\n");
    for (const bool promoted : {false, true})
    {
        std::printf(" case %s\n", promoted ? "B (promoted)" : "A (proto)");
        reset(); build_memory(promoted); build_live(promoted);
        Fix f;
        const auto snapK = snapshot_room("r3_room_3");
        const auto snapP = snapshot_room("r4_room_4");
        const auto before = mem_names();
        check(f.sw.swap(3, pose_k(), live_P), "swap accepted");
        answer("rejected");
        check(f.sw.step() == ltsm::RoomSwap::Event::Rejected, "step(): Rejected (the cycle tells the recogniser to rebase, room_recognizer_test)");
        check(f.sw.state() == ltsm::RoomSwap::State::Entering, "state Entering");
        check(snapshot_room("r3_room_3") == snapK and snapshot_room("r4_room_4") == snapP and mem_names() == before,
              "no memory change survives (K bit-identical, twin restored exactly, same node set)");
        check(not f.ev.held("r3_room_3"), "K's archive released");
        check(proposals().empty(), "the answered proposal is deleted by ltsm");
        check(live->get_node(live_P).has_value(), "P stays live: a new room");
        check(f.sw.swap(3, pose_k(), live_P) and proposals().size() == 1, "ltsm may propose again (rejection is per proposal)");
    }

    std::printf("left_before_verdict_withdraws\n");
    {
        // (a) the robot's live RT room is no longer P (room_concept's fact): step() withdraws
        reset(); build_memory(true); build_live(true);
        Fix f;
        const auto snapP = snapshot_room("r4_room_4");
        const auto before = mem_names();
        check(f.sw.swap(3, pose_k(), live_P), "swap accepted");
        live->delete_edge(robot_id, live_P, "RT");
        const auto back = live_node("room", "room_5", robot_id, 0, 0, 0, 5);
        check(f.sw.step() == ltsm::RoomSwap::Event::Withdrawn, "robot RT room changed before a verdict: Withdrawn");
        check(proposals().empty() and f.sw.state() == ltsm::RoomSwap::State::Entering and not f.ev.held("r3_room_3"), "proposal withdrawn, Entering, released");
        check(snapshot_room("r4_room_4") == snapP and mem_names() == before and passages_with_k_face() == 0, "P's twin is unstaged: P is archived as NEW, no carry-over");
        (void)back;

        // (b) no robot RT room can be resolved (two live edges, a hand-over): nothing is withdrawn on that alone
        reset(); build_memory(false); build_live(false);
        Fix g;
        check(g.sw.swap(3, pose_k(), live_P), "case A swap");
        check(g.sw.step() == ltsm::RoomSwap::Event::None and proposals().size() == 1, "ambiguous robot RT room: the proposal stands");

        // (c) the recogniser's episode ended (released): on_room_left while pending withdraws, never commits
        g.sw.on_room_left();
        check(proposals().empty() and g.sw.state() == ltsm::RoomSwap::State::Entering and passages_with_k_face() == 0,
              "on_room_left before a verdict: withdrawn, nothing carried");

        // (d) the eviction that leaves P (old = P) before a verdict: the staging is undone FIRST, so P's twin is written as new
        reset(); build_memory(true); build_live(true);
        Fix h;
        check(h.sw.swap(3, pose_k(), live_P), "case B swap");
        ltsm::Candidate c;
        c.robot_id = robot_id; c.robot_name = "Shadow";
        c.old_room_id = live_P; c.old_room_name = "room_4"; c.new_room_id = 0; c.new_room_name = "room_5";
        h.sw.on_evict(c);
        const auto t = mem->get_node("r4_room_4");
        check(proposals().empty() and h.sw.state() == ltsm::RoomSwap::State::Entering and t.has_value()
                  and mem->get_attrib_by_name<object_subtype_att>(t.value()).value_or("") != "staged",
              "on_evict(old = P) before a verdict: withdrawn, twin unstaged before the eviction writes it");

        // (e) P deleted by room_concept: Gone => withdrawn
        reset(); build_memory(false); build_live(false);
        Fix k;
        check(k.sw.swap(3, pose_k(), live_P), "case A swap");
        live->delete_node(live_P);
        check(k.sw.step() == ltsm::RoomSwap::Event::Withdrawn and k.sw.state() == ltsm::RoomSwap::State::Entering, "P gone: Withdrawn");
    }

    std::printf("verdict_before_previous_room_archived_carries_on_archive\n");
    {
        // fresh -> fresh: L (room_2) is live-only, its face of the crossed door is NOT in memory; the verdict lands before
        // room_concept's switch archives L. The carry-over must be written the moment L is archived, while the robot is in K.
        reset(); build_memory(false); build_live(false);
        for (const char *nm : {"r2_door_3", "r2_room_2"}) if (const auto n = mem->get_node(nm); n.has_value()) mem->delete_node(n->id());
        Fix f;
        check(f.sw.swap(3, pose_k(), live_P), "swap accepted");
        answer("accepted");
        check(f.sw.step() == ltsm::RoomSwap::Event::Accepted and passages_with_k_face() == 0, "accepted: L's face not in memory yet, carry deferred");
        // the adopted room under K's name (stand-in for room_concept's in-place rename, cortex reinstall pending)
        const auto K = live_node("room", "room_3", robot_id, 0, 0, 0, 3);
        ltsm::Candidate c;
        c.robot_id = robot_id; c.robot_name = "Shadow";
        c.old_room_id = live_L; c.old_room_name = "room_2"; c.new_room_id = K; c.new_room_name = "room_3";
        c.remove_old = false;
        f.sw.on_evict(c);
        const auto out = f.ev.evict(c);
        check(out.ok and mem->get_node("r2_door_3").has_value(), "L archived: its face r2_door_3 is in memory now (" + out.reason + ")");
        f.sw.on_evicted(c, out.ok);
        check(passages_with_k_face() == 1, "on the archive (a fact), the deferred carry-over is written: the way back is known");
        check(f.sw.state() == ltsm::RoomSwap::State::Adopted and f.ev.held("r3_room_3"), "while the robot is still in K (Adopted, held)");
        check(f.sw.step() == ltsm::RoomSwap::Event::None and passages_with_k_face() == 1, "idempotent on the next step");
    }

    std::printf("proposal_before_switch_does_not_self_withdraw\n");
    {
        // only robot->L is live at swap time (room_concept has not switched yet: robot->P retired)
        reset(); build_memory(false); build_live(false);
        auto ep = live->get_edge(robot_id, live_P, "RT").value();
        live->add_or_modify_attrib_local<valid_att>(ep, false);
        live->insert_or_assign_edge(ep);
        check(rc::room::robot_rt_room(*live) == std::optional<std::uint64_t>(live_L), "(robot RT room = L)");
        Fix f;
        check(f.sw.swap(3, pose_k(), live_P), "swap accepted");
        check(f.sw.step() == ltsm::RoomSwap::Event::None and proposals().size() == 1, "robot still in L, where it was at swap time: NOT withdrawn");
        // room_concept switches into P: robot->P live, robot->L retired
        live->add_or_modify_attrib_local<valid_att>(ep, true);
        live->insert_or_assign_edge(ep);
        auto el = live->get_edge(robot_id, live_L, "RT").value();
        live->add_or_modify_attrib_local<valid_att>(el, false);
        live->insert_or_assign_edge(el);
        check(f.sw.step() == ltsm::RoomSwap::Event::None and proposals().size() == 1, "switched into P: still pending");
        // the robot goes on to another room before a verdict: that IS a change away
        live->add_or_modify_attrib_local<valid_att>(ep, false);
        live->insert_or_assign_edge(ep);
        live_node("room", "room_5", robot_id, 0, 0, 0, 5);
        check(f.sw.step() == ltsm::RoomSwap::Event::Withdrawn and proposals().empty(), "moved to room_5 before a verdict: withdrawn");
    }

    std::printf("accepted_adoption_recalls_the_way_back\n");
    {
        // recog_proto6 leg 5 (Task 18): room_3 -> room_4 promoted (recall_room_3 under door_43), room_4 recognised as room_2,
        // ACCEPTED; room_concept renames room_4 -> room_2 IN PLACE (same node id, room_id rewritten) and door_concept re-seats
        // the entry mirror into K's frame. The way back (recall_<previous room> under that door) must be posted again.
        // Here: L = room_2 (archived), P = room_4 (live, mirror door_9), K = r3_room_3.
        reset(); build_memory(true); build_live(true);
        Fix f;
        ltsm::RoomRecall rc(live, mem, f.ev);
        rc.step();
        auto r = live->get_node("recall_room_2");
        check(r.has_value() and live->get_attrib_by_name<parent_att>(r.value()).value_or(0) == live_M,
              "promoted P: recall_room_2 posted under its mirror door_9 (P's memory index is memoised here)");
        check(f.sw.swap(3, pose_k(), live_P), "swap accepted");
        rc.step();
        check(not live->get_node("recall_room_2").has_value(), "staged: P's twin is not part of the map, the recall retires");
        answer("accepted");
        check(f.sw.step() == ltsm::RoomSwap::Event::Accepted and k_faces_in("passage_5") == 1, "accepted: passage_5 carries a K face");
        // the carried K face, in K's frame
        std::optional<DSR::Node> kface;
        if (const auto pw = mem->get_node("passage_5"); pw.has_value())
            for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(pw.value(), "match"))
                if (ltsm::is_descendant(*mem, e.to(), "r3_room_3")) kface = mem->get_node(e.to());
        const auto TK = kface.has_value() ? mem->get_inner_eigen_api()->get_transformation_matrix("r3_room_3", kface->name()) : std::nullopt;
        check(TK.has_value(), "the carried face resolves in K");
        // room_concept: P renamed in place to K's live name, room_id = K's index (proposal_adoption.cpp, one update_node)
        auto P = live->get_node(live_P).value();
        P.name("room_3");
        live->add_or_modify_attrib_local<room_id_att>(P, std::uint64_t{3});
        bool renamed = false;
        try { renamed = live->update_node(P); } catch (const std::exception &) { renamed = false; }
        check(renamed and live->get_node("room_3").has_value() and live->get_node("room_3")->id() == live_P, "P renamed room_4 -> room_3 in place");
        // door_concept: the mirror re-seated into K's frame, at the hole (the carried face's place)
        if (TK.has_value())
        {
            const double th = std::atan2(TK->linear()(1, 0), TK->linear()(0, 0));
            put_edge(*live, live_P, live_M, TK->translation().x(), TK->translation().y(), th);
        }
        const auto hole = f.ev.remembered_hole(live->get_node(live_M).value());
        check(hole.has_value() and hole->passage.name() == "passage_5" and kface.has_value() and hole->face.id() == kface->id(),
              "remembered_hole(door_9) in the adopted room = K's carried face in passage_5 (twin looked up as r3_room_3)");
        rc.step();
        r = live->get_node("recall_room_2");
        check(r.has_value() and live->get_attrib_by_name<parent_att>(r.value()).value_or(0) == live_M,
              "adopted: recall_room_2 (the way back) posted under the adopted room's door at that hole");
        rc.step();
        check(live->get_node("recall_room_2").has_value(), "and it stays on the next cycle");
    }

    std::printf("carry_over_reads_the_accepted_transform\n");
    for (const bool promoted : {false, true})
    {
        // Task 20 (recog_proto6): ltsm's prior T(P <- K) was the half-turn twin of the truth; room_concept's fit corrected it
        // and wrote the ACCEPTED T(P <- K) on the proposal's RT edge before "accepted". The seam must be the accepted one.
        std::printf(" case %s\n", promoted ? "B (promoted)" : "A (proto)");
        const auto se2_of = [](const Mat::RTMat &T) { return Eigen::Vector3d(T.translation().x(), T.translation().y(), std::atan2(T.linear()(1, 0), T.linear()(0, 0))); };
        // the carried K face of the crossed door, in K's frame, after the verdict (accepted_override: the edge rewritten first)
        const auto run = [&](std::optional<Mat::RTMat> accepted_override, Mat::RTMat &prior_out) -> std::optional<Eigen::Vector3d>
        {
            reset(); build_memory(promoted); build_live(promoted);
            Fix f;
            if (not f.sw.swap(3, pose_k(), live_P)) return std::nullopt;
            const auto pn = live->get_node("proposal_room_3");
            if (not pn.has_value()) return std::nullopt;
            const auto e = live->get_edge(live_P, pn->id(), "RT");
            const auto m = e.has_value() ? live->get_rt_api()->get_edge_RT_as_rtmat(e.value()) : std::nullopt;
            if (not m.has_value()) return std::nullopt;
            prior_out = m.value();
            if (accepted_override.has_value())   // room_concept: the accepted T on the edge, THEN the state
            {
                const auto a = se2_of(accepted_override.value());
                put_edge(*live, live_P, pn->id(), a.x(), a.y(), a.z());
            }
            answer("accepted");
            if (f.sw.step() != ltsm::RoomSwap::Event::Accepted) return std::nullopt;
            const char *pw = promoted ? "passage_5" : nullptr;
            for (const auto &x : mem->get_nodes_by_type("metaconcept"))
            {
                if (mem->get_attrib_by_name<object_subtype_att>(x).value_or("") != "passage") continue;
                if (pw != nullptr and x.name() != pw) continue;
                for (const auto &ed : DSR::DSRGraph::get_node_edges_by_type(x, "match"))
                    if (ltsm::is_descendant(*mem, ed.to(), "r3_room_3"))
                        if (const auto fn = mem->get_node(ed.to()); fn.has_value())
                            if (const auto T = mem->get_inner_eigen_api()->get_transformation_matrix("r3_room_3", fn->name()); T.has_value())
                                return se2_of(T.value());
            }
            return std::nullopt;
        };
        Mat::RTMat prior = Mat::RTMat::Identity();
        const auto F_prior = run(std::nullopt, prior);         // accepted == prior: the old behaviour's seam
        check(F_prior.has_value(), "accepted with the prior left on the edge: a K face is carried");
        // the truth: the prior composed with a half turn about K's (2, 2) — the twin a symmetric room allows
        const Mat::RTMat half = se2(2, 2, 0) * se2(0, 0, M_PI) * se2(-2, -2, 0);
        const Mat::RTMat acc(prior * half);
        Mat::RTMat prior2 = Mat::RTMat::Identity();
        const auto F_acc = run(acc, prior2);
        check(F_acc.has_value(), "accepted with the ACCEPTED T on the edge: a K face is carried");
        if (F_prior.has_value() and F_acc.has_value())
        {
            // expected: the same face re-based by C = T_acc(P<-K)^-1 T_prior(P<-K) = half^-1
            const Mat::RTMat C(acc.inverse() * prior);
            const Mat::RTMat Fp = se2(F_prior->x(), F_prior->y(), F_prior->z());
            const Eigen::Vector3d want = se2_of(Mat::RTMat(C * Fp));
            const double dxy = (F_acc->head<2>() - want.head<2>()).norm();
            const double dth = std::abs(std::remainder(F_acc->z() - want.z(), 2.0 * M_PI));
            char buf[256];
            std::snprintf(buf, sizeof buf, "the seam is the ACCEPTED one: face (%.3f, %.3f, %.3f) vs C.prior-face (%.3f, %.3f, %.3f), prior-face (%.3f, %.3f)",
                          F_acc->x(), F_acc->y(), F_acc->z(), want.x(), want.y(), want.z(), F_prior->x(), F_prior->y());
            check(dxy < 1e-3 and dth < 1e-3, buf);
            check((F_acc->head<2>() - F_prior->head<2>()).norm() > 1.0, "the half-turn prior does NOT leak into the seam (faces differ by > 1 m)");
        }
    }

    std::printf("stale_proposal_swept_at_start\n");
    {
        reset(); build_memory(false); build_live(false);
        ltsm::RoomProposal rp(live, mem);
        rp.post(live_P, "r3_room_3", se2(1, 2, 0.3), {});
        Fix f;
        check(f.sw.sweep_stale() >= 1 and proposals().empty(), "RoomSwap::sweep_stale removes a crashed run's proposal");
    }

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "all ok", failures);
    std::fflush(stdout);
    std::_Exit(failures ? 1 : 0);
}
