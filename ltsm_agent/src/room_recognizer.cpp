#include "room_recognizer.h"

#include "seam_fusion.h"
#include "../../common/room_resolve/room_resolve.h"   // rc::room::is_proto, current_edge_room

#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_rt_api.h>

#include <QDebug>
#include <QString>

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <ranges>
#include <unordered_set>

namespace ltsm
{
namespace
{
using seam::Mat3;
using seam::Measured;
using seam::Vec3;

constexpr const char *kRecognition = "recognition";
constexpr const char *kEvidence    = "place_evidence";

// Door-pose covariance prior per FACE: the value room_chain.cpp and seam_evict_test use (3 cm, 0.01 rad). Memory stores no
// covariance today, and a mirror whose edge states none reads as zeros; the prior is what a face that says nothing gets.
Mat3 face_prior()
{
    Mat3 C = Mat3::Zero();
    C.diagonal() << 0.03 * 0.03, 0.03 * 0.03, 0.01 * 0.01;
    return C;
}

Vec3 se2_of(const Mat::RTMat &T)
{
    return {T.translation().x(), T.translation().y(), std::atan2(T.rotation()(1, 0), T.rotation()(0, 0))};
}

/// The (x, y, yaw) block of an RT edge's 6x6 pose covariance; nullopt when the edge states none (all zeros reads as unstated).
std::optional<Mat3> cov3_of(DSR::DSRGraph &g, const DSR::Edge &e)
{
    const auto C6 = g.get_rt_api()->get_edge_RT_covariance(e);
    if (not C6.has_value()) return std::nullopt;
    static constexpr int slot[3] = {0, 1, 5};
    Mat3 S;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) S(i, j) = (*C6)(slot[i], slot[j]);
    if (not (S.cwiseAbs().maxCoeff() > 0.0)) return std::nullopt;
    return S;
}

std::string subtype_of(DSR::DSRGraph &g, const DSR::Node &n)
{
    return g.get_attrib_by_name<object_subtype_att>(n).value_or(std::string{});
}

bool is_staged(DSR::DSRGraph &g, const DSR::Node &room) { return subtype_of(g, room) == "staged"; }

std::optional<DSR::Node> owning_room(DSR::DSRGraph &g, const DSR::Node &n)
{
    auto cur = std::optional<DSR::Node>(n);
    for (int depth = 0; cur.has_value() and depth < 32; ++depth)
    {
        if (cur->type() == "room") return cur;
        const auto p = g.get_attrib_by_name<parent_att>(cur.value());
        if (not p.has_value() or p.value() == cur->id()) return std::nullopt;
        cur = g.get_node(p.value());
    }
    return std::nullopt;
}

std::optional<Eigen::Vector2d> centroid_of(DSR::DSRGraph &g, const DSR::Node &room)
{
    const auto px = g.get_attrib_by_name<delimiting_polygon_x_att>(room);
    const auto py = g.get_attrib_by_name<delimiting_polygon_y_att>(room);
    if (not px.has_value() or not py.has_value() or px->get().empty() or px->get().size() != py->get().size()) return std::nullopt;
    Eigen::Vector2d c = Eigen::Vector2d::Zero();
    for (const auto &[x, y] : std::views::zip(px->get(), py->get())) c += Eigen::Vector2d(x, y);
    return c / static_cast<double>(px->get().size());
}

/// A valid covariance for the posterior: finite, symmetric, strictly positive definite (it has no PSD guard of its own).
std::optional<Eigen::Matrix3f> as_covariance(const Eigen::Matrix3d &S)
{
    if (not S.allFinite()) return std::nullopt;
    const Eigen::Matrix3d Ss = 0.5 * (S + S.transpose());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Ss);
    if (es.info() != Eigen::Success or not (es.eigenvalues().minCoeff() > 0.0)) return std::nullopt;
    return Ss.cast<float>();
}

/// The passage nodes of `g` (metaconcept, subtype "passage") with the set of rooms their faces belong to.
struct PassageRooms { std::string name; std::vector<std::string> rooms; std::vector<std::uint64_t> faces; };

std::vector<PassageRooms> passages_of(DSR::DSRGraph &g)
{
    std::vector<PassageRooms> out;
    for (const auto &n : g.get_nodes_by_type("metaconcept"))
    {
        if (subtype_of(g, n) != "passage") continue;
        PassageRooms p{.name = n.name(), .rooms = {}, .faces = {}};
        for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(n, "match"))
        {
            p.faces.push_back(e.to());
            if (const auto f = g.get_node(e.to()); f.has_value())
                if (const auto r = owning_room(g, f.value()); r.has_value()) p.rooms.push_back(r->name());
        }
        out.push_back(std::move(p));
    }
    return out;
}
}   // namespace

//////////////////////////////////////////////////////////////////////////////////////////////////
RoomRecognizer::RoomRecognizer(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> mem, const RoomEviction &evictor)
    : live_(std::move(live)), mem_(std::move(mem)), ev_(evictor) {}

int RoomRecognizer::sweep_stale()
{
    int n = 0;
    for (const auto &node : live_->get_nodes_by_type("metaconcept"))
        if (const auto st = subtype_of(*live_, node); st == kRecognition or st == kEvidence)
            if (live_->delete_node(node.id())) ++n;
    node_id_ = 0;
    published_.clear();
    published_ids_.clear();
    return n;
}

void RoomRecognizer::end_episode()
{
    if (node_id_ != 0)
    {
        if (live_->get_node(node_id_).has_value()) live_->delete_node(node_id_);
        node_id_ = 0;
    }
    if (prov_id_ != 0)
        qInfo().noquote() << QString::fromStdString(std::format("[recog] episode for provisional room {} ended", prov_id_));
    prov_id_ = 0;
    ids_.clear();
    commit_ = {rc::recog::Commit::State::Entering, -1};
    loglr_at_commit_.clear();
    refuted_.clear();
    post_entry_.clear();
    published_.clear();
    published_ids_.clear();
}

void RoomRecognizer::track_current()
{
    const auto cur = rc::room::current_edge_room(*live_);
    std::string mem_name;
    if (cur.has_value())
        if (const auto room = live_->get_node(cur.value()); room.has_value())
            if (const auto m = ev_.memory_prefix(room.value()) + room->name(); mem_->get_node(m).has_value()) mem_name = m;
    if (mem_name == cur_mem_) return;

    // The current room changed: whatever episode was running was about the room just left behind.
    end_episode();
    prev_mem_.clear();
    prev_passage_.clear();
    if (not cur_mem_.empty() and not mem_name.empty())
        for (const auto &p : passages_of(*mem_))
            if (std::ranges::contains(p.rooms, cur_mem_) and std::ranges::contains(p.rooms, mem_name))
            {
                prev_mem_ = cur_mem_;
                prev_passage_ = p.name;
                break;
            }
    cur_mem_ = mem_name;
}

void RoomRecognizer::publish(const std::vector<float> &post)
{
    if (node_id_ == 0 or not live_->get_node(node_id_).has_value())
    {
        node_id_ = 0;
        if (const auto n = live_->get_node(kRecognition); n.has_value()) node_id_ = n->id();
    }
    if (node_id_ != 0 and ids_ == published_ids_ and post == published_) return;      // identical: no rewrite
    std::optional<DSR::Node> node = node_id_ != 0 ? live_->get_node(node_id_) : std::nullopt;
    const bool fresh = not node.has_value();
    if (fresh)
    {
        DSR::Node n;
        n.type("metaconcept");
        n.name(kRecognition);
        node = n;
        live_->add_or_modify_attrib_local<object_subtype_att>(node.value(), std::string(kRecognition));
        live_->add_or_modify_attrib_local<level_att>(node.value(), 0);
        live_->add_or_modify_attrib_local<pos_x_att>(node.value(), 0.f);
        live_->add_or_modify_attrib_local<pos_y_att>(node.value(), 120.f);
    }
    live_->add_or_modify_attrib_local<recog_room_ids_att>(node.value(), std::vector<std::uint64_t>(ids_));
    live_->add_or_modify_attrib_local<recog_posterior_att>(node.value(), std::vector<float>(post));
    if (fresh)
    {
        const auto id = live_->insert_node(node.value());      // LVALUE (the rvalue overload is not instantiated in libdsr)
        if (not id.has_value()) return;
        node_id_ = id.value();
        qInfo().noquote() << QString::fromStdString(std::format("[recog] recognition node posted: {} candidate room(s)", ids_.size()));
    }
    else
        live_->update_node(node.value());
    published_ = post;
    published_ids_ = ids_;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
RoomRecognizer::Step RoomRecognizer::step()
{
    Step out;
    track_current();

    // ── 1. ELIGIBILITY: current room L, provisional room P with an entry mirror, a crossed door nobody has used ──────
    const auto cur = rc::room::current_edge_room(*live_);
    if (not cur.has_value()) { end_episode(); return out; }
    const auto L = live_->get_node(cur.value());
    if (not L.has_value() or cur_mem_.empty()) { end_episode(); return out; }

    std::optional<DSR::Node> robot;
    for (const auto &e : live_->get_edges_by_type("current"))
        if (e.to() == cur.value()) { robot = live_->get_node(e.from()); break; }
    if (not robot.has_value()) { end_episode(); return out; }

    bool cur_retired = false;
    if (const auto ce = live_->get_edge(robot->id(), cur.value(), "RT"); ce.has_value())
        cur_retired = not live_->get_attrib_by_name<valid_att>(ce.value()).value_or(true);

    // The provisional room: hung from the robot, a proto or (the hand-over case promotion_evidence also takes) live while
    // the robot->current edge is retired.
    std::optional<DSR::Node> P;
    std::optional<DSR::Edge> P_edge;
    for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(robot.value(), "RT"))
    {
        const auto room = live_->get_node(e.to());
        if (not room.has_value() or room->type() != "room" or room->id() == cur.value()) continue;
        const bool entered = cur_retired and live_->get_attrib_by_name<valid_att>(e).value_or(true);
        if (not rc::room::is_proto(*live_, room->id()) and not entered) continue;
        P = room;
        P_edge = e;
        break;
    }
    if (not P.has_value()) { end_episode(); return out; }

    auto ie = live_->get_inner_eigen_api();
    auto rt = live_->get_rt_api();

    // the robot in P, with its covariance (promotion_evidence's reading: no covariance, no pose)
    std::optional<Measured> robot_in_P;
    if (live_->get_attrib_by_name<valid_att>(P_edge.value()).value_or(true))
        if (const auto E = rt->get_edge_RT_as_rtmat(P_edge.value()); E.has_value())
            if (const auto C = cov3_of(*live_, P_edge.value()); C.has_value())
                robot_in_P = seam::inverse_cov(Measured{se2_of(E.value()), C.value()});

    // the entry mirror: a door whose parent is P. The nearest to the robot (an argmin, nothing is excluded by distance).
    std::optional<DSR::Node> M;
    double best_m = std::numeric_limits<double>::infinity();
    for (const auto &d : live_->get_nodes_by_type("object"))
    {
        if (not is_door(*live_, d)) continue;
        const auto par = live_->get_attrib_by_name<parent_att>(d);
        if (not par.has_value() or par.value() != P->id()) continue;
        double dist = 0.0;
        if (robot_in_P.has_value())
            if (const auto T = ie->get_transformation_matrix(P->name(), d.name()); T.has_value())
                dist = (T->translation().head<2>() - robot_in_P->pose.head<2>()).norm();
        if (dist < best_m) { best_m = dist; M = d; }
    }
    if (not M.has_value()) { end_episode(); return out; }

    // its counterpart D in L: the door the live passage matched to it, else the nearest door of L (both in L's frame).
    std::optional<DSR::Node> D;
    for (const auto &n : live_->get_nodes_by_type("metaconcept"))
    {
        if (subtype_of(*live_, n) != "passage") continue;
        const auto m = DSR::DSRGraph::get_node_edges_by_type(n, "match");
        if (std::ranges::none_of(m, [&](const auto &e) { return e.to() == M->id(); })) continue;
        for (const auto &e : m)
            if (e.to() != M->id())
                if (const auto d = live_->get_node(e.to()); d.has_value())
                    if (const auto r = owning_room(*live_, d.value()); r.has_value() and r->id() == L->id()) D = d;
    }
    if (not D.has_value())
    {
        const auto TM = ie->get_transformation_matrix(L->name(), M->name());
        double best = std::numeric_limits<double>::infinity();
        if (TM.has_value())
            for (const auto &d : live_->get_nodes_by_type("object"))
            {
                if (not is_door(*live_, d)) continue;
                const auto r = owning_room(*live_, d);
                if (not r.has_value() or r->id() != L->id()) continue;
                if (const auto T = ie->get_transformation_matrix(L->name(), d.name()); T.has_value())
                    if (const double dist = (T->translation() - TM->translation()).head<2>().norm(); dist < best) { best = dist; D = d; }
            }
    }
    if (not D.has_value()) { end_episode(); return out; }

    // The crossed door must have NO passage in memory. Its face is looked up by name (how eviction/seed wrote it) and, when
    // door_concept re-made the door under a new name, by place: the passage face of L whose aperture holds D's centre.
    const std::string L_mem = cur_mem_;
    const std::string face_name = ev_.memory_prefix(L.value()) + D->name();
    const auto passages = passages_of(*mem_);
    const auto face = mem_->get_node(face_name);
    bool known_door = false;
    if (face.has_value())
        known_door = std::ranges::any_of(passages, [&](const auto &p) { return std::ranges::contains(p.faces, face->id()); });
    else if (const auto TD = ie->get_transformation_matrix(L->name(), D->name()); TD.has_value())
    {
        const double half_w = 0.5 * live_->get_attrib_by_name<width_m_att>(D.value()).value_or(1.f);
        auto mie = mem_->get_inner_eigen_api();
        for (const auto &p : passages)
            for (const auto fid : p.faces)
            {
                const auto f = mem_->get_node(fid);
                if (not f.has_value()) continue;
                const auto fr = owning_room(*mem_, f.value());
                if (not fr.has_value() or fr->name() != L_mem) continue;
                if (const auto Tm = mie->get_transformation_matrix(L_mem, f->name()); Tm.has_value())
                    if ((Tm->translation() - TD->translation()).head<2>().norm() < half_w) known_door = true;
            }
    }
    if (known_door) { end_episode(); return out; }
    if (prov_id_ != P->id())
    {
        end_episode();
        prov_id_ = P->id();
        qInfo().noquote() << QString::fromStdString(std::format("[recog] episode begins: provisional {} behind {}", P->name(), L->name()));
    }

    // ── 2. CANDIDATES and the metric query ─────────────────────────────────────────────────────────────────────────
    // The fresh seam T(L <- P) from the crossed door's two faces; nullopt (=> metric unavailable) when either is missing.
    std::optional<ChainPose> fresh;
    const auto mem_L = mem_->get_node(L_mem);
    if (face.has_value() and robot_in_P.has_value() and mem_L.has_value())
    {
        auto mie = mem_->get_inner_eigen_api();
        const auto TL = mie->get_transformation_matrix(L_mem, face_name);          // T(L <- its face), memory
        const auto edge_M = live_->get_edge(P->id(), M->id(), "RT");               // T(P <- the mirror), live
        std::optional<Mat::RTMat> TP;
        std::optional<Mat3> CP;
        if (edge_M.has_value())
        {
            TP = rt->get_edge_RT_as_rtmat(edge_M.value());
            CP = cov3_of(*live_, edge_M.value());
        }
        else
            TP = ie->get_transformation_matrix(P->name(), M->name());
        if (TL.has_value() and TP.has_value())
        {
            Mat3 CL = face_prior();
            if (const auto par = mem_->get_attrib_by_name<parent_att>(face.value()); par.has_value())
                if (const auto fe = mem_->get_edge(par.value(), face->id(), "RT"); fe.has_value())
                    if (const auto c = cov3_of(*mem_, fe.value()); c.has_value()) CL = c.value();
            const Measured fL{se2_of(TL.value()), CL};
            const Measured inv_fP = seam::inverse_cov(Measured{se2_of(TP.value()), CP.value_or(face_prior())});
            Measured seam_LP = seam::compose_cov(fL, inv_fP);

            // The two faces are ONE hole, possibly TURNED by pi (door_concept orients a door's frame from the room that fits
            // it, so the mirror seen from P is the same doorway rotated half a turn). The provisional room lies on the
            // far side of the hole from L, and the robot is in it by definition: if the seam lands the robot on L's side
            // of the face, the turned face is the one. A side test, not a tolerance; the same rule seam_through_passage
            // applies with the two outlines' centroids.
            if (const auto c_L = centroid_of(*mem_, mem_L.value()); c_L.has_value())
            {
                const Eigen::Vector2d r_in_L = seam_LP.pose.head<2>() + seam::rot(seam_LP.pose.z()) * robot_in_P->pose.head<2>();
                const Eigen::Vector2d o = fL.pose.head<2>();
                const Eigen::Vector2d n(-std::sin(fL.pose.z()), std::cos(fL.pose.z()));
                if (n.dot(c_L.value() - o) * n.dot(r_in_L - o) > 0.0)
                    seam_LP = seam::compose_cov(seam::compose_cov(fL, Measured{Vec3(0.0, 0.0, M_PI), Mat3::Zero()}), inv_fP);
            }
            fresh = ChainPose{seam_LP.pose, seam_LP.cov};
        }
    }

    // path taken: the passage that brought the robot into L (from its previous room), else L itself is the reference.
    std::vector<std::string> path;
    std::string start = L_mem;
    if (not prev_passage_.empty() and not prev_mem_.empty()) { path = {prev_passage_}; start = prev_mem_; }
    ChainPose robot_prov;
    if (robot_in_P.has_value()) robot_prov = {robot_in_P->pose, robot_in_P->cov};
    auto q = build_metric_query(*mem_, path, start, robot_in_P.has_value() ? fresh : std::nullopt, robot_prov);

    // Candidates: archived rooms that are not staged (build_metric_query already drops them), not L (the crossing rules it
    // out), and not P's own twin if memory holds one. Sorted by room_id: the order retina's evidence is keyed to.
    const auto P_rid = live_->get_attrib_by_name<room_id_att>(P.value());
    std::erase_if(q.rooms, [&](const KnownRoomMetric &k) {
        return k.mem_room == L_mem or (P_rid.has_value() and k.room_id == P_rid.value()); });
    std::ranges::sort(q.rooms, {}, &KnownRoomMetric::room_id);
    if (q.rooms.empty()) { end_episode(); return out; }

    // the candidate list changed under a running episode: carry the commit by id, drop it if its room left
    {
        std::vector<std::uint64_t> ids;
        for (const auto &k : q.rooms) ids.push_back(k.room_id);
        if (ids != ids_)
        {
            if (commit_.state == rc::recog::Commit::State::Committed)
            {
                const auto old = (commit_.room >= 0 and commit_.room < int(ids_.size())) ? std::optional(ids_[commit_.room]) : std::nullopt;
                const auto it = old.has_value() ? std::ranges::find(ids, old.value()) : ids.end();
                commit_ = it == ids.end() ? rc::recog::Commit{rc::recog::Commit::State::Entering, -1}
                                          : rc::recog::Commit{rc::recog::Commit::State::Committed, int(it - ids.begin())};
            }
            ids_ = std::move(ids);
        }
    }
    const std::size_t n = ids_.size();

    // ── 3. EVIDENCE ──────────────────────────────────────────────────────────────────────────────────────────────────
    float log_new = 0.f;
    auto metric = metric_evidence(q, log_new);
    std::vector<float> area(n, 0.f);
    for (std::size_t k = 0; k < n; ++k)
    {
        area[k] = float(q.rooms[k].area);
        // The posterior has no PSD guard: a pose density that is not a covariance (or whose room has no area to normalise
        // the pose prior) is dropped, leaving the room's inside-mass; that is validity, not a model threshold.
        auto &pm = metric[k].pose_in_k;
        for (auto &g : pm)
        {
            const auto S = as_covariance(g.S.cast<double>());
            if (S.has_value() and g.mu.allFinite()) g.S = S.value(); else g.w = -1.f;
        }
        std::erase_if(pm, [](const rc::recog::PoseGauss &g) { return not (g.w >= 0.f); });
        if (not (area[k] > 0.f)) pm.clear();
    }

    // appearance: retina's place_evidence, matched BY room id (it may carry an older candidate list)
    std::vector<rc::recog::AppearanceEvidence> app(n);
    for (const auto &pe : live_->get_nodes_by_type("metaconcept"))
    {
        if (subtype_of(*live_, pe) != kEvidence) continue;
        const auto ids = live_->get_attrib_by_name<recog_room_ids_att>(pe);
        const auto llr = live_->get_attrib_by_name<recog_loglr_att>(pe);
        const auto mix = live_->get_attrib_by_name<recog_mix_att>(pe);
        const auto mixn = live_->get_attrib_by_name<recog_mix_n_att>(pe);
        if (not ids.has_value() or not llr.has_value() or ids->get().size() != llr->get().size()) break;
        std::size_t off = 0;
        for (std::size_t j = 0; j < ids->get().size(); ++j)
        {
            const std::size_t cnt = (mixn.has_value() and j < mixn->get().size()) ? std::size_t(mixn->get()[j]) : 0;
            const auto it = std::ranges::find(ids_, ids->get()[j]);
            if (it != ids_.end() and std::isfinite(llr->get()[j]))
            {
                auto &a = app[it - ids_.begin()];
                a.available = true;
                a.loglr = llr->get()[j];
                if (mix.has_value())
                    for (std::size_t c = 0; c < cnt and (off + c + 1) * 10 <= mix->get().size(); ++c)
                    {
                        const float *f = mix->get().data() + (off + c) * 10;
                        Eigen::Matrix3d S;
                        S << f[4], f[5], f[6],
                             f[5], f[7], f[8],
                             f[6], f[8], f[9];
                        const auto cov = as_covariance(S);
                        if (cov.has_value() and std::isfinite(f[0]) and f[0] >= 0.f and std::isfinite(f[1]) and std::isfinite(f[2]) and std::isfinite(f[3]))
                            a.pose_in_k.push_back({Eigen::Vector3f(f[1], f[2], rc::place::wrap_angle(f[3])), cov.value(), f[0]});
                    }
            }
            off += cnt;
        }
        break;
    }

    // ── 4. POSTERIORS and the staged commit ──────────────────────────────────────────────────────────────────────────
    const auto since = [&](const std::map<std::uint64_t, float> &base) {
        auto a = app;
        for (std::size_t k = 0; k < n; ++k)
            if (a[k].available)
                if (const auto it = base.find(ids_[k]); it != base.end()) a[k].loglr -= it->second;
        return a; };
    post_entry_ = rc::recog::posterior(metric, app, area, log_new);

    // A room whose commit was refuted by evidence gathered after it cannot be committed again in this episode (otherwise the
    // metric, which is a state and not a sum, re-commits it the very next cycle). Excluded from the MAP only.
    auto eff = post_entry_;
    for (auto &h : eff)
        if (h.room >= 0 and h.room < int(n) and refuted_.contains(ids_[h.room])) h.log_post = -std::numeric_limits<float>::infinity();

    std::vector<rc::recog::Hypothesis> post_commit;
    if (commit_.state == rc::recog::Commit::State::Committed)
        post_commit = rc::recog::posterior(metric, since(loglr_at_commit_), area, log_new);

    const auto next = rc::recog::commit_step(commit_, eff, post_commit);
    if (commit_.state == rc::recog::Commit::State::Entering and next.state == rc::recog::Commit::State::Committed)
    {
        out.act = Step::Act::Swap;
        out.room_index = next.room;
        out.room_id = ids_[next.room];
        out.pose_in_k = post_entry_[next.room].pose_in_k;
        loglr_at_commit_.clear();
        for (std::size_t k = 0; k < n; ++k)
            if (app[k].available) loglr_at_commit_[ids_[k]] = app[k].loglr;
        qInfo().noquote() << QString::fromStdString(std::format("[recog] SWAP requested: room_id {} (posterior {:.3f})",
                                                                out.room_id, std::exp(post_entry_[next.room].log_post)));
    }
    else if (commit_.state == rc::recog::Commit::State::Committed and next.state == rc::recog::Commit::State::Entering)
    {
        out.act = Step::Act::Undo;
        out.room_index = commit_.room;
        out.room_id = ids_[commit_.room];
        out.pose_in_k = post_commit[commit_.room].pose_in_k;
        refuted_.insert(out.room_id);
        loglr_at_commit_.clear();
        qInfo().noquote() << QString::fromStdString(std::format("[recog] UNDO requested: room_id {} refuted by later evidence", out.room_id));
    }
    commit_ = next;

    std::vector<float> post;
    for (const auto &h : post_entry_) post.push_back(std::exp(h.log_post));
    publish(post);
    return out;
}
}   // namespace ltsm
