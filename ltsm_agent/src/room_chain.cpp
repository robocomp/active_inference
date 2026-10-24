#include "room_chain.h"

#include "seam_fusion.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <ranges>

namespace ltsm
{
namespace
{
using seam::Mat3;
using seam::Measured;
using seam::Vec3;

// Door-pose covariance prior per FACE (the value seam_evict_test already uses for a door node): 3 cm, 0.01 rad.
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

bool is_staged(DSR::DSRGraph &g, const DSR::Node &room)
{
    return g.get_attrib_by_name<object_subtype_att>(room).value_or(std::string{}) == "staged";
}

struct PassageEnds { std::string name, room_a, room_b; };

std::optional<PassageEnds> ends_of(DSR::DSRGraph &g, const DSR::Node &passage)
{
    PassageEnds p{.name = passage.name()};
    for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(passage, "match"))
    {
        const auto f = g.get_node(e.to());
        if (not f.has_value()) continue;
        const auto r = owning_room(g, f.value());
        if (not r.has_value()) continue;
        if (p.room_a.empty()) p.room_a = r->name();
        else if (p.room_b.empty() and r->name() != p.room_a) p.room_b = r->name();
    }
    if (p.room_a.empty() or p.room_b.empty()) return std::nullopt;
    return p;
}

std::vector<PassageEnds> all_passages(DSR::DSRGraph &g)
{
    std::vector<PassageEnds> out;
    for (const auto &n : g.get_nodes_by_type("metaconcept"))
        if (g.get_attrib_by_name<object_subtype_att>(n).value_or(std::string{}) == "passage")
            if (const auto p = ends_of(g, n); p.has_value()) out.push_back(p.value());
    return out;
}

/// The face of `passage` inside `room`: its pose there and the covariance of its own RT edge (prior when unstated).
std::optional<Measured> face_in(DSR::DSRGraph &g, const std::string &passage, const std::string &room)
{
    const auto dw = g.get_node(passage);
    if (not dw.has_value()) return std::nullopt;
    auto ie = g.get_inner_eigen_api();
    for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(dw.value(), "match"))
    {
        const auto f = g.get_node(e.to());
        if (not f.has_value() or not is_descendant(g, f->id(), room)) continue;
        const auto T = ie->get_transformation_matrix(room, f->name());
        if (not T.has_value()) continue;
        Mat3 C = face_prior();
        if (const auto par = g.get_attrib_by_name<parent_att>(f.value()); par.has_value())
            if (const auto edge = g.get_edge(par.value(), f->id(), "RT"); edge.has_value())
                if (const auto C6 = g.get_rt_api()->get_edge_RT_covariance(edge.value()); C6.has_value())
                {
                    static constexpr int slot[3] = {0, 1, 5};
                    Mat3 S;
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j) S(i, j) = (*C6)(slot[i], slot[j]);
                    // an edge that exists but states nothing reads as all zeros: that is "unstated", keep the prior
                    if (S.cwiseAbs().maxCoeff() > 0.0) C = S;
                }
        return Measured{se2_of(T.value()), C};
    }
    return std::nullopt;
}

/// T(to <- from) through `passage`, with the covariance of its two faces.
std::optional<Measured> seam_cov(DSR::DSRGraph &g, const std::string &passage, const std::string &to, const std::string &from)
{
    const auto T = seam_through_passage(g, passage, to, from);
    const auto ft = face_in(g, passage, to), ff = face_in(g, passage, from);
    if (not T.has_value() or not ft.has_value() or not ff.has_value()) return std::nullopt;
    Measured m = seam::compose_cov(ft.value(), seam::inverse_cov(ff.value()));
    m.pose = se2_of(T.value());      // the graph's own composition is the mean (it also resolves a face turned by pi)
    return m;
}

Eigen::Affine2d affine_of(const Vec3 &p)
{
    Eigen::Affine2d A = Eigen::Affine2d::Identity();
    A.rotate(p.z());
    A.pretranslate(Eigen::Vector2d(p.x(), p.y()));
    return A;
}

std::vector<Eigen::Vector2d> polygon_of(DSR::DSRGraph &g, const DSR::Node &room)
{
    std::vector<Eigen::Vector2d> out;
    const auto px = g.get_attrib_by_name<delimiting_polygon_x_att>(room);
    const auto py = g.get_attrib_by_name<delimiting_polygon_y_att>(room);
    if (not px.has_value() or not py.has_value() or px->get().size() != py->get().size()) return out;
    for (const auto &[x, y] : std::views::zip(px->get(), py->get())) out.emplace_back(x, y);
    return out;
}

double shoelace_area(const std::vector<Eigen::Vector2d> &poly)
{
    double a2 = 0.0;
    for (std::size_t i = 0; i < poly.size(); ++i)
    {
        const auto &p = poly[i];
        const auto &q = poly[(i + 1) % poly.size()];
        a2 += p.x() * q.y() - q.x() * p.y();
    }
    return 0.5 * std::abs(a2);
}

bool inside(const std::vector<Eigen::Vector2d> &poly, const Eigen::Vector2d &x)
{
    bool in = false;
    for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
        if ((poly[i].y() > x.y()) != (poly[j].y() > x.y())
            and x.x() < (poly[j].x() - poly[i].x()) * (x.y() - poly[i].y()) / (poly[j].y() - poly[i].y()) + poly[i].x())
            in = not in;
    return in;
}

// Gauss-Hermite (probabilists', weights sum to 1), n points, by Golub-Welsch.
struct Rule { std::vector<double> x, w; };
const Rule &hermite()
{
    static const Rule r = [] {
        constexpr int n = 9;
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);
        for (int i = 1; i < n; ++i) J(i, i - 1) = J(i - 1, i) = std::sqrt(double(i));
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J);
        Rule out;
        for (int i = 0; i < n; ++i) { out.x.push_back(es.eigenvalues()(i)); out.w.push_back(es.eigenvectors()(0, i) * es.eigenvectors()(0, i)); }
        return out;
    }();
    return r;
}

/// Gaussian N(mu, S) over (x, y): quadrature nodes (x, w) and the whitening map for tail distances.
struct Gauss2
{
    Eigen::Vector2d mu;
    Eigen::Matrix2d U, Linv;   // z = Linv * U^T (p - mu)
    std::vector<std::pair<Eigen::Vector2d, double>> nodes;

    Gauss2(const Eigen::Vector2d &m, const Eigen::Matrix2d &S) : mu(m)
    {
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(0.5 * (S + S.transpose()));
        U = es.eigenvectors();
        const Eigen::Vector2d lam = es.eigenvalues().cwiseMax(std::numeric_limits<double>::min());   // numerical, not a model floor
        Linv = lam.cwiseSqrt().cwiseInverse().asDiagonal();
        const Eigen::Vector2d s = lam.cwiseSqrt();
        const auto &r = hermite();
        for (std::size_t i = 0; i < r.x.size(); ++i)
            for (std::size_t j = 0; j < r.x.size(); ++j)
                nodes.push_back({mu + U * Eigen::Vector2d(s.x() * r.x[i], s.y() * r.x[j]), r.w[i] * r.w[j]});
    }

    /// Mahalanobis distance from the mean to the boundary of `poly`.
    double dist_to(const std::vector<Eigen::Vector2d> &poly) const
    {
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < poly.size(); ++i)
        {
            const Eigen::Vector2d a = Linv * U.transpose() * (poly[i] - mu);
            const Eigen::Vector2d b = Linv * U.transpose() * (poly[(i + 1) % poly.size()] - mu);
            const Eigen::Vector2d ab = b - a;
            const double len2 = ab.squaredNorm();
            const double t = len2 > 0.0 ? std::clamp(-a.dot(ab) / len2, 0.0, 1.0) : 0.0;
            best = std::min(best, (a + t * ab).norm());
        }
        return best;
    }
};

/// Mass of a region from its quadrature mass `q` plus the mass the 9-point rule cannot see. A Gaussian puts at most
/// exp(-d^2/2) beyond Mahalanobis distance d, and the rule's outermost node (r_max) is where its resolution ends, so
/// the unresolved part is exp(-(d^2 + r_max^2)/2): smooth, monotone in d, vanishing where the rule resolves, no switch.
double region_log_mass(double q, double dist)
{
    const double r = hermite().x.back();
    const double x = -0.5 * (dist * dist + r * r);
    const double lq = q > 0.0 ? std::log(q) : -std::numeric_limits<double>::infinity();   // log(0) = -inf: exact, not a switch
    const double m = std::max(lq, x);                                                    // log(q + e^x), safe from underflow
    return m + std::log(std::exp(lq - m) + std::exp(x - m));
}
}   // namespace

//////////////////////////////////////////////////////////////////////////////////////////////////
MetricQuery build_metric_query(DSR::DSRGraph &mem, const std::vector<std::string> &path_passages,
                               const std::string &path_start_room, const std::optional<ChainPose> &seam_last_to_provisional,
                               const ChainPose &robot_in_provisional)
{
    MetricQuery q;
    q.reference_room = path_start_room;
    const auto ref = mem.get_node(path_start_room);
    if (not ref.has_value()) return q;

    const auto passages = all_passages(mem);

    // ── the PATH chain ─────────────────────────────────────────────────────────────────────────
    {
        Measured acc{Vec3::Zero(), Mat3::Zero()};
        std::string here = path_start_room;
        bool ok = true;
        for (const auto &pn : path_passages)
        {
            const auto it = std::ranges::find(passages, pn, &PassageEnds::name);
            if (it == passages.end() or (it->room_a != here and it->room_b != here)) { ok = false; break; }
            const std::string next = it->room_a == here ? it->room_b : it->room_a;
            const auto s = seam_cov(mem, pn, here, next);
            if (not s.has_value()) { ok = false; break; }
            acc = seam::compose_cov(acc, s.value());
            here = next;
        }
        if (ok and seam_last_to_provisional.has_value())
        {
            // the crossed door has no passage: its seam is the fresh one measured from its two faces
            acc = seam::compose_cov(acc, Measured{seam_last_to_provisional->pose, seam_last_to_provisional->cov});
            const auto r = seam::compose_cov(acc, Measured{robot_in_provisional.pose, robot_in_provisional.cov});
            q.robot_in_ref = {r.pose, r.cov};
            q.robot_ok = true;
        }
    }

    // ── each room's OWN chain: BFS over the passages in memory (the crossed door is none of them) ──
    std::map<std::string, Measured> placed;       // room -> T(ref <- room)
    placed[path_start_room] = Measured{Vec3::Zero(), Mat3::Zero()};
    std::deque<std::string> frontier{path_start_room};
    while (not frontier.empty())
    {
        const std::string j = frontier.front(); frontier.pop_front();
        for (const auto &p : passages)
        {
            if ((p.room_a != j and p.room_b != j)) continue;
            const std::string m = p.room_a == j ? p.room_b : p.room_a;
            if (placed.contains(m)) continue;
            if (const auto rm = mem.get_node(m); not rm.has_value() or is_staged(mem, rm.value())) continue;
            const auto s = seam_cov(mem, p.name, j, m);
            if (not s.has_value()) continue;
            placed[m] = seam::compose_cov(placed.at(j), s.value());
            frontier.push_back(m);
        }
    }

    for (const auto &n : mem.get_nodes_by_type("room"))
    {
        if (is_staged(mem, n)) continue;      // a staged twin is a draft of a room, never a candidate
        KnownRoomMetric k;
        k.room_id = mem.get_attrib_by_name<room_id_att>(n).value_or(n.id());
        k.mem_room = n.name();
        if (const auto it = placed.find(n.name()); it != placed.end())
        {
            k.chain_ok = true;
            k.T_ref_k = affine_of(it->second.pose);
            k.cov_T = it->second.cov;
            for (const auto &v : polygon_of(mem, n)) k.polygon_in_ref.push_back(k.T_ref_k * v);
            k.area = shoelace_area(k.polygon_in_ref);
        }
        q.rooms.push_back(std::move(k));
    }
    return q;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<rc::recog::MetricEvidence> metric_evidence(const MetricQuery &q, float &log_new_out)
{
    std::vector<rc::recog::MetricEvidence> out(q.rooms.size());
    log_new_out = 0.f;
    if (not q.robot_ok) return out;

    double log_new = 0.0;
    for (std::size_t k = 0; k < q.rooms.size(); ++k)
    {
        const auto &r = q.rooms[k];
        if (not r.chain_ok or r.polygon_in_ref.size() < 3) continue;
        Eigen::Matrix3d S = q.robot_in_ref.cov + r.cov_T;
        S = 0.5 * (S + S.transpose());
        const Gauss2 g(q.robot_in_ref.pose.head<2>(), S.topLeftCorner<2, 2>());
        double in = 0.0, outside = 0.0;
        for (const auto &[x, w] : g.nodes)
            (inside(r.polygon_in_ref, x) ? in : outside) += w;
        const double d = g.dist_to(r.polygon_in_ref);

        // pose in k: the robot expressed in k's frame (the two chains are treated as independent: conservative)
        const Measured Tk{{r.T_ref_k.translation().x(), r.T_ref_k.translation().y(), std::atan2(r.T_ref_k.linear()(1, 0), r.T_ref_k.linear()(0, 0))}, r.cov_T};
        const Measured in_k = seam::compose_cov(seam::inverse_cov(Tk), Measured{q.robot_in_ref.pose, q.robot_in_ref.cov});
        rc::recog::PoseGauss pg;
        pg.mu = in_k.pose.cast<float>();
        pg.S = (0.5 * (in_k.cov + in_k.cov.transpose())).cast<float>();
        pg.w = 1.f;
        out[k].pose_in_k = {pg};
        out[k].log_inside = float(region_log_mass(in, d));
        out[k].available = true;
        log_new += region_log_mass(outside, d);   // log(1 - P_inside_k) under THIS room's own S
    }
    log_new_out = float(log_new);
    return out;
}
}   // namespace ltsm
