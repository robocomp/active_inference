// toy_model.h — the smallest model the shared persistent engine accepts (common/ai_belief/persistent_engine.h).
//
// One upright box, θ = (cx, cy, H, w, h, yaw) (N = 6: beam_ray.h's BeamPart::J maps six parameters), standing on the floor.
// The camera points measure its surface through a single Gaussian (no clutter, no wall: one primitive, responsibility 1);
// the LiDAR rays read it as one convex beam part. It implements ONLY the hooks PersistentModel requires — no camera extra
// terms, no energy-less accumulate_extra overload, no solver() — which is what proves the minimal concept is enough. Every
// engine test in this directory builds on it.
#pragma once

#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "ai_belief/beam_channel.h"
#include "ai_belief/persistent_engine.h"
#include "ai_belief/viewpoint.h"

namespace toy
{

struct ToyState
{
    float cx = 0.f, cy = 0.f, H = 1.f, w = 1.f, h = 1.f, yaw = 0.f;
    static ToyState from_vec(const Eigen::Matrix<float, 6, 1>& v) { return {v(0), v(1), v(2), v(3), v(4), v(5)}; }
    Eigen::Matrix<float, 6, 1> vec() const { return (Eigen::Matrix<float, 6, 1>() << cx, cy, H, w, h, yaw).finished(); }
};

struct ToyFrame
{
    std::vector<Eigen::Vector3f> points;   // camera points, room frame
    std::vector<float> R;                  // per-point variance (empty ⇒ the model's σ²)
    rc::ai::beam::BeamRays beam;           // the staged LiDAR rays
};

struct ToyBox
{
    static constexpr int N = 6;
    static constexpr int kYaw = 5;
    using State = ToyState;
    using Frame = ToyFrame;
    using MatN = Eigen::Matrix<float, N, N>;
    using VecN = Eigen::Matrix<float, N, 1>;

    State s;
    MatN  Sigma = MatN::Identity();
    float sigma_m = 0.01f;                          // camera point noise (m)
    VecN  prior_mean = VecN::Zero();                // class mean of (H, w, h) in components 2..4; position at the seed
    VecN  prior_diag = VecN::Constant(0.09f);       // Σ₀
    VecN  q_diag     = VecN::Zero();                // Q
    VecN  layer_spread = VecN::Zero();              // Δ per DOF, the class's layer spread (shape_spread(); 0 = no layering)
    rc::ai::beam::BeamConfig beam_cfg{};            // BeamConfig{}: NOT built by make (the "beam not ready" case)

    // ── the box SDF in its local frame: local = R(−yaw)(p − c), c = (cx, cy, H/2), half = (w/2, h/2, H/2) ──
    struct Local { Eigen::Vector3d pl, q; };
    static Local local(const Eigen::Vector3f& p, const State& st)
    {
        const double c = std::cos(double(st.yaw)), sn = std::sin(double(st.yaw));
        const double dx = double(p.x()) - st.cx, dy = double(p.y()) - st.cy, dz = double(p.z()) - 0.5 * st.H;
        Local L;
        L.pl = {c * dx + sn * dy, -sn * dx + c * dy, dz};
        const Eigen::Vector3d half(0.5 * st.w, 0.5 * st.h, 0.5 * st.H);
        L.q = L.pl.cwiseAbs() - half;
        return L;
    }
    static double sdf(const Local& L)
    {
        const double outside = L.q.cwiseMax(0.0).norm();
        return outside + std::min(0.0, L.q.maxCoeff());
    }
    // ∂d/∂q (the face normal inside, the closest-feature direction outside)
    static Eigen::Vector3d dsdf_dq(const Local& L)
    {
        const Eigen::Vector3d qp = L.q.cwiseMax(0.0);
        const double n = qp.norm();
        if (n > 0.0) return qp / n;
        Eigen::Vector3d g = Eigen::Vector3d::Zero();
        int k = 0;
        L.q.maxCoeff(&k);
        g(k) = 1.0;
        return g;
    }

    float sdf_prim(const Eigen::Vector3f& p, const State& st, int) const { return float(sdf(local(p, st))); }
    VecN  sdf_jacobian(const Eigen::Vector3f& p, const State& st, int) const
    {
        const Local L = local(p, st);
        const Eigen::Vector3d dq = dsdf_dq(L);
        const Eigen::Vector3d g(dq.x() * (L.pl.x() < 0 ? -1.0 : 1.0), dq.y() * (L.pl.y() < 0 ? -1.0 : 1.0),
                                dq.z() * (L.pl.z() < 0 ? -1.0 : 1.0));   // ∂d/∂p_local
        const double c = std::cos(double(st.yaw)), sn = std::sin(double(st.yaw));
        VecN J;
        J(0) = float(-(c * g.x() - sn * g.y()));                 // ∂d/∂cx = −(R g)_x
        J(1) = float(-(sn * g.x() + c * g.y()));                 // ∂d/∂cy = −(R g)_y
        J(2) = float(0.5 * (-g.z() - dq.z()));                   // cz = H/2 and half_z = H/2
        J(3) = float(-0.5 * dq.x());
        J(4) = float(-0.5 * dq.y());
        J(5) = float(g.x() * L.pl.y() - g.y() * L.pl.x());       // ∂p_local/∂yaw = (p_ly, −p_lx)
        return J;
    }
    std::array<float, 1> responsibilities(const Eigen::Vector3f&, const State&, float) const { return {1.0f}; }
    int   n_prims() const { return 1; }
    float sigma2() const { return sigma_m * sigma_m; }
    double point_energy(const Eigen::Vector3f& p, const State& st, float R) const
    {
        const double d = sdf(local(p, st));
        return 0.5 * d * d / double(R);
    }

    // the θ-only channel: the toy has no θ-only terms
    void accumulate_extra(const State&, const Frame&, MatN&, VecN&, double*) const {}

    std::vector<rc::ai::beam::BeamPart> beam_parts(const State& st) const
    {
        rc::ai::beam::BeamPart P;
        P.c = {st.cx, st.cy, 0.5 * st.H};
        P.yaw = st.yaw;
        P.half = {0.5 * st.w, 0.5 * st.h, 0.5 * st.H};
        P.J.setZero();   // rows ∂(cx, cy, cz, yaw, hx, hy, hz)/∂θ
        P.J(0, 0) = 1.0; P.J(1, 1) = 1.0; P.J(2, 2) = 0.5; P.J(3, 5) = 1.0; P.J(4, 3) = 0.5; P.J(5, 4) = 0.5; P.J(6, 2) = 0.5;
        return {P};
    }
    const rc::ai::beam::BeamRays&   beam_rays(const Frame& f) const { return f.beam; }
    const rc::ai::beam::BeamConfig& beam_config() const { return beam_cfg; }
    Eigen::Matrix<double, N, 3> chain_jacobian(const State&, const Eigen::Vector2f& lever) const { return rc::ai::view::chain_B(lever); }
    void apply_constraints(State&) const {}
    void canonicalize(State& st) const { st.yaw = std::remainder(st.yaw, float(2.0 * M_PI)); }
    VecN shape_spread() const { return layer_spread; }
    VecN prior_cov_diag() const { return prior_diag; }
    VecN process_noise_diag() const { return q_diag; }
    Eigen::Matrix<double, N, 1> class_prior_mean(const State& seed, const Frame&) const
    {
        Eigen::Matrix<double, N, 1> x = seed.vec().cast<double>();
        x.segment<3>(2) = prior_mean.segment<3>(2).cast<double>();
        return x;
    }
    const State& state() const { return s; }
    MatN covariance() const { return Sigma; }
    void set_state(const State& st) { s = st; }
    void set_covariance(const MatN& S) { Sigma = S; }
};

inline ToyBox::Frame make_frame(std::vector<Eigen::Vector3f> points, rc::ai::beam::BeamRays rays)
{
    ToyBox::Frame f;
    f.points = std::move(points);
    f.beam = std::move(rays);
    return f;
}

// Points on the four sides and the top of a box state (the floor face is never seen), `per_edge` per face side.
inline std::vector<Eigen::Vector3f> surface_points(const ToyState& t, int per_edge)
{
    std::vector<Eigen::Vector3f> pts;
    const float c = std::cos(t.yaw), sn = std::sin(t.yaw);
    const auto world = [&](float lx, float ly, float z)
    { return Eigen::Vector3f(t.cx + c * lx - sn * ly, t.cy + sn * lx + c * ly, z); };
    const float hw = 0.5f * t.w, hh = 0.5f * t.h;
    for (int i = 0; i < per_edge; ++i)
        for (int j = 0; j < per_edge; ++j)
        {
            const float a = (i + 0.5f) / per_edge, b = (j + 0.5f) / per_edge;   // interior of each face
            const float z = b * t.H;
            pts.push_back(world(-hw + a * t.w,  hh, z));
            pts.push_back(world(-hw + a * t.w, -hh, z));
            pts.push_back(world( hw, -hh + a * t.h, z));
            pts.push_back(world(-hw, -hh + a * t.h, z));
            pts.push_back(world(-hw + a * t.w, -hh + b * t.h, t.H));
        }
    return pts;
}

}  // namespace toy
