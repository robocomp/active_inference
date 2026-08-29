#include "room_gn_solver.h"

#include <algorithm>
#include <cmath>

#include "room_obs_weights.h"
#include "image_edge_accumulate.h"
#include "image_edge_ops.h"

namespace rc::gn
{
    namespace
    {
        inline float wrap_pi(float a)
        {
            return std::atan2(std::sin(a), std::cos(a));
        }

        /// d/dθ R(θ) = J·R(θ) with J = [[0,-1],[1,0]]. Used by every geometric factor here.
        inline Eigen::Matrix2f Jrot()
        {
            Eigen::Matrix2f J;
            J << 0.f, -1.f,
                 1.f,  0.f;
            return J;
        }

        // =====================================================================================
        //  SDF observation factor — one window slot's lidar scan against the room boundary.
        //
        //  loss = 0.5·σ_obs⁻²·mean_i( w_i · huber_δ(d_i) )          (identical to compute_rfe_loss)
        //
        //  Linearized as IRLS: huber_δ(d) = 0.5·u·d² with u = min(1, δ/|d|). Writing the loss as
        //  0.5·Σ a_i d_i² then requires a_i = 0.5·σ_obs⁻²·w_i·u_i/N — BOTH halves matter, the 0.5 in
        //  front of the loss and the 0.5 inside the Huber. (Getting this wrong scales H and b
        //  together, so the step is unchanged when SDF is the ONLY factor and the solver still looks
        //  like it converges; what it silently changes is the SDF term's weight against motion,
        //  corner, anchor and boundary — i.e. it moves the minimum. gn_selftest catches it.)
        //  Normal equations: H += Σ a_i JᵀJ, b += Σ a_i Jᵀd_i.
        //
        //  Row of J for point p (robot frame), g = ∇SDF at its room-frame image, q = R(θ)p:
        //      ∂d/∂(x,y) = gᵀ ,   ∂d/∂θ = gᵀ·(J·q) = -g_x·q_y + g_y·q_x
        // =====================================================================================
        class SdfFactor final : public IFactor
        {
        public:
            SdfFactor(const Input& in, int offset, const RoomConcept::WindowSlot& slot)
                : in_(in), off_(offset), points_(slot.lidar_points)
            {
                // Robot-frame xy pulled to CPU once per solve, not once per iteration.
                const auto cpu = points_.detach().to(torch::kCPU).contiguous();
                const auto n = static_cast<int>(cpu.size(0));
                const auto acc = cpu.accessor<float, 2>();
                pts_.resize(n, 2);
                for (int i = 0; i < n; ++i)
                {
                    pts_(i, 0) = acc[i][0];
                    pts_(i, 1) = acc[i][1];
                }
            }

            float evaluate(const State& x) const override
            {
                torch::NoGradGuard ng;
                const auto q = query(x, /*want_grad=*/false);
                if (not q.sdf.defined() or q.sdf.size(0) == 0) return 0.f;
                return compute_observation_loss_from_query(*in_.model, *in_.params, points_,
                                                           theta_tensor(x), q).item<float>();
            }

            float linearize(const State& x, LinearSystem& sys) const override
            {
                torch::NoGradGuard ng;
                const auto q = query(x, /*want_grad=*/true);
                if (not q.sdf.defined() or q.sdf.size(0) == 0) return 0.f;

                const auto w = build_observation_weights(*in_.model, *in_.params, points_,
                                                         theta_tensor(x), q);
                const auto sdf_cpu  = q.sdf.detach().to(torch::kCPU).contiguous();
                const auto grad_cpu = q.grad.detach().to(torch::kCPU).contiguous();
                const auto w_cpu    = w.detach().to(torch::kCPU).contiguous();

                const int n = static_cast<int>(sdf_cpu.size(0));
                const auto d_acc = sdf_cpu.accessor<float, 1>();
                const auto g_acc = grad_cpu.accessor<float, 2>();
                const auto w_acc = w_cpu.accessor<float, 1>();

                const auto& P = *in_.params;
                const float inv_var = 1.0f / (P.rfe_obs_sigma * P.rfe_obs_sigma);
                const float delta = P.rfe_huber_delta;
                const float theta = x(off_ + 2);
                const float c = std::cos(theta), s = std::sin(theta);

                Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
                Eigen::Vector3f b = Eigen::Vector3f::Zero();
                for (int i = 0; i < n; ++i)
                {
                    const float d = d_acc[i];
                    if (not std::isfinite(d)) continue;
                    const float ad = std::abs(d);
                    const float u = (ad <= delta or ad < 1e-9f) ? 1.0f : delta / ad;   // IRLS Huber
                    const float a = 0.5f * inv_var * w_acc[i] * u / static_cast<float>(n);

                    // q = R(θ)·p, so ∂d/∂θ = gᵀ(J·q) = -g_x·q_y + g_y·q_x
                    const float px = pts_(i, 0), py = pts_(i, 1);
                    const float qx = c * px - s * py;
                    const float qy = s * px + c * py;
                    const float gx = g_acc[i][0], gy = g_acc[i][1];

                    Eigen::Vector3f J;
                    J << gx, gy, -gx * qy + gy * qx;
                    H.noalias() += a * J * J.transpose();
                    b.noalias() += (a * d) * J;
                }

                sys.add_H(off_, off_, Eigen::MatrixXf(H));
                sys.add_b(off_, Eigen::VectorXf(b));
                return compute_observation_loss_from_query(*in_.model, *in_.params, points_,
                                                           theta_tensor(x), q).item<float>();
            }

        private:
            Model::SdfQueryResult query(const State& x, bool want_grad) const
            {
                const int o = off_;
                const auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(in_.device);
                const auto xy = torch::tensor({x(o), x(o + 1)}, opt);
                const auto th = torch::tensor({x(o + 2)}, opt);
                return in_.model->sdf_query_at_pose(points_, xy, th, want_grad);
            }
            torch::Tensor theta_tensor(const State& x) const
            {
                return torch::tensor({x(off_ + 2)},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(in_.device));
            }

            const Input& in_;
            int off_ = 0;
            torch::Tensor points_;
            Eigen::MatrixX2f pts_;
        };

        // =====================================================================================
        //  ImageEdgeFactor — RGB structural contours against the image gradient.
        //
        //  r_k = n_hat^T ( pi(p_cam,k(x)) - uv_meas,k )        [pixels, SIGNED, 1-D]
        //
        //  1-D on purpose: only the component along the contour NORMAL is observable (aperture
        //  problem). A 2-D residual would fabricate a tangential constraint the image does not carry.
        //
        //  The chain is  p_robot = R(-th)(p_room - t) -> p_cam = C p_robot + d -> (u,v) = pi(p_cam),
        //  where C,d are the STATIC camera<-robot extrinsic. room<-robot is x, the VARIABLE.
        //  ★ Nothing here may read room<-robot from the graph: that is this agent's own output, and
        //    a factor built on it has r == 0 by construction, carrying zero information while every
        //    diagnostic looks perfect. See image_edge_source.h.
        //
        //  All the accumulation, including the shared-nuisance Woodbury cap that stops N correlated
        //  samples out-voting the LiDAR, lives in image_edge_accumulate.h so this backend and the
        //  torch mirror cannot drift apart.
        // =====================================================================================
        class ImageEdgeFactorGn final : public IFactor
        {
        public:
            ImageEdgeFactorGn(int offset, const ::rc::ImageEdgeObs& obs)
                : off_(offset), obs_(obs), cam_(obs.cam) {}

            float evaluate(const State& x) const override { return accumulate(x, nullptr); }

            float linearize(const State& x, LinearSystem& sys) const override
            {
                Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
                Eigen::Vector3f b = Eigen::Vector3f::Zero();
                const float loss = accumulate(x, &H, &b);
                sys.add_H(off_, off_, Eigen::MatrixXf(H));
                sys.add_b(off_, Eigen::VectorXf(b));
                return loss;
            }

        private:
            /// Shared by evaluate() and linearize() so they cannot disagree — the loss returned is
            /// exactly the one whose normal equations are accumulated (IFactor's contract).
            float accumulate(const State& x, Eigen::Matrix3f* Hout, Eigen::Vector3f* bout = nullptr) const
            {
                const Eigen::Vector3f pose(x(off_), x(off_ + 1), x(off_ + 2));
                const float cth = std::cos(pose.z()), sth = std::sin(pose.z());
                Eigen::Matrix3f Rm;                     // R(-theta)
                Rm <<  cth,  sth, 0.f,
                      -sth,  cth, 0.f,
                       0.f,  0.f, 1.f;

                float loss = 0.f;
                for (const auto& seg : obs_.segments)
                {
                    const auto acc = ::rc::img::accumulate_segment(seg,
                        [&](std::size_t k, Eigen::Matrix<float, 1, 3>& J) -> float
                        {
                            const auto& smp = seg.samples[k];
                            const Eigen::Vector3f e(smp.p_room.x() - pose.x(),
                                                    smp.p_room.y() - pose.y(), smp.p_room.z());
                            const Eigen::Vector3f p_robot = Rm * e;
                            const Eigen::Vector3f p_cam   = obs_.cam_R_robot * p_robot + obs_.cam_t_robot;

                            Eigen::Vector2d uv;
                            if (not ::rc::img::project_with_model(cam_, p_cam.cast<double>(), uv))
                                return std::numeric_limits<float>::quiet_NaN();

                            double du = uv.x() - static_cast<double>(smp.uv_meas.x());
                            if (cam_.kind != ::rc::CameraModel::Kind::Pinhole)
                            {   // the panorama wraps; fold before differencing
                                while (du >  0.5 * cam_.width) du -= cam_.width;
                                while (du <= -0.5 * cam_.width) du += cam_.width;
                            }
                            const double dv = uv.y() - static_cast<double>(smp.uv_meas.y());

                            Eigen::Matrix<double, 2, 3> P;
                            if (not ::rc::img::project_jacobian_model(cam_, p_cam.cast<double>(), P))
                                return std::numeric_limits<float>::quiet_NaN();

                            Eigen::Matrix3f Jx;
                            Jx.col(0) = obs_.cam_R_robot * (-Rm.col(0));
                            Jx.col(1) = obs_.cam_R_robot * (-Rm.col(1));
                            // d/dtheta of R(-theta)e = (+p_robot.y, -p_robot.x, 0). Verified
                            // symbolically; the opposite sign converges just as prettily on the
                            // wrong answer, which is why it is spelled out here.
                            Jx.col(2) = obs_.cam_R_robot *
                                        Eigen::Vector3f(p_robot.y(), -p_robot.x(), 0.f);

                            J = smp.n_hat.transpose() * (P.cast<float>() * Jx);
                            return smp.n_hat.x() * static_cast<float>(du)
                                 + smp.n_hat.y() * static_cast<float>(dv);
                        },
                        [](std::size_t) { return 0.0f; });

                    loss += acc.loss;
                    if (Hout) { *Hout += acc.H; if (bout) *bout += acc.b; }
                }
                return loss;
            }

            int off_ = 0;
            const ::rc::ImageEdgeObs& obs_;
            ::rc::CameraModel         cam_;
        };

        // =====================================================================================
        //  Motion factor between consecutive slots: r = wrap(x_b − x_a − Δodom), loss = 0.5 rᵀΛr.
        //  Exactly linear in the state (∂r/∂x_b = I, ∂r/∂x_a = −I); only the yaw wrap is nonlinear,
        //  and it is handled in the residual.
        // =====================================================================================
        class MotionFactor final : public IFactor
        {
        public:
            MotionFactor(int off_a, int off_b, const Eigen::Vector3f& odom, const Eigen::Matrix3f& prec)
                : oa_(off_a), ob_(off_b), odom_(odom), prec_(prec) {}

            float evaluate(const State& x) const override
            {
                const Eigen::Vector3f r = residual(x);
                return 0.5f * r.dot(prec_ * r);
            }

            float linearize(const State& x, LinearSystem& sys) const override
            {
                const Eigen::Vector3f r = residual(x);
                const Eigen::MatrixXf L = prec_;
                sys.add_H(ob_, ob_,  L);
                sys.add_H(oa_, oa_,  L);
                sys.add_H(oa_, ob_, -L);
                sys.add_H(ob_, oa_, -L);
                const Eigen::Vector3f Lr = L * r;
                sys.add_b(ob_, Eigen::VectorXf( Lr));
                sys.add_b(oa_, Eigen::VectorXf(-Lr));
                return 0.5f * r.dot(Lr);
            }

        private:
            Eigen::Vector3f residual(const State& x) const
            {
                Eigen::Vector3f r = x.segment<3>(ob_) - x.segment<3>(oa_) - odom_;
                r(2) = wrap_pi(r(2));
                return r;
            }
            int oa_, ob_;
            Eigen::Vector3f odom_;
            Eigen::Matrix3f prec_;
        };

        // =====================================================================================
        //  Boundary prior on the oldest surviving slot (FEJ+Schur marginal, or the legacy anchor):
        //  loss = w·0.5·Δᵀ Λ Δ with Δ = wrap(x₀ − μ). μ is frozen, so this is a plain quadratic.
        // =====================================================================================
        class BoundaryFactor final : public IFactor
        {
        public:
            BoundaryFactor(int off, const Eigen::Vector3f& mu, const Eigen::Matrix3f& prec, float weight)
                : o_(off), mu_(mu), prec_(prec * weight) {}

            float evaluate(const State& x) const override
            {
                const Eigen::Vector3f d = delta(x);
                return 0.5f * d.dot(prec_ * d);
            }

            float linearize(const State& x, LinearSystem& sys) const override
            {
                const Eigen::Vector3f d = delta(x);
                sys.add_H(o_, o_, Eigen::MatrixXf(prec_));
                const Eigen::Vector3f Ld = prec_ * d;
                sys.add_b(o_, Eigen::VectorXf(Ld));
                return 0.5f * d.dot(Ld);
            }

        private:
            Eigen::Vector3f delta(const State& x) const
            {
                Eigen::Vector3f d = x.segment<3>(o_) - mu_;
                d(2) = wrap_pi(d(2));
                return d;
            }
            int o_;
            Eigen::Vector3f mu_;
            Eigen::Matrix3f prec_;
        };

        // =====================================================================================
        //  SE(2) landmark factor — ONE type serving both corners and object anchors, and the seat
        //  reserved for the landmarks the localizer acquires from stabilised graph objects.
        //
        //      z_hat_xy  = R(-θ)·(p_w − t),      z_hat_yaw = yaw_w − θ
        //      r         = z_obs ⊖ z_hat        (yaw wrapped)
        //      m²        = gain·rᵀΛr,  u = min(1, δ/m),  loss = scale·0.5·u·m²
        //
        //  Jacobian wrt the observing pose (Rn = R(-θ), J = [[0,-1],[1,0]], dw = p_w − t):
        //      ∂r_xy/∂(x,y) = Rn ,  ∂r_xy/∂θ = Rn·J·dw ,  ∂r_yaw/∂θ = 1
        //
        //  A corner is this with the yaw channel off (Λ is 2×2 there); an anchor with
        //  has_orientation=false arrives with its yaw row/col of Λ already zeroed, which drops the
        //  yaw term exactly as the torch factor does. When landmarks become VARIABLES, this class
        //  gains a second offset and fills the off-diagonal block — nothing else here changes.
        // =====================================================================================
        class Se2LandmarkFactor final : public IFactor
        {
        public:
            /// lm_off < 0 ⇒ the landmark is a CONSTANT at lm_world (pinned/corner case). Otherwise it is a
            /// 2-DOF variable at that offset and lm_world is only its current estimate; the factor then
            /// also fills the landmark block and the pose↔landmark cross terms.
            Se2LandmarkFactor(int off, const Eigen::Vector3f& lm_world, const Eigen::Vector3f& obs,
                              const Eigen::Matrix3f& information, bool use_yaw,
                              float gain, float scale, float huber_delta, float m2_eps,
                              int lm_off = -1)
                : o_(off), lm_off_(lm_off), lm_(lm_world), obs_(obs), lam_(information), use_yaw_(use_yaw),
                  gain_(gain), scale_(scale), huber_(huber_delta), eps_(m2_eps) {}

            float evaluate(const State& x) const override
            {
                const Eigen::Vector3f r = residual(x);
                const float m2 = gain_ * r.dot(lam_ * r);
                const float m  = std::sqrt(m2 + eps_);
                const float u  = (m <= huber_) ? 1.0f : huber_ / m;
                return scale_ * 0.5f * u * m2;
            }

            float linearize(const State& x, LinearSystem& sys) const override
            {
                const Eigen::Vector3f r = residual(x);
                const float m2 = gain_ * r.dot(lam_ * r);
                const float m  = std::sqrt(m2 + eps_);
                const float u  = (m <= huber_) ? 1.0f : huber_ / m;   // LOSS weight (mirrors torch)

                // IRLS weight for H and b — NOT the same as the loss weight u once the robust kernel
                // saturates. Writing the loss as L(s) with s = m², its exact gradient is 2ρ'(s)·gain·JᵀΛr:
                //   m ≤ δ :  L = 0.5·s          → 2ρ' = 1     = u
                //   m > δ :  L = 0.5·δ·√s       → 2ρ' = 0.5·δ/m = u/2      ← the loss is linear in m there
                // Using u in the saturated branch doubles this factor's pull, so the assembled b is zero
                // at a point that is NOT the minimum: LM converges (it accepts on the TRUE loss) but stops
                // somewhere else than Adam/L-BFGS. Found in the live shadow log (grad_relerr median 0.28,
                // 49/55 frames), reproduced offline only after gn_selftest learned to corrupt the
                // observations — consistent landmarks never leave the quadratic branch.
                const float w_irls = (m <= huber_) ? 1.0f : 0.5f * huber_ / m;

                const float th = x(o_ + 2);
                const float c = std::cos(th), s = std::sin(th);
                Eigen::Matrix2f Rn;                       // R(-θ)
                Rn << c, s,
                     -s, c;
                const Eigen::Vector2f dw = lm_xy(x) - x.segment<2>(o_);
                const Eigen::Vector2f dth = Rn * (Jrot() * dw);

                Eigen::Matrix3f J = Eigen::Matrix3f::Zero();
                J.block<2, 2>(0, 0) = Rn;
                J.block<2, 1>(0, 2) = dth;
                if (use_yaw_) J(2, 2) = 1.0f;

                // Effective information: the robust weight and both scalings fold into Λ.
                const Eigen::Matrix3f W = (scale_ * gain_ * w_irls) * lam_;
                sys.add_H(o_, o_, Eigen::MatrixXf(J.transpose() * W * J));
                sys.add_b(o_, Eigen::VectorXf(J.transpose() * (W * r)));

                if (lm_off_ >= 0)
                {
                    // ∂r_xy/∂p_lm = −Rn (the landmark enters the residual only through dw = p_lm − t,
                    // with the opposite sign to the robot translation). Yaw channel is unaffected: a
                    // position-only landmark has no yaw, and even with yaw the landmark's is not a
                    // variable here.
                    Eigen::Matrix<float, 3, 2> Jl = Eigen::Matrix<float, 3, 2>::Zero();
                    Jl.block<2, 2>(0, 0) = -Rn;
                    sys.add_H(lm_off_, lm_off_, Eigen::MatrixXf(Jl.transpose() * W * Jl));
                    sys.add_H(o_, lm_off_, Eigen::MatrixXf(J.transpose() * W * Jl));
                    sys.add_H(lm_off_, o_, Eigen::MatrixXf(Jl.transpose() * W * J));
                    sys.add_b(lm_off_, Eigen::VectorXf(Jl.transpose() * (W * r)));
                }
                return scale_ * 0.5f * u * m2;
            }

        private:
            Eigen::Vector2f lm_xy(const State& x) const
            {
                return (lm_off_ >= 0) ? Eigen::Vector2f(x.segment<2>(lm_off_)) : Eigen::Vector2f(lm_.head<2>());
            }

            Eigen::Vector3f residual(const State& x) const
            {
                const float th = x(o_ + 2);
                const float c = std::cos(th), s = std::sin(th);
                const Eigen::Vector2f dw = lm_xy(x) - x.segment<2>(o_);
                Eigen::Vector3f r = Eigen::Vector3f::Zero();
                r(0) = obs_(0) - ( c * dw.x() + s * dw.y());
                r(1) = obs_(1) - (-s * dw.x() + c * dw.y());
                if (use_yaw_) r(2) = wrap_pi(obs_(2) - (lm_(2) - th));
                return r;
            }
            int o_;
            int lm_off_ = -1;
            Eigen::Vector3f lm_, obs_;
            Eigen::Matrix3f lam_;
            bool use_yaw_;
            float gain_, scale_, huber_, eps_;
        };

        // =====================================================================================
        //  Landmark birth prior: 0.5·(p − μ)ᵀ Λ_prior (p − μ), μ/Λ from the producing agent's belief.
        //  Applied ONLY on the frame the landmark is born. Σ_o was itself derived through the robot
        //  pose, so re-applying it every frame would keep feeding the same evidence back into the
        //  estimate and shrink both sides' covariance without any new information entering.
        // =====================================================================================
        class LandmarkPriorFactor final : public IFactor
        {
        public:
            LandmarkPriorFactor(int off, const Eigen::Vector2f& mu, const Eigen::Matrix2f& prec)
                : o_(off), mu_(mu), prec_(prec) {}

            float evaluate(const State& x) const override
            {
                const Eigen::Vector2f d = x.segment<2>(o_) - mu_;
                return 0.5f * d.dot(prec_ * d);
            }

            float linearize(const State& x, LinearSystem& sys) const override
            {
                const Eigen::Vector2f d = x.segment<2>(o_) - mu_;
                sys.add_H(o_, o_, Eigen::MatrixXf(prec_));
                const Eigen::Vector2f Ld = prec_ * d;
                sys.add_b(o_, Eigen::VectorXf(Ld));
                return 0.5f * d.dot(Ld);
            }

        private:
            int o_;
            Eigen::Vector2f mu_;
            Eigen::Matrix2f prec_;
        };

        State pack(const std::vector<Eigen::Vector3f>& poses)
        {
            State x(static_cast<int>(poses.size()) * 3);
            for (size_t i = 0; i < poses.size(); ++i) x.segment<3>(static_cast<int>(i) * 3) = poses[i];
            return x;
        }

        void unpack(const State& x, std::vector<Eigen::Vector3f>& poses)
        {
            for (size_t i = 0; i < poses.size(); ++i) poses[i] = x.segment<3>(static_cast<int>(i) * 3);
        }

        float total_loss(const FactorList& fs, const State& x)
        {
            float s = 0.f;
            for (const auto& f : fs) s += f->evaluate(x);
            return s;
        }
    } // namespace

    FactorList build_factors(const Input& in, const VarIndex& idx)
    {
        FactorList fs;
        const auto& W = *in.window;
        const auto& P = *in.params;
        const int n = static_cast<int>(W.size());
        if (n == 0) return fs;

        // --- 1. Boundary prior on the oldest surviving slot (only with W > 1, as in compute_rfe_loss)
        if (in.boundary_prior->valid and n > 1)
            fs.push_back(std::make_unique<BoundaryFactor>(idx.offset(0), in.boundary_prior->mu,
                                                           in.boundary_prior->precision, in.boundary_weight));

        // --- 2. SDF observation factors
        for (int i = 0; i < n; ++i)
        {
            if (P.sdf_current_slot_only and i != n - 1) continue;
            const auto& slot = W[static_cast<size_t>(i)];
            if (not slot.lidar_points.defined() or slot.lidar_points.size(0) == 0) continue;
            fs.push_back(std::make_unique<SdfFactor>(in, idx.offset(i), slot));
        }

        // --- 3. Motion factors between consecutive slots
        for (int i = 1; i < n; ++i)
        {
            const auto& curr = W[static_cast<size_t>(i)];
            Eigen::Matrix3f prec = Eigen::Matrix3f::Identity();
            if (curr.motion_prec_tensor.defined())
            {
                const auto cpu = curr.motion_prec_tensor.detach().to(torch::kCPU).contiguous();
                const auto a = cpu.accessor<float, 2>();
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c) prec(r, c) = a[r][c];
            }
            fs.push_back(std::make_unique<MotionFactor>(idx.offset(i - 1), idx.offset(i),
                                                         curr.odometry_delta, prec));

            // ── The rest hypothesis, as the SAME factor with a ZERO delta ─────────────────────────
            // "You did not move", weighed against "you moved by Delta" by their stated precisions.
            // It needs no factor class of its own: a zero-velocity constraint between two poses IS a
            // between-factor with a zero measurement, which is exactly MotionFactor with odom = 0.
            // Present only when the interval supplied a precision, i.e. only when the feature is on
            // — and its precision already fades with the motion the interval measured, so in motion
            // it contributes nothing without anything having to switch it off.
            if (curr.zupt_prec_tensor.defined())
            {
                Eigen::Matrix3f zprec = Eigen::Matrix3f::Zero();
                const auto zcpu = curr.zupt_prec_tensor.detach().to(torch::kCPU).contiguous();
                const auto za = zcpu.accessor<float, 2>();
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c) zprec(r, c) = za[r][c];
                fs.push_back(std::make_unique<MotionFactor>(idx.offset(i - 1), idx.offset(i),
                                                             Eigen::Vector3f::Zero(), zprec));
            }
        }

        // --- 4. Corner landmarks (newest corner_max_slots slots), yaw channel off
        if (P.enable_corner_tracking)
        {
            const int start = std::max(0, n - P.corner_max_slots);
            for (int i = start; i < n; ++i)
                for (const auto& obs : W[static_cast<size_t>(i)].corner_obs)
                {
                    Eigen::Matrix3f lam = Eigen::Matrix3f::Zero();
                    lam.block<2, 2>(0, 0) = obs.information;
                    fs.push_back(std::make_unique<Se2LandmarkFactor>(
                        idx.offset(i),
                        Eigen::Vector3f(obs.model_corner_world.x(), obs.model_corner_world.y(), 0.f),
                        Eigen::Vector3f(obs.detected_robot.x(), obs.detected_robot.y(), 0.f),
                        lam, /*use_yaw=*/false,
                        P.corner_precision_gain, /*scale=*/1.0f, P.corner_huber_sigma, 1e-8f));
                }
        }

        // --- 5. Object anchors (stabilised graph objects) — the dynamic set. Rebuilt every solve,
        //        so births and retirements between frames need no special handling here.
        if (P.object_anchor.enable)
        {
            const int start = std::max(0, n - P.object_anchor_max_slots);
            for (int i = start; i < n; ++i)
                for (const auto& a : W[static_cast<size_t>(i)].object_anchors)
                {
                    // Is this anchor's object a landmark we are OPTIMISING? If so the factor gets its
                    // variable block, and the observation is weighted by R_o⁻¹ ALONE — the map's
                    // uncertainty now lives in the landmark's own state, so keeping Σ_o folded into the
                    // measurement weight (as ObjectAnchorObs::information does) would count it twice.
                    int lm_off = -1;
                    Eigen::Vector3f lm_pos = a.pose_world;
                    Eigen::Matrix3f w_info = a.information;
                    if (in.landmarks != nullptr)
                        for (size_t k = 0; k < in.landmarks->size(); ++k)
                            if ((*in.landmarks)[k].id == a.node_id)
                            {
                                lm_off = idx.offset(n + static_cast<int>(k));
                                lm_pos.head<2>() = (*in.landmarks)[k].p;
                                w_info = a.meas_information;
                                break;
                            }
                    fs.push_back(std::make_unique<Se2LandmarkFactor>(
                        idx.offset(i), lm_pos, a.obs_robot, w_info,
                        /*use_yaw=*/a.has_orientation,
                        /*gain=*/1.0f, P.object_anchor.weight, P.object_anchor.huber_delta, 1e-9f,
                        lm_off));
                }
        }

        // --- 6. Landmark birth priors (see Landmark::has_prior — birth only, never per-frame).
        if (in.landmarks != nullptr)
            for (size_t k = 0; k < in.landmarks->size(); ++k)
            {
                const auto& lm = (*in.landmarks)[k];
                if (not lm.has_prior) continue;
                fs.push_back(std::make_unique<LandmarkPriorFactor>(
                    idx.offset(n + static_cast<int>(k)), lm.prior_mean, lm.prior_information));
            }

        // --- 7. RGB structural-contour factors (OFF unless ImageEdge.enable AND .drive)
        //     Newest slots only (image_edge_max_slots), exactly like corners: the evidence is one
        //     frame's normal search, and older slots' searches were done at poses the window has
        //     since moved away from.
        if (P.image_edge.enable and P.image_edge.drive)
        {
            const int first = std::max(0, n - std::max(1, P.image_edge_max_slots));
            for (int i = first; i < n; ++i)
            {
                const auto& slot = W[static_cast<size_t>(i)];
                if (slot.image_edges.empty() or not slot.image_edges.cam.valid) continue;
                fs.push_back(std::make_unique<ImageEdgeFactorGn>(idx.offset(i), slot.image_edges));
            }
        }

        return fs;
    }

    float evaluate(const Input& in, const std::vector<Eigen::Vector3f>& poses)
    {
        if (in.window == nullptr or in.window->empty()) return std::numeric_limits<float>::quiet_NaN();
        VarIndex idx;
        for (size_t i = 0; i < in.window->size(); ++i) idx.add(3);
        const auto fs = build_factors(in, idx);
        return total_loss(fs, pack(poses));
    }

    float gradient_check(const Input& in, const std::vector<Eigen::Vector3f>& poses, float eps)
    {
        if (in.window == nullptr or in.window->empty()) return std::numeric_limits<float>::quiet_NaN();
        VarIndex idx;
        for (size_t i = 0; i < in.window->size(); ++i) idx.add(3);
        const int n = idx.total();
        const auto fs = build_factors(in, idx);
        if (fs.empty()) return std::numeric_limits<float>::quiet_NaN();

        const State x = pack(poses);
        LinearSystem sys(n);
        for (const auto& f : fs) f->linearize(x, sys);

        State num(n);
        for (int i = 0; i < n; ++i)
        {
            State xp = x, xm = x;
            xp(i) += eps;
            xm(i) -= eps;
            num(i) = (total_loss(fs, xp) - total_loss(fs, xm)) / (2.f * eps);
        }
        // Scored against the VECTOR magnitude, not per-component: where the true gradient is ~0 the
        // central difference is float noise (loss·ulp/2ε), and dividing by that noise would report a
        // 100% error on a perfectly correct Jacobian.
        const float scale = std::max({num.lpNorm<Eigen::Infinity>(),
                                      sys.b.lpNorm<Eigen::Infinity>(), 1e-6f});
        return (num - sys.b).lpNorm<Eigen::Infinity>() / scale;
    }

    NewestMarginal newest_pose_marginal(const Input& in, const std::vector<Eigen::Vector3f>& poses)
    {
        NewestMarginal out;
        if (in.model == nullptr or in.params == nullptr or in.window == nullptr or
            in.window->empty() or poses.size() != in.window->size())
            return out;

        // Same variable layout as solve(), or the blocks below would address the wrong rows.
        VarIndex idx;
        for (size_t i = 0; i < in.window->size(); ++i) idx.add(3);
        const size_t n_lm = (in.landmarks != nullptr) ? in.landmarks->size() : 0;
        for (size_t k = 0; k < n_lm; ++k) idx.add(2);
        const int n = idx.total();

        const auto fs = build_factors(in, idx);
        if (fs.empty()) return out;

        State x = State::Zero(n);
        x.head(static_cast<int>(in.window->size()) * 3) = pack(poses);
        for (size_t k = 0; k < n_lm; ++k)
            x.segment<2>(idx.offset(static_cast<int>(in.window->size() + k))) = (*in.landmarks)[k].p;

        LinearSystem sys(n);
        for (const auto& f : fs) f->linearize(x, sys);
        if (not sys.H.allFinite()) return out;

        const int nb = idx.offset(static_cast<int>(in.window->size()) - 1);   // newest pose's first row
        out.block = sys.H.block<3, 3>(nb, nb);
        out.n_marginalised = n - 3;
        if (out.n_marginalised <= 0)          // a one-slot window has nothing to fold away
        {
            out.marginal = out.block;
            out.ok = out.marginal.allFinite();
            return out;
        }

        // Gather the OTHER variables into one contiguous block. The newest pose is the last POSE but
        // not the last variable when landmarks are present, so this cannot be a simple corner slice.
        Eigen::VectorXi keep(out.n_marginalised);
        for (int i = 0, k = 0; i < n; ++i)
            if (i < nb or i >= nb + 3) keep(k++) = i;
        Eigen::MatrixXf Hoo(out.n_marginalised, out.n_marginalised);
        Eigen::MatrixXf Hno(3, out.n_marginalised);
        for (int r = 0; r < out.n_marginalised; ++r)
        {
            for (int c = 0; c < out.n_marginalised; ++c) Hoo(r, c) = sys.H(keep(r), keep(c));
            for (int c = 0; c < 3; ++c)                  Hno(c, r) = sys.H(nb + c, keep(r));
        }
        // A ridge, not a threshold: Hoo is singular whenever a window variable is unobserved this
        // frame (a landmark nobody saw), and the complement is then simply undefined for that
        // direction. The ridge makes it "known to 1e6 sigma" instead of "known exactly", which errs
        // toward the block — i.e. toward UNDERSTATING the correction this instrument exists to show.
        Hoo.diagonal().array() += 1e-6f;
        const Eigen::LDLT<Eigen::MatrixXf> ldlt(Hoo);
        if (ldlt.info() != Eigen::Success) return out;
        out.marginal = out.block - Hno * ldlt.solve(Hno.transpose());
        out.ok = out.marginal.allFinite();
        return out;
    }

    Result solve(const Input& in, std::vector<Eigen::Vector3f>& poses, const Options& opts)
    {
        Result res;
        if (in.model == nullptr or in.params == nullptr or in.window == nullptr or
            in.window->empty() or poses.size() != in.window->size())
            return res;

        VarIndex idx;
        for (size_t i = 0; i < in.window->size(); ++i) idx.add(3);
        const int n_pose = idx.total();
        const size_t n_lm = (in.landmarks != nullptr) ? in.landmarks->size() : 0;
        for (size_t k = 0; k < n_lm; ++k) idx.add(2);       // landmarks are 2-DOF (position-only)
        const int n = idx.total();

        const auto fs = build_factors(in, idx);
        if (fs.empty()) return res;

        State x = State::Zero(n);
        x.head(n_pose) = pack(poses);
        for (size_t k = 0; k < n_lm; ++k)
            x.segment<2>(idx.offset(static_cast<int>(in.window->size() + k))) = (*in.landmarks)[k].p;
        float loss = total_loss(fs, x);
        res.loss_init = loss;
        if (not std::isfinite(loss)) return res;

        float lambda = opts.lambda_init;
        // A SINGLE slow step is not convergence. Levenberg's damping shrinks after every accepted step,
        // so an early step can crawl and the next one leap; stopping on the first small relative
        // improvement left the solver short of the minimum on 11 of the 12 live shadow frames where it
        // ended above L-BFGS (grad_norm still 4.6 at the stop, up to 3% worse on the objective, 1.8 cm
        // of pose). Require the improvement to stall TWICE in a row before believing it.
        int stagnant = 0;
        float nu = 2.0f;   // Nielsen's rejection multiplier, doubling each consecutive rejection
        for (int it = 0; it < opts.max_iters; ++it)
        {
            LinearSystem sys(n);
            float lin_loss = 0.f;
            for (const auto& f : fs) lin_loss += f->linearize(x, sys);
            res.iterations = it + 1;
            res.grad_norm = sys.b.lpNorm<Eigen::Infinity>();
            if (not sys.H.allFinite() or not sys.b.allFinite()) break;
            if (res.grad_norm < 1e-9f) break;

            // Marquardt damping: scale by the diagonal so x, y and θ are damped in their own units.
            Eigen::VectorXf diag(n);
            for (int i = 0; i < n; ++i) diag(i) = std::max(sys.H(i, i), 1e-9f);
            Eigen::MatrixXf A = sys.H;
            for (int i = 0; i < n; ++i) A(i, i) += lambda * diag(i);

            const Eigen::LDLT<Eigen::MatrixXf> ldlt(A);
            const Eigen::VectorXf step = ldlt.solve(-sys.b);
            if (ldlt.info() != Eigen::Success or not step.allFinite())
            {
                ++res.rejected;
                lambda *= nu; nu *= 2.0f;
                if (lambda > opts.lambda_max) break;
                continue;
            }

            State x_try = x + step;
            for (size_t k = 0; k < in.window->size(); ++k)          // yaw wrap: pose blocks only
                x_try(idx.offset(static_cast<int>(k)) + 2) = wrap_pi(x_try(idx.offset(static_cast<int>(k)) + 2));
            const float loss_try = total_loss(fs, x_try);

            // Gain ratio: how much of the reduction the LOCAL MODEL promised did we actually get.
            // predicted = 0.5·δᵀ(λ·D·δ − b), the decrease of the damped quadratic along the step.
            const float predicted =
                0.5f * step.dot((lambda * diag.asDiagonal() * step).eval() - sys.b);
            const float rho = (predicted > 0.f) ? (loss - loss_try) / predicted : -1.f;

            if (std::isfinite(loss_try) and loss_try < loss)
            {
                const float rel = (loss - loss_try) / std::max(std::abs(loss), 1e-12f);
                x = std::move(x_try);
                res.step_norm = step.lpNorm<Eigen::Infinity>();
                loss = loss_try;
                // Nielsen's update: shrink λ by how well the model predicted the step, instead of by a
                // fixed factor. A fixed 0.33 overshoots into a rejection, which multiplies λ back up by
                // 5 — the thrash that made 57% of all steps rejections in the live shadow log (and, at
                // r=+0.79 against ms_gn, most of the runtime).
                const float r3 = 2.0f * std::clamp(rho, 0.f, 1.f) - 1.0f;
                lambda = std::max(lambda * std::max(1.0f / 3.0f, 1.0f - r3 * r3 * r3), 1e-9f);
                nu = 2.0f;
                if (res.step_norm < opts.step_tol) break;
                stagnant = (rel < opts.loss_rel_tol) ? stagnant + 1 : 0;
                if (stagnant >= 2) break;
            }
            else
            {
                ++res.rejected;
                lambda *= nu; nu *= 2.0f;      // geometric back-off, so a bad region is escaped fast
                if (lambda > opts.lambda_max) break;
            }
        }

        if (not std::isfinite(loss) or not x.allFinite()) return res;

        for (size_t i = 0; i < poses.size(); ++i)
            poses[i] = x.segment<3>(idx.offset(static_cast<int>(i)));

        if (n_lm > 0)
        {
            // Recover each landmark's marginal precision from the final H so the estimate carries its
            // own uncertainty forward. The block diagonal of H is the CONDITIONAL precision; the marginal
            // needs the Schur complement against everything else, which for a handful of variables is
            // cheap enough to take from the inverse directly.
            LinearSystem fin(n);
            for (const auto& f : fs) f->linearize(x, fin);
            Eigen::MatrixXf cov;
            const Eigen::LDLT<Eigen::MatrixXf> ldlt(fin.H);
            const bool invertible = (ldlt.info() == Eigen::Success);
            if (invertible) cov = ldlt.solve(Eigen::MatrixXf::Identity(n, n));

            for (size_t k = 0; k < n_lm; ++k)
            {
                const int o = idx.offset(static_cast<int>(in.window->size() + k));
                auto& lm = (*in.landmarks)[k];
                lm.p = x.segment<2>(o);
                if (invertible and cov.allFinite())
                {
                    const Eigen::Matrix2f c = cov.block<2, 2>(o, o);
                    if (c.allFinite() and c(0, 0) > 0.f and c(1, 1) > 0.f)
                        lm.information = c.inverse();
                }
                lm.has_prior = false;   // the birth prior is spent; never re-applied
            }
        }

        res.loss = loss;
        res.lambda_final = lambda;
        res.ok = true;
        return res;
    }
} // namespace rc::gn
