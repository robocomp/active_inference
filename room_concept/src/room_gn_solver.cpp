#include "room_gn_solver.h"

#include <algorithm>
#include <cmath>

#include "room_obs_weights.h"

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
            Se2LandmarkFactor(int off, const Eigen::Vector3f& lm_world, const Eigen::Vector3f& obs,
                              const Eigen::Matrix3f& information, bool use_yaw,
                              float gain, float scale, float huber_delta, float m2_eps)
                : o_(off), lm_(lm_world), obs_(obs), lam_(information), use_yaw_(use_yaw),
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
                const Eigen::Vector2f dw = lm_.head<2>() - x.segment<2>(o_);
                const Eigen::Vector2f dth = Rn * (Jrot() * dw);

                Eigen::Matrix3f J = Eigen::Matrix3f::Zero();
                J.block<2, 2>(0, 0) = Rn;
                J.block<2, 1>(0, 2) = dth;
                if (use_yaw_) J(2, 2) = 1.0f;

                // Effective information: the robust weight and both scalings fold into Λ.
                const Eigen::Matrix3f W = (scale_ * gain_ * w_irls) * lam_;
                sys.add_H(o_, o_, Eigen::MatrixXf(J.transpose() * W * J));
                sys.add_b(o_, Eigen::VectorXf(J.transpose() * (W * r)));
                return scale_ * 0.5f * u * m2;
            }

        private:
            Eigen::Vector3f residual(const State& x) const
            {
                const float th = x(o_ + 2);
                const float c = std::cos(th), s = std::sin(th);
                const Eigen::Vector2f dw = lm_.head<2>() - x.segment<2>(o_);
                Eigen::Vector3f r = Eigen::Vector3f::Zero();
                r(0) = obs_(0) - ( c * dw.x() + s * dw.y());
                r(1) = obs_(1) - (-s * dw.x() + c * dw.y());
                if (use_yaw_) r(2) = wrap_pi(obs_(2) - (lm_(2) - th));
                return r;
            }
            int o_;
            Eigen::Vector3f lm_, obs_;
            Eigen::Matrix3f lam_;
            bool use_yaw_;
            float gain_, scale_, huber_, eps_;
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
                    fs.push_back(std::make_unique<Se2LandmarkFactor>(
                        idx.offset(i), a.pose_world, a.obs_robot, a.information,
                        /*use_yaw=*/a.has_orientation,
                        /*gain=*/1.0f, P.object_anchor.weight, P.object_anchor.huber_delta, 1e-9f));
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

    Result solve(const Input& in, std::vector<Eigen::Vector3f>& poses, const Options& opts)
    {
        Result res;
        if (in.model == nullptr or in.params == nullptr or in.window == nullptr or
            in.window->empty() or poses.size() != in.window->size())
            return res;

        VarIndex idx;
        for (size_t i = 0; i < in.window->size(); ++i) idx.add(3);
        const int n = idx.total();

        const auto fs = build_factors(in, idx);
        if (fs.empty()) return res;

        State x = pack(poses);
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
            for (int k = 0; k < idx.count(); ++k) x_try(idx.offset(k) + 2) = wrap_pi(x_try(idx.offset(k) + 2));
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

        unpack(x, poses);
        res.loss = loss;
        res.lambda_final = lambda;
        res.ok = true;
        return res;
    }
} // namespace rc::gn
