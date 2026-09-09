#pragma once
#include <Eigen/Dense>
#include <map>
#include <algorithm>
namespace rc::mount {
    /// One corner seen by both sensors, with everything needed to form the residual.
    struct PairObs
    {
        int             vertex = -1;
        Eigen::Vector2f uv_image = Eigen::Vector2f::Zero();   ///< the triple point's measurement
        Eigen::Vector2f uv_lidar = Eigen::Vector2f::Zero();   ///< the LiDAR corner through the extrinsic
        Eigen::Vector2f r        = Eigen::Vector2f::Zero();   ///< uv_image - uv_lidar
        Eigen::Matrix2f cov      = Eigen::Matrix2f::Identity();
        Eigen::Matrix<float, 2, 5> J =
            Eigen::Matrix<float, 2, 5>::Zero();  ///< d(uv) / d(nuisance), prior-scaled
        float assoc_prob = 1.f;
        float range_m    = 0.f;
        bool  ok         = false;
    };

    /// ── THE PER-VERTEX OFFSET NUISANCE ──────────────────────────────────────────────────────────
    /// MEASURED 2026-09-02 over 395 171 ricoh pairs: the residual carries a systematic offset that
    /// belongs to the CORNER, not to the mount. Per-vertex means reach −17 px and spread sd 5.3 px
    /// across 23 vertices, while the mount signal being estimated is 1.1 px of yaw. The old model
    ///
    ///     r_i = −J_i θ + ε_i        ε independent
    ///
    /// has nowhere to put that, so it absorbs it into θ AND counts each of a vertex's thousands of
    /// sightings as independent confirmation. The reported yaw sigma came out 127x too small, against
    /// a design effect sqrt(n_rows/n_clusters) of 131 — the two agree, which names the mechanism.
    /// The fitted yaw was then smaller than its own honest error.
    ///
    /// The model gains the term it was missing, and the term is INTEGRATED OUT rather than estimated:
    ///
    ///     r_i = −J_i θ + δ_v(i) + ε_i,      δ_v ~ N(0, S),   S = offset_sigma_px² · I
    ///
    /// δ_v's value is not wanted, only its contamination removed. Marginalising is a Schur complement
    /// over the per-vertex partials below (derivation in VALIDATION_THREE_DEVICE_CORNERS §2 stage 1):
    ///
    ///     M_v = (S⁻¹ + D_v)⁻¹
    ///     H  += A_v − c_v M_v c_vᵀ,   b += b_v − c_v M_v e_v,   rTr += rTr_v − e_vᵀ M_v e_v
    ///
    /// ★ THIS IS A COVARIANCE TERM, NOT A GATE — the Woodbury common-mode marginalisation CLAUDE.md
    ///   already names for correlated mask points, applied one level up. Nothing is rejected and
    ///   nothing is clamped: `offset_sigma_px = 0` recovers the old estimator EXACTLY (that is the
    ///   default, so a running agent is unchanged until someone asks for the new term), and a large
    ///   sigma removes the level entirely.
    /// ★ THE LIMIT IS THE POINT. As n_v grows, M_v → D_v⁻¹ and the vertex contributes only
    ///   A_v − c_v D_v⁻¹ c_vᵀ — its WITHIN-vertex information. Seeing one corner a million times stops
    ///   buying anything. Yaw is a constant pixel shift with no range or bearing dependence, so within
    ///   a vertex it is EXACTLY degenerate with that vertex's own u-offset and loses nearly all of its
    ///   information; pitch and height are range-dependent (Δd = θ_p·d²/h) and survive. Marginalising
    ///   does not RECOVER yaw precision — it reveals we never had it.
    /// ⚠ A detector bias common to EVERY corner is yaw, and nothing here separates them.
    struct VertexBlock
    {
        Eigen::Matrix4d            A = Eigen::Matrix4d::Zero();              ///< Σ w Jᵀ W J
        Eigen::Matrix<double, 4, 2> c = Eigen::Matrix<double, 4, 2>::Zero(); ///< Σ w Jᵀ W
        Eigen::Matrix2d            D = Eigen::Matrix2d::Zero();              ///< Σ w W
        Eigen::Vector4d            b = Eigen::Vector4d::Zero();              ///< Σ w Jᵀ W r
        Eigen::Vector2d            e = Eigen::Vector2d::Zero();              ///< Σ w W r
        double                     rTr = 0.0;
        long                       n = 0;
        [[nodiscard]] bool finite() const
        { return A.allFinite() and c.allFinite() and D.allFinite() and b.allFinite() and e.allFinite(); }
    };

    /// The same 4-parameter normal-equation block stage 1 uses, so the two are directly comparable.
    /// Prior is the IDENTITY because J carries the prior sigma (see room_concept.h).
    struct Accum
    {
        Eigen::Matrix4d H = Eigen::Matrix4d::Zero();
        Eigen::Vector4d b = Eigen::Vector4d::Zero();
        double          rTr = 0.0;
        long            n = 0;
        /// Per-vertex partials, kept ALONGSIDE the aggregate above so the two models can be solved
        /// from one accumulation and compared without re-driving. ~23 entries in this apartment.
        std::map<int, VertexBlock> per_vertex;
        /// Prior sigma on a corner's own image offset, in PIXELS. 0 disables the nuisance and the
        /// solve is bit-for-bit the old one. ⚠ Measured from the same data it would be applied to
        /// (sd 5.3 px) this is an EMPIRICAL-BAYES hyperparameter, not a prior the data then confirms.
        /// ★ Pixels, not metres, on purpose: a 1/range fit does NOT distinguish a pixel-fixed
        ///   detector bias from a metric map offset on this data (both R² ≈ 0), so the pixel choice
        ///   assumes nothing that could not be measured. The metric version IS corner-as-landmark
        ///   refinement (DESIGN §7) and is the next step, not this one.
        double offset_sigma_px = 0.0;
        /// Set when evidence was restored from a file written before the per-vertex partials existed.
        /// Such evidence cannot be marginalised — its rows carry no vertex — and must not be mixed
        /// with evidence that can, so the solve REFUSES the nuisance while it is set.
        bool legacy_unattributed = false;

        void add(const PairObs& o)
        {
            if (not o.ok) return;
            const Eigen::Matrix2d C = o.cov.cast<double>();
            const double det = C.determinant();
            if (not (det > 1e-12)) return;
            const Eigen::Matrix2d W = C.inverse();
            // ★ Weighted by assoc_prob: the detector's own posterior that this detection belongs to
            //   this model corner. A 0.6-probability association contributes 60% of a measurement,
            //   which is what it is — not a threshold, and not a full one either.
            const double w = std::clamp(static_cast<double>(o.assoc_prob), 0.0, 1.0);
            if (not (w > 1e-3)) return;
            // Columns 0-3 only: [4] is a per-contour map offset and a single paired corner carries
            // no information about it that is separable from the mount.
            const Eigen::Matrix<double, 2, 4> J = o.J.template leftCols<4>().template cast<double>();
            const Eigen::Vector2d r = o.r.cast<double>();
            const Eigen::Matrix<double, 4, 2> JtW = w * J.transpose() * W;
            H.noalias() += JtW * J;
            b.noalias() += JtW * r;
            rTr += w * r.dot(W * r);
            ++n;
            // A pair with no vertex cannot be attributed to a cluster. It must not be dropped (the
            // aggregate above still wants it) and must not be invented into vertex 0 either, so it
            // is recorded as unattributable and the nuisance refuses to run rather than guess.
            if (o.vertex < 0) { legacy_unattributed = true; return; }
            VertexBlock& v = per_vertex[o.vertex];
            v.A.noalias() += JtW * J;
            v.c.noalias() += JtW;
            v.D.noalias() += w * W;
            v.b.noalias() += JtW * r;
            v.e.noalias() += w * W * r;
            v.rTr += w * r.dot(W * r);
            ++v.n;
        }
        void reset()
        { H.setZero(); b.setZero(); rTr = 0.0; n = 0; per_vertex.clear(); legacy_unattributed = false; }

        /// Returns {parameters in units of prior sigma, posterior sigma, chi2/dof, cond, ok}.
        /// SIGN: r = uv_image - uv_lidar and J = d(uv_pred)/d(nuisance). A mount error of x makes the
        /// PREDICTION wrong by J*x while the image measurement is right, so r = -J*x and the fit
        /// returns minus the parameter — identical to stage 1, deliberately, so the two can be
        /// compared without a sign convention standing between them.
        struct Solution
        {
            Eigen::Vector4d p = Eigen::Vector4d::Zero();
            Eigen::Vector4d sigma = Eigen::Vector4d::Ones();
            double chi2_dof = 0.0, cond = 0.0, rho = 0.0;
            int    rho_i = 0, rho_j = 1, informed = 0;
            bool   ok = false;
            // ── the nuisance, reported so a reader can tell WHICH model produced these numbers ──
            bool   marginalised = false;  ///< the per-vertex offset was integrated out
            int    clusters = 0;          ///< distinct vertices — the REAL sample size for the level
            double eff_params = 0.0;      ///< Σ tr(M_v D_v): how many of the 2·clusters offsets the
                                          ///< data actually paid for. → 2 per vertex as S grows.
        };
        /// `min_n` counts PAIRS. The nuisance does not change that: a solve with 30 pairs on one
        /// vertex is still one cluster, and reporting `clusters` is how that is made visible rather
        /// than defended against with a second minimum.
        [[nodiscard]] Solution solve(long min_n = 30) const
        {
            Solution s;
            if (n < min_n) return s;
            Eigen::Matrix4d Hm = H;
            Eigen::Vector4d bm = b;
            double          rm = rTr;
            // ⚠ REFUSED, not silently skipped: unattributed evidence cannot be marginalised, and
            //   mixing it with evidence that can would produce a number belonging to neither model.
            const bool want = offset_sigma_px > 0.0 and not legacy_unattributed and not per_vertex.empty();
            if (want)
            {
                const double s2 = offset_sigma_px * offset_sigma_px;
                const Eigen::Matrix2d Sinv = Eigen::Matrix2d::Identity() / s2;
                Hm.setZero(); bm.setZero(); rm = 0.0;
                for (const auto& [vtx, v] : per_vertex)
                {
                    if (v.n <= 0 or not v.finite()) continue;
                    const Eigen::Matrix2d M = (Sinv + v.D).inverse();
                    if (not M.allFinite()) continue;
                    Hm.noalias() += v.A - v.c * M * v.c.transpose();
                    bm.noalias() += v.b - v.c * (M * v.e);
                    rm += v.rTr - v.e.dot(M * v.e);
                    s.eff_params += (M * v.D).trace();
                    ++s.clusters;
                }
                if (s.clusters == 0) return s;
                s.marginalised = true;
            }
            else s.clusters = static_cast<int>(per_vertex.size());

            const Eigen::Matrix4d A = Hm + Eigen::Matrix4d::Identity();
            const Eigen::Matrix4d C = A.inverse();
            if (not C.allFinite()) return s;
            const Eigen::Vector4d x = C * bm;
            const double chi2 = std::max(0.0, rm - x.dot(bm));
            // The offsets consume degrees of freedom too, and softly — tr(M_v D_v) is how much of
            // each vertex's 2 the data actually paid for. Ignoring it would inflate chi2/dof and
            // then inflate every sigma through `infl` below, hiding the improvement inside the fix.
            s.chi2_dof = chi2 / std::max(1.0, 2.0 * static_cast<double>(n) - 4.0 - s.eff_params);
            const double infl = std::sqrt(std::max(1.0, s.chi2_dof));
            s.p = -x;
            for (int i = 0; i < 4; ++i)
            {
                s.sigma(i) = std::sqrt(std::max(0.0, C(i, i))) * infl;
                if (s.sigma(i) < 0.9) s.informed |= (1 << i);
            }
            Eigen::Matrix4d R = Eigen::Matrix4d::Identity();
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    R(i, j) = C(i, j) / std::sqrt(std::max(1e-300, C(i, i) * C(j, j)));
            const Eigen::Vector4d ev = Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>(R).eigenvalues();
            s.cond = ev(3) / std::max(1e-12, ev(0));
            for (int i = 0; i < 4; ++i)
                for (int j = i + 1; j < 4; ++j)
                    if (std::abs(R(i, j)) > std::abs(s.rho)) { s.rho = R(i, j); s.rho_i = i; s.rho_j = j; }
            s.ok = true;
            return s;
        }
    };

}
