/*
 *  room_gn_solver.h — Levenberg-Marquardt backend for the room localizer's sliding-window RFE.
 *
 *  WHY
 *  ---
 *  The Adam/L-BFGS backends spend a measured ~72 ms (median, 29 iterations) solving a problem with
 *  15 free scalars, because every iteration is a torch forward + autograd backward over the whole
 *  window. Nothing about the problem needs autograd: every factor in compute_rfe_loss() is already
 *  in information form (μ, Λ) or reduces to one, and the only non-trivial derivative — the SDF's —
 *  is available in closed form because Model::sdf_query_at_pose already returns ∇SDF (see
 *  SdfQueryResult::grad). So we assemble H = ΣJᵀWJ and b = ΣJᵀWr directly and solve a 15×15 LDLT.
 *
 *  THE FACTOR REGISTRY (and why it looks over-general for 15 variables)
 *  -------------------------------------------------------------------
 *  The localizer is acquiring landmarks dynamically: object_anchor_source hands over one
 *  ObjectAnchorObs per stabilised object in the graph, and that set changes every frame as objects
 *  are born, converge, and are retired. A solver that hard-codes "5 poses, 3 scalars each" has to be
 *  rewritten the first time a landmark becomes a variable rather than a constant. So:
 *
 *    - Factors are a heterogeneous list rebuilt per solve(). Adding/removing landmarks is a push/pop,
 *      not a code path.
 *    - Variables are addressed through VarIndex (id → offset, dim) instead of a hard-coded stride.
 *      Promoting landmarks from constants to variables is then "register them in the index and give
 *      Se2LandmarkFactor a second variable slot", with the pose block solved by Schur complement —
 *      an addition, not a rewrite.
 *    - Corners and object anchors are ONE factor type (Se2LandmarkFactor). They already share their
 *      geometry (z_hat = R(-θ)·(p − t)); a corner is simply a landmark whose yaw channel is off.
 *
 *  WHAT IT IS NOT
 *  --------------
 *  Not a re-derivation of the model. The loss it minimises is the same one compute_rfe_loss()
 *  evaluates, term for term, including the FEJ+Schur boundary prior, the Huber saturations (as IRLS
 *  weights, which is the standard equivalence) and the shared observation weights from
 *  room_obs_weights.h. If the two ever disagree, the shadow log is the place it shows up.
 *
 *  Known, intentional difference: params.velocity_adaptive_weights rescales Adam's gradient on the
 *  newest pose. That is a PRECONDITIONER — it changes the path, not the minimum — and a curvature-
 *  correct step does not want it, so the GN backend ignores it. Converged poses should still agree.
 */
#pragma once

#include <Eigen/Dense>
#include <deque>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "room_concept.h"
#include "room_model.h"

namespace rc::gn
{
    /// Maps a variable id to its slice of the state vector. Today every variable is one SE(2) pose
    /// (dim 3); the dim is explicit so a future landmark/extent variable does not need a new index.
    class VarIndex
    {
    public:
        int add(int dim)
        {
            const int k = static_cast<int>(offsets_.size());
            offsets_.push_back(total_);
            dims_.push_back(dim);
            total_ += dim;
            return k;
        }
        int offset(int k) const { return offsets_[static_cast<size_t>(k)]; }
        int dim(int k) const { return dims_[static_cast<size_t>(k)]; }
        int count() const { return static_cast<int>(offsets_.size()); }
        int total() const { return total_; }

    private:
        std::vector<int> offsets_;
        std::vector<int> dims_;
        int total_ = 0;
    };

    /// Concatenated variable blocks. For the pose-window problem this is [x,y,θ] per slot.
    using State = Eigen::VectorXf;

    /// Normal equations under construction. H δ = −b.
    struct LinearSystem
    {
        Eigen::MatrixXf H;
        Eigen::VectorXf b;

        explicit LinearSystem(int n) : H(Eigen::MatrixXf::Zero(n, n)), b(Eigen::VectorXf::Zero(n)) {}

        /// H(i,j) += block, keeping the matrix symmetric by construction at the call sites.
        void add_H(int i, int j, const Eigen::MatrixXf& block) { H.block(i, j, block.rows(), block.cols()) += block; }
        void add_b(int i, const Eigen::VectorXf& v) { b.segment(i, v.size()) += v; }
    };

    /// One term of the free energy. evaluate() must return exactly the loss linearize() linearizes,
    /// or Levenberg's accept/reject test is measuring the wrong thing.
    class IFactor
    {
    public:
        virtual ~IFactor() = default;
        virtual float evaluate(const State& x) const = 0;
        virtual float linearize(const State& x, LinearSystem& sys) const = 0;
    };

    using FactorList = std::vector<std::unique_ptr<IFactor>>;

    struct Options
    {
        int   max_iters    = 10;
        float lambda_init  = 1e-3f;   // Levenberg damping, relative to diag(H)
        float lambda_up    = 5.0f;    // on a rejected step
        float lambda_down  = 0.33f;   // on an accepted one
        float lambda_max   = 1e6f;    // give up past this
        float step_tol     = 1e-5f;   // ‖δ‖∞ (m / rad) below which we are done
        float loss_rel_tol = 1e-4f;   // relative loss improvement below which we are done
    };

    struct Result
    {
        bool  ok = false;
        float loss = std::numeric_limits<float>::quiet_NaN();      // final loss, same units as compute_rfe_loss
        float loss_init = std::numeric_limits<float>::quiet_NaN();
        int   iterations = 0;         // accepted + rejected LM steps taken
        int   rejected = 0;           // of those, how many were rejected (damping had to grow)
        float lambda_final = 0.f;
        float step_norm = 0.f;        // ‖δ‖∞ of the last accepted step
        float grad_norm = 0.f;        // ‖b‖∞ at the solution — the real convergence witness
    };

    /// Everything the solver reads. Assembled by the caller because WindowManager is private to
    /// RoomConcept; the solver deliberately owns no torch state of its own.
    struct Input
    {
        const Model*                              model = nullptr;
        const RoomConcept::Params*                params = nullptr;
        const std::deque<RoomConcept::WindowSlot>* window = nullptr;
        const RoomConcept::BoundaryPrior*         boundary_prior = nullptr;
        float                                     boundary_weight = 1.0f;
        torch::Device                             device = torch::kCPU;
    };

    /// Minimise the window RFE. `poses` is in/out: one [x, y, θ] per window slot, same order as the
    /// window. The caller writes them back into the slot tensors (or discards them, in shadow mode).
    Result solve(const Input& in, std::vector<Eigen::Vector3f>& poses, const Options& opts);

    /// The loss alone, at the given poses, computed by the SAME code path solve() minimises. Used to
    /// score the other backend's answer on the GN objective without touching the window tensors.
    float evaluate(const Input& in, const std::vector<Eigen::Vector3f>& poses);

    /// Builds the factor list for one window state. Exposed for testing and because it is the single
    /// place a new factor family (a new landmark type, an extent variable, …) has to be registered.
    /// The returned factors hold a reference to `in`, which must outlive them.
    FactorList build_factors(const Input& in, const VarIndex& idx);

    /// Self-check: max relative error between the analytic gradient (b of the normal equations) and a
    /// central finite difference of the SAME loss, over all state components. Anything above ~1e-2 means
    /// a Jacobian is wrong — which is the one failure mode that would make GN quietly converge to the
    /// wrong pose while every other number in the shadow log still looked plausible. Costs 2n loss
    /// evaluations, so it is opt-in (RoomConcept.GnGradCheck).
    float gradient_check(const Input& in, const std::vector<Eigen::Vector3f>& poses, float eps = 1e-3f);

} // namespace rc::gn
