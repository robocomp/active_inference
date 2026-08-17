/*
 * common/dashboard/belief_certainty.h — fill a BeliefStripRow's two certainty channels. SHARED, header-only.
 *
 * "How well does this belief know itself?" is answered the same way by every agent and every belief unit, and
 * the answer was written out EIGHT times: once per instance-family agent, and three times inside
 * cabinet_concept alone (cells, peninsula, and its instance path). It is ten lines, and every one of them is
 * a trap someone already fell into:
 *
 * ★`adequacy_gap_nats` RETURNS 0 FOR A DOF TABLE WITH NO σ* — an empty sum — and 0 is precisely the value
 * that means "adequate". So the demand has to be tested FIRST with any_sigma_star(), and the absence carried
 * as the -1 sentinel the widget knows how to read. Get that backwards and an agent that publishes no σ*
 * reports itself perfectly converged for ever, which is how a fixation looks like it is working when nothing
 * is happening.
 *
 * ★AND THE FALLBACK IS A CHOLESKY, NOT A DETERMINANT. ½·ln det Σ is computed as Σ ln L_ii, because a
 * covariance with centimetre σ has a determinant near the floor of float — log(det()) there is numerical
 * noise dressed as a certainty.
 *
 * ★★FAMILY-AGNOSTIC ON PURPOSE. This takes a covariance and a DOF table, not an instance: the belief UNIT
 * differs between the two families in this fleet — a fitted instance for bottle/chair/door/hood/
 * refrigerator/table, a (wall, tier) RUN for cabinet and the coming shelf — while the certainty rule is
 * identical for both. What varies is the loop that finds the beliefs, which stays with the agent.
 */

#pragma once

#include "belief_strip.h"            // rc::BeliefStripRow
#include "../ai_belief/dof_spec.h"    // rc::any_sigma_star / rc::adequacy_gap_nats

namespace rc::dash
{

// Fill `row.gap_nats` and `row.logdet_nats` from this belief's REPORTED covariance and its DOF table.
// `Mat` is whatever fixed-size Eigen covariance the agent's belief carries; `Dofs` its σ* table.
template <class Mat, class Dofs>
inline void fill_certainty(rc::BeliefStripRow& row, const Mat& S, const Dofs& dofs)
{
    row.gap_nats = rc::any_sigma_star(dofs)
                 ? rc::adequacy_gap_nats(dofs, [&](std::size_t j) { return S(j, j); })
                 : -1.0f;                       // -1 = "no demand", NOT "no gap" — see the header note
    const auto llt = S.llt();
    if (llt.info() == Eigen::Success)
        row.logdet_nats = llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
}

}  // namespace rc::dash
