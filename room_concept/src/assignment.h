/*
 *  assignment.h — minimum-cost bipartite assignment (Kuhn-Munkres, O(N³)).
 *
 *  Moved out of corner_detector.cpp unchanged so the wall segment ↔ wall landmark association can
 *  use the same solver (and the same termination guard) as the corner ↔ model-corner one.
 */
#pragma once

#include <algorithm>
#include <vector>

namespace rc::assign
{
    /// Infeasible pairs must be encoded as INFEASIBLE. Rows with no feasible column end up
    /// unassigned (-1).
    constexpr float INFEASIBLE = 1e9f;

    /// Returns assignment[row] = col, or -1 if that row remains unassigned.
    inline std::vector<int> solve_hungarian(const std::vector<std::vector<float>>& cost, int R, int C)
    {
        if (R == 0 or C == 0) return std::vector<int>(R, -1);

        const int N = std::max(R, C);   // pad to square

        auto cell = [&](int i, int j) -> float {
            return (i < R and j < C) ? cost[i][j] : INFEASIBLE;
        };

        // u[i]/v[j] — row/column potentials (1-indexed internally)
        // p[j]      — row currently matched to column j  (0 = free)
        // way[j]    — predecessor column on the shortest-path tree
        std::vector<float> u(N + 1, 0.f), v(N + 1, 0.f);
        std::vector<int>   p(N + 1, 0),   way(N + 1, 0);

        for (int i = 1; i <= N; ++i)
        {
            p[0] = i;
            int j0 = 0;
            std::vector<float> minv(N + 1, INFEASIBLE);
            std::vector<bool>  used(N + 1, false);

            bool augmented = true;
            do {
                used[j0] = true;
                const int i0 = p[j0];
                int   j1    = -1;          // -1 ⇒ no unused column found this step
                float delta = INFEASIBLE;
                for (int j = 1; j <= N; ++j)
                {
                    if (not used[j])
                    {
                        const float c = cell(i0 - 1, j - 1) - u[i0] - v[j];
                        if (c < minv[j]) { minv[j] = c; way[j] = j0; }
                        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                    }
                }
                // Termination guard: a valid augmenting step must reach a FEASIBLE free column. If none
                // exists (row i0 reaches only INFEASIBLE/padded columns), the classic Kuhn-Munkres loop
                // leaves delta==INFEASIBLE, never advances j0 to a free column, and spins FOREVER (caught
                // live via gdb: this thread at 100% CPU, the whole agent hung, needing kill -9). Bail
                // here WITHOUT touching the potentials so the matching stays consistent; row i then stays
                // unassigned (-1) — exactly this function's documented contract for a row with no
                // feasible column.
                if (j1 < 0 or delta >= INFEASIBLE * 0.5f) { augmented = false; break; }
                for (int j = 0; j <= N; ++j)
                {
                    if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                    else          { minv[j] -= delta; }
                }
                j0 = j1;
            } while (p[j0] != 0);

            // Augment along the discovered path only when we actually reached a free column. On a bail
            // the loop above never wrote p[], so previously matched rows/columns are left intact.
            if (augmented)
            {
                do {
                    const int j1 = way[j0];
                    p[j0] = p[j1];
                    j0 = j1;
                } while (j0);
            }
        }

        // Extract: p[j] = 1-indexed row assigned to column j. Reject padded or infeasible pairs.
        std::vector<int> assignment(R, -1);
        for (int j = 1; j <= C; ++j)
        {
            const int r = p[j] - 1;   // 0-based
            if (r >= 0 and r < R and cost[r][j - 1] < INFEASIBLE * 0.5f)
                assignment[r] = j - 1;
        }
        return assignment;
    }
} // namespace rc::assign
