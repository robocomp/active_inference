#include <torch/torch.h>

#include <algorithm>

#include "robust_metrics_torch.h"

namespace
{
float clamp_scale(float scale)
{
    return std::max(scale, 1e-6f);
}
}

torch::Tensor robust_loss_value(const torch::Tensor& residual, RobustLossType type, float scale)
{
    const float c = clamp_scale(scale);
    const float c2 = c * c;
    const auto abs_r = torch::abs(residual);
    const auto r2 = residual * residual;

    switch (type)
    {
        case RobustLossType::Quadratic:
            return r2;

        case RobustLossType::Huber:
            return torch::where(abs_r <= c, r2, 2.0f * c * abs_r - c2);

        case RobustLossType::GemanMcClure:
            return c2 * r2 / (r2 + c2);

        case RobustLossType::Welsch:
            return c2 * (1.0f - torch::exp(-r2 / c2));

        case RobustLossType::TukeyBiweight:
        {
            const auto u2 = r2 / c2;
            const auto one_minus_u2 = 1.0f - u2;
            const auto inside = (c2 / 3.0f) * (1.0f - one_minus_u2 * one_minus_u2 * one_minus_u2);
            return torch::where(abs_r <= c, inside, torch::full_like(residual, c2 / 3.0f));
        }
    }

    return r2;
}