#include "robust_metrics.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <string>

namespace
{
std::string normalize_name(std::string_view value)
{
    std::string out;
    out.reserve(value.size());
    for (const unsigned char ch : value)
    {
        if (ch == '-' || ch == ' ')
            out.push_back('_');
        else
            out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

float clamp_scale(float scale)
{
    return std::max(scale, 1e-6f);
}
}

std::optional<RobustLossType> robust_loss_type_from_string(std::string_view value)
{
    const std::string normalized = normalize_name(value);

    if (normalized == "quadratic" || normalized == "l2" || normalized == "squared")
        return RobustLossType::Quadratic;
    if (normalized == "huber")
        return RobustLossType::Huber;
    if (normalized == "geman_mcclure" || normalized == "geman" || normalized == "gm")
        return RobustLossType::GemanMcClure;
    if (normalized == "welsch" || normalized == "leclerc")
        return RobustLossType::Welsch;
    if (normalized == "tukey" || normalized == "tukey_biweight" || normalized == "biweight")
        return RobustLossType::TukeyBiweight;

    return std::nullopt;
}

std::string_view robust_loss_type_name(RobustLossType type)
{
    switch (type)
    {
        case RobustLossType::Quadratic:     return "quadratic";
        case RobustLossType::Huber:         return "huber";
        case RobustLossType::GemanMcClure:  return "geman_mcclure";
        case RobustLossType::Welsch:        return "welsch";
        case RobustLossType::TukeyBiweight: return "tukey_biweight";
    }
    return "quadratic";
}

float robust_loss_value(float residual, RobustLossType type, float scale)
{
    const float c = clamp_scale(scale);
    const float abs_r = std::abs(residual);
    const float r2 = residual * residual;
    const float c2 = c * c;

    switch (type)
    {
        case RobustLossType::Quadratic:
            return r2;

        case RobustLossType::Huber:
            return abs_r <= c ? r2 : (2.0f * c * abs_r - c2);

        case RobustLossType::GemanMcClure:
            return c2 * r2 / (r2 + c2);

        case RobustLossType::Welsch:
            return c2 * (1.0f - std::exp(-r2 / c2));

        case RobustLossType::TukeyBiweight:
            if (abs_r > c)
                return c2 / 3.0f;
            {
                const float u2 = r2 / c2;
                const float one_minus_u2 = 1.0f - u2;
                return (c2 / 3.0f) * (1.0f - one_minus_u2 * one_minus_u2 * one_minus_u2);
            }
    }

    return r2;
}