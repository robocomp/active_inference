#pragma once

#include <optional>
#include <string_view>

enum class RobustLossType
{
    Quadratic,
    Huber,
    GemanMcClure,
    Welsch,
    TukeyBiweight,
};

std::optional<RobustLossType> robust_loss_type_from_string(std::string_view value);
std::string_view robust_loss_type_name(RobustLossType type);

float robust_loss_value(float residual, RobustLossType type, float scale);