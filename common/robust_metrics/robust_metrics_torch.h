#pragma once

#include <torch/torch.h>

#include "robust_metrics.h"

torch::Tensor robust_loss_value(const torch::Tensor& residual, RobustLossType type, float scale);