#pragma once

#include <vector>
#include <tuple>

#define empty_fn doublebuffer_sync_empty_fn
#define is_iterable doublebuffer_sync_is_iterable
#include "../../../../robocomp_core/classes/doublebuffer_sync/doublebuffer_sync.h"
#undef is_iterable
#undef empty_fn

namespace rc
{

struct RawLidarPointVectors
{
    std::vector<float> xs;
    std::vector<float> ys;
    std::vector<float> zs;
};

using LidarPointVectors = std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>;
using LidarPointBuffer = BufferSync<InOut<RawLidarPointVectors, LidarPointVectors>>;

} // namespace rc