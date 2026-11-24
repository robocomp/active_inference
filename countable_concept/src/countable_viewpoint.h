// Shim: the viewpoint decorrelation moved to common/ai_belief/viewpoint.h (namespace rc::ai::view, shared engine).
// Keeps `rc::ViewpointPath`, `rc::rho_camera`, ... resolving for the countable agent and its tests.
#pragma once
#include "../../common/ai_belief/viewpoint.h"

namespace rc { using namespace rc::ai::view; }
