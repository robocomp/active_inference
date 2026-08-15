#pragma once

/*
 * onnx_providers.h
 *
 * Shared ONNX Runtime GPU execution-provider setup for the retina's detectors
 * (YoloSegDetector, YoloPoseDetector, …). Encapsulates the TensorRT availability
 * probing + EP registration so each detector doesn't re-hand-roll it (and so the
 * crash-avoidance probing — see the "TensorRT version lock" notes — lives in ONE place).
 */

#include <onnxruntime_cxx_api.h>

namespace rc::onnx
{

// Append the best available GPU execution provider(s) to `opts`:
//   - use_gpu == false  → no EP appended (ORT falls back to CPU).
//   - use_trt == true   → register the TensorRT EP (FP16, engine cache) ONLY if the TRT shared
//                         libraries are actually loadable; otherwise skip it with a warning.
//   - CUDA EP is always appended after (TRT subgraphs run on TRT, the rest on CUDA; if TRT was
//     skipped everything runs on CUDA).
// `log_tag` prefixes the diagnostic lines (e.g. "[YoloPoseDetector]"). Never throws — EP failures
// are caught and logged, leaving `opts` usable (worst case: CPU).
void append_gpu_providers(Ort::SessionOptions& opts,
                          bool use_gpu,
                          bool use_trt,
                          const char* log_tag,
                          const char* trt_cache_path = ".trt_cache");

}  // namespace rc::onnx
