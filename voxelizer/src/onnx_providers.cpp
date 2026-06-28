#include "onnx_providers.h"

#include <array>
#include <dlfcn.h>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <string>

namespace rc::onnx
{

namespace
{

// True if the TensorRT runtime + builder-resource shared libs can be dlopen'd. Registering the TRT
// EP when they can't crashes hard, so we probe first. (Lifted verbatim from YoloSegDetector.)
bool can_use_tensorrt(const char* log_tag)
{
    struct HandleGuard
    {
        void* h = nullptr;
        ~HandleGuard() { if (h) dlclose(h); }
    } nvinfer, builder_res;

    auto try_open_any = [](std::initializer_list<const char*> names) -> void*
    {
        for (const char* name : names)
            if (void* h = dlopen(name, RTLD_LAZY | RTLD_LOCAL); h)
                return h;
        return nullptr;
    };

    nvinfer.h = try_open_any({"libnvinfer.so", "libnvinfer.so.10", "libnvinfer.so.10.9.0"});
    if (!nvinfer.h)
    {
        std::cerr << log_tag << " TensorRT runtime missing (libnvinfer.so): " << dlerror() << "\n";
        return false;
    }

    builder_res.h = try_open_any({
        "libnvinfer_builder_resource.so",
        "libnvinfer_builder_resource.so.10",
        "libnvinfer_builder_resource.so.10.9.0"
    });

    if (!builder_res.h)
    {
        const std::array<std::filesystem::path, 3> probe_dirs = {
            "/usr/lib/x86_64-linux-gnu",
            "/lib/x86_64-linux-gnu",
            "/usr/local/cuda-13.2/lib64"
        };
        for (const auto& dir : probe_dirs)
        {
            std::error_code ec;
            if (!std::filesystem::exists(dir, ec) || ec)
                continue;
            for (const auto& entry : std::filesystem::directory_iterator(dir, ec))
            {
                if (ec) break;
                if (!entry.is_regular_file(ec) || ec)
                    continue;
                const auto name = entry.path().filename().string();
                if (name.rfind("libnvinfer_builder_resource_", 0) == 0 &&
                    name.find(".so") != std::string::npos)
                {
                    builder_res.h = dlopen(entry.path().c_str(), RTLD_LAZY | RTLD_LOCAL);
                    if (builder_res.h)
                        break;
                }
            }
            if (builder_res.h)
                break;
        }
    }

    if (!builder_res.h)
    {
        std::cerr << log_tag << " TensorRT builder resources missing "
                  << "(libnvinfer_builder_resource.so): " << dlerror() << "\n";
        return false;
    }

    return true;
}

// Pin the system TRT 10 stack into the global namespace so ORT binds against it (avoids picking up a
// bundled/mismatched copy). (Lifted verbatim from YoloSegDetector.)
void prefer_system_tensorrt_stack(const char* log_tag)
{
    static const std::array<const char*, 3> libs = {
        "/usr/lib/x86_64-linux-gnu/libnvinfer.so.10",
        "/usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10",
        "/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10"
    };
    for (const char* lib : libs)
        if (void* h = dlopen(lib, RTLD_NOW | RTLD_GLOBAL); h == nullptr)
            std::cerr << log_tag << " Failed to preload TensorRT library " << lib << ": " << dlerror() << "\n";
}

}  // namespace

void append_gpu_providers(Ort::SessionOptions& opts,
                          bool use_gpu,
                          bool use_trt,
                          const char* log_tag,
                          const char* trt_cache_path)
{
    if (!use_gpu)
    {
        std::cout << log_tag << " Using CPU EP\n";
        return;
    }

    if (use_trt)
    {
        if (can_use_tensorrt(log_tag))
        {
            prefer_system_tensorrt_stack(log_tag);
            try
            {
                OrtTensorRTProviderOptions trt{};
                trt.device_id = 0;
                trt.trt_fp16_enable = 1;
                trt.trt_engine_cache_enable = 1;
                trt.trt_engine_cache_path = trt_cache_path;
                trt.trt_max_partition_iterations = 1000;
                trt.trt_min_subgraph_size = 1;
                opts.AppendExecutionProvider_TensorRT(trt);
                std::cout << log_tag << " TensorRT EP registered (FP16, cache=" << trt_cache_path << ")\n";
            }
            catch (const Ort::Exception& e)
            {
                std::cerr << log_tag << " TensorRT EP unavailable (" << e.what() << "), using CUDA only\n";
            }
        }
        else
            std::cerr << log_tag << " TensorRT disabled due to missing shared libraries; using CUDA only\n";
    }

    try
    {
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = 0;
        opts.AppendExecutionProvider_CUDA(cuda);
        std::cout << log_tag << " CUDA EP registered\n";
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << log_tag << " CUDA EP unavailable (" << e.what() << "), falling back to CPU\n";
    }
}

}  // namespace rc::onnx
