// nvidia-smi memory query parsing + the status-bar label. Build+run: tests/run_tests.sh
#include "../src/gpu_memory.h"
#include <cassert>
#include <print>

int main()
{
    using rc::chat::parse_gpu_memory;
    using rc::chat::gpu_memory_label;
    int checks = 0;
    {   // the real output of --query-gpu=memory.used,memory.total --format=csv,noheader,nounits (MiB)
        const auto m = parse_gpu_memory("8316, 16303\n");
        assert(m.has_value() and m->used_mib == 8316 and m->total_mib == 16303); ++checks;
        assert(gpu_memory_label(m) == "GPU 8.1/15.9 GB"); ++checks;
    }
    {   // no trailing newline, no space, CRLF
        assert(parse_gpu_memory("100,200").has_value()); ++checks;
        const auto m = parse_gpu_memory("  512 ,  1024\r\n");
        assert(m.has_value() and m->used_mib == 512 and m->total_mib == 1024); ++checks;
        assert(gpu_memory_label(m) == "GPU 0.5/1.0 GB"); ++checks;
    }
    {   // several GPUs: the first line is the one reported
        const auto m = parse_gpu_memory("1024, 16384\n2048, 8192\n");
        assert(m.has_value() and m->used_mib == 1024 and m->total_mib == 16384); ++checks;
    }
    {   // anything else ⇒ nullopt ⇒ "GPU n/a"
        assert(not parse_gpu_memory("").has_value()); ++checks;
        assert(not parse_gpu_memory("\n").has_value()); ++checks;
        assert(not parse_gpu_memory("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.").has_value()); ++checks;
        assert(not parse_gpu_memory("[N/A], [N/A]").has_value()); ++checks;
        assert(not parse_gpu_memory("8316").has_value()); ++checks;
        assert(not parse_gpu_memory("8316, 16303x").has_value()); ++checks;
        assert(not parse_gpu_memory("-5, 16303").has_value()); ++checks;
        assert(not parse_gpu_memory("100, 0").has_value()); ++checks;   // a zero total is not a GPU reading
        assert(gpu_memory_label(std::nullopt) == "GPU n/a"); ++checks;
    }
    std::println("test_gpu: {} checks passed", checks);
    return 0;
}
