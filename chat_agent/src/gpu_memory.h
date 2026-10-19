#pragma once
/*
 * gpu_memory.h — the status bar's "GPU used/total" reading. ollama shares the robot's single GPU with retina (YOLOE,
 * DINO, ZED), so what matters is the WHOLE GPU's memory, not this process's. ChatSession runs
 *   nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
 * asynchronously (QProcess) every few seconds; its stdout ("8316, 16303\n", MiB, one line per GPU) is parsed here.
 * Anything unexpected (no driver, "[N/A]", garbage) ⇒ nullopt ⇒ "GPU n/a".
 *
 * Pure std, no Qt — tested in tests/test_gpu.cpp.
 */

#include <charconv>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>

namespace rc::chat
{

struct GpuMemory
{
    std::uint64_t used_mib = 0;
    std::uint64_t total_mib = 0;
};

/// The first line of the query's output → used/total MiB; nullopt if it is not two non-negative integers.
inline std::optional<GpuMemory> parse_gpu_memory(std::string_view out)
{
    const auto trim = [](std::string_view s)
    {
        constexpr std::string_view ws = " \t\r\n";
        const auto b = s.find_first_not_of(ws);
        if (b == std::string_view::npos) return std::string_view{};
        return s.substr(b, s.find_last_not_of(ws) - b + 1);
    };
    const auto number = [&](std::string_view s) -> std::optional<std::uint64_t>
    {
        s = trim(s);
        std::uint64_t v = 0;
        const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
        if (s.empty() or ec != std::errc{} or end != s.data() + s.size()) return std::nullopt;
        return v;
    };
    const std::string_view line = out.substr(0, out.find('\n'));
    const auto comma = line.find(',');
    if (comma == std::string_view::npos) return std::nullopt;
    const auto used = number(line.substr(0, comma));
    const auto total = number(line.substr(comma + 1));
    if (not used or not total or *total == 0) return std::nullopt;
    return GpuMemory{.used_mib = *used, .total_mib = *total};
}

/// "GPU 8.1/15.9 GB" (GiB, one decimal), or "GPU n/a".
inline std::string gpu_memory_label(const std::optional<GpuMemory>& m)
{
    if (not m) return "GPU n/a";
    constexpr double mib_per_gib = 1024.0;
    return std::format("GPU {:.1f}/{:.1f} GB", static_cast<double>(m->used_mib) / mib_per_gib,
                       static_cast<double>(m->total_mib) / mib_per_gib);
}

}   // namespace rc::chat
