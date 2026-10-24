// test_vlm_status.cpp — the remote-VLM status label: /api/ps parsing, the label formats, local-host detection.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../src/vlm_status.h"

using namespace rc::chat;

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)
#define CHECK_EQ(a, b) do { const std::string x = (a), y = (b); if (x != y) { std::printf("FAIL line %d: '%s' != '%s'\n", __LINE__, x.c_str(), y.c_str()); ++failures; } } while (0)

int main()
{
    const std::string sample = R"({"models":[{"name":"gemma4:26b","model":"gemma4:26b","size":1361072617,"digest":"x","details":{"family":"gemma"},"expires_at":"2026-10-24T10:00:00Z","size_vram":1361072617,"context_length":16384,"runner":"llama"}]})";
    const auto m = parse_ollama_ps(sample, "gemma4:26b");
    CHECK(m.valid and m.loaded and m.vram_bytes == 1361072617 and m.others_bytes == 0);
    CHECK_EQ(vlm_status_label("gemma4:26b", "192.168.1.34", "nvidia", m), "VLM gemma4:26b @ 192.168.1.34 (nvidia) · 1.4 GB loaded");
    CHECK_EQ(vlm_status_label("gemma4:26b", "192.168.1.34", "", m), "VLM gemma4:26b @ 192.168.1.34 · 1.4 GB loaded");

    const auto none = parse_ollama_ps(R"({"models":[]})", "gemma4:26b");
    CHECK(none.valid and not none.loaded);
    CHECK_EQ(vlm_status_label("gemma4:26b", "192.168.1.34", "nvidia", none), "VLM gemma4:26b @ 192.168.1.34 (nvidia) · not loaded");

    const auto other = parse_ollama_ps(R"({"models":[{"name":"qwen:7b","model":"qwen:7b","size_vram":17000000000}]})", "gemma4:26b");
    CHECK(other.valid and not other.loaded);
    CHECK_EQ(vlm_status_label("gemma4:26b", "192.168.1.34", "nvidia", other), "VLM gemma4:26b @ 192.168.1.34 (nvidia) · not loaded (others 17.0 GB)");

    for (const char* bad : {"", "not json", "{\"models\":3}", "[1,2]", "{\"models\":[{\"name\":"})
        CHECK(not parse_ollama_ps(bad, "x").valid);
    CHECK_EQ(vlm_status_label("m", "192.168.1.34", "nvidia", parse_ollama_ps("garbage", "m")), "VLM @ 192.168.1.34 (nvidia) unreachable");
    CHECK_EQ(vlm_status_label("m", "192.168.1.34", "", std::nullopt), "VLM @ 192.168.1.34 unreachable");

    const std::vector<std::string> mine{"192.168.1.50", "10.0.0.2"};
    CHECK(is_local_host("localhost", mine) and is_local_host("127.0.0.1", mine) and is_local_host("::1", mine));
    CHECK(is_local_host("[::1]", mine) and is_local_host("192.168.1.50", mine));
    CHECK(not is_local_host("192.168.1.34", mine) and not is_local_host("192.168.1.34", {}));

    if (failures == 0) std::puts("test_vlm_status: OK");
    return failures == 0 ? 0 : 1;
}
