#pragma once
/*
 * vlm_status.h — the status bar's "where does the VLM run" reading, for a REMOTE ollama (e.g. the Jetson at
 * [LLM] url). nvidia-smi on this machine says nothing about it, so ChatSession asks the server instead:
 *   GET /api/ps  →  {"models":[{"name":"gemma4:26b","model":"gemma4:26b","size_vram":1361072617,…}]}
 * and shows  "VLM gemma4:26b @ 192.168.1.34 (nvidia) · 1.4 GB loaded".  A local url keeps the nvidia-smi label
 * (gpu_memory.h). Numbers go through std::format (this machine runs es_ES); JSON through QJsonDocument.
 *
 * Qt Core only, no network — tested in tests/test_vlm_status.cpp.
 */

#include <algorithm>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <QByteArray>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QString>

namespace rc::chat
{

struct VlmMemory
{
    bool valid = false;               ///< the reply parsed as an /api/ps document
    bool loaded = false;              ///< the configured model is resident
    std::uint64_t vram_bytes = 0;     ///< its size_vram
    std::uint64_t others_bytes = 0;   ///< size_vram of every OTHER loaded model
};

/// /api/ps reply → memory of `model` (matched on "name" or "model") and of the rest; valid=false if not that JSON.
inline VlmMemory parse_ollama_ps(std::string_view json, std::string_view model)
{
    const auto doc = QJsonDocument::fromJson(QByteArray(json.data(), static_cast<qsizetype>(json.size())));
    if (not doc.isObject() or not doc.object().value("models").isArray()) return {};
    VlmMemory m{.valid = true};
    const QString want = QString::fromUtf8(model.data(), static_cast<qsizetype>(model.size()));
    for (const auto& v : doc.object().value("models").toArray())
    {
        const auto o = v.toObject();
        const auto vram = static_cast<std::uint64_t>(o.value("size_vram").toDouble(0.0));
        if (o.value("name").toString() == want or o.value("model").toString() == want)
        {
            m.loaded = true;
            m.vram_bytes += vram;
        }
        else
            m.others_bytes += vram;
    }
    return m;
}

/// "VLM <model> @ <ip> (<host>) · 1.4 GB loaded" / "· not loaded[ (others 17.0 GB)]" / "VLM @ <ip> unreachable".
/// `mem` empty or invalid ⇒ the server did not answer. `hostname` empty ⇒ no parentheses.
inline std::string vlm_status_label(std::string_view model, std::string_view ip, std::string_view hostname,
                                    const std::optional<VlmMemory>& mem)
{
    const std::string where = hostname.empty() ? std::string(ip) : std::format("{} ({})", ip, hostname);
    if (not mem or not mem->valid) return std::format("VLM @ {} unreachable", where);
    constexpr double gb = 1e9;   // ollama reports bytes; the label is decimal GB
    if (mem->loaded)
        return std::format("VLM {} @ {} · {:.1f} GB loaded", model, where, static_cast<double>(mem->vram_bytes) / gb);
    if (mem->others_bytes > 0)
        return std::format("VLM {} @ {} · not loaded (others {:.1f} GB)", model, where,
                           static_cast<double>(mem->others_bytes) / gb);
    return std::format("VLM {} @ {} · not loaded", model, where);
}

/// Is `host` this machine? Loopback names/addresses, or one of `local_addresses` (this host's interface addresses).
inline bool is_local_host(std::string_view host, const std::vector<std::string>& local_addresses)
{
    if (host.size() >= 2 and host.front() == '[' and host.back() == ']') host = host.substr(1, host.size() - 2);
    if (host.empty() or host == "localhost" or host == "::1" or host.starts_with("127.")) return true;
    return std::ranges::find(local_addresses, host) != local_addresses.end();
}

}   // namespace rc::chat
