#pragma once
/*
 * canvas_commands.h — the side canvas' fast path: "show me the zed camera", "show me the room", "show me the plan",
 * "close the canvas" are drawn at once with a stock script, without the model (2026-10-18).
 *
 * Only these fixed phrasings (English and the obvious Spanish) — anything looser ("draw the doors in red") is a
 * sentence for the model, which has the draw tool. Pure, header-only; tests/test_canvas.cpp.
 */

#include <optional>
#include <regex>
#include <string>

namespace rc::chat
{

struct ShowRequest
{
    enum class Kind { camera, layout, plan, hide } kind;
    std::string arg;   ///< camera: "zed" | "ricoh"
};

[[nodiscard]] inline std::optional<ShowRequest> parse_show(const std::string& lowered)
{
    std::string t = lowered;
    while (not t.empty() and (t.back() == '.' or t.back() == '!' or t.back() == '?' or t.back() == ' ')) t.pop_back();
    static const std::regex show(R"re(^(?:please )?(?:show(?: me)?|display|open|muestrame|muéstrame|ensename|enséñame))re"
                                 R"re( (?:the |la |el |las |los )?(.+?)(?: please)?$)re");
    static const std::regex hide(R"re(^(?:hide|close|cierra|oculta) (?:the |el )?(?:canvas|lienzo|drawing|dibujo)$)re");
    if (std::regex_match(t, hide)) return ShowRequest{ShowRequest::Kind::hide, ""};
    std::smatch m;
    if (not std::regex_match(t, m, show)) return std::nullopt;
    const std::string what = m[1].str();
    static const std::regex zed(R"re(^(?:zed|front|frontal)(?: camera)?$|^(?:camara|cámara) (?:zed|frontal|delantera)$|^camera$|^(?:camara|cámara)$)re");
    static const std::regex ricoh(R"re(^(?:ricoh|360|panorama|panoramic)(?: camera)?$|^(?:camara|cámara) (?:ricoh|360|panoramica|panorámica)$)re");
    static const std::regex layout(R"re(^(?:room|map|layout|floor plan|plan of the room|layout of the room|room layout|)re"
                                   R"re(habitacion|habitación|mapa|plano(?: de la habitaci[oó]n)?)$)re");
    static const std::regex plan(R"re(^(?:plan|mission|mission plan|misi[oó]n)$)re");
    if (std::regex_match(what, zed)) return ShowRequest{ShowRequest::Kind::camera, "zed"};
    if (std::regex_match(what, ricoh)) return ShowRequest{ShowRequest::Kind::camera, "ricoh"};
    if (std::regex_match(what, layout)) return ShowRequest{ShowRequest::Kind::layout, ""};
    if (std::regex_match(what, plan)) return ShowRequest{ShowRequest::Kind::plan, ""};
    return std::nullopt;
}

/// A library name a person or the model may give a saved script: [a-z0-9_], 1..40 chars; "" when nothing is left.
[[nodiscard]] inline std::string canvas_script_name(const std::string& raw)
{
    std::string s;
    for (char c : raw)
    {
        if (c >= 'A' and c <= 'Z') c = char(c - 'A' + 'a');
        if ((c >= 'a' and c <= 'z') or (c >= '0' and c <= '9') or c == '_') s += c;
        else if ((c == ' ' or c == '-') and not s.empty() and s.back() != '_') s += '_';
        if (s.size() == 40) break;
    }
    while (not s.empty() and s.back() == '_') s.pop_back();
    return s;
}

}   // namespace rc::chat
