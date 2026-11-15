/*
 * class_prior.h — the [prior.class] block of a <concept>.concept.toml: how far a YOLO label of this class can be
 * trusted (π_cls = P(a mask carrying this label is really this class)). Textual like back_panel.h: no Qt, no DSR,
 * numbers through std::from_chars (locale-independent; the agents run under LANG=es_ES.UTF-8).
 *
 * The reader does NOT judge the value: logit(π) is ±inf outside (0,1), so the CALLER (countable_config.cpp) must
 * refuse such a value. A present-but-non-numeric p_label_true is reported loudly here and returned as NaN, so the
 * caller's (0,1) test rejects it too (it is never silently the default).
 */
#pragma once

#include <charconv>
#include <fstream>
#include <limits>
#include <optional>
#include <print>
#include <string>
#include <string_view>

namespace rc::manifest {

// p_label_true: π_cls. fit_*_m: the per-class box-fit error (m) of the fitted (H, w, h); absent key = 0. A present key
// that is not a number is reported and returned as NaN (the caller refuses NaN and negatives).
struct ClassPrior { float p_label_true = std::numeric_limits<float>::quiet_NaN(); float fit_H_m = 0, fit_w_m = 0, fit_h_m = 0; };

inline std::optional<ClassPrior> load_class_prior(const std::string& path, bool* opened = nullptr)
{
    std::ifstream f(path);
    if (opened) *opened = static_cast<bool>(f);
    if (not f)
    {
        std::print("[manifest] ★★CANNOT OPEN {} — the class prior cannot be read (the path is relative to the agent's CWD)\n", path);
        return std::nullopt;
    }
    const auto trim = [](std::string_view s) -> std::string_view
    {
        const auto b = s.find_first_not_of(" \t\r");
        if (b == std::string_view::npos) return {};
        return s.substr(b, s.find_last_not_of(" \t\r") - b + 1);
    };
    std::string line, block;
    bool seen = false, have_p = false;
    ClassPrior out;
    while (std::getline(f, line))
    {
        std::string_view l = trim(line);
        if (l.empty() or l.front() == '#') continue;
        if (l.front() == '[')
        {
            const auto e = l.find(']');
            block = std::string(trim(l.substr(1, e == std::string_view::npos ? std::string_view::npos : e - 1)));
            if (block == "prior.class") seen = true;
            continue;
        }
        if (block != "prior.class") continue;
        const auto eq = l.find('=');
        if (eq == std::string_view::npos) continue;
        const auto key = trim(l.substr(0, eq));
        float* dst = key == "p_label_true" ? &out.p_label_true : key == "fit_error_H_m" ? &out.fit_H_m
                   : key == "fit_error_w_m" ? &out.fit_w_m : key == "fit_error_h_m" ? &out.fit_h_m : nullptr;
        if (not dst) continue;
        auto val = trim(l.substr(eq + 1));
        if (const auto c = val.find('#'); c != std::string_view::npos) val = trim(val.substr(0, c));
        if (dst == &out.p_label_true) have_p = true;
        float x = 0.0f;
        const auto [e, ec] = std::from_chars(val.data(), val.data() + val.size(), x);
        if (ec != std::errc{} or e != val.data() + val.size())
        {
            std::print("[manifest] ★{}: [prior.class] {} = \"{}\" is not a number\n", path, key, val);
            *dst = std::numeric_limits<float>::quiet_NaN();
        }
        else *dst = x;
    }
    if (not seen or not have_p) return std::nullopt;
    return out;
}

}  // namespace rc::manifest
