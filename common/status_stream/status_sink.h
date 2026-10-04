/*
 * common/status_stream/status_sink.h — the PRODUCER half of rc::StatusStream. Header-only, STDLIB ONLY.
 *
 * Any code in an agent (any thread, any translation unit, including the offline selftests that link
 * the agent's estimator sources without Qt Network) reports through the free functions below:
 *
 *     rc::status::event("loaded", rc::status::Obj{}.s("what", "layout").s("path", p).i("verts", n));
 *     rc::status::log(rc::status::Level::Info, "[camcal] resumed from ...");   // a whole line
 *     rc::status::println("[pumps] over {} s ...", secs);                       // std::println shape
 *     rc::status::cprintf("[level2] fit %.3f\n", s);                            // std::printf shape
 *
 * WITH NO SINK INSTALLED (no StatusStream: the selftests, [Status] Enable = false) every call keeps
 * the behaviour it replaced exactly: a log line goes to stdout (stderr for *_err), an event goes
 * nowhere. So converting a print site to these functions can never lose a line in a build that does
 * not run the stream — which is what makes the conversion mechanical rather than a judgement per site.
 *
 * WITH A SINK the stream decides, per line, whether it ALSO reaches the terminal (the agent's routing
 * table) and records every line as a `log` event regardless. Nothing is deleted; it is routed.
 *
 * printf-style fragments ("%.1f" in a loop, then "\n") are buffered PER THREAD until a newline, so a
 * line assembled by several calls is still one line — one event, one routing decision.
 */
#pragma once

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <format>
#include <string>
#include <string_view>

namespace rc::status
{

enum class Level : std::uint8_t { Debug = 0, Info, Warning, Critical, Fatal };

constexpr std::string_view to_string(Level l)
{
    switch (l)
    {
        case Level::Debug:    return "debug";
        case Level::Info:     return "info";
        case Level::Warning:  return "warning";
        case Level::Critical: return "critical";
        case Level::Fatal:    return "fatal";
    }
    return "info";
}

// ── A tiny JSON object builder, so producers need neither Qt nor a JSON library ──────────────────────
// Produces `{"k":v,...}`. Values are escaped; non-finite floats become null (JSON has no NaN).
class Obj
{
public:
    Obj& s(std::string_view k, std::string_view v) { key(k); quote(v); return *this; }
    Obj& i(std::string_view k, long long v) { key(k); out_ += std::to_string(v); return *this; }
    Obj& u(std::string_view k, unsigned long long v) { key(k); out_ += std::to_string(v); return *this; }
    Obj& b(std::string_view k, bool v) { key(k); out_ += v ? "true" : "false"; return *this; }
    Obj& f(std::string_view k, double v, int prec = 6)
    {
        key(k);
        num(v, prec);
        return *this;
    }
    // A pre-serialised JSON value (object, array, number) — for nesting.
    Obj& raw(std::string_view k, std::string_view json) { key(k); out_ += json; return *this; }
    Obj& null(std::string_view k) { key(k); out_ += "null"; return *this; }

    [[nodiscard]] std::string str() const { return "{" + out_ + "}"; }
    [[nodiscard]] const std::string& body() const { return out_; }   // without the braces
    [[nodiscard]] bool empty() const { return out_.empty(); }

    static void escape_into(std::string& o, std::string_view v)
    {
        for (const char c : v)
        {
            switch (c)
            {
                case '"':  o += "\\\""; break;
                case '\\': o += "\\\\"; break;
                case '\n': o += "\\n"; break;
                case '\r': o += "\\r"; break;
                case '\t': o += "\\t"; break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20)
                        o += std::format("\\u{:04x}", static_cast<unsigned>(static_cast<unsigned char>(c)));
                    else
                        o += c;
            }
        }
    }
    static std::string quoted(std::string_view v) { std::string o = "\""; escape_into(o, v); o += '"'; return o; }
    static std::string number(double v, int prec = 6)
    {
        if (not (v == v) or v > 1e300 or v < -1e300) return "null";
        return std::format("{:.{}g}", v, prec);   // std::format is locale-independent by default
    }

private:
    void key(std::string_view k)
    {
        if (not out_.empty()) out_ += ',';
        quote(k);
        out_ += ':';
    }
    void quote(std::string_view v) { out_ += '"'; escape_into(out_, v); out_ += '"'; }
    void num(double v, int prec) { out_ += number(v, prec); }

    std::string out_;
};

// A JSON array builder for the same purpose.
class Arr
{
public:
    Arr& raw(std::string_view json) { sep(); out_ += json; return *this; }
    Arr& s(std::string_view v) { sep(); out_ += Obj::quoted(v); return *this; }
    Arr& f(double v, int prec = 6) { sep(); out_ += Obj::number(v, prec); return *this; }
    Arr& i(long long v) { sep(); out_ += std::to_string(v); return *this; }
    Arr& b(bool v) { sep(); out_ += v ? "true" : "false"; return *this; }
    [[nodiscard]] std::string str() const { return "[" + out_ + "]"; }

private:
    void sep() { if (not out_.empty()) out_ += ','; }
    std::string out_;
};

// ── The sinks. Installed by rc::StatusStream on construction, cleared on destruction. ───────────────
// EventSink: (kind, body-without-braces). LogSink: (level, whole line without the trailing newline,
// to_stderr) → returns TRUE when the line must ALSO be written to the terminal by the caller.
using EventSink = void (*)(std::string_view kind, std::string_view body);
using LogSink   = bool (*)(Level level, std::string_view line, bool to_stderr);

using FlushSink = void (*)();

inline std::atomic<EventSink>& event_sink() { static std::atomic<EventSink> s{nullptr}; return s; }
inline std::atomic<LogSink>&   log_sink()   { static std::atomic<LogSink> s{nullptr}; return s; }
inline std::atomic<FlushSink>& flush_sink() { static std::atomic<FlushSink> s{nullptr}; return s; }

/// Push everything posted so far out to the file and the viewers, synchronously. For the FATAL path,
/// right before std::exit. Effective only on the thread that owns the stream (the main thread); a
/// no-op elsewhere and when no stream is running.
inline void flush()
{
    std::fflush(stdout);
    if (const auto sink = flush_sink().load(std::memory_order_acquire))
        sink();
}

inline void event(std::string_view kind, const Obj& body = {})
{
    if (const auto sink = event_sink().load(std::memory_order_acquire))
        sink(kind, body.body());
}

// One WHOLE line (no trailing newline needed; one is stripped if present).
inline void log(Level level, std::string_view line, bool to_stderr = false)
{
    while (not line.empty() and (line.back() == '\n' or line.back() == '\r'))
        line.remove_suffix(1);
    bool terminal = true;
    if (const auto sink = log_sink().load(std::memory_order_acquire))
        terminal = sink(level, line, to_stderr);
    if (terminal)
    {
        std::FILE* f = to_stderr ? stderr : stdout;
        std::fwrite(line.data(), 1, line.size(), f);
        std::fputc('\n', f);
        if (to_stderr) std::fflush(f);
    }
}

namespace detail
{
// Per-thread partial line for the printf/print shapes: a line built by several calls is one line.
inline std::string& pending(bool err)
{
    thread_local std::string out, errb;
    return err ? errb : out;
}
inline void feed(std::string_view text, bool err, Level level)
{
    auto& buf = pending(err);
    buf.append(text);
    std::size_t start = 0;
    for (std::size_t nl = buf.find('\n'); nl != std::string::npos; nl = buf.find('\n', start))
    {
        log(level, std::string_view(buf).substr(start, nl - start), err);
        start = nl + 1;
    }
    buf.erase(0, start);
}
inline void vfeed(bool err, Level level, const char* fmt, std::va_list ap)
{
    char small[512];
    std::va_list ap2;
    va_copy(ap2, ap);
    const int n = std::vsnprintf(small, sizeof small, fmt, ap);
    if (n < 0) { va_end(ap2); return; }
    if (static_cast<std::size_t>(n) < sizeof small)
        feed(std::string_view(small, static_cast<std::size_t>(n)), err, level);
    else
    {
        std::string big(static_cast<std::size_t>(n) + 1, '\0');
        std::vsnprintf(big.data(), big.size(), fmt, ap2);
        big.resize(static_cast<std::size_t>(n));
        feed(big, err, level);
    }
    va_end(ap2);
}
}  // namespace detail

// ── drop-in shapes for the print families ───────────────────────────────────────────────────────────
// std::printf(...)            → rc::status::cprintf(...)
// std::fprintf(stderr, ...)   → rc::status::cprintf_err(...)
// std::println(fmt, ...)      → rc::status::println(fmt, ...)
// std::print(fmt, ...)        → rc::status::print(fmt, ...)
[[gnu::format(printf, 1, 2)]] inline void cprintf(const char* fmt, ...)
{
    std::va_list ap;
    va_start(ap, fmt);
    detail::vfeed(false, Level::Info, fmt, ap);
    va_end(ap);
}
[[gnu::format(printf, 1, 2)]] inline void cprintf_err(const char* fmt, ...)
{
    std::va_list ap;
    va_start(ap, fmt);
    detail::vfeed(true, Level::Warning, fmt, ap);
    va_end(ap);
}
/// A fatal stop: one `fatal` event plus the line on stderr, flushed. The CALLER still exits.
inline void fatal(std::string_view msg)
{
    event("fatal", Obj{}.s("msg", msg));
    log(Level::Critical, msg, true);
    flush();
}

template <class... Args>
void print(std::format_string<Args...> fmt, Args&&... args)
{
    detail::feed(std::format(fmt, std::forward<Args>(args)...), false, Level::Info);
}
template <class... Args>
void println(std::format_string<Args...> fmt, Args&&... args)
{
    detail::feed(std::format(fmt, std::forward<Args>(args)...) + "\n", false, Level::Info);
}
// std::print(stderr, ...) shape.
template <class... Args>
void eprintln(std::format_string<Args...> fmt, Args&&... args)
{
    detail::feed(std::format(fmt, std::forward<Args>(args)...) + "\n", true, Level::Warning);
}

}  // namespace rc::status
