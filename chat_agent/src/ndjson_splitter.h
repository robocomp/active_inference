#pragma once
/*
 * ndjson_splitter.h — turn an HTTP body that arrives in arbitrary chunks into complete NDJSON lines.
 *
 * ollama's /api/chat with stream:true sends one JSON object per line, but the network delivers BYTES: a chunk may
 * hold half a line, several lines, or end in the middle of a multi-byte UTF-8 character. Only a line terminated by
 * '\n' is complete; everything after the last '\n' waits for the next chunk. flush() hands back a final line that
 * the server did not terminate (end of body).
 *
 * Pure std, no Qt — tested in tests/test_ndjson.cpp.
 */

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rc::chat
{

class NdjsonSplitter
{
public:
    /// Append a chunk; return every line it completed (without '\n' / trailing '\r'; blank lines skipped).
    std::vector<std::string> feed(std::string_view chunk)
    {
        buf_.append(chunk);
        std::vector<std::string> lines;
        std::size_t start = 0;
        for (std::size_t nl = buf_.find('\n', start); nl != std::string::npos; nl = buf_.find('\n', start))
        {
            std::string line = buf_.substr(start, nl - start);
            if (not line.empty() and line.back() == '\r') line.pop_back();
            if (not is_blank(line)) lines.push_back(std::move(line));
            start = nl + 1;
        }
        buf_.erase(0, start);
        return lines;
    }
    /// The unterminated remainder at end of stream, if it holds anything.
    std::optional<std::string> flush()
    {
        std::string rest = std::move(buf_);
        buf_.clear();
        if (not rest.empty() and rest.back() == '\r') rest.pop_back();
        if (is_blank(rest)) return std::nullopt;
        return rest;
    }
    void reset() { buf_.clear(); }
    [[nodiscard]] std::size_t pending_bytes() const { return buf_.size(); }

private:
    static bool is_blank(std::string_view s)
    {
        return s.find_first_not_of(" \t\r") == std::string_view::npos;
    }
    std::string buf_;
};

}   // namespace rc::chat
