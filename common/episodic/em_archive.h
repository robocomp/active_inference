#pragma once
/*
 * em_archive.h — the episodic memory's append-only store (2026-10-17): one JSONL file per local day,
 * "<dir>/<YYYYMMDD>.jsonl", one compact record per line. A record is appended when it OPENS and again when it
 * CLOSES; readers keep the last line per id. Nothing is ever rewritten, so a crash can only damage the tail line.
 * A truncated / unparsable line, or one with no id, is skipped with a warning on stderr — never fatal.
 *
 * Locale (CLAUDE.md): written through an ofstream imbued with the classic locale, parsed by QJsonDocument.
 */

#include "em_schema.h"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <locale>
#include <map>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rc::em
{

class Archive
{
public:
    explicit Archive(std::filesystem::path dir) : dir_(std::move(dir)) { std::filesystem::create_directories(dir_); }

    /// One compact JSON line, flushed; the file is the local day of r.t0. False (and a warning) when it could not be written.
    /// A crash mid-write leaves a last line with no '\n'; terminate it first so this record is not glued onto the fragment.
    bool append(const Record& r)
    {
        const auto path = file_of(day_of(r.t0));
        const bool heal = unterminated(path);
        std::ofstream f(path, std::ios::app);
        f.imbue(std::locale::classic());
        if (heal) f << '\n';
        f << QJsonDocument(to_json(r)).toJson(QJsonDocument::Compact).toStdString() << '\n';
        f.flush();
        if (not f)
        {
            std::println(stderr, "[episodic] cannot write {}: record {} lost", path.string(), r.id);
            return false;
        }
        return true;
    }

    /// Last line per id wins; ordered by t0 (ties keep first-seen order).
    std::vector<Record> load_day(std::string_view day) const
    {
        std::vector<Record> out;
        std::map<std::string, std::size_t, std::less<>> at;      // id -> index in out
        std::ifstream f(file_of(day));
        std::string line;
        for (int n = 1; std::getline(f, line); ++n)
        {
            if (line.empty()) continue;
            QJsonParseError err;
            const QJsonDocument doc = QJsonDocument::fromJson(QByteArray::fromStdString(line), &err);
            std::optional<Record> r;
            if (err.error == QJsonParseError::NoError and doc.isObject()) r = record_from_json(doc.object());
            if (not r or r->id.empty())
            {
                std::println(stderr, "[episodic] {}:{} skipped (truncated or invalid record)", file_of(day).string(), n);
                continue;
            }
            if (const auto it = at.find(r->id); it != at.end()) out[it->second] = std::move(*r);
            else { at.emplace(r->id, out.size()); out.push_back(std::move(*r)); }
        }
        std::ranges::stable_sort(out, {}, &Record::t0);
        return out;
    }

    /// Days that have a file, sorted.
    std::vector<std::string> days() const
    {
        std::vector<std::string> out;
        for (const auto& e : std::filesystem::directory_iterator(dir_))
            if (e.is_regular_file() and e.path().extension() == ".jsonl") out.push_back(e.path().stem().string());
        std::ranges::sort(out);
        return out;
    }

    /// 1 + the largest seq among the ids "<day>-<seq>" of that day (1 when the day is empty).
    int next_seq(std::string_view day) const
    {
        int mx = 0;
        for (const Record& r : load_day(day))
        {
            const auto dash = r.id.rfind('-');
            if (dash == std::string::npos) continue;
            int seq = 0;
            const char* b = r.id.data() + dash + 1;
            const char* e = r.id.data() + r.id.size();
            const auto res = std::from_chars(b, e, seq);
            if (res.ec == std::errc{} and res.ptr == e) mx = std::max(mx, seq);
        }
        return mx + 1;
    }

private:
    /// True when the file exists, is non-empty and its last byte is not '\n' (a torn tail).
    static bool unterminated(const std::filesystem::path& p)
    {
        std::ifstream in(p, std::ios::binary | std::ios::ate);
        if (not in or in.tellg() <= 0) return false;
        in.seekg(-1, std::ios::end);
        return in.get() != '\n';
    }
    std::filesystem::path file_of(std::string_view day) const { return dir_ / (std::string(day) + ".jsonl"); }
    std::filesystem::path dir_;
};

}   // namespace rc::em
