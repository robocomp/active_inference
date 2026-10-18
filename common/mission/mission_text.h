#pragma once
/*
 * mission_text.h — the people-facing syntax of the grounded mission (mission_ir.h): the confirmation card, logs and
 * remote_cmd.txt mission scripts (2026-10-17). The wire is JSON (mission_json.h); this is the SAME IR, written for
 * people. Design: chat_agent/docs/mission_language_2026-10-13.md §7.3.
 *
 *   mission "explore room_2 and return"                     ← column 0: the mission line, then header lines
 *   anchor start = pose:room_1(0.42, -1.1, 1.54, 0.05)      ← pose:ROOM(x, y[, yaw[, sigma_xy]])
 *   requires standoff_m
 *     cross door_27 expect=room_2 retry=2                   ← indented: steps; deeper indent = the body
 *     explore room_2 until=explored(room_2)>=0.9 budget_s=300s
 *     cross passage:passage_0@room_2 expect=room_1 retry=2  ← the face passage_0 shows in room_2
 *     go_to anchor:start                                    # comments run to the end of the line
 *
 *   step   := op positional* (key=value)*        positionals first, in the registry's declaration order
 *   ref    := NAME | anchor:NAME | passage:NAME@ROOM | pose:ROOM(…) | $NAME
 *   pred   := NAME(ref, …) [>= p | < p]
 *   step keywords: retry=N · on_fail=skip|abort · bind=NAME
 *
 * ★ GENERATED FROM THE REGISTRY: one shape for every op, so adding an op never changes this grammar.
 * ★ STRICT, UNLIKE THE JSON: an unknown keyword is an error. JSON tolerates unknown fields so builds at different
 *   versions interoperate; this text is typed by people, and a silently dropped `standof_m` is a bug.
 * ★ UNITS MUST MATCH THE FIELD: `budget_s=300s`, `turn_deg=90deg`, `standoff_m=0.8m`, or a bare number.
 * ★ IDS ARE IMPLIED: the parser assigns hierarchical ids (1, 2, 2.1) — what the resolver emits anyway.
 * ★ NUMBERS: std::from_chars to read, std::format to print — both locale-independent (CLAUDE.md, es_ES).
 */

#include <algorithm>
#include <charconv>
#include <expected>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include "mission_ir.h"

namespace rc::mission
{

namespace text_detail
{
struct Token
{
    enum class Kind { Name, Number, String, Sym, End };
    Kind kind = Kind::End;
    std::string text{};   ///< Name / Number digits / String contents / Sym spelling
    std::string unit{};   ///< Number: letters glued to it (s, m, deg)
};

inline bool name_start(char c) { return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c == '_'; }
inline bool name_char(char c) { return name_start(c) or (c >= '0' and c <= '9'); }
inline bool digit(char c) { return c >= '0' and c <= '9'; }

inline Error syntax(int line, std::string text)
{
    return {.code = "malformed", .where = std::format("line {}", line), .text = std::move(text)};
}

/// One line's tokens (comment stripped). Indentation is handled by the caller.
inline std::expected<std::vector<Token>, Error> tokenize(std::string_view s, int line)
{
    std::vector<Token> out;
    std::size_t i = 0;
    while (i < s.size())
    {
        const char c = s[i];
        if (c == ' ') { ++i; continue; }
        if (c == '\t') return std::unexpected(syntax(line, "tabs are not indentation; use spaces"));
        if (c == '#') break;
        if (c == '"')
        {
            std::string v;
            ++i;
            bool closed = false;
            while (i < s.size())
            {
                if (s[i] == '\\' and i + 1 < s.size()) { v += s[i + 1]; i += 2; continue; }
                if (s[i] == '"') { closed = true; ++i; break; }
                v += s[i++];
            }
            if (not closed) return std::unexpected(syntax(line, "unterminated string"));
            out.push_back({.kind = Token::Kind::String, .text = std::move(v)});
            continue;
        }
        if (digit(c) or ((c == '-' or c == '+' or c == '.') and i + 1 < s.size() and (digit(s[i + 1]) or s[i + 1] == '.')))
        {
            std::size_t j = i + 1;
            while (j < s.size() and (digit(s[j]) or s[j] == '.')) ++j;
            if (j + 1 < s.size() and (s[j] == 'e' or s[j] == 'E')
                and (digit(s[j + 1]) or ((s[j + 1] == '-' or s[j + 1] == '+') and j + 2 < s.size() and digit(s[j + 2]))))
            {
                j += 2;
                while (j < s.size() and digit(s[j])) ++j;
            }
            Token t{.kind = Token::Kind::Number, .text = std::string(s.substr(i, j - i))};
            std::size_t k = j;
            while (k < s.size() and name_char(s[k])) ++k;
            t.unit = std::string(s.substr(j, k - j));
            out.push_back(std::move(t));
            i = k;
            continue;
        }
        if (name_start(c))
        {
            std::size_t j = i;
            while (j < s.size() and name_char(s[j])) ++j;
            out.push_back({.kind = Token::Kind::Name, .text = std::string(s.substr(i, j - i))});
            i = j;
            continue;
        }
        if (c == '>' and i + 1 < s.size() and s[i + 1] == '=')
        {
            out.push_back({.kind = Token::Kind::Sym, .text = ">="});
            i += 2;
            continue;
        }
        if (std::string_view("=():,@$<").find(c) != std::string_view::npos)
        {
            out.push_back({.kind = Token::Kind::Sym, .text = std::string(1, c)});
            ++i;
            continue;
        }
        return std::unexpected(syntax(line, std::format("unexpected character '{}'", c)));
    }
    return out;
}

/// The unit a field's name declares: budget_s → s, standoff_m → m, turn_deg → deg, s → s; else none.
inline std::string_view unit_of(std::string_view field)
{
    if (field == "s" or field.ends_with("_s")) return "s";
    if (field.ends_with("_m")) return "m";
    if (field.ends_with("_deg")) return "deg";
    return "";
}

inline std::string fmt(double d) { return std::format("{}", d); }

class LineParser
{
public:
    LineParser(std::vector<Token> t, int line) : t_(std::move(t)), line_(line) {}

    [[nodiscard]] bool done() const { return i_ >= t_.size(); }
    [[nodiscard]] const Token& peek(std::size_t ahead = 0) const
    {
        static const Token end{};
        return i_ + ahead < t_.size() ? t_[i_ + ahead] : end;
    }
    [[nodiscard]] bool sym(std::string_view s, std::size_t ahead = 0) const
    {
        return peek(ahead).kind == Token::Kind::Sym and peek(ahead).text == s;
    }
    Token take() { return done() ? Token{} : t_[i_++]; }
    [[nodiscard]] Error error(std::string code, std::string text) const
    {
        return {.code = std::move(code), .where = std::format("line {}", line_), .text = std::move(text)};
    }
    [[nodiscard]] Error malformed(std::string text) const { return error("malformed", std::move(text)); }

    std::expected<std::string, Error> name(std::string_view what)
    {
        if (peek().kind != Token::Kind::Name) return std::unexpected(malformed(std::format("expected {}", what)));
        return take().text;
    }
    std::expected<void, Error> expect(std::string_view s)
    {
        if (not sym(s)) return std::unexpected(malformed(std::format("expected '{}'", s)));
        take();
        return {};
    }

    std::expected<double, Error> number(std::string_view unit, std::string_view field)
    {
        if (peek().kind != Token::Kind::Number)
            return std::unexpected(error("bad_type", std::format("{}: expected a number", field)));
        const Token t = take();
        if (not t.unit.empty() and t.unit != unit)
            return std::unexpected(error("bad_value", unit.empty() ? std::format("{} takes no unit, got '{}'", field, t.unit)
                                                                   : std::format("{} is in {}, got '{}'", field, unit, t.unit)));
        double d = 0;
        const char* b = t.text.data() + (t.text.starts_with('+') ? 1 : 0);
        const auto [p, ec] = std::from_chars(b, t.text.data() + t.text.size(), d);
        if (ec != std::errc{} or p != t.text.data() + t.text.size())
            return std::unexpected(malformed(std::format("'{}' is not a number", t.text)));
        return d;
    }

    std::expected<std::int64_t, Error> integer(std::string_view field)
    {
        if (peek().kind != Token::Kind::Number or not peek().unit.empty())
            return std::unexpected(error("bad_type", std::format("{}: expected an integer", field)));
        const Token t = take();
        std::int64_t n = 0;
        const auto [p, ec] = std::from_chars(t.text.data(), t.text.data() + t.text.size(), n);
        if (ec != std::errc{} or p != t.text.data() + t.text.size())
            return std::unexpected(error("bad_type", std::format("{}: '{}' is not an integer", field, t.text)));
        return n;
    }

    std::expected<Ref, Error> ref()
    {
        if (sym("$"))
        {
            take();
            auto n = name("a variable name");
            if (not n.has_value()) return std::unexpected(n.error());
            return Ref{.kind = Ref::Kind::Var, .name = *n};
        }
        auto head = name("a reference");
        if (not head.has_value()) return std::unexpected(head.error());
        if (not sym(":")) return Ref{.kind = Ref::Kind::Node, .name = *head};
        take();
        if (*head == "anchor")
        {
            auto n = name("an anchor name");
            if (not n.has_value()) return std::unexpected(n.error());
            return Ref{.kind = Ref::Kind::Anchor, .name = *n};
        }
        if (*head == "passage")
        {
            auto n = name("a passage id");
            if (not n.has_value()) return std::unexpected(n.error());
            Ref r{.kind = Ref::Kind::Passage, .name = *n};
            if (sym("@"))
            {
                take();
                auto room = name("the room the face is resolved in");
                if (not room.has_value()) return std::unexpected(room.error());
                r.from = *room;
            }
            return r;
        }
        if (*head == "pose")
        {
            auto room = name("a room");
            if (not room.has_value()) return std::unexpected(room.error());
            if (auto e = expect("("); not e.has_value()) return std::unexpected(e.error());
            std::vector<double> v;
            while (not sym(")"))
            {
                if (not v.empty())
                    if (auto e = expect(","); not e.has_value()) return std::unexpected(e.error());
                auto d = number("", "pose");
                if (not d.has_value()) return std::unexpected(d.error());
                v.push_back(*d);
            }
            take();
            if (v.size() < 2 or v.size() > 4) return std::unexpected(malformed("pose:ROOM(x, y[, yaw[, sigma_xy]])"));
            v.resize(4, 0.0);
            return Ref{.kind = Ref::Kind::Pose, .pose = {.room = *room, .x = v[0], .y = v[1], .yaw = v[2], .sigma_xy = v[3]}};
        }
        return std::unexpected(Error{.code = "unsupported", .name = "ref:" + *head, .where = std::format("line {}", line_),
                                     .text = "unknown reference kind"});
    }

    std::expected<Pred, Error> pred()
    {
        auto n = name("a predicate");
        if (not n.has_value()) return std::unexpected(n.error());
        const PredSpec* spec = pred_spec(*n);
        if (spec == nullptr)
            return std::unexpected(Error{.code = "unsupported", .name = "pred:" + *n, .where = std::format("line {}", line_),
                                         .text = "unknown predicate"});
        Pred p{.name = *n};
        if (auto e = expect("("); not e.has_value()) return std::unexpected(e.error());
        std::size_t k = 0;
        while (not sym(")"))
        {
            if (k > 0)
                if (auto e = expect(","); not e.has_value()) return std::unexpected(e.error());
            if (k >= spec->fields.size()) return std::unexpected(malformed(std::format("{} takes {} argument(s)", *n, spec->fields.size())));
            auto r = ref();
            if (not r.has_value()) return std::unexpected(r.error());
            p.refs.emplace(std::string(spec->fields[k].name), std::move(*r));
            ++k;
        }
        take();
        for (const auto& [s, op] : {std::pair{">=", Comparison::Op::Ge}, {"<", Comparison::Op::Lt}})
            if (sym(s))
            {
                take();
                auto d = number("", "a probability");
                if (not d.has_value()) return std::unexpected(d.error());
                p.cmp = Comparison{.op = op, .value = *d};
            }
        return p;
    }

    std::expected<Value, Error> value(const FieldSpec& f)
    {
        switch (f.type)
        {
            case FieldType::Number:
            {
                auto d = number(unit_of(f.name), f.name);
                if (not d.has_value()) return std::unexpected(d.error());
                return Value{*d};
            }
            case FieldType::Integer:
            {
                auto n = integer(f.name);
                if (not n.has_value()) return std::unexpected(n.error());
                return Value{*n};
            }
            case FieldType::String:
                if (peek().kind != Token::Kind::Name and peek().kind != Token::Kind::String)
                    return std::unexpected(error("bad_type", std::format("{}: expected a name or a \"string\"", f.name)));
                return Value{take().text};
            case FieldType::Ref:
            {
                auto r = ref();
                if (not r.has_value()) return std::unexpected(r.error());
                return Value{std::move(*r)};
            }
            case FieldType::Pred:
            {
                auto p = pred();
                if (not p.has_value()) return std::unexpected(p.error());
                return Value{std::move(*p)};
            }
        }
        return std::unexpected(malformed("unknown field type"));
    }

    /// op positional* (key=value)*
    std::expected<Step, Error> step()
    {
        auto op = name("an op");
        if (not op.has_value()) return std::unexpected(op.error());
        Step s{.op = *op};
        const OpSpec* spec = op_spec(s.op);
        if (spec == nullptr)
            return std::unexpected(Error{.code = "unsupported", .name = s.op, .where = std::format("line {}", line_),
                                         .text = "unknown op"});
        std::vector<const FieldSpec*> positional;
        for (const auto& f : spec->fields)
            if (f.positional) positional.push_back(&f);
        std::size_t next_positional = 0;
        bool keywords_started = false;
        while (not done())
        {
            const bool keyword = peek().kind == Token::Kind::Name and sym("=", 1);
            if (not keyword)
            {
                if (keywords_started) return std::unexpected(malformed("positional values come before key=value"));
                if (next_positional >= positional.size())
                    return std::unexpected(malformed(std::format("too many values for {}", s.op)));
                const FieldSpec& f = *positional[next_positional++];
                auto v = value(f);
                if (not v.has_value()) return std::unexpected(v.error());
                s.args.insert_or_assign(std::string(f.name), std::move(*v));
                continue;
            }
            keywords_started = true;
            const std::string key = take().text;
            take();   // '='
            if (key == "retry")
            {
                auto n = integer("retry");
                if (not n.has_value()) return std::unexpected(n.error());
                s.on_fail = {.kind = OnFail::Kind::Retry, .retries = int(*n)};
            }
            else if (key == "on_fail")
            {
                auto n = name("abort or skip");
                if (not n.has_value()) return std::unexpected(n.error());
                if (*n == "abort") s.on_fail = {};
                else if (*n == "skip") s.on_fail = {.kind = OnFail::Kind::Skip};
                else return std::unexpected(error("bad_value", "on_fail is abort or skip (or write retry=N)"));
            }
            else if (key == "bind")
            {
                auto n = name("a binding name");
                if (not n.has_value()) return std::unexpected(n.error());
                s.bind = *n;
            }
            else
            {
                const FieldSpec* f = nullptr;
                for (const auto& cand : spec->fields)
                    if (cand.name == key) f = &cand;
                if (f == nullptr) return std::unexpected(error("bad_value", std::format("{} has no field '{}'", s.op, key)));
                auto v = value(*f);
                if (not v.has_value()) return std::unexpected(v.error());
                s.args.insert_or_assign(key, std::move(*v));
            }
        }
        return s;
    }

private:
    std::vector<Token> t_;
    std::size_t i_ = 0;
    int line_;
};

// ── printing ─────────────────────────────────────────────────────────────────────────────────────────────────────

inline std::string quote(std::string_view s)
{
    std::string out = "\"";
    for (const char c : s)
    {
        if (c == '"' or c == '\\') out += '\\';
        out += c;
    }
    return out + "\"";
}

/// A string value: bare when it reads as a NAME, quoted otherwise.
inline std::string str(std::string_view s)
{
    const bool bare = not s.empty() and name_start(s.front()) and std::ranges::all_of(s, name_char);
    return bare ? std::string(s) : quote(s);
}

inline std::string ref_text(const Ref& r)
{
    switch (r.kind)
    {
        case Ref::Kind::Node: return r.name;
        case Ref::Kind::Anchor: return "anchor:" + r.name;
        case Ref::Kind::Passage: return "passage:" + r.name + (r.from.empty() ? "" : "@" + r.from);
        case Ref::Kind::Pose:
            return std::format("pose:{}({}, {}, {}, {})", r.pose.room, fmt(r.pose.x), fmt(r.pose.y), fmt(r.pose.yaw),
                               fmt(r.pose.sigma_xy));
        case Ref::Kind::Var: return "$" + r.name;
    }
    return {};
}

inline std::string pred_text(const Pred& p)
{
    std::string out = p.name + "(";
    bool first = true;
    if (const PredSpec* spec = pred_spec(p.name))
        for (const auto& f : spec->fields)
            if (const auto it = p.refs.find(f.name); it != p.refs.end())
            {
                out += (first ? "" : ", ") + ref_text(it->second);
                first = false;
            }
    out += ")";
    if (p.cmp.has_value()) out += (p.cmp->op == Comparison::Op::Ge ? ">=" : "<") + fmt(p.cmp->value);
    return out;
}

inline std::string value_text(const Value& v, std::string_view field)
{
    return std::visit(
        [field](const auto& x) -> std::string
        {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, double>) return fmt(x) + std::string(unit_of(field));
            else if constexpr (std::is_same_v<T, std::int64_t>) return std::format("{}", x);
            else if constexpr (std::is_same_v<T, std::string>) return str(x);
            else if constexpr (std::is_same_v<T, Ref>) return ref_text(x);
            else return pred_text(x);
        },
        v);
}

inline void print_steps(std::string& out, const std::vector<Step>& steps, int depth)
{
    for (const auto& s : steps)
    {
        std::string line(std::size_t(2 * depth), ' ');
        line += s.op;
        if (const OpSpec* spec = op_spec(s.op))
        {
            bool positional = true;   // positionals while they are contiguous; then everything by keyword
            std::vector<const FieldSpec*> keywords;
            for (const auto& f : spec->fields)
            {
                const auto it = s.args.find(f.name);
                if (f.positional and positional)
                {
                    if (it == s.args.end()) { positional = false; continue; }
                    line += " " + value_text(it->second, f.name);
                    continue;
                }
                if (it != s.args.end()) keywords.push_back(&f);
            }
            for (const FieldSpec* f : keywords) line += std::format(" {}={}", f->name, value_text(s.args.find(f->name)->second, f->name));
        }
        if (s.on_fail.kind == OnFail::Kind::Retry) line += std::format(" retry={}", s.on_fail.retries);
        if (s.on_fail.kind == OnFail::Kind::Skip) line += " on_fail=skip";
        if (s.bind.has_value()) line += " bind=" + *s.bind;
        out += line + "\n";
        print_steps(out, s.steps, depth + 1);
    }
}
}   // namespace text_detail

/// Parse + assign ids + validate. A returned mission is one the controller may run.
[[nodiscard]] inline std::expected<Mission, Error> parse_text(std::string_view text)
{
    using namespace text_detail;
    Mission m;
    bool have_mission_line = false, in_steps = false;
    std::vector<std::pair<int, std::vector<Step>*>> levels;   // (indent, the list steps at that indent go into)
    int line_no = 0;
    std::size_t pos = 0;
    while (pos <= text.size())
    {
        const std::size_t eol = std::min(text.find('\n', pos), text.size());
        const std::string_view raw = text.substr(pos, eol - pos);
        pos = eol + 1;
        ++line_no;
        std::size_t indent = 0;
        while (indent < raw.size() and raw[indent] == ' ') ++indent;
        auto toks = tokenize(raw, line_no);
        if (not toks.has_value()) return std::unexpected(toks.error());
        if (toks->empty()) continue;   // blank or comment-only
        LineParser lp(std::move(*toks), line_no);

        if (indent == 0)
        {
            if (in_steps) return std::unexpected(syntax(line_no, "header lines (column 0) come before the steps"));
            auto kw = lp.name("mission, anchor, requires or pref");
            if (not kw.has_value()) return std::unexpected(kw.error());
            if (not have_mission_line)
            {
                if (*kw != "mission") return std::unexpected(syntax(line_no, "a mission starts with `mission`"));
                have_mission_line = true;
                if (lp.peek().kind == Token::Kind::String) m.title = lp.take().text;
            }
            else if (*kw == "anchor")
            {
                auto n = lp.name("an anchor name");
                if (not n.has_value()) return std::unexpected(n.error());
                if (auto e = lp.expect("="); not e.has_value()) return std::unexpected(e.error());
                auto r = lp.ref();
                if (not r.has_value()) return std::unexpected(r.error());
                if (r->kind != Ref::Kind::Pose) return std::unexpected(syntax(line_no, "an anchor is a pose:ROOM(x, y, …)"));
                m.anchors.insert_or_assign(*n, r->pose);
            }
            else if (*kw == "requires")
                while (not lp.done())
                {
                    auto f = lp.name("a field name");
                    if (not f.has_value()) return std::unexpected(f.error());
                    m.requires_fields.push_back(*f);
                }
            else if (*kw == "pref")
            {
                auto k = lp.peek().kind == Token::Kind::String ? std::expected<std::string, Error>(lp.take().text)
                                                                : lp.name("a preference");
                if (not k.has_value()) return std::unexpected(k.error());
                auto w = lp.number("", "pref");
                if (not w.has_value()) return std::unexpected(w.error());
                m.prefs.insert_or_assign(*k, *w);
            }
            else return std::unexpected(syntax(line_no, std::format("unknown header line '{}'", *kw)));
            if (not lp.done()) return std::unexpected(syntax(line_no, "unexpected text at the end of the line"));
            continue;
        }

        if (not have_mission_line) return std::unexpected(syntax(line_no, "a mission starts with `mission`"));
        in_steps = true;
        auto s = lp.step();
        if (not s.has_value()) return std::unexpected(s.error());
        const int ind = int(indent);
        if (levels.empty()) levels.push_back({ind, &m.steps});
        else if (ind > levels.back().first)
        {
            if (levels.back().second->empty()) return std::unexpected(syntax(line_no, "indented under nothing"));
            levels.push_back({ind, &levels.back().second->back().steps});
        }
        else
        {
            while (not levels.empty() and levels.back().first > ind) levels.pop_back();
            if (levels.empty() or levels.back().first != ind)
                return std::unexpected(syntax(line_no, "this indentation matches no enclosing step"));
        }
        levels.back().second->push_back(std::move(*s));
    }
    if (not have_mission_line) return std::unexpected(syntax(1, "a mission starts with `mission`"));
    assign_ids(m);
    if (auto e = validate(m); e.has_value()) return std::unexpected(*e);
    return m;
}

/// The canonical text form. parse_text(print_text(m)) == m for every mission validate() accepts whose ids are the
/// hierarchical ones assign_ids gives (the text does not carry ids).
[[nodiscard]] inline std::string print_text(const Mission& m)
{
    using namespace text_detail;
    std::string out = "mission";
    if (not m.title.empty()) out += " " + quote(m.title);   // a title is always a "string"
    out += "\n";
    for (const auto& [name, p] : m.anchors) out += std::format("anchor {} = {}\n", name, ref_text(Ref{.kind = Ref::Kind::Pose, .pose = p}));
    if (not m.requires_fields.empty())
    {
        out += "requires";
        for (const auto& f : m.requires_fields) out += " " + f;
        out += "\n";
    }
    for (const auto& [k, w] : m.prefs) out += std::format("pref {} {}\n", str(k), fmt(w));
    print_steps(out, m.steps, 1);
    return out;
}

}   // namespace rc::mission
