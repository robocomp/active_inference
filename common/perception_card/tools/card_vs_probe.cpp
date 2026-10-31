// card_vs_probe: score a detect_probe CSV against the shared evaluator. See card_vs_probe.h.
//   card_vs_probe probe.csv [more.csv ...] [--agent-dir DIR] [--zed f,w,h] [--sigma ext,pos] [--csv out.csv]
// --agent-dir takes the agent's directory (the card is read from DIR/etc/perception_card.json by read_card_file),
// the same entry point the agents and the dashboard use -- simpler than rebuilding a card with an agent's make_card.
#include "card_vs_probe.h"
#include "perception_card/card_io.h"
#include <clocale>
#include <cstdlib>

static int usage()
{
    std::fprintf(stderr, "usage: card_vs_probe probe.csv... [--agent-dir DIR] [--zed f,w,h] [--sigma ext,pos] [--csv out.csv]\n"
                         "  --zed needs 3 numbers (focal px, width px, height px); --sigma needs 2 (extent m, position m)\n");
    return 2;
}

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");   // CLAUDE.md: reproduce the agents' locale; parsing is from_chars regardless
    namespace P = rc::card::probe;
    P::Options opt;
    std::vector<std::string> files;
    std::string csv_out, agent_dir;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        const auto two = [&](double& x, double& y, double& z, std::size_t need)
        {
            if (i + 1 >= argc) return false;
            const auto p = P::split(argv[++i]);
            return p.size() == need and P::num(p[0], x) and P::num(p[1], y) and (need < 3 or P::num(p[2], z));
        };
        if (a == "--agent-dir" and i + 1 < argc) agent_dir = argv[++i];
        else if (a == "--csv" and i + 1 < argc) csv_out = argv[++i];
        else if (a == "--zed") { if (not two(opt.zed_f, opt.zed_w, opt.zed_h, 3)) return usage(); }
        else if (a == "--sigma") { double e = 0, p = 0, z = 0; if (not two(e, p, z, 2)) return usage(); opt.sigma_extent = float(e); opt.sigma_pos = float(p); }
        else files.push_back(a);
    }
    if (files.empty()) return usage();
    std::optional<rc::card::Card> card;
    if (not agent_dir.empty())
    {
        card = rc::card::read_card_file(agent_dir);
        if (not card) std::fprintf(stderr, "no readable card under %s: geometric part skipped\n", agent_dir.c_str());
    }
    P::Log all;
    for (const auto& f : files)
    {
        auto l = P::read_log(f);
        if (not l.error.empty()) { std::fprintf(stderr, "%s\n", l.error.c_str()); return 1; }
        all.has_oyaw = all.rows.empty() ? l.has_oyaw : (all.has_oyaw and l.has_oyaw);
        all.rows.insert(all.rows.end(), l.rows.begin(), l.rows.end());
    }
    const auto rep = P::analyse(all, card ? &*card : nullptr, opt);
    P::print(stdout, files.front(), rep);
    if (not csv_out.empty() and not P::write_csv(csv_out, files.front(), rep)) return 1;
    return 0;
}
