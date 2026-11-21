// config_migration_test: the three class folders (nightstand, bed, toilet) moved their configs onto countable_concept's
// sections WITHOUT CHANGING ONE VALUE, and a config still using a pre-countable section is refused.
//
//  1. For each class: every key of the PRE-MIGRATION config (read from git at BASE_REV, the commit before Task 4) maps
//     through countable_concept/KEYMAP.md — written from bed's names, the class word substituted (BedModel.X ->
//     NightstandModel.X, BedConcept.BedFilterLog -> NightstandConcept.NightstandFilterLog, ...) — to a key of the
//     migrated config with the SAME value text (trimmed, trailing `# comment` stripped; nothing is reparsed, so 0.10
//     vs 0.1 cannot even arise). Keys outside the renamed sections keep their name. The migrated config holds no key
//     that did not come from an old one, except the new required `[Concept] Class = "<obj>"`.
//     Excluded: <Cls>Model.HeadboardHeightM / HeadboardThickM (bed only) — the reader stopped reading them in Task 3
//     and they were deleted from the migrated config, so they must be ABSENT there; the headboard is the manifest's
//     [prior.headboard], asserted here to be exactly 1.25 / 0.10.
//  2. rc::pre_countable_section on a ConfigLoader that loaded a temporary config with [BedModel] names it, and
//     rc::refuse_pre_countable_config throws rc::CountableConfigError carrying the KEYMAP pointer; a forked child
//     handles it exactly as generated/main.cpp does (print what(), exit non-zero), and the parent checks both.
//
// Run under the agent locale:  LANG=es_ES.UTF-8 ./config_migration_test
#include "countable_sections.h"
#include "back_panel.h"

#include <cctype>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

namespace {

std::string trim(std::string_view s)
{
    const auto b = s.find_first_not_of(" \t\r");
    if (b == std::string_view::npos) return {};
    return std::string(s.substr(b, s.find_last_not_of(" \t\r") - b + 1));
}

// The value text of a `key = value   # comment` line: trimmed, the trailing comment (a '#' outside quotes) removed.
std::string value_text(std::string_view v)
{
    bool in_str = false;
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        if (v[i] == '"') in_str = not in_str;
        else if (v[i] == '#' and not in_str) { v = v.substr(0, i); break; }
    }
    return trim(v);
}

// section.key -> value text. Textual, line by line: these configs have no multi-line values.
std::map<std::string, std::string> read_values(std::istream& in)
{
    std::map<std::string, std::string> out;
    std::string line, section;
    while (std::getline(in, line))
    {
        const std::string l = trim(line);
        if (l.empty() or l.front() == '#') continue;
        if (l.front() == '[') { section = trim(std::string_view(l).substr(1, l.find(']') - 1)); continue; }
        const auto eq = l.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = trim(std::string_view(l).substr(0, eq));
        if (out.contains(section + "." + key)) std::printf("  duplicate key %s.%s\n", section.c_str(), key.c_str());
        out[section + "." + key] = value_text(std::string_view(l).substr(eq + 1));
    }
    return out;
}

std::string run(const std::string& cmd)
{
    std::string out;
    if (FILE* p = ::popen(cmd.c_str(), "r"))
    {
        char buf[4096];
        while (const auto n = std::fread(buf, 1, sizeof buf, p)) out.append(buf, n);
        ::pclose(p);
    }
    return out;
}

// KEYMAP.md: `Old.Key -> New.Key`, bed's names.
std::map<std::string, std::string> read_keymap(const std::string& path)
{
    std::map<std::string, std::string> km;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() or line.front() == '#') continue;
        const auto a = line.find(" -> ");
        if (a == std::string::npos) continue;
        km[trim(std::string_view(line).substr(0, a))] = trim(std::string_view(line).substr(a + 4));
    }
    return km;
}

// "Bed" -> the class word, at the start of the section and at the start of the key (BedConcept.BedFilterLog ->
// NightstandConcept.NightstandFilterLog). Only a leading "Bed" followed by an upper-case letter is the class word.
std::string with_class_word(const std::string& bed_key, const std::string& word)
{
    const auto sub = [&](std::string s) {
        if (s.size() > 3 and s.starts_with("Bed") and std::isupper(static_cast<unsigned char>(s[3]))) s = word + s.substr(3);
        return s;
    };
    const auto dot = bed_key.find('.');
    return sub(bed_key.substr(0, dot)) + "." + sub(bed_key.substr(dot + 1));
}

void check_class(const std::string& obj, const std::map<std::string, std::string>& bed_km)
{
    std::string word = obj;
    word[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(word[0])));
    std::map<std::string, std::string> km;
    for (const auto& [o, n] : bed_km) km[with_class_word(o, word)] = n;

    const std::string rel = obj + "_concept/etc/config.toml";
    std::istringstream old_in(run("git -C '" REPO_DIR "' show " BASE_REV ":" + rel));
    const auto old_v = read_values(old_in);
    std::ifstream new_in(std::string(REPO_DIR) + "/" + rel);
    const auto new_v = read_values(new_in);
    CHECK(old_v.size() > 150 and new_v.size() > 150);

    const std::set<std::string> excluded = {word + "Model.HeadboardHeightM", word + "Model.HeadboardThickM"};
    // keys born after the migration base (class existence, spec 2026-11-14), not renames of an old one
    std::set<std::string> produced = {"Concept.Class", "Model.ClassAltStdM", "Model.BirthHeightNoiseM", "Model.BirthHeightOutlierFrac",
                                      "Tracker.BirthAdmitPlausibility",    // final review I1: admission gate off
                                      "Model.ExistenceCarveParts",         // carve-parts (2026-11-15), no old key
                                      "Model.LabelDumpEnabled",            // eps_px measurement (plan Task 15b), temporary
                                      "Model.LabelDumpMarginM",            // eps_px far-tail margin (plan Task 15), temporary
                                      "Model.MaskPixelErrorRate",          // eps_px itself (spec 7c.8 a0), measured per class
                                      "Model.FloorSigmaM",                 // the floor's known-surface sigma (spec 7c.8 rev 3.3.8)
                                      "Model.LidarMeasuresOnGated"};       // lidar-measures (spec 2026-11-15 §7c), no old key
    // keys removed by the class factor (spec 2026-11-14 countable class existence): the plaus_evidence accumulator's
    // clamp, its tanh gain into the existence log-odds, and the shape LLR's alternative footprint (class_log_evidence
    // reads the manifest's class prior and Model.ClassAltStdM instead). Present in the old config, ABSENT in the new.
    const std::set<std::string> removed = {"Model.PlausClamp", "Model.PlausToExistenceGain", "Model.AltSizeScale"};
    int moved = 0, same = 0, renamed = 0;
    for (const auto& [ok, ov] : old_v)
    {
        std::string nk = ok;
        if (const auto it = km.find(ok); it != km.end()) { nk = it->second; ++renamed; }
        CHECK(not (ok.starts_with(word + "Model.") or ok.starts_with(word + "Concept.")) or km.contains(ok));
        produced.insert(nk);
        if (removed.contains(nk))
        {
            if (new_v.contains(nk)) { std::printf("  %s: %s is still in the config (removed by the class factor)\n", obj.c_str(), nk.c_str()); ++bad; }
            continue;
        }
        if (excluded.contains(ok))
        {
            if (new_v.contains(nk)) { std::printf("  %s: %s is still in the config (the manifest holds it)\n", obj.c_str(), nk.c_str()); ++bad; }
            continue;
        }
        const auto it = new_v.find(nk);
        if (it == new_v.end()) { std::printf("  %s: %s -> %s MISSING\n", obj.c_str(), ok.c_str(), nk.c_str()); ++bad; continue; }
        if (it->second != ov)
        { std::printf("  %s: %s = %s  but  %s = %s\n", obj.c_str(), ok.c_str(), ov.c_str(), nk.c_str(), it->second.c_str()); ++bad; }
        ++moved; if (nk == ok) ++same;
    }
    for (const auto& [nk, nv] : new_v)
        if (not produced.contains(nk)) { std::printf("  %s: %s = %s came from no old key\n", obj.c_str(), nk.c_str(), nv.c_str()); ++bad; }
    CHECK(new_v.contains("Concept.Class") and new_v.at("Concept.Class") == "\"" + obj + "\"");
    std::set<std::string> stale;
    for (const auto& [nk, nv] : new_v)
    {
        const std::string sec = nk.substr(0, nk.find('.'));
        if ((sec.ends_with("Model") or sec.ends_with("Concept")) and sec != "Model" and sec != "Concept") stale.insert(sec);
    }
    for (const auto& s : stale) std::printf("  %s: pre-countable section [%s] still present\n", obj.c_str(), s.c_str());
    CHECK(stale.empty());
    std::printf("%s: %zu old keys, %d renamed, %d kept their name, %d value-checked\n",
                obj.c_str(), old_v.size(), renamed, same, moved);
}

// Load `text` through the real ConfigLoader (a temporary .toml file) and return the loader.
ConfigLoader load_text(const std::string& text)
{
    const auto p = std::filesystem::temp_directory_path() / ("config_migration_test_" + std::to_string(::getpid()) + ".toml");
    { std::ofstream f(p); f << text; }
    ConfigLoader cfg;
    cfg.load(p.string());
    std::filesystem::remove(p);
    return cfg;
}

// Run refuse_pre_countable_config in a child that handles rc::CountableConfigError the way generated/main.cpp does
// (print what() to stderr, exit non-zero); return {exit status, what it wrote to stderr}.
std::pair<int, std::string> refuse_in_child(const ConfigLoader& cfg)
{
    int fd[2];
    if (::pipe(fd) != 0) return {-1, {}};
    std::fflush(stdout);   // or the child's exit() flushes the parent's buffered output a second time
    const pid_t pid = ::fork();
    if (pid == 0)
    {
        ::close(fd[0]); ::dup2(fd[1], 2);
        try { rc::refuse_pre_countable_config(cfg); rc::refuse_bad_beam_config(cfg); }
        catch (const rc::CountableConfigError& e) { std::fputs(e.what(), stderr); std::fflush(stderr); std::_Exit(EXIT_FAILURE); }
        std::_Exit(0);
    }
    ::close(fd[1]);
    std::string err;
    char buf[1024];
    while (true) { const auto n = ::read(fd[0], buf, sizeof buf); if (n <= 0) break; err.append(buf, static_cast<std::size_t>(n)); }
    ::close(fd[0]);
    int st = 0;
    ::waitpid(pid, &st, 0);
    return {WIFEXITED(st) ? WEXITSTATUS(st) : -1, err};
}

}  // namespace

int main()
{
    CHECK(std::setlocale(LC_ALL, "") != nullptr);
    const auto bed_km = read_keymap(std::string(REPO_DIR) + "/countable_concept/KEYMAP.md");
    CHECK(bed_km.size() > 100);
    for (const std::string obj : {"nightstand", "bed", "toilet"}) check_class(obj, bed_km);

    // The headboard left the config in Task 3: the bed manifest's [prior.headboard] is exactly 1.25 / 0.10, and the
    // two classes without a panel declare none.
    const std::string man = std::string(REPO_DIR) + "/common/concept_manifest/";
    bool opened = false;
    const auto bp = rc::manifest::load_back_panel(man + "bed.concept.toml", &opened);
    CHECK(opened and bp.has_value());
    CHECK(bp and bp->height_m == 1.25f and bp->thickness_m == 0.10f);
    for (const std::string obj : {"nightstand", "toilet"})
    {
        bool o = false;
        CHECK(not rc::manifest::load_back_panel(man + obj + ".concept.toml", &o).has_value() and o);
    }

    // The pre-countable section check, on the real ConfigLoader.
    const ConfigLoader old_style = load_text("[Agent]\nname = \"bed_concept\"\n[BedModel]\nSigmaObs = 0.05\n[Concept]\nClass = \"bed\"\n");
    CHECK(rc::pre_countable_section(old_style) == std::optional<std::string>("BedModel"));
    const ConfigLoader old_concept = load_text("[Agent]\nname = \"n\"\n[Model]\nSigmaObs = 0.05\n[NightstandConcept]\nKStable = 30\n");
    CHECK(rc::pre_countable_section(old_concept) == std::optional<std::string>("NightstandConcept"));
    const ConfigLoader good = load_text("[Agent]\nname = \"bed_concept\"\n[Component.Debug]\nx = 1\n[Model]\nSigmaObs = 0.05\n"
                                        "[Concept]\nClass = \"bed\"\n[Tracker]\nBirthFrames = 3\n");
    CHECK(not rc::pre_countable_section(good).has_value());
    const auto [st, err] = refuse_in_child(old_style);
    std::printf("refused: exit %d, stderr: %s", st, err.c_str());
    CHECK(st != 0);
    CHECK(err.find("config uses pre-countable section [BedModel] — see countable_concept/KEYMAP.md") != std::string::npos);
    const auto [st_ok, err_ok] = refuse_in_child(good);
    CHECK(st_ok == 0 and err_ok.empty());
    // Smooth-beam config validation (spec 7c.8): eps_px in (0, 1/2), p0 c^2 > 1; only checked with LidarMeasuresOnGated.
    const std::string on = "[Agent]\nname = \"n\"\n[Model]\nLidarMeasuresOnGated = true\nLidarPrecision = 2500.0\nLidarRobustCM = 0.05\n";
    const auto beam_refused = [&](const std::string& extra) {
        const ConfigLoader c = load_text(on + extra);
        return refuse_in_child(c);
    };
    {
        const auto ok = beam_refused("MaskPixelErrorRate = 0.094\nFloorSigmaM = 0.005\n");
        CHECK(ok.first == 0 and ok.second.empty());
        for (const char* bad_fl : {"MaskPixelErrorRate = 0.094\n", "MaskPixelErrorRate = 0.094\nFloorSigmaM = 0.0\n",
                                   "MaskPixelErrorRate = 0.094\nFloorSigmaM = -0.01\n"})
        {   // rev 3.3.8: FloorSigmaM has no default with the flag on
            const auto r = beam_refused(bad_fl);
            CHECK(r.first != 0);
            CHECK(r.second.find("FloorSigmaM") != std::string::npos);
        }
        for (const char* bad_eps : {"MaskPixelErrorRate = 0.0\n", "MaskPixelErrorRate = 0.5\n", "MaskPixelErrorRate = 0.7\n", ""})
        {
            const auto r = beam_refused(bad_eps);
            CHECK(r.first != 0);
            CHECK(r.second.find("MaskPixelErrorRate") != std::string::npos);
        }
        const ConfigLoader weak = load_text("[Agent]\nname = \"n\"\n[Model]\nLidarMeasuresOnGated = true\nLidarPrecision = 300.0\n"
                                            "LidarRobustCM = 0.05\nMaskPixelErrorRate = 0.094\n");
        const auto r = refuse_in_child(weak);
        CHECK(r.first != 0 and r.second.find("p0*c^2") != std::string::npos);
        // flag off: nothing is checked
        const ConfigLoader off = load_text("[Agent]\nname = \"n\"\n[Model]\nLidarMeasuresOnGated = false\nLidarPrecision = 300.0\n");
        const auto r_off = refuse_in_child(off);
        CHECK(r_off.first == 0);
        // the sensor's own sigma lowers p0: 2500 with sigma_sensor 0.03 m => 1/p0 = 4e-4 + 9e-4, p0 c^2 = 1.92 > 1 still ok; 0.06 m is not
        CHECK(not rc::beam_config_problem(0.094, 2500.0, 0.05, 0.03).has_value());
        CHECK(rc::beam_config_problem(0.094, 2500.0, 0.05, 0.06).has_value());
    }
    // Each migrated class config passes the check.
    for (const std::string obj : {"nightstand", "bed", "toilet"})
    {
        ConfigLoader c;
        c.load(std::string(REPO_DIR) + "/" + obj + "_concept/etc/config.toml");
        CHECK(not rc::pre_countable_section(c).has_value());
        CHECK(not rc::beam_config_problem(c.get<double>("Model.MaskPixelErrorRate"), c.get<double>("Model.LidarPrecision"),
                                          c.get<double>("Model.LidarRobustCM")).has_value());
        CHECK(not rc::floor_sigma_problem(c.get<double>("Model.FloorSigmaM")).has_value());
    }

    std::printf(bad ? "config_migration_test: %d FAILED\n" : "config_migration_test: OK\n", bad);
    return bad != 0;
}
