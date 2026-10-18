// episodic_agent's per-run metrics CSV (src/em_metrics.h): header, one row per period, classic decimal point under
// es_ES (CLAUDE.md), per-interval maxima reset after each row.
#include "../src/em_metrics.h"

#include <clocale>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

static int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (not(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

int main()
{
    if (not std::setlocale(LC_ALL, "es_ES.UTF-8")) std::setlocale(LC_ALL, "");
    const auto dir = std::filesystem::temp_directory_path() / ("em_metrics_test_" + std::to_string(::getpid()));
    std::filesystem::remove_all(dir);
    const auto file = rc::em::Metrics::path_for(dir, 1792238400123ULL);
    CHECK(file.parent_path() == dir and file.filename().string().starts_with("metrics_")
          and file.filename().string().size() == std::string("metrics_YYYYMMDD_HHMMSS.csv").size());
    {
        rc::em::Metrics m(file, 1000, 10000);
        CHECK(m.ok());
        rc::em::Record r;
        r.id = "20261017-1";
        r.t0 = 1000;
        r.events = {{.dt_ms = 1, .kind = "born", .node = "a"}, {.dt_ms = 2, .kind = "died", .node = "a"}};
        rc::em::Record c = r;
        c.id = "20261017-2";
        c.t1 = 2000;
        c.events.clear();
        m.on_wire(500);
        m.on_wire(700);
        m.on_written(std::vector<rc::em::Record>{r, c});
        m.on_poll(1.5);
        CHECK(not m.maybe_row(5000, {}));                          // not due yet
        CHECK(m.maybe_row(11000, std::vector<rc::em::Record>{r}));
        CHECK(m.maybe_row(21000, {}));                             // next period, nothing happened
        CHECK(not m.maybe_row(25000, {}));
    }
    std::ifstream f(file);
    std::vector<std::string> lines;
    for (std::string l; std::getline(f, l);) lines.push_back(l);
    CHECK(lines.size() == 3);
    if (lines.size() == 3)
    {
        CHECK(lines[0] == rc::em::Metrics::kHeader);
        CHECK(lines[1].starts_with("11000,700,2,1,1,"));
        CHECK(lines[1].ends_with(",2,1.5"));                       // max events 2; decimal POINT under es_ES
        CHECK(lines[2] == "21000,700,0,0,1,0,0,1.5");             // per-interval counters reset
    }
    std::filesystem::remove_all(dir);
    if (failures) { std::fprintf(stderr, "metrics_test: %d failure(s)\n", failures); return 1; }
    std::puts("metrics_test: all passed");
    return 0;
}
