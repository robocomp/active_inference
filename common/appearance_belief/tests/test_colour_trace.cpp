// test_colour_trace.cpp — one CSV row per colour update: the slice that came in and the belief + name after it.
// Build+run: common/appearance_belief/tests/run_tests.sh
#include <clocale>
#include <cstdio>
#include <locale>
#include <sstream>
#include <string>

#include "../colour_trace.h"

static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (not (c)) { ++fails; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

namespace
{
int columns(const std::string& line) { int n = 1; for (const char c : line) n += (c == ','); return n; }
}

int main()
{
    std::setlocale(LC_ALL, "");   // the agent runs under Qt's es_ES locale (comma decimals); the file must not care
    rc::appearance::AppearanceBelief a;
    rc::appearance::ColourSample in;
    in.chroma = {0.43f, 0.13f, 0.43f}; in.chroma_var = {1e-4f, 1e-4f, 1e-4f}; in.chroma_n = 30.0f;
    in.lum = 0.46f; in.lum_var = 1e-3f; in.lum_n = 30.0f; in.range = 1.25f;
    for (int i = 0; i < 10; ++i) { a.update(in.chroma, in.chroma_var, in.chroma_n); a.update_lum(in.lum, in.lum_var, in.lum_n); }

    std::ostringstream os;
    os.imbue(std::locale::classic());
    rc::appearance::write_colour_trace_header(os);
    rc::appearance::write_colour_trace_row(os, 1234567, "chair_34", in, a);
    const std::string s = os.str();
    const auto nl = s.find('\n');
    const std::string header = s.substr(0, nl), row = s.substr(nl + 1, s.size() - nl - 2);

    CHECK(s.back() == '\n');
    CHECK(columns(header) == columns(row));                 // every column has a value
    CHECK(row.starts_with("1234567,chair_34,"));
    CHECK(row.find("0.43") != std::string::npos);           // POINT decimals, not "0,43"
    CHECK(row.find("magenta") != std::string::npos);        // the top name is in the row
    CHECK(header.find("name1") != std::string::npos and header.find("p3") != std::string::npos);   // top 3 terms
    CHECK(header.find(",w,") != std::string::npos);        // the frame's probability of being this object
    CHECK(row.find(",1,") != std::string::npos);           // ... 1 for update()-fed frames

    std::printf("test_colour_trace: %d checks, %d failures\n", checks, fails);
    return fails == 0 ? 0 : 1;
}
