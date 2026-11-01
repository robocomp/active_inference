// Standalone test for instance_identity.h (common/run_tests.sh). Runs under the es_ES locale the agents use
// (CLAUDE.md: a harness without setlocale answers a different question than the agent does).
#include "instance_identity.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <print>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using namespace rc::identity;

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::println(stderr, "FAIL {}:{}: {}", __FILE__, __LINE__, #c); ++failures; } } while (0)

static fs::path fresh_dir()
{
    static int k = 0;
    const fs::path d = fs::temp_directory_path() / ("instance_identity_test_" + std::to_string(::getpid()) + "_" + std::to_string(k++));
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

static void write_file(const fs::path& p, const std::string& text)
{
    std::ofstream f(p);
    f << text;
}

static std::string read_file(const fs::path& p)
{
    std::ifstream f(p);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

static void test_missing_file_starts_from_live()
{
    const auto f = fresh_dir() / "id.csv";
    auto c = InstanceCounter::open(f, 20);
    CHECK(c.status() == LoadStatus::missing);
    CHECK(c.issue(3) == 4);
    CHECK(c.issue(0) == 5);
}

static void test_reload_continues()
{
    const auto f = fresh_dir() / "id.csv";
    {
        auto c = InstanceCounter::open(f, 20);
        CHECK(c.issue(0) == 1);
        CHECK(c.issue(0) == 2);
    }
    auto c = InstanceCounter::open(f, 20);
    CHECK(c.status() == LoadStatus::ok);
    CHECK(c.last() == 2);
    CHECK(c.issue(0) == 3);
}

static void test_persist_before_return()
{
    const auto f = fresh_dir() / "id.csv";
    auto c = InstanceCounter::open(f, 20);
    const int n = c.issue(5);
    const std::string text = read_file(f);
    const auto pos = text.find("last ");
    CHECK(pos != std::string::npos);
    int on_disk = -1;
    if (pos != std::string::npos)
    {
        const char* b = text.data() + pos + 5;
        std::from_chars(b, text.data() + text.size(), on_disk);
    }
    CHECK(on_disk == n);
}

static void test_live_max_wins()
{
    const auto f = fresh_dir() / "id.csv";
    write_file(f, "agent_id 20\nlast 2\n");
    auto c = InstanceCounter::open(f, 20);
    CHECK(c.status() == LoadStatus::ok);
    CHECK(c.issue(9) == 10);
}

static void test_file_wins_over_live()   // the restart bug itself: chair_9 died before the restart
{
    const auto f = fresh_dir() / "id.csv";
    write_file(f, "agent_id 20\nlast 9\n");
    auto c = InstanceCounter::open(f, 20);
    CHECK(c.issue(2) == 10);
}

static void test_corrupt_file()
{
    for (const std::string bad : {"agent_id 20\nlast 7,0\n", "abc", "", "agent_id 20\n", "agent_id 20\nlast -3\n"})
    {
        const auto f = fresh_dir() / "id.csv";
        write_file(f, bad);
        auto c = InstanceCounter::open(f, 20);
        CHECK(c.status() == LoadStatus::corrupt);
        CHECK(c.issue(4) == 5);
    }
}

static void test_agent_mismatch_keeps_max()
{
    const auto f = fresh_dir() / "id.csv";
    write_file(f, "agent_id 7\nlast 12\n");
    auto c = InstanceCounter::open(f, 20);
    CHECK(c.status() == LoadStatus::agent_mismatch);
    CHECK(c.issue(3) == 13);
}

static void test_two_issues_consecutive()
{
    const auto f = fresh_dir() / "id.csv";
    {
        auto c = InstanceCounter::open(f, 20);
        CHECK(c.issue(0) == 1);
        CHECK(c.issue(0) == 2);
    }
    CHECK(InstanceCounter::open(f, 20).last() == 2);
}

static void test_make_uid()
{
    CHECK(make_uid(20, 7) == ((std::uint64_t{20} << 40) | 7u));
    CHECK(make_uid(20, 7) != make_uid(21, 7));
    CHECK(make_uid(20, 7) != make_uid(20, 8));
}

static void test_max_suffix()
{
    const std::vector<std::string> names{"chair_3", "chair_12", "table_40", "chair_x", "chair_", "chair_5b"};
    CHECK(max_suffix(names, "chair_") == 12);
    CHECK(max_suffix(std::vector<std::string>{}, "chair_") == 0);
}

static void test_unwritable_dir_still_issues()
{
    if (::geteuid() == 0) return;   // root ignores the permission bits
    const auto d = fresh_dir();
    ::chmod(d.c_str(), 0500);
    auto c = InstanceCounter::open(d / "id.csv", 20);
    CHECK(c.issue(0) == 1);
    CHECK(c.issue(0) == 2);
    ::chmod(d.c_str(), 0700);
}

int main()
{
    if (not std::setlocale(LC_ALL, "es_ES.UTF-8")) std::setlocale(LC_ALL, "");
    test_missing_file_starts_from_live();
    test_reload_continues();
    test_persist_before_return();
    test_live_max_wins();
    test_file_wins_over_live();
    test_corrupt_file();
    test_agent_mismatch_keeps_max();
    test_two_issues_consecutive();
    test_make_uid();
    test_max_suffix();
    test_unwritable_dir_still_issues();
    if (failures == 0) std::println("instance_identity_test: ok");
    return failures == 0 ? 0 : 1;
}
