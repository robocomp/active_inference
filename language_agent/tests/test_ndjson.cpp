// NDJSON splitter on partial chunks. Build+run: tests/run_tests.sh
#include "../src/ndjson_splitter.h"
#include <cassert>
#include <print>
#include <string>

int main()
{
    using rc::chat::NdjsonSplitter;
    int checks = 0;
    {   // one line split across three chunks, including mid-UTF-8 ("á" = C3 A1)
        NdjsonSplitter s;
        const std::string line = R"({"message":{"content":"está"},"done":false})";
        const auto cut1 = line.find("\xC3") + 1;   // between the two bytes of 'á'
        assert(s.feed(line.substr(0, 5)).empty());
        assert(s.feed(line.substr(5, cut1 - 5)).empty());
        auto out = s.feed(line.substr(cut1) + "\n");
        assert(out.size() == 1 and out[0] == line); checks += 3;
        assert(s.pending_bytes() == 0); ++checks;
    }
    {   // several lines in one chunk, the last incomplete; CRLF; blank lines skipped
        NdjsonSplitter s;
        auto out = s.feed("{\"a\":1}\r\n\n{\"b\":2}\n{\"c\":");
        assert(out.size() == 2 and out[0] == "{\"a\":1}" and out[1] == "{\"b\":2}"); ++checks;
        out = s.feed("3}\n");
        assert(out.size() == 1 and out[0] == "{\"c\":3}"); ++checks;
    }
    {   // unterminated final line comes back from flush(); a blank remainder does not
        NdjsonSplitter s;
        assert(s.feed("{\"done\":true}").empty()); ++checks;
        auto rest = s.flush();
        assert(rest.has_value() and *rest == "{\"done\":true}"); ++checks;
        assert(not s.flush().has_value()); ++checks;
        assert(s.feed("  \n").empty()); ++checks;
        assert(not s.flush().has_value()); ++checks;
    }
    {   // byte-at-a-time delivery
        NdjsonSplitter s;
        const std::string body = "{\"x\":1}\n{\"y\":2}\n";
        int lines = 0;
        for (char c : body) lines += static_cast<int>(s.feed(std::string(1, c)).size());
        assert(lines == 2); ++checks;
    }
    std::println("test_ndjson: {} checks passed", checks);
    return 0;
}
