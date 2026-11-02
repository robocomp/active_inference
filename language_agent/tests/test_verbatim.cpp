// verbatim_reply.h: a refusal set by a tool is taken once, unchanged; no refusal = nothing to say verbatim.
#include "../src/verbatim_reply.h"
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL  %s  (line %d)\n", #c, __LINE__); ++fails; } } while (0)

int main()
{
    rc::chat::VerbatimReply v;
    CHECK(not v.pending() and not v.take().has_value());
    v.add("I do not know which room I am in, so I cannot imagine it here.");
    CHECK(v.pending());
    CHECK(v.take() == std::optional<std::string>("I do not know which room I am in, so I cannot imagine it here."));
    CHECK(not v.pending() and not v.take().has_value());      // taken once
    v.add(""); CHECK(not v.pending());                         // an empty result is not a refusal
    v.add("A."); v.add("B.");
    CHECK(v.take() == std::optional<std::string>("A.\nB."));
    std::puts(fails == 0 ? "all verbatim checks passed" : "verbatim checks FAILED");
    return fails ? 1 : 0;
}
