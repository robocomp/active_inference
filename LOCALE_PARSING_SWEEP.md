# Locale-safe number parsing — open sweep + generator status

The RULE lives in [CLAUDE.md](CLAUDE.md) ("Parsing numbers from files: `std::from_chars` ONLY").
This file carries the parts that change over time: which sites are still unfixed, and what the
code generator now does about it.

## ⚠ PENDING FLEET-WIDE SWEEP (open as of 2026-08-03)

`retina/src/depth_dataset.cpp` and ALL of `human_concept` are fixed. **Every site still listed
below is locale-unsafe and silently truncating on these machines.** Sweep them; the fix is mechanical
(`std::from_chars` + `imbue(std::locale::classic())` on writers), but VERIFY each against real data
rather than substituting blind — some of these feed beliefs, and a truncated input may have been
silently shaping results for a long time.

- ✅ `human_concept` — DONE 2026-08-03. Shared parser `human_concept/cpp/core/csv_parse.h`
  (`rc::csv::parse_float/parse_int/parse_long`, `from_chars`). **Copy this header's approach for the
  remaining sites.**
- `controller/tools/mppi_bench.cpp:240`, `controller/tools/route_bench.cpp:378` — `std::stof` on CSV
  axis values.
- `common/media_transport/bench/media_bench.cpp:66` — `std::atof` on `--fps` (a `0.5` becomes `0`).

Re-run the grep before assuming this list is complete:
`grep -rn "strtof\|strtod\|\batof\b\|std::stof\|std::stod" --include=*.cpp --include=*.h .`

## The generator now defuses this at the source (2026-08-05)

`robocompdsl`'s C++ `main.cpp` template (`robocomp_tools/cli/src/robocompdsl/templates/templateCPP/
files/generated/main.cpp`) emits `std::setlocale(LC_NUMERIC, "C")` immediately after the
`Q*Application` constructor, so **newly generated components are born locale-safe** — the C library
and `std::ofstream` finally agree on the decimal separator. Verified against Qt6: before the line,
`strtof("0.626452")` returns `0` once the Qt ctor has run; after it, `0.626452`, and an
ofstream→strtof round-trip is clean. Only `LC_NUMERIC` is pinned — `LC_CTYPE`/`LC_TIME`/`LC_MESSAGES`
stay localised and Qt's own formatting (QLocale) is untouched.

⚠ **This does NOT retroactively fix the agents in this tree.** Their `generated/main.cpp` was emitted
by the old template and is hand-modified, so regenerating is not an option — the line has to be
added per agent, or the call site converted to `std::from_chars`. Prefer `from_chars` regardless:
it is locale-independent by definition and cannot regress if the pin is ever removed.
