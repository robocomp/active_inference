### Task 1: The `Card` type and its JSON

**Files:**
- Create: `common/perception_card/card.h`
- Test: `common/perception_card/tests/card_test.cpp`, registered in `common/perception_card/CMakeLists.txt`
  (`add_test`, as in `common/media_transport/CMakeLists.txt`)

**Interfaces (produces):**
- `enum class Level { Mass, Countable, Articulated, Animate };`
- `enum class Role { Identity, Occupancy, Both };`
- `struct Channel { std::string sensor; Role role; bool modelled; };`
  - Sensors are graph node names: `zed`, `helios`, `bpearl`, `ricoh`.
- `struct TermSpec { std::string name; std::map<std::string, float> params; };`
- `struct SubtypeParams { rc::detect::DetectorEnvelope env; float q_return = 1.0f; std::vector<TermSpec> terms; };`
  - `q_return` is the per-beam LiDAR return probability.
- `struct GeometryPrior { float w, d, h, z0, z1; };`
- The card itself:
  ```
  struct Card {
    int version = 1;
    std::string concept, graph_type;
    Level level;
    std::vector<Channel> channels;
    GeometryPrior geometry;
    std::map<std::string, SubtypeParams> subtypes;   // key "" = default
    std::vector<std::string> instance_terms;
    std::vector<std::string> blind_spots;
    std::string calibration;                          // "prior" | "fitted" | "blind_spot_map"
    std::string calibration_date;
    std::string blind_spot_map_path;
  };
  ```
- `std::string to_json(const Card&)`
- `inline constexpr std::array kTermNames{"envelope","in_fov","central","range_conf","leaf_obliquity","resolvability","kappa","coverage","motion"};`
  - This is the closed list, defined here. Task 2 implements one function per name.
- `std::expected<Card, std::string> card_from_json(std::string_view)`
  - Rejects: a term name not in `kTermNames`; a missing level; an unknown role.
- `const SubtypeParams& params_for(const Card&, std::string_view subtype)`, which falls back to the `""` entry.
- **JSON facility:** reuse the one `common/media_transport/media_transport.cpp` uses for descriptors (its
  `from_chars`-based number reads). Add no new JSON library.

Steps:
- [ ] **Step 1: write `card_test.cpp`.**
  - A full card round-trips through `to_json`/`card_from_json`, every field equal.
  - An unknown term `"magic"` → error containing `"magic"`.
  - A missing `level` → error.
  - `params_for(c, "round")` returns the round entry; `params_for(c, "oval")` returns the default.
  - The whole test runs after `CHECK(std::setlocale(LC_ALL, "") != nullptr)`.
- [ ] **Step 2: run it and see it fail** (it doesn't compile yet).
- [ ] **Step 3: implement `card.h`.**
- [ ] **Step 4: build and run the test.** Expected: `ALL OK` under `LANG=es_ES.UTF-8`.
- [ ] **Step 5: commit** `perception_card: Card type + JSON (closed term names)`.

