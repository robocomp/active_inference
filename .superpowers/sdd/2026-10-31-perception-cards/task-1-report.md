# Task 1 report: Card type + JSON

Status: DONE_WITH_CONCERNS. See the commit on feat/perception-cards.

Files: common/perception_card/card.h, tests/card_test.cpp, CMakeLists.txt (Release, -O2, add_test);
common/media_transport/flat_json.h (new); common/media_transport/media_transport.cpp (moved code out).

JSON facility: media_transport's parse_flat_json/as_float/fmt_float were in an anonymous namespace; moved
verbatim into media_transport/flat_json.h (rc::media::flat), media_transport.cpp now `using`s them. Verified
sensor_model_test and descriptor_roundtrip still build and print ALL OK.
The facility is FLAT only, so the nested card is written as one flat object with dotted keys
(channel.N.sensor, subtype.N.env.min_fill, subtype.N.term.M.p.<param>, counts for lists) and string lists as a
';'-joined quoted string. Reader rejects unknown term (subtype or instance_terms), missing/unknown level, unknown role.

Test: `cd common/perception_card && bash -ic 'cbuild -j8' && LANG=es_ES.UTF-8 ./build/card_test`
Output: `LC_NUMERIC = es_ES.UTF-8` / `ALL OK`.

Decisions / concerns:
- Field `concept` renamed `concept_` (C++20 keyword); JSON key is still "concept".
- Missing geometry/subtype numbers are an error; a missing "version" defaults to 1.
- params_for returns a static empty SubtypeParams if neither the subtype nor "" exists.
- TDD red step: first compile failed on my own bug (namespace) rather than on a missing header; test and impl were written together, so the red run was not a clean "missing header" failure.
- Other agents' includes need both common/ and common/media_transport on the include path (card.h includes "media_transport/flat_json.h" and "detectability/detectability.h").
