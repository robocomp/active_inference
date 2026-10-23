set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Shared ai_common::* libraries (see common/CMakeLists.txt). Placed FIRST so these targets carry their own flags.
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
LIST(APPEND LIBS
  ai_common::presence     # agent presence monitor + coordinator + concept_presence
  ai_common::core         # header-only: config_report (rc::cfg::Reader), ...
  ai_common::media        # zero-copy media plane (domain 7): subscriber factories
  ai_common::sensor_calib # trajectory harvester (pure Eigen)
)

LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/sensor_inputs.cpp
  ../src/dashboard_model.cpp
  ../src/specificworker_dashboard.cpp
)

LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/sensor_inputs.h
  ../src/dashboard_model.h
  ../src/dashboard_view.h
)

# libdsr_core defines symbols that libdsr_{api,gui} only reference; keep all libs (see bottle_concept).
add_link_options(-Wl,--no-as-needed)

# CRITICAL — match the prebuilt libdsr's Eigen ABI: NO -march=native, NO -DEIGEN_MAX_ALIGN_BYTES=0 (CLAUDE.md).
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)
