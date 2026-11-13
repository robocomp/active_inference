cmake_minimum_required(VERSION 3.16)
project(door_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Eigen3 REQUIRED)

# Perception card: make_card() against etc/config.toml; header-only, no Qt/DSR. Run: LANG=es_ES.UTF-8 ./door_card_test
add_executable(door_card_test door_card_test.cpp)
target_include_directories(door_card_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_definitions(door_card_test PRIVATE AGENT_DIR="${CMAKE_CURRENT_LIST_DIR}/..")
target_compile_options(door_card_test PRIVATE -O2)
target_link_libraries(door_card_test PRIVATE Eigen3::Eigen)
add_test(NAME door_card_test COMMAND door_card_test)

# Door memory codec: IdentityRow <-> door-node attributes; header-only, no Qt/DSR. Run: ./test_door_memory_note
add_executable(test_door_memory_note test_door_memory_note.cpp)
target_include_directories(test_door_memory_note PRIVATE ../src)
target_compile_options(test_door_memory_note PRIVATE -O1)
target_link_libraries(test_door_memory_note PRIVATE Eigen3::Eigen)
add_test(NAME door_memory_note COMMAND test_door_memory_note)

# Door memory on the node: write_memory_attrs read back from a standalone DSRGraph (private DDS domain). Run: ./test_door_memory_graph
find_package(Qt6 COMPONENTS Core REQUIRED)
add_executable(test_door_memory_graph test_door_memory_graph.cpp)
target_include_directories(test_door_memory_graph PRIVATE ../src)
target_compile_definitions(test_door_memory_graph PRIVATE SHADOW_JSON="${CMAKE_CURRENT_LIST_DIR}/../../robot_concept/shadow.json")
target_compile_options(test_door_memory_graph PRIVATE -O0)
target_link_libraries(test_door_memory_graph PRIVATE Eigen3::Eigen Qt6::Core dsr_api dsr_core fastcdr fastdds)
add_test(NAME door_memory_graph COMMAND test_door_memory_graph)
