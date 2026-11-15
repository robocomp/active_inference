// g++ -std=c++23 -O0 -g -Wall -Wextra common/ltsm_topology/tests/test_door_memory.cpp -o test_door_memory && ./test_door_memory
// door_memory.h: the merge keeps what only grows; door numbers parse from live names.
#include "../door_memory.h"

#include <cassert>
#include <print>
#include <string>
#include <vector>

using rc::doormem::DoorMemory;

int main()
{
    // the counter is the larger; the hazard comes whole from the record with more exposure (never mixed)
    const DoorMemory mem{27, 0, 5.28}, agent{12, 1, 9.0};
    const auto m = rc::doormem::merged(mem, agent);
    assert(m.next_index == 27 and m.changes == 1 and m.exposure_s == 9.0);
    assert(rc::doormem::merged(agent, mem) == m);                       // symmetric
    assert(rc::doormem::merged(DoorMemory{}, mem) == mem);              // a fresh agent (prior state) never overwrites memory
    assert(rc::doormem::merged(DoorMemory{3, 1, 4}, DoorMemory{3, 2, 4}).changes == 2);   // tie on exposure: more changes

    assert(rc::doormem::door_number("door_5") == 5u and rc::doormem::door_number("door_5_h1") == 5u);
    assert(rc::doormem::door_number("door_17_entry") == 17u);
    assert(not rc::doormem::door_number("door_x") and not rc::doormem::door_number("door_") and not rc::doormem::door_number("room_1"));
    assert(not rc::doormem::door_number("door_5x") and not rc::doormem::door_number("remembered_door_5"));
    const std::vector<std::string> names{"door_3", "room_2", "door_24_entry", "door_9"};
    assert(rc::doormem::next_index_floor(names) == 25u and rc::doormem::next_index_floor(std::vector<std::string>{}) == 0u);
    std::println("test_door_memory: ok");
}
