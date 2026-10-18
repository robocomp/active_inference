# Episodic memory — cortex attributes to register (Task 7, user gate)

Add to `cortex/core/include/dsr/core/types/type_checking/dsr_attr_name.h`, then reinstall cortex:

```cpp
REGISTER_TYPE(em_journal, std::string, false)  // controller agent node: episodic journal wire (common/episodic/em_journal.h)
REGISTER_TYPE(em_record,  std::string, false)  // domain-4 episode node: one Record as JSON (common/episodic/em_schema.h)
REGISTER_TYPE(em_stats,   std::string, false)  // domain-4 stats node: one Stats as JSON (common/episodic/em_consolidate.h)
```

Verify: `grep -n "em_journal\|em_record\|em_stats" /usr/local/include/dsr/core/types/type_checking/dsr_attr_name.h` → 3 lines.
