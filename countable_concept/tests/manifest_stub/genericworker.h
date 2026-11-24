// Test stub for <genericworker.h>: concept_manifest.h includes it only for ConfigLoader, which the agent's generated
// genericworker.h pulls in alongside Qt/Ice. manifest_load_test compiles the REAL ConfigLoader.cpp ($ROBOCOMP/classes),
// as the agent does, so the manifest readers run on the agent's code without the agent's Qt/DSR dependencies.
#pragma once
#include <ConfigLoader/ConfigLoader.h>
