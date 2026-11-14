/*
 * countable_config_error.h — the one exception a bad countable_concept config raises.
 *
 * The config is read inside SpecificWorker's constructor, AFTER GenericWorker has joined the shared DSR graph. A
 * std::exit() there skips every destructor, so the DDS participant is never torn down. Instead the config readers
 * throw this, generated/main.cpp catches it around the worker's construction, prints what() and returns
 * EXIT_FAILURE: the partially built worker unwinds (members, then ~GenericWorker and its graph) on the normal path.
 *
 * Header-only, no Qt / DSR, so a standalone test can include it.
 */
#pragma once

#include <stdexcept>
#include <string>

namespace rc {

struct CountableConfigError : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

}  // namespace rc
