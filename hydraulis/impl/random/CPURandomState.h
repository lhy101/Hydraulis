#pragma once

#include "hydraulis/common/macros.h"
#include <random>

namespace hydraulis {
namespace impl {

void SetCPURandomSeed(uint64_t seed);
uint64_t GenNextRandomSeed();

} // namespace impl
} // namespace hydraulis
