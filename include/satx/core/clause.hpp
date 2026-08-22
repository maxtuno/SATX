#pragma once
// satx::core — cláusula CNF (disyunción de literales)

#include "lit.hpp"

#include <vector>

namespace satx::core {

struct clause {
  std::vector<lit_t> lits;  // normalizada: ordenada por |lit|, sin duplicados, sin tautologías
};

}  // namespace satx::core
