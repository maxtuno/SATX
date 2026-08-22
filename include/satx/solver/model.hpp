#pragma once
// satx::solver — modelo SAT decodificado (asignación variable → bool)

#include "../core/lit.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace satx::solver {

// Asignación de variables 1-indexadas, materializada desde el kernel SLIME
// (model01: 1 = true, 0 = false). Variables no asignadas → false.
class model {
public:
  model() = default;

  explicit model(std::vector<std::uint8_t> bits) : bits_(std::move(bits)) {}

  // Acceso 1-indexado por variable.
  [[nodiscard]] bool get_var(std::int32_t var) const noexcept {
    if (var < 1 || static_cast<std::size_t>(var) > bits_.size()) return false;
    return bits_[static_cast<std::size_t>(var) - 1] != 0;
  }

  // Acceso por literal (respeta el signo). Literales inválidos (0, INT32_MIN)
  // → false (ADR-011: antes get(0) devolvía true silenciosamente).
  [[nodiscard]] bool get(core::lit_t l) const noexcept {
    if (l == 0 || l == std::numeric_limits<std::int32_t>::min()) return false;
    const bool v = get_var(core::var_of(l));
    return l > 0 ? v : !v;
  }

  [[nodiscard]] std::size_t variable_count() const noexcept { return bits_.size(); }

private:
  std::vector<std::uint8_t> bits_;
};

}  // namespace satx::solver
