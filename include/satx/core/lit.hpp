#pragma once
// satx::core — literales booleanos (normativa: docs/architecture.md §5.1)

#include <cstdint>

namespace satx::core {

using lit_t = std::int32_t;  // 0 = inválido; ±v = variable v (v >= 1)

inline constexpr lit_t true_lit  =  1;   // constante VERDADERO (variable 1 fijada a true)
inline constexpr lit_t false_lit = -1;   // constante FALSO

[[nodiscard]] constexpr std::int32_t var_of(lit_t l) noexcept {
  return l < 0 ? -l : l;
}

[[nodiscard]] constexpr lit_t neg(lit_t l) noexcept {
  return -l;
}

[[nodiscard]] constexpr bool sign(lit_t l) noexcept {
  return l > 0;
}

[[nodiscard]] constexpr bool is_constant(lit_t l) noexcept {
  return l == true_lit || l == false_lit;
}

}  // namespace satx::core
