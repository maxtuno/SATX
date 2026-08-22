#pragma once
// satx::num — codec negabinario (base −2), ruta concreta constexpr
// Normativa: docs/architecture.md §7.

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace satx::num {

// Rango NB en función del ancho (tiempo de ejecución, constexpr; w ∈ [0, 60]).
// min_NB(w) = −2·(4^⌊w/2⌋ − 1)/3 ;  max_NB(w) = (4^⌈w/2⌉ − 1)/3
[[nodiscard]] constexpr std::int64_t nb_min(std::size_t w) noexcept {
  std::int64_t p = 1;
  for (std::size_t i = 0; i < w / 2; ++i) p *= 4;
  return -2 * (p - 1) / 3;
}

[[nodiscard]] constexpr std::int64_t nb_max(std::size_t w) noexcept {
  std::int64_t p = 1;
  for (std::size_t i = 0; i < (w + 1) / 2; ++i) p *= 4;
  return (p - 1) / 3;
}

template<std::size_t W>
  requires (W >= 2 && W <= 60)
struct negabinary {
  std::array<std::uint8_t, W> digits{};  // LSB-first

  // Rango asimétrico del código con W dígitos.
  [[nodiscard]] static constexpr std::int64_t min() noexcept { return nb_min(W); }
  [[nodiscard]] static constexpr std::int64_t max() noexcept { return nb_max(W); }

  // Codificación por división repetida por −2 con resto normalizado a {0,1}.
  // Fuera de rango → std::out_of_range (la codificación de constantes es
  // siempre estricta — ADR-011; el pliegue wrap es complex::from_raw_wrap).
  [[nodiscard]] static constexpr negabinary encode(std::int64_t n) {
    if (n < min() || n > max()) {
      throw std::out_of_range("satx::num::negabinary::encode: value out of range");
    }
    negabinary nb;
    for (std::size_t i = 0; i < W; ++i) {
      std::int64_t q = n / -2;  // división truncada (C++)
      std::int64_t r = n % -2;  // r ∈ {−1, 0}
      if (r < 0) {
        q += 1;
        r += 2;  // r ∈ {0, 1}
      }
      nb.digits[i] = static_cast<std::uint8_t>(r);
      n = q;
    }
    return nb;
  }

  [[nodiscard]] constexpr std::int64_t decode() const noexcept {
    std::int64_t acc = 0;
    std::int64_t p = 1;  // (−2)^i
    for (std::size_t i = 0; i < W; ++i) {
      if (digits[i] != 0) acc += p;
      p *= -2;
    }
    return acc;
  }
};

}  // namespace satx::num
