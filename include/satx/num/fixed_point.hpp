#pragma once
// satx::num — punto fijo con escala 2^F (potencia de dos, no la escala decimal de fixed.py)
// Normativa: docs/architecture.md §8.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace satx::num::fixed_point {

// 2^F  (F <= 60; el núcleo impone F <= W <= 60)
[[nodiscard]] constexpr std::int64_t scale(std::size_t f) noexcept {
  return std::int64_t{1} << f;
}

// Codificación de una constante real: raw = round(x · 2^F).
// Redondeo half-away-from-zero (std::round) — ADR-003.
// x no finito o fuera del rango int64 → std::out_of_range (ADR-011: sin UB
// en la conversión float → int64).
[[nodiscard]] constexpr std::int64_t to_raw(double x, std::size_t f) {
  const double s = x * static_cast<double>(scale(f));
  if (!std::isfinite(s) || s < -9223372036854775808.0 || s >= 9223372036854775808.0)
    throw std::out_of_range("satx: to_raw: value not representable in int64");
  return static_cast<std::int64_t>(std::round(s));
}

// Decodificación exacta: x = raw / 2^F.
[[nodiscard]] constexpr double from_raw(std::int64_t raw, std::size_t f) {
  return static_cast<double>(raw) / static_cast<double>(scale(f));
}

// Desplazamiento con truncado hacia cero (ADR-003: truncado en mul).
[[nodiscard]] constexpr std::int64_t trunc_shr(std::int64_t v, std::size_t k) noexcept {
  if (v >= 0) return v >> k;
  return -((-v) >> k);
}

[[nodiscard]] constexpr std::int64_t trunc_shr_i128(__int128 v, std::size_t k) noexcept {
  if (v >= 0) return static_cast<std::int64_t>(v >> k);
  return -static_cast<std::int64_t>((-v) >> k);
}

// round(n · 2^k / den), half-away-from-zero, sin desbordes intermedios
// (división larga bit a bit sobre 241 bits). den > 0, 0 <= k <= 120.
// Lanza std::overflow_error si |resultado| >= 2^62 (fuera de todo rango NB, W <= 60)
// o si n·2^k necesita más de 241 bits (ADR-011: antes podía devolver
// cocientes truncados silenciosamente y tenía UB para k > 128).
[[nodiscard]] constexpr std::int64_t round_scaled_div(__int128 n, __int128 den,
                                                      std::size_t k) {
  if (den <= 0) throw std::domain_error("satx: round_scaled_div requires den > 0");
  if (k > 120) throw std::domain_error("satx: round_scaled_div requires k <= 120");
  if (n == 0) return 0;
  const bool neg = n < 0;
  unsigned __int128 u = neg ? static_cast<unsigned __int128>(-n)
                            : static_cast<unsigned __int128>(n);
  unsigned nbits = 0;
  for (unsigned __int128 t = u; t != 0; t >>= 1U) ++nbits;
  if (nbits + k > 241)  // la ventana de división cubre los bits 0..240 de X
    throw std::overflow_error("satx: scaled division out of range");
  // X = u · 2^k representado como hi·2^128 + lo
  unsigned __int128 hi = (k == 0) ? 0U : (u >> (128 - k));
  unsigned __int128 lo = (k == 0) ? u : (u << k);
  unsigned __int128 r = 0;
  __int128 q = 0;
  for (int b = 240; b >= 0; --b) {
    const unsigned __int128 bit = (b >= 128) ? ((hi >> (b - 128)) & 1U) : ((lo >> b) & 1U);
    r = (r << 1U) | bit;
    if (r >= static_cast<unsigned __int128>(den)) {
      r -= static_cast<unsigned __int128>(den);
      if (b >= 62) throw std::overflow_error("satx: scaled division out of range");
      q |= (__int128{1} << b);
    }
  }
  if (2 * r >= static_cast<unsigned __int128>(den)) ++q;
  if (q >= (__int128{1} << 62)) throw std::overflow_error("satx: scaled division out of range");
  return neg ? -static_cast<std::int64_t>(q) : static_cast<std::int64_t>(q);
}

}  // namespace satx::num::fixed_point
