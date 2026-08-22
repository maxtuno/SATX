// test_negabinary — roundtrip exhaustivo del rango NB y conversiones NB↔TC
// (§14.1–§14.2 de la normativa)

#include <satx/satx.hpp>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

using satx::num::negabinary;

namespace {

// Oracle TC: v mod 2^W reinterpretado con signo.
std::int64_t tc_of_nb(std::int64_t v, unsigned w) {
  const std::uint64_t mask = (std::uint64_t{1} << w) - 1U;
  std::uint64_t u = static_cast<std::uint64_t>(v) & mask;
  if (u >= (std::uint64_t{1} << (w - 1))) u -= (std::uint64_t{1} << w);
  return static_cast<std::int64_t>(u);
}

// Recurrencia TC→NB corregida (ADR-006 revisado) a nivel de enteros:
//   R_0 = patrón (mod 2^W);  d_i = R_i mod 2;  R_{i+1} = (R_i >> 1) + (i impar ? d_i : 0).
// La salida es SIEMPRE el representante canónico de la caja (wrap_nb del
// patrón). Devuelve true si el valor está en la caja NB (min_NB <= t <= max_NB).
// NOTA: la condición anterior «c_final == t[W−1]» era incorrecta.
bool nb_of_tc(std::int64_t t, unsigned w, std::int64_t& out) {
  const std::uint64_t mask = (std::uint64_t{1} << w) - 1U;
  std::uint64_t r = static_cast<std::uint64_t>(t) & mask;
  std::int64_t acc = 0;
  std::int64_t p = 1;
  for (unsigned i = 0; i < w; ++i) {
    const int d = static_cast<int>(r & 1U);
    if (d != 0) acc += p;
    r >>= 1U;
    if ((i & 1U) == 1U) r += static_cast<std::uint64_t>(d);  // (R >> 1) + d_i
    p *= -2;
  }
  out = acc;
  const std::int64_t lo = satx::num::nb_min(w);
  const std::int64_t hi = satx::num::nb_max(w);
  return t >= lo && t <= hi;
}

// Representante canónico de la caja para un valor con signo t (patrón mod 2^w).
std::int64_t wrap_nb(std::int64_t t, unsigned w) {
  const std::uint64_t mask = (std::uint64_t{1} << w) - 1U;
  std::uint64_t u = static_cast<std::uint64_t>(t) & mask;
  const std::uint64_t hi = static_cast<std::uint64_t>(satx::num::nb_max(w));
  if (u > hi) u -= (std::uint64_t{1} << w);
  return static_cast<std::int64_t>(u);
}

template<std::size_t W>
void test_roundtrip_exhaustive() {
  const auto lo = negabinary<W>::min();
  const auto hi = negabinary<W>::max();
  for (std::int64_t n = lo; n <= hi; ++n) {
    const auto nb = negabinary<W>::encode(n);
    assert(nb.decode() == n);
  }
}

template<std::size_t W>
void test_range_and_errors() {
  const auto lo = negabinary<W>::min();
  const auto hi = negabinary<W>::max();
  bool threw = false;
  try {
    (void)negabinary<W>::encode(lo - 1);
  } catch (std::out_of_range const&) {
    threw = true;
  }
  assert(threw);
  threw = false;
  try {
    (void)negabinary<W>::encode(hi + 1);
  } catch (std::out_of_range const&) {
    threw = true;
  }
  assert(threw);
  // Fronteras válidas
  assert(negabinary<W>::encode(lo).decode() == lo);
  assert(negabinary<W>::encode(hi).decode() == hi);
}

template<std::size_t W>
void test_tc_nb_oracle() {
  const auto lo = negabinary<W>::min();
  const auto hi = negabinary<W>::max();
  // Para todo el rango NB: NB→TC→NB es la identidad.
  for (std::int64_t n = lo; n <= hi; ++n) {
    const std::int64_t t = tc_of_nb(n, W);
    std::int64_t back = 0;
    (void)nb_of_tc(t, W, back);
    assert(back == n);
  }
  // Para todo el rango TC: la salida es el representante canónico de la caja
  // (wrap_nb del patrón) y la exactitud ⟺ el valor está en la caja NB.
  const std::int64_t t_lo = -(std::int64_t{1} << (W - 1));
  const std::int64_t t_hi = (std::int64_t{1} << (W - 1)) - 1;
  for (std::int64_t t = t_lo; t <= t_hi; ++t) {
    std::int64_t d = 0;
    const bool exact = nb_of_tc(t, W, d);
    assert(d == wrap_nb(t, W));
    const bool in_nb = (t >= lo && t <= hi);
    assert(exact == in_nb);
    if (exact) {
      const auto nb = negabinary<W>::encode(d);
      assert(nb.decode() == t);
    }
  }
}

}  // namespace

int main() {
  test_roundtrip_exhaustive<4>();
  test_roundtrip_exhaustive<8>();
  test_roundtrip_exhaustive<12>();
  test_roundtrip_exhaustive<16>();

  test_range_and_errors<8>();
  test_range_and_errors<16>();

  test_tc_nb_oracle<4>();
  test_tc_nb_oracle<5>();
  test_tc_nb_oracle<6>();
  test_tc_nb_oracle<8>();
  test_tc_nb_oracle<12>();

  // Tabla de rangos (§7.1)
  static_assert(negabinary<8>::min() == -170 && negabinary<8>::max() == 85);
  static_assert(negabinary<12>::min() == -2730 && negabinary<12>::max() == 1365);
  static_assert(negabinary<16>::min() == -43690 && negabinary<16>::max() == 21845);
  static_assert(negabinary<32>::min() == -2863311530LL && negabinary<32>::max() == 1431655765LL);

  // Fronteras de W=32 (no exhaustivo)
  constexpr auto nb32_lo = negabinary<32>::encode(-2863311530LL);
  constexpr auto nb32_hi = negabinary<32>::encode(1431655765LL);
  static_assert(nb32_lo.decode() == -2863311530LL);
  static_assert(nb32_hi.decode() == 1431655765LL);

  // Propiedad general: max_NB + |min_NB| == 2^W − 1
  static_assert(negabinary<8>::max() + (-negabinary<8>::min()) == 255);
  static_assert(negabinary<16>::max() + (-negabinary<16>::min()) == 65535);
  static_assert(negabinary<32>::max() + (-negabinary<32>::min()) == 4294967295LL);

  std::puts("test_negabinary: OK");
  return 0;
}
