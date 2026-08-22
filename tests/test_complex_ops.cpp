// test_complex_ops — identidades y propiedades de la ruta concreta (§14.3–§14.4)
// + regresiones de la revisión ADR-011/012/013 (wrap, truncado, guardas, gates).

#include <satx/satx.hpp>

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

using satx::complex;

namespace {

bool close(std::complex<double> a, std::complex<double> b, double tol) {
  return std::abs(a - b) <= tol;
}

}  // namespace

int main() {
  using C = complex<16, 4>;
  satx::engine e;

  // i · i == −1
  {
    const auto z = C::i_unit(e) * C::i_unit(e);
    assert((z == C{-16, 0}));
    assert(close(z.value({}), std::complex<double>{-1.0, 0.0}, 1e-12));
  }

  // z + 0 == z  y  z · 1 == z
  {
    const C z{1.5, -2.0};
    assert(z + C::zero(e) == z);
    assert(z * C::one(e) == z);
  }

  // (a+b)+c == a+(b+c)
  {
    const C a{0.5, 1.25}, b{-3.0, 0.75}, c{2.25, -1.5};
    assert((a + b) + c == a + (b + c));
  }

  // a·(b+c) == a·b + a·c (valores con productos exactos en la escala)
  {
    const C a{2.0, 0.0}, b{3.0, 1.0}, c{-1.0, 2.0};
    assert(a * (b + c) == a * b + a * c);
  }

  // z·conj(z) == |z|²
  {
    const C z{1.5, -2.0};
    const auto zz = z * satx::conj(z);
    const double norm2 = std::norm(z.value({}));
    assert(close(zz.value({}), std::complex<double>{norm2, 0.0}, 1.0 / 16.0));
  }

  // (a·b)/b == a  y  1/i == −i
  {
    const C a{1.5, -2.0}, b{0.5, 1.0};
    const auto q = (a * b) / b;
    assert(close(q.value({}), a.value({}), 1.0 / 16.0));
  }
  {
    const auto q = C::one(e) / C::i_unit(e);
    assert((q == C{0, -16}));
  }

  // División por cero → error
  {
    bool threw = false;
    try {
      (void)(C::one(e) / C::zero(e));
    } catch (std::domain_error const&) {
      threw = true;
    }
    assert(threw);
  }

  // Overflow: constantes fuera del rango NB y producto que no cabe
  {
    bool threw = false;
    try {
      (void)complex<8, 0>{100, 0};
    } catch (std::out_of_range const&) {
      threw = true;
    }
    assert(threw);
  }
  {
    bool threw = false;
    try {
      (void)(complex<8, 0>{10, 0} * complex<8, 0>{10, 0});
    } catch (std::out_of_range const&) {
      threw = true;
    }
    assert(threw);
  }
  {
    // fronteras válidas de W=8: [−170, 85]
    const complex<8, 0> lo{-170, 0}, hi{85, 0};
    assert((lo.value({}) == std::complex<double>{-170.0, 0.0}));
    assert((hi.value({}) == std::complex<double>{85.0, 0.0}));
  }

  // from_float: redondeo half-away-from-zero (ADR-003)
  {
    const C z{1.5, -2.0};
    assert((z == C{24, -32}));
    const C w{0.03125, -0.03125};  // 0.5·2^−4 → redondea a 1·2^−4 (half-away)
    assert((w == C{1, -1}));
  }

  // Anchos mixtos: W = max, F = max con alineación de escala
  {
    const complex<8, 2> a{8, 0};        // valor 2.0
    const complex<12, 4> b{-16, 16};    // valor (−1, 1)
    const auto s = a + b;               // complex<12,4>
    static_assert(decltype(s)::width == 12 && decltype(s)::fractional == 4);
    assert((s == complex<12, 4>{16, 16}));
    const auto p = a * b;               // complex<12,4>
    assert((p == complex<12, 4>{-32, 32}));
    const auto q = a / b;               // complex<12,4>: 2/(−1+i) = −1−i
    assert((q == complex<12, 4>{-16, -16}));
  }

  // constexpr: plegado de constantes en tiempo de compilación
  {
    constexpr auto z = complex<8, 2>{1, -1} * complex<8, 2>{2, 3};
    static_assert(z == complex<8, 2>{1, 0});
  }

  // wrap (pliegue mod 2^W al rango NB)
  {
    const auto w = complex<8, 0>::from_raw_wrap(100, 0);
    assert((w == complex<8, 0>{-156, 0}));
  }

  // eq_lit: pliegue a true_lit/false_lit; coincide con operator== sobre constantes;
  // eq_lit(e, z, z) ≡ true_lit también para rieles simbólicos (§14.3)
  {
    const C a{1.5, -2.0}, c{-3.0, 0.75};
    assert(satx::eq_lit(e, a, a) == satx::core::true_lit);
    assert(satx::eq_lit(e, a, c) == satx::core::false_lit);
    assert((satx::eq_lit(e, a, a) == satx::core::true_lit) == (a == a));
    assert((satx::eq_lit(e, a, c) == satx::core::true_lit) == (a == c));
    const C v{e};  // simbólica
    assert(satx::eq_lit(e, v, v) == satx::core::true_lit);
  }

  // abs_sq: z·conj(z) == |z|² con im == 0 exacto (§10.7)
  {
    const C z{1.5, -2.0};
    const auto s = satx::abs_sq(z);
    const double norm2 = std::norm(z.value({}));
    assert(s.im_raw() == 0);
    assert(close(s.value({}), std::complex<double>{norm2, 0.0}, 1.0 / 16.0));
  }

  // lt_lit / le_lit: comparación lexicográfica (re, im) sobre constantes (§10.7)
  {
    const C a{1.0, 2.0}, b{1.0, 3.0}, d{2.0, 0.0}, f{0.5, 0.0};
    assert(satx::lt_lit(e, a, b) == satx::core::true_lit);
    assert(satx::lt_lit(e, b, a) == satx::core::false_lit);
    assert(satx::lt_lit(e, a, a) == satx::core::false_lit);
    assert(satx::lt_lit(e, a, d) == satx::core::true_lit);
    assert(satx::le_lit(e, a, a) == satx::core::true_lit);
    assert(satx::le_lit(e, b, a) == satx::core::false_lit);
    assert(satx::le_lit(e, f, a) == satx::core::true_lit);
  }

  // pow: exponente entero, square & multiply (§10.7)
  {
    const C z{0.5, 1.0};
    assert((satx::pow(z, 0) == C::one(e)));
    assert((satx::pow(z, 1) == z));
    assert(close(satx::pow(z, 3).value({}), std::pow(z.value({}), 3), 1.0 / 16.0));
    assert(close(satx::pow(z, -2).value({}), std::pow(z.value({}), -2), 1.0 / 16.0));
    assert((satx::pow(z, 4) == satx::pow(z, 2) * satx::pow(z, 2)));
  }

  // root_cbe: ruta concreta (z^(1/n) con redondeo half-away) y errores (§10.7)
  {
    const C z{4.0, 0.0};
    const auto r = satx::root_cbe(z, 2);
    assert(close(std::pow(r.value({}), 2), z.value({}), 1.0 / 16.0));
    assert((satx::root_cbe(z, 1) == z));
  }
  {
    bool threw = false;
    try {
      (void)satx::root_cbe(C{1.0, 0.0}, 0);
    } catch (std::domain_error const&) {
      threw = true;
    }
    assert(threw);
  }

  // ── ADR-011/012/013: regresiones de la revisión del núcleo ──

  // add/sub con desborde envuelven mod 2^W en la ruta concreta (ADR-011)
  {
    const complex<8, 0> a{85, 0}, b{85, 0};
    assert(((a + b) == complex<8, 0>{-86, 0}));  // 170 mod 256 → −86
    assert(((complex<8, 0>{-170, 0} - complex<8, 0>{1, 0}) == complex<8, 0>{85, 0}));
  }

  // identidades neg/conj (con wrap)
  {
    const complex<8, 2> z{1.5, -2.0};
    assert(((satx::neg(z) + z) == complex<8, 2>{0, 0}));
    assert((satx::conj(satx::conj(z))) == z);
    assert(((satx::neg(z) == complex<8, 2>{-1.5, 2.0})));
  }

  // mul con truncado hacia cero simétrico (ADR-012)
  {
    const complex<8, 4> a{-0.5, 0.0}, b{0.0625, 0.0};
    assert(((a * b) == complex<8, 4>{0, 0}));  // trunc(−8/16) = 0
    assert(((complex<8, 4>{0.5, 0.0} * b) == complex<8, 4>{0, 0}));
  }

  // real() / imag()
  {
    const complex<8, 2> z{1.5, -2.0};
    assert(((satx::real(z) == complex<8, 2>{1.5, 0.0})));
    assert(((satx::imag(z) == complex<8, 2>{-2.0, 0.0})));
  }

  // to_raw fuera de rango → out_of_range (sin UB float→int64, ADR-011)
  {
    bool threw = false;
    try {
      (void)complex<8, 4>::from_float(1e300, 0.0);
    } catch (std::out_of_range const&) {
      threw = true;
    }
    assert(threw);
  }

  // round_scaled_div: guardas de k y de ventana de división (ADR-011)
  {
    bool threw = false;
    try {
      (void)satx::num::fixed_point::round_scaled_div(1, 2, 121);
    } catch (std::domain_error const&) {
      threw = true;
    }
    assert(threw);
    threw = false;
    try {
      (void)satx::num::fixed_point::round_scaled_div(__int128{1} << 122, 1, 120);
    } catch (std::overflow_error const&) {
      threw = true;
    }
    assert(threw);
    assert(satx::num::fixed_point::round_scaled_div(5, 3, 4) == 27);    // round(5·16/3)
    assert(satx::num::fixed_point::round_scaled_div(-5, 3, 4) == -27);  // half-away-from-zero
  }

  // value(): detección de pérdida de precisión (bits significativos > 53)
  {
    using C = satx::num::complex<60, 30>;
    const C exact{1LL << 58, 0};                       // potencia de dos: exacta
    const auto v = exact.value(satx::solver::model{});
    assert(v.real() == 268435456.0);
    const C lossy{384307168202282325LL, 1};            // max_NB(60): 60 bits significativos
    bool threw = false;
    try {
      (void)lossy.value(satx::solver::model{});
    } catch (std::overflow_error const&) {
      threw = true;
    }
    assert(threw);
    const auto raw = lossy.value_raw(satx::solver::model{});
    assert(raw.real() == 384307168202282325LL && raw.imag() == 1);
  }

  // model::get con literales inválidos → false (ADR-011)
  {
    const satx::solver::model m{};
    assert(m.get(0) == false);
    assert(m.get(std::numeric_limits<std::int32_t>::min()) == false);
  }

  // cnf: unidad ¬VERDADERO → UNSAT inmediato; literales inválidos → excepción
  {
    satx::engine e;
    e.add_unit(satx::core::false_lit);
    assert(e.unsat());
    bool threw = false;
    satx::engine e2;
    try {
      e2.add_clause({0});
    } catch (std::invalid_argument const&) {
      threw = true;
    }
    assert(threw);
  }

  // nuevos gates de bit-vector contra oráculo del host
  {
    using satx::gates::rail;
    const satx::lit_t T = satx::core::true_lit, F = satx::core::false_lit;
    const rail a{T, T, F, F};  // 0b0011 = 3
    const rail b{F, T, T, F};  // 0b0110 = 6
    satx::engine e;
    // shr / sra (re-enrutado, sin cláusulas)
    const rail s1 = satx::gates::shr(a, 1);  // 0b0001
    assert(s1[0] == T && s1[1] == F && s1[2] == F && s1[3] == F);
    const rail s2 = satx::gates::sra(a, 1);  // 0b0001 (signo 0)
    assert(s2[0] == T && s2[1] == F && s2[2] == F && s2[3] == F);
    const rail neg4{T, F, F, T};                 // 0b1001 = −7
    const rail s3 = satx::gates::sra(neg4, 1);   // 0b1100 = −4
    assert(s3[0] == F && s3[1] == F && s3[2] == T && s3[3] == T);
    // bit a bit (con plegado de constantes)
    const rail ab = satx::gates::and_rails(e, a, b);  // 0b0010
    assert(ab[0] == F && ab[1] == T && ab[2] == F && ab[3] == F);
    const rail ob = satx::gates::or_rails(e, a, b);   // 0b0111
    assert(ob[0] == T && ob[1] == T && ob[2] == T && ob[3] == F);
    const rail xb = satx::gates::xor_rails(e, a, b);  // 0b0101
    assert(xb[0] == T && xb[1] == F && xb[2] == T && xb[3] == F);
    const rail nb = satx::gates::not_rails(a);        // 0b1100
    assert(nb[0] == F && nb[1] == F && nb[2] == T && nb[3] == T);
    // comparadores estrictos
    assert(satx::gates::slt(e, a, b) == T);
    assert(satx::gates::ult(e, a, b) == T);
    assert(satx::gates::slt(e, b, a) == F);
    assert(satx::gates::ult(e, b, a) == F);
    // reduce_and y acarreo de salida
    assert(satx::gates::reduce_and(e, rail{T, T, T, T}) == T);
    assert(satx::gates::reduce_and(e, rail{T, T, T, F}) == F);
    const rail ff{T, T, T, T};  // 15
    const rail oo{T, F, F, F};  // 1
    const auto [sum, carry] = satx::gates::rca_carry(e, ff, oo);
    assert(carry == T);
    assert(sum[0] == F && sum[1] == F && sum[2] == F && sum[3] == F);
  }

  // width_policy::truncate recorta al ancho menor (ADR-011)
  {
    satx::engine e;
    e.set_width_policy(satx::engine::width_policy::truncate);
    using satx::gates::rail;
    const satx::lit_t T = satx::core::true_lit;
    const rail r6{T, T, T, T, T, T};  // 6 bits
    const rail r4{T, T, T, T};        // 4 bits
    const auto s = satx::gates::rca(e, r6, r4);  // 4 bits: 15 + 15 = 14 (wrap)
    assert(s.size() == 4);
    assert(s[0] == satx::core::false_lit && s[1] == T && s[2] == T && s[3] == T);
  }

  // nb_exact_lit con rieles constantes (ADR-011)
  {
    satx::engine e;
    using satx::gates::rail;
    const satx::lit_t T = satx::core::true_lit, F = satx::core::false_lit;
    const rail p3{T, T, F, F};    // patrón 3  → valor con signo 3 ∈ [−10, 5] ✓
    const rail p6{F, T, T, F};    // patrón 6  → valor con signo 6 ∉ [−10, 5] ✗
    const rail p14{F, T, T, T};   // patrón 14 → valor con signo −2 ∈ [−10, 5] ✓
    assert(satx::num::nb_exact_lit(e, p3) == T);
    assert(satx::num::nb_exact_lit(e, p6) == F);
    assert(satx::num::nb_exact_lit(e, p14) == T);
  }

  std::puts("test_complex_ops: OK");
  return 0;
}
