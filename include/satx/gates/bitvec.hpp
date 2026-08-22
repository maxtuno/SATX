#pragma once
// satx::gates — bit-vectors: riel LSB-first y operadores (RCA, RCS, PM, comparadores)
// ADR-011: los pliegues de constantes de rca/rca_carry usan __int128 (ancho
// hasta 127 sin UB); pmul pliega solo hasta 64 bits (más allá construye gates).

#include "../core/engine.hpp"
#include "../core/lit.hpp"
#include "primitive.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace satx::gates {

// Riel: vector de literales, LSB-first (posición 0 = bit 0).
using rail = std::vector<core::lit_t>;

inline rail sext(rail const& a, std::size_t w);
inline rail zext(rail const& a, std::size_t w);

// Alineación de anchos según la política del motor: sign_extend replica el
// bit más significativo (CBE); truncate recorta al ancho menor.
inline std::pair<rail, rail> align(engine& e, rail const& a, rail const& b) {
  const std::size_t wa = a.size(), wb = b.size();
  if (e.get_width_policy() == engine::width_policy::truncate) {
    const std::size_t w = std::min(wa, wb);
    return {rail(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(w)),
            rail(b.begin(), b.begin() + static_cast<std::ptrdiff_t>(w))};
  }
  const std::size_t w = std::max(wa, wb);
  return {sext(a, w), sext(b, w)};
}

// Extensión de signo (replica el bit más significativo).
// Nota: si el objetivo es menor que la fuente, trunca (hazard silencioso del
// API público; los usos internos siempre ensanchan).
inline rail sext(rail const& a, std::size_t w) {
  if (a.size() >= w) return rail(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(w));
  const core::lit_t sign = a.empty() ? core::false_lit : a.back();
  rail out = a;
  out.resize(w, sign);
  return out;
}

// Extensión con ceros.
inline rail zext(rail const& a, std::size_t w) {
  if (a.size() >= w) return rail(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(w));
  rail out = a;
  out.resize(w, core::false_lit);
  return out;
}

// Desplazamiento a la izquierda con pérdida (wrap) dentro de w bits: out[i] = i<k ? 0 : in[i−k].
inline rail shl(rail const& a, std::size_t k) {
  rail out(a.size(), core::false_lit);
  for (std::size_t i = k; i < a.size(); ++i) out[i] = a[i - k];
  return out;
}

// Desplazamiento a la derecha lógico (relleno con ceros).
inline rail shr(rail const& a, std::size_t k) {
  if (k >= a.size()) return rail(a.size(), core::false_lit);
  rail out(a.size(), core::false_lit);
  for (std::size_t i = 0; i + k < a.size(); ++i) out[i] = a[i + k];
  return out;
}

// Desplazamiento a la derecha aritmético (relleno con el signo).
inline rail sra(rail const& a, std::size_t k) {
  const core::lit_t sign = a.empty() ? core::false_lit : a.back();
  if (k >= a.size()) return rail(a.size(), sign);
  rail out(a.size(), sign);
  for (std::size_t i = 0; i + k < a.size(); ++i) out[i] = a[i + k];
  return out;
}

// Negación bit a bit (sin cláusulas: re-enrutado de literales).
inline rail not_rails(rail const& a) {
  rail out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) out[i] = core::neg(a[i]);
  return out;
}

// Operadores bit a bit por riel (alinean anchos según la política del motor).
inline rail and_rails(engine& e, rail const& a, rail const& b) {
  const auto [xa, xb] = align(e, a, b);
  rail out(xa.size());
  for (std::size_t i = 0; i < xa.size(); ++i) out[i] = and2(e, xa[i], xb[i]);
  return out;
}

inline rail or_rails(engine& e, rail const& a, rail const& b) {
  const auto [xa, xb] = align(e, a, b);
  rail out(xa.size());
  for (std::size_t i = 0; i < xa.size(); ++i) out[i] = or2(e, xa[i], xb[i]);
  return out;
}

inline rail xor_rails(engine& e, rail const& a, rail const& b) {
  const auto [xa, xb] = align(e, a, b);
  rail out(xa.size());
  for (std::size_t i = 0; i < xa.size(); ++i) out[i] = xor2(e, xa[i], xb[i]);
  return out;
}

inline bool all_constant(rail const& a) noexcept {
  for (core::lit_t l : a)
    if (!core::is_constant(l)) return false;
  return true;
}

// Suma ripple-carry con acarreo de salida. W × fas + W × fac.
// Pliegue de constantes con __int128 para w <= 127 (ADR-011).
inline std::pair<rail, core::lit_t> rca_carry(engine& e, rail const& a, rail const& b,
                                              core::lit_t cin = core::false_lit) {
  const auto [xa, xb] = align(e, a, b);
  const std::size_t w = xa.size();
  if (w == 0) return {{}, cin};

  if (w <= 127 && all_constant(xa) && all_constant(xb) && core::is_constant(cin)) {
    unsigned __int128 va = 0, vb = 0;
    for (std::size_t i = 0; i < w; ++i) {
      if (xa[i] == core::true_lit) va |= (static_cast<unsigned __int128>(1) << i);
      if (xb[i] == core::true_lit) vb |= (static_cast<unsigned __int128>(1) << i);
    }
    const unsigned __int128 s = va + vb + (cin == core::true_lit ? 1U : 0U);
    rail out(w, core::false_lit);
    for (std::size_t i = 0; i < w; ++i)
      out[i] = ((s >> i) & 1U) ? core::true_lit : core::false_lit;
    return {out, ((s >> w) & 1U) ? core::true_lit : core::false_lit};
  }

  if (cin == core::false_lit && all_constant(xb)) {
    bool zero = true;
    for (core::lit_t l : xb) zero = zero && (l == core::false_lit);
    if (zero) return {xa, core::false_lit};  // a + 0 == a, sin acarreo
  }

  rail out(w);
  core::lit_t carry = cin;
  for (std::size_t i = 0; i < w; ++i) {
    out[i] = fas(e, xa[i], xb[i], carry);
    carry = fac(e, xa[i], xb[i], carry);
  }
  return {out, carry};
}

// Suma ripple-carry (el acarreo final se descarta). W × fas + (W−1) × fac.
inline rail rca(engine& e, rail const& a, rail const& b,
                core::lit_t cin = core::false_lit) {
  return rca_carry(e, a, b, cin).first;
}

// Resta en complemento a dos: a + ¬b + 1.
inline rail rcs(engine& e, rail const& a, rail const& b) {
  if (all_constant(b)) {
    bool zero = true;
    for (core::lit_t l : b) zero = zero && (l == core::false_lit);
    if (zero) return align(e, a, b).first;
  }
  const auto [xa, xbb] = align(e, a, b);
  rail nb = xbb;
  for (core::lit_t& l : nb) l = core::neg(l);
  return rca(e, xa, nb, core::true_lit);
}

// Multiplicador por productos parciales: devuelve los w bits bajos del producto
// (complejidad O(w²)). Para el producto exacto de dos valores con signo de w bits,
// los operandos deben sext'arse previamente a 2w (normativa §10.3).
// Pliegue de constantes solo hasta 64 bits (ADR-011: sin UB).
inline rail pmul(engine& e, rail const& a, rail const& b) {
  const auto [xa, xb] = align(e, a, b);
  const std::size_t w = xa.size();
  if (w == 0) return {};

  if (w <= 64 && all_constant(xa) && all_constant(xb)) {
    std::uint64_t va = 0, vb = 0;
    for (std::size_t i = 0; i < w; ++i) {
      if (xa[i] == core::true_lit) va |= (std::uint64_t{1} << i);
      if (xb[i] == core::true_lit) vb |= (std::uint64_t{1} << i);
    }
    const std::uint64_t p = va * vb;
    rail out(w, core::false_lit);
    for (std::size_t i = 0; i < w; ++i)
      out[i] = ((p >> i) & 1U) ? core::true_lit : core::false_lit;
    return out;
  }

  // Fila 0: a[0] · b[j] en la posición j.
  rail acc(w);
  for (std::size_t j = 0; j < w; ++j) acc[j] = and2(e, xa[0], xb[j]);
  // Filas i = 1..w−1: contribuyen a las posiciones i..w−1.
  for (std::size_t i = 1; i < w; ++i) {
    rail row(w - i);
    for (std::size_t j = 0; j < w - i; ++j) row[j] = and2(e, xa[i], xb[j]);
    rail tail(w - i);
    for (std::size_t j = 0; j < w - i; ++j) tail[j] = acc[i + j];
    tail = rca(e, tail, row);
    for (std::size_t j = 0; j < w - i; ++j) acc[i + j] = tail[j];
  }
  return acc;
}

// Igualdad bit a bit (ambos rieles alineados según la política del motor).
// Plegado (ADR-010): rieles idénticos → true_lit sin circuito; rieles 100%
// constantes → literal constante (§14.3: eq_lit(e, z, z) ≡ true_lit).
inline core::lit_t eq(engine& e, rail const& a, rail const& b) {
  const auto [xa, xb] = align(e, a, b);
  if (xa == xb) return core::true_lit;
  if (all_constant(xa) && all_constant(xb)) return core::false_lit;
  core::lit_t acc = core::true_lit;
  for (std::size_t i = 0; i < xa.size(); ++i)
    acc = and2(e, acc, core::neg(xor2(e, xa[i], xb[i])));
  return acc;
}

// Comparación sin signo: a <= b.
inline core::lit_t ule(engine& e, rail const& a, rail const& b) {
  const auto [xa, xb] = align(e, a, b);
  if (xa.size() == 0) return core::true_lit;
  core::lit_t lt = core::false_lit;
  core::lit_t eqs = core::true_lit;
  for (std::size_t i = xa.size(); i-- > 0;) {
    const core::lit_t xni = core::neg(xor2(e, xa[i], xb[i]));
    lt = or2(e, lt, and2(e, and2(e, eqs, core::neg(xa[i])), xb[i]));
    eqs = and2(e, eqs, xni);
  }
  return or2(e, lt, eqs);
}

// Comparación con signo: a <= b.
inline core::lit_t sle(engine& e, rail const& a, rail const& b) {
  const auto [xa, xb] = align(e, a, b);
  const std::size_t w = xa.size();
  if (w == 0) return core::true_lit;
  if (w == 1) return or2(e, xa[0], core::neg(xb[0]));
  rail low_a(xa.begin(), xa.end() - 1);
  rail low_b(xb.begin(), xb.end() - 1);
  const core::lit_t rest = ule(e, low_a, low_b);
  const core::lit_t msb_eq = core::neg(xor2(e, xa[w - 1], xb[w - 1]));
  return mux2(e, msb_eq, xa[w - 1], rest);  // signos distintos → a<0 ∧ b>=0; iguales → rest
}

// Comparaciones estrictas (a < b ⟺ ¬(b <= a); el orden es total).
inline core::lit_t ult(engine& e, rail const& a, rail const& b) {
  return core::neg(ule(e, b, a));
}

inline core::lit_t slt(engine& e, rail const& a, rail const& b) {
  return core::neg(sle(e, b, a));
}

// Mux bit a bit: o[i] = s ? b[i] : a[i].
inline rail mux(engine& e, core::lit_t s, rail const& a, rail const& b) {
  const auto [xa, xb] = align(e, a, b);
  rail out(xa.size());
  for (std::size_t i = 0; i < xa.size(); ++i) out[i] = mux2(e, s, xa[i], xb[i]);
  return out;
}

// OR reducido de todos los bits (utilidad: restricción "distinto de cero").
inline core::lit_t reduce_or(engine& e, rail const& a) {
  core::lit_t acc = core::false_lit;
  for (core::lit_t l : a) acc = or2(e, acc, l);
  return acc;
}

// AND reducido (todos los bits a uno).
inline core::lit_t reduce_and(engine& e, rail const& a) {
  core::lit_t acc = core::true_lit;
  for (core::lit_t l : a) acc = and2(e, acc, l);
  return acc;
}

}  // namespace satx::gates
