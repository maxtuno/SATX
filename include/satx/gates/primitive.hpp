#pragma once
// satx::gates — gates booleanos primitivos (Tseitin → CNF)
// Codificaciones idénticas a la referencia Python (satx/alu.py), con
// plegado de operandos constantes (true_lit/false_lit, sin consumo de variables).

#include "../core/engine.hpp"
#include "../core/lit.hpp"

#include <array>
#include <cstddef>

namespace satx::gates {

// o ↔ (a ∨ b) — 4 cláusulas
inline core::lit_t or2(engine& e, core::lit_t a, core::lit_t b) {
  if (a == core::true_lit || b == core::true_lit) return core::true_lit;
  if (a == core::false_lit) return b;
  if (b == core::false_lit) return a;
  const core::lit_t o = e.add_variable();
  e.add_clause({a, b, core::neg(o)});
  e.add_clause({core::neg(a), b, o});
  e.add_clause({a, core::neg(b), o});
  e.add_clause({core::neg(a), core::neg(b), o});
  return o;
}

// o ↔ (a ∧ b) — 4 cláusulas
inline core::lit_t and2(engine& e, core::lit_t a, core::lit_t b) {
  if (a == core::false_lit || b == core::false_lit) return core::false_lit;
  if (a == core::true_lit) return b;
  if (b == core::true_lit) return a;
  const core::lit_t o = e.add_variable();
  e.add_clause({a, b, core::neg(o)});
  e.add_clause({core::neg(a), b, core::neg(o)});
  e.add_clause({a, core::neg(b), core::neg(o)});
  e.add_clause({core::neg(a), core::neg(b), o});
  return o;
}

// o ↔ (a ⊕ b) — 4 cláusulas
inline core::lit_t xor2(engine& e, core::lit_t a, core::lit_t b) {
  if (a == core::false_lit) return b;
  if (b == core::false_lit) return a;
  if (a == core::true_lit) return core::neg(b);
  if (b == core::true_lit) return core::neg(a);
  const core::lit_t o = e.add_variable();
  e.add_clause({a, b, core::neg(o)});
  e.add_clause({core::neg(a), core::neg(b), core::neg(o)});
  e.add_clause({a, core::neg(b), o});
  e.add_clause({core::neg(a), b, o});
  return o;
}

// o = s ? b : a — 4 cláusulas
inline core::lit_t mux2(engine& e, core::lit_t s, core::lit_t a, core::lit_t b) {
  if (s == core::true_lit) return b;
  if (s == core::false_lit) return a;
  if (a == b) return a;
  const core::lit_t o = e.add_variable();
  e.add_clause({s, a, core::neg(o)});
  e.add_clause({s, core::neg(a), o});
  e.add_clause({core::neg(s), b, core::neg(o)});
  e.add_clause({core::neg(s), core::neg(b), o});
  return o;
}

// Sumador completo, bit de suma: o = a ⊕ b ⊕ ci — 8 cláusulas
inline core::lit_t fas(engine& e, core::lit_t a, core::lit_t b, core::lit_t ci) {
  if (a == core::false_lit) return xor2(e, b, ci);
  if (b == core::false_lit) return xor2(e, a, ci);
  if (ci == core::false_lit) return xor2(e, a, b);
  if (a == core::true_lit) return core::neg(xor2(e, b, ci));
  if (b == core::true_lit) return core::neg(xor2(e, a, ci));
  if (ci == core::true_lit) return core::neg(xor2(e, a, b));
  const core::lit_t o = e.add_variable();
  e.add_clause({a, b, ci, core::neg(o)});
  e.add_clause({a, core::neg(b), core::neg(ci), core::neg(o)});
  e.add_clause({a, core::neg(b), ci, o});
  e.add_clause({a, b, core::neg(ci), o});
  e.add_clause({core::neg(a), b, ci, o});
  e.add_clause({core::neg(a), core::neg(b), core::neg(ci), o});
  e.add_clause({core::neg(a), core::neg(b), ci, core::neg(o)});
  e.add_clause({core::neg(a), b, core::neg(ci), core::neg(o)});
  return o;
}

// Sumador completo, bit de acarreo: o = mayority(a, b, ci) — 6 cláusulas
inline core::lit_t fac(engine& e, core::lit_t a, core::lit_t b, core::lit_t ci) {
  if (a == core::false_lit) return and2(e, b, ci);
  if (b == core::false_lit) return and2(e, a, ci);
  if (ci == core::false_lit) return and2(e, a, b);
  if (a == core::true_lit) return or2(e, b, ci);
  if (b == core::true_lit) return or2(e, a, ci);
  if (ci == core::true_lit) return or2(e, a, b);
  const core::lit_t o = e.add_variable();
  e.add_clause({a, b, core::neg(o)});
  e.add_clause({a, ci, core::neg(o)});
  e.add_clause({a, core::neg(b), core::neg(ci), o});
  e.add_clause({core::neg(a), b, ci, core::neg(o)});
  e.add_clause({core::neg(a), core::neg(b), o});
  e.add_clause({core::neg(a), core::neg(ci), o});
  return o;
}

}  // namespace satx::gates
