#pragma once
// satx::num — el número complejo CBE sobre rieles negabinarios (normativa §9–§10)
//
// Revisión ADR-011 (canonicalización TC):
//   · todo resultado simbólico se almacena como patrón TC de W bits (kind::tc)
//     decodificado con wrap_nb (el representante canónico de la caja NB):
//     add/sub/neg/conj envuelven mod 2^W; mul/div restringen el resultado a la
//     caja NB (desborde → UNSAT en ruta simbólica, excepción en ruta concreta);
//   · la conversión NB→TC exacta usa W+1 bits (la caja NB cabe en el rango con
//     signo de W+1 bits); la extensión de patrones negativos re-ancla con
//     −2^Wa (extend_pattern, ADR-011);
//   · eq_lit/lt_lit/le_lit alinean escalas de forma exacta (sin wrap);
//   · mul trunca hacia cero en ambas rutas (ADR-003/ADR-012), no con floor.

#include "../core/engine.hpp"
#include "../core/lit.hpp"
#include "../gates/bitvec.hpp"
#include "../solver/model.hpp"
#include "fixed_point.hpp"
#include "negabinary.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace satx::num {

template<std::size_t W, std::size_t F>
  requires (W >= 2 && W <= 60 && F <= W)
class complex;

namespace detail {

template<std::size_t W>
[[nodiscard]] std::array<core::lit_t, W> to_array(gates::rail const& r) {
  assert(r.size() == W);
  std::array<core::lit_t, W> out{};
  for (std::size_t i = 0; i < W; ++i) out[i] = r[i];
  return out;
}

// Rieles NB de una constante concreta (literales constantes, sin variables).
template<std::size_t W>
[[nodiscard]] constexpr std::array<core::lit_t, W> literal_nb_rails(std::int64_t v) {
  const auto nb = negabinary<W>::encode(v);
  std::array<core::lit_t, W> out{};
  for (std::size_t i = 0; i < W; ++i)
    out[i] = nb.digits[i] != 0 ? core::true_lit : core::false_lit;
  return out;
}

// Rieles TC (patrón mod 2^W) de una constante: (v · 2^shift) mod 2^W.
template<std::size_t W>
[[nodiscard]] constexpr std::array<core::lit_t, W> literal_tc_rails(std::int64_t v,
                                                                    std::size_t shift = 0) {
  static_assert(W >= 2 && W <= 64);
  constexpr std::uint64_t mask = (W >= 64) ? ~std::uint64_t{0}
                                           : ((std::uint64_t{1} << W) - 1U);
  std::uint64_t u = static_cast<std::uint64_t>(v) & mask;
  u = (shift >= W) ? 0U : ((u << shift) & mask);
  std::array<core::lit_t, W> out{};
  for (std::size_t i = 0; i < W; ++i)
    out[i] = ((u >> i) & 1U) != 0 ? core::true_lit : core::false_lit;
  return out;
}

// §7.3 NB → TC (patrón mod 2^W): separación de rieles pares/impares y resta.
[[nodiscard]] inline gates::rail nb_to_tc(engine& e, std::span<const core::lit_t> nb) {
  gates::rail pos(nb.size(), core::false_lit), neg(nb.size(), core::false_lit);
  for (std::size_t i = 0; i < nb.size(); ++i) {
    if ((i & 1U) == 0U) pos[i] = nb[i];
    else neg[i] = nb[i];
  }
  return gates::rcs(e, pos, neg);
}

// §7.3 TC → NB (ADR-006 revisado): la recurrencia anterior con acarreo de un
// bit («c_final == t[W−1]», acarreo AND/OR alternante) es INCORRECTA (p. ej.
// el patrón 10 en W=8 produce 30). Recurrencia correcta con resto completo:
//   R_0 = patrón (mod 2^W);  d_i = R_i[0];
//   R_{i+1} = (R_i >> 1) + (i impar ? d_i : 0).
// Produce SIEMPRE los dígitos NB del representante canónico de la caja
// (wrap_nb del patrón) para cualquier patrón de W bits. Coste O(W²);
// solo se usa en los accesores real_rail/imag_rail para kind::tc.
[[nodiscard]] inline gates::rail tc_to_nb(engine& e, std::span<const core::lit_t> tc) {
  const std::size_t w = tc.size();
  gates::rail r(tc.begin(), tc.end());
  gates::rail d(w);
  for (std::size_t i = 0; i < w; ++i) {
    d[i] = r[0];
    gates::rail sh(w - i - 1);
    for (std::size_t j = 0; j + 1 < w - i; ++j) sh[j] = r[j + 1];
    if ((i & 1U) == 1U) {
      gates::rail z(sh.size(), core::false_lit);
      sh = gates::rca(e, sh, z, d[i]);  // (R_i >> 1) + d_i
    }
    r = std::move(sh);
  }
  return d;
}

// Patrón TC → valor con signo exacto en W+1 bits (subtract 2^W si p > max_NB).
[[nodiscard]] inline gates::rail tc_to_signed(engine& e, std::span<const core::lit_t> tc) {
  const std::size_t w = tc.size();
  if (w > 60) throw std::out_of_range("satx: tc_to_signed requires width <= 60");
  gates::rail hi(w, core::false_lit);
  const std::uint64_t mx = static_cast<std::uint64_t>(nb_max(w));
  for (std::size_t i = 0; i < w; ++i)
    if (((mx >> i) & 1U) != 0U) hi[i] = core::true_lit;
  const gates::rail p(tc.begin(), tc.end());
  const core::lit_t gt = core::neg(gates::ule(e, p, hi));  // p > max_NB
  gates::rail sub(w + 1, core::false_lit);
  sub[w] = gt;  // restar 2^W cuando el patrón supera la caja
  return gates::rcs(e, gates::zext(p, w + 1), sub);
}

// Exactitud de la conversión TC→NB: valor (con signo) dentro de la caja NB
// ⟺ min_NB(W) <= v <= max_NB(W). Como la caja NB puede no caber en el rango
// con signo de W bits (W impar), la comparación se hace en W+1 bits con el
// valor sign-extendido (ADR-011). NOTA: la condición anterior «c_final ==
// t[W−1]» era incorrecta (falsos positivos/negativos, p. ej. W=3: v=−3, v=3).
[[nodiscard]] inline core::lit_t nb_exact_lit(engine& e, std::span<const core::lit_t> tc) {
  const std::size_t w = tc.size();
  if (w > 60) throw std::out_of_range("satx: nb_exact_lit requires width <= 60");
  if (w == 0) return core::true_lit;
  gates::rail p(tc.begin(), tc.end());
  p = gates::sext(p, w + 1);  // valor con signo en w+1 bits
  gates::rail lo(w + 1, core::false_lit), hi(w + 1, core::false_lit);
  const std::uint64_t lov = static_cast<std::uint64_t>(nb_min(w)) & ((std::uint64_t{1} << (w + 1)) - 1U);
  const std::uint64_t hiv = static_cast<std::uint64_t>(nb_max(w));
  for (std::size_t i = 0; i <= w; ++i) {
    if (((lov >> i) & 1U) != 0U) lo[i] = core::true_lit;
    if (((hiv >> i) & 1U) != 0U) hi[i] = core::true_lit;
  }
  return gates::and2(e, gates::sle(e, lo, p), gates::sle(e, p, hi));
}

[[nodiscard]] inline engine& common_engine(engine* a, engine* b) {
  engine* e = a != nullptr ? a : b;
  if (e == nullptr) throw std::logic_error("satx: symbolic operation requires an engine");
  if (a != nullptr && b != nullptr && a != b)
    throw std::logic_error("satx: operands belong to different engines");
  return *e;
}

// ── preparación de operandos (definidas tras la clase, §abajo) ──────────────

// Extensión de un patrón mod 2^Wa a un patrón mod 2^W (W >= Wa): si el patrón
// supera max_NB(Wa) representa un valor negativo y hay que re-anclar con
// −2^Wa (como tc_to_signed); la extensión con ceros a secas lo corrompe.
[[nodiscard]] inline gates::rail extend_pattern(engine& e, std::span<const core::lit_t> p,
                                                std::size_t W) {
  const std::size_t wa = p.size();
  if (wa == W) return gates::rail(p.begin(), p.end());
  if (wa > 60) throw std::out_of_range("satx: extend_pattern requires width <= 60");
  gates::rail hi(wa, core::false_lit);
  const std::uint64_t mx = static_cast<std::uint64_t>(nb_max(wa));
  for (std::size_t i = 0; i < wa; ++i)
    if (((mx >> i) & 1U) != 0U) hi[i] = core::true_lit;
  const gates::rail pr(p.begin(), p.end());
  const core::lit_t gt = core::neg(gates::ule(e, pr, hi));  // p > max_NB(Wa) → v < 0
  gates::rail sub(W, core::false_lit);
  sub[wa] = gt;  // restar 2^Wa
  return gates::rcs(e, gates::zext(pr, W), sub);
}

// Aritmética de patrones mod 2^W (wrap) para add/sub.
template<std::size_t Wa, std::size_t Fa, std::size_t W, std::size_t F>
[[nodiscard]] gates::rail prep_wrap(complex<Wa, Fa> const& z, engine& e, bool imag);

// Representación con signo EXACTA a ancho wprime (>= Wa + delta + 1).
template<std::size_t Wa, std::size_t Fa>
[[nodiscard]] gates::rail prep_exact(complex<Wa, Fa> const& z, engine& e, bool imag,
                                     std::size_t wprime, std::size_t delta);

}  // namespace detail

// Exactitud de la conversión TC→NB disponible en el namespace público (§9.2).
using detail::nb_exact_lit;

// ¿raw/2^F es representable EXACTAMENTE en double? Sí si los bits
// significativos de |raw| son ≤ 53 (la división por 2^F solo desplaza el
// exponente). raw se limita a la caja NB (|raw| ≤ 2^60), así que el rango
// de exponentes de double nunca se desborda (F ≤ W ≤ 60).
[[nodiscard]] constexpr bool exact_in_double(std::int64_t raw) noexcept {
  if (raw == 0) return true;
  unsigned long long u = (raw < 0) ? (0ULL - static_cast<unsigned long long>(raw))
                                   : static_cast<unsigned long long>(raw);
  while ((u & 1ULL) == 0) u >>= 1U;  // descartar ceros finales (potencias de dos)
  int sig = 0;
  while (u != 0) {
    ++sig;
    u >>= 1U;
  }
  return sig <= 53;
}

// ═══════════════════════════════════════════════════════════════════════════
// complex<W,F>: z = (re + i·im) / 2^F, re, im enteros en rango NB (constantes)
// o rieles NB/TC (ruta simbólica). Formato físico CBE(W,F) — Complejo Binario
// Entrelazado (autoría de Oscar Riveros, 2026): palabra de 2W bits entrelazada,
// Z[2k] = R[k] (carril real), Z[2k+1] = I[k] (carril imaginario), base −2 con
// exponente e(k) = k − F (normativa §9.0–§9.1).
// ═══════════════════════════════════════════════════════════════════════════
template<std::size_t W, std::size_t F>
  requires (W >= 2 && W <= 60 && F <= W)
class complex {
public:
  enum class kind { concrete, nb, tc };

  static constexpr std::size_t width = W;
  static constexpr std::size_t fractional = F;

  // ── ruta concreta (constante plegada) ──
  constexpr complex() : complex(std::int64_t{0}, std::int64_t{0}) {}

  constexpr complex(std::int64_t re_raw, std::int64_t im_raw)
      : kind_(kind::concrete), re_val_(re_raw), im_val_(im_raw) {
    check_range();
  }

  template<typename A, typename B>
    requires (std::floating_point<A> && std::floating_point<B>)
  constexpr complex(A re, B im)
      : complex(fixed_point::to_raw(re, F), fixed_point::to_raw(im, F)) {}

  // ── ruta simbólica (variable libre, rieles NB) ──
  explicit complex(engine& e) : kind_(kind::nb), engine_(&e) {
    for (std::size_t i = 0; i < W; ++i) {
      re_rail_[i] = e.add_variable();
      im_rail_[i] = e.add_variable();
    }
  }

  // ── fábricas ──
  [[nodiscard]] static constexpr complex zero(engine&) {
    return complex{std::int64_t{0}, std::int64_t{0}};
  }
  [[nodiscard]] static constexpr complex one(engine&) {
    return complex{fixed_point::scale(F), std::int64_t{0}};
  }
  [[nodiscard]] static constexpr complex i_unit(engine&) {
    return complex{std::int64_t{0}, fixed_point::scale(F)};
  }
  [[nodiscard]] static constexpr complex from_float(double re, double im) {
    return complex{re, im};
  }
  [[nodiscard]] static constexpr complex from_raw(std::int64_t re_raw, std::int64_t im_raw) {
    return complex{re_raw, im_raw};
  }
  // const_policy::wrap: pliega el valor al rango NB (mod 2^W).
  [[nodiscard]] static constexpr complex from_raw_wrap(std::int64_t re_raw, std::int64_t im_raw) {
    return complex{wrap_nb(re_raw), wrap_nb(im_raw)};
  }

  // Promoción de rieles TC (resultados internos del datapath). SEMÁNTICA:
  // los rieles se interpretan como el patrón mod 2^W, cuyo valor canónico es
  // wrap_nb(patrón) (ADR-011).
  [[nodiscard]] static complex from_tc_rails(engine& e, gates::rail re, gates::rail im) {
    if (re.size() != W || im.size() != W)
      throw std::invalid_argument("satx::num::complex: from_tc_rails requiere rieles de ancho W");
    complex z{};
    z.kind_ = kind::tc;
    z.engine_ = &e;
    z.re_rail_ = detail::to_array<W>(re);
    z.im_rail_ = detail::to_array<W>(im);
    return z;
  }

  // Rieles NB canónicos directos (resultados internos; los rieles deben ser
  // dígitos NB válidos de ancho W).
  [[nodiscard]] static complex from_nb_rails(engine& e, std::array<core::lit_t, W> re,
                                             std::array<core::lit_t, W> im) {
    complex z{};
    z.kind_ = kind::nb;
    z.engine_ = &e;
    z.re_rail_ = re;
    z.im_rail_ = im;
    return z;
  }

  // ── acceso ──
  [[nodiscard]] constexpr kind representation() const noexcept { return kind_; }
  [[nodiscard]] constexpr bool is_concrete() const noexcept { return kind_ == kind::concrete; }
  [[nodiscard]] engine* engine_of() const noexcept { return engine_; }

  [[nodiscard]] constexpr std::int64_t re_raw() const {
    if (!is_concrete()) throw std::logic_error("satx: raw value of symbolic complex");
    return re_val_;
  }
  [[nodiscard]] constexpr std::int64_t im_raw() const {
    if (!is_concrete()) throw std::logic_error("satx: raw value of symbolic complex");
    return im_val_;
  }

  // Patrón físico almacenado (dígitos NB en kind::nb; patrón mod 2^W en
  // kind::tc; indefinido en kind::concrete — usar re_raw()). Sin circuitos.
  [[nodiscard]] std::array<core::lit_t, W> re_pattern() const noexcept { return re_rail_; }
  [[nodiscard]] std::array<core::lit_t, W> im_pattern() const noexcept { return im_rail_; }

  // Pliegue mod 2^W al rango NB (representante canónico de la caja). ADR-011.
  [[nodiscard]] static constexpr std::int64_t wrap_nb(std::int64_t v) noexcept {
    constexpr std::uint64_t mask = (std::uint64_t{1} << W) - 1U;
    const std::uint64_t u = static_cast<std::uint64_t>(v) & mask;
    return u <= static_cast<std::uint64_t>(negabinary<W>::max())
               ? static_cast<std::int64_t>(u)
               : static_cast<std::int64_t>(u) - (std::int64_t{1} << W);
  }

  [[nodiscard]] static constexpr std::int64_t wrap_nb128(__int128 v) noexcept {
    constexpr __int128 mask = (std::uint64_t{1} << W) - 1U;
    const std::uint64_t u =
        static_cast<std::uint64_t>(static_cast<unsigned __int128>(v) & mask);
    return u <= static_cast<std::uint64_t>(negabinary<W>::max())
               ? static_cast<std::int64_t>(u)
               : static_cast<std::int64_t>(u - (std::uint64_t{1} << W));
  }

  // Riel NB canónico (conversión TC→NB §7.3 bajo demanda para kind::tc).
  [[nodiscard]] std::array<core::lit_t, W> real_rail(engine& e) const {
    switch (kind_) {
      case kind::concrete: return detail::literal_nb_rails<W>(re_val_);
      case kind::nb: return re_rail_;
      case kind::tc: return detail::to_array<W>(detail::tc_to_nb(e, re_rail_));
    }
    return {};
  }
  [[nodiscard]] std::array<core::lit_t, W> imag_rail(engine& e) const {
    switch (kind_) {
      case kind::concrete: return detail::literal_nb_rails<W>(im_val_);
      case kind::nb: return im_rail_;
      case kind::tc: return detail::to_array<W>(detail::tc_to_nb(e, im_rail_));
    }
    return {};
  }

  // Riel TC del datapath (patrón mod 2^W; conversión NB→TC §7.3 bajo demanda).
  [[nodiscard]] std::array<core::lit_t, W> tc_real_rail(engine& e) const {
    switch (kind_) {
      case kind::concrete: return detail::literal_tc_rails<W>(re_val_);
      case kind::nb: return detail::to_array<W>(detail::nb_to_tc(e, re_rail_));
      case kind::tc: return re_rail_;
    }
    return {};
  }
  [[nodiscard]] std::array<core::lit_t, W> tc_imag_rail(engine& e) const {
    switch (kind_) {
      case kind::concrete: return detail::literal_tc_rails<W>(im_val_);
      case kind::nb: return detail::to_array<W>(detail::nb_to_tc(e, im_rail_));
      case kind::tc: return im_rail_;
    }
    return {};
  }

  // Valor EXACTO post-solve: componentes enteras (raw) del representante
  // canónico NB — sin pasar por double, sin pérdida de precisión.
  [[nodiscard]] std::complex<std::int64_t> value_raw(solver::model const& m) const {
    if (kind_ == kind::concrete) return {re_val_, im_val_};
    if (kind_ == kind::nb) {
      return {decode_nb(m, re_rail_), decode_nb(m, im_rail_)};
    }
    return {decode_tc(m, re_rail_), decode_tc(m, im_rail_)};
  }

  // Decodificación post-solve: NB (variables/constantes) o patrón TC envuelto
  // a la caja NB (wrap_nb — ADR-011). El valor exacto es raw/2^F; si la
  // conversión a double NO puede representarlo exactamente (bits
  // significativos > 53, es decir la precisión del host no alcanza), se lanza
  // std::overflow_error: el sistema DETECTA la pérdida de precisión en lugar
  // de devolver un valor redondeado silenciosamente. Use value_raw para el
  // valor exacto (sin pasar por double).
  [[nodiscard]] std::complex<double> value(solver::model const& m) const {
    const auto raw = value_raw(m);
    if (!exact_in_double(raw.real()) || !exact_in_double(raw.imag())) {
      throw std::overflow_error(
          "satx::num::complex::value: precision loss: el valor exacto "
          "raw/2^F no es representable en double (use value_raw para el "
          "valor exacto o reduzca W/F)");
    }
    return {fixed_point::from_raw(raw.real(), F), fixed_point::from_raw(raw.imag(), F)};
  }

private:
  constexpr void check_range() const {
    if (re_val_ < negabinary<W>::min() || re_val_ > negabinary<W>::max() ||
        im_val_ < negabinary<W>::min() || im_val_ > negabinary<W>::max()) {
      throw std::out_of_range("satx::num::complex: constant out of negabinary range");
    }
  }

  [[nodiscard]] static std::int64_t decode_nb(solver::model const& m,
                                              std::array<core::lit_t, W> const& rail) noexcept {
    std::int64_t acc = 0;
    std::int64_t p = 1;
    for (std::size_t i = 0; i < W; ++i) {
      if (m.get(rail[i])) acc += p;
      p *= -2;
    }
    return acc;
  }

  // Patrón mod 2^W → valor canónico de la caja NB (wrap_nb — ADR-011).
  [[nodiscard]] static std::int64_t decode_tc(solver::model const& m,
                                              std::array<core::lit_t, W> const& rail) noexcept {
    std::uint64_t u = 0;
    for (std::size_t i = 0; i < W; ++i)
      if (m.get(rail[i])) u |= (std::uint64_t{1} << i);
    if (u > static_cast<std::uint64_t>(negabinary<W>::max()))
      u -= (std::uint64_t{1} << W);
    return static_cast<std::int64_t>(u);
  }

  kind kind_ = kind::concrete;
  engine* engine_ = nullptr;
  std::int64_t re_val_ = 0;
  std::int64_t im_val_ = 0;
  std::array<core::lit_t, W> re_rail_{};
  std::array<core::lit_t, W> im_rail_{};
};

// ═══════════════════════════════════════════════════════════════════════════
// Preparación de operandos (§detalle)
// ═══════════════════════════════════════════════════════════════════════════

namespace detail {

// Aritmética de patrones mod 2^W (wrap) para add/sub: extensión con
// re-anclaje de patrones negativos (extend_pattern) y desplazamiento de
// alineación de escala con wrap (el resultado mod 2^W es exacto porque la
// suma mod 2^W es un homomorfismo). ATENCIÓN: los dígitos NB NO son el patrón
// mod 2^W (solo coinciden mod 2); kind::nb se convierte con nb_to_tc.
template<std::size_t Wa, std::size_t Fa, std::size_t W, std::size_t F>
[[nodiscard]] gates::rail prep_wrap(complex<Wa, Fa> const& z, engine& e, bool imag) {
  static_assert(W >= Wa && F >= Fa);
  if (z.is_concrete()) {
    const auto lit = literal_tc_rails<Wa>(imag ? z.im_raw() : z.re_raw());
    gates::rail r(lit.begin(), lit.end());
    r = extend_pattern(e, r, W);
    if (F > Fa) r = gates::shl(r, F - Fa);
    return r;
  }
  const auto pat = imag ? z.im_pattern() : z.re_pattern();
  gates::rail r = (z.representation() == complex<Wa, Fa>::kind::nb) ? nb_to_tc(e, pat)
                                                                    : gates::rail(pat.begin(),
                                                                                  pat.end());
  r = extend_pattern(e, r, W);
  if (F > Fa) r = gates::shl(r, F - Fa);
  return r;
}

// Representación con signo EXACTA a ancho wprime (>= Wa + delta + 1), con
// desplazamiento de escala delta sin wrap. wprime se elige en cada operación
// para que el valor desplazado quepa (multiplicación, comparaciones y
// división simbólica).
template<std::size_t Wa, std::size_t Fa>
[[nodiscard]] gates::rail prep_exact(complex<Wa, Fa> const& z, engine& e, bool imag,
                                     std::size_t wprime, std::size_t delta) {
  assert(wprime >= Wa + 1 + delta);
  gates::rail r;
  if (z.is_concrete()) {
    const auto lit = literal_tc_rails<Wa + 1>(imag ? z.im_raw() : z.re_raw());
    r.assign(lit.begin(), lit.end());
  } else if (z.representation() == complex<Wa, Fa>::kind::nb) {
    const auto pat = imag ? z.im_pattern() : z.re_pattern();
    gates::rail pad(pat.begin(), pat.end());
    pad.resize(Wa + 1, core::false_lit);  // NB con ceros a Wa+1 = mismo valor
    r = nb_to_tc(e, pad);                 // exacto con signo en Wa+1 bits
  } else {
    const auto pat = imag ? z.im_pattern() : z.re_pattern();
    r = tc_to_signed(e, pat);             // patrón → valor con signo en Wa+1
  }
  r = gates::sext(r, wprime);
  if (delta > 0) r = gates::shl(r, delta);  // exacto: wprime >= Wa + delta + 1
  return r;
}

}  // namespace detail

// ═══════════════════════════════════════════════════════════════════════════
// Operaciones básicas (§10): W = max(Wa,Wb), F = max(Fa,Fb).
// Ruta concreta: aritmética constexpr del host + re-codificación NB.
// Ruta simbólica: circuitos de patrones TC; resultados en NB canónico.
// add/sub: wrap mod 2^W (ambas rutas) — ADR-011.
// mul/div: resultado debe caber en la caja NB (desborde → excepción / UNSAT).
// ═══════════════════════════════════════════════════════════════════════════

template<std::size_t Wa, std::size_t Fa, std::size_t Wb, std::size_t Fb>
[[nodiscard]] constexpr auto operator+(complex<Wa, Fa> const& a, complex<Wb, Fb> const& b)
    -> complex<std::max(Wa, Wb), std::max(Fa, Fb)> {
  constexpr std::size_t W = std::max(Wa, Wb);
  constexpr std::size_t F = std::max(Fa, Fb);
  using R = complex<W, F>;
  if (a.is_concrete() && b.is_concrete()) {
    const __int128 re = (static_cast<__int128>(a.re_raw()) << (F - Fa)) +
                        (static_cast<__int128>(b.re_raw()) << (F - Fb));
    const __int128 im = (static_cast<__int128>(a.im_raw()) << (F - Fa)) +
                        (static_cast<__int128>(b.im_raw()) << (F - Fb));
    return R{R::wrap_nb128(re), R::wrap_nb128(im)};  // wrap mod 2^W (ADR-011)
  }
  engine& e = detail::common_engine(a.engine_of(), b.engine_of());
  const auto re = gates::rca(e, detail::prep_wrap<Wa, Fa, W, F>(a, e, false),
                             detail::prep_wrap<Wb, Fb, W, F>(b, e, false));
  const auto im = gates::rca(e, detail::prep_wrap<Wa, Fa, W, F>(a, e, true),
                             detail::prep_wrap<Wb, Fb, W, F>(b, e, true));
  return R::from_tc_rails(e, re, im);  // patrón mod 2^W (wrap) — ADR-011
}

template<std::size_t Wa, std::size_t Fa, std::size_t Wb, std::size_t Fb>
[[nodiscard]] constexpr auto operator-(complex<Wa, Fa> const& a, complex<Wb, Fb> const& b)
    -> complex<std::max(Wa, Wb), std::max(Fa, Fb)> {
  constexpr std::size_t W = std::max(Wa, Wb);
  constexpr std::size_t F = std::max(Fa, Fb);
  using R = complex<W, F>;
  if (a.is_concrete() && b.is_concrete()) {
    const __int128 re = (static_cast<__int128>(a.re_raw()) << (F - Fa)) -
                        (static_cast<__int128>(b.re_raw()) << (F - Fb));
    const __int128 im = (static_cast<__int128>(a.im_raw()) << (F - Fa)) -
                        (static_cast<__int128>(b.im_raw()) << (F - Fb));
    return R{R::wrap_nb128(re), R::wrap_nb128(im)};  // wrap mod 2^W (ADR-011)
  }
  engine& e = detail::common_engine(a.engine_of(), b.engine_of());
  const auto re = gates::rcs(e, detail::prep_wrap<Wa, Fa, W, F>(a, e, false),
                             detail::prep_wrap<Wb, Fb, W, F>(b, e, false));
  const auto im = gates::rcs(e, detail::prep_wrap<Wa, Fa, W, F>(a, e, true),
                             detail::prep_wrap<Wb, Fb, W, F>(b, e, true));
  return R::from_tc_rails(e, re, im);  // patrón mod 2^W (wrap) — ADR-011
}

template<std::size_t Wa, std::size_t Fa, std::size_t Wb, std::size_t Fb>
[[nodiscard]] constexpr auto operator*(complex<Wa, Fa> const& a, complex<Wb, Fb> const& b)
    -> complex<std::max(Wa, Wb), std::max(Fa, Fb)> {
  constexpr std::size_t W = std::max(Wa, Wb);
  constexpr std::size_t F = std::max(Fa, Fb);
  constexpr std::size_t K = std::min(Fa, Fb);
  using R = complex<W, F>;
  if (a.is_concrete() && b.is_concrete()) {
    const __int128 ar = a.re_raw(), ai = a.im_raw();
    const __int128 br = b.re_raw(), bi = b.im_raw();
    const __int128 pre = ar * br - ai * bi;
    const __int128 pim = ar * bi + ai * br;
    return R{fixed_point::trunc_shr_i128(pre, K), fixed_point::trunc_shr_i128(pim, K)};
  }
  engine& e = detail::common_engine(a.engine_of(), b.engine_of());
  constexpr std::size_t WW = 2 * W + 2;  // entradas exactas de W+1 bits → producto exacto
  const auto ar = detail::prep_exact<Wa, Fa>(a, e, false, WW, 0);
  const auto ai = detail::prep_exact<Wa, Fa>(a, e, true, WW, 0);
  const auto br = detail::prep_exact<Wb, Fb>(b, e, false, WW, 0);
  const auto bi = detail::prep_exact<Wb, Fb>(b, e, true, WW, 0);
  const auto ac = gates::pmul(e, ar, br);
  const auto bd = gates::pmul(e, ai, bi);
  const auto ad = gates::pmul(e, ar, bi);
  const auto bc = gates::pmul(e, ai, br);
  gates::rail re_full = gates::rcs(e, ac, bd);
  gates::rail im_full = gates::rca(e, ad, bc);

  // Recorte hacia cero (ADR-012): trunc(P/2^K) = (P + corr) >> K (aritmético)
  // con corr = (signo de P) ? 2^K−1 : 0; cuando los K bits bajos son cero el
  // corr no cambia el resultado (P + 2^K−1 >> K == P >> K).
  if constexpr (K > 0) {
    const core::lit_t sre = re_full[WW - 1];
    const core::lit_t sim = im_full[WW - 1];
    gates::rail cre(WW, core::false_lit);
    gates::rail cim(WW, core::false_lit);
    for (std::size_t i = 0; i < K; ++i) {
      cre[i] = sre;
      cim[i] = sim;
    }
    re_full = gates::rca(e, re_full, cre);
    im_full = gates::rca(e, im_full, cim);
  }

  gates::rail re_tc(re_full.begin() + static_cast<std::ptrdiff_t>(K),
                    re_full.begin() + static_cast<std::ptrdiff_t>(K + W));
  gates::rail im_tc(im_full.begin() + static_cast<std::ptrdiff_t>(K),
                    im_full.begin() + static_cast<std::ptrdiff_t>(K + W));

  // Chequeo de desborde (ADR-011/ADR-012): el cociente truncado Q debe caber
  // en la caja NB. (1) los bits por encima de la ventana con signo de W+1
  // bits deben igualar al signo — Q ∈ [−2^W, 2^W−1]; (2) el valor Q completo
  // (bits [K, K+W+1), signo incluido) debe estar en [min_NB, max_NB].
  // Juntas ⟺ Q ∈ caja NB. (La ventana sola es Q mod 2^W y NO basta.)
  const std::size_t sgn = K + W;  // bit de signo de la ventana W+1
  for (std::size_t j = sgn + 1; j < WW; ++j) {
    e.add_unit(core::neg(gates::xor2(e, re_full[j], re_full[sgn])));
    e.add_unit(core::neg(gates::xor2(e, im_full[j], im_full[sgn])));
  }
  const gates::rail re_v(re_full.begin() + static_cast<std::ptrdiff_t>(K),
                         re_full.begin() + static_cast<std::ptrdiff_t>(K + W + 1));
  const gates::rail im_v(im_full.begin() + static_cast<std::ptrdiff_t>(K),
                         im_full.begin() + static_cast<std::ptrdiff_t>(K + W + 1));
  const auto lo = detail::literal_tc_rails<W + 1>(negabinary<W>::min());
  const auto hi = detail::literal_tc_rails<W + 1>(negabinary<W>::max());
  const gates::rail lo_r(lo.begin(), lo.end()), hi_r(hi.begin(), hi.end());
  e.add_unit(gates::and2(e, gates::sle(e, lo_r, re_v), gates::sle(e, re_v, hi_r)));
  e.add_unit(gates::and2(e, gates::sle(e, lo_r, im_v), gates::sle(e, im_v, hi_r)));

  return R::from_tc_rails(e, re_tc, im_tc);  // ventana (caja NB) — ADR-011
}

template<std::size_t Wa, std::size_t Fa, std::size_t Wb, std::size_t Fb>
[[nodiscard]] constexpr auto operator/(complex<Wa, Fa> const& a, complex<Wb, Fb> const& b)
    -> complex<std::max(Wa, Wb), std::max(Fa, Fb)> {
  constexpr std::size_t W = std::max(Wa, Wb);
  constexpr std::size_t F = std::max(Fa, Fb);
  constexpr std::size_t K = F + Fb - Fa;
  using R = complex<W, F>;
  if (a.is_concrete() && b.is_concrete()) {
    const __int128 ar = a.re_raw(), ai = a.im_raw();
    const __int128 cr = b.re_raw(), ci = b.im_raw();
    const __int128 den = cr * cr + ci * ci;
    if (den == 0) throw std::domain_error("satx::num::complex: division by zero");
    return R{fixed_point::round_scaled_div(ar * cr + ai * ci, den, K),
             fixed_point::round_scaled_div(ai * cr - ar * ci, den, K)};
  }
  engine& e = detail::common_engine(a.engine_of(), b.engine_of());
  R r{e};
  const auto m = r * b;  // m = mul(r, b): caja NB, escala F
  // m == a bit a bit sobre la representación exacta alineada (ADR-013).
  constexpr std::size_t Wpp = std::max(W + (F - Fa), W) + 1;
  const auto m_re = detail::prep_exact<W, F>(m, e, false, Wpp, 0);
  const auto m_im = detail::prep_exact<W, F>(m, e, true, Wpp, 0);
  const auto a_re = detail::prep_exact<Wa, Fa>(a, e, false, Wpp, F - Fa);
  const auto a_im = detail::prep_exact<Wa, Fa>(a, e, true, Wpp, F - Fa);
  for (std::size_t i = 0; i < Wpp; ++i) {
    e.add_unit(core::neg(gates::xor2(e, m_re[i], a_re[i])));
    e.add_unit(core::neg(gates::xor2(e, m_im[i], a_im[i])));
  }
  // Divisor no nulo con los rieles NATIVOS de b (sin alineación de escala).
  if (b.is_concrete()) {
    if (b.re_raw() == 0 && b.im_raw() == 0) e.add_unit(core::false_lit);
  } else {
    const auto re_pat = b.re_pattern();
    gates::rail nz(re_pat.begin(), re_pat.end());
    const auto im_pat = b.im_pattern();
    nz.insert(nz.end(), im_pat.begin(), im_pat.end());
    e.add_unit(gates::reduce_or(e, nz));
  }
  return r;
}

// ═══════════════════════════════════════════════════════════════════════════
// Utilidades (§12)
// ═══════════════════════════════════════════════════════════════════════════

template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr complex<W, F> conj(complex<W, F> const& z) {
  if (z.is_concrete())
    return complex<W, F>{z.re_raw(), complex<W, F>::wrap_nb128(-z.im_raw())};
  engine& e = *z.engine_of();
  gates::rail zero(W, core::false_lit);
  const auto im_pat = z.im_pattern();
  gates::rail im =
      (z.representation() == complex<W, F>::kind::nb) ? detail::nb_to_tc(e, im_pat)
                                                      : gates::rail(im_pat.begin(), im_pat.end());
  const auto neg_im = gates::rcs(e, zero, im);  // −im mod 2^W
  const auto re_pat = z.re_pattern();
  gates::rail re =
      (z.representation() == complex<W, F>::kind::nb) ? detail::nb_to_tc(e, re_pat)
                                                      : gates::rail(re_pat.begin(), re_pat.end());
  return complex<W, F>::from_tc_rails(e, re, neg_im);
}

template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr complex<W, F> neg(complex<W, F> const& z) {
  if (z.is_concrete())
    return complex<W, F>{complex<W, F>::wrap_nb128(-z.re_raw()),
                         complex<W, F>::wrap_nb128(-z.im_raw())};
  engine& e = *z.engine_of();
  gates::rail zero(W, core::false_lit);
  const auto re_pat = z.re_pattern();
  gates::rail re =
      (z.representation() == complex<W, F>::kind::nb) ? detail::nb_to_tc(e, re_pat)
                                                      : gates::rail(re_pat.begin(), re_pat.end());
  const auto im_pat = z.im_pattern();
  gates::rail im =
      (z.representation() == complex<W, F>::kind::nb) ? detail::nb_to_tc(e, im_pat)
                                                      : gates::rail(im_pat.begin(), im_pat.end());
  return complex<W, F>::from_tc_rails(e, gates::rcs(e, zero, re),
                                      gates::rcs(e, zero, im));
}

// Parte real / imaginaria como complex<W,F> (la otra componente es 0 exacto).
template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr complex<W, F> real(complex<W, F> const& z) {
  if (z.is_concrete()) return complex<W, F>{z.re_raw(), std::int64_t{0}};
  engine& e = *z.engine_of();
  std::array<core::lit_t, W> zero;
  zero.fill(core::false_lit);
  if (z.representation() == complex<W, F>::kind::nb)
    return complex<W, F>::from_nb_rails(e, z.re_pattern(), zero);
  const auto re_pat = z.re_pattern();
  return complex<W, F>::from_tc_rails(e, gates::rail(re_pat.begin(), re_pat.end()),
                                      gates::rail(zero.begin(), zero.end()));
}

template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr complex<W, F> imag(complex<W, F> const& z) {
  if (z.is_concrete()) return complex<W, F>{z.im_raw(), std::int64_t{0}};
  engine& e = *z.engine_of();
  std::array<core::lit_t, W> zero;
  zero.fill(core::false_lit);
  if (z.representation() == complex<W, F>::kind::nb)
    return complex<W, F>::from_nb_rails(e, z.im_pattern(), zero);
  const auto im_pat = z.im_pattern();
  return complex<W, F>::from_tc_rails(e, gates::rail(im_pat.begin(), im_pat.end()),
                                      gates::rail(zero.begin(), zero.end()));
}

// Igualdad bit a bit por riel con alineación de escala EXACTA (ADR-013) → literal.
template<std::size_t Wa, std::size_t Fa, std::size_t Wb, std::size_t Fb>
[[nodiscard]] core::lit_t eq_lit(engine& e, complex<Wa, Fa> const& a, complex<Wb, Fb> const& b) {
  constexpr std::size_t W = std::max(Wa, Wb);
  constexpr std::size_t F = std::max(Fa, Fb);
  if (a.is_concrete() && b.is_concrete()) {
    const __int128 re_a = (static_cast<__int128>(a.re_raw()) << (F - Fa));
    const __int128 re_b = (static_cast<__int128>(b.re_raw()) << (F - Fb));
    const __int128 im_a = (static_cast<__int128>(a.im_raw()) << (F - Fa));
    const __int128 im_b = (static_cast<__int128>(b.im_raw()) << (F - Fb));
    return (re_a == re_b && im_a == im_b) ? core::true_lit : core::false_lit;
  }
  // Vía rápida: el MISMO objeto simbólico NB es idéntico a sí mismo (mismo tipo).
  if constexpr (Wa == Wb && Fa == Fb) {
    if (a.engine_of() == &e && b.engine_of() == &e &&
        a.representation() == complex<Wa, Fa>::kind::nb &&
        b.representation() == complex<Wb, Fb>::kind::nb &&
        a.re_pattern() == b.re_pattern() && a.im_pattern() == b.im_pattern())
      return core::true_lit;
  }
  constexpr std::size_t Wpp = std::max(Wa + (F - Fa), Wb + (F - Fb)) + 1;
  const auto re_eq = gates::eq(e, detail::prep_exact<Wa, Fa>(a, e, false, Wpp, F - Fa),
                               detail::prep_exact<Wb, Fb>(b, e, false, Wpp, F - Fb));
  const auto im_eq = gates::eq(e, detail::prep_exact<Wa, Fa>(a, e, true, Wpp, F - Fa),
                               detail::prep_exact<Wb, Fb>(b, e, true, Wpp, F - Fb));
  return gates::and2(e, re_eq, im_eq);
}

// Igualdad de valor: solo para constantes (ruta concreta).
template<std::size_t Wa, std::size_t Fa, std::size_t Wb, std::size_t Fb>
[[nodiscard]] constexpr bool operator==(complex<Wa, Fa> const& a, complex<Wb, Fb> const& b) {
  if (!a.is_concrete() || !b.is_concrete())
    throw std::logic_error("satx: operator== on symbolic complex; use eq_lit(engine&, a, b)");
  constexpr std::size_t F = std::max(Fa, Fb);
  const __int128 re_a = (static_cast<__int128>(a.re_raw()) << (F - Fa));
  const __int128 re_b = (static_cast<__int128>(b.re_raw()) << (F - Fb));
  const __int128 im_a = (static_cast<__int128>(a.im_raw()) << (F - Fa));
  const __int128 im_b = (static_cast<__int128>(b.im_raw()) << (F - Fb));
  return re_a == re_b && im_a == im_b;
}

template<std::size_t Wa, std::size_t Fa, std::size_t Wb, std::size_t Fb>
[[nodiscard]] constexpr bool operator!=(complex<Wa, Fa> const& a, complex<Wb, Fb> const& b) {
  return !(a == b);
}

// ═══════════════════════════════════════════════════════════════════════════
// Operaciones derivadas — Etapa 2 (§10.7)
// ═══════════════════════════════════════════════════════════════════════════

// |z|² = z · conj(z). La parte imaginaria se anula estructuralmente (im == 0
// exacto); el desborde sigue la política de mul (§10.3).
template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr complex<W, F> abs_sq(complex<W, F> const& z) {
  return z * conj(z);
}

// Comparación lexicográfica (re, im): a < b ⟺ re_a < re_b ∨ (re_a == re_b ∧ im_a < im_b).
// Ruta concreta: pliega a true_lit/false_lit; ruta simbólica: 1× EQ + 2× SLE +
// 1× AND + 1× OR sobre rieles exactos alineados (ADR-013, §10.6).
template<std::size_t Wa, std::size_t Fa, std::size_t Wb, std::size_t Fb>
[[nodiscard]] core::lit_t lt_lit(engine& e, complex<Wa, Fa> const& a, complex<Wb, Fb> const& b) {
  constexpr std::size_t W = std::max(Wa, Wb);
  constexpr std::size_t F = std::max(Fa, Fb);
  if (a.is_concrete() && b.is_concrete()) {
    const __int128 re_a = (static_cast<__int128>(a.re_raw()) << (F - Fa));
    const __int128 re_b = (static_cast<__int128>(b.re_raw()) << (F - Fb));
    const __int128 im_a = (static_cast<__int128>(a.im_raw()) << (F - Fa));
    const __int128 im_b = (static_cast<__int128>(b.im_raw()) << (F - Fb));
    const bool lt = re_a < re_b || (re_a == re_b && im_a < im_b);
    return lt ? core::true_lit : core::false_lit;
  }
  if constexpr (Wa == Wb && Fa == Fb) {
    if (&a == &b) return core::false_lit;  // a < a es falso
  }
  constexpr std::size_t Wpp = std::max(Wa + (F - Fa), Wb + (F - Fb)) + 1;
  const auto a_re = detail::prep_exact<Wa, Fa>(a, e, false, Wpp, F - Fa);
  const auto a_im = detail::prep_exact<Wa, Fa>(a, e, true, Wpp, F - Fa);
  const auto b_re = detail::prep_exact<Wb, Fb>(b, e, false, Wpp, F - Fb);
  const auto b_im = detail::prep_exact<Wb, Fb>(b, e, true, Wpp, F - Fb);
  const core::lit_t re_eq = gates::eq(e, a_re, b_re);
  const core::lit_t re_lt = core::neg(gates::sle(e, b_re, a_re));  // a_re < b_re
  const core::lit_t im_lt = core::neg(gates::sle(e, b_im, a_im));  // a_im < b_im
  return gates::or2(e, re_lt, gates::and2(e, re_eq, im_lt));
}

// a <= b ⟺ re_a < re_b ∨ (re_a == re_b ∧ im_a <= im_b).
template<std::size_t Wa, std::size_t Fa, std::size_t Wb, std::size_t Fb>
[[nodiscard]] core::lit_t le_lit(engine& e, complex<Wa, Fa> const& a, complex<Wb, Fb> const& b) {
  constexpr std::size_t W = std::max(Wa, Wb);
  constexpr std::size_t F = std::max(Fa, Fb);
  if (a.is_concrete() && b.is_concrete()) {
    const __int128 re_a = (static_cast<__int128>(a.re_raw()) << (F - Fa));
    const __int128 re_b = (static_cast<__int128>(b.re_raw()) << (F - Fb));
    const __int128 im_a = (static_cast<__int128>(a.im_raw()) << (F - Fa));
    const __int128 im_b = (static_cast<__int128>(b.im_raw()) << (F - Fb));
    const bool le = re_a < re_b || (re_a == re_b && im_a <= im_b);
    return le ? core::true_lit : core::false_lit;
  }
  if constexpr (Wa == Wb && Fa == Fb) {
    if (&a == &b) return core::true_lit;  // a <= a es verdadero
  }
  constexpr std::size_t Wpp = std::max(Wa + (F - Fa), Wb + (F - Fb)) + 1;
  const auto a_re = detail::prep_exact<Wa, Fa>(a, e, false, Wpp, F - Fa);
  const auto a_im = detail::prep_exact<Wa, Fa>(a, e, true, Wpp, F - Fa);
  const auto b_re = detail::prep_exact<Wb, Fb>(b, e, false, Wpp, F - Fb);
  const auto b_im = detail::prep_exact<Wb, Fb>(b, e, true, Wpp, F - Fb);
  const core::lit_t re_eq = gates::eq(e, a_re, b_re);
  const core::lit_t re_lt = core::neg(gates::sle(e, b_re, a_re));  // a_re < b_re
  const core::lit_t im_le = gates::sle(e, a_im, b_im);
  return gates::or2(e, re_lt, gates::and2(e, re_eq, im_le));
}

// Potencia con exponente entero (square & multiply sobre operator*).
// n == 0 → one; n == 1 → z; n < 0 → z^n == (1/z)^(−n−1) / z (la división impone
// la restricción de divisor no nulo en la ruta simbólica, §10.4). La ruta
// concreta trunca en cada multiplicación (ADR-003), igual que la simbólica.
template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr complex<W, F> pow(complex<W, F> const& z, int n) {
  if (n == 0) return complex<W, F>{fixed_point::scale(F), std::int64_t{0}};
  if (n == 1) return z;
  if (n < 0) {
    const complex<W, F> one = (z.engine_of() != nullptr)
                                  ? complex<W, F>::one(*z.engine_of())
                                  : complex<W, F>{fixed_point::scale(F), std::int64_t{0}};
    return pow(one / z, -(n + 1)) / z;  // −(n+1) evita el desborde de INT_MIN
  }
  const complex<W, F> half = pow(z, n / 2);
  const complex<W, F> sq = half * half;
  return (n % 2 == 0) ? sq : sq * z;
}

// Raíz n-ésima. Ruta concreta: z^(1/n) con precisión de host y redondeo
// half-away-from-zero (ADR-003); fuera del rango NB → std::out_of_range.
// Ruta simbólica: y libre con y^n == z (eq_lit como unit, §10.4).
template<std::size_t W, std::size_t F>
[[nodiscard]] complex<W, F> root_cbe(complex<W, F> const& z, int n) {
  if (n < 1) throw std::domain_error("satx::num::root_cbe: n must be >= 1");
  if (n == 1) return z;
  if (z.is_concrete()) {
    const std::complex<double> v =
        std::pow(z.value(solver::model{}), 1.0 / static_cast<double>(n));
    return complex<W, F>{v.real(), v.imag()};
  }
  engine& e = *z.engine_of();
  complex<W, F> y{e};
  complex<W, F> power = y;
  for (int i = 1; i < n; ++i) power = power * y;
  e.add_unit(eq_lit(e, power, z));
  return y;
}

}  // namespace satx::num
