#pragma once
// satx::quantum — estado de n qubits como vector de 2^n amplitudes CBE.
// Ruta concreta: amplitudes constantes (aritmética de punto fijo plegada).
// Ruta simbólica: amplitudes como variables libres + CNF; la inicialización
// |0…0⟩ se impone con restricciones de igualdad (eq_lit).

#include "../num/complex.hpp"
#include "../solver/model.hpp"
#include "gates.hpp"

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace satx::quantum {

template<std::size_t W, std::size_t F>
class qstate {
  static_assert(W >= 4, "satx::quantum: se requiere W >= 4");
  static_assert(F + 2 <= W,
                "satx::quantum: se requiere F + 2 <= W (la unidad 2^F debe caber en el rango NB)");

public:
  using value_type = cx<W, F>;

  // Ruta concreta: |0…0⟩ con amplitudes constantes.
  // ADR-011: n ∈ [1, 20] (evita 1<<n UB y asignaciones descomunales).
  explicit qstate(std::size_t n) : n_(n) {
    if (n == 0 || n > 20) throw std::invalid_argument("satx::quantum: 1 <= n <= 20");
    amps_.assign(std::size_t{1} << n, value_type{0.0, 0.0});
    amps_[0] = value_type{1.0, 0.0};
  }

  // Ruta simbólica: amplitudes libres con el estado inicial |0…0⟩ impuesto.
  explicit qstate(satx::engine& e, std::size_t n) : n_(n), engine_(&e) {
    if (n == 0 || n > 20) throw std::invalid_argument("satx::quantum: 1 <= n <= 20");
    amps_.reserve(std::size_t{1} << n);
    for (std::size_t x = 0; x < (std::size_t{1} << n); ++x) {
      amps_.emplace_back(e);
      e.add_unit(satx::num::eq_lit(e, amps_.back(),
                                   x == 0 ? value_type::one(e) : value_type::zero(e)));
    }
  }

  [[nodiscard]] std::size_t num_qubits() const noexcept { return n_; }
  [[nodiscard]] std::size_t size() const noexcept { return amps_.size(); }
  [[nodiscard]] value_type const& amplitude(std::size_t x) const {
    if (x >= amps_.size()) throw std::out_of_range("satx::quantum: índice de amplitud fuera de rango");
    return amps_[x];
  }
  [[nodiscard]] bool is_symbolic() const noexcept { return engine_ != nullptr; }
  [[nodiscard]] satx::engine* engine() const noexcept { return engine_; }

  // Compuerta de 1 qubit sobre el qubit q (actualiza pares (x, x|2^q)).
  void apply1(std::size_t q, qgate2<W, F> const& g) {
    if (q >= n_) throw std::out_of_range("satx::quantum: qubit fuera de rango");
    const std::size_t m = std::size_t{1} << q;
    for (std::size_t x = 0; x < amps_.size(); ++x) {
      if ((x & m) != 0) continue;
      const value_type a = amps_[x];
      const value_type b = amps_[x | m];
      amps_[x] = g.m[0] * a + g.m[1] * b;
      amps_[x | m] = g.m[2] * a + g.m[3] * b;
    }
  }

  // Compuerta de 2 qubits: fila = q1, columna = q2 (índice = 2·bit(q1) + bit(q2)).
  void apply2(std::size_t q1, std::size_t q2, qgate4<W, F> const& g) {
    if (q1 >= n_ || q2 >= n_ || q1 == q2)
      throw std::out_of_range("satx::quantum: par de qubits inválido");
    const std::size_t m1 = std::size_t{1} << q1;  // fila
    const std::size_t m2 = std::size_t{1} << q2;  // columna
    for (std::size_t x = 0; x < amps_.size(); ++x) {
      if ((x & (m1 | m2)) != 0) continue;
      const std::size_t x00 = x;
      const std::size_t x01 = x | m2;
      const std::size_t x10 = x | m1;
      const std::size_t x11 = x | m1 | m2;
      const value_type a = amps_[x00];
      const value_type b = amps_[x01];
      const value_type c = amps_[x10];
      const value_type d = amps_[x11];
      amps_[x00] = g.m[0] * a + g.m[1] * b + g.m[2] * c + g.m[3] * d;
      amps_[x01] = g.m[4] * a + g.m[5] * b + g.m[6] * c + g.m[7] * d;
      amps_[x10] = g.m[8] * a + g.m[9] * b + g.m[10] * c + g.m[11] * d;
      amps_[x11] = g.m[12] * a + g.m[13] * b + g.m[14] * c + g.m[15] * d;
    }
  }

  // Valor esperado de Z en el qubit q: Σ_x s_x·|ψ_x|² (parte imaginaria ≡ 0).
  [[nodiscard]] value_type expect_z(std::size_t q) const {
    if (q >= n_) throw std::out_of_range("satx::quantum: qubit fuera de rango");
    const std::size_t m = std::size_t{1} << q;
    value_type acc{0.0, 0.0};
    for (std::size_t x = 0; x < amps_.size(); ++x) {
      const value_type p = satx::num::abs_sq(amps_[x]);
      acc = ((x & m) != 0) ? acc - p : acc + p;
    }
    return acc;
  }

  // Solo ruta concreta: Σ|ψ_x|² (debe valer ≈ 1 por unitariedad).
  [[nodiscard]] double norm_sq_concrete() const {
    if (is_symbolic())
      throw std::logic_error("satx::quantum: norm_sq_concrete sobre estado simbólico");
    const satx::solver::model m{};
    double acc = 0.0;
    for (auto const& a : amps_) acc += satx::num::abs_sq(a).value(m).real();
    return acc;
  }

  // Decodificación post-solve de todas las amplitudes.
  [[nodiscard]] std::vector<std::complex<double>> decode(satx::solver::model const& m) const {
    std::vector<std::complex<double>> out;
    out.reserve(amps_.size());
    for (auto const& a : amps_) out.push_back(a.value(m));
    return out;
  }

private:
  std::size_t n_;
  satx::engine* engine_ = nullptr;
  std::vector<value_type> amps_;
};

}  // namespace satx::quantum
