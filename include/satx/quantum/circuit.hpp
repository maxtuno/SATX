#pragma once
// satx::quantum — circuito (lista de pasos), aplicación forward/adjunta,
// construcción aleatoria sobre grilla 2D/cadena y bucle OTOC de Quantum Echoes.

#include "state.hpp"

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <random>
#include <stdexcept>
#include <variant>
#include <vector>

namespace satx::quantum {

template<std::size_t W, std::size_t F>
struct step1 {
  std::size_t q;
  qgate2<W, F> g;
};

template<std::size_t W, std::size_t F>
struct step2 {
  std::size_t q1;
  std::size_t q2;
  qgate4<W, F> g;
};

// Lista de pasos de un circuito cuántico.
template<std::size_t W, std::size_t F>
class qcircuit {
public:
  void push(std::size_t q, qgate2<W, F> const& g) { steps_.push_back(step1<W, F>{q, g}); }
  void push(std::size_t q1, std::size_t q2, qgate4<W, F> const& g) {
    steps_.push_back(step2<W, F>{q1, q2, g});
  }

  [[nodiscard]] std::size_t size() const noexcept { return steps_.size(); }
  [[nodiscard]] bool empty() const noexcept { return steps_.empty(); }

  // Aplicación forward (U) y adjunta en orden inverso (U†).
  void apply_to(qstate<W, F>& s) const {
    for (auto const& st : steps_) {
      if (auto const* p = std::get_if<step1<W, F>>(&st)) {
        s.apply1(p->q, p->g);
      } else {
        auto const& p2 = std::get<step2<W, F>>(st);
        s.apply2(p2.q1, p2.q2, p2.g);
      }
    }
  }

  void apply_adjoint_to(qstate<W, F>& s) const {
    for (auto it = steps_.rbegin(); it != steps_.rend(); ++it) {
      if (auto const* p = std::get_if<step1<W, F>>(&*it)) {
        s.apply1(p->q, p->g.adjoint());
      } else {
        auto const& p2 = std::get<step2<W, F>>(*it);
        s.apply2(p2.q1, p2.q2, p2.g.adjoint());
      }
    }
  }

private:
  std::vector<std::variant<step1<W, F>, step2<W, F>>> steps_;
};

// Circuito aleatorio de `layers` capas: 1 qubit en todos los qubits + 2 qubits
// sobre vecinos de una grilla 2D (si n es cuadrado perfecto) o de una cadena.
template<std::size_t W, std::size_t F>
[[nodiscard]] qcircuit<W, F> random_circuit(std::size_t n, std::size_t layers,
                                            std::uint32_t seed) {
  if (n == 0) throw std::invalid_argument("satx::quantum: n >= 1");
  qcircuit<W, F> c;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> pick1(0, 6);  // X Y H S T RX RY
  std::uniform_int_distribution<int> pick2(0, 3);  // CZ CNOT iSWAP fSim
  std::uniform_real_distribution<double> ang(0.0, 2.0 * std::numbers::pi_v<double>);

  auto r1 = [&]() -> qgate2<W, F> {
    switch (pick1(rng)) {
      case 0: return x2<W, F>();
      case 1: return y2<W, F>();
      case 2: return h2<W, F>();
      case 3: return s2<W, F>();
      case 4: return t2<W, F>();
      case 5: return rx2<W, F>(ang(rng));
      default: return ry2<W, F>(ang(rng));
    }
  };
  auto r2 = [&]() -> qgate4<W, F> {
    switch (pick2(rng)) {
      case 0: return cz4<W, F>();
      case 1: return cnot4<W, F>();
      case 2: return iswap4<W, F>();
      default: return fsim4<W, F>(ang(rng), ang(rng));
    }
  };

  std::size_t side = 1;
  while ((side + 1) * (side + 1) <= n) ++side;
  const bool grid2d = side >= 2 && side * side == n;

  for (std::size_t layer = 0; layer < layers; ++layer) {
    for (std::size_t q = 0; q < n; ++q) c.push(q, r1());
    if (n >= 2) {
      if (grid2d) {
        const std::size_t p = layer % 4;
        for (std::size_t r = 0; r < side; ++r) {
          for (std::size_t cc = 0; cc < side; ++cc) {
            const std::size_t i = r * side + cc;
            if ((p == 0 || p == 1) && cc + 1 < side && (cc & 1U) == p)
              c.push(i, i + 1, r2());
            if ((p == 2 || p == 3) && r + 1 < side && (r & 1U) == (p - 2U))
              c.push(i, i + side, r2());
          }
        }
      } else {
        const std::size_t off = layer & 1U;
        for (std::size_t q = off; q + 1 < n; q += 2) c.push(q, q + 1, r2());
      }
    }
  }
  return c;
}

// Señal sin inversión temporal (C(1) del artículo): U y medición ⟨Z⟩ en qM.
template<std::size_t W, std::size_t F>
[[nodiscard]] cx<W, F> forward_signal(qstate<W, F> psi, qcircuit<W, F> const& U,
                                      std::size_t qM) {
  U.apply_to(psi);
  return psi.expect_z(qM);
}

// Bucle del eco: k pasadas de (U → B → U† → M) y medición ⟨Z⟩ en qM.
// k = 1 → C(2) (1.er orden); k = 2 → C(4) (2.º orden).
template<std::size_t W, std::size_t F>
[[nodiscard]] cx<W, F> otoc_echo(qstate<W, F> psi, qcircuit<W, F> const& U,
                                 std::size_t qB, qgate2<W, F> const& B,
                                 std::size_t qM, qgate2<W, F> const& M,
                                 std::size_t k) {
  for (std::size_t i = 0; i < k; ++i) {
    U.apply_to(psi);
    psi.apply1(qB, B);
    U.apply_adjoint_to(psi);
    psi.apply1(qM, M);
  }
  return psi.expect_z(qM);
}

// Impone U†U = I sobre una compuerta de 1 qubit: columnas ortonormales
// (|c0|² = 1, |c1|² = 1, conj(c0)·c1 = 0). Uso: ruta de aprendizaje.
template<std::size_t W, std::size_t F>
void constrain_unitary2(satx::engine& e, qgate2<W, F> const& g) {
  using C = cx<W, F>;
  auto const& m = g.m;
  e.add_unit(satx::num::eq_lit(e, satx::num::abs_sq(m[0]) + satx::num::abs_sq(m[2]), C::one(e)));
  e.add_unit(satx::num::eq_lit(e, satx::num::abs_sq(m[1]) + satx::num::abs_sq(m[3]), C::one(e)));
  e.add_unit(satx::num::eq_lit(e, satx::num::conj(m[0]) * m[1] + satx::num::conj(m[2]) * m[3],
                               C::zero(e)));
}

}  // namespace satx::quantum
