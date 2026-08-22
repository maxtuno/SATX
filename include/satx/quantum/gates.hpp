#pragma once
// satx::quantum — compuertas cuánticas como matrices CBE (catálogo estándar).
// qgate2: matriz 2×2 row-major (m00, m01; m10, m11). qgate4: 4×4 row-major con
// índice de bloque = 2·bit(fila) + bit(columna); en cnot4(q1, q2), q1 es control.

#include "../num/complex.hpp"

#include <array>
#include <cmath>
#include <cstddef>

namespace satx::quantum {

template<std::size_t W, std::size_t F>
using cx = satx::num::complex<W, F>;

// Coeficiente CBE desde un par (re, im) en coma flotante del host.
template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr cx<W, F> mk(double re, double im) {
  return cx<W, F>::from_float(re, im);
}

// Compuerta de 1 qubit: matriz 2×2 row-major.
template<std::size_t W, std::size_t F>
struct qgate2 {
  std::array<cx<W, F>, 4> m{};

  [[nodiscard]] constexpr qgate2 adjoint() const {
    return qgate2{{satx::num::conj(m[0]), satx::num::conj(m[2]), satx::num::conj(m[1]),
                   satx::num::conj(m[3])}};
  }
};

// Compuerta de 2 qubits: matriz 4×4 row-major, índice = 2·bit(q1) + bit(q2).
template<std::size_t W, std::size_t F>
struct qgate4 {
  std::array<cx<W, F>, 16> m{};

  [[nodiscard]] constexpr qgate4 adjoint() const {
    qgate4 r{};
    for (std::size_t i = 0; i < 4; ++i)
      for (std::size_t j = 0; j < 4; ++j) r.m[i * 4 + j] = satx::num::conj(m[j * 4 + i]);
    return r;
  }
};

// Compuerta libre: coeficientes simbólicos (ruta de aprendizaje / Hamiltonian learning).
template<std::size_t W, std::size_t F>
[[nodiscard]] qgate2<W, F> free_gate2(satx::engine& e) {
  return qgate2<W, F>{{cx<W, F>{e}, cx<W, F>{e}, cx<W, F>{e}, cx<W, F>{e}}};
}

template<std::size_t W, std::size_t F>
[[nodiscard]] qgate4<W, F> free_gate4(satx::engine& e) {
  qgate4<W, F> g{};
  for (auto& c : g.m) c = cx<W, F>{e};
  return g;
}

// ── catálogo de 1 qubit ──
// Nota: la unidad 1.0 exige 2^F <= max_NB(W); la capa qstate impone F + 2 <= W.

template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr qgate2<W, F> id2() {
  return qgate2<W, F>{{mk<W, F>(1, 0), mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(1, 0)}};
}

template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr qgate2<W, F> x2() {
  return qgate2<W, F>{{mk<W, F>(0, 0), mk<W, F>(1, 0), mk<W, F>(1, 0), mk<W, F>(0, 0)}};
}

template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr qgate2<W, F> y2() {
  return qgate2<W, F>{{mk<W, F>(0, 0), mk<W, F>(0, -1), mk<W, F>(0, 1), mk<W, F>(0, 0)}};
}

template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr qgate2<W, F> z2() {
  return qgate2<W, F>{{mk<W, F>(1, 0), mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(-1, 0)}};
}

template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr qgate2<W, F> h2() {
  constexpr double a = 0.7071067811865475244;  // 1/√2
  return qgate2<W, F>{{mk<W, F>(a, 0), mk<W, F>(a, 0), mk<W, F>(a, 0), mk<W, F>(-a, 0)}};
}

template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr qgate2<W, F> s2() {
  return qgate2<W, F>{{mk<W, F>(1, 0), mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(0, 1)}};
}

template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr qgate2<W, F> t2() {
  constexpr double a = 0.7071067811865475244;  // e^{iπ/4} = a + a·i
  return qgate2<W, F>{{mk<W, F>(1, 0), mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(a, a)}};
}

// Rotación en X: (cos θ/2, −i·sin θ/2; −i·sin θ/2, cos θ/2).
template<std::size_t W, std::size_t F>
[[nodiscard]] qgate2<W, F> rx2(double theta) {
  const double c = std::cos(theta / 2.0);
  const double s = std::sin(theta / 2.0);
  return qgate2<W, F>{{mk<W, F>(c, 0), mk<W, F>(0, -s), mk<W, F>(0, -s), mk<W, F>(c, 0)}};
}

// Rotación en Y: (cos θ/2, −sin θ/2; sin θ/2, cos θ/2).
template<std::size_t W, std::size_t F>
[[nodiscard]] qgate2<W, F> ry2(double theta) {
  const double c = std::cos(theta / 2.0);
  const double s = std::sin(theta / 2.0);
  return qgate2<W, F>{{mk<W, F>(c, 0), mk<W, F>(-s, 0), mk<W, F>(s, 0), mk<W, F>(c, 0)}};
}

// ── catálogo de 2 qubits ──

template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr qgate4<W, F> id4() {
  return qgate4<W, F>{{mk<W, F>(1, 0), mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(0, 0),
                       mk<W, F>(0, 0), mk<W, F>(1, 0), mk<W, F>(0, 0), mk<W, F>(0, 0),
                       mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(1, 0), mk<W, F>(0, 0),
                       mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(1, 0)}};
}

template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr qgate4<W, F> cz4() {
  return qgate4<W, F>{{mk<W, F>(1, 0), mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(0, 0),
                       mk<W, F>(0, 0), mk<W, F>(1, 0), mk<W, F>(0, 0), mk<W, F>(0, 0),
                       mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(1, 0), mk<W, F>(0, 0),
                       mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(-1, 0)}};
}

// CNOT: q1 = control, q2 = objetivo. Filas 2 y 3 intercambian |10⟩ ↔ |11⟩.
template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr qgate4<W, F> cnot4() {
  return qgate4<W, F>{{mk<W, F>(1, 0), mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(0, 0),
                       mk<W, F>(0, 0), mk<W, F>(1, 0), mk<W, F>(0, 0), mk<W, F>(0, 0),
                       mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(1, 0),
                       mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(1, 0), mk<W, F>(0, 0)}};
}

// iSWAP: intercambia |01⟩ ↔ i·|10⟩.
template<std::size_t W, std::size_t F>
[[nodiscard]] constexpr qgate4<W, F> iswap4() {
  return qgate4<W, F>{{mk<W, F>(1, 0), mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(0, 0),
                       mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(0, 1), mk<W, F>(0, 0),
                       mk<W, F>(0, 0), mk<W, F>(0, 1), mk<W, F>(0, 0), mk<W, F>(0, 0),
                       mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(1, 0)}};
}

// fSim(θ, φ): sub-bloque |01⟩⟨01|↔|10⟩⟨10| de ángulo θ y fase condicional e^{−iφ}.
template<std::size_t W, std::size_t F>
[[nodiscard]] qgate4<W, F> fsim4(double theta, double phi) {
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  return qgate4<W, F>{{mk<W, F>(1, 0), mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(0, 0),
                       mk<W, F>(0, 0), mk<W, F>(c, 0), mk<W, F>(0, -s), mk<W, F>(0, 0),
                       mk<W, F>(0, 0), mk<W, F>(0, -s), mk<W, F>(c, 0), mk<W, F>(0, 0),
                       mk<W, F>(0, 0), mk<W, F>(0, 0), mk<W, F>(0, 0),
                       mk<W, F>(std::cos(-phi), std::sin(-phi))}};
}

}  // namespace satx::quantum
