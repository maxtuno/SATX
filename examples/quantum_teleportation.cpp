// quantum_teleportation — teletransporte cuántico de |φ⟩ con medición clásica.
//
// Protocolo (3 qubits): |φ⟩⊗|00⟩ → H⊗CNOT (Bell en q1,q2) → CNOT(q0,q1) →
// H(q0). El estado final es ½·Σ_{a,b} |a,b⟩ ⊗ X^b Z^a |φ⟩. Este ejemplo
// verifica la identidad del teletransporte para las CUATRO salidas de
// medición (a,b) comparando cada amplitud con el oráculo en doble precisión.
//
// La ruta es concreta (simulación de punto fijo CBE, W=24, F=12); la misma
// identidad es resoluble por SAT en la ruta simbólica (véase quantum_bell y
// quantum_learning para el patrón simbólico).

#include <satx/satx.hpp>

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>

int main() {
  using namespace satx::quantum;
  constexpr std::size_t W = 24, F = 12;
  constexpr double theta = 0.9;

  const satx::solver::model m{};

  // |φ⟩ = RY(θ)|0⟩ en doble precisión (oráculo independiente).
  const double c = std::cos(theta / 2.0);
  const double s = std::sin(theta / 2.0);
  const std::complex<double> phi[2] = {{c, 0.0}, {s, 0.0}};

  // Protocolo en la grilla CBE.
  qstate<W, F> q{3};
  q.apply1(0, ry2<W, F>(theta));   // preparar |φ⟩ en q0
  q.apply1(1, h2<W, F>());         // Bell
  q.apply2(1, 2, cnot4<W, F>());
  q.apply2(0, 1, cnot4<W, F>());   // CNOT + H de Alice
  q.apply1(0, h2<W, F>());

  const auto amps = q.decode(m);

  // Verificación: amp(x) == ½·(−1)^(a·(q⊕b))·φ[q⊕b] para x = (a,b,q).
  double max_err = 0.0;
  for (std::size_t x = 0; x < amps.size(); ++x) {
    const int a = static_cast<int>(x & 1U);
    const int b = static_cast<int>((x >> 1U) & 1U);
    const int qq = static_cast<int>((x >> 2U) & 1U);
    const int q2 = qq ^ b;
    const double phase = ((a * q2) & 1) ? -1.0 : 1.0;
    const std::complex<double> expected = 0.5 * phase * phi[q2];
    max_err = std::max(max_err, std::abs(amps[x] - expected));
  }

  std::cout << "Teleportación de |φ⟩ = RY(" << theta << ")|0⟩ (W=" << W << ", F=" << F
            << ")\n";
  std::cout << "  Amplitudes del estado final:\n";
  for (std::size_t x = 0; x < amps.size(); ++x)
    std::cout << "  |" << ((x >> 2) & 1U) << ((x >> 1) & 1U) << (x & 1U)
              << "⟩ = " << amps[x] << '\n';
  std::cout << "  Error máximo contra el oráculo: " << max_err << '\n';

  const double tol = 0.01;  // redondeo de coeficientes ~2^−13 por operación
  if (max_err <= tol) {
    std::cout << "  Teletransporte verificado (error ≤ " << tol << ").\n";
    return EXIT_SUCCESS;
  }
  std::cerr << "  VERIFICACIÓN FALLIDA\n";
  return EXIT_FAILURE;
}
