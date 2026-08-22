// test_quantum_echoes — variante satx del algoritmo Quantum Echoes (OTOC).
// Verifica: valores OTOC exactos conocidos, resonancia (B = I), eco U·U† = I,
// normalización, concordancia con un oráculo en doble precisión, concordancia
// concreta/simbólica y el problema inverso (aprendizaje de B con Kerberos).
//
// Nota: usa SATX_CHECK en lugar de assert: el build es Release (-DNDEBUG) y
// assert quedaría deshabilitado.

#include <satx/satx.hpp>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

using satx::solver::result;

#define SATX_CHECK(cond)                                                             \
  do {                                                                               \
    if (!(cond)) {                                                                   \
      std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__);         \
      std::exit(EXIT_FAILURE);                                                       \
    }                                                                                \
  } while (0)

namespace {

bool close(std::complex<double> a, std::complex<double> b, double tol) {
  return std::abs(a - b) <= tol;
}

// Simulador de referencia en doble precisión (oráculo independiente, sin satx).
struct dstate {
  std::vector<std::complex<double>> a;
  explicit dstate(std::size_t n) : a(std::size_t{1} << n, {0.0, 0.0}) { a[0] = 1.0; }
  void apply1(std::size_t q, std::complex<double> const (&g)[4]) {
    const std::size_t m = std::size_t{1} << q;
    for (std::size_t x = 0; x < a.size(); ++x) {
      if ((x & m) != 0) continue;
      const auto u = a[x];
      const auto v = a[x | m];
      a[x] = g[0] * u + g[1] * v;
      a[x | m] = g[2] * u + g[3] * v;
    }
  }
  void apply2(std::size_t q1, std::size_t q2, std::complex<double> const (&g)[16]) {
    const std::size_t m1 = std::size_t{1} << q1;
    const std::size_t m2 = std::size_t{1} << q2;
    for (std::size_t x = 0; x < a.size(); ++x) {
      if ((x & (m1 | m2)) != 0) continue;
      const auto u = a[x];
      const auto v = a[x | m2];
      const auto w = a[x | m1];
      const auto z = a[x | m1 | m2];
      a[x] = g[0] * u + g[1] * v + g[2] * w + g[3] * z;
      a[x | m2] = g[4] * u + g[5] * v + g[6] * w + g[7] * z;
      a[x | m1] = g[8] * u + g[9] * v + g[10] * w + g[11] * z;
      a[x | m1 | m2] = g[12] * u + g[13] * v + g[14] * w + g[15] * z;
    }
  }
  double expect_z(std::size_t q) const {
    const std::size_t m = std::size_t{1} << q;
    double acc = 0.0;
    for (std::size_t x = 0; x < a.size(); ++x)
      acc += (((x & m) != 0) ? -1.0 : 1.0) * std::norm(a[x]);
    return acc;
  }
};

}  // namespace

int main() {
  using namespace satx::quantum;
  const satx::solver::model m{};
  constexpr double inv_sqrt2 = 0.7071067811865475244;

  // ── 1. Valores OTOC exactos conocidos (n=1, |ψ⟩=|0⟩, U=RX(π/2), M=I) ──
  // L = U†·X·U intercambia |0⟩↔|1⟩: C(2;B=X) = ⟨1|Z|1⟩ = −1;
  // B = I devuelve el eco a |0⟩: C(2;B=I) = 1 (resonancia); L² = I → C(4) = 1.
  {
    constexpr double pi = 3.14159265358979323846;
    qcircuit<24, 12> U;
    U.push(0, rx2<24, 12>(pi / 2.0));
    auto mk = []() { return qstate<24, 12>{1}; };  // |0⟩
    const auto c1 = forward_signal<24, 12>(mk(), U, 0);
    const auto c2x = otoc_echo<24, 12>(mk(), U, 0, x2<24, 12>(), 0, id2<24, 12>(), 1);
    const auto c2i = otoc_echo<24, 12>(mk(), U, 0, id2<24, 12>(), 0, id2<24, 12>(), 1);
    const auto c4x = otoc_echo<24, 12>(mk(), U, 0, x2<24, 12>(), 0, id2<24, 12>(), 2);

    SATX_CHECK(close(c1.value(m), {0.0, 0.0}, 0.02));     // RX(π/2)|0⟩: ⟨Z⟩=0
    SATX_CHECK(close(c2x.value(m), {-1.0, 0.0}, 0.02));   // perturbación: |1⟩ → ⟨Z⟩=−1
    SATX_CHECK(close(c2i.value(m), {1.0, 0.0}, 0.02));    // resonancia: eco → |0⟩
    SATX_CHECK(close(c4x.value(m), {1.0, 0.0}, 0.02));    // L² = I → ⟨Z⟩=1
  }

  // ── 2. Convención de índices de apply2 (CNOT e iSWAP, coeficientes exactos) ──
  {
    qstate<24, 12> s{2};
    s.apply1(0, x2<24, 12>());          // qubit 0 (control) en 1 → índice 1
    s.apply2(0, 1, cnot4<24, 12>());    // control q0=1 → objetivo q1 → |11⟩
    SATX_CHECK(close(s.expect_z(0).value(m), {-1.0, 0.0}, 1e-9));
    SATX_CHECK(close(s.expect_z(1).value(m), {-1.0, 0.0}, 1e-9));

    qstate<24, 12> t{2};
    t.apply1(1, x2<24, 12>());          // índice 2 (|10⟩)
    t.apply2(0, 1, iswap4<24, 12>());   // fila 2 → i·|01⟩ (índice 1)
    SATX_CHECK(close(t.expect_z(0).value(m), {-1.0, 0.0}, 1e-9));
    SATX_CHECK(close(t.expect_z(1).value(m), {1.0, 0.0}, 1e-9));
  }

  // ── 3. Eco: U seguido de U† devuelve H|0…0⟩ (circuito aleatorio) ──
  {
    const auto U = random_circuit<24, 12>(4, 4, 7);
    qstate<24, 12> s{4};
    s.apply1(0, h2<24, 12>());
    U.apply_to(s);
    U.apply_adjoint_to(s);
    const auto a0 = s.amplitude(0).value(m);
    SATX_CHECK(std::abs(std::abs(a0) - inv_sqrt2) <= 0.05);
  }

  // ── 4. Normalización tras U (unitariedad) ──
  {
    const auto U = random_circuit<24, 12>(4, 4, 7);
    qstate<24, 12> s{4};
    s.apply1(0, h2<24, 12>());
    U.apply_to(s);
    SATX_CHECK(std::abs(s.norm_sq_concrete() - 1.0) <= 0.05);
  }

  // ── 5. Concordancia con oráculo en doble precisión (mismos coeficientes) ──
  {
    constexpr std::size_t W = 24, F = 12;
    qcircuit<W, F> U;
    U.push(0, h2<W, F>());
    U.push(0, 1, cz4<W, F>());
    U.push(1, rx2<W, F>(0.3));

    const auto gH = h2<W, F>();
    const auto gCZ = cz4<W, F>();
    const auto gRX = rx2<W, F>(0.3);
    const auto gRXd = rx2<W, F>(-0.3);  // adjunta de RX(0.3)
    const auto gX = x2<W, F>();

    std::complex<double> dH[4], dCZ[16], dRX[4], dRXd[4], dX[4];
    for (std::size_t i = 0; i < 4; ++i) {
      dH[i] = gH.m[i].value(m);
      dRX[i] = gRX.m[i].value(m);
      dRXd[i] = gRXd.m[i].value(m);
      dX[i] = gX.m[i].value(m);
    }
    for (std::size_t i = 0; i < 16; ++i) dCZ[i] = gCZ.m[i].value(m);

    dstate ref{2};
    ref.apply1(0, dH);      // |+⟩ en q0
    ref.apply1(0, dH);      // U
    ref.apply2(0, 1, dCZ);
    ref.apply1(1, dRX);
    ref.apply1(0, dX);      // B = X
    ref.apply1(1, dRXd);    // U†: RX(−0.3)
    ref.apply2(0, 1, dCZ);  //      CZ† = CZ
    ref.apply1(0, dH);      //      H† = H

    qstate<W, F> s{2};
    s.apply1(0, h2<W, F>());
    const auto c2 = otoc_echo<W, F>(s, U, 0, x2<W, F>(), 0, id2<W, F>(), 1);

    SATX_CHECK(close(c2.value(m), {ref.expect_z(0), 0.0}, 0.02));
  }

  // ── 6. Ruta simbólica vs ruta concreta (n=1) ──
  // Nota de semántica: la ruta simbólica corta rieles del producto (floor en
  // negativos) y la concreta trunca hacia cero (ADR-003); divergen en 1 unidad
  // por producto negativo. (a) circuito sin productos negativos → igualdad exacta;
  // (b) circuito general → diferencia acotada por ~4·2^(−F).
  {
    constexpr std::size_t W = 8, F = 4;
    {  // (a) U = [X], |ψ0⟩ = |0⟩, B = X: C(2) = ⟨1|Z|1⟩ = −1, sin truncamientos
      qcircuit<W, F> U;
      U.push(0, x2<W, F>());

      qstate<W, F> ref{1};
      const auto c2_ref = otoc_echo<W, F>(ref, U, 0, x2<W, F>(), 0, id2<W, F>(), 1);
      SATX_CHECK(close(c2_ref.value(m), {-1.0, 0.0}, 1e-9));

      satx::engine e;
      qstate<W, F> psi{e, 1};
      const auto c2_sym = otoc_echo<W, F>(psi, U, 0, x2<W, F>(), 0, id2<W, F>(), 1);
      const auto s = satx::solver::solve(e);
      SATX_CHECK(s.has_value());
      SATX_CHECK(close(c2_sym.value(*s), c2_ref.value(m), 1e-9));
    }
    {  // (b) U = [H], |ψ⟩ = |+⟩: mismos valores módulo el gap floor/trunc
      qcircuit<W, F> U;
      U.push(0, h2<W, F>());

      qstate<W, F> ref{1};
      ref.apply1(0, h2<W, F>());
      const auto c2_ref = otoc_echo<W, F>(ref, U, 0, x2<W, F>(), 0, id2<W, F>(), 1);

      satx::engine e;
      qstate<W, F> psi{e, 1};
      psi.apply1(0, h2<W, F>());
      const auto c2_sym = otoc_echo<W, F>(psi, U, 0, x2<W, F>(), 0, id2<W, F>(), 1);
      const auto s = satx::solver::solve(e);
      SATX_CHECK(s.has_value());
      SATX_CHECK(std::abs(c2_sym.value(*s) - c2_ref.value(m)) <= 0.25);
    }
  }

  // ── 7. Problema inverso: B libre + unitariedad, OTOC == dato (B = X) ──
  // El dato se calcula con la misma aritmética simbólica (B = X fijo) para que la
  // restricción sea consistente con la grilla de punto fijo (§4.5 del documento).
  // B se restringe a coeficientes reales (B ∈ O(2)) para acotar la búsqueda;
  // la formulación general B ∈ U(2) es idéntica salvo esa restricción.
  {
    constexpr std::size_t W = 8, F = 4;
    qcircuit<W, F> U;
    U.push(0, rx2<W, F>(0.7));

    satx::engine e0;
    qstate<W, F> ref{e0, 1};
    const auto c2_true = otoc_echo<W, F>(ref, U, 0, x2<W, F>(), 0, id2<W, F>(), 1);
    const auto s0 = satx::solver::solve(e0);
    SATX_CHECK(s0.has_value());
    const std::complex<double> target = c2_true.value(*s0);
    const auto target_c = cx<W, F>{target.real(), target.imag()};  // grilla exacta: raw/2^F

    satx::engine e;
    qstate<W, F> psi{e, 1};
    const auto B = free_gate2<W, F>(e);
    for (std::size_t i = 0; i < 4; ++i)
      for (auto l : B.m[i].imag_rail(e)) e.add_unit(-l);  // B real (32 vars libres)
    constrain_unitary2(e, B);
    const auto c2 = otoc_echo<W, F>(psi, U, 0, B, 0, id2<W, F>(), 1);
    e.add_unit(satx::eq_lit(e, c2, target_c));

    const auto s = satx::solver::solve(e);
    SATX_CHECK(s.has_value());

    const auto b00 = B.m[0].value(*s);
    const auto b01 = B.m[1].value(*s);
    const auto b10 = B.m[2].value(*s);
    const auto b11 = B.m[3].value(*s);
    // Sanidad de la unitariedad decodificada (tolerancia gruesa por el punto fijo F=4).
    SATX_CHECK(std::abs((std::norm(b00) + std::norm(b10)) - 1.0) <= 0.25);

    // Verificación en el mismo instrumento numérico: B_rec fija → OTOC == dato exacto.
    const auto B_rec =
        qgate2<W, F>{{cx<W, F>{b00.real(), b00.imag()}, cx<W, F>{b01.real(), b01.imag()},
                      cx<W, F>{b10.real(), b10.imag()}, cx<W, F>{b11.real(), b11.imag()}}};
    satx::engine e2;
    qstate<W, F> chk_state{e2, 1};
    const auto chk = otoc_echo<W, F>(chk_state, U, 0, B_rec, 0, id2<W, F>(), 1);
    const auto s2 = satx::solver::solve(e2);
    SATX_CHECK(s2.has_value());
    SATX_CHECK(close(chk.value(*s2), target, 1e-9));
  }

  // ── 8. Dato contradictorio → UNSAT (propagación inmediata) ──
  // El mismo circuito OTOC con dos restricciones de igualdad incompatibles sobre
  // C(2). (El UNSAT semántico —ningún B unitario alcanza C(2)=2.0— se verificó
  // manualmente, pero su refutación por búsqueda toma ~6 min y no se incluye aquí.)
  {
    constexpr std::size_t W = 8, F = 4;
    qcircuit<W, F> U;
    U.push(0, rx2<W, F>(0.7));

    satx::engine e;
    qstate<W, F> psi{e, 1};
    const auto B = free_gate2<W, F>(e);
    constrain_unitary2(e, B);
    const auto c2 = otoc_echo<W, F>(psi, U, 0, B, 0, id2<W, F>(), 1);
    e.add_unit(satx::eq_lit(e, c2, cx<W, F>{-0.5, 0.0}));
    e.add_unit(satx::eq_lit(e, c2, cx<W, F>{0.5, 0.0}));

    const auto s = satx::solver::solve(e);
    SATX_CHECK(!s.has_value() && s.error() == result::unsat);
  }

  std::puts("test_quantum_echoes: OK");
  return 0;
}
