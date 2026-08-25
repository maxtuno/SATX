// changeover_lotsizing — catálogo 3.2: secuenciación de coladas y cambios de
// aleación (lot-sizing + changeover), kernel WMIBO.
//
//   min  Σ c_{p,p'}·z_{p,p',t} + Σ h_p·I_{p,t}
//   s.a. AMO sobre y_{p,t}                              (una aleación por período)
//        I_{p,t} = I_{p,t-1} + q_{p,t} − d_{p,t}        (balance de inventario)
//        y_{p,t} ⇒ q_{p,t} ≥ q_min, q_{p,t} ≤ M·y_{p,t} (lote mínimo)
//        (¬y_{p,t-1} ∨ ¬y_{p',t} ∨ z_{p,p',t})          (cambio de aleación)
//
// Verificación: recomputación del balance, lotes mínimos y objetivo en el
// host.

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace satx::solver::wmibo;

int main() {
    constexpr int T = 6, P = 3;
    // demanda por (aleación, período): cada aleación exige 2 lotes mínimos
    // (q ≥ 2) en períodos distintos → la instancia es factible (p0 en t0,
    // p1 en t1, p2 en t2) y el plan es único.
    const int d[P][T] = {
        {0, 1, 0, 1, 0, 0},
        {0, 0, 1, 0, 1, 0},
        {0, 0, 0, 1, 0, 1},
    };
    const double h[P] = {1.0, 2.0, 1.0};
    const double ch[P][P] = {
        {0.0, 3.0, 4.0},
        {3.0, 0.0, 3.0},
        {4.0, 3.0, 0.0},
    };
    constexpr double QMIN = 2.0, QCAP = 6.0;

    model m{"changeover_lotsizing"};

    std::vector<std::vector<variable*>> y(P, std::vector<variable*>(T));
    std::vector<std::vector<variable*>> q(P, std::vector<variable*>(T));
    std::vector<std::vector<variable*>> I(P, std::vector<variable*>(T));
    for (int p = 0; p < P; ++p)
        for (int t = 0; t < T; ++t) {
            y[p][t] = &m.add_boolean("y" + std::to_string(p) + "_" + std::to_string(t));
            q[p][t] = &m.add_integer("q" + std::to_string(p) + "_" + std::to_string(t), 0.0, QCAP);
            I[p][t] = &m.add_integer("I" + std::to_string(p) + "_" + std::to_string(t), 0.0, 10.0);
        }
    std::vector<std::vector<std::vector<variable*>>> z(
        P, std::vector<std::vector<variable*>>(P, std::vector<variable*>(T)));
    for (int p = 0; p < P; ++p)
        for (int p2 = 0; p2 < P; ++p2)
            if (p != p2)
                for (int t = 1; t < T; ++t)
                    z[p][p2][t] = &m.add_boolean("z" + std::to_string(p) + "_" + std::to_string(p2) +
                                                 "_" + std::to_string(t));

    // una aleación por período
    for (int t = 0; t < T; ++t)
        for (int p = 0; p < P; ++p)
            for (int p2 = p + 1; p2 < P; ++p2)
                m.add_hard_clause({~lit{*y[p][t]}, ~lit{*y[p2][t]}});

    // balance de inventario
    for (int p = 0; p < P; ++p) {
        m.add_constraint((*I[p][0]) - (*q[p][0]) + static_cast<double>(d[p][0]), compare::eq, 0.0);
        for (int t = 1; t < T; ++t)
            m.add_constraint((*I[p][t]) - (*I[p][t - 1]) - (*q[p][t]) + static_cast<double>(d[p][t]),
                             compare::eq, 0.0);
    }

    // lote mínimo y acotación
    for (int p = 0; p < P; ++p)
        for (int t = 0; t < T; ++t) {
            m.add_indicator(lit{*y[p][t]}, (*q[p][t]) - QMIN, compare::ge, 0.0);
            m.add_constraint((*q[p][t]) - QCAP * (*y[p][t]), compare::le, 0.0);
        }

    // cambio de aleación: y_{p,t-1} ∧ y_{p',t} → z_{p,p',t}
    for (int p = 0; p < P; ++p)
        for (int p2 = 0; p2 < P; ++p2)
            if (p != p2)
                for (int t = 1; t < T; ++t)
                    m.add_hard_clause({~lit{*y[p][t - 1]}, ~lit{*y[p2][t]}, lit{*z[p][p2][t]}});

    expr obj;
    for (int p = 0; p < P; ++p) {
        for (int p2 = 0; p2 < P; ++p2)
            if (p != p2)
                for (int t = 1; t < T; ++t) obj += ch[p][p2] * (*z[p][p2][t]);
        for (int t = 0; t < T; ++t) obj += h[p] * (*I[p][t]);
    }
    m.set_objective(obj, sense::min);

    const auto s = m.solve();

    std::printf("estado: %s\n",
                s.state() == status::optimal ? "OPTIMAL"
                : s.state() == status::infeasible ? "INFEASIBLE" : "UNKNOWN");
    if (s.state() != status::optimal || !s.has_values()) return EXIT_FAILURE;

    std::printf("costo óptimo (wmibo) = %.0f\n", s.objective());

    // ── verificación en el host ─────────────────────────────────────────────
    double verified = 0.0;
    for (int t = 0; t < T; ++t) {
        std::printf("  período %d:", t);
        int count = 0;
        for (int p = 0; p < P; ++p) {
            const bool on = s.boolean(*y[p][t]);
            const double qv = s.integer(*q[p][t]);
            const double iv = s.integer(*I[p][t]);
            if (on) {
                ++count;
                std::printf("  aleación %d: q=%g, I=%g", p, qv, iv);
                if (qv < QMIN || qv > QCAP) {
                    std::printf("\nVERIFICACIÓN FALLIDA: lote de %d fuera de rango\n", p);
                    return EXIT_FAILURE;
                }
            } else if (qv != 0.0) {
                std::printf("\nVERIFICACIÓN FALLIDA: y=0 con q=%g\n", qv);
                return EXIT_FAILURE;
            }
            verified += h[p] * iv;
        }
        std::printf("\n");
        if (count > 1) {
            std::printf("VERIFICACIÓN FALLIDA: más de una aleación en el período %d\n", t);
            return EXIT_FAILURE;
        }
    }
    // balance
    for (int p = 0; p < P; ++p) {
        double inv = 0.0;
        for (int t = 0; t < T; ++t) {
            inv = inv + s.integer(*q[p][t]) - d[p][t];
            if (std::abs(inv - s.integer(*I[p][t])) > 1e-6) {
                std::printf("VERIFICACIÓN FALLIDA: balance del producto %d\n", p);
                return EXIT_FAILURE;
            }
        }
    }
    // cambios y objetivo
    for (int p = 0; p < P; ++p)
        for (int p2 = 0; p2 < P; ++p2)
            if (p != p2)
                for (int t = 1; t < T; ++t) {
                    if (s.boolean(*z[p][p2][t])) verified += ch[p][p2];
                    if (s.boolean(*z[p][p2][t]) && !(s.boolean(*y[p][t - 1]) && s.boolean(*y[p2][t]))) {
                        std::printf("VERIFICACIÓN FALLIDA: z sin transición %d→%d en %d\n", p, p2, t);
                        return EXIT_FAILURE;
                    }
                }
    std::printf("costo verificado = %.0f %s\n", verified,
                std::abs(verified - s.objective()) < 1e-6 ? "(ok)" : "(FAIL)");
    if (std::abs(verified - s.objective()) >= 1e-6) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
