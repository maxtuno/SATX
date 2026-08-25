// blending_mip — catálogo 4.2: mezcla de minerales/carbón con calidad (MIP),
// kernel PIXIE.
//
//   min  Σ_{i,p} c_{i,p}·x_{i,p}
//   s.a. Σ_i x_{i,p} = D_p                          (tonelaje del pedido)
//        Σ_p x_{i,p} ≤ A_i                          (disponibilidad de pila)
//        L_{p,k}·D_p ≤ Σ_i q_{i,k}·x_{i,p} ≤ U_{p,k}·D_p   (bandas de calidad)
//        x_{i,p} ≥ q_min·y_{i,p}, x_{i,p} ≤ A_i·y_{i,p}    (lote mínimo)
//
// Verificación: (a) factibilidad de la solución en el host; (b) el óptimo
// MIP no supera la relajación LP (pure_lp) — y coincide cuando el LP es
// entero.

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace satx::solver::pixie;

int main() {
    constexpr int PILES = 4, ORDERS = 2, QUALS = 2;
    constexpr double QMIN = 5.0;
    // calidades (ley Fe, azufre) y disponibilidad
    const double q[PILES][QUALS] = {
        {0.62, 0.012},
        {0.55, 0.018},
        {0.68, 0.008},
        {0.60, 0.015},
    };
    const double A[PILES] = {40.0, 30.0, 35.0, 25.0};
    const double D[ORDERS] = {50.0, 45.0};
    const double Lo[ORDERS][QUALS] = {
        {0.60, 0.000},
        {0.58, 0.000},
    };
    const double Up[ORDERS][QUALS] = {
        {0.65, 0.014},
        {0.63, 0.016},
    };
    const double cost[PILES] = {12.0, 9.0, 14.0, 10.0};

    const auto build = [&](model& m) {
        std::vector<std::vector<variable*>> x(PILES, std::vector<variable*>(ORDERS));
        std::vector<std::vector<variable*>> y(PILES, std::vector<variable*>(ORDERS));
        for (int i = 0; i < PILES; ++i)
            for (int p = 0; p < ORDERS; ++p) {
                x[i][p] = &m.add_continuous("x" + std::to_string(i) + "_" + std::to_string(p), 0.0,
                                            A[i]);
                y[i][p] = &m.add_binary("y" + std::to_string(i) + "_" + std::to_string(p));
            }
        // tonelaje del pedido
        for (int p = 0; p < ORDERS; ++p) {
            expr e;
            for (int i = 0; i < PILES; ++i) e += *x[i][p];
            m.add_constraint(e, compare::eq, D[p]);
        }
        // disponibilidad
        for (int i = 0; i < PILES; ++i) {
            expr e;
            for (int p = 0; p < ORDERS; ++p) e += *x[i][p];
            m.add_constraint(e, compare::le, A[i]);
        }
        // bandas de calidad
        for (int p = 0; p < ORDERS; ++p)
            for (int k = 0; k < QUALS; ++k) {
                expr e;
                for (int i = 0; i < PILES; ++i) e += q[i][k] * (*x[i][p]);
                m.add_constraint(e, compare::ge, Lo[p][k] * D[p]);
                if (Up[p][k] > 0.0) m.add_constraint(e, compare::le, Up[p][k] * D[p]);
            }
        // lote mínimo
        for (int i = 0; i < PILES; ++i)
            for (int p = 0; p < ORDERS; ++p) {
                m.add_constraint((*x[i][p]) - QMIN * (*y[i][p]), compare::ge, 0.0);
                m.add_constraint((*x[i][p]) - A[i] * (*y[i][p]), compare::le, 0.0);
            }
        // objetivo
        expr obj;
        for (int i = 0; i < PILES; ++i)
            for (int p = 0; p < ORDERS; ++p) obj += cost[i] * (*x[i][p]);
        m.set_objective(obj, sense::min);
        return std::pair{x, y};
    };

    model m{"blending"};
    const auto [x, y] = build(m);
    const auto s = m.solve();

    std::printf("estado: %s\n",
                s.state() == status::optimal ? "OPTIMAL"
                : s.state() == status::infeasible ? "INFEASIBLE" : "UNKNOWN");
    if (s.state() != status::optimal || !s.has_values()) return EXIT_FAILURE;

    std::printf("costo óptimo (pixie MIP) = %.3f\n", s.objective());
    for (int i = 0; i < PILES; ++i)
        for (int p = 0; p < ORDERS; ++p)
            if (s.value(*x[i][p]) > 1e-6)
                std::printf("  pila %d → pedido %d: %g t (y=%d)\n", i, p, s.value(*x[i][p]),
                            s.value(*y[i][p]) > 0.5 ? 1 : 0);

    // ── verificación (a): factibilidad en el host ───────────────────────────
    for (int p = 0; p < ORDERS; ++p) {
        double tot = 0.0;
        for (int i = 0; i < PILES; ++i) tot += s.value(*x[i][p]);
        if (std::abs(tot - D[p]) > 1e-6) {
            std::printf("VERIFICACIÓN FALLIDA: tonelaje del pedido %d\n", p);
            return EXIT_FAILURE;
        }
        for (int k = 0; k < QUALS; ++k) {
            double v = 0.0;
            for (int i = 0; i < PILES; ++i) v += q[i][k] * s.value(*x[i][p]);
            if (v + 1e-6 < Lo[p][k] * D[p] || v > Up[p][k] * D[p] + 1e-6) {
                std::printf("VERIFICACIÓN FALLIDA: calidad %d del pedido %d\n", k, p);
                return EXIT_FAILURE;
            }
        }
    }
    for (int i = 0; i < PILES; ++i) {
        double tot = 0.0;
        for (int p = 0; p < ORDERS; ++p) {
            tot += s.value(*x[i][p]);
            const bool on = s.value(*y[i][p]) > 0.5;
            if (on && s.value(*x[i][p]) + 1e-6 < QMIN) {
                std::printf("VERIFICACIÓN FALLIDA: lote mínimo pila %d\n", i);
                return EXIT_FAILURE;
            }
            if (!on && s.value(*x[i][p]) > 1e-6) {
                std::printf("VERIFICACIÓN FALLIDA: y=0 con x>0 en pila %d\n", i);
                return EXIT_FAILURE;
            }
        }
        if (tot > A[i] + 1e-6) {
            std::printf("VERIFICACIÓN FALLIDA: disponibilidad pila %d\n", i);
            return EXIT_FAILURE;
        }
    }

    // ── verificación (b): relajación LP como cota ───────────────────────────
    options opt;
    opt.pure_lp = true;
    model lp{"blending_lp"};
    build(lp);
    const auto lp_sol = lp.solve(opt);
    std::printf("relajación LP = %.3f %s\n", lp_sol.objective(),
                s.objective() >= lp_sol.objective() - 1e-6 ? "(ok: MIP ≥ LP)" : "(FAIL)");
    if (s.objective() < lp_sol.objective() - 1e-6) return EXIT_FAILURE;

    std::printf("verificación del host: OK\n");
    return EXIT_SUCCESS;
}
