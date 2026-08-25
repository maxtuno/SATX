// facility_location — catálogo 1.1: asignación de almacenes a clientes
// (Facility Location), kernel PIXIE (MIP).
//
//   min  Σ_w f_w·y_w + Σ_{w,c} c_{w,c}·d_c·x_{w,c}
//   s.a. Σ_w x_{w,c} = 1                 ∀c        (cada cliente, un almacén)
//        Σ_c d_c·x_{w,c} ≤ Q_w·y_w       ∀w        (capacidad)
//        x_{w,c} ≤ y_w                   ∀w,c      (acoplamiento)
//        y_w, x_{w,c} binarias
//
// Verificación: oráculo del host — fuerza bruta sobre los 2^W subconjuntos de
// almacenes abiertos y las |S|^C asignaciones de clientes.

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace satx::solver::pixie;

int main() {
    // Almacenes y clientes.
    constexpr int W = 4, C = 6;
    const double f[W] = {30.0, 25.0, 20.0, 28.0};  // costo fijo de apertura
    const double Q[W] = {30.0, 20.0, 25.0, 35.0};  // capacidad
    const double d[C] = {5.0, 8.0, 6.0, 9.0, 7.0, 4.0};  // demanda
    // costo de servicio c_{w,c} (por unidad de demanda).
    const double c[W][C] = {
        {3.0, 5.0, 4.0, 2.0, 6.0, 7.0},
        {5.0, 2.0, 6.0, 4.0, 3.0, 5.0},
        {4.0, 6.0, 2.0, 5.0, 4.0, 3.0},
        {6.0, 4.0, 5.0, 3.0, 2.0, 4.0},
    };

    model m{"facility_location"};

    std::vector<variable*> y;
    for (int w = 0; w < W; ++w) y.push_back(&m.add_binary("y" + std::to_string(w)));
    std::vector<variable*> x(W * C);
    for (int w = 0; w < W; ++w)
        for (int c_ = 0; c_ < C; ++c_)
            x[w * C + c_] = &m.add_binary("x" + std::to_string(w) + "_" + std::to_string(c_));

    // Cada cliente atendido por exactamente un almacén abierto.
    for (int c_ = 0; c_ < C; ++c_) {
        expr e;
        for (int w = 0; w < W; ++w) e += *x[w * C + c_];
        m.add_constraint(e, compare::eq, 1.0);
    }
    // Capacidad: Σ_c d_c·x_{w,c} ≤ Q_w·y_w.
    for (int w = 0; w < W; ++w) {
        expr cap;
        for (int c_ = 0; c_ < C; ++c_) cap += d[c_] * (*x[w * C + c_]);
        m.add_constraint(cap - Q[w] * (*y[w]), compare::le, 0.0);
    }
    // Acoplamiento: x_{w,c} ≤ y_w.
    for (int w = 0; w < W; ++w)
        for (int c_ = 0; c_ < C; ++c_)
            m.add_constraint(*x[w * C + c_] - (*y[w]), compare::le, 0.0);

    // Objetivo: apertura + transporte.
    expr obj;
    for (int w = 0; w < W; ++w) {
        obj += f[w] * (*y[w]);
        for (int c_ = 0; c_ < C; ++c_) obj += c[w][c_] * d[c_] * (*x[w * C + c_]);
    }
    m.set_objective(obj, sense::min);

    const auto s = m.solve();

    std::printf("estado: %s\n",
                s.state() == status::optimal ? "OPTIMAL"
                : s.state() == status::infeasible ? "INFEASIBLE" : "UNKNOWN");
    if (s.state() != status::optimal || !s.has_values()) return EXIT_FAILURE;

    std::printf("costo óptimo (pixie) = %.3f\n", s.objective());
    std::vector<int> open, assign(C, -1);
    for (int w = 0; w < W; ++w) {
        const bool ow = s.value(*y[w]) > 0.5;
        std::printf("  almacén %d: %s\n", w, ow ? "ABIERTO" : "cerrado");
        if (ow) open.push_back(w);
    }
    for (int c_ = 0; c_ < C; ++c_) {
        for (int w = 0; w < W; ++w)
            if (s.value(*x[w * C + c_]) > 0.5) assign[c_] = w;
        std::printf("  cliente %d (d=%g) → almacén %d\n", c_, d[c_], assign[c_]);
    }

    // ── oráculo del host ────────────────────────────────────────────────────
    // Fuerza bruta: subconjuntos de almacenes abiertos × asignaciones de
    // clientes (|S|^C), filtrando por capacidad.
    double oracle = std::numeric_limits<double>::infinity();
    for (int mask = 1; mask < (1 << W); ++mask) {
        std::vector<int> s_open;
        for (int w = 0; w < W; ++w)
            if (mask & (1 << w)) s_open.push_back(w);
        const std::size_t base = s_open.size();
        std::vector<int> a(C, 0);
        const std::size_t total = [&] {
            std::size_t t = 1;
            for (int c_ = 0; c_ < C; ++c_) t *= base;
            return t;
        }();
        for (std::size_t code = 0; code < total; ++code) {
            std::size_t k = code;
            for (int c_ = 0; c_ < C; ++c_) { a[c_] = static_cast<int>(k % base); k /= base; }
            std::vector<double> load(W, 0.0);
            double cost = 0.0;
            bool cap_ok = true;
            for (int c_ = 0; c_ < C && cap_ok; ++c_) {
                const int w = s_open[static_cast<std::size_t>(a[c_])];
                load[static_cast<std::size_t>(w)] += d[c_];
                if (load[static_cast<std::size_t>(w)] > Q[w] + 1e-9) cap_ok = false;
                cost += c[w][c_] * d[c_];
            }
            if (cap_ok) {
                for (int w : s_open) cost += f[w];
                oracle = std::min(oracle, cost);
            }
        }
    }

    std::printf("costo óptimo (oráculo) = %.3f %s\n", oracle,
                std::abs(s.objective() - oracle) < 1e-6 ? "(ok)" : "(FAIL)");
    if (std::abs(s.objective() - oracle) >= 1e-6) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
