// cutting_stock — catálogo 3.3: corte de materiales (cutting stock), kernel
// SLIME (generación de patrones) + PIXIE (maestro MIP).
//
// Rollos de longitud L; piezas de longitudes l_i con demandas d_i. Etapa 1:
// SLIME enumera los patrones de corte factibles (Σ k_i·l_i ≤ L) con
// cláusulas de bloqueo. Etapa 2: PIXIE MIP minimiza el número de rollos
// (Σ x_p s.a. Σ a_{i,p}·x_p ≥ d_i, x_p enteros).
//
// Verificación: oráculo del host — asignación de las 5 piezas a b rollos.

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    constexpr int L = 10;
    const int len[3] = {3, 4, 5};
    const int dem[3] = {2, 2, 1};

    // ── etapa 1: enumeración de patrones con SLIME ──────────────────────────
    using I = satx::integer<8>;  // conteos 0..2 y largo L=10 (rango -128..127)

    satx::engine e;
    std::vector<I> k;
    for (int i = 0; i < 3; ++i) {
        k.emplace_back(e);
        e.add_unit(satx::le_lit(e, I{0}, k[i]));
        e.add_unit(satx::le_lit(e, k[i], I{dem[i]}));
    }
    // Σ k_i·l_i ≤ L
    satx::lit_t ok = satx::le_lit(e, k[0] * I{3} + k[1] * I{4} + k[2] * I{5}, I{L});
    e.add_unit(ok);

    struct pattern {
        int a[3];
    };
    std::vector<pattern> patterns;
    while (true) {
        const auto sol = satx::solver::solve(e);
        if (!sol) break;
        pattern p;
        for (int i = 0; i < 3; ++i) p.a[i] = static_cast<int>(k[i].value(*sol));
        patterns.push_back(p);
        // cláusula de bloqueo
        e.add_clause({satx::ne_lit(e, k[0], I{p.a[0]}), satx::ne_lit(e, k[1], I{p.a[1]}),
                      satx::ne_lit(e, k[2], I{p.a[2]})});
    }

    std::printf("patrones generados con SLIME: %zu\n", patterns.size());
    for (std::size_t p = 0; p < patterns.size(); ++p)
        std::printf("  %zu: %d×3 + %d×4 + %d×5\n", p, patterns[p].a[0], patterns[p].a[1],
                    patterns[p].a[2]);

    // ── etapa 2: maestro MIP con PIXIE ──────────────────────────────────────
    satx::solver::pixie::model m{"cutting_stock"};
    std::vector<satx::solver::pixie::variable*> x;
    for (std::size_t p = 0; p < patterns.size(); ++p)
        x.push_back(&m.add_integer("x" + std::to_string(p), 0.0, 6.0));

    for (int i = 0; i < 3; ++i) {
        satx::solver::pixie::expr cov;
        for (std::size_t p = 0; p < patterns.size(); ++p)
            cov += static_cast<double>(patterns[p].a[i]) * (*x[p]);
        m.add_constraint(cov, satx::solver::pixie::compare::ge, static_cast<double>(dem[i]));
    }
    satx::solver::pixie::expr obj;
    for (std::size_t p = 0; p < patterns.size(); ++p) obj += *x[p];
    m.set_objective(obj, satx::solver::pixie::sense::min);

    const auto s = m.solve();

    std::printf("estado: %s\n",
                s.state() == satx::solver::pixie::status::optimal ? "OPTIMAL"
                : s.state() == satx::solver::pixie::status::infeasible ? "INFEASIBLE"
                                                                       : "UNKNOWN");
    if (s.state() != satx::solver::pixie::status::optimal || !s.has_values()) return EXIT_FAILURE;

    std::printf("rollos óptimos (pixie) = %.0f\n", s.objective());
    for (std::size_t p = 0; p < patterns.size(); ++p)
        if (s.value(*x[p]) > 0.5)
            std::printf("  %g× patrón %zu\n", s.value(*x[p]), p);

    // ── oráculo del host ────────────────────────────────────────────────────
    // b rollos: ¿se pueden acomodar las 5 piezas? (piezas individuales 3,3,4,4,5)
    const std::vector<int> pieces = {3, 3, 4, 4, 5};
    int oracle = -1;
    for (int b = 1; b <= 5 && oracle < 0; ++b) {
        std::vector<int> load(b, 0);
        std::size_t total = 1;
        for (int i = 0; i < 5; ++i) total *= static_cast<std::size_t>(b);
        for (std::size_t code = 0; code < total && oracle < 0; ++code) {
            std::size_t c = code;
            bool ok = true;
            std::fill(load.begin(), load.end(), 0);
            for (int piece : pieces) {
                const int r = static_cast<int>(c % static_cast<std::size_t>(b));
                c /= static_cast<std::size_t>(b);
                load[r] += piece;
                if (load[r] > L) { ok = false; break; }
            }
            if (ok) oracle = b;
        }
    }

    std::printf("rollos óptimos (oráculo) = %d %s\n", oracle,
                s.objective() == static_cast<double>(oracle) ? "(ok)" : "(FAIL)");
    if (s.objective() != static_cast<double>(oracle)) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
