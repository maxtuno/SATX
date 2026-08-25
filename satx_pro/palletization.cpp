// palletization — catálogo 3.6: paletización (patrones de apilado), kernel
// SLIME.
//
// Cajas rectangulares sobre una rejilla 4×4 (una capa), sin solapamiento y
// con límite de peso por columna (peso = área). El peso por columna se
// codifica con literales de cobertura (and2) y cláusulas que prohíben todo
// subconjunto de cajas con peso > capacidad.
//
// Verificación: comprobación geométrica y de pesos en el host.

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    constexpr int G = 4;  // rejilla 4×4
    constexpr int CAP = 6;
    // cajas (w, h); peso = w·h
    const std::vector<std::pair<int, int>> boxes = {{2, 2}, {2, 2}, {2, 1}, {1, 2}, {1, 1}, {1, 1}};
    const int n = static_cast<int>(boxes.size());

    using I = satx::integer<3>;  // coords 0..3 (rango -4..3)

    satx::engine e;
    std::vector<I> x, y;
    for (int i = 0; i < n; ++i) {
        x.emplace_back(e);
        y.emplace_back(e);
        const auto& [w, h] = boxes[static_cast<std::size_t>(i)];
        e.add_unit(satx::le_lit(e, I{0}, x[i]));
        e.add_unit(satx::le_lit(e, x[i], I{static_cast<std::int64_t>(G - w)}));
        e.add_unit(satx::le_lit(e, I{0}, y[i]));
        e.add_unit(satx::le_lit(e, y[i], I{static_cast<std::int64_t>(G - h)}));
    }
    // sin solapamiento
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const auto& [wi, hi] = boxes[static_cast<std::size_t>(i)];
            const auto& [wj, hj] = boxes[static_cast<std::size_t>(j)];
            e.add_clause({satx::le_lit(e, x[i] + I{wi}, x[j]),
                          satx::le_lit(e, x[j] + I{wj}, x[i]),
                          satx::le_lit(e, y[i] + I{hi}, y[j]),
                          satx::le_lit(e, y[j] + I{hj}, y[i])});
        }
    // peso por columna ≤ CAP
    for (int gx = 0; gx < G; ++gx) {
        std::vector<satx::lit_t> covers;
        for (int i = 0; i < n; ++i) {
            const auto& [w, h] = boxes[static_cast<std::size_t>(i)];
            const satx::lit_t lo = satx::le_lit(e, x[i], I{gx});
            const satx::lit_t hi = satx::ge_lit(e, x[i] + I{static_cast<std::int64_t>(w - 1)}, I{gx});
            covers.push_back(satx::gates::and2(e, lo, hi));
        }
        // prohibir todo subconjunto con peso > CAP
        for (int mask = 1; mask < (1 << n); ++mask) {
            int wt = 0;
            for (int i = 0; i < n; ++i)
                if (mask & (1 << i)) wt += boxes[static_cast<std::size_t>(i)].first *
                                              boxes[static_cast<std::size_t>(i)].second;
            if (wt <= CAP) continue;
            std::vector<satx::lit_t> c;
            for (int i = 0; i < n; ++i)
                if (mask & (1 << i)) c.push_back(-covers[static_cast<std::size_t>(i)]);
            e.add_clause(c);
        }
    }

    std::printf("paletización: %d cajas en %d×%d (variables %zu, cláusulas %zu)\n", n, G, G,
                e.variable_count(), e.clause_count());

    const auto sol = satx::solver::solve(e);
    if (!sol) {
        std::printf("VERIFICACIÓN FALLIDA: se esperaba SAT\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < n; ++i)
        std::printf("  caja %d (%d×%d, peso %d) en (%d, %d)\n", i, boxes[i].first, boxes[i].second,
                    boxes[i].first * boxes[i].second, static_cast<int>(x[i].value(*sol)),
                    static_cast<int>(y[i].value(*sol)));

    // ── verificación en el host ─────────────────────────────────────────────
    std::vector<int> colw(G, 0);
    for (int i = 0; i < n; ++i) {
        const int xi = static_cast<int>(x[i].value(*sol));
        const int yi = static_cast<int>(y[i].value(*sol));
        const auto& [w, h] = boxes[static_cast<std::size_t>(i)];
        if (xi < 0 || xi + w > G || yi < 0 || yi + h > G) {
            std::printf("VERIFICACIÓN FALLIDA: caja %d fuera de la rejilla\n", i);
            return EXIT_FAILURE;
        }
        for (int gx = xi; gx < xi + w; ++gx) colw[gx] += w * h;
        for (int j = i + 1; j < n; ++j) {
            const int xj = static_cast<int>(x[j].value(*sol));
            const int yj = static_cast<int>(y[j].value(*sol));
            const auto& [wj, hj] = boxes[static_cast<std::size_t>(j)];
            if (xi < xj + wj && xj < xi + w && yi < yj + hj && yj < yi + h) {
                std::printf("VERIFICACIÓN FALLIDA: solapamiento (%d, %d)\n", i, j);
                return EXIT_FAILURE;
            }
        }
    }
    for (int gx = 0; gx < G; ++gx) {
        std::printf("  columna %d: peso %d/%d\n", gx, colw[gx], CAP);
        if (colw[gx] > CAP) {
            std::printf("VERIFICACIÓN FALLIDA: sobrepeso en columna %d\n", gx);
            return EXIT_FAILURE;
        }
    }
    std::printf("verificación del host: OK\n");
    return EXIT_SUCCESS;
}
