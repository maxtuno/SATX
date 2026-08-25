// bin_packing — catálogo 1.3: cubicaje 2D (decisión), kernel SLIME.
//
// Cajas rectangulares con dimensiones enteras sobre una rejilla W×H sin
// solapamiento. Coordenadas (x_i, y_i) discretizadas con la aritmética de
// punto fijo de SATX (CBE). El no solapamiento es una disyunción de
// precedencias por eje:
//
//   (x_i + w_i ≤ x_j) ∨ (x_j + w_j ≤ x_i) ∨ (y_i + h_i ≤ y_j) ∨ (y_j + h_j ≤ y_i)
//
// Se resuelven dos instancias: una factible (todas las cajas caben en
// 10×10) y una infactible (7×7 + 6×6 en 10×10 → UNSAT por ambos ejes).

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    using I = satx::integer<5>;  // coords 0..10 (rango -16..15)

    const auto pack = [](satx::engine& e, int W, int H, std::vector<std::pair<int, int>> const& boxes,
                         std::vector<I>& x, std::vector<I>& y) {
        const int n = static_cast<int>(boxes.size());
        for (int i = 0; i < n; ++i) {
            x.emplace_back(e);
            y.emplace_back(e);
            const auto& [w, h] = boxes[static_cast<std::size_t>(i)];
            // límites del contenedor
            e.add_unit(satx::le_lit(e, I{0}, x[i]));
            e.add_unit(satx::le_lit(e, x[i], I{static_cast<std::int64_t>(W - w)}));
            e.add_unit(satx::le_lit(e, I{0}, y[i]));
            e.add_unit(satx::le_lit(e, y[i], I{static_cast<std::int64_t>(H - h)}));
        }
        // no solapamiento por pares
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                const auto& [wi, hi] = boxes[static_cast<std::size_t>(i)];
                const auto& [wj, hj] = boxes[static_cast<std::size_t>(j)];
                e.add_clause({
                    satx::le_lit(e, x[i] + I{wi}, x[j]),
                    satx::le_lit(e, x[j] + I{wj}, x[i]),
                    satx::le_lit(e, y[i] + I{hi}, y[j]),
                    satx::le_lit(e, y[j] + I{hj}, y[i]),
                });
            }
    };

    const auto verify = [](int W, int H, std::vector<std::pair<int, int>> const& boxes,
                           std::vector<I> const& x, std::vector<I> const& y,
                           satx::solver::model const& m) -> bool {
        const int n = static_cast<int>(boxes.size());
        for (int i = 0; i < n; ++i) {
            const auto& [w, h] = boxes[static_cast<std::size_t>(i)];
            const int xi = static_cast<int>(x[i].value(m));
            const int yi = static_cast<int>(y[i].value(m));
            if (xi < 0 || xi + w > W || yi < 0 || yi + h > H) return false;
            for (int j = i + 1; j < n; ++j) {
                const auto& [wj, hj] = boxes[static_cast<std::size_t>(j)];
                const int xj = static_cast<int>(x[j].value(m));
                const int yj = static_cast<int>(y[j].value(m));
                if (xi < xj + wj && xj < xi + w && yi < yj + hj && yj < yi + h) return false;
            }
        }
        return true;
    };

    // ── (a) instancia factible: 5 cajas en 10×10 ────────────────────────────
    {
        constexpr int W = 10, H = 10;
        const std::vector<std::pair<int, int>> boxes = {{6, 4}, {5, 5}, {4, 4}, {3, 3}, {2, 2}};
        satx::engine e;
        std::vector<I> x, y;
        pack(e, W, H, boxes, x, y);
        std::printf("(a) 5 cajas en %d×%d: variables %zu, cláusulas %zu\n", W, H,
                    e.variable_count(), e.clause_count());
        const auto sol = satx::solver::solve(e);
        if (!sol) {
            std::printf("VERIFICACIÓN FALLIDA: se esperaba SAT\n");
            return EXIT_FAILURE;
        }
        for (std::size_t i = 0; i < boxes.size(); ++i)
            std::printf("  caja %zu (%d×%d) en (%d, %d)\n", i, boxes[i].first, boxes[i].second,
                        static_cast<int>(x[i].value(*sol)), static_cast<int>(y[i].value(*sol)));
        if (!verify(W, H, boxes, x, y, *sol)) {
            std::printf("VERIFICACIÓN FALLIDA: empaque inválido\n");
            return EXIT_FAILURE;
        }
        std::printf("verificación geométrica del host: OK\n");
    }

    // ── (b) instancia infactible: 7×7 + 6×6 en 10×10 → UNSAT ────────────────
    {
        constexpr int W = 10, H = 10;
        const std::vector<std::pair<int, int>> boxes = {{7, 7}, {6, 6}};
        satx::engine e;
        std::vector<I> x, y;
        pack(e, W, H, boxes, x, y);
        const auto sol = satx::solver::solve(e);
        std::printf("(b) 7×7 + 6×6 en %d×%d → %s (esperado UNSAT)\n", W, H,
                    sol ? "SAT" : "UNSAT");
        if (sol) return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
