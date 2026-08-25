// workforce — catálogo 2.4: rutas de técnicos con habilidades (workforce
// scheduling), kernel SLIME.
//
// Servicios con ventana temporal, duración y habilidad requerida; técnicos
// con habilidades. Restricciones:
//   · cobertura:   cada servicio lo atiende exactamente un técnico;
//   · habilidad:   x_{t,s} = 0 si el técnico no tiene la habilidad;
//   · solapamiento: un técnico no atiende dos servicios con ventanas
//                   solapadas.
//
// Verificación: comprobación independiente en el host.

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

struct service {
    int a, b;   // ventana [a, b]
    int skill;  // 0 = A, 1 = B, 2 = C
};

int main() {
    // técnicos y habilidades: 0 = A, 1 = B, 2 = C
    const int skills[3][3] = {
        {1, 1, 0},  // T0: A, B
        {0, 1, 1},  // T1: B, C
        {1, 0, 1},  // T2: A, C
    };
    const std::vector<service> svc = {
        {0, 2, 0},  // s0: A en [0,2]
        {0, 2, 1},  // s1: B en [0,2]
        {2, 4, 2},  // s2: C en [2,4]
        {3, 5, 0},  // s3: A en [3,5]
        {3, 5, 1},  // s4: B en [3,5]
        {5, 7, 2},  // s5: C en [5,7]
    };
    constexpr int NT = 3, NS = 6;

    satx::engine e;
    std::vector<std::vector<satx::lit_t>> x(NT, std::vector<satx::lit_t>(NS));
    for (int t = 0; t < NT; ++t)
        for (int s = 0; s < NS; ++s) x[t][s] = e.add_variable();

    // cobertura: cada servicio exactamente un técnico
    for (int s = 0; s < NS; ++s) {
        e.add_clause({x[0][s], x[1][s], x[2][s]});
        for (int t1 = 0; t1 < NT; ++t1)
            for (int t2 = t1 + 1; t2 < NT; ++t2)
                e.add_unit(-satx::gates::and2(e, x[t1][s], x[t2][s]));
    }
    // habilidad
    for (int t = 0; t < NT; ++t)
        for (int s = 0; s < NS; ++s)
            if (!skills[t][svc[s].skill]) e.add_unit(-x[t][s]);
    // sin solapamiento por técnico (ventanas que se tocan)
    for (int t = 0; t < NT; ++t)
        for (int s1 = 0; s1 < NS; ++s1)
            for (int s2 = s1 + 1; s2 < NS; ++s2)
                if (svc[s1].a < svc[s2].b && svc[s2].a < svc[s1].b)
                    e.add_unit(-satx::gates::and2(e, x[t][s1], x[t][s2]));

    std::printf("técnicos: %d servicios, %d técnicos (variables %zu, cláusulas %zu)\n",
                NS, NT, e.variable_count(), e.clause_count());

    const auto sol = satx::solver::solve(e);
    if (!sol) {
        std::printf("VERIFICACIÓN FALLIDA: se esperaba SAT\n");
        return EXIT_FAILURE;
    }

    for (int s = 0; s < NS; ++s)
        for (int t = 0; t < NT; ++t)
            if (sol->get(x[t][s]))
                std::printf("  servicio %d (habilidad %d, [%d,%d]) → técnico %d\n", s, svc[s].skill,
                            svc[s].a, svc[s].b, t);

    // ── verificación en el host ─────────────────────────────────────────────
    for (int s = 0; s < NS; ++s) {
        int count = 0, who = -1;
        for (int t = 0; t < NT; ++t)
            if (sol->get(x[t][s])) { ++count; who = t; }
        if (count != 1) {
            std::printf("VERIFICACIÓN FALLIDA: cobertura del servicio %d\n", s);
            return EXIT_FAILURE;
        }
        if (!skills[who][svc[s].skill]) {
            std::printf("VERIFICACIÓN FALLIDA: habilidad del servicio %d\n", s);
            return EXIT_FAILURE;
        }
    }
    for (int t = 0; t < NT; ++t)
        for (int s1 = 0; s1 < NS; ++s1)
            for (int s2 = s1 + 1; s2 < NS; ++s2)
                if (sol->get(x[t][s1]) && sol->get(x[t][s2]) &&
                    svc[s1].a < svc[s2].b && svc[s2].a < svc[s1].b) {
                    std::printf("VERIFICACIÓN FALLIDA: solapamiento del técnico %d\n", t);
                    return EXIT_FAILURE;
                }
    std::printf("verificación del host: OK\n");
    return EXIT_SUCCESS;
}
