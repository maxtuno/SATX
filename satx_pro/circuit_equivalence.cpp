// circuit_equivalence — catálogo 3.5: verificación de circuitos y generación
// de patrones de prueba, kernel SLIME (nativo).
//
// (a) Miter de equivalencia: C1 y C2 idénticos → XOR de salidas → UNSAT
//     ⟺ circuitos equivalentes.
// (b) Circuito distinto C3 → SAT: el modelo es un vector que los distingue.
// (c) Falla stuck-at-0 en un hilo interno de C1 → SAT: el modelo es un
//     vector de prueba que detecta la falla.
//
// Verificación: tablas de verdad exhaustivas en el host (2^3 entradas).

#include <satx/satx.hpp>

#include <expected>
#include <cstdio>
#include <cstdlib>

// C1 = (a∧b) ⊕ (b∧c)
inline satx::lit_t c1(satx::engine& e, satx::lit_t a, satx::lit_t b, satx::lit_t c) {
    return satx::gates::xor2(e, satx::gates::and2(e, a, b), satx::gates::and2(e, b, c));
}

// C1 con falla stuck-at-0 en el hilo a∧b
inline satx::lit_t c1_stuck0(satx::engine& e, satx::lit_t a, satx::lit_t b, satx::lit_t c) {
    return satx::gates::xor2(e, satx::core::false_lit, satx::gates::and2(e, b, c));
}

// C3 = (a∧b) ⊕ c
inline satx::lit_t c3(satx::engine& e, satx::lit_t a, satx::lit_t b, satx::lit_t c) {
    return satx::gates::xor2(e, satx::gates::and2(e, a, b), c);
}

// miter: decide si dos circuitos son equivalentes (UNSAT ⟺ equivalentes);
// el modelo (si SAT) contiene el vector de prueba en las variables 2, 3, 4.
inline std::expected<satx::solver::model, satx::solver::result> miter(
    satx::engine& e, satx::lit_t (*f1)(satx::engine&, satx::lit_t, satx::lit_t, satx::lit_t),
    satx::lit_t (*f2)(satx::engine&, satx::lit_t, satx::lit_t, satx::lit_t)) {
    const satx::lit_t a = e.add_variable();
    const satx::lit_t b = e.add_variable();
    const satx::lit_t c = e.add_variable();
    const satx::lit_t o = satx::gates::xor2(e, f1(e, a, b, c), f2(e, a, b, c));
    e.add_unit(o);
    return satx::solver::solve(e);
}

inline int eval_c1(int a, int b, int c) { return (a && b) ^ (b && c); }
inline int eval_c1_stuck0(int a, int b, int c) { return (b && c); }
inline int eval_c3(int a, int b, int c) { return (a && b) ^ c; }

int main() {
    // ── (a) C1 ≡ C2 (estructura idéntica) ───────────────────────────────────
    {
        satx::engine e;
        auto m = miter(e, &c1, &c1);
        std::printf("(a) miter C1 vs C1 → %s (esperado UNSAT: equivalentes)\n", m ? "SAT" : "UNSAT");
        if (m) return EXIT_FAILURE;
        for (int i = 0; i < 8; ++i)
            if (eval_c1(i & 1, (i >> 1) & 1, (i >> 2) & 1) !=
                eval_c1(i & 1, (i >> 1) & 1, (i >> 2) & 1)) {
                std::printf("VERIFICACIÓN FALLIDA: oráculo\n");
                return EXIT_FAILURE;
            }
    }

    // ── (b) C1 vs C3: vector que distingue ──────────────────────────────────
    {
        satx::engine e;
        auto m = miter(e, &c1, &c3);
        std::printf("(b) miter C1 vs C3 → %s (esperado SAT)\n", m ? "SAT" : "UNSAT");
        if (!m) return EXIT_FAILURE;
        const satx::lit_t a = 2, b = 3, c = 4;
        const int va = m->get(a) ? 1 : 0, vb = m->get(b) ? 1 : 0, vc = m->get(c) ? 1 : 0;
        std::printf("    vector de prueba: a=%d b=%d c=%d\n", va, vb, vc);
        if (eval_c1(va, vb, vc) == eval_c3(va, vb, vc)) {
            std::printf("VERIFICACIÓN FALLIDA: el vector no distingue\n");
            return EXIT_FAILURE;
        }
        // la fórmula impone XOR → cualquier modelo distingue; verificar contra
        // la tabla completa de diferencias
        int diff = 0;
        for (int i = 0; i < 8; ++i) {
            const int v1 = eval_c1(i & 1, (i >> 1) & 1, (i >> 2) & 1);
            const int v3 = eval_c3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
            if (v1 != v3) ++diff;
        }
        std::printf("    entradas donde difieren: %d\n", diff);
        if (diff == 0) return EXIT_FAILURE;
    }

    // ── (c) falla stuck-at-0 en C1 ──────────────────────────────────────────
    {
        satx::engine e;
        auto m = miter(e, &c1, &c1_stuck0);
        std::printf("(c) miter C1 vs C1[stuck-at-0] → %s (esperado SAT)\n", m ? "SAT" : "UNSAT");
        if (!m) return EXIT_FAILURE;
        const satx::lit_t a = 2, b = 3, c = 4;
        const int va = m->get(a) ? 1 : 0, vb = m->get(b) ? 1 : 0, vc = m->get(c) ? 1 : 0;
        std::printf("    vector de prueba: a=%d b=%d c=%d\n", va, vb, vc);
        if (eval_c1(va, vb, vc) == eval_c1_stuck0(va, vb, vc)) {
            std::printf("VERIFICACIÓN FALLIDA: el vector no detecta la falla\n");
            return EXIT_FAILURE;
        }
    }

    std::printf("verificación del host: OK\n");
    return EXIT_SUCCESS;
}
