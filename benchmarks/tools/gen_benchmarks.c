/*
 * gen_benchmarks.c — generador determinista de instancias DIMACS CNF para
 * medir la performance del kernel SAT de Kerberos (slime.c). Sin dependencias.
 *
 * Genera familias: random 3-SAT, pigeonhole, multiplicadores Tseitin,
 * multiplicación compleja estilo CBE (workload satx), cadenas de sumadores,
 * N-reinas, paridad XOR, coloreo de grafos y Sudoku. Cada instancia incluye
 * variantes SAT y UNSAT con resultado garantizado por construcción.
 *
 * Uso: gen_benchmarks.exe <directorio_salida>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── PRNG determinista (splitmix64) ── */
static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng_next(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void rng_seed(uint64_t s) { rng_state = s; }

/* ── base de cláusulas ── */
typedef struct { int *l; int n; } Clause;
static Clause *db = NULL;
static int db_size = 0, db_cap = 0;
static int nvars = 0;

static int fresh(void) { return ++nvars; }

static void add_raw(const int *lits, int n) {
    Clause c;
    c.n = n;
    c.l = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    for (int i = 0; i < n; ++i) c.l[i] = lits[i];
    if (db_size == db_cap) {
        db_cap = db_cap ? db_cap * 2 : 1024;
        db = (Clause *)realloc(db, (size_t)db_cap * sizeof(Clause));
    }
    db[db_size++] = c;
}

static int icmp(const void *a, const void *b) { return (*(const int *)a) - (*(const int *)b); }

/* normaliza (ordena, deduplica, descarta tautologías) y agrega */
static void add_clause(const int *lits, int n) {
    static int buf[256];
    int *tmp = (n <= 256) ? buf : (int *)malloc((size_t)n * sizeof(int));
    int m = 0;
    for (int i = 0; i < n; ++i) tmp[m++] = lits[i];
    qsort(tmp, (size_t)m, sizeof(int), icmp);
    int out = 0;
    for (int i = 0; i < m; ++i) {
        if (i + 1 < m && tmp[i] == tmp[i + 1]) { /* dup */ continue; }
        if (out > 0 && tmp[out - 1] == -tmp[i]) { /* tautología */ goto done; }
        tmp[out++] = tmp[i];
    }
    if (out > 0) add_raw(tmp, out);
done:
    if (tmp != buf) free(tmp);
}

static void add_unit(int l) { int a[1] = { l }; add_clause(a, 1); }

/* ── gates Tseitin ── */
/* convención de constantes: 0 = false. Las literales ±1..±n son variables
   reales (nunca se pliegan como constantes). */
static int gate_and(int a, int b) {
    if (a == 0 || b == 0) return 0;
    int o = fresh();
    { int c[2] = { -o, a }; add_clause(c, 2); }  /* o → a */
    { int c[2] = { -o, b }; add_clause(c, 2); }  /* o → b */
    { int c[3] = { o, -a, -b }; add_clause(c, 3); }  /* a∧b → o */
    return o;
}

/* o = a XOR b (Tseitin, 4 cláusulas) */
static int gate_xor(int a, int b) {
    if (a == 0) return b;
    if (b == 0) return a;
    int o = fresh();
    { int c[3] = { -o, a, b }; add_clause(c, 3); }
    { int c[3] = { -o, -a, -b }; add_clause(c, 3); }
    { int c[3] = { o, -a, b }; add_clause(c, 3); }
    { int c[3] = { o, a, -b }; add_clause(c, 3); }
    return o;
}

/* o = a OR b (Tseitin, con pliegues correctos sobre el 0) */
static int gate_or(int a, int b) {
    if (a == 0) return b;
    if (b == 0) return a;
    int o = fresh();
    { int c[2] = { o, -a }; add_clause(c, 2); }   /* ¬a → o */
    { int c[2] = { o, -b }; add_clause(c, 2); }   /* ¬b → o */
    { int c[3] = { -o, a, b }; add_clause(c, 3); } /* o → a∨b */
    return o;
}

/* full adder: retorna bit de suma; *carry_out = literal de acarreo */
static int full_add(int a, int b, int ci, int *co) {
    int s1 = gate_xor(a, b);
    int s = gate_xor(s1, ci);
    int m1 = gate_and(a, b);
    int m2 = gate_and(s1, ci);
    *co = gate_or(m1, m2);
    return s;
}

/* sumador ripple de ancho w (LSB primero); se descarta el acarreo final */
static void ripple_add(const int *a, const int *b, int w, int *out) {
    int carry = 0; /* literal de acarreo (0 = sin acarreo) */
    for (int i = 0; i < w; ++i) {
        out[i] = full_add(a[i], b[i], carry, &carry);
    }
}

/* multiplicador Tseitin w_a × w_b (sin signo), resultado 2*w bits (LSB primero) */
static void multiplier(const int *a, int wa, const int *b, int wb, int *out, int wout) {
    int n = wa + wb;
    int **pp = (int **)malloc((size_t)wa * sizeof(int *));
    for (int i = 0; i < wa; ++i) {
        pp[i] = (int *)malloc((size_t)wb * sizeof(int));
        for (int j = 0; j < wb; ++j) pp[i][j] = gate_and(a[i], b[j]);
    }
    int *acc = (int *)calloc((size_t)n, sizeof(int));
    int *row = (int *)calloc((size_t)n, sizeof(int));
    for (int j = 0; j < wb; ++j) {
        for (int i = 0; i < wa; ++i) row[i + j] = pp[i][j];
        int carry = 0;
        for (int k = 0; k < n; ++k) {
            int nv = full_add(acc[k], row[k], carry, &carry);
            acc[k] = nv;
            row[k] = 0;
        }
        /* el acarreo que sale de la posición n-1 afecta solo bits >= n: se descarta */
    }
    for (int i = 0; i < wout; ++i) out[i] = acc[i];
    for (int i = 0; i < wa; ++i) free(pp[i]);
    free(pp); free(acc); free(row);
}

/* pin: variable k-ésima (1..n) al valor val */
static void pin(int var, int val) { add_unit(val ? var : -var); }

/* pin de un literal dado (uso interno) */
static void pin_lit(int lit) { add_unit(lit); }

/* ── utilidades de salida ── */
static void write_cnf(const char *dir, const char *name) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.cnf", dir, name);
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "no se pudo crear %s\n", path); exit(1); }
    fprintf(fp, "p cnf %d %d\n", nvars, db_size);
    for (int i = 0; i < db_size; ++i) {
        for (int j = 0; j < db[i].n; ++j) fprintf(fp, "%d ", db[i].l[j]);
        fputs("0\n", fp);
    }
    fclose(fp);
    printf("  %-28s %6d vars %7d clauses\n", name, nvars, db_size);
}

static void reset_db(void) {
    for (int i = 0; i < db_size; ++i) free(db[i].l);
    db_size = 0;
    nvars = 0;
}

/* ── generadores ── */

static void gen_random3sat(const char *dir, int n, double ratio, uint64_t seed, const char *name) {
    int m = (int)(ratio * (double)n);
    rng_seed(seed);
    reset_db();
    for (int i = 0; i < n; ++i) fresh(); /* crear las n variables */
    for (int i = 0; i < m; ++i) {
        int c[3];
        int v0 = (int)(rng_next() % (uint64_t)n) + 1;
        int v1, v2;
        do { v1 = (int)(rng_next() % (uint64_t)n) + 1; } while (v1 == v0);
        do { v2 = (int)(rng_next() % (uint64_t)n) + 1; } while (v2 == v0 || v2 == v1);
        c[0] = (rng_next() & 1) ? -v0 : v0;
        c[1] = (rng_next() & 1) ? -v1 : v1;
        c[2] = (rng_next() & 1) ? -v2 : v2;
        add_clause(c, 3);
    }
    write_cnf(dir, name);
}

static void gen_php(const char *dir, int n, int sat, const char *name) {
    /* sat=0: n+1 palomas en n agujeros (UNSAT). sat=1: n palomas en n agujeros. */
    int pigeons = sat ? n : n + 1;
    reset_db();
    int **x = (int **)malloc((size_t)pigeons * sizeof(int *));
    for (int i = 0; i < pigeons; ++i) {
        x[i] = (int *)malloc((size_t)n * sizeof(int));
        for (int j = 0; j < n; ++j) x[i][j] = fresh();
    }
    for (int i = 0; i < pigeons; ++i) { /* al menos un agujero */
        int *c = (int *)malloc((size_t)n * sizeof(int));
        for (int j = 0; j < n; ++j) c[j] = x[i][j];
        add_clause(c, n);
        free(c);
    }
    for (int i = 0; i < pigeons; ++i)
        for (int j = i + 1; j < pigeons; ++j)
            for (int k = 0; k < n; ++k) {
                int c[2] = { -x[i][k], -x[j][k] };
                add_clause(c, 2);
            }
    for (int i = 0; i < pigeons; ++i) free(x[i]);
    free(x);
    write_cnf(dir, name);
}

static void gen_mult(const char *dir, int wa, int wb, int bad, const char *name) {
    uint64_t va = 0x5A5A, vb = 0x33CC;
    rng_seed(0xABCDEF01ULL + (uint64_t)wa);
    va = rng_next() & ((1ULL << (wa > 40 ? 40 : wa)) - 1);
    vb = rng_next() & ((1ULL << (wb > 40 ? 40 : wb)) - 1);
    if (va == 0) va = 1;
    if (vb == 0) vb = 1;
    uint64_t prod = va * vb;
    reset_db();
    int *a = (int *)malloc((size_t)wa * sizeof(int));
    int *b = (int *)malloc((size_t)wb * sizeof(int));
    for (int i = 0; i < wa; ++i) { a[i] = fresh(); pin(i + 1, (int)((va >> i) & 1)); }
    int off = nvars;
    for (int i = 0; i < wb; ++i) { b[i] = fresh(); pin(off + i + 1, (int)((vb >> i) & 1)); }
    int wout = wa + wb;
    int *p = (int *)malloc((size_t)wout * sizeof(int));
    multiplier(a, wa, b, wb, p, wout);
    uint64_t expect = bad ? (prod ^ (1ULL << (wa - 1))) : prod;
    for (int i = 0; i < wout; ++i) {
        /* pinar el LITERAL de salida del circuito (p[i] puede ser un literal
           interno temprano, no necesariamente una variable fresca) */
        if (p[i] != 0) add_unit((int)(((expect >> i) & 1) ? p[i] : -p[i]));
    }
    free(a); free(b); free(p);
    write_cnf(dir, name);
}

static void gen_cbemul(const char *dir, int w, int bad, const char *name) {
    /* z = (a+bi)(c+di): real = ac−bd (mod 2^w), imag = ad+bc (mod 2^w) */
    uint64_t va, vb, vc, vd;
    rng_seed(0x55AA55AAULL + (uint64_t)w * 7);
    va = rng_next(); vb = rng_next(); vc = rng_next(); vd = rng_next();
    va &= (w >= 32) ? 0xFFFFULL : ((1ULL << w) - 1);
    vb &= (w >= 32) ? 0xFFFFULL : ((1ULL << w) - 1);
    vc &= (w >= 32) ? 0xFFFFULL : ((1ULL << w) - 1);
    vd &= (w >= 32) ? 0xFFFFULL : ((1ULL << w) - 1);
    uint64_t mw = (w >= 32) ? 0xFFFFFFFFULL : ((1ULL << w) - 1);
    uint64_t real = (va * vc - vb * vd) & mw;
    uint64_t imag = (va * vd + vb * vc) & mw;
    if (bad) real ^= 1;
    reset_db();
    int *a = (int *)malloc((size_t)w * sizeof(int));
    int *b = (int *)malloc((size_t)w * sizeof(int));
    int *c = (int *)malloc((size_t)w * sizeof(int));
    int *d = (int *)malloc((size_t)w * sizeof(int));
    for (int i = 0; i < w; ++i) { a[i] = fresh(); pin(nvars, (int)((va >> i) & 1)); }
    for (int i = 0; i < w; ++i) { b[i] = fresh(); pin(nvars, (int)((vb >> i) & 1)); }
    for (int i = 0; i < w; ++i) { c[i] = fresh(); pin(nvars, (int)((vc >> i) & 1)); }
    for (int i = 0; i < w; ++i) { d[i] = fresh(); pin(nvars, (int)((vd >> i) & 1)); }
    int *ac = (int *)malloc((size_t)(2 * w) * sizeof(int));
    int *bd = (int *)malloc((size_t)(2 * w) * sizeof(int));
    int *ad = (int *)malloc((size_t)(2 * w) * sizeof(int));
    int *bc = (int *)malloc((size_t)(2 * w) * sizeof(int));
    multiplier(a, w, c, w, ac, 2 * w);
    multiplier(b, w, d, w, bd, 2 * w);
    multiplier(a, w, d, w, ad, 2 * w);
    multiplier(b, w, c, w, bc, 2 * w);
    /* full subtractor: s = x − y − borrow; borrow_out = ¬x∧y ∨ ¬x∧borrow ∨ y∧borrow */
    int *re = (int *)malloc((size_t)w * sizeof(int));
    {
        int borrow = 0; /* literal */
        for (int i = 0; i < w; ++i) {
            int x = ac[i], y = bd[i];
            int s = gate_xor(gate_xor(x, y), borrow);
            re[i] = s;
            int b1 = gate_and(-x, y);
            int b2 = gate_and(-x, borrow);
            int b3 = gate_and(y, borrow);
            borrow = gate_or(gate_or(b1, b2), b3);
        }
    }
    /* imag = ad + bc (w bits bajos) */
    int *im = (int *)malloc((size_t)w * sizeof(int));
    {
        int carry = 0;
        for (int i = 0; i < w; ++i) im[i] = full_add(ad[i], bc[i], carry, &carry);
    }
    int off = nvars;
    for (int i = 0; i < w; ++i)
        if (re[i] != 0) add_unit((int)(((real >> i) & 1) ? re[i] : -re[i]));
    for (int i = 0; i < w; ++i)
        if (im[i] != 0) add_unit((int)(((imag >> i) & 1) ? im[i] : -im[i]));
    (void)off;
    free(a); free(b); free(c); free(d);
    free(ac); free(bd); free(ad); free(bc);
    free(re); free(im);
    write_cnf(dir, name);
}

static void gen_addchain(const char *dir, int w, int k, int bad, const char *name) {
    reset_db();
    uint64_t val = 0x12345ULL;
    int *s = (int *)malloc((size_t)w * sizeof(int));
    int *t = (int *)malloc((size_t)w * sizeof(int));
    int *acc = (int *)malloc((size_t)w * sizeof(int));
    for (int i = 0; i < w; ++i) {
        s[i] = fresh();
        pin(nvars, (int)((val >> i) & 1));
    }
    for (int i = 0; i < w; ++i) acc[i] = s[i];
    for (int j = 0; j < k; ++j) {
        uint64_t tj = val + (uint64_t)(j + 1) * 37;
        for (int i = 0; i < w; ++i) {
            t[i] = fresh();
            pin(nvars, (int)((tj >> i) & 1));
        }
        ripple_add(acc, t, w, acc);
    }
    uint64_t sum = val;
    for (int j = 0; j < k; ++j) sum += val + (uint64_t)(j + 1) * 37;
    uint64_t expect = bad ? (sum ^ 1) : sum;
    for (int i = 0; i < w; ++i)
        if (acc[i] != 0) add_unit((int)(((expect >> i) & 1) ? acc[i] : -acc[i]));
    free(s); free(t); free(acc);
    write_cnf(dir, name);
}

static void gen_nqueens(const char *dir, int n, const char *name) {
    reset_db();
    int **x = (int **)malloc((size_t)n * sizeof(int *));
    for (int i = 0; i < n; ++i) {
        x[i] = (int *)malloc((size_t)n * sizeof(int));
        for (int j = 0; j < n; ++j) x[i][j] = fresh();
    }
    for (int i = 0; i < n; ++i) {
        int *c = (int *)malloc((size_t)n * sizeof(int));
        for (int j = 0; j < n; ++j) c[j] = x[i][j];
        add_clause(c, n);
        free(c);
        for (int j = 0; j < n; ++j)
            for (int k2 = j + 1; k2 < n; ++k2) {
                int c2[2] = { -x[i][j], -x[i][k2] };
                add_clause(c2, 2);
            }
    }
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            for (int k2 = i + 1; k2 < n; ++k2) {
                int c[2] = { -x[i][j], -x[k2][j] };
                add_clause(c, 2);
            }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            for (int k2 = 1; i + k2 < n && j + k2 < n; ++k2) {
                int c[2] = { -x[i][j], -x[i + k2][j + k2] };
                add_clause(c, 2);
            }
            for (int k2 = 1; i + k2 < n && j - k2 >= 0; ++k2) {
                int c[2] = { -x[i][j], -x[i + k2][j - k2] };
                add_clause(c, 2);
            }
        }
    for (int i = 0; i < n; ++i) free(x[i]);
    free(x);
    write_cnf(dir, name);
}

static void gen_parity(const char *dir, int n, int k, int m, int unsat, const char *name) {
    /* k restricciones XOR de m variables (con repetición) sobre n variables. */
    reset_db();
    int *vars = (int *)malloc((size_t)n * sizeof(int));
    int *asgn = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; ++i) vars[i] = fresh();
    rng_seed(0x77 + (uint64_t)n);
    int **idx = (int **)malloc((size_t)k * sizeof(int *));
    int *acc_lits = (int *)malloc((size_t)k * sizeof(int));
    for (int c = 0; c < k; ++c) {
        idx[c] = (int *)malloc((size_t)m * sizeof(int));
        for (int i = 0; i < m; ++i) idx[c][i] = (int)(rng_next() % (uint64_t)n);
        int acc = 0;
        for (int i = 0; i < m; ++i) acc = gate_xor(acc, vars[idx[c][i]]);
        acc_lits[c] = acc; /* literal de paridad (0 si se canceló) */
    }
    if (unsat) {
        /* pinar una asignación arbitraria (guardándola) */
        for (int i = 0; i < n; ++i) {
            asgn[i] = (int)(rng_next() & 1);
            pin(i + 1, asgn[i]);
        }
        /* exigir lo CONTRARIO de la paridad real de cada restricción → UNSAT */
        for (int c = 0; c < k; ++c) {
            if (acc_lits[c] == 0) continue;
            int parity = 0;
            for (int i = 0; i < m; ++i) parity ^= asgn[idx[c][i]];
            /* real par → exigir impar (unidad +acc); real impar → exigir par */
            pin_lit(parity == 0 ? acc_lits[c] : -acc_lits[c]);
        }
    } else {
        /* exigir paridad par en toda restricción no vacía */
        for (int c = 0; c < k; ++c)
            if (acc_lits[c] != 0) pin_lit(-acc_lits[c]);
    }
    for (int c = 0; c < k; ++c) free(idx[c]);
    free(idx);
    free(acc_lits);
    free(vars);
    free(asgn);
    write_cnf(dir, name);
}

static void gen_coloring(const char *dir, int n, int k, double p, int clique, const char *name) {
    reset_db();
    int **x = (int **)malloc((size_t)n * sizeof(int *));
    for (int i = 0; i < n; ++i) {
        x[i] = (int *)malloc((size_t)k * sizeof(int));
        for (int j = 0; j < k; ++j) x[i][j] = fresh();
    }
    for (int i = 0; i < n; ++i) {
        int *c = (int *)malloc((size_t)k * sizeof(int));
        for (int j = 0; j < k; ++j) c[j] = x[i][j];
        add_clause(c, k);
        free(c);
        for (int a = 0; a < k; ++a)
            for (int b = a + 1; b < k; ++b) {
                int c2[2] = { -x[i][a], -x[i][b] };
                add_clause(c2, 2);
            }
    }
    rng_seed(0x1234 + (uint64_t)n);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double r = (double)(rng_next() >> 11) / (double)(1ULL << 53);
            if (r < p) {
                for (int col = 0; col < k; ++col) {
                    int c[2] = { -x[i][col], -x[j][col] };
                    add_clause(c, 2);
                }
            }
        }
    if (clique) { /* clique de k+1 nodos → no k-coloreable (UNSAT garantizado) */
        int m = k + 1;
        if (m > n) m = n;
        for (int i = 0; i < m; ++i)
            for (int j = i + 1; j < m; ++j)
                for (int col = 0; col < k; ++col) {
                    int c[2] = { -x[i][col], -x[j][col] };
                    add_clause(c, 2);
                }
    }
    for (int i = 0; i < n; ++i) free(x[i]);
    free(x);
    write_cnf(dir, name);
}

static void gen_sudoku(const char *dir, int unsat, const char *name) {
    /* Sudoku 9×9; solución completa conocida: cell(i,j) = ((i*3 + i/3 + j) % 9) + 1 */
    reset_db();
    for (int v = 0; v < 729; ++v) fresh(); /* 9×9×9 celdas */
    int id(int i, int j, int v) { return ((i * 9 + j) * 9 + v); }
    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 9; ++j) {
            int c[9];
            for (int v = 1; v <= 9; ++v) c[v - 1] = id(i, j, v);
            add_clause(c, 9);
            for (int a = 1; a <= 9; ++a)
                for (int b = a + 1; b <= 9; ++b) {
                    int c2[2] = { -id(i, j, a), -id(i, j, b) };
                    add_clause(c2, 2);
                }
        }
    for (int i = 0; i < 9; ++i)
        for (int v = 1; v <= 9; ++v) {
            int c[9];
            for (int j = 0; j < 9; ++j) c[j] = id(i, j, v);
            add_clause(c, 9);
            for (int a = 0; a < 9; ++a)
                for (int b = a + 1; b < 9; ++b) {
                    int c2[2] = { -id(i, a, v), -id(i, b, v) };
                    add_clause(c2, 2);
                }
        }
    for (int j = 0; j < 9; ++j)
        for (int v = 1; v <= 9; ++v) {
            int c[9];
            for (int i = 0; i < 9; ++i) c[i] = id(i, j, v);
            add_clause(c, 9);
            for (int a = 0; a < 9; ++a)
                for (int b = a + 1; b < 9; ++b) {
                    int c2[2] = { -id(a, j, v), -id(b, j, v) };
                    add_clause(c2, 2);
                }
        }
    for (int bi = 0; bi < 3; ++bi)
        for (int bj = 0; bj < 3; ++bj)
            for (int v = 1; v <= 9; ++v) {
                int c[9];
                int t = 0;
                for (int di = 0; di < 3; ++di)
                    for (int dj = 0; dj < 3; ++dj) c[t++] = id(bi * 3 + di, bj * 3 + dj, v);
                add_clause(c, 9);
                for (int a = 0; a < 9; ++a)
                    for (int b = a + 1; b < 9; ++b) {
                        int c2[2] = { -c[a], -c[b] };
                        add_clause(c2, 2);
                    }
            }
    /* pistas: 30 celdas de la solución */
    int clues[30][2] = {
        {0,0},{0,3},{0,6},{1,1},{1,4},{1,7},{2,2},{2,5},{2,8},
        {3,0},{3,5},{3,7},{4,2},{4,4},{4,6},{5,1},{5,3},{5,8},
        {6,0},{6,4},{6,8},{7,2},{7,5},{7,7},{8,1},{8,3},{8,6},
        {0,1},{2,0},{8,8}
    };
    for (int t = 0; t < 30; ++t) {
        int i = clues[t][0], j = clues[t][1];
        int v = ((i * 3 + i / 3 + j) % 9) + 1;
        pin(id(i, j, v), 1);
    }
    if (unsat) {
        int i = 4, j = 4;
        int v = ((i * 3 + i / 3 + j) % 9) + 1;
        int w = (v % 9) + 1; /* valor distinto en la misma celda → UNSAT */
        pin(id(i, j, w), 1);
    }
    write_cnf(dir, name);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "cnf";
    printf("generando benchmarks en %s/\n", dir);

    gen_random3sat(dir, 150, 3.5, 11, "rand3sat_n150_s1");
    gen_random3sat(dir, 150, 4.26, 21, "rand3sat_n150_u1");
    gen_random3sat(dir, 200, 3.5, 31, "rand3sat_n200_s1");
    gen_random3sat(dir, 200, 4.26, 41, "rand3sat_n200_u1");
    gen_random3sat(dir, 250, 3.5, 51, "rand3sat_n250_s1");
    gen_random3sat(dir, 250, 4.26, 61, "rand3sat_n250_u1");
    gen_random3sat(dir, 300, 3.5, 71, "rand3sat_n300_s1");
    gen_random3sat(dir, 300, 4.26, 81, "rand3sat_n300_u1");
    gen_random3sat(dir, 350, 4.26, 91, "rand3sat_n350_u1");
    gen_random3sat(dir, 400, 4.26, 101, "rand3sat_n400_u1");
    gen_random3sat(dir, 450, 4.26, 111, "rand3sat_n450_u1");
    gen_random3sat(dir, 350, 3.5, 121, "rand3sat_n350_s1");
    gen_random3sat(dir, 400, 3.5, 131, "rand3sat_n400_s1");

    gen_php(dir, 7, 0, "php_n7");
    gen_php(dir, 8, 0, "php_n8");
    gen_php(dir, 9, 0, "php_n9");
    gen_php(dir, 10, 0, "php_n10");
    gen_php(dir, 11, 0, "php_n11");
    gen_php(dir, 12, 0, "php_n12");
    gen_php(dir, 8, 1, "php_sat_n8");
    gen_php(dir, 10, 1, "php_sat_n10");
    gen_php(dir, 12, 1, "php_sat_n12");

    gen_mult(dir, 8, 7, 0, "mult_a8b7");
    gen_mult(dir, 8, 7, 1, "mult_a8b7_bad");
    gen_mult(dir, 12, 11, 0, "mult_a12b11");
    gen_mult(dir, 12, 11, 1, "mult_a12b11_bad");
    gen_mult(dir, 16, 15, 0, "mult_a16b15");
    gen_mult(dir, 16, 15, 1, "mult_a16b15_bad");
    gen_mult(dir, 20, 19, 0, "mult_a20b19");
    gen_mult(dir, 20, 19, 1, "mult_a20b19_bad");
    gen_mult(dir, 24, 23, 0, "mult_a24b23");
    gen_mult(dir, 24, 23, 1, "mult_a24b23_bad");
    gen_mult(dir, 28, 27, 0, "mult_a28b27");
    gen_mult(dir, 28, 27, 1, "mult_a28b27_bad");

    gen_cbemul(dir, 8, 0, "cbemul_w8");
    gen_cbemul(dir, 8, 1, "cbemul_w8_bad");
    gen_cbemul(dir, 10, 0, "cbemul_w10");
    gen_cbemul(dir, 10, 1, "cbemul_w10_bad");
    gen_cbemul(dir, 12, 0, "cbemul_w12");
    gen_cbemul(dir, 12, 1, "cbemul_w12_bad");
    gen_cbemul(dir, 14, 0, "cbemul_w14");
    gen_cbemul(dir, 14, 1, "cbemul_w14_bad");
    gen_cbemul(dir, 16, 0, "cbemul_w16");
    gen_cbemul(dir, 16, 1, "cbemul_w16_bad");

    gen_addchain(dir, 12, 5, 0, "addchain_w12_k5");
    gen_addchain(dir, 12, 5, 1, "addchain_w12_k5_bad");
    gen_addchain(dir, 16, 8, 0, "addchain_w16_k8");
    gen_addchain(dir, 16, 8, 1, "addchain_w16_k8_bad");
    gen_addchain(dir, 24, 12, 0, "addchain_w24_k12");
    gen_addchain(dir, 24, 12, 1, "addchain_w24_k12_bad");

    gen_nqueens(dir, 10, "nqueens_10");
    gen_nqueens(dir, 12, "nqueens_12");
    gen_nqueens(dir, 14, "nqueens_14");
    gen_nqueens(dir, 16, "nqueens_16");
    gen_nqueens(dir, 18, "nqueens_18");
    gen_nqueens(dir, 20, "nqueens_20");

    gen_parity(dir, 40, 12, 16, 0, "parity_sat_n40");
    gen_parity(dir, 40, 12, 16, 1, "parity_unsat_n40");
    gen_parity(dir, 60, 16, 20, 0, "parity_sat_n60");
    gen_parity(dir, 60, 16, 20, 1, "parity_unsat_n60");
    gen_parity(dir, 100, 24, 24, 0, "parity_sat_n100");
    gen_parity(dir, 100, 24, 24, 1, "parity_unsat_n100");

    gen_coloring(dir, 80, 5, 0.10, 0, "coloring_sat_n80");
    gen_coloring(dir, 80, 5, 0.00, 1, "coloring_unsat_n80");
    gen_coloring(dir, 100, 6, 0.08, 0, "coloring_sat_n100");
    gen_coloring(dir, 100, 6, 0.00, 1, "coloring_unsat_n100");
    gen_coloring(dir, 150, 7, 0.05, 0, "coloring_sat_n150");
    gen_coloring(dir, 150, 7, 0.00, 1, "coloring_unsat_n150");

    gen_sudoku(dir, 0, "sudoku_sat");
    gen_sudoku(dir, 1, "sudoku_unsat");

    printf("listo.\n");
    return 0;
}
