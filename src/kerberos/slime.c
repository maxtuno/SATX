/*
 * Copyright (c) 2026 Oscar Riveros.
 *
 * Licencia dual: uso personal bajo Apache License 2.0; portes a otros
 * lenguajes requieren licencia comercial con autorizacion expresa del autor.
 * Ver LICENSE.txt en la raiz del proyecto para los terminos completos.
 */

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(SATX_HAVE_THREADS)
#include <stdatomic.h>
#endif

#include "krb_parallel.h"

static int slime_log_enabled;
static int slime_validate;
static int slime_hess_max_iter = 0;
static double hessw_last_reported_unsat = DBL_MAX;

void sat_set_log_enabled(int enabled)     { slime_log_enabled = enabled; }
int sat_log_enabled(void)                 { return slime_log_enabled; }
void sat_set_validate(int enabled)        { slime_validate = enabled; }
void sat_set_hess_max_iter(int max_iter)  { slime_hess_max_iter = max_iter; }

static void hessw_reset_progress_logs(void) {
    hessw_last_reported_unsat = DBL_MAX;
}

static void hessw_log_progress(const char *label, double best, double total_w) {
    const double eps = 1e-12;
    if (!sat_log_enabled()) return;
    if (best + eps >= hessw_last_reported_unsat) return;
    hessw_last_reported_unsat = best;
    double pct = total_w > 0.0 ? 100.0 - 100.0 * best / total_w : 100.0;
    printf("c %s improve: %.12g/%.12g (%.6f%%)\n", label, best, total_w, pct);
    fflush(stdout);
}

static void hessw_eval_model(int n, int m, int *cls_sizes, int **cls_data, double *cls_weights,
                             const unsigned char *model, int *sat_count, double *unsat) {
    double total_unsat = 0.0;
    for (int ci = 0; ci < m; ++ci) {
        int sl = 0;
        for (int k = 0; k < cls_sizes[ci]; ++k) {
            int lit = cls_data[ci][k];
            int v = lit > 0 ? lit - 1 : -lit - 1;
            if (model[v] == (lit > 0 ? 1u : 0u)) sl++;
        }
        sat_count[ci] = sl;
        if (sl == 0) total_unsat += (cls_weights ? cls_weights[ci] : 1.0);
    }
    if (unsat != NULL) *unsat = total_unsat;
    (void)n;
}


typedef int Lit;
typedef struct Clause Clause;
typedef uintptr_t Reason;

typedef struct {
    int *data;
    int size;
    int cap;
} IntVec;

typedef struct {
    Clause **data;
    int size;
    int cap;
} ClauseVec;

typedef struct {
    uintptr_t ref;
} Watch;

typedef struct {
    Watch *data;
    int size;
    int cap;
} WatchVec;

typedef struct {
    uint16_t size;
    Lit *lits;
} CTCube;

typedef struct {
    CTCube *data;
    int size;
    int cap;
    int max_keep;
} CTCubeVec;

typedef struct {
    FILE *fp;
    unsigned char *buf;
    size_t cap;
    size_t pos;
    size_t len;
    int pushed;
} FastInput;

struct Clause {
    int size;
    uint32_t lbd;
    float activity;
    unsigned learnt : 1;
    unsigned deleted : 1;
    unsigned locked : 1;
    uint32_t magic; /* CLAUSE_MAGIC: integridad (doble-free/corrupción detectada) */
    Lit lits[];
};

/* Número mágico de integridad de cláusula: convierte doble-free / corrupción
   de memoria en un error detectado y ruidoso en lugar de un crash silencioso. */
#define CLAUSE_MAGIC 0x534C314Du  /* "SL1M" */

typedef struct {
    Clause *clause;
    Lit lits[2];
    int size;
} ClauseRef;

/* Registro de extensión: cláusulas que contenían una variable eliminada,
   para reconstruir su valor en el modelo (extensión de modelo).
   eq_lit != 0 → registro de sustitución equivalente (var ≡ eq_lit). */
typedef struct {
    int var;          /* variable eliminada (0-based) */
    int eq_lit;       /* literal DIMACS equivalente (0 si es un registro BVE) */
    Clause **cls;     /* copias de las cláusulas que la contenían */
    int ncls;
} ExtRec;

typedef struct {
    int *heap;
    int *pos;
    int size;
    int cap;
    double *activity;
} VarHeap;

typedef struct {
    int nvars;
    int ok;

    signed char *assigns;   // 0=unassigned, +1=true, -1=false
    int *levels;
    Reason *reasons;
    unsigned char *phases;  // preferred polarity: 1=true, 0=false
    unsigned char *seen;

    double *activity;
    double *chb_activity;
    long long *chb_last_conflict;
    double var_inc;
    double var_decay;
    double chb_step;
    double chb_step_dec;
    double chb_step_min;

    int heuristic;      // 0=VSIDS, 1=CHB
    int use_mab;
    double mabc;
    double mab_reward[2];
    unsigned mab_select[2];
    double mab_epoch_decisions;
    double mab_epoch_conflicts;

    double cla_inc;
    double cla_decay;

    VarHeap order;

    WatchVec *watches;      // indexed by literal id [0,2*nvars)

    Lit *trail;
    int trail_size;
    int trail_cap;
    int qhead;

    IntVec trail_lim;
    IntVec analyze_stack;
    IntVec lbd_marks;
    int lbd_stamp;

    ClauseVec clauses;
    ClauseVec learnts;
    long long binary_clauses;

    long long conflicts;
    long long decisions;
    long long propagations;
    long long restarts;

    int restart_count;
    long long next_restart;

    double fast_glue_ema;
    double slow_glue_ema;
    double fast_glue_alpha;
    double slow_glue_alpha;
    double fast_glue_beta;
    double slow_glue_beta;
    double restart_margin;
    long long next_ema_restart;
    long long ema_restart_interval;

    int reduce_base;
    long long next_reduce;

    CTCubeVec ct_cubes;
    int ct_enable;
    int ct_lbd_max;
    int ct_maxlen;
    int ct_buddy_merge;
    int ct_escape_rounds;
    int ct_probe_restarts;
    int hess_enable;
    uint64_t rng_state;

    IntVec orig_unit_lits;
    unsigned char *hess_unit_freeze;
    int orig_empty_clauses;

    long long deleted_freed;
    long long watch_sweeps;
    long long ct_added;
    long long ct_merged;
    long long ct_escaped;
    long long ct_probe_added;
    long long hess_calls;
    long long hess_sat_hits;

    int assumption_level;
    FILE *proof;
#if defined(SATX_HAVE_THREADS)
    const atomic_int *external_stop;
#endif
    int external_stop_hit;

    /* ── simplificación / inprocessing / mejoras CDCL ── */
    IntVec *occ;             /* listas de ocurrencias por literal (transitorias) */
    int simplified;          /* simplificación raíz ya realizada */
    int simplify_enable;
    int probe_enable;
    int inprocess_enable;
    int bve_enable;
    int chrono_enable;
    int rephase_enable;
    long long next_inprocess;
    int inprocess_count;
    long long simp_eliminated;
    long long simp_subsumed;
    long long simp_strengthened;
    long long simp_pure;
    long long simp_equivs;
    long long simp_probe_units;
    ExtRec *extend;          /* registros de extensión para el modelo */
    int extend_size;
    int extend_cap;
    IntVec eq_pairs;         /* pares (var, lit) de sustitución equivalente */
    int *lit_mark;           /* marcas por literal (stamps) para simplify */
    int lit_mark_stamp;
    unsigned char *mini_removable;   /* minimización recursiva de cláusulas */
    unsigned char *mini_poisoned;
    IntVec mini_removed_stack;
    IntVec mini_poison_stack;
    Clause *last_learnt[4];  /* anillo para subsumción eager de cláusulas aprendidas */
    long long next_rephase;
    int rephase_count;
    int chrono_conflicts;
    IntVec simp_buf1;
    IntVec simp_buf2;
} Solver;

typedef struct {
    int heuristic_mode;      /* 0=VSIDS, 1=CHB */
    int use_mab;
    double mabc;
    int use_hess;
    int use_ct;
    int ct_lbd_max;
    int ct_maxlen;
    int ct_max_cubes;
    int ct_buddy_merge;
    int ct_escape_rounds;
    int ct_probe_restarts;
    int use_simplify;        /* simplificación raíz + inprocessing */
    int use_bve;             /* eliminación acotada de variables (experimental) */
    int use_chrono;          /* backtracking cronológico */
    int use_inprocess;       /* simplificación periódica durante la búsqueda */
    int use_probe;           /* sondeo de literales fallidos (bounded) */
} SlimeSatOptions;

typedef struct {
    long long clauses;
    long long learnt;
    long long conflicts;
    long long decisions;
    long long propagations;
    long long restarts;
    long long hess_calls;
    long long hess_sat_hits;
    long long ct_added;
    long long ct_merged;
    long long ct_escaped;
    long long ct_probe_added;
} SlimeSatStats;

typedef struct SlimeSatHandle SlimeSatHandle;

static void slime_sat_options_default(SlimeSatOptions *cfg);
static void slime_sat_options_normalize(SlimeSatOptions *dst, const SlimeSatOptions *src);
static int slime_solver_load_clauses(Solver *s, int nclauses, const int *const *clauses, const int *sizes, int *base_unsat);

static void die(const char *msg) {
    fprintf(stderr, "c ERROR: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1u);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1u);
    if (!q) die("out of memory");
    return q;
}

static int parse_ll(const char *text, long long *out) {
    char *end = NULL;
    long long value;
    errno = 0;
    value = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }
    *out = value;
    return 1;
}

static int solver_poll_external_stop(Solver *s) {
#if defined(SATX_HAVE_THREADS)
    if (s != NULL && s->external_stop != NULL && atomic_load(s->external_stop) != 0) {
        s->external_stop_hit = 1;
        return 1;
    }
#else
    (void)s;
#endif
    return 0;
}

static void intvec_init(IntVec *v) {
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static void intvec_reserve(IntVec *v, int need) {
    if (need <= v->cap) return;
    int nc = v->cap ? v->cap : 4;
    while (nc < need) nc <<= 1;
    v->data = (int *)xrealloc(v->data, (size_t)nc * sizeof(int));
    v->cap = nc;
}

static void intvec_push(IntVec *v, int x) {
    if (v->size == v->cap) {
        int nc = v->cap ? (v->cap << 1) : 4;
        v->data = (int *)xrealloc(v->data, (size_t)nc * sizeof(int));
        v->cap = nc;
    }
    v->data[v->size++] = x;
}

static void intvec_free(IntVec *v) {
    free(v->data);
    v->data = NULL;
    v->size = v->cap = 0;
}

static void intvec_clear(IntVec *v) {
    v->size = 0;
}

static void clausevec_init(ClauseVec *v) {
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static void clausevec_push(ClauseVec *v, Clause *c) {
    if (v->size == v->cap) {
        int nc = v->cap ? (v->cap << 1) : 16;
        v->data = (Clause **)xrealloc(v->data, (size_t)nc * sizeof(Clause *));
        v->cap = nc;
    }
    v->data[v->size++] = c;
}

static void clausevec_free(ClauseVec *v) {
    free(v->data);
    v->data = NULL;
    v->size = v->cap = 0;
}

static void watchvec_init(WatchVec *v) {
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static void watchvec_push(WatchVec *v, Watch w) {
    if (v->size == v->cap) {
        int nc = v->cap ? (v->cap << 1) : 4;
        v->data = (Watch *)xrealloc(v->data, (size_t)nc * sizeof(Watch));
        v->cap = nc;
    }
    v->data[v->size++] = w;
}

static void watchvec_free(WatchVec *v) {
    free(v->data);
    v->data = NULL;
    v->size = v->cap = 0;
}

static void ctcubevec_init(CTCubeVec *v, int max_keep) {
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
    v->max_keep = (max_keep > 0) ? max_keep : 1;
}

static void ctcubevec_drop_oldest(CTCubeVec *v, int count) {
    if (count <= 0 || v->size == 0) return;
    if (count > v->size) count = v->size;
    for (int i = 0; i < count; ++i) {
        free(v->data[i].lits);
        v->data[i].lits = NULL;
        v->data[i].size = 0;
    }
    if (count < v->size) {
        memmove(v->data,
                v->data + count,
                (size_t)(v->size - count) * sizeof(CTCube));
    }
    v->size -= count;
}

static void ctcubevec_remove_at(CTCubeVec *v, int idx) {
    if (idx < 0 || idx >= v->size) return;
    free(v->data[idx].lits);
    if (idx + 1 < v->size) {
        memmove(v->data + idx,
                v->data + idx + 1,
                (size_t)(v->size - idx - 1) * sizeof(CTCube));
    }
    v->size--;
}

static void ctcubevec_push(CTCubeVec *v, const Lit *lits, int size) {
    if (size <= 0) return;
    if (v->size >= v->max_keep) {
        int drop = v->max_keep / 8;
        if (drop < 1) drop = 1;
        ctcubevec_drop_oldest(v, drop);
    }
    if (v->size == v->cap) {
        int nc = v->cap ? (v->cap << 1) : 64;
        v->data = (CTCube *)xrealloc(v->data, (size_t)nc * sizeof(CTCube));
        v->cap = nc;
    }
    Lit *copy = (Lit *)xmalloc((size_t)size * sizeof(Lit));
    memcpy(copy, lits, (size_t)size * sizeof(Lit));
    v->data[v->size].size = (uint16_t)size;
    v->data[v->size].lits = copy;
    v->size++;
}

static void ctcubevec_free(CTCubeVec *v) {
    for (int i = 0; i < v->size; ++i) {
        free(v->data[i].lits);
        v->data[i].lits = NULL;
        v->data[i].size = 0;
    }
    free(v->data);
    v->data = NULL;
    v->size = v->cap = 0;
    v->max_keep = 0;
}

static inline uint64_t rng_next_u64(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline uint32_t solver_rand_u32(Solver *s) {
    return (uint32_t)(rng_next_u64(&s->rng_state) >> 32);
}

static inline double solver_rand_unit(Solver *s) {
    return (double)solver_rand_u32(s) / 4294967296.0;
}

static inline Lit mk_lit(int var, int sign) {
    return (var << 1) | (sign & 1);
}

static inline int lit_var(Lit p) {
    return p >> 1;
}

static inline int lit_sign(Lit p) {
    return p & 1;
}

static inline Lit lit_neg(Lit p) {
    return p ^ 1;
}

static inline int dimacs_to_lit(int x) {
    int v = x > 0 ? x : -x;
    return mk_lit(v - 1, x < 0);
}

static inline int lit_to_dimacs(Lit p) {
    int v = lit_var(p) + 1;
    return lit_sign(p) ? -v : v;
}

static inline int lit_asgn_value(Lit p) {
    return lit_sign(p) ? -1 : 1;
}

static inline Reason reason_none(void) {
    return (Reason)0;
}

static inline int reason_is_none(Reason r) {
    return r == (Reason)0;
}

static inline int reason_is_binary(Reason r) {
    return (r & (Reason)1u) != 0;
}

static inline Reason reason_from_clause(Clause *c) {
    uintptr_t raw = (uintptr_t)c;
    assert((raw & (uintptr_t)1u) == 0u);
    return (Reason)raw;
}

static inline Clause *reason_clause_ptr(Reason r) {
    return (Clause *)(uintptr_t)r;
}

static inline Reason reason_from_binary(Lit other) {
    return (Reason)((((uintptr_t)(unsigned int)(other + 1)) << 1) | (uintptr_t)1u);
}

static inline Lit reason_binary_other(Reason r) {
    return (Lit)((int)((r >> 1) - 1u));
}

static inline Watch watch_from_clause(Clause *c) {
    Watch w;
    uintptr_t raw = (uintptr_t)c;
    assert((raw & (uintptr_t)1u) == 0u);
    w.ref = raw;
    return w;
}

static inline Watch watch_from_binary(Lit implied) {
    Watch w;
    w.ref = ((((uintptr_t)(unsigned int)(implied + 1)) << 1) | (uintptr_t)1u);
    return w;
}

static inline int watch_is_binary(Watch w) {
    return (w.ref & (uintptr_t)1u) != 0u;
}

static inline Clause *watch_clause_ptr(Watch w) {
    return (Clause *)(uintptr_t)w.ref;
}

static inline Lit watch_binary_lit(Watch w) {
    return (Lit)((int)((w.ref >> 1) - 1u));
}

static inline ClauseRef clauseref_none(void) {
    ClauseRef r;
    r.clause = NULL;
    r.lits[0] = 0;
    r.lits[1] = 0;
    r.size = 0;
    return r;
}

static inline int clauseref_is_none(ClauseRef r) {
    return r.size == 0;
}

static inline ClauseRef clauseref_from_clause(Clause *c) {
    ClauseRef r;
    r.clause = c;
    r.lits[0] = 0;
    r.lits[1] = 0;
    r.size = c ? c->size : 0;
    return r;
}

static inline ClauseRef clauseref_from_binary(Lit a, Lit b) {
    ClauseRef r;
    r.clause = NULL;
    r.lits[0] = a;
    r.lits[1] = b;
    r.size = 2;
    return r;
}

static inline ClauseRef clauseref_from_reason(Reason r, Lit implied) {
    if (reason_is_none(r)) return clauseref_none();
    if (reason_is_binary(r)) return clauseref_from_binary(implied, reason_binary_other(r));
    return clauseref_from_clause(reason_clause_ptr(r));
}

static inline int clauseref_is_learnt(const ClauseRef *r) {
    return r->clause != NULL && r->clause->learnt;
}

static inline Lit clauseref_lit(const ClauseRef *r, int idx) {
    return r->clause ? r->clause->lits[idx] : r->lits[idx];
}

static inline int solver_decision_level(const Solver *s) {
    return s->trail_lim.size;
}

static inline int solver_lit_value(const Solver *s, Lit p) {
    int v = s->assigns[lit_var(p)];
    if (v == 0) return 0;
    return lit_sign(p) ? -v : v;
}

static void heap_init(VarHeap *h, int n, double *activity) {
    h->heap = (int *)xmalloc((size_t)n * sizeof(int));
    h->pos = (int *)xmalloc((size_t)n * sizeof(int));
    h->size = 0;
    h->cap = n;
    h->activity = activity;
    for (int i = 0; i < n; ++i) h->pos[i] = -1;
}

static void heap_free(VarHeap *h) {
    free(h->heap);
    free(h->pos);
    h->heap = NULL;
    h->pos = NULL;
    h->size = h->cap = 0;
    h->activity = NULL;
}

static inline int heap_better(const VarHeap *h, int a, int b) {
    if (h->activity[a] > h->activity[b]) return 1;
    if (h->activity[a] < h->activity[b]) return 0;
    return a < b;
}

static void heap_swap(VarHeap *h, int i, int j) {
    int a = h->heap[i];
    int b = h->heap[j];
    h->heap[i] = b;
    h->heap[j] = a;
    h->pos[a] = j;
    h->pos[b] = i;
}

static void heap_sift_up(VarHeap *h, int i) {
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (!heap_better(h, h->heap[i], h->heap[p])) break;
        heap_swap(h, i, p);
        i = p;
    }
}

static void heap_sift_down(VarHeap *h, int i) {
    for (;;) {
        int l = (i << 1) + 1;
        int r = l + 1;
        int best = i;
        if (l < h->size && heap_better(h, h->heap[l], h->heap[best])) best = l;
        if (r < h->size && heap_better(h, h->heap[r], h->heap[best])) best = r;
        if (best == i) break;
        heap_swap(h, i, best);
        i = best;
    }
}

static void heap_insert(VarHeap *h, int v) {
    if (h->pos[v] >= 0) return;
    int i = h->size++;
    h->heap[i] = v;
    h->pos[v] = i;
    heap_sift_up(h, i);
}

static void heap_remove(VarHeap *h, int v) {
    int i = h->pos[v];
    if (i < 0) return;
    int last = --h->size;
    h->pos[v] = -1;
    if (i == last) return;
    int u = h->heap[last];
    h->heap[i] = u;
    h->pos[u] = i;
    heap_sift_down(h, i);
    heap_sift_up(h, i);
}

static int heap_pop_max(VarHeap *h) {
    if (h->size == 0) return -1;
    int v = h->heap[0];
    int last = --h->size;
    h->pos[v] = -1;
    if (last > 0) {
        int u = h->heap[last];
        h->heap[0] = u;
        h->pos[u] = 0;
        heap_sift_down(h, 0);
    }
    return v;
}

static void heap_increase(VarHeap *h, int v) {
    int i = h->pos[v];
    if (i >= 0) heap_sift_up(h, i);
}

static void heap_update(VarHeap *h, int v) {
    int i = h->pos[v];
    if (i < 0) return;
    heap_sift_up(h, i);
    heap_sift_down(h, i);
}

static void heap_rebuild(VarHeap *h) {
    if (h->size <= 1) return;
    for (int i = (h->size >> 1) - 1; i >= 0; --i) heap_sift_down(h, i);
}

static Clause *clause_new(const Lit *lits, int size, int learnt, uint32_t lbd) {
    if (size < 0) die("negative clause size");
    size_t bytes = sizeof(Clause) + (size_t)size * sizeof(Lit);
    Clause *c = (Clause *)xmalloc(bytes);
    c->size = size;
    c->lbd = lbd;
    c->activity = 0.0f;
    c->learnt = (unsigned)!!learnt;
    c->deleted = 0;
    c->locked = 0;
    c->magic = CLAUSE_MAGIC;
    if (size > 0) memcpy(c->lits, lits, (size_t)size * sizeof(Lit));
    return c;
}

/* Liberación con verificación de integridad: un doble free o una corrupción
   de memoria se reportan como error detectado en lugar de crash silencioso. */
static void clause_free(Clause *c) {
    if (c->magic != CLAUSE_MAGIC) {
        fprintf(stderr,
                "KERBEROS: integridad de cláusula violada (doble free o "
                "corrupción de memoria)\n");
        abort();
    }
    c->magic = 0;
    free(c);
}

static int solver_add_clause(Solver *s, const Lit *lits, int size, int learnt, uint32_t lbd);
static ClauseRef solver_propagate(Solver *s);

static void solver_attach_clause(Solver *s, Clause *c) {
    if (c->size < 2) return;
    watchvec_push(&s->watches[lit_neg(c->lits[0])], watch_from_clause(c));
    watchvec_push(&s->watches[lit_neg(c->lits[1])], watch_from_clause(c));
}

static void solver_init(Solver *s, int nvars, int need_chb, int need_hess) {
    memset(s, 0, sizeof(*s));
    s->nvars = nvars;
    s->ok = 1;

    s->assigns = (signed char *)xmalloc((size_t)nvars * sizeof(signed char));
    s->levels = (int *)xmalloc((size_t)nvars * sizeof(int));
    s->reasons = (Reason *)xmalloc((size_t)nvars * sizeof(Reason));
    s->phases = (unsigned char *)xmalloc((size_t)nvars * sizeof(unsigned char));
    s->seen = (unsigned char *)xmalloc((size_t)nvars * sizeof(unsigned char));
    s->activity = (double *)xmalloc((size_t)nvars * sizeof(double));
    s->chb_activity = need_chb ? (double *)xmalloc((size_t)nvars * sizeof(double)) : NULL;
    s->chb_last_conflict = need_chb ? (long long *)xmalloc((size_t)nvars * sizeof(long long)) : NULL;

    memset(s->assigns, 0, (size_t)nvars * sizeof(signed char));
    memset(s->levels, 0, (size_t)nvars * sizeof(int));
    for (int i = 0; i < nvars; ++i) s->reasons[i] = reason_none();
    memset(s->phases, 1, (size_t)nvars * sizeof(unsigned char));
    memset(s->seen, 0, (size_t)nvars * sizeof(unsigned char));
    memset(s->activity, 0, (size_t)nvars * sizeof(double));
    if (s->chb_activity != NULL) {
        memset(s->chb_activity, 0, (size_t)nvars * sizeof(double));
    }
    if (s->chb_last_conflict != NULL) {
        memset(s->chb_last_conflict, 0, (size_t)nvars * sizeof(long long));
    }

    heap_init(&s->order, nvars, s->activity);
    for (int v = 0; v < nvars; ++v) heap_insert(&s->order, v);

    s->watches = (WatchVec *)xmalloc((size_t)(2 * nvars) * sizeof(WatchVec));
    for (int i = 0; i < 2 * nvars; ++i) watchvec_init(&s->watches[i]);

    s->trail = NULL;
    s->trail_size = 0;
    s->trail_cap = 0;
    s->qhead = 0;

    intvec_init(&s->trail_lim);
    intvec_init(&s->analyze_stack);
    intvec_init(&s->lbd_marks);
    intvec_reserve(&s->lbd_marks, nvars + 1);
    s->lbd_marks.size = nvars + 1;
    memset(s->lbd_marks.data, 0, (size_t)(nvars + 1) * sizeof(int));
    s->lbd_stamp = 1;

    clausevec_init(&s->clauses);
    clausevec_init(&s->learnts);

    s->var_inc = 1.0;
    s->var_decay = 0.95;
    s->chb_step = 0.4;
    s->chb_step_dec = 0.000001;
    s->chb_step_min = 0.06;
    s->heuristic = 0;
    s->use_mab = 0;
    s->mabc = 4.0;
    s->mab_reward[0] = s->mab_reward[1] = 0.0;
    s->mab_select[0] = 1;
    s->mab_select[1] = 0;
    s->mab_epoch_decisions = 0.0;
    s->mab_epoch_conflicts = 0.0;
    s->cla_inc = 1.0;
    s->cla_decay = 0.999;

    s->restart_count = 1;
    s->next_restart = 50;

    s->fast_glue_ema = 0.0;
    s->slow_glue_ema = 0.0;
    s->fast_glue_alpha = 1.0 / 33.0;
    s->slow_glue_alpha = 1.0 / 5000.0;
    s->fast_glue_beta = 1.0 - s->fast_glue_alpha;
    s->slow_glue_beta = 1.0 - s->slow_glue_alpha;
    s->restart_margin = 1.2;
    s->next_ema_restart = 2000;
    s->ema_restart_interval = 500;

    s->reduce_base = 2000;
    s->next_reduce = s->reduce_base;

    ctcubevec_init(&s->ct_cubes, 40000);
    s->ct_enable = 1;
    s->ct_lbd_max = 6;
    s->ct_maxlen = 12;
    s->ct_buddy_merge = 0;
    s->ct_escape_rounds = 4;
    s->ct_probe_restarts = 4;

    s->hess_enable = 0;
    s->rng_state = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)nvars * 0xbf58476d1ce4e5b9ULL);
    intvec_init(&s->orig_unit_lits);
    s->hess_unit_freeze = need_hess ? (unsigned char *)xmalloc((size_t)nvars * sizeof(unsigned char)) : NULL;
    if (s->hess_unit_freeze != NULL) {
        memset(s->hess_unit_freeze, 0, (size_t)nvars * sizeof(unsigned char));
    }
    s->orig_empty_clauses = 0;

    s->proof = NULL;
#if defined(SATX_HAVE_THREADS)
    s->external_stop = NULL;
#endif
    s->external_stop_hit = 0;

    /* ── simplificación / inprocessing / mejoras CDCL ── */
    s->occ = NULL;
    s->simplified = 0;
    s->simplify_enable = 1;
    s->bve_enable = 0;
    s->probe_enable = 1;
    s->inprocess_enable = 0;
    s->chrono_enable = 1;
    s->rephase_enable = 1;
    s->next_inprocess = 0;
    s->inprocess_count = 0;
    s->simp_eliminated = 0;
    s->simp_subsumed = 0;
    s->simp_strengthened = 0;
    s->simp_pure = 0;
    s->simp_equivs = 0;
    s->simp_probe_units = 0;
    s->extend = NULL;
    s->extend_size = 0;
    s->extend_cap = 0;
    intvec_init(&s->eq_pairs);
    s->lit_mark = (int *)xmalloc((size_t)(2 * nvars) * sizeof(int));
    memset(s->lit_mark, 0, (size_t)(2 * nvars) * sizeof(int));
    s->lit_mark_stamp = 1;
    s->mini_removable = (unsigned char *)xmalloc((size_t)nvars * sizeof(unsigned char));
    memset(s->mini_removable, 0, (size_t)nvars * sizeof(unsigned char));
    intvec_init(&s->mini_poison_stack);
    for (int i = 0; i < 4; ++i) s->last_learnt[i] = NULL;
    s->next_rephase = 1000;
    s->rephase_count = 0;
    s->chrono_conflicts = 0;
    intvec_init(&s->simp_buf1);
    intvec_init(&s->simp_buf2);
}

static void solver_destroy(Solver *s) {
    for (int i = 0; i < s->clauses.size; ++i) free(s->clauses.data[i]);
    for (int i = 0; i < s->learnts.size; ++i) free(s->learnts.data[i]);

    clausevec_free(&s->clauses);
    clausevec_free(&s->learnts);

    for (int i = 0; i < 2 * s->nvars; ++i) watchvec_free(&s->watches[i]);
    free(s->watches);
    s->watches = NULL;

    heap_free(&s->order);

    free(s->assigns);
    free(s->levels);
    free(s->reasons);
    free(s->phases);
    free(s->seen);
    free(s->activity);
    free(s->chb_activity);
    free(s->chb_last_conflict);
    free(s->trail);

    s->assigns = NULL;
    s->levels = NULL;
    s->reasons = NULL;
    s->phases = NULL;
    s->seen = NULL;
    s->activity = NULL;
    s->chb_activity = NULL;
    s->chb_last_conflict = NULL;
    s->trail = NULL;

    ctcubevec_free(&s->ct_cubes);
    intvec_free(&s->orig_unit_lits);
    free(s->hess_unit_freeze);
    s->hess_unit_freeze = NULL;
    s->orig_empty_clauses = 0;

    intvec_free(&s->trail_lim);
    intvec_free(&s->analyze_stack);
    intvec_free(&s->lbd_marks);

    if (s->occ != NULL) {
        for (int i = 0; i < 2 * s->nvars; ++i) intvec_free(&s->occ[i]);
        free(s->occ);
        s->occ = NULL;
    }
    for (int i = 0; i < s->extend_size; ++i) {
        ExtRec *r = &s->extend[i];
        for (int j = 0; j < r->ncls; ++j) free(r->cls[j]);
        free(r->cls);
    }
    free(s->extend);
    s->extend = NULL;
    s->extend_size = s->extend_cap = 0;
    intvec_free(&s->eq_pairs);
    free(s->lit_mark);
    s->lit_mark = NULL;
    free(s->mini_removable);
    s->mini_removable = NULL;
    intvec_free(&s->mini_poison_stack);
    intvec_free(&s->simp_buf1);
    intvec_free(&s->simp_buf2);

    if (s->proof) {
        fclose(s->proof);
        s->proof = NULL;
    }
}

static void solver_trail_push(Solver *s, Lit p) {
    if (s->trail_size == s->trail_cap) {
        int nc = s->trail_cap ? (s->trail_cap << 1) : 1024;
        s->trail = (Lit *)xrealloc(s->trail, (size_t)nc * sizeof(Lit));
        s->trail_cap = nc;
    }
    s->trail[s->trail_size++] = p;
}

static int solver_enqueue(Solver *s, Lit p, Reason reason) {
    int v = lit_var(p);
    int val = lit_asgn_value(p);
    int cur = s->assigns[v];
    if (cur != 0) return cur == val;

    s->assigns[v] = (signed char)val;
    s->levels[v] = solver_decision_level(s);
    s->reasons[v] = reason;
    s->phases[v] = (unsigned char)(val > 0);
    heap_remove(&s->order, v);
    solver_trail_push(s, p);
    return 1;
}

static int solver_commit_input_clause(Solver *s, IntVec *clause, int *contradiction) {
    int out = 0;

    *contradiction = 0;
    for (int i = 0; i < clause->size; ++i) {
        Lit p = (Lit)clause->data[i];
        int val = solver_lit_value(s, p);
        if (val == 1) {
            clause->size = 0;
            return 1;
        }
        if (val == -1) continue;

        int dup = 0;
        for (int j = 0; j < out; ++j) {
            Lit q = (Lit)clause->data[j];
            if (q == p) {
                dup = 1;
                break;
            }
            if (q == lit_neg(p)) {
                clause->size = 0;
                return 1;
            }
        }
        if (!dup) clause->data[out++] = (int)p;
    }

    clause->size = out;
    if (out == 0) {
        s->ok = 0;
        *contradiction = 1;
        return 1;
    }

    if (!solver_add_clause(s, (Lit *)clause->data, out, 0, 0)) {
        if (!s->ok) {
            *contradiction = 1;
            return 1;
        }
        return 0;
    }

    if (out == 1 && s->ok) {
        ClauseRef conflict = solver_propagate(s);
        if (!clauseref_is_none(conflict)) {
            s->ok = 0;
            *contradiction = 1;
        }
    }
    return 1;
}

static ClauseRef solver_propagate(Solver *s) {
    while (s->qhead < s->trail_size) {
        if (solver_poll_external_stop(s)) {
            return clauseref_none();
        }
        Lit p = s->trail[s->qhead++];
        s->propagations++;

        WatchVec *ws = &s->watches[p];
        int i = 0;
        int j = 0;
        while (i < ws->size) {
            Watch w = ws->data[i++];
            if (watch_is_binary(w)) {
                Lit q = watch_binary_lit(w);
                int qv = solver_lit_value(s, q);
                if (qv == 1) {
                    ws->data[j++] = w;
                    continue;
                }
                ws->data[j++] = w;
                if (qv == -1) {
                    while (i < ws->size) ws->data[j++] = ws->data[i++];
                    ws->size = j;
                    return clauseref_from_binary(lit_neg(p), q);
                }
                if (!solver_enqueue(s, q, reason_from_binary(lit_neg(p)))) {
                    while (i < ws->size) ws->data[j++] = ws->data[i++];
                    ws->size = j;
                    return clauseref_from_binary(lit_neg(p), q);
                }
                continue;
            }

            Clause *c = watch_clause_ptr(w);
            if (c->magic != CLAUSE_MAGIC) {
                fprintf(stderr,
                        "KERBEROS: watch apunta a cláusula corrupta "
                        "(memoria dañada) en propagación\n");
                abort();
            }
            if (c->deleted) continue;

            if (c->size < 2) {
                ws->data[j++] = w;
                continue;
            }

            Lit false_lit = lit_neg(p);
            if (c->lits[0] == false_lit) {
                Lit t = c->lits[0];
                c->lits[0] = c->lits[1];
                c->lits[1] = t;
            }

            Lit first = c->lits[0];
            if (solver_lit_value(s, first) == 1) {
                ws->data[j++] = w;
                continue;
            }

            int found = 0;
            for (int k = 2; k < c->size; ++k) {
                Lit q = c->lits[k];
                if (solver_lit_value(s, q) != -1) {
                    c->lits[1] = q;
                    c->lits[k] = false_lit;
                    watchvec_push(&s->watches[lit_neg(q)], watch_from_clause(c));
                    found = 1;
                    break;
                }
            }
            if (found) continue;

            ws->data[j++] = w;

            if (solver_lit_value(s, first) == -1) {
                while (i < ws->size) ws->data[j++] = ws->data[i++];
                ws->size = j;
                return clauseref_from_clause(c);
            }
            if (!solver_enqueue(s, first, reason_from_clause(c))) {
                while (i < ws->size) ws->data[j++] = ws->data[i++];
                ws->size = j;
                return clauseref_from_clause(c);
            }
        }
        ws->size = j;
    }
    return clauseref_none();
}

static void solver_cancel_until(Solver *s, int level) {
    if (solver_decision_level(s) <= level) return;

    for (int dl = solver_decision_level(s); dl > level; --dl) {
        int from = s->trail_lim.data[dl - 1];
        for (int i = s->trail_size - 1; i >= from; --i) {
            Lit p = s->trail[i];
            int v = lit_var(p);
            s->assigns[v] = 0;
            s->levels[v] = 0;
            s->reasons[v] = reason_none();
            if (s->order.pos[v] < 0) heap_insert(&s->order, v);
        }
        s->trail_size = from;
        s->trail_lim.size--;
    }
    if (s->qhead > s->trail_size) s->qhead = s->trail_size;
}

static void solver_switch_heuristic(Solver *s, int heuristic) {
    int next = (heuristic == 1) ? 1 : 0;
    if (s->heuristic == next) return;
    if (next == 1 && s->chb_activity == NULL) next = 0;
    s->heuristic = next;
    s->order.activity = (s->heuristic == 1) ? s->chb_activity : s->activity;
    heap_rebuild(&s->order);
}

static void solver_var_bump(Solver *s, int v) {
    s->activity[v] += s->var_inc;
    if (s->activity[v] > 1e100) {
        for (int i = 0; i < s->nvars; ++i) s->activity[i] *= 1e-100;
        s->var_inc *= 1e-100;
    }
    if (s->heuristic == 0) heap_increase(&s->order, v);
}

static void solver_bump_chb(Solver *s, int v, double multiplier) {
    if (s->chb_activity == NULL || s->chb_last_conflict == NULL) return;
    long long age = s->conflicts - s->chb_last_conflict[v] + 1;
    if (age < 1) age = 1;
    double reward = multiplier / (double)age;
    double oldv = s->chb_activity[v];
    double newv = s->chb_step * reward + (1.0 - s->chb_step) * oldv;
    s->chb_activity[v] = newv;
    if (s->heuristic == 1) heap_update(&s->order, v);
}

static void solver_decay_chb_step(Solver *s) {
    if (s->chb_step > s->chb_step_min) {
        s->chb_step -= s->chb_step_dec;
        if (s->chb_step < s->chb_step_min) s->chb_step = s->chb_step_min;
    }
}

static void solver_update_chb_after_propagate(Solver *s, int conflict_found) {
    if (s->heuristic != 1) return;
    if (!conflict_found && (s->propagations & 63ll) != 0) return;
    int dl = solver_decision_level(s);
    if (dl <= 0) return;
    int from = s->trail_lim.data[dl - 1];
    for (int i = s->trail_size - 1; i >= from; --i) {
        int v = lit_var(s->trail[i]);
        solver_bump_chb(s, v, conflict_found ? 1.0 : 0.9);
    }
    if (conflict_found) solver_decay_chb_step(s);
}

static void solver_restart_mab(Solver *s) {
    if (!s->use_mab) return;
    if (s->heuristic < 0 || s->heuristic > 1) s->heuristic = 0;

    if (s->mab_epoch_decisions > 1.0 && s->mab_epoch_conflicts > 1.0) {
        double gain = log2(s->mab_epoch_decisions) / log2(s->mab_epoch_conflicts);
        if (isfinite(gain)) s->mab_reward[s->heuristic] += gain;
    }

    s->mab_epoch_decisions = 0.0;
    s->mab_epoch_conflicts = 0.0;

    unsigned stable_restarts = s->mab_select[0] + s->mab_select[1];
    int next = s->heuristic;
    if (stable_restarts < 2) {
        next = 1 - s->heuristic;
    } else {
        double best = -1e300;
        double c = s->mabc;
        for (int i = 0; i < 2; ++i) {
            double sel = s->mab_select[i] ? (double)s->mab_select[i] : 1.0;
            double avg = s->mab_reward[i] / sel;
            double bonus = sqrt(fmax(0.0, c * log((double)stable_restarts + 1.0) / sel));
            double ucb = avg + bonus;
            if (i == 0 || ucb > best) {
                best = ucb;
                next = i;
            }
        }
    }

    s->mab_select[next]++;
    if (next != s->heuristic) solver_switch_heuristic(s, next);
}

static void solver_clause_bump(Solver *s, Clause *c) {
    c->activity += (float)s->cla_inc;
    if (c->activity > 1e20f) {
        for (int i = 0; i < s->learnts.size; ++i) {
            Clause *d = s->learnts.data[i];
            if (d && !d->deleted) d->activity *= 1e-20f;
        }
        s->cla_inc *= 1e-20;
    }
    if (c->learnt && c->size > 2) {
        int seen_levels = 0;
        int stamp = ++s->lbd_stamp;
        if (stamp <= 0) {
            memset(s->lbd_marks.data, 0, (size_t)s->lbd_marks.size * sizeof(int));
            stamp = 1;
            s->lbd_stamp = 1;
        }
        for (int i = 0; i < c->size; ++i) {
            int lv = s->levels[lit_var(c->lits[i])];
            if (s->lbd_marks.data[lv] != stamp) {
                s->lbd_marks.data[lv] = stamp;
                seen_levels++;
                if (seen_levels >= (int)c->lbd) break;
            }
        }
        if (seen_levels > 0 && seen_levels < (int)c->lbd) {
            c->lbd = (uint32_t)seen_levels;
        }
    }
}

static inline void solver_var_decay(Solver *s) {
    s->var_inc *= (1.0 / s->var_decay);
}

static inline void solver_clause_decay(Solver *s) {
    s->cla_inc *= (1.0 / s->cla_decay);
}

static inline int assignment_lit_true(const unsigned char *values01, Lit p) {
    int val = values01[lit_var(p)] ? 1 : 0;
    int need = lit_sign(p) ? 0 : 1;
    return val == need;
}

static inline int solver_phase_lit_true(const Solver *s, Lit p) {
    return assignment_lit_true(s->phases, p);
}

static int ctcube_lit_cmp(const void *pa, const void *pb) {
    Lit a = *(const Lit *)pa;
    Lit b = *(const Lit *)pb;
    int va = lit_var(a), vb = lit_var(b);
    if (va != vb) return (va < vb) ? -1 : 1;
    int sa = lit_sign(a), sb = lit_sign(b);
    if (sa != sb) return (sa < sb) ? -1 : 1;
    return 0;
}

static int ctcube_compare_buddy_sorted(const Lit *a, const Lit *b, int n, int *diff_idx) {
    int diff = -1;
    for (int i = 0; i < n; ++i) {
        int va = lit_var(a[i]), vb = lit_var(b[i]);
        if (va != vb) return 0;
        int sa = lit_sign(a[i]), sb = lit_sign(b[i]);
        if (sa == sb) continue;
        if ((sa ^ sb) != 1) return 0;
        if (diff >= 0) return 0;
        diff = i;
    }
    if (diff_idx) *diff_idx = diff;
    return 1;
}

static int solver_covertrace_try_buddy_merge(Solver *s, Lit *cube, int *size) {
    if (*size <= 1 || s->ct_cubes.size == 0) return 0;

    int merged_any = 0;
    for (;;) {
        int found = 0;
        int start = s->ct_cubes.size > 512 ? s->ct_cubes.size - 512 : 0;
        for (int i = s->ct_cubes.size - 1; i >= start; --i) {
            CTCube *c = &s->ct_cubes.data[i];
            if ((int)c->size != *size) continue;

            int diff = -1;
            if (!ctcube_compare_buddy_sorted(cube, c->lits, *size, &diff)) continue;
            if (diff < 0) {
                return -1;  // duplicate
            }
            if (*size <= 1) continue;

            for (int k = diff; k < *size - 1; ++k) cube[k] = cube[k + 1];
            (*size)--;
            ctcubevec_remove_at(&s->ct_cubes, i);
            s->ct_merged++;
            merged_any = 1;
            found = 1;
            break;
        }
        if (!found || *size <= 1) break;
    }
    return merged_any;
}

static void solver_covertrace_feed_clause(Solver *s, const Lit *clause_lits, int size, uint32_t lbd) {
    if (!s->ct_enable) return;
    if (size <= 1 || size > s->ct_maxlen) return;
    if ((int)lbd > s->ct_lbd_max) return;

    Lit local_buf[32];
    Lit *cube = local_buf;
    if (size > (int)(sizeof(local_buf) / sizeof(local_buf[0]))) {
        cube = (Lit *)xmalloc((size_t)size * sizeof(Lit));
    }
    for (int i = 0; i < size; ++i) {
        cube[i] = lit_neg(clause_lits[i]);
    }

    qsort(cube, (size_t)size, sizeof(Lit), ctcube_lit_cmp);

    int out = 0;
    int bad = 0;
    for (int i = 0; i < size; ++i) {
        if (out > 0 && lit_var(cube[out - 1]) == lit_var(cube[i])) {
            if (lit_sign(cube[out - 1]) == lit_sign(cube[i])) {
                continue;
            }
            bad = 1;
            break;
        }
        cube[out++] = cube[i];
    }
    size = out;
    if (!bad && size > 1) {
        int skip = 0;
        if (s->ct_buddy_merge) {
            int merge_res = solver_covertrace_try_buddy_merge(s, cube, &size);
            if (merge_res < 0) skip = 1;
        }
        if (!skip && size > 1) {
            ctcubevec_push(&s->ct_cubes, cube, size);
            s->ct_added++;
        }
    }

    if (cube != local_buf) free(cube);
}

static int solver_covertrace_cube_hit_values(const Solver *s,
                                             const CTCube *cube,
                                             const unsigned char *values01) {
    (void)s;
    if (!cube || cube->size == 0) return 0;
    for (int i = 0; i < (int)cube->size; ++i) {
        if (!assignment_lit_true(values01, cube->lits[i])) return 0;
    }
    return 1;
}

static int solver_covertrace_escape_values(Solver *s,
                                           unsigned char *values01,
                                           int rounds,
                                           int use_activity) {
    if (!s->ct_enable || s->ct_cubes.size == 0) return 0;
    if (rounds <= 0) rounds = 1;
    int escaped = 0;
    for (int it = 0; it < rounds; ++it) {
        const CTCube *hit = NULL;
        int start = s->ct_cubes.size > 4096 ? s->ct_cubes.size - 4096 : 0;
        for (int i = s->ct_cubes.size - 1; i >= start; --i) {
            const CTCube *c = &s->ct_cubes.data[i];
            if (solver_covertrace_cube_hit_values(s, c, values01)) {
                hit = c;
                break;
            }
        }
        if (!hit) break;

        int pick = 0;
        if (use_activity) {
            double best = 1e300;
            for (int i = 0; i < (int)hit->size; ++i) {
                int v = lit_var(hit->lits[i]);
                double sc = s->activity[v];
                if (i == 0 || sc < best) {
                    best = sc;
                    pick = i;
                }
            }
        } else {
            pick = (int)(solver_rand_u32(s) % (uint32_t)hit->size);
        }

        int v = lit_var(hit->lits[pick]);
        values01[v] ^= 1u;
        escaped++;
    }
    return escaped;
}

static void solver_covertrace_escape_phases(Solver *s) {
    int escaped = solver_covertrace_escape_values(s, s->phases, s->ct_escape_rounds, 1);
    s->ct_escaped += escaped;
}

static uint32_t solver_compute_lbd(Solver *s, const IntVec *clause) {
    int stamp = ++s->lbd_stamp;
    if (stamp <= 0) {
        memset(s->lbd_marks.data, 0, (size_t)s->lbd_marks.size * sizeof(int));
        stamp = 1;
        s->lbd_stamp = 1;
    }
    uint32_t lbd = 0;
    for (int i = 0; i < clause->size; ++i) {
        int lv = s->levels[lit_var(clause->data[i])];
        if (s->lbd_marks.data[lv] != stamp) {
            s->lbd_marks.data[lv] = stamp;
            lbd++;
        }
    }
    if (lbd == 0) lbd = 1;
    return lbd;
}

static void solver_minimize_learnt(Solver *s, IntVec *learnt);
static void solver_eager_subsume(Solver *s, Clause *c);

static void solver_analyze(Solver *s, ClauseRef conflict, IntVec *learnt, int *out_bt) {
    learnt->size = 0;
    intvec_push(learnt, -1);

    int pathc = 0;
    Lit p = -1;
    int idx = s->trail_size - 1;

    s->analyze_stack.size = 0;

    do {
        if (clauseref_is_learnt(&conflict)) solver_clause_bump(s, conflict.clause);

        int start = (p == -1) ? 0 : 1;
        for (int j = start; j < conflict.size; ++j) {
            Lit q = clauseref_lit(&conflict, j);
            int v = lit_var(q);
            if (!s->seen[v] && s->levels[v] > 0) {
                s->seen[v] = 1;
                intvec_push(&s->analyze_stack, v);
                solver_var_bump(s, v);
                if (s->levels[v] == solver_decision_level(s)) {
                    pathc++;
                } else {
                    intvec_push(learnt, q);
                }
            }
        }

        while (!s->seen[lit_var(s->trail[idx])]) idx--;
        p = s->trail[idx--];
        s->seen[lit_var(p)] = 0;
        pathc--;
        conflict = clauseref_from_reason(s->reasons[lit_var(p)], p);
    } while (pathc > 0);

    learnt->data[0] = lit_neg(p);

    /* minimización recursiva de la cláusula aprendida (estilo CDCL moderno) */
    solver_minimize_learnt(s, learnt);

    if (learnt->size == 1) {
        *out_bt = 0;
    } else {
        int max_i = 1;
        int max_lv = s->levels[lit_var(learnt->data[1])];
        for (int i = 2; i < learnt->size; ++i) {
            int lv = s->levels[lit_var(learnt->data[i])];
            if (lv > max_lv) {
                max_lv = lv;
                max_i = i;
            }
        }
        if (max_i != 1) {
            Lit t = learnt->data[1];
            learnt->data[1] = learnt->data[max_i];
            learnt->data[max_i] = t;
        }
        *out_bt = max_lv;
    }

    for (int i = 0; i < s->analyze_stack.size; ++i) {
        int v = s->analyze_stack.data[i];
        if (s->chb_last_conflict != NULL) s->chb_last_conflict[v] = s->conflicts;
    }

    for (int i = 0; i < s->analyze_stack.size; ++i) {
        s->seen[s->analyze_stack.data[i]] = 0;
    }
    s->analyze_stack.size = 0;
}

static int solver_clause_locked(const Clause *c) {
    return c->locked && !c->deleted;
}

static int clause_cmp_quality(const void *pa, const void *pb) {
    const Clause *a = *(const Clause *const *)pa;
    const Clause *b = *(const Clause *const *)pb;

    int a_bin = (a->size <= 2);
    int b_bin = (b->size <= 2);
    if (a_bin != b_bin) return b_bin - a_bin;

    if (a->lbd != b->lbd) return (a->lbd < b->lbd) ? -1 : 1;
    if (a->activity != b->activity) return (a->activity > b->activity) ? -1 : 1;
    if (a->size != b->size) return (a->size < b->size) ? -1 : 1;
    return 0;
}

static void solver_gc_deleted_learnts(Solver *s) {
    /* limpiar el anillo de subsumción eager antes de liberar (evita dangling) */
    for (int i = 0; i < 4; ++i) {
        Clause *d = s->last_learnt[i];
        if (d != NULL && d->deleted) s->last_learnt[i] = NULL;
    }
    for (int i = 0; i < 2 * s->nvars; ++i) {
        WatchVec *ws = &s->watches[i];
        int j = 0;
        for (int k = 0; k < ws->size; ++k) {
            Watch w = ws->data[k];
            if (watch_is_binary(w)) {
                ws->data[j++] = w;
                continue;
            }
            Clause *c = watch_clause_ptr(w);
            if (c->magic != CLAUSE_MAGIC) {
                fprintf(stderr,
                        "KERBEROS: watch apunta a cláusula corrupta "
                        "(memoria dañada) en barrido\n");
                abort();
            }
            if (!c || c->deleted) continue;
            ws->data[j++] = w;
        }
        ws->size = j;
        if (ws->cap > 64 && ws->size * 4 < ws->cap) {
            int nc = ws->cap >> 1;
            if (nc < 64) nc = 64;
            if (nc < ws->size) nc = ws->size;
            ws->data = (Watch *)xrealloc(ws->data, (size_t)nc * sizeof(Watch));
            ws->cap = nc;
        }
    }
    s->watch_sweeps++;

    int out = 0;
    for (int i = 0; i < s->learnts.size; ++i) {
        Clause *c = s->learnts.data[i];
        if (!c) continue;
        if (c->deleted) {
            clause_free(c);
            s->deleted_freed++;
        } else {
            s->learnts.data[out++] = c;
        }
    }
    s->learnts.size = out;
}

static void solver_reduce_db(Solver *s) {
    int live = 0;
    for (int i = 0; i < s->learnts.size; ++i) {
        Clause *c = s->learnts.data[i];
        if (c && !c->deleted) {
            c->locked = 0;
            live++;
        }
    }
    if (live < 200) return;

    for (int i = 0; i < s->trail_size; ++i) {
        int v = lit_var(s->trail[i]);
        Reason rr = s->reasons[v];
        if (!reason_is_none(rr) && !reason_is_binary(rr)) {
            Clause *r = reason_clause_ptr(rr);
            if (r->learnt && !r->deleted) r->locked = 1;
        }
    }

    Clause **arr = (Clause **)xmalloc((size_t)live * sizeof(Clause *));
    int n = 0;
    for (int i = 0; i < s->learnts.size; ++i) {
        Clause *c = s->learnts.data[i];
        if (c && !c->deleted) arr[n++] = c;
    }

    qsort(arr, (size_t)n, sizeof(Clause *), clause_cmp_quality);

    int target = n / 2;
    int deleted_now = 0;
    for (int i = target; i < n; ++i) {
        Clause *c = arr[i];
        if (c->size <= 2) continue;
        if (c->lbd <= 2) continue;
        if (c->size <= 6) continue; /* tier1: conservar cláusulas cortas */
        if (solver_clause_locked(c)) continue;
        if (!c->deleted) {
            c->deleted = 1;
            deleted_now++;
        }
    }

    free(arr);

    if (deleted_now > 0) {
        solver_gc_deleted_learnts(s);
    } else {
        int out = 0;
        for (int i = 0; i < s->learnts.size; ++i) {
            Clause *c = s->learnts.data[i];
            if (c && !c->deleted) s->learnts.data[out++] = c;
        }
        s->learnts.size = out;
    }
}

static int luby(int y, int x) {
    int size = 1;
    int seq = 0;
    while (size < x + 1) {
        seq++;
        size = 2 * size + 1;
    }
    while (size - 1 != x) {
        size = (size - 1) >> 1;
        seq--;
        x = x % size;
    }
    int r = 1;
    for (int i = 0; i < seq; ++i) r *= y;
    return r;
}

static Lit solver_pick_branch_lit(Solver *s) {
    for (;;) {
        int v = heap_pop_max(&s->order);
        if (v < 0) return -1;
        if (s->assigns[v] != 0) continue;
        int sign = s->phases[v] ? 0 : 1;
        return mk_lit(v, sign);
    }
}

static void solver_log_proof_clause(Solver *s, const Lit *lits, int size) {
    if (!s->proof) return;
    for (int i = 0; i < size; ++i) {
        fprintf(s->proof, "%d ", lit_to_dimacs(lits[i]));
    }
    fputs("0\n", s->proof);
}

static int solver_add_binary_clause(Solver *s, Lit a, Lit b) {
    /* deduplicación global: devuelve 0 si la binaria ya existía */
    {
        WatchVec *ws = &s->watches[lit_neg(a)];
        for (int i = 0; i < ws->size; ++i) {
            Watch w = ws->data[i];
            if (watch_is_binary(w) && watch_binary_lit(w) == b) return 0;
        }
    }
    s->activity[lit_var(a)] += 1.0;
    s->activity[lit_var(b)] += 1.0;
    if (s->chb_activity != NULL) {
        s->chb_activity[lit_var(a)] += 1.0;
        s->chb_activity[lit_var(b)] += 1.0;
    }
    watchvec_push(&s->watches[lit_neg(a)], watch_from_binary(b));
    watchvec_push(&s->watches[lit_neg(b)], watch_from_binary(a));
    s->binary_clauses++;
    return 1;
}

static int solver_add_clause(Solver *s, const Lit *lits, int size, int learnt, uint32_t lbd) {
    if (!s->ok) return 0;

    if (size == 0) {
        s->ok = 0;
        return 0;
    }

    if (size == 1) {
        if (!solver_enqueue(s, lits[0], reason_none())) {
            s->ok = 0;
            return 0;
        }
        return 1;
    }

    if (!learnt && size == 2) {
        (void)solver_add_binary_clause(s, lits[0], lits[1]);
        return 1; /* la binaria está presente (aunque ya existiera) */
    }

    Clause *c = clause_new(lits, size, learnt, lbd);
    if (learnt) {
        clausevec_push(&s->learnts, c);
        solver_attach_clause(s, c);
    } else {
        for (int i = 0; i < size; ++i) {
            int v = lit_var(lits[i]);
            s->activity[v] += 1.0;
            if (s->chb_activity != NULL) s->chb_activity[v] += 1.0;
        }
        clausevec_push(&s->clauses, c);
        solver_attach_clause(s, c);
    }
    return 1;
}

static void solver_track_original_clause(Solver *s, const Lit *lits, int size) {
    if (s->hess_unit_freeze == NULL) return;
    if (size <= 0) {
        s->orig_empty_clauses++;
        return;
    }
    if (size != 1) return;

    Lit unit = lits[0];
    intvec_push(&s->orig_unit_lits, unit);
    s->hess_unit_freeze[lit_var(unit)] = 1u;
}

static int solver_verify_model(const Solver *s) {
    for (int i = 0; i < 2 * s->nvars; ++i) {
        if (solver_lit_value(s, (Lit)i) != 1) continue;
        const WatchVec *ws = &s->watches[i];
        for (int k = 0; k < ws->size; ++k) {
            Watch w = ws->data[k];
            if (!watch_is_binary(w)) continue;
            if (solver_lit_value(s, watch_binary_lit(w)) != 1) return 0;
        }
    }
    for (int i = 0; i < s->clauses.size; ++i) {
        const Clause *c = s->clauses.data[i];
        if (!c || c->deleted) continue;
        int sat = 0;
        for (int k = 0; k < c->size; ++k) {
            if (solver_lit_value(s, c->lits[k]) == 1) {
                sat = 1;
                break;
            }
        }
        if (!sat) {
            return 0;
        }
    }
    return 1;
}

static int solver_hess_flip_nonunit(int var,
                                    unsigned char *values01,
                                    IntVec *pos_occ,
                                    IntVec *neg_occ,
                                    int *sat_count,
                                    int unsat_nonunit) {
    int old = values01[var] ? 1 : 0;
    values01[var] ^= 1u;

    if (old) {
        IntVec *pv = &pos_occ[var];
        IntVec *nv = &neg_occ[var];
        for (int i = 0; i < pv->size; ++i) {
            int ci = pv->data[i];
            int before = sat_count[ci];
            sat_count[ci] = before - 1;
            if (before == 1) unsat_nonunit++;
        }
        for (int i = 0; i < nv->size; ++i) {
            int ci = nv->data[i];
            int before = sat_count[ci];
            sat_count[ci] = before + 1;
            if (before == 0) unsat_nonunit--;
        }
    } else {
        IntVec *nv = &neg_occ[var];
        IntVec *pv = &pos_occ[var];
        for (int i = 0; i < nv->size; ++i) {
            int ci = nv->data[i];
            int before = sat_count[ci];
            sat_count[ci] = before - 1;
            if (before == 1) unsat_nonunit++;
        }
        for (int i = 0; i < pv->size; ++i) {
            int ci = pv->data[i];
            int before = sat_count[ci];
            sat_count[ci] = before + 1;
            if (before == 0) unsat_nonunit--;
        }
    }

    return unsat_nonunit;
}

static int solver_hess_exact_search(Solver *s, unsigned char *best_model, int *best_unsat_out) {
    int n = s->nvars;
    int m_nonunit = s->clauses.size;
    int total_clauses = m_nonunit + s->orig_unit_lits.size + s->orig_empty_clauses;
    if (n <= 0 || total_clauses <= 0) {
        if (best_model && n > 0) memset(best_model, 0, (size_t)n * sizeof(unsigned char));
        if (best_unsat_out) *best_unsat_out = 0;
        return 1;
    }

    IntVec *pos_occ = (IntVec *)xmalloc((size_t)n * sizeof(IntVec));
    IntVec *neg_occ = (IntVec *)xmalloc((size_t)n * sizeof(IntVec));
    for (int v = 0; v < n; ++v) {
        intvec_init(&pos_occ[v]);
        intvec_init(&neg_occ[v]);
    }

    for (int ci = 0; ci < m_nonunit; ++ci) {
        Clause *c = s->clauses.data[ci];
        if (!c || c->deleted) continue;
        for (int k = 0; k < c->size; ++k) {
            Lit p = c->lits[k];
            if (lit_sign(p)) intvec_push(&neg_occ[lit_var(p)], ci);
            else intvec_push(&pos_occ[lit_var(p)], ci);
        }
    }

    int *sat_count = NULL;
    if (m_nonunit > 0) sat_count = (int *)xmalloc((size_t)m_nonunit * sizeof(int));

    unsigned char *sat = NULL;
    unsigned char *opt = NULL;
    unsigned char *freeze = NULL;
    if (n > 0) {
        sat = (unsigned char *)xmalloc((size_t)n * sizeof(unsigned char));
        opt = (unsigned char *)xmalloc((size_t)n * sizeof(unsigned char));
        freeze = (unsigned char *)xmalloc((size_t)n * sizeof(unsigned char));
        memset(freeze, 0, (size_t)n * sizeof(unsigned char));
        for (int v = 0; v < n; ++v) {
            if (s->assigns[v] != 0) {
                sat[v] = (unsigned char)(s->assigns[v] > 0 ? 1u : 0u);
                freeze[v] = 1u;
            } else {
                sat[v] = 0u;
                freeze[v] = 0u;
            }
            if (s->hess_unit_freeze != NULL && s->hess_unit_freeze[v]) freeze[v] = 1u;
        }
        memcpy(opt, sat, (size_t)n * sizeof(unsigned char));
    }

    int constant_unsat = s->orig_empty_clauses;
    for (int i = 0; i < s->orig_unit_lits.size; ++i) {
        if (!assignment_lit_true(sat, (Lit)s->orig_unit_lits.data[i])) constant_unsat++;
    }

    int unsat_nonunit = 0;
    for (int ci = 0; ci < m_nonunit; ++ci) {
        Clause *c = s->clauses.data[ci];
        int sat_literals = 0;
        if (c && !c->deleted) {
            for (int k = 0; k < c->size; ++k) {
                if (assignment_lit_true(sat, c->lits[k])) sat_literals++;
            }
        }
        sat_count[ci] = sat_literals;
        if (sat_literals == 0) unsat_nonunit++;
    }

    int cur_best = constant_unsat + unsat_nonunit;
    if (best_model && n > 0) memcpy(best_model, opt, (size_t)n * sizeof(unsigned char));
    if (cur_best == 0) {
        if (best_unsat_out) *best_unsat_out = 0;
        for (int v = 0; v < n; ++v) {
            intvec_free(&pos_occ[v]);
            intvec_free(&neg_occ[v]);
        }
        free(pos_occ);
        free(neg_occ);
        free(sat_count);
        free(sat);
        free(opt);
        free(freeze);
        return 1;
    }

    for (;;) {
        if (solver_poll_external_stop(s)) {
            break;
        }
        int done = 1;
        int glb = total_clauses + 1;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (solver_poll_external_stop(s)) {
                    done = 1;
                    glb = cur_best;
                    break;
                }
                if (!freeze[i]) {
                    unsat_nonunit = solver_hess_flip_nonunit(i, sat, pos_occ, neg_occ, sat_count, unsat_nonunit);
                }
                if (!freeze[j]) {
                    unsat_nonunit = solver_hess_flip_nonunit(j, sat, pos_occ, neg_occ, sat_count, unsat_nonunit);
                }

                int loc = constant_unsat + unsat_nonunit;
                if (loc < glb) {
                    glb = loc;
                    if (glb < cur_best) {
                        done = 0;
                        cur_best = glb;
                        if (n > 0) memcpy(opt, sat, (size_t)n * sizeof(unsigned char));
                        if (cur_best == 0) {
                            if (best_model && n > 0) memcpy(best_model, opt, (size_t)n * sizeof(unsigned char));
                            if (best_unsat_out) *best_unsat_out = 0;
                            for (int v = 0; v < n; ++v) {
                                intvec_free(&pos_occ[v]);
                                intvec_free(&neg_occ[v]);
                            }
                            free(pos_occ);
                            free(neg_occ);
                            free(sat_count);
                            free(sat);
                            free(opt);
                            free(freeze);
                            return 1;
                        }
                    }
                } else if (loc > glb) {
                    if (!freeze[j]) {
                        unsat_nonunit = solver_hess_flip_nonunit(j, sat, pos_occ, neg_occ, sat_count, unsat_nonunit);
                    }
                    if (!freeze[i]) {
                        unsat_nonunit = solver_hess_flip_nonunit(i, sat, pos_occ, neg_occ, sat_count, unsat_nonunit);
                    }
                }
            }
        }

        if (done) break;
    }

    if (best_model && n > 0) memcpy(best_model, opt, (size_t)n * sizeof(unsigned char));
    if (best_unsat_out) *best_unsat_out = cur_best;

    for (int v = 0; v < n; ++v) {
        intvec_free(&pos_occ[v]);
        intvec_free(&neg_occ[v]);
    }
    free(pos_occ);
    free(neg_occ);
    free(sat_count);
    free(sat);
    free(opt);
    free(freeze);
    return 0;
}

static int solver_probe_conflict_with_cube(Solver *s, const Lit *cube, int size) {
    int old_dl = solver_decision_level(s);
    intvec_push(&s->trail_lim, s->trail_size);
    for (int i = 0; i < size; ++i) {
        if (!solver_enqueue(s, cube[i], reason_none())) {
            solver_cancel_until(s, old_dl);
            return 1;
        }
        ClauseRef conf = solver_propagate(s);
        if (!clauseref_is_none(conf)) {
            solver_cancel_until(s, old_dl);
            return 1;
        }
    }
    solver_cancel_until(s, old_dl);
    return 0;
}

static void solver_covertrace_probe(Solver *s) {
    if (!s->ct_enable || s->proof) return;
    if (solver_decision_level(s) != 0) return;
    if (s->ct_probe_restarts <= 0) return;

    enum { SLIME_CT_PROBE_MAX_DEPTH = 6 };
    Lit cube[SLIME_CT_PROBE_MAX_DEPTH];
    int depth = 0;

    int need = SLIME_CT_PROBE_MAX_DEPTH < s->nvars ? SLIME_CT_PROBE_MAX_DEPTH : s->nvars;
    for (int i = 0; i < s->order.size && depth < need; ++i) {
        int v = s->order.heap[i];
        if (v < 0 || v >= s->nvars) continue;
        if (s->assigns[v] != 0) continue;
        cube[depth++] = mk_lit(v, s->phases[v] ? 0 : 1);
    }
    if (depth == 0) return;

    if (!solver_probe_conflict_with_cube(s, cube, depth)) return;

    int changed = 1;
    while (changed && depth > 1) {
        if (solver_poll_external_stop(s)) return;
        changed = 0;
        for (int i = 0; i < depth; ++i) {
            Lit tmp[SLIME_CT_PROBE_MAX_DEPTH];
            int tn = 0;
            for (int j = 0; j < depth; ++j) {
                if (j == i) continue;
                tmp[tn++] = cube[j];
            }
            if (tn > 0 && solver_probe_conflict_with_cube(s, tmp, tn)) {
                for (int j = 0; j < tn; ++j) cube[j] = tmp[j];
                depth = tn;
                changed = 1;
                break;
            }
        }
    }

    Lit block[SLIME_CT_PROBE_MAX_DEPTH];
    for (int i = 0; i < depth; ++i) block[i] = lit_neg(cube[i]);
    uint32_t lbd = (depth > 0) ? (uint32_t)depth : 1u;
    if (solver_add_clause(s, block, depth, 1, lbd)) {
        solver_log_proof_clause(s, block, depth);
        solver_covertrace_feed_clause(s, block, depth, lbd);
        s->ct_probe_added++;
    }
}

static int solver_apply_model01(Solver *s, const unsigned char *model01) {
    int old_dl = solver_decision_level(s);
    int opened = 0;
    for (int v = 0; v < s->nvars; ++v) {
        int val = model01[v] ? 1 : -1;
        s->phases[v] = model01[v];
        if (s->assigns[v] != 0) {
            if (s->assigns[v] != val) {
                if (opened) solver_cancel_until(s, old_dl);
                return 0;
            }
            continue;
        }
        if (!opened) {
            intvec_push(&s->trail_lim, s->trail_size);
            opened = 1;
        }
        if (!solver_enqueue(s, mk_lit(v, val < 0), reason_none())) {
            solver_cancel_until(s, old_dl);
            return 0;
        }
    }
    return 1;
}

static int solver_record_learnt_clause(Solver *s,
                                       const IntVec *learnt,
                                       uint32_t lbd,
                                       int enqueue_asserting) {
    if (learnt->size <= 0) {
        s->ok = 0;
        return 0;
    }

    solver_log_proof_clause(s, (Lit *)learnt->data, learnt->size);

    if (learnt->size == 1) {
        if (!solver_enqueue(s, learnt->data[0], reason_none())) {
            s->ok = 0;
            return 0;
        }
        return 1;
    }

    Clause *c = clause_new((Lit *)learnt->data, learnt->size, 1, lbd);
    c->activity = 0.0f;
    clausevec_push(&s->learnts, c);
    solver_attach_clause(s, c);
    solver_covertrace_feed_clause(s, (Lit *)learnt->data, learnt->size, lbd);
    if (learnt->size > 2) solver_eager_subsume(s, c);

    if (enqueue_asserting) {
        if (!solver_enqueue(s, learnt->data[0], reason_from_clause(c))) {
            s->ok = 0;
            return 0;
        }
        c->locked = 1; /* es razón activa del literal recién asignado */
    }
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SIMPLIFICACIÓN (preprocesamiento raíz + inprocessing) — mecanismos inspirados
 * en la literatura CDCL moderna (eliminación acotada de variables, subsumción/
 * fortalecimiento y sustitución de literales equivalentes), reimplementados
 * desde cero y sin dependencias.
 * Toda transformación es de equivalencia: SAT/UNSAT se preservan y el modelo
 * de las variables eliminadas se reconstruye con solver_extend_model.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* codificación de ocurrencias: v > 0 → índice de cláusula (v-1);
 * v < 0 → binaria (este literal ∨ otro) con otro = -v-2 (id de literal). */
static inline int occ_clause_idx(int v) { return v - 1; }
static inline Lit occ_binary_other(int v) { return -v - 2; }
static inline int occ_enc_binary(Lit other) { return -other - 2; }

/* asignador de sellos (stamps) para lit_mark: cada adquisición devuelve un
 * valor único; ante el desbordamiento se limpia la tabla completa. */
static int solver_next_stamp(Solver *s) {
    int stamp = ++s->lit_mark_stamp;
    if (stamp <= 0) {
        memset(s->lit_mark, 0, (size_t)(2 * s->nvars) * sizeof(int));
        stamp = 1;
        s->lit_mark_stamp = 1;
    }
    return stamp;
}

static void occ_push(Solver *s, Lit lit, int val) {
    intvec_push(&s->occ[lit], val);
}

static void solver_occ_build(Solver *s) {
    int m = 2 * s->nvars;
    if (s->occ == NULL) {
        s->occ = (IntVec *)xmalloc((size_t)m * sizeof(IntVec));
        for (int i = 0; i < m; ++i) intvec_init(&s->occ[i]);
    } else {
        for (int i = 0; i < m; ++i) s->occ[i].size = 0;
    }
    for (int idx = 0; idx < s->clauses.size; ++idx) {
        Clause *c = s->clauses.data[idx];
        if (c == NULL || c->deleted) continue;
        for (int k = 0; k < c->size; ++k) {
            occ_push(s, c->lits[k], idx + 1);
        }
    }
    for (int l = 0; l < m; ++l) {
        WatchVec *ws = &s->watches[l];
        for (int k = 0; k < ws->size; ++k) {
            Watch w = ws->data[k];
            if (!watch_is_binary(w)) continue;
            Lit q = watch_binary_lit(w);
            Lit a = l ^ 1; /* ¬l: la binaria es (¬l ∨ q) */
            if (a < q) {
                occ_push(s, a, occ_enc_binary(q));
                occ_push(s, q, occ_enc_binary(a));
            }
        }
    }
}

static void solver_occ_free(Solver *s) {
    if (s->occ == NULL) return;
    for (int i = 0; i < 2 * s->nvars; ++i) intvec_free(&s->occ[i]);
    free(s->occ);
    s->occ = NULL;
}

/* recolección de basura completa (raíz): compacta cláusulas + aprendidas,
 * barre las watch lists y libera las cláusulas eliminadas.
 * ORDEN CRÍTICO: el barrido de watch lists va ANTES de liberar (la memoria
 * liberada puede ser reutilizada y el barrido debe ver las cláusulas vivas). */
static void solver_gc_all(Solver *s) {
    /* 1. barrido de watch lists (cláusulas aún en memoria) */
    for (int i = 0; i < 2 * s->nvars; ++i) {
        WatchVec *ws = &s->watches[i];
        int j = 0;
        for (int k = 0; k < ws->size; ++k) {
            Watch w = ws->data[k];
            if (watch_is_binary(w)) {
                ws->data[j++] = w;
                continue;
            }
            Clause *c = watch_clause_ptr(w);
            if (c->magic != CLAUSE_MAGIC) {
                fprintf(stderr,
                        "KERBEROS: watch apunta a cláusula corrupta "
                        "(memoria dañada) en gc_all\n");
                abort();
            }
            if (c == NULL || c->deleted) continue;
            ws->data[j++] = w;
        }
        ws->size = j;
    }
    /* 2. limpiar el anillo de subsumción eager (evita dangling) */
    for (int i = 0; i < 4; ++i) {
        if (s->last_learnt[i] != NULL && s->last_learnt[i]->deleted) {
            s->last_learnt[i] = NULL;
        }
    }
    /* 3. compactar y liberar */
    int out = 0;
    for (int i = 0; i < s->clauses.size; ++i) {
        Clause *c = s->clauses.data[i];
        if (c == NULL) continue;
        if (c->deleted) clause_free(c);
        else s->clauses.data[out++] = c;
    }
    s->clauses.size = out;
    out = 0;
    for (int i = 0; i < s->learnts.size; ++i) {
        Clause *c = s->learnts.data[i];
        if (c == NULL) continue;
        if (c->deleted) clause_free(c);
        else s->learnts.data[out++] = c;
    }
    s->learnts.size = out;
}

static void solver_unwatch_binary(Solver *s, Lit a, Lit b) {
    int la = lit_neg(a);
    int lb = lit_neg(b);
    for (int i = 0; i < s->watches[la].size; ++i) {
        Watch w = s->watches[la].data[i];
        if (watch_is_binary(w) && watch_binary_lit(w) == b) {
            s->watches[la].data[i] = s->watches[la].data[--s->watches[la].size];
            break;
        }
    }
    for (int i = 0; i < s->watches[lb].size; ++i) {
        Watch w = s->watches[lb].data[i];
        if (watch_is_binary(w) && watch_binary_lit(w) == a) {
            s->watches[lb].data[i] = s->watches[lb].data[--s->watches[lb].size];
            break;
        }
    }
    s->binary_clauses--;
    /* invalidar las entradas binarias correspondientes en occ (evita que
       eliminaciones posteriores resuelvan contra una binaria borrada) */
    if (s->occ != NULL) {
        int ea = occ_enc_binary(b);
        for (int i = 0; i < s->occ[a].size; ++i) {
            if (s->occ[a].data[i] == ea) {
                s->occ[a].data[i] = s->occ[a].data[--s->occ[a].size];
                break;
            }
        }
        int eb = occ_enc_binary(a);
        for (int i = 0; i < s->occ[b].size; ++i) {
            if (s->occ[b].data[i] == eb) {
                s->occ[b].data[i] = s->occ[b].data[--s->occ[b].size];
                break;
            }
        }
    }
}

/* ── sondeo de literales fallidos (bounded) ── */
static int solver_probe_literals(Solver *s, int max_probes, long long max_ticks) {
    long long base = s->propagations;
    int probed = 0;
    for (int v = 0;
         v < s->nvars && probed < max_probes && (s->propagations - base) < max_ticks;
         ++v) {
        if (s->assigns[v] != 0) continue;
        if (s->order.pos[v] < 0) continue;
        for (int sign = 0; sign < 2; ++sign) {
            Lit l = mk_lit(v, sign);
            int old_dl = solver_decision_level(s);
            intvec_push(&s->trail_lim, s->trail_size);
            int ok = solver_enqueue(s, l, reason_none());
            ClauseRef conf = clauseref_none();
            if (ok) conf = solver_propagate(s);
            int conflict = !clauseref_is_none(conf);
            solver_cancel_until(s, old_dl);
            probed++;
            if (conflict) {
                if (!solver_enqueue(s, lit_neg(l), reason_none())) { return 0; }
                ClauseRef c2 = solver_propagate(s);
                if (!clauseref_is_none(c2) && solver_decision_level(s) == 0) return 0;
                s->simp_probe_units++;
                break;
            }
            if (!ok) break;
        }
    }
    return s->ok;
}

/* ── subsumción directa + fortalecimiento (self-subsuming resolution) ── */
static int solver_subsume_strengthen(Solver *s, long long max_ticks) {
    long long ticks = 0;
    for (int idx = 0; idx < s->clauses.size && ticks < max_ticks; ++idx) {
        Clause *c = s->clauses.data[idx];
        if (c == NULL || c->deleted) continue;
        if (c->size < 2 || c->size > 64) continue;
        ticks += (long long)c->size;
        int satc = 0;
        for (int k = 0; k < c->size; ++k) {
            if (solver_lit_value(s, c->lits[k]) == 1) { satc = 1; break; }
        }
        if (satc) { c->deleted = 1; continue; }
        int stamp = solver_next_stamp(s);
        for (int k = 0; k < c->size; ++k) s->lit_mark[c->lits[k]] = stamp;
        int del = 0;
        Lit remove = 0;
        for (int k = 0; k < c->size && !del; ++k) {
            Lit l = c->lits[k];
            IntVec *oc = &s->occ[l];
            int nscan = oc->size < 500 ? oc->size : 500;
            for (int e = 0; e < nscan && !del; ++e) {
                int en = oc->data[e];
                if (en > 0) {
                    Clause *d = s->clauses.data[occ_clause_idx(en)];
                    if (d == NULL || d->deleted || d == c) continue;
                    if (d->size > c->size) continue;
                    int dsat = 0;
                    for (int dk = 0; dk < d->size; ++dk) {
                        if (solver_lit_value(s, d->lits[dk]) == 1) { dsat = 1; break; }
                    }
                    if (dsat) continue;
                    int unmarked = 0;
                    Lit q = 0;
                    for (int dk = 0; dk < d->size; ++dk) {
                        Lit dl = d->lits[dk];
                        if (s->lit_mark[dl] != stamp) {
                            unmarked++;
                            q = dl;
                            if (unmarked > 1) break;
                        }
                    }
                    if (unmarked == 0) { del = 1; break; }
                    if (unmarked == 1 && s->lit_mark[lit_neg(q)] == stamp) {
                        remove = lit_neg(q);
                    }
                } else {
                    Lit other = occ_binary_other(en);
                    if (other == l) continue;
                    if (s->lit_mark[other] == stamp) { del = 1; break; }
                }
            }
        }
        if (del) {
            c->deleted = 1;
            s->simp_subsumed++;
            continue;
        }
        if (remove != 0) {
            intvec_clear(&s->simp_buf1);
            for (int k = 0; k < c->size; ++k) {
                Lit l = c->lits[k];
                if (l == remove) continue;
                if (solver_lit_value(s, l) == -1) continue;
                intvec_push(&s->simp_buf1, l);
            }
            c->deleted = 1;
            s->simp_strengthened++;
            if (!solver_add_clause(s, (Lit *)s->simp_buf1.data, s->simp_buf1.size, 0, 0)) {
                if (!s->ok) return 0;
            }
            if (s->simp_buf1.size >= 3) {
                int newidx = s->clauses.size - 1;
                for (int k = 0; k < s->simp_buf1.size; ++k) {
                    occ_push(s, s->simp_buf1.data[k], newidx + 1);
                }
            }
        }
    }
    return s->ok;
}

/* ── sustitución de literales equivalentes (SCC sobre binarias) ── */
typedef struct { uint64_t packed; } BinPack;

static int bins_cmp(const void *pa, const void *pb) {
    uint64_t x = ((const BinPack *)pa)->packed;
    uint64_t y = ((const BinPack *)pb)->packed;
    return (x > y) - (x < y);
}

static int solver_find_equiv_repr(Solver *s, int *repr) {
    int m = 2 * s->nvars;
    int *disc = (int *)xmalloc((size_t)m * sizeof(int));
    int *low = (int *)xmalloc((size_t)m * sizeof(int));
    int *curs = (int *)xmalloc((size_t)m * sizeof(int));
    unsigned char *onstk = (unsigned char *)xmalloc((size_t)m * sizeof(unsigned char));
    memset(disc, 0xFF, (size_t)m * sizeof(int));
    memset(low, 0, (size_t)m * sizeof(int));
    memset(curs, 0, (size_t)m * sizeof(int));
    memset(onstk, 0, (size_t)m * sizeof(unsigned char));
    IntVec stk, dfs;
    intvec_init(&stk);
    intvec_init(&dfs);
    int time = 0;
    int ok = 1;
    for (int root = 0; root < m && ok; ++root) {
        if (disc[root] >= 0) continue;
        intvec_push(&dfs, root);
        while (dfs.size > 0 && ok) {
            int v = dfs.data[dfs.size - 1];
            IntVec *oc = &s->occ[v ^ 1];
            if (disc[v] < 0) {
                disc[v] = low[v] = ++time;
                intvec_push(&stk, v);
                onstk[v] = 1;
            }
            int recurse = 0;
            while (curs[v] < oc->size) {
                int en = oc->data[curs[v]++];
                if (en >= 0) continue;
                Lit q = occ_binary_other(en);
                if (disc[q] < 0) {
                    intvec_push(&dfs, q);
                    recurse = 1;
                    break;
                } else if (onstk[q] && disc[q] < low[v]) {
                    low[v] = disc[q];
                }
            }
            if (recurse) continue;
            int lv = low[v];
            if (lv == disc[v]) {
                intvec_clear(&s->simp_buf2);
                for (;;) {
                    int u = stk.data[--stk.size];
                    onstk[u] = 0;
                    intvec_push(&s->simp_buf2, u);
                    if (u == v) break;
                }
                int minm = m;
                for (int i = 0; i < s->simp_buf2.size; ++i) {
                    int u = s->simp_buf2.data[i];
                    if (u < minm) minm = u;
                }
                /* lit y ¬lit en la misma SCC → contradicción */
                for (int i = 0; i < s->simp_buf2.size && ok; ++i) {
                    int u = s->simp_buf2.data[i];
                    int nb = u ^ 1;
                    for (int j = 0; j < s->simp_buf2.size; ++j) {
                        if (s->simp_buf2.data[j] == nb) {
                            ok = 0;
                            break;
                        }
                    }
                    if (!ok) break;
                }
                for (int i = 0; i < s->simp_buf2.size; ++i) {
                    repr[s->simp_buf2.data[i]] = minm;
                }
            }
            dfs.size--;
            if (dfs.size > 0) {
                int u = dfs.data[dfs.size - 1];
                if (lv < low[u]) low[u] = lv;
            }
        }
    }
    intvec_free(&stk);
    intvec_free(&dfs);
    free(disc);
    free(low);
    free(curs);
    free(onstk);
    return ok;
}

static int solver_apply_equiv_substitution(Solver *s, int *repr) {
    int m = 2 * s->nvars;
    int any = 0;
    for (int v = 0; v < s->nvars; ++v) {
        if (s->assigns[v] != 0) continue;
        if (repr[v << 1] != (v << 1)) { any = 1; break; }
    }
    if (!any) return 1;

    /* 1. SCC con literales raíz-verdaderos/falsos */
    int stampT = solver_next_stamp(s);
    int stampF = solver_next_stamp(s);
    for (int l = 0; l < m; ++l) {
        if (s->assigns[l >> 1] == 0) continue;
        int val = solver_lit_value(s, l);
        if (val == 0) continue;
        int r = repr[l];
        if (val == 1) {
            if (s->lit_mark[r] == stampF) { return 0; }
            s->lit_mark[r] = stampT;
        } else {
            if (s->lit_mark[r] == stampT) { return 0; }
            s->lit_mark[r] = stampF;
        }
    }
    for (int v = 0; v < s->nvars; ++v) {
        if (s->assigns[v] != 0) continue;
        int r = repr[v << 1];
        if (s->lit_mark[r] == stampT) {
            if (!solver_enqueue(s, mk_lit(v, 0), reason_none())) { return 0; }
        } else if (s->lit_mark[r] == stampF) {
            if (!solver_enqueue(s, mk_lit(v, 1), reason_none())) { return 0; }
        }
    }

    /* 2. recoger y transformar binarias */
    size_t nbin = 0, capbin = 1024;
    BinPack *bins = (BinPack *)xmalloc(capbin * sizeof(BinPack));
    for (int l = 0; l < m; ++l) {
        WatchVec *ws = &s->watches[l];
        for (int k = 0; k < ws->size; ++k) {
            Watch w = ws->data[k];
            if (!watch_is_binary(w)) continue;
            Lit q = watch_binary_lit(w);
            Lit a = l ^ 1;
            if (a >= q) continue;
            Lit na = repr[a];
            Lit nb = repr[q];
            if (na == (nb ^ 1)) continue;
            if (na == nb) {
                if (!solver_enqueue(s, na, reason_none())) { free(bins); return 0; }
                continue;
            }
            if (na > nb) { Lit t = na; na = nb; nb = t; }
            if (nbin == capbin) {
                capbin *= 2;
                bins = (BinPack *)xrealloc(bins, capbin * sizeof(BinPack));
            }
            bins[nbin++].packed = ((uint64_t)(uint32_t)na << 32) | (uint64_t)(uint32_t)nb;
        }
    }

    /* 3. transformar cláusulas (irredundantes + aprendidas) en sitio */
    int ok = 1;
    for (int pass = 0; pass < 2 && ok; ++pass) {
        ClauseVec *vec = (pass == 0) ? &s->clauses : &s->learnts;
        for (int idx = 0; idx < vec->size && ok; ++idx) {
            Clause *c = vec->data[idx];
            if (c == NULL || c->deleted) continue;
            int stampC = solver_next_stamp(s); /* sello fresco POR cláusula */
            int out = 0;
            int satisfied = 0;
            for (int k = 0; k < c->size; ++k) {
                Lit l = c->lits[k];
                int v_ = lit_var(l);
                if (s->assigns[v_] != 0) {
                    if (solver_lit_value(s, l) == 1) { satisfied = 1; break; }
                    continue;
                }
                Lit nl = repr[l];
                if (s->lit_mark[nl] == stampC) continue;
                if (s->lit_mark[nl ^ 1] == stampC) { satisfied = 1; break; }
                s->lit_mark[nl] = stampC;
                c->lits[out++] = nl;
            }
            if (satisfied) {
                c->deleted = 1;
                continue;
            }
            c->size = out;
            if (out == 0) {
                ok = 0;
                break;
            }
            if (out == 1) {
                if (!solver_enqueue(s, c->lits[0], reason_none())) {
                    ok = 0;
                    break;
                }
                c->deleted = 1;
                continue;
            }
            if (c->learnt && (uint32_t)out < c->lbd) c->lbd = (uint32_t)out;
        }
    }
    if (!ok) { free(bins); return 0; }

    /* 4. registrar pares equivalentes para la extensión del modelo */
    for (int v = 0; v < s->nvars; ++v) {
        if (s->assigns[v] != 0) continue;
        Lit l = mk_lit(v, 0);
        if (repr[l] != l) {
            intvec_push(&s->eq_pairs, v);
            intvec_push(&s->eq_pairs, lit_to_dimacs(repr[l]));
            if (s->extend_size == s->extend_cap) {
                int nc = s->extend_cap ? s->extend_cap * 2 : 64;
                s->extend = (ExtRec *)xrealloc(s->extend, (size_t)nc * sizeof(ExtRec));
                s->extend_cap = nc;
            }
            {
                ExtRec *rec = &s->extend[s->extend_size++];
                rec->var = v;
                rec->eq_lit = lit_to_dimacs(repr[l]);
                rec->cls = NULL;
                rec->ncls = 0;
            }
            if (s->order.pos[v] >= 0) heap_remove(&s->order, v);
            s->simp_equivs++;
        }
    }

    /* 5. reconstruir watch lists (reutilizando las asignaciones) */
    /* deduplicar binarias transformadas (pueden colapsar a la misma) */
    if (nbin > 1) {
        qsort(bins, nbin, sizeof(BinPack), bins_cmp);
        size_t nuniq = 0;
        for (size_t i = 0; i < nbin; ++i) {
            if (i == 0 || bins[i].packed != bins[i - 1].packed) {
                bins[nuniq++] = bins[i];
            }
        }
        nbin = nuniq;
    }
    for (int i = 0; i < m; ++i) s->watches[i].size = 0;
    s->binary_clauses = 0;
    for (size_t i = 0; i < nbin; ++i) {
        Lit a = (Lit)(uint32_t)(bins[i].packed >> 32);
        Lit b = (Lit)(uint32_t)bins[i].packed;
        watchvec_push(&s->watches[a ^ 1], watch_from_binary(b));
        watchvec_push(&s->watches[b ^ 1], watch_from_binary(a));
        s->binary_clauses++;
    }
    free(bins);
    for (int pass = 0; pass < 2; ++pass) {
        ClauseVec *vec = (pass == 0) ? &s->clauses : &s->learnts;
        for (int idx = 0; idx < vec->size; ++idx) {
            Clause *c = vec->data[idx];
            if (c == NULL || c->deleted) continue;
            solver_attach_clause(s, c);
        }
    }
    return s->ok;
}
static int solver_substitute_equivalents(Solver *s) {
    
    int m = 2 * s->nvars;
    int *repr = (int *)xmalloc((size_t)m * sizeof(int));
    for (int i = 0; i < m; ++i) repr[i] = i;
    int ok = solver_find_equiv_repr(s, repr);
    if (!ok) { free(repr); return 0; }
    ok = solver_apply_equiv_substitution(s, repr);
    free(repr);
    return ok;
}

/* ── eliminación acotada de variables (BVE) ── */
static int solver_bve_try_eliminate(Solver *s, int v) {
    Lit pos = mk_lit(v, 0);
    Lit neg = mk_lit(v, 1);
    intvec_clear(&s->simp_buf1); /* items pos */
    intvec_clear(&s->simp_buf2); /* items neg */
    for (int side = 0; side < 2; ++side) {
        IntVec *dst = (side == 0) ? &s->simp_buf1 : &s->simp_buf2;
        Lit lit = (side == 0) ? pos : neg;
        IntVec *oc = &s->occ[lit];
        for (int e = 0; e < oc->size; ++e) {
            int en = oc->data[e];
            if (en > 0) {
                Clause *c = s->clauses.data[occ_clause_idx(en)];
                if (c == NULL || c->deleted) continue;
                if (c->size > 32) return 1; /* no eliminable */
                int sat = 0;
                for (int k = 0; k < c->size; ++k) {
                    if (solver_lit_value(s, c->lits[k]) == 1) { sat = 1; break; }
                }
                if (sat) continue;
                intvec_push(dst, en);
            } else {
                Lit other = occ_binary_other(en);
                if (solver_lit_value(s, lit) == 1) continue;
                if (solver_lit_value(s, other) == 1) continue;
                intvec_push(dst, en);
            }
        }
    }
    int np = s->simp_buf1.size;
    int nn = s->simp_buf2.size;
    if (np == 0 || nn == 0) return 1;
    long long cost = (long long)np * nn - np - nn;
    if (cost > 16) return 1;
    if (np + nn > 96) return 1;

    IntVec resolvents;
    IntVec sizes;
    IntVec tmp;
    intvec_init(&resolvents);
    intvec_init(&sizes);
    intvec_init(&tmp);
    int ok = 1;
    for (int pi = 0; pi < np && ok; ++pi) {
        int pe = s->simp_buf1.data[pi];
        for (int ni = 0; ni < nn && ok; ++ni) {
            int stamp = solver_next_stamp(s); /* sello fresco POR par */
            int ne = s->simp_buf2.data[ni];
            intvec_clear(&tmp);
            if (pe > 0) {
                Clause *c = s->clauses.data[occ_clause_idx(pe)];
                for (int k = 0; k < c->size; ++k) {
                    Lit l = c->lits[k];
                    if (l == pos) continue;
                    s->lit_mark[l] = stamp;
                    intvec_push(&tmp, l);
                }
            } else {
                Lit other = occ_binary_other(pe);
                s->lit_mark[other] = stamp;
                intvec_push(&tmp, other);
            }
            int taut = 0;
            if (ne > 0) {
                Clause *c = s->clauses.data[occ_clause_idx(ne)];
                for (int k = 0; k < c->size; ++k) {
                    Lit l = c->lits[k];
                    if (l == neg) continue;
                    if (s->lit_mark[l] == stamp) continue;
                    if (s->lit_mark[l ^ 1] == stamp) { taut = 1; break; }
                    s->lit_mark[l] = stamp;
                    intvec_push(&tmp, l);
                }
            } else {
                Lit other = occ_binary_other(ne);
                if (s->lit_mark[other] == stamp) {
                    /* duplicado: ignorar */
                } else if (s->lit_mark[other ^ 1] == stamp) {
                    taut = 1;
                } else {
                    s->lit_mark[other] = stamp;
                    intvec_push(&tmp, other);
                }
            }
            if (taut) continue;
            if (tmp.size == 0) { ok = 0; break; }
            if (tmp.size > 40) { ok = 0; break; }
            for (int a = 0; a < tmp.size; ++a) {
                for (int b = a + 1; b < tmp.size; ++b) {
                    if (tmp.data[b] < tmp.data[a]) {
                        int t = tmp.data[a];
                        tmp.data[a] = tmp.data[b];
                        tmp.data[b] = t;
                    }
                }
            }
            int dup = 0;
            for (int r0 = 0, off = 0; r0 < sizes.size; ++r0) {
                int sz = sizes.data[r0];
                if (sz != tmp.size) { off += sz; continue; }
                int same = 1;
                for (int k = 0; k < sz; ++k) {
                    if (resolvents.data[off + k] != tmp.data[k]) { same = 0; break; }
                }
                if (same) { dup = 1; break; }
                off += sz;
            }
            if (!dup) {
                for (int k = 0; k < tmp.size; ++k) intvec_push(&resolvents, tmp.data[k]);
                intvec_push(&sizes, tmp.size);
            }
        }
    }
    if (!ok) {
        intvec_free(&resolvents);
        intvec_free(&sizes);
        intvec_free(&tmp);
        return 0;
    }
    /* registro de extensión (copias de las cláusulas que contenían v) */
    if (s->extend_size == s->extend_cap) {
        int nc = s->extend_cap ? s->extend_cap * 2 : 64;
        s->extend = (ExtRec *)xrealloc(s->extend, (size_t)nc * sizeof(ExtRec));
        s->extend_cap = nc;
    }
    {
        ExtRec *rec = &s->extend[s->extend_size++];
        rec->var = v;
        rec->eq_lit = 0;
        rec->cls = NULL;
        rec->ncls = 0;
        int total = np + nn;
        rec->cls = (Clause **)xmalloc((size_t)(total > 0 ? total : 1) * sizeof(Clause *));
        int nc_ = 0;
        for (int pi = 0; pi < np; ++pi) {
            int pe = s->simp_buf1.data[pi];
            if (pe > 0) {
                Clause *c = s->clauses.data[occ_clause_idx(pe)];
                rec->cls[nc_++] = clause_new(c->lits, c->size, 0, 0);
            } else {
                Lit other = occ_binary_other(pe);
                Lit pair[2] = { pos, other };
                rec->cls[nc_++] = clause_new(pair, 2, 0, 0);
            }
        }
        for (int ni = 0; ni < nn; ++ni) {
            int ne = s->simp_buf2.data[ni];
            if (ne > 0) {
                Clause *c = s->clauses.data[occ_clause_idx(ne)];
                rec->cls[nc_++] = clause_new(c->lits, c->size, 0, 0);
            } else {
                Lit other = occ_binary_other(ne);
                Lit pair[2] = { neg, other };
                rec->cls[nc_++] = clause_new(pair, 2, 0, 0);
            }
        }
        rec->ncls = nc_;
    }
    /* eliminar las cláusulas originales (y binarias) */
    for (int pi = 0; pi < np; ++pi) {
        int pe = s->simp_buf1.data[pi];
        if (pe > 0) {
            s->clauses.data[occ_clause_idx(pe)]->deleted = 1;
        } else {
            solver_unwatch_binary(s, pos, occ_binary_other(pe));
        }
    }
    for (int ni = 0; ni < nn; ++ni) {
        int ne = s->simp_buf2.data[ni];
        if (ne > 0) {
            s->clauses.data[occ_clause_idx(ne)]->deleted = 1;
        } else {
            solver_unwatch_binary(s, neg, occ_binary_other(ne));
        }
    }
    /* añadir resolventes (y actualizar occ incrementalmente para que las
       eliminaciones siguientes vean las cláusulas nuevas) */
    int off = 0;
    for (int r0 = 0; r0 < sizes.size && ok; ++r0) {
        int sz = sizes.data[r0];
        if (!solver_add_clause(s, (Lit *)&resolvents.data[off], sz, 0, 0)) {
            ok = 0;
            break;
        }
        if (sz >= 3) {
            int newidx = s->clauses.size - 1;
            for (int k = 0; k < sz; ++k) {
                occ_push(s, resolvents.data[off + k], newidx + 1);
            }
        } else if (sz == 2) {
            Lit a = resolvents.data[off];
            Lit b = resolvents.data[off + 1];
            if (solver_add_binary_clause(s, a, b)) {
                occ_push(s, a, occ_enc_binary(b));
                occ_push(s, b, occ_enc_binary(a));
            }
        }
        off += sz;
    }
    intvec_free(&resolvents);
    intvec_free(&sizes);
    intvec_free(&tmp);
    if (!ok) return 0;
    if (s->order.pos[v] >= 0) heap_remove(&s->order, v);
    s->simp_eliminated++;
    return 1;
}

static int solver_bve_round(Solver *s, int max_elim, long long max_added) {
    int elim = 0;
    long long added = 0;
    for (int v = 0; v < s->nvars && elim < max_elim && added < max_added; ++v) {
        if (s->assigns[v] != 0) continue;
        if (s->order.pos[v] < 0) continue;
        int before = (int)s->simp_eliminated;
        int r = solver_bve_try_eliminate(s, v);
        if (!r) return 0;
        if ((int)s->simp_eliminated > before) {
            elim++;
            added += 64;
        }
    }
    /* purgar cláusulas aprendidas que mencionan variables eliminadas */
    if (elim > 0) {
        for (int i = 0; i < s->learnts.size; ++i) {
            Clause *c = s->learnts.data[i];
            if (c == NULL || c->deleted) continue;
            for (int k = 0; k < c->size; ++k) {
                if (s->order.pos[lit_var(c->lits[k])] < 0) {
                    c->deleted = 1;
                    break;
                }
            }
        }
    }
    return s->ok;
}

/* ── reconstrucción del modelo (extensión de variables eliminadas) ── */
static void solver_extend_model(Solver *s) {
    /* El valor de una variable eliminada se deduce del conjunto COMPLETO de
       cláusulas de extensión (todas las cláusulas borradas por BVE), no solo
       de las propias: las cláusulas compartidas entre variables eliminadas
       (p. ej. binarias de "a lo más uno") se borran con la primera
       eliminación y deben seguir restringiendo a las demás. */
    /* Los valores arbitrarios que HESS asigna a las variables eliminadas se
       descartan (nivel > assumption_level); los valores de suposiciones
       (nivel <= assumption_level) y de raíz se conservan como finales. */
    for (int i = 0; i < s->extend_size; ++i) {
        int v = s->extend[i].var;
        if (s->assigns[v] != 0 && s->levels[v] > s->assumption_level) {
            s->assigns[v] = 0;
            s->levels[v] = 0;
        }
    }
    for (int iter = 0; iter < 64; ++iter) {
        int changed = 0;
        int forward = (iter & 1);
        for (int t = 0; t < s->extend_size; ++t) {
            int i = forward ? t : s->extend_size - 1 - t;
            ExtRec *rec = &s->extend[i];
            if (rec->eq_lit != 0) {
                Lit l = dimacs_to_lit(rec->eq_lit);
                int val = solver_lit_value(s, l);
                if (val != 0 && s->assigns[rec->var] != val) {
                    s->assigns[rec->var] = (signed char)val;
                    changed = 1;
                }
                continue;
            }
            if (rec->ncls == 0) continue;
            int forced = 0;
            for (int t2 = 0; t2 < s->extend_size && forced == 0; ++t2) {
                ExtRec *r2 = &s->extend[t2];
                if (r2->ncls == 0) continue;
                for (int j = 0; j < r2->ncls && forced == 0; ++j) {
                    Clause *c = r2->cls[j];
                    int has_var = 0;
                    int var_lit = 0;
                    int all_false = 1;
                    for (int k = 0; k < c->size; ++k) {
                        Lit l = c->lits[k];
                        if (lit_var(l) == rec->var) {
                            has_var = 1;
                            var_lit = lit_sign(l) ? -1 : 1;
                            continue;
                        }
                        int val = solver_lit_value(s, l);
                        if (val == 1) { all_false = 0; break; }      /* satisfecha */
                        if (val == 0) { all_false = 0; break; }      /* indeterminada */
                    }
                    if (has_var && all_false && var_lit != 0) forced = var_lit;
                }
            }
            if (forced != 0) {
                s->assigns[rec->var] = (signed char)forced;
                changed = 1;
            }
        }
        if (!changed) break;
    }
    /* Pasada final iterada (directa): re-evalúa TODAS las variables (también
       las asignadas por HESS, cuyos valores arbitrarios pueden invalidarse
       con asignaciones posteriores); sin forzado, la variable conserva su
       valor, y si está sin asignar se rompe el ciclo con 1. */
    for (int iter = 0; iter < 64; ++iter) {
        int changed = 0;
        for (int t = 0; t < s->extend_size; ++t) {
            ExtRec *rec = &s->extend[t];
            if (rec->eq_lit != 0) {
                Lit l = dimacs_to_lit(rec->eq_lit);
                int val = solver_lit_value(s, l);
                if (val != 0 && s->assigns[rec->var] != val) {
                    s->assigns[rec->var] = (signed char)val;
                    changed = 1;
                }
                continue;
            }
            if (rec->ncls == 0) continue;
            int forced = 0;
            for (int t2 = 0; t2 < s->extend_size && forced == 0; ++t2) {
                ExtRec *r2 = &s->extend[t2];
                if (r2->ncls == 0) continue;
                for (int j = 0; j < r2->ncls && forced == 0; ++j) {
                    Clause *c = r2->cls[j];
                    int has_var = 0;
                    int var_lit = 0;
                    int all_false = 1;
                    for (int k = 0; k < c->size; ++k) {
                        Lit l = c->lits[k];
                        if (lit_var(l) == rec->var) {
                            has_var = 1;
                            var_lit = lit_sign(l) ? -1 : 1;
                            continue;
                        }
                        int val = solver_lit_value(s, l);
                        if (val == 1) { all_false = 0; break; }
                        if (val == 0) { all_false = 0; break; }
                    }
                    if (has_var && all_false && var_lit != 0) forced = var_lit;
                }
            }
            if (forced != 0) {
                if (s->assigns[rec->var] != forced) {
                    s->assigns[rec->var] = (signed char)forced;
                    changed = 1;
                }
            } else if (s->assigns[rec->var] == 0) {
                s->assigns[rec->var] = 1;
                changed = 1;
            }
        }
        if (!changed) break;
    }
    /* Re-evaluación final de pares equivalentes (inversa: los representantes
       eliminados después ya están asignados por la pasada anterior). */
    for (int i = s->extend_size - 1; i >= 0; --i) {
        ExtRec *rec = &s->extend[i];
        if (rec->eq_lit != 0) {
            Lit l = dimacs_to_lit(rec->eq_lit);
            int val = solver_lit_value(s, l);
            if (val != 0) s->assigns[rec->var] = (signed char)val;
        }
    }
    /* Reparación final: si alguna cláusula de extensión quedó falsificada,
       intenta voltear una variable eliminada que la satisfaga sin romper el
       resto (iterado hasta estabilidad; los voltos son acotados). */
    for (int iter = 0; iter < 64; ++iter) {
        int fixed = 0;
        for (int i = 0; i < s->extend_size && !fixed; ++i) {
            ExtRec *rec = &s->extend[i];
            if (rec->ncls == 0) continue;
            for (int j = 0; j < rec->ncls && !fixed; ++j) {
                Clause *c = rec->cls[j];
                int sat = 0;
                for (int k = 0; k < c->size; ++k) {
                    if (solver_lit_value(s, c->lits[k]) == 1) { sat = 1; break; }
                }
                if (sat) continue;
                /* cláusula violada: buscar una variable eliminada en ella cuyo
                   volteo satisfaga TODAS las cláusulas de extensión */
                for (int k = 0; k < c->size && !fixed; ++k) {
                    int v = lit_var(c->lits[k]);
                    int is_elim = 0;
                    for (int t2 = 0; t2 < s->extend_size; ++t2) {
                        if (s->extend[t2].var == v) { is_elim = 1; break; }
                    }
                    if (!is_elim) continue;
                    int old = s->assigns[v];
                    s->assigns[v] = (signed char)(-old);
                    int ok_all = 1;
                    for (int t2 = 0; t2 < s->extend_size && ok_all; ++t2) {
                        ExtRec *r2 = &s->extend[t2];
                        if (r2->ncls == 0) continue;
                        for (int j2 = 0; j2 < r2->ncls; ++j2) {
                            Clause *c2 = r2->cls[j2];
                            int sat2 = 0;
                            for (int k2 = 0; k2 < c2->size; ++k2) {
                                if (solver_lit_value(s, c2->lits[k2]) == 1) { sat2 = 1; break; }
                            }
                            if (!sat2) { ok_all = 0; break; }
                        }
                    }
                    if (ok_all) {
                        fixed = 1;
                    } else {
                        s->assigns[v] = (signed char)old;
                    }
                }
            }
        }
        if (!fixed) break;
    }
}

/* ── simplificación raíz completa ── */
static int solver_simplify_root(Solver *s) {
    if (!s->simplify_enable || s->proof != NULL) {
        s->simplified = 1;
        return s->ok;
    }
    ClauseRef conf = solver_propagate(s);
    if (!clauseref_is_none(conf) && solver_decision_level(s) == 0) {
        s->simplified = 1;
        return 0;
    }
    if (!s->ok) {
        s->simplified = 1;
        return 0;
    }
    int rounds = 4;
    for (int r = 0; r < rounds && s->ok; ++r) {
        long long e0 = s->simp_eliminated;
        long long ss0 = s->simp_subsumed + s->simp_strengthened;
        long long eq0 = s->simp_equivs;
        long long pr0 = s->simp_probe_units;
        int trail0 = s->trail_size;

        solver_occ_build(s);
        if (s->probe_enable) {
            if (!solver_probe_literals(s, 2000, 4000000LL)) {
                solver_occ_free(s);
                s->simplified = 1;
                return 0;
            }
        }
        if (!solver_subsume_strengthen(s, 2000000LL)) {
            solver_occ_free(s);
            s->simplified = 1;
            return 0;
        }
        solver_gc_all(s);
        solver_occ_build(s);
        if (!solver_substitute_equivalents(s)) {
            solver_occ_free(s);
            s->simplified = 1;
            return 0;
        }
        solver_occ_build(s);
        if (s->bve_enable && !solver_bve_round(s, 3000, 4000000LL)) {
            solver_occ_free(s);
            s->simplified = 1;
            return 0;
        }
        solver_gc_all(s);
        solver_occ_free(s);

        /* convergencia */
        if (s->simp_eliminated == e0 &&
            s->simp_subsumed + s->simp_strengthened == ss0 &&
            s->simp_equivs == eq0 &&
            s->simp_probe_units == pr0 &&
            s->trail_size == trail0) {
            break;
        }
    }
    s->simplified = 1;
    return s->ok;
}

/* ── inprocessing ligero durante la búsqueda (estilo CDCL moderno) ── */
static int solver_inprocess(Solver *s) {
    if (!s->inprocess_enable || s->proof != NULL) return s->ok;
    solver_occ_build(s);
    if (!solver_subsume_strengthen(s, 1000000LL)) {
        solver_occ_free(s);
        return 0;
    }
    solver_gc_all(s);
    solver_occ_build(s);
    if (!solver_bve_round(s, 500, 1000000LL)) {
        solver_occ_free(s);
        return 0;
    }
    solver_gc_all(s);
    solver_occ_free(s);
    return s->ok;
}

/* ── minimización recursiva de cláusulas aprendidas (iterativa) ── */
static int solver_lit_redundant(Solver *s, Lit p) {
    int top = s->mini_poison_stack.size;
    intvec_clear(&s->simp_buf1);
    intvec_push(&s->simp_buf1, p);
    while (s->simp_buf1.size > 0) {
        Lit q = s->simp_buf1.data[s->simp_buf1.size - 1];
        s->simp_buf1.size--;
        int v = lit_var(q);
        Reason rr = s->reasons[v];
        if (reason_is_none(rr)) goto fail;
        if (!reason_is_binary(rr)) {
            Clause *c = reason_clause_ptr(rr);
            if (c->deleted) goto fail;
            for (int i = 0; i < c->size; ++i) {
                Lit r = c->lits[i];
                if (solver_lit_value(s, r) != -1) continue;
                int rv = lit_var(r);
                if (s->levels[rv] == 0) continue;
                if (s->mini_removable[rv]) continue;
                if (!reason_is_none(s->reasons[rv])) {
                    s->mini_removable[rv] = 1;
                    intvec_push(&s->mini_poison_stack, rv);
                    intvec_push(&s->simp_buf1, r);
                } else {
                    goto fail;
                }
            }
        } else {
            Lit r = reason_binary_other(rr);
            int rv = lit_var(r);
            if (s->levels[rv] == 0) continue;
            if (s->mini_removable[rv]) continue;
            if (!reason_is_none(s->reasons[rv])) {
                s->mini_removable[rv] = 1;
                intvec_push(&s->mini_poison_stack, rv);
                intvec_push(&s->simp_buf1, r);
            } else {
                goto fail;
            }
        }
    }
    return 1;
fail:
    for (int i = top; i < s->mini_poison_stack.size; ++i) {
        s->mini_removable[s->mini_poison_stack.data[i]] = 0;
    }
    s->mini_poison_stack.size = top;
    return 0;
}

static void solver_minimize_learnt(Solver *s, IntVec *learnt) {
    int top = s->mini_poison_stack.size;
    int out = 1;
    for (int i = 1; i < learnt->size; ++i) {
        Lit q = learnt->data[i];
        int v = lit_var(q);
        if (s->levels[v] == 0) continue;
        if (s->mini_removable[v]) continue;
        if (solver_lit_redundant(s, q)) continue;
        learnt->data[out++] = q;
    }
    learnt->size = out;
    for (int i = top; i < s->mini_poison_stack.size; ++i) {
        s->mini_removable[s->mini_poison_stack.data[i]] = 0;
    }
    s->mini_poison_stack.size = top;
}

/* quita UNA entrada de watch (no binaria) de la lista de ¬old0 */
static void solver_unwatch_clause_pos(Solver *s, Clause *c, Lit old0) {
    WatchVec *ws = &s->watches[lit_neg(old0)];
    for (int i = 0; i < ws->size; ++i) {
        Watch w = ws->data[i];
        if (!watch_is_binary(w) && watch_clause_ptr(w) == c) {
            ws->data[i] = ws->data[--ws->size];
            return;
        }
    }
}

/* mueve el literal `forced` a lits[0] manteniendo la invariante de watches
 * (necesario antes de usar la cláusula como razón: el análisis asume que
 * lits[0] es el literal implicado). */
static void solver_make_reason_first(Solver *s, Clause *c, Lit forced) {
    if (c->lits[0] == forced) return;
    int fi = -1;
    for (int i = 1; i < c->size; ++i) {
        if (c->lits[i] == forced) { fi = i; break; }
    }
    if (fi < 0) return;
    Lit old0 = c->lits[0];
    if (fi == 1) {
        c->lits[0] = forced;
        c->lits[1] = old0;
    } else {
        solver_unwatch_clause_pos(s, c, old0);
        c->lits[0] = forced;
        c->lits[fi] = old0;
        watchvec_push(&s->watches[lit_neg(forced)], watch_from_clause(c));
    }
}

/* ── subsumción eager de las últimas cláusulas aprendidas ── */
static void solver_eager_subsume(Solver *s, Clause *c) {
    int stamp = solver_next_stamp(s);
    for (int k = 0; k < c->size; ++k) s->lit_mark[c->lits[k]] = stamp;
    for (int i = 0; i < 4; ++i) {
        Clause *d = s->last_learnt[i];
        if (d == NULL || d->deleted || d->locked || d->size < c->size || d->size <= 2) continue;
        int sub = 1;
        for (int k = 0; k < d->size; ++k) {
            if (s->lit_mark[d->lits[k]] != stamp) { sub = 0; break; }
        }
        if (sub) {
            d->deleted = 1;
            s->last_learnt[i] = NULL;
        }
    }
    for (int i = 3; i > 0; --i) s->last_learnt[i] = s->last_learnt[i - 1];
    s->last_learnt[0] = c;
}

static int solver_solve(Solver *s) {
    IntVec learnt;
    ClauseRef confl;
    int backtrack_floor = s->assumption_level;
    int allow_hess = s->hess_enable;
    int allow_ct_probe = (backtrack_floor == 0) ? s->ct_probe_restarts : 0;
    intvec_init(&learnt);

    if (s->nvars > 200000 || ((long long)s->clauses.size + s->binary_clauses) > 1200000LL) {
        allow_hess = 0;
        allow_ct_probe = 0;
        s->use_mab = 0;
        solver_switch_heuristic(s, 0);
    }

    /* simplificación raíz (una sola vez, sin suposiciones activas) */
    if (!s->simplified && backtrack_floor == 0 && solver_decision_level(s) == 0) {
        if (!solver_simplify_root(s)) {
            s->ok = 0;
            if (s->proof) fputs("0\n", s->proof);
            intvec_free(&learnt);
            return 20;
        }
        if (!s->ok) {
            intvec_free(&learnt);
            return 20;
        }
        s->next_inprocess = s->conflicts + 2000;
    }

    confl = solver_propagate(s);
    solver_update_chb_after_propagate(s, !clauseref_is_none(confl));
    if (!clauseref_is_none(confl) && solver_decision_level(s) == 0) {
        if (s->proof) fputs("0\n", s->proof);
        intvec_free(&learnt);
        return 20;
    }

    if (clauseref_is_none(confl) &&
        allow_hess &&
        s->binary_clauses == 0 &&
        s->nvars > 0 &&
        (s->clauses.size > 0 || s->orig_unit_lits.size > 0 || s->orig_empty_clauses > 0)) {
        unsigned char *hmodel = (unsigned char *)xmalloc((size_t)s->nvars * sizeof(unsigned char));
        int best_unsat = 0;
        s->hess_calls++;
        int sat_h = solver_hess_exact_search(s, hmodel, &best_unsat);
        (void)best_unsat;
        if (sat_h) {
            if (!solver_apply_model01(s, hmodel)) {
                free(hmodel);
                intvec_free(&learnt);
                return 20;
            }
            s->hess_sat_hits++;
            solver_extend_model(s);
            free(hmodel);
            intvec_free(&learnt);
            return 10;
        }
        memcpy(s->phases, hmodel, (size_t)s->nvars * sizeof(unsigned char));
        free(hmodel);
    }

    while (s->ok) {
        if (solver_poll_external_stop(s)) {
            intvec_free(&learnt);
            return 0;
        }
        if (!clauseref_is_none(confl)) {
            s->conflicts++;
            s->mab_epoch_conflicts += 1.0;
            if (solver_decision_level(s) == 0) {
                if (s->proof) fputs("0\n", s->proof);
                intvec_free(&learnt);
                return 20;
            }

            int dl = solver_decision_level(s);
            int bt = 0;
            int reused = 0;

            /* reutilización del conflicto: exactamente un literal
               en el nivel máximo → el conflicto mismo es la cláusula assertiva */
            if (confl.size >= 2 && confl.clause != NULL) {
                Clause *cc = confl.clause;
                int maxlvl = -1;
                int cnt = 0;
                Lit forced = 0;
                int jump = 0;
                for (int i = 0; i < cc->size; ++i) {
                    int lv = s->levels[lit_var(cc->lits[i])];
                    if (lv > maxlvl) {
                        jump = maxlvl;
                        maxlvl = lv;
                        cnt = 1;
                        forced = cc->lits[i];
                    } else if (lv == maxlvl) {
                        cnt++;
                    } else if (lv > jump) {
                        jump = lv;
                    }
                }
                if (cnt == 1 && maxlvl == dl && jump >= backtrack_floor) {
                    int new_level = jump;
                    if (s->chrono_enable && (dl - 1) - jump > 100) {
                        new_level = dl - 1;
                        s->chrono_conflicts++;
                    }
                    solver_cancel_until(s, new_level);
                    solver_clause_bump(s, cc);
                    solver_make_reason_first(s, cc, forced);
                    if (cc->learnt) cc->locked = 1;
                    if (!solver_enqueue(s, forced, reason_from_clause(cc))) {
                        intvec_free(&learnt);
                        return 20;
                    }
                    reused = 1;
                }
            } else if (confl.size == 2) {
                Lit a = confl.lits[0];
                Lit b = confl.lits[1];
                int la = s->levels[lit_var(a)];
                int lb = s->levels[lit_var(b)];
                if (la != lb && (la == dl || lb == dl)) {
                    Lit forced = (la > lb) ? a : b;
                    Lit other = (la > lb) ? b : a;
                    int jump = la < lb ? la : lb;
                    if (jump >= backtrack_floor) {
                        int new_level = jump;
                        if (s->chrono_enable && (dl - 1) - jump > 100) {
                            new_level = dl - 1;
                            s->chrono_conflicts++;
                        }
                        solver_cancel_until(s, new_level);
                        if (!solver_enqueue(s, forced, reason_from_binary(other))) {
                            intvec_free(&learnt);
                            return 20;
                        }
                        reused = 1;
                    }
                }
            }

            if (!reused) {
                solver_analyze(s, confl, &learnt, &bt);

                uint32_t lbd = solver_compute_lbd(s, &learnt);

                if (bt < backtrack_floor) {
                    solver_cancel_until(s, 0);
                    if (!solver_record_learnt_clause(s, &learnt, lbd, 0)) {
                        if (s->proof) fputs("0\n", s->proof);
                    }
                    intvec_free(&learnt);
                    return 20;
                }

                int new_level = bt;
                if (s->chrono_enable && (dl - 1) - bt > 100) {
                    new_level = dl - 1;
                    s->chrono_conflicts++;
                }
                solver_cancel_until(s, new_level);
                if (!solver_record_learnt_clause(s, &learnt, lbd, 1)) {
                    if (s->proof) fputs("0\n", s->proof);
                    intvec_free(&learnt);
                    return 20;
                }

                solver_var_decay(s);
                solver_clause_decay(s);

                {
                    double f = (double)(int)lbd;
                    s->fast_glue_ema = s->fast_glue_beta * s->fast_glue_ema + s->fast_glue_alpha * f;
                    s->slow_glue_ema = s->slow_glue_beta * s->slow_glue_ema + s->slow_glue_alpha * f;
                }
            }

            if (s->conflicts >= s->next_reduce) {
                solver_reduce_db(s);
                s->next_reduce = s->conflicts + s->reduce_base + s->learnts.size / 2;
            }

            {
                int do_restart = 0;
                if (s->conflicts >= s->next_restart) {
                    do_restart = 1;
                }
                if (!do_restart &&
                    s->conflicts >= s->next_ema_restart &&
                    s->slow_glue_ema > 0.0 &&
                    s->fast_glue_ema > s->restart_margin * s->slow_glue_ema) {
                    do_restart = 1;
                }

                if (do_restart) {
                    int reuse_level = 0;
                    {
                        int next_var = -1;
                        for (int i = 0; i < s->order.size; ++i) {
                            int v = s->order.heap[i];
                            if (v >= 0 && v < s->nvars && s->assigns[v] == 0) {
                                next_var = v;
                                break;
                            }
                        }
                        if (next_var >= 0) {
                            double limit = s->activity[next_var];
                            for (int dl = 1; dl <= solver_decision_level(s); ++dl) {
                                int v = lit_var(s->trail[s->trail_lim.data[dl - 1]]);
                                if (s->activity[v] <= limit) break;
                                reuse_level = dl;
                            }
                        }
                    }
                    s->restarts++;
                    solver_cancel_until(s, reuse_level > 0 ? reuse_level : backtrack_floor);
                    solver_restart_mab(s);
                    solver_covertrace_escape_phases(s);
                    if (allow_ct_probe > 0 &&
                        (s->restarts % allow_ct_probe) == 0) {
                        solver_covertrace_probe(s);
                    }
                    if (s->rephase_enable && s->conflicts >= s->next_rephase) {
                        for (int v = 0; v < s->nvars; ++v) s->phases[v] ^= 1u;
                        s->next_rephase = s->conflicts + 1000;
                        s->rephase_count++;
                    }
                    int lub = luby(2, s->restart_count++);
                    s->next_restart = s->conflicts + (long long)(100 * lub);
                    s->next_ema_restart = s->conflicts + s->ema_restart_interval;
                }

                /* inprocessing periódico en la raíz (estilo CDCL moderno) */
                if (s->inprocess_enable &&
                    backtrack_floor == 0 &&
                    s->next_inprocess > 0 &&
                    s->conflicts >= s->next_inprocess) {
                    solver_cancel_until(s, 0);
                    confl = solver_propagate(s);
                    if (!clauseref_is_none(confl) && solver_decision_level(s) == 0) {
                        if (s->proof) fputs("0\n", s->proof);
                        intvec_free(&learnt);
                        return 20;
                    }
                    if (!solver_inprocess(s)) {
                        s->ok = 0;
                        if (s->proof) fputs("0\n", s->proof);
                        intvec_free(&learnt);
                        return 20;
                    }
                    s->inprocess_count++;
                    s->next_inprocess = s->conflicts + 2000 + 1000 * s->inprocess_count;
                    confl = solver_propagate(s);
                    if (!clauseref_is_none(confl) && solver_decision_level(s) == 0) {
                        if (s->proof) fputs("0\n", s->proof);
                        intvec_free(&learnt);
                        return 20;
                    }
                }
            }
        } else {
            Lit p = solver_pick_branch_lit(s);
            if (p < 0) {
                int sat_ok = solver_verify_model(s);
                if (sat_ok) solver_extend_model(s);
                intvec_free(&learnt);
                return sat_ok ? 10 : 0;
            }
            intvec_push(&s->trail_lim, s->trail_size);
            s->decisions++;
            s->mab_epoch_decisions += 1.0;
            if (!solver_enqueue(s, p, reason_none())) {
                intvec_free(&learnt);
                return 20;
            }
        }

        confl = solver_propagate(s);
        if (solver_poll_external_stop(s)) {
            intvec_free(&learnt);
            return 0;
        }
        solver_update_chb_after_propagate(s, !clauseref_is_none(confl));
    }

    intvec_free(&learnt);
    if (s->external_stop_hit) {
        return 0;
    }
    return s->ok ? 0 : 20;
}

static void fi_init(FastInput *in, FILE *fp) {
    in->fp = fp;
    in->cap = 1u << 20;
    in->buf = (unsigned char *)xmalloc(in->cap);
    in->pos = 0;
    in->len = 0;
    in->pushed = -1;
}

static void fi_free(FastInput *in) {
    free(in->buf);
    in->buf = NULL;
    in->cap = in->pos = in->len = 0;
    in->pushed = -1;
}

static int fi_getc(FastInput *in) {
    if (in->pushed >= 0) {
        int c = in->pushed;
        in->pushed = -1;
        return c;
    }
    if (in->pos >= in->len) {
        in->len = fread(in->buf, 1, in->cap, in->fp);
        in->pos = 0;
        if (in->len == 0) return EOF;
    }
    return in->buf[in->pos++];
}

static void fi_ungetc(FastInput *in, int c) {
    in->pushed = c;
}

static void fi_skip_line(FastInput *in) {
    int c;
    while ((c = fi_getc(in)) != EOF && c != '\n') {
    }
}

static int fi_read_word(FastInput *in, char *out, int cap) {
    int c;
    do {
        c = fi_getc(in);
        if (c == EOF) return 0;
    } while (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v');

    int n = 0;
    while (c != EOF && !(c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v')) {
        if (n + 1 < cap) out[n++] = (char)c;
        c = fi_getc(in);
    }
    if (c != EOF) fi_ungetc(in, c);
    out[n] = '\0';
    return 1;
}

static int fi_read_int_token(FastInput *in, int *out) {
    int c;
    do {
        c = fi_getc(in);
        if (c == EOF) return 0;
    } while (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v');

    int sign = 1;
    if (c == '-') {
        sign = -1;
        c = fi_getc(in);
    }
    if (c < '0' || c > '9') return 0;

    int v = c - '0';
    for (;;) {
        c = fi_getc(in);
        if (c < '0' || c > '9') break;
        v = v * 10 + (c - '0');
    }
    if (c != EOF) fi_ungetc(in, c);
    *out = sign * v;
    return 1;
}

static int parse_dimacs(const char *path,
                        Solver *s,
                        long long *parsed_clauses,
                        int need_chb,
                        int need_hess) {
    *parsed_clauses = 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "c ERROR: cannot open '%s': %s\n", path, strerror(errno));
        return 0;
    }

    FastInput in;
    fi_init(&in, fp);

    int header_found = 0;
    int nvars = 0;

    IntVec clause;
    intvec_init(&clause);

    int bol = 1;

    while (1) {
        int c = fi_getc(&in);
        if (c == EOF) break;

        if (c == '\n') {
            bol = 1;
            continue;
        }
        if (c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v') {
            continue;
        }

        if (bol && c == 'c') {
            fi_skip_line(&in);
            bol = 1;
            continue;
        }

        if (bol && c == 'p') {
            char word[16];
            if (!fi_read_word(&in, word, (int)sizeof(word))) {
                fprintf(stderr, "c ERROR: malformed header in %s\n", path);
                goto fail;
            }
            if (strcmp(word, "cnf") != 0) {
                fprintf(stderr, "c ERROR: only DIMACS 'p cnf' supported (got '%s')\n", word);
                goto fail;
            }
            int m = 0;
            if (!fi_read_int_token(&in, &nvars) || !fi_read_int_token(&in, &m) || nvars <= 0) {
                fprintf(stderr, "c ERROR: malformed 'p cnf' line in %s\n", path);
                goto fail;
            }
            solver_init(s, nvars, need_chb, need_hess);
            header_found = 1;
            fi_skip_line(&in);
            bol = 1;
            continue;
        }

        bol = 0;
        if (!header_found) {
            fprintf(stderr, "c ERROR: clause data before header in %s\n", path);
            goto fail;
        }

        int sign = 1;
        if (c == '-') {
            sign = -1;
            c = fi_getc(&in);
        }
        if (c < '0' || c > '9') {
            fprintf(stderr, "c ERROR: invalid token while parsing %s\n", path);
            goto fail;
        }

        int lit = c - '0';
        for (;;) {
            c = fi_getc(&in);
            if (c < '0' || c > '9') break;
            lit = lit * 10 + (c - '0');
        }
        if (c != EOF) fi_ungetc(&in, c);

        lit *= sign;

        if (lit == 0) {
            int contradiction = 0;
            solver_track_original_clause(s, (Lit *)clause.data, clause.size);
            if (!solver_commit_input_clause(s, &clause, &contradiction)) {
                goto fail;
            }
            clause.size = 0;
            (*parsed_clauses)++;
            if (contradiction) break;
        } else {
            int v = (lit > 0) ? lit : -lit;
            if (v < 1 || v > nvars) {
                fprintf(stderr, "c ERROR: literal %d out of range 1..%d in %s\n", lit, nvars, path);
                goto fail;
            }
            intvec_push(&clause, dimacs_to_lit(lit));
        }
    }

    if (!header_found) {
        fprintf(stderr, "c ERROR: missing 'p cnf' header in %s\n", path);
        goto fail;
    }
    if (clause.size != 0) {
        fprintf(stderr, "c ERROR: unterminated clause at EOF in %s\n", path);
        goto fail;
    }

    intvec_free(&clause);
    fi_free(&in);
    fclose(fp);
    return 1;

fail:
    intvec_free(&clause);
    fi_free(&in);
    fclose(fp);
    return 0;
}

static void print_model(const Solver *s) {
    int col = 0;
    fputs("v ", stdout);
    for (int v = 0; v < s->nvars; ++v) {
        int a = s->assigns[v];
        int lit = (a >= 0) ? (v + 1) : -(v + 1);
        printf("%d ", lit);
        if (++col == 16) {
            fputs("\nv ", stdout);
            col = 0;
        }
    }
    fputs("0\n", stdout);
}

static double now_sec(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void solver_seed_rng_from_problem(Solver *s, long long parsed_clauses) {
    uint64_t seed = 0xA0761D6478BD642FULL;
    seed ^= (uint64_t)s->nvars * 0xE7037ED1A0B428DBULL;
    seed ^= (uint64_t)(parsed_clauses + 1) * 0x8EBC6AF09C88C6E3ULL;
    int lim = s->clauses.size < 2048 ? s->clauses.size : 2048;
    for (int i = 0; i < lim; ++i) {
        Clause *c = s->clauses.data[i];
        if (!c || c->size == 0) continue;
        uint64_t x = (uint64_t)(c->size + 1) * 0xD6E8FEB86659FD93ULL;
        x ^= (uint64_t)(c->lits[0] + 1) * 0x94D049BB133111EBULL;
        seed ^= x;
        seed = rng_next_u64(&seed);
    }
    if (seed == 0) seed = 1;
    s->rng_state = seed;
}

static void solver_apply_runtime_options(Solver *s, const SlimeSatOptions *opt) {
    SlimeSatOptions cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (opt != NULL) {
        cfg = *opt;
    } else {
        cfg.heuristic_mode = 0;
        cfg.use_mab = 0;
        cfg.mabc = 4.0;
        cfg.use_hess = 0;
        cfg.use_ct = 1;
        cfg.ct_lbd_max = 6;
        cfg.ct_maxlen = 12;
        cfg.ct_max_cubes = 40000;
        cfg.ct_buddy_merge = 0;
        cfg.ct_escape_rounds = 4;
        cfg.ct_probe_restarts = 4;
        cfg.use_simplify = 1;
        cfg.use_bve = 0;
        cfg.use_chrono = 1;
        cfg.use_inprocess = 0;
        cfg.use_probe = 1;
    }

    s->heuristic = (cfg.heuristic_mode == 1) ? 1 : 0;
    s->use_mab = cfg.use_mab ? 1 : 0;
    s->mabc = (cfg.mabc >= 0.0) ? cfg.mabc : 0.0;
    s->mab_reward[0] = s->mab_reward[1] = 0.0;
    s->mab_select[0] = s->mab_select[1] = 0;
    s->mab_select[s->heuristic] = 1;
    s->mab_epoch_decisions = 0.0;
    s->mab_epoch_conflicts = 0.0;
    s->order.activity = (s->heuristic == 1) ? s->chb_activity : s->activity;

    s->ct_enable = cfg.use_ct ? 1 : 0;
    s->ct_lbd_max = (cfg.ct_lbd_max >= 1) ? cfg.ct_lbd_max : 1;
    s->ct_maxlen = (cfg.ct_maxlen >= 2) ? cfg.ct_maxlen : 2;
    s->ct_cubes.max_keep = (cfg.ct_max_cubes >= 1) ? cfg.ct_max_cubes : 1;
    s->ct_buddy_merge = cfg.ct_buddy_merge ? 1 : 0;
    s->ct_escape_rounds = (cfg.ct_escape_rounds >= 0) ? cfg.ct_escape_rounds : 0;
    s->ct_probe_restarts = (cfg.ct_probe_restarts >= 0) ? cfg.ct_probe_restarts : 0;

    s->simplify_enable = cfg.use_simplify ? 1 : 0;
    s->bve_enable = cfg.use_bve ? 1 : 0;
    s->chrono_enable = cfg.use_chrono ? 1 : 0;
    s->inprocess_enable = cfg.use_inprocess ? 1 : 0;
    s->probe_enable = cfg.use_probe ? 1 : 0;
    if (!s->simplify_enable) s->inprocess_enable = 0;
    if (!s->inprocess_enable && !s->simplify_enable) s->simplified = 1;

    s->hess_enable = cfg.use_hess ? 1 : 0;
    heap_rebuild(&s->order);
}

static size_t solver_memory_bytes(const Solver *s) {
    size_t bytes = 0;
    bytes += (size_t)s->nvars * sizeof(signed char);
    bytes += (size_t)s->nvars * sizeof(int);
    bytes += (size_t)s->nvars * sizeof(Reason);
    bytes += (size_t)s->nvars * sizeof(unsigned char);
    bytes += (size_t)s->nvars * sizeof(unsigned char);
    bytes += (size_t)s->nvars * sizeof(double);
    if (s->chb_activity != NULL) bytes += (size_t)s->nvars * sizeof(double);
    if (s->chb_last_conflict != NULL) bytes += (size_t)s->nvars * sizeof(long long);
    if (s->hess_unit_freeze != NULL) bytes += (size_t)s->nvars * sizeof(unsigned char);
    bytes += (size_t)s->order.cap * sizeof(int);
    bytes += (size_t)s->order.cap * sizeof(int);
    bytes += (size_t)s->trail_cap * sizeof(Lit);
    bytes += (size_t)s->trail_lim.cap * sizeof(int);
    bytes += (size_t)s->analyze_stack.cap * sizeof(int);
    bytes += (size_t)s->lbd_marks.cap * sizeof(int);
    bytes += (size_t)s->clauses.cap * sizeof(Clause *);
    bytes += (size_t)s->learnts.cap * sizeof(Clause *);
    if (s->hess_unit_freeze != NULL) bytes += (size_t)s->orig_unit_lits.cap * sizeof(int);

    for (int i = 0; i < s->clauses.size; ++i) {
        Clause *c = s->clauses.data[i];
        if (!c) continue;
        bytes += sizeof(Clause) + (size_t)c->size * sizeof(Lit);
    }
    for (int i = 0; i < s->learnts.size; ++i) {
        Clause *c = s->learnts.data[i];
        if (!c || c->deleted) continue;
        bytes += sizeof(Clause) + (size_t)c->size * sizeof(Lit);
    }
    for (int i = 0; i < 2 * s->nvars; ++i) {
        bytes += (size_t)s->watches[i].cap * sizeof(Watch);
    }
    bytes += (size_t)s->ct_cubes.cap * sizeof(CTCube);
    for (int i = 0; i < s->ct_cubes.size; ++i) {
        bytes += (size_t)s->ct_cubes.data[i].size * sizeof(Lit);
    }
    return bytes;
}

int slime_entry(int argc, char **argv);
#if defined(BASILISK_NO_MAIN)
int basilisk_entry(int argc, char **argv);
#endif

int slime_sat_solve_clauses(int nvars,
                            int nclauses,
                            const int *const *clauses,
                            const int *sizes,
                            const int *assumptions,
                            int num_assumptions,
                            const SlimeSatOptions *opt,
                            SlimeSatStats *stats,
                            unsigned char *model01);

SlimeSatHandle *slime_sat_handle_create(int nvars,
                                        int nclauses,
                                        const int *const *clauses,
                                        const int *sizes,
                                        const SlimeSatOptions *opt);
void slime_sat_handle_reconfigure(SlimeSatHandle *handle, const SlimeSatOptions *opt);
int slime_sat_handle_solve(SlimeSatHandle *handle,
                           const int *assumptions,
                           int num_assumptions,
                           SlimeSatStats *stats,
                           unsigned char *model01);
void slime_sat_handle_destroy(SlimeSatHandle *handle);

enum {
    SLIME_MODE_SOLVE = 0,
    SLIME_MODE_COUNT = 1,
    SLIME_MODE_PROJECT = 2
};

static int parse_slime_mode(const char *text, int *out_mode) {
    if (strcmp(text, "solve") == 0) {
        *out_mode = SLIME_MODE_SOLVE;
        return 1;
    }
    if (strcmp(text, "count") == 0) {
        *out_mode = SLIME_MODE_COUNT;
        return 1;
    }
    if (strcmp(text, "project") == 0) {
        *out_mode = SLIME_MODE_PROJECT;
        return 1;
    }
    return 0;
}

struct SlimeSatHandle {
    Solver solver;
    SlimeSatOptions opt;
    int base_unsat;
};

typedef struct {
    int nvars;
    int **clauses;
    int *sizes;
    int nclauses;
    int cap_clauses;
    long long parsed_clauses;
} SlimeCnfProblem;

typedef struct {
    int *lits;
    int size;
} SlimeCube;

typedef struct {
    SlimeCube *data;
    int size;
    int cap;
} SlimeCubeVec;

static void slime_problem_init(SlimeCnfProblem *p) {
    memset(p, 0, sizeof(*p));
}

static void slime_problem_free(SlimeCnfProblem *p) {
    if (p == NULL) return;
    for (int i = 0; i < p->nclauses; ++i) free(p->clauses[i]);
    free(p->clauses);
    free(p->sizes);
    memset(p, 0, sizeof(*p));
}

static int slime_problem_reserve(SlimeCnfProblem *p, int need) {
    int nc;
    int **new_clauses;
    int *new_sizes;
    if (need <= p->cap_clauses) return 1;
    nc = (p->cap_clauses > 0) ? p->cap_clauses : 8;
    while (nc < need) {
        if (nc > INT_MAX / 2) return 0;
        nc *= 2;
    }
    new_clauses = (int **)realloc(p->clauses, (size_t)nc * sizeof(int *));
    if (new_clauses == NULL) return 0;
    new_sizes = (int *)realloc(p->sizes, (size_t)nc * sizeof(int));
    if (new_sizes == NULL) {
        p->clauses = new_clauses;
        return 0;
    }
    p->clauses = new_clauses;
    p->sizes = new_sizes;
    p->cap_clauses = nc;
    return 1;
}

static int slime_problem_add_clause(SlimeCnfProblem *p, const IntVec *clause) {
    int *copy = NULL;
    if (!slime_problem_reserve(p, p->nclauses + 1)) return 0;
    if (clause->size > 0) {
        copy = (int *)malloc((size_t)clause->size * sizeof(int));
        if (copy == NULL) return 0;
        memcpy(copy, clause->data, (size_t)clause->size * sizeof(int));
    }
    p->clauses[p->nclauses] = copy;
    p->sizes[p->nclauses] = clause->size;
    p->nclauses++;
    p->parsed_clauses++;
    return 1;
}

static int slime_problem_parse_dimacs(const char *path, SlimeCnfProblem *p) {
    FILE *fp = NULL;
    FastInput in;
    IntVec clause;
    int header_found = 0;
    int nvars = 0;
    int bol = 1;

    slime_problem_init(p);
    fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "c ERROR: cannot open '%s': %s\n", path, strerror(errno));
        return 0;
    }
    fi_init(&in, fp);
    intvec_init(&clause);

    while (1) {
        int c = fi_getc(&in);
        if (c == EOF) break;
        if (c == '\n') {
            bol = 1;
            continue;
        }
        if (c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v') {
            continue;
        }
        if (bol && c == 'c') {
            fi_skip_line(&in);
            bol = 1;
            continue;
        }
        if (bol && c == 'p') {
            char word[16];
            int declared = 0;
            if (!fi_read_word(&in, word, (int)sizeof(word))) {
                fprintf(stderr, "c ERROR: malformed header in %s\n", path);
                goto fail;
            }
            if (strcmp(word, "cnf") != 0) {
                fprintf(stderr, "c ERROR: only DIMACS 'p cnf' supported (got '%s')\n", word);
                goto fail;
            }
            if (!fi_read_int_token(&in, &nvars) || !fi_read_int_token(&in, &declared) || nvars <= 0) {
                fprintf(stderr, "c ERROR: malformed 'p cnf' line in %s\n", path);
                goto fail;
            }
            p->nvars = nvars;
            header_found = 1;
            fi_skip_line(&in);
            bol = 1;
            (void)declared;
            continue;
        }

        bol = 0;
        if (!header_found) {
            fprintf(stderr, "c ERROR: clause data before header in %s\n", path);
            goto fail;
        }

        {
            int sign = 1;
            int lit;
            if (c == '-') {
                sign = -1;
                c = fi_getc(&in);
            }
            if (c < '0' || c > '9') {
                fprintf(stderr, "c ERROR: invalid token while parsing %s\n", path);
                goto fail;
            }
            lit = c - '0';
            for (;;) {
                c = fi_getc(&in);
                if (c < '0' || c > '9') break;
                lit = lit * 10 + (c - '0');
            }
            if (c != EOF) fi_ungetc(&in, c);
            lit *= sign;

            if (lit == 0) {
                if (!slime_problem_add_clause(p, &clause)) {
                    fprintf(stderr, "c ERROR: out of memory parsing %s\n", path);
                    goto fail;
                }
                clause.size = 0;
            } else {
                int v = (lit > 0) ? lit : -lit;
                if (v < 1 || v > nvars) {
                    fprintf(stderr, "c ERROR: literal %d out of range 1..%d in %s\n", lit, nvars, path);
                    goto fail;
                }
                intvec_push(&clause, lit);
            }
        }
    }

    if (!header_found) {
        fprintf(stderr, "c ERROR: missing 'p cnf' header in %s\n", path);
        goto fail;
    }
    if (clause.size != 0) {
        fprintf(stderr, "c ERROR: unterminated clause at EOF in %s\n", path);
        goto fail;
    }

    intvec_free(&clause);
    fi_free(&in);
    fclose(fp);
    return 1;

fail:
    intvec_free(&clause);
    fi_free(&in);
    fclose(fp);
    slime_problem_free(p);
    return 0;
}

static void slime_cubevec_init(SlimeCubeVec *v) {
    memset(v, 0, sizeof(*v));
}

static void slime_cubevec_free(SlimeCubeVec *v) {
    if (v == NULL) return;
    for (int i = 0; i < v->size; ++i) free(v->data[i].lits);
    free(v->data);
    memset(v, 0, sizeof(*v));
}

static int slime_cubevec_reserve(SlimeCubeVec *v, int need) {
    int nc;
    SlimeCube *nd;
    if (need <= v->cap) return 1;
    nc = (v->cap > 0) ? v->cap : 8;
    while (nc < need) {
        if (nc > INT_MAX / 2) return 0;
        nc *= 2;
    }
    nd = (SlimeCube *)realloc(v->data, (size_t)nc * sizeof(SlimeCube));
    if (nd == NULL) return 0;
    v->data = nd;
    v->cap = nc;
    return 1;
}

static int slime_cubevec_push_copy(SlimeCubeVec *v, const IntVec *cube) {
    int *copy = NULL;
    if (!slime_cubevec_reserve(v, v->size + 1)) return 0;
    if (cube->size > 0) {
        copy = (int *)malloc((size_t)cube->size * sizeof(int));
        if (copy == NULL) return 0;
        memcpy(copy, cube->data, (size_t)cube->size * sizeof(int));
    }
    v->data[v->size].lits = copy;
    v->data[v->size].size = cube->size;
    v->size++;
    return 1;
}

static SlimeCube slime_cubevec_pop(SlimeCubeVec *v) {
    SlimeCube cube;
    cube.lits = NULL;
    cube.size = 0;
    if (v->size <= 0) return cube;
    v->size--;
    cube = v->data[v->size];
    v->data[v->size].lits = NULL;
    v->data[v->size].size = 0;
    return cube;
}

static void slime_cube_free(SlimeCube *cube) {
    if (cube == NULL) return;
    free(cube->lits);
    cube->lits = NULL;
    cube->size = 0;
}

static int slime_solver_init_from_problem(Solver *s,
                                          const SlimeCnfProblem *problem,
                                          int need_chb,
                                          int need_hess,
                                          int *base_unsat) {
    if (s == NULL || problem == NULL || problem->nvars <= 0) return 0;
    memset(s, 0, sizeof(*s));
    solver_init(s, problem->nvars, need_chb, need_hess);
    if (!slime_solver_load_clauses(s,
                                   problem->nclauses,
                                   (const int *const *)problem->clauses,
                                   problem->sizes,
                                   base_unsat)) {
        solver_destroy(s);
        return 0;
    }
    solver_seed_rng_from_problem(s, problem->parsed_clauses);
    return 1;
}

static void slime_solver_mix_seed(Solver *s, uint64_t salt) {
    if (s == NULL) return;
    s->rng_state ^= salt + UINT64_C(0x9e3779b97f4a7c15);
    s->rng_state = rng_next_u64(&s->rng_state);
    if (s->rng_state == 0) s->rng_state = 1;
}

static void slime_print_model01(int nvars, const unsigned char *model01) {
    int col = 0;
    fputs("v ", stdout);
    for (int v = 0; v < nvars; ++v) {
        int lit = model01[v] ? (v + 1) : -(v + 1);
        printf("%d ", lit);
        if (++col == 16) {
            fputs("\nv ", stdout);
            col = 0;
        }
    }
    fputs("0\n", stdout);
}

static int slime_cube_split_dfs(Solver *s,
                                IntVec *assumptions,
                                int depth,
                                int split_depth,
                                SlimeCubeVec *out) {
    ClauseRef confl;
    Lit branch;
    int old_dl;

    if (solver_poll_external_stop(s)) return 1;

    confl = solver_propagate(s);
    if (!clauseref_is_none(confl)) {
        return 1;
    }
    if (depth >= split_depth) {
        return slime_cubevec_push_copy(out, assumptions);
    }

    branch = solver_pick_branch_lit(s);
    if (branch < 0) {
        return slime_cubevec_push_copy(out, assumptions);
    }

    old_dl = solver_decision_level(s);
    {
        Lit order[2] = { branch, lit_neg(branch) };
        for (int i = 0; i < 2; ++i) {
            Lit lit = order[i];
            int dimacs_lit = lit_to_dimacs(lit);
            intvec_push(&s->trail_lim, s->trail_size);
            if (solver_enqueue(s, lit, reason_none())) {
                intvec_push(assumptions, dimacs_lit);
                if (!slime_cube_split_dfs(s, assumptions, depth + 1, split_depth, out)) {
                    assumptions->size--;
                    solver_cancel_until(s, old_dl);
                    return 0;
                }
                assumptions->size--;
            }
            solver_cancel_until(s, old_dl);
        }
    }
    return 1;
}

static int slime_generate_cubes(const SlimeCnfProblem *problem,
                                const SlimeSatOptions *base_opt,
                                int split_depth,
                                SlimeCubeVec *out) {
    Solver s;
    SlimeSatOptions split_opt;
    IntVec assumptions;
    int base_unsat = 0;

    if (split_depth <= 0) return 1;
    slime_sat_options_normalize(&split_opt, base_opt);
    split_opt.use_hess = 0;
    split_opt.use_ct = 0;
    split_opt.use_mab = base_opt != NULL ? base_opt->use_mab : 0;

    if (!slime_solver_init_from_problem(&s,
                                        problem,
                                        (split_opt.heuristic_mode == 1 || split_opt.use_mab) ? 1 : 0,
                                        0,
                                        &base_unsat)) {
        return 0;
    }
    if (base_unsat || !s.ok) {
        solver_destroy(&s);
        return 1;
    }

    solver_apply_runtime_options(&s, &split_opt);
    intvec_init(&assumptions);
    if (!slime_cube_split_dfs(&s, &assumptions, 0, split_depth, out)) {
        intvec_free(&assumptions);
        solver_destroy(&s);
        slime_cubevec_free(out);
        return 0;
    }
    intvec_free(&assumptions);
    solver_destroy(&s);
    return 1;
}

static void slime_sat_options_default(SlimeSatOptions *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->heuristic_mode = 0;
    cfg->use_mab = 0;
    cfg->mabc = 4.0;
    cfg->use_hess = 0;
    cfg->use_ct = 1;
    cfg->ct_lbd_max = 6;
    cfg->ct_maxlen = 12;
    cfg->ct_max_cubes = 40000;
    cfg->ct_buddy_merge = 0;
    cfg->ct_escape_rounds = 4;
    cfg->ct_probe_restarts = 4;
    cfg->use_simplify = 1;
    cfg->use_bve = 0;
    cfg->use_chrono = 1;
    cfg->use_inprocess = 0;
    cfg->use_probe = 1;
}

static void slime_sat_options_normalize(SlimeSatOptions *dst, const SlimeSatOptions *src) {
    slime_sat_options_default(dst);
    if (src != NULL) *dst = *src;
}

#if defined(SATX_HAVE_THREADS)
typedef enum {
    SLIME_PAR_STRATEGY_PORTFOLIO = 0,
    SLIME_PAR_STRATEGY_CUBES = 1
} SlimeParallelStrategy;

typedef struct {
    const SlimeCnfProblem *problem;
    const SlimeCubeVec *cubes;
    SlimeSatOptions base_opt;
    SlimeParallelStrategy strategy;
    int portfolio_variants;
    const int *portfolio_assumptions;
    int portfolio_num_assumptions;
    unsigned char *winner_model;
    SlimeSatStats *worker_stats;
    atomic_int stop_flag;
    atomic_int next_cube;
    atomic_int winner_rc;
    atomic_int error_flag;
} SlimeParallelCtx;

static void slime_parallel_worker_options(const SlimeSatOptions *base,
                                          int worker_id,
                                          int portfolio_variants,
                                          SlimeSatOptions *out) {
    int variant;
    slime_sat_options_normalize(out, base);
    variant = (portfolio_variants > 0) ? (worker_id % portfolio_variants) : 0;

    switch (variant & 3) {
        case 0:
            break;
        case 1:
            out->heuristic_mode = 0;
            out->use_mab = 0;
            break;
        case 2:
            out->heuristic_mode = 1;
            out->use_mab = 0;
            break;
        default:
            out->heuristic_mode = 1;
            out->use_mab = 1;
            if (out->mabc <= 0.0) out->mabc = 0.2;
            break;
    }

    if ((worker_id & 1) != 0) {
        out->ct_probe_restarts += 1;
    }
    if (out->ct_probe_restarts < 0) out->ct_probe_restarts = 0;
}

static int slime_parallel_claim_result(SlimeParallelCtx *ctx,
                                       int rc,
                                       const unsigned char *model01) {
    int expected = 0;
    if (!atomic_compare_exchange_strong(&ctx->stop_flag, &expected, 1)) {
        return 0;
    }
    atomic_store(&ctx->winner_rc, rc);
    if (rc == 10 && model01 != NULL && ctx->winner_model != NULL) {
        memcpy(ctx->winner_model, model01, (size_t)ctx->problem->nvars * sizeof(unsigned char));
    }
    return 1;
}

static int slime_parallel_portfolio_worker(void *arg, int worker_id) {
    SlimeParallelCtx *ctx = (SlimeParallelCtx *)arg;
    SlimeSatHandle *handle;
    SlimeSatOptions opt;
    SlimeSatStats stats;
    unsigned char *model01;
    size_t model_bytes;
    int rc;

    memset(&stats, 0, sizeof(stats));
    slime_parallel_worker_options(&ctx->base_opt, worker_id, ctx->portfolio_variants, &opt);
    handle = slime_sat_handle_create(ctx->problem->nvars,
                                     ctx->problem->nclauses,
                                     (const int *const *)ctx->problem->clauses,
                                     ctx->problem->sizes,
                                     &opt);
    if (handle == NULL) {
        atomic_store(&ctx->error_flag, 1);
        atomic_store(&ctx->stop_flag, 1);
        return 1;
    }
    slime_sat_handle_mix_seed(handle, UINT64_C(0x9e3779b97f4a7c15) ^ (uint64_t)(worker_id + 1));
    slime_sat_handle_set_external_stop(handle, &ctx->stop_flag);

    model_bytes = (size_t)(ctx->problem->nvars > 0 ? ctx->problem->nvars : 1);
    model01 = (unsigned char *)malloc(model_bytes * sizeof(unsigned char));
    if (model01 == NULL) {
        slime_sat_handle_destroy(handle);
        atomic_store(&ctx->error_flag, 1);
        atomic_store(&ctx->stop_flag, 1);
        return 1;
    }

    rc = slime_sat_handle_solve(handle,
                                ctx->portfolio_assumptions,
                                ctx->portfolio_num_assumptions,
                                &stats,
                                model01);
    ctx->worker_stats[worker_id] = stats;

    if (rc == 10 || rc == 20) {
        slime_parallel_claim_result(ctx, rc, model01);
    } else if (rc != 0) {
        atomic_store(&ctx->error_flag, 1);
        atomic_store(&ctx->stop_flag, 1);
    }

    free(model01);
    slime_sat_handle_destroy(handle);
    return 0;
}

static int slime_parallel_cube_worker(void *arg, int worker_id) {
    SlimeParallelCtx *ctx = (SlimeParallelCtx *)arg;
    SlimeSatHandle *handle;
    SlimeSatOptions opt;
    SlimeSatStats stats_total;
    unsigned char *model01;
    size_t model_bytes;

    memset(&stats_total, 0, sizeof(stats_total));
    slime_parallel_worker_options(&ctx->base_opt, worker_id, ctx->portfolio_variants, &opt);
    handle = slime_sat_handle_create(ctx->problem->nvars,
                                     ctx->problem->nclauses,
                                     (const int *const *)ctx->problem->clauses,
                                     ctx->problem->sizes,
                                     &opt);
    if (handle == NULL) {
        atomic_store(&ctx->error_flag, 1);
        atomic_store(&ctx->stop_flag, 1);
        return 1;
    }
    slime_sat_handle_mix_seed(handle, UINT64_C(0xbf58476d1ce4e5b9) ^ (uint64_t)(worker_id + 1));
    slime_sat_handle_set_external_stop(handle, &ctx->stop_flag);

    model_bytes = (size_t)(ctx->problem->nvars > 0 ? ctx->problem->nvars : 1);
    model01 = (unsigned char *)malloc(model_bytes * sizeof(unsigned char));
    if (model01 == NULL) {
        slime_sat_handle_destroy(handle);
        atomic_store(&ctx->error_flag, 1);
        atomic_store(&ctx->stop_flag, 1);
        return 1;
    }

    for (;;) {
        SlimeSatStats stats;
        int idx;
        int rc;

        if (atomic_load(&ctx->stop_flag) != 0) break;
        idx = atomic_fetch_add(&ctx->next_cube, 1);
        if (idx >= ctx->cubes->size) break;

        memset(&stats, 0, sizeof(stats));
        rc = slime_sat_handle_solve(handle,
                                    ctx->cubes->data[idx].lits,
                                    ctx->cubes->data[idx].size,
                                    &stats,
                                    model01);
        stats_total.clauses += stats.clauses;
        stats_total.learnt += stats.learnt;
        stats_total.conflicts += stats.conflicts;
        stats_total.decisions += stats.decisions;
        stats_total.propagations += stats.propagations;
        stats_total.restarts += stats.restarts;
        stats_total.hess_calls += stats.hess_calls;
        stats_total.hess_sat_hits += stats.hess_sat_hits;
        stats_total.ct_added += stats.ct_added;
        stats_total.ct_merged += stats.ct_merged;
        stats_total.ct_escaped += stats.ct_escaped;
        stats_total.ct_probe_added += stats.ct_probe_added;

        if (rc == 10) {
            slime_parallel_claim_result(ctx, rc, model01);
            break;
        }
        if (rc != 20 && rc != 0) {
            atomic_store(&ctx->error_flag, 1);
            atomic_store(&ctx->stop_flag, 1);
            break;
        }
    }

    ctx->worker_stats[worker_id] = stats_total;
    free(model01);
    slime_sat_handle_destroy(handle);
    return 0;
}
#endif

static void slime_sat_stats_from_solver(SlimeSatStats *stats, const Solver *s) {
    if (stats == NULL) return;
    stats->clauses = (long long)s->clauses.size + s->binary_clauses;
    stats->learnt = s->learnts.size;
    stats->conflicts = s->conflicts;
    stats->decisions = s->decisions;
    stats->propagations = s->propagations;
    stats->restarts = s->restarts;
    stats->hess_calls = s->hess_calls;
    stats->hess_sat_hits = s->hess_sat_hits;
    stats->ct_added = s->ct_added;
    stats->ct_merged = s->ct_merged;
    stats->ct_escaped = s->ct_escaped;
    stats->ct_probe_added = s->ct_probe_added;
}

static int slime_solver_load_clauses(Solver *s,
                                     int nclauses,
                                     const int *const *clauses,
                                     const int *sizes,
                                     int *base_unsat) {
    IntVec clause;
    intvec_init(&clause);
    *base_unsat = 0;

    for (int i = 0; i < nclauses; ++i) {
        int contradiction = 0;
        int sz = sizes[i];
        if (sz < 0 || (sz > 0 && clauses[i] == NULL)) {
            intvec_free(&clause);
            return 0;
        }
        clause.size = 0;
        for (int j = 0; j < sz; ++j) {
            int lit = clauses[i][j];
            int v = (lit >= 0) ? lit : -lit;
            if (v < 1 || v > s->nvars) {
                intvec_free(&clause);
                return 0;
            }
            intvec_push(&clause, dimacs_to_lit(lit));
        }
        solver_track_original_clause(s, (Lit *)clause.data, clause.size);
        if (!solver_commit_input_clause(s, &clause, &contradiction)) {
            intvec_free(&clause);
            return 0;
        }
        if (contradiction) {
            *base_unsat = 1;
            break;
        }
    }

    intvec_free(&clause);
    return 1;
}

static void slime_solver_clear_assumptions(Solver *s) {
    solver_cancel_until(s, 0);
    s->assumption_level = 0;
}

static int slime_solver_push_assumptions(Solver *s, const int *assumptions, int num_assumptions) {
    slime_solver_clear_assumptions(s);
    for (int i = 0; i < num_assumptions; ++i) {
        Lit p;
        int lit = assumptions[i];
        int v = (lit >= 0) ? lit : -lit;
        int cur;
        if (v < 1 || v > s->nvars) return 0;
        p = dimacs_to_lit(lit);
        cur = solver_lit_value(s, p);
        if (cur == 1) continue;
        if (cur == -1) {
            slime_solver_clear_assumptions(s);
            return 20;
        }
        intvec_push(&s->trail_lim, s->trail_size);
        if (!solver_enqueue(s, p, reason_none())) {
            slime_solver_clear_assumptions(s);
            return 20;
        }
        /* mapeo de equivalencias: si la variable fue sustituida por un
           literal equivalente, la suposición propaga a su representante
           (F ⊨ v ↔ repr, luego F ∧ A ≡ F ∧ A ∧ A_mapped). */
        for (int j = 0; j + 1 < s->eq_pairs.size; j += 2) {
            if (s->eq_pairs.data[j] == v - 1) {
                Lit r = dimacs_to_lit(s->eq_pairs.data[j + 1]);
                int rcur = solver_lit_value(s, r);
                if (rcur == 0) {
                    if (!solver_enqueue(s, r, reason_none())) {
                        slime_solver_clear_assumptions(s);
                        return 20;
                    }
                } else if (rcur != 1) {
                    slime_solver_clear_assumptions(s);
                    return 20;
                }
                break;
            }
        }
    }
    s->assumption_level = solver_decision_level(s);
    return 1;
}

SlimeSatHandle *slime_sat_handle_create(int nvars,
                                        int nclauses,
                                        const int *const *clauses,
                                        const int *sizes,
                                        const SlimeSatOptions *opt) {
    SlimeSatHandle *handle;
    int need_chb;
    int need_hess;

    if (nvars < 0 || nclauses < 0) return NULL;
    if (nclauses > 0 && (clauses == NULL || sizes == NULL)) return NULL;

    handle = (SlimeSatHandle *)malloc(sizeof(*handle));
    if (handle == NULL) return NULL;
    memset(handle, 0, sizeof(*handle));
    slime_sat_options_normalize(&handle->opt, opt);

    need_chb = (handle->opt.heuristic_mode == 1 || handle->opt.use_mab) ? 1 : 0;
    need_hess = handle->opt.use_hess ? 1 : 0;
    solver_init(&handle->solver, nvars, need_chb, need_hess);
    if (!slime_solver_load_clauses(&handle->solver, nclauses, clauses, sizes, &handle->base_unsat)) {
        solver_destroy(&handle->solver);
        free(handle);
        return NULL;
    }
    solver_seed_rng_from_problem(&handle->solver, nclauses);
    solver_apply_runtime_options(&handle->solver, &handle->opt);
    return handle;
}

void slime_sat_handle_reconfigure(SlimeSatHandle *handle, const SlimeSatOptions *opt) {
    if (handle == NULL) return;
    slime_sat_options_normalize(&handle->opt, opt);
    solver_apply_runtime_options(&handle->solver, &handle->opt);
}

#if defined(SATX_HAVE_THREADS)
static void slime_sat_handle_set_external_stop(SlimeSatHandle *handle, const atomic_int *stop_flag) {
    if (handle == NULL) return;
    handle->solver.external_stop = stop_flag;
    handle->solver.external_stop_hit = 0;
}
#endif

static void slime_sat_handle_mix_seed(SlimeSatHandle *handle, uint64_t salt) {
    if (handle == NULL) return;
    slime_solver_mix_seed(&handle->solver, salt);
}

int slime_sat_handle_solve(SlimeSatHandle *handle,
                           const int *assumptions,
                           int num_assumptions,
                           SlimeSatStats *stats,
                           unsigned char *model01) {
    int rc;
    SlimeSatStats before;
    SlimeSatStats after;
    if (stats != NULL) memset(stats, 0, sizeof(*stats));
    if (handle == NULL || num_assumptions < 0) return 0;
    if (handle->base_unsat) return 20;

    memset(&before, 0, sizeof(before));
    slime_sat_stats_from_solver(&before, &handle->solver);

    rc = slime_solver_push_assumptions(&handle->solver, assumptions, num_assumptions);
    if (rc != 1) return rc;

    rc = solver_solve(&handle->solver);
    memset(&after, 0, sizeof(after));
    slime_sat_stats_from_solver(&after, &handle->solver);
    if (stats != NULL) {
        stats->clauses = after.clauses;
        stats->learnt = after.learnt;
        stats->conflicts = after.conflicts - before.conflicts;
        stats->decisions = after.decisions - before.decisions;
        stats->propagations = after.propagations - before.propagations;
        stats->restarts = after.restarts - before.restarts;
        stats->hess_calls = after.hess_calls - before.hess_calls;
        stats->hess_sat_hits = after.hess_sat_hits - before.hess_sat_hits;
        stats->ct_added = after.ct_added - before.ct_added;
        stats->ct_merged = after.ct_merged - before.ct_merged;
        stats->ct_escaped = after.ct_escaped - before.ct_escaped;
        stats->ct_probe_added = after.ct_probe_added - before.ct_probe_added;
    }
    if (model01 != NULL && rc == 10) {
        for (int v = 0; v < handle->solver.nvars; ++v) {
            model01[v] = (unsigned char)(handle->solver.assigns[v] > 0 ? 1U : 0U);
        }
    }
    slime_solver_clear_assumptions(&handle->solver);
    return rc;
}

void slime_sat_handle_destroy(SlimeSatHandle *handle) {
    if (handle == NULL) return;
    solver_destroy(&handle->solver);
    free(handle);
}

static int slime_write_text_file(const char *path, const char *text) {
    FILE *fp = fopen(path, "wb");
    size_t n;
    if (!fp) return 0;
    n = strlen(text);
    if (fwrite(text, 1, n, fp) != n) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return 1;
}

static int slime_selftest_case_opts(const char *name,
                                    const char *text,
                                    const char *const *opts,
                                    int nopts,
                                    int exp_rc) {
    char tmp_name[256];
    char *argv_run[16];
    int argc_run = 0;
    int rc;
    unsigned long stamp = (unsigned long)time(NULL);
    stamp ^= (unsigned long)clock();
    stamp ^= (unsigned long)(uintptr_t)(const void *)text;
    snprintf(tmp_name, sizeof(tmp_name), "slime_selftest_%s_%lu.cnf", name, stamp);
    if (!slime_write_text_file(tmp_name, text)) {
        fprintf(stderr, "c selftest[%s]: failed to write temp file\n", name);
        return 0;
    }

    argv_run[argc_run++] = "slime";
    for (int i = 0; i < nopts; ++i) {
        if (argc_run + 2 >= (int)(sizeof(argv_run) / sizeof(argv_run[0]))) {
            remove(tmp_name);
            fprintf(stderr, "c selftest[%s]: too many argv entries\n", name);
            return 0;
        }
        argv_run[argc_run++] = (char *)opts[i];
    }
    argv_run[argc_run++] = tmp_name;
    argv_run[argc_run] = NULL;
    rc = slime_entry(argc_run, argv_run);
    remove(tmp_name);
    if (rc != exp_rc) {
        fprintf(stderr, "c selftest[%s]: expected rc=%d got rc=%d\n", name, exp_rc, rc);
        return 0;
    }
    return 1;
}

static int slime_selftest_case(const char *name, const char *text, int exp_rc) {
    const char *opts[] = { "--no-model" };
    return slime_selftest_case_opts(name, text, opts, 1, exp_rc);
}

#if defined(SATX_HAVE_THREADS)
static int slime_parallel_selftest_case(const char *name,
                                        const char *text,
                                        int split_depth,
                                        int exp_rc) {
    const char *opts[8];
    char split_depth_text[32];
    int nopts = 0;

    opts[nopts++] = "--parallel";
    opts[nopts++] = "threads";
    opts[nopts++] = "--jobs";
    opts[nopts++] = "2";
    opts[nopts++] = "--portfolio";
    opts[nopts++] = "4";
    if (split_depth > 0) {
        snprintf(split_depth_text, sizeof(split_depth_text), "%d", split_depth);
        opts[nopts++] = "--split-depth";
        opts[nopts++] = split_depth_text;
    }

    return slime_selftest_case_opts(name, text, opts, nopts, exp_rc);
}
#endif

static int slime_incremental_selftest(void) {
    const int c1[] = { 1, 2 };
    const int c2[] = { -1, 2 };
    const int *clauses[] = { c1, c2 };
    const int sizes[] = { 2, 2 };
    const int a_unsat[] = { -2 };
    const int a_sat[] = { 2 };
    const int a_sat2[] = { -1 };
    unsigned char model[2];
    SlimeSatHandle *handle = slime_sat_handle_create(2, 2, clauses, sizes, NULL);
    if (handle == NULL) {
        fprintf(stderr, "c selftest[incremental]: failed to create handle\n");
        return 0;
    }
    if (slime_sat_handle_solve(handle, a_unsat, 1, NULL, NULL) != 20) {
        fprintf(stderr, "c selftest[incremental]: expected UNSAT under assumption -2\n");
        slime_sat_handle_destroy(handle);
        return 0;
    }
    if (slime_sat_handle_solve(handle, a_sat, 1, NULL, model) != 10 || model[1] == 0) {
        fprintf(stderr, "c selftest[incremental]: expected SAT under assumption 2\n");
        slime_sat_handle_destroy(handle);
        return 0;
    }
    if (slime_sat_handle_solve(handle, a_sat2, 1, NULL, model) != 10 || model[1] == 0) {
        fprintf(stderr, "c selftest[incremental]: expected SAT under assumption -1\n");
        slime_sat_handle_destroy(handle);
        return 0;
    }
    slime_sat_handle_destroy(handle);
    return 1;
}

static int slime_incremental_hess_selftest(void) {
    const int c1[] = { 1, 2, 3 };
    const int c2[] = { -1, 2, 3 };
    const int c3[] = { 1, -2, 3 };
    const int *clauses[] = { c1, c2, c3 };
    const int sizes[] = { 3, 3, 3 };
    const int assumptions[] = { -3 };
    unsigned char model[3];
    SlimeSatOptions opt;
    SlimeSatHandle *handle;

    slime_sat_options_default(&opt);
    opt.use_hess = 1;
    handle = slime_sat_handle_create(3, 3, clauses, sizes, &opt);
    if (handle == NULL) {
        fprintf(stderr, "c selftest[incremental_hess]: failed to create handle\n");
        return 0;
    }
    if (slime_sat_handle_solve(handle, assumptions, 1, NULL, model) != 10) {
        fprintf(stderr, "c selftest[incremental_hess]: expected SAT under assumption -3\n");
        slime_sat_handle_destroy(handle);
        return 0;
    }
    if (model[2] != 0u) {
        fprintf(stderr, "c selftest[incremental_hess]: expected model to respect assumption -3\n");
        slime_sat_handle_destroy(handle);
        return 0;
    }
    slime_sat_handle_destroy(handle);
    return 1;
}

#if defined(BASILISK_NO_MAIN)
static int slime_basilisk_selftest_case(const char *name,
                                        const char *text,
                                        const char *mode,
                                        int exp_rc) {
    char tmp_name[256];
    char *argv_run[5];
    int rc;
    unsigned long stamp = (unsigned long)time(NULL);
    stamp ^= (unsigned long)clock();
    stamp ^= (unsigned long)(uintptr_t)(const void *)text;
    stamp ^= (unsigned long)(uintptr_t)(const void *)mode;
    snprintf(tmp_name, sizeof(tmp_name), "slime_basilisk_selftest_%s_%lu.cnf", name, stamp);
    if (!slime_write_text_file(tmp_name, text)) {
        fprintf(stderr, "c selftest[%s]: failed to write temp file\n", name);
        return 0;
    }

    argv_run[0] = "slime";
    argv_run[1] = "--mode";
    argv_run[2] = (char *)mode;
    argv_run[3] = tmp_name;
    argv_run[4] = NULL;
    rc = slime_entry(4, argv_run);
    remove(tmp_name);
    if (rc != exp_rc) {
        fprintf(stderr, "c selftest[%s]: expected rc=%d got rc=%d\n", name, exp_rc, rc);
        return 0;
    }
    return 1;
}
#endif

static int slime_run_selftest(void) {
    const char *sat_case =
        "p cnf 2 2\n"
        "1 2 0\n"
        "-1 2 0\n";
    const char *unsat_case =
        "p cnf 1 2\n"
        "1 0\n"
        "-1 0\n";
    const char *unit_case =
        "p cnf 3 3\n"
        "1 0\n"
        "-1 2 0\n"
        "-2 3 0\n";
    const char *cube_unsat_case =
        "p cnf 2 4\n"
        "1 2 0\n"
        "-1 2 0\n"
        "1 -2 0\n"
        "-1 -2 0\n";

    if (!slime_selftest_case("sat", sat_case, 10)) return 1;
    if (!slime_selftest_case("unsat", unsat_case, 20)) return 1;
    if (!slime_selftest_case("unit", unit_case, 10)) return 1;
    if (!slime_incremental_selftest()) return 1;
    if (!slime_incremental_hess_selftest()) return 1;
#if defined(SATX_HAVE_THREADS)
    if (!slime_parallel_selftest_case("par_portfolio_sat", sat_case, 0, 10)) return 1;
    if (!slime_parallel_selftest_case("par_portfolio_unsat", unsat_case, 0, 20)) return 1;
    if (!slime_parallel_selftest_case("par_cube_sat", sat_case, 1, 10)) return 1;
    if (!slime_parallel_selftest_case("par_cube_unsat", cube_unsat_case, 1, 20)) return 1;
#endif
#if defined(BASILISK_NO_MAIN)
    {
        const char *count_case =
            "p cnf 2 1\n"
            "1 2 0\n";
        const char *project_case =
            "p cnf 2 1\n"
            "c ind 1 0\n"
            "1 2 0\n";
        if (!slime_basilisk_selftest_case("count_dispatch", count_case, "count", 0)) return 1;
        if (!slime_basilisk_selftest_case("project_dispatch", project_case, "project", 0)) return 1;
    }
#endif

    fprintf(stderr, "selftest: OK\n");
    return 0;
}

int slime_sat_solve_clauses(int nvars,
                            int nclauses,
                            const int *const *clauses,
                            const int *sizes,
                            const int *assumptions,
                            int num_assumptions,
                            const SlimeSatOptions *opt,
                            SlimeSatStats *stats,
                            unsigned char *model01) {
    SlimeSatHandle *handle = slime_sat_handle_create(nvars, nclauses, clauses, sizes, opt);
    int rc;
    if (handle == NULL) return 0;
    rc = slime_sat_handle_solve(handle, assumptions, num_assumptions, stats, model01);
    slime_sat_handle_destroy(handle);
    return rc;
}

#if defined(SATX_HAVE_THREADS)
typedef struct {
    int rc;
    SlimeSatStats stats;
    int cubes_generated;
    SlimeParallelStrategy strategy;
    unsigned char *model01;
} SlimeParallelRunResult;

static void slime_parallel_result_free(SlimeParallelRunResult *res) {
    if (res == NULL) return;
    free(res->model01);
    res->model01 = NULL;
}

static int slime_run_parallel_solve(const SlimeCnfProblem *problem,
                                    const SlimeSatOptions *base_opt,
                                    const KrbParallelRuntime *parallel_rt,
                                    const KrbParallelConfig *parallel_cfg,
                                    SlimeParallelRunResult *out,
                                    char *err,
                                    size_t errsz) {
    SlimeParallelCtx ctx;
    SlimeCubeVec cubes;
    KrbParallelWorkerFn worker_fn = NULL;
    size_t model_bytes;
    int jobs;

    memset(out, 0, sizeof(*out));
    model_bytes = (size_t)(problem->nvars > 0 ? problem->nvars : 1);
    out->model01 = (unsigned char *)malloc(model_bytes * sizeof(unsigned char));
    if (out->model01 == NULL) {
        snprintf(err, errsz, "out of memory allocating parallel SAT model");
        return 0;
    }

    slime_cubevec_init(&cubes);
    jobs = parallel_rt->jobs > 0 ? parallel_rt->jobs : 1;
    memset(&ctx, 0, sizeof(ctx));
    ctx.problem = problem;
    ctx.base_opt = *base_opt;
    ctx.portfolio_variants = parallel_cfg->portfolio > 0 ? parallel_cfg->portfolio : 1;
    ctx.winner_model = out->model01;
    ctx.worker_stats = (SlimeSatStats *)calloc((size_t)jobs, sizeof(SlimeSatStats));
    if (ctx.worker_stats == NULL) {
        slime_parallel_result_free(out);
        snprintf(err, errsz, "out of memory allocating parallel worker stats");
        return 0;
    }
    atomic_init(&ctx.stop_flag, 0);
    atomic_init(&ctx.next_cube, 0);
    atomic_init(&ctx.winner_rc, 0);
    atomic_init(&ctx.error_flag, 0);

    if (parallel_cfg->split_depth > 0) {
        if (!slime_generate_cubes(problem, base_opt, parallel_cfg->split_depth, &cubes)) {
            free(ctx.worker_stats);
            slime_parallel_result_free(out);
            snprintf(err, errsz, "failed to generate cube frontier");
            return 0;
        }
        out->cubes_generated = cubes.size;
        if (cubes.size == 0) {
            free(ctx.worker_stats);
            slime_cubevec_free(&cubes);
            out->rc = 20;
            out->strategy = SLIME_PAR_STRATEGY_CUBES;
            return 1;
        }
        if (cubes.size == 1) {
            ctx.strategy = SLIME_PAR_STRATEGY_PORTFOLIO;
            ctx.portfolio_assumptions = cubes.data[0].lits;
            ctx.portfolio_num_assumptions = cubes.data[0].size;
            worker_fn = slime_parallel_portfolio_worker;
        } else {
            ctx.strategy = SLIME_PAR_STRATEGY_CUBES;
            ctx.cubes = &cubes;
            worker_fn = slime_parallel_cube_worker;
        }
    } else {
        ctx.strategy = SLIME_PAR_STRATEGY_PORTFOLIO;
        worker_fn = slime_parallel_portfolio_worker;
    }

    if (!krb_parallel_run_threads(jobs, worker_fn, &ctx, err, errsz)) {
        free(ctx.worker_stats);
        slime_cubevec_free(&cubes);
        slime_parallel_result_free(out);
        return 0;
    }

    out->strategy = ctx.strategy;
    for (int i = 0; i < jobs; ++i) {
        out->stats.clauses += ctx.worker_stats[i].clauses;
        out->stats.learnt += ctx.worker_stats[i].learnt;
        out->stats.conflicts += ctx.worker_stats[i].conflicts;
        out->stats.decisions += ctx.worker_stats[i].decisions;
        out->stats.propagations += ctx.worker_stats[i].propagations;
        out->stats.restarts += ctx.worker_stats[i].restarts;
        out->stats.hess_calls += ctx.worker_stats[i].hess_calls;
        out->stats.hess_sat_hits += ctx.worker_stats[i].hess_sat_hits;
        out->stats.ct_added += ctx.worker_stats[i].ct_added;
        out->stats.ct_merged += ctx.worker_stats[i].ct_merged;
        out->stats.ct_escaped += ctx.worker_stats[i].ct_escaped;
        out->stats.ct_probe_added += ctx.worker_stats[i].ct_probe_added;
    }

    if (atomic_load(&ctx.error_flag) != 0) {
        out->rc = 0;
    } else {
        out->rc = atomic_load(&ctx.winner_rc);
        if (out->rc == 0) {
            out->rc = (ctx.strategy == SLIME_PAR_STRATEGY_CUBES) ? 20 : 0;
        }
    }

    free(ctx.worker_stats);
    slime_cubevec_free(&cubes);
    return 1;
}
#endif

int slime_maxsat_solve_mem(int nv, int nc,
                           const int *const *cls, const int *sizes,
                           const double *weights, double top_weight,
                           unsigned char *model01, double *optimal_cost);


/* ================================================================
 * Weighted HESS+LS (unified, weights=NULL → weight=1 SAT mode)
 * ================================================================ */
static void hessw_flip(int var, unsigned char *sat, int *sat_count, double *unsat,
                       const IntVec *pos, const IntVec *neg, const double *w) {
    int old = sat[var] ? 1 : 0;
    sat[var] ^= 1u;
    if (old) {
        for (int i = 0; i < pos[var].size; i++) {
            int ci = pos[var].data[i]; int b = sat_count[ci]; sat_count[ci] = b - 1;
            if (b == 1) *unsat += (w ? w[ci] : 1.0);
        }
        for (int i = 0; i < neg[var].size; i++) {
            int ci = neg[var].data[i]; int b = sat_count[ci]; sat_count[ci] = b + 1;
            if (b == 0) *unsat -= (w ? w[ci] : 1.0);
        }
    } else {
        for (int i = 0; i < neg[var].size; i++) {
            int ci = neg[var].data[i]; int b = sat_count[ci]; sat_count[ci] = b - 1;
            if (b == 1) *unsat += (w ? w[ci] : 1.0);
        }
        for (int i = 0; i < pos[var].size; i++) {
            int ci = pos[var].data[i]; int b = sat_count[ci]; sat_count[ci] = b + 1;
            if (b == 0) *unsat -= (w ? w[ci] : 1.0);
        }
    }
}

typedef struct {
    int clause;
    unsigned char sense;
} HessWLSOcc;

typedef struct {
    HessWLSOcc *data;
    int size;
    int cap;
} HessWLSOccVec;

typedef struct {
    HessWLSOccVec occs;
    IntVec neigh;
    double score;
    long long last_flip_step;
    int unsat_appear;
    unsigned char cc_value;
    unsigned char in_ccd;
} HessWLSVar;

typedef struct {
    int size;
    const int *lits;
    double base_weight;   /* objective weight, fixed */
    double weight;        /* mutable heuristic weight */
    int sat_count;
    int sat_var;
} HessWLSClause;

typedef struct {
    int n_vars;
    int n_clauses;
    HessWLSVar *vars;
    HessWLSClause *clauses;
    int *unsat_clauses;
    int *index_in_unsat_clauses;
    int *unsat_vars;
    int *index_in_unsat_vars;
    int *ccd_vars;
    unsigned char *solution;
    unsigned char *best_solution;
    double unsat;
    double best_unsat;
    long long step;
    long long mems;
    long long max_mems;
    long long max_steps;
    int aspiration_active;
    double aspiration_score;
    int swt_threshold;
    double swt_p;
    double swt_q;
    double avg_clause_weight;
    double delta_total_clause_weight;
    int unsat_clause_count;
    int unsat_var_count;
    int ccd_count;
} HessWLSState;

static void hessw_ls_occvec_push(HessWLSOccVec *v, int clause, unsigned char sense) {
    if (v->size == v->cap) {
        int nc = v->cap ? (v->cap << 1) : 4;
        v->data = (HessWLSOcc *)xrealloc(v->data, (size_t)nc * sizeof(HessWLSOcc));
        v->cap = nc;
    }
    v->data[v->size].clause = clause;
    v->data[v->size].sense = sense;
    v->size++;
}

static void hessw_ls_build_neighborhood(HessWLSState *s) {
    unsigned char *mark = (unsigned char *)xmalloc((size_t)s->n_vars);
    memset(mark, 0, (size_t)s->n_vars);
    for (int v = 0; v < s->n_vars; ++v) {
        HessWLSVar *vp = &s->vars[v];
        for (int i = 0; i < vp->occs.size; ++i) {
            int c = vp->occs.data[i].clause;
            for (int k = 0; k < s->clauses[c].size; ++k) {
                int lit = s->clauses[c].lits[k];
                int u = lit > 0 ? lit - 1 : -lit - 1;
                if (u == v || mark[u]) continue;
                mark[u] = 1;
                intvec_push(&vp->neigh, u);
            }
        }
        for (int i = 0; i < vp->neigh.size; ++i) {
            mark[vp->neigh.data[i]] = 0;
        }
    }
    free(mark);
}

static void hessw_ls_sat_clause(HessWLSState *s, int ci) {
    int index = s->index_in_unsat_clauses[ci];
    int last_item = s->unsat_clauses[--s->unsat_clause_count];
    s->unsat_clauses[index] = last_item;
    s->index_in_unsat_clauses[last_item] = index;
    s->unsat -= s->clauses[ci].base_weight;
    for (int k = 0; k < s->clauses[ci].size; ++k) {
        int lit = s->clauses[ci].lits[k];
        int v = lit > 0 ? lit - 1 : -lit - 1;
        s->vars[v].unsat_appear--;
        if (s->vars[v].unsat_appear == 0) {
            int idx = s->index_in_unsat_vars[v];
            int tail = s->unsat_vars[--s->unsat_var_count];
            s->unsat_vars[idx] = tail;
            s->index_in_unsat_vars[tail] = idx;
        }
    }
}

static void hessw_ls_unsat_clause(HessWLSState *s, int ci) {
    s->index_in_unsat_clauses[ci] = s->unsat_clause_count;
    s->unsat_clauses[s->unsat_clause_count++] = ci;
    s->unsat += s->clauses[ci].base_weight;
    for (int k = 0; k < s->clauses[ci].size; ++k) {
        int lit = s->clauses[ci].lits[k];
        int v = lit > 0 ? lit - 1 : -lit - 1;
        s->vars[v].unsat_appear++;
        if (s->vars[v].unsat_appear == 1) {
            s->index_in_unsat_vars[v] = s->unsat_var_count;
            s->unsat_vars[s->unsat_var_count++] = v;
        }
    }
}

static void hessw_ls_update_clause_weights(HessWLSState *s) {
    for (int i = 0; i < s->unsat_clause_count; ++i) {
        int c = s->unsat_clauses[i];
        s->clauses[c].weight += 1.0;
    }
    s->mems += s->unsat_var_count;
    for (int i = 0; i < s->unsat_var_count; ++i) {
        int v = s->unsat_vars[i];
        s->vars[v].score += (double)s->vars[v].unsat_appear;
        if (s->vars[v].score > 0.0 && s->vars[v].cc_value == 1 && !s->vars[v].in_ccd) {
            s->ccd_vars[s->ccd_count++] = v;
            s->vars[v].in_ccd = 1;
        }
    }
    s->delta_total_clause_weight += (double)s->unsat_clause_count;
    if (s->n_clauses > 0 && s->delta_total_clause_weight >= (double)s->n_clauses) {
        s->avg_clause_weight += 1.0;
        s->delta_total_clause_weight -= (double)s->n_clauses;
        if (s->avg_clause_weight > (double)s->swt_threshold) {
            double scale_avg = s->avg_clause_weight * s->swt_q;
            s->avg_clause_weight = 0.0;
            s->delta_total_clause_weight = 0.0;
            for (int v = 0; v < s->n_vars; ++v) {
                s->vars[v].score = 0.0;
            }
            s->unsat = 0.0;
            for (int c = 0; c < s->n_clauses; ++c) {
                HessWLSClause *cp = &s->clauses[c];
                cp->weight = cp->weight * s->swt_p + scale_avg;
                if (cp->weight < 1.0) cp->weight = 1.0;
                s->delta_total_clause_weight += cp->weight;
                if (s->delta_total_clause_weight >= (double)s->n_clauses) {
                    s->avg_clause_weight += 1.0;
                    s->delta_total_clause_weight -= (double)s->n_clauses;
                }
                if (cp->sat_count == 0) {
                    s->unsat += cp->base_weight;
                    for (int k = 0; k < cp->size; ++k) {
                        int lit = cp->lits[k];
                        int v = lit > 0 ? lit - 1 : -lit - 1;
                        s->vars[v].score += cp->weight;
                    }
                } else if (cp->sat_count == 1) {
                    s->vars[cp->sat_var].score -= cp->weight;
                }
            }
            s->ccd_count = 0;
            for (int v = 0; v < s->n_vars; ++v) {
                if (s->vars[v].score > 0.0 && s->vars[v].cc_value == 1) {
                    s->ccd_vars[s->ccd_count++] = v;
                    s->vars[v].in_ccd = 1;
                } else {
                    s->vars[v].in_ccd = 0;
                }
            }
        }
    }
}

static void hessw_ls_update_cc_after_flip(HessWLSState *s, int flipv) {
    s->vars[flipv].cc_value = 0;
    s->mems += s->ccd_count;
    for (int idx = s->ccd_count - 1; idx >= 0; --idx) {
        int v = s->ccd_vars[idx];
        if (s->vars[v].score <= 0.0) {
            int last_item = s->ccd_vars[--s->ccd_count];
            s->ccd_vars[idx] = last_item;
            s->vars[v].in_ccd = 0;
        }
    }
    for (int i = 0; i < s->vars[flipv].neigh.size; ++i) {
        int v = s->vars[flipv].neigh.data[i];
        s->vars[v].cc_value = 1;
        if (s->vars[v].score > 0.0 && !s->vars[v].in_ccd) {
            s->ccd_vars[s->ccd_count++] = v;
            s->vars[v].in_ccd = 1;
        }
    }
}

static int hessw_ls_pick_var(HessWLSState *s) {
    int best_var = -1;
    if (s->ccd_count > 0) {
        s->mems += s->ccd_count;
        best_var = s->ccd_vars[0];
        for (int i = 1; i < s->ccd_count; ++i) {
            int v = s->ccd_vars[i];
            if (s->vars[v].score > s->vars[best_var].score) {
                best_var = v;
            } else if (s->vars[v].score == s->vars[best_var].score &&
                       s->vars[v].last_flip_step < s->vars[best_var].last_flip_step) {
                best_var = v;
            }
        }
        return best_var;
    }

    if (s->aspiration_active) {
        s->aspiration_score = s->avg_clause_weight;
        for (int i = 0; i < s->unsat_var_count; ++i) {
            int v = s->unsat_vars[i];
            if (s->vars[v].score > s->aspiration_score) {
                best_var = v;
                break;
            }
        }
        if (best_var >= 0) {
            for (int i = 0; i < s->unsat_var_count; ++i) {
                int v = s->unsat_vars[i];
                if (s->vars[v].score > s->vars[best_var].score) {
                    best_var = v;
                } else if (s->vars[v].score == s->vars[best_var].score &&
                           s->vars[v].last_flip_step < s->vars[best_var].last_flip_step) {
                    best_var = v;
                }
            }
            return best_var;
        }
    }

    hessw_ls_update_clause_weights(s);
    return -1;
}

static void hessw_ls_flip(HessWLSState *s, int flipv) {
    s->solution[flipv] ^= 1u;
    double org_score = s->vars[flipv].score;
    s->mems += s->vars[flipv].occs.size;
    for (int i = 0; i < s->vars[flipv].occs.size; ++i) {
        int ci = s->vars[flipv].occs.data[i].clause;
        unsigned char sense = s->vars[flipv].occs.data[i].sense;
        HessWLSClause *cp = &s->clauses[ci];
        if (s->solution[flipv] == sense) {
            cp->sat_count++;
            if (cp->sat_count == 1) {
                hessw_ls_sat_clause(s, ci);
                cp->sat_var = flipv;
                for (int k = 0; k < cp->size; ++k) {
                    int lit = cp->lits[k];
                    int v = lit > 0 ? lit - 1 : -lit - 1;
                    s->vars[v].score -= cp->weight;
                }
            } else if (cp->sat_count == 2) {
                s->vars[cp->sat_var].score += cp->weight;
            }
        } else {
            cp->sat_count--;
            if (cp->sat_count == 0) {
                hessw_ls_unsat_clause(s, ci);
                for (int k = 0; k < cp->size; ++k) {
                    int lit = cp->lits[k];
                    int v = lit > 0 ? lit - 1 : -lit - 1;
                    s->vars[v].score += cp->weight;
                }
            } else if (cp->sat_count == 1) {
                for (int k = 0; k < cp->size; ++k) {
                    int lit = cp->lits[k];
                    int v = lit > 0 ? lit - 1 : -lit - 1;
                    if (s->solution[v] == (lit > 0 ? 1u : 0u)) {
                        s->vars[v].score -= cp->weight;
                        cp->sat_var = v;
                        break;
                    }
                }
            }
        }
    }
    s->vars[flipv].score = -org_score;
    s->vars[flipv].last_flip_step = s->step;
    hessw_ls_update_cc_after_flip(s, flipv);
}

static void hessw_ls_state_free(HessWLSState *s) {
    if (s == NULL) return;
    if (s->vars != NULL) {
        for (int v = 0; v < s->n_vars; ++v) {
            free(s->vars[v].occs.data);
            free(s->vars[v].neigh.data);
        }
        free(s->vars);
    }
    if (s->clauses != NULL) {
        free(s->clauses);
    }
    free(s->unsat_clauses);
    free(s->index_in_unsat_clauses);
    free(s->unsat_vars);
    free(s->index_in_unsat_vars);
    free(s->ccd_vars);
    free(s->solution);
    free(s->best_solution);
    memset(s, 0, sizeof(*s));
}

/*
Disabled LS cache experiment.
The active implementation keeps LS simple and rebuilds the per-run state
inside hessw_ls_state_init(). This block is preserved for later review only.

typedef struct {
    int n_vars;
    int n_clauses;
    HessWLSOccVec *occ_by_var;
    IntVec *neigh_by_var;
} HessWLSCache;

static void hessw_ls_cache_init(HessWLSCache *cache, int n, int m) {
    memset(cache, 0, sizeof(*cache));
    cache->n_vars = n;
    cache->n_clauses = m;
    cache->occ_by_var = (HessWLSOccVec *)xmalloc((size_t)n * sizeof(HessWLSOccVec));
    cache->neigh_by_var = (IntVec *)xmalloc((size_t)n * sizeof(IntVec));
    for (int v = 0; v < n; ++v) {
        cache->occ_by_var[v].data = NULL;
        cache->occ_by_var[v].size = 0;
        cache->occ_by_var[v].cap = 0;
        intvec_init(&cache->neigh_by_var[v]);
    }
}

static void hessw_ls_cache_free(HessWLSCache *cache) {
    if (cache == NULL) return;
    for (int v = 0; v < cache->n_vars; ++v) {
        free(cache->occ_by_var[v].data);
        free(cache->neigh_by_var[v].data);
    }
    free(cache->occ_by_var);
    free(cache->neigh_by_var);
    memset(cache, 0, sizeof(*cache));
}
*/

static void hessw_ls_state_init(HessWLSState *s, int n, int m, int *cls_sizes, int **cls_data, double *cls_weights,
                                const unsigned char *seed_model) {
    memset(s, 0, sizeof(*s));
    s->n_vars = n;
    s->n_clauses = m;
    s->vars = (HessWLSVar *)xmalloc((size_t)n * sizeof(HessWLSVar));
    memset(s->vars, 0, (size_t)n * sizeof(HessWLSVar));
    s->clauses = (HessWLSClause *)xmalloc((size_t)m * sizeof(HessWLSClause));
    memset(s->clauses, 0, (size_t)m * sizeof(HessWLSClause));
    s->unsat_clauses = (int *)xmalloc((size_t)m * sizeof(int));
    s->index_in_unsat_clauses = (int *)xmalloc((size_t)m * sizeof(int));
    s->unsat_vars = (int *)xmalloc((size_t)n * sizeof(int));
    s->index_in_unsat_vars = (int *)xmalloc((size_t)n * sizeof(int));
    s->ccd_vars = (int *)xmalloc((size_t)n * sizeof(int));
    s->solution = (unsigned char *)xmalloc((size_t)n * sizeof(unsigned char));
    s->best_solution = (unsigned char *)xmalloc((size_t)n * sizeof(unsigned char));
    memset(s->unsat_clauses, 0, (size_t)m * sizeof(int));
    memset(s->index_in_unsat_clauses, 0, (size_t)m * sizeof(int));
    memset(s->unsat_vars, 0, (size_t)n * sizeof(int));
    memset(s->index_in_unsat_vars, 0, (size_t)n * sizeof(int));
    memset(s->ccd_vars, 0, (size_t)n * sizeof(int));
    memset(s->solution, 0, (size_t)n * sizeof(unsigned char));
    memset(s->best_solution, 0, (size_t)n * sizeof(unsigned char));

    s->aspiration_active = 1;
    s->swt_threshold = 50;
    s->swt_p = 0.3;
    s->swt_q = 0.7;
    s->avg_clause_weight = 1.0;
    s->delta_total_clause_weight = 0.0;
    s->max_mems = 24000000LL;
    s->max_steps = (long long)(n + m) * 32LL;
    if (s->max_steps < 2500LL) s->max_steps = 2500LL;
    if (s->max_steps > 180000LL) s->max_steps = 180000LL;
    if (m > 50000 || n > 5000) {
        s->max_steps = 22000LL;
        s->max_mems = 4000000LL;
    } else if (m > 10000) {
        if (s->max_steps > 40000LL) s->max_steps = 40000LL;
        s->max_mems = 8000000LL;
    } else if (m > 3000) {
        if (s->max_steps > 60000LL) s->max_steps = 60000LL;
        s->max_mems = 12000000LL;
    }

    for (int v = 0; v < n; ++v) {
        intvec_init(&s->vars[v].neigh);
        s->vars[v].occs.data = NULL;
        s->vars[v].occs.size = 0;
        s->vars[v].occs.cap = 0;
    }

    for (int ci = 0; ci < m; ++ci) {
        HessWLSClause *cp = &s->clauses[ci];
        cp->size = cls_sizes[ci];
        cp->lits = cls_data[ci];
        cp->base_weight = cls_weights ? cls_weights[ci] : 1.0;
        cp->weight = cp->base_weight;
        cp->sat_count = 0;
        cp->sat_var = -1;
        for (int k = 0; k < cp->size; ++k) {
            int lit = cp->lits[k];
            int v = lit > 0 ? lit - 1 : -lit - 1;
            hessw_ls_occvec_push(&s->vars[v].occs, ci, (unsigned char)(lit > 0 ? 1u : 0u));
        }
    }

    hessw_ls_build_neighborhood(s);

    if (seed_model != NULL) {
        memcpy(s->solution, seed_model, (size_t)n * sizeof(unsigned char));
    }
    memcpy(s->best_solution, s->solution, (size_t)n * sizeof(unsigned char));

    for (int v = 0; v < n; ++v) {
        s->vars[v].score = 0.0;
        s->vars[v].last_flip_step = 0;
        s->vars[v].unsat_appear = 0;
        s->vars[v].cc_value = 1;
        s->vars[v].in_ccd = 0;
    }

    for (int ci = 0; ci < m; ++ci) {
        HessWLSClause *cp = &s->clauses[ci];
        for (int k = 0; k < cp->size; ++k) {
            int lit = cp->lits[k];
            int v = lit > 0 ? lit - 1 : -lit - 1;
            if (s->solution[v] == (lit > 0 ? 1u : 0u)) {
                cp->sat_count++;
                cp->sat_var = v;
            }
        }
        if (cp->sat_count == 0) {
            hessw_ls_unsat_clause(s, ci);
        }
    }

    for (int v = 0; v < n; ++v) {
        HessWLSVar *vp = &s->vars[v];
        for (int i = 0; i < vp->occs.size; ++i) {
            int ci = vp->occs.data[i].clause;
            HessWLSClause *cp = &s->clauses[ci];
            if (cp->sat_count == 0) {
                vp->score += cp->weight;
            } else if (cp->sat_count == 1 && vp->occs.data[i].sense == s->solution[v]) {
                vp->score -= cp->weight;
            }
        }
        if (vp->score > 0.0) {
            s->ccd_vars[s->ccd_count++] = v;
            vp->in_ccd = 1;
        }
    }

    s->best_unsat = s->unsat;
}

static void hessw_ls_refine(int n, int m, int *cls_sizes, int **cls_data, double *cls_weights,
                            const unsigned char *seed_model, unsigned char *best_model,
                            double total_w, double *best) {
    HessWLSState s;
    const double eps = 1e-12;

    hessw_ls_state_init(&s, n, m, cls_sizes, cls_data, cls_weights, seed_model);
    if (best != NULL) *best = s.unsat;
    if (s.unsat <= eps) {
        if (best_model != NULL) memcpy(best_model, s.best_solution, (size_t)n * sizeof(unsigned char));
        hessw_ls_state_free(&s);
        return;
    }

    for (s.step = 0; s.step < s.max_steps; ++s.step) {
        if (s.mems > s.max_mems) break;
        int flipv = hessw_ls_pick_var(&s);
        if (flipv < 0 || flipv >= n) continue;
        hessw_ls_flip(&s, flipv);
        if (s.unsat < s.best_unsat - eps) {
            s.best_unsat = s.unsat;
            memcpy(s.best_solution, s.solution, (size_t)n * sizeof(unsigned char));
            hessw_log_progress("LS", s.best_unsat, total_w);
            if (s.best_unsat <= eps) break;
        }
        if (s.unsat_clause_count == 0) {
            s.best_unsat = 0.0;
            memcpy(s.best_solution, s.solution, (size_t)n * sizeof(unsigned char));
            break;
        }
    }

    if (best_model != NULL) memcpy(best_model, s.best_solution, (size_t)n * sizeof(unsigned char));
    if (best != NULL) *best = s.best_unsat;
    hessw_ls_state_free(&s);
}

typedef struct {
    int n;
    int m;
    IntVec *pos;
    IntVec *neg;
} HessWOcc;

static void hessw_occ_free(HessWOcc *occ) {
    if (occ == NULL) return;
    if (occ->pos != NULL) {
        for (int v = 0; v < occ->n; ++v) {
            intvec_free(&occ->pos[v]);
        }
        free(occ->pos);
    }
    if (occ->neg != NULL) {
        for (int v = 0; v < occ->n; ++v) {
            intvec_free(&occ->neg[v]);
        }
        free(occ->neg);
    }
    memset(occ, 0, sizeof(*occ));
}

static int hessw_occ_init(HessWOcc *occ, int n, int m, int *cls_sizes, int **cls_data) {
    memset(occ, 0, sizeof(*occ));
    occ->n = n;
    occ->m = m;
    occ->pos = (IntVec *)xmalloc((size_t)n * sizeof(IntVec));
    occ->neg = (IntVec *)xmalloc((size_t)n * sizeof(IntVec));
    for (int v = 0; v < n; ++v) {
        intvec_init(&occ->pos[v]);
        intvec_init(&occ->neg[v]);
    }
    for (int ci = 0; ci < m; ++ci) {
        for (int k = 0; k < cls_sizes[ci]; ++k) {
            int lit = cls_data[ci][k];
            if (lit < 0) intvec_push(&occ->neg[-lit - 1], ci);
            else intvec_push(&occ->pos[lit - 1], ci);
        }
    }
    return 1;
}

static double hessw_search(int n, int m, int *cls_sizes, int **cls_data,
                           double *cls_weights, double total_w,
                           const HessWOcc *occ,
                           const unsigned char *seed_model, unsigned char *out_model,
                           int max_rounds, double stop_unsat) {
    const int hess_round_limit = (max_rounds > 0) ? max_rounds : INT_MAX;
    int hess_rounds = 0;
    const IntVec *pos = occ->pos;
    const IntVec *neg = occ->neg;

    int *sc = (int *)calloc((size_t)m, sizeof(int));
    unsigned char *sat = (unsigned char *)xmalloc((size_t)n);
    unsigned char *opt = (unsigned char *)xmalloc((size_t)n);
    if (seed_model != NULL) memcpy(sat, seed_model, (size_t)n);
    else memset(sat, 0, (size_t)n);
    memcpy(opt, sat, (size_t)n);

    double unsat = 0.0;

    hessw_eval_model(n, m, cls_sizes, cls_data, cls_weights, sat, sc, &unsat);

    double best = unsat;
    memcpy(opt, sat, (size_t)n);
    if (best < 1e-9) {
        memcpy(out_model, opt, (size_t)n);
        goto finally;
    }
    if (stop_unsat > 0.0 && best <= stop_unsat + 1e-9) {
        if (sat_log_enabled()) {
            printf("c HESS seed stop: unsat=%.0f/%.0f threshold=%.0f\n", best, total_w, stop_unsat);
            fflush(stdout);
        }
        memcpy(out_model, opt, (size_t)n);
        goto finally;
    }

    if (n > 96) {
        unsigned char *candidate = (unsigned char *)xmalloc((size_t)n);
        uint64_t rng = 0x9e3779b97f4a7c15ULL ^
                       ((uint64_t)(unsigned int)n << 32) ^
                       (uint64_t)(unsigned int)m;
        int attempts = 1;
        if (seed_model != NULL) attempts = 1;
        else if (n <= 256) attempts = (m > 3000) ? 8 : 60;
        else if (n <= 768) attempts = (m > 5000) ? 8 : 64;
        else attempts = 12;
        if (max_rounds > 0 && max_rounds < attempts) attempts = max_rounds;
        for (int attempt = 0; attempt < attempts; ++attempt) {
            double cand_best = DBL_MAX;
            if (attempt == 0) {
                if (seed_model != NULL) memcpy(sat, seed_model, (size_t)n);
                else memset(sat, 0, (size_t)n);
            } else if (best < total_w) {
                memcpy(sat, opt, (size_t)n);
                int flips = 1 + (int)((rng_next_u64(&rng) >> 32) % (uint32_t)(1 + (attempt < 24 ? attempt : 24)));
                for (int r = 0; r < flips; ++r) {
                    int v = (int)((rng_next_u64(&rng) >> 32) % (uint32_t)n);
                    sat[v] ^= 1u;
                }
            } else {
                uint32_t threshold;
                if ((attempt % 3) == 1) threshold = UINT32_MAX / 12u;
                else if ((attempt % 3) == 2) threshold = UINT32_MAX / 6u;
                else threshold = UINT32_MAX / 4u;
                for (int v = 0; v < n; ++v) {
                    sat[v] = (unsigned char)(((uint32_t)(rng_next_u64(&rng) >> 32)) < threshold ? 1u : 0u);
                }
            }
            hessw_ls_refine(n, m, cls_sizes, cls_data, cls_weights, sat, candidate, total_w, &cand_best);
            hessw_eval_model(n, m, cls_sizes, cls_data, cls_weights, candidate, sc, &unsat);
            cand_best = unsat;
            if (cand_best < best - 1e-9) {
                best = cand_best;
                memcpy(opt, candidate, (size_t)n);
                hessw_log_progress("LS", best, total_w);
                if (best < 1e-9) break;
                if (stop_unsat > 0.0 && best <= stop_unsat + 1e-9) break;
            }
        }
        if (n <= 512 && m <= 4000 && best > 1e-9) {
            memcpy(sat, opt, (size_t)n);
            hessw_eval_model(n, m, cls_sizes, cls_data, cls_weights, sat, sc, &unsat);
            best = unsat;
            for (int pass = 0; pass < 4; ++pass) {
                int improved = 0;
                for (int i = 0; i < n; ++i) {
                    hessw_flip(i, sat, sc, &unsat, pos, neg, cls_weights);
                    if (unsat < best - 1e-9) {
                        best = unsat;
                        improved = 1;
                        memcpy(opt, sat, (size_t)n);
                        hessw_log_progress("LS-1flip", best, total_w);
                        if (best < 1e-9) break;
                    } else {
                        hessw_flip(i, sat, sc, &unsat, pos, neg, cls_weights);
                    }
                }
                if (best < 1e-9) break;
                for (int i = 0; i < n; ++i) {
                    for (int j = i + 1; j < n; ++j) {
                        hessw_flip(i, sat, sc, &unsat, pos, neg, cls_weights);
                        hessw_flip(j, sat, sc, &unsat, pos, neg, cls_weights);
                        if (unsat < best - 1e-9) {
                            best = unsat;
                            improved = 1;
                            memcpy(opt, sat, (size_t)n);
                            hessw_log_progress("LS-2flip", best, total_w);
                            if (best < 1e-9) break;
                        } else {
                            hessw_flip(j, sat, sc, &unsat, pos, neg, cls_weights);
                            hessw_flip(i, sat, sc, &unsat, pos, neg, cls_weights);
                        }
                    }
                    if (best < 1e-9) break;
                }
                if (n <= 220 && best > 1e-9) {
                    for (int i = 0; i < n; ++i) {
                        for (int j = i + 1; j < n; ++j) {
                            for (int k = j + 1; k < n; ++k) {
                                hessw_flip(i, sat, sc, &unsat, pos, neg, cls_weights);
                                hessw_flip(j, sat, sc, &unsat, pos, neg, cls_weights);
                                hessw_flip(k, sat, sc, &unsat, pos, neg, cls_weights);
                                if (unsat < best - 1e-9) {
                                    best = unsat;
                                    improved = 1;
                                    memcpy(opt, sat, (size_t)n);
                                    hessw_log_progress("LS-3flip", best, total_w);
                                    if (best < 1e-9) break;
                                } else {
                                    hessw_flip(k, sat, sc, &unsat, pos, neg, cls_weights);
                                    hessw_flip(j, sat, sc, &unsat, pos, neg, cls_weights);
                                    hessw_flip(i, sat, sc, &unsat, pos, neg, cls_weights);
                                }
                            }
                            if (best < 1e-9) break;
                        }
                        if (best < 1e-9) break;
                    }
                }
                if (!improved || best < 1e-9) break;
            }
        }
        free(candidate);
        if (sat_log_enabled()) {
            printf("c LS verified: %.0f/%.0f\n", best, total_w);
            fflush(stdout);
        }
        memcpy(out_model, opt, (size_t)n);
        //goto finally;
    }

    double glb = total_w + 1.0;
        for (;;) {          
            int done = 1;   
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    //for (int k = 0; k < n; ++k) {
                        hessw_flip(i, sat, sc, &unsat, pos, neg, cls_weights);
                        hessw_flip(j, sat, sc, &unsat, pos, neg, cls_weights);
                        //hessw_flip(k, sat, sc, &unsat, pos, neg, cls_weights);
                        if (unsat < glb) {                       
                            glb = unsat;
                            if (glb < best) {
                                best = glb;
                                done = 0;           
                                memcpy(opt, sat, (size_t)n);
                                hessw_log_progress("HESS", best, total_w);
                                if (best < 1e-9) {
                                    memcpy(out_model, opt, (size_t)n);
                                    goto finally;
                                }
                                if (stop_unsat > 0.0 && best <= stop_unsat + 1e-9) {
                                    if (sat_log_enabled()) {
                                        printf("c HESS seed stop: unsat=%.0f/%.0f threshold=%.0f\n", best, total_w, stop_unsat);
                                        fflush(stdout);
                                    }
                                    memcpy(out_model, opt, (size_t)n);
                                    goto finally;
                                }
                            }                        
                        } else if (unsat > glb) {
                            //hessw_flip(k, sat, sc, &unsat, pos, neg, cls_weights);  
                            hessw_flip(j, sat, sc, &unsat, pos, neg, cls_weights);
                            hessw_flip(i, sat, sc, &unsat, pos, neg, cls_weights);                        
                        }
                    //}
                }
            }
            if (best < 1e-9) break;
            if (stop_unsat > 0.0 && best <= stop_unsat + 1e-9) {
                memcpy(out_model, opt, (size_t)n);
                goto finally;
            }
            hessw_ls_refine(n, m, cls_sizes, cls_data, cls_weights, opt, opt, total_w, &best);
            hessw_eval_model(n, m, cls_sizes, cls_data, cls_weights, opt, sc, &unsat);
            best = unsat;
            memcpy(sat, opt, (size_t)n);                   
            if (stop_unsat > 0.0 && best <= stop_unsat + 1e-9) {
                if (sat_log_enabled()) {
                    printf("c HESS seed stop: unsat=%.0f/%.0f threshold=%.0f\n", best, total_w, stop_unsat);
                    fflush(stdout);
                }
                memcpy(out_model, opt, (size_t)n);
                goto finally;
            }
            if (++hess_rounds >= hess_round_limit) break;
            if (done) break;
    }
const double eps = 1e-9;
    long long max_steps = (max_rounds > 0) ? (long long)max_rounds : LLONG_MAX;
    long long steps = 0;
    //int *sc = (int *)xmalloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    //unsigned char *sat = (unsigned char *)xmalloc((size_t)(n > 0 ? n : 1));
    //unsigned char *opt = (unsigned char *)xmalloc((size_t)(n > 0 ? n : 1));
    double cur = 0.0;
    //double best = 0.0;

    (void)occ;

    if (seed_model != NULL) {
        memcpy(sat, seed_model, (size_t)n);
    } else {
        memset(sat, 0, (size_t)n);
    }
    memcpy(opt, sat, (size_t)n);

    hessw_eval_model(n, m, cls_sizes, cls_data, cls_weights, sat, sc, &cur);
    best = cur;
    memcpy(opt, sat, (size_t)n);

    if (best < eps) {
        memcpy(out_model, opt, (size_t)n);
        free(sc);
        free(sat);
        free(opt);
        return best;
    }
    if (stop_unsat > 0.0 && best <= stop_unsat + eps) {
        if (sat_log_enabled()) {
            printf("c HESS seed stop: unsat=%.0f/%.0f threshold=%.0f\n", best, total_w, stop_unsat);
            fflush(stdout);
        }
        memcpy(out_model, opt, (size_t)n);
        free(sc);
        free(sat);
        free(opt);
        return best;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double glb = 0.0;
            int has_glb = 0;
            hessw_ls_refine(n, m, cls_sizes, cls_data, cls_weights, opt, opt, total_w, &best);
            for (int k = 0; k < n; k++) {
                for (;;) {                
                    int done = 1;
                    double loc = 0.0;
                    int cmp_glb;
                    unsigned char old_sat_k;
                    unsigned char tmp;

                    if (steps >= max_steps) {
                        memcpy(out_model, opt, (size_t)n);
                        free(sc);
                        free(sat);
                        free(opt);
                        return best;
                    }

                    old_sat_k = sat[k];
                    tmp ^= sat[i];
                    sat[i] = sat[j];
                    sat[j] = tmp;
                    sat[k] ^= 1u;
                    steps++;

                    hessw_eval_model(n, m, cls_sizes, cls_data, cls_weights, sat, sc, &loc);
                    if (!has_glb) {
                        cmp_glb = -1;
                    } else if (loc < glb - eps) {
                        cmp_glb = -1;
                    } else if (loc > glb + eps) {
                        cmp_glb = 1;
                    } else {
                        cmp_glb = 0;
                    }

                    if (cmp_glb < 0) {
                        glb = loc;
                        has_glb = 1;
                        if (glb < cur - eps) {
                            done = 0;
                            cur = glb;
                            best = cur;
                            memcpy(opt, sat, (size_t)n);
                            hessw_log_progress("HESS", best, total_w);
                            if (best < eps) {
                                memcpy(out_model, opt, (size_t)n);
                                free(sc);
                                free(sat);
                                free(opt);
                                return best;
                            }
                            if (stop_unsat > 0.0 && best <= stop_unsat + eps) {
                                if (sat_log_enabled()) {
                                    printf("c HESS seed stop: unsat=%.0f/%.0f threshold=%.0f\n", best, total_w, stop_unsat);
                                    fflush(stdout);
                                }
                                memcpy(out_model, opt, (size_t)n);
                                free(sc);
                                free(sat);
                                free(opt);
                                return best;
                            }
                        }
                    } else if (cmp_glb > 0) {
                        sat[k] = old_sat_k;
                    }

                    if (done) {
                        break;
                    }
                }
            }
        }
    }
    memcpy(out_model, opt, (size_t)n);        
finally:
    free(sc); free(sat); free(opt);
    return best;
}

/* ================================================================
 * WCNF parser + MaxSAT solve
 * ================================================================ */
typedef struct {
    int nv, nc;
    int *sizes;
    int **cls;
    double *weights;
    double tw, top;
    int owns;
    int is_wcnf;
} WCNF;

static int fi_read_double(FastInput *in, double *out) {
    int c;
    do { c = fi_getc(in); if (c == EOF) return 0; }
    while (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v');
    int sign = 1; if (c == '-') { sign = -1; c = fi_getc(in); }
    if (c < '0' || c > '9') return 0;
    double v = (double)(c - '0');
    for (;;) { c = fi_getc(in); if (c < '0' || c > '9') break; v = v * 10.0 + (double)(c - '0'); }
    if (c == '.') { double f = 0.0, d = 10.0; for (;;) { c = fi_getc(in); if (c < '0' || c > '9') break; f += (double)(c - '0') / d; d *= 10.0; } v += f; }
    if (c != EOF) fi_ungetc(in, c); *out = sign * v; return 1;
}

static int wcnf_parse(const char *path, WCNF *wp) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "c ERROR: cannot open '%s'\n", path); return 0; }
    FastInput in; fi_init(&in, fp); memset(wp, 0, sizeof(*wp));
    int hdr = 0, bol = 1, bc = 0, bs = 0, *bf = NULL, wcnf = 0, ci = 0;
    for (;;) {
        int c = fi_getc(&in); if (c == EOF) break;
        if (c == '\n') { bol = 1; continue; }
        if (c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v') continue;
        if (bol && c == 'c') { fi_skip_line(&in); bol = 1; continue; }
        if (bol && c == 'p') {
            char w[16]; if (!fi_read_word(&in, w, (int)sizeof(w))) goto fail;
            if (strcmp(w, "cnf") == 0) { int d; if (!fi_read_int_token(&in,&wp->nv)||!fi_read_int_token(&in,&d)||wp->nv<=0) goto fail; wp->nc=d; wcnf=0; wp->is_wcnf = 0; }
            else if (strcmp(w, "wcnf") == 0) { int d; if (!fi_read_int_token(&in,&wp->nv)||!fi_read_int_token(&in,&d)||wp->nv<=0) goto fail;
                if (!fi_read_double(&in, &wp->top)) goto fail; wp->nc = d; wcnf = 1; wp->is_wcnf = 1; }
            else { fprintf(stderr, "c ERROR: bad header\n"); goto fail; }
            wp->sizes = (int*)calloc((size_t)wp->nc, sizeof(int));
            wp->cls = (int**)calloc((size_t)wp->nc, sizeof(int*));
            wp->weights = (double*)calloc((size_t)wp->nc, sizeof(double));
            if (!wp->sizes||!wp->cls||!wp->weights) goto fail;
            wp->owns = 1;
            hdr = 1; fi_skip_line(&in); bol = 1; continue;
        }
        if (!hdr) { fprintf(stderr, "c ERROR: data before header\n"); goto fail; }
        bol = 0; fi_ungetc(&in, c);
        if (wcnf && bs == 0) { if (!fi_read_double(&in, &wp->weights[ci])) goto fail; wp->tw += wp->weights[ci]; }
        int lit; if (!fi_read_int_token(&in, &lit)) goto fail;
        if (lit == 0) { if (ci >= wp->nc) goto fail; wp->sizes[ci]=bs; if (bs>0) { wp->cls[ci]=(int*)xmalloc((size_t)bs*sizeof(int)); memcpy(wp->cls[ci],bf,(size_t)bs*sizeof(int)); } bs=0; ci++;
            if (!wcnf) { wp->tw += 1.0; } }
        else { int v=lit>0?lit:-lit; if (v<1||v>wp->nv) goto fail; if (bs>=bc) { bc=bc?bc<<1:64; bf=(int*)realloc(bf,(size_t)bc*sizeof(int)); } bf[bs++]=lit; }
    }
    if (!hdr || bs != 0 || ci != wp->nc) goto fail;
    free(bf); fi_free(&in); fclose(fp);
    if (!wcnf) { for (int i = 0; i < wp->nc; i++) wp->weights[i] = 1.0; wp->tw = (double)wp->nc; }
    return 1;
fail: free(bf); fi_free(&in); fclose(fp);
    if (wp->owns) { for (int i=0;i<ci;i++) free(wp->cls[i]); }
    free(wp->cls);
    free(wp->sizes);
    free(wp->weights);
    memset(wp,0,sizeof(*wp));
    return 0;
}

static void wcnf_free(WCNF *wp) { if(!wp)return; if (wp->owns) { for(int i=0;i<wp->nc;i++) free(wp->cls[i]); free(wp->cls); free(wp->sizes); free(wp->weights); } memset(wp,0,sizeof(*wp)); }

static int wcnf_soft_unit_profile(const WCNF *wp, int *soft_units_out, double *min_soft_out) {
    if (wp->top <= 0.0) return 0;
    int soft_units = 0;
    double min_soft = DBL_MAX;
    for (int i = 0; i < wp->nc; ++i) {
        if (wp->weights[i] + 1e-12 >= wp->top) continue;
        if (wp->sizes[i] != 1) return 0;
        ++soft_units;
        if (wp->weights[i] < min_soft) min_soft = wp->weights[i];
    }
    if (soft_units <= 0 || min_soft <= 0.0) return 0;
    if (soft_units_out != NULL) *soft_units_out = soft_units;
    if (min_soft_out != NULL) *min_soft_out = min_soft;
    return 1;
}

static double wcnf_partial_unsat_lower_bound(const WCNF *wp, const Solver *s) {
    double lb = 0.0;
    for (int ci = 0; ci < wp->nc; ++ci) {
        int sat = 0;
        int undecided = 0;
        for (int k = 0; k < wp->sizes[ci]; ++k) {
            int lit = wp->cls[ci][k];
            int v = lit > 0 ? lit - 1 : -lit - 1;
            int a = s->assigns[v];
            if (a == 0) {
                undecided = 1;
                continue;
            }
            if ((a > 0) == (lit > 0)) {
                sat = 1;
                break;
            }
        }
        if (!sat && !undecided) lb += wp->weights[ci];
    }
    return lb;
}

static void wcnf_build_full_block_clause(IntVec *clause, const Solver *s) {
    clause->size = 0;
    for (int i = 0; i < s->trail_size; ++i) {
        intvec_push(clause, lit_neg(s->trail[i]));
    }
}

static void wcnf_build_trail_block_clause(IntVec *clause, const Solver *s) {
    clause->size = 0;
    for (int i = 0; i < s->trail_lim.size; ++i) {
        int start = s->trail_lim.data[i];
        if (start >= 0 && start < s->trail_size) {
            intvec_push(clause, lit_neg(s->trail[start]));
        }
    }
}

/* ================================================================
 * TargetGuidedCDCL: HESS → CDCL target-guided with HESS↔CDCL recomputation
 * ================================================================ */
typedef struct {
    signed char *target;
    double *pressure;
    double *escape;
    int nvars;
} TargetGuide;

typedef struct {
    signed char *model;
    double best_score;
    double satisfied_weight;
    int matches;
    int mismatches;
    int unassigned;
} TargetSnapshot;

static const double TARGET_GUIDE_MATCH_WEIGHT = 0.75;
static const double TARGET_GUIDE_MISMATCH_PENALTY = 0.50;
static const double TARGET_GUIDE_UNASSIGNED_PENALTY = 0.25;

static void target_model_from_assigns(unsigned char *dst, const Solver *s, int nvars) {
    for (int v = 0; v < nvars; ++v) {
        dst[v] = (unsigned char)(s->assigns[v] > 0 ? 1u : 0u);
    }
}

static double target_clause_satisfied_weight(const WCNF *wp, const Solver *s) {
    double sat_weight = 0.0;
    for (int ci = 0; ci < wp->nc; ++ci) {
        int sat = 0;
        for (int k = 0; k < wp->sizes[ci]; ++k) {
            int lit = wp->cls[ci][k];
            int v = lit > 0 ? lit - 1 : -lit - 1;
            int a = s->assigns[v];
            if (a != 0 && ((a > 0) == (lit > 0))) {
                sat = 1;
                break;
            }
        }
        if (sat) sat_weight += wp->weights[ci];
    }
    return sat_weight;
}

static double target_state_score(const Solver *s, const TargetGuide *g, const WCNF *wp,
                                 int *matches_out, int *mismatches_out, int *unassigned_out,
                                 double *satisfied_weight_out) {
    int matches = 0;
    int mismatches = 0;
    int unassigned = 0;
    for (int v = 0; v < g->nvars; ++v) {
        int a = s->assigns[v];
        if (a == 0) {
            ++unassigned;
            continue;
        }
        int t = g->target[v];
        if (t == 0) continue;
        if (a == t) ++matches;
        else ++mismatches;
    }
    double sat_weight = target_clause_satisfied_weight(wp, s);
    if (matches_out) *matches_out = matches;
    if (mismatches_out) *mismatches_out = mismatches;
    if (unassigned_out) *unassigned_out = unassigned;
    if (satisfied_weight_out) *satisfied_weight_out = sat_weight;
    return TARGET_GUIDE_MATCH_WEIGHT * (double)matches -
           TARGET_GUIDE_MISMATCH_PENALTY * (double)mismatches -
           TARGET_GUIDE_UNASSIGNED_PENALTY * (double)unassigned +
           sat_weight;
}

static int target_snapshot_better(double score,
                                  int matches,
                                  int mismatches,
                                  int unassigned,
                                  double satisfied_weight,
                                  const TargetSnapshot *sn) {
    const double eps = 1e-12;
    if (score > sn->best_score + eps) return 1;
    if (score + eps < sn->best_score) return 0;
    if (matches != sn->matches) return matches > sn->matches;
    if (mismatches != sn->mismatches) return mismatches < sn->mismatches;
    if (unassigned != sn->unassigned) return unassigned < sn->unassigned;
    if (satisfied_weight != sn->satisfied_weight) return satisfied_weight > sn->satisfied_weight;
    return 0;
}

static void target_snapshot_init(TargetSnapshot *sn, const Solver *s, const TargetGuide *g, const WCNF *wp) {
    memset(sn, 0, sizeof(*sn));
    sn->best_score = target_state_score(s, g, wp,
                                        &sn->matches,
                                        &sn->mismatches,
                                        &sn->unassigned,
                                        &sn->satisfied_weight);
}

static int target_snapshot_update(const Solver *s, TargetSnapshot *sn, const TargetGuide *g, const WCNF *wp) {
    int matches = 0;
    int mismatches = 0;
    int unassigned = 0;
    double satisfied_weight = 0.0;
    double score = target_state_score(s, g, wp,
                                      &matches,
                                      &mismatches,
                                      &unassigned,
                                      &satisfied_weight);
    if (!target_snapshot_better(score, matches, mismatches, unassigned, satisfied_weight, sn)) {
        return 0;
    }

    sn->best_score = score;
    sn->matches = matches;
    sn->mismatches = mismatches;
    sn->unassigned = unassigned;
    sn->satisfied_weight = satisfied_weight;
    if (sat_log_enabled()) {
        printf("c CDCL improve: unsat=%.0f/%.0f score=%.2f matches=%d mismatches=%d unassigned=%d decisions=%lld conflicts=%lld\n",
               wp->tw - satisfied_weight,
               wp->tw,
               score,
               matches,
               mismatches,
               unassigned,
               (long long)s->decisions,
               (long long)s->conflicts);
        fflush(stdout);
    }
    return 1;
}

static int target_guide_value(const TargetGuide *g, int v) {
    int val = (g->target[v] > 0) ? 1 : -1;
    if (g->pressure != NULL && g->escape != NULL) {
        double pressure = g->pressure[v];
        double escape = g->escape[v];
        if (escape > 3.0 + 1.35 * pressure) val = -val;
    }
    return val;
}

static void target_recompute_pressure(const WCNF *wp, TargetGuide *g, const unsigned char *model) {
    if (g->pressure == NULL) return;
    memset(g->pressure, 0, (size_t)g->nvars * sizeof(double));
    for (int ci = 0; ci < wp->nc; ++ci) {
        int sz = wp->sizes[ci];
        if (sz <= 0) continue;
        int sat = 0;
        double weight = wp->weights[ci];
        double base = (weight > 0.0 ? weight : 0.0) / (double)sz;
        for (int k = 0; k < sz; ++k) {
            int lit = wp->cls[ci][k];
            int v = lit > 0 ? lit - 1 : -lit - 1;
            int lit_true = ((model[v] ? 1 : 0) == (lit > 0 ? 1 : 0));
            if (lit_true) {
                sat = 1;
                break;
            }
        }
        for (int k = 0; k < sz; ++k) {
            int lit = wp->cls[ci][k];
            int v = lit > 0 ? lit - 1 : -lit - 1;
            int lit_true = ((model[v] ? 1 : 0) == (lit > 0 ? 1 : 0));
            double bump = sat ? (lit_true ? base : 0.10 * base) : 8.0 * base;
            g->pressure[v] += bump;
        }
    }
}

static void target_apply_pressure(Solver *s, const TargetGuide *g) {
    if (g->pressure == NULL) return;
    for (int v = 0; v < g->nvars; ++v) {
        double escape = g->escape != NULL ? g->escape[v] : 0.0;
        double bump = 0.20 * g->pressure[v] + 0.45 * escape;
        if (bump <= 0.0) continue;
        s->activity[v] += bump;
        if (s->chb_activity != NULL) s->chb_activity[v] += 0.5 * bump;
    }
    heap_rebuild(&s->order);
}

static void target_absorb_conflict(TargetGuide *g, Solver *s, const IntVec *learnt) {
    if (g->escape == NULL || learnt->size <= 0) return;
    double inv = 1.0 / (double)learnt->size;
    int rescale = 0;
    for (int i = 0; i < learnt->size; ++i) {
        Lit p = (Lit)learnt->data[i];
        int v = lit_var(p);
        if (v < 0 || v >= g->nvars) continue;
        int lit_value = lit_asgn_value(p);
        double bump = (g->target[v] == lit_value) ? 0.25 * inv : 2.0 * inv;
        g->escape[v] += bump;
        if (g->escape[v] > 1000.0) rescale = 1;
        s->activity[v] += 2.0 * bump;
        if (s->chb_activity != NULL) s->chb_activity[v] += bump;
        heap_update(&s->order, v);
    }
    if (rescale) {
        for (int v = 0; v < g->nvars; ++v) g->escape[v] *= 0.01;
    }
}

static void target_guide_set_model(TargetGuide *g,
                                   Solver *s,
                                   const WCNF *wp,
                                   const unsigned char *model,
                                   TargetSnapshot *progress) {
    for (int v = 0; v < wp->nv; ++v) {
        g->target[v] = model[v] ? 1 : -1;
        s->phases[v] = model[v];
    }
    target_recompute_pressure(wp, g, model);
    target_apply_pressure(s, g);
    if (progress != NULL) {
        target_snapshot_init(progress, s, g, wp);
    }
}

static void target_build_seed(unsigned char *seed, const Solver *s, const TargetGuide *g) {
    for (int v = 0; v < g->nvars; ++v) {
        if (s->assigns[v] != 0) {
            seed[v] = (unsigned char)(s->assigns[v] > 0 ? 1u : 0u);
        } else {
            seed[v] = (unsigned char)(target_guide_value(g, v) > 0 ? 1u : 0u);
        }
    }
}

static double target_full_unsat_weight(const WCNF *wp, const unsigned char *model);

static int target_refresh_round_budget(const WCNF *wp, double best_unsat, int refresh_count) {
    const double eps = 1e-9;
    if (slime_hess_max_iter > 0) return slime_hess_max_iter;
    if (best_unsat <= eps) return 0;
    if (refresh_count > 0) {
        double near_zero = wp->tw * 0.002;
        if (near_zero < 1.0) near_zero = 1.0;
        if (best_unsat <= near_zero) return 0;
    }
    return 1;
}

static double target_convergence_zone(int soft_units, double min_soft_weight) {
    (void)soft_units;
    (void)min_soft_weight;
    /* Keep the early-stop logic conservative; do not shortcut the solver path. */
    return 0.0;
}

static int target_accept_converged_incumbent(double best_unsat,
                                             double convergence_zone,
                                             int convergence_hits,
                                             double last_delta,
                                             long long decisions_since_best,
                                             long long conflicts_since_best) {
    if (best_unsat > convergence_zone) return 0;
    if (convergence_hits < 4) return 0;
    double delta_floor = best_unsat * 0.001;
    if (delta_floor < 4.0) delta_floor = 4.0;
    if (last_delta > delta_floor) return 0;
    return decisions_since_best >= 20000 || conflicts_since_best >= 32;
}

static int target_refresh_from_state(const WCNF *wp,
                                     Solver *s,
                                     TargetGuide *guide,
                                     TargetSnapshot *progress,
                                     const HessWOcc *occ,
                                     unsigned char *best_model,
                                     double *best_unsat,
                                     unsigned char *seed,
                                     unsigned char *work_model,
                                     int refresh_rounds,
                                     int *exact_found) {
    const double eps = 1e-9;
    if (exact_found != NULL) *exact_found = 0;
    target_build_seed(seed, s, guide);
    double candidate_unsat;
    if (refresh_rounds > 0) {
        candidate_unsat = hessw_search(wp->nv, wp->nc, wp->sizes, wp->cls, wp->weights, wp->tw,
                                        occ, seed, work_model, refresh_rounds, 0.0);
    } else {
        memcpy(work_model, seed, (size_t)wp->nv);
        candidate_unsat = target_full_unsat_weight(wp, work_model);
    }
    if (candidate_unsat > *best_unsat + eps) return 0;
    if (candidate_unsat < *best_unsat - eps) {
        hessw_log_progress("HESS", candidate_unsat, wp->tw);
    }
    if (candidate_unsat < *best_unsat - eps || memcmp(work_model, best_model, (size_t)wp->nv) != 0) {
        memcpy(best_model, work_model, (size_t)wp->nv);
        *best_unsat = candidate_unsat;
        target_guide_set_model(guide, s, wp, best_model, progress);
        if (exact_found != NULL && candidate_unsat <= eps) *exact_found = 1;
        return 1;
    }
    if (exact_found != NULL && candidate_unsat <= eps) *exact_found = 1;
    return 0;
}

static double target_full_unsat_weight(const WCNF *wp, const unsigned char *model) {
    double wu = 0.0;
    for (int i = 0; i < wp->nc; ++i) {
        int ok = 0;
        for (int k = 0; k < wp->sizes[i]; ++k) {
            int lit = wp->cls[i][k];
            int v = lit > 0 ? lit - 1 : -lit - 1;
            if ((model[v] ? 1 : 0) == (lit > 0 ? 1 : 0)) {
                ok = 1;
                break;
            }
        }
        if (!ok) wu += wp->weights[i];
    }
    return wu;
}

static Lit target_pick_branch(Solver *s, const TargetGuide *g) {
    for (;;) {
        int v = heap_pop_max(&s->order);
        if (v < 0) return -1;
        if (s->assigns[v] != 0) continue;

        int sign = s->phases[v] ? 0 : 1;
        if (v < g->nvars && g->target[v] != 0) {
            sign = (target_guide_value(g, v) > 0) ? 0 : 1;
            if (s->chb_last_conflict != NULL && s->chb_activity != NULL) {
                long long age = s->conflicts - s->chb_last_conflict[v];
                if (age >= 0 && age <= 4 && s->chb_activity[v] > 1.0) {
                    sign ^= 1;
                }
            }
        }
        return mk_lit(v, sign);
    }
}

typedef struct {
    double *cost;
    int soft_count;
} MinOnesProfile;

typedef struct {
    double *pos_weight;
    double *neg_weight;
    int soft_count;
    int pos_count;
    int neg_count;
} UnitSoftProfile;

typedef struct {
    int lit;
    double gain;
} UnitSoftItem;

static int unitsoft_sort_mode;

static int unitsoft_model_has_lit(const unsigned char *model, int lit) {
    int v = lit > 0 ? lit - 1 : -lit - 1;
    return model[v] == (lit > 0 ? 1u : 0u);
}

static void unitsoft_model_set_lit(unsigned char *model, int lit) {
    int v = lit > 0 ? lit - 1 : -lit - 1;
    model[v] = (unsigned char)(lit > 0 ? 1u : 0u);
}

static void unitsoft_model_clear_lit(unsigned char *model, int lit) {
    int v = lit > 0 ? lit - 1 : -lit - 1;
    model[v] = (unsigned char)(lit > 0 ? 0u : 1u);
}

static int minones_profile_init(const WCNF *wp, MinOnesProfile *mo) {
    const double eps = 1e-12;
    memset(mo, 0, sizeof(*mo));
    if (wp->top <= 0.0) return 0;
    mo->cost = (double *)xmalloc((size_t)wp->nv * sizeof(double));
    memset(mo->cost, 0, (size_t)wp->nv * sizeof(double));
    for (int ci = 0; ci < wp->nc; ++ci) {
        if (wp->weights[ci] + eps >= wp->top) continue;
        if (wp->sizes[ci] != 1 || wp->cls[ci][0] >= 0) {
            free(mo->cost);
            memset(mo, 0, sizeof(*mo));
            return 0;
        }
        int v = -wp->cls[ci][0] - 1;
        if (v < 0 || v >= wp->nv) {
            free(mo->cost);
            memset(mo, 0, sizeof(*mo));
            return 0;
        }
        mo->cost[v] += wp->weights[ci];
        mo->soft_count++;
    }
    if (mo->soft_count <= 0) {
        free(mo->cost);
        memset(mo, 0, sizeof(*mo));
        return 0;
    }
    return 1;
}

static void minones_profile_free(MinOnesProfile *mo) {
    if (mo == NULL) return;
    free(mo->cost);
    memset(mo, 0, sizeof(*mo));
}

static int unitsoft_profile_init(const WCNF *wp, UnitSoftProfile *us) {
    const double eps = 1e-12;
    memset(us, 0, sizeof(*us));
    if (wp->top <= 0.0) return 0;
    us->pos_weight = (double *)xmalloc((size_t)wp->nv * sizeof(double));
    us->neg_weight = (double *)xmalloc((size_t)wp->nv * sizeof(double));
    memset(us->pos_weight, 0, (size_t)wp->nv * sizeof(double));
    memset(us->neg_weight, 0, (size_t)wp->nv * sizeof(double));
    for (int ci = 0; ci < wp->nc; ++ci) {
        if (wp->weights[ci] + eps >= wp->top) continue;
        if (wp->sizes[ci] != 1) {
            free(us->pos_weight);
            free(us->neg_weight);
            memset(us, 0, sizeof(*us));
            return 0;
        }
        int lit = wp->cls[ci][0];
        int v = lit > 0 ? lit - 1 : -lit - 1;
        if (v < 0 || v >= wp->nv) {
            free(us->pos_weight);
            free(us->neg_weight);
            memset(us, 0, sizeof(*us));
            return 0;
        }
        if (lit > 0) {
            us->pos_weight[v] += wp->weights[ci];
            us->pos_count++;
        } else {
            us->neg_weight[v] += wp->weights[ci];
            us->neg_count++;
        }
        us->soft_count++;
    }
    if (us->soft_count <= 0) {
        free(us->pos_weight);
        free(us->neg_weight);
        memset(us, 0, sizeof(*us));
        return 0;
    }
    return 1;
}

static void unitsoft_profile_free(UnitSoftProfile *us) {
    if (us == NULL) return;
    free(us->pos_weight);
    free(us->neg_weight);
    memset(us, 0, sizeof(*us));
}

static unsigned char unitsoft_preferred_value(const UnitSoftProfile *us, int v) {
    /* If x=true, negative unit clauses are unsatisfied. If x=false, positive units are unsatisfied. */
    return (us->neg_weight[v] <= us->pos_weight[v]) ? 1u : 0u;
}

static int unitsoft_item_cmp_desc(const void *a, const void *b) {
    const UnitSoftItem *x = (const UnitSoftItem *)a;
    const UnitSoftItem *y = (const UnitSoftItem *)b;
    if (unitsoft_sort_mode == 2 && (x->lit > 0) != (y->lit > 0)) {
        return (x->lit > 0) ? -1 : 1;
    }
    if (unitsoft_sort_mode == 3 && (x->lit > 0) != (y->lit > 0)) {
        return (x->lit < 0) ? -1 : 1;
    }
    if (unitsoft_sort_mode == 4) {
        uint32_t hx = (uint32_t)(x->lit > 0 ? x->lit : -x->lit) * 2654435761u;
        uint32_t hy = (uint32_t)(y->lit > 0 ? y->lit : -y->lit) * 2654435761u;
        if (hx < hy) return -1;
        if (hx > hy) return 1;
    }
    if (unitsoft_sort_mode == 1) {
        if (x->gain < y->gain) return -1;
        if (x->gain > y->gain) return 1;
    } else {
        if (x->gain > y->gain) return -1;
        if (x->gain < y->gain) return 1;
    }
    int ax = x->lit > 0 ? x->lit : -x->lit;
    int ay = y->lit > 0 ? y->lit : -y->lit;
    return ax - ay;
}

#define UNITSOFT_MWIS_MAX 128

typedef struct {
    int n;
    const UnitSoftItem *items;
    uint64_t compat_lo[UNITSOFT_MWIS_MAX];
    uint64_t compat_hi[UNITSOFT_MWIS_MAX];
    unsigned char cur[UNITSOFT_MWIS_MAX];
    unsigned char best_sel[UNITSOFT_MWIS_MAX];
    double best_weight;
    long long nodes;
    long long node_limit;
    double deadline;
    int aborted;
} UnitSoftMWISCtx;

static uint64_t unitsoft_mwis_bit_lo(int idx) {
    return idx < 64 ? (UINT64_C(1) << idx) : 0;
}

static uint64_t unitsoft_mwis_bit_hi(int idx) {
    return idx >= 64 ? (UINT64_C(1) << (idx - 64)) : 0;
}

static int unitsoft_mwis_compatible(const UnitSoftMWISCtx *ctx, int a, int b) {
    if (b < 64) return (ctx->compat_lo[a] & (UINT64_C(1) << b)) != 0;
    return (ctx->compat_hi[a] & (UINT64_C(1) << (b - 64))) != 0;
}

static void unitsoft_mwis_sort_by_weight(const UnitSoftMWISCtx *ctx, int *a, int n) {
    for (int i = 1; i < n; ++i) {
        int v = a[i];
        int j = i - 1;
        while (j >= 0) {
            int u = a[j];
            if (ctx->items[u].gain > ctx->items[v].gain) break;
            if (ctx->items[u].gain == ctx->items[v].gain && u < v) break;
            a[j + 1] = u;
            j--;
        }
        a[j + 1] = v;
    }
}

static int unitsoft_mwis_color_bound(const UnitSoftMWISCtx *ctx,
                                     const int *cands,
                                     int n,
                                     int *order,
                                     double *bounds) {
    int uncolored[UNITSOFT_MWIS_MAX];
    int rem[UNITSOFT_MWIS_MAX];
    int cls[UNITSOFT_MWIS_MAX];
    int un = n;
    int out = 0;
    double cumulative = 0.0;

    for (int i = 0; i < n; ++i) uncolored[i] = cands[i];
    while (un > 0) {
        int remn = 0;
        int clsn = 0;
        uint64_t class_lo = 0;
        uint64_t class_hi = 0;
        double class_weight = 0.0;

        for (int i = 0; i < un; ++i) {
            int v = uncolored[i];
            if ((class_lo & ctx->compat_lo[v]) == 0 &&
                (class_hi & ctx->compat_hi[v]) == 0) {
                cls[clsn++] = v;
                class_lo |= unitsoft_mwis_bit_lo(v);
                class_hi |= unitsoft_mwis_bit_hi(v);
                if (ctx->items[v].gain > class_weight) {
                    class_weight = ctx->items[v].gain;
                }
            } else {
                rem[remn++] = v;
            }
        }

        cumulative += class_weight;
        for (int i = 0; i < clsn; ++i) {
            order[out] = cls[i];
            bounds[out] = cumulative;
            out++;
        }
        for (int i = 0; i < remn; ++i) uncolored[i] = rem[i];
        un = remn;
    }
    return out;
}

static void unitsoft_mwis_note_best(UnitSoftMWISCtx *ctx, double weight) {
    if (weight <= ctx->best_weight + 1e-9) return;
    ctx->best_weight = weight;
    memcpy(ctx->best_sel, ctx->cur, (size_t)ctx->n * sizeof(unsigned char));
}

static void unitsoft_mwis_expand(UnitSoftMWISCtx *ctx,
                                 const int *cands,
                                 int n,
                                 double weight) {
    int order[UNITSOFT_MWIS_MAX];
    int next[UNITSOFT_MWIS_MAX];
    double bounds[UNITSOFT_MWIS_MAX];

    if (ctx->aborted) return;
    ctx->nodes++;
    if (ctx->nodes > ctx->node_limit || now_sec() > ctx->deadline) {
        ctx->aborted = 1;
        return;
    }
    if (n <= 0) {
        unitsoft_mwis_note_best(ctx, weight);
        return;
    }

    n = unitsoft_mwis_color_bound(ctx, cands, n, order, bounds);
    for (int i = n - 1; i >= 0; --i) {
        int v;
        int nextn = 0;
        double next_weight;

        if (ctx->aborted) return;
        if (weight + bounds[i] <= ctx->best_weight + 1e-9) return;

        v = order[i];
        for (int j = 0; j < i; ++j) {
            int u = order[j];
            if (unitsoft_mwis_compatible(ctx, v, u)) {
                next[nextn++] = u;
            }
        }
        unitsoft_mwis_sort_by_weight(ctx, next, nextn);

        ctx->cur[v] = 1;
        next_weight = weight + ctx->items[v].gain;
        unitsoft_mwis_note_best(ctx, next_weight);
        unitsoft_mwis_expand(ctx, next, nextn, next_weight);
        ctx->cur[v] = 0;
    }
}

static int unitsoft_try_mwis_exact(const WCNF *wp,
                                   const UnitSoftProfile *us,
                                   const UnitSoftItem *items,
                                   int item_count,
                                   unsigned char *model,
                                   double *cost,
                                   int *picked,
                                   long long *nodes,
                                   int *exact) {
    UnitSoftMWISCtx ctx;
    int *var_to_item;
    uint64_t conflict_lo[UNITSOFT_MWIS_MAX];
    uint64_t conflict_hi[UNITSOFT_MWIS_MAX];
    int cands[UNITSOFT_MWIS_MAX];
    const double eps = 1e-9;

    if (item_count <= 0 || item_count > UNITSOFT_MWIS_MAX) return 0;

    var_to_item = (int *)xmalloc((size_t)wp->nv * sizeof(int));
    for (int v = 0; v < wp->nv; ++v) var_to_item[v] = -1;
    for (int i = 0; i < item_count; ++i) {
        int lit = items[i].lit;
        int v = lit > 0 ? lit - 1 : -lit - 1;
        if (v < 0 || v >= wp->nv || var_to_item[v] >= 0) {
            free(var_to_item);
            return 0;
        }
        var_to_item[v] = i;
    }

    memset(conflict_lo, 0, sizeof(conflict_lo));
    memset(conflict_hi, 0, sizeof(conflict_hi));
    for (int ci = 0; ci < wp->nc; ++ci) {
        int a, b;
        int va, vb;
        int ia, ib;
        if (wp->weights[ci] + eps < wp->top) continue;
        if (wp->sizes[ci] != 2) {
            free(var_to_item);
            return 0;
        }
        a = wp->cls[ci][0];
        b = wp->cls[ci][1];
        va = a > 0 ? a - 1 : -a - 1;
        vb = b > 0 ? b - 1 : -b - 1;
        if (va < 0 || va >= wp->nv || vb < 0 || vb >= wp->nv || va == vb) {
            free(var_to_item);
            return 0;
        }
        ia = var_to_item[va];
        ib = var_to_item[vb];
        if (ia < 0 || ib < 0 || items[ia].lit != -a || items[ib].lit != -b) {
            free(var_to_item);
            return 0;
        }
        conflict_lo[ia] |= unitsoft_mwis_bit_lo(ib);
        conflict_hi[ia] |= unitsoft_mwis_bit_hi(ib);
        conflict_lo[ib] |= unitsoft_mwis_bit_lo(ia);
        conflict_hi[ib] |= unitsoft_mwis_bit_hi(ia);
    }
    free(var_to_item);

    memset(&ctx, 0, sizeof(ctx));
    ctx.n = item_count;
    ctx.items = items;
    ctx.node_limit = item_count <= 96 ? 10000000LL : 1000000LL;
    ctx.deadline = now_sec() + (item_count <= 96 ? 1.25 : 0.35);

    uint64_t all_lo;
    uint64_t all_hi;
    if (item_count >= 64) {
        all_lo = UINT64_MAX;
        all_hi = (item_count == 128) ? UINT64_MAX : ((UINT64_C(1) << (item_count - 64)) - 1);
    } else {
        all_lo = (UINT64_C(1) << item_count) - 1;
        all_hi = 0;
    }

    for (int i = 0; i < item_count; ++i) {
        ctx.compat_lo[i] = all_lo & ~unitsoft_mwis_bit_lo(i) & ~conflict_lo[i];
        ctx.compat_hi[i] = all_hi & ~unitsoft_mwis_bit_hi(i) & ~conflict_hi[i];
        cands[i] = i;
    }
    unitsoft_mwis_sort_by_weight(&ctx, cands, item_count);
    unitsoft_mwis_expand(&ctx, cands, item_count, 0.0);

    for (int v = 0; v < wp->nv; ++v) {
        model[v] = unitsoft_preferred_value(us, v);
    }
    for (int i = 0; i < item_count; ++i) {
        unitsoft_model_clear_lit(model, items[i].lit);
    }
    int nsel = 0;
    for (int i = 0; i < item_count; ++i) {
        if (!ctx.best_sel[i]) continue;
        unitsoft_model_set_lit(model, items[i].lit);
        nsel++;
    }

    *cost = target_full_unsat_weight(wp, model);
    *picked = nsel;
    *nodes = ctx.nodes;
    *exact = ctx.aborted ? 0 : 1;
    return 1;
}

static double minones_partial_cost(const Solver *s, const MinOnesProfile *mo) {
    double cost = 0.0;
    for (int v = 0; v < s->nvars; ++v) {
        if (s->assigns[v] > 0) cost += mo->cost[v];
    }
    return cost;
}

static void minones_model_from_partial(unsigned char *dst, const Solver *s, int nvars) {
    for (int v = 0; v < nvars; ++v) {
        dst[v] = (unsigned char)(s->assigns[v] > 0 ? 1u : 0u);
    }
}

static int minones_hard_all_satisfied(const WCNF *wp, const Solver *s) {
    const double eps = 1e-12;
    for (int ci = 0; ci < wp->nc; ++ci) {
        if (wp->weights[ci] + eps < wp->top) continue;
        int sat = 0;
        for (int k = 0; k < wp->sizes[ci]; ++k) {
            int lit = wp->cls[ci][k];
            int v = lit > 0 ? lit - 1 : -lit - 1;
            int a = s->assigns[v];
            if (a != 0 && ((a > 0) == (lit > 0))) {
                sat = 1;
                break;
            }
        }
        if (!sat) return 0;
    }
    return 1;
}

static Lit minones_pick_branch(const WCNF *wp, const Solver *s, const MinOnesProfile *mo) {
    const double eps = 1e-12;
    int best_clause = -1;
    int best_clause_unassigned = INT_MAX;
    int best_lit = 0;
    double best_delta = DBL_MAX;
    double best_activity = -DBL_MAX;

    for (int ci = 0; ci < wp->nc; ++ci) {
        if (wp->weights[ci] + eps < wp->top) continue;
        int sat = 0;
        int unassigned = 0;
        int clause_lit = 0;
        double clause_delta = DBL_MAX;
        double clause_activity = -DBL_MAX;

        for (int k = 0; k < wp->sizes[ci]; ++k) {
            int lit = wp->cls[ci][k];
            int v = lit > 0 ? lit - 1 : -lit - 1;
            int a = s->assigns[v];
            if (a != 0) {
                if ((a > 0) == (lit > 0)) {
                    sat = 1;
                    break;
                }
                continue;
            }

            unassigned++;
            double delta = (lit > 0) ? mo->cost[v] : 0.0;
            double activity = s->activity[v];
            if (delta < clause_delta - eps ||
                (delta <= clause_delta + eps && activity > clause_activity)) {
                clause_delta = delta;
                clause_activity = activity;
                clause_lit = lit;
            }
        }

        if (sat) continue;
        if (unassigned <= 0) return -2;
        if (unassigned < best_clause_unassigned ||
            (unassigned == best_clause_unassigned &&
             (clause_delta < best_delta - eps ||
              (clause_delta <= best_delta + eps && clause_activity > best_activity)))) {
            best_clause = ci;
            best_clause_unassigned = unassigned;
            best_lit = clause_lit;
            best_delta = clause_delta;
            best_activity = clause_activity;
        }
    }

    (void)best_clause;
    if (best_lit == 0) return -1;
    return dimacs_to_lit(best_lit);
}

static int minones_record_block(Solver *s, IntVec *clause) {
    if (solver_decision_level(s) == 0) return 0;
    wcnf_build_trail_block_clause(clause, s);
    solver_cancel_until(s, 0);
    if (clause->size <= 0) return 0;
    if (!solver_record_learnt_clause(s, clause,
                                     (uint32_t)(clause->size > 0 ? clause->size : 1),
                                     0)) {
        return 0;
    }
    clause->size = 0;
    return 1;
}

static int minones_pb_refine_small(const WCNF *wp,
                                   const MinOnesProfile *mo,
                                   unsigned char *best_model,
                                   double *best_cost,
                                   int *exact_out) {
    int *soft_vars = NULL;
    int *soft_costs = NULL;
    int soft_count = 0;
    int improved = 0;
    const double eps = 1e-9;

    if (exact_out != NULL) *exact_out = 0;
    if (*best_cost <= eps || *best_cost > (double)INT_MAX) return 0;

    soft_vars = (int *)xmalloc((size_t)wp->nv * sizeof(int));
    soft_costs = (int *)xmalloc((size_t)wp->nv * sizeof(int));
    int hard_count = 0;
    for (int ci = 0; ci < wp->nc; ++ci) {
        if (wp->weights[ci] + eps >= wp->top) hard_count++;
    }
    if (hard_count > 2000) {
        free(soft_vars);
        free(soft_costs);
        return 0;
    }
    for (int v = 0; v < wp->nv; ++v) {
        if (mo->cost[v] <= eps) continue;
        double rounded = (double)(int)(mo->cost[v] + 0.5);
        if (rounded < mo->cost[v] - eps || rounded > mo->cost[v] + eps) {
            free(soft_vars);
            free(soft_costs);
            return 0;
        }
        soft_vars[soft_count] = v;
        soft_costs[soft_count] = (int)rounded;
        soft_count++;
    }

    for (;;) {
        int bound = (int)(*best_cost - 1.0 + eps);
        if (bound < 0) {
            if (exact_out != NULL) *exact_out = 1;
            break;
        }
        long long states = (long long)soft_count * (long long)bound;
        if (states <= 0 || states > 250000LL) break;

        Solver P;
        memset(&P, 0, sizeof(P));
        solver_init(&P, wp->nv + (int)states, 1, 0);
        int ok = 1;
        int max_clause_size = 0;
        for (int ci = 0; ci < wp->nc; ++ci) {
            if (wp->weights[ci] + eps >= wp->top && wp->sizes[ci] > max_clause_size) {
                max_clause_size = wp->sizes[ci];
            }
        }
        Lit *tmp = (Lit *)xmalloc((size_t)(max_clause_size > 0 ? max_clause_size : 1) * sizeof(Lit));
        for (int ci = 0; ok && ci < wp->nc; ++ci) {
            if (wp->weights[ci] + eps < wp->top) continue;
            for (int k = 0; k < wp->sizes[ci]; ++k) tmp[k] = dimacs_to_lit(wp->cls[ci][k]);
            ok = solver_add_clause(&P, tmp, wp->sizes[ci], 0, 0);
        }
        free(tmp);

        for (int i = 0; ok && i < soft_count; ++i) {
            int x = soft_vars[i];
            int w = soft_costs[i];
            for (int k = 1; ok && k <= bound; ++k) {
                int curr = wp->nv + i * bound + (k - 1);
                if (i > 0) {
                    int prev = wp->nv + (i - 1) * bound + (k - 1);
                    Lit c2[2] = { mk_lit(prev, 1), mk_lit(curr, 0) };
                    ok = solver_add_clause(&P, c2, 2, 0, 0);
                }
            }
            if (!ok) break;
            if (w <= bound) {
                int curr = wp->nv + i * bound + (w - 1);
                Lit c2[2] = { mk_lit(x, 1), mk_lit(curr, 0) };
                ok = solver_add_clause(&P, c2, 2, 0, 0);
            } else {
                Lit c1[1] = { mk_lit(x, 1) };
                ok = solver_add_clause(&P, c1, 1, 0, 0);
            }
            for (int k = 1; ok && i > 0 && k <= bound; ++k) {
                int prev = wp->nv + (i - 1) * bound + (k - 1);
                if (k + w <= bound) {
                    int curr = wp->nv + i * bound + (k + w - 1);
                    Lit c3[3] = { mk_lit(prev, 1), mk_lit(x, 1), mk_lit(curr, 0) };
                    ok = solver_add_clause(&P, c3, 3, 0, 0);
                } else {
                    Lit c2[2] = { mk_lit(prev, 1), mk_lit(x, 1) };
                    ok = solver_add_clause(&P, c2, 2, 0, 0);
                }
            }
        }

        int rc = ok ? solver_solve(&P) : 20;
        if (rc == 10) {
            unsigned char *candidate = (unsigned char *)xmalloc((size_t)wp->nv);
            for (int v = 0; v < wp->nv; ++v) {
                candidate[v] = (unsigned char)(P.assigns[v] > 0 ? 1u : 0u);
            }
            double cost = target_full_unsat_weight(wp, candidate);
            if (cost < *best_cost - eps) {
                *best_cost = cost;
                memcpy(best_model, candidate, (size_t)wp->nv);
                improved = 1;
                if (sat_log_enabled()) {
                    printf("c PB improve: %.0f\n", *best_cost);
                    fflush(stdout);
                }
            }
            free(candidate);
            solver_destroy(&P);
            continue;
        }

        if (rc == 20 && exact_out != NULL) *exact_out = 1;
        solver_destroy(&P);
        break;
    }

    free(soft_vars);
    free(soft_costs);
    return improved;
}

static int minones_greedy_assumption_refine(const WCNF *wp,
                                            const MinOnesProfile *mo,
                                            unsigned char *best_model,
                                            double *best_cost) {
    const double eps = 1e-9;
    int hard_count = 0;
    int improved = 0;
    for (int i = 0; i < wp->nc; ++i) {
        if (wp->weights[i] + eps >= wp->top) hard_count++;
    }
    if (hard_count <= 0 || hard_count > 200000) return 0;

    int **hard_cls = (int **)xmalloc((size_t)hard_count * sizeof(int *));
    int *hard_sizes = (int *)xmalloc((size_t)hard_count * sizeof(int));
    int hi = 0;
    for (int i = 0; i < wp->nc; ++i) {
        if (wp->weights[i] + eps < wp->top) continue;
        hard_cls[hi] = wp->cls[i];
        hard_sizes[hi] = wp->sizes[i];
        hi++;
    }

    UnitSoftItem *items = (UnitSoftItem *)xmalloc((size_t)wp->nv * sizeof(UnitSoftItem));
    int item_count = 0;
    for (int v = 0; v < wp->nv; ++v) {
        if (mo->cost[v] <= eps) continue;
        items[item_count].lit = -(v + 1);
        items[item_count].gain = mo->cost[v];
        item_count++;
    }
    if (item_count <= 0 || item_count > 5000) {
        free(items);
        free(hard_sizes);
        free(hard_cls);
        return 0;
    }
    unitsoft_sort_mode = 0;
    qsort(items, (size_t)item_count, sizeof(UnitSoftItem), unitsoft_item_cmp_desc);

    int *assumptions = (int *)xmalloc((size_t)item_count * sizeof(int));
    unsigned char *greedy_model = (unsigned char *)xmalloc((size_t)wp->nv);
    SlimeSatOptions opt;
    slime_sat_options_default(&opt);
    opt.use_hess = 0;
    opt.use_ct = 0;
    opt.use_mab = 1;
    SlimeSatHandle *handle = slime_sat_handle_create(wp->nv, hard_count,
                                                     (const int *const *)hard_cls,
                                                     hard_sizes, &opt);
    int kept = 0;
    int old_log = slime_log_enabled;
    slime_log_enabled = 0;
    if (handle != NULL) {
        for (int i = 0; i < item_count; ++i) {
            assumptions[kept] = items[i].lit;
            int rc = slime_sat_handle_solve(handle, assumptions, kept + 1, NULL, greedy_model);
            if (rc == 10) {
                kept++;
                double cost = target_full_unsat_weight(wp, greedy_model);
                if (cost < *best_cost - eps) {
                    *best_cost = cost;
                    memcpy(best_model, greedy_model, (size_t)wp->nv);
                    improved = 1;
                }
            }
        }
        slime_sat_handle_destroy(handle);
    }
    slime_log_enabled = old_log;
    if (sat_log_enabled() && improved) {
        printf("c minones greedy assumptions: kept=%d/%d cost=%.0f\n", kept, item_count, *best_cost);
        fflush(stdout);
    }

    free(greedy_model);
    free(assumptions);
    free(items);
    free(hard_sizes);
    free(hard_cls);
    return improved;
}

static int slime_minones_solve_wp(const WCNF *wp,
                                  const MinOnesProfile *mo,
                                  unsigned char *model01,
                                  double *optimal_cost) {
    Solver S;
    HessWOcc occ;
    IntVec learnt;
    IntVec block_clause;
    ClauseRef confl;
    unsigned char *best_model = NULL;
    unsigned char *candidate = NULL;
    double best_cost = DBL_MAX;
    int found_model = 0;
    int exact = 0;
    int hard_unsat = 0;
    int solver_ready = 0;
    int max_clause_size = 0;
    const double eps = 1e-9;

    memset(&S, 0, sizeof(S));
    memset(&occ, 0, sizeof(occ));
    intvec_init(&learnt);
    intvec_init(&block_clause);

    if (sat_log_enabled()) {
        printf("c maxsat path: min-ones branch-and-bound\n");
        fflush(stdout);
    }

    best_model = (unsigned char *)xmalloc((size_t)wp->nv);
    candidate = (unsigned char *)xmalloc((size_t)wp->nv);
    memset(best_model, 0, (size_t)wp->nv);
    memset(candidate, 0, (size_t)wp->nv);

    hessw_occ_init(&occ, wp->nv, wp->nc, wp->sizes, wp->cls);
    hessw_reset_progress_logs();
    double ls_cost = hessw_search(wp->nv, wp->nc, wp->sizes, wp->cls, wp->weights, wp->tw,
                                  &occ, NULL, candidate, slime_hess_max_iter, 0.0);
    ls_cost = target_full_unsat_weight(wp, candidate);
    if (ls_cost < wp->top - eps) {
        best_cost = ls_cost;
        found_model = 1;
        memcpy(best_model, candidate, (size_t)wp->nv);
        if (sat_log_enabled()) {
            printf("c minones incumbent: %.0f\n", best_cost);
            fflush(stdout);
        }
    }

    if (found_model) {
        int pb_exact = 0;
        minones_pb_refine_small(wp, mo, best_model, &best_cost, &pb_exact);
        if (pb_exact) exact = 1;
    }

    if (wp->nv > 96 && !found_model) {
        Solver H;
        memset(&H, 0, sizeof(H));
        int hard_max_clause_size = 0;
        int hard_ok = 1;
        for (int i = 0; i < wp->nc; ++i) {
            if (wp->weights[i] + eps >= wp->top && wp->sizes[i] > hard_max_clause_size) {
                hard_max_clause_size = wp->sizes[i];
            }
        }
        solver_init(&H, wp->nv, 1, 0);
        for (int v = 0; v < wp->nv; ++v) {
            H.phases[v] = 0u;
            H.activity[v] += 1.0 / (1.0 + mo->cost[v]);
            if (H.chb_activity != NULL) H.chb_activity[v] += H.activity[v];
        }
        Lit *hard_tmp = (Lit *)xmalloc((size_t)(hard_max_clause_size > 0 ? hard_max_clause_size : 1) * sizeof(Lit));
        for (int i = 0; hard_ok && i < wp->nc; ++i) {
            if (wp->weights[i] + eps < wp->top) continue;
            for (int k = 0; k < wp->sizes[i]; ++k) {
                int lit = wp->cls[i][k];
                int v = lit > 0 ? lit - 1 : -lit - 1;
                hard_tmp[k] = dimacs_to_lit(lit);
                H.activity[v] += 1.0;
                if (lit < 0) H.activity[v] += 0.5;
            }
            hard_ok = solver_add_clause(&H, hard_tmp, wp->sizes[i], 0, 0);
        }
        free(hard_tmp);
        heap_rebuild(&H.order);
        if (hard_ok && solver_solve(&H) == 10) {
            for (int v = 0; v < wp->nv; ++v) {
                candidate[v] = (unsigned char)(H.assigns[v] > 0 ? 1u : 0u);
            }
            double hard_cost = target_full_unsat_weight(wp, candidate);
            if (hard_cost < wp->top - eps) {
                best_cost = hard_cost;
                found_model = 1;
                memcpy(best_model, candidate, (size_t)wp->nv);
                if (sat_log_enabled()) {
                    printf("c minones hard-phase incumbent: %.0f\n", best_cost);
                    fflush(stdout);
                }
            }
        }
        solver_destroy(&H);
    }

    if (wp->nv > 96 && found_model) {
        minones_greedy_assumption_refine(wp, mo, best_model, &best_cost);
        double seeded_cost = hessw_search(wp->nv, wp->nc, wp->sizes, wp->cls, wp->weights, wp->tw,
                                          &occ, best_model, candidate, 1, 0.0);
        seeded_cost = target_full_unsat_weight(wp, candidate);
        if (seeded_cost < best_cost - eps) {
            best_cost = seeded_cost;
            memcpy(best_model, candidate, (size_t)wp->nv);
            if (sat_log_enabled()) {
                printf("c minones seeded-LS incumbent: %.0f\n", best_cost);
                fflush(stdout);
            }
        }
        goto minones_finish;
    }
    if (wp->nv > 96) {
        goto minones_finish;
    }

    for (int i = 0; i < wp->nc; ++i) {
        if (wp->sizes[i] > max_clause_size) max_clause_size = wp->sizes[i];
    }

    solver_init(&S, wp->nv, 1, 0);
    solver_ready = 1;
    for (int v = 0; v < wp->nv; ++v) {
        S.phases[v] = found_model ? best_model[v] : 0u;
        S.activity[v] += 1.0 / (1.0 + mo->cost[v]);
        if (S.chb_activity != NULL) S.chb_activity[v] += S.activity[v];
    }

    Lit *tmp_clause = (Lit *)xmalloc((size_t)(max_clause_size > 0 ? max_clause_size : 1) * sizeof(Lit));
    for (int i = 0; i < wp->nc; ++i) {
        if (wp->weights[i] + eps < wp->top) continue;
        for (int k = 0; k < wp->sizes[i]; ++k) {
            tmp_clause[k] = dimacs_to_lit(wp->cls[i][k]);
            int v = wp->cls[i][k] > 0 ? wp->cls[i][k] - 1 : -wp->cls[i][k] - 1;
            S.activity[v] += 1.0;
            if (wp->cls[i][k] > 0) {
                S.activity[v] += 2.0 / (1.0 + mo->cost[v]);
            }
            if (S.chb_activity != NULL) S.chb_activity[v] += S.activity[v] * 0.25;
        }
        if (!solver_add_clause(&S, tmp_clause, wp->sizes[i], 0, 0)) {
            hard_unsat = 1;
            break;
        }
    }
    free(tmp_clause);
    heap_rebuild(&S.order);

    confl = hard_unsat ? clauseref_none() : solver_propagate(&S);
    solver_update_chb_after_propagate(&S, !clauseref_is_none(confl));
    if (!hard_unsat && !clauseref_is_none(confl) && solver_decision_level(&S) == 0) {
        hard_unsat = 1;
    }

    while (!hard_unsat && S.ok) {
        if (!clauseref_is_none(confl)) {
            S.conflicts++;
            S.mab_epoch_conflicts += 1.0;
            if (solver_decision_level(&S) == 0) {
                exact = found_model;
                if (!found_model) hard_unsat = 1;
                break;
            }

            int bt = 0;
            solver_analyze(&S, confl, &learnt, &bt);
            uint32_t lbd = solver_compute_lbd(&S, &learnt);
            solver_cancel_until(&S, bt);
            if (!solver_record_learnt_clause(&S, &learnt, lbd, 1)) {
                exact = 0;
                break;
            }
            solver_var_decay(&S);
            solver_clause_decay(&S);
            if (S.conflicts >= S.next_reduce) {
                solver_reduce_db(&S);
                S.next_reduce = S.conflicts + S.reduce_base + S.learnts.size / 2;
            }
            if (S.conflicts >= S.next_restart) {
                S.restarts++;
                solver_cancel_until(&S, 0);
                int lub = luby(2, S.restart_count++);
                S.next_restart = S.conflicts + (long long)(100 * lub);
            }
        } else {
            double lb = minones_partial_cost(&S, mo);
            if (found_model && lb >= best_cost - eps) {
                if (!minones_record_block(&S, &block_clause)) {
                    exact = 1;
                    break;
                }
                confl = clauseref_none();
                continue;
            }

            if (minones_hard_all_satisfied(wp, &S)) {
                if (lb < best_cost - eps || !found_model) {
                    best_cost = lb;
                    found_model = 1;
                    minones_model_from_partial(best_model, &S, wp->nv);
                    if (sat_log_enabled()) {
                        printf("c B&B improve: %.0f\n", best_cost);
                        fflush(stdout);
                    }
                    if (best_cost <= eps) {
                        exact = 1;
                        break;
                    }
                }
                if (!minones_record_block(&S, &block_clause)) {
                    exact = 1;
                    break;
                }
                confl = clauseref_none();
                continue;
            }

            Lit dlit = minones_pick_branch(wp, &S, mo);
            if (dlit == -2) {
                S.ok = 0;
                break;
            }
            if (dlit == -1) {
                exact = found_model;
                break;
            }
            intvec_push(&S.trail_lim, S.trail_size);
            S.decisions++;
            S.mab_epoch_decisions += 1.0;
            if (!solver_enqueue(&S, dlit, reason_none())) {
                S.ok = 0;
                break;
            }
        }

        confl = solver_propagate(&S);
        solver_update_chb_after_propagate(&S, !clauseref_is_none(confl));
    }

    if (!S.ok) exact = 0;

minones_finish:
    ;
    int rc = 0;
    if (hard_unsat && !found_model) {
        rc = 20;
    } else if (found_model) {
        rc = exact ? 10 : 0;
    }

    if (sat_log_enabled()) {
        printf("o %.0f\n", found_model ? best_cost : 0.0);
        if (slime_validate) {
            if (rc == 20) {
                printf("s UNSATISFIABLE\n");
            } else if (rc == 10) {
                printf("s OPTIMUM FOUND\n");
            } else {
                printf("s UNKNOWN\n");
            }
            if (found_model) {
                for (int i = 0; i < wp->nv; ++i) {
                    if (i % 10 == 0) printf("\nv ");
                    printf("%i ", best_model[i] ? (i + 1) : -(i + 1));
                }
                printf("0\n");
                double ck = target_full_unsat_weight(wp, best_model);
                printf("c validate: %s (unsat=%.0f %s)\n",
                       (ck >= best_cost - eps && ck <= best_cost + eps) ? "OK" : "MISMATCH",
                       ck,
                       exact ? "matches optimum" : "matches incumbent");
            }
        }
        fflush(stdout);
    }

    if (found_model && model01 != NULL) {
        memcpy(model01, best_model, (size_t)wp->nv * sizeof(unsigned char));
    }
    if (optimal_cost != NULL) {
        *optimal_cost = found_model ? best_cost : 0.0;
    }

    intvec_free(&learnt);
    intvec_free(&block_clause);
    hessw_occ_free(&occ);
    if (solver_ready) solver_destroy(&S);
    free(best_model);
    free(candidate);
    return rc;
}

static int slime_unitsoft_solve_wp(const WCNF *wp,
                                   const UnitSoftProfile *us,
                                   unsigned char *model01,
                                   double *optimal_cost) {
    Solver S;
    HessWOcc occ;
    unsigned char *best_model = NULL;
    unsigned char *work_model = NULL;
    double best_cost = DBL_MAX;
    int found_model = 0;
    int exact = 0;
    int max_clause_size = 0;
    const double eps = 1e-9;

    memset(&S, 0, sizeof(S));
    memset(&occ, 0, sizeof(occ));

    if (sat_log_enabled()) {
        printf("c maxsat path: unit-soft hard-phase search (soft=%d pos=%d neg=%d)\n",
               us->soft_count, us->pos_count, us->neg_count);
        fflush(stdout);
    }

    best_model = (unsigned char *)xmalloc((size_t)wp->nv);
    work_model = (unsigned char *)xmalloc((size_t)wp->nv);
    for (int v = 0; v < wp->nv; ++v) {
        best_model[v] = unitsoft_preferred_value(us, v);
        work_model[v] = best_model[v];
    }
    best_cost = target_full_unsat_weight(wp, best_model);
    if (best_cost < wp->top - eps) found_model = 1;

    solver_init(&S, wp->nv, 1, 0);
    for (int v = 0; v < wp->nv; ++v) {
        double posw = us->pos_weight[v];
        double negw = us->neg_weight[v];
        double diff = posw > negw ? posw - negw : negw - posw;
        S.phases[v] = unitsoft_preferred_value(us, v);
        S.activity[v] += 1.0 + diff;
        if (S.chb_activity != NULL) S.chb_activity[v] += 0.5 + diff;
    }

    for (int i = 0; i < wp->nc; ++i) {
        if (wp->weights[i] + eps >= wp->top && wp->sizes[i] > max_clause_size) {
            max_clause_size = wp->sizes[i];
        }
    }
    Lit *tmp_clause = (Lit *)xmalloc((size_t)(max_clause_size > 0 ? max_clause_size : 1) * sizeof(Lit));
    int hard_ok = 1;
    for (int i = 0; hard_ok && i < wp->nc; ++i) {
        if (wp->weights[i] + eps < wp->top) continue;
        for (int k = 0; k < wp->sizes[i]; ++k) {
            int lit = wp->cls[i][k];
            int v = lit > 0 ? lit - 1 : -lit - 1;
            tmp_clause[k] = dimacs_to_lit(lit);
            S.activity[v] += 1.0;
            if ((lit > 0 ? 1u : 0u) == S.phases[v]) S.activity[v] += 0.5;
        }
        hard_ok = solver_add_clause(&S, tmp_clause, wp->sizes[i], 0, 0);
    }
    free(tmp_clause);
    heap_rebuild(&S.order);

    if (hard_ok) {
        int sat_rc = solver_solve(&S);
        if (sat_rc == 10) {
            for (int v = 0; v < wp->nv; ++v) {
                work_model[v] = (unsigned char)(S.assigns[v] > 0 ? 1u : 0u);
            }
            double cost = target_full_unsat_weight(wp, work_model);
            if (cost < best_cost - eps || !found_model) {
                best_cost = cost;
                found_model = (cost < wp->top - eps);
                memcpy(best_model, work_model, (size_t)wp->nv);
                if (sat_log_enabled() && found_model) {
                    printf("c unit-soft hard model: %.0f\n", best_cost);
                    fflush(stdout);
                }
            }
        }
    }

    if (us->soft_count <= 3000) {
        int hard_count = 0;
        for (int i = 0; i < wp->nc; ++i) {
            if (wp->weights[i] + eps >= wp->top) hard_count++;
        }
        if (hard_count > 0 && hard_count <= 200000) {
            int **hard_cls = (int **)xmalloc((size_t)hard_count * sizeof(int *));
            int *hard_sizes = (int *)xmalloc((size_t)hard_count * sizeof(int));
            int hi = 0;
            for (int i = 0; i < wp->nc; ++i) {
                if (wp->weights[i] + eps < wp->top) continue;
                hard_cls[hi] = wp->cls[i];
                hard_sizes[hi] = wp->sizes[i];
                hi++;
            }

            UnitSoftItem *items = (UnitSoftItem *)xmalloc((size_t)wp->nv * sizeof(UnitSoftItem));
            int item_count = 0;
            for (int v = 0; v < wp->nv; ++v) {
                double posw = us->pos_weight[v];
                double negw = us->neg_weight[v];
                double gain = posw > negw ? posw - negw : negw - posw;
                if (gain <= eps) continue;
                items[item_count].lit = (posw >= negw) ? (v + 1) : -(v + 1);
                items[item_count].gain = gain;
                item_count++;
            }
            int *assumptions = (int *)xmalloc((size_t)(item_count > 0 ? item_count : 1) * sizeof(int));
            unsigned char *greedy_model = (unsigned char *)xmalloc((size_t)wp->nv);
            int best_kept = 0;
            int mwis_picked = 0;
            long long mwis_nodes = 0;
            int mwis_exact = 0;
            double mwis_cost = DBL_MAX;
            int old_log = slime_log_enabled;
            slime_log_enabled = 0;
            UnitSoftItem *ordered = (UnitSoftItem *)xmalloc((size_t)(item_count > 0 ? item_count : 1) * sizeof(UnitSoftItem));
            if (unitsoft_try_mwis_exact(wp, us, items, item_count, greedy_model,
                                        &mwis_cost, &mwis_picked, &mwis_nodes, &mwis_exact)) {
                if (mwis_cost < wp->top - eps &&
                    (mwis_cost < best_cost - eps || !found_model)) {
                    best_cost = mwis_cost;
                    found_model = 1;
                    exact = mwis_exact;
                    best_kept = mwis_picked;
                    memcpy(best_model, greedy_model, (size_t)wp->nv);
                } else if (mwis_exact && found_model) {
                    exact = 1;
                }
            }
            if (!exact) {
                int modes = item_count <= 512 ? 5 : 1;
                for (int mode = 0; mode < modes; ++mode) {
                    memcpy(ordered, items, (size_t)item_count * sizeof(UnitSoftItem));
                    unitsoft_sort_mode = mode;
                    qsort(ordered, (size_t)item_count, sizeof(UnitSoftItem), unitsoft_item_cmp_desc);

                    int kept = 0;
                    SlimeSatOptions opt;
                    slime_sat_options_default(&opt);
                    opt.use_hess = 0;
                    opt.use_ct = 0;
                    opt.use_mab = 1;
                    SlimeSatHandle *handle = slime_sat_handle_create(wp->nv, hard_count,
                                                                     (const int *const *)hard_cls,
                                                                     hard_sizes, &opt);
                    if (handle != NULL) {
                        for (int i = 0; i < item_count; ++i) {
                            assumptions[kept] = ordered[i].lit;
                            int rc = slime_sat_handle_solve(handle, assumptions, kept + 1, NULL, greedy_model);
                            if (rc == 10) {
                                kept++;
                                double cost = target_full_unsat_weight(wp, greedy_model);
                                if (cost < best_cost - eps || !found_model) {
                                    best_cost = cost;
                                    found_model = (cost < wp->top - eps);
                                    memcpy(best_model, greedy_model, (size_t)wp->nv);
                                    best_kept = kept;
                                }
                            }
                        }
                        slime_sat_handle_destroy(handle);
                    }
                }
            }
            slime_log_enabled = old_log;
            unitsoft_sort_mode = 0;
            if (sat_log_enabled() && found_model) {
                if (mwis_nodes > 0) {
                    printf("c unit-soft mwis %s: picked=%d/%d cost=%.0f nodes=%lld\n",
                           mwis_exact ? "exact" : "incumbent",
                           mwis_picked, item_count, mwis_cost, mwis_nodes);
                }
                if (!exact) {
                    printf("c unit-soft greedy assumptions: kept=%d/%d cost=%.0f\n",
                           best_kept, item_count, best_cost);
                }
                fflush(stdout);
            }
            free(ordered);
            free(greedy_model);
            free(assumptions);
            free(items);
            free(hard_sizes);
            free(hard_cls);
        }
    }

    if (found_model && !exact) {
        hessw_occ_init(&occ, wp->nv, wp->nc, wp->sizes, wp->cls);
        hessw_reset_progress_logs();
        double ls_cost = hessw_search(wp->nv, wp->nc, wp->sizes, wp->cls, wp->weights, wp->tw,
                                      &occ, best_model, work_model, 1, 0.0);
        ls_cost = target_full_unsat_weight(wp, work_model);
        if (ls_cost < best_cost - eps) {
            best_cost = ls_cost;
            memcpy(best_model, work_model, (size_t)wp->nv);
            if (sat_log_enabled()) {
                printf("c unit-soft LS model: %.0f\n", best_cost);
                fflush(stdout);
            }
        }
    }

    int rc = found_model ? (exact ? 10 : 0) : 20;
    if (sat_log_enabled()) {
        printf("o %.0f\n", found_model ? best_cost : 0.0);
        if (slime_validate) {
            if (found_model) {
                printf("%s\n", exact ? "s OPTIMUM FOUND" : "s UNKNOWN");
                for (int i = 0; i < wp->nv; ++i) {
                    if (i % 10 == 0) printf("\nv ");
                    printf("%i ", best_model[i] ? (i + 1) : -(i + 1));
                }
                printf("0\n");
                double ck = target_full_unsat_weight(wp, best_model);
                printf("c validate: %s (unsat=%.0f %s)\n",
                       (ck >= best_cost - eps && ck <= best_cost + eps) ? "OK" : "MISMATCH",
                       ck,
                       exact ? "matches optimum" : "matches incumbent");
            } else {
                printf("s UNSATISFIABLE\n");
            }
        }
        fflush(stdout);
    }

    if (found_model && model01 != NULL) {
        memcpy(model01, best_model, (size_t)wp->nv * sizeof(unsigned char));
    }
    if (optimal_cost != NULL) {
        *optimal_cost = found_model ? best_cost : 0.0;
    }

    hessw_occ_free(&occ);
    solver_destroy(&S);
    free(best_model);
    free(work_model);
    return rc;
}

int slime_maxsat_solve_mem(int nv, int nc,
                           const int *const *cls, const int *sizes,
                           const double *weights, double top_weight,
                           unsigned char *model01, double *optimal_cost) {
    WCNF wp;
    memset(&wp, 0, sizeof(wp));
    wp.nv = nv;
    wp.nc = nc;
    wp.sizes = (int *)sizes;
    wp.cls = (int **)cls;
    wp.weights = (double *)weights;
    wp.tw = 0.0;
    wp.top = (top_weight > 0.0) ? top_weight : 1.0;
    wp.owns = 0;
    
    double *local_weights = NULL;
    if (weights == NULL) {
        local_weights = (double *)xmalloc((size_t)nc * sizeof(double));
        for (int i = 0; i < nc; i++) local_weights[i] = 1.0;
        wp.weights = local_weights;
    }
    for (int i = 0; i < nc; i++) {
        wp.tw += wp.weights[i];
    }
    if (top_weight <= 0.0) {
        if (local_weights == NULL) {
            local_weights = (double *)xmalloc((size_t)nc * sizeof(double));
            wp.weights = local_weights;
        }
        for (int i = 0; i < nc; i++) wp.weights[i] = 1.0;
        wp.tw = (double)nc;
    }

    Solver S;
    TargetGuide guide;
    TargetSnapshot progress;
    IntVec learnt, prune_clause;
    HessWOcc occ;
    int solver_ready = 0, found_model = 0, search_exhausted = 0, hard_unsat = 0;
    int aborted_search = 0;
    int convergence_enabled = 0, convergence_hits = 0, soft_unit_count = 0, refresh_count = 0;
    double min_soft_weight = 0.0, convergence_zone = 0.0;
    double last_convergence_delta = DBL_MAX;
    long long last_best_decisions = 0, last_best_conflicts = 0;
    const double eps = 1e-9;
    double t0 = 0.0;
    double t1 = 0.0;
    int max_clause_size = 0;
    int hard_count = 0;
    int soft_count = 0;
    int hard_only = 0;

    for (int i = 0; i < nc; ++i) {
        if (sizes[i] > max_clause_size) max_clause_size = sizes[i];
        if (wp.weights[i] + 1e-12 >= wp.top) hard_count++;
        else soft_count++;
    }
    hard_only = (soft_count == 0);
    if (sat_log_enabled()) {
        t0 = (double)clock() / (double)CLOCKS_PER_SEC;
        printf("c maxsat setup: nv=%d nc=%d hard=%d soft=%d\n", nv, nc, hard_count, soft_count);
        fflush(stdout);
    }

    MinOnesProfile minones;
    if (minones_profile_init(&wp, &minones)) {
        int rc = slime_minones_solve_wp(&wp, &minones, model01, optimal_cost);
        minones_profile_free(&minones);
        free(local_weights);
        return rc;
    }
    UnitSoftProfile unitsoft;
    if (unitsoft_profile_init(&wp, &unitsoft)) {
        int rc = slime_unitsoft_solve_wp(&wp, &unitsoft, model01, optimal_cost);
        unitsoft_profile_free(&unitsoft);
        free(local_weights);
        return rc;
    }

    convergence_enabled = wcnf_soft_unit_profile(&wp, &soft_unit_count, &min_soft_weight);
    if (convergence_enabled)
        convergence_zone = target_convergence_zone(soft_unit_count, min_soft_weight);

    memset(&S, 0, sizeof(S));
    memset(&guide, 0, sizeof(guide));
    memset(&progress, 0, sizeof(progress));
    memset(&learnt, 0, sizeof(learnt));
    memset(&prune_clause, 0, sizeof(prune_clause));
    memset(&occ, 0, sizeof(occ));

    hessw_occ_init(&occ, wp.nv, wp.nc, wp.sizes, wp.cls);

    hessw_reset_progress_logs();
    unsigned char *best_model = (unsigned char *)xmalloc((size_t)wp.nv);
    unsigned char *seed = (unsigned char *)xmalloc((size_t)wp.nv);
    unsigned char *work_model = (unsigned char *)xmalloc((size_t)wp.nv);

    if (sat_log_enabled()) {
        t1 = (double)clock() / (double)CLOCKS_PER_SEC;
        printf("c maxsat encode: ready in %.6fs\n", t1 - t0);
        fflush(stdout);
        printf("c maxsat search: starting HESS+LS\n");
        fflush(stdout);
    }

    double hard_only_seed_limit = hard_only ? (wp.tw * 0.05) : 0.0;
    if (hard_only_seed_limit < 1.0 && hard_only_seed_limit > 0.0) {
        hard_only_seed_limit = 1.0;
    }
    double best_unsat = hessw_search(wp.nv, wp.nc, wp.sizes, wp.cls, wp.weights, wp.tw,
                                     &occ, NULL, best_model, slime_hess_max_iter, hard_only_seed_limit);
    if (best_unsat < wp.top - eps) {
        found_model = 1;
    }
    if (best_unsat <= eps) {
        search_exhausted = 1;
        progress.matches = wp.nv;
        progress.mismatches = 0;
        progress.unassigned = 0;
        progress.satisfied_weight = wp.tw;
        goto finalize;
    }
    if (!hard_only && best_unsat < wp.top - eps && wp.nv > 96) {
        if (sat_log_enabled()) {
            printf("c maxsat path: accepting local-search incumbent before proof search\n");
            fflush(stdout);
        }
        progress.matches = 0;
        progress.mismatches = 0;
        progress.unassigned = wp.nv;
        progress.satisfied_weight = wp.tw - best_unsat;
        goto finalize;
    }

    solver_init(&S, wp.nv, 1, hard_only ? 1 : 0);
    solver_ready = 1;

    Lit *tmp_clause = (Lit *)xmalloc((size_t)(max_clause_size > 0 ? max_clause_size : 1) * sizeof(Lit));
    for (int i = 0; i < wp.nc; ++i) {
        if (wp.top > 0.0 && wp.weights[i] + 1e-12 < wp.top) continue;
        for (int k = 0; k < wp.sizes[i]; ++k) tmp_clause[k] = dimacs_to_lit(wp.cls[i][k]);
        solver_add_clause(&S, tmp_clause, wp.sizes[i], 0, 0);
        double clause_weight = wp.weights[i] > 0.0 ? wp.weights[i] : 0.0;
        double clause_bump = clause_weight / (double)(wp.sizes[i] > 0 ? wp.sizes[i] : 1);
        if (clause_bump > 8.0) clause_bump = 8.0;
        if (clause_bump > 0.0) {
            for (int k = 0; k < wp.sizes[i]; ++k) {
                int v = wp.cls[i][k] > 0 ? wp.cls[i][k] - 1 : -wp.cls[i][k] - 1;
                S.activity[v] += clause_bump;
                if (S.chb_activity != NULL) S.chb_activity[v] += clause_bump;
            }
        }
    }
    free(tmp_clause);

    if (hard_only) {
        if (sat_log_enabled()) {
            printf("c maxsat path: hard-only instance; using SAT solve after HESS+LS\n");
            fflush(stdout);
        }
        memcpy(S.phases, best_model, (size_t)wp.nv);
        if (sat_log_enabled()) {
            printf("c maxsat path: incumbent phase-seeded into SAT solver\n");
            fflush(stdout);
        }
        {
            int sat_rc = solver_solve(&S);
            if (sat_rc == 10) {
                target_model_from_assigns(best_model, &S, wp.nv);
                best_unsat = 0.0;
                search_exhausted = 1;
                progress.matches = wp.nv;
                progress.mismatches = 0;
                progress.unassigned = 0;
                progress.satisfied_weight = wp.tw;
                if (sat_log_enabled()) {
                    printf("c maxsat path: SAT confirmed, using CDCL model\n");
                    fflush(stdout);
                }
            } else if (sat_rc == 20) {
                hard_unsat = 1;
                best_unsat = target_full_unsat_weight(&wp, best_model);
            }
        }
        goto finalize;
    }

    guide.nvars = wp.nv;
    guide.target = (signed char *)xmalloc((size_t)wp.nv * sizeof(signed char));
    guide.pressure = (double *)xmalloc((size_t)wp.nv * sizeof(double));
    guide.escape = (double *)xmalloc((size_t)wp.nv * sizeof(double));
    memset(guide.escape, 0, (size_t)wp.nv * sizeof(double));
    target_guide_set_model(&guide, &S, &wp, best_model, NULL);
    intvec_init(&learnt);
    intvec_init(&prune_clause);

    ClauseRef confl = solver_propagate(&S);
    solver_update_chb_after_propagate(&S, !clauseref_is_none(confl));

    target_snapshot_init(&progress, &S, &guide, &wp);

    if (!clauseref_is_none(confl) && solver_decision_level(&S) == 0) {
        if (sat_log_enabled()) {
            printf("c CDCL root conflict; refining current HESS+WalkSAT target\n");
            fflush(stdout);
        }
        hard_unsat = 1;
    } else {
        while (S.ok) {
            if (solver_poll_external_stop(&S)) break;
            if (convergence_enabled &&
                target_accept_converged_incumbent(best_unsat,
                                                  convergence_zone,
                                                  convergence_hits,
                                                  last_convergence_delta,
                                                  S.decisions - last_best_decisions,
                                                  S.conflicts - last_best_conflicts)) {
                if (sat_log_enabled()) {
                    printf("c target convergence: accepting incumbent %.0f after %lld decisions and %lld conflicts without improvement\n",
                           best_unsat,
                           (long long)(S.decisions - last_best_decisions),
                           (long long)(S.conflicts - last_best_conflicts));
                    fflush(stdout);
                }
                search_exhausted = 1;
                break;
            }

            if (!clauseref_is_none(confl)) {
                if (solver_decision_level(&S) == 0) {
                    if (found_model) {
                        search_exhausted = 1;
                    } else {
                        hard_unsat = 1;
                    }
                    break;
                }

                S.conflicts++;
                S.mab_epoch_conflicts += 1.0;

                int bt = 0;
                solver_analyze(&S, confl, &learnt, &bt);
                uint32_t lbd = solver_compute_lbd(&S, &learnt);
                solver_cancel_until(&S, bt);

                if (!solver_record_learnt_clause(&S, &learnt, lbd, 1)) {
                    if (S.proof) fputs("0\n", S.proof);
                    aborted_search = 1;
                    break;
                }
                target_absorb_conflict(&guide, &S, &learnt);

                solver_var_decay(&S);
                solver_clause_decay(&S);

                if (S.conflicts >= S.next_reduce) {
                    solver_reduce_db(&S);
                    S.next_reduce = S.conflicts + S.reduce_base + S.learnts.size / 2;
                }

                if (S.conflicts >= S.next_restart) {
                    S.restarts++;
                    solver_cancel_until(&S, 0);
                    solver_restart_mab(&S);
                    solver_covertrace_escape_phases(&S);
                    if (S.ct_probe_restarts > 0 && (S.restarts % S.ct_probe_restarts) == 0) {
                        solver_covertrace_probe(&S);
                    }
                    int lub = luby(2, S.restart_count++);
                    S.next_restart = S.conflicts + (long long)(100 * lub);
                }
            } else {
                double lb = wcnf_partial_unsat_lower_bound(&wp, &S);
                if (lb >= best_unsat - eps) {
                    if (solver_decision_level(&S) == 0) {
                        if (found_model) {
                            search_exhausted = 1;
                        } else {
                            hard_unsat = 1;
                        }
                        break;
                    }
                    wcnf_build_trail_block_clause(&prune_clause, &S);
                    solver_cancel_until(&S, 0);
                    if (!solver_record_learnt_clause(&S, &prune_clause, (uint32_t)(prune_clause.size > 0 ? prune_clause.size : 1), 0)) {
                        if (S.proof) fputs("0\n", S.proof);
                        aborted_search = 1;
                        break;
                    }
                    prune_clause.size = 0;
                    confl = clauseref_none();
                    continue;
                }

                Lit dlit = target_pick_branch(&S, &guide);
                if (dlit == -1) {
                    if (solver_verify_model(&S)) {
                        target_model_from_assigns(work_model, &S, wp.nv);
                        double wu = wcnf_partial_unsat_lower_bound(&wp, &S);
                        double wu_delta = wu - best_unsat;
                        double abs_wu_delta = wu_delta < 0.0 ? -wu_delta : wu_delta;
                        if (wu < best_unsat - 1e-9 ||
                            (abs_wu_delta <= 1e-9 &&
                             memcmp(work_model, best_model, (size_t)wp.nv) != 0)) {
                            double old_best = best_unsat;
                            best_unsat = wu;
                            memcpy(best_model, work_model, (size_t)wp.nv);
                            target_guide_set_model(&guide, &S, &wp, best_model, &progress);
                            if (sat_log_enabled()) {
                                printf("c CDCL improve: unsat=%.0f/%.0f (full model)\n", best_unsat, wp.tw);
                                fflush(stdout);
                            }
                            if (convergence_enabled && best_unsat < old_best - eps && best_unsat <= convergence_zone) {
                                convergence_hits++;
                                last_convergence_delta = old_best - best_unsat;
                                last_best_decisions = S.decisions;
                                last_best_conflicts = S.conflicts;
                            }
                        }
                        found_model = 1;
                        if (best_unsat < eps) {
                            search_exhausted = 1;
                            break;
                        }
                        int refresh_exact = 0;
                        int refresh_rounds = target_refresh_round_budget(&wp, best_unsat, refresh_count);
                        double before_refresh = best_unsat;
                        if (target_refresh_from_state(&wp, &S, &guide, &progress, &occ,
                                                      best_model, &best_unsat, seed, work_model,
                                                      refresh_rounds, &refresh_exact)) {
                            found_model = 1;
                        }
                        if (convergence_enabled && best_unsat < before_refresh - eps && best_unsat <= convergence_zone) {
                            convergence_hits++;
                            last_convergence_delta = before_refresh - best_unsat;
                            last_best_decisions = S.decisions;
                            last_best_conflicts = S.conflicts;
                        }
                        refresh_count++;
                        if (refresh_exact) {
                            search_exhausted = 1;
                            break;
                        }
                    }

                    wcnf_build_full_block_clause(&prune_clause, &S);
                    solver_cancel_until(&S, 0);
                    if (!solver_record_learnt_clause(&S, &prune_clause, (uint32_t)(prune_clause.size > 0 ? prune_clause.size : 1), 0)) {
                        if (S.proof) fputs("0\n", S.proof);
                        aborted_search = 1;
                        break;
                    }
                    prune_clause.size = 0;
                    confl = clauseref_none();
                    continue;
                }

                intvec_push(&S.trail_lim, S.trail_size);
                S.decisions++;
                S.mab_epoch_decisions += 1.0;
                if (!solver_enqueue(&S, dlit, reason_none())) {
                    S.ok = 0;
                    break;
                }
            }

            confl = solver_propagate(&S);
            if (solver_poll_external_stop(&S)) break;
            solver_update_chb_after_propagate(&S, !clauseref_is_none(confl));

            target_snapshot_update(&S, &progress, &guide, &wp);
        }
    }

finalize:
    if (solver_ready && guide.target != NULL) {
        target_snapshot_update(&S, &progress, &guide, &wp);
    }

    int rc = 0;
    if (!aborted_search && hard_unsat && !found_model) {
        rc = 20;
    } else if (search_exhausted || best_unsat < eps) {
        rc = 10;
    }

    if (sat_log_enabled()) {
        printf("c target_matches=%d target_mismatches=%d unassigned=%d\n",
            progress.matches, progress.mismatches, progress.unassigned);
        printf("o %.0lf\n", best_unsat);            
        if (rc == 20 && slime_validate) {
            printf("s UNSATISFIABLE\n");
        } else if (rc == 10 && slime_validate) {
            printf("s %s\n", best_unsat < eps ? "SATISFIABLE" : "OPTIMUM FOUND");
        } else {
            if (slime_validate) {
                printf("s UNKNOWN\n");
            }
        }
        if (found_model && slime_validate) {
            for (int i = 0; i < wp.nv; ++i) {
                if (i % 10 == 0) printf("\nv ");
                printf("%i ", best_model[i] ? (i + 1) : -(i + 1));
            }
            printf("0\n");
        }

        if (found_model && slime_validate) {
            double ck = target_full_unsat_weight(&wp, best_model);
            double d = ck - best_unsat;
            if (d < 0) d = -d;
            printf("c validate: %s (unsat=%.0f %s)\n",
                d < 1e-6 ? "OK" : "MISMATCH",
                ck,
                (rc == 10) ? "matches optimum" : "matches incumbent");
        }
        fflush(stdout);
    }

    if (found_model) {
        if (model01 != NULL) {
            memcpy(model01, best_model, (size_t)wp.nv * sizeof(unsigned char));
        }
        if (optimal_cost != NULL) {
            *optimal_cost = best_unsat;
        }
    } else if (optimal_cost != NULL) {
        *optimal_cost = 0.0;
    }

    intvec_free(&learnt);
    intvec_free(&prune_clause);
    free(guide.target);
    free(guide.pressure);
    free(guide.escape);
    free(best_model);
    free(seed);
    free(work_model);
    hessw_occ_free(&occ);
    if (solver_ready) solver_destroy(&S);
    wcnf_free(&wp);
    free(local_weights);
    return rc;
}


int slime_entry(int argc, char **argv) {
    const char *input_path = NULL;
    const char *proof_path = NULL;
    KrbParallelConfig parallel_cfg;
    KrbParallelRuntime parallel_rt;
    int print_stats = 0;
    int print_model_flag = 1;
    int mode = SLIME_MODE_SOLVE;
    int heuristic_mode = 1;
    int use_mab = 1;
    double mabc = 0.2;
    int use_hess = 1;
    int use_ct = 1;
    int ct_lbd_max = 6;
    int ct_maxlen = 12;
    int ct_max_cubes = 40000;
    int ct_buddy_merge = 1;
    int ct_escape_rounds = 4;
    int ct_probe_restarts = 8;
    int use_simplify = 1;
    int use_bve = 0;
    int use_chrono = 1;
    int use_inprocess = 0;
    int use_probe = 1;
    int run_selftest_flag = 0;

    krb_parallel_config_defaults(&parallel_cfg);

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "--cdcl") == 0) {
            continue;
        } else if (strcmp(a, "--parallel") == 0 && i + 1 < argc) {
            if (!krb_parallel_parse_mode(argv[++i], &parallel_cfg.mode)) {
                fprintf(stderr, "c ERROR: invalid --parallel '%s'\n", argv[i]);
                return 1;
            }
        } else if (strcmp(a, "--jobs") == 0 && i + 1 < argc) {
            long long jobs = 0;
            if (!parse_ll(argv[++i], &jobs) || jobs < 1 || jobs > INT_MAX) {
                fprintf(stderr, "c ERROR: invalid --jobs '%s'\n", argv[i]);
                return 1;
            }
            parallel_cfg.jobs = (int)jobs;
        } else if (strcmp(a, "--split-depth") == 0 && i + 1 < argc) {
            long long depth = 0;
            if (!parse_ll(argv[++i], &depth) || depth < 0 || depth > INT_MAX) {
                fprintf(stderr, "c ERROR: invalid --split-depth '%s'\n", argv[i]);
                return 1;
            }
            parallel_cfg.split_depth = (int)depth;
        } else if (strcmp(a, "--portfolio") == 0 && i + 1 < argc) {
            long long portfolio = 0;
            if (!parse_ll(argv[++i], &portfolio) || portfolio < 1 || portfolio > INT_MAX) {
                fprintf(stderr, "c ERROR: invalid --portfolio '%s'\n", argv[i]);
                return 1;
            }
            parallel_cfg.portfolio = (int)portfolio;
        } else if (strcmp(a, "--sync-ms") == 0 && i + 1 < argc) {
            long long sync_ms = 0;
            if (!parse_ll(argv[++i], &sync_ms) || sync_ms < 0 || sync_ms > INT_MAX) {
                fprintf(stderr, "c ERROR: invalid --sync-ms '%s'\n", argv[i]);
                return 1;
            }
            parallel_cfg.sync_ms = (int)sync_ms;
        } else if (strcmp(a, "--mode") == 0 && i + 1 < argc) {
            if (!parse_slime_mode(argv[++i], &mode)) {
                fprintf(stderr, "c ERROR: invalid mode '%s'\n", argv[i]);
                return 1;
            }
        } else if (strcmp(a, "--selftest") == 0) {
            run_selftest_flag = 1;
        } else if (strcmp(a, "--no-model") == 0) {
            print_model_flag = 0;
        } else if (strcmp(a, "--stats") == 0) {
            print_stats = 1;
        } else if (strcmp(a, "--proof") == 0 && i + 1 < argc) {
            proof_path = argv[++i];
        } else if (strcmp(a, "--heuristic") == 0 && i + 1 < argc) {
            heuristic_mode = atoi(argv[++i]);
        } else if (strcmp(a, "--mab") == 0) {
            use_mab = 1;
        } else if (strcmp(a, "--no-mab") == 0) {
            use_mab = 0;
        } else if (strcmp(a, "--mabc") == 0 && i + 1 < argc) {
            mabc = atof(argv[++i]);
        } else if (strcmp(a, "--hess") == 0) {
            use_hess = 1;
        } else if (strcmp(a, "--no-hess") == 0) {
            use_hess = 0;
        } else if (strcmp(a, "--ct") == 0) {
            use_ct = 1;
        } else if (strcmp(a, "--no-ct") == 0) {
            use_ct = 0;
        } else if (strcmp(a, "--ct-lbd-max") == 0 && i + 1 < argc) {
            ct_lbd_max = atoi(argv[++i]);
        } else if (strcmp(a, "--ct-maxlen") == 0 && i + 1 < argc) {
            ct_maxlen = atoi(argv[++i]);
        } else if (strcmp(a, "--ct-max-cubes") == 0 && i + 1 < argc) {
            ct_max_cubes = atoi(argv[++i]);
        } else if (strcmp(a, "--ct-buddy") == 0) {
            ct_buddy_merge = 1;
        } else if (strcmp(a, "--ct-no-buddy") == 0) {
            ct_buddy_merge = 0;
        } else if (strcmp(a, "--ct-escape-rounds") == 0 && i + 1 < argc) {
            ct_escape_rounds = atoi(argv[++i]);
        } else if (strcmp(a, "--ct-probe-restarts") == 0 && i + 1 < argc) {
            ct_probe_restarts = atoi(argv[++i]);
        } else if (strcmp(a, "--simplify") == 0) {
            use_simplify = 1;
        } else if (strcmp(a, "--no-simplify") == 0) {
            use_simplify = 0;
        } else if (strcmp(a, "--chrono") == 0) {
            use_chrono = 1;
        } else if (strcmp(a, "--no-chrono") == 0) {
            use_chrono = 0;
        } else if (strcmp(a, "--inprocess") == 0) {
            use_inprocess = 1;
        } else if (strcmp(a, "--no-inprocess") == 0) {
            use_inprocess = 0;
        } else if (strcmp(a, "--bve") == 0) {
            use_bve = 1;
        } else if (strcmp(a, "--no-bve") == 0) {
            use_bve = 0;
        } else if (strcmp(a, "--probe") == 0) {
            use_probe = 1;
        } else if (strcmp(a, "--no-probe") == 0) {
            use_probe = 0;
        } else if ((strcmp(a, "--slime-hess-nohit-limit") == 0 ||
                    strcmp(a, "--slime-ct-idle-soft-limit") == 0 ||
                    strcmp(a, "--slime-ct-idle-off-limit") == 0 ||
                    strcmp(a, "--slime-ct-probe-max-restarts") == 0) && i + 1 < argc) {
            ++i;
        } else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            fprintf(stderr,
                    "usage: %s [--mode solve|count|project] [--cdcl] [--no-model] [--stats]\n"
                    "          [--parallel auto|off|threads|mpi|hybrid] [--jobs N]\n"
                    "          [--split-depth N] [--portfolio N] [--sync-ms N]\n"
                    "          [--proof out.drat]\n"
                    "          [--heuristic 0|1] [--mab|--no-mab] [--mabc C]\n"
                    "          [--hess|--no-hess]\n"
                    "          [--ct|--no-ct] [--ct-lbd-max N] [--ct-maxlen N]\n"
                    "          [--ct-max-cubes N] [--ct-buddy|--ct-no-buddy]\n"
                    "          [--ct-escape-rounds N] [--ct-probe-restarts N]\n"
                    "          [--simplify|--no-simplify] [--chrono|--no-chrono]\n"
                    "          [--inprocess|--no-inprocess] [--probe|--no-probe]\n"
                    "          [--slime-hess-nohit-limit N] [--slime-ct-idle-soft-limit N]\n"
                    "          [--slime-ct-idle-off-limit N] [--slime-ct-probe-max-restarts N]\n"
                    "          count/project reuses --hess/--ct/--ct-* as residual BASILISK policy\n"
                    "          [--selftest]\n"
                    "          <input.cnf>\n",
                    argv[0]);
            return 0;
        } else if (a[0] == '-') {
            // Compatibility: ignore unsupported flags instead of failing hard.
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                // leave next token untouched unless user intended value; no reliable generic rule.
            }
        } else {
            input_path = a;
        }
    }

    if (mode != SLIME_MODE_SOLVE) {
#if defined(BASILISK_NO_MAIN)
        return basilisk_entry(argc, argv);
#else
        fprintf(stderr, "c ERROR: this slime build does not include the basilisk counting backend\n");
        return 2;
#endif
    }

    if (run_selftest_flag) {
        return slime_run_selftest();
    }

    char parallel_err[256];
    parallel_err[0] = '\0';
    if (!krb_parallel_runtime_resolve(&parallel_cfg, &parallel_rt, parallel_err, sizeof(parallel_err))) {
        fprintf(stderr, "c ERROR: %s\n", parallel_err[0] ? parallel_err : "unsupported parallel configuration");
        return 1;
    }

    if (!input_path) {
        fprintf(stderr,
                "c usage: %s [--mode solve|count|project] [--cdcl] [--no-model] [--stats] [--proof out.drat] <input.cnf>\n",
                argv[0]);
        return 1;
    }

    {
        SlimeSatOptions cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.heuristic_mode = heuristic_mode;
        cfg.use_mab = use_mab;
        cfg.mabc = mabc;
        cfg.use_hess = use_hess;
        cfg.use_ct = use_ct;
        cfg.ct_lbd_max = ct_lbd_max;
        cfg.ct_maxlen = ct_maxlen;
        cfg.ct_max_cubes = ct_max_cubes;
        cfg.ct_buddy_merge = ct_buddy_merge;
        cfg.ct_escape_rounds = ct_escape_rounds;
        cfg.ct_probe_restarts = ct_probe_restarts;
        cfg.use_simplify = use_simplify;
        cfg.use_bve = use_bve;
        cfg.use_chrono = use_chrono;
        cfg.use_inprocess = use_inprocess;
        cfg.use_probe = use_probe;

#if defined(SATX_HAVE_THREADS)
        if (parallel_rt.resolved_mode == KRB_PARALLEL_MODE_THREADS && parallel_rt.jobs > 1 && proof_path == NULL) {
            SlimeCnfProblem problem;
            SlimeParallelRunResult pres;
            double t0 = now_sec();
            double t1;

            if (!slime_problem_parse_dimacs(input_path, &problem)) {
                return 1;
            }
            if (!slime_run_parallel_solve(&problem,
                                          &cfg,
                                          &parallel_rt,
                                          &parallel_cfg,
                                          &pres,
                                          parallel_err,
                                          sizeof(parallel_err))) {
                fprintf(stderr, "c ERROR: %s\n", parallel_err[0] ? parallel_err : "parallel solve failed");
                slime_problem_free(&problem);
                return 1;
            }
            t1 = now_sec();

            if (pres.rc == 10) {
                puts("s SATISFIABLE");
                if (print_model_flag) slime_print_model01(problem.nvars, pres.model01);
            } else if (pres.rc == 20) {
                puts("s UNSATISFIABLE");
            } else {
                puts("s UNKNOWN");
            }

            if (print_stats) {
                printf("c stats vars=%d clauses=%d learnt=%lld conflicts=%lld decisions=%lld propagations=%lld restarts=%lld sec=%.3f\n",
                       problem.nvars,
                       problem.nclauses,
                       pres.stats.learnt,
                       pres.stats.conflicts,
                       pres.stats.decisions,
                       pres.stats.propagations,
                       pres.stats.restarts,
                       t1 - t0);
                printf("c parallel strategy=%s jobs=%d portfolio=%d split_depth=%d cubes=%d\n",
                       (pres.strategy == SLIME_PAR_STRATEGY_CUBES) ? "cube-and-conquer" : "portfolio",
                       parallel_rt.jobs,
                       parallel_cfg.portfolio,
                       parallel_cfg.split_depth,
                       pres.cubes_generated);
                printf("c hess calls=%lld sat_hits=%lld\n",
                       pres.stats.hess_calls,
                       pres.stats.hess_sat_hits);
                printf("c covertrace added=%lld merged=%lld escaped=%lld probe_added=%lld\n",
                       pres.stats.ct_added,
                       pres.stats.ct_merged,
                       pres.stats.ct_escaped,
                       pres.stats.ct_probe_added);
            }

            slime_parallel_result_free(&pres);
            slime_problem_free(&problem);
            if (pres.rc == 10) return 10;
            if (pres.rc == 20) return 20;
            return 0;
        }
#endif

        if (parallel_rt.resolved_mode != KRB_PARALLEL_MODE_OFF) {
            if (parallel_rt.resolved_mode == KRB_PARALLEL_MODE_THREADS && proof_path != NULL) {
                fprintf(stderr, "c warning: --proof forces serial SLIME in parallel v1\n");
            } else {
                fprintf(stderr,
                        "c warning: slime parallel mode %s requested; solve path remains serial in this build\n",
                        krb_parallel_mode_name(parallel_rt.resolved_mode));
            }
        }

        Solver S;
        memset(&S, 0, sizeof(S));

        double t0 = now_sec();
        long long parsed_clauses = 0;
        if (!parse_dimacs(input_path,
                          &S,
                          &parsed_clauses,
                          (heuristic_mode == 1 || use_mab) ? 1 : 0,
                          use_hess ? 1 : 0)) {
            return 1;
        }

        solver_seed_rng_from_problem(&S, parsed_clauses);

        if (proof_path) {
            S.proof = fopen(proof_path, "wb");
            if (!S.proof) {
                fprintf(stderr, "c ERROR: cannot open proof file '%s': %s\n", proof_path, strerror(errno));
                solver_destroy(&S);
                return 1;
            }
        }

        solver_apply_runtime_options(&S, &cfg);

        int res = solver_solve(&S);
        double t1 = now_sec();

        if (res == 10) {
            puts("s SATISFIABLE");
            if (print_model_flag) print_model(&S);
        } else if (res == 20) {
            puts("s UNSATISFIABLE");
        } else {
            puts("s UNKNOWN");
        }

        if (print_stats) {
            size_t mem_bytes = solver_memory_bytes(&S);
            long long total_clauses = (long long)S.clauses.size + S.binary_clauses;
            printf("c stats vars=%d clauses=%lld learnt=%d conflicts=%lld decisions=%lld propagations=%lld restarts=%lld sec=%.3f\n",
                   S.nvars,
                   total_clauses,
                   S.learnts.size,
                   S.conflicts,
                   S.decisions,
                   S.propagations,
                   S.restarts,
                   t1 - t0);
            printf("c parse_clauses=%lld\n", parsed_clauses);
            printf("c binary original=%lld nonbinary=%d\n", S.binary_clauses, S.clauses.size);
            printf("c heur heuristic=%s mab=%d mab_sel_vsids=%u mab_sel_chb=%u\n",
                   (S.heuristic == 1) ? "CHB" : "VSIDS",
                   S.use_mab,
                   S.mab_select[0],
                   S.mab_select[1]);
            printf("c covertrace cubes=%d added=%lld merged=%lld escaped=%lld probe_added=%lld buddy=%d\n",
                   S.ct_cubes.size,
                   S.ct_added,
                   S.ct_merged,
                   S.ct_escaped,
                   S.ct_probe_added,
                   S.ct_buddy_merge);
            printf("c hess calls=%lld sat_hits=%lld unit_clauses=%d empty=%d\n",
                   S.hess_calls,
                   S.hess_sat_hits,
                   S.orig_unit_lits.size,
                   S.orig_empty_clauses);
            printf("c simplify eliminated=%lld subsumed=%lld strengthened=%lld pure=%lld equivs=%lld probe_units=%lld inprocess=%lld chrono=%lld rephase=%lld\n",
                   (long long)S.simp_eliminated,
                   (long long)S.simp_subsumed,
                   (long long)S.simp_strengthened,
                   (long long)S.simp_pure,
                   (long long)S.simp_equivs,
                   (long long)S.simp_probe_units,
                   (long long)S.inprocess_count,
                   (long long)S.chrono_conflicts,
                   (long long)S.rephase_count);
            printf("c mem approx_mb=%.2f freed_clauses=%lld watch_sweeps=%lld\n",
                   (double)mem_bytes / (1024.0 * 1024.0),
                   S.deleted_freed,
                   S.watch_sweeps);
        }

        solver_destroy(&S);

        if (res == 10) return 10;
        if (res == 20) return 20;
        return 0;
    }
}

#ifndef SLIME_NO_MAIN
int main(int argc, char **argv) {
    return slime_entry(argc, argv);
}
#endif























