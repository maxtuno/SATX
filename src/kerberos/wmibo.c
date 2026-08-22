/*
 * Copyright (c) 2026 Oscar Riveros.
 *
 * Licencia dual: uso personal bajo Apache License 2.0; portes a otros
 * lenguajes requieren licencia comercial con autorizacion expresa del autor.
 * Ver LICENSE.txt en la raiz del proyecto para los terminos completos.
 */

/*
Description:
WMIBO-HL is a unified hybrid optimization solver that combines Boolean CNF reasoning,
weighted soft clauses, and mixed-integer linear optimization with a branch-and-bound
engine over LP relaxations. The implementation includes CDCL-style clause learning on
Boolean conflicts and a coarse theory-learning interface for LP infeasibility.

SLIME
Copyright (c) 2026 Oscar Riveros.
All rights reserved.

This source code and associated materials are proprietary.
Unauthorized use, distribution, or modification is prohibited
without explicit written permission from the copyright holder.

gcc -O3 -std=c17 -Wall -Wextra -pedantic wmibo.c krb_accel.c krb_accel_cuda_stub.c -o wmibo
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#endif

#include "krb_accel.h"
#include "krb_parallel.h"

/*
WMIBO text format (.wmibo), line-oriented:

Header:
  p wmibo <B> <I> <R> <NC> <NL> <NIND>
Only B/I/R are required by this parser; remaining declared counts are accepted.

Comments:
  # ...
  c ...

Blocks:
  begin cnf|wcnf|lin|ind|obj
  ...
  end

Clauses:
  cl hard b1 ~b2 0
  cl soft b3 0            (weight 1)
  wcl 7 soft ~b1 b2 0
  wcl 1 hard b4 0

Linear constraints (normalized to <= internally):
  lc C1 <= 4 : 1 r1 2 i1 -3 b2
  lc C2 >= 0 : 1 r1 -1 i1
  lc C3 =  5 : 1 r1

Indicators:
  ind b1  => C1
  ind ~b2 => C2

Objective:
  obj min : lin 1 r1 2 i1 -1 b3
  obj max : lin ...
Soft clause penalties are reified and automatically added to the objective.

Variable declarations:
  var b 1 [0,1]
  var i 1 [0,10]
  var i 2 bin
  var r 1 free
  var r 2 [0,5]
*/

#define WMIBO_MAX_LINE 8192
#define WMIBO_INF 1e30
#define WMIBO_BIG_BOUND 1e9
#define WMIBO_FEAS_TOL_DEFAULT 1e-8
#define WMIBO_INT_TOL_DEFAULT 1e-6
#define WMIBO_PIV_TOL_DEFAULT 1e-10
#define WMIBO_OBJ_TOL_DEFAULT 1e-9
#define WMIBO_LP_EPS 1e-9
#define WMIBO_REL_GAP_DEFAULT 0.0
#define WMIBO_MAX_DENSE_LP_CELLS 5000000.0
#define WMIBO_BOOL_LB_SINGLETON_PROBES 24
#define WMIBO_BOOL_LB_GROUP_PROBES 12
#define WMIBO_BOOL_LB_GROUP_SIZE 8

typedef enum {
    SOLVE_STATUS_UNKNOWN = 0,
    SOLVE_STATUS_OPTIMUM = 1,
    SOLVE_STATUS_INFEASIBLE = 2,
    SOLVE_STATUS_UNBOUNDED = 3
} SolveStatus;

typedef enum {
    LP_STATUS_OPTIMAL = 0,
    LP_STATUS_INFEASIBLE = 1,
    LP_STATUS_UNBOUNDED = 2,
    LP_STATUS_ERROR = 3
} LPStatus;

typedef enum {
    VAR_BOOL = 0,
    VAR_INT = 1,
    VAR_REAL = 2
} VarKind;

typedef enum {
    MODE_SOLVE = 0,
    MODE_COUNT = 1,
    MODE_PROJECT = 2,
    MODE_VOLUME = 3,
    MODE_EXPLAIN = 4
} QueryMode;

typedef struct {
    int *data;
    int size;
    int cap;
} IntVec;

typedef struct {
    int *lits;
    int size;
    bool learnt;
} Clause;

typedef struct {
    Clause *data;
    int size;
    int cap;
} ClauseVec;

typedef struct {
    int *lits;
    int size;
    double weight;
} SoftClause;

typedef struct {
    SoftClause *data;
    int size;
    int cap;
} SoftVec;

typedef struct {
    VarKind kind;
    int idx;
    double coef;
} LinTerm;

typedef struct {
    LinTerm *terms;
    int nterms;
    double rhs;
    int indicator_lit;
    char name[64];
} LinConstraint;

typedef struct {
    LinConstraint *data;
    int size;
    int cap;
} LinConstraintVec;

typedef struct {
    char name[64];
    int lit;
} PendingIndicator;

typedef struct {
    PendingIndicator *data;
    int size;
    int cap;
} PendingIndicatorVec;

typedef struct {
    bool has_feas_tol;
    bool has_int_tol;
    bool has_time_limit;
    bool has_node_limit;
    bool has_gap;
    bool has_seed;
    bool has_verbose;
    double feas_tol;
    double int_tol;
    double time_limit;
    long long node_limit;
    double rel_gap;
    uint64_t seed;
    int verbose;
} FileOptions;

typedef struct {
    int B;
    int I;
    int R;
    int nb_total;
    int nvars_total;

    double *b_lb_input;
    double *b_ub_input;
    double *i_lb;
    double *i_ub;
    double *r_lb;
    double *r_ub;

    double *obj_b_input;
    double *obj_i;
    double *obj_r;
    bool has_obj_line;
    bool obj_is_max;
    double obj_const;

    ClauseVec clauses;
    SoftVec soft;
    LinConstraintVec lin;
    PendingIndicatorVec pending_ind;

    double *obj_all;
    double *var_lb_all;
    double *var_ub_all;

    bool hard_unsat;
    FileOptions fopt;
} Model;

typedef struct {
    double feas_tol;
    double int_tol;
    double piv_tol;
    double obj_tol;
    double time_limit;
    long long node_limit;
    double rel_gap;
    uint64_t seed;
    bool seed_set;
    int verbose;
    QueryMode mode;
    const char *trace_out;
    const char *core_out;
    KrbAccelConfig accel;
    KrbParallelConfig parallel;
} SolveOptions;

typedef struct {
    int8_t *b_assign;
    double *i_lb;
    double *i_ub;
    IntVec decisions;
} NodeState;

typedef struct {
    LPStatus status;
    double obj;
    double *x;
} LPResult;

#if defined(SLIME_NO_MAIN)
typedef struct {
    int heuristic_mode;
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
SlimeSatHandle *slime_sat_handle_create(int nvars,
                                        int nclauses,
                                        const int *const *clauses,
                                        const int *sizes,
                                        const SlimeSatOptions *opt);
int slime_sat_handle_solve(SlimeSatHandle *handle,
                           const int *assumptions,
                           int num_assumptions,
                           SlimeSatStats *stats,
                           unsigned char *model01);
void slime_sat_handle_destroy(SlimeSatHandle *handle);
#endif

typedef struct {
    int m;
    int n;
    int *B;
    int *N;
    double **D;
    double *buf;
    double *buf_alt;
    int *gpu_candidate_idx;
    double *gpu_candidate_metric;
    int gpu_candidate_cap;
    double deadline;
    bool aborted;
    bool use_cuda;
    bool gpu_error;
} Simplex;

typedef struct {
    Model model;
    SolveOptions opt;

    bool stop;
    bool stopped_time;
    bool stopped_nodes;
    bool stopped_gap;

    bool have_incumbent;
    double incumbent;
    double report_incumbent;
    double root_lb;
    bool have_root_lb;

    bool found_unbounded;
    bool use_bool_satbb;

    int *best_b;
    double *best_i;
    double *best_r;

    long long nodes;
    long long lp_calls;
    long long clause_learned;
    long long bool_conflicts;
    long long theory_conflicts;

    double start_time;
    double *var_activity;
    double *branch_score;
    double *heur_x;
    int *soft_order;
    uint64_t rng_state;
#if defined(SLIME_NO_MAIN)
    SlimeSatHandle *slime_handle;
    const int **slime_clause_ptrs;
    int *slime_clause_sizes;
    int *slime_assumptions;
    unsigned char *slime_model01;
#endif
} Solver;

#if defined(_MSC_VER)
#define WMIBO_THREAD_LOCAL __declspec(thread)
#else
#define WMIBO_THREAD_LOCAL _Thread_local
#endif

static WMIBO_THREAD_LOCAL const char *g_prog_name = "wmibo";
static WMIBO_THREAD_LOCAL const SoftVec *g_soft_sort_view = NULL;

static int lit_var(int lit) { return (lit < 0) ? -lit : lit; }
static int lit_sign(int lit) { return (lit > 0) ? 1 : 0; }
static int lit_neg(int lit) { return -lit; }

static int soft_index_weight_desc_cmp(const void *pa, const void *pb) {
    int ia = *(const int *)pa;
    int ib = *(const int *)pb;
    double wa = g_soft_sort_view->data[ia].weight;
    double wb = g_soft_sort_view->data[ib].weight;
    if (wa < wb) return 1;
    if (wa > wb) return -1;
    return (ia > ib) - (ia < ib);
}

static double now_seconds(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0) {
        if (!QueryPerformanceFrequency(&freq) || freq.QuadPart <= 0) {
            return (double)GetTickCount64() * 1e-3;
        }
    }
    if (QueryPerformanceCounter(&counter)) {
        return (double)counter.QuadPart / (double)freq.QuadPart;
    }
    return (double)GetTickCount64() * 1e-3;
#endif
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static bool deadline_reached(double deadline) {
    return deadline > 0.0 && now_seconds() >= deadline;
}

static bool solver_time_limit_reached(const Solver *s) {
    return s->opt.time_limit > 0.0 && (now_seconds() - s->start_time) >= s->opt.time_limit;
}

static bool solver_poll_time_limit(Solver *s) {
    if (s != NULL && solver_time_limit_reached(s)) {
        s->stop = true;
        s->stopped_time = true;
        return true;
    }
    return false;
}

static bool str_ieq(const char *a, const char *b) {
    while (*a != '\0' && *b != '\0') {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

static bool has_ext_ci(const char *path, const char *ext) {
    size_t lp = strlen(path);
    size_t le = strlen(ext);
    if (lp < le) {
        return false;
    }
    return str_ieq(path + lp - le, ext);
}

static char *trim_left(char *s) {
    while (*s != '\0' && isspace((unsigned char)*s)) {
        ++s;
    }
    return s;
}

static void trim_right(char *s) {
    size_t n = strlen(s);
    while (n > 0U) {
        unsigned char c = (unsigned char)s[n - 1U];
        if (!isspace(c)) {
            break;
        }
        s[n - 1U] = '\0';
        --n;
    }
}

static void trim_inplace(char *s) {
    char *p = trim_left(s);
    if (p != s) {
        memmove(s, p, strlen(p) + 1U);
    }
    trim_right(s);
}

static void strip_inline_comment(char *s) {
    char *p = s;
    while (*p != '\0') {
        if (*p == '#') {
            *p = '\0';
            break;
        }
        ++p;
    }
}

static bool parse_ll(const char *s, long long *out) {
    char *end = NULL;
    long long v;
    errno = 0;
    v = strtoll(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        return false;
    }
    *out = v;
    return true;
}

static bool parse_u64(const char *s, uint64_t *out) {
    char *end = NULL;
    unsigned long long v;
    errno = 0;
    v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        return false;
    }
    *out = (uint64_t)v;
    return true;
}

static bool parse_double_str(const char *s, double *out) {
    char *end = NULL;
    double v;
    errno = 0;
    v = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0') {
        return false;
    }
    *out = v;
    return true;
}

static int eval_lit(int8_t val, int lit) {
    if (val < 0) {
        return -1;
    }
    if (lit > 0) {
        return (val == 1) ? 1 : 0;
    }
    return (val == 0) ? 1 : 0;
}

static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static int token_split(char *line, char **out, int max_tok) {
    int n = 0;
    char *p = line;
    while (*p != '\0') {
        while (*p != '\0' && isspace((unsigned char)*p)) {
            ++p;
        }
        if (*p == '\0') {
            break;
        }
        if (n >= max_tok) {
            break;
        }
        out[n++] = p;
        while (*p != '\0' && !isspace((unsigned char)*p)) {
            ++p;
        }
        if (*p == '\0') {
            break;
        }
        *p = '\0';
        ++p;
    }
    return n;
}

static int cmp_int_abs_then_val(const void *a, const void *b) {
    int x = *(const int *)a;
    int y = *(const int *)b;
    int ax = (x < 0) ? -x : x;
    int ay = (y < 0) ? -y : y;
    if (ax < ay) return -1;
    if (ax > ay) return 1;
    if (x < y) return -1;
    if (x > y) return 1;
    return 0;
}

static void intvec_init(IntVec *v) {
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static void intvec_free(IntVec *v) {
    free(v->data);
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static bool intvec_reserve(IntVec *v, int need) {
    int ncap;
    int *nd;
    if (need <= v->cap) {
        return true;
    }
    ncap = (v->cap > 0) ? v->cap : 8;
    while (ncap < need) {
        if (ncap > INT_MAX / 2) {
            return false;
        }
        ncap *= 2;
    }
    nd = (int *)realloc(v->data, (size_t)ncap * sizeof(int));
    if (nd == NULL) {
        return false;
    }
    v->data = nd;
    v->cap = ncap;
    return true;
}

static bool intvec_push(IntVec *v, int x) {
    if (!intvec_reserve(v, v->size + 1)) {
        return false;
    }
    v->data[v->size++] = x;
    return true;
}

static bool intvec_copy(IntVec *dst, const IntVec *src) {
    intvec_init(dst);
    if (src->size == 0) {
        return true;
    }
    dst->data = (int *)malloc((size_t)src->size * sizeof(int));
    if (dst->data == NULL) {
        return false;
    }
    memcpy(dst->data, src->data, (size_t)src->size * sizeof(int));
    dst->size = src->size;
    dst->cap = src->size;
    return true;
}
static void clause_free(Clause *c) {
    free(c->lits);
    c->lits = NULL;
    c->size = 0;
}

static void clausevec_init(ClauseVec *v) {
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static void clausevec_free(ClauseVec *v) {
    int i;
    for (i = 0; i < v->size; ++i) {
        clause_free(&v->data[i]);
    }
    free(v->data);
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static bool clausevec_reserve(ClauseVec *v, int need) {
    int ncap;
    Clause *nd;
    if (need <= v->cap) {
        return true;
    }
    ncap = (v->cap > 0) ? v->cap : 8;
    while (ncap < need) {
        if (ncap > INT_MAX / 2) {
            return false;
        }
        ncap *= 2;
    }
    nd = (Clause *)realloc(v->data, (size_t)ncap * sizeof(Clause));
    if (nd == NULL) {
        return false;
    }
    v->data = nd;
    v->cap = ncap;
    return true;
}

static bool clausevec_push_owned(ClauseVec *v, Clause *c) {
    if (!clausevec_reserve(v, v->size + 1)) {
        return false;
    }
    v->data[v->size++] = *c;
    c->lits = NULL;
    c->size = 0;
    return true;
}

static void softvec_init(SoftVec *v) {
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static void softvec_free(SoftVec *v) {
    int i;
    for (i = 0; i < v->size; ++i) {
        free(v->data[i].lits);
    }
    free(v->data);
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static bool softvec_reserve(SoftVec *v, int need) {
    int ncap;
    SoftClause *nd;
    if (need <= v->cap) {
        return true;
    }
    ncap = (v->cap > 0) ? v->cap : 8;
    while (ncap < need) {
        if (ncap > INT_MAX / 2) {
            return false;
        }
        ncap *= 2;
    }
    nd = (SoftClause *)realloc(v->data, (size_t)ncap * sizeof(SoftClause));
    if (nd == NULL) {
        return false;
    }
    v->data = nd;
    v->cap = ncap;
    return true;
}

static bool softvec_push_owned(SoftVec *v, SoftClause *c) {
    if (!softvec_reserve(v, v->size + 1)) {
        return false;
    }
    v->data[v->size++] = *c;
    c->lits = NULL;
    c->size = 0;
    c->weight = 0.0;
    return true;
}

static void linconstraint_free(LinConstraint *c) {
    free(c->terms);
    c->terms = NULL;
    c->nterms = 0;
    c->rhs = 0.0;
    c->indicator_lit = 0;
    c->name[0] = '\0';
}

static void linvec_init(LinConstraintVec *v) {
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static void linvec_free(LinConstraintVec *v) {
    int i;
    for (i = 0; i < v->size; ++i) {
        linconstraint_free(&v->data[i]);
    }
    free(v->data);
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static bool linvec_reserve(LinConstraintVec *v, int need) {
    int ncap;
    LinConstraint *nd;
    if (need <= v->cap) {
        return true;
    }
    ncap = (v->cap > 0) ? v->cap : 8;
    while (ncap < need) {
        if (ncap > INT_MAX / 2) {
            return false;
        }
        ncap *= 2;
    }
    nd = (LinConstraint *)realloc(v->data, (size_t)ncap * sizeof(LinConstraint));
    if (nd == NULL) {
        return false;
    }
    v->data = nd;
    v->cap = ncap;
    return true;
}

static bool linvec_push_owned(LinConstraintVec *v, LinConstraint *c) {
    if (!linvec_reserve(v, v->size + 1)) {
        return false;
    }
    v->data[v->size++] = *c;
    c->terms = NULL;
    c->nterms = 0;
    c->rhs = 0.0;
    c->indicator_lit = 0;
    c->name[0] = '\0';
    return true;
}

static void pendvec_init(PendingIndicatorVec *v) {
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static void pendvec_free(PendingIndicatorVec *v) {
    free(v->data);
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static bool pendvec_reserve(PendingIndicatorVec *v, int need) {
    int ncap;
    PendingIndicator *nd;
    if (need <= v->cap) {
        return true;
    }
    ncap = (v->cap > 0) ? v->cap : 8;
    while (ncap < need) {
        if (ncap > INT_MAX / 2) {
            return false;
        }
        ncap *= 2;
    }
    nd = (PendingIndicator *)realloc(v->data, (size_t)ncap * sizeof(PendingIndicator));
    if (nd == NULL) {
        return false;
    }
    v->data = nd;
    v->cap = ncap;
    return true;
}

static bool pendvec_push(PendingIndicatorVec *v, const PendingIndicator *p) {
    if (!pendvec_reserve(v, v->size + 1)) {
        return false;
    }
    v->data[v->size++] = *p;
    return true;
}

static void model_init(Model *m) {
    memset(m, 0, sizeof(*m));
    clausevec_init(&m->clauses);
    softvec_init(&m->soft);
    linvec_init(&m->lin);
    pendvec_init(&m->pending_ind);
}

static void model_free(Model *m) {
    free(m->b_lb_input);
    free(m->b_ub_input);
    free(m->i_lb);
    free(m->i_ub);
    free(m->r_lb);
    free(m->r_ub);
    free(m->obj_b_input);
    free(m->obj_i);
    free(m->obj_r);
    free(m->obj_all);
    free(m->var_lb_all);
    free(m->var_ub_all);
    clausevec_free(&m->clauses);
    softvec_free(&m->soft);
    linvec_free(&m->lin);
    pendvec_free(&m->pending_ind);
    memset(m, 0, sizeof(*m));
}

static void solver_defaults(SolveOptions *opt) {
    opt->feas_tol = WMIBO_FEAS_TOL_DEFAULT;
    opt->int_tol = WMIBO_INT_TOL_DEFAULT;
    opt->piv_tol = WMIBO_PIV_TOL_DEFAULT;
    opt->obj_tol = WMIBO_OBJ_TOL_DEFAULT;
    opt->time_limit = 0.0;
    opt->node_limit = 0;
    opt->rel_gap = WMIBO_REL_GAP_DEFAULT;
    opt->seed = 1U;
    opt->seed_set = false;
    opt->verbose = 1;
    opt->mode = MODE_SOLVE;
    opt->trace_out = NULL;
    opt->core_out = NULL;
    krb_accel_config_defaults(&opt->accel);
    krb_parallel_config_defaults(&opt->parallel);
}

static bool ensure_model_dims(Model *m, int B, int I, int R, char *err, size_t errsz) {
    int i;
    m->B = B;
    m->I = I;
    m->R = R;

    m->b_lb_input = (double *)calloc((size_t)B + 1U, sizeof(double));
    m->b_ub_input = (double *)calloc((size_t)B + 1U, sizeof(double));
    m->i_lb = (double *)calloc((size_t)I + 1U, sizeof(double));
    m->i_ub = (double *)calloc((size_t)I + 1U, sizeof(double));
    m->r_lb = (double *)calloc((size_t)R + 1U, sizeof(double));
    m->r_ub = (double *)calloc((size_t)R + 1U, sizeof(double));

    m->obj_b_input = (double *)calloc((size_t)B + 1U, sizeof(double));
    m->obj_i = (double *)calloc((size_t)I + 1U, sizeof(double));
    m->obj_r = (double *)calloc((size_t)R + 1U, sizeof(double));

    if ((B > 0 && (m->b_lb_input == NULL || m->b_ub_input == NULL || m->obj_b_input == NULL)) ||
        (I > 0 && (m->i_lb == NULL || m->i_ub == NULL || m->obj_i == NULL)) ||
        (R > 0 && (m->r_lb == NULL || m->r_ub == NULL || m->obj_r == NULL))) {
        snprintf(err, errsz, "out of memory allocating model variables");
        return false;
    }

    for (i = 1; i <= B; ++i) {
        m->b_lb_input[i] = 0.0;
        m->b_ub_input[i] = 1.0;
    }
    for (i = 1; i <= I; ++i) {
        m->i_lb[i] = -WMIBO_INF;
        m->i_ub[i] = WMIBO_INF;
    }
    for (i = 1; i <= R; ++i) {
        m->r_lb[i] = -WMIBO_INF;
        m->r_ub[i] = WMIBO_INF;
    }
    return true;
}

static bool parse_bounds_token(const char *tok, double *lb, double *ub) {
    const char *p = tok;
    char *end = NULL;
    double a;
    double b;
    if (strcmp(tok, "free") == 0) {
        *lb = -WMIBO_INF;
        *ub = WMIBO_INF;
        return true;
    }
    if (strcmp(tok, "bin") == 0) {
        *lb = 0.0;
        *ub = 1.0;
        return true;
    }
    if (*p != '[') {
        return false;
    }
    ++p;
    errno = 0;
    a = strtod(p, &end);
    if (errno != 0 || end == p || *end != ',') {
        return false;
    }
    p = end + 1;
    errno = 0;
    b = strtod(p, &end);
    if (errno != 0 || end == p || *end != ']') {
        return false;
    }
    if (*(end + 1) != '\0') {
        return false;
    }
    *lb = a;
    *ub = b;
    return true;
}

static bool parse_bool_lit_token(const char *tok, int B, int *out_lit) {
    int sign = 1;
    const char *p = tok;
    long long idx;
    if (*p == '~') {
        sign = -1;
        ++p;
    }
    if (*p != 'b' && *p != 'B') {
        return false;
    }
    ++p;
    if (!parse_ll(p, &idx)) {
        return false;
    }
    if (idx < 1 || idx > B) {
        return false;
    }
    *out_lit = sign * (int)idx;
    return true;
}

static bool parse_var_ref_token(const char *tok, int B, int I, int R, VarKind *out_kind, int *out_idx) {
    char t;
    long long idx;
    if (tok == NULL || tok[0] == '\0') {
        return false;
    }
    t = (char)tolower((unsigned char)tok[0]);
    if (t != 'b' && t != 'i' && t != 'r') {
        return false;
    }
    if (!parse_ll(tok + 1, &idx)) {
        return false;
    }
    if (t == 'b') {
        if (idx < 1 || idx > B) return false;
        *out_kind = VAR_BOOL;
        *out_idx = (int)idx;
    } else if (t == 'i') {
        if (idx < 1 || idx > I) return false;
        *out_kind = VAR_INT;
        *out_idx = (int)idx;
    } else {
        if (idx < 1 || idx > R) return false;
        *out_kind = VAR_REAL;
        *out_idx = (int)idx;
    }
    return true;
}
static bool add_clause_core(Model *m, const int *lits, int n, bool learnt, bool *added) {
    Clause c;
    int *tmp;
    int i;
    int w;
    if (added != NULL) {
        *added = false;
    }
    if (n < 0) {
        return false;
    }
    if (n == 0) {
        m->hard_unsat = true;
        if (added != NULL) {
            *added = true;
        }
        return true;
    }
    tmp = (int *)malloc((size_t)n * sizeof(int));
    if (tmp == NULL) {
        return false;
    }
    memcpy(tmp, lits, (size_t)n * sizeof(int));
    qsort(tmp, (size_t)n, sizeof(int), cmp_int_abs_then_val);

    w = 0;
    for (i = 0; i < n; ++i) {
        int lit = tmp[i];
        if (lit == 0) {
            continue;
        }
        if (w > 0 && tmp[w - 1] == lit) {
            continue;
        }
        if (w > 0 && tmp[w - 1] == -lit) {
            free(tmp);
            if (added != NULL) {
                *added = false;
            }
            return true;
        }
        tmp[w++] = lit;
    }
    if (w == 0) {
        free(tmp);
        if (added != NULL) {
            *added = false;
        }
        return true;
    }

    c.lits = (int *)malloc((size_t)w * sizeof(int));
    if (c.lits == NULL) {
        free(tmp);
        return false;
    }
    memcpy(c.lits, tmp, (size_t)w * sizeof(int));
    c.size = w;
    c.learnt = learnt;
    free(tmp);

    if (!clausevec_push_owned(&m->clauses, &c)) {
        free(c.lits);
        return false;
    }
    if (added != NULL) {
        *added = true;
    }
    return true;
}

static bool add_clause_hard(Model *m, const int *lits, int n) {
    return add_clause_core(m, lits, n, false, NULL);
}

static bool add_clause_learned(Model *m, const int *lits, int n, bool *added) {
    return add_clause_core(m, lits, n, true, added);
}

static bool add_soft_clause(Model *m, const int *lits, int n, double w) {
    SoftClause c;
    int *tmp;
    if (n <= 0) {
        return true;
    }
    tmp = (int *)malloc((size_t)n * sizeof(int));
    if (tmp == NULL) {
        return false;
    }
    memcpy(tmp, lits, (size_t)n * sizeof(int));
    c.lits = tmp;
    c.size = n;
    c.weight = w;
    if (!softvec_push_owned(&m->soft, &c)) {
        free(tmp);
        return false;
    }
    return true;
}

static bool add_lin_constraint_owned(Model *m, LinConstraint *c, char *err, size_t errsz) {
    if (!linvec_push_owned(&m->lin, c)) {
        snprintf(err, errsz, "out of memory storing linear constraint");
        return false;
    }
    return true;
}

static bool add_lin_constraint_normalized(Model *m,
                                          const char *name,
                                          const LinTerm *terms,
                                          int nterms,
                                          const char *sense,
                                          double rhs,
                                          char *err,
                                          size_t errsz) {
    LinConstraint c1;
    LinConstraint c2;
    int i;
    memset(&c1, 0, sizeof(c1));
    memset(&c2, 0, sizeof(c2));

    if (str_ieq(sense, "<=")) {
        c1.terms = (LinTerm *)malloc((size_t)nterms * sizeof(LinTerm));
        if (c1.terms == NULL && nterms > 0) {
            snprintf(err, errsz, "out of memory storing linear terms");
            return false;
        }
        for (i = 0; i < nterms; ++i) {
            c1.terms[i] = terms[i];
        }
        c1.nterms = nterms;
        c1.rhs = rhs;
        c1.indicator_lit = 0;
        snprintf(c1.name, sizeof(c1.name), "%s", name);
        return add_lin_constraint_owned(m, &c1, err, errsz);
    }

    if (str_ieq(sense, ">=")) {
        c1.terms = (LinTerm *)malloc((size_t)nterms * sizeof(LinTerm));
        if (c1.terms == NULL && nterms > 0) {
            snprintf(err, errsz, "out of memory storing linear terms");
            return false;
        }
        for (i = 0; i < nterms; ++i) {
            c1.terms[i] = terms[i];
            c1.terms[i].coef = -c1.terms[i].coef;
        }
        c1.nterms = nterms;
        c1.rhs = -rhs;
        c1.indicator_lit = 0;
        snprintf(c1.name, sizeof(c1.name), "%s", name);
        return add_lin_constraint_owned(m, &c1, err, errsz);
    }

    if (strcmp(sense, "=") == 0) {
        c1.terms = (LinTerm *)malloc((size_t)nterms * sizeof(LinTerm));
        c2.terms = (LinTerm *)malloc((size_t)nterms * sizeof(LinTerm));
        if ((c1.terms == NULL || c2.terms == NULL) && nterms > 0) {
            free(c1.terms);
            free(c2.terms);
            snprintf(err, errsz, "out of memory storing equality terms");
            return false;
        }
        for (i = 0; i < nterms; ++i) {
            c1.terms[i] = terms[i];
            c2.terms[i] = terms[i];
            c2.terms[i].coef = -c2.terms[i].coef;
        }
        c1.nterms = nterms;
        c1.rhs = rhs;
        c1.indicator_lit = 0;
        snprintf(c1.name, sizeof(c1.name), "%s", name);
        c2.nterms = nterms;
        c2.rhs = -rhs;
        c2.indicator_lit = 0;
        snprintf(c2.name, sizeof(c2.name), "%s", name);
        if (!add_lin_constraint_owned(m, &c1, err, errsz)) {
            linconstraint_free(&c1);
            linconstraint_free(&c2);
            return false;
        }
        if (!add_lin_constraint_owned(m, &c2, err, errsz)) {
            linconstraint_free(&c2);
            return false;
        }
        return true;
    }

    snprintf(err, errsz, "unsupported linear sense '%s'", sense);
    return false;
}

static bool parse_var_decl(Model *m, char *line, char *err, size_t errsz) {
    char *tok[64];
    int nt;
    char kind;
    long long idxll;
    int idx;
    double lb;
    double ub;

    nt = token_split(line, tok, 64);
    if (nt < 4) {
        snprintf(err, errsz, "invalid var declaration");
        return false;
    }

    if (strlen(tok[1]) != 1U) {
        snprintf(err, errsz, "invalid var kind '%s'", tok[1]);
        return false;
    }
    kind = (char)tolower((unsigned char)tok[1][0]);

    if (!parse_ll(tok[2], &idxll)) {
        snprintf(err, errsz, "invalid var index '%s'", tok[2]);
        return false;
    }
    idx = (int)idxll;

    if (!parse_bounds_token(tok[3], &lb, &ub)) {
        snprintf(err, errsz, "invalid var bounds '%s'", tok[3]);
        return false;
    }

    if (kind == 'b') {
        if (idx < 1 || idx > m->B) {
            snprintf(err, errsz, "bool var index out of range: %d", idx);
            return false;
        }
        m->b_lb_input[idx] = lb;
        m->b_ub_input[idx] = ub;
        if (m->b_lb_input[idx] < 0.0) m->b_lb_input[idx] = 0.0;
        if (m->b_ub_input[idx] > 1.0) m->b_ub_input[idx] = 1.0;
        return true;
    }

    if (kind == 'i') {
        if (idx < 1 || idx > m->I) {
            snprintf(err, errsz, "int var index out of range: %d", idx);
            return false;
        }
        m->i_lb[idx] = lb;
        m->i_ub[idx] = ub;
        return true;
    }

    if (kind == 'r') {
        if (idx < 1 || idx > m->R) {
            snprintf(err, errsz, "real var index out of range: %d", idx);
            return false;
        }
        m->r_lb[idx] = lb;
        m->r_ub[idx] = ub;
        return true;
    }

    snprintf(err, errsz, "invalid var kind '%c'", kind);
    return false;
}

static bool parse_cnf_clause_line(Model *m, char *line, bool wmode, char *err, size_t errsz) {
    char *tok[4096];
    int nt;
    int p;
    bool hard = true;
    double weight = 1.0;
    int lits[2048];
    int nl = 0;
    int i;

    nt = token_split(line, tok, 4096);
    if (nt <= 0) {
        return true;
    }

    if (!wmode) {
        if (nt < 4 || !str_ieq(tok[0], "cl")) {
            snprintf(err, errsz, "invalid cl line");
            return false;
        }
        if (str_ieq(tok[1], "hard")) {
            hard = true;
        } else if (str_ieq(tok[1], "soft")) {
            hard = false;
            weight = 1.0;
        } else {
            snprintf(err, errsz, "invalid clause type '%s'", tok[1]);
            return false;
        }
        p = 2;
    } else {
        if (nt < 5 || !str_ieq(tok[0], "wcl")) {
            snprintf(err, errsz, "invalid wcl line");
            return false;
        }
        if (!parse_double_str(tok[1], &weight) || weight < 0.0) {
            snprintf(err, errsz, "invalid weight '%s'", tok[1]);
            return false;
        }
        if (str_ieq(tok[2], "hard")) {
            hard = true;
        } else if (str_ieq(tok[2], "soft")) {
            hard = false;
        } else {
            snprintf(err, errsz, "invalid hard/soft token '%s'", tok[2]);
            return false;
        }
        p = 3;
    }

    for (i = p; i < nt; ++i) {
        int lit;
        if (strcmp(tok[i], "0") == 0) {
            break;
        }
        if (!parse_bool_lit_token(tok[i], m->B, &lit)) {
            snprintf(err, errsz, "invalid boolean literal '%s'", tok[i]);
            return false;
        }
        if (nl >= (int)(sizeof(lits) / sizeof(lits[0]))) {
            snprintf(err, errsz, "clause too long");
            return false;
        }
        lits[nl++] = lit;
    }

    if (i == nt || strcmp(tok[i], "0") != 0) {
        snprintf(err, errsz, "clause missing trailing 0");
        return false;
    }

    if (hard) {
        if (!add_clause_hard(m, lits, nl)) {
            snprintf(err, errsz, "out of memory storing hard clause");
            return false;
        }
    } else {
        if (!add_soft_clause(m, lits, nl, weight)) {
            snprintf(err, errsz, "out of memory storing soft clause");
            return false;
        }
    }

    return true;
}
static bool parse_lc_terms(const Model *m, char *expr, LinTerm **out_terms, int *out_n, char *err, size_t errsz) {
    char *tok[2048];
    int nt;
    int i;
    LinTerm *terms;

    trim_inplace(expr);
    nt = token_split(expr, tok, 2048);
    if (nt == 0 || (nt % 2) != 0) {
        snprintf(err, errsz, "linear expression must have coefficient/variable pairs");
        return false;
    }

    terms = (LinTerm *)malloc((size_t)(nt / 2) * sizeof(LinTerm));
    if (terms == NULL) {
        snprintf(err, errsz, "out of memory storing linear terms");
        return false;
    }

    for (i = 0; i < nt; i += 2) {
        LinTerm t;
        if (!parse_double_str(tok[i], &t.coef)) {
            snprintf(err, errsz, "invalid coefficient '%s'", tok[i]);
            free(terms);
            return false;
        }
        if (!parse_var_ref_token(tok[i + 1], m->B, m->I, m->R, &t.kind, &t.idx)) {
            snprintf(err, errsz, "invalid variable reference '%s'", tok[i + 1]);
            free(terms);
            return false;
        }
        terms[i / 2] = t;
    }

    *out_terms = terms;
    *out_n = nt / 2;
    return true;
}

static bool parse_lc_line(Model *m, char *line, char *err, size_t errsz) {
    char *colon = strchr(line, ':');
    char *left;
    char *right;
    char *tok[64];
    int nt;
    char cid[64];
    char sense[8];
    double rhs;
    LinTerm *terms = NULL;
    int nterms = 0;

    if (colon == NULL) {
        snprintf(err, errsz, "linear constraint missing ':'");
        return false;
    }
    *colon = '\0';
    left = line;
    right = colon + 1;
    trim_inplace(left);
    trim_inplace(right);

    nt = token_split(left, tok, 64);
    if (nt < 4 || !str_ieq(tok[0], "lc")) {
        snprintf(err, errsz, "invalid linear constraint header");
        return false;
    }

    snprintf(cid, sizeof(cid), "%s", tok[1]);
    snprintf(sense, sizeof(sense), "%s", tok[2]);
    if (!parse_double_str(tok[3], &rhs)) {
        snprintf(err, errsz, "invalid linear rhs '%s'", tok[3]);
        return false;
    }

    if (!parse_lc_terms(m, right, &terms, &nterms, err, errsz)) {
        return false;
    }

    if (!add_lin_constraint_normalized(m, cid, terms, nterms, sense, rhs, err, errsz)) {
        free(terms);
        return false;
    }

    free(terms);
    return true;
}

static bool parse_indicator_line(Model *m, char *line, char *err, size_t errsz) {
    char *tok[16];
    int nt;
    PendingIndicator p;

    nt = token_split(line, tok, 16);
    if (nt != 4 || !str_ieq(tok[0], "ind") || strcmp(tok[2], "=>") != 0) {
        snprintf(err, errsz, "invalid indicator line");
        return false;
    }
    if (!parse_bool_lit_token(tok[1], m->B, &p.lit)) {
        snprintf(err, errsz, "invalid indicator literal '%s'", tok[1]);
        return false;
    }
    snprintf(p.name, sizeof(p.name), "%s", tok[3]);

    if (!pendvec_push(&m->pending_ind, &p)) {
        snprintf(err, errsz, "out of memory storing indicator");
        return false;
    }
    return true;
}

static bool parse_obj_terms(char *expr, Model *m, char *err, size_t errsz) {
    char *tok[2048];
    int nt;
    int i;

    trim_inplace(expr);
    nt = token_split(expr, tok, 2048);
    i = 0;
    while (i < nt) {
        if (str_ieq(tok[i], "lin")) {
            ++i;
            continue;
        }
        if (str_ieq(tok[i], "pen")) {
            break;
        }
        if (i + 1 >= nt) {
            snprintf(err, errsz, "incomplete objective term");
            return false;
        }
        {
            double coef;
            VarKind kind;
            int idx;
            if (!parse_double_str(tok[i], &coef)) {
                snprintf(err, errsz, "invalid objective coefficient '%s'", tok[i]);
                return false;
            }
            if (!parse_var_ref_token(tok[i + 1], m->B, m->I, m->R, &kind, &idx)) {
                snprintf(err, errsz, "invalid objective variable '%s'", tok[i + 1]);
                return false;
            }
            if (kind == VAR_BOOL) m->obj_b_input[idx] += coef;
            else if (kind == VAR_INT) m->obj_i[idx] += coef;
            else m->obj_r[idx] += coef;
            i += 2;
        }
    }

    return true;
}

static bool parse_obj_line(Model *m, char *line, char *err, size_t errsz) {
    char *colon = strchr(line, ':');
    char *head;
    char *expr;
    char *tok[8];
    int nt;

    if (colon == NULL) {
        snprintf(err, errsz, "objective line missing ':'");
        return false;
    }
    *colon = '\0';
    head = line;
    expr = colon + 1;
    trim_inplace(head);
    trim_inplace(expr);

    nt = token_split(head, tok, 8);
    if (nt != 2 || !str_ieq(tok[0], "obj")) {
        snprintf(err, errsz, "invalid objective header");
        return false;
    }

    if (str_ieq(tok[1], "min")) {
        m->obj_is_max = false;
    } else if (str_ieq(tok[1], "max")) {
        m->obj_is_max = true;
    } else {
        snprintf(err, errsz, "invalid objective direction '%s'", tok[1]);
        return false;
    }

    if (!parse_obj_terms(expr, m, err, errsz)) {
        return false;
    }
    m->has_obj_line = true;
    return true;
}

static void parse_opt_line(Model *m, char *line) {
    char *tok[8];
    int nt;
    double x;
    long long li;
    uint64_t su;

    nt = token_split(line, tok, 8);
    if (nt != 3 || !str_ieq(tok[0], "opt")) {
        return;
    }

    if (str_ieq(tok[1], "feas_tol") && parse_double_str(tok[2], &x)) {
        m->fopt.has_feas_tol = true;
        m->fopt.feas_tol = x;
    } else if (str_ieq(tok[1], "int_tol") && parse_double_str(tok[2], &x)) {
        m->fopt.has_int_tol = true;
        m->fopt.int_tol = x;
    } else if (str_ieq(tok[1], "time_limit") && parse_double_str(tok[2], &x)) {
        m->fopt.has_time_limit = true;
        m->fopt.time_limit = x;
    } else if (str_ieq(tok[1], "node_limit") && parse_ll(tok[2], &li)) {
        m->fopt.has_node_limit = true;
        m->fopt.node_limit = li;
    } else if (str_ieq(tok[1], "gap") && parse_double_str(tok[2], &x)) {
        m->fopt.has_gap = true;
        m->fopt.rel_gap = x;
    } else if (str_ieq(tok[1], "seed") && parse_u64(tok[2], &su)) {
        m->fopt.has_seed = true;
        m->fopt.seed = su;
    } else if (str_ieq(tok[1], "verbose") && parse_ll(tok[2], &li)) {
        m->fopt.has_verbose = true;
        m->fopt.verbose = (int)li;
    }
}

static bool parse_wmibo_file(const char *path, Model *m, double deadline, char *err, size_t errsz) {
    FILE *fp;
    char line_buf[WMIBO_MAX_LINE];
    int lineno = 0;
    bool have_header = false;
    enum { BLK_NONE, BLK_CNF, BLK_WCNF, BLK_LIN, BLK_IND, BLK_OBJ } blk = BLK_NONE;

    fp = fopen(path, "rb");
    if (fp == NULL) {
        snprintf(err, errsz, "cannot open file '%s'", path);
        return false;
    }

    while (fgets(line_buf, (int)sizeof(line_buf), fp) != NULL) {
        char line_copy[WMIBO_MAX_LINE];
        char line_raw[WMIBO_MAX_LINE];
        char *line = line_copy;
        char *tok[32];
        int nt;

        ++lineno;
        if ((lineno & 1023) == 0 && deadline_reached(deadline)) {
            snprintf(err, errsz, "time limit exceeded during parsing");
            fclose(fp);
            return false;
        }
        snprintf(line_copy, sizeof(line_copy), "%s", line_buf);
        trim_inplace(line);
        if (line[0] == '\0' || line[0] == '#' ||
            ((line[0] == 'c' || line[0] == 'C') &&
             (line[1] == '\0' || isspace((unsigned char)line[1])))) {
            continue;
        }
        strip_inline_comment(line);
        trim_inplace(line);
        if (line[0] == '\0') {
            continue;
        }
        snprintf(line_raw, sizeof(line_raw), "%s", line);

        nt = token_split(line, tok, 32);
        if (nt <= 0) {
            continue;
        }

        if (str_ieq(tok[0], "p")) {
            long long B, I, R;
            if (nt < 5 || !str_ieq(tok[1], "wmibo")) {
                snprintf(err, errsz, "%s:%d invalid wmibo header", path, lineno);
                fclose(fp);
                return false;
            }
            if (!parse_ll(tok[2], &B) || !parse_ll(tok[3], &I) || !parse_ll(tok[4], &R)) {
                snprintf(err, errsz, "%s:%d invalid wmibo dimensions", path, lineno);
                fclose(fp);
                return false;
            }
            if (!ensure_model_dims(m, (int)B, (int)I, (int)R, err, errsz)) {
                fclose(fp);
                return false;
            }
            have_header = true;
            continue;
        }

        if (!have_header) {
            snprintf(err, errsz, "%s:%d data before header", path, lineno);
            fclose(fp);
            return false;
        }

        if (str_ieq(tok[0], "begin")) {
            if (nt != 2) {
                snprintf(err, errsz, "%s:%d invalid begin line", path, lineno);
                fclose(fp);
                return false;
            }
            if (str_ieq(tok[1], "cnf")) blk = BLK_CNF;
            else if (str_ieq(tok[1], "wcnf")) blk = BLK_WCNF;
            else if (str_ieq(tok[1], "lin")) blk = BLK_LIN;
            else if (str_ieq(tok[1], "ind")) blk = BLK_IND;
            else if (str_ieq(tok[1], "obj")) blk = BLK_OBJ;
            else {
                snprintf(err, errsz, "%s:%d unknown block '%s'", path, lineno, tok[1]);
                fclose(fp);
                return false;
            }
            continue;
        }

        if (str_ieq(tok[0], "end")) {
            blk = BLK_NONE;
            continue;
        }

        if (str_ieq(tok[0], "var")) {
            if (!parse_var_decl(m, line_raw, err, errsz)) {
                char msg[256];
                snprintf(msg, sizeof(msg), "%s:%d %s", path, lineno, err);
                snprintf(err, errsz, "%s", msg);
                fclose(fp);
                return false;
            }
            continue;
        }

        if (str_ieq(tok[0], "opt")) {
            parse_opt_line(m, line_raw);
            continue;
        }

        if (str_ieq(tok[0], "cl") || blk == BLK_CNF) {
            if (!parse_cnf_clause_line(m, line_raw, false, err, errsz)) {
                char msg[256];
                snprintf(msg, sizeof(msg), "%s:%d %s", path, lineno, err);
                snprintf(err, errsz, "%s", msg);
                fclose(fp);
                return false;
            }
            continue;
        }

        if (str_ieq(tok[0], "wcl") || blk == BLK_WCNF) {
            if (!parse_cnf_clause_line(m, line_raw, true, err, errsz)) {
                char msg[256];
                snprintf(msg, sizeof(msg), "%s:%d %s", path, lineno, err);
                snprintf(err, errsz, "%s", msg);
                fclose(fp);
                return false;
            }
            continue;
        }

        if (str_ieq(tok[0], "lc") || blk == BLK_LIN) {
            if (!parse_lc_line(m, line_raw, err, errsz)) {
                char msg[256];
                snprintf(msg, sizeof(msg), "%s:%d %s", path, lineno, err);
                snprintf(err, errsz, "%s", msg);
                fclose(fp);
                return false;
            }
            continue;
        }
        if (str_ieq(tok[0], "ind") || blk == BLK_IND) {
            if (!parse_indicator_line(m, line_raw, err, errsz)) {
                char msg[256];
                snprintf(msg, sizeof(msg), "%s:%d %s", path, lineno, err);
                snprintf(err, errsz, "%s", msg);
                fclose(fp);
                return false;
            }
            continue;
        }

        if (str_ieq(tok[0], "obj") || blk == BLK_OBJ) {
            if (!parse_obj_line(m, line_raw, err, errsz)) {
                char msg[256];
                snprintf(msg, sizeof(msg), "%s:%d %s", path, lineno, err);
                snprintf(err, errsz, "%s", msg);
                fclose(fp);
                return false;
            }
            continue;
        }

        snprintf(err, errsz, "%s:%d unrecognized directive '%s'", path, lineno, tok[0]);
        fclose(fp);
        return false;
    }

    fclose(fp);
    if (!have_header) {
        snprintf(err, errsz, "wmibo header not found");
        return false;
    }
    return true;
}

static bool parse_dimacs_file(const char *path, Model *m, bool force_wcnf, double deadline, char *err, size_t errsz) {
    FILE *fp;
    char line[WMIBO_MAX_LINE];
    bool have_header = false;
    bool is_wcnf = force_wcnf;
    long long nvars = 0;
    long long nclauses_decl = 0;
    double top_weight = -1.0;
    int lineno = 0;

    fp = fopen(path, "rb");
    if (fp == NULL) {
        snprintf(err, errsz, "cannot open file '%s'", path);
        return false;
    }

    while (fgets(line, (int)sizeof(line), fp) != NULL) {
        char *p = line;
        ++lineno;
        if ((lineno & 1023) == 0 && deadline_reached(deadline)) {
            snprintf(err, errsz, "time limit exceeded during parsing");
            fclose(fp);
            return false;
        }
        trim_inplace(p);
        if (p[0] == '\0' || p[0] == 'c' || p[0] == 'C' || p[0] == '#') {
            continue;
        }

        if (p[0] == 'p') {
            char *tok[8];
            int nt = token_split(p, tok, 8);
            if (nt < 4) {
                snprintf(err, errsz, "invalid DIMACS header");
                fclose(fp);
                return false;
            }
            if (str_ieq(tok[1], "cnf")) {
                is_wcnf = false;
            } else if (str_ieq(tok[1], "wcnf")) {
                is_wcnf = true;
            } else {
                snprintf(err, errsz, "unsupported DIMACS format '%s'", tok[1]);
                fclose(fp);
                return false;
            }
            if (!parse_ll(tok[2], &nvars) || !parse_ll(tok[3], &nclauses_decl)) {
                snprintf(err, errsz, "invalid DIMACS header counts");
                fclose(fp);
                return false;
            }
            if (is_wcnf && nt >= 5) {
                if (!parse_double_str(tok[4], &top_weight)) {
                    snprintf(err, errsz, "invalid WCNF top weight");
                    fclose(fp);
                    return false;
                }
            }
            if (!ensure_model_dims(m, (int)nvars, 0, 0, err, errsz)) {
                fclose(fp);
                return false;
            }
            have_header = true;
            continue;
        }

        if (!have_header) {
            snprintf(err, errsz, "DIMACS clause before header");
            fclose(fp);
            return false;
        }

        {
            char *tok[8192];
            int nt;
            int pos = 0;
            double w = 1.0;
            bool hard = true;
            int lits[4096];
            int nl = 0;
            int i;

            nt = token_split(p, tok, 8192);
            if (nt == 0) {
                continue;
            }

            if (is_wcnf) {
                if (!parse_double_str(tok[0], &w) || w < 0.0) {
                    snprintf(err, errsz, "invalid WCNF weight '%s'", tok[0]);
                    fclose(fp);
                    return false;
                }
                if (top_weight > 0.0 && w >= top_weight) {
                    hard = true;
                } else {
                    hard = false;
                }
                pos = 1;
            }

            for (i = pos; i < nt; ++i) {
                long long litll;
                int lit;
                if (!parse_ll(tok[i], &litll)) {
                    snprintf(err, errsz, "invalid literal '%s'", tok[i]);
                    fclose(fp);
                    return false;
                }
                if (litll == 0) {
                    break;
                }
                if (litll < -nvars || litll > nvars) {
                    snprintf(err, errsz, "literal out of range: %lld", litll);
                    fclose(fp);
                    return false;
                }
                if (nl >= (int)(sizeof(lits) / sizeof(lits[0]))) {
                    snprintf(err, errsz, "DIMACS clause too long");
                    fclose(fp);
                    return false;
                }
                lit = (int)litll;
                lits[nl++] = lit;
            }

            if (i == nt || strcmp(tok[i], "0") != 0) {
                snprintf(err, errsz, "DIMACS clause missing trailing 0");
                fclose(fp);
                return false;
            }

            if (hard) {
                if (!add_clause_hard(m, lits, nl)) {
                    snprintf(err, errsz, "out of memory storing hard DIMACS clause");
                    fclose(fp);
                    return false;
                }
            } else {
                if (!add_soft_clause(m, lits, nl, w)) {
                    snprintf(err, errsz, "out of memory storing soft DIMACS clause");
                    fclose(fp);
                    return false;
                }
            }
        }
    }

    fclose(fp);
    if (!have_header) {
        snprintf(err, errsz, "DIMACS header not found");
        return false;
    }
    (void)nclauses_decl;
    return true;
}

static bool apply_pending_indicators(Model *m, double deadline, char *err, size_t errsz) {
    int i;
    int j;
    for (i = 0; i < m->pending_ind.size; ++i) {
        const PendingIndicator *p = &m->pending_ind.data[i];
        bool found = false;
        if ((i & 63) == 0 && deadline_reached(deadline)) {
            snprintf(err, errsz, "time limit exceeded during model finalization");
            return false;
        }
        for (j = 0; j < m->lin.size; ++j) {
            LinConstraint *c = &m->lin.data[j];
            if (strcmp(c->name, p->name) == 0) {
                if (c->indicator_lit != 0 && c->indicator_lit != p->lit) {
                    snprintf(err, errsz, "conflicting indicators for constraint '%s'", p->name);
                    return false;
                }
                c->indicator_lit = p->lit;
                found = true;
            }
        }
        if (!found) {
            snprintf(err, errsz, "indicator references unknown constraint id '%s'", p->name);
            return false;
        }
    }
    return true;
}

static bool finalize_model(Model *m, double deadline, char *err, size_t errsz) {
    int s;
    int n;
    int i;
    double sign = m->obj_is_max ? -1.0 : 1.0;

    if (!apply_pending_indicators(m, deadline, err, errsz)) {
        return false;
    }

    m->nb_total = m->B + m->soft.size;
    m->nvars_total = m->nb_total + m->I + m->R;
    n = m->nvars_total;

    m->obj_all = (double *)calloc((size_t)n, sizeof(double));
    m->var_lb_all = (double *)calloc((size_t)n, sizeof(double));
    m->var_ub_all = (double *)calloc((size_t)n, sizeof(double));
    if ((n > 0) && (m->obj_all == NULL || m->var_lb_all == NULL || m->var_ub_all == NULL)) {
        snprintf(err, errsz, "out of memory finalizing model arrays");
        return false;
    }

    for (i = 1; i <= m->B; ++i) {
        if ((i & 1023) == 0 && deadline_reached(deadline)) {
            snprintf(err, errsz, "time limit exceeded during model finalization");
            return false;
        }
        m->var_lb_all[i - 1] = m->b_lb_input[i];
        m->var_ub_all[i - 1] = m->b_ub_input[i];
        m->obj_all[i - 1] += sign * m->obj_b_input[i];
    }
    for (s = 0; s < m->soft.size; ++s) {
        int rv = m->B + s + 1;
        int idx = rv - 1;
        SoftClause *sc = &m->soft.data[s];
        int *lits;
        int k;

        if ((s & 255) == 0 && deadline_reached(deadline)) {
            snprintf(err, errsz, "time limit exceeded during model finalization");
            return false;
        }
        m->var_lb_all[idx] = 0.0;
        m->var_ub_all[idx] = 1.0;
        m->obj_all[idx] += sign * sc->weight;

        lits = (int *)malloc((size_t)(sc->size + 1) * sizeof(int));
        if (lits == NULL) {
            snprintf(err, errsz, "out of memory reifying soft clause");
            return false;
        }
        for (k = 0; k < sc->size; ++k) {
            lits[k] = sc->lits[k];
        }
        lits[sc->size] = rv;
        if (!add_clause_hard(m, lits, sc->size + 1)) {
            free(lits);
            snprintf(err, errsz, "out of memory adding reified soft clause");
            return false;
        }
        free(lits);
    }

    for (i = 1; i <= m->I; ++i) {
        int gi = m->nb_total + (i - 1);
        if ((i & 1023) == 0 && deadline_reached(deadline)) {
            snprintf(err, errsz, "time limit exceeded during model finalization");
            return false;
        }
        m->var_lb_all[gi] = m->i_lb[i];
        m->var_ub_all[gi] = m->i_ub[i];
        m->obj_all[gi] += sign * m->obj_i[i];
    }

    for (i = 1; i <= m->R; ++i) {
        int gi = m->nb_total + m->I + (i - 1);
        if ((i & 1023) == 0 && deadline_reached(deadline)) {
            snprintf(err, errsz, "time limit exceeded during model finalization");
            return false;
        }
        m->var_lb_all[gi] = m->r_lb[i];
        m->var_ub_all[gi] = m->r_ub[i];
        m->obj_all[gi] += sign * m->obj_r[i];
    }

    for (i = 0; i < n; ++i) {
        if ((i & 2047) == 0 && deadline_reached(deadline)) {
            snprintf(err, errsz, "time limit exceeded during model finalization");
            return false;
        }
        if (m->var_lb_all[i] > m->var_ub_all[i] + WMIBO_FEAS_TOL_DEFAULT) {
            snprintf(err, errsz, "inconsistent variable bounds at variable %d", i + 1);
            return false;
        }
    }

    return true;
}

static bool load_model_from_file(const char *path, Model *m, double deadline, char *err, size_t errsz) {
    bool ok;
    model_init(m);

    if (has_ext_ci(path, ".cnf")) {
        ok = parse_dimacs_file(path, m, false, deadline, err, errsz);
    } else if (has_ext_ci(path, ".wcnf")) {
        ok = parse_dimacs_file(path, m, true, deadline, err, errsz);
    } else {
        ok = parse_wmibo_file(path, m, deadline, err, errsz);
    }

    if (!ok) {
        model_free(m);
        return false;
    }

    if (!finalize_model(m, deadline, err, errsz)) {
        model_free(m);
        return false;
    }

    return true;
}

static int map_var_global_index(const Model *m, VarKind kind, int idx) {
    if (kind == VAR_BOOL) {
        return idx - 1;
    }
    if (kind == VAR_INT) {
        return m->nb_total + (idx - 1);
    }
    return m->nb_total + m->I + (idx - 1);
}

static void simplex_init(Simplex *s) {
    memset(s, 0, sizeof(*s));
}

static void simplex_rebind_rows(Simplex *s) {
    int i;
    int rows = s->m + 2;
    int cols = s->n + 2;
    for (i = 0; i < rows; ++i) {
        s->D[i] = s->buf + (size_t)i * (size_t)cols;
    }
}

static void simplex_free(Simplex *s) {
    if (s->use_cuda) {
        krb_accel_cuda_managed_free(s->B);
        krb_accel_cuda_managed_free(s->N);
        krb_accel_cuda_managed_free(s->buf);
        krb_accel_cuda_managed_free(s->buf_alt);
        krb_accel_cuda_managed_free(s->gpu_candidate_idx);
        krb_accel_cuda_managed_free(s->gpu_candidate_metric);
    } else {
        free(s->B);
        free(s->N);
        free(s->buf);
        free(s->buf_alt);
        free(s->gpu_candidate_idx);
        free(s->gpu_candidate_metric);
    }
    free(s->D);
    s->B = NULL;
    s->N = NULL;
    s->buf = NULL;
    s->buf_alt = NULL;
    s->gpu_candidate_idx = NULL;
    s->gpu_candidate_metric = NULL;
    s->gpu_candidate_cap = 0;
    s->D = NULL;
    s->m = 0;
    s->n = 0;
    s->use_cuda = false;
    s->gpu_error = false;
}

static bool simplex_time_exceeded(const Simplex *s) {
    return s->deadline > 0.0 && now_seconds() >= s->deadline;
}

static bool simplex_build(Simplex *s, int m, int n, const double *A, const double *b, const double *c) {
    int i;
    int j;
    int rows = m + 2;
    int cols = n + 2;
    int block_cols = (n + 1 + 255) / 256;
    int block_rows = (m + 255) / 256;
    int candidate_cap = (block_cols > block_rows) ? block_cols : block_rows;
    char cuda_err[256];
    double deadline = s->deadline;
    bool use_cuda = s->use_cuda;

    simplex_init(s);
    s->deadline = deadline;
    s->use_cuda = use_cuda;
    s->m = m;
    s->n = n;

    s->D = (double **)malloc((size_t)rows * sizeof(double *));
    s->buf_alt = NULL;
    s->gpu_candidate_idx = NULL;
    s->gpu_candidate_metric = NULL;
    s->gpu_candidate_cap = 0;
    cuda_err[0] = '\0';
    if (s->use_cuda) {
        if (!krb_accel_cuda_managed_alloc_ints((size_t)((m > 0) ? m : 1), &s->B, cuda_err, sizeof(cuda_err)) ||
            !krb_accel_cuda_managed_alloc_ints((size_t)(n + 1), &s->N, cuda_err, sizeof(cuda_err)) ||
            !krb_accel_cuda_managed_alloc_ints((size_t)((candidate_cap > 0) ? candidate_cap : 1), &s->gpu_candidate_idx, cuda_err, sizeof(cuda_err)) ||
            !krb_accel_cuda_managed_alloc_doubles((size_t)((candidate_cap > 0) ? candidate_cap : 1), &s->gpu_candidate_metric, cuda_err, sizeof(cuda_err)) ||
            !krb_accel_cuda_managed_alloc_doubles((size_t)rows * (size_t)cols, &s->buf, cuda_err, sizeof(cuda_err)) ||
            !krb_accel_cuda_managed_alloc_doubles((size_t)rows * (size_t)cols, &s->buf_alt, cuda_err, sizeof(cuda_err))) {
            fprintf(stderr, "c wmibo accel error: %s\n",
                    (cuda_err[0] != '\0') ? cuda_err : "failed to allocate CUDA simplex buffers");
            s->gpu_error = true;
            simplex_free(s);
            return false;
        }
        s->gpu_candidate_cap = (candidate_cap > 0) ? candidate_cap : 1;
        memset(s->gpu_candidate_idx, 0, (size_t)s->gpu_candidate_cap * sizeof(int));
        memset(s->gpu_candidate_metric, 0, (size_t)s->gpu_candidate_cap * sizeof(double));
        memset(s->buf, 0, (size_t)rows * (size_t)cols * sizeof(double));
        memset(s->buf_alt, 0, (size_t)rows * (size_t)cols * sizeof(double));
    } else {
        s->B = (int *)malloc((size_t)((m > 0) ? m : 1) * sizeof(int));
        s->N = (int *)malloc((size_t)(n + 1) * sizeof(int));
        s->gpu_candidate_cap = (candidate_cap > 0) ? candidate_cap : 1;
        s->gpu_candidate_idx = (int *)calloc((size_t)s->gpu_candidate_cap, sizeof(int));
        s->gpu_candidate_metric = (double *)calloc((size_t)s->gpu_candidate_cap, sizeof(double));
        s->buf = (double *)calloc((size_t)rows * (size_t)cols, sizeof(double));
        s->buf_alt = (double *)calloc((size_t)rows * (size_t)cols, sizeof(double));
    }

    if ((m > 0 && s->B == NULL) || s->N == NULL || s->D == NULL ||
        s->buf == NULL || s->buf_alt == NULL ||
        s->gpu_candidate_idx == NULL || s->gpu_candidate_metric == NULL) {
        simplex_free(s);
        return false;
    }

    simplex_rebind_rows(s);

    for (i = 0; i < m; ++i) {
        if ((i & 31) == 0 && simplex_time_exceeded(s)) {
            s->aborted = true;
            simplex_free(s);
            return false;
        }
        for (j = 0; j < n; ++j) {
            s->D[i][j] = A[(size_t)i * (size_t)n + (size_t)j];
        }
    }

    for (i = 0; i < m; ++i) {
        if ((i & 31) == 0 && simplex_time_exceeded(s)) {
            s->aborted = true;
            simplex_free(s);
            return false;
        }
        s->B[i] = n + i;
        s->D[i][n] = -1.0;
        s->D[i][n + 1] = b[i];
    }

    for (j = 0; j < n; ++j) {
        if ((j & 255) == 0 && simplex_time_exceeded(s)) {
            s->aborted = true;
            simplex_free(s);
            return false;
        }
        s->N[j] = j;
        s->D[m][j] = -c[j];
    }
    s->N[n] = -1;
    s->D[m + 1][n] = 1.0;

    return true;
}

static void simplex_pivot(Simplex *s, int r, int c) {
    int i;
    int j;
    int m = s->m;
    int n = s->n;
    if (s->use_cuda) {
        char cuda_err[256];
        double *tmp;
        cuda_err[0] = '\0';
        if (!krb_accel_cuda_tableau_pivot(s->buf_alt, s->buf, m + 2, n + 2, r, c, cuda_err, sizeof(cuda_err))) {
            fprintf(stderr, "c wmibo accel error: %s\n",
                    (cuda_err[0] != '\0') ? cuda_err : "CUDA pivot update failed");
            s->gpu_error = true;
            s->aborted = true;
            return;
        }
        tmp = s->buf;
        s->buf = s->buf_alt;
        s->buf_alt = tmp;
        simplex_rebind_rows(s);
        {
            int t = s->B[r];
            s->B[r] = s->N[c];
            s->N[c] = t;
        }
        return;
    }

    double piv = s->D[r][c];

    for (i = 0; i < m + 2; ++i) {
        if (i == r) continue;
        if ((i & 31) == 0 && simplex_time_exceeded(s)) {
            s->aborted = true;
            return;
        }
        for (j = 0; j < n + 2; ++j) {
            if (j == c) continue;
            s->D[i][j] -= s->D[r][j] * s->D[i][c] / piv;
        }
    }
    for (j = 0; j < n + 2; ++j) {
        if (j != c) s->D[r][j] /= piv;
    }
    for (i = 0; i < m + 2; ++i) {
        if (i != r) s->D[i][c] /= -piv;
    }
    s->D[r][c] = 1.0 / piv;

    {
        int t = s->B[r];
        s->B[r] = s->N[c];
        s->N[c] = t;
    }
}

static bool simplex_phase(Simplex *s, int phase) {
    int x = (phase == 1) ? s->m + 1 : s->m;
    int m = s->m;
    int n = s->n;
    long long iter = 0;

    for (;;) {
        int c = -1;
        int r = -1;
        int i;
        int j;

        if (((++iter) & 3LL) == 0LL && simplex_time_exceeded(s)) {
            s->aborted = true;
            return false;
        }

        if (s->use_cuda) {
            char cuda_err[256];
            cuda_err[0] = '\0';
            if (!krb_accel_cuda_find_entering_col(s->buf,
                                                  s->N,
                                                  x,
                                                  n + 2,
                                                  n + 1,
                                                  phase,
                                                  WMIBO_LP_EPS,
                                                  s->gpu_candidate_idx,
                                                  s->gpu_candidate_metric,
                                                  s->gpu_candidate_cap,
                                                  &c,
                                                  cuda_err,
                                                  sizeof(cuda_err))) {
                fprintf(stderr, "c wmibo accel error: %s\n",
                        (cuda_err[0] != '\0') ? cuda_err : "failed to price entering column on GPU");
                s->gpu_error = true;
                s->aborted = true;
                return false;
            }
        } else {
            for (j = 0; j <= n; ++j) {
                if (phase == 2 && s->N[j] == -1) {
                    continue;
                }
                if (c < 0 || s->D[x][j] < s->D[x][c] - WMIBO_LP_EPS ||
                    (fabs(s->D[x][j] - s->D[x][c]) <= WMIBO_LP_EPS && s->N[j] < s->N[c])) {
                    c = j;
                }
            }
        }

        if (c < 0 || s->D[x][c] > -WMIBO_LP_EPS) {
            return true;
        }

        if (s->use_cuda) {
            char cuda_err[256];
            cuda_err[0] = '\0';
            if (!krb_accel_cuda_find_leaving_row(s->buf,
                                                 s->B,
                                                 m,
                                                 n + 2,
                                                 c,
                                                 n + 1,
                                                 WMIBO_LP_EPS,
                                                 s->gpu_candidate_idx,
                                                 s->gpu_candidate_metric,
                                                 s->gpu_candidate_cap,
                                                 &r,
                                                 cuda_err,
                                                 sizeof(cuda_err))) {
                fprintf(stderr, "c wmibo accel error: %s\n",
                        (cuda_err[0] != '\0') ? cuda_err : "failed to select leaving row on GPU");
                s->gpu_error = true;
                s->aborted = true;
                return false;
            }
        } else {
            for (i = 0; i < m; ++i) {
                if (s->D[i][c] > WMIBO_LP_EPS) {
                    if (r < 0) {
                        r = i;
                    } else {
                        double lhs = s->D[i][n + 1] / s->D[i][c];
                        double rhs = s->D[r][n + 1] / s->D[r][c];
                        if (lhs < rhs - WMIBO_LP_EPS ||
                            (fabs(lhs - rhs) <= WMIBO_LP_EPS && s->B[i] < s->B[r])) {
                            r = i;
                        }
                    }
                }
            }
        }

        if (r < 0) {
            return false;
        }

        simplex_pivot(s, r, c);
        if (s->aborted) {
            return false;
        }
    }
}

static LPStatus simplex_solve_max(int m,
                                  int n,
                                  const double *A,
                                  const double *b,
                                  const double *c,
                                  double deadline,
                                  bool use_cuda,
                                  double *out_obj,
                                  double *x) {
    Simplex s;
    int i;
    simplex_init(&s);
    s.deadline = deadline;
    s.use_cuda = use_cuda;

    if (!simplex_build(&s, m, n, A, b, c)) {
        return LP_STATUS_ERROR;
    }
    if (m > 0) {
        int r = 0;
        for (i = 1; i < m; ++i) {
            if (s.D[i][n + 1] < s.D[r][n + 1]) {
                r = i;
            }
        }
        if (s.D[r][n + 1] < -WMIBO_LP_EPS) {
            simplex_pivot(&s, r, n);
            if (s.aborted) {
                simplex_free(&s);
                return LP_STATUS_ERROR;
            }
            if (!simplex_phase(&s, 1) || s.D[m + 1][n + 1] < -WMIBO_LP_EPS) {
                if (s.aborted) {
                    simplex_free(&s);
                    return LP_STATUS_ERROR;
                }
                simplex_free(&s);
                return LP_STATUS_INFEASIBLE;
            }
            if (fabs(s.D[m + 1][n + 1]) > WMIBO_LP_EPS) {
                simplex_free(&s);
                return LP_STATUS_INFEASIBLE;
            }
            for (i = 0; i < m; ++i) {
                if (s.B[i] == -1) {
                    int j;
                    int ccol = -1;
                    for (j = 0; j <= n; ++j) {
                        if (ccol < 0 || s.D[i][j] < s.D[i][ccol] - WMIBO_LP_EPS ||
                            (fabs(s.D[i][j] - s.D[i][ccol]) <= WMIBO_LP_EPS && s.N[j] < s.N[ccol])) {
                            ccol = j;
                        }
                    }
                    if (ccol >= 0) {
                        simplex_pivot(&s, i, ccol);
                        if (s.aborted) {
                            simplex_free(&s);
                            return LP_STATUS_ERROR;
                        }
                    }
                }
            }
        }
    }

    if (!simplex_phase(&s, 2)) {
        if (s.aborted) {
            simplex_free(&s);
            return LP_STATUS_ERROR;
        }
        simplex_free(&s);
        return LP_STATUS_UNBOUNDED;
    }

    for (i = 0; i < n; ++i) {
        x[i] = 0.0;
    }
    for (i = 0; i < m; ++i) {
        if (s.B[i] >= 0 && s.B[i] < n) {
            x[s.B[i]] = s.D[i][n + 1];
        }
    }
    *out_obj = s.D[m][n + 1];
    simplex_free(&s);
    return LP_STATUS_OPTIMAL;
}

static LPStatus wmibo_dense_lp_dispatch(const SolveOptions *opt,
                                        int m,
                                        int n,
                                        const double *A,
                                        const double *b,
                                        const double *c,
                                        double deadline,
                                        double *out_obj,
                                        double *x) {
    KrbAccelDecision decision;
    char err[256];
    int verbose = (opt != NULL) ? opt->verbose : 0;
    const KrbAccelConfig *accel = (opt != NULL) ? &opt->accel : NULL;

    err[0] = '\0';
    if (!krb_accel_choose_dense_lp(accel, "wmibo", m, n, &decision, err, sizeof(err))) {
        fprintf(stderr, "c wmibo accel error: %s\n",
                (err[0] != '\0') ? err : "invalid acceleration configuration");
        return LP_STATUS_ERROR;
    }
    if (verbose >= 3) {
        krb_accel_log(stderr, "wmibo", &decision);
    }
    return simplex_solve_max(m, n, A, b, c, deadline, decision.path == KRB_ACCEL_PATH_CUDA, out_obj, x);
}

static bool state_has_local_int_branch(const Solver *s, const NodeState *st) {
    int i;
    for (i = 1; i <= s->model.I; ++i) {
        if (st->i_lb[i] > s->model.i_lb[i] + s->opt.feas_tol) {
            return true;
        }
        if (st->i_ub[i] < s->model.i_ub[i] - s->opt.feas_tol) {
            return true;
        }
    }
    return false;
}

static bool build_lp_relaxation(const Solver *s,
                                const NodeState *st,
                                double **out_A,
                                double **out_b,
                                double **out_cmax,
                                double **out_lower,
                                int *out_m,
                                int *out_n,
                                double *out_const,
                                IntVec *active_indicator_lits,
                                char *err,
                                size_t errsz) {
    int n = s->model.nvars_total;
    int m_est = s->model.lin.size + s->model.clauses.size + n + 8;
    double *A = NULL;
    double *b = NULL;
    double *cmax = NULL;
    double *lower = NULL;
    double *upper = NULL;
    int row = 0;
    int i;
    int ci;

    intvec_init(active_indicator_lits);

    if (n > 0 && m_est > 0) {
        double dense_cells = (double)n * (double)m_est;
        if (dense_cells > WMIBO_MAX_DENSE_LP_CELLS) {
            snprintf(err, errsz, "LP relaxation too large for dense assembly");
            return false;
        }
    }

    A = (double *)calloc((size_t)m_est * (size_t)n, sizeof(double));
    b = (double *)calloc((size_t)m_est, sizeof(double));
    cmax = (double *)calloc((size_t)n, sizeof(double));
    lower = (double *)calloc((size_t)n, sizeof(double));
    upper = (double *)calloc((size_t)n, sizeof(double));

    if ((n > 0) && (A == NULL || b == NULL || cmax == NULL || lower == NULL || upper == NULL)) {
        snprintf(err, errsz, "out of memory allocating LP buffers");
        free(A); free(b); free(cmax); free(lower); free(upper);
        return false;
    }

    for (i = 0; i < n; ++i) {
        if ((i & 255) == 0 && solver_time_limit_reached(s)) {
            snprintf(err, errsz, "time limit exceeded during LP assembly");
            free(A); free(b); free(cmax); free(lower); free(upper);
            intvec_free(active_indicator_lits);
            return false;
        }
        lower[i] = s->model.var_lb_all[i];
        upper[i] = s->model.var_ub_all[i];
        if (lower[i] <= -WMIBO_INF / 2.0) lower[i] = -WMIBO_BIG_BOUND;
        if (upper[i] >= WMIBO_INF / 2.0) upper[i] = WMIBO_BIG_BOUND;
    }

    for (i = 1; i <= s->model.nb_total; ++i) {
        int gi = i - 1;
        if (st->b_assign[i] >= 0) {
            lower[gi] = (double)st->b_assign[i];
            upper[gi] = (double)st->b_assign[i];
        } else {
            if (lower[gi] < 0.0) lower[gi] = 0.0;
            if (upper[gi] > 1.0) upper[gi] = 1.0;
        }
    }

    for (i = 1; i <= s->model.I; ++i) {
        int gi = s->model.nb_total + (i - 1);
        if (st->i_lb[i] > lower[gi]) lower[gi] = st->i_lb[i];
        if (st->i_ub[i] < upper[gi]) upper[gi] = st->i_ub[i];
    }

    for (i = 0; i < n; ++i) {
        if (lower[i] > upper[i] + s->opt.feas_tol) {
            *out_A = A;
            *out_b = b;
            *out_cmax = cmax;
            *out_lower = lower;
            *out_m = -1;
            *out_n = n;
            *out_const = 0.0;
            free(upper);
            return true;
        }
    }

    for (ci = 0; ci < s->model.lin.size; ++ci) {
        const LinConstraint *lc = &s->model.lin.data[ci];
        bool active = true;
        double rhs;
        double *rowp;
        double shift = 0.0;
        int t;

        if ((ci & 31) == 0 && solver_time_limit_reached(s)) {
            snprintf(err, errsz, "time limit exceeded during LP assembly");
            free(A); free(b); free(cmax); free(lower); free(upper);
            intvec_free(active_indicator_lits);
            return false;
        }

        if (lc->indicator_lit != 0) {
            int lit = lc->indicator_lit;
            int v = lit_var(lit);
            int ev = (v >= 1 && v <= s->model.nb_total) ? eval_lit(st->b_assign[v], lit) : 0;
            if (ev == 1) {
                active = true;
                if (!intvec_push(active_indicator_lits, lit)) {
                    snprintf(err, errsz, "out of memory storing active indicators");
                    free(A); free(b); free(cmax); free(lower); free(upper);
                    intvec_free(active_indicator_lits);
                    return false;
                }
            } else {
                active = false;
            }
        }

        if (!active) continue;

        rowp = A + (size_t)row * (size_t)n;
        rhs = lc->rhs;
        for (t = 0; t < lc->nterms; ++t) {
            int gi = map_var_global_index(&s->model, lc->terms[t].kind, lc->terms[t].idx);
            if (gi < 0 || gi >= n) {
                snprintf(err, errsz, "internal variable mapping error");
                free(A); free(b); free(cmax); free(lower); free(upper);
                intvec_free(active_indicator_lits);
                return false;
            }
            rowp[gi] += lc->terms[t].coef;
            shift += lc->terms[t].coef * lower[gi];
        }
        b[row++] = rhs - shift;
    }

    for (ci = 0; ci < s->model.clauses.size; ++ci) {
        const Clause *cl = &s->model.clauses.data[ci];
        int neg_count = 0;
        double rhs;
        double *rowp = A + (size_t)row * (size_t)n;
        double shift = 0.0;
        int j;

        if ((ci & 31) == 0 && solver_time_limit_reached(s)) {
            snprintf(err, errsz, "time limit exceeded during LP assembly");
            free(A); free(b); free(cmax); free(lower); free(upper);
            intvec_free(active_indicator_lits);
            return false;
        }
        for (j = 0; j < cl->size; ++j) {
            int lit = cl->lits[j];
            int gv = lit_var(lit) - 1;
            double coef = (lit > 0) ? -1.0 : 1.0;
            rowp[gv] += coef;
            shift += coef * lower[gv];
            if (lit < 0) neg_count++;
        }
        rhs = (double)neg_count - 1.0;
        b[row++] = rhs - shift;
    }

    for (i = 0; i < n; ++i) {
        if ((i & 255) == 0 && solver_time_limit_reached(s)) {
            snprintf(err, errsz, "time limit exceeded during LP assembly");
            free(A); free(b); free(cmax); free(lower); free(upper);
            intvec_free(active_indicator_lits);
            return false;
        }
        double *rowp = A + (size_t)row * (size_t)n;
        rowp[i] = 1.0;
        b[row++] = upper[i] - lower[i];
    }

    *out_const = s->model.obj_const;
    for (i = 0; i < n; ++i) {
        cmax[i] = -s->model.obj_all[i];
        *out_const += s->model.obj_all[i] * lower[i];
    }

    *out_A = A;
    *out_b = b;
    *out_cmax = cmax;
    *out_lower = lower;
    *out_m = row;
    *out_n = n;
    free(upper);
    return true;
}

static LPResult solve_lp_relaxation(const Solver *s, const NodeState *st, IntVec *active_indicator_lits) {
    LPResult res;
    double *A = NULL;
    double *b = NULL;
    double *cmax = NULL;
    double *lower = NULL;
    double *xprime = NULL;
    int m = 0;
    int n = 0;
    double const_part = 0.0;
    char err[256];

    res.status = LP_STATUS_ERROR;
    res.obj = NAN;
    res.x = NULL;

    if (!build_lp_relaxation(s, st, &A, &b, &cmax, &lower, &m, &n, &const_part, active_indicator_lits, err, sizeof(err))) {
        return res;
    }

    if (m < 0) {
        free(A); free(b); free(cmax); free(lower);
        res.status = LP_STATUS_INFEASIBLE;
        return res;
    }

    xprime = (double *)calloc((size_t)n, sizeof(double));
    res.x = (double *)calloc((size_t)n, sizeof(double));
    if ((n > 0) && (xprime == NULL || res.x == NULL)) {
        free(A); free(b); free(cmax); free(lower); free(xprime); free(res.x);
        res.x = NULL;
        res.status = LP_STATUS_ERROR;
        return res;
    }

    {
        double maxobj = 0.0;
        double deadline = (s->opt.time_limit > 0.0) ? (s->start_time + s->opt.time_limit) : 0.0;
        LPStatus st_lp = wmibo_dense_lp_dispatch(&s->opt, m, n, A, b, cmax, deadline, &maxobj, xprime);
        res.status = st_lp;
        if (st_lp == LP_STATUS_OPTIMAL) {
            int i;
            res.obj = const_part - maxobj;
            for (i = 0; i < n; ++i) {
                res.x[i] = xprime[i] + lower[i];
            }
        }
    }

    free(A); free(b); free(cmax); free(lower); free(xprime);
    return res;
}
static int propagate_clauses(Solver *s, const Model *m, int8_t *assign) {
    bool changed = true;
    int ci;
    while (changed) {
        changed = false;
        if (solver_poll_time_limit(s)) {
            return -2;
        }
        for (ci = 0; ci < m->clauses.size; ++ci) {
            const Clause *c = &m->clauses.data[ci];
            int un_lit = 0;
            int un_count = 0;
            bool sat = false;
            int j;
            if ((ci & 255) == 0 && solver_poll_time_limit(s)) {
                return -2;
            }
            for (j = 0; j < c->size; ++j) {
                int lit = c->lits[j];
                int v = lit_var(lit);
                int ev = eval_lit(assign[v], lit);
                if (ev == 1) {
                    sat = true;
                    break;
                }
                if (ev < 0) {
                    un_lit = lit;
                    ++un_count;
                }
            }
            if (sat) {
                continue;
            }
            if (un_count == 0) {
                return ci;
            }
            if (un_count == 1) {
                int v = lit_var(un_lit);
                int want = lit_sign(un_lit);
                if (assign[v] < 0) {
                    assign[v] = (int8_t)want;
                    changed = true;
                } else if (assign[v] != want) {
                    return ci;
                }
            }
        }
    }
    return -1;
}

static bool clause_is_satisfied(const Clause *c, const int8_t *assign) {
    int j;
    for (j = 0; j < c->size; ++j) {
        int lit = c->lits[j];
        int v = lit_var(lit);
        int ev = eval_lit(assign[v], lit);
        if (ev == 1) {
            return true;
        }
    }
    return false;
}

static bool all_bool_assigned(const Solver *s, const NodeState *st) {
    int v;
    for (v = 1; v <= s->model.nb_total; ++v) {
        if (st->b_assign[v] < 0) {
            return false;
        }
    }
    return true;
}

static bool all_hard_clauses_satisfied(Solver *s, const NodeState *st) {
    int ci;
    for (ci = 0; ci < s->model.clauses.size; ++ci) {
        if ((ci & 255) == 0 && solver_poll_time_limit(s)) {
            return false;
        }
        if (!clause_is_satisfied(&s->model.clauses.data[ci], st->b_assign)) {
            return false;
        }
    }
    return true;
}

static bool candidate_lit_true(const double *x, int lit) {
    int v = lit_var(lit) - 1;
    int val = (x[v] >= 0.5) ? 1 : 0;
    return lit_sign(lit) ? (val == 1) : (val == 0);
}

static bool solver_candidate_feasible(Solver *s,
                                      const NodeState *st,
                                      const double *cand_x,
                                      bool require_integral,
                                      double *obj_out) {
    int i;
    double tol = fmax(s->opt.feas_tol, 1e-8);

    if (cand_x == NULL) {
        return false;
    }

    for (i = 1; i <= s->model.nb_total; ++i) {
        int gi = i - 1;
        double lb = s->model.var_lb_all[gi];
        double ub = s->model.var_ub_all[gi];
        if ((i & 255) == 0 && solver_poll_time_limit(s)) {
            return false;
        }
        if (cand_x[gi] < lb - tol || cand_x[gi] > ub + tol) {
            return false;
        }
        if (st->b_assign[i] >= 0 && fabs(cand_x[gi] - (double)st->b_assign[i]) > tol) {
            return false;
        }
    }

    for (i = 1; i <= s->model.I; ++i) {
        int gi = s->model.nb_total + (i - 1);
        if ((i & 255) == 0 && solver_poll_time_limit(s)) {
            return false;
        }
        if (cand_x[gi] < st->i_lb[i] - tol || cand_x[gi] > st->i_ub[i] + tol) {
            return false;
        }
        if (require_integral && fabs(cand_x[gi] - nearbyint(cand_x[gi])) > s->opt.int_tol) {
            return false;
        }
    }

    for (i = 1; i <= s->model.R; ++i) {
        int gi = s->model.nb_total + s->model.I + (i - 1);
        if ((i & 255) == 0 && solver_poll_time_limit(s)) {
            return false;
        }
        if (cand_x[gi] < s->model.r_lb[i] - tol || cand_x[gi] > s->model.r_ub[i] + tol) {
            return false;
        }
    }

    for (i = 0; i < s->model.clauses.size; ++i) {
        const Clause *c = &s->model.clauses.data[i];
        int j;
        bool sat = false;
        if ((i & 255) == 0 && solver_poll_time_limit(s)) {
            return false;
        }
        for (j = 0; j < c->size; ++j) {
            if (candidate_lit_true(cand_x, c->lits[j])) {
                sat = true;
                break;
            }
        }
        if (!sat) {
            return false;
        }
    }

    for (i = 0; i < s->model.lin.size; ++i) {
        const LinConstraint *lc = &s->model.lin.data[i];
        double lhs = 0.0;
        int j;
        if ((i & 255) == 0 && solver_poll_time_limit(s)) {
            return false;
        }
        if (lc->indicator_lit != 0 && !candidate_lit_true(cand_x, lc->indicator_lit)) {
            continue;
        }
        for (j = 0; j < lc->nterms; ++j) {
            int gi = map_var_global_index(&s->model, lc->terms[j].kind, lc->terms[j].idx);
            if (gi < 0 || gi >= s->model.nvars_total) {
                return false;
            }
            lhs += lc->terms[j].coef * cand_x[gi];
        }
        if (lhs > lc->rhs + tol) {
            return false;
        }
    }

    if (obj_out != NULL) {
        *obj_out = s->model.obj_const;
        for (i = 0; i < s->model.nvars_total; ++i) {
            if ((i & 1023) == 0 && solver_poll_time_limit(s)) {
                return false;
            }
            *obj_out += s->model.obj_all[i] * cand_x[i];
        }
    }
    return true;
}

static bool solver_try_rounding_heuristic(Solver *s,
                                          const NodeState *st,
                                          const double *lp_x,
                                          double *cand_x,
                                          double *obj_out) {
    int i;
    double tol = fmax(s->opt.feas_tol, 1e-7);

    if (cand_x == NULL) {
        return false;
    }

    for (i = 1; i <= s->model.nb_total; ++i) {
        int gi = i - 1;
        double lb = s->model.var_lb_all[gi];
        double ub = s->model.var_ub_all[gi];
        int val;

        if ((i & 255) == 0 && solver_poll_time_limit(s)) {
            return false;
        }

        if (st->b_assign[i] >= 0) {
            val = st->b_assign[i];
        } else {
            val = (lp_x[gi] >= 0.5) ? 1 : 0;
        }
        if (lb > ub + tol) {
            return false;
        }
        if ((double)val < lb - tol) {
            val = (lb >= 0.5) ? 1 : 0;
        }
        if ((double)val > ub + tol) {
            val = (ub >= 0.5) ? 1 : 0;
        }
        if ((double)val < lb - tol || (double)val > ub + tol) {
            return false;
        }
        cand_x[gi] = (double)val;
    }

    for (i = 1; i <= s->model.I; ++i) {
        int gi = s->model.nb_total + (i - 1);
        double lb = st->i_lb[i];
        double ub = st->i_ub[i];
        double v;

        if ((i & 255) == 0 && solver_poll_time_limit(s)) {
            return false;
        }

        if (lb > ub + tol) {
            return false;
        }
        v = nearbyint(lp_x[gi]);
        if (v < lb) {
            v = ceil(lb - s->opt.int_tol);
        }
        if (v > ub) {
            v = floor(ub + s->opt.int_tol);
        }
        if (v < lb - tol || v > ub + tol) {
            return false;
        }
        cand_x[gi] = v;
    }

    for (i = 1; i <= s->model.R; ++i) {
        int gi = s->model.nb_total + s->model.I + (i - 1);
        double lb = s->model.r_lb[i];
        double ub = s->model.r_ub[i];
        double v = lp_x[gi];
        if ((i & 255) == 0 && solver_poll_time_limit(s)) {
            return false;
        }
        if (v < lb) {
            v = lb;
        }
        if (v > ub) {
            v = ub;
        }
        cand_x[gi] = v;
    }

    return solver_candidate_feasible(s, st, cand_x, true, obj_out);
}

static int choose_bool_branch_var(Solver *s, const NodeState *st, const double *lp_sol) {
    int ci;
    int v;
    int best = -1;
    double best_score = -1.0;

    if (s->branch_score != NULL) {
        memset(s->branch_score, 0, ((size_t)s->model.nb_total + 1U) * sizeof(double));
        for (ci = 0; ci < s->model.clauses.size; ++ci) {
            const Clause *c = &s->model.clauses.data[ci];
            int open = 0;
            bool sat = false;
            int j;
            double w;

            if ((ci & 255) == 0 && solver_poll_time_limit(s)) {
                return -1;
            }

            for (j = 0; j < c->size; ++j) {
                int lit = c->lits[j];
                int ev = eval_lit(st->b_assign[lit_var(lit)], lit);
                if (ev == 1) {
                    sat = true;
                    break;
                }
                if (ev < 0) {
                    ++open;
                }
            }
            if (sat || open == 0) {
                continue;
            }
            w = (open <= 2) ? 6.0 : ((open == 3) ? 3.0 : 1.0);
            for (j = 0; j < c->size; ++j) {
                int vv = lit_var(c->lits[j]);
                if (st->b_assign[vv] < 0) {
                    s->branch_score[vv] += w;
                }
            }
        }
    }

    for (v = 1; v <= s->model.nb_total; ++v) {
        if ((v & 255) == 0 && solver_poll_time_limit(s)) {
            return -1;
        }
        if (st->b_assign[v] >= 0) {
            continue;
        }
        {
            double score = s->var_activity[v];
            if (s->branch_score != NULL) {
                score += s->branch_score[v];
            }
            if (lp_sol != NULL) {
                double frac = fabs(lp_sol[v - 1] - 0.5);
                score += (0.5 - frac);
            }
            if (score > best_score + 1e-12) {
                best_score = score;
                best = v;
            }
        }
    }
    return best;
}

static int choose_input_bool_branch_var(Solver *s, const NodeState *st) {
    int ci;
    int v;
    int best = -1;
    double best_score = -1.0;

    if (s->branch_score != NULL) {
        memset(s->branch_score, 0, ((size_t)s->model.nb_total + 1U) * sizeof(double));
        for (ci = 0; ci < s->model.clauses.size; ++ci) {
            const Clause *c = &s->model.clauses.data[ci];
            int open = 0;
            bool sat = false;
            int j;
            double w;

            if ((ci & 255) == 0 && solver_poll_time_limit(s)) {
                return -1;
            }

            for (j = 0; j < c->size; ++j) {
                int lit = c->lits[j];
                int ev = eval_lit(st->b_assign[lit_var(lit)], lit);
                if (ev == 1) {
                    sat = true;
                    break;
                }
                if (ev < 0) {
                    ++open;
                }
            }
            if (sat || open == 0) {
                continue;
            }
            w = (open <= 2) ? 6.0 : ((open == 3) ? 3.0 : 1.0);
            for (j = 0; j < c->size; ++j) {
                int vv = lit_var(c->lits[j]);
                if (vv >= 1 && vv <= s->model.B && st->b_assign[vv] < 0) {
                    s->branch_score[vv] += w;
                }
            }
        }
    }

    for (v = 1; v <= s->model.B; ++v) {
        double score;
        if ((v & 255) == 0 && solver_poll_time_limit(s)) {
            return -1;
        }
        if (st->b_assign[v] >= 0) {
            continue;
        }
        score = s->var_activity[v];
        if (s->branch_score != NULL) {
            score += s->branch_score[v];
        }
        if (score > best_score + 1e-12) {
            best_score = score;
            best = v;
        }
    }
    return best;
}

static int choose_frac_int_var(Solver *s, const NodeState *st, const double *lp_sol) {
    int i;
    int best = -1;
    double best_frac = s->opt.int_tol;
    for (i = 1; i <= s->model.I; ++i) {
        int gi = s->model.nb_total + (i - 1);
        double v = lp_sol[gi];
        double frac = fabs(v - nearbyint(v));
        if ((i & 255) == 0 && solver_poll_time_limit(s)) {
            return -1;
        }
        if (st->i_lb[i] > st->i_ub[i] + s->opt.feas_tol) {
            continue;
        }
        if (frac > best_frac) {
            best_frac = frac;
            best = i;
        }
    }
    return best;
}

#if defined(SLIME_NO_MAIN)
static bool solver_use_slime_completion(const Solver *s) {
    return s->slime_handle != NULL && s->model.I == 0 && s->model.R == 0 && s->model.lin.size == 0;
}

static bool solver_build_soft_order(Solver *s) {
    int i;
    free(s->soft_order);
    s->soft_order = NULL;
    if (s->model.soft.size <= 0) {
        return true;
    }
    s->soft_order = (int *)malloc((size_t)s->model.soft.size * sizeof(int));
    if (s->soft_order == NULL) {
        return false;
    }
    for (i = 0; i < s->model.soft.size; ++i) {
        s->soft_order[i] = i;
    }
    g_soft_sort_view = &s->model.soft;
    qsort(s->soft_order, (size_t)s->model.soft.size, sizeof(int), soft_index_weight_desc_cmp);
    g_soft_sort_view = NULL;
    return true;
}

static int solver_fill_slime_assumptions(const Solver *s, const NodeState *st) {
    int i;
    int num_assumptions = 0;
    for (i = 1; i <= s->model.nb_total; ++i) {
        if (st->b_assign[i] >= 0) {
            s->slime_assumptions[num_assumptions++] = (st->b_assign[i] > 0) ? i : -i;
        }
    }
    return num_assumptions;
}

static bool solver_try_slime_completion(Solver *s, const NodeState *st, double *cand_x, double *obj_out) {
    int i;
    int num_assumptions = 0;
    int rc;

    if (!solver_use_slime_completion(s) || cand_x == NULL || obj_out == NULL) {
        return false;
    }

    num_assumptions = solver_fill_slime_assumptions(s, st);

    rc = slime_sat_handle_solve(s->slime_handle,
                                s->slime_assumptions,
                                num_assumptions,
                                NULL,
                                s->slime_model01);
    if (rc != 10) {
        return false;
    }

    *obj_out = s->model.obj_const;
    for (i = 0; i < s->model.nb_total; ++i) {
        cand_x[i] = s->slime_model01[i] ? 1.0 : 0.0;
        *obj_out += s->model.obj_all[i] * cand_x[i];
    }
    return true;
}

static bool solver_should_use_bool_satbb_fallback(const Solver *s) {
    double dense_cells;
    double m_est;
    if (!solver_use_slime_completion(s)) {
        return false;
    }
    m_est = (double)s->model.lin.size + (double)s->model.clauses.size + (double)s->model.nvars_total + 8.0;
    dense_cells = (double)s->model.nvars_total * m_est;
    return dense_cells > WMIBO_MAX_DENSE_LP_CELLS;
}
#endif

static bool node_state_init_root(const Solver *s, NodeState *st) {
    int v;
    st->b_assign = (int8_t *)malloc((size_t)s->model.nb_total + 1U);
    st->i_lb = (double *)malloc(((size_t)s->model.I + 1U) * sizeof(double));
    st->i_ub = (double *)malloc(((size_t)s->model.I + 1U) * sizeof(double));
    if ((s->model.nb_total > 0 && st->b_assign == NULL) ||
        (s->model.I > 0 && (st->i_lb == NULL || st->i_ub == NULL))) {
        free(st->b_assign); free(st->i_lb); free(st->i_ub);
        st->b_assign = NULL; st->i_lb = NULL; st->i_ub = NULL;
        return false;
    }
    intvec_init(&st->decisions);

    for (v = 1; v <= s->model.nb_total; ++v) st->b_assign[v] = -1;

    for (v = 1; v <= s->model.B; ++v) {
        double lb = s->model.var_lb_all[v - 1];
        double ub = s->model.var_ub_all[v - 1];
        if (lb > 0.5) st->b_assign[v] = 1;
        if (ub < 0.5) {
            if (st->b_assign[v] == 1) return false;
            st->b_assign[v] = 0;
        }
    }

    for (v = 1; v <= s->model.I; ++v) {
        st->i_lb[v] = s->model.i_lb[v];
        st->i_ub[v] = s->model.i_ub[v];
    }

    return true;
}

static void node_state_free(NodeState *st) {
    free(st->b_assign);
    free(st->i_lb);
    free(st->i_ub);
    st->b_assign = NULL;
    st->i_lb = NULL;
    st->i_ub = NULL;
    intvec_free(&st->decisions);
}

static bool node_state_copy(const Solver *s, const NodeState *src, NodeState *dst) {
    size_t bsz = (size_t)s->model.nb_total + 1U;
    size_t isz = (size_t)s->model.I + 1U;

    dst->b_assign = (int8_t *)malloc(bsz);
    dst->i_lb = (double *)malloc(isz * sizeof(double));
    dst->i_ub = (double *)malloc(isz * sizeof(double));
    if ((s->model.nb_total > 0 && dst->b_assign == NULL) ||
        (s->model.I > 0 && (dst->i_lb == NULL || dst->i_ub == NULL))) {
        free(dst->b_assign); free(dst->i_lb); free(dst->i_ub);
        dst->b_assign = NULL; dst->i_lb = NULL; dst->i_ub = NULL;
        return false;
    }

    memcpy(dst->b_assign, src->b_assign, bsz);
    memcpy(dst->i_lb, src->i_lb, isz * sizeof(double));
    memcpy(dst->i_ub, src->i_ub, isz * sizeof(double));

    if (!intvec_copy(&dst->decisions, &src->decisions)) {
        free(dst->b_assign); free(dst->i_lb); free(dst->i_ub);
        dst->b_assign = NULL; dst->i_lb = NULL; dst->i_ub = NULL;
        return false;
    }

    return true;
}

#if defined(SLIME_NO_MAIN)
static bool solver_init_slime_bridge(Solver *s) {
    int i;

    if (s->model.nb_total <= 0 || s->model.I != 0 || s->model.R != 0 || s->model.lin.size != 0) {
        return true;
    }

    s->slime_clause_ptrs = (const int **)calloc((size_t)(s->model.clauses.size > 0 ? s->model.clauses.size : 1), sizeof(const int *));
    s->slime_clause_sizes = (int *)calloc((size_t)(s->model.clauses.size > 0 ? s->model.clauses.size : 1), sizeof(int));
    s->slime_assumptions = (int *)calloc((size_t)s->model.nb_total, sizeof(int));
    s->slime_model01 = (unsigned char *)calloc((size_t)s->model.nb_total, sizeof(unsigned char));
    if ((s->model.clauses.size > 0 && (s->slime_clause_ptrs == NULL || s->slime_clause_sizes == NULL)) ||
        (s->model.nb_total > 0 && (s->slime_assumptions == NULL || s->slime_model01 == NULL))) {
        return false;
    }

    for (i = 0; i < s->model.clauses.size; ++i) {
        s->slime_clause_ptrs[i] = s->model.clauses.data[i].lits;
        s->slime_clause_sizes[i] = s->model.clauses.data[i].size;
    }

    s->slime_handle = slime_sat_handle_create(s->model.nb_total,
                                              s->model.clauses.size,
                                              s->slime_clause_ptrs,
                                              s->slime_clause_sizes,
                                              NULL);
    return s->slime_handle != NULL;
}

static void solver_free_slime_bridge(Solver *s) {
    slime_sat_handle_destroy(s->slime_handle);
    free(s->slime_clause_ptrs);
    free(s->slime_clause_sizes);
    free(s->slime_assumptions);
    free(s->slime_model01);
    s->slime_handle = NULL;
    s->slime_clause_ptrs = NULL;
    s->slime_clause_sizes = NULL;
    s->slime_assumptions = NULL;
    s->slime_model01 = NULL;
}
#endif

static bool should_stop(Solver *s) {
    if (s->stop) {
        return true;
    }
    if (s->opt.node_limit > 0 && s->nodes >= s->opt.node_limit) {
        s->stop = true;
        s->stopped_nodes = true;
        return true;
    }
    if (s->opt.time_limit > 0.0) {
        double t = now_seconds() - s->start_time;
        if (t >= s->opt.time_limit) {
            s->stop = true;
            s->stopped_time = true;
            return true;
        }
    }
    if (s->opt.rel_gap > 0.0 && s->have_incumbent && s->have_root_lb) {
        double den = fabs(s->incumbent);
        if (den < 1.0) den = 1.0;
        if ((s->incumbent - s->root_lb) <= s->opt.rel_gap * den + s->opt.obj_tol) {
            s->stop = true;
            s->stopped_gap = true;
            return true;
        }
    }
    return false;
}

static bool learn_clause_from_decisions(Solver *s, const NodeState *st) {
    IntVec lits;
    int i;
    bool added = false;
    intvec_init(&lits);

    for (i = 0; i < st->decisions.size; ++i) {
        if (!intvec_push(&lits, lit_neg(st->decisions.data[i]))) {
            intvec_free(&lits);
            return false;
        }
    }

    if (lits.size == 0) {
        intvec_free(&lits);
        return true;
    }

    if (!add_clause_learned(&s->model, lits.data, lits.size, &added)) {
        intvec_free(&lits);
        return false;
    }
    if (added) {
        int j;
        s->clause_learned++;
        for (j = 0; j < lits.size; ++j) {
            int v = lit_var(lits.data[j]);
            if (v >= 1 && v <= s->model.nb_total) s->var_activity[v] += 1.0;
        }
    }
    intvec_free(&lits);
    return true;
}
static bool learn_clause_from_active_indicators(Solver *s, const IntVec *active_ind) {
    IntVec lits;
    int i;
    bool added = false;

    intvec_init(&lits);
    for (i = 0; i < active_ind->size; ++i) {
        if (!intvec_push(&lits, lit_neg(active_ind->data[i]))) {
            intvec_free(&lits);
            return false;
        }
    }
    if (lits.size == 0) {
        intvec_free(&lits);
        return true;
    }
    if (!add_clause_learned(&s->model, lits.data, lits.size, &added)) {
        intvec_free(&lits);
        return false;
    }
    if (added) {
        int j;
        s->clause_learned++;
        for (j = 0; j < lits.size; ++j) {
            int v = lit_var(lits.data[j]);
            if (v >= 1 && v <= s->model.nb_total) s->var_activity[v] += 0.5;
        }
    }
    intvec_free(&lits);
    return true;
}

static bool update_incumbent(Solver *s, const NodeState *st, const double *x, double obj) {
    int i;
    if (s->have_incumbent && obj >= s->incumbent - s->opt.obj_tol) {
        return false;
    }
    s->have_incumbent = true;
    s->incumbent = obj;
    s->report_incumbent = s->model.obj_is_max ? -obj : obj;

    for (i = 1; i <= s->model.B; ++i) {
        s->best_b[i] = (st->b_assign[i] >= 0) ? st->b_assign[i] : (x[i - 1] >= 0.5 ? 1 : 0);
    }
    for (i = 1; i <= s->model.I; ++i) {
        int gi = s->model.nb_total + (i - 1);
        s->best_i[i] = x[gi];
    }
    for (i = 1; i <= s->model.R; ++i) {
        int gi = s->model.nb_total + s->model.I + (i - 1);
        s->best_r[i] = x[gi];
    }
    return true;
}

static double solver_bool_objective_lower_bound(const Solver *s, const NodeState *st) {
    double lb = s->model.obj_const;
    int i;
    for (i = 0; i < s->model.nb_total; ++i) {
        double coef = s->model.obj_all[i];
        int assign = st->b_assign[i + 1];
        if (assign >= 0) {
            lb += coef * (double)assign;
        } else if (coef < 0.0) {
            lb += coef;
        } else if (s->model.var_lb_all[i] > 0.5) {
            lb += coef;
        }
    }
    return lb;
}

static bool solver_soft_clause_satisfied_partial(const SoftClause *sc, const int8_t *assign) {
    int j;
    for (j = 0; j < sc->size; ++j) {
        int lit = sc->lits[j];
        int ev = eval_lit(assign[lit_var(lit)], lit);
        if (ev == 1) {
            return true;
        }
    }
    return false;
}

#if defined(SLIME_NO_MAIN)
static double solver_bool_soft_probe_lower_bound(Solver *s, const NodeState *st, int depth) {
    double extra = 0.0;
    int group[WMIBO_BOOL_LB_GROUP_SIZE];
    double group_min = 0.0;
    int group_fill = 0;
    int singleton_used = 0;
    int group_used = 0;
    int current_assumptions;
    int ord_pos;

    if (!solver_use_slime_completion(s) || s->soft_order == NULL || s->model.soft.size <= 0) {
        return 0.0;
    }
    if (depth > 48 || (depth > 8 && (depth & 3) != 0)) {
        return 0.0;
    }

    current_assumptions = solver_fill_slime_assumptions(s, st);
    for (ord_pos = 0; ord_pos < s->model.soft.size; ++ord_pos) {
        int soft_idx;
        int rv;
        int rc;
        const SoftClause *sc;

        if ((ord_pos & 31) == 0 && solver_poll_time_limit(s)) {
            break;
        }

        soft_idx = s->soft_order[ord_pos];
        rv = s->model.B + soft_idx + 1;
        sc = &s->model.soft.data[soft_idx];

        if (st->b_assign[rv] >= 0) {
            continue;
        }
        if (solver_soft_clause_satisfied_partial(sc, st->b_assign)) {
            continue;
        }

        if (singleton_used < WMIBO_BOOL_LB_SINGLETON_PROBES) {
            s->slime_assumptions[current_assumptions] = -rv;
            rc = slime_sat_handle_solve(s->slime_handle,
                                        s->slime_assumptions,
                                        current_assumptions + 1,
                                        NULL,
                                        s->slime_model01);
            ++singleton_used;
            if (rc == 20) {
                extra += sc->weight;
                continue;
            }
        }

        if (group_used < WMIBO_BOOL_LB_GROUP_PROBES) {
            group[group_fill++] = soft_idx;
            if (group_fill == 1 || sc->weight < group_min) {
                group_min = sc->weight;
            }
            if (group_fill == WMIBO_BOOL_LB_GROUP_SIZE) {
                int k;
                for (k = 0; k < group_fill; ++k) {
                    s->slime_assumptions[current_assumptions + k] = -(s->model.B + group[k] + 1);
                }
                rc = slime_sat_handle_solve(s->slime_handle,
                                            s->slime_assumptions,
                                            current_assumptions + group_fill,
                                            NULL,
                                            s->slime_model01);
                ++group_used;
                if (rc == 20) {
                    extra += group_min;
                }
                group_fill = 0;
                group_min = 0.0;
            }
        }
    }

    if (group_fill > 1 && group_used < WMIBO_BOOL_LB_GROUP_PROBES && !solver_poll_time_limit(s)) {
        int k;
        int rc;
        for (k = 0; k < group_fill; ++k) {
            s->slime_assumptions[current_assumptions + k] = -(s->model.B + group[k] + 1);
        }
        rc = slime_sat_handle_solve(s->slime_handle,
                                    s->slime_assumptions,
                                    current_assumptions + group_fill,
                                    NULL,
                                    s->slime_model01);
        if (rc == 20) {
            extra += group_min;
        }
    }

    return extra;
}
#endif

static void solver_search_bool_satbb(Solver *s, NodeState *st, int depth) {
    int conflict_clause;
    double lb;
    double feas_obj = NAN;
    bool have_completion = false;
    int branch_var;

    if (should_stop(s)) {
        return;
    }

    s->nodes++;

    conflict_clause = propagate_clauses(s, &s->model, st->b_assign);
    if (conflict_clause == -2) {
        return;
    }
    if (conflict_clause >= 0 || s->model.hard_unsat) {
        s->bool_conflicts++;
        if (!learn_clause_from_decisions(s, st)) {
            s->stop = true;
            return;
        }
        if (st->decisions.size == 0) {
            s->model.hard_unsat = true;
        }
        return;
    }

    lb = solver_bool_objective_lower_bound(s, st);
#if defined(SLIME_NO_MAIN)
    lb += solver_bool_soft_probe_lower_bound(s, st, depth);
#endif
    if (depth == 0) {
        s->root_lb = lb;
        s->have_root_lb = true;
    }
    if (s->have_incumbent && lb >= s->incumbent - s->opt.obj_tol) {
        return;
    }

#if defined(SLIME_NO_MAIN)
    have_completion = solver_try_slime_completion(s, st, s->heur_x, &feas_obj);
#endif
    if (!have_completion) {
        s->bool_conflicts++;
        if (!learn_clause_from_decisions(s, st)) {
            s->stop = true;
            return;
        }
        if (st->decisions.size == 0) {
            s->model.hard_unsat = true;
        }
        return;
    }

    update_incumbent(s, st, s->heur_x, feas_obj);
    if (feas_obj <= lb + s->opt.obj_tol) {
        return;
    }

    branch_var = choose_input_bool_branch_var(s, st);
    if (branch_var <= 0) {
        return;
    }

    {
        int first = (s->heur_x[branch_var - 1] >= 0.5) ? 1 : 0;
        int second = 1 - first;
        int order[2] = { first, second };
        int t;

        if (s->opt.seed_set && (xorshift64(&s->rng_state) & 1ULL)) {
            int swap = order[0];
            order[0] = order[1];
            order[1] = swap;
        }

        for (t = 0; t < 2 && !should_stop(s); ++t) {
            int val = order[t];
            int lit = (val == 1) ? branch_var : -branch_var;
            NodeState ch;
            if (!node_state_copy(s, st, &ch)) {
                s->stop = true;
                break;
            }
            ch.b_assign[branch_var] = (int8_t)val;
            if (!intvec_push(&ch.decisions, lit)) {
                s->stop = true;
                node_state_free(&ch);
                break;
            }
            solver_search_bool_satbb(s, &ch, depth + 1);
            node_state_free(&ch);
        }
    }
}

static void solver_search(Solver *s, NodeState *st, int depth) {
    LPResult lp;
    IntVec active_ind;
    int conflict_clause;

    if (s->use_bool_satbb) {
        solver_search_bool_satbb(s, st, depth);
        return;
    }

    if (should_stop(s)) {
        return;
    }

    s->nodes++;

    conflict_clause = propagate_clauses(s, &s->model, st->b_assign);
    if (conflict_clause == -2) {
        return;
    }
    if (conflict_clause >= 0 || s->model.hard_unsat) {
        s->bool_conflicts++;
        if (!learn_clause_from_decisions(s, st)) {
            s->stop = true;
            return;
        }
        if (st->decisions.size == 0) {
            s->model.hard_unsat = true;
        }
        return;
    }

    s->lp_calls++;
    lp = solve_lp_relaxation(s, st, &active_ind);

    if (depth == 0 && lp.status == LP_STATUS_OPTIMAL) {
        s->root_lb = lp.obj;
        s->have_root_lb = true;
    }

    if (lp.status == LP_STATUS_INFEASIBLE) {
        s->theory_conflicts++;
        if (!state_has_local_int_branch(s, st) && active_ind.size > 0) {
            if (!learn_clause_from_active_indicators(s, &active_ind)) {
                intvec_free(&active_ind);
                free(lp.x);
                s->stop = true;
                return;
            }
        } else {
            if (!learn_clause_from_decisions(s, st)) {
                intvec_free(&active_ind);
                free(lp.x);
                s->stop = true;
                return;
            }
            if (st->decisions.size == 0 && !state_has_local_int_branch(s, st)) {
                s->model.hard_unsat = true;
            }
        }
        intvec_free(&active_ind);
        free(lp.x);
        return;
    }

    if (lp.status == LP_STATUS_UNBOUNDED) {
        if (all_bool_assigned(s, st)) {
            s->found_unbounded = true;
            intvec_free(&active_ind);
            free(lp.x);
            s->stop = true;
            return;
        }
        intvec_free(&active_ind);
        free(lp.x);
    } else if (lp.status == LP_STATUS_ERROR) {
        intvec_free(&active_ind);
        free(lp.x);
        s->stop = true;
        return;
    } else {
        int bool_var;
        int int_var;
        double heur_obj = NAN;
        bool have_heur = false;

        if (s->have_incumbent && lp.obj >= s->incumbent - s->opt.obj_tol) {
            intvec_free(&active_ind);
            free(lp.x);
            return;
        }

#if defined(SLIME_NO_MAIN)
        if (solver_try_slime_completion(s, st, s->heur_x, &heur_obj)) {
            update_incumbent(s, st, s->heur_x, heur_obj);
            if (heur_obj <= lp.obj + s->opt.obj_tol) {
                intvec_free(&active_ind);
                free(lp.x);
                return;
            }
        }
#endif

        have_heur = solver_try_rounding_heuristic(s, st, lp.x, s->heur_x, &heur_obj);
        if (have_heur) {
            update_incumbent(s, st, s->heur_x, heur_obj);
            if (heur_obj <= lp.obj + s->opt.obj_tol) {
                intvec_free(&active_ind);
                free(lp.x);
                return;
            }
        }

        if (all_bool_assigned(s, st)) {
            bool ints_ok = true;
            int i;
            for (i = 1; i <= s->model.I; ++i) {
                int gi = s->model.nb_total + (i - 1);
                if (fabs(lp.x[gi] - nearbyint(lp.x[gi])) > s->opt.int_tol) {
                    ints_ok = false;
                    break;
                }
            }
            if (ints_ok &&
                all_hard_clauses_satisfied(s, st) &&
                solver_candidate_feasible(s, st, lp.x, true, &heur_obj)) {
                update_incumbent(s, st, lp.x, heur_obj);
                intvec_free(&active_ind);
                free(lp.x);
                return;
            }
        }

        bool_var = choose_bool_branch_var(s, st, lp.x);
        if (bool_var > 0) {
            int first = (lp.x[bool_var - 1] >= 0.5) ? 1 : 0;
            int second = 1 - first;
            NodeState ch;
            int t;
            int order[2] = { first, second };

            if (s->opt.seed_set && (xorshift64(&s->rng_state) & 1ULL)) {
                int swap = order[0];
                order[0] = order[1];
                order[1] = swap;
            }

            for (t = 0; t < 2 && !should_stop(s); ++t) {
                int val = order[t];
                int lit = (val == 1) ? bool_var : -bool_var;
                if (!node_state_copy(s, st, &ch)) {
                    s->stop = true;
                    break;
                }
                ch.b_assign[bool_var] = (int8_t)val;
                if (!intvec_push(&ch.decisions, lit)) {
                    s->stop = true;
                    node_state_free(&ch);
                    break;
                }
                solver_search(s, &ch, depth + 1);
                node_state_free(&ch);
            }

            intvec_free(&active_ind);
            free(lp.x);
            return;
        }

        int_var = choose_frac_int_var(s, st, lp.x);
        if (int_var > 0) {
            double v = lp.x[s->model.nb_total + (int_var - 1)];
            double fl = floor(v);
            double ce = ceil(v);
            double down_gap = v - fl;
            double up_gap = ce - v;
            double obj_coef = s->model.obj_all[s->model.nb_total + (int_var - 1)];
            bool prefer_down = (down_gap <= up_gap);
            NodeState left;
            NodeState right;
            bool did_branch = false;

            if (fabs(down_gap - up_gap) <= s->opt.int_tol) {
                prefer_down = (obj_coef >= 0.0);
            }

            if (prefer_down) {
                if (fl >= st->i_lb[int_var] - s->opt.feas_tol) {
                    if (node_state_copy(s, st, &left)) {
                        if (fl < left.i_ub[int_var]) left.i_ub[int_var] = fl;
                        if (left.i_lb[int_var] <= left.i_ub[int_var] + s->opt.feas_tol) {
                            solver_search(s, &left, depth + 1);
                        }
                        node_state_free(&left);
                        did_branch = true;
                    } else {
                        s->stop = true;
                    }
                }

                if (!should_stop(s) && ce <= st->i_ub[int_var] + s->opt.feas_tol) {
                    if (node_state_copy(s, st, &right)) {
                        if (ce > right.i_lb[int_var]) right.i_lb[int_var] = ce;
                        if (right.i_lb[int_var] <= right.i_ub[int_var] + s->opt.feas_tol) {
                            solver_search(s, &right, depth + 1);
                        }
                        node_state_free(&right);
                        did_branch = true;
                    } else {
                        s->stop = true;
                    }
                }
            } else {
                if (ce <= st->i_ub[int_var] + s->opt.feas_tol) {
                    if (node_state_copy(s, st, &right)) {
                        if (ce > right.i_lb[int_var]) right.i_lb[int_var] = ce;
                        if (right.i_lb[int_var] <= right.i_ub[int_var] + s->opt.feas_tol) {
                            solver_search(s, &right, depth + 1);
                        }
                        node_state_free(&right);
                        did_branch = true;
                    } else {
                        s->stop = true;
                    }
                }

                if (!should_stop(s) && fl >= st->i_lb[int_var] - s->opt.feas_tol) {
                    if (node_state_copy(s, st, &left)) {
                        if (fl < left.i_ub[int_var]) left.i_ub[int_var] = fl;
                        if (left.i_lb[int_var] <= left.i_ub[int_var] + s->opt.feas_tol) {
                            solver_search(s, &left, depth + 1);
                        }
                        node_state_free(&left);
                        did_branch = true;
                    } else {
                        s->stop = true;
                    }
                }
            }

            if (!did_branch) {
                update_incumbent(s, st, lp.x, lp.obj);
            }

            intvec_free(&active_ind);
            free(lp.x);
            return;
        }

        if (all_hard_clauses_satisfied(s, st) &&
            solver_candidate_feasible(s, st, lp.x, true, &heur_obj)) {
            update_incumbent(s, st, lp.x, heur_obj);
        }
        intvec_free(&active_ind);
        free(lp.x);
        return;
    }
}

static void solver_apply_file_options(Solver *s) {
    const FileOptions *f = &s->model.fopt;
    if (f->has_feas_tol) s->opt.feas_tol = f->feas_tol;
    if (f->has_int_tol) s->opt.int_tol = f->int_tol;
    if (f->has_time_limit && s->opt.time_limit <= 0.0) s->opt.time_limit = f->time_limit;
    if (f->has_node_limit && s->opt.node_limit <= 0) s->opt.node_limit = f->node_limit;
    if (f->has_gap && s->opt.rel_gap <= 0.0) s->opt.rel_gap = f->rel_gap;
    if (f->has_seed && !s->opt.seed_set) { s->opt.seed = f->seed; s->opt.seed_set = true; }
    if (f->has_verbose) s->opt.verbose = f->verbose;
}

static SolveStatus run_solver(Solver *s) {
    NodeState root;
    int i;

    s->nodes = 0;
    s->lp_calls = 0;
    s->clause_learned = 0;
    s->bool_conflicts = 0;
    s->theory_conflicts = 0;

    s->stop = false;
    s->stopped_time = false;
    s->stopped_nodes = false;
    s->stopped_gap = false;

    s->have_incumbent = false;
    s->incumbent = WMIBO_INF;
    s->report_incumbent = NAN;
    s->have_root_lb = false;
    s->root_lb = -WMIBO_INF;
    s->found_unbounded = false;
    s->use_bool_satbb = false;

    s->var_activity = (double *)calloc((size_t)s->model.nb_total + 1U, sizeof(double));
    s->branch_score = (double *)calloc((size_t)s->model.nb_total + 1U, sizeof(double));
    s->best_b = (int *)calloc((size_t)s->model.B + 1U, sizeof(int));
    s->best_i = (double *)calloc((size_t)s->model.I + 1U, sizeof(double));
    s->best_r = (double *)calloc((size_t)s->model.R + 1U, sizeof(double));
    s->heur_x = (double *)calloc((size_t)(s->model.nvars_total > 0 ? s->model.nvars_total : 1), sizeof(double));

    if ((s->model.nb_total > 0 && s->var_activity == NULL) ||
        (s->model.nb_total > 0 && s->branch_score == NULL) ||
        (s->model.B > 0 && s->best_b == NULL) ||
        (s->model.I > 0 && s->best_i == NULL) ||
        (s->model.R > 0 && s->best_r == NULL) ||
        (s->model.nvars_total > 0 && s->heur_x == NULL)) {
        free(s->var_activity); free(s->branch_score); free(s->best_b); free(s->best_i); free(s->best_r); free(s->heur_x);
        s->var_activity = NULL; s->branch_score = NULL; s->best_b = NULL; s->best_i = NULL; s->best_r = NULL; s->heur_x = NULL;
        return SOLVE_STATUS_UNKNOWN;
    }

#if defined(SLIME_NO_MAIN)
    if (!solver_init_slime_bridge(s)) {
        free(s->var_activity); free(s->branch_score); free(s->best_b); free(s->best_i); free(s->best_r); free(s->heur_x);
        s->var_activity = NULL; s->branch_score = NULL; s->best_b = NULL; s->best_i = NULL; s->best_r = NULL; s->heur_x = NULL;
        solver_free_slime_bridge(s);
        return SOLVE_STATUS_UNKNOWN;
    }
    if (!solver_build_soft_order(s)) {
        free(s->var_activity); free(s->branch_score); free(s->best_b); free(s->best_i); free(s->best_r); free(s->heur_x);
        s->var_activity = NULL; s->branch_score = NULL; s->best_b = NULL; s->best_i = NULL; s->best_r = NULL; s->heur_x = NULL;
        solver_free_slime_bridge(s);
        return SOLVE_STATUS_UNKNOWN;
    }
    s->use_bool_satbb = solver_should_use_bool_satbb_fallback(s);
    if (s->use_bool_satbb && s->opt.verbose >= 1) {
        fprintf(stderr, "c wmibo using SAT-based pure-boolean fallback because dense LP relaxation is too large\n");
    }
#endif

    for (i = 0; i < s->model.clauses.size; ++i) {
        const Clause *c = &s->model.clauses.data[i];
        int j;
        for (j = 0; j < c->size; ++j) {
            int v = lit_var(c->lits[j]);
            if (v >= 1 && v <= s->model.nb_total) s->var_activity[v] += c->learnt ? 1.5 : 1.0;
        }
    }

    if (!node_state_init_root(s, &root)) {
        free(s->var_activity); free(s->branch_score); free(s->best_b); free(s->best_i); free(s->best_r); free(s->heur_x);
        s->var_activity = NULL; s->branch_score = NULL; s->best_b = NULL; s->best_i = NULL; s->best_r = NULL; s->heur_x = NULL;
        return SOLVE_STATUS_UNKNOWN;
    }

    if (s->start_time <= 0.0) {
        s->start_time = now_seconds();
    }
    s->rng_state = s->opt.seed_set ? s->opt.seed : 1U;

    if (s->model.hard_unsat) {
        node_state_free(&root);
        free(s->var_activity); free(s->branch_score); free(s->heur_x);
        s->var_activity = NULL; s->branch_score = NULL; s->heur_x = NULL;
        return SOLVE_STATUS_INFEASIBLE;
    }

    solver_search(s, &root, 0);
    node_state_free(&root);
    if (s->found_unbounded) {
        free(s->var_activity); free(s->branch_score); free(s->heur_x);
        s->var_activity = NULL; s->branch_score = NULL; s->heur_x = NULL;
        return SOLVE_STATUS_UNBOUNDED;
    }

    if (s->have_incumbent && !s->stop) {
        free(s->var_activity); free(s->branch_score); free(s->heur_x);
        s->var_activity = NULL; s->branch_score = NULL; s->heur_x = NULL;
        return SOLVE_STATUS_OPTIMUM;
    }

    if (s->have_incumbent && s->stop) {
        free(s->var_activity); free(s->branch_score); free(s->heur_x);
        s->var_activity = NULL; s->branch_score = NULL; s->heur_x = NULL;
        return SOLVE_STATUS_UNKNOWN;
    }

    if (s->model.hard_unsat || (s->clause_learned > 0 && !s->have_incumbent && !s->stop)) {
        free(s->var_activity); free(s->branch_score); free(s->heur_x);
        s->var_activity = NULL; s->branch_score = NULL; s->heur_x = NULL;
        return SOLVE_STATUS_INFEASIBLE;
    }

    if (!s->have_incumbent && !s->stop) {
        free(s->var_activity); free(s->branch_score); free(s->heur_x);
        s->var_activity = NULL; s->branch_score = NULL; s->heur_x = NULL;
        return SOLVE_STATUS_INFEASIBLE;
    }

    free(s->var_activity); free(s->branch_score); free(s->heur_x);
    s->var_activity = NULL; s->branch_score = NULL; s->heur_x = NULL;
    return SOLVE_STATUS_UNKNOWN;
}

static void solver_cleanup_results(Solver *s) {
    free(s->best_b);
    free(s->best_i);
    free(s->best_r);
    free(s->var_activity);
    free(s->branch_score);
    free(s->heur_x);
    free(s->soft_order);
#if defined(SLIME_NO_MAIN)
    solver_free_slime_bridge(s);
#endif
    s->best_b = NULL;
    s->best_i = NULL;
    s->best_r = NULL;
    s->var_activity = NULL;
    s->branch_score = NULL;
    s->heur_x = NULL;
    s->soft_order = NULL;
}

static void print_solution_line(const Solver *s) {
    int i;
    printf("v");
    for (i = 1; i <= s->model.B; ++i) {
        printf(" b%d=%d", i, s->best_b[i]);
    }
    for (i = 1; i <= s->model.I; ++i) {
        printf(" i%d=%.0f", i, nearbyint(s->best_i[i]));
    }
    for (i = 1; i <= s->model.R; ++i) {
        printf(" r%d=%.17g", i, s->best_r[i]);
    }
    printf("\n");
}

static const char *status_to_text(SolveStatus st) {
    if (st == SOLVE_STATUS_OPTIMUM) return "OPTIMUM";
    if (st == SOLVE_STATUS_INFEASIBLE) return "INFEASIBLE";
    if (st == SOLVE_STATUS_UNBOUNDED) return "UNBOUNDED";
    return "UNKNOWN";
}

static bool explain_literal_true_from_solution(const Solver *s, int lit) {
    int v = lit_var(lit);
    int val = -1;
    if (v >= 1 && v <= s->model.B) {
        val = s->best_b[v];
    }
    if (val < 0) {
        return false;
    }
    return lit_sign(lit) ? (val == 1) : (val == 0);
}

static bool explain_soft_clause_satisfied(const Solver *s, const SoftClause *sc) {
    int j;
    for (j = 0; j < sc->size; ++j) {
        if (explain_literal_true_from_solution(s, sc->lits[j])) {
            return true;
        }
    }
    return false;
}

static bool explain_hard_clause_satisfied(const Solver *s, const Clause *c) {
    int j;
    for (j = 0; j < c->size; ++j) {
        if (explain_literal_true_from_solution(s, c->lits[j])) {
            return true;
        }
    }
    return false;
}

static double explain_term_value(const Solver *s, const LinTerm *t) {
    if (t->kind == VAR_BOOL) {
        if (t->idx >= 1 && t->idx <= s->model.B) {
            return (double)s->best_b[t->idx];
        }
        return 0.0;
    }
    if (t->kind == VAR_INT) {
        if (t->idx >= 1 && t->idx <= s->model.I) {
            return s->best_i[t->idx];
        }
        return 0.0;
    }
    if (t->idx >= 1 && t->idx <= s->model.R) {
        return s->best_r[t->idx];
    }
    return 0.0;
}

static double explain_constraint_lhs(const Solver *s, const LinConstraint *lc) {
    double lhs = 0.0;
    int i;
    for (i = 0; i < lc->nterms; ++i) {
        lhs += lc->terms[i].coef * explain_term_value(s, &lc->terms[i]);
    }
    return lhs;
}

static bool explain_indicator_active_from_solution(const Solver *s, int lit) {
    if (lit == 0) return true;
    return explain_literal_true_from_solution(s, lit);
}

static bool explain_name_seen(const char *const *names, int n, const char *name) {
    int i;
    for (i = 0; i < n; ++i) {
        if (strcmp(names[i], name) == 0) {
            return true;
        }
    }
    return false;
}

static void explain_print_structure_summary(const Solver *s) {
    int hard_original = s->model.clauses.size - s->model.soft.size;
    if (hard_original < 0) hard_original = 0;
    printf("x structure bool=%d int=%d real=%d hard=%d soft=%d lin=%d indicators=%d\n",
           s->model.B,
           s->model.I,
           s->model.R,
           hard_original,
           s->model.soft.size,
           s->model.lin.size,
           s->model.pending_ind.size);
}

static void explain_print_solution_details(const Solver *s, SolveStatus st) {
    int i;
    int hard_original = s->model.clauses.size - s->model.soft.size;
    int hard_sat = 0;
    int soft_violated = 0;
    int active_rows = 0;
    int inactive_rows = 0;
    int tight_rows = 0;
    int active_indicators = 0;
    double base_obj = s->model.obj_const;
    double penalty = 0.0;
    const char *tight_names[8];
    int tight_name_count = 0;

    if (hard_original < 0) hard_original = 0;
    for (i = 1; i <= s->model.B; ++i) {
        base_obj += s->model.obj_b_input[i] * (double)s->best_b[i];
    }
    for (i = 1; i <= s->model.I; ++i) {
        base_obj += s->model.obj_i[i] * s->best_i[i];
    }
    for (i = 1; i <= s->model.R; ++i) {
        base_obj += s->model.obj_r[i] * s->best_r[i];
    }

    for (i = 0; i < hard_original; ++i) {
        if (explain_hard_clause_satisfied(s, &s->model.clauses.data[i])) {
            ++hard_sat;
        }
    }

    for (i = 0; i < s->model.soft.size; ++i) {
        const SoftClause *sc = &s->model.soft.data[i];
        if (!explain_soft_clause_satisfied(s, sc)) {
            ++soft_violated;
            penalty += sc->weight;
        }
    }

    for (i = 0; i < s->model.pending_ind.size; ++i) {
        if (explain_indicator_active_from_solution(s, s->model.pending_ind.data[i].lit)) {
            ++active_indicators;
        }
    }

    for (i = 0; i < s->model.lin.size; ++i) {
        const LinConstraint *lc = &s->model.lin.data[i];
        if (!explain_indicator_active_from_solution(s, lc->indicator_lit)) {
            ++inactive_rows;
            continue;
        }
        {
            double lhs = explain_constraint_lhs(s, lc);
            double slack = lc->rhs - lhs;
            ++active_rows;
            if (fabs(slack) <= fmax(s->opt.feas_tol, 1e-7)) {
                ++tight_rows;
                if (tight_name_count < (int)(sizeof(tight_names) / sizeof(tight_names[0])) &&
                    !explain_name_seen(tight_names, tight_name_count, lc->name)) {
                    tight_names[tight_name_count++] = lc->name;
                }
            }
        }
    }

    printf("x explain status=%s mode=solution\n", status_to_text(st));
    printf("x objective direction=%s base=%.12g penalty=%.12g reported=%.12g\n",
           s->model.obj_is_max ? "max" : "min",
           base_obj,
           penalty,
           s->report_incumbent);
    printf("x hard_clauses satisfied=%d total=%d\n", hard_sat, hard_original);
    printf("x soft_clauses violated=%d total=%d penalty_paid=%.12g\n",
           soft_violated,
           s->model.soft.size,
           penalty);
    printf("x linear_rows active=%d inactive=%d tight=%d total=%d\n",
           active_rows,
           inactive_rows,
           tight_rows,
           s->model.lin.size);
    printf("x indicators active=%d total=%d\n", active_indicators, s->model.pending_ind.size);
    if (tight_name_count > 0) {
        printf("x tight_constraints");
        for (i = 0; i < tight_name_count; ++i) {
            printf(" %s", tight_names[i]);
        }
        printf("\n");
    }
}

static int explain_eval_fixed_lit(int lit, const int8_t *assign) {
    int v = lit_var(lit);
    if (assign[v] < 0) {
        return -1;
    }
    return lit_sign(lit) ? (assign[v] == 1 ? 1 : 0) : (assign[v] == 0 ? 1 : 0);
}

static void explain_print_clause_compact(const Clause *c) {
    int j;
    for (j = 0; j < c->size; ++j) {
        int lit = c->lits[j];
        if (j > 0) printf(" ");
        if (lit < 0) {
            printf("~b%d", -lit);
        } else {
            printf("b%d", lit);
        }
    }
}

static void explain_interval_update(const LinTerm *t,
                                    const Model *m,
                                    const int8_t *assign,
                                    double *lhs_min,
                                    double *lhs_max) {
    double lb = 0.0;
    double ub = 0.0;
    if (t->kind == VAR_BOOL) {
        int v = t->idx;
        if (v >= 1 && v <= m->B && assign[v] >= 0) {
            lb = ub = (double)assign[v];
        } else {
            lb = m->b_lb_input[t->idx];
            ub = m->b_ub_input[t->idx];
        }
    } else if (t->kind == VAR_INT) {
        lb = m->i_lb[t->idx];
        ub = m->i_ub[t->idx];
    } else {
        lb = m->r_lb[t->idx];
        ub = m->r_ub[t->idx];
    }
    if (t->coef >= 0.0) {
        *lhs_min += t->coef * lb;
        *lhs_max += t->coef * ub;
    } else {
        *lhs_min += t->coef * ub;
        *lhs_max += t->coef * lb;
    }
}

static void explain_print_infeasibility_details(const Solver *s) {
    int8_t *assign;
    int clause_conflict = -1;
    int reported_rows = 0;
    int i;

    assign = (int8_t *)malloc((size_t)s->model.nb_total + 1U);
    if (assign == NULL) {
        printf("x explain status=INFEASIBLE mode=diagnostic note=allocation-failed\n");
        return;
    }
    for (i = 0; i <= s->model.nb_total; ++i) {
        assign[i] = -1;
    }
    for (i = 1; i <= s->model.B; ++i) {
        if (s->model.b_lb_input[i] >= 1.0 - s->opt.feas_tol) {
            assign[i] = 1;
        } else if (s->model.b_ub_input[i] <= s->opt.feas_tol) {
            assign[i] = 0;
        }
    }

    clause_conflict = propagate_clauses(NULL, &s->model, assign);
    printf("x explain status=INFEASIBLE mode=diagnostic bool_conflicts=%lld theory_conflicts=%lld\n",
           s->bool_conflicts,
           s->theory_conflicts);

    if (clause_conflict >= 0 && clause_conflict < s->model.clauses.size) {
        const Clause *c = &s->model.clauses.data[clause_conflict];
        printf("x infeasible_clause index=%d lits=\"", clause_conflict);
        explain_print_clause_compact(c);
        printf("\"\n");
    }

    for (i = 0; i < s->model.lin.size && reported_rows < 6; ++i) {
        const LinConstraint *lc = &s->model.lin.data[i];
        double lhs_min = 0.0;
        double lhs_max = 0.0;
        bool active = true;
        int j;
        if (lc->indicator_lit != 0) {
            int ev = explain_eval_fixed_lit(lc->indicator_lit, assign);
            if (ev == 0) {
                active = false;
            } else if (ev < 0) {
                active = false;
            }
        }
        if (!active) {
            continue;
        }
        for (j = 0; j < lc->nterms; ++j) {
            explain_interval_update(&lc->terms[j], &s->model, assign, &lhs_min, &lhs_max);
        }
        if (lhs_min > lc->rhs + fmax(s->opt.feas_tol, 1e-7)) {
            printf("x infeasible_row name=%s lhs_min=%.12g rhs=%.12g\n",
                   lc->name,
                   lhs_min,
                   lc->rhs);
            ++reported_rows;
        }
    }

    if (clause_conflict < 0 && reported_rows == 0) {
        printf("x infeasibility_note no-root-certificate-found use-trace-and-core-for-more-context\n");
    }
    free(assign);
}

static void explain_print_unbounded_details(const Solver *s) {
    int shown = 0;
    int i;
    printf("x explain status=UNBOUNDED mode=diagnostic\n");
    for (i = 1; i <= s->model.B && shown < 6; ++i) {
        double c = s->model.obj_b_input[i];
        if ((!s->model.obj_is_max && c < 0.0 && s->model.b_ub_input[i] >= 1.0 - s->opt.feas_tol) ||
            (s->model.obj_is_max && c > 0.0 && s->model.b_ub_input[i] >= 1.0 - s->opt.feas_tol)) {
            printf("x unbounded_candidate var=b%d coef=%.12g bounds=[%.12g,%.12g]\n",
                   i, c, s->model.b_lb_input[i], s->model.b_ub_input[i]);
            ++shown;
        }
    }
    for (i = 1; i <= s->model.I && shown < 6; ++i) {
        double c = s->model.obj_i[i];
        if ((!s->model.obj_is_max && c < 0.0 && s->model.i_ub[i] >= WMIBO_INF / 2) ||
            (!s->model.obj_is_max && c > 0.0 && s->model.i_lb[i] <= -WMIBO_INF / 2) ||
            (s->model.obj_is_max && c > 0.0 && s->model.i_ub[i] >= WMIBO_INF / 2) ||
            (s->model.obj_is_max && c < 0.0 && s->model.i_lb[i] <= -WMIBO_INF / 2)) {
            printf("x unbounded_candidate var=i%d coef=%.12g bounds=[%.12g,%.12g]\n",
                   i, c, s->model.i_lb[i], s->model.i_ub[i]);
            ++shown;
        }
    }
    for (i = 1; i <= s->model.R && shown < 6; ++i) {
        double c = s->model.obj_r[i];
        if ((!s->model.obj_is_max && c < 0.0 && s->model.r_ub[i] >= WMIBO_INF / 2) ||
            (!s->model.obj_is_max && c > 0.0 && s->model.r_lb[i] <= -WMIBO_INF / 2) ||
            (s->model.obj_is_max && c > 0.0 && s->model.r_ub[i] >= WMIBO_INF / 2) ||
            (s->model.obj_is_max && c < 0.0 && s->model.r_lb[i] <= -WMIBO_INF / 2)) {
            printf("x unbounded_candidate var=r%d coef=%.12g bounds=[%.12g,%.12g]\n",
                   i, c, s->model.r_lb[i], s->model.r_ub[i]);
            ++shown;
        }
    }
    if (shown == 0) {
        printf("x unbounded_note no-single-variable-candidate-identified\n");
    }
}

static void print_explain_report(const Solver *s, SolveStatus st) {
    explain_print_structure_summary(s);
    if (s->have_incumbent) {
        explain_print_solution_details(s, st);
        return;
    }
    if (st == SOLVE_STATUS_INFEASIBLE) {
        explain_print_infeasibility_details(s);
        return;
    }
    if (st == SOLVE_STATUS_UNBOUNDED) {
        explain_print_unbounded_details(s);
        return;
    }
    printf("x explain status=%s mode=diagnostic note=no-incumbent-available\n", status_to_text(st));
}

static bool write_trace_jsonl_file(const char *path, const Solver *s, SolveStatus st) {
    FILE *fp;
    if (path == NULL || path[0] == '\0') return true;
    fp = fopen(path, "wb");
    if (fp == NULL) return false;
    fprintf(fp,
            "{\"event\":\"summary\",\"status\":\"%s\",\"nodes\":%lld,\"lp_calls\":%lld,"
            "\"learnt_clauses\":%lld,\"bool_conflicts\":%lld,\"theory_conflicts\":%lld}\n",
            status_to_text(st),
            s->nodes,
            s->lp_calls,
            s->clause_learned,
            s->bool_conflicts,
            s->theory_conflicts);
    if (s->have_incumbent) {
        fprintf(fp,
                "{\"event\":\"incumbent\",\"objective\":%.17g}\n",
                s->report_incumbent);
    }
    fclose(fp);
    return true;
}

static bool write_core_json_file(const char *path, SolveStatus st) {
    FILE *fp;
    if (path == NULL || path[0] == '\0') return true;
    fp = fopen(path, "wb");
    if (fp == NULL) return false;
    fprintf(fp,
            "{\"status\":\"%s\",\"hard_core\":[],\"note\":\"native core extraction not available in this build\"}\n",
            status_to_text(st));
    fclose(fp);
    return true;
}

static bool run_one_file(const char *path, const SolveOptions *cli_opt, bool print_stats, SolveStatus *out_status, double *out_obj) {
    Model m;
    Solver s;
    SolveStatus st;
    char err[512];
    double started = now_seconds();
    double deadline = (cli_opt->time_limit > 0.0) ? (started + cli_opt->time_limit) : 0.0;

    model_init(&m);
    if (!load_model_from_file(path, &m, deadline, err, sizeof(err))) {
        if (strncmp(err, "time limit exceeded", 19) == 0) {
            printf("s UNKNOWN\n");
            printf("o nan\n");
            if (out_status != NULL) *out_status = SOLVE_STATUS_UNKNOWN;
            if (out_obj != NULL) *out_obj = NAN;
            model_free(&m);
            return true;
        }
        fprintf(stderr, "ERROR: %s\n", err);
        model_free(&m);
        return false;
    }

    memset(&s, 0, sizeof(s));
    s.model = m;
    s.opt = *cli_opt;
    s.start_time = started;
    solver_apply_file_options(&s);

    st = run_solver(&s);

    if (st == SOLVE_STATUS_OPTIMUM) {
        printf("s OPTIMUM\n");
        printf("o %.17g\n", s.report_incumbent);
        print_solution_line(&s);
        if (out_obj != NULL) *out_obj = s.report_incumbent;
    } else if (st == SOLVE_STATUS_INFEASIBLE) {
        printf("s INFEASIBLE\n");
        if (out_obj != NULL) *out_obj = NAN;
    } else if (st == SOLVE_STATUS_UNBOUNDED) {
        printf("s UNBOUNDED\n");
        if (out_obj != NULL) *out_obj = -INFINITY;
    } else {
        printf("s UNKNOWN\n");
        if (s.have_incumbent) {
            printf("o %.17g\n", s.report_incumbent);
            if (out_obj != NULL) *out_obj = s.report_incumbent;
        } else {
            printf("o nan\n");
            if (out_obj != NULL) *out_obj = NAN;
        }
    }

    if (print_stats && s.opt.verbose >= 2) {
        printf("c nodes %lld\n", s.nodes);
        printf("c lp_calls %lld\n", s.lp_calls);
        printf("c learnt_clauses %lld\n", s.clause_learned);
        printf("c bool_conflicts %lld\n", s.bool_conflicts);
        printf("c theory_conflicts %lld\n", s.theory_conflicts);
    }

    if (s.opt.mode == MODE_EXPLAIN) {
        print_explain_report(&s, st);
    }

    if (!write_trace_jsonl_file(s.opt.trace_out, &s, st)) {
        fprintf(stderr, "c warning: failed to write trace file '%s'\n", s.opt.trace_out);
    }
    if (!write_core_json_file(s.opt.core_out, st)) {
        fprintf(stderr, "c warning: failed to write core file '%s'\n", s.opt.core_out);
    }

    if (out_status != NULL) *out_status = st;

    solver_cleanup_results(&s);
    model_free(&s.model);
    return true;
}

static bool write_text_file(const char *path, const char *text) {
    FILE *fp = fopen(path, "wb");
    size_t n;
    if (fp == NULL) return false;
    n = strlen(text);
    if (fwrite(text, 1U, n, fp) != n) {
        fclose(fp);
        return false;
    }
    fclose(fp);
    return true;
}

static bool selftest_case(const char *name, const char *text, SolveStatus exp_status, double exp_obj, double tol, const SolveOptions *opt) {
    char tmp_name[256];
    SolveStatus st = SOLVE_STATUS_UNKNOWN;
    double obj = NAN;
    bool ok;
    unsigned long stamp = (unsigned long)time(NULL);
    snprintf(tmp_name, sizeof(tmp_name), "wmibo_selftest_%s_%lu.wmibo", name, stamp);

    if (!write_text_file(tmp_name, text)) {
        fprintf(stderr, "selftest[%s]: failed to write temp file\n", name);
        return false;
    }

    ok = run_one_file(tmp_name, opt, false, &st, &obj);
    remove(tmp_name);

    if (!ok) {
        fprintf(stderr, "selftest[%s]: solver failed\n", name);
        return false;
    }
    if (st != exp_status) {
        fprintf(stderr, "selftest[%s]: status mismatch (got %d expected %d)\n", name, (int)st, (int)exp_status);
        return false;
    }
    if (exp_status == SOLVE_STATUS_OPTIMUM && fabs(obj - exp_obj) > tol) {
        fprintf(stderr, "selftest[%s]: objective mismatch (got %.12g expected %.12g)\n", name, obj, exp_obj);
        return false;
    }
    return true;
}

static bool run_selftests(const SolveOptions *opt) {
    const char *sat_case =
        "p wmibo 2 0 0 2 0 0\n"
        "begin cnf\n"
        "cl hard b1 b2 0\n"
        "cl hard ~b1 b2 0\n"
        "end\n";

    const char *wmax_case =
        "p wmibo 1 0 0 1 0 0\n"
        "begin wcnf\n"
        "wcl 5 soft b1 0\n"
        "end\n";

    const char *mip_case =
        "p wmibo 0 1 1 0 2 0\n"
        "var i 1 [0,3]\n"
        "var r 1 [0,10]\n"
        "begin lin\n"
        "lc C1 >= 1 : 1 r1\n"
        "lc C2 >= 0 : 1 r1 -1 i1\n"
        "end\n"
        "begin obj\n"
        "obj min : lin 1 r1 2 i1\n"
        "end\n";

    const char *ind_case =
        "p wmibo 1 0 1 0 1 1\n"
        "var r 1 [0,10]\n"
        "begin lin\n"
        "lc C1 <= 2 : 1 r1\n"
        "end\n"
        "begin ind\n"
        "ind b1 => C1\n"
        "end\n"
        "begin obj\n"
        "obj min : lin 1 r1\n"
        "end\n"
        "begin cnf\n"
        "cl hard b1 0\n"
        "end\n";
    SolveOptions t = *opt;
    t.verbose = 0;
    t.time_limit = 0.0;
    t.node_limit = 0;

    if (!selftest_case("sat", sat_case, SOLVE_STATUS_OPTIMUM, 0.0, 1e-8, &t)) return false;
    if (!selftest_case("wmaxsat", wmax_case, SOLVE_STATUS_OPTIMUM, 0.0, 1e-8, &t)) return false;
    if (!selftest_case("mip", mip_case, SOLVE_STATUS_OPTIMUM, 1.0, 1e-7, &t)) return false;
    if (!selftest_case("indicator", ind_case, SOLVE_STATUS_OPTIMUM, 0.0, 1e-8, &t)) return false;

    printf("selftest: OK\n");
    return true;
}

static void print_usage(FILE *out) {
    fprintf(out,
            "Usage: %s <file> [options]\n"
            "       %s --selftest [options]\n"
            "\n"
            "Options:\n"
            "  --parallel <mode>  auto|off|threads|mpi|hybrid\n"
            "  --jobs <n>         local worker count for threaded modes\n"
            "  --split-depth <n>  parallel splitting depth scaffold\n"
            "  --portfolio <n>    portfolio multiplicity scaffold\n"
            "  --sync-ms <n>      synchronization cadence scaffold\n"
            "  --time <sec>       Time limit in seconds\n"
            "  --node <n>         Node limit\n"
            "  --gap <g>          Relative gap stop criterion\n"
            "  --seed <s>         RNG seed (enables tie randomization)\n"
            "  --verbose <0..3>   Verbosity\n"
            "  --mode <name>      solve|count|project|volume|explain\n"
            "  --cuda <mode>      auto|on|off (on requires CUDA dense LP backend)\n"
            "  --cuda-device <n>  Select CUDA device id (-1 uses runtime default)\n"
            "  --cuda-min-cells <n> minimum dense LP cells before trying CUDA\n"
            "  --trace-out <path> Write JSONL trace events (optional)\n"
            "  --core-out <path>  Write UNSAT core JSON scaffold (optional)\n"
            "  --selftest         Run built-in tests\n",
            g_prog_name,
            g_prog_name);
}

static bool parse_mode(const char *s, QueryMode *out) {
    if (str_ieq(s, "solve")) { *out = MODE_SOLVE; return true; }
    if (str_ieq(s, "count")) { *out = MODE_COUNT; return true; }
    if (str_ieq(s, "project")) { *out = MODE_PROJECT; return true; }
    if (str_ieq(s, "volume")) { *out = MODE_VOLUME; return true; }
    if (str_ieq(s, "explain")) { *out = MODE_EXPLAIN; return true; }
    return false;
}

int wmibo_entry(int argc, char **argv) {
    SolveOptions opt;
    KrbParallelRuntime parallel_rt;
    const char *file = NULL;
    bool selftest = false;
    int i;

    g_prog_name = (argc > 0 && argv[0] != NULL) ? argv[0] : "wmibo";
    solver_defaults(&opt);

    for (i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "--selftest") == 0) {
            selftest = true;
        } else if (strcmp(a, "--parallel") == 0) {
            if (i + 1 >= argc || !krb_parallel_parse_mode(argv[++i], &opt.parallel.mode)) {
                fprintf(stderr, "ERROR: invalid --parallel\n");
                return 2;
            }
        } else if (strcmp(a, "--jobs") == 0) {
            long long jobs;
            if (i + 1 >= argc || !parse_ll(argv[++i], &jobs) || jobs < 1 || jobs > INT_MAX) {
                fprintf(stderr, "ERROR: invalid --jobs\n");
                return 2;
            }
            opt.parallel.jobs = (int)jobs;
        } else if (strcmp(a, "--split-depth") == 0) {
            long long depth;
            if (i + 1 >= argc || !parse_ll(argv[++i], &depth) || depth < 0 || depth > INT_MAX) {
                fprintf(stderr, "ERROR: invalid --split-depth\n");
                return 2;
            }
            opt.parallel.split_depth = (int)depth;
        } else if (strcmp(a, "--portfolio") == 0) {
            long long portfolio;
            if (i + 1 >= argc || !parse_ll(argv[++i], &portfolio) || portfolio < 1 || portfolio > INT_MAX) {
                fprintf(stderr, "ERROR: invalid --portfolio\n");
                return 2;
            }
            opt.parallel.portfolio = (int)portfolio;
        } else if (strcmp(a, "--sync-ms") == 0) {
            long long sync_ms;
            if (i + 1 >= argc || !parse_ll(argv[++i], &sync_ms) || sync_ms < 0 || sync_ms > INT_MAX) {
                fprintf(stderr, "ERROR: invalid --sync-ms\n");
                return 2;
            }
            opt.parallel.sync_ms = (int)sync_ms;
        } else if (strcmp(a, "--time") == 0) {
            if (i + 1 >= argc || !parse_double_str(argv[++i], &opt.time_limit)) {
                fprintf(stderr, "ERROR: invalid --time\n");
                return 2;
            }
        } else if (strcmp(a, "--node") == 0) {
            long long n;
            if (i + 1 >= argc || !parse_ll(argv[++i], &n) || n < 0) {
                fprintf(stderr, "ERROR: invalid --node\n");
                return 2;
            }
            opt.node_limit = n;
        } else if (strcmp(a, "--gap") == 0) {
            if (i + 1 >= argc || !parse_double_str(argv[++i], &opt.rel_gap) || opt.rel_gap < 0.0) {
                fprintf(stderr, "ERROR: invalid --gap\n");
                return 2;
            }
        } else if (strcmp(a, "--seed") == 0) {
            if (i + 1 >= argc || !parse_u64(argv[++i], &opt.seed)) {
                fprintf(stderr, "ERROR: invalid --seed\n");
                return 2;
            }
            opt.seed_set = true;
        } else if (strcmp(a, "--verbose") == 0) {
            long long v;
            if (i + 1 >= argc || !parse_ll(argv[++i], &v) || v < 0 || v > 3) {
                fprintf(stderr, "ERROR: invalid --verbose\n");
                return 2;
            }
            opt.verbose = (int)v;
        } else if (strcmp(a, "--cuda") == 0) {
            if (i + 1 >= argc || !krb_accel_parse_mode(argv[++i], &opt.accel.mode)) {
                fprintf(stderr, "ERROR: invalid --cuda\n");
                return 2;
            }
        } else if (strcmp(a, "--cuda-device") == 0) {
            long long dev;
            if (i + 1 >= argc || !parse_ll(argv[++i], &dev) || dev < -1 || dev > INT_MAX) {
                fprintf(stderr, "ERROR: invalid --cuda-device\n");
                return 2;
            }
            opt.accel.cuda_device = (int)dev;
        } else if (strcmp(a, "--cuda-min-cells") == 0) {
            long long cells;
            if (i + 1 >= argc || !parse_ll(argv[++i], &cells) || cells < 0) {
                fprintf(stderr, "ERROR: invalid --cuda-min-cells\n");
                return 2;
            }
            opt.accel.cuda_min_cells = (size_t)cells;
        } else if (strcmp(a, "--mode") == 0) {
            if (i + 1 >= argc || !parse_mode(argv[++i], &opt.mode)) {
                fprintf(stderr, "ERROR: invalid --mode\n");
                return 2;
            }
        } else if (strcmp(a, "--trace-out") == 0) {
            if (i + 1 >= argc || argv[i + 1][0] == '\0') {
                fprintf(stderr, "ERROR: invalid --trace-out\n");
                return 2;
            }
            opt.trace_out = argv[++i];
        } else if (strcmp(a, "--core-out") == 0) {
            if (i + 1 >= argc || argv[i + 1][0] == '\0') {
                fprintf(stderr, "ERROR: invalid --core-out\n");
                return 2;
            }
            opt.core_out = argv[++i];
        } else if (a[0] == '-') {
            fprintf(stderr, "ERROR: unknown option '%s'\n", a);
            return 2;
        } else {
            if (file != NULL) {
                fprintf(stderr, "ERROR: multiple input files provided\n");
                return 2;
            }
            file = a;
        }
    }

    if (!krb_parallel_runtime_resolve(&opt.parallel, &parallel_rt, NULL, 0)) {
        fprintf(stderr, "ERROR: unsupported parallel configuration\n");
        return 2;
    }
    if (parallel_rt.resolved_mode != KRB_PARALLEL_MODE_OFF && opt.verbose >= 1) {
        fprintf(stderr,
                "c wmibo parallel mode %s requested; backend remains serial in this build path\n",
                krb_parallel_mode_name(parallel_rt.resolved_mode));
    }

    if (selftest) {
        if (!run_selftests(&opt)) {
            return 1;
        }
        return 0;
    }

    if (opt.mode != MODE_SOLVE && opt.mode != MODE_EXPLAIN) {
        printf("s UNKNOWN\n");
        printf("o nan\n");
        if (opt.verbose >= 1) {
            if (opt.mode == MODE_COUNT) fprintf(stderr, "c COUNT mode scaffold: not implemented\n");
            else if (opt.mode == MODE_PROJECT) fprintf(stderr, "c PROJECT mode scaffold: not implemented\n");
            else if (opt.mode == MODE_VOLUME) fprintf(stderr, "c VOLUME mode scaffold: not implemented\n");
            else fprintf(stderr, "c EXPLAIN mode scaffold: not implemented\n");
        }
        return 0;
    }

    if (file == NULL) {
        print_usage(stderr);
        return 2;
    }

    if (!run_one_file(file, &opt, true, NULL, NULL)) {
        return 1;
    }

    return 0;
}

#ifndef WMIBO_NO_MAIN
int main(int argc, char **argv) {
    return wmibo_entry(argc, argv);
}
#endif
