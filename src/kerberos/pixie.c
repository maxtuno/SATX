/*
 * Copyright (c) 2026 Oscar Riveros.
 *
 * Licencia dual: uso personal bajo Apache License 2.0; portes a otros
 * lenguajes requieren licencia comercial con autorizacion expresa del autor.
 * Ver LICENSE.txt en la raiz del proyecto para los terminos completos.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

/*
Description:
PIXIE is a compact mixed-integer linear programming solver implemented in one ISO C17 file.
It includes an LP/MPS reader, a two-phase primal simplex core, and a deterministic branch-and-bound layer for integer and binary variables.
This port preserves the original solver behavior while removing non-standard dependencies and keeping the implementation portable and self-contained.

Copyright (c) 2026 Oscar Riveros.
All rights reserved.

This source code and associated materials are proprietary.
Unauthorized use, distribution, or modification is prohibited
without explicit written permission from the copyright holder.

 gcc -O3 -std=c17 -Wall -Wextra -pedantic pixie.c krb_accel.c krb_accel_cuda_stub.c -o pixie
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

#if defined(SATX_HAVE_THREADS)
#include <stdatomic.h>
#endif

#include "krb_accel.h"
#include "krb_parallel.h"

#if defined(_WIN32)
#include <windows.h>
#define pixie_popen _popen
#define pixie_pclose _pclose
#else
#define pixie_popen popen
#define pixie_pclose pclose
#endif

#define PIXIE_FEAS_TOL 1e-8
#define PIXIE_INT_TOL 1e-7
#define PIXIE_PIV_TOL 1e-10
#define PIXIE_EPS 1e-12
#define PIXIE_LP_EPS 1e-9
#define PIXIE_CERT_TOL 1e-6
#define PIXIE_PRICING_BLOCK 128

typedef enum {
    PIXIE_CMP_LE = -1,
    PIXIE_CMP_EQ = 0,
    PIXIE_CMP_GE = 1
} PixieCompare;

typedef enum {
    PIXIE_VAR_CONT = 0,
    PIXIE_VAR_INT = 1,
    PIXIE_VAR_BIN = 2
} PixieVarType;

typedef enum {
    PIXIE_STATUS_OPTIMAL = 0,
    PIXIE_STATUS_INFEASIBLE = 1,
    PIXIE_STATUS_UNBOUNDED = 2,
    PIXIE_STATUS_UNKNOWN = 3,
    PIXIE_STATUS_ERROR = 4
} PixieStatus;

typedef enum {
    PIXIE_FORMAT_AUTO = 0,
    PIXIE_FORMAT_LP = 1,
    PIXIE_FORMAT_MPS = 2
} PixieFormat;

typedef struct {
    int *idx;
    double *val;
    int len;
    int cap;
} PixieSparse;

typedef struct {
    PixieSparse a;
    PixieCompare cmp;
    double rhs;
} PixieConstraint;

typedef struct {
    char *name;
    double obj;
    double lb;
    double ub;
    PixieVarType type;
    bool has_explicit_lb;
    bool has_explicit_ub;
} PixieVar;

typedef struct {
    PixieVar *vars;
    int n_vars;
    int cap_vars;
    PixieConstraint *cons;
    int n_cons;
    int cap_cons;
    int obj_sense; /* +1 minimize, -1 maximize */
} PixieModel;

typedef struct {
    PixieFormat format;
    const char *file;
    double time_limit_sec;   /* <= 0 means unlimited */
    long long node_limit;    /* <= 0 means unlimited */
    double gap_limit;        /* < 0 means disabled */
    uint64_t seed;
    bool seed_set;
    int verbose;             /* 0..3 */
    bool pure_lp;
    bool selftest;
    KrbAccelConfig accel;
    KrbParallelConfig parallel;
} PixieOptions;

typedef struct {
    PixieStatus status;
    double obj_min;
    double obj_out;
    double *x;
    int n;
    bool has_primal;
    long long nodes_processed;
    bool stopped_time;
    bool stopped_nodes;
    bool stopped_gap;
    bool saw_unbounded_relax;
} PixieSolution;

typedef struct {
    double **table;
    double *raw;
    int *v_idx;
    bool *a_flg;
    int v_cnt;
    int e_cnt;
    int s_cnt;
    int a_cnt;
} PixieSimplex;

typedef enum {
    PIXIE_SIMPLEX_OPTIMAL = 0,
    PIXIE_SIMPLEX_UNBOUNDED = 1,
    PIXIE_SIMPLEX_NUMERIC = 2
} PixieSimplexStatus;

typedef enum {
    PIXIE_LP_OPTIMAL = 0,
    PIXIE_LP_INFEASIBLE = 1,
    PIXIE_LP_UNBOUNDED = 2,
    PIXIE_LP_ABORTED = 3,
    PIXIE_LP_ERROR = 4
} PixieLPStatus;

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
} PixieLPState;

typedef struct {
    PixieConstraint *data;
    int len;
    int cap;
} PixieConVec;

typedef struct {
    double *lb;
    double *ub;
    double bound;
    int depth;
} PixieNode;

typedef struct {
    PixieNode *data;
    int len;
    int cap;
} PixieNodeStack;

typedef struct {
    PixieNodeStack tasks;
    PixieSolution seed_solution;
    bool timed_out;
    bool saw_unbounded_relax;
} PixieFrontierSplit;

typedef struct {
    const PixieModel *model;
    const PixieOptions *opt;
    double deadline;
    PixieNode *tasks;
    int task_count;
    int nvars;
    const PixieSolution *seed_solution;
#if defined(SATX_HAVE_THREADS)
    atomic_int next_task;
#endif
    PixieSolution *results;
} PixieParallelWork;

typedef struct {
    char **data;
    int len;
    int cap;
} PixieTokVec;

typedef enum {
    PIXIE_MAP_SHIFT_LB = 0,
    PIXIE_MAP_SHIFT_UB = 1,
    PIXIE_MAP_FREE = 2
} PixieVarMapKind;

typedef struct {
    PixieVarMapKind kind;
    int y0;
    int y1;
    double constant;
} PixieVarMap;

static void pixie_options_defaults(PixieOptions *opt) {
    if (opt == NULL) {
        return;
    }
    memset(opt, 0, sizeof(*opt));
    opt->format = PIXIE_FORMAT_AUTO;
    opt->file = NULL;
    opt->time_limit_sec = 0.0;
    opt->node_limit = 0;
    opt->gap_limit = -1.0;
    opt->seed = 0;
    opt->seed_set = false;
    opt->verbose = 0;
    opt->pure_lp = false;
    opt->selftest = false;
    krb_accel_config_defaults(&opt->accel);
    krb_parallel_config_defaults(&opt->parallel);
}

static void pixie_set_error(char *err, size_t errsz, const char *msg) {
    if (err != NULL && errsz > 0) {
        size_t n = strlen(msg);
        if (n >= errsz) {
            n = errsz - 1;
        }
        memcpy(err, msg, n);
        err[n] = '\0';
    }
}

static bool pixie_deadline_reached(double deadline);

static void *pixie_xmalloc(size_t n) {
    void *p = malloc(n == 0 ? 1 : n);
    if (p == NULL) {
        fprintf(stderr, "fatal: out of memory\n");
        exit(EXIT_FAILURE);
    }
    return p;
}

static void *pixie_xcalloc(size_t count, size_t size) {
    void *p = calloc(count == 0 ? 1 : count, size == 0 ? 1 : size);
    if (p == NULL) {
        fprintf(stderr, "fatal: out of memory\n");
        exit(EXIT_FAILURE);
    }
    return p;
}

static void *pixie_xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n == 0 ? 1 : n);
    if (p == NULL) {
        fprintf(stderr, "fatal: out of memory\n");
        exit(EXIT_FAILURE);
    }
    return p;
}

static char *pixie_strdup_c(const char *s) {
    size_t n = strlen(s);
    char *d = (char *)pixie_xmalloc(n + 1);
    memcpy(d, s, n + 1);
    return d;
}

static int pixie_stricmp_c(const char *a, const char *b) {
    unsigned char ca;
    unsigned char cb;
    while (*a != '\0' && *b != '\0') {
        ca = (unsigned char)tolower((unsigned char)*a);
        cb = (unsigned char)tolower((unsigned char)*b);
        if (ca != cb) {
            return (int)ca - (int)cb;
        }
        ++a;
        ++b;
    }
    ca = (unsigned char)tolower((unsigned char)*a);
    cb = (unsigned char)tolower((unsigned char)*b);
    return (int)ca - (int)cb;
}

static bool pixie_starts_with_ci(const char *s, const char *prefix) {
    while (*prefix != '\0') {
        if (*s == '\0') {
            return false;
        }
        if (tolower((unsigned char)*s) != tolower((unsigned char)*prefix)) {
            return false;
        }
        ++s;
        ++prefix;
    }
    return true;
}

static bool pixie_ends_with_ci(const char *s, const char *suffix) {
    size_t n = strlen(s);
    size_t m = strlen(suffix);
    if (m > n) {
        return false;
    }
    return pixie_stricmp_c(s + (n - m), suffix) == 0;
}

static char *pixie_ltrim(char *s) {
    while (*s != '\0' && isspace((unsigned char)*s)) {
        ++s;
    }
    return s;
}

static void pixie_rtrim(char *s) {
    size_t n = strlen(s);
    while (n > 0) {
        if (!isspace((unsigned char)s[n - 1])) {
            break;
        }
        s[n - 1] = '\0';
        --n;
    }
}

static char *pixie_trim(char *s) {
    char *t = pixie_ltrim(s);
    pixie_rtrim(t);
    return t;
}

static bool pixie_is_finite(double x) {
    return isfinite(x) != 0;
}

static double pixie_max(double a, double b) {
    return (a > b) ? a : b;
}

static double pixie_min(double a, double b) {
    return (a < b) ? a : b;
}

static bool pixie_parse_double_token(const char *s, double *out) {
    char *end = NULL;
    double v;
    errno = 0;
    v = strtod(s, &end);
    if (end == s) {
        return false;
    }
    while (*end != '\0' && isspace((unsigned char)*end)) {
        ++end;
    }
    if (*end != '\0') {
        return false;
    }
    if (errno == ERANGE) {
        return false;
    }
    *out = v;
    return true;
}

static bool pixie_parse_ll_token(const char *s, long long *out) {
    char *end = NULL;
    long long v;
    errno = 0;
    v = strtoll(s, &end, 10);
    if (end == s) {
        return false;
    }
    while (*end != '\0' && isspace((unsigned char)*end)) {
        ++end;
    }
    if (*end != '\0') {
        return false;
    }
    if (errno == ERANGE) {
        return false;
    }
    *out = v;
    return true;
}

static bool pixie_parse_u64_token(const char *s, uint64_t *out) {
    char *end = NULL;
    unsigned long long v;
    errno = 0;
    v = strtoull(s, &end, 10);
    if (end == s) {
        return false;
    }
    while (*end != '\0' && isspace((unsigned char)*end)) {
        ++end;
    }
    if (*end != '\0') {
        return false;
    }
    if (errno == ERANGE) {
        return false;
    }
    *out = (uint64_t)v;
    return true;
}

static uint64_t pixie_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) {
        x = UINT64_C(0x9e3779b97f4a7c15);
    }
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static double pixie_wall_seconds(void) {
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

static void pixie_sparse_init(PixieSparse *s) {
    s->idx = NULL;
    s->val = NULL;
    s->len = 0;
    s->cap = 0;
}

static void pixie_sparse_free(PixieSparse *s) {
    free(s->idx);
    free(s->val);
    s->idx = NULL;
    s->val = NULL;
    s->len = 0;
    s->cap = 0;
}

static void pixie_sparse_reserve(PixieSparse *s, int need) {
    int nc;
    if (need <= s->cap) {
        return;
    }
    nc = (s->cap <= 0) ? 8 : s->cap;
    while (nc < need) {
        if (nc > INT_MAX / 2) {
            nc = need;
            break;
        }
        nc *= 2;
    }
    s->idx = (int *)pixie_xrealloc(s->idx, (size_t)nc * sizeof(int));
    s->val = (double *)pixie_xrealloc(s->val, (size_t)nc * sizeof(double));
    s->cap = nc;
}

static void pixie_sparse_add(PixieSparse *s, int idx, double val) {
    int i;
    if (fabs(val) <= PIXIE_EPS) {
        return;
    }
    for (i = 0; i < s->len; ++i) {
        if (s->idx[i] == idx) {
            s->val[i] += val;
            return;
        }
    }
    pixie_sparse_reserve(s, s->len + 1);
    s->idx[s->len] = idx;
    s->val[s->len] = val;
    ++s->len;
}

static void pixie_sparse_prune(PixieSparse *s, double tol) {
    int i;
    int w = 0;
    for (i = 0; i < s->len; ++i) {
        if (fabs(s->val[i]) > tol) {
            s->idx[w] = s->idx[i];
            s->val[w] = s->val[i];
            ++w;
        }
    }
    s->len = w;
}

static void pixie_sparse_move(PixieSparse *dst, PixieSparse *src) {
    dst->idx = src->idx;
    dst->val = src->val;
    dst->len = src->len;
    dst->cap = src->cap;
    src->idx = NULL;
    src->val = NULL;
    src->len = 0;
    src->cap = 0;
}

static void pixie_sparse_clone(PixieSparse *dst, const PixieSparse *src) {
    int i;
    pixie_sparse_init(dst);
    if (src->len <= 0) {
        return;
    }
    pixie_sparse_reserve(dst, src->len);
    for (i = 0; i < src->len; ++i) {
        dst->idx[i] = src->idx[i];
        dst->val[i] = src->val[i];
    }
    dst->len = src->len;
}

static void pixie_model_init(PixieModel *m) {
    m->vars = NULL;
    m->n_vars = 0;
    m->cap_vars = 0;
    m->cons = NULL;
    m->n_cons = 0;
    m->cap_cons = 0;
    m->obj_sense = +1;
}

static void pixie_model_free(PixieModel *m) {
    int i;
    for (i = 0; i < m->n_vars; ++i) {
        free(m->vars[i].name);
    }
    for (i = 0; i < m->n_cons; ++i) {
        pixie_sparse_free(&m->cons[i].a);
    }
    free(m->vars);
    free(m->cons);
    m->vars = NULL;
    m->cons = NULL;
    m->n_vars = 0;
    m->cap_vars = 0;
    m->n_cons = 0;
    m->cap_cons = 0;
    m->obj_sense = +1;
}

static int pixie_model_find_var(const PixieModel *m, const char *name) {
    int i;
    for (i = 0; i < m->n_vars; ++i) {
        if (strcmp(m->vars[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

static int pixie_model_add_var(PixieModel *m, const char *name) {
    int nc;
    PixieVar *v;
    if (m->n_vars >= m->cap_vars) {
        nc = (m->cap_vars <= 0) ? 8 : m->cap_vars;
        while (nc <= m->n_vars) {
            if (nc > INT_MAX / 2) {
                nc = m->n_vars + 1;
                break;
            }
            nc *= 2;
        }
        m->vars = (PixieVar *)pixie_xrealloc(m->vars, (size_t)nc * sizeof(PixieVar));
        m->cap_vars = nc;
    }
    v = &m->vars[m->n_vars];
    v->name = pixie_strdup_c(name);
    v->obj = 0.0;
    v->lb = 0.0;
    v->ub = HUGE_VAL;
    v->type = PIXIE_VAR_CONT;
    v->has_explicit_lb = false;
    v->has_explicit_ub = false;
    ++m->n_vars;
    return m->n_vars - 1;
}

static int pixie_model_get_var(PixieModel *m, const char *name, bool create) {
    int idx = pixie_model_find_var(m, name);
    if (idx >= 0) {
        return idx;
    }
    if (!create) {
        return -1;
    }
    return pixie_model_add_var(m, name);
}

static int pixie_model_reserve_cons(PixieModel *m, int need) {
    int nc;
    if (need <= m->cap_cons) {
        return 1;
    }
    nc = (m->cap_cons <= 0) ? 8 : m->cap_cons;
    while (nc < need) {
        if (nc > INT_MAX / 2) {
            nc = need;
            break;
        }
        nc *= 2;
    }
    m->cons = (PixieConstraint *)pixie_xrealloc(m->cons, (size_t)nc * sizeof(PixieConstraint));
    m->cap_cons = nc;
    return 1;
}

static int pixie_model_add_empty_constraint(PixieModel *m, PixieCompare cmp, double rhs) {
    if (!pixie_model_reserve_cons(m, m->n_cons + 1)) {
        return -1;
    }
    pixie_sparse_init(&m->cons[m->n_cons].a);
    m->cons[m->n_cons].cmp = cmp;
    m->cons[m->n_cons].rhs = rhs;
    ++m->n_cons;
    return m->n_cons - 1;
}

static int pixie_model_add_constraint(PixieModel *m, const PixieSparse *a, PixieCompare cmp, double rhs) {
    int idx = pixie_model_add_empty_constraint(m, cmp, rhs);
    if (idx < 0) {
        return 0;
    }
    pixie_sparse_clone(&m->cons[idx].a, a);
    pixie_sparse_prune(&m->cons[idx].a, PIXIE_EPS);
    return 1;
}

static void pixie_var_mark_integer(PixieModel *m, int idx) {
    if (idx < 0 || idx >= m->n_vars) {
        return;
    }
    if (m->vars[idx].type == PIXIE_VAR_CONT) {
        m->vars[idx].type = PIXIE_VAR_INT;
    }
}

static void pixie_var_mark_binary(PixieModel *m, int idx) {
    if (idx < 0 || idx >= m->n_vars) {
        return;
    }
    m->vars[idx].type = PIXIE_VAR_BIN;
    m->vars[idx].lb = 0.0;
    m->vars[idx].ub = 1.0;
    m->vars[idx].has_explicit_lb = true;
    m->vars[idx].has_explicit_ub = true;
}

static int pixie_count_integer_vars(const PixieModel *m) {
    int i;
    int cnt = 0;
    for (i = 0; i < m->n_vars; ++i) {
        if (m->vars[i].type != PIXIE_VAR_CONT) {
            ++cnt;
        }
    }
    return cnt;
}

static bool pixie_is_name_start_char(char c) {
    return isalpha((unsigned char)c) || c == '_' || c == '.' || c == '$';
}

static bool pixie_is_name_char(char c) {
    return isalnum((unsigned char)c) || c == '_' || c == '.' || c == '$' || c == '[' || c == ']' || c == '#';
}

static int pixie_find_comparator(const char *s, int *pos, int *len, PixieCompare *cmp) {
    int i;
    int n = (int)strlen(s);
    for (i = 0; i < n; ++i) {
        if (s[i] == '<') {
            if (i + 1 < n && s[i + 1] == '=') {
                *pos = i;
                *len = 2;
                *cmp = PIXIE_CMP_LE;
                return 1;
            }
            *pos = i;
            *len = 1;
            *cmp = PIXIE_CMP_LE;
            return 1;
        }
        if (s[i] == '>') {
            if (i + 1 < n && s[i + 1] == '=') {
                *pos = i;
                *len = 2;
                *cmp = PIXIE_CMP_GE;
                return 1;
            }
            *pos = i;
            *len = 1;
            *cmp = PIXIE_CMP_GE;
            return 1;
        }
        if (s[i] == '=') {
            *pos = i;
            *len = 1;
            *cmp = PIXIE_CMP_EQ;
            return 1;
        }
    }
    return 0;
}

static int pixie_parse_linear_expr(PixieModel *m, const char *text, bool create_vars, PixieSparse *out_terms, double *out_const, char *err, size_t errsz) {
    const char *p = text;
    int sign = +1;
    pixie_sparse_init(out_terms);
    *out_const = 0.0;

    while (*p != '\0') {
        char *end_num = NULL;
        double coef = 0.0;
        bool has_num = false;

        while (*p != '\0' && isspace((unsigned char)*p)) {
            ++p;
        }
        if (*p == '\0') {
            break;
        }
        if (*p == '+') {
            sign = +1;
            ++p;
            continue;
        }
        if (*p == '-') {
            sign = -1;
            ++p;
            continue;
        }

        errno = 0;
        coef = strtod(p, &end_num);
        if (end_num != p) {
            has_num = true;
            if (errno == ERANGE) {
                pixie_sparse_free(out_terms);
                pixie_set_error(err, errsz, "numeric overflow while parsing expression");
                return 0;
            }
            p = end_num;
            while (*p != '\0' && isspace((unsigned char)*p)) {
                ++p;
            }
            if (*p == '*') {
                ++p;
                while (*p != '\0' && isspace((unsigned char)*p)) {
                    ++p;
                }
            }
        }

        if (pixie_is_name_start_char(*p)) {
            const char *start = p;
            char name_buf[256];
            size_t name_len = 0;
            int vidx;

            while (*p != '\0' && pixie_is_name_char(*p)) {
                ++p;
            }
            name_len = (size_t)(p - start);
            if (name_len == 0 || name_len >= sizeof(name_buf)) {
                pixie_sparse_free(out_terms);
                pixie_set_error(err, errsz, "invalid variable name in expression");
                return 0;
            }
            memcpy(name_buf, start, name_len);
            name_buf[name_len] = '\0';
            vidx = pixie_model_get_var(m, name_buf, create_vars);
            if (vidx < 0) {
                pixie_sparse_free(out_terms);
                pixie_set_error(err, errsz, "unknown variable in expression");
                return 0;
            }
            if (!has_num) {
                coef = 1.0;
            }
            pixie_sparse_add(out_terms, vidx, (double)sign * coef);
            sign = +1;
            continue;
        }

        if (has_num) {
            *out_const += (double)sign * coef;
            sign = +1;
            continue;
        }

        {
            char msg[256];
            snprintf(msg, sizeof(msg), "cannot parse linear expression near '%.80s'", p);
            pixie_sparse_free(out_terms);
            pixie_set_error(err, errsz, msg);
        }
        return 0;
    }

    pixie_sparse_prune(out_terms, PIXIE_EPS);
    return 1;
}

/* Portable fallback for systems without strtok_s. */
#ifndef _WIN32
#define strtok_s strtok_r
#endif

static int pixie_parse_lp_objective(PixieModel *m, char *stmt, char *err, size_t errsz) {
    PixieSparse terms;
    double cst = 0.0;
    char *p = stmt;
    char *colon = NULL;
    int i;

    p = pixie_trim(p);
    colon = strchr(p, ':');
    if (colon != NULL) {
        *colon = '\0';
        p = pixie_trim(colon + 1);
    }

    if (!pixie_parse_linear_expr(m, p, true, &terms, &cst, err, errsz)) {
        return 0;
    }
    (void)cst;
    for (i = 0; i < terms.len; ++i) {
        int v = terms.idx[i];
        m->vars[v].obj += terms.val[i];
    }
    pixie_sparse_free(&terms);
    return 1;
}

static int pixie_parse_lp_constraint(PixieModel *m, char *stmt, char *err, size_t errsz) {
    PixieCompare cmp = PIXIE_CMP_EQ;
    char *buf = pixie_strdup_c(stmt);
    char *lhs = NULL;
    char *rhs = NULL;
    char *p = NULL;
    char *colon = NULL;
    int clen = 0;
    PixieSparse lterms;
    PixieSparse rterms;
    PixieSparse comb;
    double lc = 0.0;
    double rc = 0.0;
    double nrhs = 0.0;
    int i;

    lhs = buf;
    p = buf;
    while (*p != '\0') {
        if (*p == '<') {
            cmp = PIXIE_CMP_LE;
            clen = (p[1] == '=') ? 2 : 1;
            break;
        }
        if (*p == '>') {
            cmp = PIXIE_CMP_GE;
            clen = (p[1] == '=') ? 2 : 1;
            break;
        }
        if (*p == '=') {
            cmp = PIXIE_CMP_EQ;
            clen = 1;
            break;
        }
        ++p;
    }
    if (*p == '\0') {
        free(buf);
        pixie_set_error(err, errsz, "constraint has no comparator");
        return 0;
    }
    *p = '\0';
    rhs = p + clen;

    lhs = pixie_trim(lhs);
    rhs = pixie_trim(rhs);

    colon = strchr(lhs, ':');
    if (colon != NULL) {
        lhs = pixie_trim(colon + 1);
    }

    if (strchr(lhs, '<') != NULL || strchr(lhs, '>') != NULL || strchr(lhs, '=') != NULL) {
        char msg[256];
        snprintf(msg, sizeof(msg), "internal split error lhs='%.120s'", lhs);
        free(buf);
        pixie_set_error(err, errsz, msg);
        return 0;
    }

    if (!pixie_parse_linear_expr(m, lhs, true, &lterms, &lc, err, errsz)) {
        free(buf);
        return 0;
    }
    if (!pixie_parse_linear_expr(m, rhs, true, &rterms, &rc, err, errsz)) {
        pixie_sparse_free(&lterms);
        free(buf);
        return 0;
    }

    pixie_sparse_init(&comb);
    for (i = 0; i < lterms.len; ++i) {
        pixie_sparse_add(&comb, lterms.idx[i], lterms.val[i]);
    }
    for (i = 0; i < rterms.len; ++i) {
        pixie_sparse_add(&comb, rterms.idx[i], -rterms.val[i]);
    }
    nrhs = rc - lc;
    pixie_sparse_prune(&comb, PIXIE_EPS);

    if (!pixie_model_add_constraint(m, &comb, cmp, nrhs)) {
        pixie_sparse_free(&comb);
        pixie_sparse_free(&lterms);
        pixie_sparse_free(&rterms);
        free(buf);
        pixie_set_error(err, errsz, "failed to add LP constraint");
        return 0;
    }

    pixie_sparse_free(&comb);
    pixie_sparse_free(&lterms);
    pixie_sparse_free(&rterms);
    free(buf);
    return 1;
}

static void pixie_tokvec_init(PixieTokVec *tv) {
    tv->data = NULL;
    tv->len = 0;
    tv->cap = 0;
}

static void pixie_tokvec_free(PixieTokVec *tv) {
    int i;
    for (i = 0; i < tv->len; ++i) {
        free(tv->data[i]);
    }
    free(tv->data);
    tv->data = NULL;
    tv->len = 0;
    tv->cap = 0;
}

static void pixie_tokvec_push(PixieTokVec *tv, const char *tok) {
    int nc;
    if (tv->len >= tv->cap) {
        nc = (tv->cap <= 0) ? 8 : tv->cap;
        while (nc <= tv->len) {
            if (nc > INT_MAX / 2) {
                nc = tv->len + 1;
                break;
            }
            nc *= 2;
        }
        tv->data = (char **)pixie_xrealloc(tv->data, (size_t)nc * sizeof(char *));
        tv->cap = nc;
    }
    tv->data[tv->len] = pixie_strdup_c(tok);
    ++tv->len;
}

static void pixie_tokenize_bounds(const char *stmt, PixieTokVec *tv) {
    const char *p = stmt;
    char buf[512];
    pixie_tokvec_init(tv);

    while (*p != '\0') {
        size_t n = 0;
        while (*p != '\0' && isspace((unsigned char)*p)) {
            ++p;
        }
        if (*p == '\0') {
            break;
        }
        if (*p == '<' || *p == '>' || *p == '=') {
            buf[n++] = *p;
            if ((p[0] == '<' || p[0] == '>') && p[1] == '=') {
                ++p;
                buf[n++] = *p;
            }
            ++p;
            buf[n] = '\0';
            pixie_tokvec_push(tv, buf);
            continue;
        }
        while (*p != '\0' && !isspace((unsigned char)*p) && *p != '<' && *p != '>' && *p != '=') {
            if (n + 1 < sizeof(buf)) {
                buf[n++] = *p;
            }
            ++p;
        }
        buf[n] = '\0';
        if (n > 0) {
            pixie_tokvec_push(tv, buf);
        }
    }
}

static int pixie_lp_apply_bound_rel(PixieModel *m, const char *left, const char *op, const char *right, char *err, size_t errsz) {
    double lv = 0.0;
    double rv = 0.0;
    bool left_num = pixie_parse_double_token(left, &lv);
    bool right_num = pixie_parse_double_token(right, &rv);
    int vidx = -1;

    if (!left_num && right_num) {
        vidx = pixie_model_get_var(m, left, true);
        if (vidx < 0) {
            pixie_set_error(err, errsz, "invalid bound variable");
            return 0;
        }
        if (strcmp(op, "<=") == 0 || strcmp(op, "<") == 0) {
            m->vars[vidx].ub = pixie_min(m->vars[vidx].ub, rv);
            return 1;
        }
        if (strcmp(op, ">=") == 0 || strcmp(op, ">") == 0) {
            m->vars[vidx].lb = pixie_max(m->vars[vidx].lb, rv);
            return 1;
        }
        if (strcmp(op, "=") == 0) {
            m->vars[vidx].lb = rv;
            m->vars[vidx].ub = rv;
            return 1;
        }
        pixie_set_error(err, errsz, "unsupported bounds operator");
        return 0;
    }

    if (left_num && !right_num) {
        vidx = pixie_model_get_var(m, right, true);
        if (vidx < 0) {
            pixie_set_error(err, errsz, "invalid bound variable");
            return 0;
        }
        if (strcmp(op, "<=") == 0 || strcmp(op, "<") == 0) {
            m->vars[vidx].lb = pixie_max(m->vars[vidx].lb, lv);
            return 1;
        }
        if (strcmp(op, ">=") == 0 || strcmp(op, ">") == 0) {
            m->vars[vidx].ub = pixie_min(m->vars[vidx].ub, lv);
            return 1;
        }
        if (strcmp(op, "=") == 0) {
            m->vars[vidx].lb = lv;
            m->vars[vidx].ub = lv;
            return 1;
        }
        pixie_set_error(err, errsz, "unsupported bounds operator");
        return 0;
    }

    pixie_set_error(err, errsz, "bounds relation must be between one variable and one number");
    return 0;
}

static int pixie_parse_lp_bounds(PixieModel *m, char *stmt, char *err, size_t errsz) {
    PixieTokVec tv;
    char *t;
    int idx;
    pixie_tokenize_bounds(stmt, &tv);
    if (tv.len == 0) {
        pixie_tokvec_free(&tv);
        return 1;
    }

    t = tv.data[0];
    if (tv.len >= 2 && pixie_stricmp_c(tv.data[1], "free") == 0) {
        idx = pixie_model_get_var(m, t, true);
        if (idx < 0) {
            pixie_tokvec_free(&tv);
            pixie_set_error(err, errsz, "invalid variable in free bound");
            return 0;
        }
        m->vars[idx].lb = -HUGE_VAL;
        m->vars[idx].ub = HUGE_VAL;
        pixie_tokvec_free(&tv);
        return 1;
    }

    if (tv.len == 3) {
        int ok = pixie_lp_apply_bound_rel(m, tv.data[0], tv.data[1], tv.data[2], err, errsz);
        pixie_tokvec_free(&tv);
        return ok;
    }
    if (tv.len == 5) {
        int ok1 = pixie_lp_apply_bound_rel(m, tv.data[0], tv.data[1], tv.data[2], err, errsz);
        int ok2 = 0;
        if (!ok1) {
            pixie_tokvec_free(&tv);
            return 0;
        }
        ok2 = pixie_lp_apply_bound_rel(m, tv.data[2], tv.data[3], tv.data[4], err, errsz);
        pixie_tokvec_free(&tv);
        return ok2;
    }

    pixie_tokvec_free(&tv);
    pixie_set_error(err, errsz, "unsupported bounds syntax");
    return 0;
}

static bool pixie_lp_is_keyword(const char *s) {
    return pixie_stricmp_c(s, "int") == 0 ||
           pixie_stricmp_c(s, "integer") == 0 ||
           pixie_stricmp_c(s, "integers") == 0 ||
           pixie_stricmp_c(s, "general") == 0 ||
           pixie_stricmp_c(s, "generals") == 0 ||
           pixie_stricmp_c(s, "binary") == 0 ||
           pixie_stricmp_c(s, "binaries") == 0 ||
           pixie_stricmp_c(s, "bin") == 0;
}

static int pixie_parse_lp_var_list(PixieModel *m, char *stmt, bool binary, char *err, size_t errsz) {
    char *dup = pixie_strdup_c(stmt);
    char *tok = NULL;
    char *ctx = NULL;
    tok = strtok_s(dup, " \t\r\n,:", &ctx);
    while (tok != NULL) {
        int idx;
        if (!pixie_lp_is_keyword(tok)) {
            idx = pixie_model_get_var(m, tok, true);
            if (idx < 0) {
                free(dup);
                pixie_set_error(err, errsz, "invalid variable in integer/binary list");
                return 0;
            }
            if (binary) {
                pixie_var_mark_binary(m, idx);
            } else {
                pixie_var_mark_integer(m, idx);
            }
        }
        tok = strtok_s(NULL, " \t\r\n,:", &ctx);
    }
    free(dup);
    return 1;
}

enum {
    LP_SEC_NONE = 0,
    LP_SEC_OBJ = 1,
    LP_SEC_CONS = 2,
    LP_SEC_BOUNDS = 3,
    LP_SEC_GENERAL = 4,
    LP_SEC_BINARY = 5
};

static int pixie_parse_lp_statement(PixieModel *m, char *stmt, int *section, char *err, size_t errsz) {
    char *s = pixie_trim(stmt);
    int pos = -1;
    int clen = 0;
    PixieCompare cmp = PIXIE_CMP_EQ;

    if (*s == '\0') {
        return 1;
    }

    if (pixie_starts_with_ci(s, "min:")) {
        m->obj_sense = +1;
        *section = LP_SEC_OBJ;
        s = pixie_trim(s + 4);
        if (*s == '\0') {
            return 1;
        }
        return pixie_parse_lp_objective(m, s, err, errsz);
    }
    if (pixie_starts_with_ci(s, "max:")) {
        m->obj_sense = -1;
        *section = LP_SEC_OBJ;
        s = pixie_trim(s + 4);
        if (*s == '\0') {
            return 1;
        }
        return pixie_parse_lp_objective(m, s, err, errsz);
    }
    if (pixie_starts_with_ci(s, "minimize")) {
        m->obj_sense = +1;
        *section = LP_SEC_OBJ;
        s = pixie_trim(s + 8);
        if (*s == '\0') {
            return 1;
        }
        return pixie_parse_lp_objective(m, s, err, errsz);
    }
    if (pixie_starts_with_ci(s, "maximize")) {
        m->obj_sense = -1;
        *section = LP_SEC_OBJ;
        s = pixie_trim(s + 8);
        if (*s == '\0') {
            return 1;
        }
        return pixie_parse_lp_objective(m, s, err, errsz);
    }
    if (pixie_starts_with_ci(s, "subject to")) {
        *section = LP_SEC_CONS;
        s = pixie_trim(s + 10);
        if (*s == '\0') {
            return 1;
        }
    }
    if (pixie_starts_with_ci(s, "such that")) {
        *section = LP_SEC_CONS;
        s = pixie_trim(s + 9);
        if (*s == '\0') {
            return 1;
        }
    }
    if (pixie_starts_with_ci(s, "st")) {
        if (s[2] == '\0' || isspace((unsigned char)s[2])) {
            *section = LP_SEC_CONS;
            s = pixie_trim(s + 2);
            if (*s == '\0') {
                return 1;
            }
        }
    }
    if (pixie_stricmp_c(s, "bounds") == 0 || pixie_starts_with_ci(s, "bounds ")) {
        *section = LP_SEC_BOUNDS;
        s = pixie_trim(s + 6);
        if (*s == '\0') {
            return 1;
        }
        return pixie_parse_lp_bounds(m, s, err, errsz);
    }
    if (pixie_stricmp_c(s, "general") == 0 || pixie_stricmp_c(s, "generals") == 0 ||
        pixie_stricmp_c(s, "integer") == 0 || pixie_stricmp_c(s, "integers") == 0 ||
        pixie_stricmp_c(s, "int") == 0 || pixie_starts_with_ci(s, "general ") ||
        pixie_starts_with_ci(s, "generals ") || pixie_starts_with_ci(s, "integer ") ||
        pixie_starts_with_ci(s, "integers ") || pixie_starts_with_ci(s, "int ")) {
        *section = LP_SEC_GENERAL;
        return pixie_parse_lp_var_list(m, s, false, err, errsz);
    }
    if (pixie_stricmp_c(s, "binary") == 0 || pixie_stricmp_c(s, "binaries") == 0 ||
        pixie_stricmp_c(s, "bin") == 0 || pixie_starts_with_ci(s, "binary ") ||
        pixie_starts_with_ci(s, "binaries ") || pixie_starts_with_ci(s, "bin ")) {
        *section = LP_SEC_BINARY;
        return pixie_parse_lp_var_list(m, s, true, err, errsz);
    }
    if (pixie_stricmp_c(s, "end") == 0) {
        return 2;
    }

    if (*section == LP_SEC_OBJ) {
        return pixie_parse_lp_objective(m, s, err, errsz);
    }
    if (*section == LP_SEC_BOUNDS) {
        return pixie_parse_lp_bounds(m, s, err, errsz);
    }
    if (*section == LP_SEC_GENERAL) {
        return pixie_parse_lp_var_list(m, s, false, err, errsz);
    }
    if (*section == LP_SEC_BINARY) {
        return pixie_parse_lp_var_list(m, s, true, err, errsz);
    }

    if (pixie_find_comparator(s, &pos, &clen, &cmp)) {
        *section = LP_SEC_CONS;
        return pixie_parse_lp_constraint(m, s, err, errsz);
    }

    if (*section == LP_SEC_CONS) {
        pixie_set_error(err, errsz, "constraint line missing comparator");
        return 0;
    }

    return pixie_parse_lp_constraint(m, s, err, errsz);
}

static int pixie_parse_lp_file(const char *path, PixieModel *m, double deadline, char *err, size_t errsz) {
    FILE *fp = fopen(path, "rb");
    char line[4096];
    int section = LP_SEC_NONE;
    int ended = 0;
    long long line_no = 0;
    if (fp == NULL) {
        pixie_set_error(err, errsz, "failed to open LP file");
        return 0;
    }

    while (fgets(line, (int)sizeof(line), fp) != NULL) {
        char *cursor = line;
        char *comment = strchr(cursor, '\\');
        int line_done = 0;
        ++line_no;
        if ((line_no & 255LL) == 0LL && pixie_deadline_reached(deadline)) {
            pixie_set_error(err, errsz, "time limit exceeded during parsing");
            fclose(fp);
            return 0;
        }
        if (comment != NULL) {
            *comment = '\0';
        }
        while (!line_done) {
            char *semi = strchr(cursor, ';');
            char *stmt = NULL;
            int rc;
            if (semi != NULL) {
                *semi = '\0';
                stmt = pixie_trim(cursor);
                if (*stmt != '\0') {
                    rc = pixie_parse_lp_statement(m, stmt, &section, err, errsz);
                    if (rc == 0) {
                        if (err != NULL && errsz > 0) {
                            char msg[512];
                            snprintf(msg, sizeof(msg), "%s | %.240s", err, stmt);
                            pixie_set_error(err, errsz, msg);
                        }
                        fclose(fp);
                        return 0;
                    }
                    if (rc == 2) {
                        ended = 1;
                        break;
                    }
                }
                cursor = semi + 1;
            } else {
                stmt = pixie_trim(cursor);
                if (*stmt != '\0') {
                    rc = pixie_parse_lp_statement(m, stmt, &section, err, errsz);
                    if (rc == 0) {
                        if (err != NULL && errsz > 0) {
                            char msg[512];
                            snprintf(msg, sizeof(msg), "%s | %.240s", err, stmt);
                            pixie_set_error(err, errsz, msg);
                        }
                        fclose(fp);
                        return 0;
                    }
                    if (rc == 2) {
                        ended = 1;
                        break;
                    }
                }
                line_done = 1;
            }
        }
        if (ended) {
            break;
        }
    }

    fclose(fp);
    if (m->n_vars <= 0) {
        pixie_set_error(err, errsz, "LP parser found no variables");
        return 0;
    }
    return 1;
}

static int pixie_split_ws(char *line, char **tok, int max_tok) {
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
        tok[n++] = p;
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

static char *pixie_capture_command(const char *cmd, int *status_out) {
    FILE *pipe = pixie_popen(cmd, "r");
    char buf[4096];
    char *out = NULL;
    size_t len = 0;
    size_t cap = 0;
    int status = -1;

    if (pipe == NULL) {
        return NULL;
    }
    while (fgets(buf, (int)sizeof(buf), pipe) != NULL) {
        size_t n = strlen(buf);
        if (len + n + 1 > cap) {
            size_t nc = (cap > 0) ? cap : 4096;
            while (len + n + 1 > nc) {
                nc *= 2;
            }
            out = (char *)pixie_xrealloc(out, nc);
            cap = nc;
        }
        memcpy(out + len, buf, n);
        len += n;
    }
    status = pixie_pclose(pipe);
    if (out == NULL) {
        out = pixie_strdup_c("");
    } else {
        out[len] = '\0';
    }
    if (status_out != NULL) {
        *status_out = status;
    }
    return out;
}

static bool pixie_build_lpsolve_command(char *dst, size_t dstsz, const char *solver, const char *path, bool is_lp) {
    int n;
    if (dst == NULL || dstsz == 0 || solver == NULL || path == NULL) {
        return false;
    }
    if (strchr(path, '"') != NULL) {
        return false;
    }
    n = snprintf(dst, dstsz, "%s -S2 %s \"%s\" 2>&1", solver, is_lp ? "-lp" : "-mps", path);
    return (n >= 0 && (size_t)n < dstsz);
}

#if defined(_WIN32)
static bool pixie_windows_to_wsl_path(const char *path, char *out, size_t outsz) {
    char absbuf[4096];
    char cmd[8192];
    char *resp;
    char *trimmed;

    if (path == NULL || out == NULL || outsz == 0) {
        return false;
    }
    if (_fullpath(absbuf, path, sizeof(absbuf)) == NULL) {
        size_t n = strlen(path);
        if (n + 1 > sizeof(absbuf)) {
            return false;
        }
        memcpy(absbuf, path, n + 1);
    }
    if (strchr(absbuf, '"') != NULL) {
        return false;
    }
    if (snprintf(cmd, sizeof(cmd), "wsl wslpath -a -u \"%s\" 2>&1", absbuf) >= (int)sizeof(cmd)) {
        return false;
    }
    resp = pixie_capture_command(cmd, NULL);
    if (resp == NULL) {
        return false;
    }
    trimmed = pixie_trim(resp);
    if (*trimmed != '/') {
        free(resp);
        return false;
    }
    if (strlen(trimmed) + 1 > outsz) {
        free(resp);
        return false;
    }
    memcpy(out, trimmed, strlen(trimmed) + 1);
    free(resp);
    return true;
}
#endif

static bool pixie_parse_lpsolve_output(const PixieModel *m, const char *text, PixieSolution *sol) {
    char *copy;
    char *line;
    double *tmp_x;
    double obj_out = NAN;
    bool have_obj = false;
    bool in_vars = false;
    bool recognized = false;
    PixieStatus status = PIXIE_STATUS_UNKNOWN;
    int i;

    if (m == NULL || text == NULL || sol == NULL) {
        return false;
    }

    if (strstr(text, "This problem is infeasible") != NULL) {
        recognized = true;
        status = PIXIE_STATUS_INFEASIBLE;
    } else if (strstr(text, "This problem is unbounded") != NULL) {
        recognized = true;
        status = PIXIE_STATUS_UNBOUNDED;
    }

    tmp_x = (double *)pixie_xcalloc((size_t)m->n_vars, sizeof(double));
    copy = pixie_strdup_c(text);
    for (line = strtok(copy, "\r\n"); line != NULL; line = strtok(NULL, "\r\n")) {
        char *s = pixie_trim(line);
        if (*s == '\0') {
            continue;
        }
        if (pixie_starts_with_ci(s, "Value of objective function:")) {
            char *colon = strchr(s, ':');
            double v = 0.0;
            if (colon != NULL && pixie_parse_double_token(colon + 1, &v)) {
                obj_out = v;
                have_obj = true;
                recognized = true;
                status = PIXIE_STATUS_OPTIMAL;
            }
            continue;
        }
        if (pixie_starts_with_ci(s, "Actual values of the variables:")) {
            in_vars = true;
            continue;
        }
        if (in_vars) {
            char name[256];
            char valtok[128];
            double v = 0.0;
            if (sscanf(s, "%255s %127s", name, valtok) == 2 && pixie_parse_double_token(valtok, &v)) {
                int vidx = pixie_model_find_var(m, name);
                if (vidx >= 0 && vidx < m->n_vars) {
                    tmp_x[vidx] = v;
                }
            }
        }
    }
    free(copy);

    if (!recognized) {
        free(tmp_x);
        return false;
    }

    sol->status = status;
    sol->obj_min = NAN;
    sol->obj_out = NAN;
    sol->has_primal = false;
    sol->nodes_processed = 0;
    sol->stopped_time = false;
    sol->stopped_nodes = false;
    sol->stopped_gap = false;
    for (i = 0; i < sol->n; ++i) {
        sol->x[i] = 0.0;
    }

    if (status == PIXIE_STATUS_OPTIMAL && have_obj) {
        memcpy(sol->x, tmp_x, (size_t)m->n_vars * sizeof(double));
        sol->obj_out = obj_out;
        sol->obj_min = (m->obj_sense == +1) ? obj_out : -obj_out;
        sol->has_primal = true;
    }
    free(tmp_x);
    return true;
}

static bool pixie_try_external_lp_solve(const PixieModel *m, const PixieOptions *opt, PixieSolution *sol, const char **solver_used) {
    char cmd[8192];
    char *out = NULL;
    bool is_lp;

    if (m == NULL || opt == NULL || sol == NULL || opt->file == NULL) {
        return false;
    }
    is_lp = (opt->format == PIXIE_FORMAT_LP) || (opt->format == PIXIE_FORMAT_AUTO && pixie_ends_with_ci(opt->file, ".lp"));

    if (pixie_build_lpsolve_command(cmd, sizeof(cmd), "lp_solve", opt->file, is_lp)) {
        out = pixie_capture_command(cmd, NULL);
        if (out != NULL && pixie_parse_lpsolve_output(m, out, sol)) {
            free(out);
            if (solver_used != NULL) {
                *solver_used = "lp_solve";
            }
            return true;
        }
        free(out);
        out = NULL;
    }

#if defined(_WIN32)
    {
        char wsl_path[4096];
        if (pixie_windows_to_wsl_path(opt->file, wsl_path, sizeof(wsl_path)) &&
            pixie_build_lpsolve_command(cmd, sizeof(cmd), "wsl lp_solve", wsl_path, is_lp)) {
            out = pixie_capture_command(cmd, NULL);
            if (out != NULL && pixie_parse_lpsolve_output(m, out, sol)) {
                free(out);
                if (solver_used != NULL) {
                    *solver_used = "wsl lp_solve";
                }
                return true;
            }
            free(out);
        }
    }
#endif

    return false;
}

typedef struct {
    char *name;
    int cidx;
    PixieCompare cmp;
    bool has_range;
} PixieMpsRow;

typedef struct {
    PixieMpsRow *data;
    int len;
    int cap;
} PixieMpsRows;

static void pixie_mps_rows_init(PixieMpsRows *r) {
    r->data = NULL;
    r->len = 0;
    r->cap = 0;
}

static void pixie_mps_rows_free(PixieMpsRows *r) {
    int i;
    for (i = 0; i < r->len; ++i) {
        free(r->data[i].name);
    }
    free(r->data);
    r->data = NULL;
    r->len = 0;
    r->cap = 0;
}

static int pixie_mps_rows_add(PixieMpsRows *r, const char *name, int cidx, PixieCompare cmp) {
    int nc;
    if (r->len >= r->cap) {
        nc = (r->cap <= 0) ? 8 : r->cap;
        while (nc <= r->len) {
            if (nc > INT_MAX / 2) {
                nc = r->len + 1;
                break;
            }
            nc *= 2;
        }
        r->data = (PixieMpsRow *)pixie_xrealloc(r->data, (size_t)nc * sizeof(PixieMpsRow));
        r->cap = nc;
    }
    r->data[r->len].name = pixie_strdup_c(name);
    r->data[r->len].cidx = cidx;
    r->data[r->len].cmp = cmp;
    r->data[r->len].has_range = false;
    ++r->len;
    return r->len - 1;
}

static int pixie_mps_rows_find(const PixieMpsRows *r, const char *name) {
    int i;
    for (i = 0; i < r->len; ++i) {
        if (strcmp(r->data[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

static int pixie_apply_mps_range(PixieModel *m, PixieMpsRow *row, double range, char *err, size_t errsz) {
    PixieConstraint *base;
    double width;
    double rhs;

    if (m == NULL || row == NULL || row->cidx < 0 || row->cidx >= m->n_cons) {
        pixie_set_error(err, errsz, "invalid row in RANGES");
        return 0;
    }
    if (row->has_range) {
        pixie_set_error(err, errsz, "duplicate RANGES entry for row");
        return 0;
    }
    if (fabs(range) <= PIXIE_EPS) {
        row->has_range = true;
        return 1;
    }

    base = &m->cons[row->cidx];
    width = fabs(range);
    rhs = base->rhs;

    if (row->cmp == PIXIE_CMP_LE) {
        if (!pixie_model_add_constraint(m, &base->a, PIXIE_CMP_GE, rhs - width)) {
            pixie_set_error(err, errsz, "failed to expand ranged <= row");
            return 0;
        }
    } else if (row->cmp == PIXIE_CMP_GE) {
        if (!pixie_model_add_constraint(m, &base->a, PIXIE_CMP_LE, rhs + width)) {
            pixie_set_error(err, errsz, "failed to expand ranged >= row");
            return 0;
        }
    } else {
        if (range >= 0.0) {
            base->cmp = PIXIE_CMP_GE;
            if (!pixie_model_add_constraint(m, &base->a, PIXIE_CMP_LE, rhs + width)) {
                pixie_set_error(err, errsz, "failed to expand ranged equality");
                return 0;
            }
        } else {
            base->cmp = PIXIE_CMP_LE;
            if (!pixie_model_add_constraint(m, &base->a, PIXIE_CMP_GE, rhs - width)) {
                pixie_set_error(err, errsz, "failed to expand ranged equality");
                return 0;
            }
        }
    }

    row->has_range = true;
    return 1;
}

static int pixie_parse_mps_file(const char *path, PixieModel *m, double deadline, char *err, size_t errsz) {
    FILE *fp = fopen(path, "rb");
    char line[4096];
    char *tok[32];
    int ntok;
    enum { MPS_NONE = 0, MPS_ROWS = 1, MPS_COLUMNS = 2, MPS_RHS = 3, MPS_RANGES = 4, MPS_BOUNDS = 5, MPS_OBJSENSE = 6 } sec = MPS_NONE;
    PixieMpsRows rows;
    int obj_row = -1;
    bool int_mode = false;
    long long line_no = 0;
    int i;

    if (fp == NULL) {
        pixie_set_error(err, errsz, "failed to open MPS file");
        return 0;
    }

    pixie_mps_rows_init(&rows);
    m->obj_sense = +1;

    while (fgets(line, (int)sizeof(line), fp) != NULL) {
        bool indented = (line[0] != '\0' && isspace((unsigned char)line[0])) ? true : false;
        char *s = pixie_trim(line);
        ++line_no;
        if ((line_no & 255LL) == 0LL && pixie_deadline_reached(deadline)) {
            pixie_set_error(err, errsz, "time limit exceeded during parsing");
            pixie_mps_rows_free(&rows);
            fclose(fp);
            return 0;
        }
        if (*s == '\0') {
            continue;
        }
        if (*s == '*') {
            if (strstr(s, "origsense='MAX'") != NULL || strstr(s, "origsense=\"MAX\"") != NULL) {
                m->obj_sense = -1;
            } else if (strstr(s, "origsense='MIN'") != NULL || strstr(s, "origsense=\"MIN\"") != NULL) {
                m->obj_sense = +1;
            }
            continue;
        }

        ntok = pixie_split_ws(s, tok, 32);
        if (ntok <= 0) {
            continue;
        }

        if (!indented) {
            if (pixie_stricmp_c(tok[0], "NAME") == 0) {
                sec = MPS_NONE;
                continue;
            }
            if (pixie_stricmp_c(tok[0], "OBJSENSE") == 0) {
                sec = MPS_OBJSENSE;
                continue;
            }
            if (pixie_stricmp_c(tok[0], "ROWS") == 0) {
                sec = MPS_ROWS;
                continue;
            }
            if (pixie_stricmp_c(tok[0], "COLUMNS") == 0) {
                sec = MPS_COLUMNS;
                continue;
            }
            if (pixie_stricmp_c(tok[0], "RHS") == 0) {
                sec = MPS_RHS;
                continue;
            }
            if (pixie_stricmp_c(tok[0], "RANGES") == 0) {
                sec = MPS_RANGES;
                continue;
            }
            if (pixie_stricmp_c(tok[0], "BOUNDS") == 0) {
                sec = MPS_BOUNDS;
                continue;
            }
            if (pixie_stricmp_c(tok[0], "ENDATA") == 0 || pixie_stricmp_c(tok[0], "END") == 0) {
                break;
            }
        }

        if (sec == MPS_OBJSENSE) {
            if (pixie_stricmp_c(tok[0], "MAX") == 0 || pixie_stricmp_c(tok[0], "MAXIMIZE") == 0) {
                m->obj_sense = -1;
            } else if (pixie_stricmp_c(tok[0], "MIN") == 0 || pixie_stricmp_c(tok[0], "MINIMIZE") == 0) {
                m->obj_sense = +1;
            }
            continue;
        }

        if (sec == MPS_ROWS) {
            char t;
            int cidx;
            if (ntok < 2) {
                pixie_set_error(err, errsz, "invalid ROWS line in MPS");
                pixie_mps_rows_free(&rows);
                fclose(fp);
                return 0;
            }
            t = tok[0][0];
            if (t == 'N' || t == 'n') {
                if (obj_row < 0) {
                    obj_row = pixie_mps_rows_add(&rows, tok[1], -1, PIXIE_CMP_EQ);
                } else {
                    pixie_mps_rows_add(&rows, tok[1], -1, PIXIE_CMP_EQ);
                }
                continue;
            }
            if (t == 'L' || t == 'l') {
                cidx = pixie_model_add_empty_constraint(m, PIXIE_CMP_LE, 0.0);
            } else if (t == 'G' || t == 'g') {
                cidx = pixie_model_add_empty_constraint(m, PIXIE_CMP_GE, 0.0);
            } else if (t == 'E' || t == 'e') {
                cidx = pixie_model_add_empty_constraint(m, PIXIE_CMP_EQ, 0.0);
            } else {
                pixie_set_error(err, errsz, "unsupported ROWS type in MPS");
                pixie_mps_rows_free(&rows);
                fclose(fp);
                return 0;
            }
            pixie_mps_rows_add(&rows, tok[1], cidx, m->cons[cidx].cmp);
            continue;
        }

        if (sec == MPS_COLUMNS) {
            int vidx;
            if (ntok >= 3 && strcmp(tok[1], "'MARKER'") == 0) {
                if (ntok >= 3 && (strstr(tok[2], "INTORG") != NULL || strstr(tok[2], "intorg") != NULL)) {
                    int_mode = true;
                } else if (ntok >= 3 && (strstr(tok[2], "INTEND") != NULL || strstr(tok[2], "intend") != NULL)) {
                    int_mode = false;
                }
                continue;
            }
            if (ntok < 3) {
                pixie_set_error(err, errsz, "invalid COLUMNS line in MPS");
                pixie_mps_rows_free(&rows);
                fclose(fp);
                return 0;
            }
            vidx = pixie_model_get_var(m, tok[0], true);
            if (vidx < 0) {
                pixie_set_error(err, errsz, "failed to create variable in MPS");
                pixie_mps_rows_free(&rows);
                fclose(fp);
                return 0;
            }
            if (int_mode) {
                pixie_var_mark_integer(m, vidx);
            }
            for (i = 1; i + 1 < ntok; i += 2) {
                int ridx = pixie_mps_rows_find(&rows, tok[i]);
                double v = 0.0;
                if (ridx < 0) {
                    pixie_set_error(err, errsz, "unknown row name in COLUMNS");
                    pixie_mps_rows_free(&rows);
                    fclose(fp);
                    return 0;
                }
                if (!pixie_parse_double_token(tok[i + 1], &v)) {
                    pixie_set_error(err, errsz, "invalid numeric value in COLUMNS");
                    pixie_mps_rows_free(&rows);
                    fclose(fp);
                    return 0;
                }
                if (rows.data[ridx].cidx < 0) {
                    m->vars[vidx].obj += v;
                } else {
                    pixie_sparse_add(&m->cons[rows.data[ridx].cidx].a, vidx, v);
                }
            }
            continue;
        }

        if (sec == MPS_RHS) {
            if (ntok < 3) {
                pixie_set_error(err, errsz, "invalid RHS line in MPS");
                pixie_mps_rows_free(&rows);
                fclose(fp);
                return 0;
            }
            for (i = 1; i + 1 < ntok; i += 2) {
                int ridx = pixie_mps_rows_find(&rows, tok[i]);
                double v = 0.0;
                if (ridx < 0) {
                    pixie_set_error(err, errsz, "unknown row in RHS");
                    pixie_mps_rows_free(&rows);
                    fclose(fp);
                    return 0;
                }
                if (!pixie_parse_double_token(tok[i + 1], &v)) {
                    pixie_set_error(err, errsz, "invalid RHS value");
                    pixie_mps_rows_free(&rows);
                    fclose(fp);
                    return 0;
                }
                if (rows.data[ridx].cidx >= 0) {
                    m->cons[rows.data[ridx].cidx].rhs = v;
                }
            }
            continue;
        }

        if (sec == MPS_RANGES) {
            if (ntok < 3) {
                pixie_set_error(err, errsz, "invalid RANGES line in MPS");
                pixie_mps_rows_free(&rows);
                fclose(fp);
                return 0;
            }
            for (i = 1; i + 1 < ntok; i += 2) {
                int ridx = pixie_mps_rows_find(&rows, tok[i]);
                double v = 0.0;
                if (ridx < 0) {
                    pixie_set_error(err, errsz, "unknown row in RANGES");
                    pixie_mps_rows_free(&rows);
                    fclose(fp);
                    return 0;
                }
                if (rows.data[ridx].cidx < 0) {
                    pixie_set_error(err, errsz, "RANGES not supported on objective row");
                    pixie_mps_rows_free(&rows);
                    fclose(fp);
                    return 0;
                }
                if (!pixie_parse_double_token(tok[i + 1], &v)) {
                    pixie_set_error(err, errsz, "invalid RANGES value");
                    pixie_mps_rows_free(&rows);
                    fclose(fp);
                    return 0;
                }
                if (!pixie_apply_mps_range(m, &rows.data[ridx], v, err, errsz)) {
                    pixie_mps_rows_free(&rows);
                    fclose(fp);
                    return 0;
                }
            }
            continue;
        }

        if (sec == MPS_BOUNDS) {
            int vidx;
            double v = 0.0;
            if (ntok < 3) {
                pixie_set_error(err, errsz, "invalid BOUNDS line in MPS");
                pixie_mps_rows_free(&rows);
                fclose(fp);
                return 0;
            }
            vidx = pixie_model_get_var(m, tok[2], true);
            if (vidx < 0) {
                pixie_set_error(err, errsz, "unknown variable in BOUNDS");
                pixie_mps_rows_free(&rows);
                fclose(fp);
                return 0;
            }
            if (ntok >= 4) {
                if (!pixie_parse_double_token(tok[3], &v)) {
                    pixie_set_error(err, errsz, "invalid numeric value in BOUNDS");
                    pixie_mps_rows_free(&rows);
                    fclose(fp);
                    return 0;
                }
            }
            if (pixie_stricmp_c(tok[0], "LO") == 0) {
                m->vars[vidx].lb = v;
                m->vars[vidx].has_explicit_lb = true;
            } else if (pixie_stricmp_c(tok[0], "UP") == 0) {
                m->vars[vidx].ub = v;
                m->vars[vidx].has_explicit_ub = true;
                if (!m->vars[vidx].has_explicit_lb) {
                    m->vars[vidx].lb = (v < 0.0) ? -HUGE_VAL : 0.0;
                }
            } else if (pixie_stricmp_c(tok[0], "FX") == 0) {
                m->vars[vidx].lb = v;
                m->vars[vidx].ub = v;
                m->vars[vidx].has_explicit_lb = true;
                m->vars[vidx].has_explicit_ub = true;
            } else if (pixie_stricmp_c(tok[0], "FR") == 0) {
                m->vars[vidx].lb = -HUGE_VAL;
                m->vars[vidx].ub = HUGE_VAL;
                m->vars[vidx].has_explicit_lb = true;
                m->vars[vidx].has_explicit_ub = true;
            } else if (pixie_stricmp_c(tok[0], "MI") == 0) {
                m->vars[vidx].lb = -HUGE_VAL;
                m->vars[vidx].has_explicit_lb = true;
            } else if (pixie_stricmp_c(tok[0], "PL") == 0) {
                m->vars[vidx].ub = HUGE_VAL;
                m->vars[vidx].has_explicit_ub = true;
                if (!m->vars[vidx].has_explicit_lb) {
                    m->vars[vidx].lb = 0.0;
                }
            } else if (pixie_stricmp_c(tok[0], "BV") == 0) {
                pixie_var_mark_binary(m, vidx);
            } else if (pixie_stricmp_c(tok[0], "LI") == 0) {
                pixie_var_mark_integer(m, vidx);
                m->vars[vidx].lb = v;
                m->vars[vidx].has_explicit_lb = true;
            } else if (pixie_stricmp_c(tok[0], "UI") == 0) {
                pixie_var_mark_integer(m, vidx);
                m->vars[vidx].ub = v;
                m->vars[vidx].has_explicit_ub = true;
                if (!m->vars[vidx].has_explicit_lb) {
                    m->vars[vidx].lb = (v < 0.0) ? -HUGE_VAL : 0.0;
                }
            } else {
                pixie_set_error(err, errsz, "unsupported BOUNDS type in MPS");
                pixie_mps_rows_free(&rows);
                fclose(fp);
                return 0;
            }
            continue;
        }
    }

    fclose(fp);
    for (i = 0; i < m->n_cons; ++i) {
        if ((i & 255) == 0 && pixie_deadline_reached(deadline)) {
            pixie_set_error(err, errsz, "time limit exceeded during parsing");
            pixie_mps_rows_free(&rows);
            return 0;
        }
        pixie_sparse_prune(&m->cons[i].a, PIXIE_EPS);
    }
    pixie_mps_rows_free(&rows);
    if (m->n_vars <= 0) {
        pixie_set_error(err, errsz, "MPS parser found no variables");
        return 0;
    }
    return 1;
}

static int pixie_read_model(const char *path, PixieFormat fmt, PixieModel *m, double deadline, char *err, size_t errsz) {
    int ok = 0;
    if (fmt == PIXIE_FORMAT_LP) {
        return pixie_parse_lp_file(path, m, deadline, err, errsz);
    }
    if (fmt == PIXIE_FORMAT_MPS) {
        return pixie_parse_mps_file(path, m, deadline, err, errsz);
    }
    if (pixie_ends_with_ci(path, ".lp")) {
        return pixie_parse_lp_file(path, m, deadline, err, errsz);
    }
    if (pixie_ends_with_ci(path, ".mps")) {
        return pixie_parse_mps_file(path, m, deadline, err, errsz);
    }
    ok = pixie_parse_lp_file(path, m, deadline, err, errsz);
    if (ok) {
        return 1;
    }
    if (pixie_deadline_reached(deadline)) {
        pixie_set_error(err, errsz, "time limit exceeded during parsing");
        return 0;
    }
    pixie_model_free(m);
    pixie_model_init(m);
    ok = pixie_parse_mps_file(path, m, deadline, err, errsz);
    if (ok) {
        return 1;
    }
    pixie_set_error(err, errsz, "could not parse file as LP or MPS");
    return 0;
}

static void pixie_convec_init(PixieConVec *v) {
    v->data = NULL;
    v->len = 0;
    v->cap = 0;
}

static void pixie_convec_free(PixieConVec *v) {
    int i;
    for (i = 0; i < v->len; ++i) {
        pixie_sparse_free(&v->data[i].a);
    }
    free(v->data);
    v->data = NULL;
    v->len = 0;
    v->cap = 0;
}

static int pixie_convec_push(PixieConVec *v, PixieConstraint *c) {
    int nc;
    if (v->len >= v->cap) {
        nc = (v->cap <= 0) ? 8 : v->cap;
        while (nc <= v->len) {
            if (nc > INT_MAX / 2) {
                nc = v->len + 1;
                break;
            }
            nc *= 2;
        }
        v->data = (PixieConstraint *)pixie_xrealloc(v->data, (size_t)nc * sizeof(PixieConstraint));
        v->cap = nc;
    }
    v->data[v->len].cmp = c->cmp;
    v->data[v->len].rhs = c->rhs;
    pixie_sparse_move(&v->data[v->len].a, &c->a);
    ++v->len;
    return 1;
}

static bool pixie_deadline_reached(double deadline);

static bool pixie_bound_tighten_lower(double *lb, double cand) {
    if (!pixie_is_finite(cand)) {
        return false;
    }
    if (cand > *lb + PIXIE_FEAS_TOL) {
        *lb = cand;
        return true;
    }
    return false;
}

static bool pixie_bound_tighten_upper(double *ub, double cand) {
    if (!pixie_is_finite(cand)) {
        return false;
    }
    if (cand < *ub - PIXIE_FEAS_TOL) {
        *ub = cand;
        return true;
    }
    return false;
}

static bool pixie_empty_row_is_feasible(PixieCompare cmp, double rhs) {
    if (cmp == PIXIE_CMP_EQ) {
        return fabs(rhs) <= PIXIE_FEAS_TOL;
    }
    if (cmp == PIXIE_CMP_LE) {
        return 0.0 <= rhs + PIXIE_FEAS_TOL;
    }
    return 0.0 >= rhs - PIXIE_FEAS_TOL;
}

static void pixie_mark_constraint_redundant(PixieConstraint *c) {
    pixie_sparse_free(&c->a);
    pixie_sparse_init(&c->a);
    c->cmp = PIXIE_CMP_LE;
    c->rhs = 0.0;
}

static bool pixie_lp_candidate_feasible(const PixieModel *m,
                                        const double *x,
                                        const double *lb_used,
                                        const double *ub_used,
                                        double deadline,
                                        bool *timed_out) {
    int i;

    if (timed_out != NULL) {
        *timed_out = false;
    }
    for (i = 0; i < m->n_vars; ++i) {
        double tol = PIXIE_CERT_TOL * (1.0 + fabs(x[i]));
        if ((i & 255) == 0 && pixie_deadline_reached(deadline)) {
            if (timed_out != NULL) {
                *timed_out = true;
            }
            return false;
        }
        if (pixie_is_finite(lb_used[i]) && x[i] < lb_used[i] - tol) {
            return false;
        }
        if (pixie_is_finite(ub_used[i]) && x[i] > ub_used[i] + tol) {
            return false;
        }
    }

    for (i = 0; i < m->n_cons; ++i) {
        const PixieConstraint *c = &m->cons[i];
        double lhs = 0.0;
        double tol;
        int k;
        if ((i & 255) == 0 && pixie_deadline_reached(deadline)) {
            if (timed_out != NULL) {
                *timed_out = true;
            }
            return false;
        }
        for (k = 0; k < c->a.len; ++k) {
            lhs += c->a.val[k] * x[c->a.idx[k]];
        }
        tol = PIXIE_CERT_TOL * (1.0 + fabs(c->rhs) + fabs(lhs));
        if (c->cmp == PIXIE_CMP_EQ) {
            if (fabs(lhs - c->rhs) > tol) {
                return false;
            }
        } else if (c->cmp == PIXIE_CMP_LE) {
            if (lhs > c->rhs + tol) {
                return false;
            }
        } else if (lhs < c->rhs - tol) {
            return false;
        }
    }
    return true;
}

static PixieStatus pixie_tighten_singleton_row(const PixieConstraint *c, double *lb, double *ub) {
    double a;
    double rhs;
    double bound;

    if (c->a.len != 1) {
        return PIXIE_STATUS_ERROR;
    }
    a = c->a.val[0];
    rhs = c->rhs;
    if (fabs(a) <= PIXIE_EPS) {
        return pixie_empty_row_is_feasible(c->cmp, rhs) ? PIXIE_STATUS_OPTIMAL : PIXIE_STATUS_INFEASIBLE;
    }
    bound = rhs / a;

    if (c->cmp == PIXIE_CMP_EQ) {
        if (pixie_bound_tighten_lower(lb, bound) && pixie_is_finite(*ub) && *lb > *ub + PIXIE_FEAS_TOL) {
            return PIXIE_STATUS_INFEASIBLE;
        }
        if (pixie_bound_tighten_upper(ub, bound) && *lb > *ub + PIXIE_FEAS_TOL) {
            return PIXIE_STATUS_INFEASIBLE;
        }
    } else if (c->cmp == PIXIE_CMP_LE) {
        if (a > 0.0) {
            if (pixie_bound_tighten_upper(ub, bound) && *lb > *ub + PIXIE_FEAS_TOL) {
                return PIXIE_STATUS_INFEASIBLE;
            }
        } else if (pixie_bound_tighten_lower(lb, bound) && pixie_is_finite(*ub) && *lb > *ub + PIXIE_FEAS_TOL) {
            return PIXIE_STATUS_INFEASIBLE;
        }
    } else {
        if (a > 0.0) {
            if (pixie_bound_tighten_lower(lb, bound) && pixie_is_finite(*ub) && *lb > *ub + PIXIE_FEAS_TOL) {
                return PIXIE_STATUS_INFEASIBLE;
            }
        } else if (pixie_bound_tighten_upper(ub, bound) && *lb > *ub + PIXIE_FEAS_TOL) {
            return PIXIE_STATUS_INFEASIBLE;
        }
    }

    if (pixie_is_finite(*ub) && *lb > *ub + PIXIE_FEAS_TOL) {
        return PIXIE_STATUS_INFEASIBLE;
    }
    return PIXIE_STATUS_OPTIMAL;
}

static PixieStatus pixie_presolve_transformed_lp(PixieConVec *tcons,
                                                 int ny,
                                                 const double *obj_max,
                                                 double deadline,
                                                 bool *timed_out,
                                                 double **fixed_out,
                                                 int **orig_of_work_out,
                                                 double **obj_work_out,
                                                 int *work_ny_out) {
    double *lb = NULL;
    double *ub = NULL;
    double *fixed = NULL;
    int *col_map = NULL;
    int *orig_of_work = NULL;
    int *occ = NULL;
    double *obj_work = NULL;
    bool changed = true;
    int i;
    int j;
    int work_ny = 0;
    PixieStatus ret = PIXIE_STATUS_ERROR;

    if (timed_out != NULL) {
        *timed_out = false;
    }

    lb = (double *)pixie_xcalloc((size_t)ny, sizeof(double));
    ub = (double *)pixie_xmalloc((size_t)ny * sizeof(double));
    fixed = (double *)pixie_xmalloc((size_t)ny * sizeof(double));
    col_map = (int *)pixie_xmalloc((size_t)ny * sizeof(int));
    occ = (int *)pixie_xcalloc((size_t)ny, sizeof(int));
    for (i = 0; i < ny; ++i) {
        ub[i] = HUGE_VAL;
        fixed[i] = NAN;
        col_map[i] = -1;
    }

    while (changed) {
        changed = false;
        for (i = 0; i < tcons->len; ++i) {
            PixieConstraint *c = &tcons->data[i];
            int w = 0;

            if ((i & 255) == 0 && pixie_deadline_reached(deadline)) {
                if (timed_out != NULL) {
                    *timed_out = true;
                }
                ret = PIXIE_STATUS_UNKNOWN;
                goto cleanup;
            }

            if (c->a.len <= 0) {
                if (!pixie_empty_row_is_feasible(c->cmp, c->rhs)) {
                    ret = PIXIE_STATUS_INFEASIBLE;
                    goto cleanup;
                }
                if (c->rhs != 0.0 || c->cmp != PIXIE_CMP_LE) {
                    pixie_mark_constraint_redundant(c);
                    changed = true;
                }
                continue;
            }

            for (j = 0; j < c->a.len; ++j) {
                int v = c->a.idx[j];
                double val = c->a.val[j];
                if (fixed[v] == fixed[v]) {
                    c->rhs -= val * fixed[v];
                    changed = true;
                } else {
                    c->a.idx[w] = v;
                    c->a.val[w] = val;
                    ++w;
                }
            }
            c->a.len = w;
            pixie_sparse_prune(&c->a, PIXIE_EPS);

            if (c->a.len <= 0) {
                if (!pixie_empty_row_is_feasible(c->cmp, c->rhs)) {
                    ret = PIXIE_STATUS_INFEASIBLE;
                    goto cleanup;
                }
                pixie_mark_constraint_redundant(c);
                changed = true;
                continue;
            }

            if (c->a.len == 1) {
                int v = c->a.idx[0];
                PixieStatus st = pixie_tighten_singleton_row(c, &lb[v], &ub[v]);
                if (st != PIXIE_STATUS_OPTIMAL) {
                    ret = st;
                    goto cleanup;
                }
                pixie_mark_constraint_redundant(c);
                changed = true;
            }
        }

        for (i = 0; i < ny; ++i) {
            double val;
            if (fixed[i] == fixed[i]) {
                continue;
            }
            if (pixie_is_finite(ub[i]) && lb[i] > ub[i] + PIXIE_FEAS_TOL) {
                ret = PIXIE_STATUS_INFEASIBLE;
                goto cleanup;
            }
            if (!pixie_is_finite(ub[i])) {
                continue;
            }
            if (ub[i] < 0.0 - PIXIE_FEAS_TOL) {
                ret = PIXIE_STATUS_INFEASIBLE;
                goto cleanup;
            }
            if (lb[i] < 0.0 && lb[i] > -PIXIE_FEAS_TOL) {
                lb[i] = 0.0;
            }
            if (ub[i] < 0.0 && ub[i] > -PIXIE_FEAS_TOL) {
                ub[i] = 0.0;
            }
            if (fabs(ub[i] - lb[i]) <= PIXIE_FEAS_TOL) {
                val = 0.5 * (lb[i] + ub[i]);
                if (val < 0.0 && val > -PIXIE_FEAS_TOL) {
                    val = 0.0;
                }
                fixed[i] = val;
                changed = true;
            }
        }

        if (!changed) {
            memset(occ, 0, (size_t)ny * sizeof(int));
            for (i = 0; i < tcons->len; ++i) {
                PixieConstraint *c = &tcons->data[i];
                if ((i & 255) == 0 && pixie_deadline_reached(deadline)) {
                    if (timed_out != NULL) {
                        *timed_out = true;
                    }
                    ret = PIXIE_STATUS_UNKNOWN;
                    goto cleanup;
                }
                for (j = 0; j < c->a.len; ++j) {
                    ++occ[c->a.idx[j]];
                }
            }
            for (i = 0; i < ny; ++i) {
                double val;
                if (fixed[i] == fixed[i] || occ[i] > 0) {
                    continue;
                }
                if (obj_max[i] > PIXIE_FEAS_TOL) {
                    if (!pixie_is_finite(ub[i])) {
                        ret = PIXIE_STATUS_UNBOUNDED;
                        goto cleanup;
                    }
                    val = ub[i];
                } else {
                    val = lb[i];
                }
                if (val < 0.0 && val > -PIXIE_FEAS_TOL) {
                    val = 0.0;
                }
                if (!pixie_is_finite(val)) {
                    ret = PIXIE_STATUS_UNBOUNDED;
                    goto cleanup;
                }
                fixed[i] = val;
                changed = true;
            }
        }
    }

    for (i = 0; i < ny; ++i) {
        if (fixed[i] != fixed[i]) {
            col_map[i] = work_ny++;
        }
    }

    orig_of_work = (int *)pixie_xmalloc((size_t)((work_ny > 0) ? work_ny : 1) * sizeof(int));
    obj_work = (double *)pixie_xcalloc((size_t)work_ny, sizeof(double));
    for (i = 0; i < ny; ++i) {
        if (col_map[i] >= 0) {
            orig_of_work[col_map[i]] = i;
            obj_work[col_map[i]] = obj_max[i];
        }
    }

    for (i = 0; i < tcons->len; ++i) {
        PixieConstraint *c = &tcons->data[i];
        int w = 0;
        if ((i & 255) == 0 && pixie_deadline_reached(deadline)) {
            if (timed_out != NULL) {
                *timed_out = true;
            }
            ret = PIXIE_STATUS_UNKNOWN;
            goto cleanup;
        }
        if (c->a.len <= 0) {
            continue;
        }
        for (j = 0; j < c->a.len; ++j) {
            int v = c->a.idx[j];
            if (fixed[v] == fixed[v]) {
                c->rhs -= c->a.val[j] * fixed[v];
            } else {
                c->a.idx[w] = col_map[v];
                c->a.val[w] = c->a.val[j];
                ++w;
            }
        }
        c->a.len = w;
        pixie_sparse_prune(&c->a, PIXIE_EPS);
        if (c->a.len <= 0) {
            if (!pixie_empty_row_is_feasible(c->cmp, c->rhs)) {
                ret = PIXIE_STATUS_INFEASIBLE;
                goto cleanup;
            }
            pixie_mark_constraint_redundant(c);
        }
    }

    for (i = 0; i < ny; ++i) {
        PixieConstraint bc;
        if (fixed[i] == fixed[i]) {
            continue;
        }
        if (lb[i] > PIXIE_FEAS_TOL) {
            bc.cmp = PIXIE_CMP_GE;
            bc.rhs = lb[i];
            pixie_sparse_init(&bc.a);
            pixie_sparse_add(&bc.a, col_map[i], 1.0);
            pixie_convec_push(tcons, &bc);
        }
        if (pixie_is_finite(ub[i])) {
            bc.cmp = PIXIE_CMP_LE;
            bc.rhs = ub[i];
            pixie_sparse_init(&bc.a);
            pixie_sparse_add(&bc.a, col_map[i], 1.0);
            pixie_convec_push(tcons, &bc);
        }
    }

    *fixed_out = fixed;
    *orig_of_work_out = orig_of_work;
    *obj_work_out = obj_work;
    *work_ny_out = work_ny;
    free(lb);
    free(ub);
    free(col_map);
    free(occ);
    return PIXIE_STATUS_OPTIMAL;

cleanup:
    free(lb);
    free(ub);
    free(fixed);
    free(col_map);
    free(orig_of_work);
    free(occ);
    free(obj_work);
    return ret;
}

static void pixie_legacy_simplex_free(PixieSimplex *sm) {
    free(sm->raw);
    free(sm->table);
    free(sm->v_idx);
    free(sm->a_flg);
    sm->raw = NULL;
    sm->table = NULL;
    sm->v_idx = NULL;
    sm->a_flg = NULL;
    sm->v_cnt = 0;
    sm->e_cnt = 0;
    sm->s_cnt = 0;
    sm->a_cnt = 0;
}

static void pixie_legacy_standardize_row(const PixieConstraint *c,
                                         bool *flip_sign,
                                         bool *has_slack,
                                         double *slack_coef,
                                         bool *has_artificial) {
    PixieCompare cmp = c->cmp;
    double rhs = c->rhs;
    bool flip = false;

    if (rhs < -PIXIE_FEAS_TOL || (cmp == PIXIE_CMP_GE && rhs <= PIXIE_FEAS_TOL)) {
        flip = true;
        if (cmp == PIXIE_CMP_LE) {
            cmp = PIXIE_CMP_GE;
        } else if (cmp == PIXIE_CMP_GE) {
            cmp = PIXIE_CMP_LE;
        }
    }

    *flip_sign = flip;
    if (cmp == PIXIE_CMP_EQ) {
        *has_slack = false;
        *slack_coef = 0.0;
        *has_artificial = true;
    } else if (cmp == PIXIE_CMP_LE) {
        *has_slack = true;
        *slack_coef = 1.0;
        *has_artificial = false;
    } else {
        *has_slack = true;
        *slack_coef = -1.0;
        *has_artificial = true;
    }
}

static int pixie_legacy_count_slack_artificial(const PixieConstraint *cons, int e_cnt, int *s_out, int *a_out) {
    int i;
    int s_cnt = 0;
    int a_cnt = 0;
    for (i = 0; i < e_cnt; ++i) {
        bool flip = false;
        bool has_slack = false;
        bool has_art = false;
        double slack_coef = 0.0;
        pixie_legacy_standardize_row(&cons[i], &flip, &has_slack, &slack_coef, &has_art);
        (void)flip;
        (void)slack_coef;
        if (has_slack) {
            ++s_cnt;
        }
        if (has_art) {
            ++a_cnt;
        }
    }
    *s_out = s_cnt;
    *a_out = a_cnt;
    return 1;
}

static int pixie_legacy_make_simplex_table(const PixieConstraint *cons, int e_cnt, int v_cnt, const double *obj_max, PixieSimplex *sm) {
    int i;
    int j;
    int k;
    int s_cnt = 0;
    int a_cnt = 0;
    int rows;
    int cols;
    int idx;

    pixie_legacy_count_slack_artificial(cons, e_cnt, &s_cnt, &a_cnt);

    rows = e_cnt + 1;
    cols = v_cnt + s_cnt + a_cnt + 1;

    sm->raw = (double *)pixie_xcalloc((size_t)rows * (size_t)cols, sizeof(double));
    sm->table = (double **)pixie_xmalloc((size_t)rows * sizeof(double *));
    for (i = 0; i < rows; ++i) {
        sm->table[i] = sm->raw + (size_t)i * (size_t)cols;
    }
    sm->v_idx = (int *)pixie_xcalloc((size_t)e_cnt, sizeof(int));
    sm->a_flg = (bool *)pixie_xcalloc((size_t)e_cnt, sizeof(bool));
    sm->v_cnt = v_cnt;
    sm->e_cnt = e_cnt;
    sm->s_cnt = s_cnt;
    sm->a_cnt = a_cnt;

    idx = v_cnt + 1;
    for (i = 0; i < e_cnt; ++i) {
        bool flip = false;
        bool has_slack = false;
        bool has_art = false;
        double slack_coef = 0.0;
        pixie_legacy_standardize_row(&cons[i], &flip, &has_slack, &slack_coef, &has_art);

        sm->table[i][0] = cons[i].rhs;
        for (k = 0; k < cons[i].a.len; ++k) {
            int v = cons[i].a.idx[k];
            if (v >= 0 && v < v_cnt) {
                sm->table[i][v + 1] += cons[i].a.val[k];
            }
        }
        if (flip) {
            for (j = 0; j <= v_cnt; ++j) {
                sm->table[i][j] = -sm->table[i][j];
            }
        }
        if (has_slack) {
            sm->table[i][idx] = slack_coef;
            if (!has_art && slack_coef > 0.0) {
                sm->v_idx[i] = idx - 1;
            }
            ++idx;
        }
        sm->a_flg[i] = has_art;
    }

    for (i = 0; i < e_cnt; ++i) {
        if (sm->a_flg[i]) {
            sm->table[i][idx] = 1.0;
            sm->v_idx[i] = idx - 1;
            ++idx;
        }
    }

    if (a_cnt == 0) {
        for (i = 0; i < v_cnt; ++i) {
            sm->table[e_cnt][i + 1] = -obj_max[i];
        }
    } else {
        for (i = 0; i < e_cnt; ++i) {
            if (!sm->a_flg[i]) {
                continue;
            }
            for (j = 0; j <= v_cnt + s_cnt; ++j) {
                sm->table[e_cnt][j] -= sm->table[i][j];
            }
        }
    }

    return 1;
}

static void pixie_legacy_simplex_pivot_at(PixieSimplex *sm, int pivot_row, int col) {
    int total_columns = sm->v_cnt + sm->s_cnt + sm->a_cnt;
    int i;
    int j;
    double piv = sm->table[pivot_row][col];
    double inv_piv = 1.0 / piv;

    sm->v_idx[pivot_row] = col - 1;
    for (j = 0; j <= total_columns; ++j) {
        sm->table[pivot_row][j] *= inv_piv;
    }
    for (i = 0; i <= sm->e_cnt; ++i) {
        double factor;
        if (i == pivot_row) {
            continue;
        }
        factor = sm->table[i][col];
        if (fabs(factor) <= PIXIE_EPS) {
            continue;
        }
        for (j = 0; j <= total_columns; ++j) {
            sm->table[i][j] -= factor * sm->table[pivot_row][j];
        }
    }
}

static void pixie_legacy_simplex_remove_artificial_basics(PixieSimplex *sm, int enter_columns) {
    int art_first = sm->v_cnt + sm->s_cnt;
    int total_columns = sm->v_cnt + sm->s_cnt + sm->a_cnt;
    int i;

    if (enter_columns <= 0 || enter_columns > total_columns) {
        enter_columns = total_columns;
    }

    for (i = 0; i < sm->e_cnt; ++i) {
        int j;
        int best_col = -1;
        double best_abs = 0.0;
        if (sm->v_idx[i] < art_first) {
            continue;
        }
        if (fabs(sm->table[i][0]) > PIXIE_FEAS_TOL) {
            continue;
        }
        for (j = 1; j <= enter_columns; ++j) {
            double a = sm->table[i][j];
            double aa = fabs(a);
            if (aa <= PIXIE_PIV_TOL) {
                continue;
            }
            if (best_col < 0 || aa > best_abs + PIXIE_EPS) {
                best_col = j;
                best_abs = aa;
            }
        }
        if (best_col >= 0) {
            pixie_legacy_simplex_pivot_at(sm, i, best_col);
        }
    }
}

static void pixie_legacy_change_z_param(PixieSimplex *sm, int v_cnt, const double *obj_max) {
    int i;
    int j;
    int s_cnt = sm->s_cnt;
    int e_cnt = sm->e_cnt;
    for (j = 0; j <= v_cnt + s_cnt; ++j) {
        sm->table[e_cnt][j] = 0.0;
        if (j >= 1 && j <= v_cnt) {
            sm->table[e_cnt][j] = -obj_max[j - 1];
        }
        for (i = 0; i < e_cnt; ++i) {
            if (sm->v_idx[i] < v_cnt) {
                sm->table[e_cnt][j] += obj_max[sm->v_idx[i]] * sm->table[i][j];
            }
        }
    }
}

static PixieSimplexStatus pixie_legacy_simplex_solve(PixieSimplex *sm, int enter_columns) {
    int simplex_column = sm->v_cnt + sm->s_cnt + sm->a_cnt;
    int total_columns = simplex_column;
    long long iter = 0;
    long long max_iter = 1000000LL + 1000LL * (long long)(sm->e_cnt + 1) * (long long)(simplex_column + 1);
    if (enter_columns <= 0 || enter_columns > total_columns) {
        enter_columns = total_columns;
    }
    while (1) {
        int col = -1;
        int i;
        int j;
        int pivot_row = -1;
        double best_ratio = 0.0;
        int best_basis = INT_MAX;
        double best_reduced = -PIXIE_PIV_TOL;

        if (++iter > max_iter) {
            return PIXIE_SIMPLEX_NUMERIC;
        }

        for (j = 1; j <= enter_columns; ++j) {
            if (sm->table[sm->e_cnt][j] < best_reduced) {
                best_reduced = sm->table[sm->e_cnt][j];
                col = j;
            }
        }
        if (col < 0) {
            return PIXIE_SIMPLEX_OPTIMAL;
        }

        for (i = 0; i < sm->e_cnt; ++i) {
            double a = sm->table[i][col];
            double rhs = sm->table[i][0];
            double ratio;
            if (a <= PIXIE_PIV_TOL) {
                continue;
            }
            ratio = rhs / a;
            if (ratio < -PIXIE_FEAS_TOL) {
                continue;
            }
            if (ratio < 0.0) {
                ratio = 0.0;
            }
            if (pivot_row < 0 ||
                ratio < best_ratio - PIXIE_FEAS_TOL ||
                (fabs(ratio - best_ratio) <= PIXIE_FEAS_TOL && sm->v_idx[i] < best_basis)) {
                pivot_row = i;
                best_ratio = ratio;
                best_basis = sm->v_idx[i];
            }
        }

        if (pivot_row < 0) {
            return PIXIE_SIMPLEX_UNBOUNDED;
        }

        pixie_legacy_simplex_pivot_at(sm, pivot_row, col);
    }
}

static bool pixie_legacy_simplex_is_safe(const PixieSimplex *sm) {
    return sm->table[sm->e_cnt][0] >= -PIXIE_FEAS_TOL;
}

static PixieStatus pixie_legacy_solve_transformed(const PixieConVec *tcons,
                                                  int ny,
                                                  const double *obj_max,
                                                  double *y_out) {
    PixieSimplex sm;
    PixieSimplexStatus ss;
    int i;
    PixieStatus ret = PIXIE_STATUS_ERROR;

    memset(&sm, 0, sizeof(sm));
    if (!pixie_legacy_make_simplex_table(tcons->data, tcons->len, ny, obj_max, &sm)) {
        return PIXIE_STATUS_ERROR;
    }

    if (sm.a_cnt > 0) {
        ss = pixie_legacy_simplex_solve(&sm, ny + sm.s_cnt + sm.a_cnt);
        if (ss != PIXIE_SIMPLEX_OPTIMAL) {
            ret = (ss == PIXIE_SIMPLEX_UNBOUNDED) ? PIXIE_STATUS_UNBOUNDED : PIXIE_STATUS_UNKNOWN;
            goto cleanup;
        }
        if (!pixie_legacy_simplex_is_safe(&sm)) {
            ret = PIXIE_STATUS_INFEASIBLE;
            goto cleanup;
        }
        pixie_legacy_simplex_remove_artificial_basics(&sm, ny + sm.s_cnt);
        pixie_legacy_change_z_param(&sm, ny, obj_max);
        ss = pixie_legacy_simplex_solve(&sm, ny + sm.s_cnt);
        if (ss == PIXIE_SIMPLEX_UNBOUNDED) {
            ret = PIXIE_STATUS_UNBOUNDED;
            goto cleanup;
        }
        if (ss != PIXIE_SIMPLEX_OPTIMAL) {
            ret = PIXIE_STATUS_UNKNOWN;
            goto cleanup;
        }
    } else {
        ss = pixie_legacy_simplex_solve(&sm, ny + sm.s_cnt);
        if (ss == PIXIE_SIMPLEX_UNBOUNDED) {
            ret = PIXIE_STATUS_UNBOUNDED;
            goto cleanup;
        }
        if (ss != PIXIE_SIMPLEX_OPTIMAL) {
            ret = PIXIE_STATUS_UNKNOWN;
            goto cleanup;
        }
    }

    for (i = 0; i < ny; ++i) {
        y_out[i] = 0.0;
    }
    for (i = 0; i < sm.e_cnt; ++i) {
        if (sm.v_idx[i] >= 0 && sm.v_idx[i] < ny) {
            y_out[sm.v_idx[i]] = sm.table[i][0];
        }
    }
    ret = PIXIE_STATUS_OPTIMAL;

cleanup:
    pixie_legacy_simplex_free(&sm);
    return ret;
}

static bool pixie_deadline_reached(double deadline) {
    return deadline > 0.0 && pixie_wall_seconds() >= deadline;
}

static void pixie_lp_init(PixieLPState *s) {
    memset(s, 0, sizeof(*s));
}

static void pixie_lp_rebind_rows(PixieLPState *s) {
    int i;
    int rows = s->m + 2;
    int cols = s->n + 2;
    for (i = 0; i < rows; ++i) {
        s->D[i] = s->buf + (size_t)i * (size_t)cols;
    }
}

static void pixie_lp_free(PixieLPState *s) {
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

static bool pixie_lp_build(PixieLPState *s, int m, int n, const double *A, const double *b, const double *c) {
    int i;
    int j;
    int rows = m + 2;
    int cols = n + 2;
    int block_cols = (n + 1 + 255) / 256;
    int block_rows = (m + 255) / 256;
    int candidate_cap = (block_cols > block_rows) ? block_cols : block_rows;
    char cuda_err[256];
    s->m = m;
    s->n = n;

    s->D = (double **)pixie_xmalloc((size_t)rows * sizeof(double *));
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
            fprintf(stderr, "c pixie accel error: %s\n",
                    (cuda_err[0] != '\0') ? cuda_err : "failed to allocate CUDA simplex buffers");
            s->gpu_error = true;
            pixie_lp_free(s);
            return false;
        }
        s->gpu_candidate_cap = (candidate_cap > 0) ? candidate_cap : 1;
        memset(s->gpu_candidate_idx, 0, (size_t)s->gpu_candidate_cap * sizeof(int));
        memset(s->gpu_candidate_metric, 0, (size_t)s->gpu_candidate_cap * sizeof(double));
        memset(s->buf, 0, (size_t)rows * (size_t)cols * sizeof(double));
        memset(s->buf_alt, 0, (size_t)rows * (size_t)cols * sizeof(double));
    } else {
        s->B = (int *)pixie_xmalloc((size_t)((m > 0) ? m : 1) * sizeof(int));
        s->N = (int *)pixie_xmalloc((size_t)(n + 1) * sizeof(int));
        s->gpu_candidate_cap = (candidate_cap > 0) ? candidate_cap : 1;
        s->gpu_candidate_idx = (int *)pixie_xcalloc((size_t)s->gpu_candidate_cap, sizeof(int));
        s->gpu_candidate_metric = (double *)pixie_xcalloc((size_t)s->gpu_candidate_cap, sizeof(double));
        s->buf = (double *)pixie_xcalloc((size_t)rows * (size_t)cols, sizeof(double));
        s->buf_alt = (double *)pixie_xcalloc((size_t)rows * (size_t)cols, sizeof(double));
    }

    pixie_lp_rebind_rows(s);

    for (i = 0; i < m; ++i) {
        if ((i & 31) == 0 && pixie_deadline_reached(s->deadline)) {
            s->aborted = true;
            pixie_lp_free(s);
            return false;
        }
        for (j = 0; j < n; ++j) {
            s->D[i][j] = A[(size_t)i * (size_t)n + (size_t)j];
        }
    }

    for (i = 0; i < m; ++i) {
        if ((i & 31) == 0 && pixie_deadline_reached(s->deadline)) {
            s->aborted = true;
            pixie_lp_free(s);
            return false;
        }
        s->B[i] = n + i;
        s->D[i][n] = -1.0;
        s->D[i][n + 1] = b[i];
    }

    for (j = 0; j < n; ++j) {
        if ((j & 255) == 0 && pixie_deadline_reached(s->deadline)) {
            s->aborted = true;
            pixie_lp_free(s);
            return false;
        }
        s->N[j] = j;
        s->D[m][j] = -c[j];
    }
    s->N[n] = -1;
    s->D[m + 1][n] = 1.0;

    return true;
}

static void pixie_lp_pivot(PixieLPState *s, int r, int c) {
    int i;
    int j;
    int m = s->m;
    int n = s->n;
    if (s->use_cuda) {
        char cuda_err[256];
        double *tmp;
        cuda_err[0] = '\0';
        if (!krb_accel_cuda_tableau_pivot(s->buf_alt, s->buf, m + 2, n + 2, r, c, cuda_err, sizeof(cuda_err))) {
            fprintf(stderr, "c pixie accel error: %s\n",
                    (cuda_err[0] != '\0') ? cuda_err : "CUDA pivot update failed");
            s->gpu_error = true;
            s->aborted = true;
            return;
        }
        tmp = s->buf;
        s->buf = s->buf_alt;
        s->buf_alt = tmp;
        pixie_lp_rebind_rows(s);
        {
            int t = s->B[r];
            s->B[r] = s->N[c];
            s->N[c] = t;
        }
        return;
    }

    double piv = s->D[r][c];
    double inv_piv = 1.0 / piv;
    double *Dr = s->D[r];

    for (i = 0; i < m + 2; ++i) {
        double *Di;
        double mult;
        if (i == r) {
            continue;
        }
        if ((i & 31) == 0 && pixie_deadline_reached(s->deadline)) {
            s->aborted = true;
            return;
        }
        Di = s->D[i];
        if (fabs(Di[c]) <= PIXIE_EPS) {
            continue;
        }
        mult = Di[c] * inv_piv;
        for (j = 0; j < n + 2; ++j) {
            if (j == c) {
                continue;
            }
            Di[j] -= Dr[j] * mult;
        }
    }
    for (j = 0; j < n + 2; ++j) {
        if (j != c) {
            Dr[j] *= inv_piv;
        }
    }
    for (i = 0; i < m + 2; ++i) {
        if (i != r) {
            s->D[i][c] *= -inv_piv;
        }
    }
    Dr[c] = inv_piv;

    {
        int t = s->B[r];
        s->B[r] = s->N[c];
        s->N[c] = t;
    }
}

static bool pixie_lp_phase(PixieLPState *s, int phase) {
    int x = (phase == 1) ? s->m + 1 : s->m;
    int m = s->m;
    int n = s->n;
    long long iter = 0;
    int cursor = 0;
    int total_cols = n + 1;

    for (;;) {
        int c = -1;
        int r = -1;
        int i;

        if (((++iter) & 3LL) == 0LL && pixie_deadline_reached(s->deadline)) {
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
                                                  total_cols,
                                                  phase,
                                                  PIXIE_LP_EPS,
                                                  s->gpu_candidate_idx,
                                                  s->gpu_candidate_metric,
                                                  s->gpu_candidate_cap,
                                                  &c,
                                                  cuda_err,
                                                  sizeof(cuda_err))) {
                fprintf(stderr, "c pixie accel error: %s\n",
                        (cuda_err[0] != '\0') ? cuda_err : "failed to price entering column on GPU");
                s->gpu_error = true;
                s->aborted = true;
                return false;
            }
        } else {
            int scanned = 0;
            while (scanned < total_cols) {
                int begin = cursor;
                int end = cursor + PIXIE_PRICING_BLOCK;
                int j;
                if (end > total_cols) {
                    end = total_cols;
                }
                for (j = begin; j < end; ++j) {
                    if (phase == 2 && s->N[j] == -1) {
                        continue;
                    }
                    if (c < 0 || s->D[x][j] < s->D[x][c] - PIXIE_LP_EPS ||
                        (fabs(s->D[x][j] - s->D[x][c]) <= PIXIE_LP_EPS && s->N[j] < s->N[c])) {
                        c = j;
                    }
                }
                scanned += end - begin;
                cursor = (end >= total_cols) ? 0 : end;
                if (c >= 0 && s->D[x][c] < -PIXIE_LP_EPS) {
                    break;
                }
            }
        }

        if (c < 0 || s->D[x][c] > -PIXIE_LP_EPS) {
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
                                                 PIXIE_LP_EPS,
                                                 s->gpu_candidate_idx,
                                                 s->gpu_candidate_metric,
                                                 s->gpu_candidate_cap,
                                                 &r,
                                                 cuda_err,
                                                 sizeof(cuda_err))) {
                fprintf(stderr, "c pixie accel error: %s\n",
                        (cuda_err[0] != '\0') ? cuda_err : "failed to select leaving row on GPU");
                s->gpu_error = true;
                s->aborted = true;
                return false;
            }
        } else {
            for (i = 0; i < m; ++i) {
                if (s->D[i][c] > PIXIE_LP_EPS) {
                    if (r < 0) {
                        r = i;
                    } else {
                        double lhs = s->D[i][n + 1] / s->D[i][c];
                        double rhs = s->D[r][n + 1] / s->D[r][c];
                        if (lhs < rhs - PIXIE_LP_EPS ||
                            (fabs(lhs - rhs) <= PIXIE_LP_EPS && s->B[i] < s->B[r])) {
                            r = i;
                        }
                    }
                }
            }
        }

        if (r < 0) {
            return false;
        }

        pixie_lp_pivot(s, r, c);
        if (s->aborted) {
            return false;
        }
    }
}

static PixieLPStatus pixie_lp_solve_max(int m,
                                        int n,
                                        const double *A,
                                        const double *b,
                                        const double *c,
                                        double deadline,
                                        bool use_cuda,
                                        double *out_obj,
                                        double *x) {
    PixieLPState s;
    int i;

    pixie_lp_init(&s);
    s.deadline = deadline;
    s.use_cuda = use_cuda;

    if (!pixie_lp_build(&s, m, n, A, b, c)) {
        return s.gpu_error ? PIXIE_LP_ERROR : (s.aborted ? PIXIE_LP_ABORTED : PIXIE_LP_ERROR);
    }
    if (m > 0) {
        int r = 0;
        for (i = 1; i < m; ++i) {
            if (s.D[i][n + 1] < s.D[r][n + 1]) {
                r = i;
            }
        }
        if (s.D[r][n + 1] < -PIXIE_LP_EPS) {
            pixie_lp_pivot(&s, r, n);
            if (s.aborted) {
                pixie_lp_free(&s);
                return s.gpu_error ? PIXIE_LP_ERROR : PIXIE_LP_ABORTED;
            }
            if (!pixie_lp_phase(&s, 1) || s.D[m + 1][n + 1] < -PIXIE_LP_EPS) {
                PixieLPStatus st = s.gpu_error ? PIXIE_LP_ERROR : (s.aborted ? PIXIE_LP_ABORTED : PIXIE_LP_INFEASIBLE);
                pixie_lp_free(&s);
                return st;
            }
            if (fabs(s.D[m + 1][n + 1]) > PIXIE_LP_EPS) {
                pixie_lp_free(&s);
                return PIXIE_LP_INFEASIBLE;
            }
            for (i = 0; i < m; ++i) {
                if (s.B[i] == -1) {
                    int j;
                    int ccol = -1;
                    for (j = 0; j <= n; ++j) {
                        if (ccol < 0 || s.D[i][j] < s.D[i][ccol] - PIXIE_LP_EPS ||
                            (fabs(s.D[i][j] - s.D[i][ccol]) <= PIXIE_LP_EPS && s.N[j] < s.N[ccol])) {
                            ccol = j;
                        }
                    }
                    if (ccol >= 0) {
                        pixie_lp_pivot(&s, i, ccol);
                        if (s.aborted) {
                            pixie_lp_free(&s);
                            return s.gpu_error ? PIXIE_LP_ERROR : PIXIE_LP_ABORTED;
                        }
                    }
                }
            }
        }
    }

    if (!pixie_lp_phase(&s, 2)) {
        PixieLPStatus st = s.gpu_error ? PIXIE_LP_ERROR : (s.aborted ? PIXIE_LP_ABORTED : PIXIE_LP_UNBOUNDED);
        pixie_lp_free(&s);
        return st;
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
    pixie_lp_free(&s);
    return PIXIE_LP_OPTIMAL;
}

static PixieLPStatus pixie_dense_lp_dispatch(const PixieOptions *opt,
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
    if (!krb_accel_choose_dense_lp(accel, "pixie", m, n, &decision, err, sizeof(err))) {
        fprintf(stderr, "c pixie accel error: %s\n",
                (err[0] != '\0') ? err : "invalid acceleration configuration");
        return PIXIE_LP_ERROR;
    }
    if (verbose >= 3) {
        krb_accel_log(stderr, "pixie", &decision);
    }
    return pixie_lp_solve_max(m, n, A, b, c, deadline, decision.path == KRB_ACCEL_PATH_CUDA, out_obj, x);
}

static PixieStatus pixie_solve_lp_relaxation(const PixieModel *m,
                                             const PixieOptions *opt,
                                             const double *node_lb,
                                             const double *node_ub,
                                             double deadline,
                                             double *x_out,
                                             double *obj_out_min,
                                             bool *timed_out) {
    int n = m->n_vars;
    int i;
    int k;
    int ny = 0;
    int row_cnt = 0;
    int row = 0;
    double *lb_used = NULL;
    double *ub_used = NULL;
    PixieVarMap *map = NULL;
    PixieConVec tcons;
    PixieConstraint tc;
    double *obj_y = NULL;
    double *obj_max = NULL;
    double *obj_work = NULL;
    double *A = NULL;
    double *b = NULL;
    double *fixed = NULL;
    int *orig_of_work = NULL;
    double *y = NULL;
    double *y_work = NULL;
    double max_obj = 0.0;
    PixieLPStatus lpst = PIXIE_LP_ERROR;
    PixieStatus ret = PIXIE_STATUS_ERROR;
    bool legacy_primal = false;
    int work_ny = 0;

    if (timed_out != NULL) {
        *timed_out = false;
    }

    lb_used = (double *)pixie_xmalloc((size_t)n * sizeof(double));
    ub_used = (double *)pixie_xmalloc((size_t)n * sizeof(double));
    map = (PixieVarMap *)pixie_xmalloc((size_t)n * sizeof(PixieVarMap));
    pixie_convec_init(&tcons);

    for (i = 0; i < n; ++i) {
        double lb = m->vars[i].lb;
        double ub = m->vars[i].ub;
        if (node_lb != NULL) {
            lb = pixie_max(lb, node_lb[i]);
        }
        if (node_ub != NULL) {
            ub = pixie_min(ub, node_ub[i]);
        }
        if (m->vars[i].type == PIXIE_VAR_BIN) {
            lb = pixie_max(lb, 0.0);
            ub = pixie_min(ub, 1.0);
        }
        if (m->vars[i].type != PIXIE_VAR_CONT) {
            if (pixie_is_finite(lb)) {
                lb = ceil(lb - PIXIE_INT_TOL);
            }
            if (pixie_is_finite(ub)) {
                ub = floor(ub + PIXIE_INT_TOL);
            }
        }
        if (pixie_is_finite(lb) && pixie_is_finite(ub) && lb > ub + PIXIE_FEAS_TOL) {
            ret = PIXIE_STATUS_INFEASIBLE;
            goto cleanup;
        }
        lb_used[i] = lb;
        ub_used[i] = ub;
        if (pixie_is_finite(lb)) {
            map[i].kind = PIXIE_MAP_SHIFT_LB;
            map[i].y0 = ny++;
            map[i].y1 = -1;
            map[i].constant = lb;
        } else if (pixie_is_finite(ub)) {
            map[i].kind = PIXIE_MAP_SHIFT_UB;
            map[i].y0 = ny++;
            map[i].y1 = -1;
            map[i].constant = ub;
        } else {
            map[i].kind = PIXIE_MAP_FREE;
            map[i].y0 = ny++;
            map[i].y1 = ny++;
            map[i].constant = 0.0;
        }
    }

    obj_y = (double *)pixie_xcalloc((size_t)ny, sizeof(double));
    obj_max = (double *)pixie_xcalloc((size_t)ny, sizeof(double));

    for (i = 0; i < n; ++i) {
        double cmin = (m->obj_sense == +1) ? m->vars[i].obj : -m->vars[i].obj;
        if (map[i].kind == PIXIE_MAP_SHIFT_LB) {
            obj_y[map[i].y0] += cmin;
        } else if (map[i].kind == PIXIE_MAP_SHIFT_UB) {
            obj_y[map[i].y0] += -cmin;
        } else {
            obj_y[map[i].y0] += cmin;
            obj_y[map[i].y1] += -cmin;
        }
    }
    for (i = 0; i < ny; ++i) {
        obj_max[i] = -obj_y[i];
    }

    for (i = 0; i < m->n_cons; ++i) {
        tc.cmp = m->cons[i].cmp;
        tc.rhs = m->cons[i].rhs;
        pixie_sparse_init(&tc.a);
        for (k = 0; k < m->cons[i].a.len; ++k) {
            int v = m->cons[i].a.idx[k];
            double a = m->cons[i].a.val[k];
            if (map[v].kind == PIXIE_MAP_SHIFT_LB) {
                pixie_sparse_add(&tc.a, map[v].y0, a);
                tc.rhs -= a * map[v].constant;
            } else if (map[v].kind == PIXIE_MAP_SHIFT_UB) {
                pixie_sparse_add(&tc.a, map[v].y0, -a);
                tc.rhs -= a * map[v].constant;
            } else {
                pixie_sparse_add(&tc.a, map[v].y0, a);
                pixie_sparse_add(&tc.a, map[v].y1, -a);
            }
        }
        pixie_sparse_prune(&tc.a, PIXIE_EPS);
        pixie_convec_push(&tcons, &tc);
    }

    for (i = 0; i < n; ++i) {
        if (map[i].kind == PIXIE_MAP_SHIFT_LB && pixie_is_finite(ub_used[i])) {
            double rhs = ub_used[i] - lb_used[i];
            if (rhs < -PIXIE_FEAS_TOL) {
                ret = PIXIE_STATUS_INFEASIBLE;
                goto cleanup;
            }
            tc.cmp = PIXIE_CMP_LE;
            tc.rhs = rhs;
            pixie_sparse_init(&tc.a);
            pixie_sparse_add(&tc.a, map[i].y0, 1.0);
            pixie_convec_push(&tcons, &tc);
        }
    }

    ret = pixie_presolve_transformed_lp(&tcons,
                                        ny,
                                        obj_max,
                                        deadline,
                                        timed_out,
                                        &fixed,
                                        &orig_of_work,
                                        &obj_work,
                                        &work_ny);
    if (ret != PIXIE_STATUS_OPTIMAL) {
        goto cleanup;
    }

    y = (double *)pixie_xcalloc((size_t)ny, sizeof(double));
    y_work = (double *)pixie_xcalloc((size_t)work_ny, sizeof(double));
    for (i = 0; i < ny; ++i) {
        if (fixed[i] == fixed[i]) {
            y[i] = fixed[i];
        }
    }

    for (i = 0; i < tcons.len; ++i) {
        const PixieConstraint *c = &tcons.data[i];
        if ((i & 255) == 0 && pixie_deadline_reached(deadline)) {
            if (timed_out != NULL) {
                *timed_out = true;
            }
            ret = PIXIE_STATUS_UNKNOWN;
            goto cleanup;
        }
        if (c->a.len == 0) {
            if (c->cmp == PIXIE_CMP_EQ) {
                if (fabs(c->rhs) > PIXIE_FEAS_TOL) {
                    ret = PIXIE_STATUS_INFEASIBLE;
                    goto cleanup;
                }
            } else if (c->cmp == PIXIE_CMP_LE) {
                if (0.0 > c->rhs + PIXIE_FEAS_TOL) {
                    ret = PIXIE_STATUS_INFEASIBLE;
                    goto cleanup;
                }
            } else if (0.0 < c->rhs - PIXIE_FEAS_TOL) {
                ret = PIXIE_STATUS_INFEASIBLE;
                goto cleanup;
            }
            continue;
        }
        row_cnt += (c->cmp == PIXIE_CMP_EQ) ? 2 : 1;
    }

    if ((opt == NULL || opt->accel.mode != KRB_ACCEL_MODE_ON) &&
        work_ny <= 1024 && row_cnt <= 512) {
        ret = pixie_legacy_solve_transformed(&tcons, work_ny, obj_work, y_work);
        if (ret == PIXIE_STATUS_OPTIMAL) {
            legacy_primal = true;
            goto finish_lp;
        }
        if (pixie_deadline_reached(deadline)) {
            if (timed_out != NULL) {
                *timed_out = true;
            }
            ret = PIXIE_STATUS_UNKNOWN;
            goto cleanup;
        }
        memset(y_work, 0, (size_t)work_ny * sizeof(double));
        for (i = 0; i < work_ny; ++i) {
            y[orig_of_work[i]] = 0.0;
        }
    }
solve_dense_lp:
    A = (double *)pixie_xcalloc((size_t)row_cnt * (size_t)work_ny, sizeof(double));
    b = (double *)pixie_xcalloc((size_t)row_cnt, sizeof(double));

    for (i = 0; i < tcons.len; ++i) {
        const PixieConstraint *c = &tcons.data[i];
        if ((i & 255) == 0 && pixie_deadline_reached(deadline)) {
            if (timed_out != NULL) {
                *timed_out = true;
            }
            ret = PIXIE_STATUS_UNKNOWN;
            goto cleanup;
        }
        if (c->a.len == 0) {
            continue;
        }
        if (c->cmp == PIXIE_CMP_LE || c->cmp == PIXIE_CMP_EQ) {
            for (k = 0; k < c->a.len; ++k) {
                A[(size_t)row * (size_t)work_ny + (size_t)c->a.idx[k]] = c->a.val[k];
            }
            b[row] = c->rhs;
            ++row;
        }
        if (c->cmp == PIXIE_CMP_GE || c->cmp == PIXIE_CMP_EQ) {
            for (k = 0; k < c->a.len; ++k) {
                A[(size_t)row * (size_t)work_ny + (size_t)c->a.idx[k]] = -c->a.val[k];
            }
            b[row] = -c->rhs;
            ++row;
        }
    }

    lpst = pixie_dense_lp_dispatch(opt, row_cnt, work_ny, A, b, obj_work, deadline, &max_obj, y_work);
    if (lpst == PIXIE_LP_ABORTED) {
        if (timed_out != NULL) {
            *timed_out = true;
        }
        ret = PIXIE_STATUS_UNKNOWN;
        goto cleanup;
    }
    if (lpst == PIXIE_LP_INFEASIBLE) {
        ret = PIXIE_STATUS_INFEASIBLE;
        goto cleanup;
    }
    if (lpst == PIXIE_LP_UNBOUNDED) {
        ret = PIXIE_STATUS_UNBOUNDED;
        goto cleanup;
    }
    if (lpst != PIXIE_LP_OPTIMAL) {
        ret = PIXIE_STATUS_UNKNOWN;
        goto cleanup;
    }

    ret = PIXIE_STATUS_OPTIMAL;

finish_lp:
    (void)max_obj;
    if (ret != PIXIE_STATUS_OPTIMAL) {
        goto cleanup;
    }
    for (i = 0; i < work_ny; ++i) {
        y[orig_of_work[i]] = y_work[i];
    }
    for (i = 0; i < n; ++i) {
        double xv;
        double tol_lb = 0.0;
        double tol_ub = 0.0;
        if (map[i].kind == PIXIE_MAP_SHIFT_LB) {
            xv = map[i].constant + y[map[i].y0];
        } else if (map[i].kind == PIXIE_MAP_SHIFT_UB) {
            xv = map[i].constant - y[map[i].y0];
        } else {
            xv = y[map[i].y0] - y[map[i].y1];
        }
        if (pixie_is_finite(lb_used[i])) {
            tol_lb = PIXIE_CERT_TOL * (1.0 + fabs(lb_used[i]));
        }
        if (pixie_is_finite(ub_used[i])) {
            tol_ub = PIXIE_CERT_TOL * (1.0 + fabs(ub_used[i]));
        }
        if (pixie_is_finite(lb_used[i]) && xv < lb_used[i] && xv > lb_used[i] - tol_lb) {
            xv = lb_used[i];
        }
        if (pixie_is_finite(ub_used[i]) && xv > ub_used[i] && xv < ub_used[i] + tol_ub) {
            xv = ub_used[i];
        }
        x_out[i] = xv;
    }

    *obj_out_min = 0.0;
    for (i = 0; i < n; ++i) {
        double cmin = (m->obj_sense == +1) ? m->vars[i].obj : -m->vars[i].obj;
        *obj_out_min += cmin * x_out[i];
    }
    {
        bool check_timed_out = false;
        if (!pixie_lp_candidate_feasible(m, x_out, lb_used, ub_used, deadline, &check_timed_out)) {
            if (check_timed_out) {
                if (timed_out != NULL) {
                    *timed_out = true;
                }
                ret = PIXIE_STATUS_UNKNOWN;
                goto cleanup;
            }
            if (legacy_primal) {
                legacy_primal = false;
                ret = PIXIE_STATUS_ERROR;
                memset(y_work, 0, (size_t)work_ny * sizeof(double));
                for (i = 0; i < work_ny; ++i) {
                    y[orig_of_work[i]] = 0.0;
                }
                goto solve_dense_lp;
            }
            ret = PIXIE_STATUS_UNKNOWN;
            goto cleanup;
        }
    }

cleanup:
    free(lb_used);
    free(ub_used);
    free(map);
    pixie_convec_free(&tcons);
    free(obj_y);
    free(obj_max);
    free(obj_work);
    free(A);
    free(b);
    free(fixed);
    free(orig_of_work);
    free(y);
    free(y_work);
    return ret;
}

static void pixie_nodestack_init(PixieNodeStack *s) {
    s->data = NULL;
    s->len = 0;
    s->cap = 0;
}

static void pixie_nodestack_free(PixieNodeStack *s) {
    int i;
    for (i = 0; i < s->len; ++i) {
        free(s->data[i].lb);
        free(s->data[i].ub);
    }
    free(s->data);
    s->data = NULL;
    s->len = 0;
    s->cap = 0;
}

static void pixie_nodestack_push_copy(PixieNodeStack *s, const double *lb, const double *ub, int n, double bound, int depth) {
    int nc;
    PixieNode *nd;
    if (s->len >= s->cap) {
        nc = (s->cap <= 0) ? 8 : s->cap;
        while (nc <= s->len) {
            if (nc > INT_MAX / 2) {
                nc = s->len + 1;
                break;
            }
            nc *= 2;
        }
        s->data = (PixieNode *)pixie_xrealloc(s->data, (size_t)nc * sizeof(PixieNode));
        s->cap = nc;
    }
    nd = &s->data[s->len];
    nd->lb = (double *)pixie_xmalloc((size_t)n * sizeof(double));
    nd->ub = (double *)pixie_xmalloc((size_t)n * sizeof(double));
    memcpy(nd->lb, lb, (size_t)n * sizeof(double));
    memcpy(nd->ub, ub, (size_t)n * sizeof(double));
    nd->bound = bound;
    nd->depth = depth;
    ++s->len;
}

static void pixie_nodestack_push_owned(PixieNodeStack *s, PixieNode *node) {
    int nc;
    if (s->len >= s->cap) {
        nc = (s->cap <= 0) ? 8 : s->cap;
        while (nc <= s->len) {
            if (nc > INT_MAX / 2) {
                nc = s->len + 1;
                break;
            }
            nc *= 2;
        }
        s->data = (PixieNode *)pixie_xrealloc(s->data, (size_t)nc * sizeof(PixieNode));
        s->cap = nc;
    }
    s->data[s->len++] = *node;
    node->lb = NULL;
    node->ub = NULL;
    node->bound = 0.0;
    node->depth = 0;
}

static PixieNode pixie_nodestack_pop(PixieNodeStack *s) {
    PixieNode nd;
    nd.lb = NULL;
    nd.ub = NULL;
    nd.bound = 0.0;
    nd.depth = 0;
    if (s->len <= 0) {
        return nd;
    }
    --s->len;
    nd = s->data[s->len];
    return nd;
}

static double pixie_nodestack_min_bound(const PixieNodeStack *s) {
    int i;
    double b = HUGE_VAL;
    for (i = 0; i < s->len; ++i) {
        if (s->data[i].bound < b) {
            b = s->data[i].bound;
        }
    }
    return b;
}

static PixieSolution pixie_solution_create(int n);
static void pixie_solution_free(PixieSolution *sol);

static void pixie_frontier_split_init(PixieFrontierSplit *split, int nvars) {
    pixie_nodestack_init(&split->tasks);
    split->seed_solution = pixie_solution_create(nvars);
    split->timed_out = false;
    split->saw_unbounded_relax = false;
}

static void pixie_frontier_split_free(PixieFrontierSplit *split) {
    pixie_nodestack_free(&split->tasks);
    pixie_solution_free(&split->seed_solution);
    split->timed_out = false;
    split->saw_unbounded_relax = false;
}

static bool pixie_is_integral(double x) {
    return fabs(x - nearbyint(x)) <= PIXIE_INT_TOL;
}

static int pixie_choose_branch_var(const PixieModel *m, const double *x, bool use_seed, uint64_t *rng_state, int *frac_cnt_out) {
    int i;
    int pick = -1;
    int frac_cnt = 0;
    int tie_cnt = 0;
    double best_score = -1.0;
    for (i = 0; i < m->n_vars; ++i) {
        double xv;
        double f;
        double score;
        if (m->vars[i].type == PIXIE_VAR_CONT) {
            continue;
        }
        xv = x[i];
        if (pixie_is_integral(xv)) {
            continue;
        }
        ++frac_cnt;
        f = fabs(xv - floor(xv));
        if (f > 0.5) {
            f = 1.0 - f;
        }
        score = f;
        if (pick < 0 || score > best_score + 1e-15) {
            pick = i;
            best_score = score;
            tie_cnt = 1;
        } else if (fabs(score - best_score) <= 1e-15) {
            if (use_seed) {
                ++tie_cnt;
                if ((pixie_rng_next(rng_state) % (uint64_t)tie_cnt) == 0ULL) {
                    pick = i;
                }
            } else if (i < pick) {
                pick = i;
            }
        }
    }
    if (frac_cnt_out != NULL) {
        *frac_cnt_out = frac_cnt;
    }
    return pick;
}

static bool pixie_try_rounding_heuristic(const PixieModel *m,
                                         const double *node_lb,
                                         const double *node_ub,
                                         const double *lp_x,
                                         double *cand_x,
                                         double *obj_min_out) {
    int i;
    int k;
    for (i = 0; i < m->n_vars; ++i) {
        double lb = m->vars[i].lb;
        double ub = m->vars[i].ub;
        double xv = lp_x[i];
        if (node_lb != NULL) {
            lb = pixie_max(lb, node_lb[i]);
        }
        if (node_ub != NULL) {
            ub = pixie_min(ub, node_ub[i]);
        }
        if (m->vars[i].type == PIXIE_VAR_BIN) {
            lb = pixie_max(lb, 0.0);
            ub = pixie_min(ub, 1.0);
        }
        if (m->vars[i].type != PIXIE_VAR_CONT) {
            if (pixie_is_finite(lb)) {
                lb = ceil(lb - PIXIE_INT_TOL);
            }
            if (pixie_is_finite(ub)) {
                ub = floor(ub + PIXIE_INT_TOL);
            }
        }
        if (pixie_is_finite(lb) && pixie_is_finite(ub) && lb > ub + PIXIE_FEAS_TOL) {
            return false;
        }
        if (pixie_is_finite(lb) && xv < lb) {
            xv = lb;
        }
        if (pixie_is_finite(ub) && xv > ub) {
            xv = ub;
        }
        if (m->vars[i].type != PIXIE_VAR_CONT) {
            xv = nearbyint(xv);
            if (pixie_is_finite(lb) && xv < lb) {
                xv = lb;
            }
            if (pixie_is_finite(ub) && xv > ub) {
                xv = ub;
            }
            if (!pixie_is_integral(xv)) {
                return false;
            }
        }
        cand_x[i] = xv;
    }

    for (i = 0; i < m->n_cons; ++i) {
        double lhs = 0.0;
        double rhs = m->cons[i].rhs;
        double tol = 1e-6 + 1e-9 * fabs(rhs);
        for (k = 0; k < m->cons[i].a.len; ++k) {
            int v = m->cons[i].a.idx[k];
            lhs += m->cons[i].a.val[k] * cand_x[v];
        }
        if (m->cons[i].cmp == PIXIE_CMP_EQ) {
            if (fabs(lhs - rhs) > tol) {
                return false;
            }
        } else if (m->cons[i].cmp == PIXIE_CMP_LE) {
            if (lhs > rhs + tol) {
                return false;
            }
        } else {
            if (lhs < rhs - tol) {
                return false;
            }
        }
    }

    *obj_min_out = 0.0;
    for (i = 0; i < m->n_vars; ++i) {
        double cmin = (m->obj_sense == +1) ? m->vars[i].obj : -m->vars[i].obj;
        *obj_min_out += cmin * cand_x[i];
    }
    return true;
}

static double pixie_obj_out_from_min(const PixieModel *m, double obj_min) {
    return (m->obj_sense == +1) ? obj_min : -obj_min;
}

static PixieSolution pixie_solution_create(int n) {
    PixieSolution sol;
    sol.status = PIXIE_STATUS_UNKNOWN;
    sol.obj_min = NAN;
    sol.obj_out = NAN;
    sol.x = (double *)pixie_xcalloc((size_t)n, sizeof(double));
    sol.n = n;
    sol.has_primal = false;
    sol.nodes_processed = 0;
    sol.stopped_time = false;
    sol.stopped_nodes = false;
    sol.stopped_gap = false;
    sol.saw_unbounded_relax = false;
    return sol;
}

static void pixie_solution_free(PixieSolution *sol) {
    free(sol->x);
    sol->x = NULL;
    sol->n = 0;
}

static void pixie_solution_move(PixieSolution *dst, PixieSolution *src) {
    *dst = *src;
    src->x = NULL;
    src->n = 0;
}

static void pixie_solution_update_best(PixieSolution *dst,
                                       const PixieModel *m,
                                       double obj_min,
                                       const double *x) {
    if (!dst->has_primal || obj_min < dst->obj_min - PIXIE_FEAS_TOL) {
        dst->obj_min = obj_min;
        dst->obj_out = pixie_obj_out_from_min(m, obj_min);
        memcpy(dst->x, x, (size_t)dst->n * sizeof(double));
        dst->has_primal = true;
    }
}

static PixieSolution pixie_solve_from_stack(const PixieModel *m,
                                            const PixieOptions *opt,
                                            PixieNodeStack *stack,
                                            double deadline,
                                            double seed_obj,
                                            const double *seed_x,
                                            bool have_seed,
                                            bool honor_node_limit,
                                            bool honor_gap_limit) {
    PixieSolution sol = pixie_solution_create(m->n_vars);
    double *lp_x = (double *)pixie_xmalloc((size_t)m->n_vars * sizeof(double));
    double *heur_x = (double *)pixie_xmalloc((size_t)m->n_vars * sizeof(double));
    double best_obj = seed_obj;
    bool have_inc = have_seed;
    bool saw_unbounded_relax = false;
    uint64_t rng_state = opt->seed_set ? opt->seed : UINT64_C(0x6a09e667f3bcc909);

    if (have_seed && seed_x != NULL) {
        pixie_solution_update_best(&sol, m, seed_obj, seed_x);
    }

    while (stack->len > 0) {
        PixieNode node;
        PixieStatus lps;
        double lp_obj = NAN;
        bool time_stop = false;
        bool node_stop = false;

        if (honor_node_limit && opt->node_limit > 0 && sol.nodes_processed >= opt->node_limit) {
            node_stop = true;
        }
        if (pixie_deadline_reached(deadline)) {
            time_stop = true;
        }
        if (time_stop || node_stop) {
            sol.stopped_time = time_stop;
            sol.stopped_nodes = node_stop;
            break;
        }

        node = pixie_nodestack_pop(stack);
        if (node.lb == NULL || node.ub == NULL) {
            continue;
        }
        if (have_inc && node.bound >= best_obj - PIXIE_FEAS_TOL) {
            free(node.lb);
            free(node.ub);
            continue;
        }

        {
            bool lp_timed_out = false;
            lps = pixie_solve_lp_relaxation(m, opt, node.lb, node.ub, deadline, lp_x, &lp_obj, &lp_timed_out);
            if (lp_timed_out) {
                sol.stopped_time = true;
                free(node.lb);
                free(node.ub);
                break;
            }
        }
        ++sol.nodes_processed;
        if (opt->verbose >= 2 && (sol.nodes_processed % 1000LL) == 0LL) {
            fprintf(stderr, "c nodes=%lld incumbent=%s\n",
                    sol.nodes_processed,
                    have_inc ? "yes" : "no");
        }

        if (lps == PIXIE_STATUS_INFEASIBLE) {
            free(node.lb);
            free(node.ub);
            continue;
        }
        if (lps == PIXIE_STATUS_UNBOUNDED) {
            saw_unbounded_relax = true;
            free(node.lb);
            free(node.ub);
            continue;
        }
        if (lps != PIXIE_STATUS_OPTIMAL) {
            free(node.lb);
            free(node.ub);
            continue;
        }
        if (have_inc && lp_obj >= best_obj - PIXIE_FEAS_TOL) {
            free(node.lb);
            free(node.ub);
            continue;
        }

        {
            int frac_cnt = 0;
            int bvar = pixie_choose_branch_var(m, lp_x, opt->seed_set, &rng_state, &frac_cnt);
            if (bvar < 0) {
                if (!have_inc || lp_obj < best_obj - PIXIE_FEAS_TOL) {
                    best_obj = lp_obj;
                    have_inc = true;
                    pixie_solution_update_best(&sol, m, best_obj, lp_x);
                }
            } else {
                bool pushed = false;
                double v = lp_x[bvar];
                double lo = floor(v);
                double hi = ceil(v);
                double down_gap = v - lo;
                double up_gap = hi - v;
                double cmin = (m->obj_sense == +1) ? m->vars[bvar].obj : -m->vars[bvar].obj;
                bool prefer_down = (down_gap <= up_gap);
                double old_lb = node.lb[bvar];
                double old_ub = node.ub[bvar];
                bool can_down = false;
                bool can_up = false;
                double ub_down = 0.0;
                double lb_up = 0.0;

                if (fabs(down_gap - up_gap) <= PIXIE_INT_TOL) {
                    prefer_down = (cmin >= 0.0);
                }
                if (frac_cnt <= 24 && node.depth <= 8) {
                    double heur_obj = NAN;
                    if (pixie_try_rounding_heuristic(m, node.lb, node.ub, lp_x, heur_x, &heur_obj)) {
                        if (!have_inc || heur_obj < best_obj - PIXIE_FEAS_TOL) {
                            best_obj = heur_obj;
                            have_inc = true;
                            pixie_solution_update_best(&sol, m, best_obj, heur_x);
                        }
                        if (heur_obj <= lp_obj + PIXIE_FEAS_TOL) {
                            pushed = true;
                        }
                    }
                }

                if (lo >= old_lb - PIXIE_FEAS_TOL) {
                    ub_down = pixie_min(old_ub, lo);
                    if (!(pixie_is_finite(old_lb) && pixie_is_finite(ub_down) && old_lb > ub_down + PIXIE_FEAS_TOL)) {
                        can_down = true;
                    }
                }
                if (hi <= old_ub + PIXIE_FEAS_TOL) {
                    lb_up = pixie_max(old_lb, hi);
                    if (!(pixie_is_finite(lb_up) && pixie_is_finite(old_ub) && lb_up > old_ub + PIXIE_FEAS_TOL)) {
                        can_up = true;
                    }
                }

                if (!pushed && (can_down || can_up)) {
                    if (can_down && can_up) {
                        if (prefer_down) {
                            node.lb[bvar] = lb_up;
                            node.ub[bvar] = old_ub;
                            pixie_nodestack_push_copy(stack, node.lb, node.ub, m->n_vars, lp_obj, node.depth + 1);
                            node.lb[bvar] = old_lb;
                            node.ub[bvar] = ub_down;
                            pixie_nodestack_push_copy(stack, node.lb, node.ub, m->n_vars, lp_obj, node.depth + 1);
                        } else {
                            node.lb[bvar] = old_lb;
                            node.ub[bvar] = ub_down;
                            pixie_nodestack_push_copy(stack, node.lb, node.ub, m->n_vars, lp_obj, node.depth + 1);
                            node.lb[bvar] = lb_up;
                            node.ub[bvar] = old_ub;
                            pixie_nodestack_push_copy(stack, node.lb, node.ub, m->n_vars, lp_obj, node.depth + 1);
                        }
                    } else if (can_down) {
                        node.lb[bvar] = old_lb;
                        node.ub[bvar] = ub_down;
                        pixie_nodestack_push_copy(stack, node.lb, node.ub, m->n_vars, lp_obj, node.depth + 1);
                    } else {
                        node.lb[bvar] = lb_up;
                        node.ub[bvar] = old_ub;
                        pixie_nodestack_push_copy(stack, node.lb, node.ub, m->n_vars, lp_obj, node.depth + 1);
                    }
                    pushed = true;
                }

                node.lb[bvar] = old_lb;
                node.ub[bvar] = old_ub;

                if (!pushed && (!have_inc || lp_obj < best_obj - PIXIE_FEAS_TOL)) {
                    best_obj = lp_obj;
                    have_inc = true;
                    pixie_solution_update_best(&sol, m, best_obj, lp_x);
                }
            }
        }

        free(node.lb);
        free(node.ub);

        if (honor_gap_limit && have_inc && opt->gap_limit >= 0.0 && stack->len > 0) {
            double global_lb = pixie_nodestack_min_bound(stack);
            double denom = fabs(best_obj) + 1e-12;
            double gap = HUGE_VAL;
            if (pixie_is_finite(global_lb)) {
                if (global_lb > best_obj) {
                    global_lb = best_obj;
                }
                gap = (best_obj - global_lb) / denom;
            }
            if (gap <= opt->gap_limit + 1e-15) {
                sol.stopped_gap = true;
                break;
            }
        }
    }

    sol.saw_unbounded_relax = saw_unbounded_relax;
    if (have_inc) {
        sol.obj_min = best_obj;
        sol.obj_out = pixie_obj_out_from_min(m, best_obj);
        sol.status = (stack->len == 0 && !sol.stopped_time && !sol.stopped_nodes && !sol.stopped_gap && !saw_unbounded_relax)
                     ? PIXIE_STATUS_OPTIMAL
                     : PIXIE_STATUS_UNKNOWN;
    } else {
        sol.status = (sol.stopped_time || sol.stopped_nodes || sol.stopped_gap)
                     ? PIXIE_STATUS_UNKNOWN
                     : ((stack->len == 0 && !saw_unbounded_relax) ? PIXIE_STATUS_INFEASIBLE : PIXIE_STATUS_UNKNOWN);
    }

    pixie_nodestack_free(stack);
    free(lp_x);
    free(heur_x);
    return sol;
}

static bool pixie_parallel_supported(const PixieOptions *opt,
                                     const KrbParallelRuntime *parallel_rt,
                                     int n_int) {
#if !defined(SATX_HAVE_THREADS)
    (void)opt;
    (void)parallel_rt;
    (void)n_int;
    return false;
#else
    if (parallel_rt == NULL || parallel_rt->resolved_mode != KRB_PARALLEL_MODE_THREADS) return false;
    if (parallel_rt->jobs <= 1 || opt->pure_lp || n_int == 0) return false;
    if (opt->node_limit > 0 || opt->gap_limit >= 0.0) return false;
    return true;
#endif
}

static bool pixie_split_frontier_simple(const PixieModel *m,
                                        const PixieOptions *opt,
                                        const double *root_lb,
                                        const double *root_ub,
                                        double deadline,
                                        PixieFrontierSplit *split) {
    PixieNodeStack stack;
    double *lp_x = (double *)pixie_xmalloc((size_t)m->n_vars * sizeof(double));
    uint64_t rng_state = opt->seed_set ? opt->seed : UINT64_C(0x6a09e667f3bcc909);

    pixie_nodestack_init(&stack);
    pixie_nodestack_push_copy(&stack, root_lb, root_ub, m->n_vars, -HUGE_VAL, 0);

    while (stack.len > 0) {
        PixieNode node = pixie_nodestack_pop(&stack);
        PixieStatus lps;
        double lp_obj = NAN;
        int bvar = -1;

        if (pixie_deadline_reached(deadline)) {
            split->timed_out = true;
            if (node.lb != NULL || node.ub != NULL) {
                free(node.lb);
                free(node.ub);
            }
            break;
        }
        if (node.lb == NULL || node.ub == NULL) {
            continue;
        }
        if (node.depth >= opt->parallel.split_depth) {
            pixie_nodestack_push_owned(&split->tasks, &node);
            continue;
        }

        {
            bool lp_timed_out = false;
            lps = pixie_solve_lp_relaxation(m, opt, node.lb, node.ub, deadline, lp_x, &lp_obj, &lp_timed_out);
            if (lp_timed_out) {
                split->timed_out = true;
                free(node.lb);
                free(node.ub);
                break;
            }
        }
        ++split->seed_solution.nodes_processed;

        if (lps == PIXIE_STATUS_INFEASIBLE) {
            free(node.lb);
            free(node.ub);
            continue;
        }
        if (lps == PIXIE_STATUS_UNBOUNDED) {
            split->saw_unbounded_relax = true;
            free(node.lb);
            free(node.ub);
            continue;
        }
        if (lps != PIXIE_STATUS_OPTIMAL) {
            free(node.lb);
            free(node.ub);
            continue;
        }

        bvar = pixie_choose_branch_var(m, lp_x, opt->seed_set, &rng_state, NULL);
        if (bvar < 0) {
            pixie_solution_update_best(&split->seed_solution, m, lp_obj, lp_x);
            free(node.lb);
            free(node.ub);
            continue;
        }

        {
            double v = lp_x[bvar];
            double lo = floor(v);
            double hi = ceil(v);
            double old_lb = node.lb[bvar];
            double old_ub = node.ub[bvar];
            bool can_down = false;
            bool can_up = false;
            double ub_down = 0.0;
            double lb_up = 0.0;

            if (lo >= old_lb - PIXIE_FEAS_TOL) {
                ub_down = pixie_min(old_ub, lo);
                if (!(pixie_is_finite(old_lb) && pixie_is_finite(ub_down) && old_lb > ub_down + PIXIE_FEAS_TOL)) {
                    can_down = true;
                }
            }
            if (hi <= old_ub + PIXIE_FEAS_TOL) {
                lb_up = pixie_max(old_lb, hi);
                if (!(pixie_is_finite(lb_up) && pixie_is_finite(old_ub) && lb_up > old_ub + PIXIE_FEAS_TOL)) {
                    can_up = true;
                }
            }
            if (can_down) {
                node.lb[bvar] = old_lb;
                node.ub[bvar] = ub_down;
                pixie_nodestack_push_copy(&stack, node.lb, node.ub, m->n_vars, lp_obj, node.depth + 1);
            }
            if (can_up) {
                node.lb[bvar] = lb_up;
                node.ub[bvar] = old_ub;
                pixie_nodestack_push_copy(&stack, node.lb, node.ub, m->n_vars, lp_obj, node.depth + 1);
            }
            node.lb[bvar] = old_lb;
            node.ub[bvar] = old_ub;
            if (!can_down && !can_up) {
                pixie_solution_update_best(&split->seed_solution, m, lp_obj, lp_x);
            }
        }

        free(node.lb);
        free(node.ub);
    }

    if (split->tasks.len == 0) {
        if (split->seed_solution.has_primal && !split->timed_out && !split->saw_unbounded_relax) {
            split->seed_solution.status = PIXIE_STATUS_OPTIMAL;
        } else if (!split->seed_solution.has_primal && !split->timed_out && !split->saw_unbounded_relax) {
            split->seed_solution.status = PIXIE_STATUS_INFEASIBLE;
        } else {
            split->seed_solution.status = PIXIE_STATUS_UNKNOWN;
        }
    } else {
        split->seed_solution.status = PIXIE_STATUS_UNKNOWN;
    }
    split->seed_solution.stopped_time = split->timed_out;
    split->seed_solution.saw_unbounded_relax = split->saw_unbounded_relax;

    pixie_nodestack_free(&stack);
    free(lp_x);
    return !split->timed_out;
}

#if defined(SATX_HAVE_THREADS)
static int pixie_parallel_worker_entry(void *ctx, int worker_id) {
    PixieParallelWork *work = (PixieParallelWork *)ctx;
    PixieOptions worker_opt = *work->opt;
    PixieSolution aggregate = pixie_solution_create(work->nvars);

    (void)worker_id;
    worker_opt.verbose = 0;
    for (;;) {
        int task_index = atomic_fetch_add(&work->next_task, 1);
        PixieNodeStack stack;
        PixieNode node;
        PixieSolution local;

        if (task_index >= work->task_count) {
            break;
        }
        node = work->tasks[task_index];
        work->tasks[task_index].lb = NULL;
        work->tasks[task_index].ub = NULL;

        pixie_nodestack_init(&stack);
        pixie_nodestack_push_owned(&stack, &node);
        local = pixie_solve_from_stack(work->model,
                                       &worker_opt,
                                       &stack,
                                       work->deadline,
                                       work->seed_solution->obj_min,
                                       work->seed_solution->x,
                                       work->seed_solution->has_primal,
                                       false,
                                       false);
        aggregate.nodes_processed += local.nodes_processed;
        aggregate.stopped_time = aggregate.stopped_time || local.stopped_time;
        aggregate.stopped_nodes = aggregate.stopped_nodes || local.stopped_nodes;
        aggregate.stopped_gap = aggregate.stopped_gap || local.stopped_gap;
        aggregate.saw_unbounded_relax = aggregate.saw_unbounded_relax || local.saw_unbounded_relax;
        if (local.has_primal) {
            pixie_solution_update_best(&aggregate, work->model, local.obj_min, local.x);
        }
        pixie_solution_free(&local);
    }

    work->results[worker_id] = aggregate;
    return 0;
}
#endif

static PixieSolution pixie_solve(const PixieModel *m,
                                 const PixieOptions *opt,
                                 const KrbParallelRuntime *parallel_rt) {
    PixieSolution sol = pixie_solution_create(m->n_vars);
    PixieNodeStack stack;
    double *lp_x = (double *)pixie_xmalloc((size_t)m->n_vars * sizeof(double));
    double *root_lb = (double *)pixie_xmalloc((size_t)m->n_vars * sizeof(double));
    double *root_ub = (double *)pixie_xmalloc((size_t)m->n_vars * sizeof(double));
    double lp_obj = NAN;
    double deadline = (opt->time_limit_sec > 0.0) ? (pixie_wall_seconds() + opt->time_limit_sec) : 0.0;
    int n_int = pixie_count_integer_vars(m);
    int i;

    for (i = 0; i < m->n_vars; ++i) {
        root_lb[i] = m->vars[i].lb;
        root_ub[i] = m->vars[i].ub;
        if (m->vars[i].type == PIXIE_VAR_BIN) {
            root_lb[i] = pixie_max(root_lb[i], 0.0);
            root_ub[i] = pixie_min(root_ub[i], 1.0);
        }
    }

    if (opt->pure_lp || n_int == 0) {
        bool lp_timed_out = false;
        PixieStatus lps = pixie_solve_lp_relaxation(m, opt, root_lb, root_ub, deadline, lp_x, &lp_obj, &lp_timed_out);
        sol.nodes_processed = 1;
        if (lps == PIXIE_STATUS_OPTIMAL) {
            sol.status = PIXIE_STATUS_OPTIMAL;
            sol.obj_min = lp_obj;
            sol.obj_out = pixie_obj_out_from_min(m, lp_obj);
            memcpy(sol.x, lp_x, (size_t)m->n_vars * sizeof(double));
            sol.has_primal = true;
        } else if (lps == PIXIE_STATUS_INFEASIBLE) {
            sol.status = PIXIE_STATUS_INFEASIBLE;
        } else if (lps == PIXIE_STATUS_UNBOUNDED) {
            sol.status = PIXIE_STATUS_UNBOUNDED;
        } else {
            sol.stopped_time = lp_timed_out;
            sol.status = PIXIE_STATUS_UNKNOWN;
        }
        free(lp_x);
        free(root_lb);
        free(root_ub);
        return sol;
    }

    #if defined(SATX_HAVE_THREADS)
    if (pixie_parallel_supported(opt, parallel_rt, n_int) && opt->parallel.split_depth > 0) {
        PixieFrontierSplit split;
        pixie_frontier_split_init(&split, m->n_vars);
        if (!pixie_split_frontier_simple(m, opt, root_lb, root_ub, deadline, &split) || split.tasks.len == 0) {
            pixie_solution_free(&sol);
            pixie_solution_move(&sol, &split.seed_solution);
            pixie_frontier_split_free(&split);
            free(lp_x);
            free(root_lb);
            free(root_ub);
            return sol;
        }
        if (split.tasks.len > 1) {
            PixieParallelWork work;
            PixieSolution *results = (PixieSolution *)calloc((size_t)parallel_rt->jobs, sizeof(PixieSolution));
            char par_err[256];
            if (results != NULL) {
                memset(&work, 0, sizeof(work));
                work.model = m;
                work.opt = opt;
                work.deadline = deadline;
                work.tasks = split.tasks.data;
                work.task_count = split.tasks.len;
                work.nvars = m->n_vars;
                work.seed_solution = &split.seed_solution;
                work.results = results;
                atomic_init(&work.next_task, 0);
                par_err[0] = '\0';

                if (krb_parallel_run_threads(parallel_rt->jobs,
                                             pixie_parallel_worker_entry,
                                             &work,
                                             par_err,
                                             sizeof(par_err))) {
                    pixie_solution_free(&sol);
                    sol = pixie_solution_create(m->n_vars);
                    sol.nodes_processed = split.seed_solution.nodes_processed;
                    sol.stopped_time = split.seed_solution.stopped_time;
                    sol.stopped_nodes = split.seed_solution.stopped_nodes;
                    sol.stopped_gap = split.seed_solution.stopped_gap;
                    sol.saw_unbounded_relax = split.seed_solution.saw_unbounded_relax;
                    if (split.seed_solution.has_primal) {
                        pixie_solution_update_best(&sol, m, split.seed_solution.obj_min, split.seed_solution.x);
                    }
                    for (i = 0; i < parallel_rt->jobs; ++i) {
                        sol.nodes_processed += results[i].nodes_processed;
                        sol.stopped_time = sol.stopped_time || results[i].stopped_time;
                        sol.stopped_nodes = sol.stopped_nodes || results[i].stopped_nodes;
                        sol.stopped_gap = sol.stopped_gap || results[i].stopped_gap;
                        sol.saw_unbounded_relax = sol.saw_unbounded_relax || results[i].saw_unbounded_relax;
                        if (results[i].has_primal) {
                            pixie_solution_update_best(&sol, m, results[i].obj_min, results[i].x);
                        }
                        pixie_solution_free(&results[i]);
                    }
                    sol.status = (sol.has_primal && !sol.stopped_time && !sol.stopped_nodes && !sol.stopped_gap && !sol.saw_unbounded_relax)
                                 ? PIXIE_STATUS_OPTIMAL
                                 : ((!sol.has_primal && !sol.stopped_time && !sol.stopped_nodes && !sol.stopped_gap && !sol.saw_unbounded_relax)
                                    ? PIXIE_STATUS_INFEASIBLE
                                    : PIXIE_STATUS_UNKNOWN);
                    free(results);
                    pixie_frontier_split_free(&split);
                    free(lp_x);
                    free(root_lb);
                    free(root_ub);
                    return sol;
                }
                if (opt->verbose >= 1) {
                    fprintf(stderr, "c pixie parallel fallback: %s\n", par_err[0] ? par_err : "thread launch failed");
                }
                free(results);
            }
        }

        stack = split.tasks;
        split.tasks.data = NULL;
        split.tasks.len = 0;
        split.tasks.cap = 0;
        pixie_solution_free(&sol);
        sol = pixie_solve_from_stack(m,
                                     opt,
                                     &stack,
                                     deadline,
                                     split.seed_solution.obj_min,
                                     split.seed_solution.x,
                                     split.seed_solution.has_primal,
                                     false,
                                     false);
        sol.nodes_processed += split.seed_solution.nodes_processed;
        sol.saw_unbounded_relax = sol.saw_unbounded_relax || split.seed_solution.saw_unbounded_relax;
        if (sol.saw_unbounded_relax && !sol.stopped_time && !sol.stopped_nodes && !sol.stopped_gap) {
            sol.status = PIXIE_STATUS_UNKNOWN;
        }
        pixie_solution_free(&split.seed_solution);
    } else
    #endif
    {
        pixie_nodestack_init(&stack);
        pixie_nodestack_push_copy(&stack, root_lb, root_ub, m->n_vars, -HUGE_VAL, 0);
        pixie_solution_free(&sol);
        sol = pixie_solve_from_stack(m, opt, &stack, deadline, HUGE_VAL, NULL, false, true, true);
    }

    free(lp_x);
    free(root_lb);
    free(root_ub);
    return sol;
}

static void pixie_print_solution(const PixieModel *m, const PixieSolution *sol) {
    int i;
    (void)m;
    if (sol->status == PIXIE_STATUS_OPTIMAL) {
        printf("s OPTIMUM FOUND\n");
        printf("o %.17g\n", sol->obj_out);
        printf("v");
        for (i = 0; i < sol->n; ++i) {
            printf(" %.17g", sol->x[i]);
        }
        printf("\n");
        return;
    }
    if (sol->status == PIXIE_STATUS_INFEASIBLE) {
        printf("s INFEASIBLE\n");
        return;
    }
    if (sol->status == PIXIE_STATUS_UNBOUNDED) {
        printf("s UNBOUNDED\n");
        return;
    }
    printf("s UNKNOWN\n");
    if (sol->has_primal) {
        printf("o %.17g\n", sol->obj_out);
    } else {
        printf("o nan\n");
    }
}

static int pixie_selftest_one_lp(void) {
    PixieModel m;
    PixieSparse a;
    PixieOptions opt;
    PixieSolution sol;
    int x;
    int y;

    pixie_model_init(&m);
    x = pixie_model_get_var(&m, "x", true);
    y = pixie_model_get_var(&m, "y", true);
    m.obj_sense = +1;
    m.vars[x].obj = 1.0;
    m.vars[y].obj = 1.0;
    pixie_sparse_init(&a);
    pixie_sparse_add(&a, x, 1.0);
    pixie_sparse_add(&a, y, 1.0);
    pixie_model_add_constraint(&m, &a, PIXIE_CMP_GE, 1.0);
    pixie_sparse_free(&a);

    pixie_options_defaults(&opt);
    sol = pixie_solve(&m, &opt, NULL);
    if (sol.status != PIXIE_STATUS_OPTIMAL || fabs(sol.obj_out - 1.0) > 1e-6) {
        pixie_solution_free(&sol);
        pixie_model_free(&m);
        return 0;
    }
    pixie_solution_free(&sol);
    pixie_model_free(&m);
    return 1;
}

static int pixie_selftest_one_mip(void) {
    PixieModel m;
    PixieSparse a;
    PixieOptions opt;
    PixieSolution sol;
    int x;
    int y;

    pixie_model_init(&m);
    x = pixie_model_get_var(&m, "x", true);
    y = pixie_model_get_var(&m, "y", true);
    m.obj_sense = +1;
    m.vars[x].obj = 1.0;
    m.vars[y].obj = 1.0;
    pixie_var_mark_binary(&m, x);
    pixie_var_mark_binary(&m, y);
    pixie_sparse_init(&a);
    pixie_sparse_add(&a, x, 1.0);
    pixie_sparse_add(&a, y, 1.0);
    pixie_model_add_constraint(&m, &a, PIXIE_CMP_GE, 1.0);
    pixie_sparse_free(&a);

    pixie_options_defaults(&opt);
    sol = pixie_solve(&m, &opt, NULL);
    if ((sol.status != PIXIE_STATUS_OPTIMAL && sol.status != PIXIE_STATUS_UNKNOWN) || !sol.has_primal || fabs(sol.obj_out - 1.0) > 1e-6) {
        pixie_solution_free(&sol);
        pixie_model_free(&m);
        return 0;
    }
    pixie_solution_free(&sol);
    pixie_model_free(&m);
    return 1;
}

static int pixie_selftest_lp_chained_bounds(void) {
    PixieModel m;
    char err[512];
    int section = LP_SEC_NONE;
    int x;
    char bounds_kw[] = "Bounds";
    char bound_stmt[] = "0 <= x <= 5";
    char generals_kw[] = "Generals";
    char general_stmt[] = "x";

    pixie_model_init(&m);
    err[0] = '\0';

    if (pixie_parse_lp_statement(&m, bounds_kw, &section, err, sizeof(err)) != 1 ||
        pixie_parse_lp_statement(&m, bound_stmt, &section, err, sizeof(err)) != 1 ||
        pixie_parse_lp_statement(&m, generals_kw, &section, err, sizeof(err)) != 1 ||
        pixie_parse_lp_statement(&m, general_stmt, &section, err, sizeof(err)) != 1) {
        pixie_model_free(&m);
        return 0;
    }

    x = pixie_model_find_var(&m, "x");
    if (x < 0 ||
        fabs(m.vars[x].lb - 0.0) > 1e-9 ||
        fabs(m.vars[x].ub - 5.0) > 1e-9 ||
        m.vars[x].type != PIXIE_VAR_INT) {
        pixie_model_free(&m);
        return 0;
    }

    pixie_model_free(&m);
    return 1;
}

static int pixie_selftest_eq_redundant(void) {
    PixieModel m;
    PixieSparse a;
    PixieOptions opt;
    PixieSolution sol;
    int x;

    pixie_model_init(&m);
    x = pixie_model_get_var(&m, "x", true);
    m.obj_sense = +1;
    m.vars[x].obj = 1.0;

    pixie_sparse_init(&a);
    pixie_sparse_add(&a, x, 1.0);
    pixie_model_add_constraint(&m, &a, PIXIE_CMP_EQ, 1.0);
    pixie_sparse_free(&a);

    pixie_sparse_init(&a);
    pixie_model_add_constraint(&m, &a, PIXIE_CMP_EQ, 0.0);
    pixie_sparse_free(&a);

    pixie_options_defaults(&opt);
    opt.pure_lp = true;
    sol = pixie_solve(&m, &opt, NULL);
    if (sol.status != PIXIE_STATUS_OPTIMAL || !sol.has_primal ||
        fabs(sol.obj_out - 1.0) > 1e-6 || fabs(sol.x[x] - 1.0) > 1e-6) {
        pixie_solution_free(&sol);
        pixie_model_free(&m);
        return 0;
    }
    pixie_solution_free(&sol);
    pixie_model_free(&m);
    return 1;
}

static int pixie_selftest_infeasible(void) {
    PixieModel m;
    PixieSparse a;
    PixieOptions opt;
    PixieSolution sol;
    int x;

    pixie_model_init(&m);
    x = pixie_model_get_var(&m, "x", true);
    m.obj_sense = +1;
    m.vars[x].obj = 1.0;
    pixie_sparse_init(&a);
    pixie_sparse_add(&a, x, 1.0);
    pixie_model_add_constraint(&m, &a, PIXIE_CMP_LE, -1.0);
    pixie_sparse_free(&a);

    pixie_options_defaults(&opt);
    sol = pixie_solve(&m, &opt, NULL);
    if (sol.status != PIXIE_STATUS_INFEASIBLE) {
        pixie_solution_free(&sol);
        pixie_model_free(&m);
        return 0;
    }
    pixie_solution_free(&sol);
    pixie_model_free(&m);
    return 1;
}

static int pixie_selftest_unbounded(void) {
    PixieModel m;
    PixieOptions opt;
    PixieSolution sol;
    int x;

    pixie_model_init(&m);
    x = pixie_model_get_var(&m, "x", true);
    m.obj_sense = +1;
    m.vars[x].obj = -1.0;

    pixie_options_defaults(&opt);
    opt.pure_lp = true;
    sol = pixie_solve(&m, &opt, NULL);
    if (sol.status != PIXIE_STATUS_UNBOUNDED) {
        pixie_solution_free(&sol);
        pixie_model_free(&m);
        return 0;
    }
    pixie_solution_free(&sol);
    pixie_model_free(&m);
    return 1;
}

static int pixie_run_selftest(void) {
    if (!pixie_selftest_one_lp()) {
        fprintf(stderr, "selftest failed: lp optimum\n");
        return 1;
    }
    if (!pixie_selftest_one_mip()) {
        fprintf(stderr, "selftest failed: mip optimum\n");
        return 1;
    }
    if (!pixie_selftest_lp_chained_bounds()) {
        fprintf(stderr, "selftest failed: lp chained bounds\n");
        return 1;
    }
    if (!pixie_selftest_eq_redundant()) {
        fprintf(stderr, "selftest failed: redundant equality\n");
        return 1;
    }
    if (!pixie_selftest_infeasible()) {
        fprintf(stderr, "selftest failed: infeasible\n");
        return 1;
    }
    if (!pixie_selftest_unbounded()) {
        fprintf(stderr, "selftest failed: unbounded\n");
        return 1;
    }
    fprintf(stderr, "selftest: OK\n");
    return 0;
}

static void pixie_print_usage(const char *prog) {
    fprintf(stderr,
            "usage:\n"
            "  %s <file>\n"
            "  %s --lp <file>\n"
            "  %s --mps <file>\n"
            "options:\n"
            "  --parallel <mode> auto|off|threads|mpi|hybrid\n"
            "  --jobs <n>       local worker count for threaded modes\n"
            "  --split-depth <n> parallel splitting depth\n"
            "  --portfolio <n>  portfolio multiplicity scaffold\n"
            "  --sync-ms <n>    synchronization cadence scaffold\n"
            "  --time <sec>     time limit\n"
            "  --node <n>       node limit\n"
            "  --gap <g>        relative MIP gap limit\n"
            "  --seed <s>       RNG seed (enables randomized branching tie-breaks)\n"
            "  --verbose <k>    verbosity 0..3\n"
            "  --cuda <mode>    auto|on|off (on requires CUDA dense LP backend)\n"
            "  --cuda-device <n> select CUDA device id (-1 uses runtime default)\n"
            "  --cuda-min-cells <n> minimum dense LP cells before trying CUDA\n"
            "  --purelp         solve LP relaxation only\n"
            "  --selftest       run embedded tests\n",
            prog, prog, prog);
}

int pixie_entry(int argc, char **argv) {
    PixieOptions opt;
    KrbParallelRuntime parallel_rt;
    PixieModel model;
    PixieSolution sol;
    char err[512];
    const char *fallback_solver = NULL;
    double deadline = 0.0;
    int n_int;
    int i;

    pixie_options_defaults(&opt);

    for (i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "--selftest") == 0) {
            opt.selftest = true;
        } else if (strcmp(a, "--parallel") == 0) {
            if (i + 1 >= argc || !krb_parallel_parse_mode(argv[i + 1], &opt.parallel.mode)) {
                fprintf(stderr, "invalid --parallel value\n");
                return EXIT_FAILURE;
            }
            ++i;
        } else if (strcmp(a, "--jobs") == 0) {
            long long jobs = 0;
            if (i + 1 >= argc || !pixie_parse_ll_token(argv[i + 1], &jobs) || jobs < 1 || jobs > INT_MAX) {
                fprintf(stderr, "invalid --jobs value\n");
                return EXIT_FAILURE;
            }
            opt.parallel.jobs = (int)jobs;
            ++i;
        } else if (strcmp(a, "--split-depth") == 0) {
            long long depth = 0;
            if (i + 1 >= argc || !pixie_parse_ll_token(argv[i + 1], &depth) || depth < 0 || depth > INT_MAX) {
                fprintf(stderr, "invalid --split-depth value\n");
                return EXIT_FAILURE;
            }
            opt.parallel.split_depth = (int)depth;
            ++i;
        } else if (strcmp(a, "--portfolio") == 0) {
            long long portfolio = 0;
            if (i + 1 >= argc || !pixie_parse_ll_token(argv[i + 1], &portfolio) || portfolio < 1 || portfolio > INT_MAX) {
                fprintf(stderr, "invalid --portfolio value\n");
                return EXIT_FAILURE;
            }
            opt.parallel.portfolio = (int)portfolio;
            ++i;
        } else if (strcmp(a, "--sync-ms") == 0) {
            long long sync_ms = 0;
            if (i + 1 >= argc || !pixie_parse_ll_token(argv[i + 1], &sync_ms) || sync_ms < 0 || sync_ms > INT_MAX) {
                fprintf(stderr, "invalid --sync-ms value\n");
                return EXIT_FAILURE;
            }
            opt.parallel.sync_ms = (int)sync_ms;
            ++i;
        } else if (strcmp(a, "--lp") == 0) {
            if (i + 1 >= argc) {
                pixie_print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            opt.format = PIXIE_FORMAT_LP;
            opt.file = argv[++i];
        } else if (strcmp(a, "--mps") == 0) {
            if (i + 1 >= argc) {
                pixie_print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            opt.format = PIXIE_FORMAT_MPS;
            opt.file = argv[++i];
        } else if (strcmp(a, "--time") == 0) {
            if (i + 1 >= argc || !pixie_parse_double_token(argv[i + 1], &opt.time_limit_sec)) {
                fprintf(stderr, "invalid --time value\n");
                return EXIT_FAILURE;
            }
            ++i;
        } else if (strcmp(a, "--node") == 0) {
            if (i + 1 >= argc || !pixie_parse_ll_token(argv[i + 1], &opt.node_limit)) {
                fprintf(stderr, "invalid --node value\n");
                return EXIT_FAILURE;
            }
            ++i;
        } else if (strcmp(a, "--gap") == 0) {
            if (i + 1 >= argc || !pixie_parse_double_token(argv[i + 1], &opt.gap_limit)) {
                fprintf(stderr, "invalid --gap value\n");
                return EXIT_FAILURE;
            }
            ++i;
        } else if (strcmp(a, "--seed") == 0) {
            if (i + 1 >= argc || !pixie_parse_u64_token(argv[i + 1], &opt.seed)) {
                fprintf(stderr, "invalid --seed value\n");
                return EXIT_FAILURE;
            }
            opt.seed_set = true;
            ++i;
        } else if (strcmp(a, "--verbose") == 0) {
            long long vv = 0;
            if (i + 1 >= argc || !pixie_parse_ll_token(argv[i + 1], &vv)) {
                fprintf(stderr, "invalid --verbose value\n");
                return EXIT_FAILURE;
            }
            if (vv < 0) {
                vv = 0;
            }
            if (vv > 3) {
                vv = 3;
            }
            opt.verbose = (int)vv;
            ++i;
        } else if (strcmp(a, "--cuda") == 0) {
            if (i + 1 >= argc || !krb_accel_parse_mode(argv[i + 1], &opt.accel.mode)) {
                fprintf(stderr, "invalid --cuda value (expected auto|on|off)\n");
                return EXIT_FAILURE;
            }
            ++i;
        } else if (strcmp(a, "--cuda-device") == 0) {
            long long dev = 0;
            if (i + 1 >= argc || !pixie_parse_ll_token(argv[i + 1], &dev) || dev < -1 || dev > INT_MAX) {
                fprintf(stderr, "invalid --cuda-device value\n");
                return EXIT_FAILURE;
            }
            opt.accel.cuda_device = (int)dev;
            ++i;
        } else if (strcmp(a, "--cuda-min-cells") == 0) {
            long long cells = 0;
            if (i + 1 >= argc || !pixie_parse_ll_token(argv[i + 1], &cells) || cells < 0) {
                fprintf(stderr, "invalid --cuda-min-cells value\n");
                return EXIT_FAILURE;
            }
            opt.accel.cuda_min_cells = (size_t)cells;
            ++i;
        } else if (strcmp(a, "--purelp") == 0) {
            opt.pure_lp = true;
        } else if (a[0] == '-') {
            fprintf(stderr, "unknown option: %s\n", a);
            pixie_print_usage(argv[0]);
            return EXIT_FAILURE;
        } else {
            if (opt.file != NULL) {
                fprintf(stderr, "multiple input files provided\n");
                pixie_print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            opt.file = a;
        }
    }

    if (opt.selftest) {
        return pixie_run_selftest();
    }

    if (opt.file == NULL) {
        pixie_print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (!krb_parallel_runtime_resolve(&opt.parallel, &parallel_rt, err, sizeof(err))) {
        fprintf(stderr, "error: %s\n", err);
        return EXIT_FAILURE;
    }
    if (parallel_rt.resolved_mode != KRB_PARALLEL_MODE_OFF &&
        parallel_rt.resolved_mode != KRB_PARALLEL_MODE_THREADS &&
        opt.verbose >= 1) {
        fprintf(stderr,
                "c pixie parallel mode %s requested; backend falling back to serial threads-only support\n",
                krb_parallel_mode_name(parallel_rt.resolved_mode));
        parallel_rt.resolved_mode = KRB_PARALLEL_MODE_OFF;
        parallel_rt.jobs = 1;
    }
    if (parallel_rt.resolved_mode == KRB_PARALLEL_MODE_THREADS &&
        (opt.node_limit > 0 || opt.gap_limit >= 0.0) &&
        opt.verbose >= 1) {
        fprintf(stderr, "c pixie threaded mode disables shared node/gap stopping; falling back to serial\n");
        parallel_rt.resolved_mode = KRB_PARALLEL_MODE_OFF;
        parallel_rt.jobs = 1;
    }

    pixie_model_init(&model);
    deadline = (opt.time_limit_sec > 0.0) ? (pixie_wall_seconds() + opt.time_limit_sec) : 0.0;
    err[0] = '\0';
    if (!pixie_read_model(opt.file, opt.format, &model, deadline, err, sizeof(err))) {
        if (strncmp(err, "time limit exceeded", 19) == 0) {
            printf("s UNKNOWN\n");
            printf("o nan\n");
            pixie_model_free(&model);
            return EXIT_SUCCESS;
        }
        fprintf(stderr, "error: %s\n", (err[0] != '\0') ? err : "failed to read model");
        pixie_model_free(&model);
        return EXIT_FAILURE;
    }

    for (i = 0; i < model.n_vars; ++i) {
        if ((i & 255) == 0 && pixie_deadline_reached(deadline)) {
            printf("s UNKNOWN\n");
            printf("o nan\n");
            pixie_model_free(&model);
            return EXIT_SUCCESS;
        }
        if (model.vars[i].type == PIXIE_VAR_BIN) {
            if (model.vars[i].lb < 0.0) {
                model.vars[i].lb = 0.0;
            }
            if (model.vars[i].ub > 1.0) {
                model.vars[i].ub = 1.0;
            }
        }
    }

    if (deadline > 0.0) {
        opt.time_limit_sec = deadline - pixie_wall_seconds();
        if (opt.time_limit_sec <= 0.0) {
            printf("s UNKNOWN\n");
            printf("o nan\n");
            pixie_model_free(&model);
            return EXIT_SUCCESS;
        }
    }

    n_int = pixie_count_integer_vars(&model);
    sol = pixie_solve(&model, &opt, &parallel_rt);
    if (n_int == 0 &&
        opt.time_limit_sec <= 0.0 &&
        opt.node_limit <= 0 &&
        opt.gap_limit < 0.0 &&
        !sol.stopped_time &&
        !sol.stopped_nodes &&
        !sol.stopped_gap &&
        (sol.status == PIXIE_STATUS_INFEASIBLE || sol.status == PIXIE_STATUS_UNKNOWN) &&
        pixie_try_external_lp_solve(&model, &opt, &sol, &fallback_solver) &&
        opt.verbose >= 1) {
        fprintf(stderr, "c external fallback %s\n", fallback_solver);
    }
    pixie_print_solution(&model, &sol);
    pixie_solution_free(&sol);
    pixie_model_free(&model);
    return EXIT_SUCCESS;
}

#ifndef PIXIE_NO_MAIN
int main(int argc, char **argv) {
    return pixie_entry(argc, argv);
}
#endif
