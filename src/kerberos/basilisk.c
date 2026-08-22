/*
 * Copyright (c) 2026 Oscar Riveros.
 *
 * Licencia dual: uso personal bajo Apache License 2.0; portes a otros
 * lenguajes requieren licencia comercial con autorizacion expresa del autor.
 * Ver LICENSE.txt en la raiz del proyecto para los terminos completos.
 */

/*
Description:
BASILISK is a compact exact Boolean model counter implemented in a single ISO C17 file.
The initial native kernel focuses on DIMACS CNF, exact #SAT, and projected model counting
via `c ind ... 0` comments. Inside SATX, optional CUDA branch scoring is provided through
the shared `krb_accel` layer; the core counting logic remains native ISO C17.
*/

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "krb_accel.h"
#include "krb_parallel.h"

#define BASILISK_BIG_BASE 1000000000u
#define BASILISK_DEF_SLIME_HESS_NOHIT_LIMIT 8
#define BASILISK_DEF_SLIME_CT_IDLE_SOFT_LIMIT 16
#define BASILISK_DEF_SLIME_CT_IDLE_OFF_LIMIT 32
#define BASILISK_DEF_SLIME_CT_PROBE_RESTARTS_MAX 16
#define BASILISK_DEF_CUDA_MIN_LITS ((size_t)1048576)

typedef struct {
    int *data;
    int size;
    int cap;
} IntVec;

typedef struct {
    int *lits;
    int size;
} Clause;

typedef struct {
    Clause *data;
    int size;
    int cap;
} ClauseVec;

typedef struct {
    uint32_t *digits;
    int size;
    int cap;
} BigUInt;

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} StrBuf;

typedef enum {
    BASILISK_MODE_COUNT = 0,
    BASILISK_MODE_PROJECT = 1
} BasiliskMode;

typedef enum {
    BASILISK_PROJECT_AUTO = 0,
    BASILISK_PROJECT_ALL = 1,
    BASILISK_PROJECT_IND = 2
} BasiliskProjectMode;

typedef struct {
    int nvars;
    ClauseVec clauses;
    unsigned char *decl_project;
    int decl_project_count;
    int have_decl_project;
} BasiliskCNF;

typedef struct {
    KrbParallelConfig parallel;
    BasiliskMode mode;
    BasiliskProjectMode project_mode;
    int stats;
    int verbose;
    int selftest;
    KrbAccelMode cuda_mode;
    int cuda_device;
    size_t cuda_min_lits;
    int slime_init_hess;
    int slime_init_ct;
    int slime_init_ct_escape_rounds;
    int slime_init_ct_probe_restarts;
    int slime_hess_nohit_limit;
    int slime_ct_idle_soft_limit;
    int slime_ct_idle_off_limit;
    int slime_ct_probe_restarts_max;
    const char *input_path;
} BasiliskOptions;

typedef struct {
    IntVec clauses;
    IntVec vars;
} BasiliskComponent;

typedef struct {
    BasiliskComponent *data;
    int size;
    int cap;
} ComponentVec;

typedef struct {
    char *key;
    uint64_t hash;
    int used;
    BigUInt value;
} BasiliskCacheEntry;

typedef struct {
    BasiliskCacheEntry *slots;
    int cap;
    int size;
} BasiliskCache;

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
void slime_sat_handle_reconfigure(SlimeSatHandle *handle, const SlimeSatOptions *opt);
int slime_sat_handle_solve(SlimeSatHandle *handle,
                           const int *assumptions,
                           int num_assumptions,
                           SlimeSatStats *stats,
                           unsigned char *model01);
void slime_sat_handle_destroy(SlimeSatHandle *handle);
#endif

typedef struct {
    BasiliskOptions opt;
    clock_t started;
    uint64_t nodes;
    uint64_t decisions;
    uint64_t propagations;
    uint64_t sat_checks;
    uint64_t residual_sat_calls;
    uint64_t cache_hits;
    uint64_t component_splits;
    uint64_t slime_sat_calls;
    uint64_t slime_conflicts;
    uint64_t slime_decisions;
    uint64_t slime_propagations;
    uint64_t slime_restarts;
    uint64_t slime_hess_calls;
    uint64_t slime_hess_hits;
    uint64_t slime_ct_added;
    uint64_t slime_ct_merged;
    uint64_t slime_ct_escaped;
    uint64_t slime_ct_probe_added;
    uint64_t slime_tune_hess_off;
    uint64_t slime_tune_ct_soft;
    uint64_t slime_tune_ct_off;
    uint64_t cuda_branch_calls;
    uint64_t cuda_residual_branch_calls;
    uint64_t cuda_fallbacks;
    int max_depth;
    BasiliskCache cache;
} BasiliskShared;

typedef struct {
    BasiliskCNF cnf;
    signed char *assign;
    IntVec trail;
    int *score_tmp;
    int *score_pos;
    int *score_neg;
    unsigned char *active_project;
    unsigned char *phase_seen;
    unsigned char *phase_pref;
    int *flat_offsets;
    int *flat_lits;
    size_t total_lits;
    int use_cuda;
    int gpu_error;
    int active_project_count;
    const int **slime_clause_ptrs;
    int *slime_clause_sizes;
    IntVec slime_assumptions;
    void *slime_handle;
#if defined(SLIME_NO_MAIN)
    SlimeSatOptions slime_opt;
    uint64_t slime_hess_nohit_streak;
    uint64_t slime_ct_idle_streak;
#endif
    BasiliskShared *shared;
} BasiliskSolver;

typedef struct {
    int *lits;
    int size;
    int watch0;
    int watch1;
} SatResidualClause;

typedef struct {
    SatResidualClause *clauses;
    int size;
    int cap;
    int nvars;
    signed char *assign;
    unsigned char *phase_seen;
    unsigned char *phase_pref;
    int *score_tmp;
    int *score_pos;
    int *score_neg;
    int *flat_offsets;
    int *flat_lits;
    size_t total_lits;
    int use_cuda;
    int gpu_error;
    IntVec *watches;
    IntVec trail;
    IntVec trail_lim;
    IntVec decisions;
    int qhead;
    uint64_t conflicts;
    BasiliskShared *shared;
} SatResidualSolver;

static const char *g_basilisk_prog = "basilisk";

static void basilisk_die(const char *msg) {
    fprintf(stderr, "basilisk: %s\n", msg);
    exit(EXIT_FAILURE);
}

static void *basilisk_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1U);
    if (p == NULL) basilisk_die("out of memory");
    return p;
}

static void *basilisk_xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n ? n : 1U);
    if (p == NULL) basilisk_die("out of memory");
    return p;
}

static char *basilisk_xstrdup(const char *s) {
    size_t n = strlen(s);
    char *d = (char *)basilisk_xmalloc(n + 1U);
    memcpy(d, s, n + 1U);
    return d;
}

static int basilisk_cuda_alloc_bytes(size_t bytes, void **out, char *err, size_t errsz) {
    if (!krb_accel_cuda_managed_alloc_bytes(bytes, out, err, errsz)) {
        return 0;
    }
    return 1;
}

static int int_cmp(const void *a, const void *b) {
    int x = *(const int *)a;
    int y = *(const int *)b;
    return (x > y) - (x < y);
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

static void intvec_push(IntVec *v, int x) {
    int nc;
    if (v->size >= v->cap) {
        nc = (v->cap > 0) ? v->cap * 2 : 16;
        v->data = (int *)basilisk_xrealloc(v->data, (size_t)nc * sizeof(int));
        v->cap = nc;
    }
    v->data[v->size++] = x;
}

static void intvec_sort(IntVec *v) {
    if (v->size > 1) {
        qsort(v->data, (size_t)v->size, sizeof(int), int_cmp);
    }
}

static void clausevec_init(ClauseVec *v) {
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static void clausevec_free(ClauseVec *v) {
    int i;
    for (i = 0; i < v->size; ++i) {
        free(v->data[i].lits);
    }
    free(v->data);
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static void clausevec_push_copy(ClauseVec *v, const int *lits, int n) {
    int nc;
    Clause c;
    if (v->size >= v->cap) {
        nc = (v->cap > 0) ? v->cap * 2 : 16;
        v->data = (Clause *)basilisk_xrealloc(v->data, (size_t)nc * sizeof(Clause));
        v->cap = nc;
    }
    c.size = n;
    c.lits = (int *)basilisk_xmalloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    if (n > 0) memcpy(c.lits, lits, (size_t)n * sizeof(int));
    v->data[v->size++] = c;
}

static void big_init(BigUInt *b) {
    b->digits = NULL;
    b->size = 0;
    b->cap = 0;
}

static void big_free(BigUInt *b) {
    free(b->digits);
    b->digits = NULL;
    b->size = 0;
    b->cap = 0;
}

static void big_reserve(BigUInt *b, int need) {
    int nc = b->cap;
    if (need <= nc) return;
    if (nc <= 0) nc = 4;
    while (nc < need) nc *= 2;
    b->digits = (uint32_t *)basilisk_xrealloc(b->digits, (size_t)nc * sizeof(uint32_t));
    b->cap = nc;
}

static void big_set_zero(BigUInt *b) {
    b->size = 0;
}

static void big_set_u64(BigUInt *b, uint64_t x) {
    b->size = 0;
    if (x == 0ULL) return;
    while (x > 0ULL) {
        big_reserve(b, b->size + 1);
        b->digits[b->size++] = (uint32_t)(x % BASILISK_BIG_BASE);
        x /= BASILISK_BIG_BASE;
    }
}

static void big_set_one(BigUInt *b) {
    big_set_u64(b, 1ULL);
}

static int big_is_zero(const BigUInt *b) {
    return b->size == 0;
}

static void big_copy(BigUInt *dst, const BigUInt *src) {
    if (src->size == 0) {
        dst->size = 0;
        return;
    }
    big_reserve(dst, src->size);
    memcpy(dst->digits, src->digits, (size_t)src->size * sizeof(uint32_t));
    dst->size = src->size;
}

static void big_swap(BigUInt *a, BigUInt *b) {
    BigUInt t = *a;
    *a = *b;
    *b = t;
}

static void big_add_assign(BigUInt *a, const BigUInt *b) {
    int i;
    uint64_t carry = 0ULL;
    int maxn = (a->size > b->size) ? a->size : b->size;
    if (b->size == 0) return;
    big_reserve(a, maxn + 1);
    for (i = 0; i < maxn || carry != 0ULL; ++i) {
        uint64_t av = (i < a->size) ? a->digits[i] : 0U;
        uint64_t bv = (i < b->size) ? b->digits[i] : 0U;
        uint64_t sum = av + bv + carry;
        if (i >= a->size) a->digits[a->size++] = 0U;
        a->digits[i] = (uint32_t)(sum % BASILISK_BIG_BASE);
        carry = sum / BASILISK_BIG_BASE;
    }
}

static void big_add_to(BigUInt *out, const BigUInt *a, const BigUInt *b) {
    big_copy(out, a);
    big_add_assign(out, b);
}

static void big_mul_small_assign(BigUInt *a, uint32_t m) {
    int i;
    uint64_t carry = 0ULL;
    if (a->size == 0 || m == 1U) return;
    if (m == 0U) {
        a->size = 0;
        return;
    }
    big_reserve(a, a->size + 2);
    for (i = 0; i < a->size; ++i) {
        uint64_t prod = (uint64_t)a->digits[i] * (uint64_t)m + carry;
        a->digits[i] = (uint32_t)(prod % BASILISK_BIG_BASE);
        carry = prod / BASILISK_BIG_BASE;
    }
    while (carry != 0ULL) {
        a->digits[a->size++] = (uint32_t)(carry % BASILISK_BIG_BASE);
        carry /= BASILISK_BIG_BASE;
    }
}

static void big_mul_to(BigUInt *out, const BigUInt *a, const BigUInt *b) {
    int i;
    int j;
    if (a->size == 0 || b->size == 0) {
        out->size = 0;
        return;
    }
    big_reserve(out, a->size + b->size + 1);
    memset(out->digits, 0, (size_t)(a->size + b->size + 1) * sizeof(uint32_t));
    out->size = a->size + b->size;
    for (i = 0; i < a->size; ++i) {
        uint64_t carry = 0ULL;
        for (j = 0; j < b->size || carry != 0ULL; ++j) {
            uint64_t cur = out->digits[i + j] + carry;
            if (j < b->size) {
                cur += (uint64_t)a->digits[i] * (uint64_t)b->digits[j];
            }
            out->digits[i + j] = (uint32_t)(cur % BASILISK_BIG_BASE);
            carry = cur / BASILISK_BIG_BASE;
        }
    }
    while (out->size > 0 && out->digits[out->size - 1] == 0U) {
        --out->size;
    }
}

static void big_mul_assign(BigUInt *a, const BigUInt *b) {
    BigUInt tmp;
    big_init(&tmp);
    big_mul_to(&tmp, a, b);
    big_swap(a, &tmp);
    big_free(&tmp);
}

static void big_set_pow2(BigUInt *b, int exp) {
    int i;
    big_set_one(b);
    for (i = 0; i < exp; ++i) {
        big_mul_small_assign(b, 2U);
    }
}

static char *big_to_string(const BigUInt *b) {
    char *out;
    int i;
    size_t bufsz;
    size_t pos;
    if (b->size == 0) {
        out = (char *)basilisk_xmalloc(2U);
        out[0] = '0';
        out[1] = '\0';
        return out;
    }
    bufsz = (size_t)b->size * 10U + 2U;
    out = (char *)basilisk_xmalloc(bufsz);
    pos = (size_t)snprintf(out, bufsz, "%u", b->digits[b->size - 1]);
    for (i = b->size - 2; i >= 0; --i) {
        pos += (size_t)snprintf(out + pos, bufsz - pos, "%09u", b->digits[i]);
    }
    return out;
}

static void strbuf_init(StrBuf *sb) {
    sb->data = NULL;
    sb->len = 0U;
    sb->cap = 0U;
}

static void strbuf_reserve(StrBuf *sb, size_t extra) {
    size_t need = sb->len + extra + 1U;
    size_t nc = sb->cap;
    if (need <= nc) return;
    if (nc == 0U) nc = 64U;
    while (nc < need) nc *= 2U;
    sb->data = (char *)basilisk_xrealloc(sb->data, nc);
    sb->cap = nc;
}

static void strbuf_append(StrBuf *sb, const char *s) {
    size_t n = strlen(s);
    strbuf_reserve(sb, n);
    memcpy(sb->data + sb->len, s, n + 1U);
    sb->len += n;
}

static void strbuf_append_char(StrBuf *sb, char c) {
    strbuf_reserve(sb, 1U);
    sb->data[sb->len++] = c;
    sb->data[sb->len] = '\0';
}

static void strbuf_append_int(StrBuf *sb, int x) {
    char tmp[64];
    snprintf(tmp, sizeof(tmp), "%d", x);
    strbuf_append(sb, tmp);
}

static char *strbuf_detach(StrBuf *sb) {
    char *s;
    if (sb->data == NULL) {
        s = (char *)basilisk_xmalloc(1U);
        s[0] = '\0';
    } else {
        s = sb->data;
    }
    sb->data = NULL;
    sb->len = 0U;
    sb->cap = 0U;
    return s;
}

static void component_init(BasiliskComponent *c) {
    intvec_init(&c->clauses);
    intvec_init(&c->vars);
}

static void component_free(BasiliskComponent *c) {
    intvec_free(&c->clauses);
    intvec_free(&c->vars);
}

static void compvec_init(ComponentVec *v) {
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static void compvec_free(ComponentVec *v) {
    int i;
    for (i = 0; i < v->size; ++i) {
        component_free(&v->data[i]);
    }
    free(v->data);
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static BasiliskComponent *compvec_push(ComponentVec *v) {
    int nc;
    if (v->size >= v->cap) {
        nc = (v->cap > 0) ? v->cap * 2 : 8;
        v->data = (BasiliskComponent *)basilisk_xrealloc(v->data, (size_t)nc * sizeof(BasiliskComponent));
        v->cap = nc;
    }
    component_init(&v->data[v->size]);
    return &v->data[v->size++];
}

static uint64_t cache_hash_string(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s != '\0') {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ULL;
    }
    if (h == 0ULL) h = 1ULL;
    return h;
}

static void cache_init(BasiliskCache *c) {
    c->slots = NULL;
    c->cap = 0;
    c->size = 0;
}

static void cache_free(BasiliskCache *c) {
    int i;
    for (i = 0; i < c->cap; ++i) {
        if (c->slots[i].used) {
            free(c->slots[i].key);
            big_free(&c->slots[i].value);
        }
    }
    free(c->slots);
    c->slots = NULL;
    c->cap = 0;
    c->size = 0;
}

static void cache_rehash(BasiliskCache *c, int new_cap) {
    BasiliskCacheEntry *old = c->slots;
    int old_cap = c->cap;
    int i;
    c->slots = (BasiliskCacheEntry *)calloc((size_t)new_cap, sizeof(BasiliskCacheEntry));
    if (new_cap > 0 && c->slots == NULL) basilisk_die("out of memory");
    c->cap = new_cap;
    c->size = 0;
    for (i = 0; i < old_cap; ++i) {
        if (old[i].used) {
            int pos = (int)(old[i].hash % (uint64_t)c->cap);
            while (c->slots[pos].used) {
                pos = (pos + 1) % c->cap;
            }
            c->slots[pos] = old[i];
            c->slots[pos].used = 1;
            ++c->size;
        }
    }
    free(old);
}

static const BasiliskCacheEntry *cache_lookup(const BasiliskCache *c, const char *key, uint64_t hash) {
    int pos;
    int start;
    if (c->cap == 0) return NULL;
    pos = (int)(hash % (uint64_t)c->cap);
    start = pos;
    while (c->slots[pos].used) {
        if (c->slots[pos].hash == hash && strcmp(c->slots[pos].key, key) == 0) {
            return &c->slots[pos];
        }
        pos = (pos + 1) % c->cap;
        if (pos == start) break;
    }
    return NULL;
}

static void cache_store(BasiliskCache *c, const char *key, uint64_t hash, const BigUInt *value) {
    int pos;
    if (c->cap == 0) {
        cache_rehash(c, 256);
    } else if ((c->size + 1) * 10 >= c->cap * 7) {
        cache_rehash(c, c->cap * 2);
    }
    pos = (int)(hash % (uint64_t)c->cap);
    while (c->slots[pos].used) {
        if (c->slots[pos].hash == hash && strcmp(c->slots[pos].key, key) == 0) {
            big_copy(&c->slots[pos].value, value);
            return;
        }
        pos = (pos + 1) % c->cap;
    }
    c->slots[pos].used = 1;
    c->slots[pos].hash = hash;
    c->slots[pos].key = basilisk_xstrdup(key);
    big_init(&c->slots[pos].value);
    big_copy(&c->slots[pos].value, value);
    ++c->size;
}

static double basilisk_now_sec(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void trim_inplace(char *s) {
    size_t n;
    size_t start = 0;
    size_t end;
    while (s[start] != '\0' && isspace((unsigned char)s[start])) start++;
    if (start > 0) memmove(s, s + start, strlen(s + start) + 1U);
    n = strlen(s);
    end = n;
    while (end > 0 && isspace((unsigned char)s[end - 1])) end--;
    s[end] = '\0';
}

static int token_split(char *line, char **tok, int max_tok) {
    int n = 0;
    char *p = line;
    while (*p != '\0' && n < max_tok) {
        while (*p != '\0' && isspace((unsigned char)*p)) ++p;
        if (*p == '\0') break;
        tok[n++] = p;
        while (*p != '\0' && !isspace((unsigned char)*p)) ++p;
        if (*p == '\0') break;
        *p++ = '\0';
    }
    return n;
}

static int parse_ll_token(const char *s, long long *out) {
    char *end = NULL;
    long long v;
    errno = 0;
    v = strtoll(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') return 0;
    *out = v;
    return 1;
}

static void cnf_init(BasiliskCNF *cnf) {
    cnf->nvars = 0;
    clausevec_init(&cnf->clauses);
    cnf->decl_project = NULL;
    cnf->decl_project_count = 0;
    cnf->have_decl_project = 0;
}

static void cnf_free(BasiliskCNF *cnf) {
    clausevec_free(&cnf->clauses);
    free(cnf->decl_project);
    cnf->decl_project = NULL;
    cnf->decl_project_count = 0;
    cnf->have_decl_project = 0;
    cnf->nvars = 0;
}

static int cnf_parse_dimacs(BasiliskCNF *cnf, const char *path, char *err, size_t errsz) {
    FILE *fp = NULL;
    char line_buf[8192];
    int header_found = 0;
    int declared_clauses = -1;
    IntVec clause;
    IntVec pending_project;
    cnf_init(cnf);
    intvec_init(&clause);
    intvec_init(&pending_project);

    fp = fopen(path, "rb");
    if (fp == NULL) {
        snprintf(err, errsz, "cannot open '%s': %s", path, strerror(errno));
        goto fail;
    }

    while (fgets(line_buf, (int)sizeof(line_buf), fp) != NULL) {
        char line[8192];
        char *tok[4096];
        int nt;
        int i;
        snprintf(line, sizeof(line), "%s", line_buf);
        trim_inplace(line);
        if (line[0] == '\0') continue;
        nt = token_split(line, tok, 4096);
        if (nt <= 0) continue;

        if ((tok[0][0] == 'c' || tok[0][0] == 'C') && tok[0][1] == '\0') {
            if (nt >= 2 && strcmp(tok[1], "ind") == 0) {
                cnf->have_decl_project = 1;
                for (i = 2; i < nt; ++i) {
                    long long litll = 0;
                    int v;
                    if (!parse_ll_token(tok[i], &litll)) {
                        snprintf(err, errsz, "invalid c ind token '%s' in '%s'", tok[i], path);
                        goto fail;
                    }
                    if (litll == 0) break;
                    v = (int)((litll < 0) ? -litll : litll);
                    if (v < 1) {
                        snprintf(err, errsz, "projection var %d out of range in '%s'", v, path);
                        goto fail;
                    }
                    if (header_found) {
                        if (v > cnf->nvars) {
                            snprintf(err, errsz, "projection var %d out of range 1..%d in '%s'", v, cnf->nvars, path);
                            goto fail;
                        }
                        if (!cnf->decl_project[v]) {
                            cnf->decl_project[v] = 1U;
                            ++cnf->decl_project_count;
                        }
                    } else {
                        intvec_push(&pending_project, v);
                    }
                }
            }
            continue;
        }

        if (strcmp(tok[0], "p") == 0) {
            long long nvarsll = 0;
            long long nclausesll = 0;
            if (nt < 4 || strcmp(tok[1], "cnf") != 0) {
                snprintf(err, errsz, "invalid DIMACS header in '%s'", path);
                goto fail;
            }
            if (!parse_ll_token(tok[2], &nvarsll) || !parse_ll_token(tok[3], &nclausesll) ||
                nvarsll < 0 || nvarsll > 100000000 || nclausesll < 0) {
                snprintf(err, errsz, "invalid DIMACS counts in '%s'", path);
                goto fail;
            }
            cnf->nvars = (int)nvarsll;
            declared_clauses = (int)nclausesll;
            cnf->decl_project = (unsigned char *)calloc((size_t)cnf->nvars + 1U, sizeof(unsigned char));
            if (cnf->nvars > 0 && cnf->decl_project == NULL) basilisk_die("out of memory");
            for (i = 0; i < pending_project.size; ++i) {
                int v = pending_project.data[i];
                if (v < 1 || v > cnf->nvars) {
                    snprintf(err, errsz, "projection var %d out of range 1..%d in '%s'", v, cnf->nvars, path);
                    goto fail;
                }
                if (!cnf->decl_project[v]) {
                    cnf->decl_project[v] = 1U;
                    ++cnf->decl_project_count;
                }
            }
            header_found = 1;
            continue;
        }

        if (!header_found) {
            snprintf(err, errsz, "clause data before DIMACS header in '%s'", path);
            goto fail;
        }

        for (i = 0; i < nt; ++i) {
            long long litll = 0;
            int lit = 0;
            int v = 0;
            if (!parse_ll_token(tok[i], &litll)) {
                snprintf(err, errsz, "invalid literal token '%s' in '%s'", tok[i], path);
                goto fail;
            }
            if (litll < INT32_MIN || litll > INT32_MAX) {
                snprintf(err, errsz, "literal '%s' out of range in '%s'", tok[i], path);
                goto fail;
            }
            lit = (int)litll;
            if (lit == 0) {
                clausevec_push_copy(&cnf->clauses, clause.data, clause.size);
                clause.size = 0;
                continue;
            }
            v = (lit < 0) ? -lit : lit;
            if (v < 1 || v > cnf->nvars) {
                snprintf(err, errsz, "literal %d out of range 1..%d in '%s'", lit, cnf->nvars, path);
                goto fail;
            }
            intvec_push(&clause, lit);
        }
    }

    if (!header_found) {
        snprintf(err, errsz, "missing DIMACS header in '%s'", path);
        goto fail;
    }
    if (clause.size != 0) {
        snprintf(err, errsz, "unterminated clause at EOF in '%s'", path);
        goto fail;
    }
    if (declared_clauses >= 0 && declared_clauses != cnf->clauses.size) {
        snprintf(err, errsz, "declared %d clauses but parsed %d in '%s'",
                 declared_clauses, cnf->clauses.size, path);
        goto fail;
    }

    intvec_free(&clause);
    intvec_free(&pending_project);
    fclose(fp);
    return 1;

fail:
    intvec_free(&clause);
    intvec_free(&pending_project);
    if (fp != NULL) fclose(fp);
    cnf_free(cnf);
    return 0;
}

static size_t cnf_total_lits(const BasiliskCNF *cnf) {
    size_t total = 0U;
    int i;
    for (i = 0; i < cnf->clauses.size; ++i) {
        if (cnf->clauses.data[i].size > 0) {
            total += (size_t)cnf->clauses.data[i].size;
        }
    }
    return total;
}

static size_t satres_total_lits(const SatResidualSolver *rs) {
    size_t total = 0U;
    int i;
    for (i = 0; i < rs->size; ++i) {
        if (rs->clauses[i].size > 0) {
            total += (size_t)rs->clauses[i].size;
        }
    }
    return total;
}

static void cnf_build_flat(const BasiliskCNF *cnf, int *offsets, int *lits) {
    int off = 0;
    int i;
    for (i = 0; i < cnf->clauses.size; ++i) {
        const Clause *c = &cnf->clauses.data[i];
        offsets[i] = off;
        if (c->size > 0) {
            memcpy(lits + off, c->lits, (size_t)c->size * sizeof(int));
            off += c->size;
        }
    }
    offsets[cnf->clauses.size] = off;
}

static void satres_build_flat(const SatResidualSolver *rs, int *offsets, int *lits) {
    int off = 0;
    int i;
    for (i = 0; i < rs->size; ++i) {
        const SatResidualClause *c = &rs->clauses[i];
        offsets[i] = off;
        if (c->size > 0) {
            memcpy(lits + off, c->lits, (size_t)c->size * sizeof(int));
            off += c->size;
        }
    }
    offsets[rs->size] = off;
}

static int basilisk_pick_cuda(const BasiliskShared *shared,
                              size_t total_lits,
                              int nvars,
                              int nclauses,
                              const char *label,
                              int *out_use,
                              char *err,
                              size_t errsz) {
    if (out_use == NULL) {
        if (err != NULL && errsz > 0U) {
            snprintf(err, errsz, "%s: missing CUDA decision output", label != NULL ? label : "basilisk");
        }
        return 0;
    }
    *out_use = 0;
    if (shared == NULL) {
        if (err != NULL && errsz > 0U) err[0] = '\0';
        return 1;
    }
    if (shared->opt.cuda_mode == KRB_ACCEL_MODE_OFF || total_lits == 0U || nvars <= 0 || nclauses <= 0) {
        if (err != NULL && errsz > 0U) err[0] = '\0';
        return 1;
    }
    if (shared->opt.cuda_mode == KRB_ACCEL_MODE_AUTO &&
        (total_lits < shared->opt.cuda_min_lits || nvars < 2048 || nclauses < 4096)) {
        if (err != NULL && errsz > 0U) err[0] = '\0';
        return 1;
    }
    if (!krb_accel_cuda_compiled()) {
        if (shared->opt.cuda_mode == KRB_ACCEL_MODE_ON) {
            snprintf(err, errsz, "%s: CUDA requested but this build was compiled without CUDA support",
                     label != NULL ? label : "basilisk");
            return 0;
        }
        if (err != NULL && errsz > 0U) err[0] = '\0';
        return 1;
    }
    if (!krb_accel_cuda_runtime_available()) {
        if (shared->opt.cuda_mode == KRB_ACCEL_MODE_ON) {
            snprintf(err, errsz, "%s: CUDA requested but no compatible CUDA runtime/device is available",
                     label != NULL ? label : "basilisk");
            return 0;
        }
        if (err != NULL && errsz > 0U) err[0] = '\0';
        return 1;
    }
    if (!krb_accel_cuda_select_device(shared->opt.cuda_device, err, errsz)) {
        if (shared->opt.cuda_mode == KRB_ACCEL_MODE_ON || shared->opt.cuda_device >= 0) {
            return 0;
        }
        if (err != NULL && errsz > 0U) err[0] = '\0';
        return 1;
    }
    *out_use = 1;
    if (err != NULL && errsz > 0U) err[0] = '\0';
    return 1;
}

static int lit_eval(const signed char *assign, int lit) {
    int v = (lit < 0) ? -lit : lit;
    signed char a = assign[v];
    if (a < 0) return -1;
    if (lit > 0) return (a > 0) ? 1 : 0;
    return (a == 0) ? 1 : 0;
}

static int solver_assign_lit(BasiliskSolver *s, int lit) {
    int v = (lit < 0) ? -lit : lit;
    signed char want = (lit > 0) ? 1 : 0;
    signed char cur = s->assign[v];
    if (cur < 0) {
        s->assign[v] = want;
        s->phase_seen[v] = 1U;
        s->phase_pref[v] = (unsigned char)want;
        intvec_push(&s->trail, v);
        return 1;
    }
    return cur == want;
}

static void solver_undo(BasiliskSolver *s, int mark) {
    while (s->trail.size > mark) {
        int v = s->trail.data[--s->trail.size];
        s->assign[v] = -1;
    }
}

static int solver_propagate(BasiliskSolver *s) {
    int changed = 1;
    while (changed) {
        int ci;
        changed = 0;
        for (ci = 0; ci < s->cnf.clauses.size; ++ci) {
            const Clause *c = &s->cnf.clauses.data[ci];
            int sat = 0;
            int unassigned = 0;
            int unit_lit = 0;
            int j;
            for (j = 0; j < c->size; ++j) {
                int ev = lit_eval(s->assign, c->lits[j]);
                if (ev > 0) {
                    sat = 1;
                    break;
                }
                if (ev < 0) {
                    ++unassigned;
                    unit_lit = c->lits[j];
                }
            }
            if (sat) continue;
            if (unassigned == 0) return 0;
            if (unassigned == 1) {
                if (!solver_assign_lit(s, unit_lit)) return 0;
                ++s->shared->propagations;
                changed = 1;
            }
        }
    }
    return 1;
}

static int solver_all_satisfied(const BasiliskSolver *s) {
    int ci;
    for (ci = 0; ci < s->cnf.clauses.size; ++ci) {
        const Clause *c = &s->cnf.clauses.data[ci];
        int sat = 0;
        int j;
        for (j = 0; j < c->size; ++j) {
            if (lit_eval(s->assign, c->lits[j]) > 0) {
                sat = 1;
                break;
            }
        }
        if (!sat) return 0;
    }
    return 1;
}

static void solver_score_branch_cpu(BasiliskSolver *s, int project_only) {
    int ci;
    memset(s->score_pos, 0, ((size_t)s->cnf.nvars + 1U) * sizeof(int));
    memset(s->score_neg, 0, ((size_t)s->cnf.nvars + 1U) * sizeof(int));
    memset(s->score_tmp, 0, ((size_t)s->cnf.nvars + 1U) * sizeof(int));
    for (ci = 0; ci < s->cnf.clauses.size; ++ci) {
        const Clause *c = &s->cnf.clauses.data[ci];
        int sat = 0;
        int open = 0;
        int j;
        for (j = 0; j < c->size; ++j) {
            int ev = lit_eval(s->assign, c->lits[j]);
            if (ev > 0) {
                sat = 1;
                break;
            }
            if (ev < 0) ++open;
        }
        if (sat) continue;
        for (j = 0; j < c->size; ++j) {
            int lit = c->lits[j];
            int v = (lit < 0) ? -lit : lit;
            int w = (open <= 2) ? 6 : (open == 3 ? 3 : 1);
            if (s->assign[v] >= 0) continue;
            if (project_only && !s->active_project[v]) continue;
            s->score_tmp[v] += w;
            if (lit > 0) s->score_pos[v] += w;
            else s->score_neg[v] += w;
        }
    }
}

static int solver_try_score_branch_cuda(BasiliskSolver *s, int project_only) {
    char err[256];
    if (!s->use_cuda || s->gpu_error) {
        return 0;
    }
    if (!krb_accel_cuda_score_cnf_branching(s->flat_offsets,
                                            s->flat_lits,
                                            s->cnf.clauses.size,
                                            s->cnf.nvars,
                                            s->assign,
                                            s->active_project,
                                            project_only ? 1 : 0,
                                            s->score_tmp,
                                            s->score_pos,
                                            s->score_neg,
                                            err,
                                            sizeof(err))) {
        s->gpu_error = 1;
        if (s->shared != NULL) {
            ++s->shared->cuda_fallbacks;
            if (s->shared->opt.verbose >= 1) {
                fprintf(stderr, "c basilisk cuda fallback[branch]: %s\n", err);
            }
        }
        return 0;
    }
    if (s->shared != NULL) {
        ++s->shared->cuda_branch_calls;
    }
    return 1;
}

static int solver_choose_branch_lit(BasiliskSolver *s, int project_only) {
    int ci;
    int best_lit = 0;
    int best_score = -1;
    if (!solver_try_score_branch_cuda(s, project_only)) {
        solver_score_branch_cpu(s, project_only);
    }
    for (ci = 1; ci <= s->cnf.nvars; ++ci) {
        int prefer_pos;
        if (s->assign[ci] >= 0) continue;
        if (project_only && !s->active_project[ci]) continue;
        if (s->score_tmp[ci] > best_score) {
            best_score = s->score_tmp[ci];
            prefer_pos = s->phase_seen[ci] ? (s->phase_pref[ci] != 0U) : (s->score_pos[ci] >= s->score_neg[ci]);
            best_lit = prefer_pos ? ci : -ci;
        }
    }
    return best_lit;
}

static int solver_count_unassigned_all(const BasiliskSolver *s) {
    int v;
    int cnt = 0;
    for (v = 1; v <= s->cnf.nvars; ++v) {
        if (s->assign[v] < 0) ++cnt;
    }
    return cnt;
}

static int solver_count_unassigned_project(const BasiliskSolver *s) {
    int v;
    int cnt = 0;
    for (v = 1; v <= s->cnf.nvars; ++v) {
        if (s->active_project[v] && s->assign[v] < 0) ++cnt;
    }
    return cnt;
}

static int solver_has_unassigned_project(const BasiliskSolver *s) {
    return solver_count_unassigned_project(s) > 0;
}

static void basilisk_options_init(BasiliskOptions *opt) {
    memset(opt, 0, sizeof(*opt));
    krb_parallel_config_defaults(&opt->parallel);
    opt->mode = BASILISK_MODE_COUNT;
    opt->project_mode = BASILISK_PROJECT_AUTO;
    opt->cuda_mode = KRB_ACCEL_MODE_AUTO;
    opt->cuda_device = -1;
    opt->cuda_min_lits = BASILISK_DEF_CUDA_MIN_LITS;
    opt->slime_init_hess = -1;
    opt->slime_init_ct = 1;
    opt->slime_init_ct_escape_rounds = 4;
    opt->slime_init_ct_probe_restarts = 4;
    opt->slime_hess_nohit_limit = BASILISK_DEF_SLIME_HESS_NOHIT_LIMIT;
    opt->slime_ct_idle_soft_limit = BASILISK_DEF_SLIME_CT_IDLE_SOFT_LIMIT;
    opt->slime_ct_idle_off_limit = BASILISK_DEF_SLIME_CT_IDLE_OFF_LIMIT;
    opt->slime_ct_probe_restarts_max = BASILISK_DEF_SLIME_CT_PROBE_RESTARTS_MAX;
}

static void basilisk_options_normalize(BasiliskOptions *opt) {
    if (opt->slime_init_hess < -1) opt->slime_init_hess = -1;
    if (opt->slime_init_hess > 1) opt->slime_init_hess = 1;
    if (opt->cuda_device < -1) opt->cuda_device = -1;
    if (opt->cuda_min_lits < 1024U) opt->cuda_min_lits = 1024U;
    opt->slime_init_ct = opt->slime_init_ct ? 1 : 0;
    if (opt->slime_init_ct_escape_rounds < 0) opt->slime_init_ct_escape_rounds = 0;
    if (opt->slime_init_ct_probe_restarts < 0) opt->slime_init_ct_probe_restarts = 0;
    if (opt->slime_hess_nohit_limit < 1) opt->slime_hess_nohit_limit = 1;
    if (opt->slime_ct_idle_soft_limit < 1) opt->slime_ct_idle_soft_limit = 1;
    if (opt->slime_ct_idle_off_limit < opt->slime_ct_idle_soft_limit) {
        opt->slime_ct_idle_off_limit = opt->slime_ct_idle_soft_limit;
    }
    if (opt->slime_ct_probe_restarts_max < 1) {
        opt->slime_ct_probe_restarts_max = 1;
    }
    if (opt->slime_ct_probe_restarts_max < opt->slime_init_ct_probe_restarts) {
        opt->slime_ct_probe_restarts_max = opt->slime_init_ct_probe_restarts;
    }
}

static void shared_init(BasiliskShared *sh, const BasiliskOptions *opt) {
    memset(sh, 0, sizeof(*sh));
    sh->opt = *opt;
    basilisk_options_normalize(&sh->opt);
    sh->started = clock();
    cache_init(&sh->cache);
}

static void shared_free(BasiliskShared *sh) {
    cache_free(&sh->cache);
}

static void cnf_move(BasiliskCNF *dst, BasiliskCNF *src) {
    *dst = *src;
    src->nvars = 0;
    src->decl_project = NULL;
    src->decl_project_count = 0;
    src->have_decl_project = 0;
    src->clauses.data = NULL;
    src->clauses.size = 0;
    src->clauses.cap = 0;
}

static char *solver_build_residual_key(const BasiliskSolver *s) {
    StrBuf sb;
    unsigned char *var_used;
    int free_all = 0;
    int free_project = 0;
    int ci;
    int v;
    strbuf_init(&sb);
    strbuf_append(&sb, (s->shared->opt.mode == BASILISK_MODE_PROJECT) ? "p|" : "c|");
    var_used = (unsigned char *)calloc((size_t)s->cnf.nvars + 1U, sizeof(unsigned char));
    if (s->cnf.nvars > 0 && var_used == NULL) basilisk_die("out of memory");
    for (ci = 0; ci < s->cnf.clauses.size; ++ci) {
        const Clause *c = &s->cnf.clauses.data[ci];
        int sat = 0;
        int seen_unassigned = 0;
        int j;
        for (j = 0; j < c->size; ++j) {
            int ev = lit_eval(s->assign, c->lits[j]);
            if (ev > 0) {
                sat = 1;
                break;
            }
        }
        if (sat) continue;
        strbuf_append_char(&sb, '[');
        strbuf_append_int(&sb, ci);
        strbuf_append_char(&sb, ':');
        for (j = 0; j < c->size; ++j) {
            int lit = c->lits[j];
            int ev = lit_eval(s->assign, lit);
            if (ev < 0) {
                int vv = (lit < 0) ? -lit : lit;
                if (seen_unassigned) strbuf_append_char(&sb, ',');
                strbuf_append_int(&sb, lit);
                seen_unassigned = 1;
                var_used[vv] = 1U;
            }
        }
        strbuf_append_char(&sb, ']');
    }
    if (s->shared->opt.mode == BASILISK_MODE_PROJECT) {
        strbuf_append(&sb, "|proj:");
        for (v = 1; v <= s->cnf.nvars; ++v) {
            if (var_used[v] && s->active_project[v]) {
                strbuf_append_int(&sb, v);
                strbuf_append_char(&sb, ',');
            }
        }
    }
    for (v = 1; v <= s->cnf.nvars; ++v) {
        if (s->assign[v] < 0 && !var_used[v]) {
            ++free_all;
            if (s->active_project[v]) ++free_project;
        }
    }
    strbuf_append(&sb, "|free:");
    strbuf_append_int(&sb, (s->shared->opt.mode == BASILISK_MODE_PROJECT) ? free_project : free_all);
    free(var_used);
    return strbuf_detach(&sb);
}

static void solver_find_components(const BasiliskSolver *s,
                                   ComponentVec *out,
                                   int *free_all_out,
                                   int *free_project_out) {
    int n = s->cnf.nvars;
    int m = s->cnf.clauses.size;
    unsigned char *res_clause = (unsigned char *)calloc((size_t)m, sizeof(unsigned char));
    unsigned char *var_in_res = (unsigned char *)calloc((size_t)n + 1U, sizeof(unsigned char));
    int *clause_comp = (int *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    int ci;
    int v;
    if ((m > 0 && (res_clause == NULL || clause_comp == NULL)) || (n > 0 && var_in_res == NULL)) {
        basilisk_die("out of memory");
    }
    for (ci = 0; ci < m; ++ci) clause_comp[ci] = -1;
    *free_all_out = 0;
    *free_project_out = 0;
    compvec_init(out);

    for (ci = 0; ci < m; ++ci) {
        const Clause *c = &s->cnf.clauses.data[ci];
        int sat = 0;
        int j;
        for (j = 0; j < c->size; ++j) {
            int ev = lit_eval(s->assign, c->lits[j]);
            if (ev > 0) {
                sat = 1;
                break;
            }
        }
        if (sat) continue;
        res_clause[ci] = 1U;
        for (j = 0; j < c->size; ++j) {
            int lit = c->lits[j];
            int ev = lit_eval(s->assign, lit);
            if (ev < 0) {
                var_in_res[(lit < 0) ? -lit : lit] = 1U;
            }
        }
    }

    for (v = 1; v <= n; ++v) {
        if (s->assign[v] < 0 && !var_in_res[v]) {
            ++*free_all_out;
            if (s->active_project[v]) ++*free_project_out;
        }
    }

    for (ci = 0; ci < m; ++ci) {
        if (!res_clause[ci] || clause_comp[ci] >= 0) continue;
        {
            BasiliskComponent *comp = compvec_push(out);
            unsigned char *var_seen = (unsigned char *)calloc((size_t)n + 1U, sizeof(unsigned char));
            int progress = 1;
            if (n > 0 && var_seen == NULL) basilisk_die("out of memory");
            intvec_push(&comp->clauses, ci);
            clause_comp[ci] = out->size - 1;
            while (progress) {
                int k;
                progress = 0;
                for (k = 0; k < comp->clauses.size; ++k) {
                    const Clause *c = &s->cnf.clauses.data[comp->clauses.data[k]];
                    int j;
                    for (j = 0; j < c->size; ++j) {
                        int lit = c->lits[j];
                        int ev = lit_eval(s->assign, lit);
                        int vv = (lit < 0) ? -lit : lit;
                        if (ev < 0 && !var_seen[vv]) {
                            var_seen[vv] = 1U;
                            intvec_push(&comp->vars, vv);
                            progress = 1;
                        }
                    }
                }
                for (k = 0; k < m; ++k) {
                    if (!res_clause[k] || clause_comp[k] >= 0) continue;
                    {
                        const Clause *c = &s->cnf.clauses.data[k];
                        int share = 0;
                        int j;
                        for (j = 0; j < c->size; ++j) {
                            int lit = c->lits[j];
                            int ev = lit_eval(s->assign, lit);
                            int vv = (lit < 0) ? -lit : lit;
                            if (ev < 0 && var_seen[vv]) {
                                share = 1;
                                break;
                            }
                        }
                        if (share) {
                            clause_comp[k] = out->size - 1;
                            intvec_push(&comp->clauses, k);
                            progress = 1;
                        }
                    }
                }
            }
            intvec_sort(&comp->clauses);
            intvec_sort(&comp->vars);
            free(var_seen);
        }
    }

    free(res_clause);
    free(var_in_res);
    free(clause_comp);
}

static void solver_build_component_subcnf(const BasiliskSolver *s,
                                          const BasiliskComponent *comp,
                                          BasiliskCNF *out) {
    int *var_map;
    IntVec lits;
    int i;
    cnf_init(out);
    out->nvars = comp->vars.size;
    out->have_decl_project = 1;
    out->decl_project = (unsigned char *)calloc((size_t)out->nvars + 1U, sizeof(unsigned char));
    if (out->nvars > 0 && out->decl_project == NULL) basilisk_die("out of memory");
    var_map = (int *)calloc((size_t)s->cnf.nvars + 1U, sizeof(int));
    if (s->cnf.nvars > 0 && var_map == NULL) basilisk_die("out of memory");
    for (i = 0; i < comp->vars.size; ++i) {
        int orig_v = comp->vars.data[i];
        int local_v = i + 1;
        var_map[orig_v] = local_v;
        if (s->active_project[orig_v]) {
            out->decl_project[local_v] = 1U;
            ++out->decl_project_count;
        }
    }
    intvec_init(&lits);
    for (i = 0; i < comp->clauses.size; ++i) {
        const Clause *c = &s->cnf.clauses.data[comp->clauses.data[i]];
        int j;
        lits.size = 0;
        for (j = 0; j < c->size; ++j) {
            int lit = c->lits[j];
            int ev = lit_eval(s->assign, lit);
            if (ev < 0) {
                int vv = (lit < 0) ? -lit : lit;
                int local_v = var_map[vv];
                intvec_push(&lits, (lit < 0) ? -local_v : local_v);
            }
        }
        clausevec_push_copy(&out->clauses, lits.data, lits.size);
    }
    intvec_free(&lits);
    free(var_map);
}

static void solver_count_core(BasiliskSolver *s, int depth, BigUInt *out);
static void solver_release(BasiliskSolver *s);
static int solver_prepare(BasiliskSolver *s, char *err, size_t errsz);

static int sat_lit_index(int lit) {
    int v = (lit < 0) ? -lit : lit;
    return ((v - 1) << 1) | (lit < 0 ? 1 : 0);
}

static int sat_lit_eval(const SatResidualSolver *rs, int lit) {
    int v = (lit < 0) ? -lit : lit;
    signed char a = rs->assign[v];
    if (a < 0) return -1;
    if (lit > 0) return (a > 0) ? 1 : 0;
    return (a == 0) ? 1 : 0;
}

static void satres_init(SatResidualSolver *rs, int nvars, BasiliskShared *shared) {
    int i;
    memset(rs, 0, sizeof(*rs));
    rs->nvars = nvars;
    rs->shared = shared;
    rs->assign = (signed char *)malloc((size_t)nvars + 1U);
    rs->phase_seen = (unsigned char *)calloc((size_t)nvars + 1U, sizeof(unsigned char));
    rs->phase_pref = (unsigned char *)calloc((size_t)nvars + 1U, sizeof(unsigned char));
    rs->score_tmp = (int *)calloc((size_t)nvars + 1U, sizeof(int));
    rs->score_pos = (int *)calloc((size_t)nvars + 1U, sizeof(int));
    rs->score_neg = (int *)calloc((size_t)nvars + 1U, sizeof(int));
    rs->watches = (IntVec *)malloc((size_t)(2 * nvars > 0 ? 2 * nvars : 1) * sizeof(IntVec));
    if ((nvars > 0) &&
        (rs->assign == NULL || rs->phase_seen == NULL || rs->phase_pref == NULL ||
         rs->score_tmp == NULL || rs->score_pos == NULL || rs->score_neg == NULL || rs->watches == NULL)) {
        basilisk_die("out of memory");
    }
    for (i = 0; i <= nvars; ++i) rs->assign[i] = -1;
    for (i = 0; i < 2 * nvars; ++i) intvec_init(&rs->watches[i]);
    intvec_init(&rs->trail);
    intvec_init(&rs->trail_lim);
    intvec_init(&rs->decisions);
}

static void satres_free(SatResidualSolver *rs) {
    int i;
    for (i = 0; i < rs->size; ++i) free(rs->clauses[i].lits);
    free(rs->clauses);
    if (rs->watches != NULL) {
        for (i = 0; i < 2 * rs->nvars; ++i) intvec_free(&rs->watches[i]);
    }
    free(rs->watches);
    if (rs->use_cuda) {
        krb_accel_cuda_managed_free(rs->assign);
        krb_accel_cuda_managed_free(rs->phase_seen);
        krb_accel_cuda_managed_free(rs->phase_pref);
        krb_accel_cuda_managed_free(rs->score_tmp);
        krb_accel_cuda_managed_free(rs->score_pos);
        krb_accel_cuda_managed_free(rs->score_neg);
        krb_accel_cuda_managed_free(rs->flat_offsets);
        krb_accel_cuda_managed_free(rs->flat_lits);
    } else {
        free(rs->assign);
        free(rs->phase_seen);
        free(rs->phase_pref);
        free(rs->score_tmp);
        free(rs->score_pos);
        free(rs->score_neg);
        free(rs->flat_offsets);
        free(rs->flat_lits);
    }
    intvec_free(&rs->trail);
    intvec_free(&rs->trail_lim);
    intvec_free(&rs->decisions);
}

static int satres_enqueue(SatResidualSolver *rs, int lit) {
    int v = (lit < 0) ? -lit : lit;
    signed char want = (lit > 0) ? 1 : 0;
    signed char cur = rs->assign[v];
    if (cur < 0) {
        rs->assign[v] = want;
        rs->phase_seen[v] = 1U;
        rs->phase_pref[v] = (unsigned char)want;
        intvec_push(&rs->trail, lit);
        return 1;
    }
    return cur == want;
}

static int satres_add_clause(SatResidualSolver *rs, const int *lits, int size) {
    int nc;
    SatResidualClause *c;
    if (rs->size >= rs->cap) {
        nc = (rs->cap > 0) ? rs->cap * 2 : 16;
        rs->clauses = (SatResidualClause *)basilisk_xrealloc(rs->clauses, (size_t)nc * sizeof(SatResidualClause));
        rs->cap = nc;
    }
    c = &rs->clauses[rs->size];
    c->size = size;
    c->lits = (int *)basilisk_xmalloc((size_t)(size > 0 ? size : 1) * sizeof(int));
    if (size > 0) memcpy(c->lits, lits, (size_t)size * sizeof(int));
    c->watch0 = 0;
    c->watch1 = (size > 1) ? 1 : 0;
    if (size >= 1) intvec_push(&rs->watches[sat_lit_index(c->lits[c->watch0])], rs->size);
    if (size >= 2) intvec_push(&rs->watches[sat_lit_index(c->lits[c->watch1])], rs->size);
    ++rs->size;
    return rs->size - 1;
}

static int satres_prepare_accel(SatResidualSolver *rs, char *err, size_t errsz) {
    int use_cuda = 0;
    signed char *assign_managed = NULL;
    unsigned char *phase_seen_managed = NULL;
    unsigned char *phase_pref_managed = NULL;
    int *score_tmp_managed = NULL;
    int *score_pos_managed = NULL;
    int *score_neg_managed = NULL;
    int *flat_offsets_managed = NULL;
    int *flat_lits_managed = NULL;

    rs->total_lits = satres_total_lits(rs);
    if (!basilisk_pick_cuda(rs->shared,
                            rs->total_lits,
                            rs->nvars,
                            rs->size,
                            "basilisk residual",
                            &use_cuda,
                            err,
                            errsz)) {
        return 0;
    }
    rs->use_cuda = use_cuda;
    rs->gpu_error = 0;
    if (!use_cuda) {
        if (err != NULL && errsz > 0U) err[0] = '\0';
        return 1;
    }

    if (!basilisk_cuda_alloc_bytes((size_t)rs->nvars + 1U, (void **)&assign_managed, err, errsz) ||
        !basilisk_cuda_alloc_bytes((size_t)rs->nvars + 1U, (void **)&phase_seen_managed, err, errsz) ||
        !basilisk_cuda_alloc_bytes((size_t)rs->nvars + 1U, (void **)&phase_pref_managed, err, errsz) ||
        !krb_accel_cuda_managed_alloc_ints((size_t)rs->nvars + 1U, &score_tmp_managed, err, errsz) ||
        !krb_accel_cuda_managed_alloc_ints((size_t)rs->nvars + 1U, &score_pos_managed, err, errsz) ||
        !krb_accel_cuda_managed_alloc_ints((size_t)rs->nvars + 1U, &score_neg_managed, err, errsz) ||
        !krb_accel_cuda_managed_alloc_ints((size_t)rs->size + 1U, &flat_offsets_managed, err, errsz) ||
        !krb_accel_cuda_managed_alloc_ints(rs->total_lits, &flat_lits_managed, err, errsz)) {
        krb_accel_cuda_managed_free(assign_managed);
        krb_accel_cuda_managed_free(phase_seen_managed);
        krb_accel_cuda_managed_free(phase_pref_managed);
        krb_accel_cuda_managed_free(score_tmp_managed);
        krb_accel_cuda_managed_free(score_pos_managed);
        krb_accel_cuda_managed_free(score_neg_managed);
        krb_accel_cuda_managed_free(flat_offsets_managed);
        krb_accel_cuda_managed_free(flat_lits_managed);
        return 0;
    }

    memcpy(assign_managed, rs->assign, (size_t)rs->nvars + 1U);
    memcpy(phase_seen_managed, rs->phase_seen, (size_t)rs->nvars + 1U);
    memcpy(phase_pref_managed, rs->phase_pref, (size_t)rs->nvars + 1U);
    memset(score_tmp_managed, 0, ((size_t)rs->nvars + 1U) * sizeof(int));
    memset(score_pos_managed, 0, ((size_t)rs->nvars + 1U) * sizeof(int));
    memset(score_neg_managed, 0, ((size_t)rs->nvars + 1U) * sizeof(int));
    satres_build_flat(rs, flat_offsets_managed, flat_lits_managed);

    free(rs->assign);
    free(rs->phase_seen);
    free(rs->phase_pref);
    free(rs->score_tmp);
    free(rs->score_pos);
    free(rs->score_neg);

    rs->assign = assign_managed;
    rs->phase_seen = phase_seen_managed;
    rs->phase_pref = phase_pref_managed;
    rs->score_tmp = score_tmp_managed;
    rs->score_pos = score_pos_managed;
    rs->score_neg = score_neg_managed;
    rs->flat_offsets = flat_offsets_managed;
    rs->flat_lits = flat_lits_managed;
    if (err != NULL && errsz > 0U) err[0] = '\0';
    return 1;
}

static void satres_backtrack_root(SatResidualSolver *rs) {
    int i;
    for (i = 1; i <= rs->nvars; ++i) rs->assign[i] = -1;
    rs->trail.size = 0;
    rs->trail_lim.size = 0;
    rs->decisions.size = 0;
    rs->qhead = 0;
}

static int satres_propagate(SatResidualSolver *rs) {
    while (rs->qhead < rs->trail.size) {
        int lit = rs->trail.data[rs->qhead++];
        IntVec *wl = &rs->watches[sat_lit_index(-lit)];
        int i = 0;
        while (i < wl->size) {
            int clause_idx = wl->data[i];
            SatResidualClause *c = &rs->clauses[clause_idx];
            int false_watch = (c->lits[c->watch0] == -lit) ? 0 : 1;
            int false_pos = false_watch ? c->watch1 : c->watch0;
            int other_pos = false_watch ? c->watch0 : c->watch1;
            int other_lit = c->lits[other_pos];
            int other_ev = sat_lit_eval(rs, other_lit);
            int moved = 0;
            int k;
            if (other_ev > 0) {
                ++i;
                continue;
            }
            for (k = 0; k < c->size; ++k) {
                int cand_lit;
                int cand_ev;
                if (k == false_pos || k == other_pos) continue;
                cand_lit = c->lits[k];
                cand_ev = sat_lit_eval(rs, cand_lit);
                if (cand_ev != 0) {
                    if (false_watch) c->watch1 = k;
                    else c->watch0 = k;
                    intvec_push(&rs->watches[sat_lit_index(cand_lit)], clause_idx);
                    wl->data[i] = wl->data[wl->size - 1];
                    --wl->size;
                    moved = 1;
                    break;
                }
            }
            if (moved) continue;
            if (other_ev < 0) {
                if (!satres_enqueue(rs, other_lit)) return 0;
                ++i;
                continue;
            }
            return 0;
        }
    }
    return 1;
}

static void satres_score_cpu(SatResidualSolver *rs) {
    int ci;
    memset(rs->score_tmp, 0, ((size_t)rs->nvars + 1U) * sizeof(int));
    memset(rs->score_pos, 0, ((size_t)rs->nvars + 1U) * sizeof(int));
    memset(rs->score_neg, 0, ((size_t)rs->nvars + 1U) * sizeof(int));
    for (ci = 0; ci < rs->size; ++ci) {
        const SatResidualClause *c = &rs->clauses[ci];
        int sat = 0;
        int open = 0;
        int j;
        for (j = 0; j < c->size; ++j) {
            int ev = sat_lit_eval(rs, c->lits[j]);
            if (ev > 0) {
                sat = 1;
                break;
            }
            if (ev < 0) ++open;
        }
        if (sat) continue;
        for (j = 0; j < c->size; ++j) {
            int lit = c->lits[j];
            int v = (lit < 0) ? -lit : lit;
            int w = (open <= 2) ? 6 : (open == 3 ? 3 : 1);
            if (rs->assign[v] >= 0) continue;
            rs->score_tmp[v] += w;
            if (lit > 0) rs->score_pos[v] += w;
            else rs->score_neg[v] += w;
        }
    }
}

static int satres_try_score_cuda(SatResidualSolver *rs) {
    char err[256];
    if (!rs->use_cuda || rs->gpu_error) {
        return 0;
    }
    if (!krb_accel_cuda_score_cnf_branching(rs->flat_offsets,
                                            rs->flat_lits,
                                            rs->size,
                                            rs->nvars,
                                            rs->assign,
                                            NULL,
                                            0,
                                            rs->score_tmp,
                                            rs->score_pos,
                                            rs->score_neg,
                                            err,
                                            sizeof(err))) {
        rs->gpu_error = 1;
        if (rs->shared != NULL) {
            ++rs->shared->cuda_fallbacks;
            if (rs->shared->opt.verbose >= 1) {
                fprintf(stderr, "c basilisk cuda fallback[residual]: %s\n", err);
            }
        }
        return 0;
    }
    if (rs->shared != NULL) {
        ++rs->shared->cuda_residual_branch_calls;
    }
    return 1;
}

static int satres_choose_lit(SatResidualSolver *rs) {
    int ci;
    int best_lit = 0;
    int best_score = -1;
    if (!satres_try_score_cuda(rs)) {
        satres_score_cpu(rs);
    }
    for (ci = 1; ci <= rs->nvars; ++ci) {
        int prefer_pos;
        if (rs->assign[ci] >= 0) continue;
        if (rs->score_tmp[ci] > best_score) {
            best_score = rs->score_tmp[ci];
            prefer_pos = rs->phase_seen[ci] ? (rs->phase_pref[ci] != 0U) : (rs->score_pos[ci] >= rs->score_neg[ci]);
            best_lit = prefer_pos ? ci : -ci;
        }
    }
    return best_lit;
}

static int satres_solve(SatResidualSolver *rs) {
    int restart_budget = 64;
    int since_restart = 0;
    for (;;) {
        int lit;
        if (!satres_propagate(rs)) {
            IntVec learn;
            int i;
            if (rs->decisions.size == 0) return 0;
            intvec_init(&learn);
            for (i = 0; i < rs->decisions.size; ++i) {
                intvec_push(&learn, -rs->decisions.data[i]);
            }
            satres_add_clause(rs, learn.data, learn.size);
            intvec_free(&learn);
            ++rs->conflicts;
            satres_backtrack_root(rs);
            if (++since_restart >= restart_budget) {
                since_restart = 0;
                if (restart_budget < 4096) restart_budget *= 2;
            }
            continue;
        }
        lit = satres_choose_lit(rs);
        if (lit == 0) return 1;
        intvec_push(&rs->trail_lim, rs->trail.size);
        intvec_push(&rs->decisions, lit);
        if (!satres_enqueue(rs, lit)) {
            return 0;
        }
    }
}

static void solver_build_residual_subcnf(const BasiliskSolver *s, BasiliskCNF *out) {
    int *var_map;
    IntVec lits;
    int ci;
    int next_var = 0;
    cnf_init(out);
    var_map = (int *)calloc((size_t)s->cnf.nvars + 1U, sizeof(int));
    if (s->cnf.nvars > 0 && var_map == NULL) basilisk_die("out of memory");
    intvec_init(&lits);
    for (ci = 0; ci < s->cnf.clauses.size; ++ci) {
        const Clause *c = &s->cnf.clauses.data[ci];
        int sat = 0;
        int j;
        for (j = 0; j < c->size; ++j) {
            if (lit_eval(s->assign, c->lits[j]) > 0) {
                sat = 1;
                break;
            }
        }
        if (sat) continue;
        lits.size = 0;
        for (j = 0; j < c->size; ++j) {
            int lit = c->lits[j];
            int ev = lit_eval(s->assign, lit);
            if (ev < 0) {
                int v = (lit < 0) ? -lit : lit;
                if (var_map[v] == 0) {
                    var_map[v] = ++next_var;
                }
                intvec_push(&lits, (lit < 0) ? -var_map[v] : var_map[v]);
            }
        }
        clausevec_push_copy(&out->clauses, lits.data, lits.size);
    }
    out->nvars = next_var;
    intvec_free(&lits);
    free(var_map);
}

static int satres_exists_model_for_solver(BasiliskSolver *s) {
    BasiliskCNF residual;
    SatResidualSolver rs;
    int ci;
    int ok;
    char err[256];
    solver_build_residual_subcnf(s, &residual);
    satres_init(&rs, residual.nvars, s->shared);
    for (ci = 0; ci < residual.clauses.size; ++ci) {
        satres_add_clause(&rs, residual.clauses.data[ci].lits, residual.clauses.data[ci].size);
    }
    if (!satres_prepare_accel(&rs, err, sizeof(err))) {
        basilisk_die(err);
    }
    ok = satres_solve(&rs);
    satres_free(&rs);
    cnf_free(&residual);
    return ok;
}

#if defined(SLIME_NO_MAIN)
static void solver_build_slime_assumptions(BasiliskSolver *s) {
    int i;
    s->slime_assumptions.size = 0;
    for (i = 0; i < s->trail.size; ++i) {
        int v = s->trail.data[i];
        if (s->assign[v] < 0) continue;
        intvec_push(&s->slime_assumptions, (s->assign[v] > 0) ? v : -v);
    }
}

static long long slime_ct_activity(const SlimeSatStats *stats) {
    return stats->ct_added + stats->ct_merged + stats->ct_escaped + stats->ct_probe_added;
}

static void solver_autotune_slime(BasiliskSolver *s, const SlimeSatStats *stats) {
    int changed = 0;
    if (s->slime_handle == NULL) return;

    if (s->slime_opt.use_hess) {
        if (stats->hess_calls > 0) {
            if (stats->hess_sat_hits > 0) {
                s->slime_hess_nohit_streak = 0;
            } else {
                ++s->slime_hess_nohit_streak;
            }
        }
        if (s->slime_hess_nohit_streak >= (uint64_t)s->shared->opt.slime_hess_nohit_limit) {
            s->slime_opt.use_hess = 0;
            s->slime_hess_nohit_streak = 0;
            ++s->shared->slime_tune_hess_off;
            changed = 1;
        }
    } else {
        s->slime_hess_nohit_streak = 0;
    }

    if (s->slime_opt.use_ct) {
        if (slime_ct_activity(stats) > 0) {
            s->slime_ct_idle_streak = 0;
        } else {
            ++s->slime_ct_idle_streak;
        }
        if (s->slime_ct_idle_streak >= (uint64_t)s->shared->opt.slime_ct_idle_soft_limit &&
            (s->slime_opt.ct_escape_rounds > 1 ||
             (s->slime_opt.ct_probe_restarts > 0 &&
              s->slime_opt.ct_probe_restarts < s->shared->opt.slime_ct_probe_restarts_max))) {
            if (s->slime_opt.ct_escape_rounds > 1) {
                s->slime_opt.ct_escape_rounds = (s->slime_opt.ct_escape_rounds + 1) / 2;
            }
            if (s->slime_opt.ct_probe_restarts > 0 &&
                s->slime_opt.ct_probe_restarts < s->shared->opt.slime_ct_probe_restarts_max) {
                int next = s->slime_opt.ct_probe_restarts * 2;
                if (next > s->shared->opt.slime_ct_probe_restarts_max) {
                    next = s->shared->opt.slime_ct_probe_restarts_max;
                }
                s->slime_opt.ct_probe_restarts = next;
            }
            s->slime_ct_idle_streak = 0;
            ++s->shared->slime_tune_ct_soft;
            changed = 1;
        } else if (s->slime_ct_idle_streak >= (uint64_t)s->shared->opt.slime_ct_idle_off_limit) {
            s->slime_opt.use_ct = 0;
            s->slime_ct_idle_streak = 0;
            ++s->shared->slime_tune_ct_off;
            changed = 1;
        }
    } else {
        s->slime_ct_idle_streak = 0;
    }

    if (changed) {
        slime_sat_handle_reconfigure((SlimeSatHandle *)s->slime_handle, &s->slime_opt);
    }
}

static int slime_exists_model_for_solver(BasiliskSolver *s) {
    SlimeSatStats stats;
    solver_build_slime_assumptions(s);
    if (s->slime_handle == NULL) return 0;
    memset(&stats, 0, sizeof(stats));
    ++s->shared->slime_sat_calls;
    {
        int rc = slime_sat_handle_solve((SlimeSatHandle *)s->slime_handle,
                                        s->slime_assumptions.data,
                                        s->slime_assumptions.size,
                                        &stats,
                                        NULL);
        if (stats.conflicts > 0) s->shared->slime_conflicts += (uint64_t)stats.conflicts;
        if (stats.decisions > 0) s->shared->slime_decisions += (uint64_t)stats.decisions;
        if (stats.propagations > 0) s->shared->slime_propagations += (uint64_t)stats.propagations;
        if (stats.restarts > 0) s->shared->slime_restarts += (uint64_t)stats.restarts;
        if (stats.hess_calls > 0) s->shared->slime_hess_calls += (uint64_t)stats.hess_calls;
        if (stats.hess_sat_hits > 0) s->shared->slime_hess_hits += (uint64_t)stats.hess_sat_hits;
        if (stats.ct_added > 0) s->shared->slime_ct_added += (uint64_t)stats.ct_added;
        if (stats.ct_merged > 0) s->shared->slime_ct_merged += (uint64_t)stats.ct_merged;
        if (stats.ct_escaped > 0) s->shared->slime_ct_escaped += (uint64_t)stats.ct_escaped;
        if (stats.ct_probe_added > 0) s->shared->slime_ct_probe_added += (uint64_t)stats.ct_probe_added;
        solver_autotune_slime(s, &stats);
        return rc;
    }
}
#endif

static void solver_count_subproblem(BasiliskSolver *parent,
                                    BasiliskCNF *subcnf,
                                    int depth,
                                    BigUInt *out) {
    BasiliskSolver child;
    char err[256];
    memset(&child, 0, sizeof(child));
    child.shared = parent->shared;
    cnf_move(&child.cnf, subcnf);
    intvec_init(&child.trail);
    if (!solver_prepare(&child, err, sizeof(err))) {
        basilisk_die(err);
    }
    solver_count_core(&child, depth, out);
    solver_release(&child);
}

static int solver_exists_model(BasiliskSolver *s, int depth) {
    int mark = s->trail.size;
    ++s->shared->nodes;
    if (depth > s->shared->max_depth) s->shared->max_depth = depth;
    if (!solver_propagate(s)) {
        solver_undo(s, mark);
        return 0;
    }
    if (solver_all_satisfied(s)) {
        solver_undo(s, mark);
        return 1;
    }
    ++s->shared->residual_sat_calls;
#if defined(SLIME_NO_MAIN)
    {
        int rc = slime_exists_model_for_solver(s);
        if (rc == 10) {
            solver_undo(s, mark);
            return 1;
        }
        if (rc == 20) {
            solver_undo(s, mark);
            return 0;
        }
    }
#endif
    if (satres_exists_model_for_solver(s)) {
        solver_undo(s, mark);
        return 1;
    }
    solver_undo(s, mark);
    return 0;
}

static void solver_count_core(BasiliskSolver *s, int depth, BigUInt *out) {
    int mark = s->trail.size;
    int projected = (s->shared->opt.mode == BASILISK_MODE_PROJECT);
    int lit;
    char *cache_key = NULL;
    uint64_t cache_hash = 0ULL;
    const BasiliskCacheEntry *cache_hit = NULL;
    int free_all = 0;
    int free_project = 0;
    ComponentVec comps;
    BigUInt left;
    BigUInt right;
    ++s->shared->nodes;
    if (depth > s->shared->max_depth) s->shared->max_depth = depth;
    big_set_zero(out);
    if (!solver_propagate(s)) {
        solver_undo(s, mark);
        return;
    }
    if (solver_all_satisfied(s)) {
        big_set_pow2(out, projected ? solver_count_unassigned_project(s) : solver_count_unassigned_all(s));
        solver_undo(s, mark);
        return;
    }

    cache_key = solver_build_residual_key(s);
    cache_hash = cache_hash_string(cache_key);
    cache_hit = cache_lookup(&s->shared->cache, cache_key, cache_hash);
    if (cache_hit != NULL) {
        big_copy(out, &cache_hit->value);
        ++s->shared->cache_hits;
        free(cache_key);
        solver_undo(s, mark);
        return;
    }

    if (projected && !solver_has_unassigned_project(s)) {
        ++s->shared->sat_checks;
        if (solver_exists_model(s, depth + 1)) {
            big_set_one(out);
        } else {
            big_set_zero(out);
        }
        cache_store(&s->shared->cache, cache_key, cache_hash, out);
        free(cache_key);
        solver_undo(s, mark);
        return;
    }

    solver_find_components(s, &comps, &free_all, &free_project);
    if (comps.size > 1 || (!projected && free_all > 0) || (projected && free_project > 0)) {
        BigUInt acc;
        int i;
        big_init(&acc);
        big_set_one(&acc);
        if (comps.size > 1) ++s->shared->component_splits;
        for (i = 0; i < comps.size; ++i) {
            BasiliskCNF subcnf;
            BigUInt subcount;
            big_init(&subcount);
            solver_build_component_subcnf(s, &comps.data[i], &subcnf);
            solver_count_subproblem(s, &subcnf, depth + 1, &subcount);
            big_mul_assign(&acc, &subcount);
            big_free(&subcount);
            cnf_free(&subcnf);
            if (big_is_zero(&acc)) break;
        }
        if ((!projected && free_all > 0) || (projected && free_project > 0)) {
            BigUInt factor;
            big_init(&factor);
            big_set_pow2(&factor, projected ? free_project : free_all);
            big_mul_assign(&acc, &factor);
            big_free(&factor);
        }
        big_copy(out, &acc);
        cache_store(&s->shared->cache, cache_key, cache_hash, out);
        big_free(&acc);
        compvec_free(&comps);
        free(cache_key);
        solver_undo(s, mark);
        return;
    }
    compvec_free(&comps);

    if (projected) {
        ++s->shared->sat_checks;
        if (!solver_exists_model(s, depth + 1)) {
            free(cache_key);
            solver_undo(s, mark);
            return;
        }
    }

    lit = solver_choose_branch_lit(s, projected);
    if (lit == 0) {
        if (projected) big_set_one(out);
        cache_store(&s->shared->cache, cache_key, cache_hash, out);
        free(cache_key);
        solver_undo(s, mark);
        return;
    }
    big_init(&left);
    big_init(&right);
    ++s->shared->decisions;
    if (solver_assign_lit(s, lit)) {
        solver_count_core(s, depth + 1, &left);
    }
    solver_undo(s, mark);
    if (solver_assign_lit(s, -lit)) {
        solver_count_core(s, depth + 1, &right);
    }
    solver_undo(s, mark);
    big_add_to(out, &left, &right);
    cache_store(&s->shared->cache, cache_key, cache_hash, out);
    free(cache_key);
    big_free(&left);
    big_free(&right);
}

static void solver_count_total(BasiliskSolver *s, int depth, BigUInt *out) {
    solver_count_core(s, depth, out);
}

static void solver_count_projected(BasiliskSolver *s, int depth, BigUInt *out) {
    solver_count_core(s, depth, out);
}

static void solver_init(BasiliskSolver *s, BasiliskShared *shared) {
    memset(s, 0, sizeof(*s));
    s->shared = shared;
    intvec_init(&s->trail);
    intvec_init(&s->slime_assumptions);
}

static void solver_release(BasiliskSolver *s) {
    cnf_free(&s->cnf);
    if (s->use_cuda) {
        krb_accel_cuda_managed_free(s->assign);
        krb_accel_cuda_managed_free(s->score_tmp);
        krb_accel_cuda_managed_free(s->score_pos);
        krb_accel_cuda_managed_free(s->score_neg);
        krb_accel_cuda_managed_free(s->active_project);
        krb_accel_cuda_managed_free(s->phase_seen);
        krb_accel_cuda_managed_free(s->phase_pref);
        krb_accel_cuda_managed_free(s->flat_offsets);
        krb_accel_cuda_managed_free(s->flat_lits);
    } else {
        free(s->assign);
        free(s->score_tmp);
        free(s->score_pos);
        free(s->score_neg);
        free(s->active_project);
        free(s->phase_seen);
        free(s->phase_pref);
        free(s->flat_offsets);
        free(s->flat_lits);
    }
    free(s->slime_clause_ptrs);
    free(s->slime_clause_sizes);
#if defined(SLIME_NO_MAIN)
    slime_sat_handle_destroy((SlimeSatHandle *)s->slime_handle);
    memset(&s->slime_opt, 0, sizeof(s->slime_opt));
    s->slime_hess_nohit_streak = 0;
    s->slime_ct_idle_streak = 0;
#endif
    intvec_free(&s->trail);
    intvec_free(&s->slime_assumptions);
    s->assign = NULL;
    s->score_tmp = NULL;
    s->score_pos = NULL;
    s->score_neg = NULL;
    s->active_project = NULL;
    s->phase_seen = NULL;
    s->phase_pref = NULL;
    s->flat_offsets = NULL;
    s->flat_lits = NULL;
    s->slime_clause_ptrs = NULL;
    s->slime_clause_sizes = NULL;
    s->slime_handle = NULL;
    s->total_lits = 0U;
    s->use_cuda = 0;
    s->gpu_error = 0;
}

static int solver_prepare(BasiliskSolver *s, char *err, size_t errsz) {
    int v;
    int want_cuda = 0;
    signed char *assign = NULL;
    int *score_tmp = NULL;
    int *score_pos = NULL;
    int *score_neg = NULL;
    unsigned char *active_project = NULL;
    unsigned char *phase_seen = NULL;
    unsigned char *phase_pref = NULL;
    int *flat_offsets = NULL;
    int *flat_lits = NULL;

    s->total_lits = cnf_total_lits(&s->cnf);
    if (!basilisk_pick_cuda(s->shared,
                            s->total_lits,
                            s->cnf.nvars,
                            s->cnf.clauses.size,
                            "basilisk",
                            &want_cuda,
                            err,
                            errsz)) {
        return 0;
    }
    s->use_cuda = want_cuda;
    s->gpu_error = 0;

    if (want_cuda) {
        if (!basilisk_cuda_alloc_bytes((size_t)s->cnf.nvars + 1U, (void **)&assign, err, errsz) ||
            !krb_accel_cuda_managed_alloc_ints((size_t)s->cnf.nvars + 1U, &score_tmp, err, errsz) ||
            !krb_accel_cuda_managed_alloc_ints((size_t)s->cnf.nvars + 1U, &score_pos, err, errsz) ||
            !krb_accel_cuda_managed_alloc_ints((size_t)s->cnf.nvars + 1U, &score_neg, err, errsz) ||
            !basilisk_cuda_alloc_bytes((size_t)s->cnf.nvars + 1U, (void **)&active_project, err, errsz) ||
            !basilisk_cuda_alloc_bytes((size_t)s->cnf.nvars + 1U, (void **)&phase_seen, err, errsz) ||
            !basilisk_cuda_alloc_bytes((size_t)s->cnf.nvars + 1U, (void **)&phase_pref, err, errsz) ||
            !krb_accel_cuda_managed_alloc_ints((size_t)s->cnf.clauses.size + 1U, &flat_offsets, err, errsz) ||
            !krb_accel_cuda_managed_alloc_ints(s->total_lits, &flat_lits, err, errsz)) {
            krb_accel_cuda_managed_free(assign);
            krb_accel_cuda_managed_free(score_tmp);
            krb_accel_cuda_managed_free(score_pos);
            krb_accel_cuda_managed_free(score_neg);
            krb_accel_cuda_managed_free(active_project);
            krb_accel_cuda_managed_free(phase_seen);
            krb_accel_cuda_managed_free(phase_pref);
            krb_accel_cuda_managed_free(flat_offsets);
            krb_accel_cuda_managed_free(flat_lits);
            return 0;
        }
    } else {
        assign = (signed char *)malloc((size_t)s->cnf.nvars + 1U);
        score_tmp = (int *)calloc((size_t)s->cnf.nvars + 1U, sizeof(int));
        score_pos = (int *)calloc((size_t)s->cnf.nvars + 1U, sizeof(int));
        score_neg = (int *)calloc((size_t)s->cnf.nvars + 1U, sizeof(int));
        active_project = (unsigned char *)calloc((size_t)s->cnf.nvars + 1U, sizeof(unsigned char));
        phase_seen = (unsigned char *)calloc((size_t)s->cnf.nvars + 1U, sizeof(unsigned char));
        phase_pref = (unsigned char *)calloc((size_t)s->cnf.nvars + 1U, sizeof(unsigned char));
    }
    if ((s->cnf.nvars > 0) &&
        (assign == NULL || score_tmp == NULL || score_pos == NULL || score_neg == NULL ||
         active_project == NULL || phase_seen == NULL || phase_pref == NULL)) {
        snprintf(err, errsz, "out of memory allocating solver state");
        if (want_cuda) {
            krb_accel_cuda_managed_free(assign);
            krb_accel_cuda_managed_free(score_tmp);
            krb_accel_cuda_managed_free(score_pos);
            krb_accel_cuda_managed_free(score_neg);
            krb_accel_cuda_managed_free(active_project);
            krb_accel_cuda_managed_free(phase_seen);
            krb_accel_cuda_managed_free(phase_pref);
            krb_accel_cuda_managed_free(flat_offsets);
            krb_accel_cuda_managed_free(flat_lits);
        } else {
            free(assign);
            free(score_tmp);
            free(score_pos);
            free(score_neg);
            free(active_project);
            free(phase_seen);
            free(phase_pref);
        }
        return 0;
    }

    s->assign = assign;
    s->score_tmp = score_tmp;
    s->score_pos = score_pos;
    s->score_neg = score_neg;
    s->active_project = active_project;
    s->phase_seen = phase_seen;
    s->phase_pref = phase_pref;
    s->flat_offsets = flat_offsets;
    s->flat_lits = flat_lits;

    if (s->use_cuda) {
        memset(s->score_tmp, 0, ((size_t)s->cnf.nvars + 1U) * sizeof(int));
        memset(s->score_pos, 0, ((size_t)s->cnf.nvars + 1U) * sizeof(int));
        memset(s->score_neg, 0, ((size_t)s->cnf.nvars + 1U) * sizeof(int));
        memset(s->active_project, 0, (size_t)s->cnf.nvars + 1U);
        memset(s->phase_seen, 0, (size_t)s->cnf.nvars + 1U);
        memset(s->phase_pref, 0, (size_t)s->cnf.nvars + 1U);
        cnf_build_flat(&s->cnf, s->flat_offsets, s->flat_lits);
    }
    for (v = 0; v <= s->cnf.nvars; ++v) s->assign[v] = -1;

    if (s->shared->opt.mode == BASILISK_MODE_PROJECT) {
        if (s->shared->opt.project_mode == BASILISK_PROJECT_ALL) {
            for (v = 1; v <= s->cnf.nvars; ++v) {
                s->active_project[v] = 1U;
            }
            s->active_project_count = s->cnf.nvars;
        } else if (s->shared->opt.project_mode == BASILISK_PROJECT_IND || s->cnf.have_decl_project) {
            if (!s->cnf.have_decl_project && s->shared->opt.project_mode == BASILISK_PROJECT_IND) {
                snprintf(err, errsz, "projected mode 'ind' requires DIMACS 'c ind ... 0' comments");
                return 0;
            }
            for (v = 1; v <= s->cnf.nvars; ++v) {
                s->active_project[v] = s->cnf.decl_project[v];
                if (s->active_project[v]) ++s->active_project_count;
            }
        } else {
            for (v = 1; v <= s->cnf.nvars; ++v) {
                s->active_project[v] = 1U;
            }
            s->active_project_count = s->cnf.nvars;
        }
    }
#if defined(SLIME_NO_MAIN)
    if (s->cnf.clauses.size > 0) {
        int ci;
        s->slime_clause_ptrs = (const int **)calloc((size_t)s->cnf.clauses.size, sizeof(int *));
        s->slime_clause_sizes = (int *)calloc((size_t)s->cnf.clauses.size, sizeof(int));
        if (s->slime_clause_ptrs == NULL || s->slime_clause_sizes == NULL) {
            snprintf(err, errsz, "out of memory preparing slime clause view");
            return 0;
        }
        for (ci = 0; ci < s->cnf.clauses.size; ++ci) {
            s->slime_clause_ptrs[ci] = s->cnf.clauses.data[ci].lits;
            s->slime_clause_sizes[ci] = s->cnf.clauses.data[ci].size;
        }
    }
    {
        memset(&s->slime_opt, 0, sizeof(s->slime_opt));
        s->slime_hess_nohit_streak = 0;
        s->slime_ct_idle_streak = 0;
        s->slime_opt.heuristic_mode = 0;
        s->slime_opt.use_mab = 0;
        s->slime_opt.mabc = 4.0;
        s->slime_opt.use_hess = (s->shared->opt.slime_init_hess >= 0) ?
                                    s->shared->opt.slime_init_hess :
                                    ((s->cnf.nvars <= 64 && s->cnf.clauses.size <= 256) ? 1 : 0);
        s->slime_opt.use_ct = s->shared->opt.slime_init_ct;
        s->slime_opt.ct_lbd_max = 6;
        s->slime_opt.ct_maxlen = 12;
        s->slime_opt.ct_max_cubes = 40000;
        s->slime_opt.ct_buddy_merge = 0;
        s->slime_opt.ct_escape_rounds = s->shared->opt.slime_init_ct_escape_rounds;
        s->slime_opt.ct_probe_restarts = s->shared->opt.slime_init_ct_probe_restarts;
        s->slime_handle = slime_sat_handle_create(s->cnf.nvars,
                                                  s->cnf.clauses.size,
                                                  s->slime_clause_ptrs,
                                                  s->slime_clause_sizes,
                                                  &s->slime_opt);
        if (s->slime_handle == NULL) {
            snprintf(err, errsz, "failed to create incremental slime SAT handle");
            return 0;
        }
    }
#endif
    return 1;
}

static void print_usage(FILE *out) {
    fprintf(out,
            "usage: %s <input.cnf> [options]\n"
            "       %s --selftest\n"
            "\n"
            "options:\n"
            "  --parallel <mode>  auto|off|threads|mpi|hybrid\n"
            "  --jobs <n>         local worker count for threaded modes\n"
            "  --split-depth <n>  parallel splitting depth scaffold\n"
            "  --portfolio <n>    portfolio multiplicity scaffold\n"
            "  --sync-ms <n>      synchronization cadence scaffold\n"
            "  --mode <name>      count|project\n"
            "  --project <name>   auto|all|ind (only meaningful with --mode project)\n"
            "  --stats            print execution statistics\n"
            "  --cuda <mode>      auto|on|off for experimental CUDA branch scoring\n"
            "  --cuda-device <n>  CUDA device id (-1 keeps runtime default)\n"
            "  --cuda-min-lits <n>\n"
            "                     minimum residual literal volume before trying CUDA\n"
            "  --hess|--no-hess   initial residual HESS policy (default: auto)\n"
            "  --ct|--no-ct       initial residual CoverTrace policy\n"
            "  --ct-escape-rounds <n>\n"
            "                     initial residual CoverTrace escape rounds\n"
            "  --ct-probe-restarts <n>\n"
            "                     initial residual root-probe restart period\n"
            "  --slime-hess-nohit-limit <n>\n"
            "                     disable residual HESS after n no-hit calls\n"
            "  --slime-ct-idle-soft-limit <n>\n"
            "                     damp residual CoverTrace after n idle calls\n"
            "  --slime-ct-idle-off-limit <n>\n"
            "                     disable residual CoverTrace after n idle calls\n"
            "  --slime-ct-probe-max-restarts <n>\n"
            "                     ceiling for retuned residual ct_probe_restarts\n"
            "  --verbose <0..3>   verbosity level\n"
            "  --selftest         run built-in tests\n",
            g_basilisk_prog, g_basilisk_prog);
}

static int parse_mode(const char *s, BasiliskMode *out) {
    if (strcmp(s, "count") == 0) {
        *out = BASILISK_MODE_COUNT;
        return 1;
    }
    if (strcmp(s, "project") == 0) {
        *out = BASILISK_MODE_PROJECT;
        return 1;
    }
    return 0;
}

static int parse_project_mode(const char *s, BasiliskProjectMode *out) {
    if (strcmp(s, "auto") == 0) {
        *out = BASILISK_PROJECT_AUTO;
        return 1;
    }
    if (strcmp(s, "all") == 0) {
        *out = BASILISK_PROJECT_ALL;
        return 1;
    }
    if (strcmp(s, "ind") == 0) {
        *out = BASILISK_PROJECT_IND;
        return 1;
    }
    return 0;
}

static int parse_positive_int_opt(const char *s, int *out) {
    long long v = 0;
    if (!parse_ll_token(s, &v) || v < 1 || v > INT_MAX) return 0;
    *out = (int)v;
    return 1;
}

static int parse_nonnegative_int_opt(const char *s, int *out) {
    long long v = 0;
    if (!parse_ll_token(s, &v) || v < 0 || v > INT_MAX) return 0;
    *out = (int)v;
    return 1;
}

static int parse_cuda_device_opt(const char *s, int *out) {
    long long v = 0;
    if (!parse_ll_token(s, &v) || v < -1 || v > INT_MAX) return 0;
    *out = (int)v;
    return 1;
}

static int parse_nonnegative_size_opt(const char *s, size_t *out) {
    long long v = 0;
    if (!parse_ll_token(s, &v) || v < 0) return 0;
    *out = (size_t)v;
    return 1;
}

static int write_text_file(const char *path, const char *text) {
    FILE *fp = fopen(path, "wb");
    size_t n;
    if (fp == NULL) return 0;
    n = strlen(text);
    if (fwrite(text, 1U, n, fp) != n) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return 1;
}

static int run_one_file(const BasiliskOptions *opt, char *err, size_t errsz) {
    BasiliskShared shared;
    BasiliskSolver s;
    BigUInt count;
    char *count_str = NULL;
    shared_init(&shared, opt);
    solver_init(&s, &shared);
    big_init(&count);

    if (!cnf_parse_dimacs(&s.cnf, opt->input_path, err, errsz)) {
        goto fail;
    }
    if (!solver_prepare(&s, err, errsz)) {
        goto fail;
    }
    if (opt->verbose >= 2) {
        fprintf(stderr,
                "c basilisk accel=%s mode=%s total_lits=%zu cuda_min_lits=%zu\n",
                s.use_cuda ? "cuda" : "cpu",
                krb_accel_mode_name(opt->cuda_mode),
                s.total_lits,
                opt->cuda_min_lits);
    }

    if (opt->mode == BASILISK_MODE_PROJECT) {
        solver_count_projected(&s, 0, &count);
    } else {
        solver_count_total(&s, 0, &count);
    }

    if (big_is_zero(&count)) {
        puts("s UNSATISFIABLE");
        puts("s mc 0");
    } else {
        count_str = big_to_string(&count);
        printf("s mc %s\n", count_str);
    }

    if (opt->stats) {
        double elapsed = basilisk_now_sec() - ((double)shared.started / (double)CLOCKS_PER_SEC);
        printf("c basilisk stats vars=%d clauses=%d mode=%s project_vars=%d nodes=%llu decisions=%llu propagations=%llu sat_checks=%llu residual_sat=%llu cache_hits=%llu comp_splits=%llu depth=%d sec=%.6f\n",
               s.cnf.nvars,
               s.cnf.clauses.size,
               (opt->mode == BASILISK_MODE_PROJECT) ? "project" : "count",
               s.active_project_count,
               (unsigned long long)shared.nodes,
               (unsigned long long)shared.decisions,
               (unsigned long long)shared.propagations,
               (unsigned long long)shared.sat_checks,
               (unsigned long long)shared.residual_sat_calls,
               (unsigned long long)shared.cache_hits,
               (unsigned long long)shared.component_splits,
               shared.max_depth,
               elapsed);
        if (shared.slime_sat_calls > 0) {
            printf("c basilisk slime calls=%llu conflicts=%llu decisions=%llu propagations=%llu restarts=%llu hess_calls=%llu hess_hits=%llu ct_added=%llu ct_merged=%llu ct_escaped=%llu ct_probe=%llu tune_hess_off=%llu tune_ct_soft=%llu tune_ct_off=%llu\n",
                   (unsigned long long)shared.slime_sat_calls,
                   (unsigned long long)shared.slime_conflicts,
                   (unsigned long long)shared.slime_decisions,
                   (unsigned long long)shared.slime_propagations,
                   (unsigned long long)shared.slime_restarts,
                   (unsigned long long)shared.slime_hess_calls,
                   (unsigned long long)shared.slime_hess_hits,
                   (unsigned long long)shared.slime_ct_added,
                   (unsigned long long)shared.slime_ct_merged,
                   (unsigned long long)shared.slime_ct_escaped,
                   (unsigned long long)shared.slime_ct_probe_added,
                   (unsigned long long)shared.slime_tune_hess_off,
                   (unsigned long long)shared.slime_tune_ct_soft,
                   (unsigned long long)shared.slime_tune_ct_off);
        }
        if (shared.cuda_branch_calls > 0 || shared.cuda_residual_branch_calls > 0 || shared.cuda_fallbacks > 0) {
            printf("c basilisk cuda branch_calls=%llu residual_calls=%llu fallbacks=%llu\n",
                   (unsigned long long)shared.cuda_branch_calls,
                   (unsigned long long)shared.cuda_residual_branch_calls,
                   (unsigned long long)shared.cuda_fallbacks);
        }
    }

    free(count_str);
    big_free(&count);
    solver_release(&s);
    shared_free(&shared);
    return 1;

fail:
    free(count_str);
    big_free(&count);
    solver_release(&s);
    shared_free(&shared);
    return 0;
}

static int selftest_case(const char *name,
                         const char *text,
                         BasiliskMode mode,
                         BasiliskProjectMode pmode,
                         const char *expect_line) {
    char tmp_name[256];
    unsigned long stamp = (unsigned long)time(NULL);
    stamp ^= (unsigned long)clock();
    stamp ^= (unsigned long)(uintptr_t)(const void *)text;
    stamp ^= (unsigned long)(uintptr_t)(const void *)expect_line;
    BasiliskOptions opt;
    int ok = 0;
    char err[512];
    snprintf(tmp_name, sizeof(tmp_name), "basilisk_selftest_%s_%lu.cnf", name, stamp);
    if (!write_text_file(tmp_name, text)) {
        fprintf(stderr, "selftest[%s]: failed to write temp file\n", name);
        return 0;
    }
    basilisk_options_init(&opt);
    opt.mode = mode;
    opt.project_mode = pmode;
    opt.input_path = tmp_name;

    {
        BasiliskShared shared;
        BasiliskSolver s;
        BigUInt count;
        char *count_str = NULL;
        shared_init(&shared, &opt);
        solver_init(&s, &shared);
        big_init(&count);
        if (!cnf_parse_dimacs(&s.cnf, opt.input_path, err, sizeof(err))) {
            fprintf(stderr, "selftest[%s]: %s\n", name, err);
            goto cleanup_local;
        }
        if (!solver_prepare(&s, err, sizeof(err))) {
            fprintf(stderr, "selftest[%s]: %s\n", name, err);
            goto cleanup_local;
        }
        if (mode == BASILISK_MODE_PROJECT) solver_count_projected(&s, 0, &count);
        else solver_count_total(&s, 0, &count);
        count_str = big_to_string(&count);
        if (strcmp(count_str, expect_line) != 0) {
            fprintf(stderr, "selftest[%s]: expected count %s got %s\n", name, expect_line, count_str);
            free(count_str);
            goto cleanup_local;
        }
        free(count_str);
        ok = 1;
cleanup_local:
        big_free(&count);
        solver_release(&s);
        shared_free(&shared);
    }

    remove(tmp_name);
    return ok;
}

static int run_selftests(void) {
    const char *sat_case =
        "p cnf 2 1\n"
        "1 0\n";
    const char *unsat_case =
        "p cnf 1 2\n"
        "1 0\n"
        "-1 0\n";
    const char *project_case =
        "c ind 1 0\n"
        "p cnf 2 1\n"
        "1 2 0\n";
    const char *independent_case =
        "p cnf 4 2\n"
        "1 2 0\n"
        "3 4 0\n";
    const char *independent_project_case =
        "c ind 1 3 0\n"
        "p cnf 4 2\n"
        "1 2 0\n"
        "3 4 0\n";
    const char *cache_free_var_regression =
        "p cnf 4 5\n"
        "2 4 3 -1 0\n"
        "2 4 0\n"
        "-1 -3 4 2 0\n"
        "3 4 2 0\n"
        "3 -1 0\n";
    const char *empty_case =
        "p cnf 3 0\n";

    if (!selftest_case("sat_small", sat_case, BASILISK_MODE_COUNT, BASILISK_PROJECT_AUTO, "2")) return 0;
    if (!selftest_case("unsat_small", unsat_case, BASILISK_MODE_COUNT, BASILISK_PROJECT_AUTO, "0")) return 0;
    if (!selftest_case("project_ind", project_case, BASILISK_MODE_PROJECT, BASILISK_PROJECT_AUTO, "2")) return 0;
    if (!selftest_case("independent_product", independent_case, BASILISK_MODE_COUNT, BASILISK_PROJECT_AUTO, "9")) return 0;
    if (!selftest_case("independent_project", independent_project_case, BASILISK_MODE_PROJECT, BASILISK_PROJECT_AUTO, "4")) return 0;
    if (!selftest_case("cache_free_var_regression", cache_free_var_regression, BASILISK_MODE_COUNT, BASILISK_PROJECT_AUTO, "9")) return 0;
    if (!selftest_case("empty_total", empty_case, BASILISK_MODE_COUNT, BASILISK_PROJECT_AUTO, "8")) return 0;
    printf("selftest: OK\n");
    return 1;
}

int basilisk_entry(int argc, char **argv) {
    BasiliskOptions opt;
    KrbParallelRuntime parallel_rt;
    char err[512];
    int i;

    basilisk_options_init(&opt);

    g_basilisk_prog = (argc > 0 && argv[0] != NULL) ? argv[0] : "basilisk";

    for (i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "--selftest") == 0) {
            opt.selftest = 1;
        } else if (strcmp(a, "--parallel") == 0) {
            if (i + 1 >= argc || !krb_parallel_parse_mode(argv[++i], &opt.parallel.mode)) {
                fprintf(stderr, "ERROR: invalid --parallel\n");
                return 2;
            }
        } else if (strcmp(a, "--jobs") == 0) {
            long long jobs = 0;
            if (i + 1 >= argc || !parse_ll_token(argv[++i], &jobs) || jobs < 1 || jobs > INT_MAX) {
                fprintf(stderr, "ERROR: invalid --jobs\n");
                return 2;
            }
            opt.parallel.jobs = (int)jobs;
        } else if (strcmp(a, "--split-depth") == 0) {
            long long depth = 0;
            if (i + 1 >= argc || !parse_ll_token(argv[++i], &depth) || depth < 0 || depth > INT_MAX) {
                fprintf(stderr, "ERROR: invalid --split-depth\n");
                return 2;
            }
            opt.parallel.split_depth = (int)depth;
        } else if (strcmp(a, "--portfolio") == 0) {
            long long portfolio = 0;
            if (i + 1 >= argc || !parse_ll_token(argv[++i], &portfolio) || portfolio < 1 || portfolio > INT_MAX) {
                fprintf(stderr, "ERROR: invalid --portfolio\n");
                return 2;
            }
            opt.parallel.portfolio = (int)portfolio;
        } else if (strcmp(a, "--sync-ms") == 0) {
            long long sync_ms = 0;
            if (i + 1 >= argc || !parse_ll_token(argv[++i], &sync_ms) || sync_ms < 0 || sync_ms > INT_MAX) {
                fprintf(stderr, "ERROR: invalid --sync-ms\n");
                return 2;
            }
            opt.parallel.sync_ms = (int)sync_ms;
        } else if (strcmp(a, "--stats") == 0) {
            opt.stats = 1;
        } else if (strcmp(a, "--mode") == 0) {
            if (i + 1 >= argc || !parse_mode(argv[++i], &opt.mode)) {
                fprintf(stderr, "ERROR: invalid --mode\n");
                return 2;
            }
        } else if (strcmp(a, "--project") == 0) {
            if (i + 1 >= argc || !parse_project_mode(argv[++i], &opt.project_mode)) {
                fprintf(stderr, "ERROR: invalid --project\n");
                return 2;
            }
        } else if (strcmp(a, "--cuda") == 0) {
            if (i + 1 >= argc || !krb_accel_parse_mode(argv[++i], &opt.cuda_mode)) {
                fprintf(stderr, "ERROR: invalid --cuda\n");
                return 2;
            }
        } else if (strcmp(a, "--cuda-device") == 0) {
            if (i + 1 >= argc || !parse_cuda_device_opt(argv[++i], &opt.cuda_device)) {
                fprintf(stderr, "ERROR: invalid --cuda-device\n");
                return 2;
            }
        } else if (strcmp(a, "--cuda-min-lits") == 0) {
            if (i + 1 >= argc || !parse_nonnegative_size_opt(argv[++i], &opt.cuda_min_lits)) {
                fprintf(stderr, "ERROR: invalid --cuda-min-lits\n");
                return 2;
            }
        } else if (strcmp(a, "--hess") == 0) {
            opt.slime_init_hess = 1;
        } else if (strcmp(a, "--no-hess") == 0) {
            opt.slime_init_hess = 0;
        } else if (strcmp(a, "--ct") == 0) {
            opt.slime_init_ct = 1;
        } else if (strcmp(a, "--no-ct") == 0) {
            opt.slime_init_ct = 0;
        } else if (strcmp(a, "--ct-escape-rounds") == 0) {
            if (i + 1 >= argc || !parse_nonnegative_int_opt(argv[++i], &opt.slime_init_ct_escape_rounds)) {
                fprintf(stderr, "ERROR: invalid --ct-escape-rounds\n");
                return 2;
            }
        } else if (strcmp(a, "--ct-probe-restarts") == 0) {
            if (i + 1 >= argc || !parse_nonnegative_int_opt(argv[++i], &opt.slime_init_ct_probe_restarts)) {
                fprintf(stderr, "ERROR: invalid --ct-probe-restarts\n");
                return 2;
            }
        } else if (strcmp(a, "--slime-hess-nohit-limit") == 0) {
            if (i + 1 >= argc || !parse_positive_int_opt(argv[++i], &opt.slime_hess_nohit_limit)) {
                fprintf(stderr, "ERROR: invalid --slime-hess-nohit-limit\n");
                return 2;
            }
        } else if (strcmp(a, "--slime-ct-idle-soft-limit") == 0) {
            if (i + 1 >= argc || !parse_positive_int_opt(argv[++i], &opt.slime_ct_idle_soft_limit)) {
                fprintf(stderr, "ERROR: invalid --slime-ct-idle-soft-limit\n");
                return 2;
            }
        } else if (strcmp(a, "--slime-ct-idle-off-limit") == 0) {
            if (i + 1 >= argc || !parse_positive_int_opt(argv[++i], &opt.slime_ct_idle_off_limit)) {
                fprintf(stderr, "ERROR: invalid --slime-ct-idle-off-limit\n");
                return 2;
            }
        } else if (strcmp(a, "--slime-ct-probe-max-restarts") == 0) {
            if (i + 1 >= argc || !parse_positive_int_opt(argv[++i], &opt.slime_ct_probe_restarts_max)) {
                fprintf(stderr, "ERROR: invalid --slime-ct-probe-max-restarts\n");
                return 2;
            }
        } else if (strcmp(a, "--verbose") == 0) {
            long long v = 0;
            if (i + 1 >= argc || !parse_ll_token(argv[++i], &v) || v < 0 || v > 3) {
                fprintf(stderr, "ERROR: invalid --verbose\n");
                return 2;
            }
            opt.verbose = (int)v;
        } else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            print_usage(stdout);
            return 0;
        } else if (a[0] == '-') {
            fprintf(stderr, "ERROR: unknown option '%s'\n", a);
            return 2;
        } else {
            if (opt.input_path != NULL) {
                fprintf(stderr, "ERROR: multiple input files provided\n");
                return 2;
            }
            opt.input_path = a;
        }
    }

    if (opt.selftest) {
        return run_selftests() ? 0 : 1;
    }
    if (opt.input_path == NULL) {
        print_usage(stderr);
        return 2;
    }
    basilisk_options_normalize(&opt);
    if (!krb_parallel_runtime_resolve(&opt.parallel, &parallel_rt, err, sizeof(err))) {
        fprintf(stderr, "ERROR: %s\n", err);
        return 1;
    }
    if (parallel_rt.resolved_mode != KRB_PARALLEL_MODE_OFF && opt.verbose >= 1) {
        fprintf(stderr,
                "c basilisk parallel mode %s requested; backend remains serial in this build path\n",
                krb_parallel_mode_name(parallel_rt.resolved_mode));
    }
    if (opt.mode == BASILISK_MODE_COUNT && opt.project_mode != BASILISK_PROJECT_AUTO && opt.verbose >= 1) {
        fprintf(stderr, "c warning: --project is ignored in count mode\n");
    }
    if (!run_one_file(&opt, err, sizeof(err))) {
        fprintf(stderr, "ERROR: %s\n", err);
        return 1;
    }
    return 0;
}

#ifndef BASILISK_NO_MAIN
int main(int argc, char **argv) {
    return basilisk_entry(argc, argv);
}
#endif
