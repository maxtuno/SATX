/*
 * Copyright (c) 2026 Oscar Riveros.
 *
 * Licencia dual: uso personal bajo Apache License 2.0; portes a otros
 * lenguajes requieren licencia comercial con autorizacion expresa del autor.
 * Ver LICENSE.txt en la raiz del proyecto para los terminos completos.
 */

/*
Description:
KERBEROS is a unified dispatcher solver that preserves the strongest kernels from the
existing codebase:
  - `slime.c` for pure Boolean SAT;
  - `basilisk.c` for exact Boolean model counting on CNF;
  - `pixie.c` for LP/MIP over LP and MPS, and for WMIBO instances that are purely linear;
  - `wmibo.c` for weighted/hybrid Boolean-linear models and as a compatibility fallback.

Build (parte del producto satx, vía CMake): los kernels (SLIME/BASILISK/PIXIE/
WMIBO/GRINDER + aceleración) se compilan una sola vez en la biblioteca estática
`satx_kerberos` (con SLIME_NO_MAIN/BASILISK_NO_MAIN/PIXIE_NO_MAIN/WMIBO_NO_MAIN/
GRINDER_NO_MAIN) y este despachador solo aporta main(). Equivalente manual:
  gcc -O3 -std=c17 -Wall -Wextra -pedantic -DSLIME_NO_MAIN -DBASILISK_NO_MAIN -DPIXIE_NO_MAIN -DWMIBO_NO_MAIN -DGRINDER_NO_MAIN kerberos.c slime.c basilisk.c pixie.c wmibo.c grinder.c krb_accel.c krb_accel_cuda_stub.c krb_parallel_stub.c -lm -o kerberos -static
*/

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "krb_parallel.h"

int slime_entry(int argc, char **argv);
int basilisk_entry(int argc, char **argv);
int pixie_entry(int argc, char **argv);
int wmibo_entry(int argc, char **argv);
int grinder_entry(int argc, char **argv);

#ifndef SP_INF
#define SP_INF 1e30
#endif

typedef enum {
    SP_BACKEND_SLIME = 0,
    SP_BACKEND_BASILISK = 1,
    SP_BACKEND_PIXIE = 2,
    SP_BACKEND_WMIBO = 3
} SpBackend;

typedef enum {
    SP_WM_CLASS_UNKNOWN = 0,
    SP_WM_CLASS_PURE_SAT = 1,
    SP_WM_CLASS_PURE_LP = 2,
    SP_WM_CLASS_PURE_MIP = 3,
    SP_WM_CLASS_HYBRID = 4
} SpWmClass;

typedef enum {
    SP_BLK_NONE = 0,
    SP_BLK_CNF = 1,
    SP_BLK_WCNF = 2,
    SP_BLK_LIN = 3,
    SP_BLK_IND = 4,
    SP_BLK_OBJ = 5
} SpBlock;

typedef enum {
    SP_VAR_BOOL = 0,
    SP_VAR_INT = 1,
    SP_VAR_REAL = 2
} SpVarKind;

typedef struct {
    char **data;
    int size;
    int cap;
} StrVec;

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} StrBuf;

typedef struct {
    char **data;
    int size;
    int cap;
} ArgVec;

typedef struct {
    int B;
    int I;
    int R;
    bool have_header;
    bool has_clause;
    bool has_soft;
    bool has_lin;
    bool has_ind;
    bool has_obj;
    int hard_clause_count;
    int soft_clause_count;
    int clause_lits;
    int max_clause_len;
    int lin_count;
    int lin_terms;
    int max_lin_terms;
    int indicator_count;
    int obj_terms;
    int fixed_b;
    int fixed_i;
    int fixed_r;
    int binary_i;
    double max_abs_coef;
    double min_abs_coef;
    SpWmClass kind;
} WmiboSummary;

typedef struct {
    double lb;
    double ub;
    int is_binary;
} WmVarInfo;

typedef struct {
    const char *name;
    int takes_value;
} OptionSpec;

#define SP_BACKEND_MASK_SLIME (1u << SP_BACKEND_SLIME)
#define SP_BACKEND_MASK_BASILISK (1u << SP_BACKEND_BASILISK)
#define SP_BACKEND_MASK_PIXIE (1u << SP_BACKEND_PIXIE)
#define SP_BACKEND_MASK_WMIBO (1u << SP_BACKEND_WMIBO)

typedef enum {
    KOPT_CDCL = 0,
    KOPT_NO_MODEL = 1,
    KOPT_STATS = 2,
    KOPT_PROOF = 3,
    KOPT_HEURISTIC = 4,
    KOPT_MAB = 5,
    KOPT_NO_MAB = 6,
    KOPT_MABC = 7,
    KOPT_HESS = 8,
    KOPT_NO_HESS = 9,
    KOPT_CT = 10,
    KOPT_NO_CT = 11,
    KOPT_CT_LBD_MAX = 12,
    KOPT_CT_MAXLEN = 13,
    KOPT_CT_MAX_CUBES = 14,
    KOPT_CT_BUDDY = 15,
    KOPT_CT_NO_BUDDY = 16,
    KOPT_CT_ESCAPE_ROUNDS = 17,
    KOPT_CT_PROBE_RESTARTS = 18,
    KOPT_TIME = 19,
    KOPT_NODE = 20,
    KOPT_GAP = 21,
    KOPT_SEED = 22,
    KOPT_VERBOSE = 23,
    KOPT_PURELP = 24,
    KOPT_MODE = 25,
    KOPT_TRACE_OUT = 26,
    KOPT_CORE_OUT = 27,
    KOPT_CUDA = 28,
    KOPT_CUDA_DEVICE = 29,
    KOPT_CUDA_MIN_CELLS = 30,
    KOPT_CUDA_MIN_LITS = 31,
    KOPT_SLIME_HESS_NOHIT_LIMIT = 32,
    KOPT_SLIME_CT_IDLE_SOFT_LIMIT = 33,
    KOPT_SLIME_CT_IDLE_OFF_LIMIT = 34,
    KOPT_SLIME_CT_PROBE_MAX_RESTARTS = 35,
    KOPT_PARALLEL = 36,
    KOPT_JOBS = 37,
    KOPT_SPLIT_DEPTH = 38,
    KOPT_PORTFOLIO = 39,
    KOPT_SYNC_MS = 40,
    KOPT_SIMPLIFY = 41,
    KOPT_NO_SIMPLIFY = 42,
    KOPT_CHRONO = 43,
    KOPT_NO_CHRONO = 44,
    KOPT_INPROCESS = 45,
    KOPT_NO_INPROCESS = 46,
    KOPT_PROBE = 47,
    KOPT_BVE = 49,
    KOPT_NO_BVE = 50,
    KOPT_NO_PROBE = 48,
    KOPT_COUNT
} KrbOptionId;

typedef struct {
    const char *name;
    int takes_value;
    int id;
    unsigned support_mask;
    int is_meta;
    const char *description;
} CliOptionSpec;

typedef struct {
    const char *input_path;
    const char *manifest_out;
    const char *replay_path;
    bool show_help;
    bool selftest;
    bool audit_dispatch;
    bool strict_options;
    bool grinder;
    bool option_seen[KOPT_COUNT];
    bool has_mode;
    char mode_name[32];
    KrbParallelConfig parallel;
    StrVec grinder_args;
} KrbCli;

typedef struct {
    SpBackend backend;
    const char *backend_name;
    const OptionSpec *backend_opts;
    int nbackend_opts;
    const char *fmt_flag;
    const char *input_format;
    const char *reason;
    WmiboSummary summary;
    bool have_summary;
} DispatchPlan;

typedef struct {
    const char *input_path;
    const char *input_format;
    uint64_t input_hash;
    size_t input_size;
    DispatchPlan plan;
    int exit_code;
    double elapsed_sec;
    int ignored_count;
    StrVec ignored_options;
    StrVec solve_args;
    KrbParallelConfig parallel_cfg;
    KrbParallelRuntime parallel_rt;
} RunReport;

static void die_msg(const char *msg) {
    fprintf(stderr, "kerberos: %s\n", msg);
    exit(EXIT_FAILURE);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1U);
    if (p == NULL) {
        die_msg("out of memory");
    }
    return p;
}

static void *xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n ? n : 1U);
    if (p == NULL) {
        die_msg("out of memory");
    }
    return p;
}

static char *xstrdup_c(const char *s) {
    size_t n = strlen(s);
    char *d = (char *)xmalloc(n + 1U);
    memcpy(d, s, n + 1U);
    return d;
}

static void strvec_init(StrVec *v) {
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static void strvec_push_owned(StrVec *v, char *s) {
    int nc;
    if (v->size >= v->cap) {
        nc = (v->cap > 0) ? v->cap : 8;
        while (nc <= v->size) {
            nc *= 2;
        }
        v->data = (char **)xrealloc(v->data, (size_t)nc * sizeof(char *));
        v->cap = nc;
    }
    v->data[v->size++] = s;
}

static void strvec_push_copy(StrVec *v, const char *s) {
    strvec_push_owned(v, xstrdup_c(s));
}

static void strvec_free(StrVec *v) {
    int i;
    for (i = 0; i < v->size; ++i) {
        free(v->data[i]);
    }
    free(v->data);
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static void argvec_init(ArgVec *v) {
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static void argvec_push(ArgVec *v, char *s) {
    int nc;
    if (v->size >= v->cap) {
        nc = (v->cap > 0) ? v->cap : 8;
        while (nc <= v->size) {
            nc *= 2;
        }
        v->data = (char **)xrealloc(v->data, (size_t)nc * sizeof(char *));
        v->cap = nc;
    }
    v->data[v->size++] = s;
}

static void argvec_free(ArgVec *v) {
    free(v->data);
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static void strbuf_init(StrBuf *b) {
    b->data = NULL;
    b->len = 0U;
    b->cap = 0U;
}

static void strbuf_reserve(StrBuf *b, size_t need) {
    size_t nc;
    if (need <= b->cap) {
        return;
    }
    nc = (b->cap > 0U) ? b->cap : 64U;
    while (nc < need) {
        nc *= 2U;
    }
    b->data = (char *)xrealloc(b->data, nc);
    b->cap = nc;
}

static void strbuf_append(StrBuf *b, const char *s) {
    size_t n = strlen(s);
    strbuf_reserve(b, b->len + n + 1U);
    memcpy(b->data + b->len, s, n + 1U);
    b->len += n;
}

static void strbuf_appendf(StrBuf *b, const char *fmt, ...) {
    va_list ap;
    va_list aq;
    int need;
    va_start(ap, fmt);
    va_copy(aq, ap);
    need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need < 0) {
        va_end(aq);
        die_msg("formatting error");
    }
    strbuf_reserve(b, b->len + (size_t)need + 1U);
    vsnprintf(b->data + b->len, b->cap - b->len, fmt, aq);
    va_end(aq);
    b->len += (size_t)need;
}

static char *strbuf_detach(StrBuf *b) {
    char *s;
    if (b->data == NULL) {
        s = (char *)xmalloc(1U);
        s[0] = '\0';
        return s;
    }
    s = b->data;
    b->data = NULL;
    b->len = 0U;
    b->cap = 0U;
    return s;
}

static void strbuf_free(StrBuf *b) {
    free(b->data);
    b->data = NULL;
    b->len = 0U;
    b->cap = 0U;
}

static int str_ieq(const char *a, const char *b) {
    while (*a != '\0' && *b != '\0') {
        unsigned char ca = (unsigned char)tolower((unsigned char)*a);
        unsigned char cb = (unsigned char)tolower((unsigned char)*b);
        if (ca != cb) {
            return 0;
        }
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

static int has_ext_ci(const char *path, const char *ext) {
    size_t lp = strlen(path);
    size_t le = strlen(ext);
    if (lp < le) {
        return 0;
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
    while (n > 0U && isspace((unsigned char)s[n - 1U])) {
        s[n - 1U] = '\0';
        --n;
    }
}

static void trim_inplace(char *s) {
    char *t = trim_left(s);
    if (t != s) {
        memmove(s, t, strlen(t) + 1U);
    }
    trim_right(s);
}

static void strip_inline_comment_hash(char *s) {
    char *p = s;
    while (*p != '\0') {
        if (*p == '#') {
            *p = '\0';
            break;
        }
        ++p;
    }
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

static int parse_ll_token(const char *s, long long *out) {
    char *end = NULL;
    long long v;
    errno = 0;
    v = strtoll(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        return 0;
    }
    *out = v;
    return 1;
}

static int parse_double_token(const char *s, double *out) {
    char *end = NULL;
    double v;
    errno = 0;
    v = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0') {
        return 0;
    }
    *out = v;
    return 1;
}

static int parse_bounds_token_simple(const char *tok, double *lb, double *ub, int *is_binary) {
    const char *p = tok;
    char *end = NULL;
    double a;
    double b;
    if (is_binary != NULL) {
        *is_binary = 0;
    }
    if (strcmp(tok, "free") == 0) {
        *lb = -SP_INF;
        *ub = SP_INF;
        return 1;
    }
    if (strcmp(tok, "bin") == 0) {
        *lb = 0.0;
        *ub = 1.0;
        if (is_binary != NULL) {
            *is_binary = 1;
        }
        return 1;
    }
    if (*p != '[') {
        return 0;
    }
    ++p;
    errno = 0;
    a = strtod(p, &end);
    if (errno != 0 || end == p || *end != ',') {
        return 0;
    }
    p = end + 1;
    errno = 0;
    b = strtod(p, &end);
    if (errno != 0 || end == p || *end != ']') {
        return 0;
    }
    if (*(end + 1) != '\0') {
        return 0;
    }
    *lb = a;
    *ub = b;
    return 1;
}

static int parse_bool_lit_token(const char *tok, int max_b, int *out_lit) {
    int sign = 1;
    const char *p = tok;
    long long idx = 0;
    if (*p == '~') {
        sign = -1;
        ++p;
    }
    if (*p != 'b' && *p != 'B') {
        return 0;
    }
    ++p;
    if (!parse_ll_token(p, &idx)) {
        return 0;
    }
    if (idx < 1 || idx > max_b) {
        return 0;
    }
    *out_lit = sign * (int)idx;
    return 1;
}

static int parse_var_ref_token(const char *tok, SpVarKind *kind_out, int *idx_out) {
    char k;
    long long idx = 0;
    if (tok == NULL || tok[0] == '\0') {
        return 0;
    }
    k = (char)tolower((unsigned char)tok[0]);
    if (k != 'b' && k != 'i' && k != 'r') {
        return 0;
    }
    if (!parse_ll_token(tok + 1, &idx) || idx < 1) {
        return 0;
    }
    if (k == 'b') {
        *kind_out = SP_VAR_BOOL;
    } else if (k == 'i') {
        *kind_out = SP_VAR_INT;
    } else {
        *kind_out = SP_VAR_REAL;
    }
    *idx_out = (int)idx;
    return 1;
}

static void format_var_name(SpVarKind kind, int idx, char *buf, size_t bufsz) {
    char c = 'r';
    if (kind == SP_VAR_BOOL) {
        c = 'b';
    } else if (kind == SP_VAR_INT) {
        c = 'i';
    }
    snprintf(buf, bufsz, "%c%d", c, idx);
}

static int is_comment_or_blank(char *line) {
    trim_inplace(line);
    if (line[0] == '\0' || line[0] == '#') {
        return 1;
    }
    if ((line[0] == 'c' || line[0] == 'C') &&
        (line[1] == '\0' || isspace((unsigned char)line[1]))) {
        return 1;
    }
    return 0;
}

static int option_lookup(const OptionSpec *specs, int n, const char *arg, int *takes_value) {
    int i;
    for (i = 0; i < n; ++i) {
        if (strcmp(specs[i].name, arg) == 0) {
            *takes_value = specs[i].takes_value;
            return 1;
        }
    }
    return 0;
}

static const char *backend_name(SpBackend backend) {
    if (backend == SP_BACKEND_SLIME) return "slime";
    if (backend == SP_BACKEND_BASILISK) return "basilisk";
    if (backend == SP_BACKEND_PIXIE) return "pixie";
    return "wmibo";
}

static unsigned backend_mask(SpBackend backend) {
    return (backend == SP_BACKEND_SLIME) ? SP_BACKEND_MASK_SLIME :
           (backend == SP_BACKEND_BASILISK) ? SP_BACKEND_MASK_BASILISK :
           (backend == SP_BACKEND_PIXIE) ? SP_BACKEND_MASK_PIXIE :
                                           SP_BACKEND_MASK_WMIBO;
}

static const char *wm_class_name(SpWmClass kind) {
    if (kind == SP_WM_CLASS_PURE_SAT) return "pure-sat";
    if (kind == SP_WM_CLASS_PURE_LP) return "pure-lp";
    if (kind == SP_WM_CLASS_PURE_MIP) return "pure-mip";
    if (kind == SP_WM_CLASS_HYBRID) return "hybrid";
    return "unknown";
}

static const OptionSpec slime_opts[] = {
    {"--parallel", 1},
    {"--jobs", 1},
    {"--split-depth", 1},
    {"--portfolio", 1},
    {"--sync-ms", 1},
    {"--cdcl", 0},
    {"--no-model", 0},
    {"--stats", 0},
    {"--proof", 1},
    {"--heuristic", 1},
    {"--mab", 0},
    {"--no-mab", 0},
    {"--mabc", 1},
    {"--hess", 0},
    {"--no-hess", 0},
    {"--ct", 0},
    {"--no-ct", 0},
    {"--ct-lbd-max", 1},
    {"--ct-maxlen", 1},
    {"--ct-max-cubes", 1},
    {"--ct-buddy", 0},
    {"--ct-no-buddy", 0},
    {"--ct-escape-rounds", 1},
    {"--ct-probe-restarts", 1},
    {"--simplify", 0},
    {"--no-simplify", 0},
    {"--chrono", 0},
    {"--no-chrono", 0},
    {"--inprocess", 0},
    {"--no-inprocess", 0},
    {"--probe", 0},
    {"--bve", 0},
    {"--no-bve", 0},
    {"--no-probe", 0}
};

static const OptionSpec pixie_opts[] = {
    {"--parallel", 1},
    {"--jobs", 1},
    {"--split-depth", 1},
    {"--portfolio", 1},
    {"--sync-ms", 1},
    {"--time", 1},
    {"--node", 1},
    {"--gap", 1},
    {"--seed", 1},
    {"--verbose", 1},
    {"--cuda", 1},
    {"--cuda-device", 1},
    {"--cuda-min-cells", 1},
    {"--purelp", 0}
};

static const OptionSpec basilisk_opts[] = {
    {"--parallel", 1},
    {"--jobs", 1},
    {"--split-depth", 1},
    {"--portfolio", 1},
    {"--sync-ms", 1},
    {"--mode", 1},
    {"--cuda", 1},
    {"--cuda-device", 1},
    {"--cuda-min-lits", 1},
    {"--hess", 0},
    {"--no-hess", 0},
    {"--ct", 0},
    {"--no-ct", 0},
    {"--ct-escape-rounds", 1},
    {"--ct-probe-restarts", 1},
    {"--simplify", 0},
    {"--no-simplify", 0},
    {"--chrono", 0},
    {"--no-chrono", 0},
    {"--inprocess", 0},
    {"--no-inprocess", 0},
    {"--probe", 0},
    {"--bve", 0},
    {"--no-bve", 0},
    {"--no-probe", 0},
    {"--slime-hess-nohit-limit", 1},
    {"--slime-ct-idle-soft-limit", 1},
    {"--slime-ct-idle-off-limit", 1},
    {"--slime-ct-probe-max-restarts", 1},
    {"--stats", 0},
    {"--verbose", 1}
};

static const OptionSpec wmibo_opts[] = {
    {"--parallel", 1},
    {"--jobs", 1},
    {"--split-depth", 1},
    {"--portfolio", 1},
    {"--sync-ms", 1},
    {"--time", 1},
    {"--node", 1},
    {"--gap", 1},
    {"--seed", 1},
    {"--verbose", 1},
    {"--cuda", 1},
    {"--cuda-device", 1},
    {"--cuda-min-cells", 1},
    {"--mode", 1},
    {"--trace-out", 1},
    {"--core-out", 1}
};

static const CliOptionSpec cli_forward_opts[] = {
    {"--parallel", 1, KOPT_PARALLEL, SP_BACKEND_MASK_SLIME | SP_BACKEND_MASK_BASILISK | SP_BACKEND_MASK_PIXIE | SP_BACKEND_MASK_WMIBO, 0, "parallel runtime mode auto|off|threads|mpi|hybrid"},
    {"--jobs", 1, KOPT_JOBS, SP_BACKEND_MASK_SLIME | SP_BACKEND_MASK_BASILISK | SP_BACKEND_MASK_PIXIE | SP_BACKEND_MASK_WMIBO, 0, "number of local worker jobs"},
    {"--split-depth", 1, KOPT_SPLIT_DEPTH, SP_BACKEND_MASK_SLIME | SP_BACKEND_MASK_BASILISK | SP_BACKEND_MASK_PIXIE | SP_BACKEND_MASK_WMIBO, 0, "parallel root splitting depth"},
    {"--portfolio", 1, KOPT_PORTFOLIO, SP_BACKEND_MASK_SLIME | SP_BACKEND_MASK_BASILISK | SP_BACKEND_MASK_PIXIE | SP_BACKEND_MASK_WMIBO, 0, "portfolio worker multiplicity"},
    {"--sync-ms", 1, KOPT_SYNC_MS, SP_BACKEND_MASK_SLIME | SP_BACKEND_MASK_BASILISK | SP_BACKEND_MASK_PIXIE | SP_BACKEND_MASK_WMIBO, 0, "parallel synchronization cadence in milliseconds"},
    {"--cdcl", 0, KOPT_CDCL, SP_BACKEND_MASK_SLIME, 0, "compatibility flag for SLIME"},
    {"--no-model", 0, KOPT_NO_MODEL, SP_BACKEND_MASK_SLIME, 0, "suppress SAT model output"},
    {"--stats", 0, KOPT_STATS, SP_BACKEND_MASK_SLIME | SP_BACKEND_MASK_BASILISK, 0, "extended solver statistics"},
    {"--proof", 1, KOPT_PROOF, SP_BACKEND_MASK_SLIME, 0, "DRAT proof output"},
    {"--heuristic", 1, KOPT_HEURISTIC, SP_BACKEND_MASK_SLIME, 0, "SAT branching heuristic"},
    {"--mab", 0, KOPT_MAB, SP_BACKEND_MASK_SLIME, 0, "enable SLIME MAB switching"},
    {"--no-mab", 0, KOPT_NO_MAB, SP_BACKEND_MASK_SLIME, 0, "disable SLIME MAB switching"},
    {"--mabc", 1, KOPT_MABC, SP_BACKEND_MASK_SLIME, 0, "SLIME MAB confidence"},
    {"--hess", 0, KOPT_HESS, SP_BACKEND_MASK_SLIME | SP_BACKEND_MASK_BASILISK, 0, "enable HESS exact local search"},
    {"--no-hess", 0, KOPT_NO_HESS, SP_BACKEND_MASK_SLIME | SP_BACKEND_MASK_BASILISK, 0, "disable HESS exact local search"},
    {"--ct", 0, KOPT_CT, SP_BACKEND_MASK_SLIME | SP_BACKEND_MASK_BASILISK, 0, "enable covertrace"},
    {"--no-ct", 0, KOPT_NO_CT, SP_BACKEND_MASK_SLIME | SP_BACKEND_MASK_BASILISK, 0, "disable covertrace"},
    {"--ct-lbd-max", 1, KOPT_CT_LBD_MAX, SP_BACKEND_MASK_SLIME, 0, "covertrace LBD threshold"},
    {"--ct-maxlen", 1, KOPT_CT_MAXLEN, SP_BACKEND_MASK_SLIME, 0, "covertrace maximum cube length"},
    {"--ct-max-cubes", 1, KOPT_CT_MAX_CUBES, SP_BACKEND_MASK_SLIME, 0, "covertrace cube capacity"},
    {"--ct-buddy", 0, KOPT_CT_BUDDY, SP_BACKEND_MASK_SLIME, 0, "enable buddy merge"},
    {"--ct-no-buddy", 0, KOPT_CT_NO_BUDDY, SP_BACKEND_MASK_SLIME, 0, "disable buddy merge"},
    {"--ct-escape-rounds", 1, KOPT_CT_ESCAPE_ROUNDS, SP_BACKEND_MASK_SLIME | SP_BACKEND_MASK_BASILISK, 0, "covertrace escape rounds"},
    {"--ct-probe-restarts", 1, KOPT_CT_PROBE_RESTARTS, SP_BACKEND_MASK_SLIME | SP_BACKEND_MASK_BASILISK, 0, "covertrace probe restarts"},
    {"--simplify", 0, KOPT_SIMPLIFY, SP_BACKEND_MASK_SLIME, 0, "enable root simplification (BVE, subsumption, equivalent literals)"},
    {"--no-simplify", 0, KOPT_NO_SIMPLIFY, SP_BACKEND_MASK_SLIME, 0, "disable root simplification"},
    {"--chrono", 0, KOPT_CHRONO, SP_BACKEND_MASK_SLIME, 0, "enable chronological backtracking"},
    {"--no-chrono", 0, KOPT_NO_CHRONO, SP_BACKEND_MASK_SLIME, 0, "disable chronological backtracking"},
    {"--inprocess", 0, KOPT_INPROCESS, SP_BACKEND_MASK_SLIME, 0, "enable periodic inprocessing"},
    {"--no-inprocess", 0, KOPT_NO_INPROCESS, SP_BACKEND_MASK_SLIME, 0, "disable periodic inprocessing"},
    {"--probe", 0, KOPT_PROBE, SP_BACKEND_MASK_SLIME, 0, "enable failed literal probing"},
    {"--bve", 0, KOPT_BVE, SP_BACKEND_MASK_SLIME, 0, "enable bounded variable elimination (experimental)"},
    {"--no-bve", 0, KOPT_NO_BVE, SP_BACKEND_MASK_SLIME, 0, "disable bounded variable elimination"},
    {"--no-probe", 0, KOPT_NO_PROBE, SP_BACKEND_MASK_SLIME, 0, "disable failed literal probing"},
    {"--time", 1, KOPT_TIME, SP_BACKEND_MASK_PIXIE | SP_BACKEND_MASK_WMIBO, 0, "time limit in seconds"},
    {"--node", 1, KOPT_NODE, SP_BACKEND_MASK_PIXIE | SP_BACKEND_MASK_WMIBO, 0, "branch-and-bound node limit"},
    {"--gap", 1, KOPT_GAP, SP_BACKEND_MASK_PIXIE | SP_BACKEND_MASK_WMIBO, 0, "relative optimality gap"},
    {"--seed", 1, KOPT_SEED, SP_BACKEND_MASK_PIXIE | SP_BACKEND_MASK_WMIBO, 0, "random seed"},
    {"--verbose", 1, KOPT_VERBOSE, SP_BACKEND_MASK_BASILISK | SP_BACKEND_MASK_PIXIE | SP_BACKEND_MASK_WMIBO, 0, "verbosity level"},
    {"--cuda", 1, KOPT_CUDA, SP_BACKEND_MASK_BASILISK | SP_BACKEND_MASK_PIXIE | SP_BACKEND_MASK_WMIBO, 0, "CUDA acceleration mode auto|on|off"},
    {"--cuda-device", 1, KOPT_CUDA_DEVICE, SP_BACKEND_MASK_BASILISK | SP_BACKEND_MASK_PIXIE | SP_BACKEND_MASK_WMIBO, 0, "CUDA device id (-1 keeps runtime default)"},
    {"--cuda-min-cells", 1, KOPT_CUDA_MIN_CELLS, SP_BACKEND_MASK_PIXIE | SP_BACKEND_MASK_WMIBO, 0, "minimum dense LP cells before trying CUDA"},
    {"--cuda-min-lits", 1, KOPT_CUDA_MIN_LITS, SP_BACKEND_MASK_BASILISK, 0, "minimum CNF literal volume before trying CUDA"},
    {"--purelp", 0, KOPT_PURELP, SP_BACKEND_MASK_PIXIE, 0, "solve LP relaxation only"},
    {"--mode", 1, KOPT_MODE, SP_BACKEND_MASK_BASILISK | SP_BACKEND_MASK_WMIBO, 0, "query mode"},
    {"--trace-out", 1, KOPT_TRACE_OUT, SP_BACKEND_MASK_WMIBO, 0, "WMIBO JSONL trace output"},
    {"--core-out", 1, KOPT_CORE_OUT, SP_BACKEND_MASK_WMIBO, 0, "WMIBO core JSON scaffold"},
    {"--slime-hess-nohit-limit", 1, KOPT_SLIME_HESS_NOHIT_LIMIT, SP_BACKEND_MASK_BASILISK, 0, "disable residual HESS after N no-hit calls"},
    {"--slime-ct-idle-soft-limit", 1, KOPT_SLIME_CT_IDLE_SOFT_LIMIT, SP_BACKEND_MASK_BASILISK, 0, "damp residual CoverTrace after N idle calls"},
    {"--slime-ct-idle-off-limit", 1, KOPT_SLIME_CT_IDLE_OFF_LIMIT, SP_BACKEND_MASK_BASILISK, 0, "disable residual CoverTrace after N idle calls"},
    {"--slime-ct-probe-max-restarts", 1, KOPT_SLIME_CT_PROBE_MAX_RESTARTS, SP_BACKEND_MASK_BASILISK, 0, "ceiling for retuned residual ct_probe_restarts"}
};

static const CliOptionSpec *lookup_cli_forward_option(const char *arg) {
    size_t i;
    for (i = 0; i < sizeof(cli_forward_opts) / sizeof(cli_forward_opts[0]); ++i) {
        if (strcmp(cli_forward_opts[i].name, arg) == 0) {
            return &cli_forward_opts[i];
        }
    }
    return NULL;
}

static int parse_cli_args(KrbCli *cli, StrVec *solve_args, int argc, char **argv, char *err, size_t errsz) {
    int i;
    memset(cli, 0, sizeof(*cli));
    krb_parallel_config_defaults(&cli->parallel);
    strvec_init(solve_args);
    strvec_init(&cli->grinder_args);

    for (i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            cli->show_help = true;
            continue;
        }
        if (strcmp(a, "--selftest") == 0) {
            cli->selftest = true;
            continue;
        }
        if (strcmp(a, "--audit-dispatch") == 0) {
            cli->audit_dispatch = true;
            continue;
        }
        if (strcmp(a, "--strict-options") == 0) {
            cli->strict_options = true;
            continue;
        }
        if (strcmp(a, "--grinder") == 0) {
            int j;
            cli->grinder = true;
            for (j = i + 1; j < argc; j++) {
                strvec_push_copy(&cli->grinder_args, argv[j]);
            }
            break;
        }
        if (strcmp(a, "--manifest-out") == 0 || strcmp(a, "--replay") == 0) {
            const char **slot = (strcmp(a, "--manifest-out") == 0) ? &cli->manifest_out : &cli->replay_path;
            if (i + 1 >= argc) {
                snprintf(err, errsz, "missing value for option %s", a);
                strvec_free(solve_args);
                return 0;
            }
            *slot = argv[++i];
            continue;
        }
        if (a[0] == '-') {
            const CliOptionSpec *spec = lookup_cli_forward_option(a);
            if (spec == NULL) {
                snprintf(err, errsz, "unknown option %s", a);
                strvec_free(solve_args);
                return 0;
            }
            cli->option_seen[spec->id] = true;
            strvec_push_copy(solve_args, a);
            if (spec->takes_value) {
                if (i + 1 >= argc) {
                    snprintf(err, errsz, "missing value for option %s", a);
                    strvec_free(solve_args);
                    return 0;
                }
                strvec_push_copy(solve_args, argv[++i]);
                if (spec->id == KOPT_MODE) {
                    cli->has_mode = true;
                    snprintf(cli->mode_name, sizeof(cli->mode_name), "%s", argv[i]);
                } else if (spec->id == KOPT_PARALLEL) {
                    if (!krb_parallel_parse_mode(argv[i], &cli->parallel.mode)) {
                        snprintf(err, errsz, "invalid value '%s' for --parallel", argv[i]);
                        strvec_free(solve_args);
                        return 0;
                    }
                } else if (spec->id == KOPT_JOBS) {
                    long long v = 0;
                    if (!parse_ll_token(argv[i], &v) || v < 1 || v > 2147483647LL) {
                        snprintf(err, errsz, "invalid value '%s' for --jobs", argv[i]);
                        strvec_free(solve_args);
                        return 0;
                    }
                    cli->parallel.jobs = (int)v;
                } else if (spec->id == KOPT_SPLIT_DEPTH) {
                    long long v = 0;
                    if (!parse_ll_token(argv[i], &v) || v < 0 || v > 2147483647LL) {
                        snprintf(err, errsz, "invalid value '%s' for --split-depth", argv[i]);
                        strvec_free(solve_args);
                        return 0;
                    }
                    cli->parallel.split_depth = (int)v;
                } else if (spec->id == KOPT_PORTFOLIO) {
                    long long v = 0;
                    if (!parse_ll_token(argv[i], &v) || v < 1 || v > 2147483647LL) {
                        snprintf(err, errsz, "invalid value '%s' for --portfolio", argv[i]);
                        strvec_free(solve_args);
                        return 0;
                    }
                    cli->parallel.portfolio = (int)v;
                } else if (spec->id == KOPT_SYNC_MS) {
                    long long v = 0;
                    if (!parse_ll_token(argv[i], &v) || v < 0 || v > 2147483647LL) {
                        snprintf(err, errsz, "invalid value '%s' for --sync-ms", argv[i]);
                        strvec_free(solve_args);
                        return 0;
                    }
                    cli->parallel.sync_ms = (int)v;
                }
            }
            continue;
        }
        if (cli->input_path != NULL) {
            snprintf(err, errsz, "multiple input files provided");
            strvec_free(solve_args);
            return 0;
        }
        cli->input_path = argv[i];
        strvec_push_copy(solve_args, a);
    }

    return 1;
}

static int build_backend_argv(ArgVec *out,
                              const char *prog_name,
                              const OptionSpec *specs,
                              int nspecs,
                              const StrVec *solve_args,
                              const char *input_path,
                              const char *fmt_flag,
                              const char *backend_path) {
    int i;
    int skipped_input = 0;
    argvec_init(out);
    argvec_push(out, (char *)prog_name);
    for (i = 0; i < solve_args->size; ++i) {
        int takes_value = 0;
        const char *a = solve_args->data[i];
        if (!skipped_input && input_path != NULL && a[0] != '-' && strcmp(a, input_path) == 0) {
            skipped_input = 1;
            continue;
        }
        if (a[0] == '-' && option_lookup(specs, nspecs, a, &takes_value)) {
            argvec_push(out, solve_args->data[i]);
            if (takes_value) {
                if (i + 1 >= solve_args->size) {
                    fprintf(stderr, "kerberos: missing value for option %s\n", a);
                    argvec_free(out);
                    return 0;
                }
                argvec_push(out, solve_args->data[++i]);
            }
        }
    }
    if (fmt_flag != NULL) {
        argvec_push(out, (char *)fmt_flag);
    }
    argvec_push(out, (char *)backend_path);
    return 1;
}

static int make_temp_path(const char *ext, char *out, size_t outsz) {
    static unsigned long seq = 0;
    unsigned long cur = ++seq;
    unsigned long stamp = (unsigned long)time(NULL);
    if (snprintf(out, outsz, "kerberos_tmp_%lu_%lu%s", stamp, cur, ext) >= (int)outsz) {
        return 0;
    }
    return 1;
}

static int run_backend(SpBackend backend, ArgVec *argv_out) {
    if (backend == SP_BACKEND_SLIME) {
        return slime_entry(argv_out->size, argv_out->data);
    }
    if (backend == SP_BACKEND_BASILISK) {
        return basilisk_entry(argv_out->size, argv_out->data);
    }
    if (backend == SP_BACKEND_PIXIE) {
        return pixie_entry(argv_out->size, argv_out->data);
    }
    return wmibo_entry(argv_out->size, argv_out->data);
}

static void summary_note_coef(WmiboSummary *out, double coef) {
    double a = fabs(coef);
    if (a <= 0.0) {
        return;
    }
    if (a > out->max_abs_coef) {
        out->max_abs_coef = a;
    }
    if (out->min_abs_coef <= 0.0 || a < out->min_abs_coef) {
        out->min_abs_coef = a;
    }
}

static int wmibo_scan_summary(const char *path, WmiboSummary *out, char *err, size_t errsz) {
    FILE *fp;
    char line_buf[8192];
    int lineno = 0;
    SpBlock blk = SP_BLK_NONE;
    memset(out, 0, sizeof(*out));

    fp = fopen(path, "rb");
    if (fp == NULL) {
        snprintf(err, errsz, "cannot open '%s': %s", path, strerror(errno));
        return 0;
    }

    while (fgets(line_buf, (int)sizeof(line_buf), fp) != NULL) {
        char line[8192];
        char *tok[32];
        int nt;
        ++lineno;
        snprintf(line, sizeof(line), "%s", line_buf);
        if (is_comment_or_blank(line)) {
            continue;
        }
        strip_inline_comment_hash(line);
        trim_inplace(line);
        if (line[0] == '\0') {
            continue;
        }
        nt = token_split(line, tok, 32);
        if (nt <= 0) {
            continue;
        }
        if (str_ieq(tok[0], "p")) {
            long long B = 0, I = 0, R = 0;
            if (nt < 5 || !str_ieq(tok[1], "wmibo")) {
                snprintf(err, errsz, "%s:%d invalid wmibo header", path, lineno);
                fclose(fp);
                return 0;
            }
            if (!parse_ll_token(tok[2], &B) || !parse_ll_token(tok[3], &I) || !parse_ll_token(tok[4], &R)) {
                snprintf(err, errsz, "%s:%d invalid wmibo dimensions", path, lineno);
                fclose(fp);
                return 0;
            }
            out->B = (int)B;
            out->I = (int)I;
            out->R = (int)R;
            out->have_header = true;
            continue;
        }
        if (!out->have_header) {
            snprintf(err, errsz, "%s:%d data before wmibo header", path, lineno);
            fclose(fp);
            return 0;
        }
        if (str_ieq(tok[0], "var") && nt >= 4) {
            char kind = (char)tolower((unsigned char)tok[1][0]);
            double lb = 0.0;
            double ub = 0.0;
            int is_binary = 0;
            if (parse_bounds_token_simple(tok[3], &lb, &ub, &is_binary)) {
                if (kind == 'b' && fabs(lb - ub) <= 1e-12) ++out->fixed_b;
                if (kind == 'i') {
                    if (fabs(lb - ub) <= 1e-12) ++out->fixed_i;
                    if (is_binary) ++out->binary_i;
                }
                if (kind == 'r' && fabs(lb - ub) <= 1e-12) ++out->fixed_r;
            }
            continue;
        }
        if (str_ieq(tok[0], "begin")) {
            if (nt == 2) {
                if (str_ieq(tok[1], "cnf")) blk = SP_BLK_CNF;
                else if (str_ieq(tok[1], "wcnf")) blk = SP_BLK_WCNF;
                else if (str_ieq(tok[1], "lin")) blk = SP_BLK_LIN;
                else if (str_ieq(tok[1], "ind")) blk = SP_BLK_IND;
                else if (str_ieq(tok[1], "obj")) blk = SP_BLK_OBJ;
                else blk = SP_BLK_NONE;
            }
            continue;
        }
        if (str_ieq(tok[0], "end")) {
            blk = SP_BLK_NONE;
            continue;
        }
        if (str_ieq(tok[0], "cl") || blk == SP_BLK_CNF) {
            int p = 1;
            int lits = 0;
            out->has_clause = true;
            if (nt >= 2 && (str_ieq(tok[1], "hard") || str_ieq(tok[1], "soft"))) {
                p = 2;
            }
            if (nt >= 2 && str_ieq(tok[1], "soft")) {
                out->has_soft = true;
                ++out->soft_clause_count;
            } else {
                ++out->hard_clause_count;
            }
            while (p < nt && strcmp(tok[p], "0") != 0) {
                ++lits;
                ++p;
            }
            out->clause_lits += lits;
            if (lits > out->max_clause_len) {
                out->max_clause_len = lits;
            }
            continue;
        }
        if (str_ieq(tok[0], "wcl") || blk == SP_BLK_WCNF) {
            int p = 1;
            int lits = 0;
            out->has_clause = true;
            out->has_soft = true;
            ++out->soft_clause_count;
            if (str_ieq(tok[0], "wcl")) {
                double w = 0.0;
                if (nt >= 2 && parse_double_token(tok[1], &w)) {
                    summary_note_coef(out, w);
                }
                p = 3;
            }
            while (p < nt && strcmp(tok[p], "0") != 0) {
                ++lits;
                ++p;
            }
            out->clause_lits += lits;
            if (lits > out->max_clause_len) {
                out->max_clause_len = lits;
            }
            continue;
        }
        if (str_ieq(tok[0], "lc") || blk == SP_BLK_LIN) {
            int j;
            int colon = -1;
            out->has_lin = true;
            ++out->lin_count;
            for (j = 0; j < nt; ++j) {
                if (strcmp(tok[j], ":") == 0) {
                    colon = j;
                    break;
                }
            }
            if (colon >= 0 && colon + 1 < nt) {
                int terms = 0;
                for (j = colon + 1; j + 1 < nt; j += 2) {
                    double coef = 0.0;
                    if (!parse_double_token(tok[j], &coef)) {
                        break;
                    }
                    summary_note_coef(out, coef);
                    ++terms;
                }
                out->lin_terms += terms;
                if (terms > out->max_lin_terms) {
                    out->max_lin_terms = terms;
                }
            }
            continue;
        }
        if (str_ieq(tok[0], "ind") || blk == SP_BLK_IND) {
            out->has_ind = true;
            ++out->indicator_count;
            continue;
        }
        if (str_ieq(tok[0], "obj") || blk == SP_BLK_OBJ) {
            int j = 0;
            out->has_obj = true;
            for (j = 0; j < nt; ++j) {
                if (strcmp(tok[j], ":") == 0) {
                    break;
                }
            }
            if (j < nt) {
                ++j;
            }
            while (j < nt) {
                double coef = 0.0;
                if (str_ieq(tok[j], "lin")) {
                    ++j;
                    continue;
                }
                if (str_ieq(tok[j], "pen")) {
                    break;
                }
                if (j + 1 >= nt || !parse_double_token(tok[j], &coef)) {
                    break;
                }
                summary_note_coef(out, coef);
                ++out->obj_terms;
                j += 2;
            }
            continue;
        }
    }

    fclose(fp);
    if (!out->have_header) {
        snprintf(err, errsz, "wmibo header not found in '%s'", path);
        return 0;
    }

    if (out->I == 0 && out->R == 0 && !out->has_soft && !out->has_lin && !out->has_ind && !out->has_obj) {
        out->kind = SP_WM_CLASS_PURE_SAT;
    } else if (!out->has_clause && !out->has_soft && !out->has_ind) {
        if (out->B == 0 && out->I == 0) {
            out->kind = SP_WM_CLASS_PURE_LP;
        } else {
            out->kind = SP_WM_CLASS_PURE_MIP;
        }
    } else {
        out->kind = SP_WM_CLASS_HYBRID;
    }
    return 1;
}

static int wmibo_translate_pure_sat_to_cnf(const char *input_path,
                                           const char *output_path,
                                           const WmiboSummary *summary,
                                           char *err,
                                           size_t errsz) {
    FILE *fp = NULL;
    FILE *out = NULL;
    char line_buf[8192];
    WmVarInfo *bvars = NULL;
    StrVec clauses;
    SpBlock blk = SP_BLK_NONE;
    int i;

    strvec_init(&clauses);
    bvars = (WmVarInfo *)xmalloc((size_t)(summary->B + 1) * sizeof(WmVarInfo));
    for (i = 0; i <= summary->B; ++i) {
        bvars[i].lb = 0.0;
        bvars[i].ub = 1.0;
        bvars[i].is_binary = 1;
    }

    fp = fopen(input_path, "rb");
    if (fp == NULL) {
        snprintf(err, errsz, "cannot open '%s': %s", input_path, strerror(errno));
        goto fail;
    }

    while (fgets(line_buf, (int)sizeof(line_buf), fp) != NULL) {
        char line[8192];
        char *tok[4096];
        int nt;
        snprintf(line, sizeof(line), "%s", line_buf);
        if (is_comment_or_blank(line)) {
            continue;
        }
        strip_inline_comment_hash(line);
        trim_inplace(line);
        if (line[0] == '\0') {
            continue;
        }
        nt = token_split(line, tok, 4096);
        if (nt <= 0) {
            continue;
        }
        if (str_ieq(tok[0], "begin")) {
            if (nt == 2) {
                if (str_ieq(tok[1], "cnf")) blk = SP_BLK_CNF;
                else if (str_ieq(tok[1], "wcnf")) blk = SP_BLK_WCNF;
                else if (str_ieq(tok[1], "lin")) blk = SP_BLK_LIN;
                else if (str_ieq(tok[1], "ind")) blk = SP_BLK_IND;
                else if (str_ieq(tok[1], "obj")) blk = SP_BLK_OBJ;
                else blk = SP_BLK_NONE;
            }
            continue;
        }
        if (str_ieq(tok[0], "end")) {
            blk = SP_BLK_NONE;
            continue;
        }
        if (str_ieq(tok[0], "p") || str_ieq(tok[0], "opt")) {
            continue;
        }
        if (str_ieq(tok[0], "var")) {
            long long idxll = 0;
            double lb = 0.0, ub = 1.0;
            int is_bin = 0;
            int idx;
            if (nt < 4 || strlen(tok[1]) != 1U || tolower((unsigned char)tok[1][0]) != 'b') {
                continue;
            }
            if (!parse_ll_token(tok[2], &idxll) || idxll < 1 || idxll > summary->B) {
                snprintf(err, errsz, "invalid bool var declaration in '%s'", input_path);
                goto fail;
            }
            if (!parse_bounds_token_simple(tok[3], &lb, &ub, &is_bin)) {
                snprintf(err, errsz, "invalid bool bounds '%s' in '%s'", tok[3], input_path);
                goto fail;
            }
            idx = (int)idxll;
            if (lb < 0.0) lb = 0.0;
            if (ub > 1.0) ub = 1.0;
            bvars[idx].lb = lb;
            bvars[idx].ub = ub;
            continue;
        }
        if (str_ieq(tok[0], "cl") || blk == SP_BLK_CNF) {
            int p = 0;
            int j;
            StrBuf sb;
            if (nt < 2 || !str_ieq(tok[0], "cl")) {
                snprintf(err, errsz, "invalid clause syntax in '%s'", input_path);
                goto fail;
            }
            if (nt < 3 || !str_ieq(tok[1], "hard")) {
                snprintf(err, errsz, "non-hard clause reached pure SAT translator in '%s'", input_path);
                goto fail;
            }
            p = 2;
            strbuf_init(&sb);
            for (j = p; j < nt; ++j) {
                int lit = 0;
                if (strcmp(tok[j], "0") == 0) {
                    break;
                }
                if (!parse_bool_lit_token(tok[j], summary->B, &lit)) {
                    strbuf_free(&sb);
                    snprintf(err, errsz, "invalid boolean literal '%s' in '%s'", tok[j], input_path);
                    goto fail;
                }
                strbuf_appendf(&sb, "%d ", lit);
            }
            if (j == nt || strcmp(tok[j], "0") != 0) {
                strbuf_free(&sb);
                snprintf(err, errsz, "hard clause missing trailing 0 in '%s'", input_path);
                goto fail;
            }
            strbuf_append(&sb, "0\n");
            strvec_push_owned(&clauses, strbuf_detach(&sb));
            continue;
        }
    }

    for (i = 1; i <= summary->B; ++i) {
        if (bvars[i].lb > 0.5 && bvars[i].ub < 0.5) {
            strvec_push_copy(&clauses, "0\n");
        } else if (bvars[i].lb > 0.5) {
            StrBuf sb;
            strbuf_init(&sb);
            strbuf_appendf(&sb, "%d 0\n", i);
            strvec_push_owned(&clauses, strbuf_detach(&sb));
        } else if (bvars[i].ub < 0.5) {
            StrBuf sb;
            strbuf_init(&sb);
            strbuf_appendf(&sb, "-%d 0\n", i);
            strvec_push_owned(&clauses, strbuf_detach(&sb));
        }
    }

    out = fopen(output_path, "wb");
    if (out == NULL) {
        snprintf(err, errsz, "cannot create '%s': %s", output_path, strerror(errno));
        goto fail;
    }
    fprintf(out, "p cnf %d %d\n", summary->B, clauses.size);
    for (i = 0; i < clauses.size; ++i) {
        fputs(clauses.data[i], out);
    }

    fclose(out);
    fclose(fp);
    strvec_free(&clauses);
    free(bvars);
    return 1;

fail:
    if (out != NULL) fclose(out);
    if (fp != NULL) fclose(fp);
    strvec_free(&clauses);
    free(bvars);
    return 0;
}

static int wmibo_expr_to_lp_terms(char *expr, char **out_expr, char *err, size_t errsz) {
    char *tok[2048];
    int nt;
    int i;
    int first = 1;
    StrBuf sb;
    strbuf_init(&sb);
    trim_inplace(expr);
    nt = token_split(expr, tok, 2048);
    for (i = 0; i < nt; ) {
        double coef = 0.0;
        SpVarKind kind;
        int idx = 0;
        char name[64];

        if (str_ieq(tok[i], "lin")) {
            ++i;
            continue;
        }
        if (str_ieq(tok[i], "pen")) {
            break;
        }
        if (i + 1 >= nt) {
            strbuf_free(&sb);
            snprintf(err, errsz, "incomplete coefficient/variable pair");
            return 0;
        }
        if (!parse_double_token(tok[i], &coef)) {
            strbuf_free(&sb);
            snprintf(err, errsz, "invalid coefficient '%s'", tok[i]);
            return 0;
        }
        if (!parse_var_ref_token(tok[i + 1], &kind, &idx)) {
            strbuf_free(&sb);
            snprintf(err, errsz, "invalid variable reference '%s'", tok[i + 1]);
            return 0;
        }
        format_var_name(kind, idx, name, sizeof(name));
        if (first) {
            if (coef < 0.0) {
                strbuf_append(&sb, "- ");
                coef = -coef;
            }
            strbuf_appendf(&sb, "%.17g %s", coef, name);
            first = 0;
        } else {
            if (coef < 0.0) {
                strbuf_appendf(&sb, " - %.17g %s", -coef, name);
            } else {
                strbuf_appendf(&sb, " + %.17g %s", coef, name);
            }
        }
        i += 2;
    }
    if (first) {
        strbuf_append(&sb, "0");
    }
    *out_expr = strbuf_detach(&sb);
    return 1;
}

static int wmibo_translate_pure_linear_to_lp(const char *input_path,
                                             const char *output_path,
                                             const WmiboSummary *summary,
                                             char *err,
                                             size_t errsz) {
    FILE *fp = NULL;
    FILE *out = NULL;
    char line_buf[8192];
    WmVarInfo *bvars = NULL;
    WmVarInfo *ivars = NULL;
    WmVarInfo *rvars = NULL;
    StrVec constraints;
    StrVec binary_ints;
    char *objective = NULL;
    int obj_is_max = 0;
    SpBlock blk = SP_BLK_NONE;
    int i;

    strvec_init(&constraints);
    strvec_init(&binary_ints);
    bvars = (WmVarInfo *)xmalloc((size_t)(summary->B + 1) * sizeof(WmVarInfo));
    ivars = (WmVarInfo *)xmalloc((size_t)(summary->I + 1) * sizeof(WmVarInfo));
    rvars = (WmVarInfo *)xmalloc((size_t)(summary->R + 1) * sizeof(WmVarInfo));

    for (i = 0; i <= summary->B; ++i) {
        bvars[i].lb = 0.0;
        bvars[i].ub = 1.0;
        bvars[i].is_binary = 1;
    }
    for (i = 0; i <= summary->I; ++i) {
        ivars[i].lb = -SP_INF;
        ivars[i].ub = SP_INF;
        ivars[i].is_binary = 0;
    }
    for (i = 0; i <= summary->R; ++i) {
        rvars[i].lb = -SP_INF;
        rvars[i].ub = SP_INF;
        rvars[i].is_binary = 0;
    }

    fp = fopen(input_path, "rb");
    if (fp == NULL) {
        snprintf(err, errsz, "cannot open '%s': %s", input_path, strerror(errno));
        goto fail;
    }

    while (fgets(line_buf, (int)sizeof(line_buf), fp) != NULL) {
        char line[8192];
        char raw[8192];
        char *tok[256];
        int nt;

        snprintf(line, sizeof(line), "%s", line_buf);
        if (is_comment_or_blank(line)) {
            continue;
        }
        strip_inline_comment_hash(line);
        trim_inplace(line);
        if (line[0] == '\0') {
            continue;
        }
        snprintf(raw, sizeof(raw), "%s", line);
        nt = token_split(line, tok, 256);
        if (nt <= 0) {
            continue;
        }

        if (str_ieq(tok[0], "begin")) {
            if (nt == 2) {
                if (str_ieq(tok[1], "cnf")) blk = SP_BLK_CNF;
                else if (str_ieq(tok[1], "wcnf")) blk = SP_BLK_WCNF;
                else if (str_ieq(tok[1], "lin")) blk = SP_BLK_LIN;
                else if (str_ieq(tok[1], "ind")) blk = SP_BLK_IND;
                else if (str_ieq(tok[1], "obj")) blk = SP_BLK_OBJ;
                else blk = SP_BLK_NONE;
            }
            continue;
        }
        if (str_ieq(tok[0], "end")) {
            blk = SP_BLK_NONE;
            continue;
        }
        if (str_ieq(tok[0], "p") || str_ieq(tok[0], "opt")) {
            continue;
        }
        if (str_ieq(tok[0], "var")) {
            long long idxll = 0;
            double lb = 0.0, ub = 0.0;
            int is_bin = 0;
            int idx;
            char kind;
            if (nt < 4 || strlen(tok[1]) != 1U) {
                snprintf(err, errsz, "invalid var declaration in '%s'", input_path);
                goto fail;
            }
            kind = (char)tolower((unsigned char)tok[1][0]);
            if (!parse_ll_token(tok[2], &idxll)) {
                snprintf(err, errsz, "invalid var index '%s'", tok[2]);
                goto fail;
            }
            idx = (int)idxll;
            if (!parse_bounds_token_simple(tok[3], &lb, &ub, &is_bin)) {
                snprintf(err, errsz, "invalid var bounds '%s'", tok[3]);
                goto fail;
            }
            if (kind == 'b') {
                if (idx < 1 || idx > summary->B) {
                    snprintf(err, errsz, "bool var out of range");
                    goto fail;
                }
                if (lb < 0.0) lb = 0.0;
                if (ub > 1.0) ub = 1.0;
                bvars[idx].lb = lb;
                bvars[idx].ub = ub;
                bvars[idx].is_binary = 1;
            } else if (kind == 'i') {
                if (idx < 1 || idx > summary->I) {
                    snprintf(err, errsz, "int var out of range");
                    goto fail;
                }
                ivars[idx].lb = lb;
                ivars[idx].ub = ub;
                ivars[idx].is_binary = is_bin ? 1 : 0;
            } else if (kind == 'r') {
                if (idx < 1 || idx > summary->R) {
                    snprintf(err, errsz, "real var out of range");
                    goto fail;
                }
                rvars[idx].lb = lb;
                rvars[idx].ub = ub;
            } else {
                snprintf(err, errsz, "invalid variable kind '%c'", kind);
                goto fail;
            }
            continue;
        }
        if (str_ieq(tok[0], "lc") || blk == SP_BLK_LIN) {
            char *colon = strchr(raw, ':');
            char *left = raw;
            char *right = NULL;
            char *ltok[64];
            int lnt;
            char *expr_text = NULL;
            StrBuf sb;
            if (colon == NULL) {
                snprintf(err, errsz, "linear constraint missing ':' in '%s'", input_path);
                goto fail;
            }
            *colon = '\0';
            right = colon + 1;
            trim_inplace(left);
            trim_inplace(right);
            lnt = token_split(left, ltok, 64);
            if (lnt < 4 || !str_ieq(ltok[0], "lc")) {
                snprintf(err, errsz, "invalid linear constraint header in '%s'", input_path);
                goto fail;
            }
            if (!wmibo_expr_to_lp_terms(right, &expr_text, err, errsz)) {
                goto fail;
            }
            strbuf_init(&sb);
            strbuf_appendf(&sb, "%s: %s %s %s", ltok[1], expr_text, ltok[2], ltok[3]);
            strvec_push_owned(&constraints, strbuf_detach(&sb));
            free(expr_text);
            continue;
        }
        if (str_ieq(tok[0], "obj") || blk == SP_BLK_OBJ) {
            char *colon = strchr(raw, ':');
            char *head = raw;
            char *expr = NULL;
            char *htok[8];
            int hnt;
            if (colon == NULL) {
                snprintf(err, errsz, "objective line missing ':' in '%s'", input_path);
                goto fail;
            }
            *colon = '\0';
            expr = colon + 1;
            trim_inplace(head);
            trim_inplace(expr);
            hnt = token_split(head, htok, 8);
            if (hnt != 2 || !str_ieq(htok[0], "obj")) {
                snprintf(err, errsz, "invalid objective header in '%s'", input_path);
                goto fail;
            }
            obj_is_max = str_ieq(htok[1], "max") ? 1 : 0;
            free(objective);
            objective = NULL;
            if (!wmibo_expr_to_lp_terms(expr, &objective, err, errsz)) {
                goto fail;
            }
            continue;
        }
    }

    out = fopen(output_path, "wb");
    if (out == NULL) {
        snprintf(err, errsz, "cannot create '%s': %s", output_path, strerror(errno));
        goto fail;
    }

    fprintf(out, "%s\n", obj_is_max ? "Maximize" : "Minimize");
    fprintf(out, " obj: %s\n", (objective != NULL) ? objective : "0");
    fprintf(out, "Subject To\n");
    if (constraints.size == 0) {
        fprintf(out, " c0: 0 <= 0\n");
    } else {
        for (i = 0; i < constraints.size; ++i) {
            fprintf(out, " %s\n", constraints.data[i]);
        }
    }
    for (i = 1; i <= summary->B; ++i) {
        if (bvars[i].lb <= -SP_INF / 2 && bvars[i].ub >= SP_INF / 2) {
            fprintf(out, "Bounds b%d free\n", i);
        } else if (fabs(bvars[i].lb - bvars[i].ub) <= 1e-12) {
            fprintf(out, "Bounds b%d = %.17g\n", i, bvars[i].lb);
        } else if (bvars[i].lb > -SP_INF / 2 && bvars[i].ub < SP_INF / 2) {
            fprintf(out, "Bounds %.17g <= b%d <= %.17g\n", bvars[i].lb, i, bvars[i].ub);
        } else if (bvars[i].lb > -SP_INF / 2) {
            fprintf(out, "Bounds %.17g <= b%d\n", bvars[i].lb, i);
        } else {
            fprintf(out, "Bounds b%d <= %.17g\n", i, bvars[i].ub);
        }
    }
    for (i = 1; i <= summary->I; ++i) {
        if (fabs(ivars[i].lb - ivars[i].ub) <= 1e-12) {
            fprintf(out, "Bounds i%d = %.17g\n", i, ivars[i].lb);
        } else if (ivars[i].lb > -SP_INF / 2 && ivars[i].ub < SP_INF / 2) {
            fprintf(out, "Bounds %.17g <= i%d <= %.17g\n", ivars[i].lb, i, ivars[i].ub);
        } else if (ivars[i].lb > -SP_INF / 2) {
            fprintf(out, "Bounds %.17g <= i%d\n", ivars[i].lb, i);
        } else if (ivars[i].ub < SP_INF / 2) {
            fprintf(out, "Bounds i%d <= %.17g\n", i, ivars[i].ub);
        } else {
            fprintf(out, "Bounds i%d free\n", i);
        }
    }
    for (i = 1; i <= summary->R; ++i) {
        if (fabs(rvars[i].lb - rvars[i].ub) <= 1e-12) {
            fprintf(out, "Bounds r%d = %.17g\n", i, rvars[i].lb);
        } else if (rvars[i].lb > -SP_INF / 2 && rvars[i].ub < SP_INF / 2) {
            fprintf(out, "Bounds %.17g <= r%d <= %.17g\n", rvars[i].lb, i, rvars[i].ub);
        } else if (rvars[i].lb > -SP_INF / 2) {
            fprintf(out, "Bounds %.17g <= r%d\n", rvars[i].lb, i);
        } else if (rvars[i].ub < SP_INF / 2) {
            fprintf(out, "Bounds r%d <= %.17g\n", i, rvars[i].ub);
        } else {
            fprintf(out, "Bounds r%d free\n", i);
        }
    }
    if (summary->B > 0 || summary->I > 0) {
        fprintf(out, "Binaries\n");
        for (i = 1; i <= summary->B; ++i) {
            fprintf(out, " b%d\n", i);
        }
        for (i = 1; i <= summary->I; ++i) {
            if (ivars[i].is_binary) {
                StrBuf sb;
                strbuf_init(&sb);
                strbuf_appendf(&sb, "i%d", i);
                strvec_push_owned(&binary_ints, strbuf_detach(&sb));
                fprintf(out, " i%d\n", i);
            }
        }
    }
    if (summary->I > 0) {
        int wrote_generals = 0;
        for (i = 1; i <= summary->I; ++i) {
            if (!ivars[i].is_binary) {
                if (!wrote_generals) {
                    fprintf(out, "Generals\n");
                    wrote_generals = 1;
                }
                fprintf(out, " i%d\n", i);
            }
        }
    }
    fprintf(out, "End\n");

    fclose(out);
    fclose(fp);
    strvec_free(&constraints);
    strvec_free(&binary_ints);
    free(objective);
    free(bvars);
    free(ivars);
    free(rvars);
    return 1;

fail:
    if (out != NULL) fclose(out);
    if (fp != NULL) fclose(fp);
    strvec_free(&constraints);
    strvec_free(&binary_ints);
    free(objective);
    free(bvars);
    free(ivars);
    free(rvars);
    return 0;
}

static int handle_trivial_pure_sat_wmibo(const char *path, char *err, size_t errsz) {
    FILE *fp = fopen(path, "rb");
    char line_buf[8192];
    if (fp == NULL) {
        snprintf(err, errsz, "cannot open '%s': %s", path, strerror(errno));
        return -1;
    }
    while (fgets(line_buf, (int)sizeof(line_buf), fp) != NULL) {
        char line[8192];
        char *tok[64];
        int nt;
        snprintf(line, sizeof(line), "%s", line_buf);
        if (is_comment_or_blank(line)) continue;
        strip_inline_comment_hash(line);
        trim_inplace(line);
        if (line[0] == '\0') continue;
        nt = token_split(line, tok, 64);
        if (nt >= 3 && str_ieq(tok[0], "cl") && str_ieq(tok[1], "hard") && strcmp(tok[2], "0") == 0) {
            fclose(fp);
            puts("s UNSATISFIABLE");
            return 20;
        }
    }
    fclose(fp);
    puts("s SATISFIABLE");
    puts("v 0");
    return 10;
}

static const char *support_mark(unsigned mask, unsigned one) {
    return (mask & one) ? "yes" : "no";
}

static void print_kerberos_usage(const char *prog) {
    size_t i;
    fprintf(stderr,
            "usage: %s <file> [options]\n"
            "       %s --selftest\n"
            "       %s --replay <manifest> [--manifest-out <path>] [--audit-dispatch]\n"
            "       %s --grinder <cnf> <proof> [grinder-options]\n"
            "\n"
            "supported formats:\n"
            "  .cnf   -> slime by default, basilisk for count/project, wmibo for explain/trace/core modes\n"
            "  .wcnf  -> wmibo backend\n"
            "  .lp    -> pixie backend\n"
            "  .mps   -> pixie backend\n"
            "  .wmibo -> structure-based dispatch with basilisk for pure-sat count/project\n"
            "  .drat  -> grinder proof checker (via --grinder)\n"
            "\n"
            "kernel options:\n"
            "  --audit-dispatch   print structural signature, backend choice and ignored options\n"
            "  --manifest-out P   write a reproducible run manifest\n"
            "  --replay P         replay a prior manifest using stored solve arguments\n"
            "  --strict-options   fail instead of warning when an option will be ignored\n"
            "  --grinder          run grinder DRAT proof checker (passes remaining args to grinder)\n"
            "  --parallel MODE    auto|off|threads|mpi|hybrid\n"
            "  --jobs N           local worker count for threaded or hybrid modes\n"
            "  --split-depth N    root splitting depth for parallel task generation\n"
            "  --portfolio N      portfolio multiplicity for SAT/count portfolios\n"
            "  --sync-ms N        synchronization cadence in milliseconds\n"
            "  --selftest         run slime, basilisk, pixie, wmibo and dispatcher integration tests\n"
            "\n"
            "option capability matrix:\n"
            "  %-19s %-5s %-8s %-5s %-5s %s\n",
            prog, prog, prog, prog, "option", "slime", "basilisk", "pixie", "wmibo", "meaning");
    for (i = 0; i < sizeof(cli_forward_opts) / sizeof(cli_forward_opts[0]); ++i) {
        const CliOptionSpec *spec = &cli_forward_opts[i];
        fprintf(stderr,
                "  %-19s %-5s %-8s %-5s %-5s %s\n",
                spec->name,
                support_mark(spec->support_mask, SP_BACKEND_MASK_SLIME),
                support_mark(spec->support_mask, SP_BACKEND_MASK_BASILISK),
                support_mark(spec->support_mask, SP_BACKEND_MASK_PIXIE),
                support_mark(spec->support_mask, SP_BACKEND_MASK_WMIBO),
                spec->description);
    }
}

static int compute_file_digest(const char *path, uint64_t *hash_out, size_t *size_out, char *err, size_t errsz) {
    FILE *fp = fopen(path, "rb");
    unsigned char buf[4096];
    size_t total = 0U;
    uint64_t h = 1469598103934665603ULL;
    if (fp == NULL) {
        snprintf(err, errsz, "cannot open '%s': %s", path, strerror(errno));
        return 0;
    }
    for (;;) {
        size_t n = fread(buf, 1U, sizeof(buf), fp);
        size_t i;
        if (n == 0U) {
            break;
        }
        total += n;
        for (i = 0; i < n; ++i) {
            h ^= (uint64_t)buf[i];
            h *= 1099511628211ULL;
        }
    }
    if (ferror(fp)) {
        fclose(fp);
        snprintf(err, errsz, "failed to read '%s'", path);
        return 0;
    }
    fclose(fp);
    *hash_out = h;
    *size_out = total;
    return 1;
}

static int choose_dispatch_plan(const KrbCli *cli, DispatchPlan *plan, char *err, size_t errsz) {
    bool wants_wmibo_features;
    bool wants_basilisk_count;
    memset(plan, 0, sizeof(*plan));
    wants_basilisk_count = (cli->has_mode &&
                            (str_ieq(cli->mode_name, "count") || str_ieq(cli->mode_name, "project")));
    wants_wmibo_features = ((cli->has_mode && !str_ieq(cli->mode_name, "solve") && !wants_basilisk_count) ||
                            cli->option_seen[KOPT_TRACE_OUT] ||
                            cli->option_seen[KOPT_CORE_OUT]);

    if (cli->input_path == NULL) {
        snprintf(err, errsz, "missing input file");
        return 0;
    }

    if (has_ext_ci(cli->input_path, ".cnf")) {
        if (wants_basilisk_count) {
            plan->backend = SP_BACKEND_BASILISK;
            plan->backend_name = "basilisk";
            plan->backend_opts = basilisk_opts;
            plan->nbackend_opts = (int)(sizeof(basilisk_opts) / sizeof(basilisk_opts[0]));
            plan->reason = str_ieq(cli->mode_name, "project") ? "cnf-project" : "cnf-count";
        } else {
            plan->backend = wants_wmibo_features ? SP_BACKEND_WMIBO : SP_BACKEND_SLIME;
            plan->backend_name = backend_name(plan->backend);
            plan->backend_opts = wants_wmibo_features ? wmibo_opts : slime_opts;
            plan->nbackend_opts = wants_wmibo_features ? (int)(sizeof(wmibo_opts) / sizeof(wmibo_opts[0]))
                                                       : (int)(sizeof(slime_opts) / sizeof(slime_opts[0]));
            plan->reason = wants_wmibo_features ? "cnf-routed-to-wmibo-capability" : "native-cnf";
        }
        plan->input_format = "cnf";
        return 1;
    }

    if (has_ext_ci(cli->input_path, ".lp") || has_ext_ci(cli->input_path, ".mps")) {
        plan->backend = SP_BACKEND_PIXIE;
        plan->backend_name = "pixie";
        plan->backend_opts = pixie_opts;
        plan->nbackend_opts = (int)(sizeof(pixie_opts) / sizeof(pixie_opts[0]));
        plan->fmt_flag = has_ext_ci(cli->input_path, ".lp") ? "--lp" : "--mps";
        plan->input_format = has_ext_ci(cli->input_path, ".lp") ? "lp" : "mps";
        plan->reason = has_ext_ci(cli->input_path, ".lp") ? "native-lp" : "native-mps";
        return 1;
    }

    if (has_ext_ci(cli->input_path, ".wcnf")) {
        plan->backend = SP_BACKEND_WMIBO;
        plan->backend_name = "wmibo";
        plan->backend_opts = wmibo_opts;
        plan->nbackend_opts = (int)(sizeof(wmibo_opts) / sizeof(wmibo_opts[0]));
        plan->input_format = "wcnf";
        plan->reason = "native-wcnf";
        return 1;
    }

    if (has_ext_ci(cli->input_path, ".wmibo")) {
        if (!wmibo_scan_summary(cli->input_path, &plan->summary, err, errsz)) {
            return 0;
        }
        plan->have_summary = true;
        plan->input_format = "wmibo";
        if (wants_basilisk_count && plan->summary.kind == SP_WM_CLASS_PURE_SAT) {
            plan->backend = SP_BACKEND_BASILISK;
            plan->backend_name = "basilisk";
            plan->backend_opts = basilisk_opts;
            plan->nbackend_opts = (int)(sizeof(basilisk_opts) / sizeof(basilisk_opts[0]));
            plan->reason = str_ieq(cli->mode_name, "project") ? "wmibo-pure-sat-project" : "wmibo-pure-sat-count";
            return 1;
        }
        if (wants_wmibo_features) {
            plan->backend = SP_BACKEND_WMIBO;
            plan->backend_name = "wmibo";
            plan->backend_opts = wmibo_opts;
            plan->nbackend_opts = (int)(sizeof(wmibo_opts) / sizeof(wmibo_opts[0]));
            plan->reason = "wmibo-capability-override";
            return 1;
        }
        if (plan->summary.kind == SP_WM_CLASS_PURE_SAT) {
            plan->backend = SP_BACKEND_SLIME;
            plan->backend_name = "slime";
            plan->backend_opts = slime_opts;
            plan->nbackend_opts = (int)(sizeof(slime_opts) / sizeof(slime_opts[0]));
            plan->reason = "wmibo-pure-sat";
            return 1;
        }
        if (plan->summary.kind == SP_WM_CLASS_PURE_LP || plan->summary.kind == SP_WM_CLASS_PURE_MIP) {
            plan->backend = SP_BACKEND_PIXIE;
            plan->backend_name = "pixie";
            plan->backend_opts = pixie_opts;
            plan->nbackend_opts = (int)(sizeof(pixie_opts) / sizeof(pixie_opts[0]));
            plan->fmt_flag = "--lp";
            plan->reason = (plan->summary.kind == SP_WM_CLASS_PURE_LP) ? "wmibo-pure-lp" : "wmibo-pure-mip";
            return 1;
        }
        plan->backend = SP_BACKEND_WMIBO;
        plan->backend_name = "wmibo";
        plan->backend_opts = wmibo_opts;
        plan->nbackend_opts = (int)(sizeof(wmibo_opts) / sizeof(wmibo_opts[0]));
        plan->reason = "wmibo-hybrid";
        return 1;
    }

    snprintf(err, errsz, "unsupported input format '%s'", cli->input_path);
    return 0;
}

static int validate_options_for_plan(const KrbCli *cli,
                                     const DispatchPlan *plan,
                                     StrVec *ignored,
                                     char *err,
                                     size_t errsz) {
    size_t i;
    unsigned mask = backend_mask(plan->backend);
    strvec_init(ignored);
    for (i = 0; i < sizeof(cli_forward_opts) / sizeof(cli_forward_opts[0]); ++i) {
        const CliOptionSpec *spec = &cli_forward_opts[i];
        if (!cli->option_seen[spec->id]) {
            continue;
        }
        if ((spec->support_mask & mask) != 0U) {
            continue;
        }
        if (spec->id == KOPT_MODE && cli->has_mode && str_ieq(cli->mode_name, "solve")) {
            continue;
        }
        strvec_push_copy(ignored, spec->name);
    }
    if (cli->strict_options && ignored->size > 0) {
        snprintf(err, errsz, "option %s is not supported by backend %s", ignored->data[0], plan->backend_name);
        return 0;
    }
    return 1;
}

static void run_report_init(RunReport *report) {
    memset(report, 0, sizeof(*report));
    strvec_init(&report->ignored_options);
    strvec_init(&report->solve_args);
}

static void run_report_free(RunReport *report) {
    strvec_free(&report->ignored_options);
    strvec_free(&report->solve_args);
}

static void print_dispatch_audit(const RunReport *report) {
    printf("c kerberos.dispatch input=%s format=%s backend=%s reason=%s\n",
           report->input_path,
           report->input_format,
           report->plan.backend_name,
           report->plan.reason);
    printf("c kerberos.parallel requested=%s resolved=%s jobs=%d split-depth=%d portfolio=%d sync-ms=%d rank=%d/%d local-rank=%d/%d reason=%s\n",
           krb_parallel_mode_name(report->parallel_rt.requested_mode),
           krb_parallel_mode_name(report->parallel_rt.resolved_mode),
           report->parallel_rt.jobs,
           report->parallel_cfg.split_depth,
           report->parallel_cfg.portfolio,
           report->parallel_cfg.sync_ms,
           report->parallel_rt.world_rank,
           report->parallel_rt.world_size,
           report->parallel_rt.local_rank,
           report->parallel_rt.local_size,
           report->parallel_rt.reason ? report->parallel_rt.reason : "unknown");
    printf("c kerberos.manifest hash=%016llx size=%llu elapsed=%.6f\n",
           (unsigned long long)report->input_hash,
           (unsigned long long)report->input_size,
           report->elapsed_sec);
    if (report->plan.have_summary) {
        const WmiboSummary *s = &report->plan.summary;
        double avg_clause = (s->hard_clause_count + s->soft_clause_count) > 0 ?
                            ((double)s->clause_lits / (double)(s->hard_clause_count + s->soft_clause_count)) : 0.0;
        double avg_lin = s->lin_count > 0 ? ((double)s->lin_terms / (double)s->lin_count) : 0.0;
        printf("c kerberos.signature class=%s B=%d I=%d R=%d hard=%d soft=%d lin=%d ind=%d avg_clause=%.3f max_clause=%d avg_lin=%.3f max_lin=%d fixed=(%d,%d,%d) binary_i=%d coef_abs=[%.12g,%.12g]\n",
               wm_class_name(s->kind),
               s->B,
               s->I,
               s->R,
               s->hard_clause_count,
               s->soft_clause_count,
               s->lin_count,
               s->indicator_count,
               avg_clause,
               s->max_clause_len,
               avg_lin,
               s->max_lin_terms,
               s->fixed_b,
               s->fixed_i,
               s->fixed_r,
               s->binary_i,
               s->min_abs_coef,
               s->max_abs_coef);
    }
    if (report->ignored_options.size > 0) {
        int i;
        printf("c kerberos.ignored");
        for (i = 0; i < report->ignored_options.size; ++i) {
            printf(" %s", report->ignored_options.data[i]);
        }
        printf("\n");
    }
}

static int write_manifest(const char *path, const RunReport *report, char *err, size_t errsz) {
    FILE *fp;
    int i;
    fp = fopen(path, "wb");
    if (fp == NULL) {
        snprintf(err, errsz, "cannot write manifest '%s': %s", path, strerror(errno));
        return 0;
    }
    fprintf(fp, "KERBEROS_MANIFEST_V1\n");
    fprintf(fp, "build_date=%s\n", __DATE__);
    fprintf(fp, "build_time=%s\n", __TIME__);
    fprintf(fp, "input_path=%s\n", report->input_path ? report->input_path : "");
    fprintf(fp, "input_format=%s\n", report->input_format ? report->input_format : "unknown");
    fprintf(fp, "input_hash=%016llx\n", (unsigned long long)report->input_hash);
    fprintf(fp, "input_size=%llu\n", (unsigned long long)report->input_size);
    fprintf(fp, "backend=%s\n", report->plan.backend_name ? report->plan.backend_name : "unknown");
    fprintf(fp, "dispatch_reason=%s\n", report->plan.reason ? report->plan.reason : "unknown");
    fprintf(fp, "parallel_requested=%s\n", krb_parallel_mode_name(report->parallel_rt.requested_mode));
    fprintf(fp, "parallel_resolved=%s\n", krb_parallel_mode_name(report->parallel_rt.resolved_mode));
    fprintf(fp, "parallel_jobs=%d\n", report->parallel_rt.jobs);
    fprintf(fp, "parallel_split_depth=%d\n", report->parallel_cfg.split_depth);
    fprintf(fp, "parallel_portfolio=%d\n", report->parallel_cfg.portfolio);
    fprintf(fp, "parallel_sync_ms=%d\n", report->parallel_cfg.sync_ms);
    fprintf(fp, "parallel_world_rank=%d\n", report->parallel_rt.world_rank);
    fprintf(fp, "parallel_world_size=%d\n", report->parallel_rt.world_size);
    fprintf(fp, "parallel_local_rank=%d\n", report->parallel_rt.local_rank);
    fprintf(fp, "parallel_local_size=%d\n", report->parallel_rt.local_size);
    fprintf(fp, "parallel_reason=%s\n", report->parallel_rt.reason ? report->parallel_rt.reason : "unknown");
    if (report->plan.have_summary) {
        fprintf(fp, "wmibo_class=%s\n", wm_class_name(report->plan.summary.kind));
        fprintf(fp, "wmibo_signature=B:%d,I:%d,R:%d,hard:%d,soft:%d,lin:%d,ind:%d\n",
                report->plan.summary.B,
                report->plan.summary.I,
                report->plan.summary.R,
                report->plan.summary.hard_clause_count,
                report->plan.summary.soft_clause_count,
                report->plan.summary.lin_count,
                report->plan.summary.indicator_count);
    }
    fprintf(fp, "elapsed_sec=%.9f\n", report->elapsed_sec);
    fprintf(fp, "exit_code=%d\n", report->exit_code);
    fprintf(fp, "ignored_count=%d\n", report->ignored_options.size);
    for (i = 0; i < report->ignored_options.size; ++i) {
        fprintf(fp, "ignored_%d=%s\n", i, report->ignored_options.data[i]);
    }
    fprintf(fp, "solve_argc=%d\n", report->solve_args.size);
    for (i = 0; i < report->solve_args.size; ++i) {
        fprintf(fp, "solve_arg_%d=%s\n", i, report->solve_args.data[i]);
    }
    fclose(fp);
    return 1;
}

static int read_manifest_solve_args(const char *path, StrVec *out_args, char *err, size_t errsz) {
    FILE *fp;
    char line[4096];
    int saw_header = 0;
    strvec_init(out_args);
    fp = fopen(path, "rb");
    if (fp == NULL) {
        snprintf(err, errsz, "cannot open manifest '%s': %s", path, strerror(errno));
        return 0;
    }
    while (fgets(line, (int)sizeof(line), fp) != NULL) {
        trim_inplace(line);
        if (line[0] == '\0') {
            continue;
        }
        if (!saw_header) {
            if (strcmp(line, "KERBEROS_MANIFEST_V1") != 0) {
                fclose(fp);
                strvec_free(out_args);
                snprintf(err, errsz, "invalid manifest header in '%s'", path);
                return 0;
            }
            saw_header = 1;
            continue;
        }
        if (strncmp(line, "solve_arg_", 10) == 0) {
            char *eq = strchr(line, '=');
            if (eq != NULL) {
                strvec_push_copy(out_args, eq + 1);
            }
        }
    }
    fclose(fp);
    if (!saw_header || out_args->size == 0) {
        strvec_free(out_args);
        snprintf(err, errsz, "manifest '%s' does not contain replay arguments", path);
        return 0;
    }
    return 1;
}

static int execute_dispatch(const KrbCli *cli, const StrVec *solve_args, RunReport *report, char *err, size_t errsz) {
    ArgVec av;
    int rc = 0;
    double t0;
    double t1;
    int i;
    run_report_init(report);
    report->input_path = cli->input_path;
    report->parallel_cfg = cli->parallel;

    if (!krb_parallel_runtime_resolve(&cli->parallel, &report->parallel_rt, err, errsz)) {
        run_report_free(report);
        return 0;
    }
    if (krb_parallel_mode_uses_mpi(report->parallel_rt.resolved_mode) &&
        !krb_parallel_mpi_init(&report->parallel_rt, err, errsz)) {
        run_report_free(report);
        return 0;
    }

    if (!choose_dispatch_plan(cli, &report->plan, err, errsz)) {
        run_report_free(report);
        return 0;
    }
    if (!validate_options_for_plan(cli, &report->plan, &report->ignored_options, err, errsz)) {
        run_report_free(report);
        return 0;
    }
    report->ignored_count = report->ignored_options.size;
    report->input_format = report->plan.input_format;
    if (!compute_file_digest(cli->input_path, &report->input_hash, &report->input_size, err, errsz)) {
        run_report_free(report);
        return 0;
    }
    for (i = 0; i < solve_args->size; ++i) {
        strvec_push_copy(&report->solve_args, solve_args->data[i]);
    }

    t0 = (double)clock() / (double)CLOCKS_PER_SEC;

    if (has_ext_ci(cli->input_path, ".wmibo") && report->plan.backend == SP_BACKEND_SLIME) {
        char tmp_cnf[512];
        if (report->plan.summary.B == 0) {
            rc = handle_trivial_pure_sat_wmibo(cli->input_path, err, errsz);
            if (rc < 0) {
                run_report_free(report);
                return 0;
            }
        } else {
            if (!make_temp_path(".cnf", tmp_cnf, sizeof(tmp_cnf))) {
                snprintf(err, errsz, "failed to create temporary cnf path");
                run_report_free(report);
                return 0;
            }
            if (!wmibo_translate_pure_sat_to_cnf(cli->input_path, tmp_cnf, &report->plan.summary, err, errsz)) {
                run_report_free(report);
                return 0;
            }
            if (!build_backend_argv(&av, "slime", slime_opts,
                                    (int)(sizeof(slime_opts) / sizeof(slime_opts[0])),
                                    solve_args, cli->input_path, NULL, tmp_cnf)) {
                remove(tmp_cnf);
                run_report_free(report);
                snprintf(err, errsz, "failed to build backend argv");
                return 0;
            }
            rc = run_backend(SP_BACKEND_SLIME, &av);
            argvec_free(&av);
            remove(tmp_cnf);
        }
    } else if (has_ext_ci(cli->input_path, ".wmibo") && report->plan.backend == SP_BACKEND_BASILISK) {
        char tmp_cnf[512];
        if (!make_temp_path(".cnf", tmp_cnf, sizeof(tmp_cnf))) {
            snprintf(err, errsz, "failed to create temporary cnf path");
            run_report_free(report);
            return 0;
        }
        if (!wmibo_translate_pure_sat_to_cnf(cli->input_path, tmp_cnf, &report->plan.summary, err, errsz)) {
            run_report_free(report);
            return 0;
        }
        if (!build_backend_argv(&av, "basilisk", basilisk_opts,
                                (int)(sizeof(basilisk_opts) / sizeof(basilisk_opts[0])),
                                solve_args, cli->input_path, NULL, tmp_cnf)) {
            remove(tmp_cnf);
            run_report_free(report);
            snprintf(err, errsz, "failed to build backend argv");
            return 0;
        }
        rc = run_backend(SP_BACKEND_BASILISK, &av);
        argvec_free(&av);
        remove(tmp_cnf);
    } else if (has_ext_ci(cli->input_path, ".wmibo") && report->plan.backend == SP_BACKEND_PIXIE) {
        char tmp_lp[512];
        if (!make_temp_path(".lp", tmp_lp, sizeof(tmp_lp))) {
            snprintf(err, errsz, "failed to create temporary lp path");
            run_report_free(report);
            return 0;
        }
        if (!wmibo_translate_pure_linear_to_lp(cli->input_path, tmp_lp, &report->plan.summary, err, errsz)) {
            run_report_free(report);
            return 0;
        }
        if (!build_backend_argv(&av, "pixie", pixie_opts,
                                (int)(sizeof(pixie_opts) / sizeof(pixie_opts[0])),
                                solve_args, cli->input_path, "--lp", tmp_lp)) {
            remove(tmp_lp);
            run_report_free(report);
            snprintf(err, errsz, "failed to build backend argv");
            return 0;
        }
        rc = run_backend(SP_BACKEND_PIXIE, &av);
        argvec_free(&av);
        remove(tmp_lp);
    } else {
        if (!build_backend_argv(&av,
                                report->plan.backend_name,
                                report->plan.backend_opts,
                                report->plan.nbackend_opts,
                                solve_args,
                                cli->input_path,
                                report->plan.fmt_flag,
                                cli->input_path)) {
            run_report_free(report);
            snprintf(err, errsz, "failed to build backend argv");
            return 0;
        }
        rc = run_backend(report->plan.backend, &av);
        argvec_free(&av);
    }

    t1 = (double)clock() / (double)CLOCKS_PER_SEC;
    report->elapsed_sec = t1 - t0;
    report->exit_code = rc;

    if (cli->audit_dispatch) {
        print_dispatch_audit(report);
    } else if (report->ignored_options.size > 0) {
        fprintf(stderr, "kerberos: warning: option(s) ignored by backend %s:", report->plan.backend_name);
        for (i = 0; i < report->ignored_options.size; ++i) {
            fprintf(stderr, " %s", report->ignored_options.data[i]);
        }
        fprintf(stderr, "\n");
    }

    return 1;
}

static int run_slime_selftest(void) {
    char *args[] = {"slime", "--selftest", NULL};
    return slime_entry(2, args) == 0;
}

static int run_basilisk_selftest(void) {
    char *args[] = {"basilisk", "--selftest", NULL};
    return basilisk_entry(2, args) == 0;
}

static int run_pixie_selftest(void) {
    char *args[] = {"pixie", "--selftest", NULL};
    return pixie_entry(2, args) == 0;
}

static int run_wmibo_selftest(void) {
    char *args[] = {"wmibo", "--selftest", NULL};
    return wmibo_entry(2, args) == 0;
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

static int run_kerberos_selftest_case_ext_args(const char *name,
                                               const char *text,
                                               const char *ext,
                                               const char *const *opts,
                                               int nopts,
                                               int exp_rc) {
    char tmp_name[256];
    char *argv_case[16];
    int argc_case = 0;
    KrbCli cli;
    StrVec solve_args;
    RunReport report;
    char err[512];
    int ok;
    unsigned long stamp = (unsigned long)time(NULL);
    stamp ^= (unsigned long)clock();
    stamp ^= (unsigned long)(uintptr_t)(const void *)text;
    stamp ^= (unsigned long)(uintptr_t)(const void *)ext;
    snprintf(tmp_name, sizeof(tmp_name), "kerberos_selftest_%s_%lu%s", name, stamp, ext);
    if (!write_text_file(tmp_name, text)) {
        fprintf(stderr, "kerberos selftest[%s]: failed to write temp file\n", name);
        return 0;
    }
    argv_case[argc_case++] = "kerberos";
    argv_case[argc_case++] = tmp_name;
    for (int i = 0; i < nopts; ++i) {
        if (argc_case + 1 >= (int)(sizeof(argv_case) / sizeof(argv_case[0]))) {
            remove(tmp_name);
            fprintf(stderr, "kerberos selftest[%s]: too many argv entries\n", name);
            return 0;
        }
        argv_case[argc_case++] = (char *)opts[i];
    }
    argv_case[argc_case] = NULL;
    if (!parse_cli_args(&cli, &solve_args, argc_case, argv_case, err, sizeof(err))) {
        remove(tmp_name);
        fprintf(stderr, "kerberos selftest[%s]: %s\n", name, err);
        return 0;
    }
    ok = execute_dispatch(&cli, &solve_args, &report, err, sizeof(err));
    strvec_free(&solve_args);
    remove(tmp_name);
    if (!ok) {
        fprintf(stderr, "kerberos selftest[%s]: %s\n", name, err);
        return 0;
    }
    if (report.exit_code != exp_rc) {
        fprintf(stderr, "kerberos selftest[%s]: expected rc=%d got rc=%d\n", name, exp_rc, report.exit_code);
        run_report_free(&report);
        return 0;
    }
    run_report_free(&report);
    return 1;
}

static int run_kerberos_selftest_case_ext(const char *name,
                                          const char *text,
                                          const char *ext,
                                          const char *opt1,
                                          const char *opt2,
                                          int exp_rc) {
    const char *opts[2];
    int nopts = 0;
    if (opt1 != NULL) opts[nopts++] = opt1;
    if (opt2 != NULL) opts[nopts++] = opt2;
    return run_kerberos_selftest_case_ext_args(name, text, ext, opts, nopts, exp_rc);
}

static int run_kerberos_selftest_case(const char *name,
                                      const char *text,
                                      const char *opt1,
                                      const char *opt2,
                                      int exp_rc) {
    return run_kerberos_selftest_case_ext(name, text, ".wmibo", opt1, opt2, exp_rc);
}

static int run_kerberos_selftest(void) {
    const char *count_cnf =
        "c ind 1 0\n"
        "p cnf 2 1\n"
        "1 2 0\n";
    const char *sat_cnf =
        "p cnf 2 2\n"
        "1 2 0\n"
        "-1 2 0\n";
    const char *pure_sat =
        "p wmibo 2 0 0 2 0 0\n"
        "begin cnf\n"
        "cl hard b1 0\n"
        "cl hard ~b1 b2 0\n"
        "end\n";
    const char *pure_lp =
        "p wmibo 0 1 0 0 1 0\n"
        "var i 1 [0,5]\n"
        "begin lin\n"
        "lc C1 >= 2 : 1 i1\n"
        "end\n"
        "begin obj\n"
        "obj min : lin 1 i1\n"
        "end\n";
    const char *hybrid =
        "p wmibo 1 1 0 1 1 1\n"
        "var i 1 [0,4]\n"
        "begin cnf\n"
        "cl hard b1 0\n"
        "end\n"
        "begin lin\n"
        "lc CAP <= 2 : 1 i1\n"
        "end\n"
        "begin ind\n"
        "ind b1 => CAP\n"
        "end\n"
        "begin obj\n"
        "obj min : lin 1 i1\n"
        "end\n";

    if (!run_slime_selftest()) return 1;
    if (!run_basilisk_selftest()) return 1;
    if (!run_pixie_selftest()) return 1;
    if (!run_wmibo_selftest()) return 1;
    if (!run_kerberos_selftest_case_ext("cnf_count", count_cnf, ".cnf", "--mode", "count", 0)) return 1;
    if (!run_kerberos_selftest_case_ext("cnf_parallel_off", sat_cnf, ".cnf", "--parallel", "off", 10)) return 1;
    if (!run_kerberos_selftest_case("pure_sat_dispatch", pure_sat, NULL, NULL, 10)) return 1;
    if (!run_kerberos_selftest_case("pure_sat_count_dispatch", pure_sat, "--mode", "count", 0)) return 1;
    if (!run_kerberos_selftest_case("pure_lp_dispatch", pure_lp, NULL, NULL, 0)) return 1;
    if (!run_kerberos_selftest_case("hybrid_explain", hybrid, "--mode", "explain", 0)) return 1;
#if defined(SATX_HAVE_THREADS)
    {
        const char *parallel_opts[] = { "--parallel", "threads", "--jobs", "2", "--portfolio", "4" };
        const char *cube_opts[] = { "--parallel", "threads", "--jobs", "2", "--split-depth", "1" };
        if (!run_kerberos_selftest_case_ext_args("cnf_parallel_threads", sat_cnf, ".cnf", parallel_opts, 6, 10)) return 1;
        if (!run_kerberos_selftest_case_ext_args("cnf_parallel_cubes", sat_cnf, ".cnf", cube_opts, 6, 10)) return 1;
    }
#endif
    fprintf(stderr, "kerberos selftest: OK\n");
    return 0;
}

static int parse_replay_cli(const StrVec *stored_args, KrbCli *cli, StrVec *solve_args, char *err, size_t errsz) {
    ArgVec av;
    int i;
    argvec_init(&av);
    argvec_push(&av, "kerberos");
    for (i = 0; i < stored_args->size; ++i) {
        argvec_push(&av, stored_args->data[i]);
    }
    if (!parse_cli_args(cli, solve_args, av.size, av.data, err, errsz)) {
        argvec_free(&av);
        return 0;
    }
    cli->input_path = NULL;
    for (i = 0; i < solve_args->size; ++i) {
        if (solve_args->data[i][0] != '-') {
            cli->input_path = solve_args->data[i];
            break;
        }
    }
    argvec_free(&av);
    return 1;
}

int main(int argc, char **argv) {
    KrbCli cli;
    StrVec solve_args;
    RunReport report;
    char err[512];
    int rc;

    if (!parse_cli_args(&cli, &solve_args, argc, argv, err, sizeof(err))) {
        fprintf(stderr, "kerberos: %s\n", err);
        return 2;
    }

    if (cli.grinder) {
        ArgVec gav;
        int j;
        argvec_init(&gav);
        argvec_push(&gav, "grinder");
        for (j = 0; j < cli.grinder_args.size; j++) {
            argvec_push(&gav, cli.grinder_args.data[j]);
        }
        rc = grinder_entry(gav.size, gav.data);
        argvec_free(&gav);
        strvec_free(&cli.grinder_args);
        strvec_free(&solve_args);
        return rc;
    }

    if (cli.show_help) {
        print_kerberos_usage((argc > 0) ? argv[0] : "kerberos");
        strvec_free(&solve_args);
        return 0;
    }

    if (cli.selftest) {
        strvec_free(&solve_args);
        return run_kerberos_selftest();
    }

    if (cli.replay_path != NULL) {
        StrVec stored_args;
        KrbCli replay_cli;
        StrVec replay_solve_args;
        int ok;
        if (solve_args.size > 0) {
            fprintf(stderr, "kerberos: --replay does not accept additional solve arguments\n");
            strvec_free(&solve_args);
            return 2;
        }
        if (!read_manifest_solve_args(cli.replay_path, &stored_args, err, sizeof(err))) {
            fprintf(stderr, "kerberos: %s\n", err);
            strvec_free(&solve_args);
            return 1;
        }
        ok = parse_replay_cli(&stored_args, &replay_cli, &replay_solve_args, err, sizeof(err));
        strvec_free(&stored_args);
        strvec_free(&solve_args);
        if (!ok) {
            fprintf(stderr, "kerberos: %s\n", err);
            return 1;
        }
        replay_cli.audit_dispatch = cli.audit_dispatch;
        replay_cli.strict_options = cli.strict_options;
        if (cli.manifest_out != NULL) {
            replay_cli.manifest_out = cli.manifest_out;
        }
        if (replay_cli.input_path == NULL) {
            fprintf(stderr, "kerberos: replay manifest does not contain an input file\n");
            strvec_free(&replay_solve_args);
            return 1;
        }
        if (!execute_dispatch(&replay_cli, &replay_solve_args, &report, err, sizeof(err))) {
            fprintf(stderr, "kerberos: %s\n", err);
            strvec_free(&replay_solve_args);
            krb_parallel_mpi_finalize();
            return 1;
        }
        strvec_free(&replay_solve_args);
        if (replay_cli.manifest_out != NULL && !write_manifest(replay_cli.manifest_out, &report, err, sizeof(err))) {
            fprintf(stderr, "kerberos: %s\n", err);
            krb_parallel_mpi_finalize();
            run_report_free(&report);
            return 1;
        }
        rc = report.exit_code;
        krb_parallel_mpi_finalize();
        run_report_free(&report);
        return rc;
    }

    if (cli.input_path == NULL) {
        print_kerberos_usage((argc > 0) ? argv[0] : "kerberos");
        strvec_free(&solve_args);
        return 2;
    }

    if (!execute_dispatch(&cli, &solve_args, &report, err, sizeof(err))) {
        fprintf(stderr, "kerberos: %s\n", err);
        strvec_free(&solve_args);
        krb_parallel_mpi_finalize();
        return 1;
    }
    strvec_free(&solve_args);

    if (cli.manifest_out != NULL && !write_manifest(cli.manifest_out, &report, err, sizeof(err))) {
        fprintf(stderr, "kerberos: %s\n", err);
        krb_parallel_mpi_finalize();
        run_report_free(&report);
        return 1;
    }
    rc = report.exit_code;
    krb_parallel_mpi_finalize();
    run_report_free(&report);
    return rc;
}

