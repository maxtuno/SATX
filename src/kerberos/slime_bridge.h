#ifndef SATX_SLIME_BRIDGE_H
#define SATX_SLIME_BRIDGE_H
/*
 * Puente C sobre el kernel CDCL SLIME de Kerberos (src/kerberos/slime.c).
 * Declara la C API embebida (slime_sat_handle_*) con las estructuras públicas
 * exactas del fuente. Si slime.c cambia estas estructuras, actualizar este fichero.
 */

#ifdef __cplusplus
extern "C" {
#endif

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

/* Crea un handle persistente. opt/stats pueden ser NULL (opciones por defecto). */
SlimeSatHandle *slime_sat_handle_create(int nvars,
                                        int nclauses,
                                        const int *const *clauses,
                                        const int *sizes,
                                        const SlimeSatOptions *opt);

/* Resuelve. rc: 10 = SAT (model01 lleno), 20 = UNSAT, 0 = error interno. */
int slime_sat_handle_solve(SlimeSatHandle *handle,
                           const int *assumptions,
                           int num_assumptions,
                           SlimeSatStats *stats,
                           unsigned char *model01);

void slime_sat_handle_destroy(SlimeSatHandle *handle);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SATX_SLIME_BRIDGE_H */

