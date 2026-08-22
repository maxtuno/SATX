#ifndef KRB_ACCEL_H
#define KRB_ACCEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KRB_ACCEL_DEFAULT_CUDA_MIN_CELLS ((size_t)262144)

typedef enum {
    KRB_ACCEL_MODE_AUTO = 0,
    KRB_ACCEL_MODE_OFF = 1,
    KRB_ACCEL_MODE_ON = 2
} KrbAccelMode;

typedef enum {
    KRB_ACCEL_PATH_CPU = 0,
    KRB_ACCEL_PATH_CUDA = 1
} KrbAccelPath;

typedef struct {
    KrbAccelMode mode;
    int cuda_device;         /* -1 means runtime default device */
    size_t cuda_min_cells;   /* minimum dense LP cells before trying CUDA */
} KrbAccelConfig;

typedef struct {
    KrbAccelPath path;
    size_t dense_cells;
    int cuda_compiled;
    int cuda_runtime;
    int active_device;
    const char *reason;
} KrbAccelDecision;

void krb_accel_config_defaults(KrbAccelConfig *cfg);
bool krb_accel_parse_mode(const char *text, KrbAccelMode *out);
const char *krb_accel_mode_name(KrbAccelMode mode);

int krb_accel_cuda_compiled(void);
int krb_accel_cuda_runtime_available(void);
int krb_accel_cuda_dense_lp_available(void);
bool krb_accel_cuda_select_device(int device, char *err, size_t errsz);
bool krb_accel_cuda_managed_alloc_bytes(size_t bytes, void **out, char *err, size_t errsz);
bool krb_accel_cuda_managed_alloc_doubles(size_t count, double **out, char *err, size_t errsz);
bool krb_accel_cuda_managed_alloc_ints(size_t count, int **out, char *err, size_t errsz);
void krb_accel_cuda_managed_free(void *ptr);
bool krb_accel_cuda_score_cnf_branching(const int *clause_offsets,
                                        const int *clause_lits,
                                        int nclauses,
                                        int nvars,
                                        const signed char *assign,
                                        const unsigned char *active_project,
                                        int project_only,
                                        int *score_tmp,
                                        int *score_pos,
                                        int *score_neg,
                                        char *err,
                                        size_t errsz);
bool krb_accel_cuda_tableau_pivot(double *dst,
                                  const double *src,
                                  int rows,
                                  int cols,
                                  int pivot_row,
                                  int pivot_col,
                                  char *err,
                                  size_t errsz);
bool krb_accel_cuda_find_entering_col(const double *table,
                                      const int *n_index,
                                      int row_index,
                                      int cols,
                                      int total_cols,
                                      int phase,
                                      double eps,
                                      int *scratch_idx,
                                      double *scratch_metric,
                                      int scratch_cap,
                                      int *out_col,
                                      char *err,
                                      size_t errsz);
bool krb_accel_cuda_find_leaving_row(const double *table,
                                     const int *b_index,
                                     int rows,
                                     int cols,
                                     int pivot_col,
                                     int rhs_col,
                                     double eps,
                                     int *scratch_idx,
                                     double *scratch_metric,
                                     int scratch_cap,
                                     int *out_row,
                                     char *err,
                                     size_t errsz);

bool krb_accel_choose_dense_lp(const KrbAccelConfig *cfg,
                               const char *solver_name,
                               int rows,
                               int cols,
                               KrbAccelDecision *out,
                               char *err,
                               size_t errsz);

void krb_accel_log(FILE *stream, const char *solver_name, const KrbAccelDecision *decision);

#ifdef __cplusplus
}
#endif

#endif
