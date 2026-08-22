#include "krb_accel.h"

#include <stdio.h>

int krb_accel_cuda_compiled(void) {
    return 0;
}

int krb_accel_cuda_runtime_available(void) {
    return 0;
}

int krb_accel_cuda_dense_lp_available(void) {
    return 0;
}

bool krb_accel_cuda_select_device(int device, char *err, size_t errsz) {
    (void)device;
    if (err != NULL && errsz > 0U) {
        snprintf(err, errsz, "binary was built without CUDA support");
    }
    return false;
}

bool krb_accel_cuda_managed_alloc_bytes(size_t bytes, void **out, char *err, size_t errsz) {
    (void)bytes;
    if (out != NULL) {
        *out = NULL;
    }
    if (err != NULL && errsz > 0U) {
        snprintf(err, errsz, "binary was built without CUDA support");
    }
    return false;
}

bool krb_accel_cuda_managed_alloc_doubles(size_t count, double **out, char *err, size_t errsz) {
    (void)count;
    if (out != NULL) {
        *out = NULL;
    }
    if (err != NULL && errsz > 0U) {
        snprintf(err, errsz, "binary was built without CUDA support");
    }
    return false;
}

bool krb_accel_cuda_managed_alloc_ints(size_t count, int **out, char *err, size_t errsz) {
    (void)count;
    if (out != NULL) {
        *out = NULL;
    }
    if (err != NULL && errsz > 0U) {
        snprintf(err, errsz, "binary was built without CUDA support");
    }
    return false;
}

void krb_accel_cuda_managed_free(void *ptr) {
    (void)ptr;
}

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
                                        size_t errsz) {
    (void)clause_offsets;
    (void)clause_lits;
    (void)nclauses;
    (void)nvars;
    (void)assign;
    (void)active_project;
    (void)project_only;
    (void)score_tmp;
    (void)score_pos;
    (void)score_neg;
    if (err != NULL && errsz > 0U) {
        snprintf(err, errsz, "binary was built without CUDA support");
    }
    return false;
}

bool krb_accel_cuda_tableau_pivot(double *dst,
                                  const double *src,
                                  int rows,
                                  int cols,
                                  int pivot_row,
                                  int pivot_col,
                                  char *err,
                                  size_t errsz) {
    (void)dst;
    (void)src;
    (void)rows;
    (void)cols;
    (void)pivot_row;
    (void)pivot_col;
    if (err != NULL && errsz > 0U) {
        snprintf(err, errsz, "binary was built without CUDA support");
    }
    return false;
}

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
                                      size_t errsz) {
    (void)table;
    (void)n_index;
    (void)row_index;
    (void)cols;
    (void)total_cols;
    (void)phase;
    (void)eps;
    (void)scratch_idx;
    (void)scratch_metric;
    (void)scratch_cap;
    if (out_col != NULL) {
        *out_col = -1;
    }
    if (err != NULL && errsz > 0U) {
        snprintf(err, errsz, "binary was built without CUDA support");
    }
    return false;
}

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
                                     size_t errsz) {
    (void)table;
    (void)b_index;
    (void)rows;
    (void)cols;
    (void)pivot_col;
    (void)rhs_col;
    (void)eps;
    (void)scratch_idx;
    (void)scratch_metric;
    (void)scratch_cap;
    if (out_row != NULL) {
        *out_row = -1;
    }
    if (err != NULL && errsz > 0U) {
        snprintf(err, errsz, "binary was built without CUDA support");
    }
    return false;
}
