#include "krb_accel.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define KRB_ACCEL_CUDA_THREADS 256

static void krb_accel_cuda_write_error(cudaError_t st, char *err, size_t errsz, const char *prefix) {
    if (err == NULL || errsz == 0U) {
        return;
    }
    snprintf(err, errsz, "%s: %s", prefix, cudaGetErrorString(st));
}

__device__ static bool krb_accel_entering_better(double cand_val,
                                                 int cand_n,
                                                 int cand_col,
                                                 double best_val,
                                                 int best_n,
                                                 int best_col,
                                                 double eps) {
    if (cand_col < 0) {
        return false;
    }
    if (best_col < 0) {
        return true;
    }
    if (cand_val < best_val - eps) {
        return true;
    }
    if (fabs(cand_val - best_val) <= eps && cand_n < best_n) {
        return true;
    }
    return false;
}

__device__ static bool krb_accel_leaving_better(double cand_ratio,
                                                int cand_b,
                                                int cand_row,
                                                double best_ratio,
                                                int best_b,
                                                int best_row,
                                                double eps) {
    if (cand_row < 0) {
        return false;
    }
    if (best_row < 0) {
        return true;
    }
    if (cand_ratio < best_ratio - eps) {
        return true;
    }
    if (fabs(cand_ratio - best_ratio) <= eps && cand_b < best_b) {
        return true;
    }
    return false;
}

__device__ static int krb_accel_branch_lit_eval(const signed char *assign, int lit) {
    int v = (lit < 0) ? -lit : lit;
    signed char a = assign[v];
    if (a < 0) {
        return -1;
    }
    if (lit > 0) {
        return (a > 0) ? 1 : 0;
    }
    return (a == 0) ? 1 : 0;
}

__global__ static void krb_accel_tableau_pivot_kernel(double *dst,
                                                      const double *src,
                                                      int rows,
                                                      int cols,
                                                      int pivot_row,
                                                      int pivot_col,
                                                      double inv_pivot) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = rows * cols;
    if (idx >= total) {
        return;
    }

    {
        int row = idx / cols;
        int col = idx - row * cols;

        if (row == pivot_row && col == pivot_col) {
            dst[idx] = inv_pivot;
            return;
        }
        if (row == pivot_row) {
            dst[idx] = src[idx] * inv_pivot;
            return;
        }
        if (col == pivot_col) {
            dst[idx] = -src[idx] * inv_pivot;
            return;
        }

        dst[idx] = src[idx]
                 - src[pivot_row * cols + col]
                 * src[row * cols + pivot_col]
                 * inv_pivot;
    }
}

__global__ static void krb_accel_entering_col_kernel(const double *table,
                                                     const int *n_index,
                                                     int row_index,
                                                     int cols,
                                                     int total_cols,
                                                     int phase,
                                                     double eps,
                                                     int *out_idx,
                                                     double *out_metric) {
    __shared__ int s_idx[KRB_ACCEL_CUDA_THREADS];
    __shared__ int s_n[KRB_ACCEL_CUDA_THREADS];
    __shared__ double s_val[KRB_ACCEL_CUDA_THREADS];
    int tid = (int)threadIdx.x;
    int best_col = -1;
    int best_n = 0;
    double best_val = 0.0;
    const double *row = table + (size_t)row_index * (size_t)cols;
    int j;

    for (j = (int)(blockIdx.x * blockDim.x + threadIdx.x); j < total_cols; j += (int)(blockDim.x * gridDim.x)) {
        int nval;
        double val;
        if (phase == 2 && n_index[j] == -1) {
            continue;
        }
        nval = n_index[j];
        val = row[j];
        if (krb_accel_entering_better(val, nval, j, best_val, best_n, best_col, eps)) {
            best_col = j;
            best_n = nval;
            best_val = val;
        }
    }

    s_idx[tid] = best_col;
    s_n[tid] = best_n;
    s_val[tid] = best_val;
    __syncthreads();

    for (j = KRB_ACCEL_CUDA_THREADS / 2; j > 0; j >>= 1) {
        if (tid < j) {
            int cand_col = s_idx[tid + j];
            int cand_n = s_n[tid + j];
            double cand_val = s_val[tid + j];
            if (krb_accel_entering_better(cand_val, cand_n, cand_col, s_val[tid], s_n[tid], s_idx[tid], eps)) {
                s_idx[tid] = cand_col;
                s_n[tid] = cand_n;
                s_val[tid] = cand_val;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_idx[blockIdx.x] = s_idx[0];
        out_metric[blockIdx.x] = s_val[0];
    }
}

__global__ static void krb_accel_leaving_row_kernel(const double *table,
                                                    const int *b_index,
                                                    int rows,
                                                    int cols,
                                                    int pivot_col,
                                                    int rhs_col,
                                                    double eps,
                                                    int *out_idx,
                                                    double *out_metric) {
    __shared__ int s_idx[KRB_ACCEL_CUDA_THREADS];
    __shared__ int s_b[KRB_ACCEL_CUDA_THREADS];
    __shared__ double s_ratio[KRB_ACCEL_CUDA_THREADS];
    int tid = (int)threadIdx.x;
    int best_row = -1;
    int best_b = 0;
    double best_ratio = 0.0;
    int i;

    for (i = (int)(blockIdx.x * blockDim.x + threadIdx.x); i < rows; i += (int)(blockDim.x * gridDim.x)) {
        const double *row = table + (size_t)i * (size_t)cols;
        double denom = row[pivot_col];
        if (denom > eps) {
            double ratio = row[rhs_col] / denom;
            int bval = b_index[i];
            if (krb_accel_leaving_better(ratio, bval, i, best_ratio, best_b, best_row, eps)) {
                best_row = i;
                best_b = bval;
                best_ratio = ratio;
            }
        }
    }

    s_idx[tid] = best_row;
    s_b[tid] = best_b;
    s_ratio[tid] = best_ratio;
    __syncthreads();

    for (i = KRB_ACCEL_CUDA_THREADS / 2; i > 0; i >>= 1) {
        if (tid < i) {
            int cand_row = s_idx[tid + i];
            int cand_b = s_b[tid + i];
            double cand_ratio = s_ratio[tid + i];
            if (krb_accel_leaving_better(cand_ratio, cand_b, cand_row, s_ratio[tid], s_b[tid], s_idx[tid], eps)) {
                s_idx[tid] = cand_row;
                s_b[tid] = cand_b;
                s_ratio[tid] = cand_ratio;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_idx[blockIdx.x] = s_idx[0];
        out_metric[blockIdx.x] = s_ratio[0];
    }
}

__global__ static void krb_accel_branch_score_kernel(const int *clause_offsets,
                                                     const int *clause_lits,
                                                     int nclauses,
                                                     const signed char *assign,
                                                     const unsigned char *active_project,
                                                     int project_only,
                                                     int *score_tmp,
                                                     int *score_pos,
                                                     int *score_neg) {
    int ci = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(blockDim.x * gridDim.x);
    while (ci < nclauses) {
        int begin = clause_offsets[ci];
        int end = clause_offsets[ci + 1];
        int sat = 0;
        int open = 0;
        int j;

        for (j = begin; j < end; ++j) {
            int ev = krb_accel_branch_lit_eval(assign, clause_lits[j]);
            if (ev > 0) {
                sat = 1;
                break;
            }
            if (ev < 0) {
                ++open;
            }
        }

        if (!sat) {
            int w = (open <= 2) ? 6 : ((open == 3) ? 3 : 1);
            for (j = begin; j < end; ++j) {
                int lit = clause_lits[j];
                int v = (lit < 0) ? -lit : lit;
                if (assign[v] >= 0) {
                    continue;
                }
                if (project_only && active_project != NULL && !active_project[v]) {
                    continue;
                }
                atomicAdd(&score_tmp[v], w);
                if (lit > 0) {
                    atomicAdd(&score_pos[v], w);
                } else {
                    atomicAdd(&score_neg[v], w);
                }
            }
        }

        ci += stride;
    }
}

int krb_accel_cuda_compiled(void) {
    return 1;
}

int krb_accel_cuda_runtime_available(void) {
    int count = 0;
    cudaError_t st = cudaGetDeviceCount(&count);
    return (st == cudaSuccess && count > 0) ? 1 : 0;
}

int krb_accel_cuda_dense_lp_available(void) {
    return 1;
}

bool krb_accel_cuda_select_device(int device, char *err, size_t errsz) {
    int count = 0;
    cudaError_t st;

    if (device < 0) {
        if (err != NULL && errsz > 0U) {
            err[0] = '\0';
        }
        return true;
    }

    st = cudaGetDeviceCount(&count);
    if (st != cudaSuccess) {
        krb_accel_cuda_write_error(st, err, errsz, "failed to query CUDA devices");
        return false;
    }
    if (device >= count) {
        if (err != NULL && errsz > 0U) {
            snprintf(err, errsz, "requested CUDA device %d but only %d device(s) are visible", device, count);
        }
        return false;
    }

    st = cudaSetDevice(device);
    if (st != cudaSuccess) {
        krb_accel_cuda_write_error(st, err, errsz, "failed to select CUDA device");
        return false;
    }

    if (err != NULL && errsz > 0U) {
        err[0] = '\0';
    }
    return true;
}

bool krb_accel_cuda_managed_alloc_bytes(size_t bytes, void **out, char *err, size_t errsz) {
    cudaError_t st;
    if (out == NULL) {
        krb_accel_cuda_write_error(cudaErrorInvalidValue, err, errsz, "invalid managed allocation output");
        return false;
    }
    *out = NULL;
    if (bytes == 0U) {
        return true;
    }
    st = cudaMallocManaged(out, bytes, cudaMemAttachGlobal);
    if (st != cudaSuccess) {
        krb_accel_cuda_write_error(st, err, errsz, "cudaMallocManaged failed");
        *out = NULL;
        return false;
    }
    if (err != NULL && errsz > 0U) {
        err[0] = '\0';
    }
    return true;
}

bool krb_accel_cuda_managed_alloc_doubles(size_t count, double **out, char *err, size_t errsz) {
    cudaError_t st;
    if (out == NULL) {
        krb_accel_cuda_write_error(cudaErrorInvalidValue, err, errsz, "invalid managed allocation output");
        return false;
    }
    *out = NULL;
    if (count == 0U) {
        return true;
    }
    st = cudaMallocManaged((void **)out, count * sizeof(double), cudaMemAttachGlobal);
    if (st != cudaSuccess) {
        krb_accel_cuda_write_error(st, err, errsz, "cudaMallocManaged failed");
        *out = NULL;
        return false;
    }
    if (err != NULL && errsz > 0U) {
        err[0] = '\0';
    }
    return true;
}

bool krb_accel_cuda_managed_alloc_ints(size_t count, int **out, char *err, size_t errsz) {
    cudaError_t st;
    if (out == NULL) {
        krb_accel_cuda_write_error(cudaErrorInvalidValue, err, errsz, "invalid managed allocation output");
        return false;
    }
    *out = NULL;
    if (count == 0U) {
        return true;
    }
    st = cudaMallocManaged((void **)out, count * sizeof(int), cudaMemAttachGlobal);
    if (st != cudaSuccess) {
        krb_accel_cuda_write_error(st, err, errsz, "cudaMallocManaged failed");
        *out = NULL;
        return false;
    }
    if (err != NULL && errsz > 0U) {
        err[0] = '\0';
    }
    return true;
}

void krb_accel_cuda_managed_free(void *ptr) {
    if (ptr != NULL) {
        (void)cudaFree(ptr);
    }
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
    cudaError_t st;
    int blocks;
    size_t scores_bytes;

    if (clause_offsets == NULL || clause_lits == NULL || assign == NULL ||
        score_tmp == NULL || score_pos == NULL || score_neg == NULL ||
        nclauses < 0 || nvars < 0) {
        if (err != NULL && errsz > 0U) {
            snprintf(err, errsz, "invalid branch-scoring buffers");
        }
        return false;
    }
    if (project_only && active_project == NULL) {
        if (err != NULL && errsz > 0U) {
            snprintf(err, errsz, "project-only branch scoring requires an active-project mask");
        }
        return false;
    }

    scores_bytes = ((size_t)nvars + 1U) * sizeof(int);
    st = cudaMemset(score_tmp, 0, scores_bytes);
    if (st == cudaSuccess) {
        st = cudaMemset(score_pos, 0, scores_bytes);
    }
    if (st == cudaSuccess) {
        st = cudaMemset(score_neg, 0, scores_bytes);
    }
    if (st != cudaSuccess) {
        krb_accel_cuda_write_error(st, err, errsz, "branch-score buffer reset failed");
        return false;
    }

    if (nclauses == 0) {
        if (err != NULL && errsz > 0U) {
            err[0] = '\0';
        }
        return true;
    }

    blocks = (nclauses + KRB_ACCEL_CUDA_THREADS - 1) / KRB_ACCEL_CUDA_THREADS;
    if (blocks < 1) {
        blocks = 1;
    }

    krb_accel_branch_score_kernel<<<blocks, KRB_ACCEL_CUDA_THREADS>>>(clause_offsets,
                                                                      clause_lits,
                                                                      nclauses,
                                                                      assign,
                                                                      active_project,
                                                                      project_only,
                                                                      score_tmp,
                                                                      score_pos,
                                                                      score_neg);
    st = cudaGetLastError();
    if (st != cudaSuccess) {
        krb_accel_cuda_write_error(st, err, errsz, "branch-score kernel launch failed");
        return false;
    }
    st = cudaDeviceSynchronize();
    if (st != cudaSuccess) {
        krb_accel_cuda_write_error(st, err, errsz, "branch-score kernel failed");
        return false;
    }

    if (err != NULL && errsz > 0U) {
        err[0] = '\0';
    }
    return true;
}

bool krb_accel_cuda_tableau_pivot(double *dst,
                                  const double *src,
                                  int rows,
                                  int cols,
                                  int pivot_row,
                                  int pivot_col,
                                  char *err,
                                  size_t errsz) {
    cudaError_t st;
    double pivot;
    double inv_pivot;
    int total;
    int blocks;

    if (dst == NULL || src == NULL || rows <= 0 || cols <= 0) {
        if (err != NULL && errsz > 0U) {
            snprintf(err, errsz, "invalid tableau buffers");
        }
        return false;
    }
    if (pivot_row < 0 || pivot_row >= rows || pivot_col < 0 || pivot_col >= cols) {
        if (err != NULL && errsz > 0U) {
            snprintf(err, errsz, "invalid pivot coordinates");
        }
        return false;
    }

    pivot = src[pivot_row * cols + pivot_col];
    if (pivot == 0.0) {
        if (err != NULL && errsz > 0U) {
            snprintf(err, errsz, "zero pivot");
        }
        return false;
    }
    inv_pivot = 1.0 / pivot;
    total = rows * cols;
    blocks = (total + KRB_ACCEL_CUDA_THREADS - 1) / KRB_ACCEL_CUDA_THREADS;

    krb_accel_tableau_pivot_kernel<<<blocks, KRB_ACCEL_CUDA_THREADS>>>(dst,
                                                                       src,
                                                                       rows,
                                                                       cols,
                                                                       pivot_row,
                                                                       pivot_col,
                                                                       inv_pivot);
    st = cudaGetLastError();
    if (st != cudaSuccess) {
        krb_accel_cuda_write_error(st, err, errsz, "tableau pivot kernel launch failed");
        return false;
    }

    st = cudaDeviceSynchronize();
    if (st != cudaSuccess) {
        krb_accel_cuda_write_error(st, err, errsz, "tableau pivot kernel failed");
        return false;
    }

    if (err != NULL && errsz > 0U) {
        err[0] = '\0';
    }
    return true;
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
    cudaError_t st;
    int blocks;
    int b;
    int best_col = -1;
    double best_val = 0.0;

    if (out_col != NULL) {
        *out_col = -1;
    }
    if (table == NULL || n_index == NULL || scratch_idx == NULL || scratch_metric == NULL || out_col == NULL) {
        if (err != NULL && errsz > 0U) {
            snprintf(err, errsz, "invalid entering-column buffers");
        }
        return false;
    }
    if (row_index < 0 || cols <= 0 || total_cols <= 0 || total_cols > cols || scratch_cap <= 0) {
        if (err != NULL && errsz > 0U) {
            snprintf(err, errsz, "invalid entering-column dimensions");
        }
        return false;
    }

    blocks = (total_cols + KRB_ACCEL_CUDA_THREADS - 1) / KRB_ACCEL_CUDA_THREADS;
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > scratch_cap) {
        if (err != NULL && errsz > 0U) {
            snprintf(err, errsz, "insufficient entering-column scratch capacity");
        }
        return false;
    }

    krb_accel_entering_col_kernel<<<blocks, KRB_ACCEL_CUDA_THREADS>>>(table,
                                                                      n_index,
                                                                      row_index,
                                                                      cols,
                                                                      total_cols,
                                                                      phase,
                                                                      eps,
                                                                      scratch_idx,
                                                                      scratch_metric);
    st = cudaGetLastError();
    if (st != cudaSuccess) {
        krb_accel_cuda_write_error(st, err, errsz, "entering-column kernel launch failed");
        return false;
    }
    st = cudaDeviceSynchronize();
    if (st != cudaSuccess) {
        krb_accel_cuda_write_error(st, err, errsz, "entering-column kernel failed");
        return false;
    }

    for (b = 0; b < blocks; ++b) {
        int cand_col = scratch_idx[b];
        if (cand_col >= 0) {
            double cand_val = scratch_metric[b];
            if (best_col < 0 ||
                cand_val < best_val - eps ||
                (fabs(cand_val - best_val) <= eps && n_index[cand_col] < n_index[best_col])) {
                best_col = cand_col;
                best_val = cand_val;
            }
        }
    }

    *out_col = best_col;
    if (err != NULL && errsz > 0U) {
        err[0] = '\0';
    }
    return true;
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
    cudaError_t st;
    int blocks;
    int b;
    int best_row = -1;
    double best_ratio = 0.0;

    if (out_row != NULL) {
        *out_row = -1;
    }
    if (table == NULL || b_index == NULL || scratch_idx == NULL || scratch_metric == NULL || out_row == NULL) {
        if (err != NULL && errsz > 0U) {
            snprintf(err, errsz, "invalid leaving-row buffers");
        }
        return false;
    }
    if (rows <= 0 || cols <= 0 || pivot_col < 0 || pivot_col >= cols || rhs_col < 0 || rhs_col >= cols || scratch_cap <= 0) {
        if (err != NULL && errsz > 0U) {
            snprintf(err, errsz, "invalid leaving-row dimensions");
        }
        return false;
    }

    blocks = (rows + KRB_ACCEL_CUDA_THREADS - 1) / KRB_ACCEL_CUDA_THREADS;
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > scratch_cap) {
        if (err != NULL && errsz > 0U) {
            snprintf(err, errsz, "insufficient leaving-row scratch capacity");
        }
        return false;
    }

    krb_accel_leaving_row_kernel<<<blocks, KRB_ACCEL_CUDA_THREADS>>>(table,
                                                                     b_index,
                                                                     rows,
                                                                     cols,
                                                                     pivot_col,
                                                                     rhs_col,
                                                                     eps,
                                                                     scratch_idx,
                                                                     scratch_metric);
    st = cudaGetLastError();
    if (st != cudaSuccess) {
        krb_accel_cuda_write_error(st, err, errsz, "leaving-row kernel launch failed");
        return false;
    }
    st = cudaDeviceSynchronize();
    if (st != cudaSuccess) {
        krb_accel_cuda_write_error(st, err, errsz, "leaving-row kernel failed");
        return false;
    }

    for (b = 0; b < blocks; ++b) {
        int cand_row = scratch_idx[b];
        if (cand_row >= 0) {
            double cand_ratio = scratch_metric[b];
            if (best_row < 0 ||
                cand_ratio < best_ratio - eps ||
                (fabs(cand_ratio - best_ratio) <= eps && b_index[cand_row] < b_index[best_row])) {
                best_row = cand_row;
                best_ratio = cand_ratio;
            }
        }
    }

    *out_row = best_row;
    if (err != NULL && errsz > 0U) {
        err[0] = '\0';
    }
    return true;
}
