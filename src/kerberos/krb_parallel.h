#ifndef KRB_PARALLEL_H
#define KRB_PARALLEL_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KRB_PARALLEL_DEFAULT_JOBS 1
#define KRB_PARALLEL_DEFAULT_SPLIT_DEPTH 1
#define KRB_PARALLEL_DEFAULT_PORTFOLIO 1
#define KRB_PARALLEL_DEFAULT_SYNC_MS 100

typedef enum {
    KRB_PARALLEL_MODE_AUTO = 0,
    KRB_PARALLEL_MODE_OFF = 1,
    KRB_PARALLEL_MODE_THREADS = 2,
    KRB_PARALLEL_MODE_MPI = 3,
    KRB_PARALLEL_MODE_HYBRID = 4
} KrbParallelMode;

typedef struct {
    KrbParallelMode mode;
    int jobs;
    int split_depth;
    int portfolio;
    int sync_ms;
} KrbParallelConfig;

typedef struct {
    KrbParallelMode requested_mode;
    KrbParallelMode resolved_mode;
    int compiled_threads;
    int compiled_mpi;
    int jobs;
    int world_rank;
    int world_size;
    int local_rank;
    int local_size;
    const char *reason;
} KrbParallelRuntime;

typedef int (*KrbParallelWorkerFn)(void *ctx, int worker_id);

void krb_parallel_config_defaults(KrbParallelConfig *cfg);
bool krb_parallel_parse_mode(const char *text, KrbParallelMode *out);
const char *krb_parallel_mode_name(KrbParallelMode mode);
int krb_parallel_effective_jobs(const KrbParallelConfig *cfg);
bool krb_parallel_mode_uses_threads(KrbParallelMode mode);
bool krb_parallel_mode_uses_mpi(KrbParallelMode mode);

void krb_parallel_runtime_defaults(KrbParallelRuntime *rt);
bool krb_parallel_runtime_resolve(const KrbParallelConfig *cfg,
                                  KrbParallelRuntime *rt,
                                  char *err,
                                  size_t errsz);
bool krb_parallel_mpi_init(KrbParallelRuntime *rt, char *err, size_t errsz);
void krb_parallel_mpi_finalize(void);
bool krb_parallel_run_threads(int jobs,
                              KrbParallelWorkerFn worker,
                              void *ctx,
                              char *err,
                              size_t errsz);

#ifdef __cplusplus
}
#endif

#endif
