#include "krb_parallel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(SATX_HAVE_THREADS)
bool krb_parallel_run_threads_impl(int jobs,
                                   KrbParallelWorkerFn worker,
                                   void *ctx,
                                   char *err,
                                   size_t errsz);
#endif

#if defined(SATX_HAVE_MPI)
bool krb_parallel_mpi_init_impl(KrbParallelRuntime *rt, char *err, size_t errsz);
void krb_parallel_mpi_finalize_impl(void);
#endif

static void krb_parallel_set_error(char *err, size_t errsz, const char *msg) {
    if (err != NULL && errsz > 0U) {
        size_t n = strlen(msg);
        if (n >= errsz) {
            n = errsz - 1U;
        }
        memcpy(err, msg, n);
        err[n] = '\0';
    }
}

static int krb_parallel_parse_env_int(const char *name, int fallback) {
    const char *text = getenv(name);
    long long value;
    char *end = NULL;
    if (text == NULL || text[0] == '\0') {
        return fallback;
    }
    value = strtoll(text, &end, 10);
    if (end == text || *end != '\0' || value < 0LL || value > 2147483647LL) {
        return fallback;
    }
    return (int)value;
}

static int krb_parallel_detect_world_size_hint(void) {
    int size = krb_parallel_parse_env_int("OMPI_COMM_WORLD_SIZE", 0);
    if (size > 0) return size;
    size = krb_parallel_parse_env_int("PMI_SIZE", 0);
    if (size > 0) return size;
    size = krb_parallel_parse_env_int("MV2_COMM_WORLD_SIZE", 0);
    if (size > 0) return size;
    size = krb_parallel_parse_env_int("WORLD_SIZE", 0);
    if (size > 0) return size;
    return 0;
}

static const char *krb_parallel_reason_string(KrbParallelMode mode) {
    if (mode == KRB_PARALLEL_MODE_THREADS) return "threads";
    if (mode == KRB_PARALLEL_MODE_MPI) return "mpi";
    if (mode == KRB_PARALLEL_MODE_HYBRID) return "hybrid";
    return "serial";
}

void krb_parallel_config_defaults(KrbParallelConfig *cfg) {
    if (cfg == NULL) {
        return;
    }
    cfg->mode = KRB_PARALLEL_MODE_AUTO;
    cfg->jobs = KRB_PARALLEL_DEFAULT_JOBS;
    cfg->split_depth = KRB_PARALLEL_DEFAULT_SPLIT_DEPTH;
    cfg->portfolio = KRB_PARALLEL_DEFAULT_PORTFOLIO;
    cfg->sync_ms = KRB_PARALLEL_DEFAULT_SYNC_MS;
}

bool krb_parallel_parse_mode(const char *text, KrbParallelMode *out) {
    if (text == NULL || out == NULL) {
        return false;
    }
    if (strcmp(text, "auto") == 0) {
        *out = KRB_PARALLEL_MODE_AUTO;
        return true;
    }
    if (strcmp(text, "off") == 0) {
        *out = KRB_PARALLEL_MODE_OFF;
        return true;
    }
    if (strcmp(text, "threads") == 0) {
        *out = KRB_PARALLEL_MODE_THREADS;
        return true;
    }
    if (strcmp(text, "mpi") == 0) {
        *out = KRB_PARALLEL_MODE_MPI;
        return true;
    }
    if (strcmp(text, "hybrid") == 0) {
        *out = KRB_PARALLEL_MODE_HYBRID;
        return true;
    }
    return false;
}

const char *krb_parallel_mode_name(KrbParallelMode mode) {
    if (mode == KRB_PARALLEL_MODE_AUTO) return "auto";
    if (mode == KRB_PARALLEL_MODE_OFF) return "off";
    if (mode == KRB_PARALLEL_MODE_THREADS) return "threads";
    if (mode == KRB_PARALLEL_MODE_MPI) return "mpi";
    if (mode == KRB_PARALLEL_MODE_HYBRID) return "hybrid";
    return "unknown";
}

int krb_parallel_effective_jobs(const KrbParallelConfig *cfg) {
    if (cfg == NULL || cfg->jobs <= 0) {
        return KRB_PARALLEL_DEFAULT_JOBS;
    }
    return cfg->jobs;
}

bool krb_parallel_mode_uses_threads(KrbParallelMode mode) {
    return mode == KRB_PARALLEL_MODE_THREADS || mode == KRB_PARALLEL_MODE_HYBRID;
}

bool krb_parallel_mode_uses_mpi(KrbParallelMode mode) {
    return mode == KRB_PARALLEL_MODE_MPI || mode == KRB_PARALLEL_MODE_HYBRID;
}

void krb_parallel_runtime_defaults(KrbParallelRuntime *rt) {
    if (rt == NULL) {
        return;
    }
    memset(rt, 0, sizeof(*rt));
    rt->requested_mode = KRB_PARALLEL_MODE_AUTO;
    rt->resolved_mode = KRB_PARALLEL_MODE_OFF;
    rt->jobs = 1;
    rt->world_size = 1;
    rt->local_size = 1;
    rt->reason = "serial";
}

bool krb_parallel_runtime_resolve(const KrbParallelConfig *cfg,
                                  KrbParallelRuntime *rt,
                                  char *err,
                                  size_t errsz) {
    KrbParallelMode requested = KRB_PARALLEL_MODE_AUTO;
    int jobs = KRB_PARALLEL_DEFAULT_JOBS;
    int mpi_hint = 0;

    krb_parallel_runtime_defaults(rt);
    if (cfg != NULL) {
        requested = cfg->mode;
        jobs = krb_parallel_effective_jobs(cfg);
    }
    if (rt != NULL) {
        rt->requested_mode = requested;
#if defined(SATX_HAVE_THREADS)
        rt->compiled_threads = 1;
#endif
#if defined(SATX_HAVE_MPI)
        rt->compiled_mpi = 1;
#endif
        rt->jobs = jobs;
    }

    mpi_hint = krb_parallel_detect_world_size_hint();

    if (requested == KRB_PARALLEL_MODE_OFF) {
        if (rt != NULL) {
            rt->resolved_mode = KRB_PARALLEL_MODE_OFF;
            rt->reason = "disabled-by-cli";
        }
        return true;
    }

    if (requested == KRB_PARALLEL_MODE_THREADS) {
#if !defined(SATX_HAVE_THREADS)
        krb_parallel_set_error(err, errsz, "threads mode requested but this build does not provide threads.h support");
        return false;
#else
        if (jobs <= 1) {
            if (rt != NULL) {
                rt->resolved_mode = KRB_PARALLEL_MODE_OFF;
                rt->reason = "jobs=1";
            }
            return true;
        }
        if (rt != NULL) {
            rt->resolved_mode = KRB_PARALLEL_MODE_THREADS;
            rt->reason = krb_parallel_reason_string(rt->resolved_mode);
        }
        return true;
#endif
    }

    if (requested == KRB_PARALLEL_MODE_MPI) {
#if !defined(SATX_HAVE_MPI)
        krb_parallel_set_error(err, errsz, "mpi mode requested but this build does not provide MPI support");
        return false;
#else
        if (rt != NULL) {
            rt->resolved_mode = KRB_PARALLEL_MODE_MPI;
            rt->reason = krb_parallel_reason_string(rt->resolved_mode);
        }
        return true;
#endif
    }

    if (requested == KRB_PARALLEL_MODE_HYBRID) {
#if !defined(SATX_HAVE_MPI)
        krb_parallel_set_error(err, errsz, "hybrid mode requested but this build does not provide MPI support");
        return false;
#elif !defined(SATX_HAVE_THREADS)
        krb_parallel_set_error(err, errsz, "hybrid mode requested but this build does not provide threads.h support");
        return false;
#elif !defined(SATX_ENABLE_HYBRID_PARALLEL)
        krb_parallel_set_error(err, errsz, "hybrid mode requested but hybrid parallel support is disabled in this build");
        return false;
#else
        if (rt != NULL) {
            rt->resolved_mode = (jobs > 1) ? KRB_PARALLEL_MODE_HYBRID : KRB_PARALLEL_MODE_MPI;
            rt->reason = krb_parallel_reason_string(rt->resolved_mode);
        }
        return true;
#endif
    }

#if defined(SATX_HAVE_MPI)
    if (mpi_hint > 1) {
#if defined(SATX_HAVE_THREADS) && defined(SATX_ENABLE_HYBRID_PARALLEL)
        if (jobs > 1) {
            if (rt != NULL) {
                rt->resolved_mode = KRB_PARALLEL_MODE_HYBRID;
                rt->reason = krb_parallel_reason_string(rt->resolved_mode);
            }
            return true;
        }
#endif
        if (rt != NULL) {
            rt->resolved_mode = KRB_PARALLEL_MODE_MPI;
            rt->reason = krb_parallel_reason_string(rt->resolved_mode);
        }
        return true;
    }
#endif

#if defined(SATX_HAVE_THREADS)
    if (jobs > 1) {
        if (rt != NULL) {
            rt->resolved_mode = KRB_PARALLEL_MODE_THREADS;
            rt->reason = krb_parallel_reason_string(rt->resolved_mode);
        }
        return true;
    }
#endif

    if (rt != NULL) {
        rt->resolved_mode = KRB_PARALLEL_MODE_OFF;
        rt->reason = (jobs <= 1) ? "jobs=1" : "serial";
    }
    return true;
}

bool krb_parallel_mpi_init(KrbParallelRuntime *rt, char *err, size_t errsz) {
#if defined(SATX_HAVE_MPI)
    return krb_parallel_mpi_init_impl(rt, err, errsz);
#else
    (void)rt;
    if (err != NULL && errsz > 0U) {
        err[0] = '\0';
    }
    return true;
#endif
}

void krb_parallel_mpi_finalize(void) {
#if defined(SATX_HAVE_MPI)
    krb_parallel_mpi_finalize_impl();
#endif
}

bool krb_parallel_run_threads(int jobs,
                              KrbParallelWorkerFn worker,
                              void *ctx,
                              char *err,
                              size_t errsz) {
    if (worker == NULL) {
        krb_parallel_set_error(err, errsz, "invalid null worker callback");
        return false;
    }
    if (jobs <= 1) {
        if (worker(ctx, 0) != 0) {
            krb_parallel_set_error(err, errsz, "single-thread worker failed");
            return false;
        }
        return true;
    }
#if defined(SATX_HAVE_THREADS)
    return krb_parallel_run_threads_impl(jobs, worker, ctx, err, errsz);
#else
    krb_parallel_set_error(err, errsz, "threads requested but this build does not provide threads.h support");
    return false;
#endif
}
