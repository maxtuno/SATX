#include "krb_parallel.h"

#include <mpi.h>
#include <stdlib.h>
#include <string.h>

static int krb_parallel_mpi_owns_runtime = 0;

static void krb_parallel_mpi_set_error(char *err, size_t errsz, const char *msg) {
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
    char *end = NULL;
    long long value;
    if (text == NULL || text[0] == '\0') {
        return fallback;
    }
    value = strtoll(text, &end, 10);
    if (end == text || *end != '\0' || value < 0LL || value > 2147483647LL) {
        return fallback;
    }
    return (int)value;
}

static int krb_parallel_env_local_only(void) {
    return krb_parallel_parse_env_int("MSMPI_LOCAL_ONLY", 0) != 0;
}

static int krb_parallel_env_local_rank(int fallback) {
    const char *vars[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "PMI_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "MV2_COMM_WORLD_LOCAL_RANK"
    };
    size_t i;
    for (i = 0; i < sizeof(vars) / sizeof(vars[0]); ++i) {
        int value = krb_parallel_parse_env_int(vars[i], -1);
        if (value >= 0) {
            return value;
        }
    }

    if (krb_parallel_env_local_only()) {
        return krb_parallel_parse_env_int("PMI_RANK", fallback);
    }

    return fallback;
}

static int krb_parallel_env_local_size(int fallback) {
    const char *vars[] = {
        "OMPI_COMM_WORLD_LOCAL_SIZE",
        "PMI_LOCAL_SIZE",
        "MPI_LOCALNRANKS",
        "MV2_COMM_WORLD_LOCAL_SIZE"
    };
    size_t i;
    for (i = 0; i < sizeof(vars) / sizeof(vars[0]); ++i) {
        int value = krb_parallel_parse_env_int(vars[i], -1);
        if (value > 0) {
            return value;
        }
    }

    if (krb_parallel_env_local_only()) {
        return krb_parallel_parse_env_int("PMI_SIZE", fallback);
    }

    return fallback;
}

bool krb_parallel_mpi_init_impl(KrbParallelRuntime *rt, char *err, size_t errsz) {
    int initialized = 0;
    int finalized = 0;
    int provided = MPI_THREAD_SINGLE;
    int rc;

    if (rt == NULL) {
        krb_parallel_mpi_set_error(err, errsz, "invalid null runtime for MPI initialization");
        return false;
    }

    MPI_Initialized(&initialized);
    if (!initialized) {
        rc = MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
        if (rc != MPI_SUCCESS) {
            krb_parallel_mpi_set_error(err, errsz, "MPI_Init_thread failed");
            return false;
        }
        krb_parallel_mpi_owns_runtime = 1;
    }

    MPI_Finalized(&finalized);
    if (finalized) {
        krb_parallel_mpi_set_error(err, errsz, "MPI already finalized");
        return false;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rt->world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rt->world_size);
#if defined(MPI_COMM_TYPE_SHARED)
    {
        MPI_Comm local_comm = MPI_COMM_NULL;
        if (MPI_Comm_split_type(MPI_COMM_WORLD,
                                MPI_COMM_TYPE_SHARED,
                                0,
                                MPI_INFO_NULL,
                                &local_comm) == MPI_SUCCESS &&
            local_comm != MPI_COMM_NULL) {
            MPI_Comm_rank(local_comm, &rt->local_rank);
            MPI_Comm_size(local_comm, &rt->local_size);
            MPI_Comm_free(&local_comm);
        } else {
            rt->local_rank = krb_parallel_env_local_rank(rt->world_rank);
            rt->local_size = krb_parallel_env_local_size((rt->world_size > 0) ? rt->world_size : 1);
        }
    }
#else
    rt->local_rank = krb_parallel_env_local_rank(rt->world_rank);
    rt->local_size = krb_parallel_env_local_size((rt->world_size > 0) ? rt->world_size : 1);
#endif
    rt->reason = "mpi-runtime";
    return true;
}

void krb_parallel_mpi_finalize_impl(void) {
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (initialized && !finalized && krb_parallel_mpi_owns_runtime) {
        MPI_Finalize();
        krb_parallel_mpi_owns_runtime = 0;
    }
}
