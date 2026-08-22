#include "krb_parallel.h"

#include <stdlib.h>
#include <string.h>
#include <threads.h>

typedef struct {
    KrbParallelWorkerFn worker;
    void *ctx;
    int worker_id;
    int rc;
} KrbParallelThreadLaunch;

static void krb_parallel_threads_set_error(char *err, size_t errsz, const char *msg) {
    if (err != NULL && errsz > 0U) {
        size_t n = strlen(msg);
        if (n >= errsz) {
            n = errsz - 1U;
        }
        memcpy(err, msg, n);
        err[n] = '\0';
    }
}

static int krb_parallel_thread_entry(void *arg) {
    KrbParallelThreadLaunch *launch = (KrbParallelThreadLaunch *)arg;
    launch->rc = launch->worker(launch->ctx, launch->worker_id);
    return launch->rc;
}

bool krb_parallel_run_threads_impl(int jobs,
                                   KrbParallelWorkerFn worker,
                                   void *ctx,
                                   char *err,
                                   size_t errsz) {
    KrbParallelThreadLaunch *launches = NULL;
    thrd_t *threads = NULL;
    int i;
    int ok = 1;

    if (jobs <= 1) {
        return worker(ctx, 0) == 0;
    }

    launches = (KrbParallelThreadLaunch *)calloc((size_t)jobs, sizeof(*launches));
    threads = (thrd_t *)calloc((size_t)jobs, sizeof(*threads));
    if (launches == NULL || threads == NULL) {
        free(launches);
        free(threads);
        krb_parallel_threads_set_error(err, errsz, "out of memory creating thread launch table");
        return false;
    }

    for (i = 0; i < jobs; ++i) {
        launches[i].worker = worker;
        launches[i].ctx = ctx;
        launches[i].worker_id = i;
        launches[i].rc = 0;
        if (thrd_create(&threads[i], krb_parallel_thread_entry, &launches[i]) != thrd_success) {
            ok = 0;
            krb_parallel_threads_set_error(err, errsz, "failed to create worker thread");
            jobs = i;
            break;
        }
    }

    for (i = 0; i < jobs; ++i) {
        int thread_rc = 0;
        thrd_join(threads[i], &thread_rc);
        if (thread_rc != 0) {
            ok = 0;
            if (err != NULL && errsz > 0U && err[0] == '\0') {
                krb_parallel_threads_set_error(err, errsz, "worker thread reported failure");
            }
        }
    }

    free(launches);
    free(threads);
    return ok != 0;
}
