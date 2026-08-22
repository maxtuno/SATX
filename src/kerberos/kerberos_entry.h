#ifndef KERBEROS_ENTRY_H
#define KERBEROS_ENTRY_H

#include <stddef.h>

#include "krb_parallel.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Stable C entry point for the standalone Kerberos dispatcher or FFI hosts. */
int kerberos_entry(int argc, char **argv);

typedef struct {
    size_t struct_size;
    const char *input_path;
    const char *manifest_out;
    const char *result_out;
    const char *replay_path;
    const char *mode_name;
    int show_help;
    int selftest;
    int audit_dispatch;
    int strict_options;
    KrbParallelConfig parallel;
    const char *const *solve_args;
    size_t solve_argc;
} KerberosBackendRequest;

void kerberos_backend_request_defaults(KerberosBackendRequest *request);
int kerberos_backend_request_run(const KerberosBackendRequest *request, char *err, size_t errsz);
const char *kerberos_backend_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
