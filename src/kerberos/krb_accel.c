#include "krb_accel.h"

#include <ctype.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static void krb_accel_set_error(char *err, size_t errsz, const char *fmt, ...) {
    va_list ap;
    if (err == NULL || errsz == 0U) {
        return;
    }
    va_start(ap, fmt);
    vsnprintf(err, errsz, fmt, ap);
    va_end(ap);
}

static bool krb_accel_streq_ci(const char *a, const char *b) {
    while (*a != '\0' && *b != '\0') {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

void krb_accel_config_defaults(KrbAccelConfig *cfg) {
    if (cfg == NULL) {
        return;
    }
    cfg->mode = KRB_ACCEL_MODE_AUTO;
    cfg->cuda_device = -1;
    cfg->cuda_min_cells = KRB_ACCEL_DEFAULT_CUDA_MIN_CELLS;
}

bool krb_accel_parse_mode(const char *text, KrbAccelMode *out) {
    if (text == NULL || out == NULL) {
        return false;
    }
    if (krb_accel_streq_ci(text, "auto")) {
        *out = KRB_ACCEL_MODE_AUTO;
        return true;
    }
    if (krb_accel_streq_ci(text, "off") || krb_accel_streq_ci(text, "cpu")) {
        *out = KRB_ACCEL_MODE_OFF;
        return true;
    }
    if (krb_accel_streq_ci(text, "on") || krb_accel_streq_ci(text, "cuda")) {
        *out = KRB_ACCEL_MODE_ON;
        return true;
    }
    return false;
}

const char *krb_accel_mode_name(KrbAccelMode mode) {
    switch (mode) {
    case KRB_ACCEL_MODE_AUTO:
        return "auto";
    case KRB_ACCEL_MODE_OFF:
        return "off";
    case KRB_ACCEL_MODE_ON:
        return "on";
    default:
        return "unknown";
    }
}

bool krb_accel_choose_dense_lp(const KrbAccelConfig *cfg,
                               const char *solver_name,
                               int rows,
                               int cols,
                               KrbAccelDecision *out,
                               char *err,
                               size_t errsz) {
    KrbAccelConfig local_cfg;
    const char *name = (solver_name != NULL) ? solver_name : "solver";
    size_t cells = 0U;

    if (out == NULL) {
        krb_accel_set_error(err, errsz, "%s: missing acceleration decision output", name);
        return false;
    }

    if (cfg == NULL) {
        krb_accel_config_defaults(&local_cfg);
        cfg = &local_cfg;
    }

    memset(out, 0, sizeof(*out));
    out->path = KRB_ACCEL_PATH_CPU;
    out->active_device = cfg->cuda_device;
    out->reason = "disabled";

    if (rows > 0 && cols > 0) {
        if ((size_t)rows > (SIZE_MAX / (size_t)cols)) {
            cells = SIZE_MAX;
        } else {
            cells = (size_t)rows * (size_t)cols;
        }
    }
    out->dense_cells = cells;

    if (cfg->mode == KRB_ACCEL_MODE_OFF) {
        out->reason = "disabled";
        return true;
    }

    if (rows <= 0 || cols <= 0) {
        out->reason = "empty lp";
        return true;
    }

    if (cfg->mode == KRB_ACCEL_MODE_AUTO && cells < cfg->cuda_min_cells) {
        out->reason = "below threshold";
        return true;
    }

    out->cuda_compiled = krb_accel_cuda_compiled();
    if (!out->cuda_compiled) {
        if (cfg->mode == KRB_ACCEL_MODE_ON) {
            krb_accel_set_error(err, errsz,
                                "%s: CUDA requested but this build was compiled without CUDA support",
                                name);
            return false;
        }
        out->reason = "not built";
        return true;
    }

    out->cuda_runtime = krb_accel_cuda_runtime_available();
    if (!out->cuda_runtime) {
        if (cfg->mode == KRB_ACCEL_MODE_ON) {
            krb_accel_set_error(err, errsz,
                                "%s: CUDA requested but no compatible CUDA runtime/device is available",
                                name);
            return false;
        }
        out->reason = "runtime unavailable";
        return true;
    }

    if (!krb_accel_cuda_dense_lp_available()) {
        if (cfg->mode == KRB_ACCEL_MODE_ON) {
            krb_accel_set_error(err, errsz,
                                "%s: CUDA build detected but dense LP kernels are not implemented yet",
                                name);
            return false;
        }
        out->reason = "dense kernels unavailable";
        return true;
    }

    if (!krb_accel_cuda_select_device(cfg->cuda_device, err, errsz)) {
        if (cfg->mode == KRB_ACCEL_MODE_ON || cfg->cuda_device >= 0) {
            return false;
        }
        out->reason = "device unavailable";
        return true;
    }

    out->path = KRB_ACCEL_PATH_CUDA;
    out->reason = "selected";
    return true;
}

void krb_accel_log(FILE *stream, const char *solver_name, const KrbAccelDecision *decision) {
    const char *name = (solver_name != NULL) ? solver_name : "solver";
    const char *path_name;

    if (stream == NULL || decision == NULL) {
        return;
    }

    path_name = (decision->path == KRB_ACCEL_PATH_CUDA) ? "cuda" : "cpu";
    fprintf(stream,
            "c %s accel=%s reason=%s cells=%zu cuda_compiled=%d cuda_runtime=%d",
            name,
            path_name,
            (decision->reason != NULL) ? decision->reason : "n/a",
            decision->dense_cells,
            decision->cuda_compiled,
            decision->cuda_runtime);
    if (decision->path == KRB_ACCEL_PATH_CUDA && decision->active_device >= 0) {
        fprintf(stream, " device=%d", decision->active_device);
    }
    fputc('\n', stream);
}
