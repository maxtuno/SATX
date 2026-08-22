/*
 * check_model.c — validador externo de modelos SAT (DIMACS).
 * Uso: check_model.exe <archivo.cnf> <línea-v>
 * Salida: "OK" (exit 0) o "MAL MODELO" (exit 1).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXL 4096

int main(int argc, char **argv) {
    FILE *fp;
    char line[MAXL];
    long nvars = 0;
    signed char *model = NULL;
    int header = 0;

    if (argc < 2) { fprintf(stderr, "uso: check_model <cnf> [linea-v]\n"); return 2; }
    fp = fopen(argv[1], "r");
    if (!fp) { fprintf(stderr, "no se pudo abrir %s\n", argv[1]); return 2; }

    /* primer pase: header */
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (line[0] == 'p') {
            sscanf(line, "p cnf %ld", &nvars);
            header = 1;
            break;
        }
    }
    if (!header || nvars < 1) { fprintf(stderr, "sin header valido\n"); return 2; }

    model = (signed char *)calloc((size_t)nvars + 1, 1);

    /* modelo desde los argumentos (línea v) o desde un fichero */
    for (int a = 2; a < argc; ++a) {
        const char *src = argv[a];
        FILE *vf = NULL;
        if (src[0] == '@') vf = fopen(src + 1, "r");
        if (vf) {
            char vbuf[MAXL];
            while (fgets(vbuf, sizeof(vbuf), vf) != NULL) {
                char *tok = strtok(vbuf, " \t\r\n");
                while (tok != NULL) {
                    if (tok[0] == 'v' && tok[1] == '\0') { tok = strtok(NULL, " \t\r\n"); continue; }
                    long l = atol(tok);
                    if (l != 0) {
                        long v = l > 0 ? l : -l;
                        if (v >= 1 && v <= nvars) model[v] = (signed char)(l > 0 ? 1 : -1);
                    }
                    tok = strtok(NULL, " \t\r\n");
                }
            }
            fclose(vf);
            continue;
        }
        char *tok = strtok((char *)src, " \t\r\n");
        while (tok != NULL) {
            if (tok[0] == 'v' && tok[1] == '\0') { tok = strtok(NULL, " \t\r\n"); continue; }
            long l = atol(tok);
            if (l != 0) {
                long v = l > 0 ? l : -l;
                if (v >= 1 && v <= nvars) model[v] = (signed char)(l > 0 ? 1 : -1);
            }
            tok = strtok(NULL, " \t\r\n");
        }
    }

    /* segundo pase: verificar cláusulas */
    rewind(fp);
    header = 0;
    int bad = 0;
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (line[0] == 'c' || line[0] == '\n' || line[0] == '\r') continue;
        if (line[0] == 'p') { header = 1; continue; }
        if (line[0] == 'v' || line[0] == 's') continue;
        if (!header) continue;
        char *tok = strtok(line, " \t\r\n");
        int sat = 0;
        while (tok != NULL) {
            long l = atol(tok);
            if (l == 0) break;
            long v = l > 0 ? l : -l;
            if (v >= 1 && v <= nvars && model[v] == (l > 0 ? 1 : -1)) sat = 1;
            tok = strtok(NULL, " \t\r\n");
        }
        if (!sat) {
            bad = 1;
            fprintf(stderr, "clausula falsificada: %s", line);
            break;
        }
    }
    free(model);
    fclose(fp);
    if (bad) { printf("MAL MODELO\n"); return 1; }
    printf("OK\n");
    return 0;
}
