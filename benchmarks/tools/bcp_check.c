/* bcp_check.c — verificación independiente por BCP con depuración. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    FILE *fp = fopen(argv[1], "r");
    if (!fp) return 2;
    long nv = 0, nc = 0;
    char line[1 << 20];
    int ncl = 0;
    int **cls = NULL; int *sz = NULL;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == 'c') continue;
        if (line[0] == 'p') { sscanf(line, "p cnf %ld %ld", &nv, &nc); continue; }
        char *tok = strtok(line, " \t\r\n");
        int lits[4096]; int n = 0;
        while (tok && n < 4096) {
            long l = atol(tok);
            if (l == 0) break;
            lits[n++] = (int)l;
            tok = strtok(NULL, " \t\r\n");
        }
        cls = realloc(cls, (size_t)(ncl + 1) * sizeof(int *));
        sz = realloc(sz, (size_t)(ncl + 1) * sizeof(int));
        cls[ncl] = malloc((size_t)(n ? n : 1) * sizeof(int));
        memcpy(cls[ncl], lits, (size_t)(n ? n : 1) * sizeof(int));
        sz[ncl] = n;
        ncl++;
    }
    fclose(fp);
    signed char *val = calloc((size_t)nv + 1, 1);
    int *why = calloc((size_t)nv + 1, sizeof(int)); /* cláusula que asignó cada var */
    int changed = 1;
    int conflict = 0;
    int rounds = 0;
    while (changed && !conflict && rounds < 1000000) {
        changed = 0;
        rounds++;
        for (int c = 0; c < ncl; ++c) {
            int sat = 0, unassigned = -1;
            for (int i = 0; i < sz[c]; ++i) {
                int l = cls[c][i];
                long v = l > 0 ? l : -l;
                if (val[v] == (l > 0 ? 1 : -1)) { sat = 1; break; }
                if (val[v] == 0) unassigned = i;
            }
            if (sat) continue;
            if (unassigned < 0) {
                conflict = 1;
                printf("CONFLICTO en clausula %d: ", c);
                for (int i = 0; i < sz[c]; ++i) printf("%d ", cls[c][i]);
                printf("0\n");
                printf("asignaciones que causan el conflicto:\n");
                for (int i = 0; i < sz[c]; ++i) {
                    int l = cls[c][i];
                    long v = l > 0 ? l : -l;
                    printf("  lit %d -> %d (por clausula %d)\n", l, val[v], why[v]);
                }
                /* si la variable conflictuada tiene razón, inspeccionar la
                   cláusula de razón para rastrear la cadena */
                {
                    long v0 = 0;
                    for (int i = 0; i < sz[c]; ++i) {
                        int l = cls[c][i];
                        long v = l > 0 ? l : -l;
                        if (val[v] != 0) { v0 = v; break; }
                    }
                    if (v0 && why[v0] >= 0) {
                        printf("razon de la var %ld (clausula %d): ", v0, why[v0]);
                        for (int i = 0; i < sz[why[v0]]; ++i) printf("%d ", cls[why[v0]][i]);
                        printf("0\n");
                        printf("valores de esa razon:\n");
                        for (int i = 0; i < sz[why[v0]]; ++i) {
                            int l = cls[why[v0]][i];
                            long v = l > 0 ? l : -l;
                            printf("  lit %d -> %d (por clausula %d)\n", l, val[v], why[v]);
                        }
                    }
                }
                for (long dv = 1; dv <= 22 && dv <= nv; ++dv) {
                    printf("var %ld -> %d (por %d)\n", dv, val[dv], why[dv]);
                }
                break;
            }
            int l = cls[c][unassigned];
            long v = l > 0 ? l : -l;
            val[v] = (signed char)(l > 0 ? 1 : -1);
            why[v] = c;
            changed = 1;
        }
    }
    if (conflict) { printf("UNSAT (conflicto por BCP)\n"); return 20; }
    for (int c = 0; c < ncl; ++c) {
        int sat = 0;
        for (int i = 0; i < sz[c]; ++i) {
            int l = cls[c][i];
            long v = l > 0 ? l : -l;
            if (val[v] == (l > 0 ? 1 : -1)) { sat = 1; break; }
        }
        if (!sat) { printf("NI SAT NI UNSAT (no BCP-completa)\n"); return 1; }
    }
    printf("SAT (modelo encontrado solo con BCP)\n");
    return 10;
}
