#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

extern int requested_matrix_size;

static double *generate_matrix(int n) {
    double *a = (double*)calloc((size_t)n*(size_t)n, sizeof(double));
    for (int i = 0; i < n*n; ++i) a[i] = (double)rand() / RAND_MAX;
    return a;
}
static double *generate_vector(int n) {
    double *x = (double*)calloc((size_t)n, sizeof(double));
    for (int i = 0; i < n; ++i) x[i] = (double)rand() / RAND_MAX;
    return x;
}

static void split_1d(int n, int r, int size, int *start, int *count) {
    int base = (size > 0) ? n / size : 0;
    int rem  = (size > 0) ? n % size : 0;
    *start = r * base + (r < rem ? r : rem);
    *count = base + (r < rem);
}

static void row(int n, double *Aroot, double *x) {
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int r0, rc; split_1d(n, rank, size, &r0, &rc);

    double *A = (rc > 0) ? (double*)malloc((size_t)rc*(size_t)n*sizeof(double)) : NULL;
    double *y_local = (rc > 0) ? (double*)calloc((size_t)rc, sizeof(double)) : NULL;

    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            int s, c; split_1d(n, r, size, &s, &c);
            if (c <= 0) continue;
            if (r == 0) {
                memcpy(A, Aroot + (size_t)s*(size_t)n, (size_t)c*(size_t)n*sizeof(double));
            } else {
                MPI_Send(Aroot + (size_t)s*(size_t)n, c*n, MPI_DOUBLE, r, 10, MPI_COMM_WORLD);
            }
        }
    } else if (rc > 0) {
        MPI_Recv(A, rc*n, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int i = 0; i < rc; ++i) {
        const double *rowp = A + (size_t)i*(size_t)n;
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += rowp[j] * x[j];
        y_local[i] = s;
    }

    double *y = NULL;
    int *counts = NULL, *displs = NULL;
    if (rank == 0) {
        y = (double*)calloc((size_t)n, sizeof(double));
        counts = (int*)malloc((size_t)size*sizeof(int));
        displs = (int*)malloc((size_t)size*sizeof(int));
        for (int r = 0, pos = 0; r < size; ++r) {
            int s, c; split_1d(n, r, size, &s, &c);
            counts[r] = c; displs[r] = pos; pos += c;
        }
    }
    MPI_Gatherv(y_local, rc, MPI_DOUBLE, y, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();
    double t_local = t1 - t0, t_max = 0.0;
    MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("ROW|%d|%.13lf\n", n, t_max);

    free(A); free(y_local);
    if (rank == 0) { free(y); free(counts); free(displs); }
}

static void column(int n, double *Aroot, double *x) {
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int c0, cc; split_1d(n, rank, size, &c0, &cc);

    double *cols = (cc > 0) ? (double*)malloc((size_t)n*(size_t)cc*sizeof(double)) : NULL;

    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            int s, c; split_1d(n, r, size, &s, &c);
            if (c <= 0) continue;

            if (r == 0) {
                for (int i = 0; i < n; ++i) {
                    const double *rowp = Aroot + (size_t)i*(size_t)n + s;
                    memcpy(cols + (size_t)i*(size_t)c, rowp, (size_t)c*sizeof(double));
                }
            } else {
                double *buf = (double*)malloc((size_t)n*(size_t)c*sizeof(double));
                for (int i = 0; i < n; ++i) {
                    const double *rowp = Aroot + (size_t)i*(size_t)n + s;
                    memcpy(buf + (size_t)i*(size_t)c, rowp, (size_t)c*sizeof(double));
                }
                MPI_Send(buf, n*c, MPI_DOUBLE, r, 20, MPI_COMM_WORLD);
                free(buf);
            }
        }
    } else if (cc > 0) {
        MPI_Recv(cols, n*cc, MPI_DOUBLE, 0, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double *y_partial = (double*)calloc((size_t)n, sizeof(double));
    if (cc > 0) {
        for (int i = 0; i < n; ++i) {
            const double *rowc = cols + (size_t)i*(size_t)cc;
            double s = 0.0;
            for (int j = 0; j < cc; ++j) s += rowc[j] * x[c0 + j];
            y_partial[i] = s;
        }
    }

    double *y = NULL;
    if (rank == 0) y = (double*)calloc((size_t)n, sizeof(double));
    MPI_Reduce(y_partial, y, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();
    double t_local = t1 - t0, t_max = 0.0;
    MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("COLUMN|%d|%.13lf\n", n, t_max);

    free(cols); free(y_partial);
    if (rank == 0) free(y);
}

static void block(int n, double *Aroot, double *x) {
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int c0, cc; split_1d(n, rank, size, &c0, &cc);

    int B = 512;
    if (B > cc) B = cc;

    double *cols = (cc > 0) ? (double*)malloc((size_t)n*(size_t)cc*sizeof(double)) : NULL;

    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            int s, c; split_1d(n, r, size, &s, &c);
            if (c <= 0) continue;

            if (r == 0) {
                for (int i = 0; i < n; ++i) {
                    const double *rowp = Aroot + (size_t)i*(size_t)n + s;
                    memcpy(cols + (size_t)i*(size_t)c, rowp, (size_t)c*sizeof(double));
                }
            } else {
                double *buf = (double*)malloc((size_t)n*(size_t)c*sizeof(double));
                for (int i = 0; i < n; ++i) {
                    const double *rowp = Aroot + (size_t)i*(size_t)n + s;
                    memcpy(buf + (size_t)i*(size_t)c, rowp, (size_t)c*sizeof(double));
                }
                MPI_Send(buf, n*c, MPI_DOUBLE, r, 30, MPI_COMM_WORLD);
                free(buf);
            }
        }
    } else if (cc > 0) {
        MPI_Recv(cols, n*cc, MPI_DOUBLE, 0, 30, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double *y_partial = (double*)calloc((size_t)n, sizeof(double));
    if (cc > 0) {
        for (int jb = 0; jb < cc; jb += B) {
            int w = (jb + B <= cc) ? B : (cc - jb);
            const double *xseg = x + c0 + jb;
            for (int i = 0; i < n; ++i) {
                const double *rowc = cols + (size_t)i*(size_t)cc + jb;
                double s = 0.0;
                for (int j = 0; j < w; ++j) s += rowc[j] * xseg[j];
                y_partial[i] += s;
            }
        }
    }

    double *y = NULL;
    if (rank == 0) y = (double*)calloc((size_t)n, sizeof(double));
    MPI_Reduce(y_partial, y, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();
    double t_local = t1 - t0, t_max = 0.0;
    MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("BLOCK|%d|%.13lf\n", n, t_max);

    free(cols); free(y_partial);
    if (rank == 0) free(y);
}

int runTask2() {
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand((unsigned int)(time(NULL) + rank));

    if (rank == 0)
        printf("  Size |   Row Time (s)  | Column Time (s) | Block Time (s)\n");

    // Если задан конкретный размер - выполняем только его
    if (requested_matrix_size > 0) {
        int n = requested_matrix_size;
        double *A = NULL;
        double *x = NULL;

        if (rank == 0) {
            A = generate_matrix(n);
            x = generate_vector(n);
        } else {
            x = (double*)calloc((size_t)n, sizeof(double));
        }

        MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        row(n, A, x);
        column(n, A, x);
        block(n, A, x);

        if (rank == 0) free(A);
        free(x);
        return 0;
    }

    for (int n = 500; n <= 20000; n += 500) {
        double *A = NULL;
        double *x = NULL;

        if (rank == 0) {
            A = generate_matrix(n);
            x = generate_vector(n);
        } else {
            x = (double*)calloc((size_t)n, sizeof(double));
        }

        MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        row(n, A, x);
        column(n, A, x);
        block(n, A, x);

        if (rank == 0) free(A);
        free(x);
    }
    return 0;
}