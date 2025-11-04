#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

double *generate_matrix(int n) {
    double *matrix = (double*)calloc(n * n, sizeof(double));
    for (int i = 0; i < n * n; ++i)
        matrix[i] = ((double)rand()) / RAND_MAX;
    return matrix;
}

double *generate_vector(int n) {
    double *vector = (double*)calloc(n, sizeof(double));
    for (int i = 0; i < n; ++i)
        vector[i] = ((double)rand()) / RAND_MAX;
    return vector;
}

void row(int n, double* matrix, double* vector);
void column(int n, double* matrix, double* vector);
void block(int n, double* matrix, double* vector);

int runTask2() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand(time(NULL) + rank);

    if (rank == 0)
        printf("  Size |   Row Time (s)  | Column Time (s) | Block Time (s)\n");

    for (int n = 0; n <= 15000; n += 500) {
        double *matrix = NULL;
        double *vector = NULL;
        if (rank == 0) {
            matrix = generate_matrix(n);
            vector = generate_vector(n);
        } else {
            matrix = (double*)calloc(n * n, sizeof(double));
            vector = (double*)calloc(n, sizeof(double));
        }

        MPI_Bcast(matrix, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(vector, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        row(n, matrix, vector);
        column(n, matrix, vector);
        block(n, matrix, vector);

        if (matrix) free(matrix);
        free(vector);
    }
    return 0;
}

void row(int n, double *matrix, double *vector) {
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int base = n / size, rem = n % size;
    int start = rank * base + (rank < rem ? rank : rem);
    int count = base + (rank < rem);

    double *local_matrix = (double*)calloc(count * n, sizeof(double));
    double *local_result = (double*)calloc(count, sizeof(double));

    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            int s = r * base + (r < rem ? r : rem);
            int c = base + (r < rem);
            if (r == 0) {
                for (int i = 0; i < c * n; ++i)
                    local_matrix[i] = matrix[i];
            } else {
                MPI_Send(matrix + s * n, c * n, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(local_matrix, count * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int i = 0; i < count; ++i) {
        local_result[i] = 0.0;
        for (int j = 0; j < n; ++j)
            local_result[i] += local_matrix[i * n + j] * vector[j];
    }

    double *result = NULL;
    int *recvcounts = NULL, *displs = NULL;
    if (rank == 0) {
        result = (double*)calloc(n, sizeof(double));
        recvcounts = (int*)calloc(size, sizeof(int));
        displs = (int*)calloc(size, sizeof(int));
        for (int r = 0, pos = 0; r < size; ++r) {
            int c = base + (r < rem);
            recvcounts[r] = c;
            displs[r] = pos;
            pos += c;
        }
    }

    MPI_Gatherv(local_result, count, MPI_DOUBLE,
                result, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);


    double t1 = MPI_Wtime();
    if (rank == 0)
        printf("%6d | %.13lf", n, t1 - t0);

    free(local_matrix);
    free(local_result);
    if (rank == 0) {
        free(result);
        free(recvcounts);
        free(displs);
    }
}


void column(int n, double *matrix, double *vector) {
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int base = n / size, rem = n % size;
    int col_start = rank * base + (rank < rem ? rank : rem);
    int col_count = base + (rank < rem);

    double *local_result = (double*)calloc(n, sizeof(double));

    for (int col = col_start; col < col_start + col_count; ++col) {
        for (int row = 0; row < n; ++row) {
            local_result[row] += matrix[row * n + col] * vector[col];
        }
    }

    double *result = NULL;
    if (rank == 0) result = (double*)calloc(n, sizeof(double));

    MPI_Reduce(local_result, result, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    if (rank == 0)
        printf(" | %.13lf", t1 - t0);

    free(local_result);
    if (rank == 0)
        free(result);
}

void block(int n, double *matrix, double *vector) {
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int B = 256;
    if (n > 0 && B > n) B = n;

    double *local_result = (double*)calloc(n, sizeof(double));

    int block_idx = 0;
    for (int jb = 0; jb < n; jb += B, ++block_idx) {
        if ((block_idx % size) == rank) {
            int width = (jb + B <= n) ? B : (n - jb);

            for (int i = 0; i < n; ++i) {
                const double *row = matrix + i * n + jb;
                double s = 0.0;
                for (int j = 0; j < width; ++j) {
                    s += row[j] * vector[jb + j];
                }
                local_result[i] += s;
            }
        }
    }

    double *result = NULL;
    if (rank == 0) result = (double*)calloc(n, sizeof(double));
    MPI_Reduce(local_result, result, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();
    if (rank == 0)
        printf(" | %.13lf\n", t1 - t0);

    free(local_result);
    if (rank == 0) free(result);
}