#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static void *xmalloc(const size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "malloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return p;
}

void pack_blocks_from_full(const double *full, double *sendbuf, const int Nfull, const int block_size, const int sqrtP) {
    int idx = 0;
    for (int br = 0; br < sqrtP; ++br)
        for (int bc = 0; bc < sqrtP; ++bc)
            for (int i = 0; i < block_size; ++i) {
                const int gr = br * block_size + i;
                for (int j = 0; j < block_size; ++j) {
                    const int gc = bc * block_size + j;
                    sendbuf[idx++] = full[gr * Nfull + gc];
                }
            }
}

void unpack_blocks_to_full(const double *recvbuf, double *full, const int Nfull, const int block_size, const int sqrtP) {
    int idx = 0;
    for (int br = 0; br < sqrtP; ++br)
        for (int bc = 0; bc < sqrtP; ++bc)
            for (int i = 0; i < block_size; ++i) {
                const int gr = br * block_size + i;
                for (int j = 0; j < block_size; ++j) {
                    const int gc = bc * block_size + j;
                    full[gr * Nfull + gc] = recvbuf[idx++];
                }
            }
}

void local_matmul_accumulate(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; ++i) {
        const int ri = i * n;
        for (int k = 0; k < n; ++k) {
            const double a = A[ri + k];
            const int rk = k * n;
            for (int j = 0; j < n; ++j)
                C[ri + j] += a * B[rk + j];
        }
    }
}

void seq_matmul(const double *A, const double *B, double *C, int N) {
    for (int i = 0; i < N*N; ++i) C[i] = 0.0;
    for (int i = 0; i < N; ++i) {
        const int ri = i * N;
        for (int k = 0; k < N; ++k) {
            const double a = A[ri + k];
            const int rk = k * N;
            for (int j = 0; j < N; ++j)
                C[ri + j] += a * B[rk + j];
        }
    }
}

void runTask3() {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int sqrtP = (int)round(sqrt(world_size));
    if (sqrtP * sqrtP != world_size) {
        if (world_rank == 0)
            fprintf(stderr, "Number of processes must be a perfect square.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int num_sizes = 15;
    const int N_min = 64;
    const int N_max = 2048;
    const double growth = pow((double)N_max / N_min, 1.0 / (num_sizes - 1));
    int *sizes = malloc(num_sizes * sizeof(int));
    for (int i = 0; i < num_sizes; ++i) {
        int Ni = (int)(N_min * pow(growth, i) + 0.5);
        Ni = (Ni + sqrtP - 1) / sqrtP * sqrtP;
        sizes[i] = Ni;
    }

    if (world_rank == 0) {
        printf("Size  | Processes | AvgTime(s)  | Speedup | Efficiency(%%)| Check\n");
        printf("-----------------------------------------------------------------\n");
    }

    for (int sidx = 0; sidx < num_sizes; ++sidx) {
        const int iterations = 3;
        const int N = sizes[sidx];
        const int block_size = N / sqrtP;
        const int block_elems = block_size * block_size;

        const int dims[2] = {sqrtP, sqrtP};
        const int periods[2] = {1, 1};
        MPI_Comm grid;
        MPI_Cart_create(MPI_COMM_WORLD, 2, (int*)dims, (int*)periods, 0, &grid);

        int grid_rank;
        MPI_Comm_rank(grid, &grid_rank);
        int coords[2];
        MPI_Cart_coords(grid, grid_rank, 2, coords);

        int left, right, up, down;
        MPI_Cart_shift(grid, 1, 1, &left, &right);
        MPI_Cart_shift(grid, 0, 1, &up, &down);

        double *A = xmalloc(block_elems * sizeof(double));
        double *B = xmalloc(block_elems * sizeof(double));
        double *C = xmalloc(block_elems * sizeof(double));

        double *A_full = NULL, *B_full = NULL, *sendA = NULL, *sendB = NULL;
        if (world_rank == 0) {
            A_full = xmalloc(N * N * sizeof(double));
            B_full = xmalloc(N * N * sizeof(double));
            srand(42);
            for (int i = 0; i < N * N; ++i) {
                A_full[i] = ((double)rand())/RAND_MAX;
                B_full[i] = ((double)rand())/RAND_MAX;
            }
            sendA = xmalloc(world_size * block_elems * sizeof(double));
            sendB = xmalloc(world_size * block_elems * sizeof(double));
            pack_blocks_from_full(A_full, sendA, N, block_size, sqrtP);
            pack_blocks_from_full(B_full, sendB, N, block_size, sqrtP);
        }

        MPI_Scatter(sendA, block_elems, MPI_DOUBLE, A, block_elems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(sendB, block_elems, MPI_DOUBLE, B, block_elems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (world_rank == 0) { free(sendA); free(sendB); }

        {
            double *tmpA = xmalloc(block_elems * sizeof(double));
            double *tmpB = xmalloc(block_elems * sizeof(double));
            double *tmpC = xmalloc(block_elems * sizeof(double));
            memcpy(tmpA, A, block_elems * sizeof(double));
            memcpy(tmpB, B, block_elems * sizeof(double));
            for (int i = 0; i < block_elems; ++i) tmpC[i] = 0.0;
            local_matmul_accumulate(tmpA, tmpB, tmpC, block_size);
            free(tmpA); free(tmpB); free(tmpC);
        }

        double total_time = 0.0;

        MPI_Barrier(grid);

        for (int iter = 0; iter < iterations; ++iter) {
            for (int i = 0; i < block_elems; ++i) C[i] = 0.0;

            MPI_Barrier(grid);
            const double t0 = MPI_Wtime();

            for (int s = 0; s < coords[0]; ++s)
                MPI_Sendrecv_replace(A, block_elems, MPI_DOUBLE, left, 1, right, 1, grid, MPI_STATUS_IGNORE);
            for (int s = 0; s < coords[1]; ++s)
                MPI_Sendrecv_replace(B, block_elems, MPI_DOUBLE, up, 2, down, 2, grid, MPI_STATUS_IGNORE);

            for (int step = 0; step < sqrtP; ++step) {
                local_matmul_accumulate(A, B, C, block_size);
                MPI_Sendrecv_replace(A, block_elems, MPI_DOUBLE, left, 3, right, 3, grid, MPI_STATUS_IGNORE);
                MPI_Sendrecv_replace(B, block_elems, MPI_DOUBLE, up, 4, down, 4, grid, MPI_STATUS_IGNORE);
            }

            const double t1 = MPI_Wtime();
            double local_time = t1 - t0;

            double iter_max;
            MPI_Reduce(&local_time, &iter_max, 1, MPI_DOUBLE, MPI_MAX, 0, grid);

            if (grid_rank == 0)
                total_time += iter_max;

            if (world_rank == 0) {
                sendA = xmalloc(world_size * block_elems * sizeof(double));
                sendB = xmalloc(world_size * block_elems * sizeof(double));
                pack_blocks_from_full(A_full, sendA, N, block_size, sqrtP);
                pack_blocks_from_full(B_full, sendB, N, block_size, sqrtP);
            }
            MPI_Scatter(sendA, block_elems, MPI_DOUBLE, A, block_elems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Scatter(sendB, block_elems, MPI_DOUBLE, B, block_elems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (world_rank == 0) { free(sendA); free(sendB); }
        }

        if (world_rank == 0) {
            const double par_time = total_time / iterations;

            double seq_total = 0.0;
            double *C_seq = xmalloc(N * N * sizeof(double));
            for (int it = 0; it < iterations; ++it) {
                const double seq_t0 = MPI_Wtime();
                seq_matmul(A_full, B_full, C_seq, N);
                const double seq_t1 = MPI_Wtime();
                seq_total += (seq_t1 - seq_t0);
            }
            const double seq_time = seq_total / iterations;

            double *recvC = xmalloc(world_size * block_elems * sizeof(double));
            MPI_Gather(C, block_elems, MPI_DOUBLE, recvC, block_elems, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            double *C_par_full = xmalloc(N * N * sizeof(double));
            unpack_blocks_to_full(recvC, C_par_full, N, block_size, sqrtP);

            double max_abs_diff = 0.0;
            double max_ref = 0.0;
            for (int i = 0; i < N * N; ++i) {
                const double diff = fabs(C_par_full[i] - C_seq[i]);
                if (diff > max_abs_diff) max_abs_diff = diff;
                const double aval = fabs(C_seq[i]);
                if (aval > max_ref) max_ref = aval;
            }
            const double rel = max_abs_diff / (max_ref > 0.0 ? max_ref : 1.0);
            const double eps = 1e-8;

            const double speedup = seq_time / par_time;
            const double efficiency = speedup / (double)world_size * 100.0;

            printf("%5d | %9d | %11.6f | %7.3f | %12.2f | ", N, world_size, par_time, speedup, efficiency);
            if (max_abs_diff > eps && rel > eps) {
                printf("Mismatch (abs=%e rel=%e)\n", max_abs_diff, rel);
            } else {
                printf("OK (abs=%e rel=%e)\n", max_abs_diff, rel);
            }

            free(C_seq);
            free(recvC);
            free(C_par_full);
            free(A_full); free(B_full);
        } else {
            MPI_Gather(C, block_elems, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        free(A); free(B); free(C);
        MPI_Comm_free(&grid);
    }

    free(sizes);
}