#include <stdio.h>
#include "tasks.h"
#include <mpi.h>
#include <time.h>
#include <stdlib.h>

typedef struct {
    double execution_time;
    double pi_value;
    int total_hits;
} AlgorithmResult;

AlgorithmResult naive(int n, int iterations, int rank, int size);
AlgorithmResult manual_collect(int n, int iterations, int rank, int size);
AlgorithmResult gather_collect(int n, int iterations, int rank, int size);

int runTask1() {
    int n, rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the total iterations number:\n");
        if (scanf("%d", &n) != 1) {
            fprintf(stderr, "Error: Invalid input for iterations\n");
            n = -1;
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Number of iterations must be positive\n");
        }
        return 1;
    }

    int iterations_per_process = n / size;
    int remainder = n % size;
    int iterations = iterations_per_process + (rank < remainder ? 1 : 0);

    const AlgorithmResult naive_result = naive(n, iterations, rank, size);
    const AlgorithmResult manual_result = manual_collect(n, iterations, rank, size);
    const AlgorithmResult gather_result = gather_collect(n, iterations, rank, size);

    if (rank == 0) {
        printf("\n\n=== RESULTS SUMMARY ===\n");
        printf("NAIVE|%.15f|%lf\n", naive_result.pi_value, naive_result.execution_time);
        printf("MANUAL|%.15f|%lf\n", manual_result.pi_value, manual_result.execution_time);
        printf("GATHER|%.15f|%lf\n", gather_result.pi_value, gather_result.execution_time);
    }

    return 0;
}

AlgorithmResult naive(int n, int iterations, int rank, int size) {
    AlgorithmResult result = {0, 0, 0};
    int total_iterations_count = 0;

    if (rank == 0) printf("\n\nNAIVE IMPLEMENTATION:\n\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int hits = 0;
    unsigned int seed = (unsigned int)time(NULL) + rank * 101;
    srand(seed);

    for (int i = 0; i < iterations; ++i) {
        double x = ((double)rand()/RAND_MAX) * 2.0 - 1.0;
        double y = ((double)rand()/RAND_MAX) * 2.0 - 1.0;
        if (x*x + y*y <= 1.0)
            ++hits;
    }

    int total_hits = 0;
    MPI_Reduce(&hits, &total_hits, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&iterations, &total_iterations_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    double total_time = t1 - t0;

    if (rank == 0) {
        result.pi_value = 4 * (double)total_hits / total_iterations_count;
        result.execution_time = total_time;
        result.total_hits = total_hits;

        printf("PI = %.15f\n", result.pi_value);
        printf("Total Time = %lf seconds (compute+comm, barrier-synced)\n", total_time);
        printf("Total iterations: %d, Total hits: %d\n", total_iterations_count, total_hits);
    }

    return result;
}

AlgorithmResult manual_collect(int n, int iterations, int rank, int size) {
    AlgorithmResult result = {0, 0, 0};
    int total_iterations_count = 0;

    if (rank == 0) printf("\n\nMANUAL COLLECT (MPI_SEND/RECV) IMPLEMENTATION:\n\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int hits = 0;
    unsigned int seed = (unsigned int)time(NULL) + rank * 101;
    srand(seed);

    for (int i = 0; i < iterations; ++i) {
        double x = ((double)rand()/RAND_MAX) * 2.0 - 1.0;
        double y = ((double)rand()/RAND_MAX) * 2.0 - 1.0;
        if (x*x + y*y <= 1.0)
            ++hits;
    }

    int total_hits = hits;

    if (rank == 0) {
        total_iterations_count = iterations;
        for (int src = 1; src < size; ++src) {
            int recv_iterations, recv_hits;
            MPI_Recv(&recv_iterations, 1, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&recv_hits, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_iterations_count += recv_iterations;
            total_hits += recv_hits;
        }
    } else {
        MPI_Send(&iterations, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&hits, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        result.pi_value = 4 * (double)total_hits / total_iterations_count;
        result.execution_time = t1 - t0;
        result.total_hits = total_hits;

        printf("PI = %.15f\n", result.pi_value);
        printf("Time = %lf seconds (compute+comm, barrier-synced)\n", result.execution_time);
        printf("Total iterations: %d, Total hits: %d\n", total_iterations_count, total_hits);
    }

    return result;
}

AlgorithmResult gather_collect(int n, int iterations, int rank, int size) {
    AlgorithmResult result = {0, 0, 0};
    int total_iterations_count = 0;

    if (rank == 0) printf("\n\nGATHER IMPLEMENTATION:\n\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int hits = 0;
    unsigned int seed = (unsigned int)time(NULL) + rank * 101;
    srand(seed);

    for (int i = 0; i < iterations; ++i) {
        double x = ((double)rand()/RAND_MAX) * 2.0 - 1.0;
        double y = ((double)rand()/RAND_MAX) * 2.0 - 1.0;
        if (x*x + y*y <= 1.0)
            ++hits;
    }

    int total_hits = hits;

    int *all_hits = NULL;
    int *all_iterations = NULL;

    if (rank == 0) {
        all_hits = malloc(size * sizeof(int));
        all_iterations = malloc(size * sizeof(int));
    }

    MPI_Gather(&hits, 1, MPI_INT, all_hits, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&iterations, 1, MPI_INT, all_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int total_hits_local = 0;
        total_iterations_count = 0;
        for (int i = 0; i < size; ++i) {
            total_hits_local += all_hits[i];
            total_iterations_count += all_iterations[i];
        }
        total_hits = total_hits_local;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        result.pi_value = 4 * (double)total_hits / total_iterations_count;
        result.execution_time = t1 - t0;
        result.total_hits = total_hits;

        printf("PI = %.15f\n", result.pi_value);
        printf("Time = %lf seconds (compute+comm, barrier-synced)\n", result.execution_time);
        printf("Total iterations: %d, Total hits: %d\n", total_iterations_count, total_hits);

        free(all_hits);
        free(all_iterations);
    }

    return result;
}