#include <stdio.h>
#include "tasks.h"
#include <mpi.h>
#include <time.h>
#include <stdlib.h>

void naive(int n, int iterations, int rank);
void manual_collect(int n, int iterations, int rank, int size);
void gather_collect(int n, int iterations, int rank, int size);

int runTask1() {
    int n, rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the total iterations number:\n");
        scanf("%d", &n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n < 0) return 1;


    naive(n, n / size, rank);
    manual_collect(n, n / size, rank, size);
    gather_collect(n, n / size, rank, size);

    return 0;
}

void naive(int n, int iterations, int rank) {
    if (rank == 0) printf("\n\nNAIVE IMPLEMNTATION:\n\n");

    double t0 = MPI_Wtime();
    int hits = 0;
    unsigned int seed = time(NULL) + rank; // seed for random numbers

    srand(seed);

    for (int i = 0; i < iterations; ++i) {
        double x = ((double)rand()/RAND_MAX) * 2.0 - 1.0;
        double y = ((double)rand()/RAND_MAX) * 2.0 - 1.0;
        if (x*x + y*y <= 1.0)
            ++hits;
     }


     int total_hits = 0;
     MPI_Reduce(&hits, &total_hits, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

     double t1 = MPI_Wtime();

     if (rank == 0) {
        float pi = 4 * (float)total_hits / n;
        printf("PI = %.15f\n\n", pi);

        printf("Done in %lfs\n", t1 - t0);
    }
}

void manual_collect(int n, int iterations, int rank, int size) {
    if (rank == 0) printf("\n\nMPI_SEND/MPI_REDUCE  IMPLEMNTATION:\n\n");

    double t0 = MPI_Wtime();
    int hits = 0;
    unsigned int seed = time(NULL) + rank;

    srand(seed);

    for (int i = 0; i < iterations; ++i) {
        double x = ((double)rand()/RAND_MAX) * 2.0 - 1.0;
        double y = ((double)rand()/RAND_MAX) * 2.0 - 1.0;
        if (x*x + y*y <= 1.0)
            ++hits;
    }

    int total_hits = hits;

    if (rank == 0) {
        for (int src = 1; src < size; ++src) {
            int recv_hits;
            MPI_Recv(&recv_hits, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_hits += recv_hits;
        }
        double t1 = MPI_Wtime();

        float pi = 4 * (float)total_hits / n;
        printf("PI = %.15f\n\n", pi);
        printf("Done in %lfs\n", t1 - t0);
    } else {
        MPI_Send(&hits, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

void gather_collect(int n, int iterations, int rank, int size) {
    if (rank == 0) printf("\n\nMPI_GATHER IMPLEMNTATION:\n\n");

    double t0 = MPI_Wtime();
    int hits = 0;
    unsigned int seed = time(NULL) + rank;
    srand(seed);

    for (int i = 0; i < iterations; ++i) {
        double x = ((double)rand()/RAND_MAX) * 2.0 - 1.0;
        double y = ((double)rand()/RAND_MAX) * 2.0 - 1.0;
        if (x*x + y*y <= 1.0)
            ++hits;
    }

    int *all_hits = NULL;
    if (rank == 0) {
        all_hits = malloc(size * sizeof(int));
    }

    MPI_Gather(&hits, 1, MPI_INT, all_hits, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    if (rank == 0) {
        int total_hits = 0;
        for (int i = 0; i < size; ++i) {
            total_hits += all_hits[i];
        }
        float pi = 4 * (float)total_hits / n;
        printf("PI = %.15f\n\n", pi);
        printf("Done in %lfs\n", t1 - t0);
        free(all_hits);
    }
}