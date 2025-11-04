#include <stdio.h>
#include "tasks.h"
#include <mpi.h>
#include <time.h>
#include <stdlib.h>

void naive(int n, int iterations, int rank);

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