#include <stdio.h>
#include "tasks.h"
#include <mpi.h>

int runTask1() {
    int n, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("Enter the iterations number:\n");
        scanf("%d", &n);
        if (n < 0) return 1;

        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }


    return 0;
}
