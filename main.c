#include <stdio.h>
#include "tasks/tasks.h"
#include <mpi.h>

int runTask(int taskNumber);

int main(int argc, char *argv[]) {
    int rank;
    int selectedTask = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("LAB 1\n");
        printf("Enter task number:\n1 - Finding PI\n2 - Matrix Multiplication\n3 - Cannon's Matrix Multiplication\n");

        while (selectedTask < 1 || selectedTask > 3) {
            scanf("%d",&selectedTask);
            if (0 < selectedTask && selectedTask <= 3) {
                printf("-------------Task %d---------------\n\n", selectedTask);
                break;
            }
            printf("No such task\n");
        }

    }

    MPI_Bcast(&selectedTask, 1, MPI_INT, 0, MPI_COMM_WORLD);
    runTask(selectedTask);

    MPI_Finalize();
    return 0;
}

int runTask(int taskNumber) {
    switch (taskNumber) {
        case 1:
            return runTask1();
        case 2:
            break;
        case 3:
            break;
        default:
           break;
    }
    return 0;
}