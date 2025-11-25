#include <stdio.h>
#include "tasks/tasks.h"
#include <mpi.h>
#include <stdlib.h>

int requested_matrix_size = -1;

int runTask(const int taskNumber) {
    switch (taskNumber) {
        case 1:
            return runTask1();
        case 2:
            return runTask2();
        case 3:
            runTask3();
            return 0;
        default: {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0) {
                printf("Invalid task number: %d\n", taskNumber);
            }
        }
            break;
    }
    return 0;
}

int main(int argc, char *argv[]) {
    int rank;
    int selectedTask = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc > 1) {
        selectedTask = atoi(argv[1]);
    } else if (rank == 0) {
        printf("LAB 1\n");
        printf("Enter task number:\n1 - Finding PI\n2 - Matrix Multiplication\n3 - Cannon's Matrix Multiplication\n");

        while (selectedTask < 1 || selectedTask > 3) {
            const int scan_result = scanf("%d", &selectedTask);
            if (scan_result != 1) {
                printf("Error reading input. Please enter a number.\n");
                while (getchar() != '\n');
                continue;
            }

            if (selectedTask == 1) {
                printf("-------------Task %d---------------\n\n", selectedTask);
                break;
            }
            printf("No such task\n");
        }
    }

    if (argc > 2) {
        requested_matrix_size = atoi(argv[2]);
        if (requested_matrix_size <= 0) requested_matrix_size = -1;
    }

    MPI_Bcast(&selectedTask, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&requested_matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (selectedTask == 1 || selectedTask == 2 || selectedTask == 3) {
        runTask(selectedTask);
    } else {
        if (rank == 0) printf("No such task\n");
    }

    MPI_Finalize();
    return 0;
}