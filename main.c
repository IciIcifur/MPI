#include <stdio.h>
#include "tasks/tasks.h"

int runTask(int taskNumber);

int main(void) {
    printf("LAB 1\n");
    printf("Enter task number:\n1 - Finding PI\n2 - Matrix Multiplication\n3 - Cannon's Matrix Multiplication\n");
    int selectedTask = 0;
    while (0 >= selectedTask || selectedTask >= 3) {
        scanf("%d",&selectedTask);
        if (0 < selectedTask && selectedTask <= 3) {
            printf("-------------Task %d---------------\n\n", selectedTask);
            break;
        }
        printf("No such task\n");
    }

    runTask(selectedTask);
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