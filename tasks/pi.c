#include <stdio.h>
#include "tasks.h"

int runTask1() {
    int n;
    printf("Enter the iterations number: ");
    scanf("%d", n);
    if (n < 0) return 1;

    return 0;
}
