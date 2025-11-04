#!/bin/bash

mpicc -o main main.c tasks/*.c

NPROCS=4

mpirun -np $NPROCS ./main