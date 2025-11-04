#!/usr/bin/env bash

mpicc -o main main.c tasks/*.c -lm

NPROCS=4

mpirun -np $NPROCS ./main