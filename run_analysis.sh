#!/bin/bash

set -e

# Переменная для выбора задания (по умолчанию 1 - PI)
TASK=${1:-1}

echo "============================================"
echo "Parallel Algorithms Analysis"
echo "Task: $TASK"
echo "============================================"

# Check if MPI is installed
if ! command -v mpicc &> /dev/null; then
    echo "Error: MPI compiler (mpicc) not found. Please install OpenMPI or MPICH."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3."
    exit 1
fi

# Step 1: Build the project
echo -e "\n[1/3] Building project..."
make clean
make

# Step 2: Run benchmarks based on task
echo -e "\n[2/3] Running benchmarks..."

case $TASK in
    1)
        echo "Running PI calculation benchmarks..."
        python3 scripts/run_benchmark_pi.py
        ;;
    2)
        echo "Running matrix multiplication benchmarks..."
        python3 scripts/run_benchmark_matrix.py
        ;;
    3)
        echo "Running Cannon matrix-matrix multiplication benchmarks..."
        python3 scripts/run_benchmark_cannon.py
        ;;
    *)
        echo "Error: Unknown task number: $TASK"
        echo "Available tasks: 1 (PI calculation), 2 (Matrix multiplication), 3 (Cannon matrix multiplication)"
        exit 1
        ;;
esac

# Step 3: Analyze results
echo -e "\n[3/3] Analyzing results and generating graphs..."

case $TASK in
    1)
        python3 scripts/analyze_results_pi.py
        ;;
    2)
        python3 scripts/analyze_results_matrix.py
        ;;
    3)
        python3 scripts/analyze_results_cannon.py
        ;;
esac

echo -e "\n============================================"
echo "Analysis complete!"
echo "Results saved in 'results/' directory"
echo "Task: $TASK"
echo "============================================"