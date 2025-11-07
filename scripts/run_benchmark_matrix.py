#!/usr/bin/env python3
"""
Benchmark script for matrix-vector multiplication algorithms.
"""

import os
import subprocess
import csv
import sys
from datetime import datetime

RESULTS_DIR = "results/matrix_results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "matrix_results.csv")
EXECUTABLE = "bin/matrix_multiplier"

PROCESS_COUNTS = [1, 2, 4, 8]
MATRIX_SIZES = [
    500, 1000, 2000, 3000, 4000, 5000,
    6000, 7000, 8000, 9000, 10000,
    12000, 14000, 16000, 18000, 20000
]

ALGORITHMS = ["ROW", "COLUMN", "BLOCK"]

def ensure_directories():
    """Create necessary directories."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

def run_matrix_test(num_processes, matrix_size):
    """
    Run a single matrix test with specified parameters.
    """
    print(f"Running test: {num_processes} processes, matrix {matrix_size}x{matrix_size}")

    cmd = ["mpirun", "-np", str(num_processes), EXECUTABLE, "2", str(matrix_size)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            return None

        output = result.stdout
        results = {}

        for line in output.split('\n'):
            if '|' in line:
                parts = line.split('|')
                if len(parts) == 3:
                    algo = parts[0].strip()
                    try:
                        size = int(parts[1].strip())
                        time = float(parts[2].strip())
                    except:
                        continue

                    if algo in ALGORITHMS and size == matrix_size:
                        results[algo] = time

        return results

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: Test exceeded 10 minutes")
        return None
    except Exception as e:
        print(f"  EXCEPTION: {str(e)}")
        return None

def calculate_metrics(single_process_time, parallel_time, num_processes):
    """Calculate speedup and efficiency."""
    if parallel_time == 0 or single_process_time == 0:
        return None, None

    speedup = single_process_time / parallel_time
    efficiency = (speedup / num_processes) * 100.0

    return speedup, efficiency

def main():
    """Main benchmark execution."""
    print("=" * 70)
    print("Matrix-Vector Multiplication Benchmark")
    print("Testing different process counts AND matrix sizes")
    print("=" * 70)

    ensure_directories()

    print("\nStarting benchmarks...")
    print(f"Results will be saved to: {RESULTS_FILE}")

    csv_header = [
        'timestamp',
        'num_processes',
        'matrix_size',
        'algorithm',
        'execution_time',
        'speedup',
        'efficiency'
    ]

    single_process_times = {}

    with open(RESULTS_FILE, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()

        print("\n=== Phase 1: Single process baseline ===")
        for matrix_size in MATRIX_SIZES:
            test_results = run_matrix_test(1, matrix_size)
            if test_results:
                timestamp = datetime.now().isoformat()

                for algo in ALGORITHMS:
                    if algo in test_results:
                        exec_time = test_results[algo]
                        single_process_times[(algo, matrix_size)] = exec_time

                        row = {
                            'timestamp': timestamp,
                            'num_processes': 1,
                            'matrix_size': matrix_size,
                            'algorithm': algo,
                            'execution_time': f"{exec_time:.6f}",
                            'speedup': "1.0000",
                            'efficiency': "100.0000"
                        }

                        writer.writerow(row)
                        csvfile.flush()
                        print(f"  {algo:8} | Time={exec_time:.6f}s | Speedup=1.0000x | Eff=100.00%")

        print("\n=== Phase 2: Multi-process testing ===")
        for num_processes in PROCESS_COUNTS:
            if num_processes == 1:
                continue

            for matrix_size in MATRIX_SIZES:
                print(f"\n--- Config: {num_processes} processes, size {matrix_size} ---")

                test_results = run_matrix_test(num_processes, matrix_size)

                if test_results is None:
                    print(f"  Skipping due to error")
                    continue

                timestamp = datetime.now().isoformat()

                for algo in ALGORITHMS:
                    if algo in test_results:
                        exec_time = test_results[algo]

                        speedup = None
                        efficiency = None
                        key = (algo, matrix_size)
                        if key in single_process_times:
                            single_time = single_process_times[key]
                            speedup, efficiency = calculate_metrics(
                                single_time, exec_time, num_processes
                            )

                        row = {
                            'timestamp': timestamp,
                            'num_processes': num_processes,
                            'matrix_size': matrix_size,
                            'algorithm': algo,
                            'execution_time': f"{exec_time:.6f}",
                            'speedup': f"{speedup:.4f}" if speedup is not None else "",
                            'efficiency': f"{efficiency:.4f}" if efficiency is not None else ""
                        }

                        writer.writerow(row)
                        csvfile.flush()

                        if speedup is not None and efficiency is not None:
                            print(f"  {algo:8} | Time={exec_time:.6f}s | Speedup={speedup:.4f}x | Eff={efficiency:.2f}%")
                        else:
                            print(f"  {algo:8} | Time={exec_time:.6f}s")

    print("\n" + "=" * 70)
    print(f"Benchmark complete! Results saved to {RESULTS_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    main()