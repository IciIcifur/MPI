#!/usr/bin/env python3
"""
Benchmark script for PI calculation using different MPI algorithms.
"""

import os
import subprocess
import csv
import sys
from datetime import datetime

RESULTS_DIR = "results/pi_results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "pi_results.csv")
EXECUTABLE = "bin/pi_calculator"

TEST_CONFIGS = [
    # 1 процесс
    (1, 1000000),      # 1M iterations
    (1, 5000000),      # 5M iterations
    (1, 10000000),     # 10M iterations
    (1, 50000000),     # 50M iterations
    (1, 100000000),    # 100M iterations

    # 2 процесса
    (2, 1000000),
    (2, 5000000),
    (2, 10000000),
    (2, 50000000),
    (2, 100000000),

    # 4 процесса
    (4, 1000000),
    (4, 5000000),
    (4, 10000000),
    (4, 50000000),
    (4, 100000000),

    # 8 процессов
    (8, 1000000),
    (8, 5000000),
    (8, 10000000),
    (8, 50000000),
    (8, 100000000),
]

ALGORITHMS = ["NAIVE", "MANUAL", "GATHER"]

def ensure_directories():
    """Create necessary directories."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    if not os.path.exists("bin"):
        print("Error: bin directory not found. Please run 'make' first.")
        sys.exit(1)

def build_project():
    """Build the project using make."""
    print("Building project...")
    result = subprocess.run(["make", "clean"], capture_output=True)
    result = subprocess.run(["make"], capture_output=True)
    if result.returncode != 0:
        print("Build failed!")
        try:
            print(result.stderr.decode())
        except Exception:
            print(result.stderr)
        sys.exit(1)
    print("Build successful!")

def run_test(num_processes, iterations):
    """
    Run a single test with specified parameters.
    Returns a dict with results for each algorithm.
    """
    print(f"Running test: {num_processes} processes, {iterations} iterations")

    cmd = ["mpirun", "-np", str(num_processes), EXECUTABLE, "1"]

    try:
        result = subprocess.run(
            cmd,
            input=str(iterations),
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            return None

        output = result.stdout
        results = {}

        for line in output.split('\n'):
            if line.startswith(('NAIVE', 'MANUAL', 'GATHER')):
                parts = line.split('|')
                if len(parts) == 3:
                    algo = parts[0].strip()
                    try:
                        pi_value = float(parts[1].strip())
                        exec_time = float(parts[2].strip())
                    except Exception:
                        continue
                    results[algo] = {
                        'pi_value': pi_value,
                        'execution_time': exec_time
                    }

        return results

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: Test exceeded 5 minutes")
        return None
    except Exception as e:
        print(f"  EXCEPTION: {str(e)}")
        return None

def calculate_metrics(single_thread_time, parallel_time, num_processes):
    """Calculate speedup and efficiency (efficiency as percent)."""
    if parallel_time == 0 or single_thread_time == 0:
        return None, None

    speedup = single_thread_time / parallel_time
    efficiency_percent = (speedup / num_processes) * 100.0

    return speedup, efficiency_percent

def main():
    """Main benchmark execution."""
    print("=" * 70)
    print("PI Calculation Benchmark - Multi-algorithm Analysis")
    print("=" * 70)

    ensure_directories()
    build_project()

    print("\nStarting benchmarks...")
    print(f"Results will be saved to: {RESULTS_FILE}")

    csv_header = [
        'timestamp',
        'num_processes',
        'num_iterations',
        'algorithm',
        'pi_value',
        'execution_time',
        'speedup',
        'efficiency'
    ]

    single_thread_times = {}

    with open(RESULTS_FILE, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()

        for num_processes, iterations in TEST_CONFIGS:
            print(f"\n--- Config: {num_processes} process(es), {iterations} iterations ---")

            test_results = run_test(num_processes, iterations)

            if test_results is None:
                print(f"  Skipping due to error")
                continue

            timestamp = datetime.now().isoformat()

            for algo in ALGORITHMS:
                if algo in test_results:
                    result = test_results[algo]
                    pi_val = result['pi_value']
                    exec_time = result['execution_time']

                    if num_processes == 1:
                        single_thread_times[(algo, iterations)] = exec_time
                        speedup = 1.0
                        efficiency = 100.0
                    else:
                        key = (algo, iterations)
                        if key in single_thread_times:
                            s, e = calculate_metrics(single_thread_times[key], exec_time, num_processes)
                            speedup = s
                            efficiency = e
                        else:
                            speedup = None
                            efficiency = None

                    row = {
                        'timestamp': timestamp,
                        'num_processes': num_processes,
                        'num_iterations': iterations,
                        'algorithm': algo,
                        'pi_value': f"{pi_val:.15f}",
                        'execution_time': f"{exec_time:.6f}",
                        'speedup': f"{speedup:.4f}" if speedup is not None else "",
                        'efficiency': f"{efficiency:.4f}" if efficiency is not None else ""
                    }

                    if num_processes == 1:
                        row['speedup'] = "1.0000"
                        row['efficiency'] = "100.0000"

                    writer.writerow(row)
                    csvfile.flush()

                    if row['speedup'] and row['efficiency']:
                        print(f"  {algo:10} | π={pi_val:.10f} | Time={exec_time:.6f}s | Speedup={row['speedup']} | Eff={row['efficiency']}%")
                    else:
                        print(f"  {algo:10} | π={pi_val:.10f} | Time={exec_time:.6f}s")

    print("\n" + "=" * 70)
    print(f"Benchmark complete! Results saved to {RESULTS_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    main()