#!/usr/bin/env python3
"""
Analysis script for matrix multiplication benchmark results.
Generates graphs and statistical analysis.
"""

import os
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_FILE = "results/matrix_results/matrix_results.csv"
RESULTS_DIR = "results/matrix_results"

def ensure_results_dir():
    """Create results directory if needed."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

def load_results():
    """Load results from CSV file."""
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found")
        sys.exit(1)

    data = {}

    with open(RESULTS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row['algorithm']
            processes = int(row['num_processes'])
            size = int(row['matrix_size'])

            if algo not in data:
                data[algo] = {}
            if size not in data[algo]:
                data[algo][size] = {
                    'execution_times': {},
                    'speedups': {},
                    'efficiencies': {}
                }

            if processes not in data[algo][size]['execution_times']:
                data[algo][size]['execution_times'][processes] = []
            if processes not in data[algo][size]['speedups']:
                data[algo][size]['speedups'][processes] = []
            if processes not in data[algo][size]['efficiencies']:
                data[algo][size]['efficiencies'][processes] = []

            exec_time = float(row['execution_time'])
            data[algo][size]['execution_times'][processes].append(exec_time)

            if row['speedup'] and row['speedup'].strip():
                try:
                    speedup = float(row['speedup'])
                    data[algo][size]['speedups'][processes].append(speedup)
                except ValueError:
                    pass

            if row['efficiency'] and row['efficiency'].strip():
                try:
                    efficiency = float(row['efficiency'])
                    data[algo][size]['efficiencies'][processes].append(efficiency)
                except ValueError:
                    pass

    return data

def plot_execution_time_vs_size(data):
    """Plot execution time vs matrix size for different process counts."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    process_counts = [1, 2, 4, 8]
    algorithms = list(data.keys())

    for idx, algo in enumerate(algorithms):
        ax = axes[idx]

        for processes in process_counts:
            sizes = []
            times = []

            for size in sorted(data[algo].keys()):
                if processes in data[algo][size]['execution_times']:
                    times_data = data[algo][size]['execution_times'][processes]
                    if times_data:
                        sizes.append(size)
                        times.append(np.mean(times_data))

            if sizes:
                ax.plot(sizes, times, marker='o', label=f'{processes} processes', linewidth=2)

        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'{algo} Algorithm\nExecution Time vs Matrix Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/execution_time_vs_size.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR}/execution_time_vs_size.png")

def plot_speedup_vs_processes(data):
    """Plot speedup vs number of processes for different matrix sizes."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    algorithms = list(data.keys())
    characteristic_sizes = [1000, 5000, 10000, 20000]

    for idx, algo in enumerate(algorithms):
        ax = axes[idx]

        for size in characteristic_sizes:
            if size in data[algo]:
                speedups_data = data[algo][size]['speedups']
                processes = sorted([p for p in speedups_data.keys() if p > 1 and speedups_data[p]])

                if processes:
                    speedups = [np.mean(speedups_data[p]) for p in processes]
                    ax.plot(processes, speedups, marker='o', label=f'Size {size}', linewidth=2)

        ideal_processes = list(range(1, 9))
        ideal_speedup = ideal_processes
        ax.plot(ideal_processes, ideal_speedup, 'k--', label='Ideal', linewidth=2, alpha=0.7)

        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Speedup')
        ax.set_title(f'{algo} Algorithm\nSpeedup vs Processes')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/speedup_vs_processes.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR}/speedup_vs_processes.png")

def plot_efficiency_vs_processes(data):
    """Plot efficiency vs number of processes for different matrix sizes."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    algorithms = list(data.keys())
    characteristic_sizes = [1000, 5000, 10000, 20000]

    for idx, algo in enumerate(algorithms):
        ax = axes[idx]

        for size in characteristic_sizes:
            if size in data[algo]:
                efficiencies_data = data[algo][size]['efficiencies']
                processes = sorted([p for p in efficiencies_data.keys() if p > 1 and efficiencies_data[p]])

                if processes:
                    efficiencies = [np.mean(efficiencies_data[p]) for p in processes]
                    ax.plot(processes, efficiencies, marker='o', label=f'Size {size}', linewidth=2)

        ax.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='100% efficiency')
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Efficiency (%)')
        ax.set_title(f'{algo} Algorithm\nEfficiency vs Processes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 120])

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/efficiency_vs_processes.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR}/efficiency_vs_processes.png")

def plot_algorithm_comparison(data):
    """Plot comparison of all algorithms for fixed process counts."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    process_counts = [1, 2, 4, 8]
    algorithms = list(data.keys())

    colors = {'ROW': 'blue', 'COLUMN': 'red', 'BLOCK': 'green'}

    for idx, processes in enumerate(process_counts):
        ax = axes[idx // 2, idx % 2]

        for algo in algorithms:
            sizes = []
            times = []

            for size in sorted(data[algo].keys()):
                if processes in data[algo][size]['execution_times']:
                    times_data = data[algo][size]['execution_times'][processes]
                    if times_data:
                        sizes.append(size)
                        times.append(np.mean(times_data))

            if sizes:
                ax.plot(sizes, times, marker='o', label=algo,
                        color=colors.get(algo, 'black'), linewidth=2)

        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'Algorithm Comparison\n{processes} Process(es)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR}/algorithm_comparison.png")

def plot_scalability_analysis(data):
    """Plot scalability analysis for large matrix size."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    large_size = 20000

    ax1 = axes[0]
    for algo in data.keys():
        if large_size in data[algo]:
            speedups_data = data[algo][large_size]['speedups']
            processes = sorted([p for p in speedups_data.keys() if p > 1 and speedups_data[p]])

            if processes:
                speedups = [np.mean(speedups_data[p]) for p in processes]
                ax1.plot(processes, speedups, marker='o', label=algo, linewidth=2)

    ideal_processes = list(range(1, 9))
    ideal_speedup = ideal_processes
    ax1.plot(ideal_processes, ideal_speedup, 'k--', label='Ideal', linewidth=2, alpha=0.7)

    ax1.set_xlabel('Number of Processes')
    ax1.set_ylabel('Speedup')
    ax1.set_title(f'Speedup Comparison\nMatrix Size {large_size}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for algo in data.keys():
        if large_size in data[algo]:
            efficiencies_data = data[algo][large_size]['efficiencies']
            processes = sorted([p for p in efficiencies_data.keys() if p > 1 and efficiencies_data[p]])

            if processes:
                efficiencies = [np.mean(efficiencies_data[p]) for p in processes]
                ax2.plot(processes, efficiencies, marker='o', label=algo, linewidth=2)

    ax2.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='100% efficiency')
    ax2.set_xlabel('Number of Processes')
    ax2.set_ylabel('Efficiency (%)')
    ax2.set_title(f'Efficiency Comparison\nMatrix Size {large_size}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 120])

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/scalability_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR}/scalability_analysis.png")

def print_statistics(data):
    """Print statistical analysis."""
    print("\n" + "=" * 70)
    print("MATRIX MULTIPLICATION - STATISTICAL ANALYSIS")
    print("=" * 70)

    algorithms = list(data.keys())
    process_counts = [1, 2, 4, 8]

    for algo in algorithms:
        print(f"\n{algo} Algorithm:")
        print("-" * 40)

        display_sizes = [1000, 5000, 10000, 20000]

        for size in display_sizes:
            if size in data[algo]:
                print(f"\n  Matrix size: {size}")

                for processes in process_counts:
                    if processes in data[algo][size]['execution_times']:
                        times = data[algo][size]['execution_times'][processes]
                        if times:
                            avg_time = np.mean(times)
                            std_time = np.std(times)

                            if processes == 1:
                                print(f"    {processes} process:  {avg_time:.6f}s (±{std_time:.6f}s)")
                            else:
                                speedups = data[algo][size]['speedups'].get(processes, [])
                                efficiencies = data[algo][size]['efficiencies'].get(processes, [])

                                if speedups and efficiencies:
                                    avg_speedup = np.mean(speedups)
                                    avg_efficiency = np.mean(efficiencies)
                                    print(f"    {processes} processes: {avg_time:.6f}s (±{std_time:.6f}s), "
                                          f"Speedup: {avg_speedup:.4f}x, Efficiency: {avg_efficiency:.2f}%")
                                else:
                                    print(f"    {processes} processes: {avg_time:.6f}s (±{std_time:.6f}s)")

def save_statistics_to_file(data):
    """Save detailed statistics to a text file."""
    stats_file = f'{RESULTS_DIR}/statistics.txt'

    with open(stats_file, 'w') as f:
        f.write("MATRIX MULTIPLICATION - STATISTICAL ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        algorithms = list(data.keys())
        process_counts = [1, 2, 4, 8]

        for algo in algorithms:
            f.write(f"{algo} Algorithm:\n")
            f.write("-" * 40 + "\n")

            # Все размеры матриц
            sizes = sorted(data[algo].keys())

            for size in sizes:
                if size in data[algo]:
                    f.write(f"\nMatrix size: {size}\n")

                    for processes in process_counts:
                        if processes in data[algo][size]['execution_times']:
                            times = data[algo][size]['execution_times'][processes]
                            if times:
                                avg_time = np.mean(times)
                                std_time = np.std(times)

                                if processes == 1:
                                    f.write(f"  {processes} process:  {avg_time:.6f}s (±{std_time:.6f}s)\n")
                                else:
                                    speedups = data[algo][size]['speedups'].get(processes, [])
                                    efficiencies = data[algo][size]['efficiencies'].get(processes, [])

                                    if speedups and efficiencies:
                                        avg_speedup = np.mean(speedups)
                                        avg_efficiency = np.mean(efficiencies)
                                        f.write(f"  {processes} processes: {avg_time:.6f}s (±{std_time:.6f}s), "
                                                f"Speedup: {avg_speedup:.4f}x, Efficiency: {avg_efficiency:.2f}%\n")
                                    else:
                                        f.write(f"  {processes} processes: {avg_time:.6f}s (±{std_time:.6f}s)\n")

            f.write("\n" + "=" * 70 + "\n\n")

    print(f"Saved: {stats_file}")

def main():
    """Main analysis execution."""
    print("Loading results from CSV...")
    data = load_results()

    if not data:
        print("No data found in results file!")
        sys.exit(1)

    print(f"Loaded data for {len(data)} algorithms")

    ensure_results_dir()

    print("\nGenerating graphs...")
    plot_execution_time_vs_size(data)
    plot_speedup_vs_processes(data)
    plot_efficiency_vs_processes(data)
    plot_algorithm_comparison(data)
    plot_scalability_analysis(data)

    print_statistics(data)
    save_statistics_to_file(data)

    print("\n" + "=" * 70)
    print("Matrix analysis complete!")
    print(f"All results saved to: {RESULTS_DIR}/")
    print("=" * 70)

if __name__ == "__main__":
    main()