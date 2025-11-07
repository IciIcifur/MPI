#!/usr/bin/env python3
"""
Analysis script for PI calculation benchmark results.
"""

import os
import csv
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_FILE = "results/pi_results/pi_results.csv"
RESULTS_DIR = "results/pi_results"

def ensure_results_dir():
    """Create results directory if needed."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

def load_results():
    """Load results from CSV file with proper iteration grouping."""
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found")
        sys.exit(1)

    data = {}

    with open(RESULTS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row['algorithm']
            processes = int(row['num_processes'])
            iterations = int(row['num_iterations'])

            if algo not in data:
                data[algo] = {}
            if iterations not in data[algo]:
                data[algo][iterations] = {
                    'execution_times': {},
                    'pi_values': {},
                    'speedups': {},
                    'efficiencies': {}
                }

            if processes not in data[algo][iterations]['execution_times']:
                data[algo][iterations]['execution_times'][processes] = []
            if processes not in data[algo][iterations]['pi_values']:
                data[algo][iterations]['pi_values'][processes] = []
            if processes not in data[algo][iterations]['speedups']:
                data[algo][iterations]['speedups'][processes] = []
            if processes not in data[algo][iterations]['efficiencies']:
                data[algo][iterations]['efficiencies'][processes] = []

            exec_time = float(row['execution_time'])
            data[algo][iterations]['execution_times'][processes].append(exec_time)

            try:
                pi_val = float(row['pi_value'])
                data[algo][iterations]['pi_values'][processes].append(pi_val)
            except Exception:
                pass

            if row.get('speedup') and row['speedup'].strip():
                try:
                    speedup = float(row['speedup'])
                    data[algo][iterations]['speedups'][processes].append(speedup)
                except ValueError:
                    pass

            if row.get('efficiency') and row['efficiency'].strip():
                try:
                    efficiency = float(row['efficiency'])
                    data[algo][iterations]['efficiencies'][processes].append(efficiency)
                except ValueError:
                    pass

            if processes == 1:
                if not data[algo][iterations]['speedups'][processes]:
                    data[algo][iterations]['speedups'][processes].append(1.0)
                if not data[algo][iterations]['efficiencies'][processes]:
                    data[algo][iterations]['efficiencies'][processes].append(100.0)

    return data

def compute_precision_digits(abs_error):
    """Return estimated correct decimal digits given absolute error."""
    if abs_error <= 0.0:
        return ">=15"
    if abs_error >= 1.0:
        return 0
    try:
        digits = int(math.floor(-math.log10(abs_error)))
        if digits < 0:
            digits = 0
        return digits
    except Exception:
        return 0

def plot_execution_time(data):
    """Plot execution time vs number of processes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    iterations_list = [1000000, 10000000, 100000000]

    for idx, iterations in enumerate(iterations_list):
        ax = axes[idx]

        for algo in data.keys():
            if iterations not in data[algo]:
                continue

            processes_data = data[algo][iterations]['execution_times']
            processes = sorted(processes_data.keys())
            times = [np.mean(processes_data[p]) for p in processes]

            ax.plot(processes, times, marker='o', label=algo, linewidth=2)

        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'Execution Time ({iterations:,} iterations)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/execution_time.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR}/execution_time.png")

def plot_speedup(data):
    """Plot speedup vs number of processes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    iterations_list = [1000000, 10000000, 100000000]

    for idx, iterations in enumerate(iterations_list):
        ax = axes[idx]

        for algo in data.keys():
            if iterations not in data[algo]:
                continue

            speedups_data = data[algo][iterations]['speedups']
            processes = sorted([p for p in speedups_data.keys() if p > 1 and speedups_data[p]])

            if processes:
                speedups = [np.mean(speedups_data[p]) for p in processes]
                ax.plot(processes, speedups, marker='o', label=f"{algo} (actual)", linewidth=2)

        max_process = 8
        ideal_processes = list(range(1, max_process + 1))
        ideal_speedup = ideal_processes
        ax.plot(ideal_processes, ideal_speedup, 'k--', label='Ideal', linewidth=2)

        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Speedup')
        ax.set_title(f'Speedup ({iterations:,} iterations)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/speedup.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR}/speedup.png")

def plot_efficiency(data):
    """Plot efficiency vs number of processes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    iterations_list = [1000000, 10000000, 100000000]

    for idx, iterations in enumerate(iterations_list):
        ax = axes[idx]

        for algo in data.keys():
            if iterations not in data[algo]:
                continue

            efficiencies_data = data[algo][iterations]['efficiencies']
            processes = sorted([p for p in efficiencies_data.keys() if p > 1 and efficiencies_data[p]])

            if processes:
                efficiencies = [np.mean(efficiencies_data[p]) for p in processes]
                ax.plot(processes, efficiencies, marker='o', label=algo, linewidth=2)

        ax.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='100% efficiency')
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Efficiency (%)')
        ax.set_title(f'Efficiency ({iterations:,} iterations)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 120])

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/efficiency.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR}/efficiency.png")

def print_statistics(data):
    """Print statistical analysis grouped by iterations."""
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    iterations_list = [1000000, 10000000, 100000000]

    for algo in sorted(data.keys()):
        print(f"\n{algo} Algorithm:")
        print("-" * 40)

        for iterations in iterations_list:
            if iterations not in data[algo]:
                continue

            print(f"\n  {iterations:,} iterations:")

            iterations_data = data[algo][iterations]

            for processes in sorted(iterations_data['pi_values'].keys()):
                pi_vals = iterations_data['pi_values'][processes]
                if not pi_vals:
                    continue
                avg_pi = float(np.mean(pi_vals))
                std_pi = float(np.std(pi_vals))
                abs_err = abs(avg_pi - math.pi)
                rel_err_pct = (abs_err / math.pi) * 100.0 if math.pi != 0 else float('inf')
                digits = compute_precision_digits(abs_err)

                if std_pi > 0:
                    pi_str = f"{avg_pi:.15f} (±{std_pi:.15f})"
                else:
                    pi_str = f"{avg_pi:.15f}"

                print(f"    [{processes} proc] Pi: {pi_str} | Abs err={abs_err:.15g} | Rel err={rel_err_pct:.6f}% | Prec ≈ {digits} dec. digits")

            for processes in sorted(iterations_data['execution_times'].keys()):
                times = iterations_data['execution_times'][processes]
                avg_time = np.mean(times)
                std_time = np.std(times)

                if std_time > 0:
                    time_str = f"{avg_time:.6f}s (±{std_time:.6f}s)"
                else:
                    time_str = f"{avg_time:.6f}s"

                speedups = iterations_data['speedups'].get(processes, [])
                efficiencies = iterations_data['efficiencies'].get(processes, [])

                if speedups and efficiencies:
                    avg_speedup = np.mean(speedups)
                    avg_efficiency = np.mean(efficiencies)
                    print(f"    {processes} process{'es' if processes>1 else ''}: {time_str}, "
                          f"Speedup: {avg_speedup:.4f}x, Efficiency: {avg_efficiency:.2f}%")
                else:
                    print(f"    {processes} process{'es' if processes>1 else ''}: {time_str}")

def plot_comparison_summary(data):
    """Create a summary comparison plot for all algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    iterations_list = [1000000, 10000000, 100000000]

    ax = axes[0, 0]
    for algo in data.keys():
        avg_times = []
        for iterations in iterations_list:
            if iterations in data[algo]:
                times_1proc = data[algo][iterations]['execution_times'].get(1, [])
                if times_1proc:
                    avg_times.append(np.mean(times_1proc))

        if avg_times:
            ax.plot([f"{it//1000000}M" for it in iterations_list], avg_times,
                    marker='o', label=algo, linewidth=2)

    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Single Process Execution Time\nby Iteration Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    iterations = 100000000
    for algo in data.keys():
        if iterations in data[algo]:
            speedups_data = data[algo][iterations]['speedups']
            processes = sorted([p for p in speedups_data.keys() if p > 1 and speedups_data[p]])
            if processes:
                speedups = [np.mean(speedups_data[p]) for p in processes]
                ax.plot(processes, speedups, marker='o', label=algo, linewidth=2)

    ideal_processes = list(range(1, 9))
    ideal_speedup = ideal_processes
    ax.plot(ideal_processes, ideal_speedup, 'k--', label='Ideal', linewidth=2)

    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Speedup')
    ax.set_title(f'Speedup Comparison\n({iterations:,} iterations)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    iterations = 100000000
    for algo in data.keys():
        if iterations in data[algo]:
            efficiencies_data = data[algo][iterations]['efficiencies']
            processes = sorted([p for p in efficiencies_data.keys() if p > 1 and efficiencies_data[p]])
            if processes:
                efficiencies = [np.mean(efficiencies_data[p]) for p in processes]
                ax.plot(processes, efficiencies, marker='o', label=algo, linewidth=2)

    ax.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='100% efficiency')
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title(f'Efficiency Comparison\n({iterations:,} iterations)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 120])

    ax = axes[1, 1]
    for algo in data.keys():
        pi_errors = []
        for iterations in iterations_list:
            if iterations in data[algo]:
                pi_values = data[algo][iterations]['pi_values'].get(1, [])
                if pi_values:
                    avg_pi = np.mean(pi_values)
                    error = abs(avg_pi - math.pi) / math.pi * 100
                    pi_errors.append(error)

        if pi_errors:
            ax.plot([f"{it//1000000}M" for it in iterations_list], pi_errors,
                    marker='o', label=algo, linewidth=2)

    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('PI Calculation Accuracy\n(Relative to π)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/comparison_summary.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR}/comparison_summary.png")

def save_statistics_to_file(data):
    """Save detailed statistics to a text file, including pi estimates and precision."""
    stats_file = f'{RESULTS_DIR}/statistics.txt'

    with open(stats_file, 'w') as f:
        f.write("PI CALCULATION BENCHMARK - STATISTICAL ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        iterations_list = [1000000, 10000000, 100000000]

        for algo in sorted(data.keys()):
            f.write(f"{algo} Algorithm:\n")
            f.write("-" * 40 + "\n")

            for iterations in iterations_list:
                if iterations not in data[algo]:
                    continue

                f.write(f"\n{iterations:,} iterations:\n")

                iterations_data = data[algo][iterations]

                for processes in sorted(iterations_data['pi_values'].keys()):
                    pi_vals = iterations_data['pi_values'][processes]
                    if not pi_vals:
                        continue
                    avg_pi = float(np.mean(pi_vals))
                    std_pi = float(np.std(pi_vals))
                    abs_err = abs(avg_pi - math.pi)
                    rel_err_pct = (abs_err / math.pi) * 100.0 if math.pi != 0 else float('inf')
                    digits = compute_precision_digits(abs_err)

                    if std_pi > 0:
                        pi_str = f"{avg_pi:.15f} (±{std_pi:.15f})"
                    else:
                        pi_str = f"{avg_pi:.15f}"

                    f.write(f"  [{processes} proc] Pi: {pi_str} | Abs err={abs_err:.15g} | Rel err={rel_err_pct:.6f}% | Prec ≈ {digits} dec. digits\n")

                for processes in sorted(iterations_data['execution_times'].keys()):
                    times = iterations_data['execution_times'][processes]
                    avg_time = np.mean(times)
                    std_time = np.std(times)

                    if std_time > 0:
                        time_str = f"{avg_time:.6f}s (±{std_time:.6f}s)"
                    else:
                        time_str = f"{avg_time:.6f}s"

                    speedups = iterations_data['speedups'].get(processes, [])
                    efficiencies = iterations_data['efficiencies'].get(processes, [])

                    if speedups and efficiencies:
                        avg_speedup = np.mean(speedups)
                        avg_efficiency = np.mean(efficiencies)
                        f.write(f"  {processes} processes: {time_str}, Speedup: {avg_speedup:.4f}x, Efficiency: {avg_efficiency:.2f}%\n")
                    else:
                        f.write(f"  {processes} processes: {time_str}\n")

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
    plot_execution_time(data)
    plot_speedup(data)
    plot_efficiency(data)
    plot_comparison_summary(data)

    print_statistics(data)

    save_statistics_to_file(data)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"All results saved to: {RESULTS_DIR}/")
    print("=" * 70)

if __name__ == "__main__":
    main()