#!/usr/bin/env python3
"""
Analysis and plotting for Cannon's algorithm benchmark results.

Reads results from results/cannon_results/cannon_results.csv and generates:
 - execution_time_vs_size.png
 - speedup_vs_processes.png
 - efficiency_vs_processes.png
 - statistics.txt
"""
import os
import csv
import sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

RESULTS_FILE = "results/cannon_results/cannon_results.csv"
RESULTS_DIR = "results/cannon_results"

def ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

def load_results():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found")
        sys.exit(1)

    data_time = defaultdict(lambda: defaultdict(list))
    data_speedup = defaultdict(lambda: defaultdict(list))
    data_eff = defaultdict(lambda: defaultdict(list))
    checks = defaultdict(lambda: defaultdict(list))

    with open(RESULTS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                p = int(row['num_processes'])
                n = int(row['matrix_size'])
                t = float(row['avg_time'])
            except Exception:
                continue

            data_time[n][p].append(t)

            try:
                s = float(row['speedup']) if row.get('speedup') else None
                if s is not None:
                    data_speedup[n][p].append(s)
            except Exception:
                pass

            try:
                e = float(row['efficiency']) if row.get('efficiency') else None
                if e is not None:
                    data_eff[n][p].append(e)
            except Exception:
                pass

            ch = row.get('check', '').strip()
            checks[n][p].append(ch)

    return data_time, data_speedup, data_eff, checks

def plot_execution_time_vs_size(data_time):
    plt.figure(figsize=(10,6))
    process_counts = sorted({p for n in data_time for p in data_time[n]})
    markers = ['o','s','^','D','v','<','>','x']
    for i, p in enumerate(process_counts):
        sizes = sorted([n for n in data_time if p in data_time[n]])
        times = [np.mean(data_time[n][p]) for n in sizes]
        plt.plot(sizes, times, marker=markers[i%len(markers)], label=f'{p} proc', linewidth=2)

    plt.xlabel('Matrix size (N)')
    plt.ylabel('Average time (s)')
    plt.title("Cannon: Execution time vs Matrix size")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(RESULTS_DIR, 'execution_time_vs_size.png')
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

def plot_speedup_vs_processes(data_speedup):
    plt.figure(figsize=(10,6))
    sizes_all = sorted(data_speedup.keys())
    characteristic_sizes = []
    for target in [64, 128, 256, 512, 1024, 2048]:
        candidates = [s for s in sizes_all if abs(s - target) <= target*0.2]
        if candidates:
            characteristic_sizes.append(candidates[0])
    if not characteristic_sizes:
        characteristic_sizes = sizes_all[-4:]

    markers = ['o','s','^','D','v','<','>','x']
    for i, n in enumerate(characteristic_sizes):
        processes = sorted([p for p in data_speedup[n].keys() if p > 1])
        if not processes:
            continue
        speedups = [np.mean(data_speedup[n][p]) for p in processes]
        plt.plot(processes, speedups, marker=markers[i%len(markers)], label=f'N={n}', linewidth=2)

    if plt.gca().get_xlim()[1] < 1:
        ideal_max = max([p for n in data_speedup for p in data_speedup[n].keys()] + [1])
    else:
        ideal_max = int(max([p for n in data_speedup for p in data_speedup[n].keys()] + [1]))
    ideal = list(range(1, ideal_max+1))
    plt.plot(ideal, ideal, 'k--', label='Ideal', linewidth=1)

    plt.xlabel('Number of processes')
    plt.ylabel('Speedup')
    plt.title("Cannon: Speedup vs Processes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = os.path.join(RESULTS_DIR, 'speedup_vs_processes.png')
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

def plot_efficiency_vs_processes(data_eff):
    plt.figure(figsize=(10,6))
    sizes_all = sorted(data_eff.keys())
    characteristic_sizes = sizes_all[-4:] if sizes_all else []
    markers = ['o','s','^','D','v','<','>','x']
    for i, n in enumerate(characteristic_sizes):
        processes = sorted([p for p in data_eff[n].keys() if p > 1])
        if not processes:
            continue
        effs = [np.mean(data_eff[n][p]) for p in processes]
        plt.plot(processes, effs, marker=markers[i%len(markers)], label=f'N={n}', linewidth=2)

    plt.axhline(100, color='k', linestyle='--', alpha=0.5, label='100%')
    plt.xlabel('Number of processes')
    plt.ylabel('Efficiency (%)')
    plt.title("Cannon: Efficiency vs Processes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = os.path.join(RESULTS_DIR, 'efficiency_vs_processes.png')
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

def print_and_save_statistics(data_time, data_speedup, data_eff, checks):
    stats_file = os.path.join(RESULTS_DIR, 'statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("CANNON ALGORITHM - STATISTICAL ANALYSIS\n")
        f.write("="*70 + "\n")
        sizes = sorted(data_time.keys())
        for n in sizes:
            f.write(f"\nMatrix size: {n}\n")
            f.write("-"*40 + "\n")
            processes = sorted(data_time[n].keys())
            for p in processes:
                times = data_time[n][p]
                if not times:
                    continue
                avg_t = np.mean(times)
                std_t = np.std(times)
                line = f"  {p:3d} proc: {avg_t:.6f}s (±{std_t:.6f}s)"
                sp_list = data_speedup[n].get(p, [])
                ef_list = data_eff[n].get(p, [])
                if sp_list and ef_list:
                    avg_sp = np.mean(sp_list)
                    avg_ef = np.mean(ef_list)
                    line += f", Speedup={avg_sp:.4f}x, Eff={avg_ef:.2f}%"
                chk = checks[n].get(p, [])
                if chk:
                    from collections import Counter
                    most = Counter(chk).most_common(1)[0][0]
                    line += f", Check={most}"
                f.write(line + "\n")
    print(f"Saved: {stats_file}")

def main():
    print("Loading cannon results...")
    data_time, data_speedup, data_eff, checks = load_results()
    if not data_time:
        print("No data loaded.")
        sys.exit(1)

    ensure_results_dir()

    print("Generating plots...")
    plot_execution_time_vs_size(data_time)
    plot_speedup_vs_processes(data_speedup)
    plot_efficiency_vs_processes(data_eff)

    print("Saving statistics...")
    print_and_save_statistics(data_time, data_speedup, data_eff, checks)

    print("\nCannon analysis complete. Results saved to:", RESULTS_DIR)

if __name__ == "__main__":
    main()