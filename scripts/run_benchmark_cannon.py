#!/usr/bin/env python3
"""
Benchmark runner for Cannon's algorithm (task 3).
"""
import os
import subprocess
import csv
import sys
from datetime import datetime

RESULTS_DIR = "results/cannon_results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "cannon_results.csv")

EXECUTABLE_ENV = os.environ.get("CANNON_EXEC", "").strip()
CANDIDATES = []
if EXECUTABLE_ENV:
    CANDIDATES.append(EXECUTABLE_ENV)
CANDIDATES += ["bin/main", "bin/pi_calculator", "bin/pi_calculator.exe", "bin/pi_calculator.out"]

def find_executable():
    for c in CANDIDATES:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    if os.path.isdir("bin"):
        for fname in sorted(os.listdir("bin")):
            path = os.path.join("bin", fname)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
    return None

EXECUTABLE = find_executable()

PROCESS_COUNTS = [1, 4, 9, 16]

MPIRUN_OPTS = os.environ.get("MPIRUN_OPTS", "").strip()
if not MPIRUN_OPTS:
    MPIRUN_OPTS = "--oversubscribe"

RUN_TIMEOUT = int(os.environ.get("CANNON_RUN_TIMEOUT", "1200"))  # seconds

def ensure_directories():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

def run_cannon(num_processes):
    if not EXECUTABLE:
        print("ERROR: No executable found. Set CANNON_EXEC or put an executable in bin/ (e.g. bin/main or bin/pi_calculator).")
        return None

    print(f"Running Cannon: {num_processes} processes")
    cmd = ["mpirun"] + MPIRUN_OPTS.split() + ["-np", str(num_processes), EXECUTABLE, "3"]

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT: run exceeded timeout")
        return None
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return None

    if completed.returncode != 0:
        print(f"  ERROR: non-zero exit code. stderr:\n{completed.stderr.strip()}")

    out_lines = (completed.stdout or "").splitlines()

    parsed = []
    for line in out_lines:
        if '|' not in line:
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 6:
            continue
        try:
            n = int(parts[0])
            p = int(parts[1])
            if len(parts) >= 7:
                seq_time = float(parts[2])
                avg_time = float(parts[3])
                speedup = float(parts[4])
                efficiency = float(parts[5])
                check = parts[6]
            else:
                avg_time = float(parts[2])
                speedup = float(parts[3]) if len(parts) > 3 else 0.0
                efficiency = float(parts[4]) if len(parts) > 4 else 0.0
                check = parts[5] if len(parts) > 5 else ''
        except Exception:
            # malformed numeric conversion => skip
            continue
        parsed.append({
            'matrix_size': n,
            'processes': p,
            'avg_time': avg_time,
            'speedup': speedup,
            'efficiency': efficiency,
            'check': check
        })

    if not parsed:
        if completed.stdout:
            print("  STDOUT:")
            print(completed.stdout)
        if completed.stderr:
            print("  STDERR:")
            print(completed.stderr)
        return None

    return parsed

def main():
    print("="*70)
    print("CANNON'S ALGORITHM BENCHMARK (TASK 3)")
    print(f"Using executable: {EXECUTABLE if EXECUTABLE else '(none found)'}")
    print(f"mpirun options: {MPIRUN_OPTS}")
    print("Results will be saved to:", RESULTS_FILE)
    print("="*70)

    if not EXECUTABLE:
        print("Please compile the project and ensure an executable is available in bin/,")
        print("or set environment variable CANNON_EXEC to point to the MPI binary.")
        sys.exit(1)

    ensure_directories()

    csv_header = [
        'timestamp',
        'num_processes',
        'matrix_size',
        'avg_time',
        'speedup',
        'efficiency',
        'check'
    ]

    with open(RESULTS_FILE, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()

        for p in PROCESS_COUNTS:
            if p <= 0:
                continue
            results = run_cannon(p)
            if results is None:
                print(f"  Skipping process count {p} due to error or no output.")
                continue

            ts = datetime.now().isoformat()
            for entry in results:
                if entry['processes'] == 1:
                    speedup_str = "1.000000"
                    efficiency_str = "100.000000"
                else:
                    speedup_str = f"{entry['speedup']:.6f}"
                    efficiency_str = f"{entry['efficiency']:.6f}"

                row = {
                    'timestamp': ts,
                    'num_processes': entry['processes'],
                    'matrix_size': entry['matrix_size'],
                    'avg_time': f"{entry['avg_time']:.6f}",
                    'speedup': speedup_str,
                    'efficiency': efficiency_str,
                    'check': entry['check']
                }
                writer.writerow(row)
                csvfile.flush()
                try:
                    sp_display = float(speedup_str)
                    ef_display = float(efficiency_str)
                    print(f"  N={entry['matrix_size']:5d} | P={entry['processes']:3d} | "
                          f"Time={entry['avg_time']:.6f}s | S={sp_display:.3f} | E={ef_display:.2f}% | {entry['check']}")
                except Exception:
                    print(f"  N={entry['matrix_size']:5d} | P={entry['processes']:3d} | Time={entry['avg_time']:.6f}s | {entry['check']}")

    print("\nBenchmark finished. CSV saved to:", RESULTS_FILE)

if __name__ == "__main__":
    main()