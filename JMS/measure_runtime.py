import cProfile
import pstats
import time
import os
import runpy
from datetime import datetime

# ===== Configuration =====
TARGET_SCRIPT = r"JMS/main_script_parallel_grasp_valid_save_candidate_pt.py"
N_RUNS = 45                               # Number of runs
OUTPUT_DIR = os.path.join("results", "runtime_record")
SHOW_TOP = 20                              # Show only the top N most time-consuming custom functions
# =========================

script_abs_path = os.path.abspath(TARGET_SCRIPT)

def is_own_function(key):
    """
    key: (filename, lineno, funcname)
    Keep only functions/methods defined in the target script file.
    """
    filename, _, _ = key
    try:
        return os.path.abspath(filename) == script_abs_path
    except Exception:
        return False

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Containers for aggregated results
run_times = []
all_stats_summary = []

print(f"Starting {N_RUNS} benchmark runs...\n")

for run_idx in range(1, N_RUNS + 1):
    pr = cProfile.Profile()
    start_time = time.time()

    try:
        pr.enable()
        runpy.run_path(TARGET_SCRIPT, run_name="__main__")
        pr.disable()
    except SystemExit:
        # Handle sys.exit() in the target script gracefully
        pr.disable()
    finally:
        total_time = time.time() - start_time

    run_times.append(total_time)

    # Extract cumulative time for custom functions in this run
    stats = pstats.Stats(pr)
    own_functions = {k: v for k, v in stats.stats.items() if is_own_function(k)}
    sorted_funcs = sorted(
        own_functions.items(),
        key=lambda kv: kv[1][3],  # kv[1] = (cc, nc, tt, ct, callers); [3] is cumulative time
        reverse=True
    )
    all_stats_summary.append((total_time, sorted_funcs))

    print(f"Run {run_idx:02d}/{N_RUNS}  Elapsed: {total_time:.6f} s")

# ===== Generate Summary Report =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(OUTPUT_DIR, f"benchmark_{N_RUNS}runs_{timestamp}.txt")

avg_time = sum(run_times) / len(run_times)
min_time = min(run_times)
max_time = max(run_times)

# Standard deviation
variance = sum((t - avg_time) ** 2 for t in run_times) / len(run_times)
std_dev = variance ** 0.5

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"{'='*60}\n")
    f.write(f"Benchmark Report — {N_RUNS} Runs\n")
    f.write(f"Target script: {TARGET_SCRIPT}\n")
    f.write(f"{'='*60}\n\n")

    # Overall statistics
    f.write(f"[Overall Statistics]\n")
    f.write(f"  Average time: {avg_time:.6f} s\n")
    f.write(f"  Min time:     {min_time:.6f} s  (Run {run_times.index(min_time)+1})\n")
    f.write(f"  Max time:     {max_time:.6f} s  (Run {run_times.index(max_time)+1})\n")
    f.write(f"  Std deviation:{std_dev:.6f} s\n\n")

    # Per-run breakdown
    f.write(f"[Per-Run Elapsed Times]\n")
    for i, t in enumerate(run_times, 1):
        marker = " <- min" if t == min_time else (" <- max" if t == max_time else "")
        f.write(f"  Run {i:02d}: {t:.6f} s{marker}\n")

    # Top N custom functions per run
    f.write(f"\n{'='*60}\n")
    f.write(f"[Top {SHOW_TOP} Custom Functions by Cumulative Time — Per Run]\n")
    for run_idx, (total_time, sorted_funcs) in enumerate(all_stats_summary, 1):
        f.write(f"\n--- Run {run_idx:02d}  Total: {total_time:.6f} s ---\n")
        for idx, (func_key, stat) in enumerate(sorted_funcs[:SHOW_TOP], start=1):
            filename, lineno, funcname = func_key
            cc, nc, tt, ct, callers = stat
            f.write(
                f"  {idx:02d}. {funcname}  cumtime: {ct:.6f} s  "
                f"calls: {nc}  {os.path.basename(filename)}:{lineno}\n"
            )

print(f"\n{'='*60}")
print(f"Benchmark complete! {N_RUNS} runs summary:")
print(f"  Average time:  {avg_time:.6f} s")
print(f"  Min time:      {min_time:.6f} s")
print(f"  Max time:      {max_time:.6f} s")
print(f"  Std deviation: {std_dev:.6f} s")
print(f"Report saved to: {output_path}")