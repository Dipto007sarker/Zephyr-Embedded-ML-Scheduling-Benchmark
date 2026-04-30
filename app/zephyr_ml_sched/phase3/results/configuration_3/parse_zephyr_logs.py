#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import pandas as pd

RT_PAT = re.compile(r"\[RT\] run=(\d+) exec_ms=(-?\d+) lateness_ms=(-?\d+) gap_ms=(-?\d+) misses=(\d+)")
ML_PAT = re.compile(r"\[ML\] run=(\d+) repeats=(\d+) total_cycles=(\d+) avg_cycles=(\d+)")
SUMMARY_PAT = re.compile(r"\[SUMMARY\] ml_runs=(\d+) rt_runs=(\d+) rt_misses=(\d+)")

def infer_config(name: str) -> str:
    if "with_deadline" in name:
        return "equal_with_deadline"
    if "equal_priority" in name and "no_deadline" in name:
        return "equal_no_deadline"
    if "priority_baseline_no_deadline" in name:
        return "priority_no_deadline"
    return "unknown"

def parse_one_file(path: Path) -> dict:
    text = path.read_text(errors="ignore")

    rt = [tuple(map(int, m)) for m in RT_PAT.findall(text)]
    ml = [tuple(map(int, m)) for m in ML_PAT.findall(text)]
    su = [tuple(map(int, m)) for m in SUMMARY_PAT.findall(text)]

    rt_df = pd.DataFrame(rt, columns=["run", "exec_ms", "lateness_ms", "gap_ms", "misses"]) if rt else pd.DataFrame(columns=["run", "exec_ms", "lateness_ms", "gap_ms", "misses"])
    ml_df = pd.DataFrame(ml,columns=["run", "repeats", "total_cycles", "avg_cycles"]) if ml else pd.DataFrame(columns=["run", "repeats", "total_cycles", "avg_cycles"])

    final_ml_runs = int(su[-1][0]) if su else (int(ml_df["run"].max()) if not ml_df.empty else 0)
    final_rt_runs = int(su[-1][1]) if su else (int(rt_df["run"].max()) if not rt_df.empty else 0)
    final_rt_misses = int(su[-1][2]) if su else (int(rt_df["misses"].max()) if not rt_df.empty else 0)

    m = re.search(r"_(\d+)\.txt$", path.name)
    trial = int(m.group(1)) if m else -1

    return {
        "file": path.name,
        "config": infer_config(path.name),
        "trial": trial,
        "final_ml_runs": final_ml_runs,
        "final_rt_runs": final_rt_runs,
        "final_rt_misses": final_rt_misses,
        "avg_rt_exec_ms": rt_df["exec_ms"].mean() if not rt_df.empty else float("nan"),
        "max_rt_exec_ms": rt_df["exec_ms"].max() if not rt_df.empty else float("nan"),
        "avg_rt_lateness_ms": rt_df["lateness_ms"].mean() if not rt_df.empty else float("nan"),
        "max_rt_lateness_ms": rt_df["lateness_ms"].max() if not rt_df.empty else float("nan"),
        "avg_rt_gap_ms": rt_df["gap_ms"].mean() if not rt_df.empty else float("nan"),
        "max_rt_gap_ms": rt_df["gap_ms"].max() if not rt_df.empty else float("nan"),
        "avg_ml_cycles": ml_df["avg_cycles"].mean() if not ml_df.empty else float("nan"),
        "max_ml_cycles": ml_df["avg_cycles"].max() if not ml_df.empty else float("nan"),
        "n_rt_lines": len(rt_df),
        "n_ml_lines": len(ml_df),
    }

def main():
    ap = argparse.ArgumentParser(description="Parse Zephyr scheduling benchmark logs into CSV summaries.")
    ap.add_argument("--input-dir", default=".", help="Directory containing run_*.txt log files")
    ap.add_argument("--pattern", default="run_*.txt", help="Glob pattern for input logs")
    ap.add_argument("--trial-csv", default="zephyr_trial_summary.csv", help="Output CSV for per-trial rows")
    ap.add_argument("--config-csv", default="zephyr_config_summary.csv", help="Output CSV for per-config summary")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched pattern {args.pattern!r} in {input_dir}")

    trial_rows = [parse_one_file(f) for f in files]
    trial_df = pd.DataFrame(trial_rows).sort_values(["config", "trial"]).reset_index(drop=True)

    agg = trial_df.groupby("config").agg(
        trials=("trial", "count"),
        mean_final_ml_runs=("final_ml_runs", "mean"),
        std_final_ml_runs=("final_ml_runs", "std"),
        mean_final_rt_runs=("final_rt_runs", "mean"),
        std_final_rt_runs=("final_rt_runs", "std"),
        mean_final_rt_misses=("final_rt_misses", "mean"),
        std_final_rt_misses=("final_rt_misses", "std"),
        mean_avg_rt_exec_ms=("avg_rt_exec_ms", "mean"),
        std_avg_rt_exec_ms=("avg_rt_exec_ms", "std"),
        mean_avg_rt_lateness_ms=("avg_rt_lateness_ms", "mean"),
        std_avg_rt_lateness_ms=("avg_rt_lateness_ms", "std"),
        mean_max_rt_lateness_ms=("max_rt_lateness_ms", "mean"),
        mean_avg_rt_gap_ms=("avg_rt_gap_ms", "mean"),
        std_avg_rt_gap_ms=("avg_rt_gap_ms", "std"),
        mean_max_rt_gap_ms=("max_rt_gap_ms", "mean"),
        mean_avg_ml_cycles=("avg_ml_cycles", "mean"),
        std_avg_ml_cycles=("avg_ml_cycles", "std"),
    ).reset_index()

    # keep raw values in CSV, not rounded, for later plotting/statistics
    trial_df.to_csv(args.trial_csv, index=False)
    agg.to_csv(args.config_csv, index=False)

    print(f"Wrote per-trial summary: {args.trial_csv}")
    print(f"Wrote per-config summary: {args.config_csv}")
    print("\nConfiguration summary:")
    print(agg.to_string(index=False))

if __name__ == "__main__":
    main()
