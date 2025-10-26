import os
import sys
import csv
import json
import time
import uuid
import argparse
import platform
import subprocess
from datetime import datetime

# Ensure src is importable
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from src.algos.mcts import evaluate, MCTS
import numpy as np


def get_git_info():
    def run(cmd):
        try:
            out = subprocess.check_output(cmd, shell=True, cwd=ROOT).decode().strip()
            return out
        except Exception:
            return None
    return {
        "git_commit": run("git --no-pager rev-parse --short HEAD"),
        "branch": run("git rev-parse --abbrev-ref HEAD"),
    }


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def summarize(values):
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    def pct(p):
        k = int(round((p/100.0)*(n-1)))
        return s[k]
    return {
        "count": n,
        "min": float(s[0]),
        "p50": float(pct(50)),
        "p90": float(pct(90)),
        "p95": float(pct(95)),
        "max": float(s[-1]),
        "mean": float(sum(s)/n),
        "std": float(np.std(s, ddof=0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-refactor baseline runner")
    parser.add_argument("--n", nargs="+", type=int, default=[6], help="Grid sizes (n) to test; default: 6")
    parser.add_argument("--repeats", type=int, default=20, help="Runs per n; default: 20")
    parser.add_argument("--env", type=str, default="N3il_with_symmetry", choices=["N3il","N3il_with_symmetry"], help="Environment")
    parser.add_argument("--algo", type=str, default="MCTS", choices=["MCTS","ParallelMCTS","LeafChildParallelMCTS","MCGS"], help="Algorithm")
    parser.add_argument("--base-seed", type=int, default=12345, help="Base seed for cohort")
    parser.add_argument("--node-compression", action="store_true", default=True, help="Enable node compression where supported")
    parser.add_argument("--output-root", type=str, default="results/baseline/pre_refactor", help="Output directory root (gitignored)")
    parser.add_argument("--allow-large-n", action="store_true", help="Bypass the n<=10 test constraint for ad-hoc baselines (e.g., n=20)")
    args = parser.parse_args()

    # Build output directory name with timestamp + short commit
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    git_info = get_git_info()
    short_commit = git_info.get("git_commit") or "nogit"
    cohort_dir = ensure_dir(os.path.join(ROOT, args.output_root, f"{ts}_{short_commit}"))

    # IDs
    cohort_id = uuid.uuid4().hex  # UUID for simplicity; ULID not required as dependency

    # Environment metadata
    env_meta = {
        **git_info,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    }
    try:
        import numba
        env_meta["numba_version"] = numba.__version__
    except Exception:
        env_meta["numba_version"] = None
    env_meta["platform"] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    # Persist metadata/config
    with open(os.path.join(cohort_dir, "environment.json"), "w") as f:
        json.dump(env_meta, f, indent=2)
    config_payload = {
        "n_list": args.n,
        "repeats": args.repeats,
        "environment": args.env,
        "algorithm": args.algo,
        "node_compression": args.node_compression,
        "constraints": {
            "max_n": 10,
            "num_searches_rule": "<= 5 * n^2",
            "replications": ">= 20",
            "allow_large_n_override": bool(getattr(args, "allow_large_n", False)),
        },
    }
    with open(os.path.join(cohort_dir, "config.json"), "w") as f:
        json.dump(config_payload, f, indent=2)

    seeds_payload = {
        "base_seed": args.base_seed,
        "worker_seed_strategy": "base_seed + worker_id*10000",
        "numba_warmup": True,
    }
    with open(os.path.join(cohort_dir, "seeds.json"), "w") as f:
        json.dump(seeds_payload, f, indent=2)

    # Prepare CSV writers
    runs_path = os.path.join(cohort_dir, "runs.csv")
    with open(runs_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "cohort_id","run_id","timestamp",
            "git_commit","branch","python_version","numpy_version","numba_version","platform_system",
            "environment","algorithm","n","num_searches","C","virtual_loss","node_compression","child_parallel","simulations_per_leaf","symmetry_enabled","priority_enabled",
            "base_seed","run_seed","time_seconds","terminal_num_points"
        ])
        writer.writeheader()

        # Iterate Ns and runs
        for n in args.n:
            if n > 10 and not args.allow_large_n:
                print(f"Skipping n={n} (exceeds test constraint n<=10). Use --allow-large-n to override for baseline runs.")
                continue
            num_searches = min(5 * (n ** 2), 5 * (n ** 2))  # enforce rule
            for i in range(args.repeats):
                run_id = uuid.uuid4().hex
                run_seed = args.base_seed + i

                eval_args = {
                    'environment': args.env,
                    'algorithm': args.algo,
                    'node_compression': bool(args.node_compression),
                    'max_level_to_use_symmetry': 1,
                    'n': n,
                    'C': 1.41,
                    'num_searches': num_searches,
                    'num_workers': 1,
                    'virtual_loss': 1.0,
                    'process_bar': False,
                    'display_state': False,
                    'logging_mode': True,
                    'TopN': n,
                    'simulate_with_priority': False,
                    'table_dir': cohort_dir,
                    'figure_dir': os.path.join(cohort_dir, 'figures'),
                    'random_seed': run_seed,
                    'tree_visualization': False,
                    'pause_at_each_step': False,
                }

                t0 = time.time()
                try:
                    num_points = evaluate(eval_args)
                except Exception as e:
                    # Ensure we still record the failure
                    num_points = float('nan')
                    print(f"Run failed for n={n}, i={i}: {e}")
                t1 = time.time()
                dur = t1 - t0

                writer.writerow({
                    "cohort_id": cohort_id,
                    "run_id": run_id,
                    "timestamp": datetime.utcnow().isoformat()+"Z",
                    "git_commit": env_meta.get("git_commit"),
                    "branch": env_meta.get("branch"),
                    "python_version": env_meta.get("python_version"),
                    "numpy_version": env_meta.get("numpy_version"),
                    "numba_version": env_meta.get("numba_version"),
                    "platform_system": env_meta["platform"].get("system"),
                    "environment": args.env,
                    "algorithm": args.algo,
                    "n": n,
                    "num_searches": num_searches,
                    "C": 1.41,
                    "virtual_loss": 1.0,
                    "node_compression": bool(args.node_compression),
                    "child_parallel": False,
                    "simulations_per_leaf": 1,
                    "symmetry_enabled": (args.env == "N3il_with_symmetry"),
                    "priority_enabled": False,
                    "base_seed": args.base_seed,
                    "run_seed": run_seed,
                    "time_seconds": f"{dur:.6f}",
                    "terminal_num_points": num_points,
                })
                csvfile.flush()
                print(f"n={n} run={i+1}/{args.repeats}: points={num_points} time={dur:.3f}s")

    # Build summary.csv
    # Group by (n)
    import collections
    rows = []
    with open(runs_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    by_n = collections.defaultdict(lambda: {"points": [], "time": []})
    for r in rows:
        try:
            n = int(r["n"])
            p = float(r["terminal_num_points"]) if r["terminal_num_points"] != "nan" else float('nan')
            t = float(r["time_seconds"]) if r["time_seconds"] != "" else float('nan')
            if not np.isnan(p):
                by_n[n]["points"].append(p)
            if not np.isnan(t):
                by_n[n]["time"].append(t)
        except Exception:
            pass

    summary_path = os.path.join(cohort_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        fieldnames = [
            "n",
            "count_points","min_points","p50_points","p90_points","p95_points","max_points","mean_points","std_points",
            "count_time","min_time","p50_time","p90_time","p95_time","max_time","mean_time","std_time",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for n, vals in sorted(by_n.items()):
            sp = summarize(vals["points"]) if vals["points"] else {}
            st = summarize(vals["time"]) if vals["time"] else {}
            writer.writerow({
                "n": n,
                "count_points": sp.get("count", 0),
                "min_points": sp.get("min", ""),
                "p50_points": sp.get("p50", ""),
                "p90_points": sp.get("p90", ""),
                "p95_points": sp.get("p95", ""),
                "max_points": sp.get("max", ""),
                "mean_points": sp.get("mean", ""),
                "std_points": sp.get("std", ""),
                "count_time": st.get("count", 0),
                "min_time": st.get("min", ""),
                "p50_time": st.get("p50", ""),
                "p90_time": st.get("p90", ""),
                "p95_time": st.get("p95", ""),
                "max_time": st.get("max", ""),
                "mean_time": st.get("mean", ""),
                "std_time": st.get("std", ""),
            })
    print(f"Baseline written to: {cohort_dir}")


if __name__ == "__main__":
    main()
