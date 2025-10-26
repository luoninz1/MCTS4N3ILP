import os
import sys
import csv
import argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def read_summary(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


essential_fields = [
    "n",
    "count_points","min_points","p50_points","p90_points","p95_points","max_points","mean_points","std_points",
    "count_time","min_time","p50_time","p90_time","p95_time","max_time","mean_time","std_time",
]


def main():
    parser = argparse.ArgumentParser(description="Aggregate multiple baseline summary.csv files into one CSV")
    parser.add_argument("inputs", nargs="+", help="Paths to summary.csv files to aggregate")
    parser.add_argument("--output-root", default=os.path.join("results","baseline","pre_refactor"))
    args = parser.parse_args()

    all_rows = []
    for p in args.inputs:
        if not os.path.exists(p):
            print(f"Warning: missing {p}, skipping")
            continue
        rows = read_summary(p)
        # Keep only columns of interest; assume one row per n per file
        for r in rows:
            all_rows.append({k: r.get(k, "") for k in essential_fields})

    # Sort by n
    try:
        all_rows.sort(key=lambda r: int(float(r["n"])) )
    except Exception:
        pass

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = ensure_dir(os.path.join(ROOT, args.output_root))
    out_path = os.path.join(out_dir, f"combined_summary_{ts}.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=essential_fields)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"Combined summary written to: {out_path}")


if __name__ == "__main__":
    main()
