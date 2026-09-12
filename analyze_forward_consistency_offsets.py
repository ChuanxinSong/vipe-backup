import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List


METRIC_KEYS = [
    "ssim",
    # "psnr",
    # "masked_l1",
    # "coverage",
    # "depth_absrel",
    # "depth_delta_1_25",
]


def load_metrics(metrics_path: Path) -> List[Dict[str, float]]:
    with metrics_path.open("r") as f:
        raw = json.load(f)

    flattened: List[Dict[str, float]] = []
    for sequence_metrics in raw.values():
        flattened.extend(sequence_metrics)
    return flattened


def aggregate_metrics(entries: Iterable[Dict[str, float]], keys: Iterable[str]) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    entries_list = list(entries)
    if not entries_list:
        return results

    for key in keys:
        values = [float(item[key]) for item in entries_list if key in item]
        finite_values = [v for v in values if math.isfinite(v)]
        if not finite_values:
            continue

        mean_val = statistics.fmean(finite_values)
        median_val = statistics.median(finite_values)
        std_val = statistics.pstdev(finite_values) if len(finite_values) > 1 else 0.0

        results[key] = {
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "count": len(finite_values),
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Summarize forward consistency metrics for each SAVE_OFFSET."
    )
    parser.add_argument(
        "metrics_json",
        type=Path,
        help="Path to metrics.json produced by run_eval_forward_consistency_with_depth_vipe.py",
    )
    parser.add_argument(
        "--offsets",
        type=int,
        nargs="*",
        default=None,
        help="Offsets to summarize (defaults to all present in the metrics file).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=METRIC_KEYS,
        help="Metric fields to include in the summary.",
    )

    args = parser.parse_args()
    metrics_path: Path = args.metrics_json

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")

    metrics = load_metrics(metrics_path)
    if not metrics:
        raise ValueError("metrics file does not contain any entries")

    offsets = args.offsets
    if offsets is None:
        offsets = sorted({int(entry["offset"]) for entry in metrics if "offset" in entry})

    offsets_set = set(offsets)
    metrics_by_offset: Dict[int, List[Dict[str, float]]] = {offset: [] for offset in offsets}

    for entry in metrics:
        offset = int(entry.get("offset", -1))
        if offset in offsets_set:
            metrics_by_offset[offset].append(entry)

    for offset in offsets:
        entries = metrics_by_offset[offset]
        if not entries:
            print(f"Offset {offset}: no entries found\n")
            continue

        summary = aggregate_metrics(entries, args.metrics)

        print(f"Offset {offset} (count={len(entries)}):")
        for metric_name in args.metrics:
            metric_summary = summary.get(metric_name)
            if metric_summary is None:
                continue
            mean_val = metric_summary["mean"]
            median_val = metric_summary["median"]
            std_val = metric_summary["std"]
            count_val = metric_summary["count"]
            print(
                f"  {metric_name:>16}: mean={mean_val:.4f}, median={median_val:.4f}, "
                f"std={std_val:.4f}, samples={count_val}"
            )
        print()


if __name__ == "__main__":
    main()
