import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a directory of ViPE pose-I2V trajectories.")
    parser.add_argument("--input_root", type=Path, default=None, help="Original Pose-I2V result root for metadata.")
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Scope output produced by run_vipe_pose_i2v.sh.",
    )
    parser.add_argument("--scope", choices=["first_segment", "all_segments"], required=True)
    parser.add_argument("--split_json", type=Path, required=True, help="JSON containing the selected scene split.")
    parser.add_argument("--split_subset", default="test", help="Split key in --split_json.")
    parser.add_argument("--interiorgs_root", type=Path, required=True, help="Root containing <scene>/transforms.json.")
    parser.add_argument("--frames_subdir", default="pano_camera0")
    parser.add_argument("--frame_ext", default="png")
    parser.add_argument("--align_mode", choices=["none", "scale", "sim3"], default="sim3")
    parser.add_argument(
        "--rotation_mode",
        choices=["absolute", "relative", "constant_offset"],
        default="constant_offset",
    )
    parser.add_argument("--gt_position_source", choices=["location", "rt"], default="location")
    parser.add_argument(
        "--gt_rotation_source",
        choices=["rt", "identity", "interiorgs_erp"],
        default="interiorgs_erp",
    )
    parser.add_argument("--worldscore_mode", choices=["relative", "raw"], default="relative")
    parser.add_argument(
        "--worldscore_scale_solver",
        choices=["auto", "scipy", "cvxpy", "least_squares"],
        default="scipy",
    )
    parser.add_argument("--preview_rows", type=int, default=0)
    return parser.parse_args()


def load_scene_ids(split_json: Path, subset: str) -> list[str]:
    with split_json.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    if not isinstance(split, dict) or not isinstance(split.get(subset), list):
        raise ValueError(f"{split_json} must contain split[{subset!r}] as a list")
    scene_ids = split[subset]
    if not scene_ids:
        raise ValueError(f"No scenes found in split[{subset!r}] of {split_json}")
    if any(not isinstance(scene_id, str) or not scene_id for scene_id in scene_ids):
        raise ValueError(f"Invalid scene id in split[{subset!r}] of {split_json}")
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError(f"Duplicate scene ids in split[{subset!r}] of {split_json}")
    return scene_ids


def write_outputs(args: argparse.Namespace, rows: list[dict], failures: list[dict]) -> None:
    per_scene_path = args.output_root / "metrics_per_scene.csv"
    summary_path = args.output_root / "metrics_summary.json"
    per_scene_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene_id",
        "status",
        "frame_count",
        "worldscore_rotation_mean_deg",
        "worldscore_translation_mean",
        "worldscore_scale",
        "worldscore_scale_solver",
        "summary_json",
        "error",
    ]
    with per_scene_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful = [row for row in rows if row["status"] == "success"]
    rotation_values = np.asarray(
        [float(row["worldscore_rotation_mean_deg"]) for row in successful], dtype=np.float64
    )
    translation_values = np.asarray(
        [float(row["worldscore_translation_mean"]) for row in successful], dtype=np.float64
    )
    summary = {
        "input_root": str(args.input_root) if args.input_root is not None else None,
        "output_root": str(args.output_root),
        "scope": args.scope,
        "split_json": str(args.split_json),
        "split_subset": args.split_subset,
        "interiorgs_root": str(args.interiorgs_root),
        "scene_count": len(rows),
        "succeeded": len(successful),
        "failed": len(failures),
        "macro_average": {
            "worldscore_rotation_mean_deg": float(rotation_values.mean()) if len(rotation_values) else None,
            "worldscore_translation_mean": float(translation_values.mean()) if len(translation_values) else None,
        },
        "config": {
            "align_mode": args.align_mode,
            "rotation_mode": args.rotation_mode,
            "gt_position_source": args.gt_position_source,
            "gt_rotation_source": args.gt_rotation_source,
            "worldscore_mode": args.worldscore_mode,
            "worldscore_scale_solver": args.worldscore_scale_solver,
        },
        "failures": failures,
        "metrics_per_scene_csv": str(per_scene_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    if not args.split_json.is_file():
        raise FileNotFoundError(f"Split JSON not found: {args.split_json}")
    if not args.interiorgs_root.is_dir():
        raise FileNotFoundError(f"InteriorGS root not found: {args.interiorgs_root}")
    scene_ids = load_scene_ids(args.split_json, args.split_subset)
    metrics_dir = args.output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    for index, scene_id in enumerate(scene_ids, start=1):
        pose_path = args.output_root / "poses" / f"{scene_id}_poses.json"
        manifest_path = args.output_root / "poses" / f"{scene_id}_input_manifest.json"
        transforms_path = args.interiorgs_root / scene_id / "transforms.json"
        csv_path = metrics_dir / f"{scene_id}_vipe_vs_gt.csv"
        summary_path = metrics_dir / f"{scene_id}_vipe_vs_gt_summary.json"
        print(f"[{index}/{len(scene_ids)}] Evaluating {scene_id}", flush=True)
        try:
            for required_path, label in (
                (pose_path, "pose"),
                (manifest_path, "frame manifest"),
                (transforms_path, "GT transforms"),
            ):
                if not required_path.is_file():
                    raise FileNotFoundError(f"Missing {label}: {required_path}")
            command = [
                sys.executable,
                str(REPO_ROOT / "compare_vipe_omniroam_pose.py"),
                "--vipe_poses",
                str(pose_path),
                "--frame_manifest",
                str(manifest_path),
                "--gt_transforms_json",
                str(transforms_path),
                "--frames_subdir",
                args.frames_subdir,
                "--frame_ext",
                args.frame_ext,
                "--align_mode",
                args.align_mode,
                "--rotation_mode",
                args.rotation_mode,
                "--gt_position_source",
                args.gt_position_source,
                "--gt_rotation_source",
                args.gt_rotation_source,
                "--worldscore_mode",
                args.worldscore_mode,
                "--worldscore_scale_solver",
                args.worldscore_scale_solver,
                "--preview_rows",
                str(args.preview_rows),
                "--output_csv",
                str(csv_path),
                "--output_summary_json",
                str(summary_path),
            ]
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout, end="", flush=True)
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr, flush=True)
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
                raise RuntimeError(f"comparison exited with status {result.returncode}: {detail[-4000:]}")
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            rows.append(
                {
                    "scene_id": scene_id,
                    "status": "success",
                    "frame_count": summary["frame_count"],
                    "worldscore_rotation_mean_deg": summary["worldscore_rotation_mean_deg"],
                    "worldscore_translation_mean": summary["worldscore_translation_mean"],
                    "worldscore_scale": summary["worldscore_scale"],
                    "worldscore_scale_solver": summary["worldscore_scale_solver"],
                    "summary_json": str(summary_path),
                    "error": "",
                }
            )
        except Exception as error:
            failure = {"scene_id": scene_id, "error": str(error)}
            failures.append(failure)
            rows.append(
                {
                    "scene_id": scene_id,
                    "status": "failed",
                    "frame_count": "",
                    "worldscore_rotation_mean_deg": "",
                    "worldscore_translation_mean": "",
                    "worldscore_scale": "",
                    "worldscore_scale_solver": "",
                    "summary_json": str(summary_path),
                    "error": str(error),
                }
            )
            print(f"[ERROR] {scene_id}: {error}", file=sys.stderr, flush=True)

    write_outputs(args, rows, failures)
    print(
        f"Evaluation complete: succeeded={len(rows) - len(failures)} failed={len(failures)} "
        f"summary={args.output_root / 'metrics_summary.json'}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
