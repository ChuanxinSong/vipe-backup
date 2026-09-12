import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


DEFAULT_INPUT = Path("openMVG/test_equi/reconstruction/sfm_data.json")


def extract_frame_index(filename: str) -> Optional[int]:
    match = re.search(r"(\d+)", filename)
    if match is None:
        return None
    return int(match.group(1))


def rotation_matrix_to_euler_zyx_deg(rotation: np.ndarray) -> np.ndarray:
    sy = math.hypot(rotation[0, 0], rotation[1, 0])
    singular = sy < 1e-8

    if not singular:
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        pitch = math.atan2(-rotation[2, 0], sy)
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
    else:
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        roll = 0.0

    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))


def rotation_angle_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = rotation_a.T @ rotation_b
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def trajectory_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())


def default_output_prefix(input_path: Path) -> Path:
    safe_parts = [part for part in input_path.with_suffix("").parts if part not in ("", ".", "..")]
    safe_name = "_".join(safe_parts)
    return Path("pose_exports") / safe_name


def build_summary(centers: np.ndarray, rotations_c2w: np.ndarray) -> Dict[str, object]:
    steps = np.zeros(len(centers), dtype=np.float64)
    cumulative = np.zeros(len(centers), dtype=np.float64)
    if len(centers) > 1:
        steps[1:] = np.linalg.norm(centers[1:] - centers[:-1], axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(steps[1:])])

    start = centers[0]
    end = centers[-1]
    net = end - start
    path_length = float(cumulative[-1])
    straightness = float(np.linalg.norm(net) / path_length) if path_length > 0 else 0.0

    centered = centers - centers.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centers) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    principal_direction = eigenvectors[:, 0]
    if principal_direction[2] < 0:
        principal_direction = -principal_direction
    variance_ratio = eigenvalues / max(float(eigenvalues.sum()), 1e-12)

    projected = ((centers - centers.mean(axis=0)) @ principal_direction)
    projected_deltas = np.diff(projected)
    line_projection = (
        np.outer(projected, principal_direction) + centers.mean(axis=0, keepdims=True)
    )
    line_deviation = np.linalg.norm(centers - line_projection, axis=1)

    rotation_steps = np.zeros(len(rotations_c2w), dtype=np.float64)
    if len(rotations_c2w) > 1:
        rotation_steps[1:] = np.array(
            [
                rotation_angle_deg(rotations_c2w[idx - 1], rotations_c2w[idx])
                for idx in range(1, len(rotations_c2w))
            ],
            dtype=np.float64,
        )

    return {
        "frame_count": int(len(centers)),
        "start_center": start.tolist(),
        "end_center": end.tolist(),
        "net_displacement": net.tolist(),
        "net_displacement_norm": float(np.linalg.norm(net)),
        "path_length": path_length,
        "straightness_ratio": straightness,
        "axis_min": centers.min(axis=0).tolist(),
        "axis_max": centers.max(axis=0).tolist(),
        "axis_range": (centers.max(axis=0) - centers.min(axis=0)).tolist(),
        "mean_step": float(steps[1:].mean()) if len(steps) > 1 else 0.0,
        "median_step": float(np.median(steps[1:])) if len(steps) > 1 else 0.0,
        "max_step": float(steps[1:].max()) if len(steps) > 1 else 0.0,
        "principal_direction": principal_direction.tolist(),
        "principal_variance_ratio": variance_ratio.tolist(),
        "principal_span": float(projected.max() - projected.min()) if len(projected) else 0.0,
        "principal_progress_positive_fraction": float((projected_deltas > 0).mean()) if len(projected_deltas) else 0.0,
        "principal_progress_negative_fraction": float((projected_deltas < 0).mean()) if len(projected_deltas) else 0.0,
        "line_fit_rms_deviation": float(np.sqrt(np.mean(line_deviation ** 2))),
        "line_fit_max_deviation": float(line_deviation.max()),
        "total_rotation_change_deg": rotation_angle_deg(rotations_c2w[0], rotations_c2w[-1]),
        "mean_rotation_step_deg": float(rotation_steps[1:].mean()) if len(rotation_steps) > 1 else 0.0,
        "max_rotation_step_deg": float(rotation_steps[1:].max()) if len(rotation_steps) > 1 else 0.0,
    }


def build_summary_text(summary: Dict[str, object]) -> List[str]:
    axis_range = np.asarray(summary["axis_range"], dtype=np.float64)
    principal_direction = np.asarray(summary["principal_direction"], dtype=np.float64)
    principal_var = np.asarray(summary["principal_variance_ratio"], dtype=np.float64)

    dominant_axes = sorted(
        zip(["x", "y", "z"], axis_range.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    dominant_axis_names = "/".join(axis for axis, _ in dominant_axes[:2])

    motion_shape = "mostly a straight line"
    if principal_var[0] < 0.9:
        motion_shape = "a curved or multi-directional path"
    elif float(summary["straightness_ratio"]) < 0.95:
        motion_shape = "a mostly forward path with some backtracking"

    heading_change = "orientation stays fairly stable"
    if float(summary["total_rotation_change_deg"]) > 15.0:
        heading_change = "orientation changes noticeably"
    elif float(summary["total_rotation_change_deg"]) > 5.0:
        heading_change = "orientation changes mildly"

    return [
        (
            f"Motion pattern: {motion_shape}; translation is dominated by {dominant_axis_names}, "
            f"with principal direction about [{principal_direction[0]:.3f}, {principal_direction[1]:.3f}, {principal_direction[2]:.3f}]."
        ),
        (
            f"Track length {summary['path_length']:.3f}, net displacement {summary['net_displacement_norm']:.3f}, "
            f"straightness {summary['straightness_ratio']:.3f}, line-fit RMS deviation {summary['line_fit_rms_deviation']:.3f}."
        ),
        (
            f"Per-step translation mean/median/max: {summary['mean_step']:.3f} / {summary['median_step']:.3f} / {summary['max_step']:.3f}; "
            f"{heading_change} (total rotation {summary['total_rotation_change_deg']:.3f} deg, "
            f"mean step rotation {summary['mean_rotation_step_deg']:.3f} deg)."
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract camera poses from an openMVG sfm_data.json file and summarize motion."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to openMVG sfm_data.json.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to save extracted [N,4,4] poses. Defaults to <input_stem>_c2w_poses.json.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to save per-frame pose metadata. Defaults to <input_stem>_pose_table.csv.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path to save the motion summary JSON. Defaults to <input_stem>_motion_summary.json.",
    )
    parser.add_argument(
        "--matrix-convention",
        choices=["c2w", "w2c"],
        default="c2w",
        help="Which pose matrix convention to save in --output-json.",
    )
    args = parser.parse_args()

    input_path = args.input
    output_prefix = default_output_prefix(input_path)
    output_json = args.output_json or output_prefix.with_name(output_prefix.name + f"_{args.matrix_convention}_poses.json")
    output_csv = args.output_csv or output_prefix.with_name(output_prefix.name + "_pose_table.csv")
    summary_json = args.summary_json or output_prefix.with_name(output_prefix.name + "_motion_summary.json")

    with input_path.open("r", encoding="utf-8") as handle:
        sfm_data = json.load(handle)

    view_info: Dict[int, Dict[str, object]] = {}
    for item in sfm_data.get("views", []):
        data = item["value"]["ptr_wrapper"]["data"]
        pose_id = data.get("id_pose")
        if pose_id is None:
            continue
        view_info[pose_id] = {
            "filename": data.get("filename", ""),
            "id_view": data.get("id_view"),
            "width": data.get("width"),
            "height": data.get("height"),
            "frame_index": extract_frame_index(data.get("filename", "")),
        }

    rows = []
    for item in sfm_data.get("extrinsics", []):
        pose_id = item["key"]
        value = item["value"]
        rotation_cw = np.asarray(value["rotation"], dtype=np.float64)
        center = np.asarray(value["center"], dtype=np.float64)
        rotation_wc = rotation_cw.T
        translation_wc = center
        translation_cw = -rotation_cw @ center

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rotation_wc
        c2w[:3, 3] = translation_wc

        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = rotation_cw
        w2c[:3, 3] = translation_cw

        info = view_info.get(pose_id, {})
        frame_index = info.get("frame_index")
        if frame_index is None:
            frame_index = int(pose_id)

        rows.append(
            {
                "pose_id": int(pose_id),
                "frame_index": int(frame_index),
                "filename": str(info.get("filename", "")),
                "id_view": info.get("id_view"),
                "width": info.get("width"),
                "height": info.get("height"),
                "center": center,
                "rotation_cw": rotation_cw,
                "rotation_c2w": rotation_wc,
                "c2w": c2w,
                "w2c": w2c,
            }
        )

    rows.sort(key=lambda row: (row["frame_index"], row["pose_id"]))
    if not rows:
        raise ValueError(f"No extrinsics found in {input_path}")

    centers = np.stack([row["center"] for row in rows], axis=0)
    rotations_c2w = np.stack([row["rotation_c2w"] for row in rows], axis=0)
    pose_matrices = np.stack([row[args.matrix_convention] for row in rows], axis=0)

    steps = np.zeros(len(rows), dtype=np.float64)
    cumulative = np.zeros(len(rows), dtype=np.float64)
    eulers_deg = np.stack(
        [rotation_matrix_to_euler_zyx_deg(row["rotation_c2w"]) for row in rows],
        axis=0,
    )
    if len(rows) > 1:
        steps[1:] = np.linalg.norm(centers[1:] - centers[:-1], axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(steps[1:])])

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(pose_matrices.tolist(), handle, indent=2)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "pose_id",
                "filename",
                "id_view",
                "width",
                "height",
                "center_x",
                "center_y",
                "center_z",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "step_distance",
                "cumulative_distance",
            ]
        )
        for idx, row in enumerate(rows):
            writer.writerow(
                [
                    row["frame_index"],
                    row["pose_id"],
                    row["filename"],
                    row["id_view"],
                    row["width"],
                    row["height"],
                    float(centers[idx, 0]),
                    float(centers[idx, 1]),
                    float(centers[idx, 2]),
                    float(eulers_deg[idx, 0]),
                    float(eulers_deg[idx, 1]),
                    float(eulers_deg[idx, 2]),
                    float(steps[idx]),
                    float(cumulative[idx]),
                ]
            )

    summary = build_summary(centers, rotations_c2w)
    summary.update(
        {
            "input_path": str(input_path),
            "output_json": str(output_json),
            "output_csv": str(output_csv),
            "matrix_convention": args.matrix_convention,
            "first_frame_index": int(rows[0]["frame_index"]),
            "last_frame_index": int(rows[-1]["frame_index"]),
            "first_filename": rows[0]["filename"],
            "last_filename": rows[-1]["filename"],
            "trajectory_length_check": trajectory_length(centers),
        }
    )
    summary["summary_text"] = build_summary_text(summary)

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved {len(rows)} poses to {output_json}")
    print(f"Saved pose table to {output_csv}")
    print(f"Saved motion summary to {summary_json}")
    print("")
    for line in summary["summary_text"]:
        print(line)


if __name__ == "__main__":
    main()
