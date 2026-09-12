import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np


DEFAULT_REFERENCE = Path("equi_data/frame_trajectory.txt")
DEFAULT_ESTIMATE = Path(
    "carla_benchmark_results/genex_results/genex_1024_unik3d/076_2_segment_09_poses.json"
)
DEFAULT_OUTPUT_CSV = Path("pose_comparison_errors.csv")


def load_txt_poses(path: Path) -> np.ndarray:
    poses = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            values = [float(token) for token in stripped.split()]
            if len(values) != 12:
                raise ValueError(f"{path}:{line_idx} expected 12 values, got {len(values)}")
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :4] = np.array(values, dtype=np.float64).reshape(3, 4)
            poses.append(pose)
    if not poses:
        raise ValueError(f"No poses found in {path}")
    return np.stack(poses, axis=0)


def load_json_poses(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    poses = np.asarray(data, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{path} expected shape [N, 4, 4], got {poses.shape}")
    if poses.shape[0] == 0:
        raise ValueError(f"No poses found in {path}")
    return poses


def load_poses(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".txt":
        return load_txt_poses(path)
    if path.suffix.lower() == ".json":
        return load_json_poses(path)
    raise ValueError(f"Unsupported pose file format: {path}")


def solve_scale_translation(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray]:
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matched point sets of shape [N, 3], got {src.shape} and {dst.shape}")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    denominator = np.sum(src_centered ** 2)
    if denominator <= 0:
        raise ValueError("Degenerate trajectory: source translation variance is zero")

    scale = float(np.sum(src_centered * dst_centered) / denominator)
    translation = dst_mean - scale * src_mean
    return scale, translation


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matched point sets of shape [N, 3], got {src.shape} and {dst.shape}")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    covariance = (dst_centered.T @ src_centered) / src.shape[0]
    u, singular_values, vh = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vh
    variance = np.mean(np.sum(src_centered ** 2, axis=1))
    if variance <= 0:
        raise ValueError("Degenerate trajectory: source translation variance is zero")

    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = rotation_a.T @ rotation_b
    trace_value = np.trace(relative)
    cosine = np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def trajectory_length(translations: np.ndarray) -> float:
    if len(translations) < 2:
        return 0.0
    deltas = translations[1:] - translations[:-1]
    return float(np.linalg.norm(deltas, axis=1).sum())


def format_table(rows, headers) -> str:
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(row_values):
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row_values))

    separator = "-+-".join("-" * width for width in widths)
    table_lines = [format_row(headers), separator]
    table_lines.extend(format_row(row) for row in string_rows)
    return "\n".join(table_lines)


def main():
    parser = argparse.ArgumentParser(description="Align two pose files frame-by-frame and export error metrics.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="Reference pose file (.txt or .json).")
    parser.add_argument("--estimate", type=Path, default=DEFAULT_ESTIMATE, help="Estimated pose file (.txt or .json).")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to save the per-frame error table as CSV.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=10,
        help="Number of per-frame rows to print to stdout.",
    )
    parser.add_argument(
        "--align-mode",
        choices=["scale", "sim3"],
        default="scale",
        help="Alignment mode applied to the estimate trajectory before computing errors.",
    )
    args = parser.parse_args()

    reference_poses = load_poses(args.reference)
    estimate_poses = load_poses(args.estimate)

    num_frames = min(len(reference_poses), len(estimate_poses))
    if num_frames == 0:
        raise ValueError("No overlapping frames to compare")
    if len(reference_poses) != len(estimate_poses):
        print(
            f"Warning: frame count mismatch, using first {num_frames} frames "
            f"(reference={len(reference_poses)}, estimate={len(estimate_poses)})"
        )

    reference_poses = reference_poses[:num_frames]
    estimate_poses = estimate_poses[:num_frames]

    reference_rotations = reference_poses[:, :3, :3]
    estimate_rotations = estimate_poses[:, :3, :3]
    reference_translations = reference_poses[:, :3, 3]
    estimate_translations = estimate_poses[:, :3, 3]

    if args.align_mode == "sim3":
        scale, global_rotation, global_translation = umeyama_similarity(
            estimate_translations,
            reference_translations,
        )
    else:
        scale, global_translation = solve_scale_translation(
            estimate_translations,
            reference_translations,
        )
        global_rotation = np.eye(3, dtype=np.float64)

    aligned_rotations = np.einsum("ij,njk->nik", global_rotation, estimate_rotations)
    aligned_translations = scale * np.einsum("ij,nj->ni", global_rotation, estimate_translations) + global_translation

    translation_error_before = np.linalg.norm(estimate_translations - reference_translations, axis=1)
    translation_error_after = np.linalg.norm(aligned_translations - reference_translations, axis=1)
    rotation_error_before = np.array(
        [rotation_error_deg(reference_rotations[idx], estimate_rotations[idx]) for idx in range(num_frames)]
    )
    rotation_error_after = np.array(
        [rotation_error_deg(reference_rotations[idx], aligned_rotations[idx]) for idx in range(num_frames)]
    )

    if num_frames > 1:
        reference_step = np.zeros(num_frames, dtype=np.float64)
        estimate_step = np.zeros(num_frames, dtype=np.float64)
        aligned_step = np.zeros(num_frames, dtype=np.float64)
        step_error_after = np.zeros(num_frames, dtype=np.float64)

        reference_step[1:] = np.linalg.norm(reference_translations[1:] - reference_translations[:-1], axis=1)
        estimate_step[1:] = np.linalg.norm(estimate_translations[1:] - estimate_translations[:-1], axis=1)
        aligned_step[1:] = np.linalg.norm(aligned_translations[1:] - aligned_translations[:-1], axis=1)
        step_error_after[1:] = np.linalg.norm(
            (aligned_translations[1:] - aligned_translations[:-1])
            - (reference_translations[1:] - reference_translations[:-1]),
            axis=1,
        )
    else:
        reference_step = np.zeros(1, dtype=np.float64)
        estimate_step = np.zeros(1, dtype=np.float64)
        aligned_step = np.zeros(1, dtype=np.float64)
        step_error_after = np.zeros(1, dtype=np.float64)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_idx",
                "ref_tx",
                "ref_ty",
                "ref_tz",
                "est_tx",
                "est_ty",
                "est_tz",
                "aligned_tx",
                "aligned_ty",
                "aligned_tz",
                "translation_error_before",
                "translation_error_after",
                "rotation_error_before_deg",
                "rotation_error_after_deg",
                "ref_step",
                "est_step",
                "aligned_step",
                "step_error_after",
            ]
        )
        for idx in range(num_frames):
            writer.writerow(
                [
                    idx,
                    *reference_translations[idx].tolist(),
                    *estimate_translations[idx].tolist(),
                    *aligned_translations[idx].tolist(),
                    float(translation_error_before[idx]),
                    float(translation_error_after[idx]),
                    float(rotation_error_before[idx]),
                    float(rotation_error_after[idx]),
                    float(reference_step[idx]),
                    float(estimate_step[idx]),
                    float(aligned_step[idx]),
                    float(step_error_after[idx]),
                ]
            )

    summary_rows = [
        ["frames", num_frames],
        ["reference_path_length", f"{trajectory_length(reference_translations):.6f}"],
        ["estimate_path_length", f"{trajectory_length(estimate_translations):.6f}"],
        ["aligned_path_length", f"{trajectory_length(aligned_translations):.6f}"],
        ["align_mode", args.align_mode],
        ["similarity_scale", f"{scale:.6f}"],
        ["translation_rmse_before", f"{np.sqrt(np.mean(translation_error_before ** 2)):.6f}"],
        ["translation_rmse_after", f"{np.sqrt(np.mean(translation_error_after ** 2)):.6f}"],
        ["translation_mean_after", f"{translation_error_after.mean():.6f}"],
        ["translation_max_after", f"{translation_error_after.max():.6f}"],
        ["rotation_mean_before_deg", f"{rotation_error_before.mean():.6f}"],
        ["rotation_mean_after_deg", f"{rotation_error_after.mean():.6f}"],
        ["rotation_max_after_deg", f"{rotation_error_after.max():.6f}"],
        ["csv_path", str(args.output_csv)],
    ]
    print("Summary")
    print(format_table(summary_rows, headers=["metric", "value"]))

    preview_count = max(0, min(args.preview_rows, num_frames))
    if preview_count > 0:
        preview_rows = []
        for idx in range(preview_count):
            preview_rows.append(
                [
                    idx,
                    f"{translation_error_before[idx]:.6f}",
                    f"{translation_error_after[idx]:.6f}",
                    f"{rotation_error_before[idx]:.6f}",
                    f"{rotation_error_after[idx]:.6f}",
                    f"{reference_step[idx]:.6f}",
                    f"{aligned_step[idx]:.6f}",
                    f"{step_error_after[idx]:.6f}",
                ]
            )

        print("\nPer-frame preview")
        print(
            format_table(
                preview_rows,
                headers=[
                    "frame",
                    "t_err_before",
                    "t_err_after",
                    "r_err_before_deg",
                    "r_err_after_deg",
                    "ref_step",
                    "aligned_step",
                    "step_err_after",
                ],
            )
        )


if __name__ == "__main__":
    main()
