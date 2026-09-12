import argparse
import json
from pathlib import Path

import numpy as np


def load_poses(path: str) -> np.ndarray:
    with open(path, "r") as f:
        poses = np.array(json.load(f), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{path} is expected to have shape [T, 4, 4], got {poses.shape}")
    return poses


def rotation_angle_deg(rot: np.ndarray) -> float:
    trace = np.trace(rot)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def summarize(name: str, values: np.ndarray) -> str:
    return (
        f"{name}: min={values.min():.9f}, mean={values.mean():.9f}, "
        f"median={np.median(values):.9f}, max={values.max():.9f}"
    )


def compare_poses(ref_poses: np.ndarray, test_poses: np.ndarray) -> dict:
    abs_diff = np.abs(ref_poses - test_poses)

    rel_trans_errors = []
    rel_rot_errors_deg = []
    fro_errors = []

    for ref_pose, test_pose in zip(ref_poses, test_poses):
        rel_pose = np.linalg.inv(ref_pose) @ test_pose
        rel_trans_errors.append(np.linalg.norm(rel_pose[:3, 3]))
        rel_rot_errors_deg.append(rotation_angle_deg(rel_pose[:3, :3]))
        fro_errors.append(np.linalg.norm(ref_pose - test_pose, ord="fro"))

    rel_trans_errors = np.array(rel_trans_errors)
    rel_rot_errors_deg = np.array(rel_rot_errors_deg)
    fro_errors = np.array(fro_errors)

    return {
        "max_abs_matrix_diff": float(abs_diff.max()),
        "mean_abs_matrix_diff": float(abs_diff.mean()),
        "translation_errors": rel_trans_errors,
        "rotation_errors_deg": rel_rot_errors_deg,
        "fro_errors": fro_errors,
        "worst_translation_idx": int(rel_trans_errors.argmax()),
        "worst_rotation_idx": int(rel_rot_errors_deg.argmax()),
        "worst_fro_idx": int(fro_errors.argmax()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two pose JSON files of shape [T, 4, 4].")
    parser.add_argument("reference", type=str, help="Reference pose JSON path")
    parser.add_argument("target", type=str, help="Target pose JSON path")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for allclose check")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    target_path = Path(args.target)
    ref_poses = load_poses(str(ref_path))
    target_poses = load_poses(str(target_path))

    if ref_poses.shape != target_poses.shape:
        raise ValueError(f"Shape mismatch: {ref_poses.shape} vs {target_poses.shape}")

    result = compare_poses(ref_poses, target_poses)

    print(f"reference: {ref_path}")
    print(f"target:    {target_path}")
    print(f"shape:     {ref_poses.shape}")
    print(f"allclose(atol={args.atol}, rtol=0): {np.allclose(ref_poses, target_poses, atol=args.atol, rtol=0)}")
    print(f"max_abs_matrix_diff:  {result['max_abs_matrix_diff']:.9f}")
    print(f"mean_abs_matrix_diff: {result['mean_abs_matrix_diff']:.9f}")
    print(summarize("translation_error", result["translation_errors"]))
    print(summarize("rotation_error_deg", result["rotation_errors_deg"]))
    print(summarize("frobenius_error", result["fro_errors"]))

    trans_idx = result["worst_translation_idx"]
    rot_idx = result["worst_rotation_idx"]
    fro_idx = result["worst_fro_idx"]
    print(
        f"worst_translation_frame: {trans_idx} "
        f"(error={result['translation_errors'][trans_idx]:.9f})"
    )
    print(
        f"worst_rotation_frame:    {rot_idx} "
        f"(error_deg={result['rotation_errors_deg'][rot_idx]:.9f})"
    )
    print(
        f"worst_frobenius_frame:   {fro_idx} "
        f"(error={result['fro_errors'][fro_idx]:.9f})"
    )


if __name__ == "__main__":
    main()
