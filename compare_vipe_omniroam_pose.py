import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ViPE C2W pose JSON against OmniRoam/InteriorGS transforms.json GT."
    )
    parser.add_argument("--vipe_poses", type=Path, required=True, help="ViPE pose JSON with shape [T,4,4].")
    parser.add_argument(
        "--gt_transforms_json",
        type=Path,
        required=True,
        help="OmniRoam/InteriorGS transforms.json containing per_image R/t/location.",
    )
    parser.add_argument("--frames_subdir", default="pano_camera0", help="Frame subdirectory key in transforms.json.")
    parser.add_argument("--frame_ext", default="png", help="Frame extension used in transforms.json keys.")
    parser.add_argument("--frame_start", type=int, default=1, help="First GT frame id to compare.")
    parser.add_argument(
        "--frame_count",
        type=int,
        default=None,
        help="Number of frames to compare. Defaults to ViPE pose length.",
    )
    parser.add_argument(
        "--frame_manifest",
        type=Path,
        default=None,
        help="Optional ViPE pose-I2V input manifest containing the exact ordered GT frame ids.",
    )
    parser.add_argument(
        "--align_mode",
        choices=["none", "scale", "sim3"],
        default="sim3",
        help="Alignment applied to ViPE trajectory before computing absolute errors.",
    )
    parser.add_argument(
        "--rotation_mode",
        choices=["absolute", "relative", "constant_offset"],
        default="constant_offset",
        help=(
            "Rotation comparison mode. absolute compares C2W rotations after trajectory alignment; "
            "relative compares rotation changes from the first frame; constant_offset additionally fits one "
            "fixed camera-axis rotation offset."
        ),
    )
    parser.add_argument(
        "--gt_rotation_source",
        choices=["rt", "identity", "interiorgs_erp"],
        default="interiorgs_erp",
        help=(
            "GT rotation source. interiorgs_erp uses InteriorGS world axes "
            "[right,forward,up] with ViPE/OpenCV camera axes [right,down,forward]; "
            "rt uses transforms.json OpenCV R/t; identity ignores GT rotation."
        ),
    )
    parser.add_argument(
        "--gt_position_source",
        choices=["location", "rt"],
        default="location",
        help="GT camera center source. rt computes C=-R.T@t.",
    )
    parser.add_argument(
        "--worldscore_mode",
        choices=["relative", "raw"],
        default="relative",
        help=(
            "WorldScore-style translation origin. relative subtracts the first frame before scalar scale fitting; "
            "raw follows WorldScore source more literally and uses absolute camera centers."
        ),
    )
    parser.add_argument(
        "--worldscore_scale_solver",
        choices=["auto", "scipy", "cvxpy", "least_squares"],
        default="auto",
        help="Scalar scale solver. auto preserves the historical cvxpy/scipy/fallback order.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional CSV for per-frame errors. Defaults to <vipe_poses>_vs_gt.csv.",
    )
    parser.add_argument(
        "--output_summary_json",
        type=Path,
        default=None,
        help="Optional summary JSON. Defaults to <vipe_poses>_vs_gt_summary.json.",
    )
    parser.add_argument("--preview_rows", type=int, default=10, help="Number of per-frame rows to print.")
    return parser.parse_args()


def load_vipe_poses(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        poses = np.asarray(json.load(handle), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{path} must have shape [T,4,4], got {poses.shape}")
    if len(poses) == 0:
        raise ValueError(f"{path} contains no poses")
    return poses


def load_manifest_frame_ids(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} must contain a JSON object")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{path} must contain a non-empty frames list")
    frame_ids = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"{path} frames[{index}] must be an object")
        frame_id = frame.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int):
            raise ValueError(f"{path} frames[{index}].frame_id must be an integer")
        if frame_ids and frame_id <= frame_ids[-1]:
            raise ValueError(f"{path} frame ids must be strictly increasing: {frame_ids[-1]} then {frame_id}")
        frame_ids.append(frame_id)
    if manifest.get("frame_count") not in (None, len(frame_ids)):
        raise ValueError(
            f"{path} frame_count={manifest.get('frame_count')!r} does not match frames length {len(frame_ids)}"
        )
    return frame_ids


def load_gt_c2w(args: argparse.Namespace, frame_ids: list[int]) -> np.ndarray:
    with args.gt_transforms_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    per_image = data.get("per_image")
    if not isinstance(per_image, dict):
        raise ValueError(f"{args.gt_transforms_json} must contain object field 'per_image'")

    frame_ext = args.frame_ext.lower().lstrip(".")
    frames_subdir = args.frames_subdir.strip("/")
    poses = []
    missing = []

    for frame_id in frame_ids:
        key = f"{frames_subdir}/frame_{frame_id:04d}.{frame_ext}"
        info = per_image.get(key)
        if not isinstance(info, dict):
            missing.append(key)
            continue

        if args.gt_rotation_source == "identity":
            r_cw = np.eye(3, dtype=np.float64)
        elif args.gt_rotation_source == "interiorgs_erp":
            # InteriorGS ERP world axes are [right, forward, up].
            # ViPE/OpenCV camera-local axes are [right, down, forward].
            # Columns are camera axes expressed in InteriorGS world coordinates:
            #   camera +X/right   -> world +X/right
            #   camera +Y/down    -> world -Z/up
            #   camera +Z/forward -> world +Y/forward
            r_cw = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                ],
                dtype=np.float64,
            )
        else:
            r_wc = np.asarray(info.get("R"), dtype=np.float64)
            if r_wc.shape != (3, 3):
                raise ValueError(f"{key} has invalid R shape {r_wc.shape}")
            r_cw = r_wc.T

        if args.gt_position_source == "location":
            loc = info.get("location")
            if not isinstance(loc, dict):
                raise ValueError(f"{key} has no location object")
            c_w = np.array([loc["x"], loc["y"], loc["z"]], dtype=np.float64)
        else:
            r_wc = np.asarray(info.get("R"), dtype=np.float64)
            t_wc = np.asarray(info.get("t"), dtype=np.float64)
            if r_wc.shape != (3, 3) or t_wc.shape != (3,):
                raise ValueError(f"{key} has invalid R/t shapes {r_wc.shape}, {t_wc.shape}")
            c_w = -r_wc.T @ t_wc

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = r_cw
        pose[:3, 3] = c_w
        poses.append(pose)

    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} GT frames, first missing: {missing[:5]}")
    return np.stack(poses, axis=0)


def check_rotation_matrices(name: str, rotations: np.ndarray) -> None:
    det = np.linalg.det(rotations)
    if not np.all(np.isfinite(rotations)):
        raise ValueError(f"{name} rotations contain NaN or Inf")
    max_det_error = np.max(np.abs(det - 1.0))
    if max_det_error > 1e-2:
        raise ValueError(f"{name} rotations are not valid enough: max |det(R)-1|={max_det_error:.6f}")


def solve_scale_translation(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    denom = np.sum(src_centered**2)
    if denom <= 0:
        raise ValueError("Degenerate source trajectory: zero translation variance")
    scale = float(np.sum(src_centered * dst_centered) / denom)
    translation = dst_mean - scale * src_mean
    return scale, translation


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = (dst_centered.T @ src_centered) / len(src)
    u, singular_values, vh = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vh
    variance = np.mean(np.sum(src_centered**2, axis=1))
    if variance <= 0:
        raise ValueError("Degenerate source trajectory: zero translation variance")
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def fit_rotation_offset(src_rot: np.ndarray, dst_rot: np.ndarray) -> np.ndarray:
    accum = np.zeros((3, 3), dtype=np.float64)
    for src, dst in zip(src_rot, dst_rot):
        accum += src.T @ dst
    u, _, vh = np.linalg.svd(accum)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vh) < 0:
        correction[-1, -1] = -1.0
    return u @ correction @ vh


def rotation_angle_deg(rot: np.ndarray) -> float:
    cosine = np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rotation_errors_deg(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    return np.array([rotation_angle_deg(ref.T @ est) for ref, est in zip(reference, estimate)], dtype=np.float64)


def solve_worldscore_scale(
    gt_positions: np.ndarray,
    pred_positions: np.ndarray,
    solver: str = "auto",
) -> tuple[float, str]:
    if gt_positions.shape != pred_positions.shape or gt_positions.ndim != 2 or gt_positions.shape[1] != 3:
        raise ValueError(
            f"Expected matched position arrays of shape [N,3], got {gt_positions.shape} and {pred_positions.shape}"
        )
    pred_energy = float(np.sum(pred_positions**2))
    if pred_energy <= 1e-12:
        return 0.0, "degenerate_zero_prediction"
    if solver not in {"auto", "scipy", "cvxpy", "least_squares"}:
        raise ValueError(f"Unsupported WorldScore scale solver: {solver}")

    def objective(scale_value: float) -> float:
        return float(np.linalg.norm(gt_positions - scale_value * pred_positions, axis=1).sum())

    if solver in {"auto", "cvxpy"}:
        try:
            import cvxpy as cp

            scale_var = cp.Variable()
            problem = cp.Problem(cp.Minimize(cp.sum(cp.norm(gt_positions - scale_var * pred_positions, axis=1))))
            problem.solve()
            if scale_var.value is None:
                raise RuntimeError(f"cvxpy returned no scale value (status={problem.status})")
            scale_value = float(scale_var.value)
            if not np.isfinite(scale_value):
                raise RuntimeError(f"cvxpy returned non-finite scale {scale_value}")
            return scale_value, "cvxpy"
        except Exception as error:
            if solver == "cvxpy":
                raise RuntimeError("Requested cvxpy WorldScore scale solver failed") from error

    if solver in {"auto", "scipy"}:
        try:
            from scipy.optimize import minimize_scalar

            least_squares_scale = float(np.sum(gt_positions * pred_positions) / pred_energy)
            gt_distance = np.linalg.norm(gt_positions, axis=1).mean()
            pred_distance = np.linalg.norm(pred_positions, axis=1).mean()
            distance_scale = float(gt_distance / (pred_distance + 1e-12))
            span = max(1.0, abs(least_squares_scale) * 10.0, abs(distance_scale) * 10.0)
            result = None
            for _ in range(6):
                result = minimize_scalar(objective, bounds=(-span, span), method="bounded")
                if result.success and abs(float(result.x)) < span * 0.95:
                    break
                span *= 10.0
            if result is None or not result.success or not np.isfinite(result.x):
                raise RuntimeError(f"scipy minimize_scalar failed: {result}")
            return float(result.x), "scipy_minimize_scalar"
        except Exception as error:
            if solver == "scipy":
                raise RuntimeError("Requested scipy WorldScore scale solver failed") from error

    if solver in {"auto", "least_squares"}:
        least_squares_scale = float(np.sum(gt_positions * pred_positions) / pred_energy)
        label = "least_squares" if solver == "least_squares" else "least_squares_fallback"
        return least_squares_scale, label
    raise AssertionError(f"Unhandled WorldScore scale solver: {solver}")


def compute_worldscore_metrics(
    gt_rot: np.ndarray,
    gt_pos: np.ndarray,
    vipe_rot: np.ndarray,
    vipe_pos: np.ndarray,
    mode: str,
    scale_solver: str = "auto",
) -> dict:
    if mode not in {"relative", "raw"}:
        raise ValueError(f"Unsupported worldscore_mode: {mode}")

    fixed_world_rotation = gt_rot[0]
    pred_rot_world = np.einsum("ij,njk->nik", fixed_world_rotation, vipe_rot)
    pred_pos_world = (fixed_world_rotation @ vipe_pos.T).T

    if mode == "relative":
        gt_eval_pos = gt_pos - gt_pos[0]
        pred_eval_pos = pred_pos_world - pred_pos_world[0]
    else:
        gt_eval_pos = gt_pos
        pred_eval_pos = pred_pos_world

    scale, scale_solver_name = solve_worldscore_scale(gt_eval_pos, pred_eval_pos, scale_solver)
    scaled_pred_eval_pos = scale * pred_eval_pos
    translation_error = np.linalg.norm(gt_eval_pos - scaled_pred_eval_pos, axis=1)
    rotation_error = rotation_errors_deg(gt_rot, pred_rot_world)

    return {
        "mode": mode,
        "scale": float(scale),
        "scale_solver": scale_solver_name,
        "fixed_world_rotation": fixed_world_rotation,
        "pred_rot_world": pred_rot_world,
        "pred_pos_world": pred_pos_world,
        "gt_eval_pos": gt_eval_pos,
        "pred_eval_pos": pred_eval_pos,
        "scaled_pred_eval_pos": scaled_pred_eval_pos,
        "translation_error": translation_error,
        "rotation_error_deg": rotation_error,
    }


def trajectory_length(translations: np.ndarray) -> float:
    if len(translations) < 2:
        return 0.0
    return float(np.linalg.norm(translations[1:] - translations[:-1], axis=1).sum())


def summarize_values(values: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def format_table(rows: list[list[object]], headers: list[str]) -> str:
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row_values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row_values))

    return "\n".join([fmt(headers), "-+-".join("-" * w for w in widths), *[fmt(row) for row in string_rows]])


def default_output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}")


def main() -> int:
    args = parse_args()
    vipe_poses = load_vipe_poses(args.vipe_poses)
    if args.frame_manifest is not None:
        if args.frame_count is not None:
            raise ValueError("--frame_count cannot be combined with --frame_manifest")
        frame_ids = load_manifest_frame_ids(args.frame_manifest)
        frame_count = len(frame_ids)
        if len(vipe_poses) != frame_count:
            raise ValueError(
                f"ViPE pose length {len(vipe_poses)} does not match manifest frame count {frame_count}"
            )
    else:
        frame_count = args.frame_count or len(vipe_poses)
        frame_ids = list(range(args.frame_start, args.frame_start + frame_count))
    if frame_count <= 0:
        raise ValueError("--frame_count must be positive")
    if len(vipe_poses) < frame_count:
        raise ValueError(f"ViPE pose length {len(vipe_poses)} is shorter than frame_count={frame_count}")

    vipe_poses = vipe_poses[:frame_count]
    gt_poses = load_gt_c2w(args, frame_ids)

    gt_rot = gt_poses[:, :3, :3]
    vipe_rot = vipe_poses[:, :3, :3]
    gt_pos = gt_poses[:, :3, 3]
    vipe_pos = vipe_poses[:, :3, 3]
    check_rotation_matrices("GT", gt_rot)
    check_rotation_matrices("ViPE", vipe_rot)
    worldscore = compute_worldscore_metrics(
        gt_rot,
        gt_pos,
        vipe_rot,
        vipe_pos,
        args.worldscore_mode,
        args.worldscore_scale_solver,
    )

    if args.align_mode == "none":
        scale = 1.0
        global_rotation = np.eye(3, dtype=np.float64)
        global_translation = np.zeros(3, dtype=np.float64)
    elif args.align_mode == "scale":
        scale, global_translation = solve_scale_translation(vipe_pos, gt_pos)
        global_rotation = np.eye(3, dtype=np.float64)
    else:
        scale, global_rotation, global_translation = umeyama_similarity(vipe_pos, gt_pos)

    aligned_pos = scale * (global_rotation @ vipe_pos.T).T + global_translation
    aligned_rot = np.einsum("ij,njk->nik", global_rotation, vipe_rot)

    if args.rotation_mode == "absolute":
        rot_ref = gt_rot
        rot_est = aligned_rot
        rotation_offset = np.eye(3, dtype=np.float64)
    elif args.rotation_mode == "relative":
        rot_ref = np.einsum("ij,njk->nik", gt_rot[0].T, gt_rot)
        rot_est = np.einsum("ij,njk->nik", aligned_rot[0].T, aligned_rot)
        rotation_offset = np.eye(3, dtype=np.float64)
    else:
        rotation_offset = fit_rotation_offset(aligned_rot, gt_rot)
        rot_ref = gt_rot
        rot_est = np.einsum("nij,jk->nik", aligned_rot, rotation_offset)

    trans_error_before = np.linalg.norm(vipe_pos - gt_pos, axis=1)
    trans_error_after = np.linalg.norm(aligned_pos - gt_pos, axis=1)
    rot_error = rotation_errors_deg(rot_ref, rot_est)

    gt_step = np.zeros(frame_count, dtype=np.float64)
    aligned_step = np.zeros(frame_count, dtype=np.float64)
    step_error = np.zeros(frame_count, dtype=np.float64)
    rel_rot_error = np.zeros(frame_count, dtype=np.float64)
    if frame_count > 1:
        gt_delta = gt_pos[1:] - gt_pos[:-1]
        aligned_delta = aligned_pos[1:] - aligned_pos[:-1]
        gt_step[1:] = np.linalg.norm(gt_delta, axis=1)
        aligned_step[1:] = np.linalg.norm(aligned_delta, axis=1)
        step_error[1:] = np.linalg.norm(aligned_delta - gt_delta, axis=1)
        gt_rel_rot = np.einsum("nij,njk->nik", np.swapaxes(gt_rot[:-1], 1, 2), gt_rot[1:])
        est_rel_rot = np.einsum("nij,njk->nik", np.swapaxes(aligned_rot[:-1], 1, 2), aligned_rot[1:])
        rel_rot_error[1:] = rotation_errors_deg(gt_rel_rot, est_rel_rot)

    output_csv = args.output_csv or default_output_path(args.vipe_poses, "_vs_gt.csv")
    output_summary_json = args.output_summary_json or default_output_path(args.vipe_poses, "_vs_gt_summary.json")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_id",
                "gt_x",
                "gt_y",
                "gt_z",
                "vipe_x",
                "vipe_y",
                "vipe_z",
                "aligned_x",
                "aligned_y",
                "aligned_z",
                "translation_error_before",
                "translation_error_after",
                "rotation_error_deg",
                "relative_rotation_step_error_deg",
                "worldscore_rotation_error_deg",
                "worldscore_translation_error",
                "worldscore_scaled_pred_rel_x",
                "worldscore_scaled_pred_rel_y",
                "worldscore_scaled_pred_rel_z",
                "gt_rel_x",
                "gt_rel_y",
                "gt_rel_z",
                "pred_rel_x",
                "pred_rel_y",
                "pred_rel_z",
                "gt_step",
                "aligned_step",
                "translation_step_error",
            ]
        )
        for idx in range(frame_count):
            writer.writerow(
                [
                    frame_ids[idx],
                    *gt_pos[idx].tolist(),
                    *vipe_pos[idx].tolist(),
                    *aligned_pos[idx].tolist(),
                    float(trans_error_before[idx]),
                    float(trans_error_after[idx]),
                    float(rot_error[idx]),
                    float(rel_rot_error[idx]),
                    float(worldscore["rotation_error_deg"][idx]),
                    float(worldscore["translation_error"][idx]),
                    *worldscore["scaled_pred_eval_pos"][idx].tolist(),
                    *worldscore["gt_eval_pos"][idx].tolist(),
                    *worldscore["pred_eval_pos"][idx].tolist(),
                    float(gt_step[idx]),
                    float(aligned_step[idx]),
                    float(step_error[idx]),
                ]
            )

    summary = {
        "frame_count": frame_count,
        "frame_start": frame_ids[0],
        "frame_ids": frame_ids,
        "frame_manifest": str(args.frame_manifest) if args.frame_manifest is not None else None,
        "vipe_poses": str(args.vipe_poses),
        "gt_transforms_json": str(args.gt_transforms_json),
        "align_mode": args.align_mode,
        "rotation_mode": args.rotation_mode,
        "gt_rotation_source": args.gt_rotation_source,
        "gt_position_source": args.gt_position_source,
        "similarity_scale": float(scale),
        "similarity_rotation": global_rotation.tolist(),
        "similarity_translation": global_translation.tolist(),
        "rotation_offset": rotation_offset.tolist(),
        "gt_path_length": trajectory_length(gt_pos),
        "vipe_path_length": trajectory_length(vipe_pos),
        "aligned_path_length": trajectory_length(aligned_pos),
        "translation_before": summarize_values(trans_error_before),
        "translation_after": summarize_values(trans_error_after),
        "translation_step_error": summarize_values(step_error),
        "rotation_error_deg": summarize_values(rot_error),
        "relative_rotation_step_error_deg": summarize_values(rel_rot_error),
        "worldscore_mode": args.worldscore_mode,
        "worldscore_scale": float(worldscore["scale"]),
        "worldscore_scale_solver": worldscore["scale_solver"],
        "worldscore_fixed_world_rotation": worldscore["fixed_world_rotation"].tolist(),
        "worldscore_rotation_mean_deg": float(worldscore["rotation_error_deg"].mean()),
        "worldscore_translation_mean": float(worldscore["translation_error"].mean()),
        "worldscore_rotation_error_deg": summarize_values(worldscore["rotation_error_deg"]),
        "worldscore_translation_error": summarize_values(worldscore["translation_error"]),
        "output_csv": str(output_csv),
        "output_summary_json": str(output_summary_json),
    }
    with output_summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    rows = [
        ["worldscore_mode", args.worldscore_mode],
        ["worldscore_scale", f"{summary['worldscore_scale']:.9f}"],
        ["worldscore_scale_solver", summary["worldscore_scale_solver"]],
        ["worldscore_rotation_mean_deg", f"{summary['worldscore_rotation_mean_deg']:.9f}"],
        ["worldscore_rotation_rmse_deg", f"{summary['worldscore_rotation_error_deg']['rmse']:.9f}"],
        ["worldscore_rotation_max_deg", f"{summary['worldscore_rotation_error_deg']['max']:.9f}"],
        ["worldscore_translation_mean", f"{summary['worldscore_translation_mean']:.9f}"],
        ["worldscore_translation_rmse", f"{summary['worldscore_translation_error']['rmse']:.9f}"],
        ["worldscore_translation_max", f"{summary['worldscore_translation_error']['max']:.9f}"],
        ["frames", frame_count],
        ["align_mode", args.align_mode],
        ["rotation_mode", args.rotation_mode],
        ["gt_position_source", args.gt_position_source],
        ["gt_rotation_source", args.gt_rotation_source],
        ["similarity_scale", f"{scale:.9f}"],
        ["gt_path_length", f"{summary['gt_path_length']:.9f}"],
        ["vipe_path_length", f"{summary['vipe_path_length']:.9f}"],
        ["aligned_path_length", f"{summary['aligned_path_length']:.9f}"],
        ["translation_rmse_after", f"{summary['translation_after']['rmse']:.9f}"],
        ["translation_mean_after", f"{summary['translation_after']['mean']:.9f}"],
        ["translation_max_after", f"{summary['translation_after']['max']:.9f}"],
        ["translation_step_rmse", f"{summary['translation_step_error']['rmse']:.9f}"],
        ["rotation_rmse_deg", f"{summary['rotation_error_deg']['rmse']:.9f}"],
        ["rotation_mean_deg", f"{summary['rotation_error_deg']['mean']:.9f}"],
        ["rotation_max_deg", f"{summary['rotation_error_deg']['max']:.9f}"],
        ["relative_rotation_step_rmse_deg", f"{summary['relative_rotation_step_error_deg']['rmse']:.9f}"],
        ["csv", str(output_csv)],
        ["summary_json", str(output_summary_json)],
    ]
    print("Summary")
    print(format_table(rows, ["metric", "value"]))

    preview_count = max(0, min(args.preview_rows, frame_count))
    if preview_count:
        preview_rows = []
        for idx in range(preview_count):
            preview_rows.append(
                [
                    frame_ids[idx],
                    f"{trans_error_after[idx]:.9f}",
                    f"{rot_error[idx]:.9f}",
                    f"{worldscore['translation_error'][idx]:.9f}",
                    f"{worldscore['rotation_error_deg'][idx]:.9f}",
                    f"{rel_rot_error[idx]:.9f}",
                    f"{gt_step[idx]:.9f}",
                    f"{aligned_step[idx]:.9f}",
                    f"{step_error[idx]:.9f}",
                ]
            )
        print("\nPer-frame preview")
        print(
            format_table(
                preview_rows,
                [
                    "frame_id",
                    "t_err_after",
                    "rot_err_deg",
                    "ws_t_err",
                    "ws_rot_deg",
                    "rel_rot_step_deg",
                    "gt_step",
                    "aligned_step",
                    "step_err",
                ],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
