import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def load_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"{path} is expected to have shape [T, H, W], got {arr.shape}")
    return arr


def load_mask(path: Optional[str], depth: np.ndarray) -> np.ndarray:
    if path is None:
        return np.isfinite(depth) & (depth > 0)
    mask = np.load(path)
    if mask.shape != depth.shape:
        raise ValueError(f"Mask shape mismatch: {mask.shape} vs {depth.shape}")
    return mask.astype(bool)


def summarize(name: str, values: np.ndarray) -> str:
    return (
        f"{name}: min={values.min():.9f}, mean={values.mean():.9f}, "
        f"median={np.median(values):.9f}, max={values.max():.9f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two depth npy files on overlapping valid pixels.")
    parser.add_argument("reference", type=str, help="Reference depth npy path")
    parser.add_argument("target", type=str, help="Target depth npy path")
    parser.add_argument("--reference-mask", type=str, default=None, help="Optional reference valid mask npy path")
    parser.add_argument("--target-mask", type=str, default=None, help="Optional target valid mask npy path")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    tgt_path = Path(args.target)
    ref = load_npy(str(ref_path)).astype(np.float64)
    tgt = load_npy(str(tgt_path)).astype(np.float64)

    if ref.shape != tgt.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {tgt.shape}")

    ref_valid = load_mask(args.reference_mask, ref)
    tgt_valid = load_mask(args.target_mask, tgt)
    overlap = ref_valid & tgt_valid

    print(f"reference: {ref_path}")
    print(f"target:    {tgt_path}")
    print(f"shape:     {ref.shape}")
    print(f"reference_valid_ratio: {ref_valid.mean():.9f}")
    print(f"target_valid_ratio:    {tgt_valid.mean():.9f}")
    print(f"overlap_valid_ratio:   {overlap.mean():.9f}")

    if not overlap.any():
        print("No overlapping valid pixels.")
        return

    abs_err = np.abs(ref[overlap] - tgt[overlap])
    rel_err_ref = abs_err / np.maximum(ref[overlap], 1e-6)
    rel_err_tgt = abs_err / np.maximum(tgt[overlap], 1e-6)
    sq_err = (ref[overlap] - tgt[overlap]) ** 2
    rmse = float(np.sqrt(np.mean(sq_err)))

    frame_overlap = overlap.reshape(overlap.shape[0], -1).mean(axis=1)
    frame_abs_mean = np.zeros(overlap.shape[0], dtype=np.float64)
    valid_frame_ids = np.where(frame_overlap > 0)[0]
    for idx in valid_frame_ids:
        frame_mask = overlap[idx]
        frame_abs_mean[idx] = np.abs(ref[idx][frame_mask] - tgt[idx][frame_mask]).mean()

    worst_frame = int(frame_abs_mean.argmax())

    print(summarize("abs_error", abs_err))
    print(summarize("rel_error_vs_reference", rel_err_ref))
    print(summarize("rel_error_vs_target", rel_err_tgt))
    print(f"rmse: {rmse:.9f}")
    print(
        f"worst_frame_by_mean_abs_error: {worst_frame} "
        f"(mean_abs_error={frame_abs_mean[worst_frame]:.9f}, overlap_ratio={frame_overlap[worst_frame]:.9f})"
    )


if __name__ == "__main__":
    main()
