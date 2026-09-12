#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


DEFAULT_INPUT_ROOT = Path("/data3/songcx/dataset/genex_realword_top50")
DEFAULT_OUTPUT_ROOT = Path("/data3/songcx/dataset/genex_realword_top50_res512")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize all MP4 videos under a directory to a fixed resolution with bilinear interpolation."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Source directory containing MP4 videos. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Target directory for resized videos. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument("--width", type=int, default=1024, help="Target width. Default: 1024")
    parser.add_argument("--height", type=int, default=512, help="Target height. Default: 512")
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="Path to the ffmpeg binary. Default: ffmpeg",
    )
    parser.add_argument(
        "--fps-mode",
        choices=("passthrough", "auto", "cfr", "vfr"),
        default="passthrough",
        help="ffmpeg fps handling mode. Default: passthrough",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files. By default existing outputs are skipped.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="CRF for libx264 re-encoding. Lower means higher quality. Default: 18",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="libx264 preset. Default: medium",
    )
    return parser.parse_args()


def ensure_ffmpeg(ffmpeg_bin: str) -> None:
    if shutil.which(ffmpeg_bin) is None:
        raise FileNotFoundError(
            f"Cannot find ffmpeg binary: {ffmpeg_bin}. Please install ffmpeg or pass --ffmpeg-bin."
        )


@lru_cache(maxsize=None)
def ffmpeg_supports_fps_mode(ffmpeg_bin: str) -> bool:
    result = subprocess.run(
        [ffmpeg_bin, "-h", "full"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return "-fps_mode" in result.stdout


def build_sync_args(ffmpeg_bin: str, fps_mode: str) -> list[str]:
    if ffmpeg_supports_fps_mode(ffmpeg_bin):
        return ["-fps_mode", fps_mode]

    vsync_mapping = {
        "passthrough": "0",
        "cfr": "1",
        "vfr": "2",
        "auto": "-1",
    }
    return ["-vsync", vsync_mapping[fps_mode]]


def find_mp4_files(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.rglob("*.mp4") if path.is_file())


def build_ffmpeg_command(
    ffmpeg_bin: str,
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps_mode: str,
    overwrite: bool,
    crf: int,
    preset: str,
) -> list[str]:
    sync_args = build_sync_args(ffmpeg_bin, fps_mode)
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-map",
        "0:s?",
        "-map",
        "0:d?",
        "-vf",
        f"scale={width}:{height}:flags=bilinear",
        *sync_args,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-c:s",
        "copy",
        "-c:d",
        "copy",
        "-map_metadata",
        "0",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return command


def resize_videos(args: argparse.Namespace) -> int:
    input_root = args.input_root.expanduser()
    output_root = args.output_root.expanduser()

    if not input_root.exists():
        print(f"Input directory does not exist: {input_root}", file=sys.stderr)
        return 1
    if not input_root.is_dir():
        print(f"Input path is not a directory: {input_root}", file=sys.stderr)
        return 1

    ensure_ffmpeg(args.ffmpeg_bin)

    mp4_files = find_mp4_files(input_root)
    if not mp4_files:
        print(f"No .mp4 files found under: {input_root}", file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[Path, int]] = []

    print(f"Found {len(mp4_files)} MP4 files under {input_root}")
    print(f"Output directory: {output_root}")
    print(f"Target size: {args.width}x{args.height}")

    for index, input_path in enumerate(mp4_files, start=1):
        relative_path = input_path.relative_to(input_root)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(mp4_files)}] Skip existing: {output_path}")
            continue

        print(f"[{index}/{len(mp4_files)}] Resizing: {input_path} -> {output_path}")
        command = build_ffmpeg_command(
            ffmpeg_bin=args.ffmpeg_bin,
            input_path=input_path,
            output_path=output_path,
            width=args.width,
            height=args.height,
            fps_mode=args.fps_mode,
            overwrite=args.overwrite,
            crf=args.crf,
            preset=args.preset,
        )

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            failures.append((input_path, exc.returncode))
            print(
                f"[{index}/{len(mp4_files)}] Failed with exit code {exc.returncode}: {input_path}",
                file=sys.stderr,
            )

    success_count = len(mp4_files) - len(failures)
    print(f"Finished. Success: {success_count}, Failed: {len(failures)}")

    if failures:
        print("Failed files:", file=sys.stderr)
        for failed_path, return_code in failures:
            print(f"  {failed_path} (exit code: {return_code})", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(resize_videos(parse_args()))
