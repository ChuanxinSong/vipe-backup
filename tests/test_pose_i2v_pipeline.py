import argparse
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import compare_vipe_omniroam_pose as compare
import infer_vipe_panorama_pose as infer


REPO_ROOT = Path(__file__).resolve().parents[1]


class CudaReservationTest(unittest.TestCase):
    def test_reservation_fills_only_to_target_and_keeps_safety_margin(self):
        gib = 1024**3
        with (
            mock.patch.object(infer.torch.cuda, "is_available", return_value=True),
            mock.patch.object(infer.torch.cuda, "init") as cuda_init,
            mock.patch.object(infer.torch.cuda, "current_device", return_value=0),
            mock.patch.object(
                infer.torch.cuda,
                "memory_reserved",
                side_effect=[2 * gib, 10 * gib],
            ),
            mock.patch.object(
                infer.torch.cuda,
                "memory_allocated",
                side_effect=[1 * gib, 1 * gib],
            ),
            mock.patch.object(
                infer.torch.cuda,
                "mem_get_info",
                side_effect=[(20 * gib, 24 * gib), (12 * gib, 24 * gib)],
            ),
            mock.patch.object(infer.torch.cuda, "synchronize") as synchronize,
            mock.patch.object(infer.torch, "empty", return_value=object()) as empty,
        ):
            result = infer.reserve_cuda_memory(10.0, 2.0)

        cuda_init.assert_called_once_with()
        empty.assert_called_once_with(
            8 * gib,
            dtype=infer.torch.uint8,
            device=infer.torch.device("cuda", 0),
        )
        synchronize.assert_called_once_with(0)
        self.assertEqual(result["reserved_before_gib"], 2.0)
        self.assertEqual(result["reserved_after_gib"], 10.0)
        self.assertEqual(result["free_after_gib"], 12.0)

    def test_reservation_rejects_request_that_would_cross_safety_margin(self):
        gib = 1024**3
        with (
            mock.patch.object(infer.torch.cuda, "is_available", return_value=True),
            mock.patch.object(infer.torch.cuda, "init"),
            mock.patch.object(infer.torch.cuda, "current_device", return_value=0),
            mock.patch.object(infer.torch.cuda, "memory_reserved", return_value=2 * gib),
            mock.patch.object(infer.torch.cuda, "memory_allocated", return_value=1 * gib),
            mock.patch.object(infer.torch.cuda, "mem_get_info", return_value=(5 * gib, 24 * gib)),
            mock.patch.object(infer.torch, "empty") as empty,
        ):
            with self.assertRaisesRegex(RuntimeError, "Cannot satisfy requested CUDA reservation"):
                infer.reserve_cuda_memory(10.0, 2.0)
        empty.assert_not_called()


class PoseI2VInputTest(unittest.TestCase):
    def make_scene(self, root: Path, segments: list[dict]) -> Path:
        scene_id = "scene_a"
        scene_root = root / scene_id
        metadata_segments = []
        for segment_id, spec in enumerate(segments):
            saved_ids = spec["saved_ids"]
            crop = spec.get("crop", [0, 0, 8, 4])
            metadata_segments.append(
                {
                    "segment_id": segment_id,
                    "saved_frame_indices": saved_ids,
                    "frame_layout": {"generated_crop": crop},
                }
            )
            frames_dir = scene_root / f"segment_{segment_id:02d}" / "frames"
            frames_dir.mkdir(parents=True)
            for frame_id in saved_ids:
                image = np.zeros((9, 8, 3), dtype=np.uint8)
                image[:4, :, :] = np.array([10, 20, frame_id], dtype=np.uint8)
                image[5:, :, :] = np.array([100, 110, 120], dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(frames_dir / f"frame_{frame_id:04d}.png"), image))
        metadata = {
            "scene_id": scene_id,
            "output_format_version": "test_structural_format",
            "segments": metadata_segments,
        }
        (scene_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return scene_root

    @staticmethod
    def args(root: Path, scope: str, expected_segments: int = 2) -> argparse.Namespace:
        return argparse.Namespace(
            pose_i2v_results_root=root,
            pose_i2v_scope=scope,
            pose_i2v_expected_segments=expected_segments,
        )

    def test_first_segment_uses_native_metadata_crop_and_ignores_later_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_root = self.make_scene(root, [{"saved_ids": [1, 3]}, {"saved_ids": [4]}])
            metadata_path = scene_root / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["segments"][1]["segment_id"] = 99
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            source = infer.validate_pose_i2v_clip(
                {"id": "scene_a"}, self.args(root, "first_segment")
            )
            self.assertEqual(source["frame_ids"], [1, 3])
            self.assertEqual(source["frame_size"], (4, 8))
            image = cv2.imread(str(source["records"][0]["image_path"]), cv2.IMREAD_COLOR)
            cropped = infer.crop_comparison_image(image, source["generated_crop"])
            self.assertEqual(cropped.shape, (4, 8, 3))
            np.testing.assert_array_equal(cropped[0, 0], np.array([10, 20, 1], dtype=np.uint8))

    def test_all_segments_preserves_saved_ids_without_boundary_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_scene(root, [{"saved_ids": [1, 2]}, {"saved_ids": [3, 4]}])
            source = infer.validate_pose_i2v_clip({"id": "scene_a"}, self.args(root, "all_segments"))
            self.assertEqual(source["frame_ids"], [1, 2, 3, 4])
            self.assertEqual([record["segment_id"] for record in source["records"]], [0, 0, 1, 1])

    def test_segments_may_use_different_crop_coordinates_with_the_same_erp_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_scene(
                root,
                [
                    {"saved_ids": [1], "crop": [0, 0, 8, 4]},
                    {"saved_ids": [2], "crop": [0, 5, 8, 9]},
                ],
            )
            source = infer.validate_pose_i2v_clip({"id": "scene_a"}, self.args(root, "all_segments"))
            self.assertIsNone(source["generated_crop"])
            self.assertEqual(source["records"][0]["generated_crop"], (0, 0, 8, 4))
            self.assertEqual(source["records"][1]["generated_crop"], (0, 5, 8, 9))

    def test_all_segments_rejects_duplicate_or_non_increasing_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_scene(root, [{"saved_ids": [1, 2]}, {"saved_ids": [2, 3]}])
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                infer.validate_pose_i2v_clip({"id": "scene_a"}, self.args(root, "all_segments"))

    def test_all_segments_requires_exact_segment_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_scene(root, [{"saved_ids": [1, 2]}])
            with self.assertRaisesRegex(ValueError, "expected exactly 2"):
                infer.validate_pose_i2v_clip({"id": "scene_a"}, self.args(root, "all_segments"))

    def test_corrupt_comparison_png_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_root = self.make_scene(root, [{"saved_ids": [1]}])
            (scene_root / "segment_00" / "frames" / "frame_0001.png").write_bytes(b"not a png")
            with self.assertRaisesRegex(ValueError, "cannot be decoded"):
                infer.validate_pose_i2v_clip({"id": "scene_a"}, self.args(root, "first_segment"))

    def test_crop_rejects_out_of_bounds_coordinates(self):
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "outside image size"):
            infer.crop_comparison_image(image, (0, 0, 9, 4))

    def test_virtual_and_slam_sizes_match_selected_pipeline(self):
        virtual_cfg = argparse.Namespace(fovx=100.0)
        height, width, _ = infer.compute_virtual_view_size(virtual_cfg, 256)
        self.assertEqual((height, width), (256, 256))
        self.assertEqual(infer.compute_standard_slam_size((height, width)), (440, 440))

    def test_legacy_png_stream_still_resizes_to_requested_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "legacy.png"
            self.assertTrue(cv2.imwrite(str(image_path), np.zeros((4, 8, 3), dtype=np.uint8)))
            source = {"kind": "png_paths", "length": 1, "image_paths": [image_path]}
            stream = infer.create_video_stream(source, resolution=(16, 8), name="legacy")
            frame = next(iter(stream))
            self.assertEqual(stream.frame_size(), (8, 16))
            self.assertEqual(tuple(frame.rgb.shape), (8, 16, 3))


class PoseComparisonTest(unittest.TestCase):
    def test_scipy_scale_is_explicit_and_repeatable(self):
        prediction = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        ground_truth = prediction * 2.5
        first, first_solver = compare.solve_worldscore_scale(ground_truth, prediction, "scipy")
        second, second_solver = compare.solve_worldscore_scale(ground_truth, prediction, "scipy")
        self.assertAlmostEqual(first, 2.5, places=5)
        self.assertEqual(first, second)
        self.assertEqual(first_solver, "scipy_minimize_scalar")
        self.assertEqual(first_solver, second_solver)

    def test_manifest_non_contiguous_frame_ids_drive_gt_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_ids = [1, 3, 7]
            poses = []
            per_image = {}
            for index, frame_id in enumerate(frame_ids):
                pose = np.eye(4)
                pose[0, 3] = float(index)
                poses.append(pose.tolist())
                per_image[f"pano_camera0/frame_{frame_id:04d}.png"] = {
                    "location": {"x": float(index * 2), "y": 0.0, "z": 0.0},
                    "R": np.eye(3).tolist(),
                    "t": [float(-index * 2), 0.0, 0.0],
                }
            pose_path = root / "poses.json"
            manifest_path = root / "manifest.json"
            transforms_path = root / "transforms.json"
            csv_path = root / "metrics.csv"
            summary_path = root / "summary.json"
            pose_path.write_text(json.dumps(poses), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "frame_count": len(frame_ids),
                        "frames": [{"frame_id": frame_id} for frame_id in frame_ids],
                    }
                ),
                encoding="utf-8",
            )
            transforms_path.write_text(json.dumps({"per_image": per_image}), encoding="utf-8")

            command = [
                sys.executable,
                str(REPO_ROOT / "compare_vipe_omniroam_pose.py"),
                "--vipe_poses",
                str(pose_path),
                "--frame_manifest",
                str(manifest_path),
                "--gt_transforms_json",
                str(transforms_path),
                "--worldscore_scale_solver",
                "scipy",
                "--output_csv",
                str(csv_path),
                "--output_summary_json",
                str(summary_path),
                "--preview_rows",
                "0",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["frame_id"]) for row in rows], frame_ids)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["frame_ids"], frame_ids)
            self.assertEqual(summary["worldscore_scale_solver"], "scipy_minimize_scalar")

    def test_batch_evaluation_writes_macro_average_and_success_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_id = "scene_a"
            output_root = root / "outputs"
            poses_dir = output_root / "poses"
            gt_scene_dir = root / "gt" / scene_id
            poses_dir.mkdir(parents=True)
            gt_scene_dir.mkdir(parents=True)
            frame_ids = [2, 5, 9]
            poses = []
            per_image = {}
            for index, frame_id in enumerate(frame_ids):
                pose = np.eye(4)
                pose[0, 3] = float(index)
                poses.append(pose.tolist())
                per_image[f"pano_camera0/frame_{frame_id:04d}.png"] = {
                    "location": {"x": float(index * 3), "y": 0.0, "z": 0.0},
                    "R": np.eye(3).tolist(),
                    "t": [float(-index * 3), 0.0, 0.0],
                }
            (poses_dir / f"{scene_id}_poses.json").write_text(json.dumps(poses), encoding="utf-8")
            (poses_dir / f"{scene_id}_input_manifest.json").write_text(
                json.dumps(
                    {
                        "frame_count": len(frame_ids),
                        "frames": [{"frame_id": frame_id} for frame_id in frame_ids],
                    }
                ),
                encoding="utf-8",
            )
            (gt_scene_dir / "transforms.json").write_text(
                json.dumps({"per_image": per_image}), encoding="utf-8"
            )
            split_path = root / "split.json"
            split_path.write_text(json.dumps({"test": [scene_id]}), encoding="utf-8")
            command = [
                sys.executable,
                str(REPO_ROOT / "evaluate_vipe_pose_i2v.py"),
                "--input_root",
                str(root / "input"),
                "--output_root",
                str(output_root),
                "--scope",
                "first_segment",
                "--split_json",
                str(split_path),
                "--interiorgs_root",
                str(root / "gt"),
                "--worldscore_scale_solver",
                "scipy",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            summary = json.loads((output_root / "metrics_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["scope"], "first_segment")
            self.assertEqual(summary["succeeded"], 1)
            self.assertEqual(summary["failed"], 0)
            self.assertAlmostEqual(summary["macro_average"]["worldscore_translation_mean"], 0.0, places=5)
            with (output_root / "metrics_per_scene.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["status"], "success")


if __name__ == "__main__":
    unittest.main()
