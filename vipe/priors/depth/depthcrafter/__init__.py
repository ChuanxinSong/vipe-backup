# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import torch
import numpy as np
from pathlib import Path
from typing import Literal, Optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType

class DepthCrafterModel(DepthEstimationModel):
    def __init__(self, 
                 unet_path: str = "tencent/DepthCrafter",
                 num_denoising_steps: int = 5,
                 guidance_scale: float = 1.0,
                 window_size: int = 110,
                 overlap: int = 25,
                 ppl_type: str = "depthcrafter",
                 cpu_offload: Optional[str] = None,
                 npy_path: Optional[str] = None) -> None:
        super().__init__()
        
        self.num_denoising_steps = num_denoising_steps
        self.guidance_scale = guidance_scale
        self.window_size = window_size
        self.overlap = overlap
        self.ppl_type = ppl_type
        self.npy_path = npy_path
        
        if npy_path is not None:
            print(f"DepthCrafter initialized in Cached Mode. Loading from: {npy_path}")
            return

        # Setup sys.path to import from DepthCrafter
        self.depthcrafter_root = Path(__file__).parents[4] / "DepthCrafter"
        if str(self.depthcrafter_root) not in sys.path:
            sys.path.append(str(self.depthcrafter_root))
            
        # Lazy imports of DepthCrafter components
        try:
            from cube_temporal_alignment import (
                apply_custom_cube_temporal_fusion_w_cube_fea_for_kv_processors_for_unet, 
                apply_custom_cube_temporal_fusion_processors_for_unet,
                inject_cube_branch_layers
            )
            from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
            from depthcrafter.depth_crafter_ppl_w_cube_temporal_w_fusion_w_cube_fea_for_kv import (
                DepthCrafterPipelinew_w_cube_temporal_w_fusion_w_cube_fea_for_kv, 
                DepthCrafterPipelinew_w_cube_temporal_w_fusion_w_cube_fea_for_kv_w_distortion_noise_annealed_weighting, 
                DepthCrafterPipelinew_w_cube_temporal_w_fusion_w_cube_fea_for_kv_w_distortion_noise_annealed_weighting_normal_0
            )
            from depthcrafter.depth_crafter_ppl_w_distortion_noise import (
                DepthCrafterPipeline_w_distortion_noise, 
                DepthCrafterPipeline_w_distortion_noise_annealed_weighting,
                DepthCrafterPipeline_w_distortion_noise_annealed_weighting_normal_0, 
                DepthCrafterPipeline_w_distortion_noise_weighting
            )
            from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
            from depthcrafter.unet_w_cube_temporal_w_fusion_w_cube_fea_for_kv import DiffusersUNetSpatioTemporalConditionModelDepthCrafter_ww_cube_temporal_w_fusion_w_cube_fea_for_kv
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(f"Failed to import DepthCrafter components. Error: {e}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 1. Initialize UNet ---
        print(f"Loading DepthCrafter UNet ({ppl_type}) from {unet_path}...")
        if "cube_temporal_fusion" in ppl_type:
            if "w_cube_fea_for_kv" in ppl_type:
                unet_cls = DiffusersUNetSpatioTemporalConditionModelDepthCrafter_ww_cube_temporal_w_fusion_w_cube_fea_for_kv
            else:
                unet_cls = DiffusersUNetSpatioTemporalConditionModelDepthCrafter
            
            unet = unet_cls.from_pretrained(unet_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, ignore_mismatched_sizes=True)
            inject_cube_branch_layers(unet)
            
            if "w_cube_fea_for_kv" in ppl_type: 
                apply_custom_cube_temporal_fusion_w_cube_fea_for_kv_processors_for_unet(unet)
            else: 
                apply_custom_cube_temporal_fusion_processors_for_unet(unet)
                
            weights_path = os.path.join(unet_path, "diffusion_pytorch_model.safetensors")
            unet.load_state_dict(load_file(weights_path), strict=False)
            unet.to(dtype=torch.float16, device=device)
        else:
            unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
                unet_path, low_cpu_mem_usage=True, torch_dtype=torch.float16
            )

        # --- 2. Initialize Pipeline ---
        print("Loading Pipeline...")
        svd_path = "stabilityai/stable-video-diffusion-img2vid-xt"
        if ppl_type == "depthcrafter":
            pipe = DepthCrafterPipeline.from_pretrained(svd_path, unet=unet, torch_dtype=torch.float16, variant="fp16")
        elif ppl_type in ["distortion_noise", "distortion_noise_annealed_weighting_normal_0"]:
            pipe = DepthCrafterPipeline_w_distortion_noise.from_pretrained(svd_path, unet=unet, torch_dtype=torch.float16, variant="fp16")
        elif ppl_type == "distortion_noise_w_weighting":
            pipe = DepthCrafterPipeline_w_distortion_noise_weighting.from_pretrained(svd_path, unet=unet, torch_dtype=torch.float16, variant="fp16")
        elif ppl_type == "distortion_noise_annealed_weighting":
            pipe = DepthCrafterPipeline_w_distortion_noise_annealed_weighting.from_pretrained(svd_path, unet=unet, torch_dtype=torch.float16, variant="fp16")
        elif "cube_temporal_fusion" in ppl_type:
            if "w_cube_fea_for_kv" in ppl_type:
                if "distortion_noise_annealed_weighting_normal_0" in ppl_type: 
                    pipe = DepthCrafterPipelinew_w_cube_temporal_w_fusion_w_cube_fea_for_kv_w_distortion_noise_annealed_weighting_normal_0.from_pretrained(svd_path, unet=unet, torch_dtype=torch.float16, variant="fp16")
                elif "distortion_noise_annealed_weighting" in ppl_type: 
                    pipe = DepthCrafterPipelinew_w_cube_temporal_w_fusion_w_cube_fea_for_kv_w_distortion_noise_annealed_weighting.from_pretrained(svd_path, unet=unet, torch_dtype=torch.float16, variant="fp16")
                else: 
                    pipe = DepthCrafterPipelinew_w_cube_temporal_w_fusion_w_cube_fea_for_kv.from_pretrained(svd_path, unet=unet, torch_dtype=torch.float16, variant="fp16")
            else:
                pipe = DepthCrafterPipeline.from_pretrained(svd_path, unet=unet, torch_dtype=torch.float16, variant="fp16")
        else:
            raise ValueError(f"Unknown pipeline type: {ppl_type}")

        if cpu_offload == "sequential": pipe.enable_sequential_cpu_offload()
        elif cpu_offload == "model": pipe.enable_model_cpu_offload()
        else: pipe.to(device)
        
        pipe.enable_attention_slicing()
        self.pipe = pipe

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DISTANCE

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        # Fallback to single frame if called
        rgb = src.rgb
        if rgb is None:
             raise ValueError("Input RGB is required for DepthCrafter estimation")
        
        depths = self.estimate_sequence(rgb[None])
        return DepthEstimationResult(metric_depth=depths[0])

    def estimate_sequence(self, rgb_sequence: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth for a sequence of frames.
        Args:
            rgb_sequence: [T, H, W, 3] float32 tensor in range [0, 1]
        Returns:
            depths: [T, H, W] float32 tensor (Distance, NOT disparity)
        """
        T, H, W, C = rgb_sequence.shape
        
        if self.npy_path is not None:
            if not os.path.exists(self.npy_path):
                raise FileNotFoundError(f"Cached disparity file not found: {self.npy_path}")
            
            print(f"Loading cached disparity from {self.npy_path}...")
            # Load disparity [T, H, W]
            res_disparity = np.load(self.npy_path)
            
            # Check shape - if disparity is [T, H, W, 3], average channels
            if len(res_disparity.shape) == 4:
                res_disparity = res_disparity.mean(-1)
            
            # Re-resize if dimensions don't match (Vipe usually uses different resolutions than depthcrafter outputs)
            if res_disparity.shape[1] != H or res_disparity.shape[2] != W:
                print(f"Resizing disparity from {res_disparity.shape[1:]} to {H}x{W}")
                import cv2
                resized_disp = []
                for t in range(T):
                    resized_disp.append(cv2.resize(res_disparity[t], (W, H), interpolation=cv2.INTER_LINEAR))
                res_disparity = np.stack(resized_disp)

            # Convert Disparity to Distance
            res_distance = 1.0 / (res_disparity + 1e-6)
            return torch.from_numpy(res_distance).cuda().float()

        frames_np = rgb_sequence.cpu().numpy()
        
        print(f"Running DepthCrafter on {T} frames ({W}x{H})...")
        
        with torch.inference_mode():
            res_3channel = self.pipe(
                frames_np, height=H, width=W, output_type="np", 
                guidance_scale=self.guidance_scale, 
                num_inference_steps=self.num_denoising_steps, 
                window_size=min(self.window_size, T), 
                overlap=0 if T <= self.window_size else self.overlap
            ).frames[0]
        
        # Post-process: Convert Disparity to Distance
        # res_3channel is in [0, 1] and represents disparity
        res_disparity = res_3channel.sum(-1) / res_3channel.shape[-1]
        res_distance = 1.0 / (res_disparity + 1e-6)
        
        # Convert back to torch tensor on GPU
        depths = torch.from_numpy(res_distance).cuda().float()
        
        return depths
