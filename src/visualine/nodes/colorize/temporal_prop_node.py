import logging
from typing import Dict, Any
import torch
import torch.nn.functional as F
import kornia

from visualine.core.node_base import NodeBase
from visualine.core.resource_manager import ResourceManager
from visualine.models.archs.dis_wrapper import DISFlowWrapper
from visualine.models.archs.colorization_wrapper import DDColorArchWrapper

logger = logging.getLogger(__name__)

class TemporalColorPropNode(NodeBase):
    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.colorizer_model = self.config.get("model_filename", "ddcolor_artistic.pth")
        self.render_res = self.config.get("render_res", 512)
        self.fp16 = self.config.get("fp16", True)
        
        self.blend_alpha = self.config.get("blend_alpha", 0.85) 
        
        self.color_wrapper: DDColorArchWrapper | None = None
        self.flow_wrapper: DISFlowWrapper | None = None
        self._resource_manager: ResourceManager = ResourceManager()

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return

        logger.info(f"Setting up {self.node_name} with RAFT EMA Blending...")

        self.last_smoothed_lab = None
        self.last_gray_frame = None
        
        def color_loader(): return DDColorArchWrapper(
            model_filename=self.colorizer_model, render_res=self.render_res, fp16=self.fp16
        )
        color_cache_key = f"colorizer_{self.colorizer_model}_res{self.render_res}_fp16{self.fp16}"
        self.color_wrapper = self._resource_manager.get_model(
            model_name=color_cache_key, model_loader=color_loader, device=str(device)
        )

        def flow_loader(): return DISFlowWrapper()
        self.flow_wrapper = self._resource_manager.get_model(
            model_name="dis_flow_fast", model_loader=flow_loader, device=str(device)
        )
        
        self.is_setup = True

    def _rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        return kornia.color.rgb_to_lab(rgb / 255.0)
        
    def _lab_to_rgb(self, lab: torch.Tensor) -> torch.Tensor:
        return kornia.color.lab_to_rgb(lab) * 255.0

    def process(self, data: torch.Tensor) -> torch.Tensor:
        self.validate_input(data)
        T, C, H, W = data.shape
        device = data.device
        
        raw_rgb_batch = self.color_wrapper.predict(data)
        raw_lab_batch = self._rgb_to_lab(raw_rgb_batch)

        smoothed_lab_batch = torch.zeros_like(raw_lab_batch)

        if self.last_smoothed_lab is None:
            smoothed_lab_batch[0] = raw_lab_batch[0]
            prev_smoothed = raw_lab_batch[0:1]
            prev_luma = raw_lab_batch[0:1, 0:1]
            start_idx = 1
        else:
            prev_smoothed = self.last_smoothed_lab
            prev_luma = self.last_luma_frame
            start_idx = 0

        xx = torch.arange(0, W, device=device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=device).view(-1, 1).repeat(1, W)
        base_grid = torch.stack((xx, yy), dim=0).unsqueeze(0).float()

        for i in range(start_idx, T):
            curr_raw_lab = raw_lab_batch[i:i+1]
            curr_luma = curr_raw_lab[:, 0:1]
            
            flow = self.flow_wrapper.predict(prev_luma / 100.0, curr_luma / 100.0) 
            
            motion_mag = torch.sqrt(flow[:, 0:1, :, :]**2 + flow[:, 1:2, :, :]**2 + 1e-8)

            curr_ab = curr_raw_lab[:, 1:3, :, :]
            chroma = torch.sqrt(curr_ab[:, 0:1, :, :]**2 + curr_ab[:, 1:2, :, :]**2 + 1e-8)

            is_faint = torch.sigmoid((5.0 - chroma) / 2.0)
            is_static = torch.sigmoid((1.0 - motion_mag) / 0.5)
            suppression_mask = is_faint * is_static * 0.30 
            curr_ab = curr_ab * (1.0 - suppression_mask)

            grid = base_grid + flow
            grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / max(W - 1, 1) - 1.0
            grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / max(H - 1, 1) - 1.0
            grid = grid.permute(0, 2, 3, 1) 
            
            warped_prev_lab = F.grid_sample(prev_smoothed, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
            
            warped_prev_L = warped_prev_lab[:, 0:1] 
            warped_prev_ab = warped_prev_lab[:, 1:3]
            
            motion_factor = torch.clamp(motion_mag / 4.0, 0.0, 1.0)
            base_alpha = 0.98 - (motion_factor * 0.28)
            
            luma_diff = torch.abs(warped_prev_L - curr_luma) / 100.0 
            occlusion_penalty = torch.exp(-luma_diff * 20.0)
            
            ab_diff = torch.sqrt((warped_prev_ab[:, 0:1] - curr_ab[:, 0:1])**2 + 
                                 (warped_prev_ab[:, 1:2] - curr_ab[:, 1:2])**2 + 1e-8)
            ab_diff_norm = torch.clamp(ab_diff / 40.0, 0.0, 1.0)
            
            hallucination_gate = torch.exp(-luma_diff * 50.0) * ab_diff_norm
            hallucination_boost = hallucination_gate * 0.15 
            
            dynamic_alpha = torch.clamp((base_alpha * occlusion_penalty) + hallucination_boost, 0.0, 0.99)
            
            blended_ab = (dynamic_alpha * warped_prev_ab) + ((1.0 - dynamic_alpha) * curr_ab)
            blended_lab = torch.cat([curr_luma, blended_ab], dim=1)
            
            smoothed_lab_batch[i] = blended_lab[0]
            
            prev_smoothed = blended_lab
            prev_luma = curr_luma

        self.last_smoothed_lab = smoothed_lab_batch[-1:]
        self.last_luma_frame = prev_luma 

        return self._lab_to_rgb(smoothed_lab_batch)
        
    def teardown(self) -> None:
        self.last_smoothed_lab = None
        self.last_gray_frame = None
        self.color_wrapper = None
        self.flow_wrapper = None
        self.is_setup = False
        
        logger.info(f"{self.node_name} teardown complete.")