import logging
import cv2
import torch
import numpy as np
from visualine.models.base_wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)

class DISFlowWrapper(BaseModelWrapper):
    use_torch: bool = True

    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def to(self, device: torch.device) -> 'DISFlowWrapper':
        self.device = device
        logger.info("DIS Optical Flow Engine initialized (CPU/GPU Hybrid).")
        return self

    def predict(self, prev_gray: torch.Tensor, curr_gray: torch.Tensor) -> torch.Tensor:
        prev_np = (prev_gray.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
        curr_np = (curr_gray.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
        
        flow = self.dis.calc(curr_np, prev_np, None)
        
        flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return flow_tensor
        
    def cleanup(self) -> None:
        self.dis = None