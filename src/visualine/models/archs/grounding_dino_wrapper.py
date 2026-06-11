import logging
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Any, Dict, List

# Grounding DINO imports
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict as gd_predict

from visualine.models.loader import get_model_path
from visualine.models.base_wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)

class GroundingDinoWrapper(BaseModelWrapper):
    use_torch: bool = True

    def __init__(self, model_filename: str, config_path: str):
        """
        Args:
            model_filename: Name of the weights file (e.g., 'groundingdino_swinb_cogcoor.pth')
            config_path: Absolute or relative path to the GroundingDINO config (.py file)
        """
        self.model_filename = model_filename
        self.config_path = config_path
        self._device_str: str = 'cpu'
        
        logger.info("Booting Grounding DINO engine...")
        model_path_str = str(get_model_path(self.model_filename))
        
        # Load model to CPU initially; Resource Manager will call to() later
        self.model = load_model(self.config_path, model_path_str)
        self.model.eval()
        
        logger.info("Switching Grounding DINO to deploy mode...")
        for param in self.model.parameters():
            param.requires_grad = False

    def to(self, device: torch.device) -> 'GroundingDinoWrapper':
        target_device_str = str(device)

        if self._device_str != target_device_str:
            self._device_str = target_device_str
            self.model = self.model.to(device)
            logger.info(f"Grounding DINO Model loaded on {self._device_str}.")
            
        return self

    @torch.inference_mode()
    def predict(self, data: Dict[str, Any]) -> List[np.ndarray]:
        """
        Runs object detection based on a text prompt.

        Args:
            data:
                - image: np.ndarray, BGR format from cv2
                - prompt: str
                - conf: float

        Returns:
            List of absolute pixel boxes:
                [x1, y1, x2, y2]
        """
        frame = data["image"]
        prompt = data.get("prompt", "")
        conf = float(data.get("conf", 0.35))
        text_threshold = float(data.get("text_threshold", 0.25))

        if frame is None or frame.ndim != 3:
            raise ValueError("GroundingDinoWrapper.predict expected a BGR image with shape (H, W, 3).")

        if not prompt:
            return []

        # GroundingDINO generally behaves better when prompts end with a period.
        prompt = prompt.strip()
        if not prompt.endswith("."):
            prompt += "."

        h_img, w_img, _ = frame.shape

        # BGR -> RGB -> PIL -> normalized tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_tensor, _ = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])(Image.fromarray(frame_rgb), None)

        img_tensor = img_tensor.to(self._device_str)

        # GroundingDINO returns normalized cx, cy, w, h.
        g_boxes, logits, phrases = gd_predict(
            self.model,
            img_tensor,
            prompt,
            conf,
            text_threshold,
        )

        found_boxes: List[np.ndarray] = []

        for box in g_boxes:
            cx, cy, bw, bh = box.detach().cpu().numpy().astype(np.float32)

            x1 = (cx - bw / 2.0) * w_img
            y1 = (cy - bh / 2.0) * h_img
            x2 = (cx + bw / 2.0) * w_img
            y2 = (cy + bh / 2.0) * h_img

            # Clip to valid image boundaries.
            x1 = float(np.clip(x1, 0, w_img - 1))
            y1 = float(np.clip(y1, 0, h_img - 1))
            x2 = float(np.clip(x2, 0, w_img))
            y2 = float(np.clip(y2, 0, h_img))

            if x2 <= x1 or y2 <= y1:
                continue

            found_boxes.append(
                np.array([x1, y1, x2, y2], dtype=np.float32)
            )

        return found_boxes

    def cleanup(self) -> None:
        logger.debug(f"Cleaning up resources for Grounding DINO ({self.model_filename})...")
        if getattr(self, 'model', None) is not None:
            try:
                self.model.to('cpu')
            except Exception as e:
                logger.warning(f"Could not move Grounding DINO model to CPU: {e}")
            del self.model
            self.model = None