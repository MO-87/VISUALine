import logging
from contextlib import nullcontext
from typing import Any, Dict, Generator

import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor

from visualine.models.loader import get_model_path
from visualine.models.base_wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)


class SAM2VideoWrapper(BaseModelWrapper):
    use_torch: bool = True

    def __init__(self, model_filename: str, model_cfg: str, half: bool = True):
        """
        Args:
            model_filename: Name of the weights file, e.g. 'sam2_hiera_small.pt'
            model_cfg: SAM2 config name, e.g. 'sam2_hiera_s.yaml'

        Important:
            `half=True` here means "use CUDA autocast where safe".
            We do NOT call predictor.half(), because SAM2 video memory encoding
            can create float32 masks internally, causing dtype mismatch.
        """
        self.model_filename = model_filename
        self.model_cfg = model_cfg

        # Treat this as AMP preference, not hard model FP16 conversion.
        self.half = bool(half)

        self._device_str = "cpu"

        logger.info(f"Booting SAM2 Video Predictor ({self.model_cfg})...")

        model_path_str = str(get_model_path(self.model_filename))

        self.predictor = build_sam2_video_predictor(
            self.model_cfg,
            model_path_str,
            device="cpu",
        )

        self.predictor.eval()

    def to(self, device: torch.device) -> "SAM2VideoWrapper":
        target_device = torch.device(device)
        target_device_str = str(target_device)

        if target_device.type == "cpu" and self.half:
            logger.warning("CPU device detected. Disabling AMP/FP16 for SAM2.")
            self.half = False

        if self._device_str == target_device_str:
            return self

        self._device_str = target_device_str

        if hasattr(self.predictor, "to"):
            self.predictor = self.predictor.to(target_device)
        else:
            raise RuntimeError("SAM2 predictor does not support .to(device).")

        # Do NOT call self.predictor.half().
        # Keeping SAM2 weights in FP32 avoids:
        # Input type (float) and bias type (c10::Half) mismatch.

        self.predictor.eval()

        logger.info(
            f"SAM2 loaded on {self._device_str}. "
            f"AMP requested: {self.half}. Model dtype kept FP32."
        )

        return self

    def _amp_context(self):
        if self.half and "cuda" in self._device_str:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def predict(self, data: Dict[str, Any]) -> Any:
        """
        Dispatcher for SAM2's stateful video tracking API.

        Actions:
        - init
        - add_box
        - add_mask
        - propagate
        - reset
        """
        action = data.get("action")

        if action == "propagate":
            return self._propagate_generator(data["state"])

        with torch.inference_mode(), self._amp_context():
            if action == "init":
                return self.predictor.init_state(
                    video_path=data["video_path"]
                )

            if action == "add_box":
                box = self._prepare_box_for_sam2(data["box"])

                return self.predictor.add_new_points_or_box(
                    inference_state=data["state"],
                    frame_idx=int(data["frame_idx"]),
                    obj_id=int(data["obj_id"]),
                    box=box,
                )

            if action == "add_mask":
                mask = self._prepare_mask_for_sam2(data["mask"])

                return self.predictor.add_new_mask(
                    inference_state=data["state"],
                    frame_idx=int(data["frame_idx"]),
                    obj_id=int(data["obj_id"]),
                    mask=mask,
                )

            if action == "reset":
                self.predictor.reset_state(data["state"])
                return None

        raise ValueError(f"Unknown SAM2 action: {action}")

    def _propagate_generator(self, state: Any) -> Generator[Any, None, None]:
        """
        Wrap SAM2's propagate generator so inference_mode/autocast remain active
        while the generator is being iterated.
        """
        with torch.inference_mode(), self._amp_context():
            for item in self.predictor.propagate_in_video(state):
                yield item

    def _prepare_box_for_sam2(self, box: Any) -> np.ndarray:
        """
        Pass boxes as CPU numpy arrays.

        This avoids CPU/CUDA mismatch inside SAM2's add_new_points_or_box().
        SAM2 will move/convert internally as needed.
        """
        if isinstance(box, torch.Tensor):
            box_np = box.detach().float().cpu().numpy()
        else:
            box_np = np.asarray(box, dtype=np.float32)

        box_np = box_np.reshape(-1)

        if box_np.size < 4:
            raise ValueError(f"SAM2 box must contain 4 values, got shape {box_np.shape}")

        return box_np[:4].astype(np.float32)

    def _prepare_mask_for_sam2(self, mask: Any) -> np.ndarray:
        """
        Pass masks as CPU numpy arrays for SAM2 compatibility.
        """
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().float().cpu().numpy()
        else:
            mask_np = np.asarray(mask, dtype=np.float32)

        return mask_np.astype(np.float32)

    def cleanup(self) -> None:
        logger.debug(f"Cleaning up resources for SAM2 ({self.model_filename})...")

        if getattr(self, "predictor", None) is not None:
            try:
                if hasattr(self.predictor, "to"):
                    self.predictor = self.predictor.to("cpu")
            except Exception as e:
                logger.warning(f"Could not move SAM2 predictor to CPU: {e}")

            del self.predictor
            self.predictor = None