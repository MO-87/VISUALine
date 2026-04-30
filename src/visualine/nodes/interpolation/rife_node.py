import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.amp import autocast

from visualine.core.node_base import NodeBase
from visualine.core.resource_manager import ResourceManager
from visualine.models.loader import get_model_path
from visualine.models.archs.IFNet_HDv3 import IFNet

logger = logging.getLogger(__name__)


class RIFENode(NodeBase):
    """
    VISUALine RIFE frame interpolation node.

    Supports:
    - 4D video sequence batch: (N, 3, H, W)
    - 5D video window:         (1, T, 3, H, W)

    Output:
    - First call with one frame:
        returns the original frame and stores it as history.

    - First call with N frames:
        returns:
            frame0, interp01, frame1, interp12, frame2, ...

    - Subsequent calls:
        uses the previous stored frame and returns:
            interp(prev, frame0), frame0, interp(frame0, frame1), frame1, ...

    This gives an overall output length of approximately:
        (input_frames * 2) - 1

    and doubles the effective FPS.
    """

    use_torch: bool = True

    @property
    def fps_multiplier(self) -> float:
        return 2.0

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model_filename: str = self.config.get("model_filename", "flownet.pkl")
        self.scale: float = float(self.config.get("scale", 1.0))
        self.fp16: bool = bool(self.config.get("fp16", False))

        self.model: IFNet | None = None
        self._resource_manager: ResourceManager = ResourceManager()

        # Stores the last normalized frame: (1, 3, H, W), range 0-1.
        self.last_frame_buffer: torch.Tensor | None = None

        logger.debug(f"{self.node_name} initialized with config: {self.config}")

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return

        device = torch.device(device)

        if device.type != "cuda" and self.fp16:
            logger.warning(
                f"{self.node_name}: fp16=True requested on non-CUDA device. "
                "Disabling fp16."
            )
            self.fp16 = False

        logger.info(f"Setting up {self.node_name}...")

        self.last_frame_buffer = None

        model_cache_key = (
            f"rife_ifnet_{self.model_filename}_scale{self.scale}_fp16{self.fp16}"
        )

        def model_loader():
            model_instance = IFNet()

            model_path = get_model_path(self.model_filename)
            load_net = torch.load(model_path, map_location="cpu")

            weights = {
                k.replace("module.", ""): v
                for k, v in load_net.items()
            }

            model_instance.load_state_dict(weights, strict=False)

            model_instance.eval()

            for param in model_instance.parameters():
                param.requires_grad = False

            if self.fp16:
                model_instance.half()

            model_instance = model_instance.to(memory_format=torch.channels_last)

            logger.info(f"IFNet model loaded from {self.model_filename}")
            return model_instance

        try:
            self.model = self._resource_manager.get_model(
                model_name=model_cache_key,
                model_loader=model_loader,
                device=str(device),
            )

            self.is_setup = True
            logger.info(f"{self.node_name} setup complete.")

        except Exception as e:
            logger.critical(
                f"Failed during {self.node_name} setup: {e}",
                exc_info=True,
            )
            raise

    def reset_state(self) -> None:
        """
        Clear temporal history.

        Important:
        PipelineManager may run a probe/sample pass before the real video pass.
        Without reset_state(), that probe frame can pollute interpolation state.
        """
        self.last_frame_buffer = None

    @torch.inference_mode()
    def process(self, data: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        if not self.is_setup or self.model is None:
            raise RuntimeError(
                f"{self.node_name} process called before successful setup."
            )

        if isinstance(data, dict):
            tensor = data.get("tensor")
        else:
            tensor = data

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{self.node_name} expects torch.Tensor, got {type(tensor)}"
            )

        sequence, original_ndim = self._flatten_input(tensor)

        N, C, H, W = sequence.shape

        if C != 3:
            raise ValueError(
                f"{self.node_name} expects RGB input with 3 channels, got {C}. "
                f"Input shape: {tuple(tensor.shape)}"
            )

        calc_dtype = torch.float16 if self.fp16 else torch.float32

        sequence_norm = (
            sequence
            .to(dtype=calc_dtype, copy=True)
            .clamp(0.0, 255.0)
            .mul_(1.0 / 255.0)
        )

        # First ever frame/window.
        if self.last_frame_buffer is None:
            self.last_frame_buffer = sequence_norm[-1:].detach().clone()

            # If only one frame exists, there is no pair to interpolate yet.
            if N == 1:
                output = sequence.float().clamp(0.0, 255.0)
                return self._restore_output(output, original_ndim)

            # Interpolate within the first window:
            # frame0, interp01, frame1, interp12, frame2, ...
            img0 = sequence_norm[:-1]
            img1 = sequence_norm[1:]

            interpolated = self._interpolate_pairs(
                img0=img0,
                img1=img1,
                height=H,
                width=W,
            )

            output_norm = torch.empty(
                (N * 2 - 1, C, H, W),
                dtype=sequence_norm.dtype,
                device=sequence_norm.device,
            )

            output_norm[0::2] = sequence_norm
            output_norm[1::2] = interpolated

            output = output_norm.mul(255.0).clamp(0.0, 255.0).float()
            return self._restore_output(output, original_ndim)

        # Subsequent frames/windows.
        # Pairs are:
        # previous -> current0
        # current0 -> current1
        # current1 -> current2
        # ...
        img0 = torch.cat(
            [
                self.last_frame_buffer.to(
                    device=sequence_norm.device,
                    dtype=sequence_norm.dtype,
                ),
                sequence_norm[:-1],
            ],
            dim=0,
        )

        img1 = sequence_norm

        self.last_frame_buffer = sequence_norm[-1:].detach().clone()

        interpolated = self._interpolate_pairs(
            img0=img0,
            img1=img1,
            height=H,
            width=W,
        )

        output_norm = torch.empty(
            (N * 2, C, H, W),
            dtype=sequence_norm.dtype,
            device=sequence_norm.device,
        )

        # Do not repeat the previous frame.
        # Previous frame was already emitted during the previous call.
        output_norm[0::2] = interpolated
        output_norm[1::2] = sequence_norm

        output = output_norm.mul(255.0).clamp(0.0, 255.0).float()

        return self._restore_output(output, original_ndim)

    def _flatten_input(self, tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Convert supported inputs into a sequence tensor:
            (N, 3, H, W)
        """
        original_ndim = tensor.ndim

        if original_ndim == 4:
            # Already sequence-like: (N, C, H, W)
            return tensor, original_ndim

        if original_ndim == 5:
            B, T, C, H, W = tensor.shape

            if B != 1:
                raise ValueError(
                    f"{self.node_name} currently supports 5D video windows with B=1, "
                    f"got B={B}. Shape: {tuple(tensor.shape)}"
                )

            return tensor.reshape(B * T, C, H, W), original_ndim

        raise ValueError(
            f"{self.node_name} expects 4D or 5D tensor, got {tuple(tensor.shape)}"
        )

    def _restore_output(self, output: torch.Tensor, original_ndim: int) -> torch.Tensor:
        """
        Restore output dimensionality.

        4D input:
            returns (N_out, 3, H, W)

        5D input:
            returns (1, T_out, 3, H, W)
        """
        if original_ndim == 4:
            return output

        if original_ndim == 5:
            N_out, C, H, W = output.shape
            return output.reshape(1, N_out, C, H, W)

        raise ValueError(f"Unsupported original_ndim: {original_ndim}")

    def _interpolate_pairs(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Interpolate between paired normalized RGB frames.

        Args:
            img0: (N, 3, H, W), range 0-1
            img1: (N, 3, H, W), range 0-1

        Returns:
            (N, 3, H, W), range approximately 0-1
        """
        if self.model is None:
            raise RuntimeError(f"{self.node_name} model is not initialized.")

        divisor = max(32, int(64.0 / max(self.scale, 1e-6)))

        pad_h = (divisor - (height % divisor)) % divisor
        pad_w = (divisor - (width % divisor)) % divisor

        if pad_h != 0 or pad_w != 0:
            img0 = F.pad(img0, (0, pad_w, 0, pad_h), mode="replicate")
            img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode="replicate")

        x = torch.cat((img0, img1), dim=1)

        if x.is_cuda:
            x = x.to(memory_format=torch.channels_last)

        scale_list = [
            16.0 / self.scale,
            8.0 / self.scale,
            4.0 / self.scale,
            2.0 / self.scale,
            1.0 / self.scale,
        ]

        device_type = "cuda" if x.is_cuda else "cpu"

        with autocast(device_type, enabled=self.fp16 and x.is_cuda):
            interpolated = self.model(
                x,
                timestep=0.5,
                scale_list=scale_list,
            )[2][-1]

        interpolated = interpolated[..., :height, :width]

        return interpolated.clamp(0.0, 1.0)

    def teardown(self) -> None:
        logger.debug(f"Tearing down {self.node_name}...")

        self.model = None
        self.last_frame_buffer = None
        self.is_setup = False

        logger.info(f"{self.node_name} teardown complete.")