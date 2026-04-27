import logging
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Union

from visualine.core.node_base import NodeBase
from visualine.core.resource_manager import ResourceManager
from visualine.models.archs.realesrgan_wrapper import RealESRGANArchWrapper
from visualine.models.archs.span_wrapper import SPANArchWrapper

logger = logging.getLogger(__name__)

class DynamicUpscaleNode(NodeBase):
    """
    An optimized node for dynamic VSR. Reduces Python overhead by using 
    vectorized tensor operations for tiling and stitching.
    Supports both RealESRGAN and SPAN architectures.
    """
    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Configuration parameters
        self.model_filename: str = self.config.get("model_filename", "realesr-animevideov3.pth")
        self.model_type: str = self.config.get("model_type", "realesrgan").lower()
        self.scale: int = self.config.get("scale", 4)
        self.threshold: float = self.config.get("threshold", 0.01)
        self.tile_size: int = self.config.get("tile_size", 32)
        self.padding: int = self.config.get("padding", 16)
        self.refresh_interval: int = self.config.get("refresh_interval", 60)
        self.batch_size: int = self.config.get("batch_size", 32)
        self.fp16: bool = self.config.get("fp16", True)
        self.debug_view: bool = self.config.get("debug_view", False)
        self.dilation: int = self.config.get("dilation", 1)
        self.edge_weight: float = self.config.get("edge_weight", 2.0)
        self.persistence: int = self.config.get("persistence", 2)
        self.global_threshold: float = self.config.get("global_threshold", 0.0001)
        
        # SPAN specific
        self.feature_channels: int = self.config.get("feature_channels", 48)

        # State for temporal logic
        self.prev_input: torch.Tensor | None = None
        self.prev_input_gray: torch.Tensor | None = None
        self.prev_output: torch.Tensor | None = None
        self.active_counters: torch.Tensor | None = None
        self.frame_count: int = 0
        
        # Internals
        self.model_wrapper: Union[RealESRGANArchWrapper, SPANArchWrapper, None] = None
        self._resource_manager: ResourceManager = ResourceManager()
        self.block_size = self.tile_size + 2 * self.padding
        self.sobel_x = None
        self.sobel_y = None

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return
        
        # Initialize Sobel kernels for edge detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        if self.fp16:
            self.sobel_x = self.sobel_x.half()
            self.sobel_y = self.sobel_y.half()
        
        # Auto-detect model type if not explicitly provided
        if "span" in self.model_filename.lower() and self.model_type == "realesrgan":
            self.model_type = "span"
            logger.info(f"Auto-detected SPAN model type for {self.model_filename}")

        logger.info(f"Setting up {self.node_name} ({self.model_type.upper()}) with {self.model_filename}...")
        model_cache_key = f"dynamic_vector_{self.model_type}_{self.model_filename}_s{self.scale}_fp16{self.fp16}"

        def model_loader():
            if self.model_type == "span":
                return SPANArchWrapper(
                    model_filename=self.model_filename,
                    scale=self.scale,
                    feature_channels=self.feature_channels,
                    half=self.fp16
                )
            else: # Default to RealESRGAN
                return RealESRGANArchWrapper(
                    model_filename=self.model_filename,
                    scale=self.scale,
                    tile=0,
                    half=self.fp16
                )
        
        self.model_wrapper = self._resource_manager.get_model(
            model_name=model_cache_key,
            model_loader=model_loader,
            device=str(device)
        )
        self.is_setup = True

    @torch.inference_mode()
    def process(self, batch: torch.Tensor) -> torch.Tensor:
        if not self.is_setup or self.model_wrapper is None:
            raise RuntimeError(f"{self.node_name} process called before successful setup.")

        device = batch.device
        
        # Extract the raw model from wrapper
        if self.model_type == "span":
            model = self.model_wrapper.model
        else:
            model = self.model_wrapper.upsampler.model
            
        dtype = torch.float16 if self.fp16 else torch.float32
        
        batch_bgr = batch[:, [2, 1, 0], :, :].to(dtype)
        processed_frames = []

        for i in range(batch_bgr.shape[0]):
            curr_tensor = batch_bgr[i:i+1] / 255.0
            _, _, h, w = curr_tensor.shape
            
            # Convert to grayscale for efficient difference and edge detection
            curr_gray = 0.299 * curr_tensor[:, 0:1] + 0.587 * curr_tensor[:, 1:2] + 0.114 * curr_tensor[:, 2:3]
            
            is_refresh = (self.frame_count % self.refresh_interval == 0) or (self.prev_input is None)
            rows, cols = math.ceil(h / self.tile_size), math.ceil(w / self.tile_size)
            
            # Initialize persistence counters
            if self.active_counters is None or self.active_counters.shape != (rows, cols):
                self.active_counters = torch.zeros((rows, cols), dtype=torch.int32, device=device)

            if is_refresh:
                mask = torch.ones((rows, cols), dtype=torch.bool, device=device)
                self.active_counters.fill_(self.persistence)
            else:
                # 1. Global Skip (Fast Reject)
                global_diff = torch.mean(torch.abs(curr_gray - self.prev_input_gray))
                if global_diff < self.global_threshold:
                    processed_frames.append(self.prev_output)
                    self.frame_count += 1
                    continue

                # 2. Luma-only difference and L1 Edge Detection
                diff = torch.abs(curr_gray - self.prev_input_gray)
                grad_x = F.conv2d(curr_gray, self.sobel_x, padding=1)
                grad_y = F.conv2d(curr_gray, self.sobel_y, padding=1)
                edge_mag = torch.abs(grad_x) + torch.abs(grad_y) # L1 is faster than sqrt
                
                # Weight the difference
                weighted_diff = diff * (1.0 + self.edge_weight * edge_mag)
                
                # 3. Block-level max pooling
                block_max = F.max_pool2d(weighted_diff, self.tile_size, self.tile_size, ceil_mode=True)
                current_mask = (block_max > self.threshold).squeeze()
                
                # Reshape if necessary
                if current_mask.ndim == 0: current_mask = current_mask.unsqueeze(0).unsqueeze(0)
                elif current_mask.ndim == 1: current_mask = current_mask.view(rows, cols)

                # 4. Update Persistence Counters
                self.active_counters[current_mask] = self.persistence
                mask = (self.active_counters > 0)
                # Decay counters
                self.active_counters = (self.active_counters - 1).clamp(min=0)

                # 5. Optional Mask Dilation (applied to the persistent mask)
                if self.dilation > 0:
                    mask_float = mask.float().unsqueeze(0).unsqueeze(0)
                    dilated_mask = F.max_pool2d(mask_float, kernel_size=2*self.dilation + 1, stride=1, padding=self.dilation)
                    mask = (dilated_mask > 0).squeeze()
                    if mask.ndim == 0: mask = mask.unsqueeze(0).unsqueeze(0)
                    elif mask.ndim == 1: mask = mask.view(rows, cols)

            active_mask = mask.flatten()
            num_active = active_mask.sum().item()
            
            # Prepare Output Buffer
            out_h, out_w = h * self.scale, w * self.scale
            grid_h, grid_w = rows * self.tile_size * self.scale, cols * self.tile_size * self.scale
            
            if is_refresh or self.prev_output is None:
                curr_output = torch.zeros((1, 3, grid_h, grid_w), dtype=dtype, device=device)
            else:
                # Copy prev_output into the new grid buffer
                curr_output = torch.zeros((1, 3, grid_h, grid_w), dtype=dtype, device=device)
                ph_c = min(self.prev_output.shape[2], grid_h)
                pw_c = min(self.prev_output.shape[3], grid_w)
                curr_output[0, :, :ph_c, :pw_c] = self.prev_output[0, :, :ph_c, :pw_c]

            # Vectorized Tiling and Inference
            if num_active > 0:
                pad_t, pad_l = self.padding, self.padding
                pad_b, pad_r = self.padding + (rows * self.tile_size - h), self.padding + (cols * self.tile_size - w)
                padded_input = F.pad(curr_tensor, (pad_l, pad_r, pad_t, pad_b), mode='reflect')
                
                # Extract all tiles
                tiles_unfolded = padded_input.unfold(2, self.block_size, self.tile_size).unfold(3, self.block_size, self.tile_size)
                all_tiles = tiles_unfolded.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, self.block_size, self.block_size)
                active_tiles = all_tiles[active_mask]
                
                out_tiles_list = []
                for j in range(0, num_active, self.batch_size):
                    out_tiles_list.append(model(active_tiles[j : j + self.batch_size]))
                
                out_tiles_all = torch.cat(out_tiles_list, dim=0)
                s_p, e_p = self.padding * self.scale, (self.padding + self.tile_size) * self.scale
                out_tiles_final = out_tiles_all[:, :, s_p:e_p, s_p:e_p]
                
                # Stitching
                out_ts = self.tile_size * self.scale
                curr_output_reshaped = curr_output.view(1, 3, rows, out_ts, cols, out_ts)
                curr_output_reshaped = curr_output_reshaped.permute(0, 2, 4, 1, 3, 5).reshape(-1, 3, out_ts, out_ts)
                curr_output_reshaped[active_mask] = out_tiles_final
                curr_output = curr_output_reshaped.reshape(1, rows, cols, 3, out_ts, out_ts).permute(0, 3, 1, 4, 2, 5).reshape(1, 3, grid_h, grid_w)

            # Crop grid to actual output resolution
            curr_output_final = curr_output[:, :, :out_h, :out_w]
            
            # STATE PERSISTENCE
            self.prev_input = curr_tensor.clone()
            self.prev_input_gray = curr_gray.clone()
            self.prev_output = curr_output_final.clone()
            self.frame_count += 1

            # DEBUG VIEW: Draw borders only on the current frame output
            if self.debug_view and num_active > 0:
                # Reuse the reshaped view to draw borders efficiently
                # We use out_tiles_final and re-stitch to avoid complex slice logic
                out_tiles_final[:, :, 0, :] = 1.0
                out_tiles_final[:, :, -1, :] = 1.0
                out_tiles_final[:, :, :, 0] = 1.0
                out_tiles_final[:, :, :, -1] = 1.0
                
                curr_output_reshaped = curr_output.view(1, 3, rows, out_ts, cols, out_ts)
                curr_output_reshaped = curr_output_reshaped.permute(0, 2, 4, 1, 3, 5).reshape(-1, 3, out_ts, out_ts)
                curr_output_reshaped[active_mask] = out_tiles_final
                
                # Refresh curr_output_final view
                curr_output_final = curr_output[:, :, :out_h, :out_w]

            processed_frames.append(curr_output_final)
        # Back to RGB uint8
        final_batch_bgr = torch.cat(processed_frames, dim=0)
        final_batch_rgb = final_batch_bgr[:, [2, 1, 0], :, :]
        return (torch.clamp(final_batch_rgb, 0.0, 1.0) * 255.0).float()

    def teardown(self) -> None:
        self.model_wrapper = None 
        self.prev_input = None
        self.prev_output = None
        self.frame_count = 0
        self.is_setup = False
        logger.info(f"{self.node_name} teardown complete.")
