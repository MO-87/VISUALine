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
        
        # SPAN specific
        self.feature_channels: int = self.config.get("feature_channels", 48)

        # State for temporal logic
        self.prev_input: torch.Tensor | None = None
        self.prev_output: torch.Tensor | None = None
        self.frame_count: int = 0
        
        # Internals
        self.model_wrapper: Union[RealESRGANArchWrapper, SPANArchWrapper, None] = None
        self._resource_manager: ResourceManager = ResourceManager()
        self.block_size = self.tile_size + 2 * self.padding

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return
        
        # Auto-detect model type if not explicitly provided or if it's the default
        model_name = self.model_filename.lower()
        if "span" in model_name:
            self.model_type = "span"
        else:
            self.model_type = "realesrgan"

        logger.info(f"Setting up {self.node_name} ({self.model_type.upper()}) with {self.model_filename}...")
        model_cache_key = f"dynamic_vector_{self.model_type}_{self.model_filename}_s{self.scale}_fp16{self.fp16}"

        def model_loader():
            if self.model_type == "span":
                return SPANArchWrapper(
                    model_filename=self.model_filename,
                    scale=self.scale,
                    feature_channels=self.feature_channels,
                    half=self.fp16,
                    tile_size=0 # We handle tiling in the node
                )
            else: # Default to RealESRGAN
                return RealESRGANArchWrapper(
                    model_filename=self.model_filename,
                    scale=self.scale,
                    tile=0, # We handle tiling in the node
                    half=self.fp16
                )
        
        self.model_wrapper = self._resource_manager.get_model(
            model_name=model_cache_key,
            model_loader=model_loader,
            device=str(device)
        )

        # SMART SYNC: If using TensorRT, force parameters to match the compiled engine
        if self.model_filename.endswith(".ts"):
            logger.info("TensorRT model detected. Syncing node parameters to engine specs.")
            self.tile_size = 64
            self.padding = 16
            self.block_size = 96
            self.batch_size = 1
            
        self.is_setup = True

    @torch.inference_mode()
    def process(self, data: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        if not self.is_setup or self.model_wrapper is None:
            raise RuntimeError(f"{self.node_name} process called before successful setup.")

        # 1. Handle Input (PipelineManager provides [0, 255] RGB)
        batch = data.get("tensor") if isinstance(data, dict) else data
        if not isinstance(batch, torch.Tensor):
            raise TypeError(f"{self.node_name} expects torch.Tensor, got {type(batch)}")

        device = batch.device
        dtype = torch.float16 if self.fp16 else torch.float32
        
        # 2. Preparation
        processed_frames = []

        for i in range(batch.shape[0]):
            curr_frame = batch[i:i+1] # [1, 3, H, W] RGB 0-255
            _, _, h, w = curr_frame.shape
            
            # Gridding logic
            rows, cols = math.ceil(h / self.tile_size), math.ceil(w / self.tile_size)
            is_refresh = (self.frame_count % self.refresh_interval == 0) or (self.prev_input is None)
            
            if is_refresh:
                mask = torch.ones((rows, cols), dtype=torch.bool, device=device)
            else:
                # Motion detection
                diff = torch.abs(curr_frame.float() / 255.0 - self.prev_input.float() / 255.0).mean(dim=1, keepdim=True)
                block_max = F.max_pool2d(diff, self.tile_size, self.tile_size, ceil_mode=True)
                mask = (block_max > self.threshold).squeeze()
                if mask.ndim == 0: mask = mask.view(1, 1)
                elif mask.ndim == 1: mask = mask.view(rows, cols)

            active_mask = mask.flatten()
            num_active = active_mask.sum().item()
            
            # Prepare Output Canvas
            out_h, out_w = h * self.scale, w * self.scale
            grid_h, grid_w = rows * self.tile_size * self.scale, cols * self.tile_size * self.scale
            
            # Create a clean canvas for every frame
            canvas = torch.zeros((1, 3, grid_h, grid_w), dtype=dtype, device=device)
            
            if not is_refresh and self.prev_output is not None:
                # Copy previous result into new grid (handles resolution changes if any)
                ph, pw = self.prev_output.shape[2], self.prev_output.shape[3]
                canvas[0, :, :ph, :pw] = self.prev_output[0].to(dtype)

            # 3. Processing Tiles
            if num_active > 0:
                # Padding
                pad_t, pad_l = self.padding, self.padding
                pad_b, pad_r = self.padding + (rows * self.tile_size - h), self.padding + (cols * self.tile_size - w)
                padded_input = F.pad(curr_frame, (pad_l, pad_r, pad_t, pad_b), mode='reflect')
                
                # Unfold into tiles
                tiles = padded_input.unfold(2, self.block_size, self.tile_size).unfold(3, self.block_size, self.tile_size)
                tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, self.block_size, self.block_size)
                
                active_tiles = tiles[active_mask]
                processed_tiles = []
                
                # Process tiles through the wrapper to ensure correct normalization
                # RealESRGANArchWrapper.predict expects [B, 3, H, W] RGB 0-255
                for j in range(0, num_active, self.batch_size):
                    t_batch = active_tiles[j : j + self.batch_size]
                    # The wrapper handles all complexity (BGR conversion, 0-1, SPAN mean, etc.)
                    t_out = self.model_wrapper.predict(t_batch)
                    processed_tiles.append(t_out.to(dtype))
                
                all_processed = torch.cat(processed_tiles, dim=0)
                
                # Crop padding from processed tiles
                s, e = self.padding * self.scale, (self.padding + self.tile_size) * self.scale
                final_tiles = all_processed[:, :, s:e, s:e]
                
                # 4. Stitching
                ts = self.tile_size * self.scale
                canvas_view = canvas.view(1, 3, rows, ts, cols, ts).permute(0, 2, 4, 1, 3, 5).reshape(-1, 3, ts, ts)
                canvas_view[active_mask] = final_tiles
                canvas = canvas_view.view(1, rows, cols, 3, ts, ts).permute(0, 3, 1, 4, 2, 5).reshape(1, 3, grid_h, grid_w)

            # Crop canvas to exact frame resolution
            final_frame = canvas[:, :, :out_h, :out_w]
            
            # Update state
            self.prev_input = curr_frame.clone()
            self.prev_output = final_frame.clone()
            self.frame_count += 1
            processed_frames.append(final_frame)

        # 5. Finalize Batch
        output_tensor = torch.cat(processed_frames, dim=0)
        return output_tensor.float().contiguous()

    def teardown(self) -> None:
        self.model_wrapper = None 
        self.prev_input = None
        self.prev_output = None
        self.frame_count = 0
        self.is_setup = False
        logger.info(f"{self.node_name} teardown complete.")
