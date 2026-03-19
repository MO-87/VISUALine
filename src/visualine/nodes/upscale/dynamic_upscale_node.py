import logging
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List

from visualine.core.node_base import NodeBase
from visualine.core.resource_manager import ResourceManager
from visualine.models.archs.realesrgan_wrapper import RealESRGANArchWrapper

logger = logging.getLogger(__name__)

class DynamicUpscaleNode(NodeBase):
    """
    A node that performs resource-efficient upscaling by only processing 
    dynamic (moving) regions of a video.
    """
    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Configuration parameters
        self.model_filename: str = self.config.get("model_filename", "RealESRGAN_x4plus.pth")
        self.scale: int = self.config.get("scale", 4)
        self.threshold: float = self.config.get("threshold", 0.02)
        self.tile_size: int = self.config.get("tile_size", 64)
        self.padding: int = self.config.get("padding", 16)
        self.refresh_interval: int = self.config.get("refresh_interval", 60)
        self.batch_size: int = self.config.get("batch_size", 12)
        self.fp16: bool = self.config.get("fp16", True)

        # State for temporal logic
        self.prev_input: torch.Tensor | None = None
        self.prev_output: torch.Tensor | None = None
        self.frame_count: int = 0
        
        # Internals
        self.model_wrapper: RealESRGANArchWrapper | None = None
        self._resource_manager: ResourceManager = ResourceManager()
        self.block_size = self.tile_size + 2 * self.padding

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return

        logger.info(f"Setting up {self.node_name}...")
        
        # We use a unique key for the resource manager
        model_cache_key = f"dynamic_realesrgan_{self.model_filename}_s{self.scale}_fp16{self.fp16}"

        def model_loader():
            # We reuse the existing wrapper but we will call its internal .upsampler.model directly
            return RealESRGANArchWrapper(
                model_filename=self.model_filename,
                scale=self.scale,
                tile=0, # We handle tiling manually here
                half=self.fp16
            )
        
        self.model_wrapper = self._resource_manager.get_model(
            model_name=model_cache_key,
            model_loader=model_loader,
            device=str(device)
        )
        self.is_setup = True
        logger.info(f"{self.node_name} setup complete.")

    def process(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Processes a batch of frames. Since each frame depends on the previous one,
        we iterate through the batch.
        """
        if not self.is_setup or self.model_wrapper is None:
            raise RuntimeError(f"{self.node_name} process called before successful setup.")

        device = batch.device
        model = self.model_wrapper.upsampler.model
        dtype = torch.float16 if self.fp16 else torch.float32
        
        processed_frames = []

        for i in range(batch.shape[0]):
            # 1. Prepare current frame (B, C, H, W) -> (1, C, H, W), normalized 0-1
            curr_tensor = batch[i:i+1].to(dtype) / 255.0
            _, _, h, w = curr_tensor.shape
            
            # 2. Determine if we need a full refresh or just dynamic tiling
            is_refresh = (self.frame_count % self.refresh_interval == 0)
            
            # 3. Calculate dynamic mask
            rows, cols = math.ceil(h / self.tile_size), math.ceil(w / self.tile_size)
            
            if self.prev_input is None or is_refresh:
                # Force all tiles to be active
                mask = torch.ones((rows, cols), dtype=torch.bool, device=device)
            else:
                # Calculate difference with previous frame
                # Note: We assume same resolution as prev_input
                diff = torch.abs(curr_tensor - self.prev_input).mean(dim=1, keepdim=True)
                # Max pool to find which tiles have significant changes
                block_max = F.max_pool2d(diff, self.tile_size, self.tile_size, ceil_mode=True)
                mask = (block_max > self.threshold).squeeze()
                
                # Handle edge case for single-tile frames
                if mask.ndim == 0: mask = mask.unsqueeze(0).unsqueeze(0)

            active_idx = torch.nonzero(mask)
            
            # 4. Prepare output buffer
            out_h, out_w = h * self.scale, w * self.scale
            if self.prev_output is None or is_refresh:
                curr_output = torch.zeros((1, 3, out_h, out_w), dtype=dtype, device=device)
            else:
                curr_output = self.prev_output.clone()

            # 5. Process Active Tiles
            if len(active_idx) > 0:
                # Padding for tiles to avoid edge artifacts (mirror padding like the notebook)
                pad_t, pad_l = self.padding, self.padding
                pad_b, pad_r = self.padding + (rows * self.tile_size - h), self.padding + (cols * self.tile_size - w)
                
                padded = F.pad(curr_tensor, (pad_l, pad_r, pad_t, pad_b), mode='reflect').squeeze(0)
                
                # Extract tiles
                tiles = []
                for r, c in active_idx:
                    r_val, c_val = r.item(), c.item()
                    tile = padded[:, 
                                  r_val * self.tile_size : r_val * self.tile_size + self.block_size, 
                                  c_val * self.tile_size : c_val * self.tile_size + self.block_size]
                    tiles.append(tile)
                
                # Process tiles in mini-batches
                for j in range(0, len(tiles), self.batch_size):
                    tile_batch = torch.stack(tiles[j : j + self.batch_size])
                    
                    with torch.no_grad():
                        # Run the core RRDBNet / SRVGG model
                        out_tile_batch = model(tile_batch)
                    
                    # Stitch back
                    for k, out_tile in enumerate(out_tile_batch):
                        r, c = active_idx[j + k]
                        y, x = r.item() * self.tile_size * self.scale, c.item() * self.tile_size * self.scale
                        
                        # Remove padding from the upscaled tile
                        start_p = self.padding * self.scale
                        end_p = start_p + self.tile_size * self.scale
                        block = out_tile[:, start_p:end_p, start_p:end_p]
                        
                        # Handle boundaries
                        bh, bw = block.shape[1], block.shape[2]
                        ph, pw = min(bh, out_h - y), min(bw, out_w - x)
                        
                        if ph > 0 and pw > 0:
                            curr_output[0, :, y:y+ph, x:x+pw] = block[:, :ph, :pw]

            # 6. Finalize frame
            # Store states for next iteration
            self.prev_input = curr_tensor.clone()
            self.prev_output = curr_output.clone()
            self.frame_count += 1
            
            processed_frames.append(curr_output)

        # Combine processed batch and convert back to uint8 [0-255]
        final_batch = torch.cat(processed_frames, dim=0)
        return (torch.clamp(final_batch, 0.0, 1.0) * 255.0).float()

    def teardown(self) -> None:
        logger.debug(f"Tearing down {self.node_name}...")
        self.model_wrapper = None 
        self.prev_input = None
        self.prev_output = None
        self.is_setup = False
        logger.info(f"{self.node_name} teardown complete.")
