import logging
import torch
import torch.nn.functional as F

from visualine.models.loader import get_model_path
from visualine.models.base_wrapper import BaseModelWrapper

from visualine.models.archs.ddcolor_arch import DDColor 

logger = logging.getLogger(__name__)

class DDColorArchWrapper(BaseModelWrapper):
    use_torch: bool = True

    def __init__(self, model_filename: str = "ddcolor_paper_tiny.pth", render_res: int = 512, fp16: bool = True):
        self.model_filename = model_filename
        self.render_res = render_res 
        self.fp16 = fp16
        self.model = None
        self._device_str = 'cpu'

    def to(self, device: torch.device) -> 'DDColorArchWrapper':
        target_device_str = str(device)
        if self.model is None or self._device_str != target_device_str:
            self._device_str = target_device_str
            model_path_str = str(get_model_path(self.model_filename))
            
            if "tiny" in self.model_filename.lower():
                logger.info("Tiny Colorizer detected. Calibrating for paper-tiny weights...")
                self.model = DDColor(
                    encoder_name='convnext-t',
                    decoder_name='MultiScaleColorDecoder',
                    last_norm='Spectral',
                    num_queries=100,
                    num_output_channels=2,
                    nf=512, 
                    dec_layers=9
                )
            else:
                logger.info("High-End Colorizer detected. Calibrating for Artistic/Large weights...")
                self.model = DDColor(
                    encoder_name='convnext-l',
                    decoder_name='MultiScaleColorDecoder', 
                    last_norm='Spectral',
                    num_queries=100,
                    num_output_channels=2,
                    nf=512,
                    dec_layers=9
                )
            
            state_dict = torch.load(model_path_str, map_location=device, weights_only=True)
            
            if 'params' in state_dict:
                state_dict = state_dict['params']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            self.model.load_state_dict(state_dict, strict=True)
            
            self.model.eval()
            self.model.to(device)
            
            for param in self.model.parameters():
                param.requires_grad = False
                
            # if self.fp16:
            #     self.model.half()

            logger.info(f"DDColor Engine '{self.model_filename}' initialized successfully.")
            
        return self

    @torch.inference_mode()
    def predict(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = batch_tensor.shape
        batch_norm = batch_tensor / 255.0

        r_chan = batch_norm[:, 0:1, :, :]
        mask_r = r_chan > 0.04045
        y = torch.where(mask_r, ((r_chan + 0.055) / 1.055) ** 2.4, r_chan / 12.92)
        
        mask_y = y > 0.008856
        f_y = torch.where(mask_y, torch.pow(torch.clamp(y, min=1e-5), 1.0/3.0), (7.787 * y) + (16.0 / 116.0))
        l_highres = (116.0 * f_y) - 16.0


        lowres_input = F.interpolate(batch_norm, size=(self.render_res, self.render_res), mode='bilinear', align_corners=False)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.fp16):
            ab_pred_lowres = self.model(lowres_input)

        ab_pred_lowres = ab_pred_lowres.float()

        ab_pred_highres = F.interpolate(ab_pred_lowres, size=(H, W), mode='bicubic', align_corners=False)

        y_inv = (l_highres + 16.0) / 116.0
        x_inv = (ab_pred_highres[:, 0:1, :, :] / 500.0) + y_inv
        z_inv = y_inv - (ab_pred_highres[:, 1:2, :, :] / 200.0)

        xyz = torch.cat([x_inv, y_inv, z_inv], dim=1)
        mask_xyz = xyz > 0.2068966
        xyz = torch.where(mask_xyz, xyz ** 3.0, (xyz - 16.0 / 116.0) / 7.787)

        xyz[:, 0, :, :] *= 0.95047
        xyz[:, 1, :, :] *= 1.00000
        xyz[:, 2, :, :] *= 1.08883

        r = xyz[:, 0:1, :, :] * 3.2404542 + xyz[:, 1:2, :, :] * -1.5371385 + xyz[:, 2:3, :, :] * -0.4985314
        g = xyz[:, 0:1, :, :] * -0.9692660 + xyz[:, 1:2, :, :] * 1.8760108 + xyz[:, 2:3, :, :] * 0.0415560
        b = xyz[:, 0:1, :, :] * 0.0556434 + xyz[:, 1:2, :, :] * -0.2040259 + xyz[:, 2:3, :, :] * 1.0572252

        final_rgb = torch.cat([r, g, b], dim=1)
        
        mask_rgb = final_rgb > 0.0031308
        final_rgb = torch.where(mask_rgb, 1.055 * (torch.clamp(final_rgb, min=1e-5) ** (1.0 / 2.4)) - 0.055, final_rgb * 12.92)

        final_rgb = torch.clamp(final_rgb, 0.0, 1.0)
        return (final_rgb * 255.0).float()

    def cleanup(self) -> None:
        logger.debug(f"Cleaning up resources for DDColor ({self.model_filename})...")
        if self.model:
            try:
                self.model.to('cpu')
            except Exception as e:
                logger.warning(f"Could not move torch model to CPU: {e}")
            del self.model
            self.model = None