# Utils
from .utils import grayscale_node

# Upscale (add your others as you build them)
from .upscale import realesrgan_node
from .upscale import span_node

# Colorize
from .colorize import colorization_node
from .colorize import temporal_prop_node

# Denoise
from .denoise import scunet_node

# Interpolation
from .interpolation import rife_node