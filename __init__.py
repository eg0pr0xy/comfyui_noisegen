"""
ComfyUI-NoiseGen: Professional noise generation nodes for ComfyUI
"""

__version__ = "1.0.0"
__author__ = "eg0pr0xy"

from .noise_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# ComfyUI will look for these variables
WEB_DIRECTORY = "./web" 