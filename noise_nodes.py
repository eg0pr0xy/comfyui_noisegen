import os
import numpy as np
import torch
import torchaudio

# Import audio utils with fallback for direct execution
try:
    from .audio_utils import *
except ImportError:
    from audio_utils import *

class NoiseGeneratorNode:
    """Universal noise generator node with multiple noise types."""
    
    NOISE_TYPES = [
        "white",
        "pink", 
        "brown",
        "blue",
        "violet",
        "perlin",
        "bandlimited"
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_type": (cls.NOISE_TYPES, {"default": "white"}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {"default": 44100}),
                "amplitude": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "channels": ([1, 2], {"default": 1}),
                "stereo_mode": (["independent", "correlated", "decorrelated"], {"default": "independent"}),
                "stereo_width": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "low_freq": ("FLOAT", {"default": 100.0, "min": 20.0, "max": 20000.0, "step": 1.0}),
                "high_freq": ("FLOAT", {"default": 8000.0, "min": 20.0, "max": 20000.0, "step": 1.0}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 8}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_noise"
    CATEGORY = "🎵 NoiseGen"
    
    def generate_noise(self, noise_type, duration, sample_rate, amplitude, seed, channels, 
                      stereo_mode, stereo_width, frequency=1.0, low_freq=100.0, high_freq=8000.0, octaves=4):
        """Generate different types of noise based on the selected type."""
        
        # Validate parameters
        duration, sample_rate, amplitude, channels = validate_audio_params(duration, sample_rate, amplitude, channels)
        
        try:
            # Common parameters for all noise types
            common_params = {
                'duration': duration,
                'sample_rate': sample_rate, 
                'amplitude': amplitude,
                'seed': seed,
                'channels': channels,
                'stereo_mode': stereo_mode,
                'stereo_width': stereo_width
            }
            
            if noise_type == "white":
                audio_array = generate_white_noise(**common_params)
            elif noise_type == "pink":
                audio_array = generate_pink_noise(**common_params)
            elif noise_type == "brown":
                audio_array = generate_brown_noise(**common_params)
            elif noise_type == "blue":
                audio_array = generate_blue_noise(**common_params)
            elif noise_type == "violet":
                audio_array = generate_violet_noise(**common_params)
            elif noise_type == "perlin":
                audio_array = generate_perlin_noise(frequency=frequency, octaves=octaves, **common_params)
            elif noise_type == "bandlimited":
                audio_array = generate_bandlimited_noise(low_freq=low_freq, high_freq=high_freq, **common_params)
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")
            
            # Convert to ComfyUI audio format
            audio_output = numpy_to_comfy_audio(audio_array, sample_rate)
            
            return (audio_output,)
            
        except Exception as e:
            print(f"Error generating {noise_type} noise: {str(e)}")
            # Return silence on error
            if channels == 1:
                silence = np.zeros(int(duration * sample_rate), dtype=np.float32)
            else:
                silence = np.zeros((channels, int(duration * sample_rate)), dtype=np.float32)
            audio_output = numpy_to_comfy_audio(silence, sample_rate)
            return (audio_output,)

class WhiteNoiseNode:
    """Dedicated white noise generator with stereo support."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {"default": 44100}),
                "amplitude": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "channels": ([1, 2], {"default": 1}),
                "stereo_mode": (["independent", "correlated", "decorrelated"], {"default": "independent"}),
                "stereo_width": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🎵 NoiseGen/Basic"
    
    def generate(self, duration, sample_rate, amplitude, seed, channels, stereo_mode, stereo_width):
        """Generate white noise with optional stereo support."""
        duration, sample_rate, amplitude, channels = validate_audio_params(duration, sample_rate, amplitude, channels)
        
        try:
            audio_array = generate_white_noise(duration, sample_rate, amplitude, seed, 
                                             channels, stereo_mode, stereo_width)
            audio_output = numpy_to_comfy_audio(audio_array, sample_rate)
            return (audio_output,)
        except Exception as e:
            print(f"Error generating white noise: {str(e)}")
            if channels == 1:
                silence = np.zeros(int(duration * sample_rate), dtype=np.float32)
            else:
                silence = np.zeros((channels, int(duration * sample_rate)), dtype=np.float32)
            audio_output = numpy_to_comfy_audio(silence, sample_rate)
            return (audio_output,)

class PinkNoiseNode:
    """Dedicated pink noise generator."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {"default": 44100}),
                "amplitude": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🎵 NoiseGen/Basic"
    
    def generate(self, duration, sample_rate, amplitude, seed):
        """Generate pink noise."""
        duration, sample_rate, amplitude = validate_audio_params(duration, sample_rate, amplitude)
        
        try:
            audio_array = generate_pink_noise(duration, sample_rate, amplitude, seed)
            audio_output = numpy_to_comfy_audio(audio_array, sample_rate)
            return (audio_output,)
        except Exception as e:
            print(f"Error generating pink noise: {str(e)}")
            silence = np.zeros(int(duration * sample_rate), dtype=np.float32)
            audio_output = numpy_to_comfy_audio(silence, sample_rate)
            return (audio_output,)

class BrownNoiseNode:
    """Dedicated brown/red noise generator."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {"default": 44100}),
                "amplitude": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🎵 NoiseGen/Basic"
    
    def generate(self, duration, sample_rate, amplitude, seed):
        """Generate brown noise."""
        duration, sample_rate, amplitude = validate_audio_params(duration, sample_rate, amplitude)
        
        try:
            audio_array = generate_brown_noise(duration, sample_rate, amplitude, seed)
            audio_output = numpy_to_comfy_audio(audio_array, sample_rate)
            return (audio_output,)
        except Exception as e:
            print(f"Error generating brown noise: {str(e)}")
            silence = np.zeros(int(duration * sample_rate), dtype=np.float32)
            audio_output = numpy_to_comfy_audio(silence, sample_rate)
            return (audio_output,)

class PerlinNoiseNode:
    """Advanced Perlin-like noise generator for natural variations."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 8}),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {"default": 44100}),
                "amplitude": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🎵 NoiseGen/Advanced"
    
    def generate(self, duration, frequency, octaves, sample_rate, amplitude, seed):
        """Generate Perlin-like noise."""
        duration, sample_rate, amplitude = validate_audio_params(duration, sample_rate, amplitude)
        
        try:
            audio_array = generate_perlin_noise(duration, frequency, sample_rate, amplitude, octaves, seed)
            audio_output = numpy_to_comfy_audio(audio_array, sample_rate)
            return (audio_output,)
        except Exception as e:
            print(f"Error generating Perlin noise: {str(e)}")
            silence = np.zeros(int(duration * sample_rate), dtype=np.float32)
            audio_output = numpy_to_comfy_audio(silence, sample_rate)
            return (audio_output,)

class BandLimitedNoiseNode:
    """Band-limited noise generator with frequency filtering."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "low_frequency": ("FLOAT", {"default": 100.0, "min": 20.0, "max": 20000.0, "step": 1.0}),
                "high_frequency": ("FLOAT", {"default": 8000.0, "min": 20.0, "max": 20000.0, "step": 1.0}),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {"default": 44100}),
                "amplitude": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🎵 NoiseGen/Advanced"
    
    def generate(self, duration, low_frequency, high_frequency, sample_rate, amplitude, seed):
        """Generate band-limited noise."""
        duration, sample_rate, amplitude = validate_audio_params(duration, sample_rate, amplitude)
        
        # Ensure frequency order is correct
        if low_frequency >= high_frequency:
            low_frequency, high_frequency = high_frequency, low_frequency
        
        try:
            audio_array = generate_bandlimited_noise(duration, low_frequency, high_frequency, sample_rate, amplitude, seed)
            audio_output = numpy_to_comfy_audio(audio_array, sample_rate)
            return (audio_output,)
        except Exception as e:
            print(f"Error generating band-limited noise: {str(e)}")
            silence = np.zeros(int(duration * sample_rate), dtype=np.float32)
            audio_output = numpy_to_comfy_audio(silence, sample_rate)
            return (audio_output,)

class AudioSaveNode:
    """Save generated audio to file."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "NoiseGen_"}),
                "format": (["wav", "flac", "mp3"], {"default": "wav"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "filepath")
    FUNCTION = "save_audio"
    CATEGORY = "🎵 NoiseGen/Utils"
    OUTPUT_NODE = True
    
    def save_audio(self, audio, filename_prefix, format):
        """Save audio to file and return path."""
        try:
            # Get ComfyUI's output directory from folder_paths
            import folder_paths
            
            # Create audio subdirectory in ComfyUI's output folder
            output_dir = folder_paths.get_output_directory()
            audio_dir = os.path.join(output_dir, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            
            # Generate unique filename with timestamp
            import time
            timestamp = str(int(time.time()))
            filename = f"{filename_prefix}{timestamp}.{format}"
            filepath = os.path.join(audio_dir, filename)
            
            # Extract audio data
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # CRITICAL: Ensure tensor format for torchaudio.save compatibility
            # Convert to CPU if on GPU
            if hasattr(waveform, 'cpu'):
                waveform = waveform.cpu()
            
            # Ensure waveform is 2D for torchaudio.save [channels, samples]
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # [samples] -> [1, samples]
            elif waveform.ndim > 2:
                # Flatten to 2D if somehow higher dimension
                waveform = waveform.view(waveform.size(0), -1)
            
            # Ensure proper data type
            waveform = waveform.float()
            
            # Validate final tensor shape
            if waveform.ndim != 2:
                raise ValueError(f"Expected 2D tensor for torchaudio.save, got {waveform.ndim}D with shape {waveform.shape}")
            
            # Save using torchaudio
            torchaudio.save(filepath, waveform, sample_rate)
            
            print(f"✅ Audio saved to: {filepath}")
            print(f"📁 Output location: ComfyUI/output/audio/{filename}")
            return (audio, filepath)
            
        except Exception as e:
            print(f"❌ Error saving audio: {str(e)}")
            import traceback
            traceback.print_exc()
            return (audio, "Error: Could not save audio")

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "NoiseGenerator": NoiseGeneratorNode,
    "WhiteNoise": WhiteNoiseNode,
    "PinkNoise": PinkNoiseNode,
    "BrownNoise": BrownNoiseNode,
    "PerlinNoise": PerlinNoiseNode,
    "BandLimitedNoise": BandLimitedNoiseNode,
    "AudioSave": AudioSaveNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NoiseGenerator": "🎵 Noise Generator (Universal)",
    "WhiteNoise": "🎵 White Noise",
    "PinkNoise": "🎵 Pink Noise",
    "BrownNoise": "🎵 Brown Noise",
    "PerlinNoise": "🎵 Perlin Noise",
    "BandLimitedNoise": "🎵 Band-Limited Noise",
    "AudioSave": "🎵 Save Audio",
} 