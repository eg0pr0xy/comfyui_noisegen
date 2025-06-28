import os
import numpy as np
import torch
import torchaudio

# Import audio utils with fallback for direct execution
try:
    from .audio_utils import *
except ImportError:
    from audio_utils import *

# Import ComfyUI dependencies
try:
    import folder_paths
    import comfy.model_management
except ImportError:
    print("Warning: ComfyUI dependencies not found. Some features may not work.")
    folder_paths = None

# ComfyUI Audio Type Definition - This helps with node suggestions
AUDIO_TYPE = "AUDIO"
AUDIO_EXTENSIONS = ["wav", "mp3", "flac", "ogg", "aiff", "au"]

class NoiseGeneratorNode:
    """Universal noise generator for all noise types with stereo support.
    
    This node provides access to 7 different types of scientifically-accurate noise:
    - White: Flat frequency spectrum (equal energy per frequency)
    - Pink: 1/f spectrum (equal energy per octave) 
    - Brown: 1/f² spectrum (Brownian motion)
    - Blue: +3dB/octave (opposite of pink noise)
    - Violet: +6dB/octave (opposite of brown noise)
    - Perlin: Organic variations with controllable octaves
    - Band-limited: Filtered to specific frequency range
    """
    
    NOISE_TYPES = [
        "white",        # Pure static - flat frequency spectrum
        "pink",         # Natural balance - 1/f slope  
        "brown",        # Deep rumble - 1/f² slope
        "blue",         # Bright/harsh - +3dB/octave
        "violet",       # Ultra-bright - +6dB/octave
        "perlin",       # Organic textures - natural variations
        "bandlimited"   # Frequency filtered - targeted ranges
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_type": (cls.NOISE_TYPES, {
                    "default": "white",
                    "tooltip": "Type of noise to generate. White=flat spectrum, Pink=1/f, Brown=1/f², Blue=+3dB/oct, Violet=+6dB/oct, Perlin=organic, Band-limited=frequency filtered"
                }),
                "duration": ("FLOAT", {
                    "default": 5.0, 
                    "min": 0.1, 
                    "max": 300.0, 
                    "step": 0.1,
                    "tooltip": "Duration of the generated audio in seconds (0.1 to 300.0)"
                }),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {
                    "default": 44100,
                    "tooltip": "Audio sample rate in Hz. Higher rates = better quality, more processing"
                }),
                "amplitude": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.01,
                    "tooltip": "Peak amplitude/volume (0.0=silence, 1.0=normal, 2.0=loud)"
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 2147483647,
                    "tooltip": "Random seed for reproducible results. Same seed = same noise pattern"
                }),
                "channels": ([1, 2], {
                    "default": 1,
                    "tooltip": "Number of audio channels (1=mono, 2=stereo)"
                }),
                "stereo_mode": (["independent", "correlated", "decorrelated"], {
                    "default": "independent",
                    "tooltip": "Stereo mode: independent=different L/R, correlated=same L/R, decorrelated=phase-shifted"
                }),
                "stereo_width": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.1,
                    "tooltip": "Stereo image width (0.0=mono, 1.0=normal, 2.0=wide stereo)"
                }),
            },
            "optional": {
                "frequency": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1, 
                    "max": 100.0, 
                    "step": 0.1,
                    "tooltip": "[Perlin only] Base frequency for oscillations"
                }),
                "low_freq": ("FLOAT", {
                    "default": 100.0, 
                    "min": 1.0, 
                    "max": 20000.0, 
                    "step": 1.0,
                    "tooltip": "[Band-limited only] Low frequency cutoff in Hz"
                }),
                "high_freq": ("FLOAT", {
                    "default": 8000.0, 
                    "min": 1.0, 
                    "max": 20000.0, 
                    "step": 1.0,
                    "tooltip": "[Band-limited only] High frequency cutoff in Hz"
                }),
                "octaves": ("INT", {
                    "default": 4, 
                    "min": 1, 
                    "max": 8,
                    "tooltip": "[Perlin only] Number of octaves for complexity (more = more detail)"
                }),
            }
        }
    
    RETURN_TYPES = (AUDIO_TYPE,)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_noise"
    CATEGORY = "🎵 NoiseGen"
    DESCRIPTION = "Universal noise generator supporting 7 types of scientifically-accurate noise with professional stereo options"
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs and provide helpful error messages for ComfyUI"""
        errors = []
        
        if kwargs.get("duration", 5.0) < 0.1 or kwargs.get("duration", 5.0) > 300.0:
            errors.append("Duration must be between 0.1 and 300.0 seconds")
            
        if kwargs.get("amplitude", 0.8) < 0.0 or kwargs.get("amplitude", 0.8) > 2.0:
            errors.append("Amplitude must be between 0.0 and 2.0")
            
        if kwargs.get("high_freq", 8000.0) <= kwargs.get("low_freq", 100.0):
            errors.append("High frequency must be greater than low frequency for band-limited noise")
            
        # Return True for success, or error string for failure
        return True if len(errors) == 0 else "; ".join(errors)
    
    def generate_noise(self, noise_type, duration, sample_rate, amplitude, seed, channels, 
                      stereo_mode, stereo_width, frequency=1.0, low_freq=100.0, high_freq=8000.0, octaves=4):
        """Generate different types of noise based on the selected type."""
        
        # Fix parameter issues from malformed workflows
        try:
            # Convert string numbers to proper types if needed
            if isinstance(channels, str) and channels.replace('.', '').isdigit():
                channels = int(float(channels))
            elif isinstance(channels, str):
                # Handle string values that should be numeric
                channel_map = {"independent": 1, "correlated": 2, "decorrelated": 2}
                channels = channel_map.get(channels, 1)
                
            if isinstance(stereo_mode, str) and stereo_mode.replace('.', '').isdigit():
                # Handle numeric stereo_mode values
                mode_map = {"1": "independent", "2": "correlated", "1.5": "decorrelated", "1.8": "decorrelated"}
                stereo_mode = mode_map.get(stereo_mode, "independent")
                
            # Ensure valid stereo_mode
            if stereo_mode not in ["independent", "correlated", "decorrelated"]:
                stereo_mode = "independent"
                
            # Ensure valid channels
            if channels not in [1, 2]:
                channels = 1 if channels < 1.5 else 2
                
            # Fix frequency range issues
            low_freq = max(1.0, float(low_freq))
            high_freq = max(low_freq + 1.0, float(high_freq))
            
        except Exception as e:
            print(f"Parameter fix warning: {e}, using defaults")
            channels = 1
            stereo_mode = "independent"
            low_freq = 100.0
            high_freq = 8000.0
        
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
            
            # Convert to ComfyUI audio format with enhanced compatibility
            audio_output = numpy_to_comfy_audio(audio_array, sample_rate)
            
            # Add metadata for better ComfyUI integration
            audio_output["_metadata"] = {
                "noise_type": noise_type,
                "duration": duration,
                "channels": channels,
                "sample_rate": sample_rate,
                "generated_by": "NoiseGen"
            }
            
            return (audio_output,)
            
        except Exception as e:
            print(f"Error generating {noise_type} noise: {str(e)}")
            # Return silence on error with proper format
            if channels == 1:
                silence = np.zeros(int(duration * sample_rate), dtype=np.float32)
            else:
                silence = np.zeros((channels, int(duration * sample_rate)), dtype=np.float32)
            audio_output = numpy_to_comfy_audio(silence, sample_rate)
            return (audio_output,)

class WhiteNoiseNode:
    """Legacy white noise generator with stereo support."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {"default": 44100}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "channels": ([1, 2], {"default": 1}),
                "stereo_mode": (["independent", "correlated", "decorrelated"], {"default": "independent"}),
                "stereo_width": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = (AUDIO_TYPE,)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🎵 NoiseGen/Legacy"
    DESCRIPTION = "Dedicated white noise generator (use Universal NoiseGenerator for new workflows)"
    
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
    """Legacy pink noise generator."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {"default": 44100}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = (AUDIO_TYPE,)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🎵 NoiseGen/Basic"
    DESCRIPTION = "Dedicated pink noise generator (1/f frequency response)"
    
    def generate(self, duration, sample_rate, amplitude, seed):
        """Generate pink noise."""
        duration, sample_rate, amplitude, _ = validate_audio_params(duration, sample_rate, amplitude, 1)
        
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
    """Legacy brown/red noise generator."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {"default": 44100}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = (AUDIO_TYPE,)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🎵 NoiseGen/Basic"
    DESCRIPTION = "Dedicated brown noise generator (1/f² frequency response)"
    
    def generate(self, duration, sample_rate, amplitude, seed):
        """Generate brown noise."""
        duration, sample_rate, amplitude, _ = validate_audio_params(duration, sample_rate, amplitude, 1)
        
        try:
            audio_array = generate_brown_noise(duration, sample_rate, amplitude, seed)
            audio_output = numpy_to_comfy_audio(audio_array, sample_rate)
            return (audio_output,)
        except Exception as e:
            print(f"Error generating brown noise: {str(e)}")
            silence = np.zeros(int(duration * sample_rate), dtype=np.float32)
            audio_output = numpy_to_comfy_audio(silence, sample_rate)
            return (audio_output,)

class BlueNoiseNode:
    """Legacy blue noise generator."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {"default": 44100}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = (AUDIO_TYPE,)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🎵 NoiseGen/Basic"
    DESCRIPTION = "Dedicated blue noise generator (+3dB/octave frequency response)"
    
    def generate(self, duration, sample_rate, amplitude, seed):
        """Generate blue noise."""
        duration, sample_rate, amplitude, _ = validate_audio_params(duration, sample_rate, amplitude, 1)
        
        try:
            audio_array = generate_blue_noise(duration, sample_rate, amplitude, seed)
            audio_output = numpy_to_comfy_audio(audio_array, sample_rate)
            return (audio_output,)
        except Exception as e:
            print(f"Error generating blue noise: {str(e)}")
            silence = np.zeros(int(duration * sample_rate), dtype=np.float32)
            audio_output = numpy_to_comfy_audio(silence, sample_rate)
            return (audio_output,)

class VioletNoiseNode:
    """Legacy violet noise generator."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {"default": 44100}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = (AUDIO_TYPE,)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🎵 NoiseGen/Basic"
    DESCRIPTION = "Dedicated violet noise generator (+6dB/octave frequency response)"
    
    def generate(self, duration, sample_rate, amplitude, seed):
        """Generate violet noise."""
        duration, sample_rate, amplitude, _ = validate_audio_params(duration, sample_rate, amplitude, 1)
        
        try:
            audio_array = generate_violet_noise(duration, sample_rate, amplitude, seed)
            audio_output = numpy_to_comfy_audio(audio_array, sample_rate)
            return (audio_output,)
        except Exception as e:
            print(f"Error generating violet noise: {str(e)}")
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
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "NoiseGen/Advanced"
    
    def generate(self, duration, frequency, octaves, sample_rate, amplitude, seed):
        """Generate Perlin-like noise."""
        duration, sample_rate, amplitude, _ = validate_audio_params(duration, sample_rate, amplitude, 1)
        
        try:
            audio_array = generate_perlin_noise(duration, frequency, sample_rate, amplitude, octaves, seed)
            audio_output = numpy_to_comfy_audio(audio_array, sample_rate)
            return (audio_output,)
        except Exception as e:
            print(f"Error generating Perlin noise: {str(e)}")
            silence = np.zeros(int(duration * sample_rate), dtype=np.float32)
            audio_output = numpy_to_comfy_audio(silence, sample_rate)
            return (audio_output,)

class AudioMixerNode:
    """Professional audio mixer with individual channel controls.
    
    Provides traditional mixing capabilities with:
    - Individual gain controls for each input
    - Pan controls for stereo positioning
    - Clean addition-based mixing
    - Professional audio standards
    
    Perfect for: Music production, sound design, traditional audio work
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_a": ("AUDIO", {
                    "tooltip": "First audio input - any audio source"
                }),
                "gain_a": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.01,
                    "tooltip": "Volume level for input A (0.0=mute, 1.0=unity, 2.0=+6dB)"
                }),
                "pan_a": ("FLOAT", {
                    "default": 0.0, 
                    "min": -1.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Pan position for input A (-1.0=left, 0.0=center, 1.0=right)"
                }),
            },
            "optional": {
                "audio_b": ("AUDIO", {
                    "tooltip": "Second audio input (optional)"
                }),
                "gain_b": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.01,
                    "tooltip": "Volume level for input B"
                }),
                "pan_b": ("FLOAT", {
                    "default": 0.0, 
                    "min": -1.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Pan position for input B"
                }),
                "audio_c": ("AUDIO", {
                    "tooltip": "Third audio input (optional)"
                }),
                "gain_c": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.01,
                    "tooltip": "Volume level for input C"
                }),
                "pan_c": ("FLOAT", {
                    "default": 0.0, 
                    "min": -1.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Pan position for input C"
                }),
                "audio_d": ("AUDIO", {
                    "tooltip": "Fourth audio input (optional)"
                }),
                "gain_d": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.01,
                    "tooltip": "Volume level for input D"
                }),
                "pan_d": ("FLOAT", {
                    "default": 0.0, 
                    "min": -1.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Pan position for input D"
                }),
                "master_gain": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.01,
                    "tooltip": "Master output level (applied after mixing)"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("mixed_audio",)
    FUNCTION = "mix_audio"
    CATEGORY = "🎵 NoiseGen/Utils"
    DESCRIPTION = "Professional audio mixer with individual gain and pan controls for up to 4 inputs"
    
    def mix_audio(self, audio_a, gain_a, pan_a, audio_b=None, gain_b=1.0, pan_b=0.0, 
                  audio_c=None, gain_c=1.0, pan_c=0.0, audio_d=None, gain_d=1.0, pan_d=0.0, 
                  master_gain=1.0):
        """Mix multiple audio inputs with professional controls."""
        try:
            # Validate primary input
            if audio_a is None:
                raise ValueError("Audio input A is required")
            if not isinstance(audio_a, dict) or 'waveform' not in audio_a or 'sample_rate' not in audio_a:
                raise ValueError("Audio input A is not a valid audio object")
            
            # Get primary audio properties
            sample_rate = audio_a["sample_rate"]
            
            # Collect all valid inputs
            inputs = []
            gains = []
            pans = []
            
            # Process each input
            for audio, gain, pan, name in [
                (audio_a, gain_a, pan_a, "A"),
                (audio_b, gain_b, pan_b, "B"), 
                (audio_c, gain_c, pan_c, "C"),
                (audio_d, gain_d, pan_d, "D")
            ]:
                if audio is not None:
                    if not isinstance(audio, dict) or 'waveform' not in audio or 'sample_rate' not in audio:
                        print(f"Warning: Invalid audio input {name}, skipping")
                        continue
                    
                    # Extract and convert waveform
                    waveform = audio["waveform"]
                    if hasattr(waveform, 'cpu'):
                        waveform = waveform.cpu().numpy()
                    
                    # Ensure proper shape [channels, samples]
                    if waveform.ndim == 1:
                        waveform = waveform[np.newaxis, :]
                    elif waveform.ndim > 2:
                        waveform = waveform.view(waveform.size(0), -1)
                    
                    inputs.append(waveform.astype(np.float32))
                    gains.append(float(gain))
                    pans.append(float(pan))
                    
                    print(f"✅ Input {name}: {waveform.shape} @ {audio['sample_rate']}Hz, gain={gain:.2f}, pan={pan:.2f}")
            
            if len(inputs) == 0:
                raise ValueError("No valid audio inputs provided")
            
            # Determine output format
            max_channels = max(inp.shape[0] for inp in inputs)
            max_length = max(inp.shape[1] for inp in inputs)
            
            # Force stereo output for proper panning
            output_channels = max(2, max_channels)
            
            print(f"🎛️ Mixing {len(inputs)} inputs → {output_channels}ch, {max_length} samples")
            
            # Initialize output buffer
            mixed = np.zeros((output_channels, max_length), dtype=np.float32)
            
            # Mix each input
            for i, (waveform, gain, pan) in enumerate(zip(inputs, gains, pans)):
                # Apply gain
                scaled = waveform * gain
                
                # Extend to match output length
                if scaled.shape[1] < max_length:
                    padded = np.zeros((scaled.shape[0], max_length), dtype=np.float32)
                    padded[:, :scaled.shape[1]] = scaled
                    scaled = padded
                elif scaled.shape[1] > max_length:
                    scaled = scaled[:, :max_length]
                
                # Apply panning and add to mix
                if output_channels == 1:
                    # Mono output - ignore panning
                    if scaled.shape[0] == 1:
                        mixed[0] += scaled[0]
                    else:
                        # Mix down stereo to mono
                        mixed[0] += np.mean(scaled, axis=0)
                        
                elif output_channels == 2:
                    # Stereo output with panning
                    if scaled.shape[0] == 1:
                        # Mono input - apply panning
                        left_gain = np.sqrt((1.0 - pan) / 2.0) if pan >= 0 else 1.0
                        right_gain = np.sqrt((1.0 + pan) / 2.0) if pan <= 0 else 1.0
                        
                        mixed[0] += scaled[0] * left_gain
                        mixed[1] += scaled[0] * right_gain
                    else:
                        # Stereo input - apply pan as balance
                        if pan < 0:
                            # Pan left - reduce right channel
                            mixed[0] += scaled[0]
                            mixed[1] += scaled[1] * (1.0 + pan)
                        elif pan > 0:
                            # Pan right - reduce left channel  
                            mixed[0] += scaled[0] * (1.0 - pan)
                            mixed[1] += scaled[1]
                        else:
                            # Center - no change
                            mixed[:2] += scaled[:2]
                else:
                    # Multi-channel - simple addition (no panning)
                    channels_to_mix = min(scaled.shape[0], output_channels)
                    mixed[:channels_to_mix] += scaled[:channels_to_mix]
            
            # Apply master gain
            mixed *= master_gain
            
            # Prevent clipping with soft limiting
            max_amplitude = np.max(np.abs(mixed))
            if max_amplitude > 1.0:
                # Soft limiting
                mixed = np.tanh(mixed)
                print(f"⚠️ Applied soft limiting (peak was {max_amplitude:.2f})")
            
            # Convert back to ComfyUI format
            audio_output = numpy_to_comfy_audio(mixed, sample_rate)
            
            # Add mixing metadata
            audio_output["_metadata"] = {
                "mixed_inputs": len(inputs),
                "output_channels": output_channels,
                "master_gain": master_gain,
                "_generated_by": "ComfyUI-NoiseGen-AudioMixer"
            }
            
            print(f"🎵 Mixed audio: {output_channels}ch, {max_length} samples, {len(inputs)} inputs")
            
            return (audio_output,)
            
        except Exception as e:
            print(f"❌ Error in AudioMixer: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Fallback - return first input with gain applied
            try:
                if audio_a is not None:
                    fallback_waveform = audio_a["waveform"]
                    if hasattr(fallback_waveform, 'cpu'):
                        fallback_waveform = fallback_waveform.cpu().numpy()
                    
                    if fallback_waveform.ndim == 1:
                        fallback_waveform = fallback_waveform[np.newaxis, :]
                    
                    fallback_waveform = fallback_waveform.astype(np.float32) * gain_a * master_gain
                    audio_output = numpy_to_comfy_audio(fallback_waveform, audio_a["sample_rate"])
                    return (audio_output,)
            except:
                pass
            
            # Final emergency fallback
            emergency_audio = np.random.normal(0, 0.01, (1, 44100)).astype(np.float32)
            audio_output = numpy_to_comfy_audio(emergency_audio, 44100)
            return (audio_output,)

class ChaosNoiseMixNode:
    """
    Chaos Noise Mix - Extreme processing for harsh noise / Merzbow-style chaos
    
    MIX MODES EXPLAINED:
    - ADD       - Standard mixing (gentle)
    - MULTIPLY  - Ring modulation style (metallic)  
    - XOR       - Digital harsh mixing (glitchy)
    - MODULO    - Chaotic wrapping (unpredictable)
    - SUBTRACT  - Phase cancellation (hollow)
    - MAX/MIN   - Peak/trough selection (dynamic)
    - RING_MOD  - Classic ring modulation (carrier frequency)
    - AM_MOD    - Amplitude modulation (tremolo-like)
    - FM_MOD    - Frequency modulation style (complex harmonics)
    - CHAOS     - Extreme non-linear mixing (total devastation)
    
    Perfect for: Japanese noise, power electronics, experimental music, harsh textures
    """
    
    MIX_MODES = [
        "add",           # Standard addition
        "multiply",      # Ring modulation style
        "xor",          # Digital harsh mixing
        "modulo",       # Chaotic wrapping
        "subtract",     # Phase cancellation effects
        "max",          # Peak selection
        "min",          # Trough selection
        "ring_mod",     # Classic ring modulation
        "am_mod",       # Amplitude modulation
        "fm_mod",       # Frequency modulation style
        "chaos"         # Chaotic non-linear mixing
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_a": ("AUDIO", {
                    "tooltip": "First audio input - any audio source (NoiseGen, audio file, etc.)"
                }),
                "noise_b": ("AUDIO", {
                    "tooltip": "Second audio input - will be mixed with first audio"
                }),
                "mix_mode": (cls.MIX_MODES, {
                    "default": "add",
                    "tooltip": "Mixing algorithm: add=clean mixing, chaos=extreme, xor=digital harsh, ring_mod=carrier freq"
                }),
                "mix_ratio": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Mix balance (0.0=only A, 0.5=equal, 1.0=only B)"
                }),
                "chaos_amount": ("FLOAT", {
                    "default": 0.3, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Random chaos injection amount (0.0=clean, 1.0=maximum chaos)"
                }),
                "distortion": ("FLOAT", {
                    "default": 0.4, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Harsh distortion/saturation amount (0.0=clean, 1.0=maximum drive)"
                }),
                "bit_crush": ("INT", {
                    "default": 16, 
                    "min": 1, 
                    "max": 16,
                    "tooltip": "Bit depth for crushing (16=clean, 1=extreme digital artifacts)"
                }),
                "feedback": ("FLOAT", {
                    "default": 0.2, 
                    "min": 0.0, 
                    "max": 0.8, 
                    "step": 0.01,
                    "tooltip": "Feedback delay amount for metallic resonance (careful: can create runaway feedback!)"
                }),
                "ring_freq": ("FLOAT", {
                    "default": 440.0, 
                    "min": 1.0, 
                    "max": 5000.0, 
                    "step": 1.0,
                    "tooltip": "Carrier frequency for ring modulation and FM effects (Hz)"
                }),
                "amplitude": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.01,
                    "tooltip": "Final output amplitude/volume (0.0=silence, 1.0=normal, 2.0=loud)"
                }),
            },
            "optional": {
                "noise_c": ("AUDIO", {
                    "tooltip": "Optional third audio input for complex 3-way mixing"
                }),
                "modulation": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Amount of third input to blend in (requires noise_c connected)"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("chaos_audio",)
    FUNCTION = "mix_chaos"
    CATEGORY = "🎵 NoiseGen/Processing"
    DESCRIPTION = "Extreme audio mixing for harsh noise, power electronics, and experimental textures"
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate that required audio inputs are provided and compatible."""
        errors = []
        
        # More lenient validation - only check if values are explicitly provided
        # ComfyUI calls this during workflow loading when inputs might be None
        noise_a = kwargs.get('noise_a')
        noise_b = kwargs.get('noise_b')
        
        # Only validate if both inputs are actually provided (not None)
        if noise_a is not None and noise_b is not None:
            # Validate audio format if provided
            for input_name, audio_input in [('noise_a', noise_a), ('noise_b', noise_b)]:
                if audio_input is not None:
                    if not isinstance(audio_input, dict):
                        errors.append(f"{input_name}: Must be a valid audio object")
                    elif 'waveform' not in audio_input or 'sample_rate' not in audio_input:
                        errors.append(f"{input_name}: Audio object missing required waveform or sample_rate")
        
        # Always validate parameter ranges regardless of audio inputs
        mix_ratio = kwargs.get('mix_ratio', 0.5)
        if not (0.0 <= mix_ratio <= 1.0):
            errors.append("mix_ratio: Must be between 0.0 and 1.0")
            
        amplitude = kwargs.get('amplitude', 0.8)
        if not (0.0 <= amplitude <= 2.0):
            errors.append("amplitude: Must be between 0.0 and 2.0")
        
        # Return True for success, or error string for failure
        return True if len(errors) == 0 else "; ".join(errors)
    
    def mix_chaos(self, noise_a, noise_b, mix_mode, mix_ratio, chaos_amount, 
                  distortion, bit_crush, feedback, ring_freq, amplitude,
                  noise_c=None, modulation=0.0):
        """Create chaotic noise mixes for extreme audio textures."""
        try:
            # Validate inputs at runtime (more thorough than VALIDATE_INPUTS)
            if noise_a is None:
                raise ValueError("Audio input A (noise_a) is required but not connected")
            if noise_b is None:
                raise ValueError("Audio input B (noise_b) is required but not connected")
            
            if not isinstance(noise_a, dict) or 'waveform' not in noise_a or 'sample_rate' not in noise_a:
                raise ValueError("Audio input A is not a valid audio object")
            if not isinstance(noise_b, dict) or 'waveform' not in noise_b or 'sample_rate' not in noise_b:
                raise ValueError("Audio input B is not a valid audio object")
            
            # Extract waveforms and ensure same sample rate
            waveform_a = noise_a["waveform"]
            waveform_b = noise_b["waveform"]
            sample_rate = noise_a["sample_rate"]
            
            # Ensure sample rates match - use the first audio's sample rate
            if noise_b["sample_rate"] != sample_rate:
                print(f"Warning: Sample rate mismatch in ChaosNoiseMix - using {sample_rate}Hz")
            
            # Convert to CPU and numpy for processing
            if hasattr(waveform_a, 'cpu'):
                waveform_a = waveform_a.cpu().numpy()
            if hasattr(waveform_b, 'cpu'):
                waveform_b = waveform_b.cpu().numpy()
            
            # Handle different channel counts - expand mono to match stereo if needed
            if waveform_a.ndim == 1:
                waveform_a = waveform_a[np.newaxis, :]
            if waveform_b.ndim == 1:
                waveform_b = waveform_b[np.newaxis, :]
            
            # Match channel counts (broadcast smaller to larger)
            if waveform_a.shape[0] != waveform_b.shape[0]:
                target_channels = max(waveform_a.shape[0], waveform_b.shape[0])
                if waveform_a.shape[0] == 1 and target_channels == 2:
                    waveform_a = np.repeat(waveform_a, 2, axis=0)
                if waveform_b.shape[0] == 1 and target_channels == 2:
                    waveform_b = np.repeat(waveform_b, 2, axis=0)
            
            # Ensure same length (trim to shorter)
            min_len = min(waveform_a.shape[-1], waveform_b.shape[-1])
            waveform_a = waveform_a[..., :min_len]
            waveform_b = waveform_b[..., :min_len]
            
            # Validate audio data
            if min_len == 0:
                raise ValueError("Audio inputs have no samples")
            
            # Apply chaos mixing
            mixed = self._apply_chaos_mix(waveform_a, waveform_b, mix_mode, mix_ratio, 
                                        chaos_amount, sample_rate, ring_freq)
            
            # Add third noise source if provided
            if noise_c is not None:
                try:
                    if not isinstance(noise_c, dict) or 'waveform' not in noise_c:
                        print("Warning: Third audio input (noise_c) is invalid, skipping")
                    else:
                        waveform_c = noise_c["waveform"]
                        if hasattr(waveform_c, 'cpu'):
                            waveform_c = waveform_c.cpu().numpy()
                        
                        # Match dimensions
                        if waveform_c.ndim == 1:
                            waveform_c = waveform_c[np.newaxis, :]
                        
                        # Match channels and length
                        if waveform_c.shape[0] != mixed.shape[0]:
                            if waveform_c.shape[0] == 1 and mixed.shape[0] == 2:
                                waveform_c = np.repeat(waveform_c, 2, axis=0)
                            elif mixed.shape[0] == 1 and waveform_c.shape[0] == 2:
                                mixed = np.repeat(mixed, 2, axis=0)
                        
                        waveform_c = waveform_c[..., :min_len]
                        
                        # Mix in the third source with modulation
                        mixed = mixed * (1 - modulation) + waveform_c * modulation
                except Exception as c_error:
                    print(f"Warning: Failed to mix third noise source: {c_error}")
                    # Continue without third source
            
            # Apply harsh processing effects with error handling
            try:
                mixed = self._apply_distortion(mixed, distortion)
                mixed = self._apply_bit_crush(mixed, bit_crush)
                mixed = self._apply_feedback(mixed, feedback)
            except Exception as fx_error:
                print(f"Warning: Effects processing failed: {fx_error}")
                # Continue with unprocessed audio
            
            # Final amplitude scaling with safety checks
            max_amplitude = np.max(np.abs(mixed))
            if max_amplitude > 0 and not np.isnan(max_amplitude) and not np.isinf(max_amplitude):
                mixed = mixed / max_amplitude * amplitude
            else:
                # If audio is silent or has invalid values, create minimal noise
                mixed = np.random.normal(0, 0.01, mixed.shape).astype(np.float32) * amplitude
            
            # Ensure no NaN or Inf values
            mixed = np.nan_to_num(mixed, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Final safety clipping
            mixed = np.clip(mixed, -2.0, 2.0)
            
            # Convert back to ComfyUI format
            audio_output = numpy_to_comfy_audio(mixed, sample_rate)
            
            # Validate output format
            if not isinstance(audio_output, dict) or "waveform" not in audio_output:
                raise ValueError("Failed to create valid audio output")
            
            return (audio_output,)
            
        except Exception as e:
            print(f"❌ Error in ChaosNoiseMix: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Create robust fallback audio
            try:
                # Use the first input's properties for fallback if available
                if noise_a is not None and isinstance(noise_a, dict):
                    fallback_sample_rate = noise_a.get("sample_rate", 44100)
                    fallback_waveform = noise_a.get("waveform", None)
                    
                    if fallback_waveform is not None:
                        if hasattr(fallback_waveform, 'cpu'):
                            fallback_audio = fallback_waveform.cpu().numpy()
                        else:
                            fallback_audio = np.array(fallback_waveform)
                        
                        # Scale down the fallback audio
                        if np.max(np.abs(fallback_audio)) > 0:
                            fallback_audio = fallback_audio * 0.1 * amplitude
                    else:
                        # Generate minimal noise as last resort
                        fallback_audio = np.random.normal(0, 0.05, (1, int(fallback_sample_rate * 1.0))).astype(np.float32) * amplitude
                else:
                    # Complete fallback when no valid input
                    fallback_sample_rate = 44100
                    fallback_audio = np.random.normal(0, 0.05, (1, int(fallback_sample_rate * 1.0))).astype(np.float32) * amplitude
                
                # Ensure proper format
                if fallback_audio.ndim == 1:
                    fallback_audio = fallback_audio[np.newaxis, :]
                
                audio_output = numpy_to_comfy_audio(fallback_audio, fallback_sample_rate)
                return (audio_output,)
                
            except Exception as fallback_error:
                print(f"❌ Fallback audio creation failed: {fallback_error}")
                # Final emergency fallback
                emergency_audio = np.random.normal(0, 0.01, (1, 44100)).astype(np.float32)
                audio_output = numpy_to_comfy_audio(emergency_audio, 44100)
                return (audio_output,)
    
    def _apply_chaos_mix(self, a, b, mode, ratio, chaos, sample_rate, ring_freq):
        """Apply various chaotic mixing algorithms."""
        # Add some chaos/randomization
        if chaos > 0:
            chaos_mod = np.random.uniform(-chaos, chaos, a.shape) * 0.1
            a = a + chaos_mod
        
        if mode == "add":
            return a * (1 - ratio) + b * ratio
        
        elif mode == "multiply":
            return a * b * (1 + ratio)
        
        elif mode == "xor":
            # Digital XOR-style harsh mixing
            a_int = (a * 32767).astype(np.int16)
            b_int = (b * 32767).astype(np.int16)
            result = np.bitwise_xor(a_int, b_int) / 32767.0
            return result * ratio + a * (1 - ratio)
        
        elif mode == "modulo":
            # Chaotic modulo wrapping
            return np.fmod(a + b * ratio, 1.0) * 2 - 1
        
        elif mode == "subtract":
            return a - b * ratio
        
        elif mode == "max":
            return np.maximum(a, b * ratio)
        
        elif mode == "min":
            return np.minimum(a, b * ratio)
        
        elif mode == "ring_mod":
            # Classic ring modulation
            t = np.arange(a.shape[-1]) / sample_rate
            carrier = np.sin(2 * np.pi * ring_freq * t)
            if a.ndim == 2:
                carrier = np.broadcast_to(carrier, a.shape)
            return a * (1 + b * carrier * ratio)
        
        elif mode == "am_mod":
            # Amplitude modulation
            return a * (1 + b * ratio)
        
        elif mode == "fm_mod":
            # Frequency modulation style
            t = np.arange(a.shape[-1]) / sample_rate
            fm_signal = np.sin(2 * np.pi * ring_freq * t + b * ratio * 10)
            if a.ndim == 2:
                fm_signal = np.broadcast_to(fm_signal, a.shape)
            return a * fm_signal
        
        elif mode == "chaos":
            # Extreme chaotic non-linear mixing
            return np.tanh((a * 2) + (b * ratio * 3)) * np.sin(a * b * 50 * ratio)
        
        else:
            # Default to add
            return a * (1 - ratio) + b * ratio
    
    def _apply_distortion(self, audio, amount):
        """Apply harsh distortion/saturation."""
        if amount <= 0:
            return audio
        
        # Multiple distortion stages for extreme harshness
        drive = 1 + amount * 20
        distorted = np.tanh(audio * drive) / drive
        
        # Add some asymmetric clipping for extra harshness
        clip_threshold = 1.0 - amount * 0.5
        distorted = np.clip(distorted, -clip_threshold, clip_threshold * 0.8)
        
        return distorted
    
    def _apply_bit_crush(self, audio, bits):
        """Apply bit crushing for digital artifacts."""
        if bits >= 16:
            return audio
        
        # Quantize to fewer bits
        levels = 2 ** bits
        quantized = np.round(audio * levels) / levels
        return quantized
    
    def _apply_feedback(self, audio, amount):
        """Apply feedback delay for metallic resonance."""
        if amount <= 0:
            return audio
        
        # Simple feedback delay
        delay_samples = int(0.001 * 44100)  # 1ms delay
        feedback_audio = audio.copy()
        
        if audio.ndim == 1:
            for i in range(delay_samples, len(audio)):
                feedback_audio[i] += feedback_audio[i - delay_samples] * amount
        else:
            for i in range(delay_samples, audio.shape[1]):
                feedback_audio[:, i] += feedback_audio[:, i - delay_samples] * amount
        
        # Prevent runaway feedback
        if np.max(np.abs(feedback_audio)) > 2.0:
            feedback_audio = feedback_audio / np.max(np.abs(feedback_audio)) * 1.5
        
        return feedback_audio

class BandLimitedNoiseNode:
    """Band-limited noise generator with frequency filtering."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "low_frequency": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 20000.0, "step": 1.0}),
                "high_frequency": ("FLOAT", {"default": 8000.0, "min": 1.0, "max": 20000.0, "step": 1.0}),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {"default": 44100}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "NoiseGen/Advanced"
    
    def generate(self, duration, low_frequency, high_frequency, sample_rate, amplitude, seed):
        """Generate band-limited noise."""
        duration, sample_rate, amplitude, _ = validate_audio_params(duration, sample_rate, amplitude, 1)
        
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
    """Professional audio export with metadata preservation.
    
    Saves generated audio to various formats while preserving generation metadata
    for compatibility with audio analysis tools and DAWs.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (AUDIO_TYPE, {
                    "tooltip": "Audio data to save (connects to any NoiseGen audio output)"
                }),
                "filename_prefix": ("STRING", {
                    "default": "NoiseGen_",
                    "tooltip": "Prefix for the saved filename (timestamp will be appended)"
                }),
                "format": (["wav", "flac", "mp3"], {
                    "default": "wav",
                    "tooltip": "Audio format: WAV=uncompressed, FLAC=lossless, MP3=compressed"
                }),
            }
        }
    
    RETURN_TYPES = (AUDIO_TYPE, "STRING")
    RETURN_NAMES = ("audio", "filepath")
    FUNCTION = "save_audio"
    CATEGORY = "🎵 NoiseGen/Utils"
    OUTPUT_NODE = True
    DESCRIPTION = "Save audio to file with metadata preservation and return the file path"
    
    def save_audio(self, audio, filename_prefix, format):
        """Save audio to file and return path."""
        filepath = None  # Initialize to prevent UnboundLocalError
        try:
            # Validate inputs first
            if audio is None:
                raise ValueError("Audio input is required")
            if not isinstance(audio, dict) or 'waveform' not in audio or 'sample_rate' not in audio:
                raise ValueError("Invalid audio object - missing waveform or sample_rate")
            
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
            
            # Validate audio data
            if waveform is None:
                raise ValueError("Audio waveform is None")
            if sample_rate is None or sample_rate <= 0:
                raise ValueError(f"Invalid sample rate: {sample_rate}")
            
            # Ensure proper tensor format for torchaudio.save
            if hasattr(waveform, 'cpu'):
                waveform = waveform.cpu()
            
            # Ensure waveform is 2D [channels, samples] for torchaudio compatibility
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # [samples] -> [1, samples]
            elif waveform.ndim > 2:
                waveform = waveform.view(waveform.size(0), -1)
            
            waveform = waveform.float()
            
            # Validate tensor dimensions for torchaudio
            if waveform.ndim != 2:
                raise ValueError(f"Expected 2D tensor for torchaudio.save, got {waveform.ndim}D with shape {waveform.shape}")
            
            # Check for valid audio content
            if waveform.shape[1] == 0:
                raise ValueError("Audio waveform has no samples")
            
            # Save with enhanced metadata
            torchaudio.save(filepath, waveform, sample_rate)
            
            # Verify file was created successfully
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Failed to create audio file: {filepath}")
            
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                raise ValueError(f"Created audio file is empty: {filepath}")
            
            # Print detailed save information
            channels, samples = waveform.shape
            duration = samples / sample_rate
            print(f"✅ Audio saved: {filename}")
            print(f"   📁 Location: ComfyUI/output/audio/")
            print(f"   📊 Format: {format.upper()}, {sample_rate}Hz, {channels}ch, {duration:.2f}s")
            print(f"   💾 File size: {file_size} bytes")
            
            # Enhanced return with metadata
            return (audio, filepath)
            
        except Exception as e:
            error_msg = f"Error saving audio: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Return safe fallback
            fallback_path = filepath if filepath is not None else f"Error: Could not save audio - {str(e)}"
            return (audio, fallback_path)

class FeedbackProcessorNode:
    """
    Feedback Processor - Essential for self-generating Merzbow-style textures
    
    Creates complex feedback loops with filtering, saturation, and modulation.
    Perfect for: Self-generating textures, metallic resonances, runaway chaos
    
    FEEDBACK MODES:
    - SIMPLE    - Basic delay feedback
    - FILTERED  - HP/LP/BP filtering in feedback loop  
    - SATURATED - Nonlinear saturation in loop
    - MODULATED - LFO modulation of delay time
    - COMPLEX   - All effects combined
    - RUNAWAY   - Intentionally unstable feedback (use carefully!)
    """
    
    FEEDBACK_MODES = [
        "simple",      # Basic delay feedback
        "filtered",    # Filtering in feedback loop
        "saturated",   # Nonlinear saturation
        "modulated",   # LFO modulation of delay time
        "complex",     # All effects combined
        "runaway"      # Intentionally unstable (dangerous!)
    ]
    
    FILTER_TYPES = [
        "lowpass",     # Low-pass filtering
        "highpass",    # High-pass filtering
        "bandpass",    # Band-pass filtering
        "notch",       # Notch filtering
        "allpass"      # All-pass (phase only)
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Input audio to process through feedback system"
                }),
                "feedback_mode": (cls.FEEDBACK_MODES, {
                    "default": "filtered",
                    "tooltip": "Feedback processing mode: simple=basic, complex=all effects, runaway=dangerous!"
                }),
                "feedback_amount": ("FLOAT", {
                    "default": 0.4, 
                    "min": 0.0, 
                    "max": 0.95, 
                    "step": 0.01,
                    "tooltip": "Feedback intensity (0.0=none, 0.95=maximum safe, >0.8=chaotic)"
                }),
                "delay_time": ("FLOAT", {
                    "default": 2.0, 
                    "min": 0.1, 
                    "max": 100.0, 
                    "step": 0.1,
                    "tooltip": "Delay time in milliseconds (0.1ms=metallic, 100ms=echo-like)"
                }),
                "filter_type": (cls.FILTER_TYPES, {
                    "default": "lowpass",
                    "tooltip": "Filter type in feedback loop (shapes the resonance character)"
                }),
                "filter_freq": ("FLOAT", {
                    "default": 2000.0, 
                    "min": 20.0, 
                    "max": 20000.0, 
                    "step": 10.0,
                    "tooltip": "Filter frequency in Hz (controls resonance pitch)"
                }),
                "filter_resonance": ("FLOAT", {
                    "default": 0.3, 
                    "min": 0.0, 
                    "max": 0.99, 
                    "step": 0.01,
                    "tooltip": "Filter resonance (0.0=gentle, 0.99=self-oscillating)"
                }),
                "saturation": ("FLOAT", {
                    "default": 0.2, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Nonlinear saturation amount in feedback loop"
                }),
                "modulation_rate": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.01, 
                    "max": 20.0, 
                    "step": 0.01,
                    "tooltip": "LFO rate for delay time modulation (Hz)"
                }),
                "modulation_depth": ("FLOAT", {
                    "default": 0.2, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "LFO depth for delay time modulation"
                }),
                "amplitude": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.01,
                    "tooltip": "Output amplitude (0.0=silence, 1.0=normal, 2.0=loud)"
                }),
            },
            "optional": {
                "wet_dry_mix": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Wet/dry mix (0.0=original only, 1.0=feedback only)"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("feedback_audio",)
    FUNCTION = "process_feedback"
    CATEGORY = "🎵 NoiseGen/Processing"
    DESCRIPTION = "Advanced feedback processor for self-generating Merzbow-style textures and metallic resonances"
    
    def process_feedback(self, audio, feedback_mode, feedback_amount, delay_time, 
                        filter_type, filter_freq, filter_resonance, saturation, 
                        modulation_rate, modulation_depth, amplitude, wet_dry_mix=0.7):
        """Process audio through advanced feedback system."""
        try:
            # Extract audio data
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # Convert to numpy
            if hasattr(waveform, 'cpu'):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = waveform
            
            # Ensure 2D array [channels, samples]
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            
            # Process each channel
            processed_channels = []
            for channel in range(audio_np.shape[0]):
                channel_audio = audio_np[channel]
                processed = self._apply_feedback_processing(
                    channel_audio, sample_rate, feedback_mode, feedback_amount, 
                    delay_time, filter_type, filter_freq, filter_resonance, 
                    saturation, modulation_rate, modulation_depth, wet_dry_mix
                )
                processed_channels.append(processed)
            
            # Stack channels back together
            result = np.stack(processed_channels, axis=0) * amplitude
            
            # Convert back to tensor format
            result_tensor = torch.from_numpy(result).float()
            
            # Create output audio
            output_audio = {
                "waveform": result_tensor,
                "sample_rate": sample_rate
            }
            
            return (output_audio,)
            
        except Exception as e:
            print(f"❌ Error in feedback processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return (audio,)  # Return original on error
    
    def _apply_feedback_processing(self, audio, sample_rate, mode, feedback_amount, 
                                 delay_time_ms, filter_type, filter_freq, filter_resonance,
                                 saturation, mod_rate, mod_depth, wet_dry_mix):
        """Apply sophisticated feedback processing to audio."""
        
        # Convert delay time to samples
        delay_samples = int((delay_time_ms / 1000.0) * sample_rate)
        delay_samples = max(1, min(delay_samples, len(audio) // 2))  # Safety bounds
        
        # Initialize feedback buffer
        feedback_buffer = np.zeros(len(audio) + delay_samples)
        output = np.zeros_like(audio)
        
        # Filter state variables (simple IIR)
        filter_state = {"x1": 0.0, "x2": 0.0, "y1": 0.0, "y2": 0.0}
        
        # LFO for modulation
        lfo_phase = 0.0
        lfo_increment = 2 * np.pi * mod_rate / sample_rate
        
        # Process sample by sample for feedback
        for i in range(len(audio)):
            # Read from delay line
            delay_pos = i + delay_samples
            
            # Modulated delay time (if enabled)
            if mode in ["modulated", "complex"]:
                lfo_value = np.sin(lfo_phase)
                mod_delay = int(delay_samples * (1 + mod_depth * lfo_value * 0.5))
                mod_delay = max(1, min(mod_delay, len(feedback_buffer) - i - 1))
                lfo_phase += lfo_increment
                if lfo_phase > 2 * np.pi:
                    lfo_phase -= 2 * np.pi
            else:
                mod_delay = delay_samples
            
            # Get delayed signal
            if delay_pos < len(feedback_buffer):
                delayed_signal = feedback_buffer[delay_pos - mod_delay]
            else:
                delayed_signal = 0.0
            
            # Apply filtering to delayed signal
            if mode in ["filtered", "complex"]:
                delayed_signal = self._apply_filter(delayed_signal, filter_type, 
                                                  filter_freq, filter_resonance, 
                                                  sample_rate, filter_state)
            
            # Apply saturation
            if mode in ["saturated", "complex", "runaway"]:
                saturation_drive = 1.0 + saturation * 10.0
                delayed_signal = np.tanh(delayed_signal * saturation_drive) / saturation_drive
            
            # Apply feedback
            feedback_signal = delayed_signal * feedback_amount
            
            # For runaway mode, intentionally increase feedback over time
            if mode == "runaway":
                time_factor = min(i / (sample_rate * 2.0), 1.0)  # Ramp over 2 seconds
                feedback_signal *= (1.0 + time_factor * 0.3)  # Gradually increase
            
            # Mix input with feedback
            mixed_signal = audio[i] + feedback_signal
            
            # Store in feedback buffer for next iteration
            if i < len(feedback_buffer):
                feedback_buffer[i] = mixed_signal
            
            # Apply safety limiting to prevent runaway
            if abs(mixed_signal) > 3.0:
                mixed_signal = np.sign(mixed_signal) * 3.0
            
            # Wet/dry mix
            output[i] = audio[i] * (1 - wet_dry_mix) + mixed_signal * wet_dry_mix
        
        return output
    
    def _apply_filter(self, signal, filter_type, cutoff, resonance, sample_rate, state):
        """Apply IIR filter to signal with state preservation."""
        
        # Calculate filter coefficients
        nyquist = sample_rate / 2.0
        normalized_cutoff = max(20.0, min(cutoff, nyquist - 100)) / nyquist
        
        # Simple biquad coefficients (approximate)
        w = 2 * np.pi * normalized_cutoff
        cos_w = np.cos(w)
        sin_w = np.sin(w)
        alpha = sin_w / (2.0 * max(0.1, (1.0 - resonance * 0.9)))
        
        if filter_type == "lowpass":
            b0 = (1 - cos_w) / 2
            b1 = 1 - cos_w
            b2 = (1 - cos_w) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w
            a2 = 1 - alpha
        elif filter_type == "highpass":
            b0 = (1 + cos_w) / 2
            b1 = -(1 + cos_w)
            b2 = (1 + cos_w) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w
            a2 = 1 - alpha
        elif filter_type == "bandpass":
            b0 = alpha
            b1 = 0
            b2 = -alpha
            a0 = 1 + alpha
            a1 = -2 * cos_w
            a2 = 1 - alpha
        else:  # Default to lowpass for notch/allpass
            b0 = (1 - cos_w) / 2
            b1 = 1 - cos_w
            b2 = (1 - cos_w) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w
            a2 = 1 - alpha
        
        # Normalize coefficients
        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0
        
        # Apply biquad filter
        output = (b0 * signal + 
                 b1 * state["x1"] + 
                 b2 * state["x2"] - 
                 a1 * state["y1"] - 
                 a2 * state["y2"])
        
        # Update state
        state["x2"] = state["x1"]
        state["x1"] = signal
        state["y2"] = state["y1"]
        state["y1"] = output
        
        return output

class HarshFilterNode:
    """
    Harsh Filter - Extreme filtering for noise sculpting and self-oscillation
    
    Advanced filter system designed for harsh noise and extreme audio processing.
    Perfect for: Resonant peaks, self-oscillating drones, metallic textures, frequency destruction
    
    FILTER MODES:
    - LOWPASS   - Classic low-pass filtering with resonance
    - HIGHPASS  - High-pass filtering for brightness
    - BANDPASS  - Band-pass for frequency isolation
    - NOTCH     - Notch filtering for frequency removal
    - COMB      - Comb filtering for metallic textures
    - ALLPASS   - Phase-only filtering for phasing effects
    - MORPH     - Morphing between filter types
    - CHAOS     - Chaotic filter modulation
    """
    
    FILTER_TYPES = [
        "lowpass",      # Classic LP filter
        "highpass",     # Classic HP filter  
        "bandpass",     # Band-pass isolation
        "notch",        # Frequency removal
        "comb",         # Metallic textures
        "allpass",      # Phase manipulation
        "morph",        # Filter morphing
        "chaos"         # Chaotic modulation
    ]
    
    DRIVE_MODES = [
        "clean",        # No saturation
        "tube",         # Tube-style saturation
        "transistor",   # Transistor saturation
        "digital",      # Digital clipping
        "chaos"         # Chaotic saturation
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Input audio to process through extreme filtering"
                }),
                "filter_type": (cls.FILTER_TYPES, {
                    "default": "lowpass",
                    "tooltip": "Filter type: lowpass=warm, highpass=bright, comb=metallic, chaos=unpredictable"
                }),
                "cutoff_freq": ("FLOAT", {
                    "default": 1000.0, 
                    "min": 10.0, 
                    "max": 20000.0, 
                    "step": 1.0,
                    "tooltip": "Filter cutoff frequency in Hz (10Hz=deep, 20kHz=bright)"
                }),
                "resonance": ("FLOAT", {
                    "default": 0.3, 
                    "min": 0.0, 
                    "max": 0.999, 
                    "step": 0.001,
                    "tooltip": "Filter resonance (0.0=gentle, 0.999=self-oscillating, >0.9=extreme)"
                }),
                "drive": ("FLOAT", {
                    "default": 0.2, 
                    "min": 0.0, 
                    "max": 5.0, 
                    "step": 0.01,
                    "tooltip": "Filter drive/saturation amount (0.0=clean, 1.0=saturated, 5.0=extreme)"
                }),
                "drive_mode": (cls.DRIVE_MODES, {
                    "default": "tube",
                    "tooltip": "Saturation character: tube=warm, transistor=harsh, digital=glitchy, chaos=unpredictable"
                }),
                "filter_slope": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.5, 
                    "max": 4.0, 
                    "step": 0.1,
                    "tooltip": "Filter slope steepness (0.5=gentle, 4.0=extreme brick-wall)"
                }),
                "morph_amount": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Filter morphing amount (0.0=pure type, 1.0=full morph, requires morph mode)"
                }),
                "modulation_rate": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.001, 
                    "max": 50.0, 
                    "step": 0.001,
                    "tooltip": "Modulation rate for cutoff frequency (Hz) - creates movement"
                }),
                "modulation_depth": ("FLOAT", {
                    "default": 0.1, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Modulation depth (0.0=static, 1.0=extreme movement)"
                }),
                "amplitude": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.01,
                    "tooltip": "Output amplitude (0.0=silence, 1.0=normal, 2.0=loud)"
                }),
            },
            "optional": {
                "wet_dry_mix": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Wet/dry mix (0.0=original only, 1.0=filtered only)"
                }),
                "stereo_spread": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Stereo frequency spread (0.0=mono, 1.0=different L/R frequencies)"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("filtered_audio",)
    FUNCTION = "process_harsh_filter"
    CATEGORY = "🎵 NoiseGen/Processing"
    DESCRIPTION = "Extreme filtering with self-oscillation, saturation, and morphing for harsh noise sculpting"
    
    def process_harsh_filter(self, audio, filter_type, cutoff_freq, resonance, drive, 
                           drive_mode, filter_slope, morph_amount, modulation_rate, 
                           modulation_depth, amplitude, wet_dry_mix=1.0, stereo_spread=0.0):
        """Process audio through extreme harsh filtering."""
        try:
            # Extract audio data
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # Convert to numpy
            if hasattr(waveform, 'cpu'):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = waveform
            
            # Ensure 2D array [channels, samples]
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            
            # Process each channel
            processed_channels = []
            for channel in range(audio_np.shape[0]):
                channel_audio = audio_np[channel]
                
                # Calculate stereo frequency offset
                freq_offset = 0.0
                if audio_np.shape[0] > 1 and stereo_spread > 0.0:
                    # Different frequencies for L/R channels
                    freq_multiplier = 1.0 + (stereo_spread * 0.2 * (1 if channel == 0 else -1))
                    actual_cutoff = cutoff_freq * freq_multiplier
                else:
                    actual_cutoff = cutoff_freq
                
                processed = self._apply_harsh_filtering(
                    channel_audio, sample_rate, filter_type, actual_cutoff, 
                    resonance, drive, drive_mode, filter_slope, morph_amount,
                    modulation_rate, modulation_depth, wet_dry_mix
                )
                processed_channels.append(processed)
            
            # Stack channels back together
            result = np.stack(processed_channels, axis=0) * amplitude
            
            # Convert back to tensor format
            result_tensor = torch.from_numpy(result).float()
            
            # Create output audio
            output_audio = {
                "waveform": result_tensor,
                "sample_rate": sample_rate
            }
            
            return (output_audio,)
            
        except Exception as e:
            print(f"❌ Error in harsh filtering: {str(e)}")
            import traceback
            traceback.print_exc()
            return (audio,)  # Return original on error
    
    def _apply_harsh_filtering(self, audio, sample_rate, filter_type, cutoff_freq, 
                             resonance, drive, drive_mode, filter_slope, morph_amount,
                             mod_rate, mod_depth, wet_dry_mix):
        """Apply sophisticated harsh filtering to audio."""
        
        # Initialize filter states (multiple stages for steeper slopes)
        num_stages = max(1, int(filter_slope * 2))
        filter_states = []
        for _ in range(num_stages):
            filter_states.append({"x1": 0.0, "x2": 0.0, "y1": 0.0, "y2": 0.0})
        
        # LFO for modulation
        lfo_phase = 0.0
        lfo_increment = 2 * np.pi * mod_rate / sample_rate
        
        # Chaos state for chaotic modes
        chaos_state = 0.5
        
        output = np.zeros_like(audio)
        
        # Process sample by sample for real-time modulation
        for i in range(len(audio)):
            # Calculate modulated cutoff frequency
            if mod_depth > 0.0:
                lfo_value = np.sin(lfo_phase)
                if filter_type == "chaos":
                    # Chaotic modulation
                    chaos_state = (chaos_state * 3.8 * (1 - chaos_state)) % 1.0
                    mod_value = (chaos_state - 0.5) * 2  # -1 to 1
                else:
                    mod_value = lfo_value
                
                modulated_cutoff = cutoff_freq * (1 + mod_depth * mod_value * 0.8)
                modulated_cutoff = max(10.0, min(modulated_cutoff, sample_rate * 0.49))
                lfo_phase += lfo_increment
                if lfo_phase > 2 * np.pi:
                    lfo_phase -= 2 * np.pi
            else:
                modulated_cutoff = cutoff_freq
            
            # Apply drive/saturation before filtering (if enabled)
            driven_sample = audio[i]
            if drive > 0.0:
                driven_sample = self._apply_drive(driven_sample, drive, drive_mode)
            
            # Apply filtering (multiple stages for steeper slopes)
            filtered_sample = driven_sample
            for stage in range(num_stages):
                filtered_sample = self._apply_filter_stage(
                    filtered_sample, filter_type, modulated_cutoff, 
                    resonance, sample_rate, filter_states[stage], morph_amount
                )
            
            # Apply wet/dry mix
            output[i] = audio[i] * (1 - wet_dry_mix) + filtered_sample * wet_dry_mix
            
            # Safety limiting
            if abs(output[i]) > 3.0:
                output[i] = np.sign(output[i]) * 3.0
        
        return output
    
    def _apply_filter_stage(self, signal, filter_type, cutoff, resonance, sample_rate, state, morph_amount):
        """Apply a single filter stage with specified type."""
        
        # Calculate filter coefficients
        nyquist = sample_rate / 2.0
        normalized_cutoff = max(10.0, min(cutoff, nyquist - 100)) / nyquist
        
        # Biquad coefficients calculation
        w = 2 * np.pi * normalized_cutoff
        cos_w = np.cos(w)
        sin_w = np.sin(w)
        
        # Resonance with extreme values (can self-oscillate)
        Q = max(0.1, 1.0 / (1.0 - resonance * 0.999))
        alpha = sin_w / (2.0 * Q)
        
        # Filter coefficient selection based on type
        if filter_type == "lowpass":
            b0 = (1 - cos_w) / 2
            b1 = 1 - cos_w
            b2 = (1 - cos_w) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w
            a2 = 1 - alpha
            
        elif filter_type == "highpass":
            b0 = (1 + cos_w) / 2
            b1 = -(1 + cos_w)
            b2 = (1 + cos_w) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w
            a2 = 1 - alpha
            
        elif filter_type == "bandpass":
            b0 = alpha
            b1 = 0
            b2 = -alpha
            a0 = 1 + alpha
            a1 = -2 * cos_w
            a2 = 1 - alpha
            
        elif filter_type == "notch":
            b0 = 1
            b1 = -2 * cos_w
            b2 = 1
            a0 = 1 + alpha
            a1 = -2 * cos_w
            a2 = 1 - alpha
            
        elif filter_type == "comb":
            # Comb filter implementation
            delay_samples = max(1, int(sample_rate / cutoff))
            if 'comb_buffer' not in state:
                state['comb_buffer'] = np.zeros(delay_samples)
            elif len(state['comb_buffer']) != delay_samples:
                state['comb_buffer'] = np.zeros(delay_samples)
            
            # Comb filter processing
            delayed = state['comb_buffer'][0]
            output = signal + resonance * delayed
            
            # Shift buffer
            state['comb_buffer'][:-1] = state['comb_buffer'][1:]
            state['comb_buffer'][-1] = output
            
            return output
            
        elif filter_type == "allpass":
            # All-pass filter
            b0 = -alpha
            b1 = 1
            b2 = alpha
            a0 = 1 + alpha
            a1 = 1
            a2 = alpha
            
        elif filter_type == "morph":
            # Morph between lowpass and highpass
            # Lowpass coefficients
            lp_b0 = (1 - cos_w) / 2
            lp_b1 = 1 - cos_w
            lp_b2 = (1 - cos_w) / 2
            
            # Highpass coefficients  
            hp_b0 = (1 + cos_w) / 2
            hp_b1 = -(1 + cos_w)
            hp_b2 = (1 + cos_w) / 2
            
            # Morph between them
            b0 = lp_b0 * (1 - morph_amount) + hp_b0 * morph_amount
            b1 = lp_b1 * (1 - morph_amount) + hp_b1 * morph_amount
            b2 = lp_b2 * (1 - morph_amount) + hp_b2 * morph_amount
            a0 = 1 + alpha
            a1 = -2 * cos_w
            a2 = 1 - alpha
            
        elif filter_type == "chaos":
            # Chaotic filter - randomize coefficients slightly
            chaos_factor = 0.1 * morph_amount
            b0 = (1 - cos_w) / 2 * (1 + chaos_factor * (np.random.random() - 0.5))
            b1 = (1 - cos_w) * (1 + chaos_factor * (np.random.random() - 0.5))
            b2 = (1 - cos_w) / 2 * (1 + chaos_factor * (np.random.random() - 0.5))
            a0 = 1 + alpha
            a1 = -2 * cos_w * (1 + chaos_factor * (np.random.random() - 0.5))
            a2 = 1 - alpha
        
        else:  # Default to lowpass
            b0 = (1 - cos_w) / 2
            b1 = 1 - cos_w
            b2 = (1 - cos_w) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w
            a2 = 1 - alpha
        
        # Normalize coefficients
        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0
        
        # Apply biquad filter
        output = (b0 * signal + 
                 b1 * state["x1"] + 
                 b2 * state["x2"] - 
                 a1 * state["y1"] - 
                 a2 * state["y2"])
        
        # Update state
        state["x2"] = state["x1"]
        state["x1"] = signal
        state["y2"] = state["y1"]
        state["y1"] = output
        
        return output
    
    def _apply_drive(self, signal, drive_amount, drive_mode):
        """Apply saturation/drive to the signal."""
        
        if drive_amount <= 0.0:
            return signal
        
        # Scale drive amount
        drive_scale = 1.0 + drive_amount * 4.0
        driven_signal = signal * drive_scale
        
        if drive_mode == "clean":
            return driven_signal
        
        elif drive_mode == "tube":
            # Tube-style soft saturation
            if abs(driven_signal) > 1.0:
                driven_signal = np.tanh(driven_signal * 0.7) * 1.4
            return driven_signal
        
        elif drive_mode == "transistor":
            # Transistor-style asymmetric saturation
            if driven_signal > 0:
                driven_signal = min(driven_signal, 1.0 + 0.3 * np.tanh((driven_signal - 1.0) * 3))
            else:
                driven_signal = max(driven_signal, -1.0 - 0.2 * np.tanh((-driven_signal - 1.0) * 2))
            return driven_signal
        
        elif drive_mode == "digital":
            # Digital clipping
            return np.clip(driven_signal, -1.0, 1.0)
        
        elif drive_mode == "chaos":
            # Chaotic saturation
            chaos_factor = abs(driven_signal) * 0.1
            random_factor = (np.random.random() - 0.5) * chaos_factor
            return np.tanh(driven_signal * (1 + random_factor))
        
        return driven_signal

class MultiDistortionNode:
    """
    Multi-Distortion - Comprehensive distortion palette for extreme audio processing
    
    Professional multi-stage distortion system with 12 distortion types and advanced controls.
    Perfect for: Harsh noise textures, power electronics, extreme sound design, brutal saturation
    
    DISTORTION TYPES:
    - TUBE         - Warm analog tube saturation
    - TRANSISTOR   - Asymmetric transistor clipping  
    - DIODE        - Classic diode clipping
    - DIGITAL      - Hard digital clipping
    - BITCRUSH     - Bit reduction and sample rate reduction
    - WAVESHAPER   - Custom waveshaping curves
    - FOLDBACK     - Wave folding distortion
    - RING_MOD     - Ring modulation distortion
    - CHAOS        - Chaotic nonlinear distortion
    - FUZZ         - Classic fuzz box saturation
    - OVERDRIVE    - Smooth overdrive saturation
    - DESTRUCTION  - Extreme multi-stage destruction
    """
    
    DISTORTION_TYPES = [
        "tube",         # Warm analog saturation
        "transistor",   # Asymmetric clipping
        "diode",        # Classic diode clipping
        "digital",      # Hard clipping
        "bitcrush",     # Digital artifacts
        "waveshaper",   # Custom curves
        "foldback",     # Wave folding
        "ring_mod",     # Ring modulation
        "chaos",        # Chaotic distortion
        "fuzz",         # Fuzz box
        "overdrive",    # Smooth overdrive
        "destruction"   # Multi-stage chaos
    ]
    
    FILTER_TYPES = [
        "none",         # No filtering
        "lowpass",      # Pre-filtering
        "highpass",     # Pre-filtering
        "bandpass",     # Pre-filtering
        "notch"         # Pre-filtering
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Input audio to process through multi-stage distortion"
                }),
                "distortion_type": (cls.DISTORTION_TYPES, {
                    "default": "tube",
                    "tooltip": "Primary distortion type: tube=warm, digital=harsh, chaos=unpredictable, destruction=extreme"
                }),
                "drive": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.01,
                    "tooltip": "Distortion drive amount (0.0=clean, 1.0=moderate, 10.0=extreme destruction)"
                }),
                "output_gain": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.01,
                    "tooltip": "Output gain compensation (0.0=silence, 1.0=unity, 2.0=boost)"
                }),
                "wet_dry_mix": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Wet/dry mix (0.0=clean only, 1.0=distorted only)"
                }),
                "stages": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 4, 
                    "step": 1,
                    "tooltip": "Number of distortion stages (1=single, 4=extreme multi-stage)"
                }),
                "stage_feedback": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 0.8, 
                    "step": 0.01,
                    "tooltip": "Feedback between stages (0.0=none, 0.8=extreme interdependence)"
                }),
                "asymmetry": ("FLOAT", {
                    "default": 0.0, 
                    "min": -1.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Positive/negative asymmetry (-1.0=negative bias, 1.0=positive bias)"
                }),
                "harmonic_emphasis": ("FLOAT", {
                    "default": 0.3, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Harmonic generation emphasis (0.0=minimal, 1.0=maximum harmonics)"
                }),
            },
            "optional": {
                "pre_filter_type": (cls.FILTER_TYPES, {
                    "default": "none",
                    "tooltip": "Pre-distortion filtering to shape input spectrum"
                }),
                "pre_filter_freq": ("FLOAT", {
                    "default": 1000.0, 
                    "min": 20.0, 
                    "max": 20000.0, 
                    "step": 1.0,
                    "tooltip": "Pre-filter cutoff frequency in Hz"
                }),
                "bitcrush_bits": ("INT", {
                    "default": 8, 
                    "min": 1, 
                    "max": 16, 
                    "step": 1,
                    "tooltip": "Bit depth for bitcrush mode (1=extreme, 16=subtle)"
                }),
                "bitcrush_sample_rate": ("FLOAT", {
                    "default": 11025.0, 
                    "min": 100.0, 
                    "max": 44100.0, 
                    "step": 100.0,
                    "tooltip": "Sample rate for bitcrush mode (100Hz=extreme, 44100Hz=clean)"
                }),
                "ring_mod_freq": ("FLOAT", {
                    "default": 440.0, 
                    "min": 10.0, 
                    "max": 2000.0, 
                    "step": 0.1,
                    "tooltip": "Ring modulation frequency in Hz"
                }),
                "chaos_amount": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Chaos injection amount for unpredictable variations"
                }),
                "stereo_spread": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Stereo parameter spread (0.0=mono, 1.0=different L/R processing)"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("distorted_audio",)
    FUNCTION = "process_multi_distortion"
    CATEGORY = "🎵 NoiseGen/Processing"
    DESCRIPTION = "Comprehensive multi-stage distortion with 12 types and advanced controls for extreme audio processing"
    
    def process_multi_distortion(self, audio, distortion_type, drive, output_gain, wet_dry_mix, 
                                stages, stage_feedback, asymmetry, harmonic_emphasis,
                                pre_filter_type="none", pre_filter_freq=1000.0, bitcrush_bits=8,
                                bitcrush_sample_rate=11025.0, ring_mod_freq=440.0, chaos_amount=0.5,
                                stereo_spread=0.0):
        """Process audio through comprehensive multi-distortion."""
        try:
            # Extract audio data
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # Convert to numpy
            if hasattr(waveform, 'cpu'):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = waveform
            
            # Ensure 2D array [channels, samples]
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            
            # Store original for wet/dry mixing
            original_audio = audio_np.copy()
            
            # Process each channel
            processed_channels = []
            for channel in range(audio_np.shape[0]):
                channel_audio = audio_np[channel]
                
                # Calculate stereo parameter variations
                if audio_np.shape[0] > 1 and stereo_spread > 0.0:
                    # Different parameters for L/R channels
                    channel_multiplier = 1.0 + (stereo_spread * 0.3 * (1 if channel == 0 else -1))
                    actual_drive = drive * channel_multiplier
                    actual_chaos = chaos_amount * (1.0 + stereo_spread * 0.2 * (1 if channel == 0 else -1))
                    actual_ring_freq = ring_mod_freq * channel_multiplier
                else:
                    actual_drive = drive
                    actual_chaos = chaos_amount
                    actual_ring_freq = ring_mod_freq
                
                processed = self._apply_multi_distortion(
                    channel_audio, sample_rate, distortion_type, actual_drive, 
                    stages, stage_feedback, asymmetry, harmonic_emphasis,
                    pre_filter_type, pre_filter_freq, bitcrush_bits, 
                    bitcrush_sample_rate, actual_ring_freq, actual_chaos
                )
                processed_channels.append(processed)
            
            # Stack channels back together
            processed_audio = np.stack(processed_channels, axis=0)
            
            # Apply wet/dry mixing
            result = original_audio * (1 - wet_dry_mix) + processed_audio * wet_dry_mix
            
            # Apply output gain
            result = result * output_gain
            
            # Convert back to tensor format
            result_tensor = torch.from_numpy(result).float()
            
            # Create output audio
            output_audio = {
                "waveform": result_tensor,
                "sample_rate": sample_rate
            }
            
            return (output_audio,)
            
        except Exception as e:
            print(f"❌ Error in multi-distortion: {str(e)}")
            import traceback
            traceback.print_exc()
            return (audio,)  # Return original on error
    
    def _apply_multi_distortion(self, audio, sample_rate, distortion_type, drive, 
                               stages, stage_feedback, asymmetry, harmonic_emphasis,
                               pre_filter_type, pre_filter_freq, bitcrush_bits,
                               bitcrush_sample_rate, ring_mod_freq, chaos_amount):
        """Apply sophisticated multi-stage distortion to audio."""
        
        # Apply pre-filtering if requested
        if pre_filter_type != "none":
            audio = self._apply_pre_filter(audio, sample_rate, pre_filter_type, pre_filter_freq)
        
        # Initialize stage buffers for feedback
        stage_buffers = [np.zeros(len(audio)) for _ in range(stages)]
        
        # Ring modulation oscillator
        ring_osc_phase = 0.0
        ring_osc_increment = 2 * np.pi * ring_mod_freq / sample_rate
        
        # Chaos state for chaotic distortion
        chaos_state = 0.5
        
        # Process through multiple stages
        current_signal = audio.copy()
        
        for stage in range(stages):
            # Calculate stage-specific drive
            stage_drive = drive * (1.0 + stage * 0.2)  # Increasing drive per stage
            
            # Apply distortion to current signal
            if distortion_type == "tube":
                distorted = self._apply_tube_distortion(current_signal, stage_drive, asymmetry)
            elif distortion_type == "transistor":
                distorted = self._apply_transistor_distortion(current_signal, stage_drive, asymmetry)
            elif distortion_type == "diode":
                distorted = self._apply_diode_distortion(current_signal, stage_drive, asymmetry)
            elif distortion_type == "digital":
                distorted = self._apply_digital_distortion(current_signal, stage_drive, asymmetry)
            elif distortion_type == "bitcrush":
                distorted = self._apply_bitcrush_distortion(current_signal, stage_drive, 
                                                          bitcrush_bits, bitcrush_sample_rate, sample_rate)
            elif distortion_type == "waveshaper":
                distorted = self._apply_waveshaper_distortion(current_signal, stage_drive, harmonic_emphasis)
            elif distortion_type == "foldback":
                distorted = self._apply_foldback_distortion(current_signal, stage_drive, asymmetry)
            elif distortion_type == "ring_mod":
                distorted, ring_osc_phase = self._apply_ring_mod_distortion(
                    current_signal, stage_drive, ring_osc_phase, ring_osc_increment, asymmetry)
            elif distortion_type == "chaos":
                distorted, chaos_state = self._apply_chaos_distortion(
                    current_signal, stage_drive, chaos_state, chaos_amount, asymmetry)
            elif distortion_type == "fuzz":
                distorted = self._apply_fuzz_distortion(current_signal, stage_drive, asymmetry)
            elif distortion_type == "overdrive":
                distorted = self._apply_overdrive_distortion(current_signal, stage_drive, asymmetry)
            elif distortion_type == "destruction":
                distorted = self._apply_destruction_distortion(current_signal, stage_drive, 
                                                             harmonic_emphasis, chaos_amount)
            else:
                distorted = current_signal  # Default passthrough
            
            # Store stage output
            stage_buffers[stage] = distorted.copy()
            
            # Apply stage feedback if not the last stage
            if stage < stages - 1 and stage_feedback > 0.0:
                # Mix feedback from previous stages
                feedback_signal = np.zeros_like(current_signal)
                for prev_stage in range(stage + 1):
                    feedback_weight = stage_feedback * (0.8 ** (stage - prev_stage))
                    feedback_signal += stage_buffers[prev_stage] * feedback_weight
                
                # Mix feedback with current signal for next stage
                current_signal = distorted * 0.7 + feedback_signal * 0.3
                
                # Safety limiting
                current_signal = np.clip(current_signal, -2.0, 2.0)
            else:
                current_signal = distorted
        
        # Apply harmonic emphasis post-processing
        if harmonic_emphasis > 0.0:
            current_signal = self._apply_harmonic_emphasis(current_signal, harmonic_emphasis)
        
        # Final safety limiting
        current_signal = np.clip(current_signal, -3.0, 3.0)
        
        return current_signal
    
    def _apply_pre_filter(self, audio, sample_rate, filter_type, cutoff_freq):
        """Apply pre-distortion filtering."""
        nyquist = sample_rate / 2.0
        normalized_cutoff = min(cutoff_freq, nyquist - 100) / nyquist
        
        # Simple first-order filters for pre-processing
        alpha = 1.0 - np.exp(-2 * np.pi * normalized_cutoff)
        
        filtered = np.zeros_like(audio)
        state = 0.0
        
        for i in range(len(audio)):
            if filter_type == "lowpass":
                state += alpha * (audio[i] - state)
                filtered[i] = state
            elif filter_type == "highpass":
                state += alpha * (audio[i] - state)
                filtered[i] = audio[i] - state
            elif filter_type == "bandpass":
                # Simple bandpass approximation
                state += alpha * (audio[i] - state)
                filtered[i] = audio[i] - state if i % 2 == 0 else state
            elif filter_type == "notch":
                # Simple notch approximation  
                state += alpha * (audio[i] - state)
                filtered[i] = audio[i] - state * 0.5
            else:
                filtered[i] = audio[i]
        
        return filtered
    
    def _apply_tube_distortion(self, signal, drive, asymmetry):
        """Apply warm tube-style distortion."""
        drive_signal = signal * (1.0 + drive * 3.0)
        
        # Asymmetric tube characteristics
        positive_drive = 1.0 + asymmetry * 0.3
        negative_drive = 1.0 - asymmetry * 0.2
        
        distorted = np.zeros_like(signal)
        for i in range(len(signal)):
            if drive_signal[i] >= 0:
                distorted[i] = np.tanh(drive_signal[i] * positive_drive) * 0.7
            else:
                distorted[i] = np.tanh(drive_signal[i] * negative_drive) * 0.8
        
        return distorted
    
    def _apply_transistor_distortion(self, signal, drive, asymmetry):
        """Apply transistor-style asymmetric clipping."""
        drive_signal = signal * (1.0 + drive * 4.0)
        
        # Asymmetric clipping thresholds
        pos_threshold = 1.0 + asymmetry * 0.5
        neg_threshold = -1.0 - asymmetry * 0.3
        
        distorted = np.clip(drive_signal, neg_threshold, pos_threshold)
        
        # Soft saturation at extremes
        distorted = np.tanh(distorted * 0.8)
        
        return distorted
    
    def _apply_diode_distortion(self, signal, drive, asymmetry):
        """Apply classic diode clipping distortion."""
        drive_signal = signal * (1.0 + drive * 2.0)
        
        # Diode characteristic curve
        distorted = np.zeros_like(signal)
        for i in range(len(signal)):
            x = drive_signal[i]
            if x > 0:
                # Forward diode bias
                threshold = 0.7 + asymmetry * 0.2
                if x > threshold:
                    distorted[i] = threshold + np.tanh((x - threshold) * 5) * 0.1
                else:
                    distorted[i] = x
            else:
                # Reverse diode bias
                threshold = -0.7 - asymmetry * 0.1
                if x < threshold:
                    distorted[i] = threshold + np.tanh((x - threshold) * 3) * 0.05
                else:
                    distorted[i] = x
        
        return distorted
    
    def _apply_digital_distortion(self, signal, drive, asymmetry):
        """Apply hard digital clipping."""
        drive_signal = signal * (1.0 + drive * 5.0)
        
        # Asymmetric clipping
        pos_clip = 1.0 + asymmetry * 0.3
        neg_clip = -1.0 - asymmetry * 0.3
        
        return np.clip(drive_signal, neg_clip, pos_clip)
    
    def _apply_bitcrush_distortion(self, signal, drive, bits, target_sample_rate, actual_sample_rate):
        """Apply bit crushing and sample rate reduction."""
        drive_signal = signal * (1.0 + drive * 2.0)
        
        # Bit reduction
        max_value = 2 ** (bits - 1) - 1
        quantized = np.round(drive_signal * max_value) / max_value
        
        # Sample rate reduction
        if target_sample_rate < actual_sample_rate:
            downsample_factor = int(actual_sample_rate / target_sample_rate)
            if downsample_factor > 1:
                # Simple downsampling by taking every Nth sample
                downsampled = np.zeros_like(quantized)
                for i in range(len(quantized)):
                    sample_index = (i // downsample_factor) * downsample_factor
                    if sample_index < len(quantized):
                        downsampled[i] = quantized[sample_index]
                quantized = downsampled
        
        return quantized
    
    def _apply_waveshaper_distortion(self, signal, drive, harmonic_emphasis):
        """Apply custom waveshaping curves."""
        drive_signal = signal * (1.0 + drive * 3.0)
        
        # Custom waveshaping function with harmonic emphasis
        distorted = np.zeros_like(signal)
        for i in range(len(signal)):
            x = drive_signal[i]
            # Chebyshev polynomial-inspired waveshaping
            shaped = x - (harmonic_emphasis * 0.3) * (x**3) + (harmonic_emphasis * 0.1) * (x**5)
            distorted[i] = np.tanh(shaped)
        
        return distorted
    
    def _apply_foldback_distortion(self, signal, drive, asymmetry):
        """Apply wave folding distortion."""
        drive_signal = signal * (1.0 + drive * 4.0)
        
        # Asymmetric folding thresholds
        pos_threshold = 1.0 + asymmetry * 0.5
        neg_threshold = -1.0 - asymmetry * 0.5
        
        folded = np.zeros_like(signal)
        for i in range(len(signal)):
            x = drive_signal[i]
            
            # Positive folding
            while x > pos_threshold:
                x = 2 * pos_threshold - x
            
            # Negative folding  
            while x < neg_threshold:
                x = 2 * neg_threshold - x
                
            folded[i] = x
        
        return folded
    
    def _apply_ring_mod_distortion(self, signal, drive, phase, phase_increment, asymmetry):
        """Apply ring modulation distortion."""
        drive_signal = signal * (1.0 + drive * 2.0)
        
        modulated = np.zeros_like(signal)
        for i in range(len(signal)):
            # Ring modulation
            modulator = np.sin(phase) * (1.0 + asymmetry * 0.3)
            modulated[i] = drive_signal[i] * modulator
            
            phase += phase_increment
            if phase > 2 * np.pi:
                phase -= 2 * np.pi
        
        return modulated, phase
    
    def _apply_chaos_distortion(self, signal, drive, chaos_state, chaos_amount, asymmetry):
        """Apply chaotic nonlinear distortion."""
        drive_signal = signal * (1.0 + drive * 3.0)
        
        distorted = np.zeros_like(signal)
        for i in range(len(signal)):
            # Chaotic map evolution
            chaos_state = (chaos_state * 3.8 * (1 - chaos_state)) % 1.0
            
            # Chaotic modulation of distortion
            chaos_mod = (chaos_state - 0.5) * 2 * chaos_amount  # -1 to 1
            
            # Apply chaotic distortion
            x = drive_signal[i]
            chaos_factor = 1.0 + chaos_mod * 0.5 + asymmetry * 0.3
            distorted[i] = np.tanh(x * chaos_factor)
        
        return distorted, chaos_state
    
    def _apply_fuzz_distortion(self, signal, drive, asymmetry):
        """Apply classic fuzz box distortion."""
        drive_signal = signal * (1.0 + drive * 6.0)
        
        # Fuzz characteristics with asymmetry
        distorted = np.zeros_like(signal)
        for i in range(len(signal)):
            x = drive_signal[i]
            
            # Asymmetric fuzz response
            if x >= 0:
                fuzz_factor = 1.0 + asymmetry * 0.4
                distorted[i] = np.sign(x) * min(abs(x * fuzz_factor), 1.0)
            else:
                fuzz_factor = 1.0 - asymmetry * 0.2
                distorted[i] = np.sign(x) * min(abs(x * fuzz_factor), 0.9)
        
        return distorted
    
    def _apply_overdrive_distortion(self, signal, drive, asymmetry):
        """Apply smooth overdrive distortion."""
        drive_signal = signal * (1.0 + drive * 2.5)
        
        # Smooth overdrive curve with asymmetry
        distorted = np.zeros_like(signal)
        for i in range(len(signal)):
            x = drive_signal[i]
            
            # Asymmetric smooth clipping
            if x >= 0:
                threshold = 0.5 + asymmetry * 0.2
                if abs(x) > threshold:
                    sign = np.sign(x)
                    distorted[i] = sign * (threshold + (1 - threshold) * np.tanh((abs(x) - threshold) * 3))
                else:
                    distorted[i] = x
            else:
                threshold = 0.5 - asymmetry * 0.1
                if abs(x) > threshold:
                    sign = np.sign(x)
                    distorted[i] = sign * (threshold + (1 - threshold) * np.tanh((abs(x) - threshold) * 2))
                else:
                    distorted[i] = x
        
        return distorted
    
    def _apply_destruction_distortion(self, signal, drive, harmonic_emphasis, chaos_amount):
        """Apply extreme multi-stage destruction."""
        current = signal * (1.0 + drive * 8.0)
        
        # Stage 1: Hard clipping
        current = np.clip(current, -1.0, 1.0)
        
        # Stage 2: Bit crushing
        current = np.round(current * 7) / 7
        
        # Stage 3: Waveshaping
        current = current - 0.3 * (current**3) * harmonic_emphasis
        
        # Stage 4: Chaotic modulation
        chaos_state = 0.5
        for i in range(len(current)):
            chaos_state = (chaos_state * 3.9 * (1 - chaos_state)) % 1.0
            chaos_mod = (chaos_state - 0.5) * 2 * chaos_amount
            current[i] = np.tanh(current[i] * (1 + chaos_mod * 0.5))
        
        # Stage 5: Final limiting
        current = np.clip(current, -1.5, 1.5)
        
        return current
    
    def _apply_harmonic_emphasis(self, signal, emphasis):
        """Apply harmonic emphasis post-processing."""
        if emphasis <= 0.0:
            return signal
        
        # Generate harmonics through nonlinear processing
        emphasized = signal.copy()
        
        # Add harmonic content
        for i in range(len(signal)):
            x = signal[i]
            # Add controlled harmonic distortion
            harmonics = emphasis * 0.2 * (x**3 - 0.1 * x**5)
            emphasized[i] = x + harmonics
        
        # Prevent excessive amplitude
        emphasized = np.tanh(emphasized)
        
        return emphasized

# Node mappings for ComfyUI - OPTIMIZED UX
NODE_CLASS_MAPPINGS = {
    # MAIN NODES - Clean and focused
    "NoiseGenerator": NoiseGeneratorNode,        # Universal - handles all basic types
    "PerlinNoise": PerlinNoiseNode,             # Unique parameters (frequency, octaves)
    "BandLimitedNoise": BandLimitedNoiseNode,   # Unique parameters (freq filtering)
    "ChaosNoiseMix": ChaosNoiseMixNode,         # Advanced mixing
    "AudioMixer": AudioMixerNode,               # Professional mixing
    "FeedbackProcessor": FeedbackProcessorNode, # Feedback systems - NEW!
    "AudioSave": AudioSaveNode,                 # Utility
    "HarshFilter": HarshFilterNode,              # Advanced filtering
    "MultiDistortion": MultiDistortionNode,      # Comprehensive distortion
    
    # LEGACY NODES - Hidden from main menu, kept for compatibility
    # Users can still access these if needed, but they're not promoted
    "_WhiteNoise": WhiteNoiseNode,
    "_PinkNoise": PinkNoiseNode,
    "_BrownNoise": BrownNoiseNode,
    "_BlueNoise": BlueNoiseNode,
    "_VioletNoise": VioletNoiseNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # MAIN INTERFACE - Clean and discoverable
    "NoiseGenerator": "🎵 Noise Generator",
    "PerlinNoise": "🌊 Perlin Noise", 
    "BandLimitedNoise": "📡 Band Limited Noise",
    "ChaosNoiseMix": "💥 Chaos Noise Mix",
    "AudioMixer": "🎛️ Audio Mixer",
    "FeedbackProcessor": "🔄 Feedback Processor",
    "AudioSave": "💾 Audio Save",
    "HarshFilter": "🎛️ Harsh Filter",
    "MultiDistortion": "🎛️ Multi-Distortion",
    
    # LEGACY - Hidden with underscore prefix
    "_WhiteNoise": "⚪ White Noise (Legacy)",
    "_PinkNoise": "🌸 Pink Noise (Legacy)", 
    "_BrownNoise": "🟤 Brown Noise (Legacy)",
    "_BlueNoise": "🔵 Blue Noise (Legacy)",
    "_VioletNoise": "🟣 Violet Noise (Legacy)",
} 