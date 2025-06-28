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

# Node mappings for ComfyUI - OPTIMIZED UX
NODE_CLASS_MAPPINGS = {
    # MAIN NODES - Clean and focused
    "NoiseGenerator": NoiseGeneratorNode,        # Universal - handles all basic types
    "PerlinNoise": PerlinNoiseNode,             # Unique parameters (frequency, octaves)
    "BandLimitedNoise": BandLimitedNoiseNode,   # Unique parameters (freq filtering)
    "ChaosNoiseMix": ChaosNoiseMixNode,         # Advanced mixing
    "AudioMixer": AudioMixerNode,               # Professional mixing
    "AudioSave": AudioSaveNode,                 # Utility
    
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
    "BandLimitedNoise": "📻 Band-Limited Noise", 
    "ChaosNoiseMix": "🔥 Chaos Noise Mix",
    "AudioMixer": "🎛️ Audio Mixer",
    "AudioSave": "💾 Save Audio",
    
    # LEGACY - Hidden with underscore prefix
    "_WhiteNoise": "⚪ White Noise (Legacy)",
    "_PinkNoise": "🌸 Pink Noise (Legacy)", 
    "_BrownNoise": "🟤 Brown Noise (Legacy)",
    "_BlueNoise": "🔵 Blue Noise (Legacy)",
    "_VioletNoise": "🟣 Violet Noise (Legacy)",
} 