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
    """
    Universal Noise Generator - All noise types in one node
    
    NOISE TYPES EXPLAINED:
    - WHITE   - Pure static chaos (flat frequency spectrum) - Audio testing, masking
    - PINK    - Natural balance (1/f slope) - Relaxation, natural ambience  
    - BROWN   - Deep rumble (1/f² slope) - Deep relaxation, bass-heavy textures
    - BLUE    - Bright/harsh (+3dB/octave) - High-freq testing, cutting textures
    - VIOLET  - Ultra-bright (+6dB/octave) - Extreme high-freq, digital artifacts
    - PERLIN  - Organic textures (natural variations) - Wind, water, organic sounds
    - BANDLIMITED - Frequency filtered (targeted ranges) - Precise freq testing
    
    Perfect for: Experimental music, audio testing, relaxation, sound design
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
                    "tooltip": "WHITE=static | PINK=natural | BROWN=deep | BLUE=bright | VIOLET=harsh | PERLIN=organic | BANDLIMITED=filtered"
                }),
                "duration": ("FLOAT", {
                    "default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1,
                    "tooltip": "Length of generated audio in seconds"
                }),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {
                    "default": 44100,
                    "tooltip": "Audio quality: 44100=CD quality, 48000=pro audio, 96000=hi-res"
                }),
                "amplitude": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Volume/loudness: 0.0=silent, 1.0=full scale, >1.0=boosted"
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2147483647,
                    "tooltip": "Random seed for reproducible results (same seed = same noise)"
                }),
                "channels": ([1, 2], {
                    "default": 1,
                    "tooltip": "1=mono, 2=stereo"
                }),
                "stereo_mode": (["independent", "correlated", "decorrelated"], {
                    "default": "independent",
                    "tooltip": "INDEPENDENT=different L/R | CORRELATED=same L/R | DECORRELATED=opposite L/R"
                }),
                "stereo_width": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Stereo spread: 0.0=mono, 1.0=normal stereo, 2.0=wide stereo"
                }),
            },
            "optional": {
                "frequency": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1,
                    "tooltip": "Base frequency for Perlin noise (Hz) - controls texture speed"
                }),
                "low_freq": ("FLOAT", {
                    "default": 100.0, "min": 1.0, "max": 20000.0, "step": 1.0,
                    "tooltip": "Low cutoff for band-limited noise (Hz) - removes frequencies below this"
                }),
                "high_freq": ("FLOAT", {
                    "default": 8000.0, "min": 1.0, "max": 20000.0, "step": 1.0,
                    "tooltip": "High cutoff for band-limited noise (Hz) - removes frequencies above this"
                }),
                "octaves": ("INT", {
                    "default": 4, "min": 1, "max": 8,
                    "tooltip": "Perlin noise complexity: 1=simple, 8=very detailed/layered"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_noise"
    CATEGORY = "NoiseGen"
    
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
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "channels": ([1, 2], {"default": 1}),
                "stereo_mode": (["independent", "correlated", "decorrelated"], {"default": "independent"}),
                "stereo_width": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "NoiseGen/Legacy"
    
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
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "NoiseGen/Basic"
    
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
    """Dedicated brown/red noise generator."""
    
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
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "NoiseGen/Basic"
    
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
                "noise_a": ("AUDIO",),
                "noise_b": ("AUDIO",),
                "mix_mode": (cls.MIX_MODES, {"default": "chaos"}),
                "mix_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chaos_amount": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "distortion": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bit_crush": ("INT", {"default": 16, "min": 1, "max": 16}),
                "feedback": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.8, "step": 0.01}),
                "ring_freq": ("FLOAT", {"default": 440.0, "min": 1.0, "max": 5000.0, "step": 1.0}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "noise_c": ("AUDIO",),
                "modulation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("chaos_audio",)
    FUNCTION = "mix_chaos"
    CATEGORY = "NoiseGen"
    
    def mix_chaos(self, noise_a, noise_b, mix_mode, mix_ratio, chaos_amount, 
                  distortion, bit_crush, feedback, ring_freq, amplitude,
                  noise_c=None, modulation=0.0):
        """Create chaotic noise mixes for extreme audio textures."""
        try:
            # Extract waveforms and ensure same sample rate
            waveform_a = noise_a["waveform"]
            waveform_b = noise_b["waveform"]
            sample_rate = noise_a["sample_rate"]
            
            # Convert to CPU and numpy for processing
            if hasattr(waveform_a, 'cpu'):
                waveform_a = waveform_a.cpu().numpy()
            if hasattr(waveform_b, 'cpu'):
                waveform_b = waveform_b.cpu().numpy()
            
            # Ensure same length (trim to shorter)
            min_len = min(waveform_a.shape[-1], waveform_b.shape[-1])
            waveform_a = waveform_a[..., :min_len]
            waveform_b = waveform_b[..., :min_len]
            
            # Apply chaos mixing
            mixed = self._apply_chaos_mix(waveform_a, waveform_b, mix_mode, mix_ratio, 
                                        chaos_amount, sample_rate, ring_freq)
            
            # Add third noise source if provided
            if noise_c is not None:
                waveform_c = noise_c["waveform"]
                if hasattr(waveform_c, 'cpu'):
                    waveform_c = waveform_c.cpu().numpy()
                waveform_c = waveform_c[..., :min_len]
                # Mix in the third source with modulation
                mixed = mixed * (1 - modulation) + waveform_c * modulation
            
            # Apply harsh processing effects
            mixed = self._apply_distortion(mixed, distortion)
            mixed = self._apply_bit_crush(mixed, bit_crush)
            mixed = self._apply_feedback(mixed, feedback)
            
            # Final amplitude scaling
            if np.max(np.abs(mixed)) > 0:
                mixed = mixed / np.max(np.abs(mixed)) * amplitude
            
            # Convert back to ComfyUI format
            audio_output = numpy_to_comfy_audio(mixed, sample_rate)
            return (audio_output,)
            
        except Exception as e:
            print(f"Error in chaos noise mixing: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return silence on error
            silence = np.zeros_like(waveform_a, dtype=np.float32)
            audio_output = numpy_to_comfy_audio(silence, sample_rate)
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
    CATEGORY = "NoiseGen/Utils"
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
            
            print(f"Audio saved to: {filepath}")
            print(f"Output location: ComfyUI/output/audio/{filename}")
            return (audio, filepath)
            
        except Exception as e:
            print(f"Error saving audio: {str(e)}")
            import traceback
            traceback.print_exc()
            return (audio, "Error: Could not save audio")

class BlueNoiseNode:
    """Dedicated blue noise generator."""
    
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
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "NoiseGen/Advanced"
    
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
    """Dedicated violet noise generator."""
    
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
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "NoiseGen/Advanced"
    
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

class AudioPreviewNode:
    """Preview generated audio directly in ComfyUI interface."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "NoiseGen_Preview_"}),
            }
        }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "preview_audio"
    CATEGORY = "NoiseGen/Utils"
    OUTPUT_NODE = True  # This makes it a preview/output node
    
    def preview_audio(self, audio, filename_prefix="NoiseGen_Preview_"):
        """Preview audio in ComfyUI interface with temporary file."""
        try:
            import tempfile
            import time
            import folder_paths
            
            # Extract audio data
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # Convert to CPU if on GPU
            if hasattr(waveform, 'cpu'):
                waveform = waveform.cpu()
            
            # Ensure waveform is 2D for torchaudio.save [channels, samples]
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # [samples] -> [1, samples]
            elif waveform.ndim > 2:
                waveform = waveform.view(waveform.size(0), -1)
            
            # Ensure proper data type
            waveform = waveform.float()
            
            # Create temporary preview file in ComfyUI's temp directory
            temp_dir = folder_paths.get_temp_directory()
            timestamp = str(int(time.time()))
            temp_filename = f"{filename_prefix}{timestamp}.wav"
            temp_filepath = os.path.join(temp_dir, temp_filename)
            
            # Save to temporary file for preview
            torchaudio.save(temp_filepath, waveform, sample_rate)
            
            print(f"Audio preview ready: {temp_filename}")
            print(f"Duration: {waveform.shape[-1] / sample_rate:.2f}s")
            print(f"Sample Rate: {sample_rate}Hz")
            print(f"Channels: {waveform.shape[0]}")
            
            # Return the temporary filepath for ComfyUI to handle preview
            return {"ui": {"audio": [temp_filepath]}}
            
        except Exception as e:
            print(f"Error creating audio preview: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"ui": {"audio": []}}

# Node mappings for ComfyUI - OPTIMIZED UX
NODE_CLASS_MAPPINGS = {
    # MAIN NODES - Clean and focused
    "NoiseGenerator": NoiseGeneratorNode,        # Universal - handles all basic types
    "PerlinNoise": PerlinNoiseNode,             # Unique parameters (frequency, octaves)
    "BandLimitedNoise": BandLimitedNoiseNode,   # Unique parameters (freq filtering)
    "ChaosNoiseMix": ChaosNoiseMixNode,         # Advanced mixing
    "AudioSave": AudioSaveNode,                 # Utility
    "AudioPreview": AudioPreviewNode,           # Preview playback
    
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
    "NoiseGenerator": "Noise Generator",
    "PerlinNoise": "Perlin Noise",
    "BandLimitedNoise": "Band-Limited Noise", 
    "ChaosNoiseMix": "Chaos Noise Mix",
    "AudioSave": "Save Audio",
    "AudioPreview": "Preview Audio",
    
    # LEGACY - Hidden with underscore prefix
    "_WhiteNoise": "White Noise (Legacy)",
    "_PinkNoise": "Pink Noise (Legacy)", 
    "_BrownNoise": "Brown Noise (Legacy)",
    "_BlueNoise": "Blue Noise (Legacy)",
    "_VioletNoise": "Violet Noise (Legacy)",
} 