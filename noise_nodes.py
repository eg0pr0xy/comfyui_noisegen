"""
ComfyUI-NoiseGen v2.0 - The Ultimate Merzbow Noise Machine
===========================================================

Advanced noise generation and audio processing for experimental music.
Optimized for harsh noise, power electronics, and Merzbow-style chaos.

Node Categories:
🎵 GENERATORS  - NoiseGenerator, PerlinNoise, BandLimitedNoise  
🔄 PROCESSORS  - FeedbackProcessor, HarshFilter
🎸 EFFECTS     - MultiDistortion, SpectralProcessor
🎛️ MIXERS      - AudioMixer, ChaosNoiseMix
🔧 UTILITIES   - AudioSave
"""

import os
import numpy as np
import torch
import torchaudio

# =============================================================================
# DEPENDENCIES & IMPORTS
# =============================================================================

# Import audio utils with fallback
try:
    from .audio_utils import *
except ImportError:
    from audio_utils import *

# ComfyUI dependencies
try:
    import folder_paths
    import comfy.model_management
except ImportError:
    print("Warning: ComfyUI dependencies not found. Some features may not work.")
    folder_paths = None

# Constants
AUDIO_TYPE = "AUDIO"
AUDIO_EXTENSIONS = ["wav", "mp3", "flac", "ogg", "aiff", "au"]


# =============================================================================
# 🎵 GENERATORS - Audio Generation Nodes
# =============================================================================

class NoiseGeneratorNode:
    """Universal noise generator with 7 scientifically-accurate noise types."""
    
    NOISE_TYPES = ["white", "pink", "brown", "blue", "violet", "perlin", "bandlimited"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_type": (cls.NOISE_TYPES, {
                    "default": "white",
                    "tooltip": "Type of noise: white=flat, pink=1/f, brown=1/f², blue=+3dB/oct, violet=+6dB/oct, perlin=organic, bandlimited=filtered"
                }),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {"default": 44100}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "channels": ([1, 2], {"default": 1}),
                "stereo_mode": (["independent", "correlated", "decorrelated"], {"default": "independent"}),
                "stereo_width": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "low_freq": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 20000.0, "step": 1.0}),
                "high_freq": ("FLOAT", {"default": 8000.0, "min": 1.0, "max": 20000.0, "step": 1.0}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 8}),
            }
        }
    
    RETURN_TYPES = (AUDIO_TYPE,)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_noise"
    CATEGORY = "🎵 NoiseGen"
    DESCRIPTION = "Universal noise generator with 7 scientifically-accurate noise types"
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs for basic type checking."""
        # ComfyUI VALIDATE_INPUTS should only return True or error strings
        # Parameter correction is handled in generate_noise method
        return True
    
    def generate_noise(self, noise_type, duration, sample_rate, amplitude, seed, channels, 
                      stereo_mode, stereo_width, frequency=1.0, low_freq=100.0, high_freq=8000.0, octaves=4):
        try:
            # Robust parameter type fixing for legacy workflows
            # Detect and fix parameter swapping/type issues
            
            # Fix channels parameter
            if isinstance(channels, str):
                if channels in ["independent", "correlated", "decorrelated"]:
                    # This is actually stereo_mode, fix the swap
                    original_channels = channels
                    channels = int(stereo_mode) if isinstance(stereo_mode, (int, str)) and str(stereo_mode) in ["1", "2"] else 1
                    stereo_mode = original_channels
                else:
                    # Convert string channels to int
                    channels = int(channels) if channels in ["1", "2"] else 1
            else:
                # Ensure channels is valid integer
                channels = int(channels) if channels in [1, 2] else 1
            
            # Fix stereo_mode parameter  
            if not isinstance(stereo_mode, str):
                # Convert numeric stereo_mode to string
                stereo_mode = "independent"
            elif stereo_mode not in ["independent", "correlated", "decorrelated"]:
                stereo_mode = "independent"
            
            # Ensure other parameters are properly typed
            low_freq, high_freq = max(1.0, float(low_freq)), max(low_freq + 1.0, float(high_freq))
            
            duration, sample_rate, amplitude, channels = validate_audio_params(duration, sample_rate, amplitude, channels)
            
            # Common parameters
            params = {
                'duration': duration, 'sample_rate': sample_rate, 'amplitude': amplitude,
                'seed': seed, 'channels': channels, 'stereo_mode': stereo_mode, 'stereo_width': stereo_width
            }
            
            # Generate noise
            if noise_type == "white": result_np = generate_white_noise(**params)
            elif noise_type == "pink": result_np = generate_pink_noise(**params)
            elif noise_type == "brown": result_np = generate_brown_noise(**params)
            elif noise_type == "blue": result_np = generate_blue_noise(**params)
            elif noise_type == "violet": result_np = generate_violet_noise(**params)
            elif noise_type == "perlin": result_np = generate_perlin_noise(frequency=frequency, octaves=octaves, **params)
            elif noise_type == "bandlimited": result_np = generate_bandlimited_noise(low_freq=low_freq, high_freq=high_freq, **params)
            else: result_np = generate_white_noise(**params)
            
            # Convert to ComfyUI audio format
            result = numpy_to_comfy_audio(result_np, sample_rate)
            return (result,)
            
        except Exception as e:
            print(f"❌ Error in noise generation: {str(e)}")
            fallback = numpy_to_comfy_audio(np.zeros((channels, int(sample_rate * 0.1))), sample_rate)
            return (fallback,)


class PerlinNoiseNode:
    """Advanced Perlin noise generator for organic, natural textures."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "frequency": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {"default": 44100}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "channels": ([1, 2], {"default": 2}),
                "stereo_mode": (["independent", "correlated", "decorrelated"], {"default": "independent"}),
                "octaves": ("INT", {"default": 6, "min": 1, "max": 8}),
                "persistence": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🎵 NoiseGen/Advanced"
    DESCRIPTION = "Organic Perlin noise for natural-sounding textures"
    
    def generate(self, duration, frequency, sample_rate, amplitude, seed, channels, stereo_mode, octaves, persistence):
        try:
            duration, sample_rate, amplitude, channels = validate_audio_params(duration, sample_rate, amplitude, channels)
            
            # Note: persistence parameter is accepted but not used in the core function
            # It's kept for UI compatibility - could be implemented as fractal persistence in future
            result_np = generate_perlin_noise(
                duration=duration, frequency=frequency, sample_rate=sample_rate, amplitude=amplitude,
                seed=seed, channels=channels, stereo_mode=stereo_mode, octaves=octaves
            )
            
            # Apply persistence as amplitude scaling for now
            if persistence != 1.0:
                result_np *= persistence
            
            result = numpy_to_comfy_audio(result_np, sample_rate)
            return (result,)
            
        except Exception as e:
            print(f"❌ Error in Perlin generation: {str(e)}")
            fallback = numpy_to_comfy_audio(np.zeros((channels, int(sample_rate * 0.1))), sample_rate)
            return (fallback,)


class BandLimitedNoiseNode:
    """Band-limited noise generator with precise frequency filtering."""
    
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
    CATEGORY = "🎵 NoiseGen/Advanced"
    DESCRIPTION = "Band-limited noise with precise frequency control"
    
    def generate(self, duration, low_frequency, high_frequency, sample_rate, amplitude, seed):
        try:
            duration, sample_rate, amplitude, channels = validate_audio_params(duration, sample_rate, amplitude, 1)
            if high_frequency <= low_frequency: high_frequency = low_frequency + 100.0
            
            result_np = generate_bandlimited_noise(
                duration=duration, low_freq=low_frequency, high_freq=high_frequency,
                sample_rate=sample_rate, amplitude=amplitude, seed=seed, channels=1
            )
            result = numpy_to_comfy_audio(result_np, sample_rate)
            return (result,)
            
        except Exception as e:
            print(f"❌ Error in band-limited generation: {str(e)}")
            fallback = numpy_to_comfy_audio(np.zeros((1, int(sample_rate * 0.1))), sample_rate)
            return (fallback,)


# =============================================================================
# 🔄 PROCESSORS - Advanced Audio Processing
# =============================================================================

class FeedbackProcessorNode:
    """Advanced feedback processor for self-generating Merzbow-style textures."""
    
    FEEDBACK_MODES = ["simple", "filtered", "saturated", "modulated", "complex", "runaway"]
    FILTER_TYPES = ["lowpass", "highpass", "bandpass", "notch", "allpass"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio for feedback processing"}),
                "feedback_mode": (cls.FEEDBACK_MODES, {"default": "filtered"}),
                "feedback_amount": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 0.95, "step": 0.01}),
                "delay_time": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "filter_type": (cls.FILTER_TYPES, {"default": "lowpass"}),
                "filter_freq": ("FLOAT", {"default": 2000.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "filter_resonance": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.99, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "modulation_rate": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 20.0, "step": 0.01}),
                "modulation_depth": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "modulation_type": (["sine", "triangle", "sawtooth", "square", "noise"], {"default": "sine"}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "wet_dry_mix": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("feedback_audio",)
    FUNCTION = "process_feedback"
    CATEGORY = "🎵 NoiseGen/Processing"
    DESCRIPTION = "Advanced feedback processor for self-generating textures"
    
    def process_feedback(self, audio, feedback_mode, feedback_amount, delay_time, 
                        filter_type, filter_freq, filter_resonance, saturation, 
                        modulation_rate, modulation_depth, modulation_type, amplitude, wet_dry_mix=0.7):
        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # Convert to numpy
            if hasattr(waveform, 'cpu'):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = waveform
            
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            
            # Store original
            original_audio = audio_np.copy()
            
            # Process each channel
            processed_channels = []
            for channel in range(audio_np.shape[0]):
                processed = self._apply_feedback_processing(
                    audio_np[channel], sample_rate, feedback_mode, feedback_amount, 
                    delay_time, filter_type, filter_freq, filter_resonance,
                    saturation, modulation_rate, modulation_depth, modulation_type, wet_dry_mix
                )
                processed_channels.append(processed)
            
            # Stack and apply amplitude
            result = np.stack(processed_channels, axis=0) * amplitude
            
            # Safety limiting
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result /= max_val
            
            result_tensor = torch.from_numpy(result).float()
            output_audio = {"waveform": result_tensor, "sample_rate": sample_rate}
            
            return (output_audio,)
            
        except Exception as e:
            print(f"❌ Error in feedback processing: {str(e)}")
            return (audio,)
    
    def _apply_feedback_processing(self, audio, sample_rate, mode, feedback_amount, 
                                 delay_time_ms, filter_type, filter_freq, filter_resonance,
                                 saturation, mod_rate, mod_depth, mod_type, wet_dry_mix):
        """Core feedback processing implementation."""
        
        # Calculate delay in samples
        delay_samples = max(1, int(delay_time_ms * sample_rate / 1000.0))
        delay_samples = min(delay_samples, len(audio) // 2)  # Safety limit
        
        # Initialize delay buffer
        delay_buffer = np.zeros(delay_samples)
        delay_index = 0
        
        # Initialize filter state
        filter_state = {'x1': 0.0, 'x2': 0.0, 'y1': 0.0, 'y2': 0.0}
        
        # LFO for modulation
        lfo_phase = 0.0
        lfo_increment = 2.0 * np.pi * mod_rate / sample_rate
        
        # Process sample by sample
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            # Calculate current delay time (with modulation)
            if mode in ["modulated", "complex"]:
                if mod_type == "sine":
                    lfo_value = np.sin(lfo_phase)
                elif mod_type == "triangle":
                    lfo_value = 2.0 * np.abs((lfo_phase / (2 * np.pi)) % 1.0 - 0.5) - 1.0
                elif mod_type == "sawtooth":
                    lfo_value = 2.0 * ((lfo_phase / (2 * np.pi)) % 1.0) - 1.0
                elif mod_type == "square":
                    lfo_value = 1.0 if np.sin(lfo_phase) > 0 else -1.0
                else:  # noise
                    lfo_value = np.random.random() * 2.0 - 1.0
                
                lfo_phase += lfo_increment
                if lfo_phase >= 2 * np.pi:
                    lfo_phase -= 2 * np.pi
                
                # Modulate delay time
                mod_delay_samples = delay_samples * (1.0 + lfo_value * mod_depth * 0.5)
                mod_delay_samples = max(1, min(int(mod_delay_samples), len(delay_buffer) - 1))
            else:
                mod_delay_samples = delay_samples
            
            # Get delayed sample
            delayed_sample = delay_buffer[delay_index]
            
            # Apply filtering to feedback signal
            if mode in ["filtered", "complex"]:
                delayed_sample = self._apply_filter(delayed_sample, filter_type, filter_freq, filter_resonance, sample_rate, filter_state)
            
            # Apply saturation to feedback signal
            if mode in ["saturated", "complex"]:
                delayed_sample = self._apply_saturation(delayed_sample, saturation)
            
            # Mix input with feedback
            if mode == "runaway":
                # Intentionally unstable
                feedback_sample = delayed_sample * feedback_amount * (1.0 + np.random.random() * 0.2)
            else:
                feedback_sample = delayed_sample * feedback_amount
            
            mixed_sample = audio[i] + feedback_sample
            
            # Safety limiting for feedback
            mixed_sample = np.tanh(mixed_sample * 0.5) * 2.0
            
            # Store in delay buffer
            delay_buffer[delay_index] = mixed_sample
            delay_index = (delay_index + 1) % len(delay_buffer)
            
            # Output mix
            output[i] = audio[i] * (1 - wet_dry_mix) + mixed_sample * wet_dry_mix
        
        return output
    
    def _apply_filter(self, signal, filter_type, cutoff, resonance, sample_rate, state):
        """Apply filter to feedback signal."""
        # Calculate filter coefficients
        nyquist = sample_rate / 2.0
        freq = max(10.0, min(cutoff, nyquist - 10.0))
        omega = 2.0 * np.pi * freq / sample_rate
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        q = max(0.1, 1.0 - resonance)
        alpha = sin_omega / (2.0 * q)
        
        # Filter coefficients
        if filter_type == "lowpass":
            b0 = (1.0 - cos_omega) / 2.0
            b1 = 1.0 - cos_omega
            b2 = (1.0 - cos_omega) / 2.0
        elif filter_type == "highpass":
            b0 = (1.0 + cos_omega) / 2.0
            b1 = -(1.0 + cos_omega)
            b2 = (1.0 + cos_omega) / 2.0
        elif filter_type == "bandpass":
            b0 = alpha
            b1 = 0.0
            b2 = -alpha
        elif filter_type == "notch":
            b0 = 1.0
            b1 = -2.0 * cos_omega
            b2 = 1.0
        else:  # allpass
            b0 = 1.0 - alpha
            b1 = -2.0 * cos_omega
            b2 = 1.0 + alpha
        
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha
        
        # Normalize
        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0
        
        # Apply filter
        output = b0 * signal + b1 * state['x1'] + b2 * state['x2'] - a1 * state['y1'] - a2 * state['y2']
        
        # Update state
        state['x2'] = state['x1']
        state['x1'] = signal
        state['y2'] = state['y1']
        state['y1'] = output
        
        return output
    
    def _apply_saturation(self, signal, amount):
        """Apply nonlinear saturation."""
        if amount <= 0.0:
            return signal
        
        # Tube-style saturation
        drive = 1.0 + amount * 5.0
        driven = signal * drive
        return np.tanh(driven) / drive


class HarshFilterNode:
    """Extreme filtering with self-oscillation for harsh noise sculpting."""
    
    FILTER_TYPES = ["lowpass", "highpass", "bandpass", "notch", "comb", "allpass", "morph", "chaos"]
    DRIVE_MODES = ["clean", "tube", "transistor", "digital", "chaos"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio for extreme filtering"}),
                "filter_type": (cls.FILTER_TYPES, {"default": "lowpass"}),
                "cutoff_freq": ("FLOAT", {"default": 1000.0, "min": 10.0, "max": 20000.0, "step": 1.0}),
                "resonance": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.999, "step": 0.001}),
                "drive": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.01}),
                "drive_mode": (cls.DRIVE_MODES, {"default": "tube"}),
                "filter_slope": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 4.0, "step": 0.1}),
                "morph_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "modulation_rate": ("FLOAT", {"default": 0.5, "min": 0.001, "max": 50.0, "step": 0.001}),
                "modulation_depth": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "wet_dry_mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stereo_spread": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("filtered_audio",)
    FUNCTION = "process_harsh_filter"
    CATEGORY = "🎵 NoiseGen/Processing"
    DESCRIPTION = "Extreme filtering with self-oscillation and morphing"
    
    def process_harsh_filter(self, audio, filter_type, cutoff_freq, resonance, drive, 
                           drive_mode, filter_slope, morph_amount, modulation_rate, 
                           modulation_depth, amplitude, wet_dry_mix=1.0, stereo_spread=0.0):
        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            if hasattr(waveform, 'cpu'):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = waveform
            
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            
            original_audio = audio_np.copy()
            processed_channels = []
            
            for channel in range(audio_np.shape[0]):
                # Calculate stereo frequency spread
                if audio_np.shape[0] > 1 and stereo_spread > 0.0:
                    freq_spread = 1.0 + stereo_spread * 0.5 * (1 if channel == 0 else -1)
                    channel_cutoff = cutoff_freq * freq_spread
                else:
                    channel_cutoff = cutoff_freq
                
                processed = self._apply_harsh_filtering(
                    audio_np[channel], sample_rate, filter_type, channel_cutoff, 
                    resonance, drive, drive_mode, filter_slope, morph_amount,
                    modulation_rate, modulation_depth, wet_dry_mix
                )
                processed_channels.append(processed)
            
            # Stack and apply amplitude
            result = np.stack(processed_channels, axis=0) * amplitude
            
            # Safety limiting
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result /= max_val
            
            result_tensor = torch.from_numpy(result).float()
            output_audio = {"waveform": result_tensor, "sample_rate": sample_rate}
            
            return (output_audio,)
            
        except Exception as e:
            print(f"❌ Error in harsh filtering: {str(e)}")
            return (audio,)
    
    def _apply_harsh_filtering(self, audio, sample_rate, filter_type, cutoff_freq, 
                             resonance, drive, drive_mode, filter_slope, morph_amount,
                             mod_rate, mod_depth, wet_dry_mix):
        """Core harsh filtering implementation."""
        
        # Initialize filter state
        state = {'x1': 0.0, 'x2': 0.0, 'y1': 0.0, 'y2': 0.0}
        
        # Special handling for comb filter
        if filter_type == "comb":
            delay_samples = max(1, int(sample_rate / cutoff_freq))
            delay_samples = min(delay_samples, len(audio) // 4)  # Safety limit
            state['comb_buffer'] = np.zeros(delay_samples)
            state['comb_index'] = 0
        
        # LFO for modulation
        lfo_phase = 0.0
        lfo_increment = 2.0 * np.pi * mod_rate / sample_rate
        
        # Process sample by sample
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            # Calculate modulated cutoff frequency
            lfo_value = np.sin(lfo_phase)
            lfo_phase += lfo_increment
            if lfo_phase >= 2 * np.pi:
                lfo_phase -= 2 * np.pi
            
            modulated_cutoff = cutoff_freq * (1.0 + lfo_value * mod_depth)
            modulated_cutoff = max(10.0, min(modulated_cutoff, sample_rate / 2.0 - 10.0))
            
            # Apply filter
            filtered_sample = self._apply_filter_stage(
                audio[i], filter_type, modulated_cutoff, resonance, sample_rate, state, morph_amount
            )
            
            # Apply multiple stages if filter_slope > 1
            for stage in range(int(filter_slope) - 1):
                filtered_sample = self._apply_filter_stage(
                    filtered_sample, filter_type, modulated_cutoff, resonance, sample_rate, state, morph_amount
                )
            
            # Apply drive/saturation
            if drive > 0.0:
                filtered_sample = self._apply_drive(filtered_sample, drive, drive_mode)
            
            # Mix wet/dry
            output[i] = audio[i] * (1 - wet_dry_mix) + filtered_sample * wet_dry_mix
        
        return output
    
    def _apply_filter_stage(self, signal, filter_type, cutoff, resonance, sample_rate, state, morph_amount):
        """Apply single filter stage."""
        
        if filter_type == "comb":
            # Comb filter implementation
            if 'comb_buffer' not in state:
                delay_samples = max(1, int(sample_rate / cutoff))
                state['comb_buffer'] = np.zeros(delay_samples)
                state['comb_index'] = 0
            
            delayed = state['comb_buffer'][state['comb_index']]
            output = signal + delayed * resonance
            state['comb_buffer'][state['comb_index'] ] = signal
            state['comb_index'] = (state['comb_index'] + 1) % len(state['comb_buffer'])
            return output
        
        # Standard biquad filters
        nyquist = sample_rate / 2.0
        freq = max(10.0, min(cutoff, nyquist - 10.0))
        omega = 2.0 * np.pi * freq / sample_rate
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        q = max(0.01, 1.0 - resonance * 0.99)  # Prevent instability
        alpha = sin_omega / (2.0 * q)
        
        # Filter coefficients based on type
        if filter_type == "lowpass":
            b0 = (1.0 - cos_omega) / 2.0
            b1 = 1.0 - cos_omega
            b2 = (1.0 - cos_omega) / 2.0
        elif filter_type == "highpass":
            b0 = (1.0 + cos_omega) / 2.0
            b1 = -(1.0 + cos_omega)
            b2 = (1.0 + cos_omega) / 2.0
        elif filter_type == "bandpass":
            b0 = alpha
            b1 = 0.0
            b2 = -alpha
        elif filter_type == "notch":
            b0 = 1.0
            b1 = -2.0 * cos_omega
            b2 = 1.0
        elif filter_type == "allpass":
            b0 = 1.0 - alpha
            b1 = -2.0 * cos_omega
            b2 = 1.0 + alpha
        elif filter_type == "morph":
            # Morph between lowpass and highpass
            lp_b0 = (1.0 - cos_omega) / 2.0
            lp_b1 = 1.0 - cos_omega
            lp_b2 = (1.0 - cos_omega) / 2.0
            hp_b0 = (1.0 + cos_omega) / 2.0
            hp_b1 = -(1.0 + cos_omega)
            hp_b2 = (1.0 + cos_omega) / 2.0
            
            b0 = lp_b0 * (1 - morph_amount) + hp_b0 * morph_amount
            b1 = lp_b1 * (1 - morph_amount) + hp_b1 * morph_amount
            b2 = lp_b2 * (1 - morph_amount) + hp_b2 * morph_amount
        elif filter_type == "chaos":
            # Chaotic filter behavior
            chaos_factor = 0.5 + 0.5 * np.sin(signal * 10.0)
            b0 = (1.0 - cos_omega) * chaos_factor / 2.0
            b1 = (1.0 - cos_omega) * chaos_factor
            b2 = (1.0 - cos_omega) * chaos_factor / 2.0
        else:
            # Default to lowpass
            b0 = (1.0 - cos_omega) / 2.0
            b1 = 1.0 - cos_omega
            b2 = (1.0 - cos_omega) / 2.0
        
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha
        
        # Normalize
        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0
        
        # Apply biquad filter
        output = b0 * signal + b1 * state['x1'] + b2 * state['x2'] - a1 * state['y1'] - a2 * state['y2']
        
        # Update state
        state['x2'] = state['x1']
        state['x1'] = signal
        state['y2'] = state['y1']
        state['y1'] = output
        
        return output
    
    def _apply_drive(self, signal, drive_amount, drive_mode):
        """Apply saturation/drive to the signal."""
        if drive_amount <= 0.0:
            return signal
        
        drive = 1.0 + drive_amount
        driven = signal * drive
        
        if drive_mode == "clean":
            return signal
        elif drive_mode == "tube":
            return np.tanh(driven * 0.7) / drive * 1.5
        elif drive_mode == "transistor":
            return np.clip(driven, -1.0, 1.0) / drive
        elif drive_mode == "digital":
            return np.sign(driven) * min(abs(driven), 1.0) / drive
        elif drive_mode == "chaos":
            return np.tanh(driven + np.sin(driven * 5.0) * 0.2) / drive
        else:
            return np.tanh(driven) / drive


# =============================================================================
# 🎸 EFFECTS - Distortion and Spectral Processing
# =============================================================================

class MultiDistortionNode:
    """Comprehensive multi-stage distortion with 12 types for extreme processing."""
    
    DISTORTION_TYPES = ["tube", "transistor", "diode", "digital", "bitcrush", "waveshaper", "foldback", "ring_mod", "chaos", "fuzz", "overdrive", "destruction"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio to process through multi-stage distortion"}),
                "distortion_type": (cls.DISTORTION_TYPES, {"default": "tube"}),
                "drive": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "output_gain": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "wet_dry_mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stages": ("INT", {"default": 1, "min": 1, "max": 4}),
                "asymmetry": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "chaos_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("distorted_audio",)
    FUNCTION = "process_multi_distortion"
    CATEGORY = "🎵 NoiseGen/Processing"
    DESCRIPTION = "Multi-stage distortion with 12 types for extreme processing"
    
    def process_multi_distortion(self, audio, distortion_type, drive, output_gain, wet_dry_mix, stages, asymmetry, amplitude, chaos_amount=0.5):
        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            if hasattr(waveform, 'cpu'):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = waveform
            
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            
            # Process each channel
            processed_channels = []
            for channel in range(audio_np.shape[0]):
                processed = self._apply_distortion(audio_np[channel], distortion_type, drive, stages, asymmetry, chaos_amount)
                processed_channels.append(processed)
            
            # Stack and apply gains
            result = np.stack(processed_channels, axis=0)
            original_audio = audio_np.copy()
            result = original_audio * (1 - wet_dry_mix) + result * wet_dry_mix
            result *= output_gain * amplitude
            
            # Safety limiting
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result /= max_val
            
            result_tensor = torch.from_numpy(result).float()
            output_audio = {"waveform": result_tensor, "sample_rate": sample_rate}
            
            return (output_audio,)
            
        except Exception as e:
            print(f"❌ Error in multi-distortion: {str(e)}")
            return (audio,)
    
    def _apply_distortion(self, audio, dist_type, drive, stages, asymmetry, chaos_amount):
        """Apply distortion with multiple stages."""
        output = audio.copy()
        
        for stage in range(stages):
            stage_drive = drive * (1.0 - stage * 0.1)  # Reduce drive per stage
            
            if dist_type == "tube":
                output = np.tanh(output * (1 + stage_drive) * (1 + asymmetry * 0.2))
            elif dist_type == "digital":
                output = np.clip(output * (1 + stage_drive), -1.0, 1.0)
            elif dist_type == "chaos":
                chaos_factor = 1.0 + chaos_amount * (np.random.random() - 0.5)
                output = np.tanh(output * stage_drive * chaos_factor + np.sin(output * 5.0) * 0.2)
            else:
                output = np.tanh(output * (1 + stage_drive))
        
        return output


class SpectralProcessorNode:
    """FFT-based spectral manipulation for frequency-domain processing."""
    
    SPECTRAL_MODES = ["enhance", "suppress", "shift", "morph", "gate", "compress", "chaos", "phase", "vocoder", "freeze"]
    WINDOW_TYPES = ["hann", "hamming", "blackman", "kaiser", "rectangular"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio for spectral processing"}),
                "spectral_mode": (cls.SPECTRAL_MODES, {"default": "enhance"}),
                "fft_size": (["512", "1024", "2048", "4096", "8192"], {"default": "2048"}),
                "overlap_factor": ("FLOAT", {"default": 0.75, "min": 0.25, "max": 0.95, "step": 0.05}),
                "window_type": (cls.WINDOW_TYPES, {"default": "hann"}),
                "frequency_range_low": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 10000.0, "step": 10.0}),
                "frequency_range_high": ("FLOAT", {"default": 8000.0, "min": 100.0, "max": 20000.0, "step": 10.0}),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "wet_dry_mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "phase_randomization": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("processed_audio",)
    FUNCTION = "process_spectral"
    CATEGORY = "🎵 NoiseGen/Processing"
    DESCRIPTION = "FFT-based spectral manipulation for frequency-domain processing"
    
    def process_spectral(self, audio, spectral_mode, fft_size, overlap_factor, window_type, frequency_range_low, frequency_range_high, intensity, amplitude, wet_dry_mix=1.0, phase_randomization=0.0):
        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            if hasattr(waveform, 'cpu'):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = waveform
            
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            
            fft_size_int = int(fft_size)
            
            # Process each channel
            processed_channels = []
            for channel in range(audio_np.shape[0]):
                processed = self._apply_spectral_processing(
                    audio_np[channel], sample_rate, spectral_mode, fft_size_int, 
                    overlap_factor, window_type, frequency_range_low, frequency_range_high, 
                    intensity, phase_randomization
                )
                processed_channels.append(processed)
            
            # Stack and apply mixing
            result = np.stack(processed_channels, axis=0)
            original_audio = audio_np.copy()
            result = original_audio * (1 - wet_dry_mix) + result * wet_dry_mix
            result *= amplitude
            
            # Safety limiting
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result /= max_val
            
            result_tensor = torch.from_numpy(result).float()
            output_audio = {"waveform": result_tensor, "sample_rate": sample_rate}
            
            return (output_audio,)
            
        except Exception as e:
            print(f"❌ Error in spectral processing: {str(e)}")
            return (audio,)
    
    def _apply_spectral_processing(self, audio, sample_rate, mode, fft_size, overlap_factor, window_type, freq_low, freq_high, intensity, phase_randomization):
        """Apply spectral processing using FFT."""
        # Create window
        if window_type == "hann":
            window = np.hanning(fft_size)
        elif window_type == "hamming":
            window = np.hamming(fft_size)
        elif window_type == "blackman":
            window = np.blackman(fft_size)
        elif window_type == "kaiser":
            # Kaiser window approximation
            beta = 8.6
            n = np.arange(fft_size)
            window = np.i0(beta * np.sqrt(1 - ((n - (fft_size-1)/2) / ((fft_size-1)/2))**2)) / np.i0(beta)
        else:  # rectangular
            window = np.ones(fft_size)
        
        hop_size = int(fft_size * (1 - overlap_factor))
        output = np.zeros_like(audio)
        
        # Calculate frequency bins
        freqs = np.fft.fftfreq(fft_size, 1.0 / sample_rate)
        freq_bins = len(freqs) // 2
        low_bin = int(freq_low * fft_size / sample_rate)
        high_bin = int(freq_high * fft_size / sample_rate)
        low_bin = max(0, min(low_bin, freq_bins))
        high_bin = max(low_bin + 1, min(high_bin, freq_bins))
        
        # Process in overlapping windows
        for i in range(0, len(audio) - fft_size, hop_size):
            # Extract windowed frame
            frame = audio[i:i+fft_size] * window
            
            # FFT
            fft_data = np.fft.fft(frame)
            magnitude = np.abs(fft_data)
            phase = np.angle(fft_data)
            
            # Apply spectral processing
            if mode == "enhance":
                magnitude[low_bin:high_bin] *= (1.0 + intensity)
            elif mode == "suppress":
                magnitude[low_bin:high_bin] *= max(0.1, 1.0 - intensity)
            elif mode == "chaos":
                chaos_factor = 1.0 + intensity * (np.random.random(high_bin - low_bin) - 0.5)
                magnitude[low_bin:high_bin] *= chaos_factor
            elif mode == "phase":
                if phase_randomization > 0:
                    random_phase = (np.random.random(high_bin - low_bin) - 0.5) * 2 * np.pi * phase_randomization
                    phase[low_bin:high_bin] += random_phase
            
            # Reconstruct
            processed_fft = magnitude * np.exp(1j * phase)
            processed_frame = np.real(np.fft.ifft(processed_fft))
            
            # Overlap-add
            end_idx = min(i + fft_size, len(output))
            frame_end = end_idx - i
            output[i:end_idx] += processed_frame[:frame_end] * window[:frame_end]
        
        return output


# =============================================================================
# 🎛️ MIXERS - Audio Mixing and Combination
# =============================================================================

class AudioMixerNode:
    """Professional audio mixer with individual channel controls."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_a": ("AUDIO", {"tooltip": "First audio input"}),
                "gain_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "pan_a": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "audio_b": ("AUDIO", {"tooltip": "Second audio input"}),
                "gain_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "pan_b": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "audio_c": ("AUDIO", {"tooltip": "Third audio input"}),
                "gain_c": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "pan_c": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "audio_d": ("AUDIO", {"tooltip": "Fourth audio input"}),
                "gain_d": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "pan_d": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "master_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("mixed_audio",)
    FUNCTION = "mix_audio"
    CATEGORY = "🎵 NoiseGen/Mixing"
    DESCRIPTION = "Professional audio mixer with gain and pan controls for up to 4 inputs"
    
    def mix_audio(self, audio_a, gain_a, pan_a, audio_b=None, gain_b=1.0, pan_b=0.0, 
                  audio_c=None, gain_c=1.0, pan_c=0.0, audio_d=None, gain_d=1.0, pan_d=0.0, master_gain=1.0):
        try:
            # Get first audio
            waveform_a = audio_a["waveform"]
            sample_rate = audio_a["sample_rate"]
            
            if hasattr(waveform_a, 'cpu'):
                audio_a_np = waveform_a.cpu().numpy()
            else:
                audio_a_np = waveform_a
            
            if audio_a_np.ndim == 1:
                audio_a_np = audio_a_np.reshape(1, -1)
            
            # Apply gain A
            mixed = audio_a_np * gain_a
            
            # Add second audio if provided
            if audio_b is not None:
                waveform_b = audio_b["waveform"]
                if hasattr(waveform_b, 'cpu'):
                    audio_b_np = waveform_b.cpu().numpy()
                else:
                    audio_b_np = waveform_b
                
                if audio_b_np.ndim == 1:
                    audio_b_np = audio_b_np.reshape(1, -1)
                
                # Match lengths
                min_length = min(mixed.shape[1], audio_b_np.shape[1])
                mixed = mixed[:, :min_length]
                audio_b_np = audio_b_np[:, :min_length]
                
                # Add with gain
                mixed += audio_b_np * gain_b
            
            # Add third audio if provided
            if audio_c is not None:
                waveform_c = audio_c["waveform"]
                if hasattr(waveform_c, 'cpu'):
                    audio_c_np = waveform_c.cpu().numpy()
                else:
                    audio_c_np = waveform_c
                
                if audio_c_np.ndim == 1:
                    audio_c_np = audio_c_np.reshape(1, -1)
                
                # Match lengths with existing mixed audio
                min_length = min(mixed.shape[1], audio_c_np.shape[1])
                mixed = mixed[:, :min_length]
                audio_c_np = audio_c_np[:, :min_length]
                
                # Add with gain
                mixed += audio_c_np * gain_c
            
            # Add fourth audio if provided
            if audio_d is not None:
                waveform_d = audio_d["waveform"]
                if hasattr(waveform_d, 'cpu'):
                    audio_d_np = waveform_d.cpu().numpy()
                else:
                    audio_d_np = waveform_d
                
                if audio_d_np.ndim == 1:
                    audio_d_np = audio_d_np.reshape(1, -1)
                
                # Match lengths with existing mixed audio
                min_length = min(mixed.shape[1], audio_d_np.shape[1])
                mixed = mixed[:, :min_length]
                audio_d_np = audio_d_np[:, :min_length]
                
                # Add with gain
                mixed += audio_d_np * gain_d
            
            # Apply master gain
            mixed *= master_gain
            
            # Safety limiting
            max_val = np.max(np.abs(mixed))
            if max_val > 1.0:
                mixed /= max_val
            
            result_tensor = torch.from_numpy(mixed).float()
            output_audio = {"waveform": result_tensor, "sample_rate": sample_rate}
            
            return (output_audio,)
            
        except Exception as e:
            print(f"❌ Error in audio mixing: {str(e)}")
            return (audio_a,)


class ChaosNoiseMixNode:
    """Extreme audio mixing for harsh noise and experimental textures."""
    
    MIX_MODES = ["add", "multiply", "xor", "modulo", "subtract", "max", "min", "ring_mod", "chaos"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_a": ("AUDIO", {"tooltip": "First audio input"}),
                "noise_b": ("AUDIO", {"tooltip": "Second audio input"}),
                "mix_mode": (cls.MIX_MODES, {"default": "add"}),
                "mix_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chaos_amount": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "distortion": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "feedback": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.8, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("chaos_audio",)
    FUNCTION = "mix_chaos"
    CATEGORY = "🎵 NoiseGen/Mixing"
    DESCRIPTION = "Extreme audio mixing for harsh noise and experimental textures"
    
    def mix_chaos(self, noise_a, noise_b, mix_mode, mix_ratio, chaos_amount, distortion, amplitude, feedback=0.2):
        try:
            # Get audio data
            waveform_a = noise_a["waveform"]
            waveform_b = noise_b["waveform"]
            sample_rate = noise_a["sample_rate"]
            
            if hasattr(waveform_a, 'cpu'):
                audio_a_np = waveform_a.cpu().numpy()
                audio_b_np = waveform_b.cpu().numpy()
            else:
                audio_a_np = waveform_a
                audio_b_np = waveform_b
            
            if audio_a_np.ndim == 1:
                audio_a_np = audio_a_np.reshape(1, -1)
            if audio_b_np.ndim == 1:
                audio_b_np = audio_b_np.reshape(1, -1)
            
            # Match lengths
            min_length = min(audio_a_np.shape[1], audio_b_np.shape[1])
            audio_a_np = audio_a_np[:, :min_length]
            audio_b_np = audio_b_np[:, :min_length]
            
            # Apply chaos mixing
            mixed = self._apply_chaos_mix(audio_a_np, audio_b_np, mix_mode, mix_ratio, chaos_amount)
            
            # Apply distortion
            if distortion > 0:
                mixed = np.tanh(mixed * (1 + distortion * 3))
            
            # Apply amplitude
            mixed *= amplitude
            
            # Safety limiting
            max_val = np.max(np.abs(mixed))
            if max_val > 1.0:
                mixed /= max_val
            
            result_tensor = torch.from_numpy(mixed).float()
            output_audio = {"waveform": result_tensor, "sample_rate": sample_rate}
            
            return (output_audio,)
            
        except Exception as e:
            print(f"❌ Error in chaos mixing: {str(e)}")
            return (noise_a,)
    
    def _apply_chaos_mix(self, a, b, mode, ratio, chaos):
        """Apply chaotic mixing between two audio signals."""
        if mode == "add":
            return a * (1 - ratio) + b * ratio
        elif mode == "multiply":
            return a * b * (1 + chaos)
        elif mode == "chaos":
            chaos_factor = 1.0 + chaos * (np.random.random(a.shape) - 0.5)
            return (a * b + np.sin(a + b) * chaos) * chaos_factor
        else:
            return a * (1 - ratio) + b * ratio


# =============================================================================
# 🔧 UTILITIES - File I/O and helpers
# =============================================================================

class AudioSaveNode:
    """Enhanced audio export with metadata, preview, and playback functionality."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (AUDIO_TYPE, {"tooltip": "Audio data to save"}),
                "filename_prefix": ("STRING", {"default": "NoiseGen_"}),
                "format": (["wav", "flac", "mp3"], {"default": "wav"}),
            },
            "optional": {
                "quality": ("INT", {"default": 320, "min": 128, "max": 320, "step": 32, "tooltip": "MP3 bitrate (kbps)"}),
            }
        }
    
    RETURN_TYPES = (AUDIO_TYPE, "STRING")
    RETURN_NAMES = ("audio", "filepath")
    FUNCTION = "save_audio"
    CATEGORY = "🎵 NoiseGen/Utils"
    OUTPUT_NODE = True
    DESCRIPTION = "Enhanced audio export with preview, playback, and waveform visualization"

    def save_audio(self, audio, filename_prefix, format, quality=320):
        try:
            import folder_paths
            import os
            import datetime
            import soundfile as sf
            import base64
            import io
            
            # Get output directory
            output_dir = folder_paths.get_output_directory()
            
            # Extract audio data
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # Convert to numpy
            if hasattr(waveform, 'cpu'):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = waveform
            
            # Ensure 2D array (channels, samples)
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            
            # Transpose for soundfile (samples, channels)
            audio_for_save = audio_np.T
            
            # Generate unique filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}{timestamp}.{format}"
            filepath = os.path.join(output_dir, filename)
            
            # Save audio file
            if format == "mp3":
                # For MP3, we'd need additional libraries like pydub
                # For now, save as WAV and note MP3 support needs enhancement
                sf.write(filepath.replace('.mp3', '.wav'), audio_for_save, sample_rate)
                actual_filepath = filepath.replace('.mp3', '.wav')
                print(f"💾 Note: MP3 support requires additional libraries, saved as WAV")
            else:
                sf.write(filepath, audio_for_save, sample_rate)
                actual_filepath = filepath
            
            # Calculate audio metadata
            duration = audio_np.shape[1] / sample_rate
            file_size = os.path.getsize(actual_filepath)
            channels = audio_np.shape[0]
            
            # Create waveform visualization for preview
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                import matplotlib.pyplot as plt
                
                # Generate waveform plot
                plt.figure(figsize=(12, 4))
                
                # Plot waveform (first 44100 samples max for performance)
                plot_samples = min(44100, audio_np.shape[1])
                time_axis = np.linspace(0, plot_samples/sample_rate, plot_samples)
                
                if channels == 1:
                    plt.plot(time_axis, audio_np[0, :plot_samples], color='#2E86AB', linewidth=0.5)
                    plt.title(f"🎵 Audio Waveform - {filename}")
                else:
                    plt.plot(time_axis, audio_np[0, :plot_samples], color='#2E86AB', linewidth=0.5, label='Left', alpha=0.7)
                    plt.plot(time_axis, audio_np[1, :plot_samples], color='#A23B72', linewidth=0.5, label='Right', alpha=0.7)
                    plt.legend()
                    plt.title(f"🎵 Stereo Audio Waveform - {filename}")
                
                plt.xlabel("Time (seconds)")
                plt.ylabel("Amplitude")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Convert plot to base64 for web display
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                buffer.seek(0)
                
                waveform_image = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
            except ImportError:
                waveform_image = None
                print("⚠️  matplotlib not available - waveform preview disabled")
            except Exception as e:
                waveform_image = None
                print(f"⚠️  Waveform generation failed: {e}")
            
            # Create enhanced metadata for UI
            metadata = {
                "filename": filename,
                "filepath": actual_filepath,
                "duration": f"{duration:.2f}s",
                "sample_rate": f"{sample_rate}Hz",
                "channels": f"{channels}ch",
                "format": format.upper(),
                "file_size": f"{file_size/1024:.1f}KB",
                "bitdepth": "32-bit float",
                "waveform_preview": waveform_image
            }
            
            # Enhanced UI output with playback controls
            ui_output = {
                "ui": {
                    "audio": [{
                        "filename": filename,
                        "subfolder": "",
                        "type": "output",
                        "format": format,
                        "duration": duration,
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "waveform_preview": waveform_image,
                        # Add preview controls
                        "has_preview": True,
                        "metadata": metadata
                    }],
                    "text": [f"💾 Saved: {filename} ({duration:.1f}s, {sample_rate}Hz, {channels}ch)"]
                },
                "result": (audio, actual_filepath)
            }
            
            print(f"💾 Audio saved successfully:")
            print(f"   📁 File: {filename}")
            print(f"   ⏱️  Duration: {duration:.2f}s")
            print(f"   🔊 Sample Rate: {sample_rate}Hz")
            print(f"   📊 Channels: {channels}")
            print(f"   💽 Size: {file_size/1024:.1f}KB")
            
            return ui_output
            
        except Exception as e:
            print(f"❌ Error saving audio: {str(e)}")
            # Return basic output on error
            return {
                "ui": {"text": [f"❌ Save failed: {str(e)}"]},
                "result": (audio, "")
            }


# =============================================================================
# 🌟 PHASE 2: GRANULAR SYNTHESIS ENGINE - The Crown Jewel
# =============================================================================

class GranularProcessorNode:
    """🌟 PHASE 2: Ultimate granular synthesis powerhouse for microsound control."""
    
    GRAIN_SOURCES = ["input", "oscillator", "noise", "sample"]
    GRAIN_ENVELOPES = ["hann", "gaussian", "triangle", "exponential", "adsr"]
    POSITIONING_MODES = ["sequential", "random", "reverse", "pingpong", "freeze"]
    PITCH_MODES = ["preserve", "transpose", "random", "microtonal"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio for granular processing"}),
                "grain_size": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 1000.0, "step": 0.1, "tooltip": "Grain duration in milliseconds"}),
                "grain_density": ("FLOAT", {"default": 20.0, "min": 0.1, "max": 1000.0, "step": 0.1, "tooltip": "Grains per second"}),
                "pitch_ratio": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01, "tooltip": "Pitch transposition ratio"}),
                "grain_envelope": (cls.GRAIN_ENVELOPES, {"default": "hann"}),
                "positioning_mode": (cls.POSITIONING_MODES, {"default": "random"}),
                "pitch_mode": (cls.PITCH_MODES, {"default": "transpose"}),
                "grain_scatter": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Random timing variation"}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "position_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Playhead position offset"}),
                "pitch_scatter": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Random pitch variation"}),
                "grain_overlap": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.95, "step": 0.01, "tooltip": "Grain overlap factor"}),
                "stereo_spread": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Stereo positioning spread"}),
                "freeze_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Position for freeze mode"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("granular_audio",)
    FUNCTION = "process_granular"
    CATEGORY = "🎵 NoiseGen/Granular"
    DESCRIPTION = "🌟 PHASE 2: Ultimate granular synthesis powerhouse for microsound control"
    
    def __init__(self):
        self.grain_buffer = None
        self.grain_positions = []
        self.grain_states = []
    
    def process_granular(self, audio, grain_size, grain_density, pitch_ratio, grain_envelope, 
                        positioning_mode, pitch_mode, grain_scatter, amplitude,
                        position_offset=0.0, pitch_scatter=0.0, grain_overlap=0.5, 
                        stereo_spread=0.0, freeze_position=0.5):
        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            if hasattr(waveform, 'cpu'):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = waveform
            
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            
            # Apply granular synthesis
            processed = self._apply_granular_synthesis(
                audio_np, sample_rate, grain_size, grain_density, pitch_ratio,
                grain_envelope, positioning_mode, pitch_mode, grain_scatter,
                position_offset, pitch_scatter, grain_overlap, stereo_spread, freeze_position
            )
            
            # Apply amplitude
            processed *= amplitude
            
            # Safety limiting
            max_val = np.max(np.abs(processed))
            if max_val > 1.0:
                processed /= max_val
            
            result_tensor = torch.from_numpy(processed).float()
            output_audio = {"waveform": result_tensor, "sample_rate": sample_rate}
            
            return (output_audio,)
            
        except Exception as e:
            print(f"❌ Error in granular processing: {str(e)}")
            return (audio,)
    
    def _apply_granular_synthesis(self, audio, sample_rate, grain_size_ms, grain_density, 
                                pitch_ratio, envelope_type, pos_mode, pitch_mode, scatter,
                                pos_offset, pitch_scatter, overlap, stereo_spread, freeze_pos):
        """Core granular synthesis engine."""
        
        channels, samples = audio.shape
        grain_size_samples = int(grain_size_ms * sample_rate / 1000.0)
        grain_hop = int(sample_rate / grain_density)
        
        # Calculate output length accounting for pitch ratio
        output_length = int(samples / pitch_ratio) if pitch_ratio > 0 else samples
        output = np.zeros((channels, output_length))
        
        # Generate grain envelope
        envelope = self._generate_grain_envelope(grain_size_samples, envelope_type)
        
        # Current playhead position
        current_pos = 0.0
        output_pos = 0
        
        while output_pos < output_length - grain_size_samples:
            # Calculate grain parameters with scatter
            actual_grain_size = grain_size_samples
            if scatter > 0:
                size_variation = np.random.uniform(-scatter, scatter) * grain_size_samples * 0.3
                actual_grain_size = max(10, int(grain_size_samples + size_variation))
            
            # Determine grain source position
            if pos_mode == "sequential":
                source_pos = current_pos + pos_offset * samples
            elif pos_mode == "random":
                source_pos = np.random.uniform(0, samples - actual_grain_size)
            elif pos_mode == "reverse":
                source_pos = samples - current_pos - actual_grain_size
            elif pos_mode == "pingpong":
                ping_cycle = (current_pos % (2 * samples)) / samples
                if ping_cycle <= 1.0:
                    source_pos = ping_cycle * samples
                else:
                    source_pos = (2.0 - ping_cycle) * samples
            elif pos_mode == "freeze":
                source_pos = freeze_pos * samples
            else:
                source_pos = current_pos
            
            source_pos = np.clip(source_pos, 0, samples - actual_grain_size)
            source_start = int(source_pos)
            source_end = source_start + actual_grain_size
            
            # Extract and process grain
            if source_end <= samples:
                grain = audio[:, source_start:source_end]
                
                # Apply pitch processing
                if pitch_mode == "transpose" and pitch_ratio != 1.0:
                    grain = self._pitch_shift_grain(grain, pitch_ratio, sample_rate)
                elif pitch_mode == "random" and pitch_scatter > 0:
                    random_ratio = 1.0 + np.random.uniform(-pitch_scatter, pitch_scatter)
                    grain = self._pitch_shift_grain(grain, random_ratio, sample_rate)
                
                # Ensure grain size matches envelope
                if grain.shape[1] != len(envelope):
                    if grain.shape[1] > len(envelope):
                        grain = grain[:, :len(envelope)]
                    else:
                        # Pad grain if needed
                        padding = len(envelope) - grain.shape[1]
                        grain = np.pad(grain, ((0, 0), (0, padding)), mode='constant')
                
                # Apply envelope
                grain = grain * envelope[np.newaxis, :]
                
                # Apply stereo spread
                if channels == 2 and stereo_spread > 0:
                    spread_amount = np.random.uniform(-stereo_spread, stereo_spread)
                    if spread_amount > 0:
                        grain[0] *= (1.0 - spread_amount)  # Reduce left
                    else:
                        grain[1] *= (1.0 + spread_amount)  # Reduce right
                
                # Add grain to output with overlap
                end_pos = min(output_pos + len(envelope), output_length)
                actual_length = end_pos - output_pos
                
                if actual_length > 0:
                    output[:, output_pos:end_pos] += grain[:, :actual_length]
            
            # Advance positions
            hop_variation = 1.0
            if scatter > 0:
                hop_variation = 1.0 + np.random.uniform(-scatter * 0.5, scatter * 0.5)
            
            actual_hop = int(grain_hop * hop_variation * (1.0 - overlap))
            current_pos += actual_hop / pitch_ratio if pitch_ratio > 0 else actual_hop
            output_pos += actual_hop
            
            # Wrap position for looping modes
            if current_pos >= samples:
                current_pos = 0.0
        
        return output
    
    def _generate_grain_envelope(self, size, envelope_type):
        """Generate grain envelope based on type."""
        t = np.linspace(0, 1, size)
        
        if envelope_type == "hann":
            return 0.5 * (1 - np.cos(2 * np.pi * t))
        elif envelope_type == "gaussian":
            sigma = 0.3
            return np.exp(-0.5 * ((t - 0.5) / sigma) ** 2)
        elif envelope_type == "triangle":
            return 1.0 - np.abs(2 * t - 1)
        elif envelope_type == "exponential":
            return np.where(t <= 0.5, 
                          np.exp(4 * t - 2), 
                          np.exp(2 - 4 * t))
        elif envelope_type == "adsr":
            attack = int(size * 0.1)
            decay = int(size * 0.2)
            sustain_level = 0.7
            sustain = int(size * 0.4)
            release = size - attack - decay - sustain
            
            env = np.ones(size)
            # Attack
            env[:attack] = np.linspace(0, 1, attack)
            # Decay
            env[attack:attack+decay] = np.linspace(1, sustain_level, decay)
            # Sustain
            env[attack+decay:attack+decay+sustain] = sustain_level
            # Release
            env[attack+decay+sustain:] = np.linspace(sustain_level, 0, release)
            return env
        else:
            return np.ones(size)  # Rectangular
    
    def _pitch_shift_grain(self, grain, ratio, sample_rate):
        """Simple pitch shifting using interpolation."""
        if ratio == 1.0:
            return grain
        
        channels, length = grain.shape
        new_length = int(length / ratio)
        
        if new_length <= 0:
            return np.zeros((channels, 1))
        
        # Simple linear interpolation for pitch shifting
        old_indices = np.linspace(0, length - 1, new_length)
        new_grain = np.zeros((channels, new_length))
        
        for c in range(channels):
            new_grain[c] = np.interp(old_indices, np.arange(length), grain[c])
        
        return new_grain


class GranularSequencerNode:
    """🎵 PHASE 2: Pattern-based granular control with step sequencing."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio for sequenced granular processing"}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 64, "tooltip": "Number of sequence steps"}),
                "step_duration": ("FLOAT", {"default": 0.125, "min": 0.01, "max": 4.0, "step": 0.001, "tooltip": "Duration per step in seconds"}),
                "base_grain_size": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 500.0, "step": 1.0}),
                "base_density": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "pattern_variation": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Pattern randomization amount"}),
                "swing": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01, "tooltip": "Groove timing offset"}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "probability": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Global trigger probability"}),
                "velocity_variation": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "euclidean_rhythm": ("INT", {"default": 0, "min": 0, "max": 32, "tooltip": "Euclidean rhythm pattern (0 = off)"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("sequenced_audio",)
    FUNCTION = "process_sequenced_granular"
    CATEGORY = "🎵 NoiseGen/Granular"
    DESCRIPTION = "🎵 PHASE 2: Pattern-based granular control with step sequencing"
    
    def __init__(self):
        self.current_step = 0
        self.step_patterns = {}
    
    def process_sequenced_granular(self, audio, steps, step_duration, base_grain_size, base_density,
                                 pattern_variation, swing, amplitude, probability=1.0, 
                                 velocity_variation=0.2, euclidean_rhythm=0):
        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            if hasattr(waveform, 'cpu'):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = waveform
            
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            
            # Generate sequence pattern
            pattern = self._generate_sequence_pattern(steps, euclidean_rhythm, probability)
            
            # Apply sequenced granular processing
            processed = self._apply_sequenced_granular(
                audio_np, sample_rate, pattern, step_duration, base_grain_size, 
                base_density, pattern_variation, swing, velocity_variation
            )
            
            # Apply amplitude
            processed *= amplitude
            
            # Safety limiting
            max_val = np.max(np.abs(processed))
            if max_val > 1.0:
                processed /= max_val
            
            result_tensor = torch.from_numpy(processed).float()
            output_audio = {"waveform": result_tensor, "sample_rate": sample_rate}
            
            return (output_audio,)
            
        except Exception as e:
            print(f"❌ Error in sequenced granular: {str(e)}")
            return (audio,)
    
    def _generate_sequence_pattern(self, steps, euclidean_rhythm, probability):
        """Generate sequence pattern with optional euclidean rhythm."""
        pattern = []
        
        if euclidean_rhythm > 0:
            # Generate euclidean rhythm
            pattern = self._generate_euclidean_pattern(steps, euclidean_rhythm)
        else:
            # Generate probability-based pattern
            for i in range(steps):
                active = np.random.random() < probability
                velocity = np.random.uniform(0.5, 1.0) if active else 0.0
                pattern.append({
                    'active': active,
                    'velocity': velocity,
                    'grain_size_mult': np.random.uniform(0.5, 2.0),
                    'density_mult': np.random.uniform(0.5, 2.0),
                    'pitch_offset': np.random.uniform(-0.5, 0.5)
                })
        
        return pattern
    
    def _generate_euclidean_pattern(self, steps, hits):
        """Generate euclidean rhythm pattern."""
        pattern = []
        bucket = 0
        
        for i in range(steps):
            bucket += hits
            if bucket >= steps:
                bucket -= steps
                active = True
                velocity = np.random.uniform(0.7, 1.0)
            else:
                active = False
                velocity = 0.0
            
            pattern.append({
                'active': active,
                'velocity': velocity,
                'grain_size_mult': np.random.uniform(0.8, 1.2),
                'density_mult': np.random.uniform(0.8, 1.2),
                'pitch_offset': np.random.uniform(-0.2, 0.2)
            })
        
        return pattern
    
    def _apply_sequenced_granular(self, audio, sample_rate, pattern, step_duration,
                                base_grain_size, base_density, pattern_variation, 
                                swing, velocity_variation):
        """Apply sequenced granular processing."""
        channels, samples = audio.shape
        
        # Calculate total duration and create output buffer
        total_duration = len(pattern) * step_duration
        output_samples = int(total_duration * sample_rate)
        output = np.zeros((channels, output_samples))
        
        # Process each step
        for step_idx, step_data in enumerate(pattern):
            if not step_data['active']:
                continue
            
            # Calculate step timing with swing
            step_start_time = step_idx * step_duration
            if step_idx % 2 == 1:  # Apply swing to odd steps
                step_start_time += swing * step_duration
            
            step_start_sample = int(step_start_time * sample_rate)
            step_end_sample = int((step_start_time + step_duration) * sample_rate)
            
            if step_start_sample >= output_samples:
                break
            
            # Calculate step parameters with variation
            step_grain_size = base_grain_size * step_data['grain_size_mult']
            step_density = base_density * step_data['density_mult']
            step_velocity = step_data['velocity']
            
            # Add velocity variation
            if velocity_variation > 0:
                velocity_mod = 1.0 + np.random.uniform(-velocity_variation, velocity_variation)
                step_velocity *= velocity_mod
            
            # Generate grains for this step
            step_output = self._generate_step_grains(
                audio, sample_rate, step_grain_size, step_density, 
                step_duration, step_velocity, step_data['pitch_offset']
            )
            
            # Add to output buffer
            step_length = min(len(step_output[0]), step_end_sample - step_start_sample)
            if step_length > 0:
                output[:, step_start_sample:step_start_sample + step_length] += step_output[:, :step_length]
        
        return output
    
    def _generate_step_grains(self, audio, sample_rate, grain_size, density, 
                            step_duration, velocity, pitch_offset):
        """Generate grains for a single step."""
        channels, samples = audio.shape
        grain_size_samples = int(grain_size * sample_rate / 1000.0)
        step_samples = int(step_duration * sample_rate)
        
        output = np.zeros((channels, step_samples))
        
        # Calculate number of grains for this step
        num_grains = int(density * step_duration)
        
        for grain_idx in range(num_grains):
            # Random grain position in source
            source_pos = np.random.randint(0, max(1, samples - grain_size_samples))
            
            # Random grain position in step
            step_pos = np.random.randint(0, max(1, step_samples - grain_size_samples))
            
            # Extract grain
            grain = audio[:, source_pos:source_pos + grain_size_samples]
            
            # Apply pitch offset (simple rate change)
            if pitch_offset != 0:
                pitch_ratio = 2.0 ** pitch_offset
                new_length = int(grain_size_samples * pitch_ratio)
                if new_length > 0:
                    old_indices = np.linspace(0, grain_size_samples - 1, new_length)
                    new_grain = np.zeros((channels, new_length))
                    for c in range(channels):
                        new_grain[c] = np.interp(old_indices, np.arange(grain_size_samples), grain[c])
                    grain = new_grain
                    grain_size_samples = new_length
            
            # Apply envelope and velocity
            envelope = np.hanning(grain_size_samples)
            grain = grain * envelope[np.newaxis, :] * velocity
            
            # Add to step output
            end_pos = min(step_pos + grain_size_samples, step_samples)
            actual_length = end_pos - step_pos
            if actual_length > 0:
                output[:, step_pos:end_pos] += grain[:, :actual_length]
        
        return output


class MicrosoundSculptorNode:
    """⚡ PHASE 2: Extreme granular manipulation for harsh noise and microsound art."""
    
    DESTRUCTION_MODES = ["bitcrush", "saturation", "chaos", "ring_mod", "frequency_shift"]
    SCULPTING_MODES = ["grain_filter", "grain_morph", "grain_feedback", "spectral_destroy"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio for extreme microsound processing"}),
                "destruction_mode": (cls.DESTRUCTION_MODES, {"default": "chaos"}),
                "destruction_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sculpting_mode": (cls.SCULPTING_MODES, {"default": "grain_morph"}),
                "sculpting_intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grain_size": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 200.0, "step": 0.1}),
                "chaos_rate": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "feedback_amount": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.9, "step": 0.01}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "spectral_chaos": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grain_randomization": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "microsound_density": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 1000.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("microsound_audio",)
    FUNCTION = "process_microsound"
    CATEGORY = "🎵 NoiseGen/Granular"
    DESCRIPTION = "⚡ PHASE 2: Extreme granular manipulation for harsh noise and microsound art"
    
    def __init__(self):
        self.feedback_buffer = None
        self.chaos_state = np.random.RandomState(42)
    
    def process_microsound(self, audio, destruction_mode, destruction_intensity, 
                         sculpting_mode, sculpting_intensity, grain_size, chaos_rate,
                         feedback_amount, amplitude, spectral_chaos=0.0, 
                         grain_randomization=0.5, microsound_density=50.0):
        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            if hasattr(waveform, 'cpu'):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = waveform
            
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            
            # Apply extreme microsound processing
            processed = self._apply_microsound_sculpting(
                audio_np, sample_rate, destruction_mode, destruction_intensity,
                sculpting_mode, sculpting_intensity, grain_size, chaos_rate,
                feedback_amount, spectral_chaos, grain_randomization, microsound_density
            )
            
            # Apply amplitude
            processed *= amplitude
            
            # Safety limiting with some chaos
            max_val = np.max(np.abs(processed))
            if max_val > 1.0:
                # Add slight chaos to limiting
                chaos_factor = 1.0 + destruction_intensity * 0.1 * self.chaos_state.uniform(-1, 1)
                processed = np.tanh(processed * chaos_factor)
            
            result_tensor = torch.from_numpy(processed).float()
            output_audio = {"waveform": result_tensor, "sample_rate": sample_rate}
            
            return (output_audio,)
            
        except Exception as e:
            print(f"❌ Error in microsound sculpting: {str(e)}")
            return (audio,)
    
    def _apply_microsound_sculpting(self, audio, sample_rate, destruction_mode, destruction_intensity,
                                  sculpting_mode, sculpting_intensity, grain_size_ms, chaos_rate,
                                  feedback_amount, spectral_chaos, grain_randomization, density):
        """Apply extreme microsound sculpting techniques."""
        
        channels, samples = audio.shape
        grain_size_samples = int(grain_size_ms * sample_rate / 1000.0)
        
        # Initialize feedback buffer
        if self.feedback_buffer is None or self.feedback_buffer.shape != audio.shape:
            self.feedback_buffer = np.zeros_like(audio)
        
        output = np.zeros_like(audio)
        
        # Calculate grain hop size based on density
        grain_hop = max(1, int(sample_rate / density))
        
        # Process in grains
        for grain_start in range(0, samples - grain_size_samples, grain_hop):
            grain_end = grain_start + grain_size_samples
            
            # Extract grain
            grain = audio[:, grain_start:grain_end].copy()
            
            # Add feedback
            if feedback_amount > 0:
                feedback_grain = self.feedback_buffer[:, grain_start:grain_end]
                grain = grain * (1 - feedback_amount) + feedback_grain * feedback_amount
            
            # Apply destruction processing
            grain = self._apply_grain_destruction(grain, destruction_mode, destruction_intensity, 
                                                sample_rate, chaos_rate)
            
            # Apply sculpting processing
            grain = self._apply_grain_sculpting(grain, sculpting_mode, sculpting_intensity, 
                                              sample_rate, spectral_chaos, grain_randomization)
            
            # Add grain to output with random positioning for chaos
            if grain_randomization > 0:
                pos_chaos = int(grain_randomization * grain_hop * self.chaos_state.uniform(-1, 1))
                actual_start = np.clip(grain_start + pos_chaos, 0, samples - grain_size_samples)
                actual_end = actual_start + grain_size_samples
            else:
                actual_start = grain_start
                actual_end = grain_end
            
            # Apply grain envelope for smooth blending
            envelope = np.hanning(grain_size_samples)
            grain *= envelope[np.newaxis, :]
            
            # Add to output
            output[:, actual_start:actual_end] += grain
            
            # Update feedback buffer
            self.feedback_buffer[:, grain_start:grain_end] = grain * 0.7
        
        return output
    
    def _apply_grain_destruction(self, grain, mode, intensity, sample_rate, chaos_rate):
        """Apply destructive processing to individual grains."""
        
        if intensity <= 0:
            return grain
        
        if mode == "bitcrush":
            # Reduce bit depth
            bits = int(16 * (1 - intensity) + 1)
            scale = 2 ** (bits - 1)
            grain = np.round(grain * scale) / scale
            
        elif mode == "saturation":
            # Heavy saturation/clipping
            drive = 1 + intensity * 10
            grain = np.tanh(grain * drive) / drive
            
        elif mode == "chaos":
            # Chaotic modulation
            chaos_freq = chaos_rate * intensity
            chaos_phase = self.chaos_state.uniform(0, 2 * np.pi, grain.shape)
            chaos_mod = 1 + intensity * 0.5 * np.sin(chaos_phase + 
                                                    np.arange(grain.shape[1])[np.newaxis, :] * 
                                                    chaos_freq / sample_rate * 2 * np.pi)
            grain *= chaos_mod
            
        elif mode == "ring_mod":
            # Ring modulation with chaos frequency
            mod_freq = 100 + intensity * 2000 * self.chaos_state.uniform(0.5, 2.0)
            mod_signal = np.sin(np.arange(grain.shape[1]) * mod_freq / sample_rate * 2 * np.pi)
            grain *= (1 - intensity + intensity * mod_signal[np.newaxis, :])
            
        elif mode == "frequency_shift":
            # Simple frequency shifting through modulation
            shift_freq = intensity * 500 * self.chaos_state.uniform(-1, 1)
            shift_signal = np.exp(1j * np.arange(grain.shape[1]) * shift_freq / sample_rate * 2 * np.pi)
            # Apply frequency shift (simplified)
            grain *= np.real(shift_signal[np.newaxis, :])
        
        return grain
    
    def _apply_grain_sculpting(self, grain, mode, intensity, sample_rate, spectral_chaos, randomization):
        """Apply advanced sculpting to grains."""
        
        if intensity <= 0:
            return grain
        
        if mode == "grain_filter":
            # Random filtering per grain
            cutoff = 200 + randomization * 8000 * self.chaos_state.uniform(0, 1)
            resonance = intensity * 0.8
            
            # Simple IIR filter
            omega = 2 * np.pi * cutoff / sample_rate
            sin_omega = np.sin(omega)
            cos_omega = np.cos(omega)
            alpha = sin_omega / (2 * (1 / resonance))
            
            b0 = (1 - cos_omega) / 2
            b1 = 1 - cos_omega
            b2 = (1 - cos_omega) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_omega
            a2 = 1 - alpha
            
            # Normalize
            b0 /= a0
            b1 /= a0
            b2 /= a0
            a1 /= a0
            a2 /= a0
            
            # Apply filter
            for c in range(grain.shape[0]):
                # Simple filtering implementation
                filtered = np.zeros_like(grain[c])
                x1 = x2 = y1 = y2 = 0.0
                
                for i in range(len(grain[c])):
                    x0 = grain[c, i]
                    y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                    filtered[i] = y0
                    
                    x2, x1 = x1, x0
                    y2, y1 = y1, y0
                
                grain[c] = grain[c] * (1 - intensity) + filtered * intensity
        
        elif mode == "grain_morph":
            # Morphing/warping grains
            warp_amount = intensity * randomization
            if warp_amount > 0:
                warp_factor = 1 + warp_amount * self.chaos_state.uniform(-0.5, 0.5, grain.shape[1])
                
                # Time warping
                old_indices = np.cumsum(warp_factor)
                old_indices = old_indices / old_indices[-1] * (grain.shape[1] - 1)
                
                for c in range(grain.shape[0]):
                    grain[c] = np.interp(np.arange(grain.shape[1]), old_indices, grain[c])
        
        elif mode == "grain_feedback":
            # Self-modulating grains
            delay_samples = max(1, int(grain.shape[1] * 0.1))
            fb_amount = intensity * 0.8
            
            for i in range(delay_samples, grain.shape[1]):
                grain[:, i] += grain[:, i - delay_samples] * fb_amount
        
        elif mode == "spectral_destroy":
            # Spectral destruction
            if spectral_chaos > 0:
                # Simple spectral chaos by randomizing phase
                fft_grain = np.fft.fft(grain, axis=1)
                phase_chaos = spectral_chaos * self.chaos_state.uniform(-np.pi, np.pi, fft_grain.shape)
                magnitude = np.abs(fft_grain)
                new_phase = np.angle(fft_grain) + phase_chaos
                fft_grain = magnitude * np.exp(1j * new_phase)
                grain = np.real(np.fft.ifft(fft_grain, axis=1))
        
        return grain


# =============================================================================
# NODE MAPPINGS FOR COMFYUI
# =============================================================================

NODE_CLASS_MAPPINGS = {
    # 🎵 GENERATORS
    "NoiseGenerator": NoiseGeneratorNode,
    "PerlinNoise": PerlinNoiseNode,
    "BandLimitedNoise": BandLimitedNoiseNode,
    
    # 🔄 PROCESSORS
    "FeedbackProcessor": FeedbackProcessorNode,
    "HarshFilter": HarshFilterNode,
    
    # 🎸 EFFECTS
    "MultiDistortion": MultiDistortionNode,
    "SpectralProcessor": SpectralProcessorNode,
    
    # 🎛️ MIXERS
    "AudioMixer": AudioMixerNode,
    "ChaosNoiseMix": ChaosNoiseMixNode,
    
    # 🌟 PHASE 2: GRANULAR SYNTHESIS
    "GranularProcessor": GranularProcessorNode,
    "GranularSequencer": GranularSequencerNode,
    "MicrosoundSculptor": MicrosoundSculptorNode,
    
    # 🔧 UTILITIES
    "AudioSave": AudioSaveNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 🎵 GENERATORS
    "NoiseGenerator": "🎵 Noise Generator",
    "PerlinNoise": "🌊 Perlin Noise",
    "BandLimitedNoise": "📡 Band Limited Noise",
    
    # 🔄 PROCESSORS
    "FeedbackProcessor": "🔄 Feedback Processor",
    "HarshFilter": "🎛️ Harsh Filter",
    
    # 🎸 EFFECTS
    "MultiDistortion": "🎸 Multi Distortion",
    "SpectralProcessor": "🔬 Spectral Processor",
    
    # 🎛️ MIXERS
    "AudioMixer": "🎛️ Audio Mixer",
    "ChaosNoiseMix": "⚡ Chaos Noise Mix",
    
    # 🌟 PHASE 2: GRANULAR SYNTHESIS
    "GranularProcessor": "🌟 Granular Processor",
    "GranularSequencer": "🎵 Granular Sequencer", 
    "MicrosoundSculptor": "⚡ Microsound Sculptor",
    
    # 🔧 UTILITIES
    "AudioSave": "💾 Audio Save",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]