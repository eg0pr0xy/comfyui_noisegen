"""
ComfyUI-NoiseGen - The Ultimate Merzbow Noise Machine

Advanced noise generation and audio processing nodes for ComfyUI.
Comprehensive suite for experimental audio, harsh noise, and industrial processing.

Author: eg0pr0xy
License: MIT
"""

import os
import numpy as np
import torch
import torchaudio
import scipy.signal
import scipy.integrate

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
    CATEGORY = "🎵 NoiseGen/Generate"
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
    CATEGORY = "🎵 NoiseGen/Generate"
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
    CATEGORY = "🎵 NoiseGen/Generate"
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
    CATEGORY = "🎵 NoiseGen/Process"
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
    CATEGORY = "🎵 NoiseGen/Process"
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
    CATEGORY = "🎵 NoiseGen/Process"
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
    CATEGORY = "🎵 NoiseGen/Process"
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
    CATEGORY = "🎵 NoiseGen/Utility"
    DESCRIPTION = "Professional audio mixer with gain and pan controls for up to 4 inputs"
    
    def mix_audio(self, audio_a, gain_a, pan_a, audio_b=None, gain_b=1.0, pan_b=0.0, 
                  audio_c=None, gain_c=1.0, pan_c=0.0, audio_d=None, gain_d=1.0, pan_d=0.0, master_gain=1.0):
        try:
            # Process input A
            waveform_a = audio_a["waveform"]
            sample_rate = audio_a["sample_rate"]
            
            if hasattr(waveform_a, 'cpu'):
                audio_a_np = waveform_a.cpu().numpy()
            else:
                audio_a_np = waveform_a
            
            # Fix tensor shape issues
            if audio_a_np.ndim == 1:
                audio_a_np = audio_a_np.reshape(1, -1)
            elif audio_a_np.ndim == 3:
                audio_a_np = audio_a_np.squeeze()
                if audio_a_np.ndim == 1:
                    audio_a_np = audio_a_np.reshape(1, -1)
            
            # Ensure stereo for mixing
            if audio_a_np.shape[0] == 1:
                audio_a_np = np.repeat(audio_a_np, 2, axis=0)
            
            # Apply gain and pan to A
            mixed = self._apply_gain_and_pan(audio_a_np, gain_a, pan_a)
            
            # Find the maximum length for proper buffer allocation
            max_length = mixed.shape[1]
            input_audios = []
            
            # Process additional inputs
            for audio_input, gain, pan in [(audio_b, gain_b, pan_b), (audio_c, gain_c, pan_c), (audio_d, gain_d, pan_d)]:
                if audio_input is not None:
                    waveform = audio_input["waveform"]
                    if hasattr(waveform, 'cpu'):
                        audio_np = waveform.cpu().numpy()
                    else:
                        audio_np = waveform
                    
                    # Fix tensor shape issues
                    if audio_np.ndim == 1:
                        audio_np = audio_np.reshape(1, -1)
                    elif audio_np.ndim == 3:
                        audio_np = audio_np.squeeze()
                        if audio_np.ndim == 1:
                            audio_np = audio_np.reshape(1, -1)
                    
                    # Ensure stereo
                    if audio_np.shape[0] == 1:
                        audio_np = np.repeat(audio_np, 2, axis=0)
                    
                    max_length = max(max_length, audio_np.shape[1])
                    input_audios.append((audio_np, gain, pan))
            
            # Resize mixed output to maximum length
            if mixed.shape[1] < max_length:
                padded_mixed = np.zeros((2, max_length))
                padded_mixed[:, :mixed.shape[1]] = mixed
                mixed = padded_mixed
            
            # Mix additional inputs
            for audio_np, gain, pan in input_audios:
                processed = self._apply_gain_and_pan(audio_np, gain, pan)
                
                # Ensure compatible lengths by padding or truncating
                if processed.shape[1] < max_length:
                    # Pad shorter audio
                    padded = np.zeros((2, max_length))
                    padded[:, :processed.shape[1]] = processed
                    processed = padded
                elif processed.shape[1] > max_length:
                    # Truncate longer audio
                    processed = processed[:, :max_length]
                
                # Add to mix
                mixed += processed
            
            # Apply master gain
            mixed *= master_gain
            
            # Safety limiting
            max_val = np.max(np.abs(mixed))
            if max_val > 1.0:
                mixed /= max_val
                print(f"⚠️  Audio limited: reduced by {max_val:.2f}x")
            
            # Convert back to tensor
            result_tensor = torch.from_numpy(mixed).float()
            output_audio = {"waveform": result_tensor, "sample_rate": sample_rate}
            
            return (output_audio,)
            
        except Exception as e:
            print(f"❌ Error in audio mixing: {str(e)}")
            # Return the first audio input on error
            return (audio_a,)
    
    def _apply_gain_and_pan(self, audio, gain, pan):
        """Apply gain and pan to the audio signal."""
        # Apply gain
        processed = audio * gain
        
        # Apply pan if stereo
        if processed.shape[0] == 2 and pan != 0:
            # Pan range: -1 (full left) to 1 (full right)
            pan_left = np.sqrt((1 - pan) / 2) if pan >= 0 else 1.0
            pan_right = np.sqrt((1 + pan) / 2) if pan <= 0 else 1.0
            
            processed[0] *= pan_left   # Left channel
            processed[1] *= pan_right  # Right channel
        
        return processed


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
    CATEGORY = "🎵 NoiseGen/Utility"
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
                "format": (["wav", "flac"], {"default": "wav"}),  # Removed MP3 for now since quality param only applies there
            }
        }
    
    RETURN_TYPES = (AUDIO_TYPE, "STRING")
    RETURN_NAMES = ("audio", "filepath")
    FUNCTION = "save_audio"
    CATEGORY = "🎵 NoiseGen/Utility"
    OUTPUT_NODE = True
    DESCRIPTION = "Enhanced audio export with preview, playback, and waveform visualization"

    def save_audio(self, audio, filename_prefix, format):
        try:
            import folder_paths
            import os
            import datetime
            import soundfile as sf
            import base64
            import io
            
            # Create dedicated audio output directory (same as original)
            if folder_paths:
                base_output_dir = folder_paths.get_output_directory()
                audio_dir = os.path.join(base_output_dir, "audio")
            else:
                audio_dir = os.path.join("outputs", "audio")
            
            os.makedirs(audio_dir, exist_ok=True)
            
            # Extract audio data
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # Convert to numpy and fix tensor shape issues
            if hasattr(waveform, 'cpu'):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = waveform
            
            # Fix any tensor shape issues - ensure 2D array (channels, samples)
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            elif audio_np.ndim == 3:
                # Handle tensors with extra dimensions like (samples, channels, 1)
                audio_np = audio_np.squeeze()  # Remove singleton dimensions
                if audio_np.ndim == 1:
                    audio_np = audio_np.reshape(1, -1)
                elif audio_np.shape[0] > audio_np.shape[1]:
                    # If shape is (samples, channels), transpose to (channels, samples)
                    audio_np = audio_np.T
            elif audio_np.shape[0] > audio_np.shape[1] and audio_np.shape[1] <= 2:
                # Transpose if likely (samples, channels) format
                audio_np = audio_np.T
            
            # Ensure 2D array (channels, samples)
            if audio_np.ndim != 2:
                raise ValueError(f"Cannot handle audio tensor with {audio_np.ndim} dimensions")
            
            # Transpose for soundfile (samples, channels)
            audio_for_save = audio_np.T
            
            # Generate unique filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}{timestamp}.{format}"
            filepath = os.path.join(audio_dir, filename)
            
            # Save audio file (WAV and FLAC are lossless - no quality setting needed)
            sf.write(filepath, audio_for_save, sample_rate)
            
            # Calculate audio metadata
            duration = audio_np.shape[1] / sample_rate
            file_size = os.path.getsize(filepath)
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
                "filepath": filepath,
                "duration": f"{duration:.2f}s",
                "sample_rate": f"{sample_rate}Hz",
                "channels": f"{channels}ch",
                "format": format.upper(),
                "file_size": f"{file_size/1024:.1f}KB",
                "bitdepth": "32-bit float",
                "compression": "Lossless" if format in ["wav", "flac"] else "Lossy",
                "waveform_preview": waveform_image
            }
            
            # Enhanced UI output with playback controls
            ui_output = {
                "ui": {
                    "audio": [{
                        "filename": filename,
                        "subfolder": "audio",  # Specify correct subfolder for UI
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
                    "text": [f"💾 Saved: audio/{filename} ({duration:.1f}s, {sample_rate}Hz, {channels}ch, Lossless)"]
                },
                "result": (audio, filepath)
            }
            
            print(f"💾 Audio saved successfully:")
            print(f"   📁 File: audio/{filename}")
            print(f"   ⏱️  Duration: {duration:.2f}s")
            print(f"   🔊 Sample Rate: {sample_rate}Hz")
            print(f"   📊 Channels: {channels}")
            print(f"   💽 Size: {file_size/1024:.1f}KB")
            print(f"   🎵 Quality: Lossless ({format.upper()})")
            
            return ui_output
            
        except Exception as e:
            print(f"❌ Error saving audio: {str(e)}")
            # Return basic output on error
            return {
                "ui": {"text": [f"❌ Save failed: {str(e)}"]},
                "result": (audio, "")
            }


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
    CATEGORY = "🎵 NoiseGen/Process"
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
            
            # Convert to numpy and ensure proper shape
            if hasattr(waveform, 'cpu'):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = waveform
            
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            elif audio_np.ndim == 3:  # Fix tensor shape issues
                audio_np = audio_np.squeeze()
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
            original_grain_size = grain_size_samples
            
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
            
            # Apply envelope and velocity - use actual grain size for envelope
            envelope = np.hanning(grain_size_samples)  # Use actual grain size
            grain = grain * envelope[np.newaxis, :] * velocity
            
            # Add to step output - ensure we don't exceed bounds
            end_pos = min(step_pos + grain_size_samples, step_samples)
            actual_length = end_pos - step_pos
            if actual_length > 0:
                # Ensure grain is cropped to fit
                grain_to_add = grain[:, :actual_length]
                output[:, step_pos:end_pos] += grain_to_add
        
        return output


class GranularProcessorNode:
    """🌟 PHASE 2: Ultimate granular synthesis engine with professional controls."""
    
    GRAIN_ENVELOPES = ["hann", "gaussian", "triangle", "exponential", "adsr"]
    POSITIONING_MODES = ["sequential", "random", "reverse", "pingpong", "freeze"]
    PITCH_MODES = ["transpose", "random", "lfo", "envelope"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio for granular processing"}),
                "grain_size": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 1000.0, "step": 0.1, "tooltip": "Grain size in milliseconds"}),
                "grain_density": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 1000.0, "step": 0.1, "tooltip": "Grains per second"}),
                "pitch_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.01, "tooltip": "Pitch transposition ratio"}),
                "grain_envelope": (cls.GRAIN_ENVELOPES, {"default": "hann"}),
                "positioning_mode": (cls.POSITIONING_MODES, {"default": "sequential"}),
                "pitch_mode": (cls.PITCH_MODES, {"default": "transpose"}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "grain_scatter": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "position_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pitch_scatter": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "grain_overlap": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.95, "step": 0.01}),
                "stereo_spread": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "freeze_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("granular_audio",)
    FUNCTION = "process_granular"
    CATEGORY = "🎵 NoiseGen/Process"
    DESCRIPTION = "🌟 PHASE 2: Ultimate granular synthesis engine with professional controls"
    
    def process_granular(self, audio, grain_size, grain_density, pitch_ratio, grain_envelope, 
                        positioning_mode, pitch_mode, amplitude, grain_scatter=0.0, 
                        position_offset=0.0, pitch_scatter=0.0, grain_overlap=0.5, 
                        stereo_spread=0.0, freeze_position=0.5):
        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # Convert to numpy and ensure proper shape
            if hasattr(waveform, 'cpu'):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = waveform
            
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            elif audio_np.ndim == 3:  # Fix tensor shape issues
                audio_np = audio_np.squeeze()
                if audio_np.ndim == 1:
                    audio_np = audio_np.reshape(1, -1)
            
            # Apply granular processing
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
        grain_hop = max(1, int(sample_rate / grain_density))
        
        # Calculate output length accounting for pitch ratio
        output_length = int(samples / pitch_ratio) if pitch_ratio > 0 else samples
        output = np.zeros((channels, output_length))
        
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
                
                # Generate envelope for actual grain size
                envelope = self._generate_grain_envelope(grain.shape[1], envelope_type)
                
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
                end_pos = min(output_pos + grain.shape[1], output_length)
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
            if attack > 0:
                env[:attack] = np.linspace(0, 1, attack)
            # Decay
            if decay > 0:
                env[attack:attack+decay] = np.linspace(1, sustain_level, decay)
            # Sustain
            if sustain > 0:
                env[attack+decay:attack+decay+sustain] = sustain_level
            # Release
            if release > 0:
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
    CATEGORY = "🎵 NoiseGen/Process"
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
            
            # Convert to numpy and ensure proper shape
            if hasattr(waveform, 'cpu'):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = waveform
            
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            elif audio_np.ndim == 3:  # Fix tensor shape issues
                audio_np = audio_np.squeeze()
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
# 🔬 PHASE 3: ADVANCED PROCESSING & ANALYSIS
# =============================================================================

class AudioAnalyzerNode:
    """🔬 PHASE 3: Comprehensive real-time audio analysis and feature extraction."""
    
    ANALYSIS_MODES = ["basic", "spectral", "dynamic", "comprehensive"]
    WINDOW_TYPES = ["hann", "hamming", "blackman", "rectangular"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio for analysis"}),
                "analysis_mode": (cls.ANALYSIS_MODES, {"default": "comprehensive"}),
                "frame_size": (["512", "1024", "2048", "4096"], {"default": "2048"}),
                "overlap_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.9, "step": 0.1}),
                "window_type": (cls.WINDOW_TYPES, {"default": "hann"}),
                "frequency_bands": ("INT", {"default": 8, "min": 4, "max": 32, "tooltip": "Number of frequency bands for analysis"}),
                "smoothing_factor": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.9, "step": 0.01, "tooltip": "Analysis smoothing (0=instant, 0.9=very smooth)"}),
            },
            "optional": {
                "threshold_db": ("FLOAT", {"default": -60.0, "min": -120.0, "max": 0.0, "step": 1.0}),
                "enable_visualization": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("audio", "analysis_report", "rms_level", "peak_level", "spectral_centroid", "dynamic_range")
    FUNCTION = "analyze_audio"
    CATEGORY = "🎵 NoiseGen/Utility"
    DESCRIPTION = "🔬 PHASE 3: Comprehensive real-time audio analysis and feature extraction"
    
    def __init__(self):
        self.previous_analysis = None
        self.analysis_history = []
        
    def analyze_audio(self, audio, analysis_mode, frame_size, overlap_factor, window_type, 
                     frequency_bands, smoothing_factor, threshold_db=-60.0, enable_visualization=False):
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
            elif audio_np.ndim == 3:
                audio_np = audio_np.squeeze()
                if audio_np.ndim == 1:
                    audio_np = audio_np.reshape(1, -1)
            
            # Perform comprehensive analysis
            analysis_results = self._perform_audio_analysis(
                audio_np, sample_rate, analysis_mode, int(frame_size), 
                overlap_factor, window_type, frequency_bands, smoothing_factor, threshold_db
            )
            
            # Generate analysis report
            report = self._generate_analysis_report(analysis_results, sample_rate)
            
            # Extract key metrics
            rms_level = float(analysis_results.get('rms_db', -60.0))
            peak_level = float(analysis_results.get('peak_db', -60.0))
            spectral_centroid = float(analysis_results.get('spectral_centroid', 1000.0))
            dynamic_range = float(analysis_results.get('dynamic_range', 0.0))
            
            # Optional visualization
            if enable_visualization:
                self._create_analysis_visualization(analysis_results, audio_np, sample_rate)
            
            return (audio, report, rms_level, peak_level, spectral_centroid, dynamic_range)
            
        except Exception as e:
            print(f"❌ Error in audio analysis: {str(e)}")
            fallback_report = f"Analysis Error: {str(e)}"
            return (audio, fallback_report, -60.0, -60.0, 1000.0, 0.0)
    
    def _perform_audio_analysis(self, audio, sample_rate, mode, frame_size, overlap_factor, 
                              window_type, freq_bands, smoothing, threshold_db):
        """Perform comprehensive audio analysis."""
        
        channels, samples = audio.shape
        
        # Initialize analysis results
        results = {
            'sample_rate': sample_rate,
            'duration': samples / sample_rate,
            'channels': channels,
            'samples': samples
        }
        
        # Basic level analysis
        if mode in ["basic", "dynamic", "comprehensive"]:
            results.update(self._analyze_levels(audio, threshold_db))
        
        # Spectral analysis  
        if mode in ["spectral", "comprehensive"]:
            results.update(self._analyze_spectrum(audio, sample_rate, frame_size, 
                                                overlap_factor, window_type, freq_bands))
        
        # Dynamic analysis
        if mode in ["dynamic", "comprehensive"]:
            results.update(self._analyze_dynamics(audio, sample_rate, frame_size))
        
        # Apply smoothing if previous analysis exists
        if self.previous_analysis and smoothing > 0:
            results = self._apply_smoothing(results, self.previous_analysis, smoothing)
        
        self.previous_analysis = results.copy()
        return results
    
    def _analyze_levels(self, audio, threshold_db):
        """Analyze audio levels and basic statistics."""
        
        # RMS analysis
        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(max(rms, 10**(threshold_db/20))) if rms > 0 else threshold_db
        
        # Peak analysis
        peak = np.max(np.abs(audio))
        peak_db = 20 * np.log10(max(peak, 10**(threshold_db/20))) if peak > 0 else threshold_db
        
        # Crest factor
        crest_factor = peak / max(rms, 1e-10)
        crest_factor_db = 20 * np.log10(crest_factor)
        
        # DC offset
        dc_offset = np.mean(audio)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(audio), axis=1))
        zcr = zero_crossings / audio.shape[1]
        
        return {
            'rms': rms,
            'rms_db': rms_db,
            'peak': peak,
            'peak_db': peak_db,
            'crest_factor': crest_factor,
            'crest_factor_db': crest_factor_db,
            'dc_offset': dc_offset,
            'zero_crossing_rate': float(np.mean(zcr))
        }
    
    def _analyze_spectrum(self, audio, sample_rate, frame_size, overlap_factor, window_type, freq_bands):
        """Analyze spectral characteristics."""
        
        # Create window
        if window_type == "hann":
            window = np.hanning(frame_size)
        elif window_type == "hamming":
            window = np.hamming(frame_size)
        elif window_type == "blackman":
            window = np.blackman(frame_size)
        else:
            window = np.ones(frame_size)  # rectangular
        
        # Calculate hop size
        hop_size = int(frame_size * (1 - overlap_factor))
        
        # Initialize spectral accumulators
        magnitude_spectrum = np.zeros(frame_size // 2 + 1)
        centroid_acc = 0.0
        rolloff_acc = 0.0
        flux_acc = 0.0
        frame_count = 0
        
        previous_magnitude = None
        
        # Process frames
        for start in range(0, audio.shape[1] - frame_size, hop_size):
            # Extract frame (use first channel for mono analysis)
            frame = audio[0, start:start + frame_size] * window
            
            # FFT
            fft = np.fft.fft(frame)
            magnitude = np.abs(fft[:frame_size // 2 + 1])
            magnitude_spectrum += magnitude
            
            # Spectral centroid
            freqs = np.fft.fftfreq(frame_size, 1/sample_rate)[:frame_size // 2 + 1]
            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                centroid_acc += centroid
            
            # Spectral rolloff (85% of energy)
            cumsum = np.cumsum(magnitude)
            if cumsum[-1] > 0:
                rolloff_threshold = 0.85 * cumsum[-1]
                rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
                if len(rolloff_idx) > 0:
                    rolloff_freq = freqs[rolloff_idx[0]]
                    rolloff_acc += rolloff_freq
            
            # Spectral flux
            if previous_magnitude is not None:
                flux = np.sum(np.diff(magnitude - previous_magnitude)**2)
                flux_acc += flux
            previous_magnitude = magnitude.copy()
            
            frame_count += 1
        
        # Average results
        if frame_count > 0:
            magnitude_spectrum /= frame_count
            centroid_acc /= frame_count
            rolloff_acc /= frame_count
            flux_acc /= max(1, frame_count - 1)
        
        # Frequency band analysis
        freqs = np.fft.fftfreq(frame_size, 1/sample_rate)[:frame_size // 2 + 1]
        band_energies = self._analyze_frequency_bands(magnitude_spectrum, freqs, freq_bands)
        
        return {
            'spectral_centroid': centroid_acc,
            'spectral_rolloff': rolloff_acc,
            'spectral_flux': flux_acc,
            'magnitude_spectrum': magnitude_spectrum,
            'frequency_bands': band_energies,
            'spectral_bandwidth': self._calculate_spectral_bandwidth(magnitude_spectrum, freqs, centroid_acc)
        }
    
    def _analyze_dynamics(self, audio, sample_rate, frame_size):
        """Analyze dynamic characteristics."""
        
        # Calculate frame-wise RMS for dynamic analysis
        hop_size = frame_size // 4
        rms_frames = []
        
        for start in range(0, audio.shape[1] - frame_size, hop_size):
            frame = audio[:, start:start + frame_size]
            frame_rms = np.sqrt(np.mean(frame**2))
            rms_frames.append(frame_rms)
        
        rms_frames = np.array(rms_frames)
        
        # Dynamic range
        if len(rms_frames) > 0:
            dynamic_range = np.max(rms_frames) - np.min(rms_frames[rms_frames > 0])
            dynamic_range_db = 20 * np.log10(dynamic_range) if dynamic_range > 0 else 0
        else:
            dynamic_range_db = 0
        
        # Loudness variation (standard deviation of RMS)
        loudness_variation = np.std(rms_frames) if len(rms_frames) > 0 else 0
        
        # Attack/decay analysis (simplified)
        if len(rms_frames) > 1:
            diff = np.diff(rms_frames)
            attack_rate = np.mean(diff[diff > 0]) if np.any(diff > 0) else 0
            decay_rate = np.mean(np.abs(diff[diff < 0])) if np.any(diff < 0) else 0
        else:
            attack_rate = decay_rate = 0
        
        return {
            'dynamic_range': dynamic_range_db,
            'loudness_variation': loudness_variation,
            'attack_rate': attack_rate,
            'decay_rate': decay_rate,
            'rms_frames': rms_frames
        }
    
    def _analyze_frequency_bands(self, magnitude_spectrum, freqs, num_bands):
        """Analyze energy in frequency bands."""
        
        # Create logarithmic frequency bands
        min_freq = max(20, freqs[1])  # Avoid DC
        max_freq = min(20000, freqs[-1])
        
        band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), num_bands + 1)
        band_energies = []
        
        for i in range(num_bands):
            # Find frequency indices for this band
            band_start = band_edges[i]
            band_end = band_edges[i + 1]
            
            band_indices = np.where((freqs >= band_start) & (freqs < band_end))[0]
            
            if len(band_indices) > 0:
                band_energy = np.sum(magnitude_spectrum[band_indices]**2)
                band_energies.append(band_energy)
            else:
                band_energies.append(0.0)
        
        return np.array(band_energies)
    
    def _calculate_spectral_bandwidth(self, magnitude, freqs, centroid):
        """Calculate spectral bandwidth around centroid."""
        
        if np.sum(magnitude) == 0:
            return 0.0
        
        # Weighted deviation from centroid
        deviation = (freqs - centroid) ** 2
        bandwidth = np.sqrt(np.sum(deviation * magnitude) / np.sum(magnitude))
        
        return bandwidth
    
    def _apply_smoothing(self, current, previous, smoothing):
        """Apply temporal smoothing to analysis results."""
        
        smoothed = {}
        
        for key, value in current.items():
            if isinstance(value, (int, float)):
                if key in previous:
                    # Apply exponential smoothing
                    smoothed[key] = smoothing * previous[key] + (1 - smoothing) * value
                else:
                    smoothed[key] = value
            else:
                # Non-numeric values pass through
                smoothed[key] = value
        
        return smoothed
    
    def _generate_analysis_report(self, results, sample_rate):
        """Generate human-readable analysis report."""
        
        report = f"""🔬 AUDIO ANALYSIS REPORT 🔬
        
📊 Basic Properties:
  • Duration: {results.get('duration', 0):.2f} seconds
  • Sample Rate: {sample_rate} Hz
  • Channels: {results.get('channels', 1)}
  • Samples: {results.get('samples', 0):,}

📈 Level Analysis:
  • RMS Level: {results.get('rms_db', -60):.1f} dB
  • Peak Level: {results.get('peak_db', -60):.1f} dB
  • Crest Factor: {results.get('crest_factor_db', 0):.1f} dB
  • DC Offset: {results.get('dc_offset', 0):.6f}
  • Zero Crossing Rate: {results.get('zero_crossing_rate', 0):.3f}

🌈 Spectral Analysis:
  • Spectral Centroid: {results.get('spectral_centroid', 1000):.0f} Hz
  • Spectral Rolloff: {results.get('spectral_rolloff', 5000):.0f} Hz
  • Spectral Bandwidth: {results.get('spectral_bandwidth', 1000):.0f} Hz
  • Spectral Flux: {results.get('spectral_flux', 0):.3f}

⚡ Dynamic Analysis:
  • Dynamic Range: {results.get('dynamic_range', 0):.1f} dB
  • Loudness Variation: {results.get('loudness_variation', 0):.3f}
  • Attack Rate: {results.get('attack_rate', 0):.3f}
  • Decay Rate: {results.get('decay_rate', 0):.3f}
"""
        
        # Add frequency band analysis if available
        if 'frequency_bands' in results:
            bands = results['frequency_bands']
            report += "\n🎼 Frequency Band Energy:\n"
            for i, energy in enumerate(bands):
                freq_range = f"Band {i+1}"
                report += f"  • {freq_range}: {energy:.3f}\n"
        
        return report
    
    def _create_analysis_visualization(self, results, audio, sample_rate):
        """Create analysis visualization (optional)."""
        try:
            import matplotlib.pyplot as plt
            
            # This would create plots - placeholder for now
            # Could be expanded to save visualization images
            print("📊 Analysis visualization enabled - plots would be generated here")
            
        except ImportError:
            print("⚠️ Matplotlib not available for visualization")


class SpectrumAnalyzerNode:
    """📊 PHASE 3: Real-time spectrum analyzer with FFT visualization and spectral gating."""
    
    DISPLAY_MODES = ["magnitude", "power", "log", "mel", "bark"]
    SCALE_TYPES = ["linear", "log", "mel", "bark"] 
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio for spectrum analysis"}),
                "fft_size": (["512", "1024", "2048", "4096", "8192"], {"default": "2048"}),
                "overlap_factor": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 0.95, "step": 0.05}),
                "window_type": (["hann", "hamming", "blackman", "kaiser", "rectangular"], {"default": "hann"}),
                "display_mode": (cls.DISPLAY_MODES, {"default": "magnitude"}),
                "frequency_scale": (cls.SCALE_TYPES, {"default": "log"}),
                "frequency_range_low": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 1000.0, "step": 1.0}),
                "frequency_range_high": ("FLOAT", {"default": 20000.0, "min": 1000.0, "max": 48000.0, "step": 10.0}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "gate_threshold": ("FLOAT", {"default": -60.0, "min": -120.0, "max": 0.0, "step": 1.0}),
                "spectral_gate": ("BOOLEAN", {"default": False}),
                "smoothing_factor": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.9, "step": 0.01}),
                "peak_tracking": ("BOOLEAN", {"default": True}),
                "save_spectrum": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("audio", "spectrum_report", "peak_frequency", "spectral_centroid", "spectral_energy")
    FUNCTION = "analyze_spectrum"
    CATEGORY = "🎵 NoiseGen/Utility"
    DESCRIPTION = "📊 PHASE 3: Real-time spectrum analyzer with FFT visualization and spectral gating"
    
    def __init__(self):
        self.previous_spectrum = None
        self.peak_history = []
        
    def analyze_spectrum(self, audio, fft_size, overlap_factor, window_type, display_mode, 
                        frequency_scale, frequency_range_low, frequency_range_high, amplitude,
                        gate_threshold=-60.0, spectral_gate=False, smoothing_factor=0.1, 
                        peak_tracking=True, save_spectrum=False):
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
            elif audio_np.ndim == 3:
                audio_np = audio_np.squeeze()
                if audio_np.ndim == 1:
                    audio_np = audio_np.reshape(1, -1)
            
            # Perform spectrum analysis
            spectrum_results = self._analyze_spectrum_detailed(
                audio_np, sample_rate, int(fft_size), overlap_factor, window_type,
                display_mode, frequency_scale, frequency_range_low, frequency_range_high,
                gate_threshold, spectral_gate, smoothing_factor, peak_tracking
            )
            
            # Apply spectral gating if enabled
            if spectral_gate:
                audio_np = self._apply_spectral_gating(audio_np, sample_rate, spectrum_results, gate_threshold)
                audio = {"waveform": torch.from_numpy(audio_np), "sample_rate": sample_rate}
            
            # Apply amplitude
            if amplitude != 1.0:
                audio_np *= amplitude
                audio = {"waveform": torch.from_numpy(audio_np), "sample_rate": sample_rate}
            
            # Generate spectrum report
            report = self._generate_spectrum_report(spectrum_results, sample_rate)
            
            # Extract key metrics
            peak_frequency = float(spectrum_results.get('peak_frequency', 1000.0))
            spectral_centroid = float(spectrum_results.get('spectral_centroid', 1000.0))
            spectral_energy = float(spectrum_results.get('total_energy', 0.0))
            
            # Optional spectrum saving
            if save_spectrum:
                self._save_spectrum_data(spectrum_results, sample_rate)
            
            return (audio, report, peak_frequency, spectral_centroid, spectral_energy)
            
        except Exception as e:
            print(f"❌ Error in spectrum analysis: {str(e)}")
            fallback_report = f"Spectrum Analysis Error: {str(e)}"
            return (audio, fallback_report, 1000.0, 1000.0, 0.0)
    
    def _analyze_spectrum_detailed(self, audio, sample_rate, fft_size, overlap_factor, window_type,
                                 display_mode, freq_scale, freq_low, freq_high, gate_threshold,
                                 spectral_gate, smoothing, peak_tracking):
        """Perform detailed spectrum analysis."""
        
        # Create analysis window
        if window_type == "hann":
            window = np.hanning(fft_size)
        elif window_type == "hamming":
            window = np.hamming(fft_size)
        elif window_type == "blackman":
            window = np.blackman(fft_size)
        elif window_type == "kaiser":
            window = np.kaiser(fft_size, 8.0)
        else:
            window = np.ones(fft_size)
        
        hop_size = int(fft_size * (1 - overlap_factor))
        
        # Initialize spectrum accumulator
        accumulated_spectrum = np.zeros(fft_size // 2 + 1)
        frame_count = 0
        
        # Frequency bins
        freqs = np.fft.fftfreq(fft_size, 1/sample_rate)[:fft_size // 2 + 1]
        
        # Find frequency range indices
        freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
        analysis_freqs = freqs[freq_mask]
        
        # Process audio frames
        peak_frequencies = []
        spectral_centroids = []
        
        for start in range(0, audio.shape[1] - fft_size, hop_size):
            # Extract and window frame
            frame = audio[0, start:start + fft_size] * window
            
            # Compute FFT
            fft_result = np.fft.fft(frame)
            magnitude = np.abs(fft_result[:fft_size // 2 + 1])
            
            # Apply display mode transformation
            if display_mode == "magnitude":
                spectrum = magnitude
            elif display_mode == "power":
                spectrum = magnitude ** 2
            elif display_mode == "log":
                spectrum = 20 * np.log10(np.maximum(magnitude, 1e-10))
            elif display_mode == "mel":
                spectrum = self._convert_to_mel_scale(magnitude, freqs, sample_rate)
            elif display_mode == "bark":
                spectrum = self._convert_to_bark_scale(magnitude, freqs, sample_rate)
            else:
                spectrum = magnitude
            
            # Accumulate spectrum
            accumulated_spectrum += spectrum
            
            # Peak frequency tracking
            if peak_tracking:
                analysis_spectrum = spectrum[freq_mask]
                if len(analysis_spectrum) > 0:
                    peak_idx = np.argmax(analysis_spectrum)
                    peak_freq = analysis_freqs[peak_idx]
                    peak_frequencies.append(peak_freq)
                    
                    # Spectral centroid
                    if np.sum(analysis_spectrum) > 0:
                        centroid = np.sum(analysis_freqs * analysis_spectrum) / np.sum(analysis_spectrum)
                        spectral_centroids.append(centroid)
            
            frame_count += 1
        
        # Average accumulated spectrum
        if frame_count > 0:
            accumulated_spectrum /= frame_count
        
        # Apply smoothing if previous spectrum exists
        if self.previous_spectrum is not None and smoothing > 0:
            accumulated_spectrum = (smoothing * self.previous_spectrum + 
                                  (1 - smoothing) * accumulated_spectrum)
        
        self.previous_spectrum = accumulated_spectrum.copy()
        
        # Calculate analysis results
        results = {
            'spectrum': accumulated_spectrum,
            'frequencies': freqs,
            'frame_count': frame_count,
            'fft_size': fft_size,
            'hop_size': hop_size,
            'window_type': window_type,
            'display_mode': display_mode,
            'frequency_scale': freq_scale
        }
        
        # Peak frequency statistics
        if peak_frequencies:
            results['peak_frequency'] = np.mean(peak_frequencies)
            results['peak_frequency_std'] = np.std(peak_frequencies)
            self.peak_history.extend(peak_frequencies[-10:])  # Keep recent history
            
            if len(self.peak_history) > 100:
                self.peak_history = self.peak_history[-100:]  # Limit history size
        else:
            results['peak_frequency'] = 0.0
            results['peak_frequency_std'] = 0.0
        
        # Spectral centroid statistics
        if spectral_centroids:
            results['spectral_centroid'] = np.mean(spectral_centroids)
            results['spectral_centroid_std'] = np.std(spectral_centroids)
        else:
            results['spectral_centroid'] = 0.0
            results['spectral_centroid_std'] = 0.0
        
        # Energy calculations
        total_energy = np.sum(accumulated_spectrum[freq_mask])
        results['total_energy'] = total_energy
        
        # Band energy analysis
        results['band_energies'] = self._calculate_band_energies(
            accumulated_spectrum, freqs, freq_low, freq_high
        )
        
        # Spectral features
        results.update(self._calculate_spectral_features(accumulated_spectrum, freqs))
        
        return results
    
    def _convert_to_mel_scale(self, magnitude, freqs, sample_rate):
        """Convert magnitude spectrum to mel scale."""
        
        # Mel filter bank (simplified)
        mel_points = np.linspace(0, 2595 * np.log10(1 + sample_rate/2 / 700), 40)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        
        # Create filter bank
        mel_spectrum = np.zeros(len(hz_points) - 2)
        
        for i in range(len(mel_spectrum)):
            # Find frequency indices for this mel band
            start_freq = hz_points[i]
            center_freq = hz_points[i + 1]
            end_freq = hz_points[i + 2]
            
            # Triangular filter
            mask = (freqs >= start_freq) & (freqs <= end_freq)
            if np.any(mask):
                # Triangular weighting
                weights = np.zeros_like(freqs)
                ascending_mask = (freqs >= start_freq) & (freqs <= center_freq)
                descending_mask = (freqs > center_freq) & (freqs <= end_freq)
                
                if np.any(ascending_mask):
                    weights[ascending_mask] = (freqs[ascending_mask] - start_freq) / (center_freq - start_freq)
                if np.any(descending_mask):
                    weights[descending_mask] = (end_freq - freqs[descending_mask]) / (end_freq - center_freq)
                
                mel_spectrum[i] = np.sum(magnitude * weights)
        
        return mel_spectrum
    
    def _convert_to_bark_scale(self, magnitude, freqs, sample_rate):
        """Convert magnitude spectrum to bark scale."""
        
        # Bark scale conversion (simplified)
        bark_freqs = 600 * np.sinh(np.linspace(0, 15, 24) / 4)
        bark_spectrum = np.zeros(len(bark_freqs) - 1)
        
        for i in range(len(bark_spectrum)):
            # Find frequency range for this bark band
            start_freq = bark_freqs[i]
            end_freq = bark_freqs[i + 1]
            
            mask = (freqs >= start_freq) & (freqs < end_freq)
            if np.any(mask):
                bark_spectrum[i] = np.mean(magnitude[mask])
        
        return bark_spectrum
    
    def _calculate_band_energies(self, spectrum, freqs, freq_low, freq_high):
        """Calculate energy in standard frequency bands."""
        
        # Define standard frequency bands
        bands = [
            ("Sub Bass", 20, 60),
            ("Bass", 60, 250),
            ("Low Mids", 250, 500),
            ("Mids", 500, 2000),
            ("High Mids", 2000, 4000),
            ("Presence", 4000, 8000),
            ("Brilliance", 8000, 20000)
        ]
        
        band_energies = {}
        
        for band_name, low, high in bands:
            # Only include bands within analysis range
            if high > freq_low and low < freq_high:
                mask = (freqs >= max(low, freq_low)) & (freqs <= min(high, freq_high))
                if np.any(mask):
                    energy = np.sum(spectrum[mask])
                    band_energies[band_name] = energy
                else:
                    band_energies[band_name] = 0.0
        
        return band_energies
    
    def _calculate_spectral_features(self, spectrum, freqs):
        """Calculate advanced spectral features."""
        
        if np.sum(spectrum) == 0:
            return {
                'spectral_rolloff': 0.0,
                'spectral_flux': 0.0,
                'spectral_flatness': 0.0,
                'spectral_spread': 0.0
            }
        
        # Spectral rolloff (85% energy point)
        cumsum = np.cumsum(spectrum)
        rolloff_threshold = 0.85 * cumsum[-1]
        rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
        rolloff_freq = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0.0
        
        # Spectral flux (change from previous frame)
        flux = 0.0
        if self.previous_spectrum is not None:
            flux = np.sum((spectrum - self.previous_spectrum) ** 2)
        
        # Spectral flatness (geometric mean / arithmetic mean)
        geometric_mean = np.exp(np.mean(np.log(np.maximum(spectrum, 1e-10))))
        arithmetic_mean = np.mean(spectrum)
        flatness = geometric_mean / max(arithmetic_mean, 1e-10)
        
        # Spectral spread (variance around centroid)
        centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / np.sum(spectrum))
        
        return {
            'spectral_rolloff': rolloff_freq,
            'spectral_flux': flux,
            'spectral_flatness': flatness,
            'spectral_spread': spread
        }
    
    def _apply_spectral_gating(self, audio, sample_rate, spectrum_results, threshold_db):
        """Apply spectral gating based on analysis."""
        
        # This is a simplified spectral gate
        # In practice, this would be more sophisticated
        fft_size = spectrum_results['fft_size']
        hop_size = spectrum_results['hop_size']
        
        # Convert threshold to linear
        threshold_linear = 10 ** (threshold_db / 20)
        
        gated_audio = audio.copy()
        
        for start in range(0, audio.shape[1] - fft_size, hop_size):
            frame = audio[:, start:start + fft_size]
            
            # Compute frame energy
            frame_energy = np.sqrt(np.mean(frame ** 2))
            
            # Apply gate
            if frame_energy < threshold_linear:
                gated_audio[:, start:start + fft_size] *= 0.1  # Reduce by 20dB
        
        return gated_audio
    
    def _generate_spectrum_report(self, results, sample_rate):
        """Generate detailed spectrum analysis report."""
        
        report = f"""📊 SPECTRUM ANALYSIS REPORT 📊

🔧 Analysis Settings:
  • FFT Size: {results.get('fft_size', 2048)}
  • Hop Size: {results.get('hop_size', 512)}
  • Window: {results.get('window_type', 'hann')}
  • Display Mode: {results.get('display_mode', 'magnitude')}
  • Frequency Scale: {results.get('frequency_scale', 'log')}
  • Frames Analyzed: {results.get('frame_count', 0)}

🎯 Peak Analysis:
  • Peak Frequency: {results.get('peak_frequency', 0):.1f} Hz
  • Peak Frequency Std: {results.get('peak_frequency_std', 0):.1f} Hz
  • Spectral Centroid: {results.get('spectral_centroid', 0):.1f} Hz
  • Centroid Std: {results.get('spectral_centroid_std', 0):.1f} Hz

📈 Spectral Features:
  • Spectral Rolloff: {results.get('spectral_rolloff', 0):.1f} Hz
  • Spectral Spread: {results.get('spectral_spread', 0):.1f} Hz
  • Spectral Flatness: {results.get('spectral_flatness', 0):.3f}
  • Spectral Flux: {results.get('spectral_flux', 0):.3f}
  • Total Energy: {results.get('total_energy', 0):.3f}
"""
        
        # Add band energies if available
        if 'band_energies' in results:
            report += "\n🎼 Frequency Band Energies:\n"
            for band_name, energy in results['band_energies'].items():
                report += f"  • {band_name}: {energy:.3f}\n"
        
        return report
    
    def _save_spectrum_data(self, results, sample_rate):
        """Save spectrum data to file (optional)."""
        try:
            # This would save spectrum data - placeholder for now
            print("💾 Spectrum data saved (feature placeholder)")
        except Exception as e:
            print(f"⚠️ Could not save spectrum data: {e}")


class TrueChaosNode:
    """🌀 PHASE 3: Mathematical chaos systems for ultimate unpredictable audio generation."""
    
    CHAOS_SYSTEMS = ["lorenz", "chua", "rossler", "henon", "duffing", "hybrid"]
    MODULATION_MODES = ["amplitude", "frequency", "phase", "ring_mod", "waveshape"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chaos_system": (cls.CHAOS_SYSTEMS, {"default": "lorenz"}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "sample_rate": ([8000, 16000, 22050, 44100, 48000, 96000], {"default": 44100}),
                "time_scale": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 1000.0, "step": 0.1, "tooltip": "Time scaling factor for chaos evolution"}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "modulation_mode": (cls.MODULATION_MODES, {"default": "amplitude"}),
                "chaos_intensity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "carrier_frequency": ("FLOAT", {"default": 440.0, "min": 10.0, "max": 10000.0, "step": 1.0}),
                "initial_x": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "initial_y": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "initial_z": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "chaos_parameter_a": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 0.01}),
                "chaos_parameter_b": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 0.01}),
                "chaos_parameter_c": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 0.01}),
                "channels": ([1, 2], {"default": 2}),
                "audio_input": ("AUDIO", {"tooltip": "Optional audio input for chaos modulation"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("chaos_audio", "chaos_report")
    FUNCTION = "generate_chaos"
    CATEGORY = "🎵 NoiseGen/Generate"
    DESCRIPTION = "🌀 PHASE 3: Mathematical chaos systems for ultimate unpredictable audio generation"
    
    def __init__(self):
        self.chaos_state = None
        self.attractor_history = []
        
    def generate_chaos(self, chaos_system, duration, sample_rate, time_scale, amplitude, 
                      modulation_mode, chaos_intensity, seed, carrier_frequency=440.0,
                      initial_x=1.0, initial_y=1.0, initial_z=1.0, 
                      chaos_parameter_a=0.0, chaos_parameter_b=0.0, chaos_parameter_c=0.0,
                      channels=2, audio_input=None):
        try:
            # Set random seed for reproducibility
            np.random.seed(seed)
            
            # Validate parameters
            duration = max(0.1, min(300.0, duration))
            sample_rate = int(sample_rate)
            num_samples = int(duration * sample_rate)
            
            # Generate chaos attractor trajectory
            chaos_trajectory = self._generate_chaos_system(
                chaos_system, duration, sample_rate, time_scale,
                initial_x, initial_y, initial_z,
                chaos_parameter_a, chaos_parameter_b, chaos_parameter_c
            )
            
            # Convert chaos trajectory to audio
            if audio_input is not None:
                # Modulate input audio with chaos
                chaos_audio = self._modulate_audio_with_chaos(
                    audio_input, chaos_trajectory, modulation_mode, chaos_intensity, amplitude
                )
            else:
                # Generate pure chaos audio
                chaos_audio = self._chaos_to_audio(
                    chaos_trajectory, sample_rate, carrier_frequency, 
                    modulation_mode, chaos_intensity, amplitude, channels
                )
            
            # Generate analysis report
            report = self._generate_chaos_report(
                chaos_system, chaos_trajectory, duration, sample_rate, time_scale
            )
            
            # Convert to ComfyUI format
            result_audio = {"waveform": torch.from_numpy(chaos_audio), "sample_rate": sample_rate}
            
            return (result_audio, report)
            
        except Exception as e:
            print(f"❌ Error in chaos generation: {str(e)}")
            # Fallback to white noise
            fallback = np.random.normal(0, 0.1, (channels, int(sample_rate * 0.1)))
            fallback_audio = {"waveform": torch.from_numpy(fallback), "sample_rate": sample_rate}
            fallback_report = f"Chaos Error: {str(e)} - Fallback to white noise"
            return (fallback_audio, fallback_report)
    
    def _generate_chaos_system(self, system_type, duration, sample_rate, time_scale,
                             init_x, init_y, init_z, param_a, param_b, param_c):
        """Generate chaos attractor trajectory using numerical integration."""
        
        # Validate inputs
        if duration <= 0 or sample_rate <= 0 or time_scale <= 0:
            raise ValueError(f"Invalid parameters: duration={duration}, sample_rate={sample_rate}, time_scale={time_scale}")
        
        # Calculate integration parameters
        dt = time_scale / sample_rate  # Time step
        num_steps = int(duration * sample_rate)
        
        if num_steps <= 0:
            raise ValueError(f"Invalid number of steps: {num_steps}")
        
        # Initialize state
        x, y, z = init_x, init_y, init_z
        trajectory = np.zeros((3, num_steps))
        
        # Integration using 4th order Runge-Kutta
        for i in range(num_steps):
            # Store current state
            trajectory[0, i] = x
            trajectory[1, i] = y
            trajectory[2, i] = z
            
            # Check for numerical instability
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                print(f"⚠️  Chaos system {system_type} became unstable at step {i}, reinitializing...")
                x, y, z = init_x + np.random.normal(0, 0.1), init_y + np.random.normal(0, 0.1), init_z + np.random.normal(0, 0.1)
            
            # Compute derivatives based on chaos system
            dx_dt, dy_dt, dz_dt = self._compute_chaos_derivatives(
                system_type, x, y, z, param_a, param_b, param_c
            )
            
            # Runge-Kutta 4th order integration
            k1_x, k1_y, k1_z = dx_dt * dt, dy_dt * dt, dz_dt * dt
            
            k2_dx, k2_dy, k2_dz = self._compute_chaos_derivatives(
                system_type, x + k1_x/2, y + k1_y/2, z + k1_z/2, param_a, param_b, param_c
            )
            k2_x, k2_y, k2_z = k2_dx * dt, k2_dy * dt, k2_dz * dt
            
            k3_dx, k3_dy, k3_dz = self._compute_chaos_derivatives(
                system_type, x + k2_x/2, y + k2_y/2, z + k2_z/2, param_a, param_b, param_c
            )
            k3_x, k3_y, k3_z = k3_dx * dt, k3_dy * dt, k3_dz * dt
            
            k4_dx, k4_dy, k4_dz = self._compute_chaos_derivatives(
                system_type, x + k3_x, y + k3_y, z + k3_z, param_a, param_b, param_c
            )
            k4_x, k4_y, k4_z = k4_dx * dt, k4_dy * dt, k4_dz * dt
            
            # Update state
            x += (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
            y += (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
            z += (k1_z + 2*k2_z + 2*k3_z + k4_z) / 6
        
        # Validate trajectory before returning
        if trajectory.shape != (3, num_steps):
            raise ValueError(f"Trajectory shape mismatch: expected (3, {num_steps}), got {trajectory.shape}")
        
        if not np.all(np.isfinite(trajectory)):
            print(f"⚠️  Warning: Trajectory contains non-finite values, applying safety clipping...")
            trajectory = np.clip(trajectory, -1000, 1000)
        
        return trajectory
    
    def _compute_chaos_derivatives(self, system_type, x, y, z, param_a, param_b, param_c):
        """Compute derivatives for different chaos systems."""
        
        if system_type == "lorenz":
            # Lorenz attractor
            sigma = 10.0 if param_a == 0.0 else param_a
            rho = 28.0 if param_b == 0.0 else param_b
            beta = 8.0/3.0 if param_c == 0.0 else param_c
            
            dx_dt = sigma * (y - x)
            dy_dt = x * (rho - z) - y
            dz_dt = x * y - beta * z
            
        elif system_type == "chua":
            # Chua's circuit
            alpha = 15.6 if param_a == 0.0 else param_a
            beta = 28.0 if param_b == 0.0 else param_b
            m0, m1 = -1.143, -0.714
            
            # Chua's nonlinearity
            h_x = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))
            
            dx_dt = alpha * (y - x - h_x)
            dy_dt = x - y + z
            dz_dt = -beta * y
            
        elif system_type == "rossler":
            # Rössler attractor
            a = 0.2 if param_a == 0.0 else param_a
            b = 0.2 if param_b == 0.0 else param_b
            c = 5.7 if param_c == 0.0 else param_c
            
            dx_dt = -y - z
            dy_dt = x + a * y
            dz_dt = b + z * (x - c)
            
        elif system_type == "henon":
            # Hénon map (discrete, adapted to continuous)
            a = 1.4 if param_a == 0.0 else param_a
            b = 0.3 if param_b == 0.0 else param_b
            
            dx_dt = 1 - a * x * x + y
            dy_dt = b * x
            dz_dt = 0.1 * (x - z)  # Add third dimension
            
        elif system_type == "duffing":
            # Duffing oscillator
            alpha = -1.0 if param_a == 0.0 else param_a
            beta = 1.0 if param_b == 0.0 else param_b
            gamma = 0.3 if param_c == 0.0 else param_c
            omega = 1.0
            
            dx_dt = y
            dy_dt = -alpha * x - beta * x**3 + gamma * np.cos(omega * z)
            dz_dt = omega  # Time variable
            
        elif system_type == "hybrid":
            # Hybrid system combining multiple attractors
            # Mix Lorenz and Rössler characteristics
            sigma = 10.0
            rho = 28.0
            beta = 8.0/3.0
            
            # Lorenz component
            lorenz_x = sigma * (y - x)
            lorenz_y = x * (rho - z) - y
            lorenz_z = x * y - beta * z
            
            # Rössler component  
            rossler_x = -y - z
            rossler_y = x + 0.2 * y
            rossler_z = 0.2 + z * (x - 5.7)
            
            # Mix based on current state
            mix_factor = np.sin(param_a * x) if param_a != 0 else 0.5
            
            dx_dt = mix_factor * lorenz_x + (1 - mix_factor) * rossler_x
            dy_dt = mix_factor * lorenz_y + (1 - mix_factor) * rossler_y
            dz_dt = mix_factor * lorenz_z + (1 - mix_factor) * rossler_z
            
        else:
            # Default to Lorenz
            dx_dt = 10.0 * (y - x)
            dy_dt = x * (28.0 - z) - y
            dz_dt = x * y - (8.0/3.0) * z
        
        return dx_dt, dy_dt, dz_dt
    
    def _chaos_to_audio(self, trajectory, sample_rate, carrier_freq, modulation_mode, 
                       chaos_intensity, amplitude, channels):
        """Convert chaos trajectory to audio signal."""
        
        # Unpack trajectory correctly - trajectory is shape (3, num_steps)
        x_signal, y_signal, z_signal = trajectory[0], trajectory[1], trajectory[2]
        num_samples = len(x_signal)
        
        # Normalize chaos signals to [-1, 1]
        x_norm = self._normalize_signal(x_signal)
        y_norm = self._normalize_signal(y_signal)
        z_norm = self._normalize_signal(z_signal)
        
        # Generate time array
        t = np.linspace(0, num_samples / sample_rate, num_samples)
        
        # Generate carrier signal
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        
        # Apply chaos modulation
        if modulation_mode == "amplitude":
            # Amplitude modulation
            chaos_mod = 1 + chaos_intensity * x_norm
            audio_signal = carrier * chaos_mod
            
        elif modulation_mode == "frequency":
            # Frequency modulation
            freq_deviation = carrier_freq * chaos_intensity
            instantaneous_freq = carrier_freq + freq_deviation * x_norm
            phase = np.cumsum(2 * np.pi * instantaneous_freq / sample_rate)
            audio_signal = np.sin(phase)
            
        elif modulation_mode == "phase":
            # Phase modulation
            phase_deviation = np.pi * chaos_intensity
            phase_mod = phase_deviation * x_norm
            audio_signal = np.sin(2 * np.pi * carrier_freq * t + phase_mod)
            
        elif modulation_mode == "ring_mod":
            # Ring modulation
            ring_carrier = np.sin(2 * np.pi * carrier_freq * 0.1 * t)  # Lower frequency ring
            chaos_mod = chaos_intensity * x_norm + (1 - chaos_intensity)
            audio_signal = carrier * ring_carrier * chaos_mod
            
        elif modulation_mode == "waveshape":
            # Waveshaping with chaos
            drive = 1 + chaos_intensity * 10
            shaped = np.tanh(carrier * drive * (1 + x_norm))
            audio_signal = shaped * (1 + 0.5 * chaos_intensity * y_norm)
            
        else:
            # Default amplitude modulation
            audio_signal = carrier * (1 + chaos_intensity * x_norm)
        
        # Apply amplitude scaling
        audio_signal *= amplitude
        
        # Create stereo output if requested
        if channels == 2:
            # Use different chaos dimensions for left/right
            left_channel = audio_signal
            
            # Right channel uses y-dimension for variation
            if modulation_mode == "amplitude":
                right_chaos_mod = 1 + chaos_intensity * y_norm
                right_channel = carrier * right_chaos_mod * amplitude
            else:
                # Similar processing but with y_norm
                right_channel = audio_signal * (1 + 0.3 * chaos_intensity * y_norm)
            
            audio_output = np.array([left_channel, right_channel])
        else:
            audio_output = audio_signal.reshape(1, -1)
        
        return audio_output
    
    def _modulate_audio_with_chaos(self, audio_input, chaos_trajectory, modulation_mode, 
                                 chaos_intensity, amplitude):
        """Modulate input audio with chaos trajectory."""
        
        waveform = audio_input["waveform"]
        sample_rate = audio_input["sample_rate"]
        
        if hasattr(waveform, 'cpu'):
            audio_np = waveform.cpu().numpy()
        else:
            audio_np = waveform
        
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)
        elif audio_np.ndim == 3:
            audio_np = audio_np.squeeze()
        
        # Unpack trajectory correctly - trajectory is shape (3, num_steps)
        x_signal, y_signal, z_signal = chaos_trajectory[0], chaos_trajectory[1], chaos_trajectory[2]
        
        # Match lengths by interpolation if needed
        if len(x_signal) != audio_np.shape[1]:
            try:
                from scipy import interpolate
                old_indices = np.linspace(0, len(x_signal) - 1, len(x_signal))
                new_indices = np.linspace(0, len(x_signal) - 1, audio_np.shape[1])
                
                interp_x = interpolate.interp1d(old_indices, x_signal, kind='linear')
                x_signal = interp_x(new_indices)
            except ImportError:
                # Fallback to simple numpy interpolation
                x_signal = np.interp(np.linspace(0, len(x_signal) - 1, audio_np.shape[1]), 
                                   np.linspace(0, len(x_signal) - 1, len(x_signal)), x_signal)
        
        # Normalize chaos signal
        x_norm = self._normalize_signal(x_signal)
        
        # Apply modulation
        modulated_audio = audio_np.copy()
        
        for channel in range(audio_np.shape[0]):
            if modulation_mode == "amplitude":
                chaos_mod = 1 + chaos_intensity * x_norm
                modulated_audio[channel] = audio_np[channel] * chaos_mod
                
            elif modulation_mode == "ring_mod":
                modulated_audio[channel] = audio_np[channel] * (1 + chaos_intensity * x_norm)
                
            elif modulation_mode == "waveshape":
                drive = 1 + chaos_intensity * 5
                modulated_audio[channel] = np.tanh(audio_np[channel] * drive * (1 + x_norm))
                
            else:
                # Default amplitude modulation
                modulated_audio[channel] = audio_np[channel] * (1 + chaos_intensity * x_norm)
        
        # Apply amplitude scaling
        modulated_audio *= amplitude
        
        return modulated_audio
    
    def _normalize_signal(self, signal):
        """Normalize signal to [-1, 1] range."""
        
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        
        if signal_max - signal_min == 0:
            return np.zeros_like(signal)
        
        normalized = 2 * (signal - signal_min) / (signal_max - signal_min) - 1
        return normalized
    
    def _generate_chaos_report(self, system_type, trajectory, duration, sample_rate, time_scale):
        """Generate chaos analysis report."""
        
        # Unpack trajectory correctly - trajectory is shape (3, num_steps)
        x_signal, y_signal, z_signal = trajectory[0], trajectory[1], trajectory[2]
        
        # Calculate statistics
        x_mean, x_std = np.mean(x_signal), np.std(x_signal)
        y_mean, y_std = np.mean(y_signal), np.std(y_signal)
        z_mean, z_std = np.mean(z_signal), np.std(z_signal)
        
        # Calculate correlation dimension (simplified)
        correlation_sum = np.corrcoef([x_signal, y_signal, z_signal])
        correlation_avg = np.mean(np.abs(correlation_sum[np.triu_indices_from(correlation_sum, k=1)]))
        
        # Lyapunov exponent estimate (simplified)
        lyapunov_estimate = self._estimate_lyapunov(x_signal)
        
        report = f"""🌀 CHAOS ANALYSIS REPORT 🌀

🔧 System Configuration:
  • Chaos System: {system_type.upper()}
  • Duration: {duration:.2f} seconds
  • Sample Rate: {sample_rate} Hz
  • Time Scale: {time_scale:.2f}
  • Total Samples: {len(x_signal):,}

📊 Attractor Statistics:
  • X-dimension: μ={x_mean:.3f}, σ={x_std:.3f}
  • Y-dimension: μ={y_mean:.3f}, σ={y_std:.3f}
  • Z-dimension: μ={z_mean:.3f}, σ={z_std:.3f}

🌪️ Chaos Characteristics:
  • Average Correlation: {correlation_avg:.3f}
  • Lyapunov Estimate: {lyapunov_estimate:.3f}
  • Phase Space Volume: {x_std * y_std * z_std:.3f}
  • Attractor Complexity: {"High" if lyapunov_estimate > 0.1 else "Medium" if lyapunov_estimate > 0.01 else "Low"}

⚡ Dynamics:
  • X-range: [{np.min(x_signal):.2f}, {np.max(x_signal):.2f}]
  • Y-range: [{np.min(y_signal):.2f}, {np.max(y_signal):.2f}]
  • Z-range: [{np.min(z_signal):.2f}, {np.max(z_signal):.2f}]
"""
        
        return report
    
    def _estimate_lyapunov(self, signal):
        """Estimate largest Lyapunov exponent (simplified method)."""
        
        try:
            # Use simple finite difference method
            n = len(signal)
            if n < 100:
                return 0.0
            
            # Calculate local divergence rates
            divergences = []
            
            for i in range(10, n - 10):
                # Find nearby points
                current_point = signal[i]
                nearby_indices = []
                
                for j in range(max(0, i - 50), min(n, i + 50)):
                    if abs(signal[j] - current_point) < 0.1 * np.std(signal) and abs(j - i) > 10:
                        nearby_indices.append(j)
                
                if len(nearby_indices) > 0:
                    # Calculate average divergence
                    distances = []
                    for j in nearby_indices:
                        if j + 10 < n and i + 10 < n:
                            initial_dist = abs(signal[j] - signal[i])
                            final_dist = abs(signal[j + 10] - signal[i + 10])
                            if initial_dist > 0:
                                divergence = np.log(final_dist / initial_dist) / 10
                                distances.append(divergence)
                    
                    if distances:
                        divergences.append(np.mean(distances))
            
            if divergences:
                return np.mean(divergences)
            else:
                return 0.0
                
        except Exception:
            return 0.0


class ModulationMatrixNode:
    """🎛️ PHASE 3: 8x8 parameter modulation matrix for complex control routing."""
    
    MODULATION_SOURCES = ["lfo_1", "lfo_2", "envelope", "audio_follower", "chaos", "random", "step_seq", "manual"]
    MODULATION_TARGETS = ["frequency", "amplitude", "filter_cutoff", "resonance", "delay_time", "feedback", "pan", "custom"]
    LFO_SHAPES = ["sine", "triangle", "sawtooth", "square", "random", "smooth_random"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio for modulation processing"}),
                "matrix_size": ([4, 6, 8], {"default": 8}),
                "global_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "update_rate": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 1000.0, "step": 1.0, "tooltip": "Modulation update rate in Hz"}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                # LFO Controls
                "lfo1_frequency": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "lfo1_shape": (cls.LFO_SHAPES, {"default": "sine"}),
                "lfo2_frequency": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "lfo2_shape": (cls.LFO_SHAPES, {"default": "triangle"}),
                
                # Envelope Controls
                "envelope_attack": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 10.0, "step": 0.001}),
                "envelope_decay": ("FLOAT", {"default": 0.2, "min": 0.001, "max": 10.0, "step": 0.001}),
                "envelope_sustain": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "envelope_release": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 10.0, "step": 0.001}),
                
                # Matrix Routing (simplified - in practice would be 8x8 matrix of connections)
                "modulation_amount_1": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "modulation_amount_2": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "modulation_amount_3": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "modulation_amount_4": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "modulation_amount_5": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "modulation_amount_6": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "modulation_amount_7": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "modulation_amount_8": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                
                # Audio Follower
                "follower_sensitivity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "follower_attack": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 1.0, "step": 0.001}),
                "follower_release": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.001}),
                
                # Random Source
                "random_rate": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "random_smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("modulated_audio", "matrix_report", "lfo1_value", "lfo2_value", "envelope_value", "follower_value")
    FUNCTION = "process_modulation_matrix"
    CATEGORY = "🎵 NoiseGen/Process"
    DESCRIPTION = "🎛️ PHASE 3: 8x8 parameter modulation matrix for complex control routing"
    
    def __init__(self):
        self.modulation_state = {
            'lfo1_phase': 0.0,
            'lfo2_phase': 0.0,
            'envelope_stage': 'idle',
            'envelope_time': 0.0,
            'envelope_value': 0.0,
            'follower_value': 0.0,
            'random_value': 0.0,
            'random_target': 0.0,
            'previous_audio_level': 0.0,
            'matrix_values': np.zeros((8, 8))
        }
        
    def process_modulation_matrix(self, audio, matrix_size, global_intensity, update_rate, amplitude,
                                lfo1_frequency=2.0, lfo1_shape="sine", lfo2_frequency=5.0, lfo2_shape="triangle",
                                envelope_attack=0.1, envelope_decay=0.2, envelope_sustain=0.5, envelope_release=1.0,
                                modulation_amount_1=0.0, modulation_amount_2=0.0, modulation_amount_3=0.0, modulation_amount_4=0.0,
                                modulation_amount_5=0.0, modulation_amount_6=0.0, modulation_amount_7=0.0, modulation_amount_8=0.0,
                                follower_sensitivity=1.0, follower_attack=0.01, follower_release=0.1,
                                random_rate=10.0, random_smoothing=0.5):
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
            elif audio_np.ndim == 3:
                audio_np = audio_np.squeeze()
                if audio_np.ndim == 1:
                    audio_np = audio_np.reshape(1, -1)
            
            # Create modulation amounts array
            modulation_amounts = np.array([
                modulation_amount_1, modulation_amount_2, modulation_amount_3, modulation_amount_4,
                modulation_amount_5, modulation_amount_6, modulation_amount_7, modulation_amount_8
            ])
            
            # Process modulation matrix
            modulated_audio, modulation_values = self._process_modulation_matrix(
                audio_np, sample_rate, matrix_size, global_intensity, update_rate,
                lfo1_frequency, lfo1_shape, lfo2_frequency, lfo2_shape,
                envelope_attack, envelope_decay, envelope_sustain, envelope_release,
                modulation_amounts, follower_sensitivity, follower_attack, follower_release,
                random_rate, random_smoothing
            )
            
            # Apply amplitude scaling
            modulated_audio *= amplitude
            
            # Generate modulation report
            report = self._generate_modulation_report(
                modulation_values, matrix_size, global_intensity, update_rate
            )
            
            # Extract current values for outputs
            lfo1_value = float(modulation_values.get('lfo1_current', 0.0))
            lfo2_value = float(modulation_values.get('lfo2_current', 0.0))
            envelope_value = float(modulation_values.get('envelope_current', 0.0))
            follower_value = float(modulation_values.get('follower_current', 0.0))
            
            # Convert back to ComfyUI format
            result_audio = {"waveform": torch.from_numpy(modulated_audio), "sample_rate": sample_rate}
            
            return (result_audio, report, lfo1_value, lfo2_value, envelope_value, follower_value)
            
        except Exception as e:
            print(f"❌ Error in modulation matrix: {str(e)}")
            fallback_report = f"Modulation Matrix Error: {str(e)}"
            return (audio, fallback_report, 0.0, 0.0, 0.0, 0.0)
    
    def _process_modulation_matrix(self, audio, sample_rate, matrix_size, global_intensity, update_rate,
                                 lfo1_freq, lfo1_shape, lfo2_freq, lfo2_shape,
                                 env_attack, env_decay, env_sustain, env_release,
                                 mod_amounts, follower_sens, follower_attack, follower_release,
                                 random_rate, random_smoothing):
        """Process the modulation matrix and apply to audio."""
        
        channels, num_samples = audio.shape
        modulated_audio = audio.copy()
        
        # Calculate update intervals
        update_interval = int(sample_rate / update_rate)
        
        # Initialize modulation values tracking
        modulation_values = {
            'lfo1_current': 0.0,
            'lfo2_current': 0.0,
            'envelope_current': 0.0,
            'follower_current': 0.0,
            'random_current': 0.0,
            'matrix_outputs': np.zeros(matrix_size)
        }
        
        # Process in blocks for efficiency
        for block_start in range(0, num_samples, update_interval):
            block_end = min(block_start + update_interval, num_samples)
            block_size = block_end - block_start
            
            # Current time
            current_time = block_start / sample_rate
            
            # Generate modulation sources
            mod_sources = self._generate_modulation_sources(
                current_time, block_size, sample_rate,
                lfo1_freq, lfo1_shape, lfo2_freq, lfo2_shape,
                env_attack, env_decay, env_sustain, env_release,
                audio[:, block_start:block_end], follower_sens, follower_attack, follower_release,
                random_rate, random_smoothing
            )
            
            # Update current values for output
            modulation_values['lfo1_current'] = mod_sources['lfo1']
            modulation_values['lfo2_current'] = mod_sources['lfo2']
            modulation_values['envelope_current'] = mod_sources['envelope']
            modulation_values['follower_current'] = mod_sources['audio_follower']
            modulation_values['random_current'] = mod_sources['random']
            
            # Apply modulation matrix (simplified 8x8 routing)
            matrix_outputs = self._apply_modulation_matrix(
                mod_sources, mod_amounts[:matrix_size], global_intensity
            )
            
            modulation_values['matrix_outputs'] = matrix_outputs
            
            # Apply modulations to audio
            modulated_audio = self._apply_modulations_to_audio(
                modulated_audio, matrix_outputs, block_start, block_end
            )
        
        return modulated_audio, modulation_values
    
    def _generate_modulation_sources(self, current_time, block_size, sample_rate,
                                   lfo1_freq, lfo1_shape, lfo2_freq, lfo2_shape,
                                   env_attack, env_decay, env_sustain, env_release,
                                   audio_block, follower_sens, follower_attack, follower_release,
                                   random_rate, random_smoothing):
        """Generate all modulation sources."""
        
        # LFO 1
        lfo1_value = self._generate_lfo(current_time, lfo1_freq, lfo1_shape, 'lfo1_phase')
        
        # LFO 2
        lfo2_value = self._generate_lfo(current_time, lfo2_freq, lfo2_shape, 'lfo2_phase')
        
        # Envelope (simplified ADSR)
        envelope_value = self._generate_envelope(
            current_time, env_attack, env_decay, env_sustain, env_release
        )
        
        # Audio follower
        follower_value = self._generate_audio_follower(
            audio_block, follower_sens, follower_attack, follower_release, sample_rate
        )
        
        # Random source
        random_value = self._generate_random_source(
            current_time, random_rate, random_smoothing
        )
        
        # Chaos source (simplified)
        chaos_value = np.sin(current_time * 7.3) * np.cos(current_time * 11.7)
        
        # Step sequencer (simplified 8-step)
        step_seq_value = self._generate_step_sequencer(current_time)
        
        # Manual value (static for now)
        manual_value = 0.5
        
        return {
            'lfo1': lfo1_value,
            'lfo2': lfo2_value,
            'envelope': envelope_value,
            'audio_follower': follower_value,
            'chaos': chaos_value,
            'random': random_value,
            'step_seq': step_seq_value,
            'manual': manual_value
        }
    
    def _generate_lfo(self, time, frequency, shape, phase_key):
        """Generate LFO output."""
        
        # Update phase
        self.modulation_state[phase_key] = (self.modulation_state[phase_key] + 
                                          frequency * 2 * np.pi / 100.0) % (2 * np.pi)
        phase = self.modulation_state[phase_key]
        
        if shape == "sine":
            return np.sin(phase)
        elif shape == "triangle":
            return 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1
        elif shape == "sawtooth":
            return 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))
        elif shape == "square":
            return 1.0 if np.sin(phase) >= 0 else -1.0
        elif shape == "random":
            return np.random.uniform(-1, 1) if phase < np.pi else self.modulation_state.get('lfo_random_hold', 0.0)
        elif shape == "smooth_random":
            # Smoothed random with interpolation
            if phase < 0.1:  # Update random target occasionally
                self.modulation_state['lfo_random_target'] = np.random.uniform(-1, 1)
            current = self.modulation_state.get('lfo_random_current', 0.0)
            target = self.modulation_state.get('lfo_random_target', 0.0)
            # Smooth interpolation
            self.modulation_state['lfo_random_current'] = current + (target - current) * 0.01
            return self.modulation_state['lfo_random_current']
        else:
            return np.sin(phase)
    
    def _generate_envelope(self, time, attack, decay, sustain, release):
        """Generate ADSR envelope."""
        
        # Simplified envelope - assumes trigger at t=0
        if time < attack:
            # Attack phase
            value = time / attack
            self.modulation_state['envelope_stage'] = 'attack'
        elif time < attack + decay:
            # Decay phase
            decay_progress = (time - attack) / decay
            value = 1.0 - decay_progress * (1.0 - sustain)
            self.modulation_state['envelope_stage'] = 'decay'
        elif time < attack + decay + 2.0:  # Sustain for 2 seconds
            # Sustain phase
            value = sustain
            self.modulation_state['envelope_stage'] = 'sustain'
        else:
            # Release phase
            release_time = time - (attack + decay + 2.0)
            if release_time < release:
                value = sustain * (1.0 - release_time / release)
                self.modulation_state['envelope_stage'] = 'release'
            else:
                value = 0.0
                self.modulation_state['envelope_stage'] = 'idle'
        
        self.modulation_state['envelope_value'] = value
        return value
    
    def _generate_audio_follower(self, audio_block, sensitivity, attack, release, sample_rate):
        """Generate audio follower output."""
        
        # Calculate RMS of audio block
        rms = np.sqrt(np.mean(audio_block ** 2))
        
        # Apply sensitivity
        target_level = rms * sensitivity
        
        # Smooth with attack/release
        current_level = self.modulation_state['follower_value']
        
        if target_level > current_level:
            # Attack
            rate = 1.0 - np.exp(-1.0 / (attack * sample_rate))
        else:
            # Release
            rate = 1.0 - np.exp(-1.0 / (release * sample_rate))
        
        new_level = current_level + rate * (target_level - current_level)
        self.modulation_state['follower_value'] = new_level
        
        return np.clip(new_level, 0.0, 1.0)
    
    def _generate_random_source(self, time, rate, smoothing):
        """Generate smooth random modulation."""
        
        # Update random target at specified rate
        update_interval = 1.0 / rate
        if time % update_interval < 0.01:  # Update when close to interval
            self.modulation_state['random_target'] = np.random.uniform(-1, 1)
        
        # Smooth towards target
        current = self.modulation_state['random_value']
        target = self.modulation_state.get('random_target', 0.0)
        
        smoothed = current + (target - current) * (1.0 - smoothing) * 0.1
        self.modulation_state['random_value'] = smoothed
        
        return smoothed
    
    def _generate_step_sequencer(self, time):
        """Generate simple 8-step sequencer."""
        
        # Fixed pattern for demo
        pattern = [1.0, 0.0, 0.5, 0.0, 0.8, 0.0, 0.2, 0.0]
        step_duration = 0.5  # seconds per step
        
        step_index = int(time / step_duration) % len(pattern)
        return pattern[step_index]
    
    def _apply_modulation_matrix(self, mod_sources, mod_amounts, global_intensity):
        """Apply 8x8 modulation matrix routing."""
        
        # Convert sources to array
        source_values = np.array([
            mod_sources['lfo1'],
            mod_sources['lfo2'],
            mod_sources['envelope'],
            mod_sources['audio_follower'],
            mod_sources['chaos'],
            mod_sources['random'],
            mod_sources['step_seq'],
            mod_sources['manual']
        ])
        
        # Apply modulation amounts (simplified linear routing)
        matrix_outputs = source_values * mod_amounts * global_intensity
        
        return matrix_outputs
    
    def _apply_modulations_to_audio(self, audio, matrix_outputs, block_start, block_end):
        """Apply matrix outputs to audio processing."""
        
        # This is a simplified version - in practice would route to specific parameters
        # For now, apply as amplitude, filter, and effects modulations
        
        # Amplitude modulation
        amp_mod = 1.0 + matrix_outputs[0] * 0.5
        audio[:, block_start:block_end] *= amp_mod
        
        # Simple filter-like effect using matrix outputs
        if len(matrix_outputs) > 1:
            filter_mod = matrix_outputs[1]
            # Simple high-pass/low-pass effect
            if filter_mod > 0:
                # Emphasize higher frequencies
                audio[:, block_start:block_end] = self._simple_highpass(audio[:, block_start:block_end], filter_mod)
            elif filter_mod < 0:
                # Emphasize lower frequencies  
                audio[:, block_start:block_end] = self._simple_lowpass(audio[:, block_start:block_end], abs(filter_mod))
        
        return audio
    
    def _simple_highpass(self, audio_block, intensity):
        """Simple highpass filter effect."""
        # Very basic first-order highpass
        alpha = intensity * 0.1
        filtered = audio_block.copy()
        for channel in range(audio_block.shape[0]):
            for i in range(1, audio_block.shape[1]):
                filtered[channel, i] = alpha * (filtered[channel, i-1] + audio_block[channel, i] - audio_block[channel, i-1])
        return filtered
    
    def _simple_lowpass(self, audio_block, intensity):
        """Simple lowpass filter effect."""
        # Very basic first-order lowpass
        alpha = 1.0 - intensity * 0.1
        filtered = audio_block.copy()
        for channel in range(audio_block.shape[0]):
            for i in range(1, audio_block.shape[1]):
                filtered[channel, i] = alpha * filtered[channel, i-1] + (1 - alpha) * audio_block[channel, i]
        return filtered
    
    def _generate_modulation_report(self, mod_values, matrix_size, global_intensity, update_rate):
        """Generate modulation matrix report."""
        
        report = f"""🎛️ MODULATION MATRIX REPORT 🎛️

🔧 Matrix Configuration:
  • Matrix Size: {matrix_size}x{matrix_size}
  • Global Intensity: {global_intensity:.2f}
  • Update Rate: {update_rate:.1f} Hz

📊 Current Source Values:
  • LFO 1: {mod_values.get('lfo1_current', 0):.3f}
  • LFO 2: {mod_values.get('lfo2_current', 0):.3f}
  • Envelope: {mod_values.get('envelope_current', 0):.3f}
  • Audio Follower: {mod_values.get('follower_current', 0):.3f}
  • Random: {mod_values.get('random_current', 0):.3f}

🎯 Matrix Outputs:"""
        
        matrix_outputs = mod_values.get('matrix_outputs', np.zeros(matrix_size))
        for i, output in enumerate(matrix_outputs):
            report += f"\n  • Output {i+1}: {output:.3f}"
        
        return report


class ConvolutionReverbNode:
    """🏛️ PHASE 3: Advanced convolution reverb with impulse response processing."""
    
    REVERB_TYPES = ["hall", "room", "cathedral", "plate", "spring", "chamber", "ambient", "custom"]
    CONVOLUTION_MODES = ["fft", "overlap_add", "uniform_partitioned", "adaptive"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio for reverb processing"}),
                "reverb_type": (cls.REVERB_TYPES, {"default": "hall"}),
                "convolution_mode": (cls.CONVOLUTION_MODES, {"default": "fft"}),
                "wet_dry_mix": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reverb_time": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Reverb decay time in seconds"}),
                "pre_delay": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.001, "tooltip": "Pre-delay before reverb in seconds"}),
                "amplitude": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                # Impulse Response Controls
                "ir_length": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 20.0, "step": 0.1, "tooltip": "Impulse response length in seconds"}),
                "early_reflections": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "late_reflections": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "room_size": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
                "damping": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "diffusion": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # Modulation
                "modulation_rate": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "modulation_depth": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.001}),
                
                # Filtering
                "high_cut": ("FLOAT", {"default": 8000.0, "min": 1000.0, "max": 20000.0, "step": 10.0}),
                "low_cut": ("FLOAT", {"default": 100.0, "min": 20.0, "max": 1000.0, "step": 1.0}),
                
                # Advanced
                "stereo_width": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "freeze": ("BOOLEAN", {"default": False, "tooltip": "Freeze reverb tail"}),
                "reverse": ("BOOLEAN", {"default": False, "tooltip": "Reverse impulse response"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("reverb_audio", "reverb_report")
    FUNCTION = "process_convolution_reverb"
    CATEGORY = "🎵 NoiseGen/Process"
    DESCRIPTION = "🏛️ PHASE 3: Advanced convolution reverb with impulse response processing"
    
    def __init__(self):
        self.impulse_responses = {}
        self.convolution_buffer = None
        self.modulation_phase = 0.0
        
    def process_convolution_reverb(self, audio, reverb_type, convolution_mode, wet_dry_mix, reverb_time, 
                                 pre_delay, amplitude, ir_length=4.0, early_reflections=0.3, 
                                 late_reflections=0.7, room_size=0.5, damping=0.3, diffusion=0.6,
                                 modulation_rate=0.5, modulation_depth=0.05, high_cut=8000.0, low_cut=100.0,
                                 stereo_width=1.0, freeze=False, reverse=False):
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
            elif audio_np.ndim == 3:
                audio_np = audio_np.squeeze()
                if audio_np.ndim == 1:
                    audio_np = audio_np.reshape(1, -1)
            
            # Generate or retrieve impulse response
            impulse_response = self._generate_impulse_response(
                reverb_type, sample_rate, ir_length, reverb_time, room_size, 
                damping, diffusion, early_reflections, late_reflections, reverse
            )
            
            # Apply pre-delay
            if pre_delay > 0:
                delay_samples = int(pre_delay * sample_rate)
                audio_np = self._apply_pre_delay(audio_np, delay_samples)
            
            # Apply filtering
            filtered_audio = self._apply_filtering(audio_np, sample_rate, low_cut, high_cut)
            
            # Apply convolution reverb
            reverb_audio = self._apply_convolution(
                filtered_audio, impulse_response, sample_rate, convolution_mode,
                modulation_rate, modulation_depth, freeze
            )
            
            # Mix wet/dry
            mixed_audio = self._mix_wet_dry(audio_np, reverb_audio, wet_dry_mix)
            
            # Apply stereo width
            if mixed_audio.shape[0] == 2 and stereo_width != 1.0:
                mixed_audio = self._apply_stereo_width(mixed_audio, stereo_width)
            
            # Apply amplitude scaling
            mixed_audio *= amplitude
            
            # Generate report
            report = self._generate_reverb_report(
                reverb_type, convolution_mode, wet_dry_mix, reverb_time, 
                ir_length, sample_rate, audio_np.shape
            )
            
            # Convert back to ComfyUI format
            result_audio = {"waveform": torch.from_numpy(mixed_audio), "sample_rate": sample_rate}
            
            return (result_audio, report)
            
        except Exception as e:
            print(f"❌ Error in convolution reverb: {str(e)}")
            fallback_report = f"Convolution Reverb Error: {str(e)}"
            return (audio, fallback_report)
    
    def _generate_impulse_response(self, reverb_type, sample_rate, ir_length, reverb_time, 
                                 room_size, damping, diffusion, early_refl, late_refl, reverse):
        """Generate impulse response based on reverb type and parameters."""
        
        # Cache key for impulse response
        cache_key = f"{reverb_type}_{sample_rate}_{ir_length}_{reverb_time}_{room_size}_{damping}_{diffusion}_{reverse}"
        
        if cache_key in self.impulse_responses:
            return self.impulse_responses[cache_key]
        
        ir_samples = int(ir_length * sample_rate)
        
        if reverb_type == "hall":
            impulse_response = self._generate_hall_ir(
                ir_samples, sample_rate, reverb_time, room_size, damping, diffusion, early_refl, late_refl
            )
        elif reverb_type == "room":
            impulse_response = self._generate_room_ir(
                ir_samples, sample_rate, reverb_time, room_size, damping, diffusion
            )
        elif reverb_type == "cathedral":
            impulse_response = self._generate_cathedral_ir(
                ir_samples, sample_rate, reverb_time, room_size, damping
            )
        elif reverb_type == "plate":
            impulse_response = self._generate_plate_ir(
                ir_samples, sample_rate, reverb_time, damping, diffusion
            )
        elif reverb_type == "spring":
            impulse_response = self._generate_spring_ir(
                ir_samples, sample_rate, reverb_time, damping
            )
        elif reverb_type == "chamber":
            impulse_response = self._generate_chamber_ir(
                ir_samples, sample_rate, reverb_time, room_size, damping, diffusion
            )
        elif reverb_type == "ambient":
            impulse_response = self._generate_ambient_ir(
                ir_samples, sample_rate, reverb_time, diffusion
            )
        else:
            # Default hall
            impulse_response = self._generate_hall_ir(
                ir_samples, sample_rate, reverb_time, room_size, damping, diffusion, early_refl, late_refl
            )
        
        # Apply reverse if requested
        if reverse:
            impulse_response = impulse_response[:, ::-1]
        
        # Cache the impulse response
        self.impulse_responses[cache_key] = impulse_response
        
        return impulse_response
    
    def _generate_hall_ir(self, ir_samples, sample_rate, reverb_time, room_size, damping, diffusion, early_refl, late_refl):
        """Generate concert hall impulse response."""
        
        # Create stereo impulse response
        impulse_response = np.zeros((2, ir_samples))
        
        # Early reflections (first 80ms)
        early_duration = min(int(0.08 * sample_rate), ir_samples // 4)
        
        # Generate early reflections with delays based on room geometry
        reflection_delays = [0.01, 0.023, 0.035, 0.047, 0.059, 0.071]  # seconds
        reflection_gains = [0.8, 0.6, 0.5, 0.4, 0.35, 0.3]
        
        for delay, gain in zip(reflection_delays, reflection_gains):
            delay_samples = int(delay * sample_rate * room_size)
            if delay_samples < early_duration:
                # Add slight stereo spread
                impulse_response[0, delay_samples] += gain * early_refl * (1 + diffusion * 0.1)
                impulse_response[1, delay_samples + 1] += gain * early_refl * (1 - diffusion * 0.1)
        
        # Late reflections (exponential decay)
        for i in range(early_duration, ir_samples):
            time = i / sample_rate
            
            # Exponential decay
            decay_factor = np.exp(-time * 6.91 / reverb_time)  # -60dB decay
            
            # Add random reflections with frequency-dependent damping
            for channel in range(2):
                # Noise-based diffuse field
                noise = np.random.normal(0, 1) * decay_factor * late_refl
                
                # Frequency-dependent damping (high-frequency rolloff)
                if i > early_duration + 1:
                    # Simple high-frequency damping
                    noise = noise * (1 - damping) + impulse_response[channel, i-1] * damping * 0.3
                
                # Stereo decorrelation
                stereo_factor = 1 + (channel * 2 - 1) * diffusion * 0.2
                impulse_response[channel, i] = noise * stereo_factor
        
        return impulse_response
    
    def _generate_room_ir(self, ir_samples, sample_rate, reverb_time, room_size, damping, diffusion):
        """Generate small room impulse response."""
        
        impulse_response = np.zeros((2, ir_samples))
        
        # Room dimensions affect early reflection pattern
        wall_delays = [0.005, 0.012, 0.018, 0.025] * room_size  # seconds
        
        for i, delay in enumerate(wall_delays):
            delay_samples = int(delay * sample_rate)
            if delay_samples < ir_samples:
                gain = 0.7 - i * 0.15  # Decreasing gain
                impulse_response[0, delay_samples] += gain * (1 + diffusion * 0.1)
                impulse_response[1, delay_samples + 2] += gain * (1 - diffusion * 0.1)
        
        # Shorter, denser late reflections
        start_late = int(0.03 * sample_rate)
        for i in range(start_late, ir_samples):
            time = i / sample_rate
            decay_factor = np.exp(-time * 6.91 / reverb_time)
            
            for channel in range(2):
                noise = np.random.normal(0, 1) * decay_factor * 0.6
                # More aggressive damping for small rooms
                if i > start_late:
                    noise = noise * (1 - damping * 1.5) + impulse_response[channel, i-1] * damping * 0.5
                
                impulse_response[channel, i] = noise
        
        return impulse_response
    
    def _generate_cathedral_ir(self, ir_samples, sample_rate, reverb_time, room_size, damping):
        """Generate cathedral impulse response with long, lush decay."""
        
        impulse_response = np.zeros((2, ir_samples))
        
        # Very long early reflections
        reflection_delays = np.linspace(0.02, 0.15, 12) * room_size
        
        for i, delay in enumerate(reflection_delays):
            delay_samples = int(delay * sample_rate)
            if delay_samples < ir_samples:
                gain = 0.9 - i * 0.05
                impulse_response[0, delay_samples] += gain * 0.8
                impulse_response[1, delay_samples + 3] += gain * 0.8
        
        # Very long, smooth decay
        start_late = int(0.2 * sample_rate)
        for i in range(start_late, ir_samples):
            time = i / sample_rate
            # Slower decay for cathedral
            decay_factor = np.exp(-time * 3.0 / reverb_time)
            
            for channel in range(2):
                noise = np.random.normal(0, 1) * decay_factor * 0.8
                # Gentle damping to preserve lushness
                if i > start_late:
                    noise = noise * (1 - damping * 0.3) + impulse_response[channel, i-1] * damping * 0.7
                
                impulse_response[channel, i] = noise
        
        return impulse_response
    
    def _generate_plate_ir(self, ir_samples, sample_rate, reverb_time, damping, diffusion):
        """Generate plate reverb impulse response."""
        
        impulse_response = np.zeros((2, ir_samples))
        
        # Plate characteristics: dense early reflections, bright sound
        for i in range(0, min(int(0.01 * sample_rate), ir_samples)):
            for channel in range(2):
                # Dense initial reflections
                noise = np.random.normal(0, 1) * (0.8 - i / (0.01 * sample_rate) * 0.7)
                impulse_response[channel, i] = noise * diffusion
        
        # Metallic resonances
        resonant_freqs = [440, 880, 1320, 1760]  # Hz
        for freq in resonant_freqs:
            for i in range(ir_samples):
                time = i / sample_rate
                decay = np.exp(-time * 8.0 / reverb_time)
                resonance = np.sin(2 * np.pi * freq * time) * decay * 0.1
                
                # Add to both channels with slight phase difference
                impulse_response[0, i] += resonance
                impulse_response[1, i] += resonance * np.cos(0.1 * time)
        
        # Bright, dense late field
        start_late = int(0.02 * sample_rate)
        for i in range(start_late, ir_samples):
            time = i / sample_rate
            decay_factor = np.exp(-time * 8.0 / reverb_time)
            
            for channel in range(2):
                noise = np.random.normal(0, 1) * decay_factor * 0.7
                # Less damping to maintain brightness
                if i > start_late:
                    noise = noise * (1 - damping * 0.2) + impulse_response[channel, i-1] * damping * 0.1
                
                impulse_response[channel, i] = noise
        
        return impulse_response
    
    def _generate_spring_ir(self, ir_samples, sample_rate, reverb_time, damping):
        """Generate spring reverb impulse response."""
        
        impulse_response = np.zeros((2, ir_samples))
        
        # Spring characteristics: boing, flutter, metallic
        flutter_freq = 12.0  # Hz
        boing_freq = 180.0   # Hz
        
        for i in range(ir_samples):
            time = i / sample_rate
            decay = np.exp(-time * 10.0 / reverb_time)
            
            # Spring flutter
            flutter = np.sin(2 * np.pi * flutter_freq * time) * decay * 0.3
            
            # Boing resonance
            boing = np.sin(2 * np.pi * boing_freq * time) * np.exp(-time * 15.0) * 0.5
            
            # Metallic noise
            noise = np.random.normal(0, 1) * decay * 0.4
            
            # High-frequency emphasis with damping
            if i > 0:
                noise = noise * (1 - damping * 0.8) + impulse_response[0, i-1] * damping * 0.2
            
            signal = flutter + boing + noise
            
            # Mono to stereo conversion with slight delay
            impulse_response[0, i] = signal
            if i < ir_samples - 2:
                impulse_response[1, i + 2] = signal * 0.8
        
        return impulse_response
    
    def _generate_chamber_ir(self, ir_samples, sample_rate, reverb_time, room_size, damping, diffusion):
        """Generate chamber reverb impulse response."""
        
        # Similar to room but warmer and more intimate
        impulse_response = self._generate_room_ir(ir_samples, sample_rate, reverb_time, 
                                                room_size * 0.7, damping * 1.2, diffusion)
        
        # Add warmth by emphasizing lower frequencies
        # Simple low-pass filtering effect
        for channel in range(2):
            for i in range(1, ir_samples):
                impulse_response[channel, i] = (impulse_response[channel, i] * 0.7 + 
                                              impulse_response[channel, i-1] * 0.3)
        
        return impulse_response
    
    def _generate_ambient_ir(self, ir_samples, sample_rate, reverb_time, diffusion):
        """Generate ambient reverb impulse response."""
        
        impulse_response = np.zeros((2, ir_samples))
        
        # Smooth, washy, no distinct early reflections
        for i in range(ir_samples):
            time = i / sample_rate
            decay_factor = np.exp(-time * 2.0 / reverb_time)  # Very slow decay
            
            for channel in range(2):
                # Smooth, diffuse noise
                noise = np.random.normal(0, 1) * decay_factor * 0.6
                
                # Heavy smoothing for ambient character
                if i > 4:
                    smoothed = np.mean(impulse_response[channel, i-5:i])
                    noise = noise * 0.3 + smoothed * 0.7 * diffusion
                
                impulse_response[channel, i] = noise
        
        return impulse_response
    
    def _apply_pre_delay(self, audio, delay_samples):
        """Apply pre-delay to audio."""
        
        delayed_audio = np.zeros((audio.shape[0], audio.shape[1] + delay_samples))
        delayed_audio[:, delay_samples:] = audio
        
        return delayed_audio
    
    def _apply_filtering(self, audio, sample_rate, low_cut, high_cut):
        """Apply low-cut and high-cut filtering."""
        
        # Simple first-order filters
        filtered_audio = audio.copy()
        
        # High-pass filter (low cut)
        if low_cut > 20:
            alpha_hp = np.exp(-2 * np.pi * low_cut / sample_rate)
            for channel in range(audio.shape[0]):
                for i in range(1, audio.shape[1]):
                    filtered_audio[channel, i] = (alpha_hp * filtered_audio[channel, i-1] + 
                                                 (1 - alpha_hp) * (audio[channel, i] - audio[channel, i-1]))
        
        # Low-pass filter (high cut)
        if high_cut < 20000:
            alpha_lp = np.exp(-2 * np.pi * high_cut / sample_rate)
            for channel in range(audio.shape[0]):
                for i in range(1, audio.shape[1]):
                    filtered_audio[channel, i] = (alpha_lp * filtered_audio[channel, i-1] + 
                                                 (1 - alpha_lp) * filtered_audio[channel, i])
        
        return filtered_audio
    
    def _apply_convolution(self, audio, impulse_response, sample_rate, conv_mode, 
                         mod_rate, mod_depth, freeze):
        """Apply convolution with the impulse response."""
        
        channels = audio.shape[0]
        ir_channels = impulse_response.shape[0]
        
        # Ensure matching channel count
        if ir_channels == 1 and channels == 2:
            # Mono IR to stereo audio
            impulse_response = np.repeat(impulse_response, 2, axis=0)
        elif ir_channels == 2 and channels == 1:
            # Stereo IR to mono audio
            impulse_response = np.mean(impulse_response, axis=0, keepdims=True)
        
        convolved_audio = np.zeros((channels, audio.shape[1] + impulse_response.shape[1] - 1))
        
        # Apply modulation to impulse response
        if mod_rate > 0 and mod_depth > 0:
            modulated_ir = self._apply_modulation_to_ir(impulse_response, sample_rate, mod_rate, mod_depth)
        else:
            modulated_ir = impulse_response
        
        # Freeze effect: loop the tail of the impulse response
        if freeze:
            modulated_ir = self._apply_freeze_effect(modulated_ir)
        
        # Perform convolution based on mode
        if conv_mode == "fft":
            # FFT-based convolution (most efficient for long IRs)
            for channel in range(channels):
                convolved_audio[channel] = np.convolve(audio[channel], modulated_ir[channel], mode='full')
        else:
            # Direct convolution (simplified)
            for channel in range(channels):
                convolved_audio[channel] = np.convolve(audio[channel], modulated_ir[channel], mode='full')
        
        # Trim to original length plus some reverb tail
        output_length = audio.shape[1] + min(impulse_response.shape[1], int(2 * sample_rate))
        convolved_audio = convolved_audio[:, :output_length]
        
        return convolved_audio
    
    def _apply_modulation_to_ir(self, impulse_response, sample_rate, mod_rate, mod_depth):
        """Apply modulation to impulse response for chorus-like effect."""
        
        modulated_ir = impulse_response.copy()
        
        for i in range(impulse_response.shape[1]):
            time = i / sample_rate
            mod_amount = mod_depth * np.sin(2 * np.pi * mod_rate * time + self.modulation_phase)
            
            # Apply amplitude modulation
            modulated_ir[:, i] *= (1 + mod_amount)
        
        # Update modulation phase
        self.modulation_phase += 2 * np.pi * mod_rate * impulse_response.shape[1] / sample_rate
        self.modulation_phase = self.modulation_phase % (2 * np.pi)
        
        return modulated_ir
    
    def _apply_freeze_effect(self, impulse_response):
        """Apply freeze effect to impulse response."""
        
        # Find the point where amplitude drops below threshold
        freeze_threshold = 0.1
        freeze_start = 0
        
        for i in range(impulse_response.shape[1]):
            if np.max(np.abs(impulse_response[:, i])) < freeze_threshold:
                freeze_start = i
                break
        
        if freeze_start > 0:
            # Loop the frozen section
            freeze_section = impulse_response[:, freeze_start:]
            frozen_ir = impulse_response.copy()
            
            # Repeat the freeze section multiple times with decay
            for repeat in range(3):
                decay_factor = 0.7 ** repeat
                section_length = freeze_section.shape[1]
                start_idx = freeze_start + repeat * section_length
                end_idx = min(start_idx + section_length, frozen_ir.shape[1])
                
                if start_idx < frozen_ir.shape[1]:
                    frozen_ir[:, start_idx:end_idx] += freeze_section[:, :end_idx-start_idx] * decay_factor
            
            return frozen_ir
        
        return impulse_response
    
    def _mix_wet_dry(self, dry_audio, wet_audio, wet_mix):
        """Mix wet and dry signals."""
        
        # Ensure same length
        min_length = min(dry_audio.shape[1], wet_audio.shape[1])
        dry_trimmed = dry_audio[:, :min_length]
        wet_trimmed = wet_audio[:, :min_length]
        
        # Mix
        mixed = dry_trimmed * (1 - wet_mix) + wet_trimmed * wet_mix
        
        return mixed
    
    def _apply_stereo_width(self, audio, width):
        """Apply stereo width control."""
        
        if audio.shape[0] != 2:
            return audio
        
        # Mid-side processing
        mid = (audio[0] + audio[1]) * 0.5
        side = (audio[0] - audio[1]) * 0.5 * width
        
        # Convert back to stereo
        left = mid + side
        right = mid - side
        
        return np.array([left, right])
    
    def _generate_reverb_report(self, reverb_type, conv_mode, wet_mix, reverb_time, 
                              ir_length, sample_rate, audio_shape):
        """Generate convolution reverb report."""
        
        report = f"""🏛️ CONVOLUTION REVERB REPORT 🏛️

🔧 Reverb Configuration:
  • Type: {reverb_type.title()}
  • Convolution Mode: {conv_mode.title()}
  • Wet/Dry Mix: {wet_mix:.1%}
  • Reverb Time: {reverb_time:.1f} seconds
  • IR Length: {ir_length:.1f} seconds

📊 Processing Details:
  • Input Channels: {audio_shape[0]}
  • Input Samples: {audio_shape[1]:,}
  • Sample Rate: {sample_rate} Hz
  • IR Samples: {int(ir_length * sample_rate):,}

🎭 Characteristics:
  • Room Type: {reverb_type.title()}
  • Processing: {"FFT Convolution" if conv_mode == "fft" else "Direct Convolution"}
  • Latency: {"Low" if ir_length < 2.0 else "Medium" if ir_length < 5.0 else "High"}
  • Quality: {"High-End Professional"}
"""
        
        return report


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
    
    # 🔬 PHASE 3: ANALYSIS
    "AudioAnalyzer": AudioAnalyzerNode,
    "SpectrumAnalyzer": SpectrumAnalyzerNode,
    "TrueChaos": TrueChaosNode,
    "ModulationMatrix": ModulationMatrixNode,
    "ConvolutionReverb": ConvolutionReverbNode,
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
    
    # 🔬 PHASE 3: ANALYSIS
    "AudioAnalyzer": "🔬 Audio Analyzer",
    "SpectrumAnalyzer": "📊 Spectrum Analyzer",
    "TrueChaos": "🌀 True Chaos",
    "ModulationMatrix": "🎛️ Modulation Matrix",
    "ConvolutionReverb": "🏛️ Convolution Reverb",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]