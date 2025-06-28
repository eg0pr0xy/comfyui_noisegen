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
    
    def generate_noise(self, noise_type, duration, sample_rate, amplitude, seed, channels, 
                      stereo_mode, stereo_width, frequency=1.0, low_freq=100.0, high_freq=8000.0, octaves=4):
        try:
            # Parameter validation and fixing
            if isinstance(channels, str):
                channels = {"independent": 1, "correlated": 2, "decorrelated": 2}.get(channels, 1)
            if stereo_mode not in ["independent", "correlated", "decorrelated"]:
                stereo_mode = "independent"
            channels = 1 if channels < 1.5 else 2
            low_freq, high_freq = max(1.0, float(low_freq)), max(low_freq + 1.0, float(high_freq))
            
            duration, sample_rate, amplitude, channels = validate_audio_params(duration, sample_rate, amplitude, channels)
            
            # Common parameters
            params = {
                'duration': duration, 'sample_rate': sample_rate, 'amplitude': amplitude,
                'seed': seed, 'channels': channels, 'stereo_mode': stereo_mode, 'stereo_width': stereo_width
            }
            
            # Generate noise
            if noise_type == "white": result = generate_white_noise(**params)
            elif noise_type == "pink": result = generate_pink_noise(**params)
            elif noise_type == "brown": result = generate_brown_noise(**params)
            elif noise_type == "blue": result = generate_blue_noise(**params)
            elif noise_type == "violet": result = generate_violet_noise(**params)
            elif noise_type == "perlin": result = generate_perlin_noise(frequency=frequency, octaves=octaves, **params)
            elif noise_type == "bandlimited": result = generate_bandlimited_noise(low_frequency=low_freq, high_frequency=high_freq, **params)
            else: result = generate_white_noise(**params)
            
            return (result,)
            
        except Exception as e:
            print(f"❌ Error in noise generation: {str(e)}")
            fallback = create_audio_dict(np.zeros((channels, int(sample_rate * 0.1))), sample_rate)
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
            
            result = generate_perlin_noise(
                duration=duration, frequency=frequency, sample_rate=sample_rate, amplitude=amplitude,
                seed=seed, channels=channels, stereo_mode=stereo_mode, octaves=octaves, persistence=persistence
            )
            return (result,)
            
        except Exception as e:
            print(f"❌ Error in Perlin generation: {str(e)}")
            fallback = create_audio_dict(np.zeros((channels, int(sample_rate * 0.1))), sample_rate)
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
            
            result = generate_bandlimited_noise(
                duration=duration, low_frequency=low_frequency, high_frequency=high_frequency,
                sample_rate=sample_rate, amplitude=amplitude, seed=seed, channels=1
            )
            return (result,)
            
        except Exception as e:
            print(f"❌ Error in band-limited generation: {str(e)}")
            fallback = create_audio_dict(np.zeros((1, int(sample_rate * 0.1))), sample_rate)
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
                "master_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("mixed_audio",)
    FUNCTION = "mix_audio"
    CATEGORY = "🎵 NoiseGen/Mixing"
    DESCRIPTION = "Professional audio mixer with gain and pan controls"
    
    def mix_audio(self, audio_a, gain_a, pan_a, audio_b=None, gain_b=1.0, pan_b=0.0, master_gain=1.0):
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
# 🔧 UTILITIES - File I/O and Utilities
# =============================================================================

class AudioSaveNode:
    """Professional audio export with metadata preservation."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (AUDIO_TYPE, {"tooltip": "Audio data to save"}),
                "filename_prefix": ("STRING", {"default": "NoiseGen_"}),
                "format": (["wav", "flac"], {"default": "wav"}),
            }
        }
    
    RETURN_TYPES = (AUDIO_TYPE, "STRING")
    RETURN_NAMES = ("audio", "filepath")
    FUNCTION = "save_audio"
    CATEGORY = "🎵 NoiseGen/Utils"
    OUTPUT_NODE = True
    DESCRIPTION = "Save audio to file with metadata preservation"
    
    def save_audio(self, audio, filename_prefix, format):
        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            if hasattr(waveform, 'cpu'):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = waveform
            
            # Ensure output directory exists
            if folder_paths:
                output_dir = folder_paths.get_output_directory()
            else:
                output_dir = "outputs"
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}{timestamp}.{format}"
            filepath = os.path.join(output_dir, filename)
            
            # Convert to torch tensor for torchaudio
            if not hasattr(audio_np, 'float'):
                audio_tensor = torch.from_numpy(audio_np).float()
            else:
                audio_tensor = audio_np
            
            # Save audio file
            torchaudio.save(filepath, audio_tensor, sample_rate)
            
            print(f"✅ Audio saved: {filepath}")
            return (audio, filepath)
            
        except Exception as e:
            print(f"❌ Error saving audio: {str(e)}")
            return (audio, "error")


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
    
    # 🔧 UTILITIES
    "AudioSave": "💾 Audio Save",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]