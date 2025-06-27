import numpy as np
import torch
from scipy import signal
from typing import Tuple, Optional

# Audio utility constants
DEFAULT_SAMPLE_RATE = 44100
MAX_DURATION = 300  # 5 minutes max
MIN_DURATION = 0.1  # 0.1 seconds min

def generate_white_noise(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE, 
                        amplitude: float = 1.0, seed: Optional[int] = None,
                        channels: int = 1, stereo_mode: str = "independent", 
                        stereo_width: float = 1.0) -> np.ndarray:
    """Generate white noise with flat frequency spectrum.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Peak amplitude (0.0 to 2.0)
        seed: Random seed for reproducibility
        channels: Number of channels (1=mono, 2=stereo)
        stereo_mode: "independent", "correlated", or "decorrelated"
        stereo_width: Stereo width factor (0.0=mono, 1.0=normal, 2.0=wide)
    
    Returns:
        Audio array: (channels, samples) or (samples,) for mono
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = int(duration * sample_rate)
    # Generate normalized white noise and scale to desired amplitude
    noise = np.random.normal(0, 1, num_samples)
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise)) * amplitude
    
    # Convert to stereo if requested
    if channels == 2:
        noise = create_stereo_from_mono(noise, stereo_mode, stereo_width, seed)
    elif channels > 2:
        # For multi-channel, create independent channels
        multi_channel = np.zeros((channels, num_samples), dtype=np.float32)
        for ch in range(channels):
            if seed is not None:
                np.random.seed(seed + ch)
            ch_noise = np.random.normal(0, 1, num_samples)
            if np.max(np.abs(ch_noise)) > 0:
                ch_noise = ch_noise / np.max(np.abs(ch_noise)) * amplitude
            multi_channel[ch] = ch_noise
        return multi_channel
    
    return noise.astype(np.float32)

def generate_pink_noise(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE, 
                       amplitude: float = 1.0, seed: Optional[int] = None,
                       channels: int = 1, stereo_mode: str = "independent", 
                       stereo_width: float = 1.0) -> np.ndarray:
    """Generate pink noise with 1/f frequency spectrum."""
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = int(duration * sample_rate)
    
    # Generate white noise
    white = np.random.normal(0, 1, num_samples)
    
    # Simplified pink noise using integration method
    # This is more stable than filtering approach
    pink = np.cumsum(white)
    
    # Remove DC component
    pink = pink - np.mean(pink)
    
    # Apply some high-frequency restoration for better pink noise characteristics
    pink = pink + 0.1 * white
    
    # Normalize and apply amplitude
    if np.max(np.abs(pink)) > 0:
        pink = pink / np.max(np.abs(pink)) * amplitude
    
    # Convert to stereo if requested
    if channels == 2:
        pink = create_stereo_from_mono(pink, stereo_mode, stereo_width, seed)
    elif channels > 2:
        # For multi-channel, create independent channels
        multi_channel = np.zeros((channels, num_samples), dtype=np.float32)
        for ch in range(channels):
            if seed is not None:
                np.random.seed(seed + ch)
            ch_white = np.random.normal(0, 1, num_samples)
            ch_pink = np.cumsum(ch_white)
            ch_pink = ch_pink - np.mean(ch_pink)
            ch_pink = ch_pink + 0.1 * ch_white
            if np.max(np.abs(ch_pink)) > 0:
                ch_pink = ch_pink / np.max(np.abs(ch_pink)) * amplitude
            multi_channel[ch] = ch_pink
        return multi_channel
    
    return pink.astype(np.float32)

def generate_brown_noise(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE, 
                        amplitude: float = 1.0, seed: Optional[int] = None,
                        channels: int = 1, stereo_mode: str = "independent", 
                        stereo_width: float = 1.0) -> np.ndarray:
    """Generate brown (red) noise with 1/f² frequency spectrum."""
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = int(duration * sample_rate)
    
    # Generate white noise
    white = np.random.normal(0, 1, num_samples)
    
    # Integrate to get brown noise (cumulative sum approximation)
    brown = np.cumsum(white)
    
    # Remove DC component and normalize
    brown = brown - np.mean(brown)
    if np.max(np.abs(brown)) > 0:
        brown = brown / np.max(np.abs(brown)) * amplitude
    
    # Convert to stereo if requested
    if channels == 2:
        brown = create_stereo_from_mono(brown, stereo_mode, stereo_width, seed)
    elif channels > 2:
        # For multi-channel, create independent channels
        multi_channel = np.zeros((channels, num_samples), dtype=np.float32)
        for ch in range(channels):
            if seed is not None:
                np.random.seed(seed + ch)
            ch_white = np.random.normal(0, 1, num_samples)
            ch_brown = np.cumsum(ch_white)
            ch_brown = ch_brown - np.mean(ch_brown)
            if np.max(np.abs(ch_brown)) > 0:
                ch_brown = ch_brown / np.max(np.abs(ch_brown)) * amplitude
            multi_channel[ch] = ch_brown
        return multi_channel
    
    return brown.astype(np.float32)

def generate_blue_noise(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE, 
                       amplitude: float = 1.0, seed: Optional[int] = None,
                       channels: int = 1, stereo_mode: str = "independent", 
                       stereo_width: float = 1.0) -> np.ndarray:
    """Generate blue noise with +3dB/octave frequency spectrum."""
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = int(duration * sample_rate)
    
    # Generate white noise
    white = np.random.normal(0, 1, num_samples)
    
    # Apply high-pass filter for blue noise characteristic
    nyquist = sample_rate / 2
    cutoff = 1000  # Hz
    b, a = signal.butter(2, cutoff / nyquist, btype='high')
    blue = signal.filtfilt(b, a, white)
    
    # Normalize and apply amplitude
    if np.max(np.abs(blue)) > 0:
        blue = blue / np.max(np.abs(blue)) * amplitude
    
    # Convert to stereo if requested
    if channels == 2:
        blue = create_stereo_from_mono(blue, stereo_mode, stereo_width, seed)
    elif channels > 2:
        # For multi-channel, create independent channels
        multi_channel = np.zeros((channels, num_samples), dtype=np.float32)
        for ch in range(channels):
            if seed is not None:
                np.random.seed(seed + ch)
            ch_white = np.random.normal(0, 1, num_samples)
            ch_blue = signal.filtfilt(b, a, ch_white)
            if np.max(np.abs(ch_blue)) > 0:
                ch_blue = ch_blue / np.max(np.abs(ch_blue)) * amplitude
            multi_channel[ch] = ch_blue
        return multi_channel
    
    return blue.astype(np.float32)

def generate_violet_noise(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE, 
                         amplitude: float = 1.0, seed: Optional[int] = None,
                         channels: int = 1, stereo_mode: str = "independent", 
                         stereo_width: float = 1.0) -> np.ndarray:
    """Generate violet noise with +6dB/octave frequency spectrum."""
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = int(duration * sample_rate)
    
    # Generate white noise
    white = np.random.normal(0, 1, num_samples)
    
    # Differentiate to get violet noise
    violet = np.diff(white, prepend=white[0])
    
    # Normalize and apply amplitude
    if np.max(np.abs(violet)) > 0:
        violet = violet / np.max(np.abs(violet)) * amplitude
    
    # Convert to stereo if requested
    if channels == 2:
        violet = create_stereo_from_mono(violet, stereo_mode, stereo_width, seed)
    elif channels > 2:
        # For multi-channel, create independent channels
        multi_channel = np.zeros((channels, num_samples), dtype=np.float32)
        for ch in range(channels):
            if seed is not None:
                np.random.seed(seed + ch)
            ch_white = np.random.normal(0, 1, num_samples)
            ch_violet = np.diff(ch_white, prepend=ch_white[0])
            if np.max(np.abs(ch_violet)) > 0:
                ch_violet = ch_violet / np.max(np.abs(ch_violet)) * amplitude
            multi_channel[ch] = ch_violet
        return multi_channel
    
    return violet.astype(np.float32)

def generate_perlin_noise(duration: float, frequency: float = 1.0, 
                         sample_rate: int = DEFAULT_SAMPLE_RATE, 
                         amplitude: float = 1.0, octaves: int = 4,
                         seed: Optional[int] = None, channels: int = 1, 
                         stereo_mode: str = "independent", stereo_width: float = 1.0) -> np.ndarray:
    """Generate Perlin-like noise for natural variations."""
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)
    
    noise = np.zeros(num_samples)
    
    for octave in range(octaves):
        freq = frequency * (2 ** octave)
        amp = 1.0 / (2 ** octave)  # Normalized amplitude per octave
        
        # Generate smooth noise using sine waves with random phases
        phase = np.random.uniform(0, 2 * np.pi)
        noise += amp * np.sin(2 * np.pi * freq * t + phase)
    
    # Normalize and scale to desired amplitude
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise)) * amplitude
    
    # Convert to stereo if requested
    if channels == 2:
        noise = create_stereo_from_mono(noise, stereo_mode, stereo_width, seed)
    elif channels > 2:
        # For multi-channel, create independent channels
        multi_channel = np.zeros((channels, num_samples), dtype=np.float32)
        for ch in range(channels):
            if seed is not None:
                np.random.seed(seed + ch)
            ch_noise = np.zeros(num_samples)
            for octave in range(octaves):
                freq = frequency * (2 ** octave)
                amp = 1.0 / (2 ** octave)
                phase = np.random.uniform(0, 2 * np.pi)
                ch_noise += amp * np.sin(2 * np.pi * freq * t + phase)
            if np.max(np.abs(ch_noise)) > 0:
                ch_noise = ch_noise / np.max(np.abs(ch_noise)) * amplitude
            multi_channel[ch] = ch_noise
        return multi_channel
    
    return noise.astype(np.float32)

def generate_bandlimited_noise(duration: float, low_freq: float, high_freq: float,
                              sample_rate: int = DEFAULT_SAMPLE_RATE, 
                              amplitude: float = 1.0, seed: Optional[int] = None,
                              channels: int = 1, stereo_mode: str = "independent", 
                              stereo_width: float = 1.0) -> np.ndarray:
    """Generate band-limited noise between specified frequencies."""
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = int(duration * sample_rate)
    
    # Generate white noise
    white = np.random.normal(0, 1, num_samples)
    
    # Apply band-pass filter
    nyquist = sample_rate / 2
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    # Ensure frequencies are valid
    low_norm = max(0.01, min(low_norm, 0.99))
    high_norm = max(low_norm + 0.01, min(high_norm, 0.99))
    
    b, a = signal.butter(4, [low_norm, high_norm], btype='band')
    filtered = signal.filtfilt(b, a, white)
    
    # Normalize and apply amplitude
    if np.max(np.abs(filtered)) > 0:
        filtered = filtered / np.max(np.abs(filtered)) * amplitude
    
    # Convert to stereo if requested
    if channels == 2:
        filtered = create_stereo_from_mono(filtered, stereo_mode, stereo_width, seed)
    elif channels > 2:
        # For multi-channel, create independent channels
        multi_channel = np.zeros((channels, num_samples), dtype=np.float32)
        for ch in range(channels):
            if seed is not None:
                np.random.seed(seed + ch)
            ch_white = np.random.normal(0, 1, num_samples)
            ch_filtered = signal.filtfilt(b, a, ch_white)
            if np.max(np.abs(ch_filtered)) > 0:
                ch_filtered = ch_filtered / np.max(np.abs(ch_filtered)) * amplitude
            multi_channel[ch] = ch_filtered
        return multi_channel
    
    return filtered.astype(np.float32)

def numpy_to_comfy_audio(audio_array: np.ndarray, sample_rate: int) -> dict:
    """Convert numpy array to ComfyUI audio format.
    
    This follows the standard ComfyUI audio format which should be compatible
    with most audio nodes including ComfyUI-audio and VideoHelperSuite.
    
    Args:
        audio_array: Shape (channels, samples) or (samples,) for mono
        sample_rate: Sample rate in Hz
    
    Returns:
        dict: {"waveform": tensor, "sample_rate": int}
    """
    # Ensure audio is 2D (channels, samples)
    if audio_array.ndim == 1:
        audio_array = audio_array[np.newaxis, :]  # Add channel dimension
    
    # Convert to torch tensor and ensure proper format
    waveform = torch.from_numpy(audio_array).float()
    
    # Ensure waveform is in the correct range [-1, 1]
    max_val = torch.max(torch.abs(waveform))
    if max_val > 1.0:
        waveform = waveform / max_val
    
    return {
        "waveform": waveform,
        "sample_rate": sample_rate
    }

def validate_audio_params(duration: float, sample_rate: int, amplitude: float, 
                         channels: int = 1) -> Tuple[float, int, float, int]:
    """Validate and clamp audio parameters to safe ranges."""
    # Clamp duration
    duration = max(MIN_DURATION, min(duration, MAX_DURATION))
    
    # Validate sample rate
    if sample_rate not in [8000, 16000, 22050, 44100, 48000, 96000]:
        sample_rate = DEFAULT_SAMPLE_RATE
    
    # Clamp amplitude
    amplitude = max(0.0, min(amplitude, 2.0))
    
    # Validate channels
    channels = max(1, min(channels, 8))  # Support up to 7.1 surround
    
    return duration, sample_rate, amplitude, channels

def create_stereo_from_mono(mono_audio: np.ndarray, stereo_mode: str = "independent", 
                           stereo_width: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
    """Convert mono audio to stereo with various modes.
    
    Args:
        mono_audio: Mono audio array (samples,)
        stereo_mode: "independent", "correlated", or "decorrelated"
        stereo_width: Stereo width factor (0.0 = mono, 1.0 = normal, 2.0 = wide)
        seed: Random seed for decorrelated mode
    
    Returns:
        Stereo audio array (2, samples)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if stereo_mode == "independent":
        # Generate completely independent audio for each channel
        left = mono_audio.copy()
        right = generate_independent_noise_like(mono_audio, seed)
        
    elif stereo_mode == "correlated":
        # Use same audio for both channels but apply stereo width
        left = mono_audio.copy()
        right = mono_audio.copy()
        
    elif stereo_mode == "decorrelated":
        # Create decorrelated stereo using all-pass filters
        left = mono_audio.copy()
        right = apply_decorrelation_filter(mono_audio)
        
    else:
        # Default to correlated
        left = mono_audio.copy()
        right = mono_audio.copy()
    
    # Apply stereo width
    if stereo_width != 1.0:
        # Stereo width processing
        mid = (left + right) / 2
        side = (left - right) / 2
        side *= stereo_width
        left = mid + side
        right = mid - side
        
        # Normalize after stereo width processing to prevent clipping
        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
        if max_val > 1.0:
            left = left / max_val
            right = right / max_val
    
    return np.stack([left, right], axis=0)

def generate_independent_noise_like(reference_audio: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """Generate independent noise with similar characteristics to reference audio."""
    if seed is not None:
        np.random.seed(seed + 1)  # Different seed for right channel
    
    # Simple approach: generate new random noise with same length
    return np.random.normal(0, np.std(reference_audio), len(reference_audio)).astype(np.float32)

def apply_decorrelation_filter(audio: np.ndarray) -> np.ndarray:
    """Apply decorrelation filter to create psychoacoustically pleasant stereo."""
    # Simple decorrelation using phase shift
    # This is a simplified approach - professional decorrelators are more complex
    
    # Apply slight delay and filtering to create decorrelation
    delayed = np.concatenate([np.zeros(7), audio[:-7]])  # ~0.16ms delay at 44.1kHz
    
    # Mix original with delayed version
    decorrelated = 0.7 * audio + 0.3 * delayed
    
    return decorrelated.astype(np.float32) 