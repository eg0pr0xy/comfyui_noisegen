#!/usr/bin/env python3
"""
Test script for ComfyUI NoiseGen nodes.
Run this to verify all noise generation functions work correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from audio_utils import *

def test_all_noise_types():
    """Test all noise generation functions."""
    
    print("🎵 Testing ComfyUI NoiseGen Functions")
    print("=" * 50)
    
    # Test parameters
    duration = 2.0  # seconds
    sample_rate = 44100
    amplitude = 0.5
    seed = 42
    
    # Test basic noise types (mono)
    noise_functions = [
        ("White Noise (Mono)", lambda: generate_white_noise(duration, sample_rate, amplitude, seed)),
        ("Pink Noise (Mono)", lambda: generate_pink_noise(duration, sample_rate, amplitude, seed)),
        ("Brown Noise (Mono)", lambda: generate_brown_noise(duration, sample_rate, amplitude, seed)),
        ("Blue Noise (Mono)", lambda: generate_blue_noise(duration, sample_rate, amplitude, seed)),
        ("Violet Noise (Mono)", lambda: generate_violet_noise(duration, sample_rate, amplitude, seed)),
        ("Perlin Noise (Mono)", lambda: generate_perlin_noise(duration, 1.0, sample_rate, amplitude, 4, seed)),
        ("Band-Limited Noise (Mono)", lambda: generate_bandlimited_noise(duration, 1000, 4000, sample_rate, amplitude, seed)),
    ]
    
    # Test stereo versions
    stereo_functions = [
        ("White Noise (Stereo)", lambda: generate_white_noise(duration, sample_rate, amplitude, seed, 2, "independent", 1.0)),
        ("Pink Noise (Decorrelated)", lambda: generate_pink_noise(duration, sample_rate, amplitude, seed, 2, "decorrelated", 1.5)),
        ("Brown Noise (Correlated)", lambda: generate_brown_noise(duration, sample_rate, amplitude, seed, 2, "correlated", 0.8)),
        ("Perlin Noise (Wide Stereo)", lambda: generate_perlin_noise(duration, 1.0, sample_rate, amplitude, 4, seed, 2, "independent", 2.0)),
    ]
    
    results = {}
    
    # Test mono functions
    for name, func in noise_functions:
        try:
            print(f"Testing {name}...")
            audio = func()
            
            # Basic validation for mono
            assert len(audio) == int(duration * sample_rate), f"Wrong length for {name}"
            assert audio.dtype == np.float32, f"Wrong dtype for {name}"
            assert np.max(np.abs(audio)) <= amplitude * 1.1, f"Amplitude too high for {name}"  # Small tolerance
            
            # Store for analysis
            results[name] = audio
            print(f"✅ {name} - OK (length: {len(audio)}, max: {np.max(np.abs(audio)):.3f})")
            
        except Exception as e:
            print(f"❌ {name} - FAILED: {str(e)}")
            results[name] = None
    
    # Test stereo functions
    for name, func in stereo_functions:
        try:
            print(f"Testing {name}...")
            audio = func()
            
            # Basic validation for stereo
            assert audio.shape[0] == 2, f"Wrong number of channels for {name}"
            assert audio.shape[1] == int(duration * sample_rate), f"Wrong length for {name}"
            assert audio.dtype == np.float32, f"Wrong dtype for {name}"
            assert np.max(np.abs(audio)) <= amplitude * 1.1, f"Amplitude too high for {name}"  # Small tolerance
            
            # Test stereo separation
            left_max = np.max(np.abs(audio[0]))
            right_max = np.max(np.abs(audio[1]))
            print(f"   📊 L: {left_max:.3f}, R: {right_max:.3f}")
            
            # Store for analysis (use left channel for compatibility)
            results[name] = audio[0]  # Store left channel for plotting
            print(f"✅ {name} - OK (shape: {audio.shape}, max: {np.max(np.abs(audio)):.3f})")
            
        except Exception as e:
            print(f"❌ {name} - FAILED: {str(e)}")
            results[name] = None
    
    return results

def test_parameter_validation():
    """Test parameter validation function."""
    print("\n🔧 Testing Parameter Validation")
    print("=" * 30)
    
    # Test cases: (input, expected_output)
    test_cases = [
        ((5.0, 44100, 0.5, 1), (5.0, 44100, 0.5, 1)),  # Normal case
        ((0.05, 44100, 0.5, 1), (0.1, 44100, 0.5, 1)),  # Duration too small
        ((500.0, 44100, 0.5, 1), (300.0, 44100, 0.5, 1)),  # Duration too large
        ((5.0, 12345, 0.5, 1), (5.0, 44100, 0.5, 1)),  # Invalid sample rate
        ((5.0, 44100, -0.1, 1), (5.0, 44100, 0.0, 1)),  # Amplitude too small
        ((5.0, 44100, 3.0, 1), (5.0, 44100, 2.0, 1)),  # Amplitude too large
        ((5.0, 44100, 0.5, 2), (5.0, 44100, 0.5, 2)),  # Stereo case
        ((5.0, 44100, 0.5, 10), (5.0, 44100, 0.5, 8)),  # Too many channels
    ]
    
    for i, (input_params, expected) in enumerate(test_cases):
        result = validate_audio_params(*input_params)
        if result == expected:
            print(f"✅ Test case {i+1} - OK")
        else:
            print(f"❌ Test case {i+1} - FAILED: {result} != {expected}")

def test_comfy_audio_conversion():
    """Test conversion to ComfyUI audio format."""
    print("\n🎛️ Testing ComfyUI Audio Conversion")
    print("=" * 35)
    
    try:
        # Generate test audio
        audio_array = generate_white_noise(1.0, 44100, 0.5, 42)
        
        # Convert to ComfyUI format
        comfy_audio = numpy_to_comfy_audio(audio_array, 44100)
        
        # Validate format
        assert "waveform" in comfy_audio, "Missing waveform key"
        assert "sample_rate" in comfy_audio, "Missing sample_rate key"
        assert comfy_audio["sample_rate"] == 44100, "Wrong sample rate"
        assert comfy_audio["waveform"].shape[0] == 1, "Wrong number of channels"
        assert comfy_audio["waveform"].shape[1] == len(audio_array), "Wrong number of samples"
        
        print("✅ ComfyUI audio conversion - OK")
        
    except Exception as e:
        print(f"❌ ComfyUI audio conversion - FAILED: {str(e)}")

def plot_frequency_analysis(results):
    """Create frequency analysis plots for generated noise."""
    print("\n📊 Generating Frequency Analysis Plots")
    print("=" * 40)
    
    try:
        import matplotlib.pyplot as plt
        from scipy import signal
        
        # Filter out failed results
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            print("❌ No valid results to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, len(valid_results), figsize=(4*len(valid_results), 8))
        if len(valid_results) == 1:
            axes = axes.reshape(-1, 1)
        
        sample_rate = 44100
        
        for i, (name, audio) in enumerate(valid_results.items()):
            # Time domain plot
            time = np.linspace(0, len(audio)/sample_rate, len(audio))
            axes[0, i].plot(time[:int(sample_rate*0.1)], audio[:int(sample_rate*0.1)])  # First 0.1 seconds
            axes[0, i].set_title(f"{name} - Time Domain")
            axes[0, i].set_xlabel("Time (s)")
            axes[0, i].set_ylabel("Amplitude")
            
            # Frequency domain plot
            freqs, psd = signal.welch(audio, sample_rate, nperseg=2048)
            axes[1, i].semilogx(freqs[1:], 10*np.log10(psd[1:]))  # Skip DC component
            axes[1, i].set_title(f"{name} - Frequency Domain")
            axes[1, i].set_xlabel("Frequency (Hz)")
            axes[1, i].set_ylabel("Power (dB)")
            axes[1, i].grid(True)
            axes[1, i].set_xlim(20, sample_rate//2)
        
        plt.tight_layout()
        plt.savefig("noise_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Frequency analysis plots created: noise_analysis.png")
        
    except ImportError:
        print("⚠️ Matplotlib not available - skipping plots")
    except Exception as e:
        print(f"❌ Plotting failed: {str(e)}")

def main():
    """Run all tests."""
    print("🎵 ComfyUI NoiseGen Test Suite")
    print("=" * 60)
    
    # Test noise generation
    results = test_all_noise_types()
    
    # Test parameter validation
    test_parameter_validation()
    
    # Test ComfyUI format conversion
    test_comfy_audio_conversion()
    
    # Generate analysis plots
    plot_frequency_analysis(results)
    
    print("\n" + "=" * 60)
    print("🎉 Test suite completed!")
    print("If all tests passed, your NoiseGen nodes are ready to use!")

if __name__ == "__main__":
    main() 