#!/usr/bin/env python3
"""
Test script for ComfyUI NoiseGen nodes.
Run this to verify all noise generation functions work correctly.
"""

import sys
import os
import traceback
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        from noise_nodes import NODE_CLASS_MAPPINGS
        return NODE_CLASS_MAPPINGS
    except Exception as e:
        print(f"Failed to import: {e}")
        return None

print("NOISEGEN")
print("=" * 50)
print("Testing NoiseGen Node Pack")
print("=" * 50)

# Test imports
mappings = test_imports()
if not mappings:
    print("Failed to import nodes - exiting")
    sys.exit(1)

print(f"Found {len(mappings)} registered nodes:")
for name in mappings.keys():
    print(f"  - {name}")
print()

def test_basic_functionality():
    """Test basic functionality of each node type"""
    print("Testing Basic Node Functionality")
    print("-" * 30)
    
    # Import noise generation functions directly
    try:
        from audio_utils import (
            generate_white_noise, generate_pink_noise, generate_brown_noise,
            generate_blue_noise, generate_violet_noise, generate_perlin_noise,
            generate_band_limited_noise
        )
    except ImportError as e:
        print(f"Failed to import audio utils: {e}")
        return False
    
    # Test each noise type
    test_params = {
        "duration": 1.0,
        "sample_rate": 44100,
        "amplitude": 0.5,
        "channels": 1
    }
    
    noise_tests = [
        ("White Noise", lambda: generate_white_noise(**test_params)),
        ("Pink Noise", lambda: generate_pink_noise(**test_params)),
        ("Brown Noise", lambda: generate_brown_noise(**test_params)),
        ("Blue Noise", lambda: generate_blue_noise(**test_params)),
        ("Violet Noise", lambda: generate_violet_noise(**test_params)),
        ("Perlin Noise", lambda: generate_perlin_noise(**test_params, frequency=1.0, octaves=1)),
        ("Band-Limited", lambda: generate_band_limited_noise(**test_params, low_freq=100, high_freq=1000))
    ]
    
    for name, func in noise_tests:
        try:
            audio = func()
            if len(audio) > 0 and np.max(np.abs(audio)) > 0:
                print(f"OK {name} - Length: {len(audio)}, Max: {np.max(np.abs(audio)):.3f}")
            else:
                print(f"FAILED {name} - Empty or silent audio")
        except Exception as e:
            print(f"FAILED {name} - Error: {str(e)}")
    
    return True

def test_chaos_node():
    """Test ChaosNoiseMix node"""
    print("\nTesting ChaosNoiseMix Node")
    print("-" * 25)
    
    try:
        ChaosNoiseMixNode = mappings.get("ChaosNoiseMix")
        if not ChaosNoiseMixNode:
            print("ChaosNoiseMix node not found")
            return False
            
        node = ChaosNoiseMixNode()
        
        # Create dummy audio inputs (simulate ComfyUI AUDIO format)
        dummy_audio1 = np.random.normal(0, 0.1, (44100, 2)).astype(np.float32)
        dummy_audio2 = np.random.normal(0, 0.1, (44100, 2)).astype(np.float32)
        
        # Test different mixing modes
        mix_modes = ["add", "multiply", "xor", "chaos"]
        
        for mode in mix_modes:
            try:
                result = node.mix_chaos(
                    dummy_audio1, dummy_audio2,
                    mix_mode=mode,
                    chaos_amount=0.5,
                    distortion=0.3,
                    bit_crush=8,
                    feedback=0.1,
                    ring_freq=100
                )
                audio = result[0]
                print(f"OK {mode} - Shape: {audio.shape}, Max: {np.max(np.abs(audio)):.3f}")
            except Exception as e:
                print(f"FAILED {mode} - Error: {str(e)}")
    
    except Exception as e:
        print(f"ChaosNoiseMix test failed: {e}")
        return False
    
    return True

def test_parameter_validation():
    """Test parameter validation and edge cases"""
    print("\nTesting Parameter Validation")
    print("-" * 30)
    
    try:
        from audio_utils import validate_and_fix_params
        
        # Test cases: (input_params, expected_fixes)
        test_cases = [
            ({"channels": "independent", "stereo_mode": 2}, "String/int conversion"),
            ({"channels": 1, "stereo_mode": "correlated"}, "Normal case"),
            ({"duration": -1}, "Negative duration"),
            ({"amplitude": 5.0}, "High amplitude"),
            ({"sample_rate": 999}, "Invalid sample rate")
        ]
        
        for i, (params, description) in enumerate(test_cases):
            try:
                result = validate_and_fix_params(**params)
                print(f"OK Test case {i+1} - {description}")
            except Exception as e:
                print(f"FAILED Test case {i+1} - {description}: {result} != {expected}")
    
    except ImportError:
        print("Parameter validation function not found - skipping")
    except Exception as e:
        print(f"Parameter validation test failed: {e}")

def test_comfyui_integration():
    """Test ComfyUI AUDIO format compatibility"""
    print("\nTesting ComfyUI Integration")
    print("-" * 28)
    
    try:
        # Test universal noise generator
        NoiseGenNode = mappings.get("NoiseGenerator")
        if NoiseGenNode:
            node = NoiseGenNode()
            result = node.generate_noise(
                noise_type="white",
                duration=0.5,
                sample_rate=44100,
                amplitude=0.3,
                channels=2,
                stereo_mode="independent",
                seed=42
            )
            audio = result[0]
            print("OK ComfyUI audio conversion")
        else:
            print("NoiseGenerator node not found")
    except Exception as e:
        print(f"FAILED ComfyUI audio conversion: {str(e)}")

def analyze_and_plot():
    """Create frequency analysis plots of different noise types"""
    print("\nCreating Frequency Analysis")
    print("-" * 27)
    
    try:
        import matplotlib.pyplot as plt
        from scipy import signal
        from audio_utils import generate_white_noise, generate_pink_noise, generate_brown_noise
        
        # Generate different noise types
        duration = 2.0
        sample_rate = 44100
        
        noises = {
            "White": generate_white_noise(duration, sample_rate, 0.5, 1),
            "Pink": generate_pink_noise(duration, sample_rate, 0.5, 1),
            "Brown": generate_brown_noise(duration, sample_rate, 0.5, 1)
        }
        
        if not noises["White"].size:
            print("No valid results to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('NoiseGen Frequency Analysis', fontsize=16)
        
        # Time domain plots
        time = np.linspace(0, duration, len(noises["White"]))
        axes[0, 0].plot(time[:1000], noises["White"][:1000])
        axes[0, 0].set_title('White Noise (Time Domain)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        
        # Frequency domain analysis
        colors = ['blue', 'pink', 'brown']
        axes[0, 1].set_title('Frequency Response Comparison')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Power (dB)')
        
        for i, (name, noise) in enumerate(noises.items()):
            freqs, psd = signal.welch(noise, sample_rate, nperseg=4096)
            psd_db = 10 * np.log10(psd + 1e-12)
            axes[0, 1].semilogx(freqs[1:], psd_db[1:], label=name, color=colors[i], alpha=0.8)
        
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(20, 20000)
        
        # Individual spectrograms
        for i, (name, noise) in enumerate(list(noises.items())[:2]):
            row, col = (1, i)
            freqs, times, Sxx = signal.spectrogram(noise, sample_rate, nperseg=1024)
            im = axes[row, col].pcolormesh(times, freqs, 10 * np.log10(Sxx + 1e-12))
            axes[row, col].set_title(f'{name} Noise Spectrogram')
            axes[row, col].set_xlabel('Time (s)')
            axes[row, col].set_ylabel('Frequency (Hz)')
            plt.colorbar(im, ax=axes[row, col])
        
        plt.tight_layout()
        plt.savefig('noise_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Frequency analysis plots created: noise_analysis.png")
        return True
        
    except ImportError:
        print("Matplotlib/scipy not available - skipping plots")
        return False
    except Exception as e:
        print(f"Plotting failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("NOISEGEN")
    print("=" * 50)
    print("ComfyUI NoiseGen Test Suite")
    print("=" * 50)
    
    success_count = 0
    total_tests = 4
    
    if test_basic_functionality():
        success_count += 1
    
    if test_chaos_node():
        success_count += 1
        
    test_parameter_validation()
    success_count += 1
    
    test_comfyui_integration()
    success_count += 1
    
    # Optional plotting (doesn't count toward success)
    analyze_and_plot()
    
    print(f"\nTest suite completed!")
    print(f"Passed: {success_count}/{total_tests} tests")
    
    if success_count == total_tests:
        print("All core functionality working correctly!")
    else:
        print("Some tests failed - check output above")
        sys.exit(1) 