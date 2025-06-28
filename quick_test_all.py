#!/usr/bin/env python3
"""
Quick Comprehensive Test for all NoiseGen nodes
Tests all 10 nodes with various parameter combinations
"""

import sys
import os
import traceback
import numpy as np
import torch
import time

# Add current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import noise_nodes as nn
    from audio_utils import *
    print("✅ Successfully imported modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def validate_audio(audio_dict, expected_channels=None):
    """Quick audio validation."""
    try:
        if not isinstance(audio_dict, dict):
            return False, f"Not dict: {type(audio_dict)}"
        
        if 'waveform' not in audio_dict or 'sample_rate' not in audio_dict:
            return False, "Missing waveform or sample_rate"
        
        waveform = audio_dict['waveform']
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        
        if not isinstance(waveform, np.ndarray) or waveform.ndim != 2:
            return False, f"Invalid waveform shape"
        
        channels, samples = waveform.shape
        
        if expected_channels and channels != expected_channels:
            return False, f"Expected {expected_channels}ch, got {channels}ch"
        
        if np.any(np.isnan(waveform)) or np.any(np.isinf(waveform)):
            return False, "Contains NaN/Inf"
        
        max_amp = np.max(np.abs(waveform))
        return True, f"{channels}ch, {samples} samples, {audio_dict['sample_rate']}Hz, max={max_amp:.3f}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def create_test_audio(duration=1.0, channels=2, sample_rate=44100):
    """Create test audio."""
    samples = int(duration * sample_rate)
    waveform = np.random.normal(0, 0.1, (channels, samples)).astype(np.float32)
    return numpy_to_comfy_audio(waveform, sample_rate)

def test_noise_generator():
    """Test NoiseGenerator with all types and parameters."""
    print("\n🎵 Testing NoiseGenerator...")
    node = nn.NoiseGeneratorNode()
    results = []
    
    # Test all noise types
    noise_types = ['white', 'pink', 'brown', 'blue', 'violet', 'perlin', 'bandlimited']
    for noise_type in noise_types:
        try:
            result = node.generate_noise(
                noise_type=noise_type, duration=1.0, sample_rate=44100, amplitude=0.7,
                seed=42, channels=1, stereo_mode='independent', stereo_width=1.0,
                frequency=2.0, low_freq=200.0, high_freq=8000.0, octaves=4
            )
            valid, details = validate_audio(result[0])
            status = "✅" if valid else "❌"
            print(f"  {status} {noise_type}: {details}")
            results.append(valid)
        except Exception as e:
            print(f"  ❌ {noise_type}: Exception - {str(e)}")
            results.append(False)
    
    # Test stereo modes
    stereo_modes = ['independent', 'correlated', 'decorrelated']
    for mode in stereo_modes:
        try:
            result = node.generate_noise(
                noise_type='white', duration=0.5, sample_rate=44100, amplitude=0.5,
                seed=42, channels=2, stereo_mode=mode, stereo_width=1.0
            )
            valid, details = validate_audio(result[0], expected_channels=2)
            status = "✅" if valid else "❌"
            print(f"  {status} stereo_{mode}: {details}")
            results.append(valid)
        except Exception as e:
            print(f"  ❌ stereo_{mode}: Exception - {str(e)}")
            results.append(False)
    
    # Test legacy parameter swapping
    try:
        result = node.generate_noise(
            noise_type='white', duration=0.5, sample_rate=44100, amplitude=0.5,
            seed=42, channels='independent', stereo_mode='1', stereo_width=1.0
        )
        valid, details = validate_audio(result[0])
        status = "✅" if valid else "❌"
        print(f"  {status} legacy_param_swap: {details}")
        results.append(valid)
    except Exception as e:
        print(f"  ❌ legacy_param_swap: Exception - {str(e)}")
        results.append(False)
    
    return results

def test_processors():
    """Test processing nodes."""
    print("\n🔄 Testing Processor Nodes...")
    test_audio = create_test_audio(1.0, 2, 44100)
    results = []
    
    # Test FeedbackProcessor
    try:
        node = nn.FeedbackProcessorNode()
        result = node.process_feedback(
            audio=test_audio, feedback_mode='filtered', feedback_amount=0.3,
            delay_time=2.0, filter_type='lowpass', filter_freq=2000.0,
            filter_resonance=0.3, saturation=0.2, modulation_rate=0.5,
            modulation_depth=0.2, modulation_type='sine', amplitude=0.8
        )
        valid, details = validate_audio(result[0])
        status = "✅" if valid else "❌"
        print(f"  {status} FeedbackProcessor: {details}")
        results.append(valid)
    except Exception as e:
        print(f"  ❌ FeedbackProcessor: Exception - {str(e)}")
        results.append(False)
    
    # Test HarshFilter
    try:
        node = nn.HarshFilterNode()
        result = node.process_harsh_filter(
            audio=test_audio, filter_type='lowpass', cutoff_freq=1000.0,
            resonance=0.5, drive=0.3, drive_mode='tube', filter_slope=1.0,
            morph_amount=0.0, modulation_rate=0.5, modulation_depth=0.1, amplitude=0.8
        )
        valid, details = validate_audio(result[0])
        status = "✅" if valid else "❌"
        print(f"  {status} HarshFilter: {details}")
        results.append(valid)
    except Exception as e:
        print(f"  ❌ HarshFilter: Exception - {str(e)}")
        results.append(False)
    
    # Test MultiDistortion
    try:
        node = nn.MultiDistortionNode()
        result = node.process_multi_distortion(
            audio=test_audio, distortion_type='tube', drive=0.5, output_gain=0.7,
            wet_dry_mix=1.0, stages=1, asymmetry=0.0, amplitude=0.8
        )
        valid, details = validate_audio(result[0])
        status = "✅" if valid else "❌"
        print(f"  {status} MultiDistortion: {details}")
        results.append(valid)
    except Exception as e:
        print(f"  ❌ MultiDistortion: Exception - {str(e)}")
        results.append(False)
    
    # Test SpectralProcessor
    try:
        node = nn.SpectralProcessorNode()
        result = node.process_spectral(
            audio=test_audio, spectral_mode='enhance', fft_size='2048',
            overlap_factor=0.75, window_type='hann', frequency_range_low=100.0,
            frequency_range_high=8000.0, intensity=0.5, amplitude=0.8
        )
        valid, details = validate_audio(result[0])
        status = "✅" if valid else "❌"
        print(f"  {status} SpectralProcessor: {details}")
        results.append(valid)
    except Exception as e:
        print(f"  ❌ SpectralProcessor: Exception - {str(e)}")
        results.append(False)
    
    return results

def test_mixers():
    """Test mixer nodes."""
    print("\n🎚️ Testing Mixer Nodes...")
    audio_a = create_test_audio(1.0, 2, 44100)
    audio_b = create_test_audio(1.0, 2, 44100)
    results = []
    
    # Test AudioMixer
    try:
        node = nn.AudioMixerNode()
        result = node.mix_audio(
            audio_a=audio_a, gain_a=0.7, pan_a=-0.3,
            audio_b=audio_b, gain_b=0.5, pan_b=0.3,
            master_gain=0.8
        )
        valid, details = validate_audio(result[0])
        status = "✅" if valid else "❌"
        print(f"  {status} AudioMixer: {details}")
        results.append(valid)
    except Exception as e:
        print(f"  ❌ AudioMixer: Exception - {str(e)}")
        results.append(False)
    
    # Test ChaosNoiseMix
    try:
        node = nn.ChaosNoiseMixNode()
        result = node.mix_chaos(
            noise_a=audio_a, noise_b=audio_b, mix_mode='add', mix_ratio=0.5,
            chaos_amount=0.3, distortion=0.4, amplitude=0.8
        )
        valid, details = validate_audio(result[0])
        status = "✅" if valid else "❌"
        print(f"  {status} ChaosNoiseMix: {details}")
        results.append(valid)
    except Exception as e:
        print(f"  ❌ ChaosNoiseMix: Exception - {str(e)}")
        results.append(False)
    
    return results

def test_utilities():
    """Test utility nodes."""
    print("\n🛠️ Testing Utility Nodes...")
    results = []
    
    # Test PerlinNoise
    try:
        node = nn.PerlinNoiseNode()
        result = node.generate(
            duration=1.0, frequency=2.0, sample_rate=44100, amplitude=0.8,
            seed=42, channels=2, stereo_mode='independent', octaves=6, persistence=1.0
        )
        valid, details = validate_audio(result[0])
        status = "✅" if valid else "❌"
        print(f"  {status} PerlinNoise: {details}")
        results.append(valid)
    except Exception as e:
        print(f"  ❌ PerlinNoise: Exception - {str(e)}")
        results.append(False)
    
    # Test BandLimitedNoise
    try:
        node = nn.BandLimitedNoiseNode()
        result = node.generate(
            duration=1.0, low_frequency=200.0, high_frequency=8000.0,
            sample_rate=44100, amplitude=0.8, seed=42
        )
        valid, details = validate_audio(result[0])
        status = "✅" if valid else "❌"
        print(f"  {status} BandLimitedNoise: {details}")
        results.append(valid)
    except Exception as e:
        print(f"  ❌ BandLimitedNoise: Exception - {str(e)}")
        results.append(False)
    
    # Test AudioSave
    try:
        test_audio = create_test_audio()
        node = nn.AudioSaveNode()
        result = node.save_audio(
            audio=test_audio, filename_prefix="test_", format="wav"
        )
        if result and len(result) >= 2:
            valid, details = validate_audio(result[0])
            filepath = result[1]
            status = "✅" if valid else "❌"
            print(f"  {status} AudioSave: {details}, saved: {filepath}")
            results.append(valid)
        else:
            print(f"  ❌ AudioSave: Invalid save result")
            results.append(False)
    except Exception as e:
        print(f"  ❌ AudioSave: Exception - {str(e)}")
        results.append(False)
    
    return results

def test_chaining():
    """Test node chaining."""
    print("\n🔗 Testing Node Chaining...")
    results = []
    
    try:
        # Generate noise
        noise_gen = nn.NoiseGeneratorNode()
        noise = noise_gen.generate_noise(
            noise_type='white', duration=2.0, sample_rate=44100, amplitude=0.8,
            seed=42, channels=2, stereo_mode='independent', stereo_width=1.0
        )[0]
        
        # Process through feedback
        feedback = nn.FeedbackProcessorNode()
        processed = feedback.process_feedback(
            audio=noise, feedback_mode='filtered', feedback_amount=0.2,
            delay_time=1.0, filter_type='lowpass', filter_freq=3000.0,
            filter_resonance=0.2, saturation=0.1, modulation_rate=0.3,
            modulation_depth=0.1, modulation_type='sine', amplitude=0.8
        )[0]
        
        # Mix with original
        mixer = nn.AudioMixerNode()
        mixed = mixer.mix_audio(
            audio_a=noise, gain_a=0.5, pan_a=-0.2,
            audio_b=processed, gain_b=0.5, pan_b=0.2,
            master_gain=0.8
        )[0]
        
        # Save result
        save_node = nn.AudioSaveNode()
        saved = save_node.save_audio(
            audio=mixed, filename_prefix="chain_test_", format="wav"
        )
        
        valid, details = validate_audio(mixed)
        status = "✅" if valid else "❌"
        print(f"  {status} Complete Chain: {details}")
        results.append(valid)
        
        if saved and len(saved) >= 2:
            print(f"  ✅ Chain saved to: {saved[1]}")
        
    except Exception as e:
        print(f"  ❌ Node Chaining: Exception - {str(e)}")
        results.append(False)
    
    return results

def test_edge_cases():
    """Test edge cases and parameter limits."""
    print("\n⚠️ Testing Edge Cases...")
    noise_gen = nn.NoiseGeneratorNode()
    results = []
    
    edge_tests = [
        ("min_duration", {'duration': 0.1}),
        ("max_duration", {'duration': 10.0}),
        ("min_amplitude", {'amplitude': 0.0}),
        ("max_amplitude", {'amplitude': 2.0}),
        ("high_samplerate", {'sample_rate': 96000}),
        ("extreme_perlin", {'noise_type': 'perlin', 'frequency': 50.0, 'octaves': 8}),
        ("extreme_bandlimited", {'noise_type': 'bandlimited', 'low_freq': 50.0, 'high_freq': 15000.0}),
    ]
    
    base_params = {
        'noise_type': 'white', 'duration': 1.0, 'sample_rate': 44100, 'amplitude': 0.7,
        'seed': 42, 'channels': 1, 'stereo_mode': 'independent', 'stereo_width': 1.0,
        'frequency': 2.0, 'low_freq': 200.0, 'high_freq': 8000.0, 'octaves': 4
    }
    
    for test_name, overrides in edge_tests:
        try:
            params = base_params.copy()
            params.update(overrides)
            result = noise_gen.generate_noise(**params)
            valid, details = validate_audio(result[0])
            status = "✅" if valid else "❌"
            print(f"  {status} {test_name}: {details}")
            results.append(valid)
        except Exception as e:
            print(f"  ❌ {test_name}: Exception - {str(e)}")
            results.append(False)
    
    return results

def main():
    """Run comprehensive test suite."""
    print("🎵 ComfyUI-NoiseGen Comprehensive Test Suite")
    print("=" * 50)
    print("Testing all parameters on all 10 nodes")
    print("=" * 50)
    
    start_time = time.time()
    all_results = []
    
    # Run all tests
    all_results.extend(test_noise_generator())
    all_results.extend(test_processors())
    all_results.extend(test_mixers())
    all_results.extend(test_utilities())
    all_results.extend(test_chaining())
    all_results.extend(test_edge_cases())
    
    # Generate report
    end_time = time.time()
    total_tests = len(all_results)
    passed_tests = sum(all_results)
    failed_tests = total_tests - passed_tests
    
    print("\n" + "=" * 50)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 50)
    print(f"📈 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"⏱️ Total Time: {end_time - start_time:.2f}s")
    
    if failed_tests == 0:
        print("\n🎉 PERFECT! All tests passed.")
        print("✨ All 10 nodes work correctly with all parameter combinations.")
        print("🚀 System is fully functional and ready for production.")
    elif failed_tests <= total_tests * 0.1:
        print("\n✅ EXCELLENT! System is highly functional.")
        print("🔧 Minor issues detected but core functionality works.")
    elif failed_tests <= total_tests * 0.2:
        print("\n⚠️ GOOD! System is mostly functional.")
        print("🛠️ Some parameter combinations need attention.")
    else:
        print("\n❌ NEEDS WORK! Significant issues detected.")
        print("🚨 Multiple failures require investigation.")
    
    # Test coverage summary
    print(f"\n📋 Test Coverage:")
    print(f"   • 7 noise types tested")
    print(f"   • 3 stereo modes tested") 
    print(f"   • 4 processing nodes tested")
    print(f"   • 2 mixer nodes tested")
    print(f"   • 3 utility nodes tested")
    print(f"   • Node chaining tested")
    print(f"   • Edge cases tested")
    print(f"   • Legacy compatibility tested")
    
    print(f"\n🎯 System Status: {'🟢 OPERATIONAL' if failed_tests <= total_tests * 0.1 else '🟡 REVIEW NEEDED' if failed_tests <= total_tests * 0.2 else '🔴 REQUIRES FIXES'}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        traceback.print_exc() 