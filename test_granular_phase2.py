#!/usr/bin/env python3
"""
🌟 PHASE 2: Granular Synthesis Test Suite
Comprehensive testing for the three new granular synthesis nodes.
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from noise_nodes import (
    GranularProcessorNode, 
    GranularSequencerNode, 
    MicrosoundSculptorNode,
    NoiseGeneratorNode  # For creating test audio
)

def create_test_audio(duration=3.0, sample_rate=44100, noise_type="white"):
    """Create test audio for granular processing."""
    noise_gen = NoiseGeneratorNode()
    result = noise_gen.generate_noise(
        noise_type=noise_type,
        duration=duration,
        sample_rate=sample_rate,
        amplitude=0.5,
        seed=42,
        channels=2,
        stereo_mode="independent",
        stereo_width=1.0
    )
    return result[0]  # Return the audio dict

def test_granular_processor():
    """Test the GranularProcessor node with various settings."""
    print("🌟 Testing GranularProcessor...")
    
    # Create test audio
    test_audio = create_test_audio(duration=2.0, noise_type="pink")
    
    # Create granular processor
    granular = GranularProcessorNode()
    
    # Test different granular settings
    test_configs = [
        {
            "name": "Classic Granular",
            "grain_size": 100.0,
            "grain_density": 20.0,
            "pitch_ratio": 1.0,
            "grain_envelope": "hann",
            "positioning_mode": "random",
            "pitch_mode": "preserve",
            "grain_scatter": 0.3,
            "amplitude": 0.8
        },
        {
            "name": "Dense Microsound",
            "grain_size": 25.0,
            "grain_density": 100.0,
            "pitch_ratio": 1.5,
            "grain_envelope": "gaussian",
            "positioning_mode": "sequential",
            "pitch_mode": "transpose",
            "grain_scatter": 0.5,
            "amplitude": 0.7
        },
        {
            "name": "Freeze Mode",
            "grain_size": 200.0,
            "grain_density": 10.0,
            "pitch_ratio": 0.5,
            "grain_envelope": "triangle",
            "positioning_mode": "freeze",
            "pitch_mode": "random",
            "grain_scatter": 0.1,
            "amplitude": 0.9,
            "freeze_position": 0.3
        },
        {
            "name": "Chaotic Grains",
            "grain_size": 50.0,
            "grain_density": 50.0,
            "pitch_ratio": 2.0,
            "grain_envelope": "adsr",
            "positioning_mode": "pingpong",
            "pitch_mode": "random",
            "grain_scatter": 0.8,
            "amplitude": 0.6,
            "pitch_scatter": 0.5,
            "stereo_spread": 0.7
        }
    ]
    
    for i, config in enumerate(test_configs):
        print(f"  Testing: {config['name']}")
        
        try:
            result = granular.process_granular(
                audio=test_audio,
                **{k: v for k, v in config.items() if k != "name"}
            )
            
            audio_output = result[0]
            waveform = audio_output["waveform"]
            
            # Validate output
            assert isinstance(waveform, torch.Tensor), "Output should be torch tensor"
            assert waveform.shape[0] == 2, "Should maintain stereo channels"
            assert waveform.shape[1] > 0, "Should have audio samples"
            assert torch.max(torch.abs(waveform)) <= 1.1, "Should be within reasonable amplitude range"
            
            print(f"    ✅ {config['name']}: Shape {waveform.shape}, Max amplitude {torch.max(torch.abs(waveform)):.3f}")
            
        except Exception as e:
            print(f"    ❌ {config['name']}: Error - {str(e)}")
            return False
    
    print("✅ GranularProcessor tests passed!")
    return True

def test_granular_sequencer():
    """Test the GranularSequencer node with various patterns."""
    print("🎵 Testing GranularSequencer...")
    
    # Create test audio
    test_audio = create_test_audio(duration=1.5, noise_type="white")
    
    # Create granular sequencer
    sequencer = GranularSequencerNode()
    
    # Test different sequencer settings
    test_configs = [
        {
            "name": "Simple 8-Step",
            "steps": 8,
            "step_duration": 0.125,
            "base_grain_size": 100.0,
            "base_density": 15.0,
            "pattern_variation": 0.3,
            "swing": 0.0,
            "amplitude": 0.8,
            "probability": 1.0
        },
        {
            "name": "Euclidean Pattern",
            "steps": 16,
            "step_duration": 0.0625,
            "base_grain_size": 75.0,
            "base_density": 20.0,
            "pattern_variation": 0.5,
            "swing": 0.1,
            "amplitude": 0.7,
            "euclidean_rhythm": 5
        },
        {
            "name": "Sparse Probability",
            "steps": 12,
            "step_duration": 0.1,
            "base_grain_size": 150.0,
            "base_density": 8.0,
            "pattern_variation": 0.7,
            "swing": -0.2,
            "amplitude": 0.9,
            "probability": 0.6,
            "velocity_variation": 0.4
        },
        {
            "name": "Dense Microsound",
            "steps": 32,
            "step_duration": 0.03125,
            "base_grain_size": 25.0,
            "base_density": 50.0,
            "pattern_variation": 0.8,
            "swing": 0.3,
            "amplitude": 0.6,
            "probability": 0.8
        }
    ]
    
    for i, config in enumerate(test_configs):
        print(f"  Testing: {config['name']}")
        
        try:
            result = sequencer.process_sequenced_granular(
                audio=test_audio,
                **{k: v for k, v in config.items() if k != "name"}
            )
            
            audio_output = result[0]
            waveform = audio_output["waveform"]
            
            # Validate output
            assert isinstance(waveform, torch.Tensor), "Output should be torch tensor"
            assert waveform.shape[0] == 2, "Should maintain stereo channels"
            assert waveform.shape[1] > 0, "Should have audio samples"
            
            # Calculate expected duration
            expected_samples = int(config["steps"] * config["step_duration"] * audio_output["sample_rate"])
            actual_samples = waveform.shape[1]
            
            print(f"    ✅ {config['name']}: Shape {waveform.shape}, Expected ~{expected_samples} samples, got {actual_samples}")
            
        except Exception as e:
            print(f"    ❌ {config['name']}: Error - {str(e)}")
            return False
    
    print("✅ GranularSequencer tests passed!")
    return True

def test_microsound_sculptor():
    """Test the MicrosoundSculptor node with extreme processing."""
    print("⚡ Testing MicrosoundSculptor...")
    
    # Create test audio
    test_audio = create_test_audio(duration=1.0, noise_type="brown")
    
    # Create microsound sculptor
    sculptor = MicrosoundSculptorNode()
    
    # Test different destruction and sculpting modes
    test_configs = [
        {
            "name": "Bitcrush Destruction",
            "destruction_mode": "bitcrush",
            "destruction_intensity": 0.7,
            "sculpting_mode": "grain_filter",
            "sculpting_intensity": 0.5,
            "grain_size": 50.0,
            "chaos_rate": 15.0,
            "feedback_amount": 0.2,
            "amplitude": 0.8
        },
        {
            "name": "Chaotic Morphing",
            "destruction_mode": "chaos",
            "destruction_intensity": 0.8,
            "sculpting_mode": "grain_morph",
            "sculpting_intensity": 0.6,
            "grain_size": 25.0,
            "chaos_rate": 30.0,
            "feedback_amount": 0.4,
            "amplitude": 0.7,
            "spectral_chaos": 0.3,
            "grain_randomization": 0.8
        },
        {
            "name": "Ring Mod Feedback",
            "destruction_mode": "ring_mod",
            "destruction_intensity": 0.6,
            "sculpting_mode": "grain_feedback",
            "sculpting_intensity": 0.7,
            "grain_size": 75.0,
            "chaos_rate": 20.0,
            "feedback_amount": 0.6,
            "amplitude": 0.6,
            "microsound_density": 100.0
        },
        {
            "name": "Spectral Destruction",
            "destruction_mode": "saturation",
            "destruction_intensity": 0.9,
            "sculpting_mode": "spectral_destroy",
            "sculpting_intensity": 0.8,
            "grain_size": 30.0,
            "chaos_rate": 50.0,
            "feedback_amount": 0.3,
            "amplitude": 0.5,
            "spectral_chaos": 0.7,
            "grain_randomization": 0.9,
            "microsound_density": 200.0
        }
    ]
    
    for i, config in enumerate(test_configs):
        print(f"  Testing: {config['name']}")
        
        try:
            result = sculptor.process_microsound(
                audio=test_audio,
                **{k: v for k, v in config.items() if k != "name"}
            )
            
            audio_output = result[0]
            waveform = audio_output["waveform"]
            
            # Validate output
            assert isinstance(waveform, torch.Tensor), "Output should be torch tensor"
            assert waveform.shape[0] == 2, "Should maintain stereo channels"
            assert waveform.shape[1] > 0, "Should have audio samples"
            assert torch.max(torch.abs(waveform)) <= 1.1, "Should be within reasonable amplitude range"
            
            print(f"    ✅ {config['name']}: Shape {waveform.shape}, Max amplitude {torch.max(torch.abs(waveform)):.3f}")
            
        except Exception as e:
            print(f"    ❌ {config['name']}: Error - {str(e)}")
            return False
    
    print("✅ MicrosoundSculptor tests passed!")
    return True

def test_granular_chain():
    """Test chaining granular processors together."""
    print("🔗 Testing Granular Processing Chain...")
    
    try:
        # Create test audio
        test_audio = create_test_audio(duration=1.0, noise_type="violet")
        
        # Create nodes
        granular = GranularProcessorNode()
        sculptor = MicrosoundSculptorNode()
        sequencer = GranularSequencerNode()
        
        # Chain 1: Granular -> Sculptor
        print("  Testing: Granular -> Microsound Sculptor")
        granular_result = granular.process_granular(
            audio=test_audio,
            grain_size=80.0,
            grain_density=25.0,
            pitch_ratio=1.2,
            grain_envelope="hann",
            positioning_mode="random",
            pitch_mode="transpose",
            grain_scatter=0.4,
            amplitude=0.7
        )
        
        sculptor_result = sculptor.process_microsound(
            audio=granular_result[0],
            destruction_mode="chaos",
            destruction_intensity=0.5,
            sculpting_mode="grain_morph",
            sculpting_intensity=0.4,
            grain_size=40.0,
            chaos_rate=20.0,
            feedback_amount=0.3,
            amplitude=0.8
        )
        
        # Validate chain result
        final_waveform = sculptor_result[0]["waveform"]
        assert isinstance(final_waveform, torch.Tensor), "Chain output should be torch tensor"
        assert final_waveform.shape[0] == 2, "Should maintain stereo channels"
        print(f"    ✅ Chain 1: Final shape {final_waveform.shape}, Max amplitude {torch.max(torch.abs(final_waveform)):.3f}")
        
        # Chain 2: Simple sequencing test
        print("  Testing: Granular Sequencer standalone")
        seq_result = sequencer.process_sequenced_granular(
            audio=test_audio,
            steps=8,
            step_duration=0.125,
            base_grain_size=60.0,
            base_density=12.0,
            pattern_variation=0.4,
            swing=0.1,
            amplitude=0.8,
            probability=0.8
        )
        
        seq_waveform = seq_result[0]["waveform"]
        assert isinstance(seq_waveform, torch.Tensor), "Sequencer output should be torch tensor"
        print(f"    ✅ Sequencer: Shape {seq_waveform.shape}, Max amplitude {torch.max(torch.abs(seq_waveform)):.3f}")
        
    except Exception as e:
        print(f"    ❌ Chain test failed: {str(e)}")
        return False
    
    print("✅ Granular chain tests passed!")
    return True

def run_all_tests():
    """Run all granular synthesis tests."""
    print("🌟 PHASE 2: Granular Synthesis Test Suite")
    print("=" * 60)
    
    tests = [
        test_granular_processor,
        test_granular_sequencer,
        test_microsound_sculptor,
        test_granular_chain
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {str(e)}")
            results.append(False)
            print()
    
    # Summary
    print("=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    test_names = [
        "GranularProcessor",
        "GranularSequencer", 
        "MicrosoundSculptor",
        "Granular Chains"
    ]
    
    passed = sum(results)
    total = len(results)
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL GRANULAR SYNTHESIS TESTS PASSED!")
        print("🌟 Phase 2 granular synthesis engine is ready!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed - needs debugging")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 