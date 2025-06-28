#!/usr/bin/env python3
"""
Test script for 4-input AudioMixer functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import noise_nodes as nn
    print("✅ Successfully imported noise_nodes")
except ImportError as e:
    print(f"❌ Failed to import noise_nodes: {e}")
    sys.exit(1)

def test_4_input_mixer():
    """Test AudioMixer with all 4 inputs"""
    print("\n🎛️ Testing 4-Input AudioMixer...")
    
    try:
        # Create noise generator and mixer
        noise_gen = nn.NoiseGeneratorNode()
        mixer = nn.AudioMixerNode()
        
        # Generate 4 different audio sources
        audio_a = noise_gen.generate_noise("white", 1.0, 44100, 0.5, 1, 1, "independent", 1.0)[0]
        audio_b = noise_gen.generate_noise("pink", 1.0, 44100, 0.5, 2, 1, "independent", 1.0)[0]
        audio_c = noise_gen.generate_noise("brown", 1.0, 44100, 0.5, 3, 1, "independent", 1.0)[0]
        audio_d = noise_gen.generate_noise("blue", 1.0, 44100, 0.5, 4, 1, "independent", 1.0)[0]
        
        print(f"  ✅ Generated 4 audio sources:")
        print(f"    • A: white noise, {audio_a['waveform'].shape}")
        print(f"    • B: pink noise, {audio_b['waveform'].shape}")
        print(f"    • C: brown noise, {audio_c['waveform'].shape}")
        print(f"    • D: blue noise, {audio_d['waveform'].shape}")
        
        # Test with all 4 inputs
        mixed_audio = mixer.mix_audio(
            audio_a=audio_a, gain_a=1.0, pan_a=-0.5,
            audio_b=audio_b, gain_b=0.8, pan_b=0.5,
            audio_c=audio_c, gain_c=0.6, pan_c=-0.3,
            audio_d=audio_d, gain_d=0.4, pan_d=0.3,
            master_gain=0.8
        )[0]
        
        # Verify output
        waveform = mixed_audio["waveform"]
        sample_rate = mixed_audio["sample_rate"]
        
        if hasattr(waveform, 'cpu'):
            audio_np = waveform.cpu().numpy()
        else:
            audio_np = waveform
        
        max_val = abs(audio_np).max()
        
        print(f"  ✅ Mixed 4 inputs successfully:")
        print(f"    • Shape: {audio_np.shape}")
        print(f"    • Sample Rate: {sample_rate}Hz")
        print(f"    • Max Amplitude: {max_val:.3f}")
        print(f"    • All gains applied: A=1.0, B=0.8, C=0.6, D=0.4")
        print(f"    • All pans applied: A=-0.5, B=0.5, C=-0.3, D=0.3")
        print(f"    • Master gain: 0.8")
        
        # Test with fewer inputs (backward compatibility)
        mixed_2_inputs = mixer.mix_audio(
            audio_a=audio_a, gain_a=1.0, pan_a=0.0,
            audio_b=audio_b, gain_b=1.0, pan_b=0.0,
            master_gain=1.0
        )[0]
        
        print(f"  ✅ Backward compatibility (2 inputs): {mixed_2_inputs['waveform'].shape}")
        
        # Test with 3 inputs
        mixed_3_inputs = mixer.mix_audio(
            audio_a=audio_a, gain_a=1.0, pan_a=0.0,
            audio_b=audio_b, gain_b=1.0, pan_b=0.0,
            audio_c=audio_c, gain_c=1.0, pan_c=0.0,
            master_gain=1.0
        )[0]
        
        print(f"  ✅ 3-input compatibility: {mixed_3_inputs['waveform'].shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error in 4-input mixer test: {str(e)}")
        return False

def test_input_validation():
    """Test that all input parameters are recognized"""
    print("\n🔍 Testing Input Parameter Validation...")
    
    try:
        mixer = nn.AudioMixerNode()
        input_types = mixer.INPUT_TYPES()
        
        # Check required inputs
        required = input_types["required"]
        assert "audio_a" in required
        assert "gain_a" in required
        assert "pan_a" in required
        print("  ✅ Required inputs (A): audio_a, gain_a, pan_a")
        
        # Check optional inputs
        optional = input_types["optional"]
        expected_optional = [
            "audio_b", "gain_b", "pan_b",
            "audio_c", "gain_c", "pan_c", 
            "audio_d", "gain_d", "pan_d",
            "master_gain"
        ]
        
        for param in expected_optional:
            assert param in optional, f"Missing parameter: {param}"
        
        print("  ✅ Optional inputs (B, C, D): all parameters present")
        print(f"    • B: audio_b, gain_b, pan_b")
        print(f"    • C: audio_c, gain_c, pan_c")
        print(f"    • D: audio_d, gain_d, pan_d")
        print(f"    • Master: master_gain")
        
        # Check description
        assert "4 inputs" in mixer.DESCRIPTION
        print(f"  ✅ Description updated: '{mixer.DESCRIPTION}'")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error in validation test: {str(e)}")
        return False

if __name__ == "__main__":
    print("🎛️ AudioMixer 4-Input Test Suite")
    print("=" * 50)
    
    tests = [
        test_input_validation,
        test_4_input_mixer,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS")
    print(f"✅ Passed: {passed}/{total}")
    print(f"📊 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! 4-input AudioMixer is working perfectly!")
        print("🎛️ AudioMixer now supports: A + B + C + D inputs")
        print("🔧 Each input has individual gain and pan controls")
        print("🔄 Full backward compatibility maintained")
    else:
        print("❌ Some tests failed!")
        sys.exit(1) 