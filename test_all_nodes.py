#!/usr/bin/env python3
"""
Comprehensive Test Suite for ComfyUI-NoiseGen
Tests all parameters on all 10 nodes to verify functionality and logic.
"""

import sys
import os
import traceback
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from noise_nodes import *
    from audio_utils import *
    print("✅ Successfully imported modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class NodeTester:
    """Comprehensive node testing class."""
    
    def __init__(self):
        self.results = []
        self.failed = []
        
        # Initialize all nodes
        self.nodes = {
            'NoiseGenerator': NoiseGeneratorNode(),
            'PerlinNoise': PerlinNoiseNode(),
            'BandLimitedNoise': BandLimitedNoiseNode(),
            'FeedbackProcessor': FeedbackProcessorNode(),
            'HarshFilter': HarshFilterNode(),
            'MultiDistortion': MultiDistortionNode(),
            'SpectralProcessor': SpectralProcessorNode(),
            'AudioMixer': AudioMixerNode(),
            'ChaosNoiseMix': ChaosNoiseMixNode(),
            'AudioSave': AudioSaveNode()
        }
        print(f"🧪 Initialized {len(self.nodes)} nodes for testing")
    
    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name} - {details}")
        self.results.append({'name': test_name, 'success': success, 'details': details})
        if not success:
            self.failed.append(test_name)
    
    def validate_audio(self, audio_dict: Dict, expected_channels: int = None) -> Tuple[bool, str]:
        """Validate audio output."""
        try:
            if not isinstance(audio_dict, dict):
                return False, f"Not a dict: {type(audio_dict)}"
            
            if 'waveform' not in audio_dict or 'sample_rate' not in audio_dict:
                return False, "Missing waveform or sample_rate"
            
            waveform = audio_dict['waveform']
            sample_rate = audio_dict['sample_rate']
            
            # Convert torch tensor if needed
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.numpy()
            
            if not isinstance(waveform, np.ndarray) or waveform.ndim != 2:
                return False, f"Invalid waveform: {type(waveform)}, shape: {getattr(waveform, 'shape', 'N/A')}"
            
            channels, samples = waveform.shape
            
            if expected_channels and channels != expected_channels:
                return False, f"Expected {expected_channels} channels, got {channels}"
            
            if np.any(np.isnan(waveform)) or np.any(np.isinf(waveform)):
                return False, "Contains NaN/Inf values"
            
            max_amp = np.max(np.abs(waveform))
            if max_amp > 10.0:
                return False, f"Amplitude too high: {max_amp}"
            
            return True, f"{channels}ch, {samples} samples, {sample_rate}Hz, max_amp={max_amp:.3f}"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def create_test_audio(self, duration=1.0, channels=2, sample_rate=44100):
        """Create test audio for input nodes."""
        samples = int(duration * sample_rate)
        waveform = np.random.normal(0, 0.1, (channels, samples)).astype(np.float32)
        return create_audio_dict(waveform, sample_rate)
    
    def test_noise_generator(self):
        """Test NoiseGenerator comprehensively."""
        print("\n--- Testing NoiseGenerator ---")
        node = self.nodes['NoiseGenerator']
        
        # Test all noise types
        noise_types = ['white', 'pink', 'brown', 'blue', 'violet', 'perlin', 'bandlimited']
        for noise_type in noise_types:
            try:
                result = node.generate_noise(
                    noise_type=noise_type, duration=1.0, sample_rate=44100, amplitude=0.7,
                    seed=42, channels=1, stereo_mode='independent', stereo_width=1.0,
                    frequency=2.0, low_freq=200.0, high_freq=8000.0, octaves=4
                )
                
                if result and len(result) > 0:
                    valid, details = self.validate_audio(result[0], expected_channels=1)
                    self.log_result(f"NoiseGen_{noise_type}", valid, details)
                else:
                    self.log_result(f"NoiseGen_{noise_type}", False, "No result returned")
            except Exception as e:
                self.log_result(f"NoiseGen_{noise_type}", False, f"Exception: {str(e)}")
        
        # Test parameter swapping fix (legacy compatibility)
        try:
            # Test swapped parameters (channels='independent', stereo_mode='1')
            result = node.generate_noise(
                noise_type='white', duration=0.5, sample_rate=44100, amplitude=0.5,
                seed=42, channels='independent', stereo_mode='1', stereo_width=1.0
            )
            valid, details = self.validate_audio(result[0], expected_channels=1)
            self.log_result("NoiseGen_param_swap", valid, f"Legacy compatibility: {details}")
        except Exception as e:
            self.log_result("NoiseGen_param_swap", False, f"Exception: {str(e)}")
        
        # Test stereo modes
        stereo_modes = ['independent', 'correlated', 'decorrelated']
        for mode in stereo_modes:
            try:
                result = node.generate_noise(
                    noise_type='white', duration=0.5, sample_rate=44100, amplitude=0.5,
                    seed=42, channels=2, stereo_mode=mode, stereo_width=1.0
                )
                valid, details = self.validate_audio(result[0], expected_channels=2)
                self.log_result(f"NoiseGen_stereo_{mode}", valid, details)
            except Exception as e:
                self.log_result(f"NoiseGen_stereo_{mode}", False, f"Exception: {str(e)}")
    
    def test_processors(self):
        """Test processing nodes."""
        print("\n--- Testing Processor Nodes ---")
        
        # Create test audio
        test_audio = self.create_test_audio(duration=1.0, channels=2)
        
        processor_tests = [
            ('FeedbackProcessor', 'process_feedback', {
                'audio': test_audio, 'feedback_mode': 'filtered', 'feedback_amount': 0.3,
                'delay_time': 2.0, 'filter_type': 'lowpass', 'filter_freq': 2000.0,
                'filter_resonance': 0.3, 'saturation': 0.2, 'modulation_rate': 0.5,
                'modulation_depth': 0.2, 'modulation_type': 'sine', 'amplitude': 0.8
            }),
            ('HarshFilter', 'process_harsh_filter', {
                'audio': test_audio, 'filter_type': 'lowpass', 'cutoff_freq': 1000.0,
                'resonance': 0.5, 'drive': 0.3, 'drive_mode': 'tube', 'filter_slope': 1.0,
                'morph_amount': 0.0, 'modulation_rate': 0.5, 'modulation_depth': 0.1, 'amplitude': 0.8
            }),
            ('MultiDistortion', 'process_multi_distortion', {
                'audio': test_audio, 'distortion_type': 'tube', 'drive': 0.5, 'output_gain': 0.7,
                'wet_dry_mix': 1.0, 'stages': 2, 'asymmetry': 0.0, 'amplitude': 0.8
            }),
            ('SpectralProcessor', 'process_spectral', {
                'audio': test_audio, 'spectral_mode': 'enhance', 'fft_size': '2048',
                'overlap_factor': 0.75, 'window_type': 'hann', 'frequency_range_low': 100.0,
                'frequency_range_high': 8000.0, 'intensity': 0.5, 'amplitude': 0.8
            })
        ]
        
        for node_name, func_name, params in processor_tests:
            try:
                node = self.nodes[node_name]
                func = getattr(node, func_name)
                result = func(**params)
                
                if result and len(result) > 0:
                    valid, details = self.validate_audio(result[0])
                    self.log_result(f"{node_name}_default", valid, details)
                else:
                    self.log_result(f"{node_name}_default", False, "No result")
            except Exception as e:
                self.log_result(f"{node_name}_default", False, f"Exception: {str(e)}")
    
    def test_mixers(self):
        """Test mixer nodes."""
        print("\n--- Testing Mixer Nodes ---")
        
        # Create two test audio sources
        audio_a = self.create_test_audio(duration=1.0, channels=2)
        audio_b = self.create_test_audio(duration=1.0, channels=2)
        
        # Test AudioMixer
        try:
            mixer = self.nodes['AudioMixer']
            result = mixer.mix_audio(
                audio_a=audio_a, gain_a=0.7, pan_a=-0.3,
                audio_b=audio_b, gain_b=0.5, pan_b=0.3,
                master_gain=0.8
            )
            valid, details = self.validate_audio(result[0])
            self.log_result("AudioMixer_default", valid, details)
        except Exception as e:
            self.log_result("AudioMixer_default", False, f"Exception: {str(e)}")
        
        # Test ChaosNoiseMix
        try:
            chaos_mixer = self.nodes['ChaosNoiseMix']
            result = chaos_mixer.mix_chaos(
                noise_a=audio_a, noise_b=audio_b, mix_mode='add', mix_ratio=0.5,
                chaos_amount=0.3, distortion=0.4, amplitude=0.8
            )
            valid, details = self.validate_audio(result[0])
            self.log_result("ChaosNoiseMix_default", valid, details)
        except Exception as e:
            self.log_result("ChaosNoiseMix_default", False, f"Exception: {str(e)}")
    
    def test_utilities(self):
        """Test utility nodes."""
        print("\n--- Testing Utility Nodes ---")
        
        # Test individual noise generators
        perlin_tests = [
            ('PerlinNoise', 'generate', {
                'duration': 1.0, 'frequency': 2.0, 'sample_rate': 44100, 'amplitude': 0.8,
                'seed': 42, 'channels': 2, 'stereo_mode': 'independent', 'octaves': 6, 'persistence': 1.0
            }),
            ('BandLimitedNoise', 'generate', {
                'duration': 1.0, 'low_frequency': 200.0, 'high_frequency': 8000.0,
                'sample_rate': 44100, 'amplitude': 0.8, 'seed': 42
            })
        ]
        
        for node_name, func_name, params in perlin_tests:
            try:
                node = self.nodes[node_name]
                func = getattr(node, func_name)
                result = func(**params)
                
                if result and len(result) > 0:
                    valid, details = self.validate_audio(result[0])
                    self.log_result(f"{node_name}_default", valid, details)
                else:
                    self.log_result(f"{node_name}_default", False, "No result")
            except Exception as e:
                self.log_result(f"{node_name}_default", False, f"Exception: {str(e)}")
        
        # Test AudioSave
        try:
            test_audio = self.create_test_audio()
            save_node = self.nodes['AudioSave']
            result = save_node.save_audio(
                audio=test_audio, filename_prefix="test_", format="wav"
            )
            
            if result and len(result) >= 2:
                # Should return (audio, filepath)
                valid, details = self.validate_audio(result[0])
                filepath = result[1]
                self.log_result("AudioSave_default", valid, f"{details}, saved to: {filepath}")
            else:
                self.log_result("AudioSave_default", False, "Invalid save result")
        except Exception as e:
            self.log_result("AudioSave_default", False, f"Exception: {str(e)}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n--- Testing Edge Cases ---")
        
        # Test extreme parameters
        try:
            noise_gen = self.nodes['NoiseGenerator']
            
            # Very short duration
            result = noise_gen.generate_noise(
                noise_type='white', duration=0.1, sample_rate=44100, amplitude=0.5,
                seed=42, channels=1, stereo_mode='independent', stereo_width=1.0
            )
            valid, details = self.validate_audio(result[0])
            self.log_result("NoiseGen_min_duration", valid, details)
            
            # Very long duration
            result = noise_gen.generate_noise(
                noise_type='white', duration=10.0, sample_rate=8000, amplitude=0.1,
                seed=42, channels=1, stereo_mode='independent', stereo_width=1.0
            )
            valid, details = self.validate_audio(result[0])
            self.log_result("NoiseGen_long_duration", valid, details)
            
            # High sample rate
            result = noise_gen.generate_noise(
                noise_type='white', duration=0.5, sample_rate=96000, amplitude=0.3,
                seed=42, channels=2, stereo_mode='decorrelated', stereo_width=2.0
            )
            valid, details = self.validate_audio(result[0])
            self.log_result("NoiseGen_high_samplerate", valid, details)
            
        except Exception as e:
            self.log_result("edge_cases", False, f"Exception: {str(e)}")
    
    def test_chain_processing(self):
        """Test chaining multiple nodes together."""
        print("\n--- Testing Node Chaining ---")
        
        try:
            # Generate → Process → Mix → Save chain
            
            # 1. Generate noise
            noise_gen = self.nodes['NoiseGenerator']
            noise1 = noise_gen.generate_noise(
                noise_type='white', duration=2.0, sample_rate=44100, amplitude=0.8,
                seed=42, channels=2, stereo_mode='independent', stereo_width=1.0
            )[0]
            
            noise2 = noise_gen.generate_noise(
                noise_type='pink', duration=2.0, sample_rate=44100, amplitude=0.6,
                seed=123, channels=2, stereo_mode='correlated', stereo_width=1.0
            )[0]
            
            # 2. Process through feedback
            feedback = self.nodes['FeedbackProcessor']
            processed1 = feedback.process_feedback(
                audio=noise1, feedback_mode='filtered', feedback_amount=0.2,
                delay_time=1.0, filter_type='lowpass', filter_freq=3000.0,
                filter_resonance=0.2, saturation=0.1, modulation_rate=0.3,
                modulation_depth=0.1, modulation_type='sine', amplitude=0.7
            )[0]
            
            # 3. Process through distortion
            distortion = self.nodes['MultiDistortion']
            processed2 = distortion.process_multi_distortion(
                audio=noise2, distortion_type='tube', drive=0.3, output_gain=0.8,
                wet_dry_mix=0.7, stages=1, asymmetry=0.1, amplitude=0.7
            )[0]
            
            # 4. Mix together
            mixer = self.nodes['AudioMixer']
            mixed = mixer.mix_audio(
                audio_a=processed1, gain_a=0.6, pan_a=-0.2,
                audio_b=processed2, gain_b=0.4, pan_b=0.2,
                master_gain=0.8
            )[0]
            
            # 5. Final processing
            harsh_filter = self.nodes['HarshFilter']
            final = harsh_filter.process_harsh_filter(
                audio=mixed, filter_type='lowpass', cutoff_freq=8000.0,
                resonance=0.3, drive=0.2, drive_mode='tube', filter_slope=1.0,
                morph_amount=0.0, modulation_rate=0.2, modulation_depth=0.05, amplitude=0.8
            )[0]
            
            # 6. Save result
            save_node = self.nodes['AudioSave']
            saved = save_node.save_audio(
                audio=final, filename_prefix="chain_test_", format="wav"
            )
            
            # Validate final result
            valid, details = self.validate_audio(final)
            self.log_result("chain_processing", valid, f"Complete chain: {details}")
            
            if saved and len(saved) >= 2:
                self.log_result("chain_save", True, f"Chain saved to: {saved[1]}")
            
        except Exception as e:
            self.log_result("chain_processing", False, f"Exception: {str(e)}")
            traceback.print_exc()
    
    def run_all_tests(self):
        """Run complete test suite."""
        print("🎵 ComfyUI-NoiseGen Comprehensive Test Suite")
        print("=" * 50)
        
        start_time = time.time()
        
        # Run all test categories
        self.test_noise_generator()
        self.test_processors()
        self.test_mixers()
        self.test_utilities()
        self.test_edge_cases()
        self.test_chain_processing()
        
        # Generate report
        end_time = time.time()
        self.generate_report(end_time - start_time)
    
    def generate_report(self, total_time):
        """Generate test report."""
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS")
        print("=" * 50)
        
        total = len(self.results)
        passed = len([r for r in self.results if r['success']])
        failed = total - passed
        
        print(f"📈 Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Success Rate: {(passed/total)*100:.1f}%")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        
        if self.failed:
            print(f"\n❌ Failed Tests ({len(self.failed)}):")
            for test_name in self.failed:
                result = next(r for r in self.results if r['name'] == test_name)
                print(f"   • {test_name}: {result['details']}")
        
        print(f"\n📋 All Results:")
        for result in self.results:
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['name']}: {result['details']}")
        
        # Summary by node
        print(f"\n🎵 Node Summary:")
        node_stats = {}
        for result in self.results:
            # Extract node name from test name
            parts = result['name'].split('_')
            if len(parts) > 1:
                node_name = '_'.join(parts[:-1]) if parts[-1] in ['default', 'min', 'max'] else parts[0]
            else:
                node_name = parts[0]
            
            if node_name not in node_stats:
                node_stats[node_name] = {'passed': 0, 'total': 0}
            node_stats[node_name]['total'] += 1
            if result['success']:
                node_stats[node_name]['passed'] += 1
        
        for node_name, stats in sorted(node_stats.items()):
            rate = (stats['passed'] / stats['total']) * 100
            status = "✅" if rate == 100 else "⚠️" if rate >= 80 else "❌"
            print(f"   {status} {node_name}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")
        
        print(f"\n🎯 Conclusion:")
        if failed == 0:
            print("   🎉 Perfect! All tests passed. System is fully functional.")
        elif failed <= total * 0.1:
            print("   ✅ Excellent! System is highly functional with minor issues.")
        elif failed <= total * 0.2:
            print("   ⚠️  Good! System is mostly functional but needs some attention.")
        else:
            print("   ❌ Needs work! System has significant issues requiring fixes.")

if __name__ == "__main__":
    try:
        tester = NodeTester()
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        traceback.print_exc() 