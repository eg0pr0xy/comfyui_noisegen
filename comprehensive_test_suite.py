#!/usr/bin/env python3
"""
Comprehensive Test Suite for ComfyUI-NoiseGen
=============================================

Tests all 10 Phase 1 nodes with comprehensive parameter validation,
edge cases, error handling, and audio output verification.

Usage: python comprehensive_test_suite.py
"""

import sys
import os
import traceback
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
import time
import tempfile

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from noise_nodes import *
    from audio_utils import *
    print("✅ Successfully imported noise_nodes and audio_utils")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class ComprehensiveTestSuite:
    """Comprehensive test suite for all NoiseGen nodes."""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
        self.audio_outputs = []
        
        # Initialize all node instances
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
        
        print(f"🧪 Initialized test suite with {len(self.nodes)} nodes")
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results."""
        status = "✅ PASS" if success else "❌ FAIL"
        message = f"{status}: {test_name}"
        if details:
            message += f" - {details}"
        print(message)
        
        self.test_results.append({
            'name': test_name,
            'success': success,
            'details': details
        })
        
        if not success:
            self.failed_tests.append(test_name)
    
    def validate_audio_output(self, audio_dict: Dict, expected_channels: int = None) -> Tuple[bool, str]:
        """Validate audio output format and properties."""
        try:
            # Check basic structure
            if not isinstance(audio_dict, dict):
                return False, f"Output is not a dict: {type(audio_dict)}"
            
            required_keys = ['waveform', 'sample_rate']
            for key in required_keys:
                if key not in audio_dict:
                    return False, f"Missing key: {key}"
            
            waveform = audio_dict['waveform']
            sample_rate = audio_dict['sample_rate']
            
            # Check waveform properties
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.numpy()
            elif not isinstance(waveform, np.ndarray):
                return False, f"Waveform is not tensor or array: {type(waveform)}"
            
            # Check dimensions
            if waveform.ndim != 2:
                return False, f"Waveform should be 2D, got {waveform.ndim}D"
            
            channels, samples = waveform.shape
            
            # Check channel count
            if expected_channels and channels != expected_channels:
                return False, f"Expected {expected_channels} channels, got {channels}"
            
            # Check sample rate
            if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
                return False, f"Invalid sample rate: {sample_rate}"
            
            # Check for NaN or infinite values
            if np.any(np.isnan(waveform)) or np.any(np.isinf(waveform)):
                return False, "Waveform contains NaN or infinite values"
            
            # Check amplitude range (should be reasonable)
            max_amp = np.max(np.abs(waveform))
            if max_amp > 10.0:  # Very generous limit
                return False, f"Amplitude too high: {max_amp}"
            
            return True, f"Valid audio: {channels}ch, {samples} samples, {sample_rate}Hz, max_amp={max_amp:.3f}"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def test_node_defaults(self, node_name: str):
        """Test node with default parameters."""
        print(f"\n--- Testing {node_name} with defaults ---")
        
        try:
            node = self.nodes[node_name]
            input_types = node.INPUT_TYPES()
            
            # Build default parameters
            kwargs = {}
            for param_name, param_spec in input_types['required'].items():
                if param_name == 'audio':
                    # Create test audio for audio input nodes
                    test_audio = self.create_test_audio()
                    kwargs[param_name] = test_audio
                elif param_name == 'audio_a':
                    kwargs[param_name] = self.create_test_audio()
                elif param_name in ['audio_b', 'noise_a', 'noise_b']:
                    kwargs[param_name] = self.create_test_audio()
                else:
                    # Extract default value
                    if isinstance(param_spec, tuple) and len(param_spec) > 1 and isinstance(param_spec[1], dict):
                        default = param_spec[1].get('default')
                        if default is not None:
                            kwargs[param_name] = default
                        elif isinstance(param_spec[0], list):
                            kwargs[param_name] = param_spec[0][0]  # First option
                    elif isinstance(param_spec, tuple):
                        kwargs[param_name] = param_spec[0][0] if isinstance(param_spec[0], list) else param_spec[0]
            
            # Add optional parameters with defaults
            if 'optional' in input_types:
                for param_name, param_spec in input_types['optional'].items():
                    if isinstance(param_spec, tuple) and len(param_spec) > 1 and isinstance(param_spec[1], dict):
                        default = param_spec[1].get('default')
                        if default is not None:
                            kwargs[param_name] = default
            
            # Execute node function
            func_name = node.FUNCTION
            func = getattr(node, func_name)
            
            start_time = time.time()
            result = func(**kwargs)
            execution_time = time.time() - start_time
            
            # Validate result
            if not isinstance(result, tuple):
                self.log_test(f"{node_name}_defaults", False, f"Result not tuple: {type(result)}")
                return None
            
            # Check primary output (should be audio for most nodes)
            primary_output = result[0]
            
            if node_name == 'AudioSave':
                # AudioSave returns (audio, filepath)
                audio_valid, audio_details = self.validate_audio_output(primary_output)
                self.log_test(f"{node_name}_defaults", audio_valid, 
                            f"{audio_details}, exec_time={execution_time:.3f}s")
                return result
            else:
                # Regular audio node
                audio_valid, audio_details = self.validate_audio_output(primary_output)
                self.log_test(f"{node_name}_defaults", audio_valid, 
                            f"{audio_details}, exec_time={execution_time:.3f}s")
                return result
                
        except Exception as e:
            self.log_test(f"{node_name}_defaults", False, f"Exception: {str(e)}")
            traceback.print_exc()
            return None
    
    def test_node_edge_cases(self, node_name: str):
        """Test node with edge case parameters."""
        print(f"\n--- Testing {node_name} edge cases ---")
        
        try:
            node = self.nodes[node_name]
            input_types = node.INPUT_TYPES()
            
            # Test cases: [min_values, max_values]
            test_cases = ['min_values', 'max_values']
            
            for case_name in test_cases:
                kwargs = {}
                
                for param_name, param_spec in input_types['required'].items():
                    if param_name in ['audio', 'audio_a', 'audio_b', 'noise_a', 'noise_b']:
                        kwargs[param_name] = self.create_test_audio()
                    else:
                        # Extract min/max values
                        if isinstance(param_spec, tuple) and len(param_spec) > 1 and isinstance(param_spec[1], dict):
                            spec_dict = param_spec[1]
                            if case_name == 'min_values' and 'min' in spec_dict:
                                kwargs[param_name] = spec_dict['min']
                            elif case_name == 'max_values' and 'max' in spec_dict:
                                kwargs[param_name] = spec_dict['max']
                            else:
                                # Use default
                                default = spec_dict.get('default')
                                if default is not None:
                                    kwargs[param_name] = default
                                elif isinstance(param_spec[0], list):
                                    kwargs[param_name] = param_spec[0][0]
                        elif isinstance(param_spec, tuple):
                            kwargs[param_name] = param_spec[0][0] if isinstance(param_spec[0], list) else param_spec[0]
                
                # Execute test
                try:
                    func = getattr(node, node.FUNCTION)
                    result = func(**kwargs)
                    
                    if isinstance(result, tuple) and len(result) > 0:
                        audio_valid, details = self.validate_audio_output(result[0])
                        self.log_test(f"{node_name}_{case_name}", audio_valid, details)
                    else:
                        self.log_test(f"{node_name}_{case_name}", False, "Invalid result format")
                        
                except Exception as e:
                    self.log_test(f"{node_name}_{case_name}", False, f"Exception: {str(e)}")
                    
        except Exception as e:
            self.log_test(f"{node_name}_edge_cases", False, f"Setup error: {str(e)}")
    
    def test_parameter_types(self, node_name: str):
        """Test parameter type conversion and validation."""
        print(f"\n--- Testing {node_name} parameter types ---")
        
        try:
            node = self.nodes[node_name]
            
            # Special test for NoiseGenerator parameter swapping issue
            if node_name == 'NoiseGenerator':
                print("Testing legacy parameter swapping...")
                
                # Test case 1: Normal parameters
                result1 = node.generate_noise(
                    noise_type='white', duration=1.0, sample_rate=44100, amplitude=0.5,
                    seed=42, channels=1, stereo_mode='independent', stereo_width=1.0
                )
                valid1, details1 = self.validate_audio_output(result1[0], expected_channels=1)
                self.log_test(f"{node_name}_normal_params", valid1, details1)
                
                # Test case 2: Swapped parameters (legacy workflow format)
                result2 = node.generate_noise(
                    noise_type='pink', duration=1.0, sample_rate=44100, amplitude=0.5,
                    seed=42, channels='independent', stereo_mode='1', stereo_width=1.0
                )
                valid2, details2 = self.validate_audio_output(result2[0], expected_channels=1)
                self.log_test(f"{node_name}_swapped_params", valid2, details2)
                
                # Test case 3: String numbers
                result3 = node.generate_noise(
                    noise_type='brown', duration=1.0, sample_rate=44100, amplitude=0.5,
                    seed=42, channels='2', stereo_mode='correlated', stereo_width=1.0
                )
                valid3, details3 = self.validate_audio_output(result3[0], expected_channels=2)
                self.log_test(f"{node_name}_string_numbers", valid3, details3)
                
        except Exception as e:
            self.log_test(f"{node_name}_param_types", False, f"Exception: {str(e)}")
    
    def test_audio_chaining(self):
        """Test chaining nodes together (output of one as input to another)."""
        print(f"\n--- Testing audio chaining ---")
        
        try:
            # Generate noise
            noise_gen = self.nodes['NoiseGenerator']
            noise_result = noise_gen.generate_noise(
                noise_type='white', duration=2.0, sample_rate=44100, amplitude=0.8,
                seed=42, channels=2, stereo_mode='independent', stereo_width=1.0
            )
            
            if not noise_result or len(noise_result) == 0:
                self.log_test("chaining_noise_gen", False, "No noise generated")
                return
            
            noise_audio = noise_result[0]
            valid_noise, details = self.validate_audio_output(noise_audio)
            self.log_test("chaining_noise_gen", valid_noise, details)
            
            if not valid_noise:
                return
            
            # Process through FeedbackProcessor
            feedback_proc = self.nodes['FeedbackProcessor']
            feedback_result = feedback_proc.process_feedback(
                audio=noise_audio, feedback_mode='filtered', feedback_amount=0.3,
                delay_time=2.0, filter_type='lowpass', filter_freq=2000.0,
                filter_resonance=0.3, saturation=0.2, modulation_rate=0.5,
                modulation_depth=0.2, modulation_type='sine', amplitude=0.8
            )
            
            if feedback_result and len(feedback_result) > 0:
                valid_feedback, details = self.validate_audio_output(feedback_result[0])
                self.log_test("chaining_feedback", valid_feedback, details)
                
                # Process through HarshFilter
                if valid_feedback:
                    harsh_filter = self.nodes['HarshFilter']
                    filter_result = harsh_filter.process_harsh_filter(
                        audio=feedback_result[0], filter_type='lowpass', cutoff_freq=1000.0,
                        resonance=0.5, drive=0.3, drive_mode='tube', filter_slope=1.0,
                        morph_amount=0.0, modulation_rate=0.5, modulation_depth=0.1, amplitude=0.8
                    )
                    
                    if filter_result and len(filter_result) > 0:
                        valid_filter, details = self.validate_audio_output(filter_result[0])
                        self.log_test("chaining_harsh_filter", valid_filter, details)
                        
                        # Mix two audio sources
                        mixer = self.nodes['AudioMixer']
                        mix_result = mixer.mix_audio(
                            audio_a=noise_audio, gain_a=0.5, pan_a=-0.3,
                            audio_b=filter_result[0], gain_b=0.5, pan_b=0.3,
                            master_gain=0.8
                        )
                        
                        if mix_result and len(mix_result) > 0:
                            valid_mix, details = self.validate_audio_output(mix_result[0])
                            self.log_test("chaining_mixer", valid_mix, details)
                            
                            # Save final result
                            save_node = self.nodes['AudioSave']
                            save_result = save_node.save_audio(
                                audio=mix_result[0], filename_prefix="test_chain_", format="wav"
                            )
                            
                            if save_result and len(save_result) >= 2:
                                self.log_test("chaining_save", True, f"Saved to: {save_result[1]}")
                            else:
                                self.log_test("chaining_save", False, "Save failed")
            
        except Exception as e:
            self.log_test("audio_chaining", False, f"Exception: {str(e)}")
            traceback.print_exc()
    
    def test_noise_types(self):
        """Test all noise types in NoiseGenerator."""
        print(f"\n--- Testing all noise types ---")
        
        noise_gen = self.nodes['NoiseGenerator']
        noise_types = ['white', 'pink', 'brown', 'blue', 'violet', 'perlin', 'bandlimited']
        
        for noise_type in noise_types:
            try:
                result = noise_gen.generate_noise(
                    noise_type=noise_type, duration=1.0, sample_rate=44100, amplitude=0.7,
                    seed=42, channels=1, stereo_mode='independent', stereo_width=1.0,
                    frequency=2.0, low_freq=200.0, high_freq=8000.0, octaves=4
                )
                
                if result and len(result) > 0:
                    valid, details = self.validate_audio_output(result[0])
                    self.log_test(f"noise_type_{noise_type}", valid, details)
                else:
                    self.log_test(f"noise_type_{noise_type}", False, "No result")
                    
            except Exception as e:
                self.log_test(f"noise_type_{noise_type}", False, f"Exception: {str(e)}")
    
    def create_test_audio(self, duration: float = 1.0, channels: int = 2, sample_rate: int = 44100) -> Dict:
        """Create test audio for nodes that require audio input."""
        samples = int(duration * sample_rate)
        waveform = np.random.normal(0, 0.1, (channels, samples)).astype(np.float32)
        return create_audio_dict(waveform, sample_rate)
    
    def run_all_tests(self):
        """Run the complete test suite."""
        print("🎵 ComfyUI-NoiseGen Comprehensive Test Suite")
        print("=" * 50)
        
        start_time = time.time()
        
        # Test each node with defaults
        print("\n🧪 PHASE 1: Default Parameters")
        for node_name in self.nodes.keys():
            self.test_node_defaults(node_name)
        
        # Test edge cases
        print("\n🧪 PHASE 2: Edge Cases")
        for node_name in self.nodes.keys():
            self.test_node_edge_cases(node_name)
        
        # Test parameter types
        print("\n🧪 PHASE 3: Parameter Type Conversion")
        for node_name in self.nodes.keys():
            self.test_parameter_types(node_name)
        
        # Test noise types
        print("\n🧪 PHASE 4: Noise Type Coverage")
        self.test_noise_types()
        
        # Test audio chaining
        print("\n🧪 PHASE 5: Audio Chaining")
        self.test_audio_chaining()
        
        # Generate report
        self.generate_report(time.time() - start_time)
    
    def generate_report(self, total_time: float):
        """Generate comprehensive test report."""
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results if t['success']])
        failed_tests = total_tests - passed_tests
        
        print(f"📈 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        
        if self.failed_tests:
            print("\n❌ FAILED TESTS:")
            for test_name in self.failed_tests:
                test_result = next(t for t in self.test_results if t['name'] == test_name)
                print(f"   • {test_name}: {test_result['details']}")
        
        print("\n📋 DETAILED RESULTS:")
        for result in self.test_results:
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['name']}: {result['details']}")
        
        # Node-specific summary
        print("\n🎵 NODE SUMMARY:")
        node_stats = {}
        for result in self.test_results:
            node_name = result['name'].split('_')[0]
            if node_name not in node_stats:
                node_stats[node_name] = {'passed': 0, 'total': 0}
            node_stats[node_name]['total'] += 1
            if result['success']:
                node_stats[node_name]['passed'] += 1
        
        for node_name, stats in node_stats.items():
            success_rate = (stats['passed'] / stats['total']) * 100
            status = "✅" if success_rate == 100 else "⚠️" if success_rate >= 80 else "❌"
            print(f"   {status} {node_name}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
        
        print("\n🎯 CONCLUSION:")
        if failed_tests == 0:
            print("   🎉 All tests passed! The NoiseGen system is fully functional.")
        elif failed_tests <= total_tests * 0.1:
            print("   ✅ System is mostly functional with minor issues.")
        else:
            print("   ⚠️  System has significant issues that need attention.")


if __name__ == "__main__":
    print("Starting comprehensive test suite...")
    
    try:
        suite = ComprehensiveTestSuite()
        suite.run_all_tests()
    except KeyboardInterrupt:
        print("\n⚠️ Test suite interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {str(e)}")
        traceback.print_exc() 