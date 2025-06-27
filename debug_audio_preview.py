#!/usr/bin/env python3
"""
Debug script for AudioPreview functionality
"""

import sys
import os
sys.path.append('.')

try:
    from noise_nodes import AudioPreviewNode
    from audio_utils import generate_white_noise, numpy_to_comfy_audio
    import numpy as np
    
    print("=== AUDIO PREVIEW DEBUG TEST ===")
    print()
    
    # Create a simple test audio
    print("1. Generating test audio...")
    duration = 2.0
    sample_rate = 44100
    amplitude = 0.5
    seed = 42
    
    audio_array = generate_white_noise(duration, sample_rate, amplitude, seed)
    audio_data = numpy_to_comfy_audio(audio_array, sample_rate)
    
    print(f"   ✅ Audio generated: {duration}s, {sample_rate}Hz, amplitude {amplitude}")
    print(f"   📊 Audio shape: {audio_data['waveform'].shape}")
    print(f"   🎵 Audio type: {type(audio_data['waveform'])}")
    
    # Test AudioPreview node
    print("\n2. Testing AudioPreview node...")
    preview_node = AudioPreviewNode()
    
    try:
        result = preview_node.preview_audio(audio_data, "DebugTest_")
        print(f"   ✅ Preview result: {result}")
        
        if "ui" in result and "audio" in result["ui"]:
            audio_files = result["ui"]["audio"]
            if audio_files:
                print(f"   🎧 Audio file for UI: {audio_files[0]}")
                
                # Check if file exists
                import folder_paths
                temp_dir = folder_paths.get_temp_directory() if folder_paths else "/tmp"
                full_path = os.path.join(temp_dir, audio_files[0])
                
                if os.path.exists(full_path):
                    file_size = os.path.getsize(full_path)
                    print(f"   ✅ File exists: {full_path} ({file_size} bytes)")
                else:
                    print(f"   ❌ File not found: {full_path}")
            else:
                print("   ❌ No audio files returned")
        else:
            print("   ❌ Invalid result structure")
            
    except Exception as e:
        print(f"   ❌ Error in preview: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. Checking ComfyUI dependencies...")
    try:
        import folder_paths
        temp_dir = folder_paths.get_temp_directory()
        print(f"   ✅ folder_paths available, temp dir: {temp_dir}")
    except ImportError:
        print("   ❌ folder_paths not available")
    
    try:
        import comfy.model_management
        print("   ✅ comfy.model_management available")
    except ImportError:
        print("   ❌ comfy.model_management not available")
    
    print("\n=== DEBUG COMPLETE ===")
    
except Exception as e:
    print(f"❌ Debug script failed: {e}")
    import traceback
    traceback.print_exc() 