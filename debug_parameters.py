#!/usr/bin/env python3
"""
Debug script to verify NoiseGenerator parameter order
"""

import sys
sys.path.append('.')

try:
    from noise_nodes import NoiseGeneratorNode
    
    print("=== NOISEGEN PARAMETER DEBUG ===")
    print()
    
    # Get the INPUT_TYPES
    input_types = NoiseGeneratorNode.INPUT_TYPES()
    required_params = input_types["required"]
    optional_params = input_types.get("optional", {})
    
    print("REQUIRED PARAMETERS (in order):")
    for i, (param_name, param_config) in enumerate(required_params.items()):
        print(f"  {i+1:2d}. {param_name:15s} = {param_config}")
    
    print()
    print("OPTIONAL PARAMETERS:")
    for i, (param_name, param_config) in enumerate(optional_params.items()):
        print(f"  {i+1:2d}. {param_name:15s} = {param_config}")
    
    print()
    print("CURRENT WORKFLOW VALUES:")
    current_values = ["white", 3.0, 44100, 0.5, 42, 1, "independent", 1.0, 1.0, 100.0, 8000.0, 4]
    
    # Map current values to parameters
    all_params = list(required_params.keys()) + list(optional_params.keys())
    
    print("Expected mapping:")
    for i, (param_name, value) in enumerate(zip(all_params, current_values)):
        param_info = required_params.get(param_name, optional_params.get(param_name, "UNKNOWN"))
        print(f"  {i+1:2d}. {param_name:15s} = {value!r:15s} (expected: {param_info})")
    
    print()
    print("VALIDATION CHECK:")
    
    # Check channels parameter
    channels_config = required_params.get("channels", ["MISSING"])
    print(f"Channels expects: {channels_config}")
    print(f"Channels getting: {current_values[5]} (position 6)")
    
    # Check stereo_mode parameter  
    stereo_config = required_params.get("stereo_mode", ["MISSING"])
    print(f"Stereo_mode expects: {stereo_config}")
    print(f"Stereo_mode getting: {current_values[6]} (position 7)")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 