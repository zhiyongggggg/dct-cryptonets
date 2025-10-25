#!/usr/bin/env python3
"""
Verify GPU setup for Concrete ML
Save as: verify_gpu_setup.py
Run: python verify_gpu_setup.py
"""

import os
import sys

print("=" * 60)
print("GPU Setup Verification for Concrete ML")
print("=" * 60)

# Check 1: PyTorch CUDA
print("\n1. Checking PyTorch CUDA...")
try:
    import torch
    print(f"   ✓ PyTorch version: {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA version: {torch.version.cuda}")
        print(f"   ✓ GPU count: {torch.cuda.device_count()}")
        print(f"   ✓ GPU name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Check 2: Concrete ML
print("\n2. Checking Concrete ML...")
try:
    import concrete
    from concrete.fhe import Configuration
    print(f"   ✓ Concrete version: {concrete.__version__}")
    
    # Try creating a configuration
    config = Configuration(enable_unsafe_features=True)
    print(f"   ✓ Configuration created successfully")
    print(f"   ✓ enable_unsafe_features: {config.enable_unsafe_features}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Check 3: Environment variables
print("\n3. Checking environment variables...")
env_vars = {
    'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'),
    'CONCRETE_USE_GPU': os.environ.get('CONCRETE_USE_GPU', 'Not set'),
    'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', 'Not set'),
}
for key, value in env_vars.items():
    print(f"   {key}: {value}")

# Check 4: CUDA libraries
print("\n4. Checking CUDA libraries...")
try:
    import ctypes
    try:
        ctypes.CDLL('libcudart.so')
        print("   ✓ libcudart.so found")
    except:
        print("   ✗ libcudart.so not found")
    
    try:
        ctypes.CDLL('libcublas.so')
        print("   ✓ libcublas.so found")
    except:
        print("   ✗ libcublas.so not found")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Check 5: Brevitas
print("\n5. Checking Brevitas...")
try:
    import brevitas
    print(f"   ✓ Brevitas version: {brevitas.__version__}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Check 6: Concrete ML compilation test
print("\n6. Testing simple Concrete ML compilation...")
try:
    from concrete.ml.torch.compile import compile_torch_model
    import torch.nn as nn
    
    # Simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleModel()
    dummy_input = torch.randn(1, 10)
    
    config = Configuration(
        enable_unsafe_features=True,
        show_progress=False
    )
    
    print("   Compiling test model...")
    q_module = compile_torch_model(
        model,
        dummy_input,
        n_bits=4,
        configuration=config,
        verbose=False
    )
    print("   ✓ Compilation successful!")
    
except Exception as e:
    print(f"   ✗ Compilation failed: {e}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if torch.cuda.is_available():
    print("✓ GPU is available for PyTorch")
else:
    print("✗ GPU is NOT available for PyTorch")

if os.environ.get('CONCRETE_USE_GPU') == '1':
    print("✓ GPU is enabled for Concrete ML (via CONCRETE_USE_GPU=1)")
else:
    print("✗ GPU is NOT enabled for Concrete ML")
    print("  Set: export CONCRETE_USE_GPU=1")

print("\n" + "=" * 60)