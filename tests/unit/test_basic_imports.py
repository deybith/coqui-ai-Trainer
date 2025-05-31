#!/usr/bin/env python3
"""
Test basic torch imports only to isolate the hanging issue.
"""

import sys
import os
import time

print("Testing basic imports...")

print("1. Testing torch...")
import torch
print("   ✓ torch imported")

print("2. Testing torch.nn...")
import torch.nn as nn
print("   ✓ torch.nn imported")

print("3. Testing torch.nn.functional...")
import torch.nn.functional as F
print("   ✓ torch.nn.functional imported")

print("4. Testing if transformers is available...")
try:
    from transformers import GPT2PreTrainedModel
    print("   ✓ transformers imported successfully")
    has_transformers = True
except ImportError as e:
    print(f"   ✗ transformers not available: {e}")
    has_transformers = False

print("5. Testing basic torch operations...")
x = torch.randn(2, 3)
print(f"   ✓ Created tensor: {x.shape}")

print("6. Testing nn.Module creation...")
class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModule()
print("   ✓ Simple module created")

print("\nAll basic tests passed!")
print(f"Transformers available: {has_transformers}")
