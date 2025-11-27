#!/usr/bin/env python3
"""
Debug script to understand the saw's actual coordinate frame.
We need to figure out which axis is which in the saw model.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

print("SAW DIMENSIONS: 0.7m × 0.1m × 0.02m")
print("We ASSUME: X=0.7m (length), Y=0.1m (width), Z=0.02m (blade thickness)")
print()

print("CURRENT ROTATION: RotZ(90°) = Quat(0.707, 0, 0, 0.707)")
print("This transforms:")
r = R.from_quat([0, 0, 0.707, 0.707])  # x,y,z,w
print(f"  Saw X (0.7m) → {r.apply([1,0,0]).round(3)}")
print(f"  Saw Y (0.1m) → {r.apply([0,1,0]).round(3)}")
print(f"  Saw Z (0.02m) → {r.apply([0,0,1]).round(3)}")
print()

print("ACTUAL OBSERVED: Saw is lying flat (horizontal)")
print("This suggests the 0.02m thin dimension is NOT what we think!")
print()

print("ALTERNATIVE INTERPRETATION:")
print("What if the saw model has a different axis assignment?")
print("Let's try all possibilities:")
print()

# The saw should be vertical. That means the THIN dimension should be vertical.
# If RotZ(90°) makes it horizontal, then maybe:
# - The thin dimension is actually X or Y, not Z
# - Or the model is oriented differently in its local frame

tests = [
    ("If thin is X-axis", "Need X vertical → rotate to make X point up"),
    ("If thin is Y-axis", "Need Y vertical → rotate to make Y point up"),
    ("If thin is Z-axis", "Need Z vertical → it already is! (current assumption)"),
]

for name, desc in tests:
    print(f"• {name}: {desc}")

print()
print("RECOMMENDATION: Check the saw USD file to see actual dimensions along each axis")
