#!/usr/bin/env python3
"""
Find rotation for VERTICAL blade orientation.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

# Saw's local frame
saw_x = np.array([1, 0, 0])  # 0.7m length - want along World Y
saw_y = np.array([0, 1, 0])  # 0.1m width - want along World -Z (down)
saw_z = np.array([0, 0, 1])  # 0.02m blade - want VERTICAL (World X or -X)

print("TARGET FOR VERTICAL BLADE:")
print("  Saw length (X) → World Y = [0, 1, 0]")
print("  Saw width (Y) → World -Z = [0, 0, -1] (down)")
print("  Saw blade (Z) → World X or -X (vertical/sideways)")
print()

# Test RotZ(90°) then RotX(90°)
print("RotZ(90°) THEN RotX(90°):")
r1 = R.from_euler("zx", [90, 90], degrees=True)
print(f"   Saw X → {r1.apply(saw_x)}")
print(f"   Saw Y → {r1.apply(saw_y)}")
print(f"   Saw Z → {r1.apply(saw_z)}")
quat = r1.as_quat()
print(
    f"   Quaternion (w,x,y,z): [{quat[3]:.3f}, {quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}]"
)
print()

# Test RotZ(90°) then RotX(-90°)
print("RotZ(90°) THEN RotX(-90°):")
r2 = R.from_euler("zx", [90, -90], degrees=True)
print(f"   Saw X → {r2.apply(saw_x)}")
print(f"   Saw Y → {r2.apply(saw_y)}")
print(f"   Saw Z → {r2.apply(saw_z)}")
quat = r2.as_quat()
print(
    f"   Quaternion (w,x,y,z): [{quat[3]:.3f}, {quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}]"
)
