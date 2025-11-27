#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

print("SAW CUBOID: size=(0.7, 0.1, 0.02)")
print("Saw local X=0.7m, Y=0.1m, Z=0.02m (blade thickness)\n")

print("GOAL: Make Z-axis (0.02m blade) VERTICAL")
print("This means saw Z should point up/down in world frame\n")

saw_z = np.array([0, 0, 1])  # Saw's Z-axis in local frame

tests = [
    ("No rotation (identity)", "", []),
    ("RotX(90°)", "x", [90]),
    ("RotX(-90°)", "x", [-90]),
    ("RotY(90°)", "y", [90]),
    ("RotY(-90°)", "y", [-90]),
    ("RotZ(90°) [CURRENT]", "z", [90]),
]

for name, axes, angles in tests:
    if not axes:
        result_z = saw_z
        print(f"{name}:")
    else:
        r = R.from_euler(axes, angles, degrees=True)
        result_z = r.apply(saw_z)
        quat = r.as_quat()
        print(
            f"{name}: Quat(w,x,y,z)=[{quat[3]:.3f}, {quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}]"
        )

    print(f"  Saw Z-axis → {result_z.round(3)}")

    # Check if Z is vertical (pointing up [0,0,1] or down [0,0,-1])
    if np.allclose(result_z, [0, 0, 1], atol=0.01):
        print("  ✓ Z points UP (vertical)")
    elif np.allclose(result_z, [0, 0, -1], atol=0.01):
        print("  ✓ Z points DOWN (vertical)")
    else:
        print("  ✗ Z is HORIZONTAL")
    print()

print("\nCONCLUSION: RotZ(90°) keeps Z vertical IF it starts vertical.")
print("But the image shows the saw is flat → Z is currently HORIZONTAL.")
print("We need RotX or RotY to make Z vertical!")
