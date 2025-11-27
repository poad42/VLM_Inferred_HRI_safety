#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

print("CORRECT GOAL: Saw blade CUTTING INTO LOG")
print("Log is along Y-axis")
print("\nAfter spawn RotX(90°):")
print("  Saw X (0.7m length) → [1, 0, 0] ✓ PERPENDICULAR to log!")
print("  Saw Y (0.1m width) → [0, 0, 1] ✗ pointing UP")
print("  Saw Z (0.02m blade) → [0, -1, 0] ✗ along log\n")

print("GOAL: Make Y (0.1m width) point DOWN for blade edge to cut")
print("      Keep X (0.7m length) perpendicular to log\n")

r_spawn = R.from_quat([0.707, 0.0, 0.0, 0.707])  # Spawn RotX(90°)

tests = [
    ("Identity", R.from_quat([0, 0, 0, 1])),  # No rotation
    ("RotX(90°)", R.from_euler("x", 90, degrees=True)),
    ("RotX(180°)", R.from_euler("x", 180, degrees=True)),
    ("RotX(-90°)", R.from_euler("x", -90, degrees=True)),
]

for name, r_attach in tests:
    r_total = r_attach * r_spawn
    saw_x = r_total.apply([1, 0, 0])
    saw_y = r_total.apply([0, 1, 0])
    saw_z = r_total.apply([0, 0, 1])

    quat = r_attach.as_quat()
    print(
        f"{name}: Quat(w,x,y,z)=[{quat[3]:.3f}, {quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}]"
    )
    print(f"  X (0.7m length)  → {saw_x.round(3)}")
    print(f"  Y (0.1m WIDTH)   → {saw_y.round(3)}", end="")

    # Check if Y (width) points down - this is the cutting edge!
    if np.allclose(saw_y, [0, 0, -1], atol=0.01):
        print(" ✓ CUTTING EDGE DOWN!")
        # Also check X is perpendicular to log (not along Y)
        if np.allclose(saw_x, [1, 0, 0], atol=0.01) or np.allclose(
            saw_x, [-1, 0, 0], atol=0.01
        ):
            print(f"  Z (0.02m blade)  → {saw_z.round(3)}")
            print("  ✓✓✓ PERFECT! Length perpendicular to log, width edge cuts down!")
        else:
            print(f"  Z (0.02m blade)  → {saw_z.round(3)}")
    else:
        print()
        print(f"  Z (0.02m blade)  → {saw_z.round(3)}")
    print()
