#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

print("KEY INSIGHT: Saw has INITIAL rotation at spawn!")
print("Init rotation: (0.707, 0.707, 0.0, 0.0) = RotX(90°)\n")

# Initial cuboid: X=0.7m, Y=0.1m, Z=0.02m
print("Cuboid dimensions: X=0.7m(length), Y=0.1m(width), Z=0.02m(blade)")
print()

# After initial RotX(90°) spawn rotation
r_spawn = R.from_quat([0.707, 0.0, 0.0, 0.707])  # x,y,z,w format
print("After spawn RotX(90°):")
saw_x_after_spawn = r_spawn.apply([1, 0, 0])
saw_y_after_spawn = r_spawn.apply([0, 1, 0])
saw_z_after_spawn = r_spawn.apply([0, 0, 1])
print(f"  Saw X (0.7m) → {saw_x_after_spawn.round(3)}")
print(f"  Saw Y (0.1m) → {saw_y_after_spawn.round(3)}")
print(f"  Saw Z (0.02m blade) → {saw_z_after_spawn.round(3)}")
print()

# Then we apply RotZ(90°) in attachment
r_attachment = R.from_quat([0, 0, 0.707, 0.707])  # Current attachment rotation
print("Then attachment applies RotZ(90°):")
print("This rotates RELATIVE to the saw's current frame")
print()

# TOTAL rotation = spawn * attachment
r_total = r_attachment * r_spawn
print("TOTAL rotation (spawn ∘ attachment):")
total_x = r_total.apply([1, 0, 0])
total_y = r_total.apply([0, 1, 0])
total_z = r_total.apply([0, 0, 1])
print(f"  Saw X (0.7m) → {total_x.round(3)}")
print(f"  Saw Y (0.1m) → {total_y.round(3)}")
print(f"  Saw Z (0.02m blade) → {total_z.round(3)}")

quat_total = r_total.as_quat()
print(
    f"\nTotal Quat (w,x,y,z): [{quat_total[3]:.6f}, {quat_total[0]:.6f}, {quat_total[1]:.6f}, {quat_total[2]:.6f}]"
)

if np.allclose(total_z, [0, 0, 1], atol=0.01) or np.allclose(
    total_z, [0, 0, -1], atol=0.01
):
    print("\n✓ Blade (Z) is VERTICAL")
else:
    print("\n✗ Blade (Z) is HORIZONTAL - THIS IS THE PROBLEM!")
