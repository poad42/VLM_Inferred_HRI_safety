#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

print("AFTER SPAWN RotX(90°), the saw is:")
print("  Saw X (0.7m) → [1, 0, 0] (world X)")
print("  Saw Y (0.1m) → [0, 0, 1] (world Z - already vertical!)")
print("  Saw Z (0.02m blade) → [0, -1, 0] (world -Y)\n")

print("GOAL: Make blade (saw Z-axis) vertical")
print("Want: Saw Z → [0, 0, 1] or [0, 0, -1]\n")

# Spawn rotation (already applied)
r_spawn = R.from_quat([0.707, 0.0, 0.0, 0.707])  # RotX(90°)

# Test different attachment rotations
tests = [
    ("No attachment rotation (identity)", [1, 0, 0, 0]),  # w,x,y,z
    ("RotX(90°)", R.from_euler("x", 90, degrees=True).as_quat()),
    ("RotX(-90°)", R.from_euler("x", -90, degrees=True).as_quat()),
    ("RotY(90°)", R.from_euler("y", 90, degrees=True).as_quat()),
    ("RotY(-90°)", R.from_euler("y", -90, degrees=True).as_quat()),
    ("RotZ(90°) [CURRENT]", R.from_euler("z", 90, degrees=True).as_quat()),
]

for name, quat_xyzw in tests:
    if isinstance(quat_xyzw, np.ndarray):
        quat_xyzw_arr = quat_xyzw
    else:
        # Convert w,x,y,z to x,y,z,w
        quat_xyzw_arr = np.array(
            [quat_xyzw[1], quat_xyzw[2], quat_xyzw[3], quat_xyzw[0]]
        )

    r_attach = R.from_quat(quat_xyzw_arr)
    r_total = r_attach * r_spawn

    saw_x = r_total.apply([1, 0, 0])
    saw_y = r_total.apply([0, 1, 0])
    saw_z = r_total.apply([0, 0, 1])

    print(f"{name}:")
    print(f"  Saw X (0.7m) → {saw_x.round(3)}")
    print(f"  Saw Y (0.1m) → {saw_y.round(3)}")
    print(f"  Saw Z (blade) → {saw_z.round(3)}", end="")

    if np.allclose(saw_z, [0, 0, 1], atol=0.01):
        quat_total = r_total.as_quat()
        print(" ✓ BLADE VERTICAL UP!")
        print(
            f"  Total Quat(w,x,y,z): [{quat_total[3]:.6f}, {quat_total[0]:.6f}, {quat_total[1]:.6f}, {quat_total[2]:.6f}]"
        )
    elif np.allclose(saw_z, [0, 0, -1], atol=0.01):
        quat_total = r_total.as_quat()
        print(" ✓ BLADE VERTICAL DOWN!")
        print(
            f"  Total Quat(w,x,y,z): [{quat_total[3]:.6f}, {quat_total[0]:.6f}, {quat_total[1]:.6f}, {quat_total[2]:.6f}]"
        )
    else:
        print(" ✗ horizontal")
    print()
