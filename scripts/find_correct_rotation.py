#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

saw_x = np.array([1, 0, 0])
saw_y = np.array([0, 1, 0])
saw_z = np.array([0, 0, 1])

print("TARGET: X→[0,1,0], Y→[0,0,1], Z→[-1,0,0]\n")

# Test 2-axis and 3-axis rotations
tests = [
    ("RotY(-90°) RotX(90°)", "yx", [-90, 90]),
    ("RotX(90°) RotY(-90°)", "xy", [90, -90]),
    ("RotY(180°) RotX(-90°) RotZ(90°)", "yxz", [180, -90, 90]),
    ("RotZ(90°) RotY(180°) RotX(-90°)", "zyx", [90, 180, -90]),
    ("RotX(90°) RotZ(-90°) RotY(180°)", "xzy", [90, -90, 180]),
    ("RotY(-90°) RotZ(90°)", "yz", [-90, 90]),
    ("RotZ(90°) RotY(-90°)", "zy", [90, -90]),
]

for name, axes, angles in tests:
    print(f"{name}:")
    r = R.from_euler(axes, angles, degrees=True)
    result_x = r.apply(saw_x)
    result_y = r.apply(saw_y)
    result_z = r.apply(saw_z)
    print(
        f"   X → {result_x.round(3)}, Y → {result_y.round(3)}, Z → {result_z.round(3)}"
    )

    if (
        np.allclose(result_x, [0, 1, 0], atol=0.01)
        and np.allclose(result_y, [0, 0, 1], atol=0.01)
        and np.allclose(result_z, [-1, 0, 0], atol=0.01)
    ):
        quat = r.as_quat()
        print(
            f"   ✓✓✓ PERFECT! Quat (w,x,y,z): [{quat[3]:.3f}, {quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}]"
        )
    print()
