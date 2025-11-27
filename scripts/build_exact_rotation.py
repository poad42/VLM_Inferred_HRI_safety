#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

print("ISSUE: The target axes form a left-handed system!")
print("If saw X → [0,1,0] and saw Y → [0,0,1]")
print("Then by right-hand rule, saw Z MUST → [0,1,0] × [0,0,1] = [1,0,0]")
print("But we want saw Z → [-1,0,0]")
print()

# Let's find rotations that match X and Y, and see what Z becomes
tests = [
    ("RotZ(90°) RotY(-90°)", "zy", [90, -90]),
    ("RotY(-90°) RotX(90°)", "yx", [-90, 90]),
]

for name, axes, angles in tests:
    print(f"{name}:")
    r = R.from_euler(axes, angles, degrees=True)

    saw_x = np.array([1, 0, 0])
    saw_y = np.array([0, 1, 0])
    saw_z = np.array([0, 0, 1])

    result_x = r.apply(saw_x)
    result_y = r.apply(saw_y)
    result_z = r.apply(saw_z)

    print(f"   Saw X → {result_x.round(3)}")
    print(f"   Saw Y → {result_y.round(3)}")
    print(f"   Saw Z → {result_z.round(3)} (must be [1,0,0] by right-hand rule)")

    if np.allclose(result_x, [0, 1, 0], atol=0.01) and np.allclose(
        result_y, [0, 0, 1], atol=0.01
    ):
        quat = r.as_quat()
        print(
            f"   ✓ BEST POSSIBLE! Quat (w,x,y,z): [{quat[3]:.6f}, {quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}]"
        )
        print(f"   NOTE: Blade points +X (away from log) due to right-hand rule")
    print()

print("\nOPTIONS:")
print(
    "1. Accept Z→[1,0,0] (blade away from log) - may still work if we move EE to other side"
)
print("2. Flip the saw model's geometry in the USD file")
print("3. Reconsider which saw dimension should be vertical")
