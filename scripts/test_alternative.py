#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

print("ALTERNATIVE: Make 0.02m blade VERTICAL instead of 0.1m width")
print("Target:")
print("  Saw X (0.7m length) → World Y = [0, 1, 0]  (along log)")
print("  Saw Y (0.1m width)  → World -X = [-1, 0, 0] (into log)")
print("  Saw Z (0.02m blade) → World Z = [0, 0, 1]  (vertical!)")
print()

saw_x = np.array([1, 0, 0])
saw_y = np.array([0, 1, 0])
saw_z = np.array([0, 0, 1])

# Test just RotZ(90°) which looked perfect
print("RotZ(90°):")
r = R.from_euler("z", 90, degrees=True)

result_x = r.apply(saw_x)
result_y = r.apply(saw_y)
result_z = r.apply(saw_z)

print(f"   Saw X → {result_x.round(3)}")
print(f"   Saw Y → {result_y.round(3)}")
print(f"   Saw Z → {result_z.round(3)}")

if (
    np.allclose(result_x, [0, 1, 0], atol=0.01)
    and np.allclose(result_y, [-1, 0, 0], atol=0.01)
    and np.allclose(result_z, [0, 0, 1], atol=0.01)
):
    quat = r.as_quat()  # Returns [x, y, z, w]
    print(f"\n✓✓✓ PERFECT MATCH!")
    print(
        f"Quaternion (x,y,z,w): [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]"
    )
    print(
        f"Quaternion (w,x,y,z): [{quat[3]:.6f}, {quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}]"
    )
    print()
    print("This is a SIMPLE 90° rotation around Z!")
    print(
        "The 0.02m blade thickness is vertical, 0.7m length along log, 0.1m width into log."
    )
