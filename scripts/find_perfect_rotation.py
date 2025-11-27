#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

print("TARGET: Saw length along Y (log), blade vertical")
print("After spawn RotX(90°): X→[1,0,0], Y→[0,0,1], Z→[0,-1,0]\n")

r_spawn = R.from_quat([0.707, 0.0, 0.0, 0.707])  # RotX(90°)

# Need: Saw X → world Y, Saw Z → world Z (or -Z)
# Test combinations
tests = [
    ("RotX(-90°) ∘ RotZ(90°)", "zx", [90, -90]),
    ("RotZ(90°) ∘ RotX(-90°)", "xz", [-90, 90]),
    ("RotX(90°) ∘ RotZ(90°)", "zx", [90, 90]),
    ("RotZ(90°) ∘ RotX(90°)", "xz", [90, 90]),
    ("RotZ(-90°) ∘ RotX(-90°)", "xz", [-90, -90]),
    ("RotZ(-90°) ∘ RotX(90°)", "xz", [90, -90]),
]

for name, axes, angles in tests:
    r_attach = R.from_euler(axes, angles, degrees=True)
    r_total = r_attach * r_spawn

    saw_x = r_total.apply([1, 0, 0])
    saw_y = r_total.apply([0, 1, 0])
    saw_z = r_total.apply([0, 0, 1])

    # Check if: X along Y AND Z vertical
    x_along_y = np.allclose(saw_x, [0, 1, 0], atol=0.01) or np.allclose(
        saw_x, [0, -1, 0], atol=0.01
    )
    z_vertical = np.allclose(saw_z, [0, 0, 1], atol=0.01) or np.allclose(
        saw_z, [0, 0, -1], atol=0.01
    )

    if x_along_y and z_vertical:
        quat_attach = r_attach.as_quat()  # x,y,z,w
        quat_total = r_total.as_quat()
        print(f"✓✓✓ {name}")
        print(f"  Saw X (0.7m) → {saw_x.round(3)} (along log!)")
        print(f"  Saw Y (0.1m) → {saw_y.round(3)}")
        print(f"  Saw Z (blade) → {saw_z.round(3)} (vertical!)")
        print(
            f"  Attachment Quat(w,x,y,z): [{quat_attach[3]:.6f}, {quat_attach[0]:.6f}, {quat_attach[1]:.6f}, {quat_attach[2]:.6f}]"
        )
        print(
            f"  Total Quat(w,x,y,z): [{quat_total[3]:.6f}, {quat_total[0]:.6f}, {quat_total[1]:.6f}, {quat_total[2]:.6f}]"
        )
        print()
