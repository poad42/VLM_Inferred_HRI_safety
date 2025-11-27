#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. Define Desired Saw Orientation in World Frame (FLIPPED)
# Previous: X->[1,0,0], Y->[0,0,-1], Z->[0,1,0]
# New (Flipped 180 around vertical/cutting plane):
# X (Length) -> World -X [-1, 0, 0] (Flipped direction)
# Y (Width) -> World -Z [0, 0, -1] (Still cutting edge down)
# Z (Thickness) -> World -Y [0, -1, 0] (Flipped to maintain right-hand rule)

r_saw_des_mat = np.array(
    [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]  # Wait, Y is col 2.  # Z is col 3.
)
# Re-verify columns:
# Col 1 (X): [-1, 0, 0]
# Col 2 (Y): [0, 0, -1]
# Col 3 (Z): [0, -1, 0]
r_saw_des_mat = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]).T

print(f"Det(R_saw_des): {np.linalg.det(r_saw_des_mat)}")

r_saw_des = R.from_matrix(r_saw_des_mat)
print(f"Desired Saw World Quat: {r_saw_des.as_quat()}")

# 2. Define EE Orientation in World Frame
# Target is RotY(180) -> Fingers pointing down
r_ee = R.from_euler("y", 180, degrees=True)
print(f"EE World Quat: {r_ee.as_quat()}")

# 3. Calculate Local Rotation `local_rot_saw`
# R_local1 = (R_ee.inv() * R_saw_des).inv()
# Actually: Saw = EE * Local
# Local = inv(EE) * Saw
r_local = r_ee.inv() * r_saw_des

quat = r_local.as_quat()  # x, y, z, w
print(f"\nCalculated local_rot_saw (x,y,z,w): {quat}")
print(f"Gf.Quatf format (w, x, y, z): {quat[3]}, {quat[0]}, {quat[1]}, {quat[2]}")

# Verify
print("\nVerification:")
r_saw_check = r_ee * r_local
print(f"Resulting Saw X: {r_saw_check.apply([1,0,0]).round(3)} (Should be [-1,0,0])")
print(f"Resulting Saw Y: {r_saw_check.apply([0,1,0]).round(3)} (Should be [0,0,-1])")
print(f"Resulting Saw Z: {r_saw_check.apply([0,0,1]).round(3)} (Should be [0,-1,0])")
