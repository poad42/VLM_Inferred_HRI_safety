#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. Define Desired Saw Orientation in World Frame
# X (Length) -> World X [1, 0, 0] (Perpendicular to log which is Y)
# Y (Width) -> World -Z [0, 0, -1] (Cutting edge down)
# Z (Thickness) -> World Y [0, 1, 0] (Along log)
r_saw_des_mat = np.array(
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
).T  # Transpose because vectors are columns?
# Wait, R * e1 = col1.
# R * [1,0,0] = [1,0,0]. Col 1 = [1,0,0].
# R * [0,1,0] = [0,0,-1]. Col 2 = [0,0,-1].
# R * [0,0,1] = [0,1,0]. Col 3 = [0,1,0].
r_saw_des_mat = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
# Check determinant
print(f"Det(R_saw_des): {np.linalg.det(r_saw_des_mat)}")

r_saw_des = R.from_matrix(r_saw_des_mat)
print(f"Desired Saw World Quat: {r_saw_des.as_quat()}")

# 2. Define EE Orientation in World Frame
# Target is RotY(180) -> Fingers pointing down
r_ee = R.from_euler("y", 180, degrees=True)
print(f"EE World Quat: {r_ee.as_quat()}")

# 3. Calculate Local Rotation `local_rot_saw` (Local1)
# Constraint: Frame_EE * I = Frame_Saw * Local1
# Frame_Saw = Frame_EE * inv(Local1)
# R_saw_des = R_ee * inv(R_local1)
# inv(R_local1) = inv(R_ee) * R_saw_des
# R_local1 = inv(inv(R_ee) * R_saw_des)
# R_local1 = (R_ee.inv() * R_saw_des).inv()

r_rel = r_ee.inv() * r_saw_des
r_local1 = r_rel.inv()

quat = r_local1.as_quat()  # x, y, z, w
print(f"\nCalculated local_rot_saw (x,y,z,w): {quat}")
print(f"Gf.Quatf format (w, x, y, z): {quat[3]}, {quat[0]}, {quat[1]}, {quat[2]}")

# Verify
print("\nVerification:")
# Saw = EE * inv(Local1)
r_saw_check = r_ee * r_local1.inv()
print(f"Resulting Saw X: {r_saw_check.apply([1,0,0]).round(3)} (Should be [1,0,0])")
print(f"Resulting Saw Y: {r_saw_check.apply([0,1,0]).round(3)} (Should be [0,0,-1])")
print(f"Resulting Saw Z: {r_saw_check.apply([0,0,1]).round(3)} (Should be [0,1,0])")
