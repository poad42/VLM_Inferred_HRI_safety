import torch
from scipy.spatial.transform import Rotation as R
import numpy as np


def analyze_quat(q_list):
    # q is w, x, y, z
    # scipy uses x, y, z, w
    q_scipy = [q_list[1], q_list[2], q_list[3], q_list[0]]
    r = R.from_quat(q_scipy)
    print(f"Quaternion: {q_list}")
    print(f"Euler (XYZ) degrees: {r.as_euler('xyz', degrees=True)}")
    print(f"Matrix:\n{r.as_matrix()}")

    # Check vectors
    x_vec = r.apply([1, 0, 0])
    y_vec = r.apply([0, 1, 0])
    z_vec = r.apply([0, 0, 1])
    print(f"X axis (Forward) maps to: {x_vec}")
    print(f"Y axis (Left) maps to: {y_vec}")
    print(f"Z axis (Up) maps to: {z_vec}")
    print("-" * 20)


print("Analyzing previous stable quaternion:")
analyze_quat([0.5, -0.5, 0.5, -0.5])

print("Analyzing new unstable quaternion:")
analyze_quat([0.0, 1.0, 0.0, 0.0])
