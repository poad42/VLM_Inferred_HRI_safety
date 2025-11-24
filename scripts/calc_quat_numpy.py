import numpy as np
from scipy.spatial.transform import Rotation as R

# Target Rotation Matrix
# Col 0: Saw X in EE frame -> [0, 0, 1] (EE Z)
# Col 1: Saw Y in EE frame -> [0, -1, 0] (-EE Y)
# Col 2: Saw Z in EE frame -> [1, 0, 0] (EE X)
matrix = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])

r = R.from_matrix(matrix)
quat = r.as_quat()  # x, y, z, w
print(f"Quaternion (x, y, z, w): {quat}")
print(f"Quaternion (w, x, y, z): {[quat[3], quat[0], quat[1], quat[2]]}")
print(f"Euler: {r.as_euler('xyz', degrees=True)}")
