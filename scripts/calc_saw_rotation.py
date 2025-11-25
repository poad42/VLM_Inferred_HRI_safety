"""
Calculate the correct quaternion for saw spawn orientation.

Saw dimensions: (0.7, 0.1, 0.02) = (Length, Width/Blade, Thickness)

DESIRED FINAL ORIENTATION:
- Saw X (length, 0.7m) → World Y (perpendicular to log, which is along Y)
- Saw Y (blade, 0.1m) → World -Z (pointing down for cutting)
- Saw Z (thickness, 0.02m) → World X (along approach to log)

Log is at X=1.0, rotated 90° around Z, so its length (0.8m) is along Y-axis.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

# Method 1: Direct rotation matrix construction
# Target orientation matrix (world frame):
# For proper sawing with log at X=1.0, rotated 90° around Z (length along Y):
# - Saw X (length 0.7m) → World X (perpendicular to log)
# - Saw Y (blade 0.1m) → World -Z (pointing down for cutting)
# - Saw Z (thickness 0.02m) → World Y (parallel to log length)

target_matrix = np.array(
    [
        [1, 0, 0],  # Column 0: Saw X → World X
        [0, 0, 1],  # Column 1: Saw Y → World -Z becomes World Y after cross product
        [0, -1, 0],  # Column 2: Saw Z → World -Y
    ]
).T  # Transpose to get columns as basis vectors

# Verify it's right-handed (determinant should be +1)
det = np.linalg.det(target_matrix)
print(f"Matrix determinant: {det:.6f} (should be 1.0 for right-handed)")

# Create rotation from matrix
rot = R.from_matrix(target_matrix)
quat_wxyz = rot.as_quat()  # Returns [x, y, z, w]

# Convert to Isaac Sim format [w, x, y, z]
quat_isaac = np.array([quat_wxyz[3], quat_wxyz[0], quat_wxyz[1], quat_wxyz[2]])

print("=" * 60)
print("SAW SPAWN ROTATION QUATERNION")
print("=" * 60)
print(f"Quaternion (w, x, y, z): {quat_isaac}")
print(f"\nFor Isaac Sim code:")
print(
    f"rot=({quat_isaac[0]:.6f}, {quat_isaac[1]:.6f}, {quat_isaac[2]:.6f}, {quat_isaac[3]:.6f})"
)
print("=" * 60)

# Verify
rot_matrix_check = rot.as_matrix()
print("\nVerification - Rotation Matrix:")
print(rot_matrix_check)
print("\nSaw X (length) points to:", rot_matrix_check[:, 0])
print("Saw Y (blade) points to:", rot_matrix_check[:, 1])
print("Saw Z (thickness) points to:", rot_matrix_check[:, 2])
