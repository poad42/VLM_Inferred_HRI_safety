import numpy as np
from scipy.spatial.transform import Rotation as R

# Data from Frame 878 (compliant mode - actual attachment relationship)
ee_quat_wxyz = np.array([0.05259514, 0.93752, 0.32035586, 0.12514836])
saw_quat_wxyz = np.array([-0.2623905, -0.74749327, -0.19071503, 0.5796833])

# Convert to xyzw for scipy
ee_quat_xyzw = np.array(
    [ee_quat_wxyz[1], ee_quat_wxyz[2], ee_quat_wxyz[3], ee_quat_wxyz[0]]
)
saw_quat_xyzw = np.array(
    [saw_quat_wxyz[1], saw_quat_wxyz[2], saw_quat_wxyz[3], saw_quat_wxyz[0]]
)

# Create rotation objects
R_ee = R.from_quat(ee_quat_xyzw)
R_saw = R.from_quat(saw_quat_xyzw)

# Calculate attachment offset: saw = ee * offset
# Therefore: offset = ee^-1 * saw
R_offset = R_ee.inv() * R_saw

# Get offset quaternion
offset_quat_xyzw = R_offset.as_quat()
offset_quat_wxyz = np.array(
    [offset_quat_xyzw[3], offset_quat_xyzw[0], offset_quat_xyzw[1], offset_quat_xyzw[2]]
)

print("=" * 60)
print("EMPIRICAL ATTACHMENT OFFSET CALCULATION")
print("=" * 60)
print(f"\\nEE Quat (wxyz):  {ee_quat_wxyz}")
print(f"SAW Quat (wxyz): {saw_quat_wxyz}")
print(f"\\nAttachment Offset (wxyz): {offset_quat_wxyz}")

# Convert to euler for understanding
euler_offset = R_offset.as_euler("xyz", degrees=True)
print(f"Offset in Euler (XYZ, degrees): {euler_offset}")

# Now calculate what EE orientation we need for perpendicular saw
# Desired saw: perpendicular to log = Identity
desired_saw = R.identity()

# Target EE = desired_saw * offset^-1
R_target_ee = desired_saw * R_offset.inv()

target_ee_quat_xyzw = R_target_ee.as_quat()
target_ee_quat_wxyz = np.array(
    [
        target_ee_quat_xyzw[3],
        target_ee_quat_xyzw[0],
        target_ee_quat_xyzw[1],
        target_ee_quat_xyzw[2],
    ]
)

print("\\n" + "=" * 60)
print("TARGET EE FOR PERPENDICULAR SAW")
print("=" * 60)
print(f"Desired SAW: Identity (perpendicular)")
print(f"Target EE Quat (wxyz): {target_ee_quat_wxyz}")

# Verify by applying offset
R_verify_saw = R_target_ee * R_offset
verify_saw_euler = R_verify_saw.as_euler("xyz", degrees=True)
print(f"\\nVerification - resulting saw orientation (euler): {verify_saw_euler}")
print(f"Should be close to [0, 0, 0] for identity")
