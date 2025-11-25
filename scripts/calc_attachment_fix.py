import numpy as np
from scipy.spatial.transform import Rotation as R

print("=" * 60)
print("CALCULATING NEW ATTACHMENT ROTATION")
print("=" * 60)

# Current attachment: RotY(90)
# This aligns Saw X (length) with EE Z (out)
# But Saw Y (blade) is likely aligned with EE Y (side)
q_current_wxyz = np.array([0.707, 0.0, 0.707, 0.0])
q_current_xyzw = np.array([0.0, 0.707, 0.0, 0.707])
R_current = R.from_quat(q_current_xyzw)

print(f"Current Rotation (Euler XYZ): {R_current.as_euler('xyz', degrees=True)}")

# We want to rotate the saw 90 degrees around its own X-axis (handle)
# so that the blade (Y) points in a different direction (e.g. Down)
R_fix = R.from_euler("x", 90, degrees=True)

# New rotation = Current * Fix (intrinsic)
R_new = R_current * R_fix

q_new_xyzw = R_new.as_quat()
q_new_wxyz = np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]])

print(f"New Rotation (Euler XYZ): {R_new.as_euler('xyz', degrees=True)}")
print(f"New Quaternion (wxyz): {q_new_wxyz}")
print(f"  (Use this for local_rot_saw)")

# Alternative: RotX(-90) if blade is the other way
R_fix_neg = R.from_euler("x", -90, degrees=True)
R_new_neg = R_current * R_fix_neg
q_new_neg_xyzw = R_new_neg.as_quat()
q_new_neg_wxyz = np.array(
    [q_new_neg_xyzw[3], q_new_neg_xyzw[0], q_new_neg_xyzw[1], q_new_neg_xyzw[2]]
)

print(f"\nAlternative (RotX -90):")
print(f"New Quaternion (wxyz): {q_new_neg_wxyz}")
