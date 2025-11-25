import numpy as np
from scipy.spatial.transform import Rotation as R

print("=" * 60)
print("LOG POSITIONING FOR SAW ALIGNMENT")
print("=" * 60)

# Current saw state from compliant mode (Frame 878)
saw_pos = np.array([0.254, -0.014, 0.512])  # Saw position in world
saw_quat_wxyz = np.array([-0.2624, -0.7475, -0.1907, 0.5797])

# Convert to xyzw for scipy
saw_quat_xyzw = np.array(
    [saw_quat_wxyz[1], saw_quat_wxyz[2], saw_quat_wxyz[3], saw_quat_wxyz[0]]
)
R_saw = R.from_quat(saw_quat_xyzw)

# Get saw orientation in euler
saw_euler = R_saw.as_euler("xyz", degrees=True)
print(f"\nCurrent SAW state:")
print(f"  Position: {saw_pos}")
print(f"  Euler (XYZ, deg): {saw_euler}")

# The saw blade should cut downward (-Z direction in saw's local frame)
# After RotX(-90°) spawn, the saw's Y-axis points down
# We need to find what direction that is in world frame

# Saw's local Y-axis in world frame (blade cutting direction)
saw_local_y = np.array([0, 1, 0])  # Local Y
blade_direction_world = R_saw.apply(saw_local_y)

print(f"\nSaw blade cutting direction (world): {blade_direction_world}")
print(f"  (This should point into the log)")

# For the log to be cuttable:
# - Log's surface normal should be parallel to blade direction
# - Log should be positioned so saw blade touches its surface

# Current log
log_pos_current = np.array([0.45, 0.0, 0.4])
log_rot_current_deg = 90  # Currently rotated 90° around Z

print(f"\nCurrent LOG state:")
print(f"  Position: {log_pos_current}")
print(f"  Rotation: {log_rot_current_deg}° around Z")

# Calculate new log orientation
# We want log's top surface (+Z in log frame) perpendicular to blade direction
# Blade points: blade_direction_world
# We want log rotated so its +Z aligns with -blade_direction

# If blade points down-ish, log should have its top surface facing that direction
# Let's rotate log so its +Z points opposite to blade direction
target_log_up = -blade_direction_world

# Calculate rotation to align log's +Z with target
current_up = np.array([0, 0, 1])
rotation_axis = np.cross(current_up, target_log_up)
rotation_axis_norm = np.linalg.norm(rotation_axis)

if rotation_axis_norm > 1e-6:
    rotation_axis = rotation_axis / rotation_axis_norm
    rotation_angle = np.arccos(np.dot(current_up, target_log_up))
    R_log_new = R.from_rotvec(rotation_angle * rotation_axis)
else:
    R_log_new = R.identity()

log_quat_new_xyzw = R_log_new.as_quat()
log_quat_new_wxyz = np.array(
    [
        log_quat_new_xyzw[3],
        log_quat_new_xyzw[0],
        log_quat_new_xyzw[1],
        log_quat_new_xyzw[2],
    ]
)
log_euler_new = R_log_new.as_euler("xyz", degrees=True)

print(f"\nNEW LOG orientation:")
print(f"  Quat (wxyz): {log_quat_new_wxyz}")
print(f"  Euler (XYZ, deg): {log_euler_new}")

# Calculate new log position
# Position log so its top surface is at saw blade height
# Assume log dimensions: height=0.2m, top surface at pos_z + 0.1
log_top_z = saw_pos[2]  # Align log top with saw Z
log_pos_new_z = log_top_z - 0.1  # Assuming log half-height = 0.1

# X, Y: position log in front of saw
log_pos_new = np.array([saw_pos[0] + 0.2, saw_pos[1], log_pos_new_z])

print(f"\nNEW LOG position:")
print(f"  Position: {log_pos_new}")
print(f"  (Placed {0.2}m in front of saw in X direction)")

print("\n" + "=" * 60)
print("RECOMMENDED CHANGES TO run_hri_demo.py")
print("=" * 60)
print(f"\nLog init_state:")
print(f"  pos=({log_pos_new[0]:.2f}, {log_pos_new[1]:.2f}, {log_pos_new[2]:.2f})")
print(
    f"  rot=({log_quat_new_wxyz[0]:.4f}, {log_quat_new_wxyz[1]:.4f}, {log_quat_new_wxyz[2]:.4f}, {log_quat_new_wxyz[3]:.4f})"
)
