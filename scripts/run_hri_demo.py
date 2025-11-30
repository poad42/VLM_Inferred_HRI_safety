"""
This is the final, working baseline script for Isaac Lab v2.3.0.
It successfully controls the Franka arm's end-effector with an OSC.

-- MODIFIED VERSION (v12) --
This script has been modified to include runtime dynamic attachment.

1.  **Attachment API Fix (FINAL):**
    - Per your last log (`TypeError: 'Stage' object is not callable`),
      the fix is to access `sim._stage` as an *attribute*, not a method.
    - `AttachmentHelper` now correctly gets the stage via `self.stage = sim._stage`.
2.  **TypeError Fix (FINAL):**
    - The `TypeError` log proves `find_bodies` returns a list of lists.
    - The correct index `ee_frame_idx_int = ee_frame_idx_list[0][0]` is
      correct and remains in this file.
3.  **Default Attachment:**
    - The saw remains attached by default.
"""

import argparse
from isaaclab.app import AppLauncher
import copy
import torch
import numpy as np

# Boilerplate
parser = argparse.ArgumentParser(
    description="A working baseline for the Operational Space Controller."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- MOVED IMPORT TO TOP ---
# --- MOVED IMPORT TO TOP ---
from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)

# Debug Drawing
try:
    from omni.isaac.debug_draw import _debug_draw
except ImportError:
    print("[WARNING] omni.isaac.debug_draw not found. HUD will be disabled.")
    _debug_draw = None

# Camera and VLM Integration
from isaaclab.sensors import CameraCfg, Camera
from shared_buffer import SharedImageBuffer
from camera_utils import add_camera_to_scene, save_camera_image
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.controllers import (
    OperationalSpaceController,
    OperationalSpaceControllerCfg,
)
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_inv,
    subtract_frame_transforms,
    quat_apply_inverse,
    combine_frame_transforms,
    quat_apply,
)
import carb.input
import omni.appwindow
from pxr import Gf
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# --- NEW ATTACHMENT IMPORT (v11 FIX) ---
import omni.physx.scripts.physicsUtils
import omni.usd
import omni.physx.scripts.physicsUtils
import omni.usd
from camera_utils import add_camera_to_scene, save_camera_image

# --- END NEW IMPORTS ---

from material_zones import get_material_at_position, material_to_stiffness

# --- KEYBOARD IMPL (IMPLEMENTATION PLAN SECTION 6) ---
# Force command in EE frame [fx, fy, fz]
g_human_saw_force_cmd_ee = torch.zeros(1, 3)
g_force_magnitude = 5.0  # N (start with lower force to avoid breaking joint)

# IMPLEMENTATION PLAN FIX: Force ramping for downward pressure (Section 6.3)
# Variables for smooth force application (slew rate limiter)
g_downward_force_target = 0.0  # Target downward force (Z-axis)
g_downward_force_applied = 0.0  # Current applied force (smoothed)
MAX_DOWNWARD_FORCE = 50.0  # N (Section 6.3)
FORCE_RAMP_RATE = 1.0  # N per simulation step (Section 6.3 - prevents instability)

# Y-AXIS TRAVERSAL: Material Zone Navigation (LEFT/RIGHT across zones)
g_target_y_position = 0.0  # Starts at center (knot zone at Y=0.0)
Y_MOVE_INCREMENT = 0.05  # 5cm per arrow key press
Y_MIN_LIMIT = -0.40  # Left boundary (extended to allow more reach)
Y_MAX_LIMIT = 0.40  # Right boundary (extended)

# Z-AXIS CONTROL: Manual Height Adjustment (UP/DOWN)
g_target_z_position = 0.53  # Default height (resting on log)
Z_MOVE_INCREMENT = 0.02  # 2cm per arrow key press
Z_MIN_LIMIT = 0.45  # Don't go too low (collision)
Z_MAX_LIMIT = 0.70  # Don't go too high (reach limit)

# ORACLE BENCHMARK (Day 2.5)
g_oracle_mode = False  # Toggle with 'O'
g_penalty_mode = False  # Toggle with 'P'

# STIFFNESS SMOOTHING (Stability Fix)
g_smoothed_stiffness = 400.0  # Start at baseline
STIFFNESS_RAMP_RATE = (
    5.0  # Max change per step (N/m per 10ms) -> 500 N/m/s (Slower for stability)
)


# --- NEW ATTACHMENT HELPER CLASS (v12) ---
class AttachmentHelper:
    """
    Manages the state of a runtime-created fixed joint to attach
    the end-effector to the saw.

    MODIFIED (v12): Uses sim._stage (attribute)
    """

    # --- MODIFIED __init__ ---
    def __init__(self, sim: SimulationContext, env_idx=0):
        """
        Initializes the helper and immediately creates the attachment.
        """
        # --- API FIX (v12) ---
        # Get the physics utils and the stage
        self.physx_utils = omni.physx.scripts.physicsUtils
        # Access `_stage` as an attribute (no parentheses)
        self.stage = sim._stage
        # --- END API FIX ---

        self.attachment_joint = None
        self.is_attached = False

        # Define a unique path for the new joint prim
        self.joint_path = f"/World/envs/env_{env_idx}/DynamicAttachmentJoint"

        print("[AttachmentHelper]: Initialized. Attaching by default...")

        # --- ATTACH ON INIT ---
        self._attach_saw_to_ee(env_idx)
        # --- END ATTACH ON INIT ---

    def _attach_saw_to_ee(self, env_idx=0):
        """Creates a fixed joint between the EE and the saw."""
        if self.attachment_joint is not None:
            return

        # Define the prim paths for the two bodies to be joined
        # These must be acquired from your scene/asset definitions
        ee_prim_path = (
            f"/World/envs/env_{env_idx}/Robot/panda_hand"  # Matches ee_frame_name
        )
        saw_prim_path = f"/World/envs/env_{env_idx}/Saw"

        # IMPLEMENTATION PLAN FIX: Updated TCP offset and rotation
        # TCP offset: 0.107m out from wrist (standard Franka TCP)
        local_pos_ee = Gf.Vec3f(0.0, 0.0, 0.107)
        local_rot_ee = Gf.Quatf(1.0, 0.0, 0.0, 0.0)

        # IMPLEMENTATION PLAN FIX: Saw attachment point and rotation
        # Position offset from saw origin to attachment point
        # For vertical blade alignment with robot grasp
        # Attachment: Center of the saw (0.0, 0.0, 0.0)
        # This reduces the required reach height and improves stability
        local_pos_saw = Gf.Vec3f(0.0, 0.0, 0.0)
        # Rotation: -90° around X-axis (w, x, y, z) for Horizontal Knife orientation
        # This aligns Length (X) Horizontal and Width (Y) Vertical
        local_rot_saw = Gf.Quatf(0.707, -0.707, 0.0, 0.0)

        try:
            # --- API FIX (v11) ---
            # Call the function from the user's provided API
            joint_prim = self.physx_utils.add_joint_fixed(
                stage=self.stage,
                jointPath=self.joint_path,
                actor0=ee_prim_path,
                actor1=saw_prim_path,
                localPos0=local_pos_ee,
                localRot0=local_rot_ee,
                localPos1=local_pos_saw,
                localRot1=local_rot_saw,
                breakForce=1.0e30,  # 0.0 = unbreakable
                breakTorque=1.0e30,  # 0.0 = unbreakable
            )
            # --- END API FIX ---

            if joint_prim:
                self.attachment_joint = joint_prim
                self.is_attached = True
                print(f"SUCCESS: Created joint at {self.joint_path}")
            else:
                print(f"ERROR: Failed to create joint at {self.joint_path}")

        except Exception as e:
            print(f"Exception while creating joint: {e}")

    def _detach_saw_from_ee(self):
        """Removes the fixed joint from the simulation."""
        if self.attachment_joint:
            try:
                # Remove the joint prim from the stage
                self.stage.RemovePrim(self.joint_path)
                self.attachment_joint = None
                self.is_attached = False
                print(f"SUCCESS: Removed joint at {self.joint_path}")
            except Exception as e:
                print(f"Exception while removing joint: {e}")


# --- END NEW ATTACHMENT HELPER CLASS ---


# --- KEYBOARD HANDLER (IMPLEMENTATION PLAN SECTION 6.1) ---
def _on_keyboard_event(event, saw_object: RigidObject):
    """Callback to apply force to the saw object and control Y/Z position"""
    global g_human_saw_force_cmd_ee, g_force_magnitude, g_downward_force_target, g_target_y_position, g_target_z_position, g_oracle_mode, g_penalty_mode

    if event.type in (
        carb.input.KeyboardEventType.KEY_PRESS,
        carb.input.KeyboardEventType.KEY_REPEAT,
    ):
        if event.input == carb.input.KeyboardInput.K:
            # "Pull" saw (+ X direction in EE frame = along saw length)
            g_human_saw_force_cmd_ee[0, 0] = g_force_magnitude
        elif event.input == carb.input.KeyboardInput.J:
            # "Push" saw (- X direction in EE frame = along saw length)
            g_human_saw_force_cmd_ee[0, 0] = -g_force_magnitude
        elif event.input == carb.input.KeyboardInput.F:
            # IMPLEMENTATION PLAN: Downward cutting force (Section 6.1)
            # Press 'F' to apply downward pressure for cutting
            g_downward_force_target = MAX_DOWNWARD_FORCE
        elif event.input == carb.input.KeyboardInput.LEFT:
            # Y-AXIS TRAVERSAL: Move left (decrease Y)
            g_target_y_position = max(
                Y_MIN_LIMIT, g_target_y_position - Y_MOVE_INCREMENT
            )
            print(f"[Y-Traverse] Target Y: {g_target_y_position:.3f}m")
        elif event.input == carb.input.KeyboardInput.RIGHT:
            # Y-AXIS TRAVERSAL: Move right (increase Y)
            g_target_y_position = min(
                Y_MAX_LIMIT, g_target_y_position + Y_MOVE_INCREMENT
            )
            print(f"[Y-Traverse] Target Y: {g_target_y_position:.3f}m")
        elif event.input == carb.input.KeyboardInput.UP:
            # Z-AXIS CONTROL: Move up (increase Z)
            g_target_z_position = min(
                Z_MAX_LIMIT, g_target_z_position + Z_MOVE_INCREMENT
            )
            print(f"[Z-Control] Target Z: {g_target_z_position:.3f}m")
        elif event.input == carb.input.KeyboardInput.DOWN:
            # Z-AXIS CONTROL: Move down (decrease Z)
            g_target_z_position = max(
                Z_MIN_LIMIT, g_target_z_position - Z_MOVE_INCREMENT
            )
            print(f"[Z-Control] Target Z: {g_target_z_position:.3f}m")
        elif event.input == carb.input.KeyboardInput.U:
            # Increase force magnitude
            g_force_magnitude += 5.0
            print(f"[Force Control] Increased force to {g_force_magnitude:.1f} N")
        elif event.input == carb.input.KeyboardInput.M:
            # Decrease force magnitude (minimum 0)
            g_force_magnitude = max(0.0, g_force_magnitude - 5.0)
            print(f"[Force Control] Decreased force to {g_force_magnitude:.1f} N")
        elif event.input == carb.input.KeyboardInput.R:
            # Reset force magnitude to default (Section 6.1)
            g_force_magnitude = 5.0
            g_downward_force_target = 0.0  # Also reset downward force
            print(f"[Force Control] Reset force to {g_force_magnitude:.1f} N")
        elif event.input == carb.input.KeyboardInput.O:
            # Toggle Oracle Mode
            g_oracle_mode = not g_oracle_mode
            status = "ENABLED" if g_oracle_mode else "DISABLED"
            print(f"[ORACLE] Mode {status}")
        elif event.input == carb.input.KeyboardInput.P:
            # Toggle Penalty Mode
            g_penalty_mode = not g_penalty_mode
            status = "ENABLED" if g_penalty_mode else "DISABLED"
            print(f"[PENALTY] Mode {status}")
        # --- REMOVED 'T' KEY ---

    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        if event.input in (carb.input.KeyboardInput.J, carb.input.KeyboardInput.K):
            # Stop horizontal force
            g_human_saw_force_cmd_ee[0, 0] = 0.0
        elif event.input == carb.input.KeyboardInput.F:
            # IMPLEMENTATION PLAN: Release downward force (Section 6.1)
            g_downward_force_target = 0.0


# --- END MODIFIED KEYBOARD HANDLER ---


# --- MODIFIED update_states (v11) ---
def update_states(robot: Articulation, ee_frame_idx: int, arm_joint_ids: list[int]):
    """
    MODIFIED: Signature now accepts `ee_frame_idx` as an integer.
    """
    # `ee_frame_idx` is now an int, no indexing needed
    ee_jacobi_idx = ee_frame_idx - 1

    jacobian_w = robot.root_physx_view.get_jacobians()[
        :, ee_jacobi_idx, :, arm_joint_ids
    ]
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[
        :, arm_joint_ids, :
    ][:, :, arm_joint_ids]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])
    root_pos_w, root_quat_w = robot.data.root_pos_w, robot.data.root_quat_w

    # Use the integer index
    ee_pos_w, ee_quat_w = (
        robot.data.body_pos_w[:, ee_frame_idx],
        robot.data.body_quat_w[:, ee_frame_idx],
    )
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # Use the integer index
    ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]
    root_vel_w = robot.data.root_vel_w
    relative_vel_w = ee_vel_w - root_vel_w
    ee_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])
    ee_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    joint_vel = robot.data.joint_vel[:, arm_joint_ids]
    return jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, joint_pos, joint_vel


# --- END MODIFIED update_states ---


def main():
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.5])

    # Create the scene_cfg instance
    scene_cfg = InteractiveSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)

    # Add the robot, ground, and light
    # Configure robot with initial joint positions to extend arm toward log
    scene_cfg.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            # Joint positions to extend arm forward toward log (X~0.5-0.6)
            # These values position the EE approximately at [0.5, 0.0, 0.55]
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.5,  # Shoulder down
                "panda_joint3": 0.0,
                "panda_joint4": -2.0,  # Elbow bent to reach forward
                "panda_joint5": 0.0,
                "panda_joint6": 1.5,  # Wrist up
                "panda_joint7": 0.785,  # Rotated 45°
                "panda_finger_joint.*": 0.04,  # Gripper open
            },
        ),
    )
    scene_cfg.ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    scene_cfg.light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0)
    )

    # Add the "saw" object
    scene_cfg.saw = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Saw",
        spawn=sim_utils.CuboidCfg(
            size=(0.7, 0.1, 0.02),  # (length, width, height)
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.7, 0.7), metallic=0.8
            ),
            # PHYSICS FIX 1: Dynamic mode with negligible mass
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.001
            ),  # Negligible mass to prevent sagging
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,  # Dynamic to allow movement and force application
                disable_gravity=True,
                max_depenetration_velocity=0.1,  # Prevent explosive ejection
                solver_position_iteration_count=12,  # Increase from default 4
                solver_velocity_iteration_count=4,  # Increase from default 1
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            # PHYSICS FIX 2: Material properties for stable contact
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,  # High friction to bite into wood
                dynamic_friction=0.5,  # Moderate friction for sawing
                restitution=0.0,  # NO BOUNCING (critical!)
                friction_combine_mode="max",
                restitution_combine_mode="min",
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, 0.0, 0.55),
            rot=(0.707107, -0.707107, 0.0, 0.0),  # RotX(-90°) - restoring original
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    # ==============================================================================
    # MULTI-ZONE LOG FOR VLM MATERIAL DETECTION
    # ==============================================================================
    # Zone 1: Soft Wood (Light tan)
    scene_cfg.log_soft = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LogSoft",
        spawn=sim_utils.CuboidCfg(
            size=(0.30, 0.15, 0.15),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.5, 0.3)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=False
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3, dynamic_friction=0.2, restitution=0.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, -0.25, 0.4), rot=(0.707, 0.0, 0.0, 0.707)
        ),
    )

    # Zone 2: Hard Knot (Dark brown)
    scene_cfg.log_knot = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LogKnot",
        spawn=sim_utils.CuboidCfg(
            size=(0.20, 0.15, 0.15),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.25, 0.15)
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=False
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9, dynamic_friction=0.7, restitution=0.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.0, 0.4), rot=(0.707, 0.0, 0.0, 0.707)
        ),
    )

    # Zone 3: Cracked (Medium brown)
    scene_cfg.log_crack = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LogCrack",
        spawn=sim_utils.CuboidCfg(
            size=(0.30, 0.15, 0.15),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=False
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.2, dynamic_friction=0.1, restitution=0.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.25, 0.4), rot=(0.707, 0.0, 0.0, 0.707)
        ),
    )

    # --- NEW: ADD CAMERA TO SCENE ---
    add_camera_to_scene(scene_cfg)
    # --- END CAMERA ADDITION ---

    # --- FRAME CALIBRATION: Add FrameTransformer for TCP visualization ---
    # This visualizes the tool center point (TCP) offset for calibration
    scene_cfg.ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=True,  # Show RGB axes in viewport
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="ee_tcp",
                offset=OffsetCfg(
                    # TCP offset from panda_hand to saw blade center
                    # Saw is 0.7m long, attached at handle (one end)
                    # Blade center is ~0.35m from attachment point
                    pos=(0.35, 0.0, 0.0),
                    # Match saw rotation: RotX(-90°)
                    rot=(0.707107, -0.707107, 0.0, 0.0),
                ),
            ),
        ],
    )
    # --- END FRAME CALIBRATION ---

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    saw = scene["saw"]
    log = scene["log_knot"]  # Use middle zone (knot) as primary log reference
    camera = scene["camera"]  # NEW: Get camera reference

    # --- MODIFIED (v12) ---
    # Instantiate the helper, passing the sim object.
    # This will now create the joint by default using the correct API.
    attachment_helper = AttachmentHelper(sim, env_idx=0)
    # --- END MODIFIED ---

    print(f"Successfully spawned robot: {robot.cfg.prim_path}")
    print(f"Successfully spawned saw: {saw.cfg.prim_path}")
    print(f"Successfully spawned log (knot zone): {log.cfg.prim_path}")
    print(f"Successfully spawned camera: {camera.cfg.prim_path}")

    # --- Initialize Shared Memory Buffer for VLM ---
    shared_buffer = SharedImageBuffer(
        name="hri_camera_buffer",
        buffer_size=10,
        height=480,
        width=640,
        channels=3,
        create=True,
    )
    print("[SharedBuffer] Initialized for camera→VLM communication")
    # --- End Buffer Init ---  # NEW

    sim_dt = sim.get_physics_dt()

    # --- MODIFIED ---
    # We subscribe the velocity-only callback function
    carb_input = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    keyboard_sub = carb_input.subscribe_to_keyboard_events(
        app_window.get_keyboard(),
        # Pass only the saw object
        lambda e, s=saw: _on_keyboard_event(e, s),
    )
    print(f"Successfully spawned saw: {saw.cfg.prim_path}")
    print("--------------------")
    print(" Keyboard Handler Initialized...")
    print(" K:   'Pull' saw (+ X direction)")
    print(" J:   'Push' saw (- X direction)")
    print(" F:   Apply downward cutting force (hold)")
    print(" U:   Increase force magnitude (+5 N)")
    print(" M:   Decrease force magnitude (-5 N)")
    print(" R:   Reset all forces to default")
    print(" T:   Toggle REMOVED. Attached by default.")
    print(" Camera: Capturing every 30 frames")
    print("--------------------")
    # --- END MODIFIED ---

    # Buffer for VELOCITY control
    saw_vel_b = torch.zeros((scene.num_envs, 6), device=sim.device)

    # Move the global buffer to the correct device
    global g_human_saw_force_cmd_ee
    g_human_saw_force_cmd_ee = g_human_saw_force_cmd_ee.to(sim.device)

    ee_frame_name = "panda_hand"
    # --- MODIFIED (v11) ---
    # Get the list of lists
    ee_frame_idx_list = robot.find_bodies(ee_frame_name)
    # --- TYPEERROR FIX ---
    # Get list for env 0, then get first index
    ee_frame_idx_int = ee_frame_idx_list[0][0]
    # --- END TYPEERROR FIX ---
    # find_joints() returns a list of lists, one for each environment
    arm_joint_ids_list = robot.find_joints("panda_joint.*")
    # Get the flat list of joint indices for the first environment (env 0)
    arm_joint_ids = arm_joint_ids_list[0]

    # --- OSC CONFIGURATION: IMPLEMENTATION PLAN VALUES ---
    # Anisotropic stiffness for perpendicular orientation constraints (Table 2)
    # Translational: [X, Y, Z]
    #   - X/Y: 800.0 (high stiffness to track cutting line accurately)
    #   - Z: 100.0 (low stiffness for force compliance with log surface)
    # Rotational: [Roll, Pitch, Yaw]
    #   - Roll/Pitch: 1500.0 (very high to prevent tilting - perpendicular constraint)
    #   - Yaw: 600.0 (moderate to allow cut steering if needed)
    default_stiffness_tuple = (
        400.0,  # Translation X - reduced from 800 for stability
        400.0,  # Translation Y - reduced from 800 for stability
        200.0,  # Translation Z - increased from 100 to prevent sag
        600.0,  # Rotation Roll - reduced from 1500 for stability
        600.0,  # Rotation Pitch - reduced from 1500 for stability
        400.0,  # Rotation Yaw - reduced from 600 for stability
    )

    # Damping ratios per implementation plan (Section 5.2, Table 2)
    # CRITICAL FIX: These are RATIOS, not absolute values!
    # OSC formula: d_gains = 2 * sqrt(p_gains) * damping_ratio
    # Lower damping on Z-axis for compliance, higher on rotations for stability
    # UPDATED (User Request): Increased damping to prevent "swinging" and "flatness"
    default_damping_ratio_tuple = (
        2.0,  # Translation X - overdamped
        4.0,  # Translation Y - heavily overdamped (prevent swing)
        1.0,  # Translation Z - critically damped (was 0.7)
        5.0,  # Rotation Roll - very heavily overdamped (lock orientation)
        5.0,  # Rotation Pitch - very heavily overdamped (lock orientation)
        2.0,  # Rotation Yaw - overdamped
    )

    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="variable_kp",
        inertial_dynamics_decoupling=True,
        gravity_compensation=True,  # Ensure arm weight doesn't bias contact
        nullspace_control="position",
        motion_stiffness_task=default_stiffness_tuple,
        motion_damping_ratio_task=default_damping_ratio_tuple,
        # NOTE: body_offset not supported in this OSC version
        # Using FrameTransformer for TCP visualization instead
    )

    osc = OperationalSpaceController(osc_cfg, scene.num_envs, sim.device)
    robot.update(sim_dt)

    # --- MODIFIED (v11) ---
    # Pass the *integer* index to update_states
    _, _, _, initial_ee_pose_b, _, _, _ = update_states(
        robot, ee_frame_idx_int, arm_joint_ids
    )
    # --- END MODIFIED ---

    # We only need the initial ORIENTATION
    initial_ee_quat_b = initial_ee_pose_b[:, 3:].clone()

    joint_centers = torch.mean(
        robot.data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1
    )

    # --- We only need a buffer for stiffness ---
    default_stiffness_tensor = torch.tensor(
        [default_stiffness_tuple], device=sim.device
    )
    current_stiffness = default_stiffness_tensor.repeat(scene.num_envs, 1)

    # --- NEW: Camera capture variables ---
    # --- NEW: Camera capture variables ---
    frame_count = 0
    capture_every = 30  # Capture every 30 frames (~0.3 seconds at 100 Hz)
    # --- END CAMERA VARIABLES ---

    try:
        print("[DEBUG] Entering simulation loop...")
        while simulation_app.is_running():
            sim.step(render=True)
            robot.update(sim_dt)
            scene.update(sim_dt)  # This updates camera data

            # --- ROBOT CONTROL LOGIC (MODIFIED FOR ATTACHMENT) ---
            # --- TYPEERROR FIX (v11) ---
            # Pass the *integer* index
            (
                jacobian_b,
                mass_matrix,
                gravity,
                ee_pose_b,
                ee_vel_b,
                joint_pos,
                joint_vel,
            ) = update_states(robot, ee_frame_idx_int, arm_joint_ids)
            # --- END TYPEERROR FIX ---

            # --- HRI (Human) Control Loop - FORCE CONTROL ---
            # Get EE orientation quaternion (base frame)
            ee_quat_b = ee_pose_b[:, 3:7]  # [num_envs, 4] quaternion (w, x, y, z)

            # Transform force from EE frame to world/base frame
            # quat_apply rotates a vector by a quaternion
            force_ee = g_human_saw_force_cmd_ee.clone()  # [1, 3]
            force_world = quat_apply(ee_quat_b, force_ee)  # [num_envs, 3]

            # Apply force to saw as external wrench
            # Create indices for all environments
            indices = torch.arange(scene.num_envs, dtype=torch.int32, device=sim.device)
            saw.root_physx_view.apply_forces(
                force_world, indices=indices, is_global=True
            )
            # --- END HRI CONTROL LOOP ---

            # --- IMPLEMENTATION PLAN: Hybrid Force/Position Control (Section 6.2, Option B) ---
            # Apply force ramping (slew rate limiter - Section 6.3)
            global g_downward_force_applied, g_downward_force_target
            force_error = g_downward_force_target - g_downward_force_applied
            # Clamp the change to prevent step-function instability
            force_delta = max(min(force_error, FORCE_RAMP_RATE), -FORCE_RAMP_RATE)
            g_downward_force_applied += force_delta

            # Modulate target Z-position based on applied downward force
            # This implements hybrid control: position control on X/Y, force control on Z
            target_ee_pose_b = ee_pose_b.clone()

            # --- VLM-READY SAW DESCENT ---
            # Goal: Lower the saw blade to contact the log surface
            # Challenge: Saw is rotated, so we need to find blade tip in world coords

            # Get current saw state
            saw_pos_w = saw.data.root_pos_w[0]  # Saw center of mass in world frame
            saw_quat_w = saw.data.root_quat_w[0]  # [w, x, y, z] in world frame

            # Saw geometry: 0.7m (length) x 0.1m (width) x 0.02m (thickness)
            # In saw's local frame: blade extends along X-axis
            # The blade TIP (cutting edge) is at (+0.35, 0, 0) in saw's local frame

            # Transform blade tip offset through saw's rotation
            # Convert quaternion to rotation matrix for transformation
            from scipy.spatial.transform import Rotation as R

            quat_scipy = [
                saw_quat_w[1].item(),
                saw_quat_w[2].item(),
                saw_quat_w[3].item(),
                saw_quat_w[0].item(),
            ]  # Isaac (w,x,y,z) -> scipy (x,y,z,w)
            saw_rotation = R.from_quat(quat_scipy)

            # Blade tip offset in saw's local frame
            # TESTING: Try opposite end since +0.35 gives blade 33cm below log
            blade_tip_local = torch.tensor([-0.35, 0.0, 0.0], device=saw_pos_w.device)

            # Transform to world frame
            blade_tip_offset_world = torch.tensor(
                saw_rotation.apply(blade_tip_local.cpu().numpy()),
                device=saw_pos_w.device,
                dtype=torch.float32,
            )

            # Blade tip position in world frame
            blade_tip_world = saw_pos_w + blade_tip_offset_world
            blade_tip_z = blade_tip_world[2]

            # Log surface (top): Z = 0.4 (center) + 0.1 (half height) = 0.5m
            log_surface_z = 0.5

            # Gap between blade tip and log surface
            gap_to_log = blade_tip_z - log_surface_z

            # Apply proportional control to close the gap
            descent_gain = 0.3  # More conservative: 30% per step for stability

            # Debug output every 10 frames
            if frame_count % 10 == 0:
                print(f"\n--- SAW DESCENT DEBUG (Frame {frame_count}) ---")
                print(f"Saw Center (World): {saw_pos_w.cpu().numpy()}")
                print(f"Saw Quat (World): {saw_quat_w.cpu().numpy()}")
                print(f"Blade Tip Local: {blade_tip_local.cpu().numpy()}")
                print(
                    f"Blade Tip Offset (World): {blade_tip_offset_world.cpu().numpy()}"
                )
                print(f"Blade Tip Z (World): {blade_tip_z:.4f}m")
                print(f"Log Surface Z: {log_surface_z:.4f}m")
                print(f"Gap to Log: {gap_to_log:.4f}m ({gap_to_log*1000:.1f}mm)")
                if gap_to_log > 0.001:
                    descent_cmd = gap_to_log * descent_gain
                    print(
                        f"Descent Command: {descent_cmd:.4f}m ({descent_cmd*1000:.1f}mm)"
                    )
                else:
                    print("Descent Command: NONE (gap < 1mm, contact achieved)")

            # --- KINEMATIC SAW CONTROL ---
            # Strategy: Direct Position/Orientation Command
            # 1. Force EE orientation to standard top-down (180 deg X-axis rotation)
            # 2. Command EE Z to place saw on log
            # 3. Keep X/Y aligned with log

            # Restore stable stiffness
            # Increase rotational stiffness to force "straight" alignment
            current_stiffness[:, 2] = 200.0
            current_stiffness[:, 3:] = 1500.0

            # --- ORACLE BENCHMARK LOGIC (Day 2.5) ---
            if g_oracle_mode:
                # 1. Get Ground Truth Position
                # CRITICAL FIX: Use COMMANDED Y (g_target_y_position) instead of ACTUAL Y (saw_pos_w)
                # Using actual Y creates a feedback loop: Swing -> Y changes -> Stiffness changes -> More swing
                saw_y = g_target_y_position

                # 2. Query Material Zone
                zone_name, props = get_material_at_position(saw_y)
                target_stiffness = props["recommended_stiffness"]
                target_force_limit = props.get(
                    "target_force", 20.0
                )  # Default 20N if missing

                # 3. Apply Force-Dependent Stiffness (REMOVED per user request)
                # User: "I don't like that regulation for increasing stiffness in response, remove that for now"
                current_human_force = g_force_magnitude
                # if current_human_force > target_force_limit:
                #     excess_force = current_human_force - target_force_limit
                #     force_penalty_stiffness = excess_force * 10.0
                #     target_stiffness += force_penalty_stiffness

                # 4. Apply Penalty (if enabled)
                if g_penalty_mode:
                    # Invert stiffness to cause failure
                    # If hard (600) -> make soft (400) -> Instability on knot
                    # If soft (500) -> make stiff (600) -> Rigid/Vibration
                    if target_stiffness >= 550:
                        target_stiffness = 400.0
                    else:
                        target_stiffness = 600.0

                # 5. Apply to Translational Stiffness (X/Y/Z) with SMOOTHING
                # We modulate the translational stiffness to simulate "giving way" or "resisting"

                # Smooth the stiffness change to prevent instability
                global g_smoothed_stiffness
                stiffness_error = target_stiffness - g_smoothed_stiffness
                stiffness_delta = max(
                    min(stiffness_error, STIFFNESS_RAMP_RATE), -STIFFNESS_RAMP_RATE
                )
                g_smoothed_stiffness += stiffness_delta

                current_stiffness[:, :3] = g_smoothed_stiffness

                # CRITICAL FIX: Enforce high rotational stiffness to prevent "swinging"
                # The user reported "swings around like crazy", which means rotational control is too loose
                current_stiffness[:, 3:] = (
                    2000.0  # Increased from 1500.0 for extra stability
                )

                # Also increase damping for rotation (if possible via OSC config, but here we control stiffness)
                # We can't easily change damping ratio dynamically in this script without accessing the cfg
                # But higher stiffness with the same damping ratio = higher damping force

                # Debug Print
                if frame_count % 30 == 0:
                    mode_str = "PENALTY" if g_penalty_mode else "ORACLE"
                    print(
                        f"[{mode_str}] Zone: {zone_name.upper()} | Force: {current_human_force:.1f}/{target_force_limit:.1f}N | Target K: {target_stiffness:.0f} | Actual K: {g_smoothed_stiffness:.0f}"
                    )

                # --- IN-SCENE VISUALIZATION ---
                if _debug_draw:
                    draw = _debug_draw.acquire_debug_draw_interface()
                    # Position text above the saw
                    text_pos = saw_pos_w.cpu().numpy() + np.array([0.0, 0.0, 0.3])

                    # Format text
                    status_text = f"ZONE: {zone_name.upper()}\n"
                    status_text += f"FORCE: {current_human_force:.1f} / {target_force_limit:.1f} N\n"
                    status_text += f"STIFFNESS: {g_smoothed_stiffness:.0f} N/m"

                    # Color based on compliance (Green=Good, Red=Overload)
                    if current_human_force > target_force_limit:
                        color = (1.0, 0.0, 0.0, 1.0)  # Red
                    else:
                        color = (0.0, 1.0, 0.0, 1.0)  # Green

                    draw.draw_text(text_pos.tolist(), status_text, 20, color)

            # Target Orientation: [0, 1, 0, 0] (w, x, y, z) -> 180 deg around X
            # This points the gripper Z-axis DOWN
            target_ee_pose_b[:, 3] = 0.0
            target_ee_pose_b[:, 4] = 1.0
            target_ee_pose_b[:, 5] = 0.0
            target_ee_pose_b[:, 6] = 0.0

            # Target Position:
            # Log Center = [0.45, 0.0, 0.4]
            # Zones: soft(Y=-0.25), knot(Y=0.0), crack(Y=0.25)
            target_ee_pose_b[:, 0] = 0.45  # Fixed X (aligned with log)
            # Y-AXIS TRAVERSAL: Use global target position (controlled by LEFT/RIGHT arrows)
            target_ee_pose_b[:, 1] = g_target_y_position

            # Target Z Calculation:
            # Saw Center at 0.0 (Attachment)
            # Log Top = 0.5m
            # Blade Length = 0.35m (Half-length)
            # If vertical, Tip is at Center - 0.35m
            # We want Tip at 0.5m -> Center at 0.85m
            # EE Offset is approx 0.24m -> EE Z = 1.09m
            # Target Z Calculation:
            # Saw Center at 0.0 (Attachment)
            # Log Top = 0.5m
            # Saw Width = 0.1m (Vertical dimension now) -> Half-Width = 0.05m
            # We want Bottom Edge at 0.5m -> Center at 0.55m
            # User wants it to "rest straight". Let's lower slightly to 0.53m
            # to ensure it sits firmly and flat.
            # Z-AXIS CONTROL: Use global target position (controlled by UP/DOWN arrows)
            target_ee_pose_b[:, 2] = g_target_z_position

            # Debug output
            if frame_count % 10 == 0:
                print(f"\n--- KINEMATIC SAW DEBUG (Frame {frame_count}) ---")
                print(f"Target EE: {target_ee_pose_b[0, :3].cpu().numpy()}")
                print(f"Target Quat: {target_ee_pose_b[0, 3:].cpu().numpy()}")
                print(f"Actual EE: {ee_pose_b[0, :3].cpu().numpy()}")
                print(f"Saw Center: {saw_pos_w.cpu().numpy()}")

                # Calculate Attachment Error (Distance between EE and Saw Center)
                dist = torch.norm(saw_pos_w - ee_pose_b[0, :3])
                print(f"Attachment Dist (Joint Error): {dist:.4f}m")

                # Print Saw Orientation (Euler) to verify vertical
                saw_euler = R.from_quat(saw_quat_w.cpu().numpy()).as_euler(
                    "xyz", degrees=True
                )
                print(f"Saw Euler (deg): {saw_euler}")

            # Z-compliance shift: Based on dynamic stiffness
            # This creates a "virtual spring" that generates force proportional to displacement
            # F = K * x  ->  x = F / K
            # We use the smoothed stiffness to ensure consistent reaction
            if g_smoothed_stiffness > 1.0:
                raw_shift = g_downward_force_applied / g_smoothed_stiffness
                # Clamp shift to 5cm max to prevent "falling" feeling at low stiffness
                z_compliance_shift = min(raw_shift, 0.05)
            else:
                z_compliance_shift = 0.0

            target_ee_pose_b[:, 2] -= z_compliance_shift  # Lower Z target

            # Action dim = 7 (pose) + 6 (stiffness) = 13
            command = torch.cat([target_ee_pose_b, current_stiffness], dim=1)
            # --- END ROBOT CONTROL LOGIC ---

            osc.set_command(command)
            joint_efforts = osc.compute(
                current_ee_pose_b=ee_pose_b,
                current_ee_vel_b=ee_vel_b,
                mass_matrix=mass_matrix,
                jacobian_b=jacobian_b,
                gravity=gravity,
                current_joint_pos=joint_pos,
                current_joint_vel=joint_vel,
                nullspace_joint_pos_target=joint_centers,
            )
            robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
            robot.write_data_to_sim()

            # --- Camera capture: Write to shared memory buffer ---
            if frame_count % 30 == 0:  # Every 30 frames (~0.3s at 100Hz = ~3 FPS)
                try:
                    if (
                        not hasattr(camera.data, "output")
                        or "rgb" not in camera.data.output
                    ):
                        pass
                    else:
                        rgb_data = camera.data.output["rgb"]
                        if rgb_data is not None:
                            rgb_np = rgb_data[0].cpu().numpy()
                            if rgb_np.dtype in (np.float32, np.float64):
                                rgb_np = (np.clip(rgb_np, 0, 1) * 255).astype(np.uint8)
                            if rgb_np.shape[2] == 4:
                                rgb_np = rgb_np[:, :, :3]
                            shared_buffer.write(rgb_np)
                            if frame_count % 300 == 0:  # Print every 3 seconds
                                stats = shared_buffer.get_stats()
                                print(
                                    f"[Camera→Buffer] Frame {frame_count} → Buffer #{stats['frame_count']}"
                                )
                except Exception as e:
                    if frame_count % 300 == 0:
                        print(f"[Camera→Buffer Error] {e}")
            frame_count += 1
            # --- END CAMERA CAPTURE ---

            # --- NEW: Camera capture logic ---
            if frame_count % capture_every == 0:
                save_camera_image(camera, frame_count)
            frame_count += 1

            # --- COMPREHENSIVE DEBUG OUTPUT (EVERY FRAME) ---
            if True:  # Always print for debugging
                # Calculate orientation error (quaternion difference)
                from isaaclab.utils.math import quat_error_magnitude

                # Target vs Actual EE
                ee_quat_error = quat_error_magnitude(
                    target_ee_pose_b[:, 3:7], ee_pose_b[:, 3:7]
                )
                ee_pos_error = torch.norm(
                    target_ee_pose_b[:, :3] - ee_pose_b[:, :3], dim=-1
                )

                # Saw pose
                saw_pos_w = saw.data.root_pos_w[0]
                saw_quat_w = saw.data.root_quat_w[0]

                # Log pose
                log_pos_w = log.data.root_pos_w[0]
                log_quat_w = log.data.root_quat_w[0]

                # Applied force
                force_magnitude = torch.norm(force_world[0])

                print(f"\n========== FRAME {frame_count} DEBUG ==========")
                print("TARGET EE:")
                print(f"  Pos: {target_ee_pose_b[0, :3].cpu().numpy()}")
                print(f"  Quat: {target_ee_pose_b[0, 3:7].cpu().numpy()}")
                print("\nACTUAL EE:")
                print(f"  Pos: {ee_pose_b[0, :3].cpu().numpy()}")
                print(f"  Quat: {ee_pose_b[0, 3:7].cpu().numpy()}")
                print("\nEE ERROR:")
                print(f"  Position Error: {ee_pos_error[0].item():.4f} m")
                print(f"  Orientation Error: {ee_quat_error[0].item():.4f} rad")
                print("\nSAW STATE:")
                print(f"  Pos (World): {saw_pos_w.cpu().numpy()}")
                print(f"  Quat (World): {saw_quat_w.cpu().numpy()}")
                print("\nLOG STATE:")
                print(f"  Pos (World): {log_pos_w.cpu().numpy()}")
                print(f"  Quat (World): {log_quat_w.cpu().numpy()}")
                print("\nFORCES APPLIED:")
                print(
                    f"  Human Force (World): {force_world[0].cpu().numpy()} ({force_magnitude.item():.2f}N)"
                )
                print("\nSTIFFNESS:")
                print(f"  Translational: {current_stiffness[0, :3].cpu().numpy()}")
                print(f"  Rotational: {current_stiffness[0, 3:].cpu().numpy()}")
                print(f"{'='*50}")

            # Old debug output (keep for compatibility)
            if frame_count % 30 == 0:
                pass  # Comprehensive output above replaces this

        print("[DEBUG] Exited simulation loop normally")

    except Exception as e:
        print(f"[ERROR] Exception in simulation loop: {e}")
        import traceback

        traceback.print_exc()  # --- END CAMERA CAPTURE ---

    finally:
        # Clean up shared memory buffer
        if "shared_buffer" in locals():
            print("[SharedBuffer] Cleaning up...")
            shared_buffer.close()
            shared_buffer.unlink()

        # --- CLEANUP ---
        if (
            "carb_input" in locals()
            and "keyboard_sub" in locals()
            and keyboard_sub is not None
        ):
            carb_input.unsubscribe_to_keyboard_events(
                app_window.get_keyboard(), keyboard_sub
            )

        if "attachment_helper" in locals():
            attachment_helper._detach_saw_from_ee()
        print("KeyboardHandler shutdown and joint detached.")
        simulation_app.close()


if __name__ == "__main__":
    main()
