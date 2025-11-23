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

# from isaaclab.sensors import CameraCfg # REMOVED CAMERA
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

# --- KEYBOARD IMPL (MODIFIED) ---
g_human_saw_force_cmd_ee = torch.zeros(1, 3)  # Force in EE frame [fx, fy, fz]
g_force_magnitude = 5.0  # N (start with lower force to avoid breaking joint)


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

        # This attaches to the Tool Center Point (TCP), 10.7cm "out" from the wrist
        local_pos_ee = Gf.Vec3f(0.0, 0.0, 0.107)
        local_rot_ee = Gf.Quatf(1.0, 0.0, 0.0, 0.0)

        # This addresses the "one end of the saw" request.
        # Assumes the saw is 0.7 long, so -0.35 is one end.
        local_pos_saw = Gf.Vec3f(-0.35, 0.0, 0.0)
        # This is a 90-degree rotation around the Y-axis (w, x, y, z)
        local_rot_saw = Gf.Quatf(0.707, 0.0, 0.707, 0.0)

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


# --- MODIFIED KEYBOARD HANDLER ---
def _on_keyboard_event(event, saw_object: RigidObject):
    """Callback to apply force to the saw object"""
    global g_human_saw_force_cmd_ee, g_force_magnitude

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
        # --- REMOVED 'T' KEY ---

    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        if event.input in (carb.input.KeyboardInput.J, carb.input.KeyboardInput.K):
            # Stop force
            g_human_saw_force_cmd_ee[0, 0] = 0.0


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
    scene_cfg.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, 0.0, 0.5),  # Offset Y to avoid collision
            rot=(0.707, 0.707, 0.0, 0.0),  # Rotated 90-deg around X-axis
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    # --- NEW: ADD CAMERA TO SCENE ---
    add_camera_to_scene(scene_cfg)
    # --- END CAMERA ADDITION ---

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    saw = scene["saw"]
    camera = scene["camera"]  # NEW: Get camera reference

    # --- MODIFIED (v12) ---
    # Instantiate the helper, passing the sim object.
    # This will now create the joint by default using the correct API.
    attachment_helper = AttachmentHelper(sim, env_idx=0)
    # --- END MODIFIED ---

    print(f"Successfully spawned robot: {robot.cfg.prim_path}")
    print(f"Successfully spawned saw: {saw.cfg.prim_path}")
    print(f"Successfully spawned camera: {camera.cfg.prim_path}")  # NEW

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
    print(" T:   Toggle REMOVED. Attached by default.")
    print(" Camera: Capturing every 30 frames")  # NEW
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

    # --- Use stable "Idle" values from Plan Table 1 ---
    default_stiffness_tuple = (5000.0, 5000.0, 5000.0, 500.0, 500.0, 500.0)
    default_damping_ratio_tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="variable_kp",
        inertial_dynamics_decoupling=True,
        nullspace_control="position",
        motion_stiffness_task=default_stiffness_tuple,
        motion_damping_ratio_task=default_damping_ratio_tuple,
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

            # --- MODIFIED ---
            # Since we are always attached, we are always compliant.
            # Set OSC target to its *current* pose.
            target_ee_pose_b = initial_ee_pose_b.clone()
            # --- END MODIFIED ---

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

            # --- NEW: Camera capture logic ---
            if frame_count % capture_every == 0:
                save_camera_image(camera, frame_count)
            frame_count += 1

            # Debug output every 30 frames
            if frame_count % 30 == 0:
                print(f"[DEBUG] Frame {frame_count} - Loop running...")

        print("[DEBUG] Exited simulation loop normally")

    except Exception as e:
        print(f"[ERROR] Exception in simulation loop: {e}")
        import traceback

        traceback.print_exc()  # --- END CAMERA CAPTURE ---

    finally:
        # --- Unsubscribe the global function ---
        if (
            "carb_input" in locals()
            and "keyboard_sub" in locals()
            and keyboard_sub is not None
        ):
            carb_input.unsubscribe_to_keyboard_events(
                app_window.get_keyboard(), keyboard_sub
            )

        # --- Detach on exit ---
        if "attachment_helper" in locals():
            attachment_helper._detach_saw_from_ee()
        print("KeyboardHandler shutdown and joint detached.")
        # --- END ---
        simulation_app.close()


if __name__ == "__main__":
    main()
