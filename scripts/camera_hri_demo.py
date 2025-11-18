"""
This is the final, working baseline script for Isaac Lab v2.3.0.
It successfully controls the Franka arm's end-effector with an OSC.

-- MODIFIED VERSION (v13 - Camera Added) --
This script adds minimal camera functionality to capture RGB frames.
"""

import argparse
from isaaclab.app import AppLauncher
import copy
import torch
import numpy as np
from PIL import Image
import os

# Boilerplate
parser = argparse.ArgumentParser(description="A working baseline for the Operational Space Controller.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- MOVED IMPORT TO TOP ---
from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors import CameraCfg, Camera  # ADDED CAMERA BACK
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms, quat_apply_inverse
import carb.input
import omni.appwindow
from pxr import Gf
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# --- NEW ATTACHMENT IMPORT (v11 FIX) ---
import omni.physx.scripts.physicsUtils
import omni.usd
# --- END NEW IMPORTS ---

# --- KEYBOARD IMPL (MODIFIED) ---
g_human_saw_velocity_cmd = torch.zeros(1, 6) # [vx, vy, vz, wx, wy, wz]
SAW_VELOCITY = 0.5 # m/s

# --- NEW ATTACHMENT HELPER CLASS (v12) ---
class AttachmentHelper:
    """
    Manages the state of a runtime-created fixed joint to attach
    the end-effector to the saw.
    
    MODIFIED (v12): Uses sim._stage (attribute)
    """
    def __init__(self, sim: SimulationContext, env_idx=0):
        """
        Initializes the helper and immediately creates the attachment.
        """
        self.physx_utils = omni.physx.scripts.physicsUtils
        self.stage = sim._stage 
        
        self.attachment_joint = None
        self.is_attached = False
        
        self.joint_path = f"/World/envs/env_{env_idx}/DynamicAttachmentJoint"
        
        print("[AttachmentHelper]: Initialized. Attaching by default...")
        
        self._attach_saw_to_ee(env_idx)

    def _attach_saw_to_ee(self, env_idx=0):
        """Creates a fixed joint between the EE and the saw."""
        if self.attachment_joint is not None:
            return

        ee_prim_path = f"/World/envs/env_{env_idx}/Robot/panda_hand"
        saw_prim_path = f"/World/envs/env_{env_idx}/Saw"   
        
        local_pos_ee = Gf.Vec3f(0.0, 0.0, 0.107) 
        local_rot_ee = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        local_pos_saw = Gf.Vec3f(-0.35, 0.0, 0.0) 
        local_rot_saw = Gf.Quatf(0.707, 0.0, 0.707, 0.0)

        try:
            joint_prim = self.physx_utils.add_joint_fixed(
                stage=self.stage,
                jointPath=self.joint_path,
                actor0=ee_prim_path,
                actor1=saw_prim_path,
                localPos0=local_pos_ee,
                localRot0=local_rot_ee,
                localPos1=local_pos_saw,
                localRot1=local_rot_saw,
                breakForce=1.0e30,
                breakTorque=1.0e30
            )
            
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
                omni.usd.delete_prim(self.joint_path)
                self.attachment_joint = None
                self.is_attached = False
                print(f"SUCCESS: Removed joint at {self.joint_path}")
            except Exception as e:
                print(f"Exception while removing joint: {e}")
# --- END NEW ATTACHMENT HELPER CLASS ---


# --- MODIFIED KEYBOARD HANDLER ---
def _on_keyboard_event(event, saw_object: RigidObject):
    """Callback to apply velocity to the saw object"""
    global g_human_saw_velocity_cmd
    
    if event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
        if event.input == carb.input.KeyboardInput.K:
            g_human_saw_velocity_cmd[0, 0] = SAW_VELOCITY
        elif event.input == carb.input.KeyboardInput.J:
            g_human_saw_velocity_cmd[0, 0] = -SAW_VELOCITY
            
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        if event.input in (carb.input.KeyboardInput.J, carb.input.KeyboardInput.K):
            g_human_saw_velocity_cmd[0, 0] = 0.0
# --- END MODIFIED KEYBOARD HANDLER ---


# --- MODIFIED update_states (v11) ---
def update_states(robot: Articulation, ee_frame_idx: int, arm_joint_ids: list[int]):
    """
    MODIFIED: Signature now accepts `ee_frame_idx` as an integer.
    """
    ee_jacobi_idx = ee_frame_idx - 1 
    
    jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])
    root_pos_w, root_quat_w = robot.data.root_pos_w, robot.data.root_quat_w
    
    ee_pos_w, ee_quat_w = robot.data.body_pos_w[:, ee_frame_idx], robot.data.body_quat_w[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)
    
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


# --- NEW: MINIMAL CAMERA SAVE FUNCTION ---
def save_camera_image(camera: Camera, frame_count: int, save_dir: str = "./camera_output"):
    """Minimal camera capture function - saves RGB as-is from Isaac Lab"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # Check camera data availability
        if not hasattr(camera.data, 'output') or "rgb" not in camera.data.output:
            return None
            
        # Get raw RGB data [num_envs, H, W, 3 or 4]
        rgb_data = camera.data.output["rgb"]
        if rgb_data is None:
            return None
        
        # Extract first environment, convert to numpy
        rgb_np = rgb_data[0].cpu().numpy()
        
        # Handle data type: Isaac Lab outputs float32 in [0, 1] range
        if rgb_np.dtype in (np.float32, np.float64):
            rgb_np = (np.clip(rgb_np, 0, 1) * 255).astype(np.uint8)
        
        # Drop alpha if present
        if rgb_np.shape[2] == 4:
            rgb_np = rgb_np[:, :, :3]
        
        # Save directly
        pil_image = Image.fromarray(rgb_np, mode='RGB')
        filepath = os.path.join(save_dir, f"frame_{frame_count:06d}.png")
        pil_image.save(filepath)
        
        print(f"[Camera] Saved frame {frame_count}")
        return filepath
        
    except Exception as e:
        print(f"[Camera Error] Frame {frame_count}: {e}")
        return None
# --- END CAMERA SAVE FUNCTION ---


def main():
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.5])

    scene_cfg = InteractiveSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    
    scene_cfg.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene_cfg.ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    scene_cfg.light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    
    scene_cfg.saw = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Saw",
        spawn=sim_utils.CuboidCfg(
            size=(0.7, 0.1, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.7, 0.7),
                metallic=0.8
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, 0.0, 0.5),
            rot=(0.707, 0.707, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0)
        ),
    )
    
    # --- NEW: ADD CAMERA TO SCENE ---
    # This matches your GUI camera perspective exactly
    scene_cfg.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0,  # Update every frame
        offset=CameraCfg.OffsetCfg(
            pos=(0.5, 4.0, 0.75),  # Same as your sim.set_camera_view
            rot=(0.7071, 0.0, 0.0, -0.7071),  # Looking down at the scene
            convention="world"
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1000.0),
        ),
        width=640,
        height=480,
        data_types=["rgb"],
    )
    # --- END CAMERA ADDITION ---

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    saw = scene["saw"]
    camera = scene["camera"]  # NEW: Get camera reference

    attachment_helper = AttachmentHelper(sim, env_idx=0)

    print(f"Successfully spawned robot: {robot.cfg.prim_path}")
    print(f"Successfully spawned saw: {saw.cfg.prim_path}")
    print(f"Successfully spawned camera: {camera.cfg.prim_path}")  # NEW

    sim_dt = sim.get_physics_dt()
    
    carb_input = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    keyboard_sub = carb_input.subscribe_to_keyboard_events(
        app_window.get_keyboard(),
        lambda e, s=saw: _on_keyboard_event(e, s)
    )
    print("--------------------")
    print(" Keyboard Handler Initialized...")
    print(" K:   'Pull' saw (+ X direction)")
    print(" J:   'Push' saw (- X direction)")
    print(" Camera: Capturing every 30 frames")  # NEW
    print("--------------------")
    
    saw_vel_b = torch.zeros((scene.num_envs, 6), device=sim.device)
    
    global g_human_saw_velocity_cmd
    g_human_saw_velocity_cmd = g_human_saw_velocity_cmd.to(sim.device)
    
    ee_frame_name = "panda_hand"
    ee_frame_idx_list = robot.find_bodies(ee_frame_name)
    ee_frame_idx_int = ee_frame_idx_list[0][0] 
    arm_joint_ids_list = robot.find_joints("panda_joint.*")
    arm_joint_ids = arm_joint_ids_list[0]

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
    
    _, _, _, initial_ee_pose_b, _, _, _ = update_states(robot, ee_frame_idx_int, arm_joint_ids)
    initial_ee_quat_b = initial_ee_pose_b[:, 3:].clone()
    
    joint_centers = torch.mean(robot.data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1)

    default_stiffness_tensor = torch.tensor([default_stiffness_tuple], device=sim.device)
    current_stiffness = default_stiffness_tensor.repeat(scene.num_envs, 1)
    
    # --- NEW: Camera capture variables ---
    frame_count = 0
    capture_every = 30  # Capture every 30 frames (~0.3 seconds at 100 Hz)
    # --- END CAMERA VARIABLES ---
    
    try:
        while simulation_app.is_running():
            sim.step(render=True)
            robot.update(sim_dt)
            scene.update(sim_dt)  # This updates camera data
            
            saw_vel_b[:, :6] = g_human_saw_velocity_cmd.clone()
            saw.write_root_velocity_to_sim(saw_vel_b)

            jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, joint_pos, joint_vel = update_states(robot, ee_frame_idx_int, arm_joint_ids)
            
            target_ee_pose_b = initial_ee_pose_b.clone()
            
            command = torch.cat([
                target_ee_pose_b, 
                current_stiffness
            ], dim=1)

            osc.set_command(command)
            joint_efforts = osc.compute(
                current_ee_pose_b=ee_pose_b, current_ee_vel_b=ee_vel_b, mass_matrix=mass_matrix,
                jacobian_b=jacobian_b, gravity=gravity, current_joint_pos=joint_pos,
                current_joint_vel=joint_vel, nullspace_joint_pos_target=joint_centers,
            )
            robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
            robot.write_data_to_sim()
            
            # --- NEW: Camera capture logic ---
            if frame_count % capture_every == 0:
                save_camera_image(camera, frame_count)
            frame_count += 1
            # --- END CAMERA CAPTURE ---
            
    finally:
        if 'carb_input' in locals() and 'keyboard_sub' in locals() and keyboard_sub is not None:
            carb_input.unsubscribe_to_keyboard_events(app_window.get_keyboard(), keyboard_sub)
        
        if 'attachment_helper' in locals():
            attachment_helper._detach_saw_from_ee()
        print("KeyboardHandler shutdown and joint detached.")
        simulation_app.close()

if __name__ == "__main__":
    main()
