"""
This script implements the fixed 2-Day HRI prototype for adaptive impedance control
in NVIDIA Isaac Lab.

The final runtime error is fixed by updating the arguments for physx_utils.createJoint
to use positional arguments for the prim paths, which is the necessary API fix.
"""

import argparse
from isaaclab.app import AppLauncher
import copy


def main():
    """Main function to launch the simulation."""

    parser = argparse.ArgumentParser(
        description="Fixed HRI Adaptive Controller Prototype."
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to spawn."
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
    import torch
    import carb.input
    import omni.timeline
    from isaaclab.sim import SimulationCfg, SimulationContext
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.assets import Articulation
    from isaaclab.controllers import (
        OperationalSpaceController,
        OperationalSpaceControllerCfg,
    )
    import omni.physx.scripts.utils as physx_utils
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdGeom, Gf, UsdPhysics

    class KeyboardHandler:
        """
        Handles keyboard inputs by subscribing to carb.input events.
        """
        def __init__(self, device: str = "cpu"):
            self.pull_command = False
            self.push_command = False
            self.stuck_blade = False
            self.stiffness_mode = "high"

            self._appwindow = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._appwindow.get_keyboard(), self._on_key_event
            )
            print("--------------------")
            print(" Keyboard Handler Initialized. Use keyboard to control:")
            print(" J/K:   Pull / Push robot (+/- X direction)")
            print(" G/S:   'Good' (High Stiffness) / 'Safe' (Low Stiffness)")
            print(" Z (Hold): Simulate 'Stuck Blade' (Apply opposing force)")
            print("--------------------")

        def _on_key_event(self, event: carb.input.KeyboardEvent, *args, **kwargs):
            if event.type == carb.input.KeyboardEventType.KEY_PRESS or \
               event.type == carb.input.KeyboardEventType.KEY_REPEAT:
                if event.input == carb.input.KeyboardInput.J:
                    self.pull_command = True
                elif event.input == carb.input.KeyboardInput.K:
                    self.push_command = True
                elif event.input == carb.input.KeyboardInput.G:
                    self.stiffness_mode = "high"
                elif event.input == carb.input.KeyboardInput.S:
                    self.stiffness_mode = "low"
                elif event.input == carb.input.KeyboardInput.Z:
                    self.stuck_blade = True

            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                if event.input == carb.input.KeyboardInput.J:
                    self.pull_command = False
                elif event.input == carb.input.KeyboardInput.K:
                    self.push_command = False
                elif event.input == carb.input.KeyboardInput.Z:
                    self.stuck_blade = False

        def shutdown(self):
            if self._input and self._keyboard_sub is not None:
                self._input.unsubscribe_to_keyboard_events(
                    self._appwindow.get_keyboard(), self._keyboard_sub
                )
            print("KeyboardHandler shutdown.")

    def create_saw_prim(stage, prim_path: str, position: Gf.Vec3f, scale: Gf.Vec3f):
        saw_prim = UsdGeom.Cube.Define(stage, prim_path)
        xformable = UsdGeom.Xformable(saw_prim)
        xformable.AddTranslateOp().Set(position)
        xformable.AddScaleOp().Set(scale)
        physx_utils.setRigidBody(saw_prim.GetPrim(), "convexHull", False)
        mass_api = UsdPhysics.MassAPI.Apply(saw_prim.GetPrim())
        mass_api.CreateMassAttr(0.5)
        print(f"Created physics-enabled 'saw' prim at {prim_path}")

    class HriDemo:
        """Main simulation class for the HRI prototype."""
        def __init__(self, sim_cfg, num_envs: int):
            self.sim = SimulationContext(sim_cfg)
            self.device = self.sim.device
            self.num_envs = num_envs
            
            self.robot_prim_path = f"/World/envs/env_.*/Robot"
            self.saw_prim_path = f"/World/envs/env_0/Saw"

            scene_cfg = InteractiveSceneCfg(num_envs=self.num_envs, env_spacing=2.0)
            
            franka_cfg = copy.deepcopy(FRANKA_PANDA_CFG)
            franka_cfg.prim_path = self.robot_prim_path
            franka_cfg.init_state.pos = (0.0, 0.0, 0.0)

            # --- BUG FIX STARTS HERE ---
            # Explicitly define the 7-DoF arm joints.
            # This excludes the 2 gripper joints ("panda_finger_joint1", "panda_finger_joint2").
            arm_joint_names = [f"panda_joint{i}" for i in range(1, 8)] # panda_joint1 to panda_joint7
            
            # Set the config to only use these joints.
            # This will make get_...() methods return 7-DoF tensors.
            franka_cfg.joint_names = arm_joint_names
            
            # Also define the end-effector body name, which is good practice.
            franka_cfg.ee_body_name = "panda_hand"

            scene_cfg.robot = franka_cfg
            
            self.scene = InteractiveScene(scene_cfg)

            osc_cfg = OperationalSpaceControllerCfg(
                target_types=["wrench_abs"],
                impedance_mode="variable_kp",
                motion_stiffness_task=(100.0, 100.0, 100.0, 50.0, 50.0, 50.0),
                motion_damping_ratio_task=(0.7, 0.7, 0.7, 0.7, 0.7, 0.7),
            )
            self.osc_controller = OperationalSpaceController(
                osc_cfg, self.num_envs, self.device
            )

            self.keyboard_handler = KeyboardHandler(device=self.device)

            self.high_stiffness = torch.tensor(
                [[150.0, 150.0, 150.0, 75.0, 75.0, 75.0]], device=self.device
            )
            self.low_stiffness = torch.tensor(
                [[15.0, 15.0, 15.0, 7.5, 7.5, 7.5]], device=self.device
            )
            self.joint_created = False
            
            self.robot_articulation = None

            self.hand_body_index = None # Add this line
            
            self.setup_scene()
            
            print("Controller ready.")

        def setup_scene(self):
            stage = get_current_stage()
            
            ground_plane_cfg = sim_utils.GroundPlaneCfg()
            ground_plane_cfg.func("/World/defaultGroundPlane", ground_plane_cfg)

            wall_prim = stage.GetPrimAtPath("/World/Wall")
            if wall_prim:
                stage.RemovePrim(wall_prim.GetPath())
                print("Removed default wall prim.")

            create_saw_prim(
                stage,
                prim_path=self.saw_prim_path,
                position=Gf.Vec3f(0.4, 0.0, 0.5),
                scale=Gf.Vec3f(0.05, 0.8, 0.1),
            )

            self.sim.reset()
            self.scene.reset()
            
            self.robot_articulation = self.scene.articulations["robot"]

            print("Simulation scene setup complete.")

        def _on_simulation_step(self, dt):
            if not self.joint_created:
                # ... (joint creation is fine) ...
                stage = get_current_stage()
                hand_prim_path = self.robot_prim_path.replace(".*", "0") + "/panda_hand"
                hand_prim = stage.GetPrimAtPath(hand_prim_path)
                saw_prim = stage.GetPrimAtPath(self.saw_prim_path)
                physx_utils.createJoint(stage, "Fixed", hand_prim, saw_prim)
                self.joint_created = True
                print(
                    f"Created Fixed Joint between '{hand_prim_path}' and '{self.saw_prim_path}'"
                )

            # 1. One-time setup: Find the tensor index
            if self.hand_body_index is None:
                self.hand_body_index = self.robot_articulation.find_bodies("panda_hand")[0][0]
                print(f"Found 'panda_hand' at body index: {self.hand_body_index}")

            # 2. Get 9-DoF data from physx
            all_jacobians = self.robot_articulation.root_physx_view.get_jacobians()
            
            # --- BUG FIX Part 1: Keep the 'num_bodies' dimension ---
            # Slice with a range to keep the dimension: shape (1, 1, 6, 9)
            jacobian_9dof = all_jacobians[:, self.hand_body_index:self.hand_body_index+1, :, :]
            # --- END FIX Part 1 ---
            
            # Mass matrix is fine, it's not per-body
            mass_matrix_9dof = self.robot_articulation.root_physx_view.get_generalized_mass_matrices()

            all_body_states = self.robot_articulation.data.body_state_w
            hand_state = all_body_states[:, self.hand_body_index, :]
            
            ee_pose = hand_state[:, 0:7]  # (pos, quat)
            ee_vel = hand_state[:, 7:13]  # (lin_vel, ang_vel)

            # 3. Slice inputs from 9-DoF to 7-DoF
            # jacobian_9dof is (1, 1, 6, 9) -> jacobian_7dof is (1, 1, 6, 7)
            jacobian_7dof = jacobian_9dof[..., 0:7]
            # mass_matrix_9dof is (1, 9, 9) -> mass_matrix_7dof is (1, 7, 7)
            mass_matrix_7dof = mass_matrix_9dof[..., 0:7, 0:7]

            # 4. Get commands
            # ... (command logic is fine) ...
            wrench_command = torch.zeros((self.num_envs, 6), device=self.device)
            if self.keyboard_handler.pull_command:
                wrench_command[:, 0] = -15.0
            elif self.keyboard_handler.push_command:
                wrench_command[:, 0] = 15.0

            stiffness_command = torch.zeros((self.num_envs, 6), device=self.device)
            if self.keyboard_handler.stiffness_mode == 'high':
                stiffness_command[:] = self.high_stiffness
            else:
                stiffness_command[:] = self.low_stiffness
            
            final_command_tensor = torch.cat([wrench_command, stiffness_command], dim=1)
            self.osc_controller.set_command(final_command_tensor)

            # 5. DEBUGGING STATEMENTS
            print("--- OSC DEBUG ---")
            try:
                # Print shapes (Jacobian should now be 4D)
                print(f"  ee_pose shape: {ee_pose.shape}")
                print(f"  ee_vel shape: {ee_vel.shape}")
                print(f"  jacobian shape: {jacobian_7dof.shape}") # Should be [1, 1, 6, 7]
                print(f"  mass_matrix shape: {mass_matrix_7dof.shape}") # Should be [1, 7, 7]

                # 6. Compute torques
                joint_torques_7dof = self.osc_controller.compute(
                    ee_pose, ee_vel, jacobian_7dof, mass_matrix_7dof
                )
                print(f"  OSC compute SUCCEEDED. Output torques shape: {joint_torques_7dof.shape}")
                print("-------------------")

            except Exception as e:
                print(f"  !! OSC compute FAILED !!")
                import traceback
                traceback.print_exc()
                print("-------------------")
                raise e
            
            # 7. Pad outputs from 7-DoF to 9-DoF
            # (This logic is correct from the previous step)
            gripper_torques = torch.zeros((self.num_envs, 2), device=self.device)
            joint_torques_9dof = torch.cat([joint_torques_7dof, gripper_torques], dim=1)

            # 8. Apply 9-DoF torques
            self.robot_articulation.set_joint_effort_target(joint_torques_9dof)

            if self.keyboard_handler.stuck_blade:
                # ... (apply wrench logic is fine) ...
                opposing_wrench = torch.zeros((self.num_envs, 6), device=self.device)
                opposing_wrench[:, 0] = 50.0
                self.robot_articulation.apply_body_wrench(
                    opposing_wrench, body_names=["panda_hand"]
                )
                
        def run(self):
            self.sim.step()

            while simulation_app.is_running():
                self.scene.update(self.sim.get_physics_dt())
                self._on_simulation_step(self.sim.get_physics_dt())
                self.sim.step()

    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    demo = HriDemo(sim_cfg, num_envs=args_cli.num_envs)
    
    try:
        demo.run()
    except Exception as e:
        print(f"Error in simulation loop: {e}")
    finally:
        demo.keyboard_handler.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()