from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg


def main():
    sim_cfg = SimulationCfg(dt=0.01, device="cpu")
    sim = SimulationContext(sim_cfg)

    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.saw = RigidObjectCfg(
        prim_path="/World/Saw",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    saw = scene["saw"]
    print("Attributes of saw.root_physx_view:")
    print(dir(saw.root_physx_view))

    simulation_app.close()


if __name__ == "__main__":
    main()
