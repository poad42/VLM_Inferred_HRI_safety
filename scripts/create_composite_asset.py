import argparse
import os
import sys

# Import SimulationApp first!
from isaaclab.app import AppLauncher

# Create the parser
parser = argparse.ArgumentParser(description="Generate composite USD asset.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# NOW import pxr and other modules
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR


def create_composite_asset(output_path):
    print(f"Creating composite asset at: {output_path}")

    # 1. Create a new Stage
    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # 2. Define Rail Base as ROOT (no /World prefix for proper USD spawning)
    # This will be the default prim and articulation root
    rail_base_path = "/RailBase"
    rail_base = UsdGeom.Cube.Define(stage, rail_base_path)
    # Set as default prim so IsaacLab can spawn it correctly
    stage.SetDefaultPrim(rail_base.GetPrim())

    # Visuals for Rail Base (Long track)
    rail_base.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.05))
    rail_base.AddScaleOp().Set(Gf.Vec3d(2.0, 0.1, 0.05))  # 4m long rail
    rail_base.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.2, 0.2)])

    # Physics for Rail Base
    UsdPhysics.RigidBodyAPI.Apply(rail_base.GetPrim())
    # Add mass to make it a proper dynamic body (required for ArticulationRootAPI)
    mass_api_rail = UsdPhysics.MassAPI.Apply(rail_base.GetPrim())
    mass_api_rail.CreateMassAttr(50.0)  # Heavy base to anchor the system

    UsdPhysics.CollisionAPI.Apply(rail_base.GetPrim())

    # Apply Articulation Root to the fixed base
    UsdPhysics.ArticulationRootAPI.Apply(rail_base.GetPrim())

    # Create FixedJoint to World to hold RailBase in place
    # Body0 is None (implicit world), Body1 is RailBase
    fixed_joint_path = "/RootFixedJoint"
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)
    fixed_joint.CreateBody1Rel().AddTarget(rail_base_path)

    # 3. Define Carriage (Moving) - sibling of RailBase
    carriage_path = "/Carriage"
    carriage = UsdGeom.Xform.Define(stage, carriage_path)
    UsdPhysics.RigidBodyAPI.Apply(carriage.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(carriage.GetPrim())
    mass_api.CreateMassAttr(10.0)
    UsdPhysics.CollisionAPI.Apply(carriage.GetPrim())

    # Visual Carriage
    carriage_vis = UsdGeom.Cube.Define(stage, carriage_path + "/Visual")
    carriage_vis.AddScaleOp().Set(Gf.Vec3d(0.2, 0.2, 0.05))
    carriage_vis.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))  # Local to carriage
    carriage_vis.GetDisplayColorAttr().Set([Gf.Vec3f(0.5, 0.0, 0.0)])

    # 4. Define Prismatic Joint (Rail -> Carriage)
    joint_path = "/RailJoint"
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().AddTarget(rail_base_path)
    joint.CreateBody1Rel().AddTarget(carriage_path)
    joint.CreateAxisAttr("x")
    joint.CreateLowerLimitAttr(-1.5)
    joint.CreateUpperLimitAttr(1.5)

    # 5. Add Franka Robot (Sibling of Carriage and RailBase)
    # CRITICAL: Use NON-instanceable USD so we can modify the ArticulationRootAPI
    # The instanceable version is read-only and prevents API removal
    franka_usd_path = (
        f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    )
    robot_prim_path = "/Robot"

    robot_prim = stage.DefinePrim(robot_prim_path)
    robot_prim.GetReferences().AddReference(franka_usd_path)

    # The non-instanceable Franka USD has ArticulationRootAPI on panda_link0
    # We need to remove it so RailBase remains the only articulation root
    # Wait for the reference to load by saving and reloading
    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(output_path)

    # Now find and remove ArticulationRootAPI from the Franka robot
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    for child in robot_prim.GetAllChildren():
        if child.HasAPI(UsdPhysics.ArticulationRootAPI):
            child.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            print(f"Removed ArticulationRootAPI from {child.GetPath()}")

    # 6. Connect Carriage to Robot (Fixed Joint)
    # Robot's root link is panda_link0 (standard Franka structure)
    robot_root_link_path = robot_prim_path + "/panda_link0"

    carriage_robot_joint_path = "/CarriageToRobotJoint"
    cr_joint = UsdPhysics.FixedJoint.Define(stage, carriage_robot_joint_path)
    cr_joint.CreateBody0Rel().AddTarget(carriage_path)
    cr_joint.CreateBody1Rel().AddTarget(robot_root_link_path)

    # 7. Add Saw to Robot End-Effector
    panda_hand_path = robot_prim_path + "/panda_hand"

    # Define Saw Prim
    saw_path = panda_hand_path + "/Saw"
    saw = UsdGeom.Cube.Define(stage, saw_path)

    # Saw Geometry: (0.7, 0.1, 0.02)
    saw.AddScaleOp().Set(Gf.Vec3d(0.7, 0.1, 0.02))
    saw.GetDisplayColorAttr().Set([Gf.Vec3f(0.7, 0.7, 0.7)])

    # Saw Transform relative to Hand
    xform = saw.AddTransformOp()
    mat_trans = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, 0.0, 0.107))
    rot = Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
    mat_rot = Gf.Matrix4d().SetRotate(rot)
    mat_final = mat_rot * mat_trans
    xform.Set(mat_final)

    # Physics for Saw (Collision only, attached to hand)
    UsdPhysics.CollisionAPI.Apply(saw.GetPrim())

    # Physical Material
    mat_path = "/Looks/SawMaterial"
    material = UsdShade.Material.Define(stage, mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    phys_mat.CreateStaticFrictionAttr(0.8)
    phys_mat.CreateDynamicFrictionAttr(0.5)
    phys_mat.CreateRestitutionAttr(0.0)

    binding_api = UsdShade.MaterialBindingAPI.Apply(saw.GetPrim())
    binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

    # Save
    stage.GetRootLayer().Save()
    print("Done!")


if __name__ == "__main__":
    # Output path in the same directory as scripts
    output_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to VLM_Inferred_HRI_safety
    base_dir = os.path.dirname(output_dir)
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    output_usd = os.path.join(assets_dir, "rail_franka_saw.usd")
    create_composite_asset(output_usd)

    # Close app
    simulation_app.close()
