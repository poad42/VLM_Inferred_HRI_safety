"""
Runtime attachment function for decoupled rail architecture.
Creates a FixedJoint between the rail and robot base.
"""

from pxr import UsdPhysics, Gf


def attach_robot_to_rail(stage, rail_prim_path: str, robot_prim_path: str):
    """
    Creates a FixedJoint between the Rail and Robot Base.

    Args:
        stage: The USD stage
        rail_prim_path: Path to the rail rigid body (e.g., "/World/envs/env_0/Rail")
        robot_prim_path: Path to the robot root (e.g., "/World/envs/env_0/Robot")
    """
    # Define the path for the new joint
    # Attach it under the robot for organization
    joint_path = f"{robot_prim_path}/RailAttachment"

    # Create the FixedJoint
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)

    # Body0: The Rail (driver - kinematic body)
    fixed_joint.CreateBody0Rel().SetTargets([rail_prim_path])

    # Body1: The Robot Base (passenger - panda_link0)
    robot_base_path = f"{robot_prim_path}/panda_link0"
    fixed_joint.CreateBody1Rel().SetTargets([robot_base_path])

    # Set local offsets
    # Rail attachment point: Top surface of rail (Z = +0.1m from rail center)
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.1))

    # Robot base attachment point: Bottom of robot base (origin)
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # Identity rotation (no relative rotation needed)
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # Setting high break forces ensures the joint won't break
    fixed_joint.CreateBreakForceAttr().Set(1e30)
    fixed_joint.CreateBreakTorqueAttr().Set(1e30)

    print(f"[RailAttachment] Created FixedJoint at {joint_path}")
    print(f"[RailAttachment] Connected {rail_prim_path} to {robot_base_path}")

    return fixed_joint
