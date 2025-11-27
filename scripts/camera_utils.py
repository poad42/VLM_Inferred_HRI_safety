import os
import numpy as np
from PIL import Image
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, Camera


def add_camera_to_scene(scene_cfg):
    """
    Adds a camera configuration to the scene config.
    Matches the GUI camera perspective.
    """
    scene_cfg.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0,  # Update every frame
        offset=CameraCfg.OffsetCfg(
            pos=(0.5, 4.0, 0.75),  # Same as sim.set_camera_view
            rot=(0.7071, 0.0, 0.0, -0.7071),  # Looking down at the scene
            convention="world",
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


def save_camera_image(
    camera: Camera, frame_count: int, save_dir: str = "./camera_output"
):
    """Minimal camera capture function - saves RGB as-is from Isaac Lab"""
    try:
        os.makedirs(save_dir, exist_ok=True)

        # Check camera data availability
        if not hasattr(camera.data, "output") or "rgb" not in camera.data.output:
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
        pil_image = Image.fromarray(rgb_np, mode="RGB")
        filepath = os.path.join(save_dir, f"frame_{frame_count:06d}.png")
        pil_image.save(filepath)

        print(f"[Camera] Saved frame {frame_count}")
        return filepath

    except Exception as e:
        print(f"[Camera Error] Frame {frame_count}: {e}")
        return None
