#!/usr/bin/env python3
"""
Octo-Base VLA for HRI Safety Assessment
========================================

Real-time vision-action model using UC Berkeley's Octo-Base (93M parameters).
Trained on 800K robot trajectories from Open X-Embodiment dataset.

Performance: ~30-50ms inference on GPU

NOTE: This implementation is compatible with Octo v1.0+ API
If you get errors, check: https://github.com/octo-models/octo
"""

import numpy as np
from PIL import Image
from typing import Dict, Union
import time


class OctoVLA:
    """
    Octo-Base VLA (93M parameters) for real-time safety inference.
    
    Model: UC Berkeley's Octo-Base
    Training: 800K robot trajectories, diverse embodiments
    Output: 7-DOF actions → safety parameters
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize Octo-Base model."""
        self.device = device
        self.model_loaded = False
        
        print(f"[OctoVLA] Initializing Octo-Base (93M parameters)")
        print(f"[OctoVLA] Device: {device}")
        
        try:
            # Try importing Octo
            from octo.model.octo_model import OctoModel
            import jax
            
            print(f"[OctoVLA] Loading pre-trained model from HuggingFace...")
            print(f"[OctoVLA] This may take a few minutes on first run (downloads ~400MB)...")
            
            # Try multiple model names (API may vary)
            model_names = [
                "hf://rail-berkeley/octo-base-1.5",
                "hf://rail-berkeley/octo-base",
                "octo-base-1.5",
                "octo-base"
            ]
            
            model_loaded_success = False
            for model_name in model_names:
                try:
                    print(f"[OctoVLA] Trying: {model_name}")
                    self.model = OctoModel.load_pretrained(model_name)
                    model_loaded_success = True
                    print(f"[OctoVLA] ✓ Successfully loaded: {model_name}")
                    break
                except Exception as e:
                    print(f"[OctoVLA]   Failed: {e}")
                    continue
            
            if not model_loaded_success:
                raise RuntimeError("Could not load any Octo model variant")
            
            self.model_loaded = True
            print(f"[OctoVLA] ✓ Model loaded successfully!")
            
            # Check JAX devices
            devices = jax.devices()
            print(f"[OctoVLA] JAX devices: {devices}")
            if device == "cuda" and devices[0].platform != 'gpu':
                print(f"[OctoVLA] WARNING: Requested CUDA but JAX using {devices[0].platform}")
                
        except ImportError as e:
            print(f"[OctoVLA] ✗ Octo not installed: {e}")
            print(f"[OctoVLA] Install: pip install octo-models")
            print(f"[OctoVLA] Using placeholder mode...")
        except Exception as e:
            print(f"[OctoVLA] ✗ Error loading model: {e}")
            print(f"[OctoVLA] Using placeholder mode...")
            import traceback
            traceback.print_exc()
    
    def infer(self, image: Union[np.ndarray, Image.Image], 
              instruction: str = "Monitor human proximity and adjust robot stiffness for safe collaboration") -> Dict:
        """
        Run Octo inference on camera frame.
        
        Args:
            image: RGB image (numpy array or PIL Image)
            instruction: Task description for VLA
            
        Returns:
            Safety parameters: {impedance_xy, impedance_z, safety_score, force_limit, action_command}
        """
        start_time = time.time()
        
        if not self.model_loaded:
            return self._placeholder(start_time)
        
        try:
            # Convert to PIL and resize to Octo's expected size (256×256)
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image = image.resize((256, 256))
            
            # Convert to numpy array [256, 256, 3] in range [0, 1]
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # Prepare observation - Octo expects dict with specific keys
            # Format: {batch_size: 1, height: 256, width: 256, channels: 3}
            observation = {
                "image_primary": image_array[np.newaxis, ...]  # Add batch dimension [1, H, W, C]
            }
            
            # Octo uses language task directly as string
            task = instruction
            
            # Run Octo inference
            # Octo API: sample_actions(observation_dict, task_string, rng=None)
            import jax
            rng = jax.random.PRNGKey(0)  # Fixed seed for deterministic results
            
            action = self.model.sample_actions(
                observation,
                task,
                rng=rng
            )
            
            # Extract action from JAX array
            # Octo returns: {action_dim: 7} = [x, y, z, roll, pitch, yaw, gripper]
            if hasattr(action, '__array__'):
                action_np = np.array(action).squeeze()
            elif isinstance(action, dict) and 'action' in action:
                action_np = np.array(action['action']).squeeze()
            else:
                action_np = np.array(action).squeeze()
            
            # Convert to safety parameters
            safety_params = self._action_to_safety(action_np)
            safety_params["inference_time"] = time.time() - start_time
            safety_params["model_type"] = "Octo-Base-93M"
            
            return safety_params
            
        except Exception as e:
            print(f"[OctoVLA] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return self._placeholder(start_time)
    
    def _action_to_safety(self, action: np.ndarray) -> Dict:
        """
        Convert Octo's 7-DOF action to safety parameters.
        
        Logic: Large predicted actions indicate dynamic/unsafe scene
               Small predicted actions indicate stable/safe scene
        """
        # Ensure action is 1D array
        action = np.array(action).flatten()
        
        # Use first 6 dimensions (ignore gripper if present)
        if len(action) >= 6:
            action_6d = action[:6]
        else:
            # Handle unexpected action dimensions
            print(f"[OctoVLA] Warning: Expected 7-DOF action, got {len(action)}-DOF")
            action_6d = np.zeros(6)
            action_6d[:len(action)] = action
        
        # Calculate action magnitude (proxy for scene dynamics)
        magnitude = np.linalg.norm(action_6d)
        
        # Normalize (empirical Octo action range: 0-1.5)
        normalized_mag = np.clip(magnitude / 1.5, 0, 1)
        
        # Inverse relationship: large action = low safety
        safety_score = np.clip(1.0 - normalized_mag * 0.8, 0.1, 1.0)
        
        # Check for erratic behavior (high variance)
        action_std = np.std(np.abs(action_6d))
        if action_std > 0.3:
            safety_score *= 0.85
        
        # Map to robot control parameters
        impedance_xy = float(np.clip(safety_score * 0.7 + 0.2, 0.2, 0.9))
        impedance_z = float(np.clip(safety_score * 0.6 + 0.3, 0.3, 0.9))
        force_limit = float(np.clip(safety_score * 0.5 + 0.4, 0.4, 0.9))
        
        # Action command based on safety threshold
        if safety_score > 0.75:
            action_command = "continue"
        elif safety_score > 0.5:
            action_command = "slow"
        else:
            action_command = "stop"
        
        return {
            "impedance_xy": impedance_xy,
            "impedance_z": impedance_z,
            "safety_score": float(safety_score),
            "force_limit": force_limit,
            "action_command": action_command,
            "confidence": 0.85,
            "action_magnitude": float(magnitude),
            "action_dims": len(action)  # For debugging
        }
    
    def _placeholder(self, start_time) -> Dict:
        """Fallback when model not loaded."""
        return {
            "impedance_xy": 0.65,
            "impedance_z": 0.80,
            "safety_score": 0.75,
            "action_command": "continue",
            "force_limit": 0.70,
            "confidence": 0.50,
            "inference_time": time.time() - start_time,
            "model_type": "Placeholder",
            "action_magnitude": 0.0
        }


def test_inference():
    """Test Octo-Base inference."""
    print("\n" + "="*60)
    print("Octo-Base VLA Test")
    print("="*60)
    
    # Check device availability
    try:
        import jax
        devices = jax.devices()
        device_type = "cuda" if devices[0].platform == 'gpu' else "cpu"
        print(f"\nJAX devices available: {devices}")
        print(f"Using device: {device_type}")
    except:
        device_type = "cpu"
        print(f"\nJAX not available, using CPU")
    
    model = OctoVLA(device=device_type)
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\n[Test] Running inference on random image...")
    result = model.infer(test_image)
    
    print("\n[Test] Results:")
    print(f"  Model Type:     {result['model_type']}")
    print(f"  Inference Time: {result['inference_time']:.3f}s")
    print(f"  Safety Score:   {result['safety_score']:.2f}")
    print(f"  Action:         {result['action_command'].upper()}")
    print(f"  Impedance XY:   {result['impedance_xy']:.2f}")
    print(f"  Impedance Z:    {result['impedance_z']:.2f}")
    print(f"  Force Limit:    {result['force_limit']:.2f}")
    print(f"  Confidence:     {result['confidence']:.2f}")
    
    if result['model_type'] == "Placeholder":
        print("\n⚠️  Running in PLACEHOLDER mode")
        print("   Octo model not loaded - install: pip install octo-models")
    else:
        print(f"\n✓  Octo model loaded successfully")
        print(f"   Action dims: {result.get('action_dims', 'N/A')}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    test_inference()
