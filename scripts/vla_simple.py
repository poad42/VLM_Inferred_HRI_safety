#!/usr/bin/env python3
"""
SmolVLA for HRI Safety Assessment
==================================

Real-time vision-action model using LeRobot's SmolVLA (450M parameters).
Built on SmolVLM2-500M backbone, fine-tuned for robot control.

Performance: ~50-100ms inference on GPU

Model: https://huggingface.co/lerobot/smolvla_base
"""

import numpy as np
from PIL import Image
from typing import Dict, Union
import time
import torch


class SmolVLA:
    """
    SmolVLA (450M parameters) for real-time safety inference.
    
    Model: LeRobot's SmolVLA Base
    Training: Robot manipulation tasks with vision-language grounding
    Output: 7-DOF actions → safety parameters
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize SmolVLA model."""
        self.device = device
        self.model_loaded = False
        
        print(f"[SmolVLA] Initializing SmolVLA (450M parameters)")
        print(f"[SmolVLA] Device: {device}")
        
        try:
            # Import LeRobot SmolVLA
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.factory import make_pre_post_processors
            
            print(f"[SmolVLA] Loading pre-trained model from HuggingFace...")
            print(f"[SmolVLA] Model: lerobot/smolvla_base (~2GB download on first run)...")
            
            # Load SmolVLA model
            self.model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
            self.model = self.model.to(device)
            self.model.eval()
            
            # Setup preprocessor and postprocessor
            self.preprocess, self.postprocess = make_pre_post_processors(
                self.model.config,
                "lerobot/smolvla_base",
                preprocessor_overrides={"device_processor": {"device": device}}
            )
            
            self.model_loaded = True
            print(f"[SmolVLA] ✓ Model loaded successfully!")
            print(f"[SmolVLA] Device: {next(self.model.parameters()).device}")
            print(f"[SmolVLA] Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
                
        except ImportError as e:
            print(f"[SmolVLA] ✗ LeRobot not installed: {e}")
            print(f"[SmolVLA] Install LeRobot: pip install lerobot")
            print(f"[SmolVLA] Using placeholder mode...")
        except Exception as e:
            print(f"[SmolVLA] ✗ Error loading model: {e}")
            print(f"[SmolVLA] Using placeholder mode...")
            import traceback
            traceback.print_exc()
    
    def infer(self, image: Union[np.ndarray, Image.Image], 
              instruction: str = "Monitor human proximity and adjust robot stiffness for safe collaboration") -> Dict:
        """
        Run SmolVLA inference on camera frame.
        
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
            # Convert to PIL if numpy array
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # SmolVLA expects 256x256 images (as per model config)
            image = image.resize((256, 256))
            image_np = np.array(image, dtype=np.uint8)
            
            # Convert to CHW format and add batch dimension
            if image_np.ndim == 3 and image_np.shape[2] == 3:
                image_np = image_np.transpose(2, 0, 1)  # HWC -> CHW
            image_np = image_np[np.newaxis, ...]  # Add batch: CHW -> BCHW (1, 3, 256, 256)
            
            # Convert to torch tensors with float32 dtype (required for interpolation)
            image_tensor = torch.from_numpy(image_np).float().to(self.device)
            state_tensor = torch.zeros((1, 6), dtype=torch.float32).to(self.device)
            
            # Prepare raw observation dict
            raw_obs = {
                "observation.images.camera1": image_tensor,
                "observation.state": state_tensor,
                "task": instruction,
            }
            
            # Use preprocessor to prepare observation for model
            processed_obs = self.preprocess(raw_obs)
            
            # Run SmolVLA inference
            with torch.no_grad():
                action = self.model.select_action(processed_obs)
                
                # Extract action tensor
                if isinstance(action, dict) and "action" in action:
                    action_tensor = action["action"]
                else:
                    action_tensor = action
                
                # Convert to numpy
                if isinstance(action_tensor, torch.Tensor):
                    action_np = action_tensor.cpu().numpy().squeeze()
                else:
                    action_np = np.array(action_tensor).squeeze()
            
            # Convert to safety parameters
            safety_params = self._action_to_safety(action_np)
            safety_params["inference_time"] = time.time() - start_time
            safety_params["model_type"] = "SmolVLA-450M"
            
            return safety_params
            
        except Exception as e:
            print(f"[SmolVLA] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return self._placeholder(start_time)
    
    def _action_to_safety(self, action: np.ndarray) -> Dict:
        """
        Convert SmolVLA's 7-DOF action to safety parameters.
        
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
            print(f"[SmolVLA] Warning: Expected 7-DOF action, got {len(action)}-DOF")
            action_6d = np.zeros(6)
            action_6d[:len(action)] = action
        
        # Calculate action magnitude (proxy for scene dynamics)
        magnitude = np.linalg.norm(action_6d)
        
        # Normalize (empirical SmolVLA action range: 0-2.0)
        normalized_mag = np.clip(magnitude / 2.0, 0, 1)
        
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
    """Test SmolVLA inference."""
    print("\n" + "="*60)
    print("SmolVLA Test")
    print("="*60)
    
    # Check device availability
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nPyTorch device: {device_type}")
    if device_type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = SmolVLA(device=device_type)
    
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
        print("   SmolVLA not loaded - install: pip install lerobot")
    else:
        print(f"\n✓  SmolVLA model loaded successfully")
        print(f"   Action dims: {result.get('action_dims', 'N/A')}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    test_inference()
