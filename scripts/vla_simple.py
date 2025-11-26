#!/usr/bin/env python3
"""
Lightweight VLA Model for Safety Assessment
============================================

Standalone vision-action model without LeRobot dependency.
Uses MobileNetV3 (vision) + action head for safety inference.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Union
import time


class SimplifiedVLA:
    """
    Lightweight VLA for HRI safety assessment.
    
    Uses pre-trained vision encoder + simple action head.
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize model."""
        self.device = device
        
        print(f"[VLA] Initializing lightweight VLA model")
        print(f"[VLA] Device: {device}")
        
        try:
            import torchvision.models as models
            from torchvision import transforms
            
            # Load MobileNetV3 as vision encoder
            print(f"[VLA] Loading vision encoder...")
            mobilenet = models.mobilenet_v3_small(pretrained=True)
            self.vision_encoder = torch.nn.Sequential(*list(mobilenet.children())[:-1])
            self.vision_encoder = self.vision_encoder.to(device)
            self.vision_encoder.eval()
            
            # Action head: vision → 6-DOF actions
            self.action_head = torch.nn.Sequential(
                torch.nn.Linear(576, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 6)
            ).to(device)
            
            # Initialize with small weights
            for layer in self.action_head:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    torch.nn.init.zeros_(layer.bias)
            
            # Image preprocessing
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            self.model_loaded = True
            print(f"[VLA] Model loaded successfully!")
            if device == "cuda":
                print(f"[VLA] Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
        except Exception as e:
            print(f"[VLA] Error: {e}")
            self.model_loaded = False
    
    def infer(self, image: Union[np.ndarray, Image.Image], instruction: str = "") -> Dict:
        """
        Run inference on image.
        
        Args:
            image: RGB image (numpy or PIL)
            instruction: Safety instruction (for logging)
            
        Returns:
            Safety parameters dict
        """
        start_time = time.time()
        
        # Convert to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if not self.model_loaded:
            return self._placeholder(start_time)
        
        try:
            # Preprocess
            img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                features = self.vision_encoder(img_tensor)
                features = features.view(features.size(0), -1)
                action = self.action_head(features)
            
            action_np = action.cpu().numpy()[0]
            
            # Convert action to safety params
            params = self._action_to_safety(action_np)
            params["inference_time"] = time.time() - start_time
            params["model_type"] = "VLA-Standalone"
            params["raw_output"] = f"Action: [{', '.join([f'{a:.2f}' for a in action_np])}]"
            
            return params
            
        except Exception as e:
            print(f"[VLA] Inference error: {e}")
            return self._placeholder(start_time)
    
    def _action_to_safety(self, action: np.ndarray) -> Dict:
        """Convert 6-DOF action to safety parameters."""
        # Action magnitude as proxy for scene dynamics
        magnitude = np.linalg.norm(action)
        normalized_mag = min(magnitude / 2.5, 1.0)
        
        # Inverse: large action = active scene = lower safety
        safety_score = max(0.2, 1.0 - normalized_mag * 0.7)
        
        # Check action variance (erratic = less safe)
        action_std = np.std(np.abs(action))
        if action_std > 0.3:
            safety_score *= 0.8
        
        # Map to parameters
        impedance_xy = np.clip(safety_score * 0.7 + 0.2, 0.2, 0.9)
        impedance_z = np.clip(safety_score * 0.6 + 0.3, 0.3, 0.9)
        force_limit = np.clip(safety_score * 0.5 + 0.4, 0.4, 0.9)
        
        # Action command
        if safety_score > 0.7:
            cmd = "continue"
        elif safety_score > 0.45:
            cmd = "slow"
        else:
            cmd = "stop"
        
        return {
            "impedance_xy": float(impedance_xy),
            "impedance_z": float(impedance_z),
            "safety_score": float(safety_score),
            "force_limit": float(force_limit),
            "action_command": cmd,
            "confidence": 0.75
        }
    
    def _placeholder(self, start_time) -> Dict:
        """Fallback placeholder."""
        return {
            "impedance_xy": 0.65,
            "impedance_z": 0.80,
            "safety_score": 0.75,
            "action_command": "continue",
            "force_limit": 0.70,
            "confidence": 0.85,
            "inference_time": time.time() - start_time,
            "model_type": "Placeholder",
            "raw_output": "Model not loaded"
        }


def test_inference():
    """Quick test."""
    print("[Test] Initializing VLA...")
    model = SimplifiedVLA(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("[Test] Running inference...")
    result = model.infer(test_image, instruction="Test safety assessment")
    
    print("[Test] Results:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_inference()
