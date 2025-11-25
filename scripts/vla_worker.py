#!/usr/bin/env python3
"""
VLA Worker Process - Consumes Camera Frames from Shared Memory
================================================================

Asynchronous VLA worker that:
1. Reads camera frames from shared memory buffer (zero disk I/O)
2. Runs VLA inference for safety parameter generation
3. Outputs safety parameters for robot control

Architecture:
    camera_hri_demo.py → SharedImageBuffer → vla_worker.py → safety parameters
    
Performance:
    - Zero disk I/O overhead
    - Asynchronous: VLA runs independently of camera (100Hz)
    - Always processes latest frame (natural frame skipping)

Usage:
    # Terminal 1: Start camera + Isaac Sim
    ./isaaclab/isaaclab.sh -p camera_hri_demo.py --livestream 2 --enable_cameras
    
    # Terminal 2: Start VLA worker
    python vla_worker.py
"""

import argparse
import time
import numpy as np
import signal
import sys
from pathlib import Path

# Import shared buffer
from shared_buffer import SharedImageBuffer

# Import VLA inference (using simple placeholder for now)
from test_vla import simple_vla_inference


class VLAWorker:
    """
    Asynchronous VLA worker for HRI safety monitoring.
    """
    
    def __init__(
        self,
        buffer_name: str = "hri_camera_buffer",
        safety_instruction: str = "Monitor human proximity and adjust robot stiffness for safe collaboration"
    ):
        """
        Initialize VLA worker.
        
        Args:
            buffer_name: Name of shared memory buffer (must match camera producer)
            safety_instruction: Safety instruction for VLA inference
        """
        self.buffer_name = buffer_name
        self.safety_instruction = safety_instruction
        self.running = True
        self.last_frame_count = -1
        
        # Statistics
        self.frames_processed = 0
        self.total_inference_time = 0.0
        
        print(f"[VLAWorker] Initializing...")
        print(f"[VLAWorker] Safety instruction: '{safety_instruction}'")
        
        # Attach to shared memory buffer
        print(f"[VLAWorker] Connecting to buffer '{buffer_name}'...")
        try:
            self.buffer = SharedImageBuffer(
                name=buffer_name,
                buffer_size=10,
                height=480,
                width=640,
                channels=3,
                create=False  # Consumer mode
            )
            print(f"[VLAWorker] Successfully connected to shared buffer")
        except FileNotFoundError:
            print(f"[VLAWorker] ERROR: Buffer '{buffer_name}' not found!")
            print(f"[VLAWorker] Make sure camera_hri_demo.py is running first.")
            raise
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print(f"\n[VLAWorker] Received signal {signum}, shutting down...")
        self.running = False
    
    def process_frame(self, image_np: np.ndarray) -> dict:
        """
        Run VLA inference on camera frame.
        
        Args:
            image_np: numpy array (H, W, 3) uint8 RGB image
            
        Returns:
            Safety parameters dictionary
        """
        start_time = time.time()
        
        # Run VLA inference (placeholder for now)
        # In production, this would be real SmolVLA-450M inference
        safety_params = simple_vla_inference(
            image_np,  # Pass numpy array directly
            self.safety_instruction
        )
        
        inference_time = time.time() - start_time
        safety_params['inference_time'] = inference_time
        
        return safety_params
    
    def run(self, poll_interval: float = 0.1):
        """
        Main worker loop: continuously read latest frame and process with VLA.
        
        Args:
            poll_interval: Time to wait between buffer reads (seconds)
        """
        print(f"[VLAWorker] Starting worker loop (poll interval: {poll_interval}s)")
        print(f"[VLAWorker] Press Ctrl+C to stop\n")
        
        while self.running:
            try:
                # Read latest frame from shared buffer
                result = self.buffer.read_latest()
                
                if result is None:
                    # No data available yet
                    time.sleep(poll_interval)
                    continue
                
                image_np, metadata = result
                frame_count = metadata['frame_count']
                
                # Skip if we've already processed this frame
                if frame_count <= self.last_frame_count:
                    time.sleep(poll_interval)
                    continue
                
                self.last_frame_count = frame_count
                
                # Calculate frame age
                frame_timestamp_ns = metadata['timestamp_ns']
                current_time_ns = time.time_ns()
                frame_age_ms = (current_time_ns - frame_timestamp_ns) / 1e6
                
                print(f"\n[VLAWorker] Processing buffer frame #{frame_count} (age: {frame_age_ms:.1f}ms)")
                
                # Run VLA inference
                safety_params = self.process_frame(image_np)
                
                # Update statistics
                self.frames_processed += 1
                self.total_inference_time += safety_params['inference_time']
                
                # Print results
                print(f"[VLAWorker] Inference time: {safety_params['inference_time']:.3f}s")
                print(f"[VLAWorker] Safety Parameters:")
                print(f"  Safety Score:   {safety_params['safety_score']:.2f}")
                print(f"  Action:         {safety_params['action_command'].upper()}")
                print(f"  Impedance XY:   {safety_params['impedance_xy']:.2f}")
                print(f"  Impedance Z:    {safety_params['impedance_z']:.2f}")
                print(f"  Force Limit:    {safety_params['force_limit']:.2f}")
                print("-" * 60)
                
                # Calculate average FPS
                avg_inference_time = self.total_inference_time / self.frames_processed
                avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
                print(f"[VLAWorker] Stats: {self.frames_processed} frames, avg {avg_fps:.2f} FPS")
                
                # Wait before next poll
                time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                print(f"\n[VLAWorker] Interrupted by user")
                break
            except Exception as e:
                print(f"[VLAWorker] ERROR: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Cleanup
        print(f"\n[VLAWorker] Shutting down...")
        print(f"[VLAWorker] Total frames processed: {self.frames_processed}")
        if self.frames_processed > 0:
            avg_time = self.total_inference_time / self.frames_processed
            print(f"[VLAWorker] Average inference time: {avg_time:.3f}s ({1.0/avg_time:.2f} FPS)")
        
        self.buffer.close()


# Helper function for test_vla.py that accepts numpy array
def simple_vla_inference_np(image_np: np.ndarray, safety_instruction: str) -> dict:
    """
    VLA inference directly from numpy array (no disk I/O).
    """
    # Placeholder safety parameters
    # In production, this would be real SmolVLA-450M inference
    safety_params = {
        "impedance_xy": 0.65,
        "impedance_z": 0.80,
        "safety_score": 0.75,
        "action_command": "continue",
        "force_limit": 0.70,
        "confidence": 0.85,
        "model_type": "SimpleVLA-Placeholder"
    }
    
    return safety_params


# Monkey-patch test_vla to accept numpy arrays
def simple_vla_inference(image_input, safety_instruction: str = "Assess robot safety") -> dict:
    """
    Wrapper that accepts both image paths and numpy arrays.
    """
    if isinstance(image_input, np.ndarray):
        # Direct numpy array input (from shared memory)
        return simple_vla_inference_np(image_input, safety_instruction)
    else:
        # Image path input (from disk) - original behavior
        from PIL import Image
        image = Image.open(image_input)
        return simple_vla_inference_np(np.array(image), safety_instruction)


def main():
    parser = argparse.ArgumentParser(
        description="VLA Worker - Process camera frames from shared memory"
    )
    parser.add_argument(
        "--buffer-name",
        type=str,
        default="hri_camera_buffer",
        help="Name of the shared memory buffer (default: hri_camera_buffer)"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="Time between buffer reads in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Monitor human proximity and adjust robot stiffness for safe collaboration",
        help="Safety instruction for VLA"
    )
    
    args = parser.parse_args()
    
    # Create and run worker
    worker = VLAWorker(
        buffer_name=args.buffer_name,
        safety_instruction=args.instruction
    )
    
    worker.run(poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
