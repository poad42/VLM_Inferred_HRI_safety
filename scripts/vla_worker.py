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

# Import simplified VLA model (no LeRobot dependency)
from vla_simple import SimplifiedVLA


class VLAWorker:
    """
    Asynchronous VLA worker for HRI safety monitoring.
    """
    
    def __init__(
        self,
        buffer_name: str = "hri_camera_buffer",
        safety_instruction: str = "Monitor human proximity and adjust robot stiffness for safe collaboration",
        device: str = "cuda"
    ):
        """
        Initialize VLA worker.
        
        Args:
            buffer_name: Name of shared memory buffer (must match camera producer)
            safety_instruction: Safety instruction for VLA inference
            device: 'cuda' or 'cpu' for model inference
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
        
        # Initialize VLA model
        print(f"[VLAWorker] Loading VLA model on {device}...")
        self.vla_model = SimplifiedVLA(device=device)
        
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
        # Run SmolVLA inference
        safety_params = self.vla_model.infer(
            image_np,
            instruction=self.safety_instruction
        )
        
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
                print(f"[VLAWorker] Model: {safety_params.get('model_type', 'Unknown')}")
                print(f"[VLAWorker] Safety Parameters:")
                print(f"  Safety Score:   {safety_params['safety_score']:.2f}")
                print(f"  Action:         {safety_params['action_command'].upper()}")
                print(f"  Impedance XY:   {safety_params['impedance_xy']:.2f}")
                print(f"  Impedance Z:    {safety_params['impedance_z']:.2f}")
                print(f"  Force Limit:    {safety_params['force_limit']:.2f}")
                if 'raw_output' in safety_params:
                    print(f"  VLA Output:     {safety_params['raw_output'][:100]}...")
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
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for VLA inference (default: cuda)"
    )
    
    args = parser.parse_args()
    
    # Create and run worker
    worker = VLAWorker(
        buffer_name=args.buffer_name,
        safety_instruction=args.instruction,
        device=args.device
    )
    
    worker.run(poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
