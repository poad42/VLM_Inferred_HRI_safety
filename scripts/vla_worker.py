#!/usr/bin/env python3
"""
SmolVLA Worker - Real-time Safety Inference
============================================

Consumes camera frames from shared memory and runs SmolVLA inference.

Architecture:
    camera_hri_demo.py → Shared Memory → vla_worker.py (SmolVLA) → Safety Parameters
    
Performance:
    - Inference: ~50-100ms (SmolVLA 450M)
    - Zero disk I/O
    - Real-time processing

Usage:
    # Terminal 1: Start Isaac Sim camera demo (MUST use isaaclab.sh)
    cd /workspace
    ./isaaclab/isaaclab.sh -p VLM_Inferred_HRI_safety/scripts/camera_hri_demo.py --livestream 2 --enable_cameras
    
    # Terminal 2: Start VLA worker (use lerobot conda environment)
    cd /workspace/VLM_Inferred_HRI_safety/scripts
    /isaac-sim/miniforge3/envs/lerobot/bin/python vla_worker.py
    
    Note: Terminal 2 must use lerobot conda Python, NOT isaaclab.sh
"""

import argparse
import time
import signal
import sys

from shared_buffer import SharedImageBuffer
from vla_simple import SmolVLA


class SmolVLAWorker:
    """
    Real-time SmolVLA worker for HRI safety monitoring.
    """
    
    def __init__(
        self,
        buffer_name: str = "hri_camera_buffer",
        instruction: str = "Monitor human proximity and adjust robot stiffness for safe collaboration",
        device: str = "cuda"
    ):
        """
        Initialize SmolVLA worker.
        
        Args:
            buffer_name: Shared memory buffer name
            instruction: Task instruction for SmolVLA
            device: 'cuda' or 'cpu'
        """
        self.buffer_name = buffer_name
        self.instruction = instruction
        self.running = True
        self.last_frame_count = -1
        
        # Statistics
        self.frames_processed = 0
        self.total_inference_time = 0.0
        
        print("\n" + "="*70)
        print("SmolVLA Worker - Real-time Safety Inference")
        print("="*70)
        print(f"Buffer: {buffer_name}")
        print(f"Device: {device}")
        print(f"Instruction: '{instruction}'")
        print("="*70 + "\n")
        
        # Initialize SmolVLA model
        print(f"[Worker] Loading SmolVLA...")
        self.vla_model = SmolVLA(device=device)
        
        # Attach to shared memory buffer
        print(f"\n[Worker] Connecting to shared memory buffer...")
        try:
            self.buffer = SharedImageBuffer(
                name=buffer_name,
                buffer_size=10,
                height=480,
                width=640,
                channels=3,
                create=False  # Consumer mode
            )
            print(f"[Worker] ✓ Connected to buffer successfully\n")
        except FileNotFoundError:
            print(f"[Worker] ✗ ERROR: Buffer '{buffer_name}' not found!")
            print(f"[Worker] Start camera_hri_demo.py first:\n")
            print(f"  cd /workspace")
            print(f"  ./isaaclab/isaaclab.sh -p VLM_Inferred_HRI_safety/scripts/camera_hri_demo.py --livestream 2 --enable_cameras\n")
            raise
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print(f"\n[Worker] Shutting down...")
        self.running = False
    
    def run(self, poll_interval: float = 0.05):
        """
        Main worker loop: read frames and run SmolVLA inference.
        
        Args:
            poll_interval: Time between buffer reads (seconds, default: 0.05 = 20Hz)
        """
        print(f"[Worker] Starting inference loop")
        print(f"[Worker] Poll interval: {poll_interval}s")
        print(f"[Worker] Press Ctrl+C to stop\n")
        print("-" * 70)
        
        while self.running:
            try:
                # Read latest frame from shared buffer
                result = self.buffer.read_latest()
                
                if result is None:
                    time.sleep(poll_interval)
                    continue
                
                image_np, metadata = result
                frame_count = metadata['frame_count']
                
                # Skip if already processed
                if frame_count <= self.last_frame_count:
                    time.sleep(poll_interval)
                    continue
                
                self.last_frame_count = frame_count
                
                # Calculate latency
                frame_age_ms = (time.time_ns() - metadata['timestamp_ns']) / 1e6
                
                # Run SmolVLA inference
                safety_params = self.vla_model.infer(image_np, instruction=self.instruction)
                
                # Update statistics
                self.frames_processed += 1
                self.total_inference_time += safety_params['inference_time']
                avg_fps = self.frames_processed / self.total_inference_time if self.total_inference_time > 0 else 0
                
                # Print results
                print(f"\n[Frame #{frame_count}] Age: {frame_age_ms:.1f}ms | Inference: {safety_params['inference_time']*1000:.1f}ms")
                print(f"  Safety Score:   {safety_params['safety_score']:.3f}")
                print(f"  Action:         {safety_params['action_command'].upper()}")
                print(f"  Impedance XY:   {safety_params['impedance_xy']:.3f}")
                print(f"  Impedance Z:    {safety_params['impedance_z']:.3f}")
                print(f"  Force Limit:    {safety_params['force_limit']:.3f}")
                print(f"  Action Mag:     {safety_params.get('action_magnitude', 0):.3f}")
                print(f"  Model:          {safety_params['model_type']}")
                print(f"  Stats:          {self.frames_processed} frames | {avg_fps:.2f} FPS avg")
                print("-" * 70)
                
                time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n[Worker] ERROR: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Cleanup
        print(f"\n{'='*70}")
        print(f"Shutdown Summary")
        print(f"{'='*70}")
        print(f"Total frames processed: {self.frames_processed}")
        if self.frames_processed > 0:
            avg_time = self.total_inference_time / self.frames_processed
            print(f"Average inference time: {avg_time*1000:.1f}ms ({1.0/avg_time:.2f} FPS)")
        print(f"{'='*70}\n")
        
        self.buffer.close()


def main():
    parser = argparse.ArgumentParser(
        description="SmolVLA Worker - Real-time Safety Inference from Camera Frames"
    )
    parser.add_argument(
        "--buffer-name",
        type=str,
        default="hri_camera_buffer",
        help="Shared memory buffer name (default: hri_camera_buffer)"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
        help="Time between buffer reads in seconds (default: 0.05 = 20Hz)"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Monitor human proximity and adjust robot stiffness for safe collaboration",
        help="Task instruction for SmolVLA"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)"
    )
    
    args = parser.parse_args()
    
    # Create and run worker
    worker = SmolVLAWorker(
        buffer_name=args.buffer_name,
        instruction=args.instruction,
        device=args.device
    )
    
    worker.run(poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
