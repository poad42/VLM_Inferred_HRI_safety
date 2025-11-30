#!/usr/bin/env python3
"""
VLM Material Detector - Simplified for Material Classification
================================================================

Reads camera frames from shared buffer and classifies wood material.

Usage:
    # Terminal 1: Run Isaac Sim with camera
    ./isaaclab.sh -p VLM_Inferred_HRI_safety/scripts/run_hri_demo.py --enable_cameras

    # Terminal 2: Run VLM detector (use lerobot conda environment)
    /isaac-sim/miniforge3/envs/lerobot/bin/python vlm_material_detector.py
"""

import time
import signal
import sys
from shared_buffer import SharedImageBuffer
from shared_result_buffer import SharedResultBuffer


# Placeholder for VLM model - replace with actual SmolVLA
class MaterialVLM:
    """Simplified VLM for material classification."""

    def __init__(self, device="cuda"):
        self.device = device
        print(f"[VLM] Initializing material classifier on {device}...")
        # TODO: Load actual SmolVLA model here
        print(f"[VLM] ✓ Model loaded")

    def classify_material(self, image_np):
        """
        Classify wood material from image.

        Args:
            image_np: numpy array (H, W, 3) RGB image

        Returns:
            dict with 'material_type', 'confidence', 'inference_time'
        """
        start_time = time.time()

        # TODO: Replace with actual VLM inference
        # For now, return mock data
        material_type = "soft_wood"  # or "knot" or "cracked"
        confidence = 0.95

        inference_time = time.time() - start_time

        return {
            "material_type": material_type,
            "confidence": confidence,
            "inference_time": inference_time,
        }


class MaterialDetectorWorker:
    """Real-time material detection worker."""

    def __init__(
        self,
        buffer_name="hri_camera_buffer",
        result_buffer_name="vlm_results",
        device="cuda",
    ):
        self.buffer_name = buffer_name
        self.result_buffer_name = result_buffer_name
        self.running = True
        self.last_frame_count = -1

        print("\n" + "=" * 70)
        print("VLM Material Detector")
        print("=" * 70)

        # Initialize VLM
        self.vlm = MaterialVLM(device=device)

        # Connect to camera buffer (consumer)
        print(f"\n[Worker] Connecting to camera buffer '{buffer_name}'...")
        try:
            self.buffer = SharedImageBuffer(
                name=buffer_name,
                buffer_size=10,
                height=480,
                width=640,
                channels=3,
                create=False,  # Consumer mode
            )
            print(f"[Worker] ✓ Connected to camera buffer")
        except FileNotFoundError:
            print(f"[Worker] ✗ ERROR: Camera buffer not found!")
            print(f"[Worker] Start run_hri_demo.py with --enable_cameras first\n")
            raise

        # Create result buffer (producer)
        print(f"[Worker] Creating result buffer '{result_buffer_name}'...")
        try:
            self.result_buffer = SharedResultBuffer(
                name=result_buffer_name, create=True
            )
            print(f"[Worker] ✓ Result buffer created\n")
        except Exception as e:
            print(f"[Worker] ✗ ERROR creating result buffer: {e}\n")
            raise

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\n[Worker] Shutting down...")
        self.running = False

    def run(self, poll_interval=0.33):
        """
        Main loop: classify material at ~3 FPS.

        Args:
            poll_interval: Time between classifications (default: 0.33s = 3 FPS)
        """
        print(f"[Worker] Starting at ~{1.0/poll_interval:.1f} FPS")
        print(f"[Worker] Press Ctrl+C to stop\n")
        print("-" * 70)

        while self.running:
            try:
                # Read latest frame
                result = self.buffer.read_latest()

                if result is None:
                    time.sleep(poll_interval)
                    continue

                image_np, metadata = result
                frame_count = metadata["frame_count"]

                # Skip if already processed
                if frame_count <= self.last_frame_count:
                    time.sleep(poll_interval)
                    continue

                self.last_frame_count = frame_count

                # Classify material
                result = self.vlm.classify_material(image_np)

                # Print result
                print(f"\n[Frame #{frame_count}]")
                print(f"  Material:   {result['material_type'].upper()}")
                print(f"  Confidence: {result['confidence']:.2%}")
                print(f"  Inference:  {result['inference_time']*1000:.1f}ms")
                print("-" * 70)

                # Write result to shared buffer for run_hri_demo.py to read
                self.result_buffer.write_json(
                    {
                        "material_type": result["material_type"],
                        "confidence": result["confidence"],
                        "inference_time": result["inference_time"],
                    }
                )

                time.sleep(poll_interval)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n[Worker] ERROR: {e}")
                import traceback

                traceback.print_exc()
                break

        self.buffer.close()
        self.result_buffer.close()
        self.result_buffer.unlink()  # Clean up result buffer


if __name__ == "__main__":
    worker = MaterialDetectorWorker(device="cuda")
    worker.run(poll_interval=0.33)  # 3 FPS
