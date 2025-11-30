#!/usr/bin/env python3
"""
Quick test to check if camera buffer is being written.
Run this WHILE run_hri_demo.py is running.
"""

from shared_buffer import SharedImageBuffer
import time

try:
    # Attach to existing buffer
    buffer = SharedImageBuffer(
        name="hri_camera_buffer",
        buffer_size=10,
        height=480,
        width=640,
        channels=3,
        create=False,  # Consumer mode
    )

    print("✓ Buffer found!")
    print("\nMonitoring buffer (Ctrl+C to stop)...\n")

    last_frame_count = -1

    while True:
        result = buffer.read_latest()

        if result is None:
            print("⚠ No frames in buffer yet...")
        else:
            image, meta = result
            frame_count = meta["frame_count"]

            if frame_count != last_frame_count:
                print(
                    f"✓ Frame #{frame_count} | Shape: {image.shape} | Age: {(time.time_ns() - meta['timestamp_ns']) / 1e6:.1f}ms"
                )
                last_frame_count = frame_count

        time.sleep(0.5)

except FileNotFoundError:
    print("✗ Buffer 'hri_camera_buffer' not found!")
    print("\nMake sure run_hri_demo.py is running with --enable_cameras")
except KeyboardInterrupt:
    print("\n\nStopped monitoring.")
finally:
    if "buffer" in locals():
        buffer.close()
