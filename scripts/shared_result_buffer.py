#!/usr/bin/env python3
"""
Shared Result Buffer - JSON Result Sharing via Shared Memory
==============================================================

Enables bidirectional communication between VLM worker and Isaac Sim demo.
VLM detector writes classification results, demo reads them.

Usage:
    # Writer (VLM detector)
    buffer = SharedResultBuffer('vlm_results', create=True)
    buffer.write_json({'material_type': 'knot', 'confidence': 0.95})

    # Reader (Demo)
    buffer = SharedResultBuffer('vlm_results', create=False)
    result = buffer.read_latest_json()
    if result:
        print(result['material_type'])
"""

import json
import time
from multiprocessing import shared_memory
import numpy as np


class SharedResultBuffer:
    """Shared memory buffer for JSON results (VLM → Demo communication)."""

    def __init__(
        self,
        name: str = "vlm_results",
        max_json_size: int = 1024,  # 1KB sufficient for classification results
        create: bool = False,
    ):
        """
        Initialize shared result buffer.

        Args:
            name: Shared memory name
            max_json_size: Maximum JSON string size in bytes
            create: True to create buffer (producer), False to attach (consumer)
        """
        self.name = name
        self.max_json_size = max_json_size
        self.create = create

        # Buffer layout: [timestamp_ns (8 bytes)] + [json_string (max_json_size bytes)]
        self.buffer_size = 8 + max_json_size

        if create:
            # Create new buffer (producer)
            self.shm = shared_memory.SharedMemory(
                name=name, create=True, size=self.buffer_size
            )
            # Initialize with zeros
            self.shm.buf[:] = b"\x00" * self.buffer_size
            print(
                f"[SharedResultBuffer] Created '{name}' (size: {self.buffer_size} bytes)"
            )
        else:
            # Attach to existing buffer (consumer)
            self.shm = shared_memory.SharedMemory(name=name, create=False)
            print(f"[SharedResultBuffer] Attached to '{name}'")

    def write_json(self, result_dict: dict) -> bool:
        """
        Write JSON result to buffer.

        Args:
            result_dict: Dictionary to write (will be JSON-encoded)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Add timestamp
            result_dict["timestamp"] = time.time()
            result_dict["timestamp_ns"] = time.time_ns()

            # Encode to JSON
            json_str = json.dumps(result_dict)
            json_bytes = json_str.encode("utf-8")

            if len(json_bytes) > self.max_json_size:
                print(
                    f"[SharedResultBuffer] ERROR: JSON too large ({len(json_bytes)} > {self.max_json_size})"
                )
                return False

            # Write timestamp (first 8 bytes)
            timestamp_ns = result_dict["timestamp_ns"]
            self.shm.buf[0:8] = timestamp_ns.to_bytes(8, byteorder="little")

            # Write JSON string (pad with zeros)
            self.shm.buf[8 : 8 + len(json_bytes)] = json_bytes
            # Zero out remainder
            if len(json_bytes) < self.max_json_size:
                self.shm.buf[8 + len(json_bytes) : self.buffer_size] = b"\x00" * (
                    self.max_json_size - len(json_bytes)
                )

            return True

        except Exception as e:
            print(f"[SharedResultBuffer] Write error: {e}")
            return False

    def read_latest_json(self) -> dict | None:
        """
        Read latest JSON result from buffer.

        Returns:
            Dictionary with result or None if no valid data
        """
        try:
            # Read timestamp
            timestamp_ns = int.from_bytes(self.shm.buf[0:8], byteorder="little")

            # If timestamp is zero, no data written yet
            if timestamp_ns == 0:
                return None

            # Read JSON bytes
            json_bytes = bytes(self.shm.buf[8 : self.buffer_size])

            # Find null terminator
            null_idx = json_bytes.find(b"\x00")
            if null_idx != -1:
                json_bytes = json_bytes[:null_idx]

            # Decode JSON
            json_str = json_bytes.decode("utf-8")
            result_dict = json.loads(json_str)

            return result_dict

        except Exception as e:
            # Silent failure - buffer may not be initialized yet
            return None

    def close(self):
        """Close shared memory (consumer should call this)."""
        self.shm.close()
        print(f"[SharedResultBuffer] Closed '{self.name}'")

    def unlink(self):
        """Unlink shared memory (producer should call this on shutdown)."""
        if self.create:
            self.shm.unlink()
            print(f"[SharedResultBuffer] Unlinked '{self.name}'")

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        try:
            result = self.read_latest_json()
            if result:
                age_ms = (time.time_ns() - result["timestamp_ns"]) / 1e6
                return {
                    "has_data": True,
                    "age_ms": age_ms,
                    "material_type": result.get("material_type", "unknown"),
                    "confidence": result.get("confidence", 0.0),
                }
            else:
                return {"has_data": False}
        except:
            return {"has_data": False, "error": True}


# Test if run directly
if __name__ == "__main__":
    print("Testing SharedResultBuffer...")

    # Test 1: Create and write
    print("\n[Test 1] Creating buffer and writing...")
    buf_writer = SharedResultBuffer("test_results", create=True)

    test_data = {"material_type": "knot", "confidence": 0.95, "inference_time": 0.05}

    success = buf_writer.write_json(test_data)
    print(f"Write successful: {success}")

    # Test 2: Read back
    print("\n[Test 2] Reading data...")
    buf_reader = SharedResultBuffer("test_results", create=False)
    result = buf_reader.read_latest_json()

    if result:
        print(f"✓ Read successful!")
        print(f"  Material: {result['material_type']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Age: {(time.time_ns() - result['timestamp_ns']) / 1e6:.1f}ms")
    else:
        print("✗ Read failed")

    # Test 3: Stats
    print("\n[Test 3] Getting stats...")
    stats = buf_reader.get_stats()
    print(f"Stats: {stats}")

    # Cleanup
    print("\n[Cleanup]")
    buf_reader.close()
    buf_writer.close()
    buf_writer.unlink()

    print("\n✓ All tests passed!")
