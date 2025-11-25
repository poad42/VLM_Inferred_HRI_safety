#!/usr/bin/env python3
"""
Simple SmolVLA Test Script for HRI Safety
==========================================

Minimal VLA inference script that outputs safety parameters from camera images.
Uses SmolVLM-500M adapted for action output (no LeRobot dependency conflicts).

Usage:
    python test_vla.py --image frame_030.png
    python test_vla.py --batch ./camera_output
"""

import argparse
import os
import json
import time
import re
from pathlib import Path
from PIL import Image


def simple_vla_inference(image_path: str, safety_instruction: str = "Assess robot safety and human proximity") -> dict:
    """
    Simple VLA-style inference using rule-based heuristics.
    This is a placeholder until we properly set up SmolVLA in a clean environment.
    
    Args:
        image_path: Path to camera image
        safety_instruction: Safety assessment instruction
        
    Returns:
        Dictionary with safety parameters
    """
    print(f"\n[VLA] Processing: {os.path.basename(image_path)}")
    
    # Load and analyze image
    try:
        image = Image.open(image_path)
        width, height = image.size
        
        # Simple heuristic-based safety assessment
        # In real implementation, this would be SmolVLA inference
        
        # For now, use placeholder values that demonstrate the output format
        safety_params = {
            "impedance_xy": 0.65,         # Lateral stiffness [0-1]
            "impedance_z": 0.80,          # Vertical stiffness [0-1]
            "safety_score": 0.75,         # Overall safety [0-1]
            "action_command": "continue",  # continue/slow/stop
            "force_limit": 0.70,          # Force limitation [0-1]
            "confidence": 0.85,           # Model confidence
            "inference_time": 0.05,       # Placeholder inference time
            "model_type": "SimpleVLA-Placeholder"
        }
        
        print(f"[VLA] Safety Score: {safety_params['safety_score']:.2f}")
        print(f"[VLA] Action: {safety_params['action_command'].upper()}")
        print(f"[VLA] Impedance XY: {safety_params['impedance_xy']:.2f}")
        print(f"[VLA] Impedance Z: {safety_params['impedance_z']:.2f}")
        
        return safety_params
        
    except Exception as e:
        print(f"[VLA] Error processing image: {e}")
        return create_safe_fallback()


def create_safe_fallback() -> dict:
    """Conservative safe parameters when inference fails"""
    return {
        "impedance_xy": 0.3,
        "impedance_z": 0.4,
        "safety_score": 0.2,
        "action_command": "slow",
        "force_limit": 0.5,
        "confidence": 0.1,
        "error": "fallback_mode"
    }


def process_single_image(image_path: str, safety_instruction: str, verbose: bool = True):
    """Process single image and display results"""
    if not os.path.exists(image_path):
        print(f"[VLA] Error: Image file {image_path} not found")
        return None
    
    print(f"[VLA] Single image mode")
    print(f"[VLA] Safety instruction: '{safety_instruction}'")
    
    result = simple_vla_inference(image_path, safety_instruction)
    
    if verbose:
        print(f"\n[VLA] Final Safety Parameters:")
        print("=" * 50)
        for key, value in result.items():
            print(f"{key:18}: {value}")
    
    return result


def process_batch(input_dir: str, safety_instruction: str, output_file: str = "vla_safety_results.json"):
    """Process all images in directory"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"[VLA] Error: Input directory {input_dir} not found")
        return
    
    # Find image files
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    image_files = sorted([f for f in input_path.iterdir() 
                         if f.suffix.lower() in image_extensions])
    
    if not image_files:
        print(f"[VLA] No images found in {input_dir}")
        return
    
    print(f"[VLA] Batch processing {len(image_files)} images...")
    print(f"[VLA] Safety instruction: '{safety_instruction}'")
    
    results = []
    total_start = time.time()
    
    for i, image_file in enumerate(image_files):
        print(f"\n[VLA] Progress: {i+1}/{len(image_files)}")
        
        safety_params = simple_vla_inference(str(image_file), safety_instruction)
        
        result = {
            "image_file": image_file.name,
            "timestamp": time.time(),
            **safety_params
        }
        results.append(result)
    
    total_time = time.time() - total_start
    
    # Save results
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Statistics
    avg_time = total_time / len(image_files)
    print(f"\n[VLA] Batch processing complete!")
    print(f"[VLA] Results saved to: {output_path}")
    print(f"[VLA] Total time: {total_time:.2f}s")
    print(f"[VLA] Average per image: {avg_time:.3f}s ({1/avg_time:.1f} FPS)")
    
    # Summary statistics
    safe_count = sum(1 for r in results if r.get("safety_score", 0) > 0.7)
    print(f"\n[VLA] Safety Summary:")
    print(f"  High safety (>0.7): {safe_count}/{len(results)} images")


def main():
    parser = argparse.ArgumentParser(
        description="Simple VLA Safety Parameter Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_vla.py --image frame_030.png
  python test_vla.py --batch ./camera_output
  python test_vla.py --image frame_030.png --instruction "Check human proximity to saw"
"""
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Process single image file"
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Process all images in directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vla_safety_results.json",
        help="Output file for batch results (default: vla_safety_results.json)"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Monitor human proximity and adjust robot stiffness for safe collaboration",
        help="Safety instruction for VLA"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.batch:
        parser.print_help()
        print("\n[VLA] Error: Must specify either --image or --batch")
        return
    
    print("[VLA] Simple VLA Safety Assessment")
    print("[VLA] NOTE: This is a placeholder implementation.")
    print("[VLA] For production, integrate SmolVLA-450M in clean Python environment.\n")
    
    if args.image:
        # Single image mode
        process_single_image(args.image, args.instruction, args.verbose)
    
    elif args.batch:
        # Batch processing mode
        process_batch(args.batch, args.instruction, args.output)


if __name__ == "__main__":
    main()
