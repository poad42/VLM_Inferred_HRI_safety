#!/usr/bin/env python3
"""
Standalone VLM Test Script for SmolVLM-500M-Instruct
Tests the VLM on existing camera output images
"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import os
import json
from pathlib import Path
import time

class SmolVLMTester:
    """Handler for SmolVLM-500M-Instruct model"""
    
    def __init__(self, model_name="HuggingFaceTB/SmolVLM-500M-Instruct", device="auto"):
        """Initialize the VLM model"""
        print(f"[VLM] Initializing {model_name}...")
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[VLM] Using device: {self.device}")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        )
        
        print(f"[VLM] Model loaded successfully!")
        
        # Print GPU memory usage if using CUDA
        if self.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            print(f"[VLM] GPU memory allocated: {memory_allocated:.2f} GB")
    
    def describe_scene(self, image_path, prompt=None):
        """
        Process an image and get scene description
        
        Args:
            image_path: Path to image file
            prompt: Custom prompt (if None, uses default HRI safety prompt)
            
        Returns:
            dict with response, timing, and metadata
        """
        # Default HRI safety-focused prompt
        if prompt is None:
            prompt = (
                "You are a safety monitoring system for human-robot collaboration. "
                "Describe this scene focusing on: "
                "1) Robot arm position and configuration, "
                "2) Tool/saw position and state, "
                "3) Any visible motion or interaction, "
                "4) Potential safety concerns. "
                "Be concise and factual."
            )
        
        start_time = time.time()
        
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare inputs for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            prompt_text = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=prompt_text,
                images=[image],
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,  # Deterministic for consistency
                    temperature=None,
                    top_p=None,
                )
            
            # Decode response
            generated_text = self.processor.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Extract assistant's response (after the last "Assistant:" in output)
            if "Assistant:" in generated_text:
                response = generated_text.split("Assistant:")[-1].strip()
            else:
                response = generated_text.strip()
            
            processing_time = time.time() - start_time
            
            result = {
                "image_path": image_path,
                "response": response,
                "processing_time_s": processing_time,
                "success": True,
                "model": "SmolVLM-500M-Instruct"
            }
            
            print(f"\n[VLM] Processed: {os.path.basename(image_path)}")
            print(f"[VLM] Time: {processing_time:.3f}s")
            print(f"[VLM] Response: {response[:100]}...")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_result = {
                "image_path": image_path,
                "response": f"Error: {str(e)}",
                "processing_time_s": processing_time,
                "success": False,
                "model": "SmolVLM-500M-Instruct"
            }
            print(f"\n[VLM ERROR] {image_path}: {e}")
            return error_result
    
    def batch_process(self, image_dir, output_json, max_images=None):
        """
        Process all images in a directory
        
        Args:
            image_dir: Directory containing images
            output_json: Path to save results JSON
            max_images: Maximum number of images to process (None = all)
        """
        print(f"\n[VLM] Starting batch processing...")
        print(f"[VLM] Image directory: {image_dir}")
        
        # Get all image files
        image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"[VLM] Found {len(image_files)} images to process")
        
        results = []
        
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(image_dir, image_file)
            print(f"\n[VLM] Processing {i}/{len(image_files)}: {image_file}")
            
            result = self.describe_scene(image_path)
            results.append(result)
        
        # Save results
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[VLM] Batch processing complete!")
        print(f"[VLM] Results saved to: {output_json}")
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        avg_time = sum(r['processing_time_s'] for r in results) / len(results)
        
        print(f"\n[VLM] Summary:")
        print(f"  - Total images: {len(results)}")
        print(f"  - Successful: {successful}")
        print(f"  - Failed: {len(results) - successful}")
        print(f"  - Avg processing time: {avg_time:.3f}s")


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SmolVLM on camera output images")
    parser.add_argument(
        "--image_dir", 
        type=str, 
        default="./camera_output",
        help="Directory containing camera output images"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./vlm_test_results.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--max_images", 
        type=int, 
        default=5,
        help="Maximum number of images to process (for testing)"
    )
    parser.add_argument(
        "--single_image",
        type=str,
        default=None,
        help="Process single image instead of batch"
    )
    
    args = parser.parse_args()
    
    # Initialize VLM
    print("="*60)
    print("SmolVLM-500M-Instruct Test Script")
    print("="*60)
    
    vlm = SmolVLMTester()
    
    if args.single_image:
        # Test single image
        print(f"\n[TEST] Processing single image: {args.single_image}")
        result = vlm.describe_scene(args.single_image)
        
        print("\n" + "="*60)
        print("RESULT:")
        print("="*60)
        print(f"Image: {result['image_path']}")
        print(f"Time: {result['processing_time_s']:.3f}s")
        print(f"Response:\n{result['response']}")
        print("="*60)
        
    else:
        # Batch process
        vlm.batch_process(
            image_dir=args.image_dir,
            output_json=args.output,
            max_images=args.max_images
        )


if __name__ == "__main__":
    main()