# VLM-Inferred HRI Safety Project - Implementation Log

## Project Goal
Implement "VLM-Inferred Safety Constraints for Verifiable and Temporally-Aware Human-Robot Interaction"

### Core Objective
Use Vision-Language-Action (VLA) model to infer safety parameters in real-time for a collaborative robot-human sawing task, then map those to impedance control parameters with verifiable safety guarantees via energy tank.

## Architecture Pipeline
```
Camera (100Hz) → Shared Memory Buffer → VLA Inference → Safety Parameters → 
→ GRU (temporal) → Impedance Control → Energy Tank → OSC Controller → Robot
```

## Hardware
- NVIDIA L40S GPU (46GB VRAM)
- Isaac Sim in Brev dev container
- Isaac Lab v2.3.0

## Implementation Progress

### ✅ COMPLETED Steps

#### Step 1: Camera Setup (Week 1)
- **Issue**: Camera initially produced black/grayscale images
- **Solution**: Changed camera convention from "ros" to "world", adjusted position/rotation
- **Current Config**: 
  - Position: (0.5, 4.0, 0.75)
  - Rotation: (0.7071, 0.0, 0.0, -0.7071)
  - Resolution: 640×480 RGB
  - Update rate: Every 30 frames (~3.3 FPS)
- **File**: `camera_hri_demo.py` - Working Franka arm + saw + camera capture

#### Step 2: VLA Model Selection
- **Initial Plan**: Use VLM (SmolVLM-500M) + GRU
- **Pivot Decision**: Switch from VLM to VLA for direct action output
- **Final Choice**: SmolVLA-450M (lerobot/smolvla_base)
  - Size: 450M parameters (~1-2GB VRAM)
  - Speed: <100ms inference target
  - Output: Continuous action space → safety parameters
- **Reasoning**: VLA natively outputs actions, VLM requires text parsing

#### Step 3: VLA Implementation
- **Challenge**: LeRobot + Isaac Sim dependency conflicts (circular imports)
- **Solution**: Created placeholder VLA that demonstrates interface
- **File**: `test_vla.py` - Simple VLA inference with correct output format
- **Output Format**:
```python
{
    "impedance_xy": 0.65,      # Lateral stiffness [0-1]
    "impedance_z": 0.80,       # Vertical stiffness [0-1]  
    "safety_score": 0.75,      # Overall safety [0-1]
    "action_command": "continue", # continue/slow/stop
    "force_limit": 0.70,       # Force limitation [0-1]
    "confidence": 0.85         # Model confidence
}
```

#### Step 4: Shared Memory Buffer (CURRENT)
- **Motivation**: Eliminate 10-20ms disk I/O overhead for real-time operation
- **Implementation**: Lock-free ring buffer using multiprocessing.shared_memory
- **Files Created**:
  1. `shared_buffer.py` - SharedImageBuffer class (285 lines)
     - Ring buffer: 10 frames (9.2MB RAM)
     - Zero-copy numpy array sharing
     - Atomic metadata updates (write_idx, frame_count, timestamp_ns)
  
  2. `camera_hri_demo.py` - Modified to use shared memory
     - Import order fixed (SimulationApp before torch)
     - Writes frames via `write_camera_to_buffer()`
     - Initializes buffer in main()
     - Cleanup in finally block
  
  3. `vla_worker.py` - Async VLA consumer (238 lines)
     - Attaches to shared buffer (consumer mode)
     - Reads latest frame continuously
     - Runs VLA inference asynchronously
     - Outputs safety parameters
     - Graceful shutdown (Ctrl+C handling)

- **Usage**:
```bash
# Terminal 1: Camera producer
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/VLM_Inferred_HRI_safety/scripts/camera_hri_demo.py --livestream 2 --enable_cameras

# Terminal 2: VLA consumer
cd /workspace/VLM_Inferred_HRI_safety/scripts
python vla_worker.py
```

### ⏳ PENDING Steps

#### Step 5: Integrate VLA with Robot Control
- Map VLA safety outputs to OSC impedance parameters
- Connect to existing OSC controller in camera_hri_demo.py
- Real-time impedance adjustment based on VLA inference

#### Step 6: Add GRU for Temporal Reasoning
- Process sequence of VLA outputs
- Smooth safety parameter transitions
- Detect temporal patterns (approaching hazard, etc.)

#### Step 7: Energy Tank Safety System
- Map GRU outputs to energy tank parameters
- Implement verifiable safety constraints
- Connect to OSC controller

## Key Technical Decisions

### Why VLA over VLM?
- VLA: Direct continuous action output (native robotics)
- VLM: Text output requires parsing, less reliable
- VLA trained on robot data, better for safety parameters

### Why Shared Memory over Disk?
- Disk I/O: 10-20ms overhead per frame
- Shared Memory: <0.5ms overhead (zero-copy)
- Required for real-time <100ms inference loop

### Why SmolVLA-450M?
- Lightweight: 450M params (vs 7B alternatives)
- Fast: <100ms inference target
- Consumer hardware compatible
- HuggingFace available: lerobot/smolvla_base

## Current Blockers

### Isaac Sim PyTorch Corruption (Temporary)
- **Error**: Missing `/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/_structures.py`
- **Impact**: Both original and modified scripts fail
- **Solution**: Restart Brev container for fresh Isaac Sim
- **Note**: Not caused by our code changes (verified with git stash)

## Files in Repository

### Core Implementation
- `camera_hri_demo.py` - Camera + robot + shared memory (modified)
- `shared_buffer.py` - Ring buffer for camera→VLA (new)
- `test_vla.py` - VLA inference placeholder (new)
- `vla_worker.py` - Async VLA worker process (new)

### To Delete
- `test_vla_complex.py.bak` - Failed LeRobot attempt (backup)

## Next Session Instructions

After container restart:

1. **Verify Isaac Sim works**:
```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/VLM_Inferred_HRI_safety/scripts/camera_hri_demo.py --livestream 2 --enable_cameras
```

2. **Test Shared Memory Pipeline** (2 terminals):
```bash
# Terminal 1
./isaaclab/isaaclab.sh -p camera_hri_demo.py --livestream 2 --enable_cameras

# Terminal 2  
cd /workspace/VLM_Inferred_HRI_safety/scripts
python vla_worker.py
```

3. **Expected Behavior**:
   - Terminal 1: Shows `[Camera→Buffer] Frame X → Buffer #Y`
   - Terminal 2: Shows VLA processing frames with safety parameters
   - Zero disk writes (no files in camera_output/)

4. **Then Continue to Step 5**: Integrate VLA outputs with robot control

## Important Context for AI Assistant

### User's Use Case
- Collaborative sawing task (human + robot)
- Robot: Franka Panda arm
- Tool: Saw object in simulation
- Safety: Human proximity must adjust robot stiffness
- Real-time: <100ms response for safety

### Technical Constraints
- Must run in Isaac Sim environment
- VLA runs in separate Python process (avoid dependency conflicts)
- Shared memory bridges the two processes
- Output must be verifiable (energy tank requirement)

### User Preferences
- Step-by-step implementation (not all at once)
- Simplify code when possible (avoid over-engineering)
- Placeholder implementations OK if dependencies conflict
- Real model integration comes later in clean environment

## Research References

### VLA Models Evaluated
1. **SmolVLA-450M** (CHOSEN)
   - HuggingFace: lerobot/smolvla_base
   - 450M params, consumer GPU ready
   - Continuous 7-DoF action output

2. **OpenVLA-7B** (Alternative)
   - Larger, more capable
   - 3-4GB with compression
   - 200-300ms inference

3. **Pi-0.6 VLA** (Alternative)
   - Physical Intelligence model
   - Production robotics focus
   - ~4-5GB VRAM

### Papers/Resources Referenced
- LeRobot framework (HuggingFace)
- SmolVLA architecture documentation
- VLA compression techniques (VLA-Pruner, Compressor-VLA)

## Development Timeline
- Week 1: Camera setup and debugging
- Week 2: VLA model research and selection
- Week 3: Shared memory buffer implementation
- Week 4 (NEXT): Robot control integration

---

**Last Updated**: 2025-11-25
**Status**: Awaiting container restart to test shared memory pipeline
**Next Milestone**: Complete Step 5 (VLA → Robot Control Integration)
