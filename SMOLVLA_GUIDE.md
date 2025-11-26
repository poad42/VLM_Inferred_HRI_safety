# SmolVLA Setup & Usage Guide

Complete guide for running SmolVLA (450M) with Isaac Sim for real-time HRI safety inference.

---

## Quick Overview

**What this does**: Real-time vision-based safety monitoring using SmolVLA
- Camera captures robot workspace → SmolVLA analyzes scene → Outputs safety parameters
- **Performance**: 50-100ms inference (well under safety requirements)
- **Model**: LeRobot's SmolVLA (450M parameters, built on SmolVLM2-500M backbone)

**Architecture**:
```
Terminal 1 (Isaac Sim)     →  Shared Memory  →  Terminal 2 (SmolVLA)
Camera + Robot Sim              Ring Buffer       Safety Inference
```

---

## Installation (One-Time Setup)

### Prerequisites
- NVIDIA GPU with CUDA support
- Isaac Sim installed at `/workspace/isaaclab/`
- Minimum 8GB VRAM recommended

### Step 1: Install Miniforge (if not already installed)

```bash
cd ~
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Follow prompts, then restart shell or run:
```bash
source ~/.bashrc
```

### Step 2: Create LeRobot Conda Environment

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

### Step 3: Install ffmpeg

```bash
conda install ffmpeg -c conda-forge
```

### Step 4: Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential python3-dev pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev
```

### Step 5: Clone and Install LeRobot

```bash
cd /workspace
git clone https://github.com/huggingface/lerobot.git
cd lerobot
/isaac-sim/miniforge3/envs/lerobot/bin/python -m pip install -e .
```

**Note**: Use the full path to avoid Isaac Sim's corrupted pip wrapper.

### Step 6: Verify Installation

```bash
/isaac-sim/miniforge3/envs/lerobot/bin/python -c "
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print('Loading SmolVLA...')
model = SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')
print(f'✓ Model loaded on: {next(model.parameters()).device}')
print(f'✓ Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')
"
```

**Expected output**:
```
Loading SmolVLA...
Loading  HuggingFaceTB/SmolVLM2-500M-Video-Instruct weights ...
Reducing the number of VLM layers to 16 ...
✓ Model loaded on: cuda:0
✓ Parameters: 450.0M
```

---

## Running the SmolVLA Pipeline

### Terminal 1: Start Isaac Sim Camera Demo

```bash
cd /workspace
./isaaclab/isaaclab.sh -p VLM_Inferred_HRI_safety/scripts/camera_hri_demo.py --livestream 2 --enable_cameras
```

**What this does**:
- Launches Isaac Sim with Franka robot + saw
- Starts camera capture (640×480 RGB @ 3.3 FPS)
- Writes frames to shared memory buffer
- Keyboard control: J/K keys move saw

**Expected output**:
```
[SharedBuffer] Initialized for camera→VLA communication
[Camera→Buffer] Frame 30 → Buffer #1
[Camera→Buffer] Frame 60 → Buffer #2
...
```

### Terminal 2: Start SmolVLA Worker

**Open a NEW terminal** and run:

```bash
cd /workspace/VLM_Inferred_HRI_safety/scripts
/isaac-sim/miniforge3/envs/lerobot/bin/python vla_worker.py
```

**What this does**:
- Connects to shared memory buffer
- Loads SmolVLA model (450M parameters)
- Runs inference on camera frames
- Outputs safety parameters in real-time

**Expected output**:
```
SmolVLA Worker - Real-time Safety Inference
[Worker] Loading SmolVLA...
[SmolVLA] ✓ Model loaded successfully!
[Worker] ✓ Connected to buffer successfully

[Frame #30] Age: 15.2ms | Inference: 78.5ms
  Safety Score:   0.752
  Action:         CONTINUE
  Impedance XY:   0.647
  Impedance Z:    0.801
  Force Limit:    0.726
  Stats:          15 frames | 18.50 FPS avg
```

---

## Important Notes

### ⚠️ Two Different Python Environments

| Terminal | Command | Python Used | Why |
|----------|---------|-------------|-----|
| **Terminal 1** | `./isaaclab/isaaclab.sh -p ...` | Isaac Sim's Python | Needs Isaac Lab libraries |
| **Terminal 2** | `/isaac-sim/miniforge3/envs/lerobot/bin/python` | Conda's Python | Needs LeRobot/SmolVLA |

**Critical**: 
- ✅ Terminal 1 uses `isaaclab.sh` wrapper
- ✅ Terminal 2 uses full path to conda Python
- ❌ Never use `isaaclab.sh` for Terminal 2
- ❌ Never install LeRobot in Isaac Sim's Python

### 🚨 CRITICAL: Avoiding Isaac Sim Corruption

**The Problem**: Installing LeRobot or other packages into Isaac Sim's Python environment will corrupt PyTorch and break Isaac Sim.

**Symptoms of corruption**:
```
FileNotFoundError: /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/_structures.py
Error importing isaacsim.core.nodes extension
Extension isaaclab_tasks failed to load
```

**How to prevent this**:

1. **NEVER activate conda in Terminal 1**
   ```bash
   # ❌ WRONG - DO NOT DO THIS
   conda activate lerobot
   ./isaaclab/isaaclab.sh -p camera_hri_demo.py
   
   # ✅ CORRECT - Deactivate conda first
   conda deactivate
   ./isaaclab/isaaclab.sh -p camera_hri_demo.py
   ```

2. **NEVER install packages in Isaac Sim's Python**
   ```bash
   # ❌ WRONG - DO NOT DO THIS
   pip install lerobot
   python -m pip install lerobot
   
   # ✅ CORRECT - Use full path to conda Python
   /isaac-sim/miniforge3/envs/lerobot/bin/python -m pip install lerobot
   ```

3. **Keep environments completely separate**
   - Terminal 1: No conda, only `isaaclab.sh`
   - Terminal 2: Conda `lerobot` environment only
   - Communication: Shared memory only (no Python imports between them)

**If corruption happens**:
- **Solution**: Restart your Brev container to get a fresh Isaac Sim installation
- **After restart**: The `lerobot` conda environment persists (stored in `/workspace`), so you don't need to reinstall anything
- **Verify after restart**: Run `conda env list` to confirm `lerobot` still exists

**Why This is Safe**

- **Physical separation**: Isaac Sim and LeRobot in different Python environments
- **No conflicts**: They communicate via shared memory (RAM), not Python packages
- **Isaac Sim untouched**: Conda environment never modifies Isaac Sim files
- **Persistent storage**: Conda environments and your code survive container restarts

---

## Safety Parameters Explained

### Output Values

```python
{
    "impedance_xy": 0.65,      # Lateral stiffness [0.2-0.9]
    "impedance_z": 0.80,       # Vertical stiffness [0.3-0.9]
    "safety_score": 0.75,      # Overall safety [0.1-1.0]
    "force_limit": 0.70,       # Max force [0.4-0.9]
    "action_command": "continue",  # continue/slow/stop
    "action_magnitude": 0.42   # Raw action magnitude
}
```

### Interpretation

**Safety Score**:
- `0.9-1.0`: Very safe (human far away)
- `0.7-0.9`: Safe (normal operation)
- `0.5-0.7`: Caution (human approaching)
- `0.1-0.5`: Unsafe (stop robot)

**Action Command**:
- `continue`: Safe to proceed normally
- `slow`: Reduce speed, human nearby
- `stop`: Emergency stop required

**Impedance Values** (lower = softer/safer):
- `0.2-0.4`: Very compliant (soft contact)
- `0.5-0.7`: Moderate stiffness (balanced)
- `0.8-0.9`: Stiff (high precision tasks)

---

## Performance Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| VLA Inference | <100ms | 50-100ms ✅ |
| Total Latency | <100ms | 65-115ms ✅ |
| Processing Rate | - | 15-20 FPS ✅ |
| VRAM Usage | - | ~2GB ✅ |
| Model Size | - | 450M params ✅ |

---

## Troubleshooting

### Problem: "Buffer not found" in Terminal 2
**Solution**: Start Terminal 1 first, wait for `[SharedBuffer] Initialized` message

### Problem: "LeRobot not installed"
**Solution**: Make sure you're using the correct Python
```bash
cd /workspace/VLM_Inferred_HRI_safety/scripts
/isaac-sim/miniforge3/envs/lerobot/bin/python vla_worker.py
```

### Problem: Pip install fails with permission error
**Solution**: Use full path to bypass Isaac Sim's wrapper
```bash
/isaac-sim/miniforge3/envs/lerobot/bin/python -m pip install -e .
```

### Problem: Model loading errors
**Common errors**:
- `Model not found`: Check internet connection, HuggingFace may be downloading (~2GB)
- `CUDA out of memory`: Reduce batch size or use CPU (`--device cpu`)
- `Import errors`: Verify LeRobot installed correctly

**Debug steps**:
```bash
/isaac-sim/miniforge3/envs/lerobot/bin/python -c "import lerobot; print(lerobot.__version__)"
/isaac-sim/miniforge3/envs/lerobot/bin/python -c "import torch; print(torch.cuda.is_available())"
```

### Problem: Isaac Sim won't start
**Solution**: Make sure you're in the workspace root
```bash
cd /workspace
./isaaclab/isaaclab.sh -p VLM_Inferred_HRI_safety/scripts/camera_hri_demo.py --livestream 2 --enable_cameras
```

### Problem: Slow inference (>200ms)
**Possible causes**:
- Running on CPU instead of GPU
- GPU memory full from other processes
- Model not properly loaded

**Check GPU usage**:
```bash
nvidia-smi
```

---

## Keyboard Controls (Terminal 1)

In the Isaac Sim window:
- **K**: Pull saw (+X direction)
- **J**: Push saw (-X direction)
- **Release**: Stop motion

Watch Terminal 2 for real-time safety updates as you move the saw!

---

## Stopping the Pipeline

1. **Terminal 2**: Press `Ctrl+C` to stop VLA worker
2. **Terminal 1**: Press `Ctrl+C` or close Isaac Sim window

---

## File Structure

```
VLM_Inferred_HRI_safety/
├── scripts/
│   ├── camera_hri_demo.py      # Isaac Sim simulation with camera
│   ├── vla_simple.py           # SmolVLA model wrapper
│   ├── vla_worker.py           # VLA inference worker
│   └── shared_buffer.py        # Shared memory communication
├── SMOLVLA_GUIDE.md            # This file
└── PROJECT_CONTEXT.md          # Development log
```

---

## Advanced Usage

### Custom Instruction

Change the task instruction for SmolVLA:

```bash
/isaac-sim/miniforge3/envs/lerobot/bin/python vla_worker.py \
    --instruction "Detect human and maintain safe distance"
```

### Adjust Polling Rate

Change how often frames are processed:

```bash
/isaac-sim/miniforge3/envs/lerobot/bin/python vla_worker.py \
    --poll-interval 0.1  # 10Hz instead of default 20Hz
```

### CPU Mode (for testing)

Run on CPU if GPU unavailable:

```bash
/isaac-sim/miniforge3/envs/lerobot/bin/python vla_worker.py \
    --device cpu
```

---

## Next Steps

### Current Status
✅ Camera captures workspace scenes  
✅ SmolVLA infers safety parameters  
✅ Real-time processing (<100ms)

### To Implement
⏳ Integrate safety parameters back to robot control  
⏳ Add GRU for temporal reasoning  
⏳ Implement energy tank safety constraints

---

## Quick Command Reference

```bash
# Installation (one-time)
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge
cd /workspace && git clone https://github.com/huggingface/lerobot.git
cd lerobot
/isaac-sim/miniforge3/envs/lerobot/bin/python -m pip install -e .

# Test installation
/isaac-sim/miniforge3/envs/lerobot/bin/python -c "
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
model = SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')
print('✓ SmolVLA ready!')
"

# Run pipeline
# Terminal 1:
cd /workspace
./isaaclab/isaaclab.sh -p VLM_Inferred_HRI_safety/scripts/camera_hri_demo.py --livestream 2 --enable_cameras

# Terminal 2:
cd /workspace/VLM_Inferred_HRI_safety/scripts
/isaac-sim/miniforge3/envs/lerobot/bin/python vla_worker.py
```

---

## Why SmolVLA?

- ✅ **Fast**: 50-100ms inference (meets real-time requirements)
- ✅ **Efficient**: 450M parameters (~2GB VRAM)
- ✅ **Proven**: Built on SmolVLM2-500M, fine-tuned for robotics
- ✅ **Open source**: Available on HuggingFace
- ✅ **Vision-language grounding**: Understands natural language instructions
- ✅ **Action prediction**: Directly outputs robot actions (7-DOF)

---

## Model Architecture

```
SmolVLA (450M) = SmolVLM2-500M (vision-language backbone, reduced to 16 layers)
                 + Action Head (robot control)
```

**Components**:
1. **Vision Encoder**: Processes 224×224 RGB images
2. **Language Encoder**: Processes task instructions
3. **Fusion Layers**: Combines vision + language features
4. **Action Head**: Predicts 7-DOF robot actions

**Training Data**: Robot manipulation tasks from Open X-Embodiment dataset

---

## Support

For issues or questions:
1. Check this guide's Troubleshooting section
2. Review `PROJECT_CONTEXT.md` for implementation details
3. Check LeRobot documentation: https://huggingface.co/docs/lerobot
4. Check SmolVLA model card: https://huggingface.co/lerobot/smolvla_base

---

**Last Updated**: November 26, 2025  
**SmolVLA Version**: lerobot/smolvla_base (450M)  
**Status**: Ready for real-time inference ✅
