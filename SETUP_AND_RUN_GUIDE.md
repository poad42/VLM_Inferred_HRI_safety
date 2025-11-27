# VLM-Inferred HRI Safety - Complete Setup & Run Guide

**Date**: November 27, 2025  
**Platform**: NVIDIA Brev + Isaac Lab 2.3.0  
**GPU**: NVIDIA L40S (46GB VRAM)

---

## 📋 Overview

This guide walks you through setting up and running the VLM-Inferred HRI Safety system from scratch on a fresh NVIDIA Brev machine with Isaac Lab pre-installed.

**What you'll get:**
- Real-time vision-based safety monitoring using SmolVLA (450M parameters)
- Camera captures robot workspace → SmolVLA analyzes scene → Outputs safety parameters
- ~6-12ms inference time (real-time capable)

**Architecture:**
```
Terminal 1: Isaac Sim Camera Demo (Producer)
     ↓
Shared Memory Buffer (Zero-copy, ~9MB RAM)
     ↓
Terminal 2: SmolVLA Worker (Consumer)
     ↓
Safety Parameters (impedance, force limits, action commands)
```

---

## 🚀 One-Time Setup (First Time Only)

### Prerequisites
- NVIDIA Brev machine with Isaac Lab 2.3.0 installed
- Isaac Sim 5.1.0 at `/isaac-sim/`
- Internet connection for downloading models (~2GB)

---

### Step 1: Install Miniforge (Conda)

**Environment**: Default shell (no conda)

<!-- ```bash
cd /tmp
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

**Follow the prompts:**
- Press Enter to review license
- Type `yes` to accept license
- Press Enter to use default location (`/root/miniforge3` or `$HOME/miniforge3`)
- Type `yes` to initialize Miniforge
- Type `no` to use conda python installation as default throughout (so as to be able to use it only in out conda virtual environment)


**Reload your shell:**
```bash
source ~/.bashrc
```

**Verify installation:**
```bash
conda --version
# Should show: conda 24.x.x (or similar)
```

---

### Step 2: Create LeRobot Conda Environment

**Environment**: Base conda (auto-activated after Step 1)

```bash
# Create environment with Python 3.10
conda create -y -n lerobot python=3.10

# Activate the environment
conda activate lerobot

# Install ffmpeg (required for video processing)
conda install ffmpeg -c conda-forge
```

**Verify environment:**
```bash
python --version
# Should show: Python 3.10.x
```

---

### Step 3: Clone and Install LeRobot

**Environment**: lerobot conda

```bash
# Clone LeRobot repository
cd /workspace
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Install LeRobot with SmolVLA dependencies
# Use full path to ensure correct environment and correct version of python being used

/isaac-sim/miniforge3/envs/lerobot/bin/python -m pip install -e .
/isaac-sim/miniforge3/envs/lerobot/bin/python -m pip install lerobot
/isaac-sim/miniforge3/envs/lerobot/bin/python -m pip install -e ".[smolvla]"
``` -->

**This will take 5-10 minutes.** You'll see many packages being installed.

**Verify installation:**
```bash
/isaac-sim/miniforge3/envs/lerobot/bin/python -c "import lerobot; print('✓ LeRobot installed:', lerobot.__version__)"
```

Expected output:
```
✓ LeRobot installed: 0.4.2
```

---

### Step 5: Download SmolVLA Model

**Environment**: lerobot conda

```bash
cd /workspace/VLM_Inferred_HRI_safety/scripts

# Download SmolVLA-450M from HuggingFace (~2GB)
/isaac-sim/miniforge3/envs/lerobot/bin/python -c "
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
print('Downloading SmolVLA-450M model from HuggingFace...')
print('This will download ~2GB (first time only)')
print('-' * 60)
model = SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')
print('-' * 60)
print('✓ Model downloaded successfully!')
print(f'✓ Device: {next(model.parameters()).device}')
print(f'✓ Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')
"
```

**Expected output:**
```
Downloading SmolVLA-450M model from HuggingFace...
This will download ~2GB (first time only)
------------------------------------------------------------
Loading  HuggingFaceTB/SmolVLM2-500M-Video-Instruct weights ...
model.safetensors: 100%|████████████| 2.03G/2.03G [00:02<00:00, 832MB/s]
...
Reducing the number of VLM layers to 16 ...
------------------------------------------------------------
✓ Model downloaded successfully!
✓ Device: cuda:0
✓ Parameters: 450.0M
```

The model is now cached in `~/.cache/huggingface/hub/` and won't need to be downloaded again.

---

## 🎯 Running the System (Every Time)

You'll need **two terminals** running simultaneously.

---

### Terminal 1: Isaac Sim Camera Demo (Producer)

**Environment**: NO CONDA (default shell)

**Purpose**: Runs Isaac Sim with Franka robot, saw tool, and camera capture. Writes frames to shared memory.

**Commands:**
```bash
# Make sure NO conda environment is active
conda deactivate

# Navigate to workspace root
cd /workspace


# Declare environment variable 
export ENABLE_CAMERAS=1

# Run Isaac Sim with camera demo
./isaaclab/isaaclab.sh -p /workspace/VLM_Inferred_HRI_safety/scripts/camera_hri_demo.py --livestream 2 --enable_cameras
```

**Expected output:**
```
[SharedBuffer] Initialized for camera→VLA communication
Successfully spawned robot: /World/envs/env_0/Robot
Successfully spawned saw: /World/envs/env_0/Saw
Successfully spawned camera: /World/envs/env_0/Camera
--------------------
 Keyboard Handler Initialized...
 K:   'Pull' saw (+ X direction)
 J:   'Push' saw (- X direction)
 Camera: Capturing every 30 frames
--------------------
[Camera→Buffer] Frame 30 → Buffer #1
[Camera→Buffer] Frame 60 → Buffer #2
[Camera→Buffer] Frame 90 → Buffer #3
...
```

**What to check:**
- ✅ Look for `[SharedBuffer] Initialized` - confirms shared memory buffer created
- ✅ Look for `[Camera→Buffer]` messages every ~0.3 seconds
- ✅ Web viewer at `http://<your-brev-instance>/viewer` shows Franka robot + saw

**Keep this terminal running!** Do not close it.

---

### Terminal 2: SmolVLA Worker (Consumer)

**Environment**: lerobot conda

**Purpose**: Reads camera frames from shared memory, runs SmolVLA inference, outputs safety parameters.

**Commands (in a NEW terminal):**
```bash
# Activate lerobot environment
conda activate lerobot

# Navigate to scripts directory
cd /workspace/VLM_Inferred_HRI_safety/scripts

# Run VLA worker (use full path to be safe)
/isaac-sim/miniforge3/envs/lerobot/bin/python vla_worker.py
```

**Expected output:**
```
======================================================================
SmolVLA Worker - Real-time Safety Inference
======================================================================
Buffer: hri_camera_buffer
Device: cuda
Instruction: 'Monitor human proximity and adjust robot stiffness for safe collaboration'
======================================================================

[Worker] Loading SmolVLA...
[SmolVLA] Initializing SmolVLA (450M parameters)
[SmolVLA] Device: cuda
[SmolVLA] Loading pre-trained model from HuggingFace...
Loading  HuggingFaceTB/SmolVLM2-500M-Video-Instruct weights ...
Reducing the number of VLM layers to 16 ...
[SmolVLA] ✓ Model loaded successfully!
[SmolVLA] Device: cuda:0
[SmolVLA] Parameters: 450.0M

[Worker] Connecting to shared memory buffer...
[SharedBuffer] Attached to buffer 'hri_camera_buffer'
[Worker] ✓ Connected to buffer successfully

[Worker] Starting inference loop
[Worker] Poll interval: 0.05s
[Worker] Press Ctrl+C to stop

----------------------------------------------------------------------

[Frame #118] Age: 609.6ms | Inference: 1368.8ms
  Safety Score:   0.651
  Action:         SLOW
  Impedance XY:   0.656
  Impedance Z:    0.690
  Force Limit:    0.725
  Action Mag:     0.873
  Model:          SmolVLA-450M
  Stats:          1 frames | 0.73 FPS avg
----------------------------------------------------------------------

[Frame #120] Age: 548.7ms | Inference: 10.7ms
  Safety Score:   0.686
  Action:         SLOW
  Impedance XY:   0.681
  Impedance Z:    0.712
  Force Limit:    0.743
  Action Mag:     0.784
  Model:          SmolVLA-450M
  Stats:          2 frames | 1.45 FPS avg
----------------------------------------------------------------------
```

**What to check:**
- ✅ **NOT** `Using placeholder mode` (means LeRobot installed correctly)
- ✅ `Model: SmolVLA-450M` (not `Placeholder`)
- ✅ First inference: ~1000-2000ms (model warmup - normal)
- ✅ Subsequent inferences: ~6-12ms (real-time!)
- ✅ Safety values changing each frame

**Keep this terminal running!** You'll monitor safety parameters here.

---

## 🎮 Interactive Testing

With both terminals running, you can interact with the system:

### Test 1: Move the Saw

**In Terminal 1 (Isaac Sim viewer window):**

1. **Press and hold 'K' key** - Pull saw in +X direction
   - Watch Terminal 2 for safety parameter changes
   
2. **Press and hold 'J' key** - Push saw in -X direction
   - Watch Terminal 2 for different safety values

3. **Release keys** - Stop motion
   - Safety parameters should stabilize

### Test 2: Observe Safety Changes

**Expected behavior in Terminal 2:**

- `Action Mag` increases when saw moves (higher activity detected)
- `Safety Score` may decrease with motion (dynamic = potentially less safe)
- `Impedance XY/Z` values adjust accordingly
- `Action Command` might change:
  - `CONTINUE` - Safe, proceed normally
  - `SLOW` - Caution, reduce speed
  - `STOP` - Unsafe, emergency stop

### Example Output During Motion:

```
[Frame #150] Age: 15.2ms | Inference: 8.5ms
  Safety Score:   0.752      # Higher = safer
  Action:         CONTINUE    # Safe to proceed
  Impedance XY:   0.726       # Moderate stiffness
  Impedance Z:    0.851
  Force Limit:    0.776
  Action Mag:     0.412       # Low activity
  Model:          SmolVLA-450M

[Frame #151] Age: 12.8ms | Inference: 7.3ms
  Safety Score:   0.621       # Decreased (motion detected)
  Action:         SLOW        # Caution!
  Impedance XY:   0.635       # Lower (softer/safer)
  Impedance Z:    0.673
  Force Limit:    0.711
  Action Mag:     0.947       # Higher activity
  Model:          SmolVLA-450M
```

---

## 📊 Understanding Safety Parameters

| Parameter | Range | Meaning |
|-----------|-------|---------|
| **Safety Score** | 0.1 - 1.0 | Overall safety (higher = safer) |
| **Action Command** | continue / slow / stop | Robot action recommendation |
| **Impedance XY** | 0.2 - 0.9 | Lateral stiffness (lower = softer/compliant) |
| **Impedance Z** | 0.3 - 0.9 | Vertical stiffness (lower = softer/compliant) |
| **Force Limit** | 0.4 - 0.9 | Maximum allowed force |
| **Action Mag** | 0.0 - 2.0+ | VLA-predicted motion magnitude |

### Safety Score Interpretation:

- `0.9 - 1.0`: Very safe (human far away, static scene)
- `0.7 - 0.9`: Safe (normal operation)
- `0.5 - 0.7`: Caution (human approaching, motion detected)
- `0.1 - 0.5`: Unsafe (emergency stop recommended)

---

## 🛑 Stopping the System

### Graceful Shutdown:

1. **Terminal 2**: Press `Ctrl+C` to stop VLA worker
   - Shared memory will be automatically cleaned up
   
2. **Terminal 1**: Press `Ctrl+C` or close Isaac Sim window
   - Simulation will stop and clean up resources

---

## 🔧 Troubleshooting

### Problem 1: "Buffer not found" in Terminal 2

**Cause**: Consumer started before producer

**Solution**:
```bash
# Make sure Terminal 1 is running first
# Look for this message in Terminal 1:
[SharedBuffer] Initialized for camera→VLA communication

# Then start Terminal 2
```

---

### Problem 2: "Using placeholder mode" in Terminal 2

**Cause**: LeRobot not installed in lerobot conda environment

**Solution**:
```bash
conda activate lerobot
cd /workspace/lerobot
/isaac-sim/miniforge3/envs/lerobot/bin/python -m pip install -e ".[smolvla]"
```

**Verify:**
```bash
/isaac-sim/miniforge3/envs/lerobot/bin/python -c "import lerobot; print('✓ LeRobot installed')"
```

---

### Problem 3: Terminal 1 fails with import errors

**Cause**: Conda environment is active when running Isaac Sim

**Solution**:
```bash
# Deactivate ALL conda environments
conda deactivate

# Run Isaac Sim again
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/VLM_Inferred_HRI_safety/scripts/camera_hri_demo.py --livestream 2 --enable_cameras
```

---

### Problem 4: Slow inference (>200ms)

**Check GPU usage:**
```bash
nvidia-smi
```

**Expected**: Should show Python process using GPU

**If running on CPU:**
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Should return `True`

---

### Problem 5: Model download fails

**Cause**: Network issue or HuggingFace outage

**Solution**:
```bash
# Check internet connectivity
ping huggingface.co

# Retry download
/isaac-sim/miniforge3/envs/lerobot/bin/python -c "
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
model = SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')
"
```

---

## 📁 Repository Structure

```
VLM_Inferred_HRI_safety/
├── scripts/
│   ├── camera_hri_demo.py        # Isaac Sim simulation (Terminal 1)
│   ├── vla_worker.py             # SmolVLA inference (Terminal 2)
│   ├── vla_simple.py             # SmolVLA model wrapper
│   └── shared_buffer.py          # Shared memory communication
├── PROJECT_CONTEXT.md            # Development log
├── SMOLVLA_GUIDE.md              # Detailed SmolVLA guide
├── VERIFICATION_REPORT.md        # Testing results
├── SETUP_AND_RUN_GUIDE.md        # This file
└── README.md                     # General overview
```

---

## 🎓 Key Concepts

### Why Two Python Environments?

| Environment | Python | Purpose | Packages |
|-------------|--------|---------|----------|
| **Default Shell** | Isaac Sim's Python | Robot simulation | Isaac Lab, Omniverse |
| **lerobot conda** | Miniforge Python | VLA inference | LeRobot, SmolVLA |

**Communication**: Shared memory buffer (no direct Python imports between them)

**Critical Rule**: ⚠️ NEVER install LeRobot in Isaac Sim's Python - causes corruption!

---

### How Shared Memory Works

```
Producer (Terminal 1):
  camera.capture() → numpy array → buffer.write() → RAM

Consumer (Terminal 2):
  buffer.read() → numpy array → SmolVLA.infer() → safety params
```

**Benefits**:
- **Zero-copy**: NumPy arrays share memory directly
- **Fast**: <0.5ms overhead (vs 10-20ms disk I/O)
- **Real-time**: Enables <100ms total latency

---

## 🚀 Next Steps

Current implementation (Steps 1-4 complete):
- ✅ Camera capture working
- ✅ SmolVLA inference working
- ✅ Safety parameters generated

Future work (Steps 5-7):
- ⏳ Integrate safety params → robot impedance control
- ⏳ Add GRU for temporal smoothing
- ⏳ Implement energy tank safety constraints

---

## 📚 Additional Resources

- **Isaac Lab Docs**: https://isaac-sim.github.io/IsaacLab
- **SmolVLA Model**: https://huggingface.co/lerobot/smolvla_base
- **LeRobot Docs**: https://huggingface.co/docs/lerobot
- **Project Repository**: https://github.com/poad42/VLM_Inferred_HRI_safety

---

## ✅ Quick Command Reference

### Setup (one-time):
```bash
# Install Miniforge
cd ~ && wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
source ~/.bashrc

# Create environment
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge

# Install LeRobot
cd /workspace
git clone https://github.com/huggingface/lerobot.git
cd lerobot
/isaac-sim/miniforge3/envs/lerobot/bin/python -m pip install -e ".[smolvla]"

# Download model
cd /workspace/VLM_Inferred_HRI_safety/scripts
/isaac-sim/miniforge3/envs/lerobot/bin/python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; model = SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')"
```

### Run (every time):
```bash
# Terminal 1 (NO CONDA):
conda deactivate
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/VLM_Inferred_HRI_safety/scripts/camera_hri_demo.py --livestream 2 --enable_cameras

# Terminal 2 (LEROBOT CONDA):
conda activate lerobot
cd /workspace/VLM_Inferred_HRI_safety/scripts
/isaac-sim/miniforge3/envs/lerobot/bin/python vla_worker.py
```

---

**Last Updated**: November 27, 2025  
**Status**: Fully operational ✅  
**Performance**: 6-12ms inference, real-time capable

---

*For issues or questions, check the troubleshooting section above or refer to other documentation files in this repository.*
