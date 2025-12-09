# VLM-Inferred HRI Safety Demo

This document provides step-by-step instructions to run the human-robot interaction demo with vision-language model material detection.

---

## Prerequisites
- **Official Requirements Documentation** https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html 
- **NVIDIA GPU** (recommended for Isaac Sim, RTX 3060+ or better)
- **Isaac Sim 4.0.0+** installed
- **Isaac Lab 2.3.0+** installed
- **Conda** or **Miniconda** installed

---

## Installation

### Step 1: Install Isaac Lab

Follow the official [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

We recommend using the **conda installation** for easier Python environment management.

### Step 2: Clone This Repository

```bash
# Clone outside the IsaacLab directory
cd ~/Downloads
git clone https://github.com/poad42/VLM_Inferred_HRI_safety.git
cd VLM_Inferred_HRI_safety
```

### Step 3: Create Conda Environment

```bash
# Create new conda environment with Python 3.10
conda create -n vlm_hri python=3.10 -y
conda activate vlm_hri
```

### Step 4: Install Python Dependencies

```bash
# Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python pillow scipy
pip install transformers accelerate

# Install Isaac Lab extension (from this repository)
pip install -e source/VLM_Inferred_HRI_safety
```

**Note**: Adjust the CUDA version (`cu118`) based on your system. For CPU-only: `pip install torch torchvision`

---

## Running the Demo

The demo consists of two components that run in separate terminals:

### Terminal 1: Isaac Sim Main Simulation

This runs the robot simulation with interactive keyboard controls.

```bash
# Navigate to repository root
cd ~/Downloads/VLM_Inferred_HRI_safety

# Activate conda environment
conda activate vlm_hri

# Run simulation (Oracle mode - ground truth material detection)
./isaaclab.sh -p scripts/run_hri_demo.py --enable_cameras

# OR run with VLM mode enabled
./isaaclab.sh -p scripts/run_hri_demo.py --enable_cameras --vlm_mode
```

**If `isaaclab.sh` is not in your PATH:**
```bash
# Use absolute path to Isaac Lab installation
/path/to/IsaacLab/isaaclab.sh -p scripts/run_hri_demo.py --enable_cameras
```

### Terminal 2: VLM Material Detector (Optional)

Only needed if running in `--vlm_mode`. This process reads camera frames and detects materials using the vision-language model.

```bash
# In a new terminal
cd ~/Downloads/VLM_Inferred_HRI_safety
conda activate vlm_hri

# Run VLM detector
python scripts/vlm_material_detector.py
```

---

## Keyboard Controls

Once the simulation is running, use these keys to interact:

### Force Application
- **J**: Push saw forward (+20 N along gripper X-axis)
- **K**: Pull saw backward (-20 N along gripper X-axis)
- **F**: Press saw downward (-10 N along gripper Z-axis)

### Position Control
- **Arrow Up**: Move gripper upward (+1 cm Z-axis)
- **Arrow Down**: Move gripper downward (-1 cm Z-axis)
- **Arrow Left**: Move gripper left (-1 cm Y-axis)
- **Arrow Right**: Move gripper right (+1 cm Y-axis)

### Mode Selection
- **O**: Oracle Mode (ground truth material detection from Y-position)
- **V**: VLM Mode (vision-based material detection using camera)
- **P**: Penalty Mode (inverted stiffness for user study)

---

## Operating Modes

### 1. Oracle Mode (Default)
- Uses blade tip Y-position to determine material zone
- **Soft Wood** (Y < -0.125m): Stiffness = 500 N/m
- **Knot** (-0.125m ≤ Y < 0.125m): Stiffness = 600 N/m
- **Cracked** (Y ≥ 0.125m): Stiffness = 600 N/m
- No VLM detector needed

### 2. VLM Mode
- Uses camera + SmolVLM2-500M model for material detection
- Requires Terminal 2 (VLM detector) running
- Real-time vision processing at ~10 Hz
- Validates detection using scene graphs + self-correction

### 3. Penalty Mode
- Intentionally provides **incorrect** stiffness values
- Used for user study to compare performance
- Soft wood → 600 N/m (should be 500)
- Knot/Cracked → 400 N/m (should be 600)

---

## Expected Behavior

When you **press F key** (downward force):
- **Soft Wood Zone**: Saw descends ~2.0 cm (500 N/m stiffness)
- **Knot/Cracked Zone**: Saw descends ~1.7 cm (600 N/m stiffness)

The difference in compliance creates a **haptic perception** of material hardness.

Stiffness transitions smoothly at **200 N/m/s** (0.5 seconds to fully transition).

---

## File Structure

```
VLM_Inferred_HRI_safety/
├── DEMO.md                          # This file
├── README.md                        # Template documentation
├── scripts/
│   ├── run_hri_demo.py              # Main simulation (Terminal 1)
│   ├── vlm_material_detector.py     # VLM detector (Terminal 2)
│   ├── shared_buffer.py             # Camera → VLM shared memory
│   ├── shared_result_buffer.py      # VLM → Simulation shared memory
│   ├── material_zones.py            # Material property definitions
│   └── camera_utils.py              # Camera configuration
├── assets/
│   └── rail_franka_saw.usd          # 3D assets (robot, saw, log)
└── source/
    └── VLM_Inferred_HRI_safety/     # Isaac Lab extension
```

---

## Quick Start (Oracle Mode - No VLM)

For a simple demo **without vision processing**:

```bash
# Single terminal only
cd ~/Downloads/VLM_Inferred_HRI_safety
conda activate vlm_hri
./isaaclab.sh -p scripts/run_hri_demo.py --enable_cameras

# Press 'O' for Oracle mode (default)
# Use J/K/F keys to apply forces
# Use Arrow keys to move gripper
# Move Left/Right to traverse across wood zones
```

You should feel different compliance when pressing **F** in different Y-positions (left/right).

---

## Advanced: Container Setup (Optional)

If you prefer using Docker/Singularity containers:

### Dockerfile

```dockerfile
FROM nvcr.io/nvidia/isaac-sim:4.0.0

# Install dependencies
RUN pip install torch torchvision opencv-python pillow scipy transformers accelerate

# Copy repository
COPY . /workspace/VLM_Inferred_HRI_safety
WORKDIR /workspace/VLM_Inferred_HRI_safety

# Install Isaac Lab extension
RUN pip install -e source/VLM_Inferred_HRI_safety

CMD ["/isaac-sim/isaaclab.sh", "-p", "scripts/run_hri_demo.py", "--enable_cameras"]
```

### Build and Run

```bash
# Build container
docker build -t vlm-hri-demo .

# Run with GPU support
docker run --gpus all -it \
  --env DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  vlm-hri-demo
```

---