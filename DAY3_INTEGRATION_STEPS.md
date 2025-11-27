# Day 3: VLM Integration - Step-by-Step Guide

## Morning Session (2 hours)

### Step 1: Add VLM Mode to run_hri_demo.py (15 min)

Add these globals after line 117:
```python
# VLM MODE (Day 3)
g_vlm_mode = False  # Toggle with 'V' key
g_detected_material = "soft_wood"  # Updated by VLM worker
```

Add keyboard handler in `_on_keyboard_event` function:
```python
elif event.input == carb.input.KeyboardInput.V:
    global g_vlm_mode
    g_vlm_mode = not g_vlm_mode
    status = "ENABLED" if g_vlm_mode else "DISABLED"
    print(f"[VLM] Mode {status}")
```

### Step 2: Replace Oracle Lookup with VLM (10 min)

In the Oracle block (around line 778), change:
```python
# OLD (Oracle - uses position)
saw_y = g_target_y_position
zone_name, props = get_material_at_position(saw_y)

# NEW (VLM - uses detected material)
if g_vlm_mode:
    zone_name = g_detected_material
    props = MATERIAL_ZONES[zone_name]
else:
    # Oracle fallback
    saw_y = g_target_y_position
    zone_name, props = get_material_at_position(saw_y)
```

### Step 3: Test with Mock Data (30 min)

Run simulation and toggle VLM mode ('V' key). It should use "soft_wood" by default.

Manually change `g_detected_material` in the code to test:
- `g_detected_material = "knot"` → Should apply 600 N/m
- `g_detected_material = "cracked"` → Should apply 400 N/m

### Step 4: Add Shared Buffer Reader (45 min)

At the top of `run_hri_demo.py`, import:
```python
from shared_buffer import SharedImageBuffer
```

In the main function, after camera setup:
```python
# VLM Shared Buffer (optional - for reading VLM results)
vlm_result_buffer = None
if args_cli.enable_cameras:
    try:
        vlm_result_buffer = SharedImageBuffer(
            name="vlm_material_results",
            buffer_size=1,
            height=1,
            width=1,
            channels=1,
            create=False
        )
        print("[VLM] Connected to VLM result buffer")
    except:
        print("[VLM] No VLM worker detected (OK)")
```

In the main loop, read VLM results:
```python
if g_vlm_mode and vlm_result_buffer and frame_count % 10 == 0:
    result = vlm_result_buffer.read_latest()
    if result:
        # Parse material from buffer
        # g_detected_material = parse_result(result)
        pass
```

---

## Afternoon Session (2 hours)

### Step 5: Connect Real VLM Worker (1 hour)

Terminal 1 (Isaac Sim):
```bash
cd /workspace
./isaaclab/isaaclab.sh -p VLM_Inferred_HRI_safety/scripts/run_hri_demo.py --enable_cameras
```

Terminal 2 (VLM Worker):
```bash
cd /workspace/VLM_Inferred_HRI_safety/scripts
/isaac-sim/miniforge3/envs/lerobot/bin/python vlm_material_detector.py
```

### Step 6: Test All Zones (30 min)

1. Enable VLM mode ('V' key)
2. Move to each zone (LEFT/RIGHT arrows)
3. Verify VLM detects correct material
4. Check stiffness changes in HUD

### Step 7: Compare Oracle vs VLM (30 min)

Record metrics:
- Oracle accuracy: 100% (ground truth)
- VLM accuracy: Count correct classifications
- VLM latency: Check inference time in VLM worker output

---

## Evening Session (1 hour)

### Step 8: Record Demo Video

1. **Baseline** (No Oracle, No VLM): Fixed 500 N/m
2. **Oracle Mode**: Perfect adaptation
3. **VLM Mode**: Real-time vision-based adaptation
4. **Penalty Mode**: Show failure case

### Step 9: Update Documentation

Update `task.md`:
```markdown
## Day 3: VLM Material Detection ✅ COMPLETE
- [x] Add VLM mode toggle
- [x] Connect shared buffer
- [x] Test VLM classifications
- [x] Compare Oracle vs VLM
- [x] Record demo video
```

---

## Quick Reference

### Keyboard Controls
- `O` - Oracle Mode (ground truth)
- `P` - Penalty Mode (wrong stiffness)
- `V` - VLM Mode (vision-based)
- `LEFT/RIGHT` - Move across zones
- `UP/DOWN` - Adjust height

### Expected Stiffness Values
- Cracked: 400 N/m
- Soft: 500 N/m
- Knot: 600 N/m

### Troubleshooting

**VLM worker can't connect to buffer:**
- Make sure Isaac Sim is running with `--enable_cameras`
- Check buffer name matches: "hri_camera_buffer"

**VLM detections are wrong:**
- Check camera is pointing at log
- Verify lighting in scene
- Try adjusting VLM prompt

**Stiffness not changing:**
- Verify `g_vlm_mode = True`
- Check `g_detected_material` value in debugger
- Ensure Oracle block is using VLM lookup

---

## Success Criteria

✅ VLM mode toggles on/off
✅ VLM worker runs at 3 FPS
✅ Material classifications are reasonable (>70% accuracy)
✅ Stiffness changes match detected material
✅ Demo video shows all 4 modes

**Estimated Total Time: 5 hours**
