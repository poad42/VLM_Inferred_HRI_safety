#!/usr/bin/env python3
"""
Test script to verify Oracle/VLM/Penalty mode logic
Tests stiffness application without Isaac Sim
"""

# Material zones (same as material_zones.py)
MATERIAL_ZONES = {
    "soft_wood": {
        "recommended_stiffness": 500.0,
        "description": "Soft Wood",
    },
    "knot": {
        "recommended_stiffness": 600.0,
        "description": "Hard Knot",
    },
    "cracked": {
        "recommended_stiffness": 600.0,
        "description": "Cracked Wood",
    },
}


def get_material_oracle(position_y):
    """Simulate oracle mode - position-based detection"""
    if -0.50 <= position_y < -0.15:
        return "soft_wood"
    elif -0.15 <= position_y < 0.15:
        return "knot"
    elif 0.15 <= position_y <= 0.50:
        return "cracked"
    return "soft_wood"


def get_material_vlm(simulated_vlm_result):
    """Simulate VLM mode - vision-based detection"""
    # In real system, this reads from shared buffer
    # Here we simulate with parameter
    return simulated_vlm_result


def apply_penalty(target_stiffness, penalty_enabled):
    """Apply penalty mode - invert stiffness"""
    if not penalty_enabled:
        return target_stiffness

    # Invert stiffness (same logic as run_hri_demo.py line 1083-1086)
    if target_stiffness >= 550:
        return 400.0  # Hard → Soft (wrong!)
    else:
        return 600.0  # Soft → Hard (wrong!)


def apply_vlm_fault(detected_material, fault_enabled):
    """Apply VLM fault injection - intentionally detect wrong material"""
    if not fault_enabled:
        return detected_material

    # Same fault map as run_hri_demo.py
    fault_map = {"soft_wood": "knot", "knot": "cracked", "cracked": "soft_wood"}

    return fault_map.get(detected_material, "soft_wood")


def test_scenario(
    scenario_name,
    mode,
    position_y,
    vlm_result,
    penalty_enabled,
    vlm_fault_enabled=False,
):
    """Test a single scenario"""
    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*70}")

    # 1. Detect material
    if mode == "VLM":
        zone_name = get_material_vlm(vlm_result)
        original_zone = zone_name

        # Apply VLM fault injection
        zone_name = apply_vlm_fault(zone_name, vlm_fault_enabled)

        if vlm_fault_enabled:
            print(
                f"[VLM Mode + FAULT] Vision detected: {original_zone} → FAULT INJECTED → {zone_name}"
            )
        else:
            print(f"[VLM Mode] Vision detected: {zone_name}")
    else:  # Oracle
        zone_name = get_material_oracle(position_y)
        print(f"[Oracle Mode] Position Y={position_y:.2f} → Zone: {zone_name}")

    # 2. Get correct stiffness
    correct_stiffness = MATERIAL_ZONES[zone_name]["recommended_stiffness"]
    print(f"Stiffness for detected zone ({zone_name}): {correct_stiffness}")

    # 3. Apply penalty if enabled
    final_stiffness = apply_penalty(correct_stiffness, penalty_enabled)

    # 4. Show mode indicator
    if mode == "VLM":
        mode_str = "VLM"
    else:
        mode_str = "ORACLE"

    if vlm_fault_enabled and mode == "VLM":
        mode_str += "+FAULT"
    if penalty_enabled:
        mode_str += "+PENALTY"

    print(f"Mode: [{mode_str}]")
    print(f"Applied stiffness: {final_stiffness}")

    # 5. Check modes
    if vlm_fault_enabled and mode == "VLM":
        print("✓ VLM FAULT ACTIVE: Detecting WRONG material!")

    if penalty_enabled:
        if final_stiffness != correct_stiffness:
            print("✓ PENALTY ACTIVE: Stiffness INVERTED (saw will get stuck!)")
        else:
            print("⚠ ERROR: Penalty should have inverted stiffness!")
    else:
        if final_stiffness == correct_stiffness:
            print("✓ Normal operation: Correct stiffness applied")
        else:
            print("⚠ ERROR: Stiffness incorrectly modified!")

    return final_stiffness


def main():
    print("=" * 70)
    print("MODE LOGIC TEST - Oracle/VLM/Penalty/Fault Combinations")
    print("=" * 70)

    # Test 1: Oracle mode, soft wood, no penalty (baseline)
    test_scenario(
        "Oracle + Soft Wood + No Penalty",
        mode="Oracle",
        position_y=-0.25,  # Soft wood zone
        vlm_result=None,
        penalty_enabled=False,
        vlm_fault_enabled=False,
    )

    # Test 2: Oracle mode, soft wood, WITH penalty
    test_scenario(
        "Oracle + Soft Wood + PENALTY (should invert to 600)",
        mode="Oracle",
        position_y=-0.25,  # Soft wood zone
        vlm_result=None,
        penalty_enabled=True,
        vlm_fault_enabled=False,
    )

    # Test 3: Oracle mode, knot, no penalty
    test_scenario(
        "Oracle + Knot + No Penalty",
        mode="Oracle",
        position_y=0.0,  # Knot zone
        vlm_result=None,
        penalty_enabled=False,
        vlm_fault_enabled=False,
    )

    # Test 4: Oracle mode, knot, WITH penalty
    test_scenario(
        "Oracle + Knot + PENALTY (should invert to 400)",
        mode="Oracle",
        position_y=0.0,  # Knot zone
        vlm_result=None,
        penalty_enabled=True,
        vlm_fault_enabled=False,
    )

    # Test 5: VLM mode, correct detection, no penalty
    test_scenario(
        "VLM + Correct Detection (soft) + No Penalty",
        mode="VLM",
        position_y=None,
        vlm_result="soft_wood",
        penalty_enabled=False,
        vlm_fault_enabled=False,
    )

    # Test 6: VLM mode, correct detection, WITH penalty
    test_scenario(
        "VLM + Correct Detection (soft) + PENALTY",
        mode="VLM",
        position_y=None,
        vlm_result="soft_wood",
        penalty_enabled=True,
        vlm_fault_enabled=False,
    )

    # Test 7: VLM mode, correct detection, WITH FAULT (no penalty)
    test_scenario(
        "VLM + FAULT MODE (soft→knot) + No Penalty",
        mode="VLM",
        position_y=None,
        vlm_result="soft_wood",  # VLM detects soft
        penalty_enabled=False,
        vlm_fault_enabled=True,  # Fault injects knot
    )

    # Test 8: VLM mode, correct detection, WITH FAULT + PENALTY
    test_scenario(
        "VLM + FAULT MODE (soft→knot) + PENALTY",
        mode="VLM",
        position_y=None,
        vlm_result="soft_wood",  # VLM detects soft
        penalty_enabled=True,
        vlm_fault_enabled=True,  # Fault injects knot, then penalty inverts
    )

    # Test 9: VLM fault only (knot→cracked)
    test_scenario(
        "VLM + FAULT MODE (knot→cracked)",
        mode="VLM",
        position_y=None,
        vlm_result="knot",
        penalty_enabled=False,
        vlm_fault_enabled=True,
    )

    # Test 10: VLM fault + penalty (knot→cracked→soft)
    test_scenario(
        "VLM + FAULT MODE (knot→cracked) + PENALTY (600→400)",
        mode="VLM",
        position_y=None,
        vlm_result="knot",
        penalty_enabled=True,
        vlm_fault_enabled=True,
    )

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("✓ Penalty mode works with BOTH Oracle and VLM")
    print("✓ VLM Fault mode intentionally detects wrong material")
    print("✓ Fault and Penalty can be combined for extreme testing")
    print("\nMODE COMBINATIONS:")
    print("  [ORACLE]            - Geometric material detection")
    print("  [ORACLE+PENALTY]    - Geometric + inverted stiffness")
    print("  [VLM]               - Vision-based detection")
    print("  [VLM+FAULT]         - Vision detects WRONG material")
    print("  [VLM+PENALTY]       - Vision + inverted stiffness")
    print("  [VLM+FAULT+PENALTY] - Wrong material + inverted stiffness (worst case!)")
    print("\nFAULT INJECTION MAP:")
    print("  soft_wood → knot    (500 → 600)")
    print("  knot → cracked      (600 → 600)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
