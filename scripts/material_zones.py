"""
Material zone configuration for VLM-based impedance modulation.
Defines visual and physical properties of different wood sections.
"""

# Material zones along the log (Y-axis)
# Log is centered at Y=0.0
# Soft (-0.25), Knot (0.0), Crack (+0.25)

MATERIAL_ZONES = {
    "soft_wood": {
        "y_range": (-0.40, -0.15),  # Left section
        "color": (0.7, 0.5, 0.3),  # Light tan
        "friction": 0.3,  # Low resistance
        "recommended_stiffness": 500.0,
        "target_force": 10.0,  # N (Ideal cutting force)
        "description": "Soft pine, easy cutting",
    },
    "knot": {
        "y_range": (-0.15, 0.15),  # Middle section
        "color": (0.4, 0.25, 0.15),  # Dark brown
        "friction": 0.9,  # High resistance
        "recommended_stiffness": 600.0,
        "target_force": 40.0,  # N (Requires more force)
        "description": "Hard knot, dense grain",
    },
    "cracked": {
        "y_range": (0.15, 0.40),  # Right section
        "color": (0.6, 0.4, 0.2),  # Medium brown with visual cracks
        "friction": 0.2,  # Very low (weak structure)
        "recommended_stiffness": 400.0,
        "target_force": 5.0,  # N (Gentle handling required)
        "description": "Cracked section, will split easily",
    },
}


def get_material_at_position(y_pos):
    """
    Determine material zone based on Y position.

    Args:
        y_pos: Y coordinate in world frame

    Returns:
        Material zone name and properties
    """
    for zone_name, props in MATERIAL_ZONES.items():
        y_min, y_max = props["y_range"]
        if y_min <= y_pos <= y_max:
            return zone_name, props

    # Default to soft wood if outside ranges
    return "soft_wood", MATERIAL_ZONES["soft_wood"]


# VLM Prompts for material detection
VLM_MATERIAL_PROMPT = """You are analyzing wood surface for a sawing robot.

Classify the visible wood section as ONE of:
- SOFT: Light colored wood, visible grain, no knots
- HARD: Dark patch or knot visible, dense appearance  
- CRACK: Visible crack, split, or structural weakness

Respond with ONLY ONE WORD: SOFT, HARD, or CRACK"""


def parse_vlm_material_response(vlm_text):
    """
    Parse VLM response into material classification.

    Args:
        vlm_text: Raw VLM response string

    Returns:
        Material classification: 'soft_wood', 'knot', or 'cracked'
    """
    text_upper = vlm_text.upper()

    if "HARD" in text_upper or "KNOT" in text_upper:
        return "knot"
    elif "CRACK" in text_upper or "SPLIT" in text_upper:
        return "cracked"
    else:  # Default to soft
        return "soft_wood"


def material_to_stiffness(material_zone):
    """
    Map material zone to OSC stiffness value.

    Args:
        material_zone: One of 'soft_wood', 'knot', 'cracked'

    Returns:
        Stiffness value for motion_stiffness_task
    """
    if material_zone in MATERIAL_ZONES:
        return MATERIAL_ZONES[material_zone]["recommended_stiffness"]
    return 400.0  # Default baseline
