"""
Material zone configuration for VLM-based impedance modulation.
Defines visual and physical properties of different wood sections.
"""

# Material zones along the log (X-axis)
# Log is 0.8m long, centered at X=0.45, so ranges from 0.05 to 0.85

MATERIAL_ZONES = {
    "soft_wood": {
        "x_range": (0.05, 0.35),  # First 30cm
        "color": (0.7, 0.5, 0.3),  # Light tan
        "friction": 0.3,  # Low resistance
        "recommended_stiffness": 300.0,
        "description": "Soft pine, easy cutting",
    },
    "knot": {
        "x_range": (0.35, 0.55),  # Middle 20cm
        "color": (0.4, 0.25, 0.15),  # Dark brown
        "friction": 0.9,  # High resistance
        "recommended_stiffness": 800.0,
        "description": "Hard knot, dense grain",
    },
    "cracked": {
        "x_range": (0.55, 0.85),  # Last 30cm
        "color": (0.6, 0.4, 0.2),  # Medium brown with visual cracks
        "friction": 0.2,  # Very low (weak structure)
        "recommended_stiffness": 200.0,
        "description": "Cracked section, will split easily",
    },
}


def get_material_at_position(x_pos):
    """
    Determine material zone based on X position.

    Args:
        x_pos: X coordinate in world frame

    Returns:
        Material zone name and properties
    """
    for zone_name, props in MATERIAL_ZONES.items():
        x_min, x_max = props["x_range"]
        if x_min <= x_pos <= x_max:
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
