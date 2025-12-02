"""
Vehicle Classifier Node

Classifies the user's request into one of the supported vehicle types:
- drone (multicopter UAV)
- fixed_wing (airplane)
- helicopter (rotorcraft)
- rocket (launch vehicle)
- satellite (spacecraft)
- glider (sailplane)
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# FIXED: Import from graph.state
sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.state import DesignPhase, DesignState, VehicleType

# ... rest of the file remains the same, just update the import


# Keywords associated with each vehicle type
VEHICLE_KEYWORDS = {
    VehicleType.DRONE: [
        "drone",
        "quadcopter",
        "multicopter",
        "uav",
        "quadrotor",
        "hexacopter",
        "octocopter",
        "fpv",
        "aerial photography",
        "multirotor",
        "unmanned aerial",
        "rc quad",
        "racing drone",
        "hover",
        "vtol drone",
    ],
    VehicleType.FIXED_WING: [
        "airplane",
        "aircraft",
        "plane",
        "fixed wing",
        "fixed-wing",
        "cessna",
        "jet",
        "airliner",
        "transport",
        "fighter",
        "passenger",
        "cargo plane",
        "propeller plane",
        "wing",
        "takeoff",
        "landing",
        "runway",
        "cruise flight",
    ],
    VehicleType.HELICOPTER: [
        "helicopter",
        "rotorcraft",
        "chopper",
        "helo",
        "heli",
        "rotor",
        "autogyro",
        "gyrocopter",
        "vtol helicopter",
        "tail rotor",
        "main rotor",
        "hovering vehicle",
    ],
    VehicleType.ROCKET: [
        "rocket",
        "launch vehicle",
        "missile",
        "sounding rocket",
        "model rocket",
        "space launch",
        "booster",
        "staging",
        "delta-v",
        "thrust",
        "propellant",
        "altitude record",
        "reach space",
        "suborbital",
        "motor",
    ],
    VehicleType.SATELLITE: [
        "satellite",
        "spacecraft",
        "space station",
        "cubesat",
        "orbit",
        "leo",
        "geo",
        "meo",
        "solar panel",
        "payload bay",
        "earth observation",
        "communication satellite",
        "space",
    ],
    VehicleType.GLIDER: [
        "glider",
        "sailplane",
        "soaring",
        "thermal",
        "unpowered",
        "motorglider",
        "hang glider",
        "paraglider",
        "glide ratio",
        "sink rate",
        "no engine",
        "silent flight",
    ],
}


# Mutual exclusion rules - if these keywords appear, reduce score for type
EXCLUSION_KEYWORDS = {
    VehicleType.DRONE: ["airplane", "helicopter rotor", "rocket motor", "orbit"],
    VehicleType.FIXED_WING: ["hover in place", "quadcopter", "rotor blade", "rocket"],
    VehicleType.HELICOPTER: ["fixed wing", "quadcopter", "rocket", "orbit"],
    VehicleType.ROCKET: ["wing", "rotor", "propeller", "hover"],
    VehicleType.SATELLITE: ["wing", "rotor", "propeller", "atmosphere flight"],
    VehicleType.GLIDER: ["powered", "engine", "motor", "fuel", "propeller"],
}


def calculate_keyword_scores(text: str) -> Dict[VehicleType, float]:
    """
    Calculate keyword-based scores for each vehicle type.

    Args:
        text: User input text (lowercase)

    Returns:
        Dict mapping vehicle type to score
    """
    scores = {vtype: 0.0 for vtype in VehicleType if vtype != VehicleType.UNKNOWN}

    # Count keyword matches
    for vtype, keywords in VEHICLE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                # Longer keywords get more weight
                weight = 1.0 + len(keyword) / 20
                scores[vtype] += weight

    # Apply exclusion penalties
    for vtype, exclusions in EXCLUSION_KEYWORDS.items():
        for keyword in exclusions:
            if keyword in text:
                scores[vtype] *= 0.5  # Reduce score if exclusion keyword found

    return scores


def classify_by_requirements(text: str) -> Optional[VehicleType]:
    """
    Classify based on specific requirement patterns.

    Args:
        text: User input text

    Returns:
        Vehicle type if pattern matches, None otherwise
    """
    # Patterns that strongly indicate vehicle type
    patterns = [
        # Drone patterns
        (r"(\d+)\s*min(ute)?s?\s*(of\s+)?flight\s*time.*hover", VehicleType.DRONE),
        (r"aerial\s+photo|fpv\s+racing", VehicleType.DRONE),
        # Fixed wing patterns
        (r"(\d+)\s*(passenger|seat|person)", VehicleType.FIXED_WING),
        (r"(\d+)\s*km\s*range.*cruise", VehicleType.FIXED_WING),
        (r"takeoff.*runway|runway.*land", VehicleType.FIXED_WING),
        # Helicopter patterns
        (r"hover.*transport|transport.*hover", VehicleType.HELICOPTER),
        (r"autorotation|disk\s*loading", VehicleType.HELICOPTER),
        # Rocket patterns
        (r"reach\s*(\d+)\s*(km|m|ft).*altitude", VehicleType.ROCKET),
        (r"delta-?v|specific\s*impulse|staging", VehicleType.ROCKET),
        (r"launch.*payload.*space", VehicleType.ROCKET),
        # Satellite patterns
        (r"(\d+)\s*km\s*orbit", VehicleType.SATELLITE),
        (r"(leo|geo|meo)\s*(orbit|satellite)?", VehicleType.SATELLITE),
        (r"earth\s*observation|communication\s*satellite", VehicleType.SATELLITE),
        # Glider patterns
        (r"soaring|thermal\s*climb", VehicleType.GLIDER),
        (r"glide\s*ratio|l/d\s*ratio", VehicleType.GLIDER),
        (r"unpowered|no\s*engine", VehicleType.GLIDER),
    ]

    for pattern, vtype in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return vtype

    return None


def classify_vehicle(text: str) -> Tuple[VehicleType, float, str]:
    """
    Classify the vehicle type from user input.

    Uses a combination of:
    1. Keyword matching
    2. Pattern-based classification
    3. Heuristic rules

    Args:
        text: User input text

    Returns:
        (vehicle_type, confidence, reasoning)
    """
    text_lower = text.lower()

    # Try pattern-based classification first (high confidence)
    pattern_match = classify_by_requirements(text_lower)
    if pattern_match:
        return pattern_match, 0.9, f"Strong pattern match for {pattern_match.value}"

    # Calculate keyword scores
    scores = calculate_keyword_scores(text_lower)

    # Find best match
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Calculate confidence based on score margin
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2 and sorted_scores[0] > 0:
        margin = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
        confidence = min(0.5 + margin * 0.4, 0.95)
    else:
        confidence = 0.3 if best_score > 0 else 0.1

    # Build reasoning
    if best_score > 0:
        matched_keywords = [
            kw for kw in VEHICLE_KEYWORDS[best_type] if kw in text_lower
        ]
        reasoning = f"Matched keywords: {', '.join(matched_keywords[:5])}"
    else:
        reasoning = "No strong keyword matches found"
        best_type = VehicleType.UNKNOWN
        confidence = 0.0

    return best_type, confidence, reasoning


def classifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for vehicle classification.

    Args:
        state: Current graph state

    Returns:
        State updates
    """
    user_input = state["user_input"]

    # Perform classification
    vehicle_type, confidence, reasoning = classify_vehicle(user_input)

    # Determine next phase
    if vehicle_type == VehicleType.UNKNOWN:
        next_phase = DesignPhase.ERROR
        error_msg = "Could not determine vehicle type from input. Please specify if you want a drone, airplane, helicopter, rocket, satellite, or glider."
    else:
        next_phase = DesignPhase.PARSING
        error_msg = None

    return {
        "vehicle_type": vehicle_type,
        "classification_confidence": confidence,
        "classification_reasoning": reasoning,
        "current_phase": next_phase,
        "error_message": error_msg,
        "messages": [
            {
                "role": "system",
                "content": f"Classified as {vehicle_type.value} with {confidence:.0%} confidence. {reasoning}",
            }
        ],
    }


if __name__ == "__main__":
    # Test classification
    test_inputs = [
        "Design a drone that can carry 2kg payload for 30 minutes",
        "I need a small airplane for 2 passengers with 500km range",
        "Design a helicopter for medical evacuation",
        "Build a model rocket to reach 1km altitude with 500g payload",
        "LEO satellite for Earth observation, 400km orbit, 5 year mission",
        "Competition glider with 50:1 glide ratio for cross-country",
        "Flying vehicle for transportation",  # Ambiguous
    ]

    print("=== Vehicle Classification Tests ===\n")

    for input_text in test_inputs:
        vtype, conf, reason = classify_vehicle(input_text)
        print(f"Input: {input_text[:60]}...")
        print(f"  Type: {vtype.value}")
        print(f"  Confidence: {conf:.0%}")
        print(f"  Reason: {reason}")
        print()
