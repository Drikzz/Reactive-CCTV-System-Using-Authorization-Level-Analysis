"""
Authorization Level Configuration for Main Integrated System

Defines access levels for recognized individuals and their monitoring requirements:
- AUTHORIZED: Full access, face tracking only (no behavior monitoring)
- PARTIAL: Limited access, face + behavior tracking (alerts suppressed if authorized present)
- UNAUTHORIZED: No access, immediate alerts + full logging

Usage:
    from authorization_config_main import get_authorization_level, AUTHORIZED, PARTIAL, UNAUTHORIZED
    
    level = get_authorization_level("Aldrikz")
    if level == AUTHORIZED:
        # Track face only, no behavior monitoring
    elif level == PARTIAL:
        # Track face + behavior, conditional alerts
    else:
        # Track everything, immediate alerts
"""

# ==================== AUTHORIZATION LEVELS ====================

# Access level constants
AUTHORIZED = "Authorized"      # Full access - Face tracking only - Green
PARTIAL = "Partial"            # Limited access - Face + Behavior tracking - Yellow
UNAUTHORIZED = "Unauthorized"  # No access - Immediate alerts - Red

# All possible levels
ALL_LEVELS = [AUTHORIZED, PARTIAL, UNAUTHORIZED]

# ==================== PERSON TO LEVEL MAPPING ====================

# Map each person's name to their authorization level
# Names must match folder names in datasets/faces/
AUTHORIZATION_MAP = {
    # Authorized personnel (full access, no behavior monitoring)
    "Aldrikz": AUTHORIZED,
    "Art": AUTHORIZED,
    
    # Partial access personnel (monitored, conditional alerts)
    "Jude": PARTIAL,
    "Myke": PARTIAL,
    "Speed": PARTIAL,
    
    # Add more people here:
    # "John_Doe": AUTHORIZED,
    # "Jane_Smith": PARTIAL,
}

# ==================== DEFAULT SETTINGS ====================

# Default level for recognized people not in the map
DEFAULT_LEVEL = PARTIAL

# Treat Unknown faces as Unauthorized
UNKNOWN_LEVEL = UNAUTHORIZED

# ==================== BEHAVIOR MONITORING SETTINGS ====================

# Behavior classes that trigger alerts (suspicious behaviors)
SUSPICIOUS_BEHAVIORS = [
    "opening-cabinet",
    "holding-object",
    "using-computer", 
    "opening-door"
]

# Safe/neutral behaviors (no alerts)
SAFE_BEHAVIORS = [
    "Neutral",
]

# Minimum confidence threshold for behavior detection
BEHAVIOR_CONFIDENCE_THRESHOLD = 0.7

# ==================== DISPLAY SETTINGS ====================

# Colors for each level (BGR format for OpenCV)
LEVEL_COLORS = {
    AUTHORIZED: (0, 255, 0),      # Green
    PARTIAL: (0, 255, 255),       # Yellow
    UNAUTHORIZED: (0, 0, 255),    # Red
}

# Box thickness for each level
LEVEL_THICKNESS = {
    AUTHORIZED: 3,    # Thicker box for authorized
    PARTIAL: 2,       # Medium thickness
    UNAUTHORIZED: 2,  # Medium thickness
}

# Label prefix for each level
LEVEL_PREFIX = {
    AUTHORIZED: "✓",     # Checkmark
    PARTIAL: "◐",        # Half-filled circle
    UNAUTHORIZED: "✗",   # Cross
}

# ==================== LOGGING SETTINGS ====================

# Base directories
LOGS_BASE = "logs/main_system"
ANNOTATED_BASE = "annotated_frames/main_system"

# Separate log directories for each level
LOG_STRUCTURE = {
    AUTHORIZED: f"{LOGS_BASE}/authorized",
    PARTIAL: f"{LOGS_BASE}/partial",
    UNAUTHORIZED: f"{LOGS_BASE}/unauthorized",
}

# Separate annotated frame directories
ANNOTATED_STRUCTURE = {
    AUTHORIZED: f"{ANNOTATED_BASE}/authorized",
    PARTIAL: f"{ANNOTATED_BASE}/partial",
    UNAUTHORIZED: f"{ANNOTATED_BASE}/unauthorized",
}

# Behavior-specific logging
BEHAVIOR_LOGS = {
    "suspicious": f"{LOGS_BASE}/suspicious_behavior",
    "alerts": f"{LOGS_BASE}/alerts",
}

# ==================== ALERT SETTINGS ====================

# Alert trigger conditions
ALERT_CONDITIONS = {
    AUTHORIZED: {
        "suspicious_behavior": False,  # Never alert for authorized
        "presence": False,
    },
    PARTIAL: {
        "suspicious_behavior": True,   # Alert if suspicious + no authorized present
        "presence": False,
    },
    UNAUTHORIZED: {
        "suspicious_behavior": True,   # Always alert on suspicious behavior
        "presence": True,              # Alert on presence alone
    }
}

# ==================== HELPER FUNCTIONS ====================

def get_authorization_level(name):
    """
    Get authorization level for a person by name
    
    Args:
        name: Person's name (must match folder name in datasets/faces/)
    
    Returns:
        Authorization level (AUTHORIZED, PARTIAL, or UNAUTHORIZED)
    """
    if name == "Unknown":
        return UNKNOWN_LEVEL
    
    # Check if person is in authorization map
    level = AUTHORIZATION_MAP.get(name)
    
    if level is not None:
        return level
    
    # Return default level for recognized but unmapped people
    return DEFAULT_LEVEL


def should_monitor_behavior(level):
    """
    Check if behavior monitoring is required for this level
    
    Args:
        level: Authorization level
    
    Returns:
        bool: True if behavior should be monitored
    """
    return level in [PARTIAL, UNAUTHORIZED]


def is_suspicious_behavior(behavior_class):
    """
    Check if a behavior class is considered suspicious
    
    Args:
        behavior_class: Predicted behavior class name
    
    Returns:
        bool: True if behavior is suspicious
    """
    return behavior_class in SUSPICIOUS_BEHAVIORS


def should_trigger_alert(person_level, behavior_class, authorized_present):
    """
    Determine if an alert should be triggered based on conditions
    
    Args:
        person_level: Authorization level of the person
        behavior_class: Detected behavior class
        authorized_present: Whether an authorized person is in the same frame
    
    Returns:
        bool: True if alert should be triggered
    """
    # Unauthorized always triggers alert on suspicious behavior
    if person_level == UNAUTHORIZED and is_suspicious_behavior(behavior_class):
        return True
    
    # Partial triggers alert only if no authorized person present
    if person_level == PARTIAL and is_suspicious_behavior(behavior_class):
        return not authorized_present
    
    # Authorized never triggers alerts
    return False


def get_level_color(level):
    """Get display color for authorization level"""
    return LEVEL_COLORS.get(level, LEVEL_COLORS[UNAUTHORIZED])


def get_level_thickness(level):
    """Get box thickness for authorization level"""
    return LEVEL_THICKNESS.get(level, 2)


def get_level_prefix(level):
    """Get label prefix for authorization level"""
    return LEVEL_PREFIX.get(level, "")


def get_log_directory(level):
    """Get log directory for authorization level"""
    return LOG_STRUCTURE.get(level, f"{LOGS_BASE}/unknown")


def get_annotated_directory(level):
    """Get annotated frames directory for authorization level"""
    return ANNOTATED_STRUCTURE.get(level, f"{ANNOTATED_BASE}/unknown")


def format_display_name(name, level, behavior=None):
    """
    Format display name with authorization level and optional behavior
    
    Args:
        name: Person's name
        level: Authorization level
        behavior: Optional behavior class
    
    Returns:
        Formatted string like "✓ Aldrikz (Authorized)" or "◐ Jude (Partial) - Fighting"
    """
    prefix = get_level_prefix(level)
    base = f"{prefix} {name} ({level})"
    
    if behavior and should_monitor_behavior(level):
        base += f" - {behavior}"
    
    return base


def is_authorized(name):
    """Check if person has full authorization"""
    return get_authorization_level(name) == AUTHORIZED


def has_partial_access(name):
    """Check if person has partial access"""
    return get_authorization_level(name) == PARTIAL


def is_unauthorized(name):
    """Check if person is unauthorized"""
    return get_authorization_level(name) == UNAUTHORIZED


def get_all_people_by_level():
    """
    Get all people grouped by authorization level
    
    Returns:
        Dictionary mapping level -> list of names
    """
    people_by_level = {level: [] for level in ALL_LEVELS}
    
    for name, level in AUTHORIZATION_MAP.items():
        people_by_level[level].append(name)
    
    return people_by_level


def print_authorization_summary():
    """Print summary of authorization mappings"""
    print("\n" + "="*70)
    print("MAIN SYSTEM - AUTHORIZATION CONFIGURATION")
    print("="*70)
    
    people_by_level = get_all_people_by_level()
    
    for level in ALL_LEVELS:
        people = people_by_level[level]
        count = len(people)
        
        monitoring = "Face only" if level == AUTHORIZED else "Face + Behavior"
        
        print(f"\n{get_level_prefix(level)} {level.upper()}: {count} person(s) - {monitoring}")
        if people:
            for person in people:
                print(f"   - {person}")
        else:
            print(f"   (none)")
    
    print(f"\nDefault level for unmapped people: {DEFAULT_LEVEL}")
    print(f"Unknown faces treated as: {UNKNOWN_LEVEL}")
    print(f"\nSuspicious behaviors: {', '.join(SUSPICIOUS_BEHAVIORS)}")
    print(f"Safe behaviors: {', '.join(SAFE_BEHAVIORS)}")
    print("="*70 + "\n")


def print_monitoring_rules():
    """Print behavior monitoring and alert rules"""
    print("\n" + "="*70)
    print("MONITORING & ALERT RULES")
    print("="*70)
    
    print("\n✓ AUTHORIZED Persons:")
    print("  - Face tracking only (no behavior monitoring)")
    print("  - Never triggers alerts")
    print("  - Presence suppresses alerts for PARTIAL persons in same frame")
    
    print("\n◐ PARTIAL Persons:")
    print("  - Face + Behavior tracking")
    print("  - Alerts triggered ONLY if:")
    print("    → Suspicious behavior detected AND")
    print("    → No AUTHORIZED person in same frame")
    print("  - Suspicious behavior still logged even if alert suppressed")
    
    print("\n✗ UNAUTHORIZED Persons:")
    print("  - Face + Behavior tracking")
    print("  - Immediate alert on suspicious behavior")
    print("  - Always logs full frame + cropped frame")
    print("  - Not affected by presence of AUTHORIZED persons")
    
    print("="*70 + "\n")


# ==================== VALIDATION ====================

def validate_authorization_map():
    """Validate that all mapped people have face data"""
    import os
    
    faces_dir = "datasets/faces"
    if not os.path.exists(faces_dir):
        print(f"[WARN] Faces directory not found: {faces_dir}")
        return
    
    existing_people = [d for d in os.listdir(faces_dir) 
                      if os.path.isdir(os.path.join(faces_dir, d))]
    mapped_people = list(AUTHORIZATION_MAP.keys())
    
    # Check for mapped people without face data
    missing_faces = [p for p in mapped_people if p not in existing_people]
    if missing_faces:
        print(f"[WARN] These people are mapped but have no face data:")
        for person in missing_faces:
            print(f"   - {person} ({AUTHORIZATION_MAP[person]})")
    
    # Check for people with face data but not mapped
    unmapped_people = [p for p in existing_people if p not in mapped_people]
    if unmapped_people:
        print(f"[INFO] These people have face data but use default level ({DEFAULT_LEVEL}):")
        for person in unmapped_people:
            print(f"   - {person}")


if __name__ == "__main__":
    # Test configuration
    print_authorization_summary()
    print_monitoring_rules()
    validate_authorization_map()
    
    # Test examples
    print("\nTesting authorization levels:")
    test_cases = [
        ("Aldrikz", "Assault"),
        ("Jude", "Fighting"),
        ("Unknown", "Robbery"),
    ]
    
    print(f"\n{'Person':<15} {'Behavior':<15} {'Alert (No Auth)':<20} {'Alert (Auth Present)':<20}")
    print("-" * 70)
    
    for name, behavior in test_cases:
        level = get_authorization_level(name)
        alert_no_auth = should_trigger_alert(level, behavior, authorized_present=False)
        alert_with_auth = should_trigger_alert(level, behavior, authorized_present=True)
        
        print(f"{name:<15} {behavior:<15} {str(alert_no_auth):<20} {str(alert_with_auth):<20}")
