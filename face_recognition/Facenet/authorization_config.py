"""
Authorization Level Configuration for FaceNet Recognition System

Defines access levels for recognized individuals:
- AUTHORIZED: Full access to all areas/systems
- PARTIAL: Limited access to certain areas/systems
- UNAUTHORIZED: No access (treated same as Unknown)

Usage:
    from authorization_config import get_authorization_level, AUTHORIZED, PARTIAL, UNAUTHORIZED
    
    level = get_authorization_level("Aldrikz")
    if level == AUTHORIZED:
        # Grant full access
    elif level == PARTIAL:
        # Grant limited access
    else:
        # Deny access
"""

# ==================== AUTHORIZATION LEVELS ====================

# Access level constants
AUTHORIZED = "Authorized"      # Full access - Green
PARTIAL = "Partial"            # Limited access - Yellow
UNAUTHORIZED = "Unauthorized"  # No access - Red (same as Unknown)

# All possible levels
ALL_LEVELS = [AUTHORIZED, PARTIAL, UNAUTHORIZED]

# ==================== PERSON TO LEVEL MAPPING ====================

# Map each person's name to their authorization level
# Names must match folder names in datasets/faces/
AUTHORIZATION_MAP = {
    # Authorized personnel (full access)
    "Aldrikz": AUTHORIZED,
    "Art": AUTHORIZED,
    
    # Partial access personnel (limited access)
    "Jude": PARTIAL,
    "Myke": PARTIAL,
    "Speed": PARTIAL,
    
    # You can add more people here:
    # "John_Doe": AUTHORIZED,
    # "Jane_Smith": PARTIAL,
}

# ==================== DEFAULT SETTINGS ====================

# Default level for recognized people not in the map
DEFAULT_LEVEL = PARTIAL

# Treat Unknown faces as Unauthorized
UNKNOWN_LEVEL = UNAUTHORIZED

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

# Separate log directories for each level
LOG_STRUCTURE = {
    AUTHORIZED: "logs/FaceNet/authorized",
    PARTIAL: "logs/FaceNet/partial",
    UNAUTHORIZED: "logs/FaceNet/unauthorized",
}

# Separate annotated frame directories
ANNOTATED_STRUCTURE = {
    AUTHORIZED: "annotated_frames/FaceNet/authorized",
    PARTIAL: "annotated_frames/FaceNet/partial",
    UNAUTHORIZED: "annotated_frames/FaceNet/unauthorized",
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
    return LOG_STRUCTURE.get(level, "logs/FaceNet/unknown")


def get_annotated_directory(level):
    """Get annotated frames directory for authorization level"""
    return ANNOTATED_STRUCTURE.get(level, "annotated_frames/FaceNet/unknown")


def format_display_name(name, level):
    """
    Format display name with authorization level indicator
    
    Args:
        name: Person's name
        level: Authorization level
    
    Returns:
        Formatted string like "✓ Aldrikz (Authorized)"
    """
    prefix = get_level_prefix(level)
    return f"{prefix} {name} ({level})"


def is_authorized(name):
    """Check if person has full authorization"""
    return get_authorization_level(name) == AUTHORIZED


def has_partial_access(name):
    """Check if person has at least partial access"""
    level = get_authorization_level(name)
    return level in [AUTHORIZED, PARTIAL]


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
    print("\n" + "="*60)
    print("AUTHORIZATION CONFIGURATION")
    print("="*60)
    
    people_by_level = get_all_people_by_level()
    
    for level in ALL_LEVELS:
        people = people_by_level[level]
        count = len(people)
        color_bgr = LEVEL_COLORS[level]
        
        print(f"\n{get_level_prefix(level)} {level.upper()}: {count} person(s)")
        if people:
            for person in people:
                print(f"   - {person}")
        else:
            print(f"   (none)")
    
    print("\nDefault level for unmapped people: " + DEFAULT_LEVEL)
    print("Unknown faces treated as: " + UNKNOWN_LEVEL)
    print("="*60 + "\n")


# ==================== VALIDATION ====================

def validate_authorization_map():
    """Validate that all mapped people have face data"""
    import os
    
    faces_dir = "datasets/faces"
    if not os.path.exists(faces_dir):
        print(f"[WARN] Faces directory not found: {faces_dir}")
        return
    
    existing_people = [d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))]
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
    validate_authorization_map()
    
    # Test examples
    print("\nTesting authorization levels:")
    test_names = ["Aldrikz", "Art", "Jude", "Myke", "Speed", "Unknown", "NonExistent"]
    
    for name in test_names:
        level = get_authorization_level(name)
        formatted = format_display_name(name, level)
        color = get_level_color(level)
        print(f"{name:15s} → {formatted} (Color: BGR{color})")
