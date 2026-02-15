"""
Authorization Manager
---------------------
Manages authorization levels for face recognition system.
Automatically syncs with datasets/faces/ folder structure and persists to JSON.

Authorization Levels:
    - "Authorized": Full access
    - "Partially Authorized": Limited access
    - "Unauthorized": No access (default)
"""

import os
import json
from pathlib import Path
from typing import Dict, List


class AuthorizationManager:
    """Manages authorization levels for registered personnel."""
    
    # Authorization level constants
    AUTHORIZED = "Authorized"
    PARTIAL = "Partially Authorized"
    UNAUTHORIZED = "Unauthorized"
    
    VALID_LEVELS = [AUTHORIZED, PARTIAL, UNAUTHORIZED]
    DEFAULT_LEVEL = UNAUTHORIZED
    
    def __init__(self, faces_dir: str = "datasets/faces", config_file: str = "authorization_map.json"):
        """
        Initialize the authorization manager.
        
        Args:
            faces_dir: Path to the faces dataset directory
            config_file: Path to save/load authorization configuration
        """
        self.faces_dir = Path(faces_dir)
        self.config_file = Path(config_file)
        self.authorization_map: Dict[str, str] = {}
        
        # Ensure faces directory exists
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and sync
        self._load_or_create()
        self._sync_with_folders()
    
    def _load_or_create(self) -> None:
        """Load existing authorization map or create new one."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Validate loaded data
                if isinstance(data, dict):
                    # Filter out invalid authorization levels
                    self.authorization_map = {
                        k: v for k, v in data.items() 
                        if v in self.VALID_LEVELS
                    }
                    print(f"[INFO] Loaded authorization map with {len(self.authorization_map)} entries")
                else:
                    print("[WARN] Invalid authorization map format, creating new one")
                    self.authorization_map = {}
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse authorization map: {e}")
                print("[INFO] Creating new authorization map")
                self.authorization_map = {}
            except Exception as e:
                print(f"[ERROR] Failed to load authorization map: {e}")
                self.authorization_map = {}
        else:
            print("[INFO] No existing authorization map found, creating new one")
            self.authorization_map = {}
    
    def _get_person_folders(self) -> List[str]:
        """
        Scan the faces directory and return list of person names (folder names).
        
        Returns:
            List of person names (directory names only)
        """
        if not self.faces_dir.exists():
            print(f"[WARN] Faces directory not found: {self.faces_dir}")
            return []
        
        person_folders = []
        try:
            for item in self.faces_dir.iterdir():
                # Only include directories, exclude files like embeddings.npy
                if item.is_dir():
                    person_folders.append(item.name)
        except Exception as e:
            print(f"[ERROR] Failed to scan faces directory: {e}")
            return []
        
        return sorted(person_folders)
    
    def _sync_with_folders(self) -> None:
        """
        Sync authorization map with current folder structure.
        - Add new folders with default "Unauthorized"
        - Keep existing authorization levels
        - Remove entries for deleted folders (optional cleanup)
        """
        current_folders = set(self._get_person_folders())
        existing_entries = set(self.authorization_map.keys())
        
        # Add new folders with default authorization
        new_folders = current_folders - existing_entries
        if new_folders:
            print(f"[INFO] Found {len(new_folders)} new person(s): {sorted(new_folders)}")
            for folder in new_folders:
                self.authorization_map[folder] = self.DEFAULT_LEVEL
                print(f"[INFO]   Added '{folder}' with default level: {self.DEFAULT_LEVEL}")
        
        # Optional: Remove entries for deleted folders
        deleted_folders = existing_entries - current_folders
        if deleted_folders:
            print(f"[INFO] Found {len(deleted_folders)} deleted folder(s): {sorted(deleted_folders)}")
            for folder in deleted_folders:
                old_level = self.authorization_map.pop(folder)
                print(f"[INFO]   Removed '{folder}' (was: {old_level})")
        
        # Save changes if any
        if new_folders or deleted_folders:
            self._save()
        else:
            print("[INFO] Authorization map is up to date")
    
    def _save(self) -> None:
        """Save authorization map to JSON file."""
        try:
            # Create backup if file exists
            if self.config_file.exists():
                backup_file = self.config_file.with_suffix('.json.backup')
                self.config_file.replace(backup_file)
            
            # Save with pretty formatting
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.authorization_map, f, indent=4, ensure_ascii=False, sort_keys=True)
            
            print(f"[INFO] Saved authorization map to: {self.config_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save authorization map: {e}")
    
    def get_authorization(self, person_name: str) -> str:
        """
        Get authorization level for a person (case-insensitive).
        
        Args:
            person_name: Name of the person
            
        Returns:
            Authorization level (defaults to UNAUTHORIZED if not found)
        """
        # Case-insensitive lookup
        person_lower = person_name.lower()
        for name, level in self.authorization_map.items():
            if name.lower() == person_lower:
                return level
        return self.DEFAULT_LEVEL
    
    def set_authorization(self, person_name: str, level: str) -> bool:
        """
        Set authorization level for a person and save to file.
        
        Args:
            person_name: Name of the person
            level: Authorization level (must be one of VALID_LEVELS)
            
        Returns:
            True if successful, False otherwise
        """
        # Validate level
        if level not in self.VALID_LEVELS:
            print(f"[ERROR] Invalid authorization level: {level}")
            print(f"[INFO] Valid levels: {self.VALID_LEVELS}")
            return False
        
        # Check if person exists in folders
        current_folders = self._get_person_folders()
        if person_name not in current_folders:
            print(f"[WARN] Person '{person_name}' not found in {self.faces_dir}")
            print(f"[INFO] Available persons: {current_folders}")
            return False
        
        # Update and save
        old_level = self.authorization_map.get(person_name, "Not set")
        self.authorization_map[person_name] = level
        self._save()
        
        print(f"[INFO] Updated '{person_name}': {old_level} → {level}")
        return True
    
    def get_all_authorizations(self) -> Dict[str, str]:
        """
        Get a copy of the entire authorization map.
        
        Returns:
            Dictionary mapping person names to authorization levels
        """
        return self.authorization_map.copy()
    
    def refresh(self) -> None:
        """
        Refresh authorization map by re-syncing with folder structure.
        Useful after adding/removing person folders.
        """
        print("[INFO] Refreshing authorization map...")
        self._sync_with_folders()
    
    def get_persons_by_level(self, level: str) -> List[str]:
        """
        Get list of persons with specific authorization level.
        
        Args:
            level: Authorization level to filter by
            
        Returns:
            List of person names
        """
        return [name for name, auth_level in self.authorization_map.items() if auth_level == level]
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about authorization levels.
        
        Returns:
            Dictionary with counts for each authorization level
        """
        stats = {level: 0 for level in self.VALID_LEVELS}
        for level in self.authorization_map.values():
            if level in stats:
                stats[level] += 1
        return stats
    
    def export_to_dict(self) -> Dict[str, str]:
        """
        Export authorization map for use in other modules.
        Returns lowercase keys for case-insensitive matching.
        
        Returns:
            Dictionary with lowercase person names as keys
        """
        return {name.lower(): level for name, level in self.authorization_map.items()}


# Convenience functions for quick usage
def load_authorization_map(faces_dir: str = "datasets/faces", 
                          config_file: str = "authorization_map.json") -> Dict[str, str]:
    """
    Quick function to load authorization map.
    
    Args:
        faces_dir: Path to faces dataset
        config_file: Path to authorization config file
        
    Returns:
        Dictionary mapping person names (lowercase) to authorization levels
    """
    manager = AuthorizationManager(faces_dir, config_file)
    return manager.export_to_dict()


def update_authorization(person_name: str, level: str,
                        faces_dir: str = "datasets/faces",
                        config_file: str = "authorization_map.json") -> bool:
    """
    Quick function to update a person's authorization level.
    
    Args:
        person_name: Name of the person
        level: Authorization level
        faces_dir: Path to faces dataset
        config_file: Path to authorization config file
        
    Returns:
        True if successful, False otherwise
    """
    manager = AuthorizationManager(faces_dir, config_file)
    return manager.set_authorization(person_name, level)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Authorization Manager Demo")
    print("=" * 60)
    
    # Initialize manager
    manager = AuthorizationManager()
    
    print("\n--- Current Authorization Map ---")
    for person, level in manager.get_all_authorizations().items():
        print(f"  {person}: {level}")
    
    print("\n--- Statistics ---")
    stats = manager.get_statistics()
    for level, count in stats.items():
        print(f"  {level}: {count}")
    
    print("\n--- Example: Update Authorization ---")
    # Uncomment to test setting authorization
    # manager.set_authorization("dean", AuthorizationManager.AUTHORIZED)
    # manager.set_authorization("myke", AuthorizationManager.PARTIAL)
    
    print("\n--- Export for Face Recognition ---")
    auth_dict = manager.export_to_dict()
    print(f"Exported {len(auth_dict)} entries (lowercase keys)")
    
    print("\n" + "=" * 60)
