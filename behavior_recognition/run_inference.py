"""
Launcher for Behavior Recognition Inference
Handles import paths automatically.
"""
import sys
from pathlib import Path

# Add behavior_recognition directory to Python path
behavior_dir = Path(__file__).parent
sys.path.insert(0, str(behavior_dir))

# Import and run inference
from inference_behavior import main

if __name__ == "__main__":
    main()
