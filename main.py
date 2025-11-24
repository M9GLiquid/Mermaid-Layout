"""
Main entry point for the Mermaid Layout Editor.

This script launches the grid editor that allows users to mark obstacles and home
positions on a rectified GPS camera feed. The editor uses the GPS overlay API
for coordinate transformations and grid management.

Usage:
    python main.py

Controls:
    - Left click: Cycle cell states (FREE → OBSTACLE → HOME)
    - 's': Save grid to JSON file
    - 'i': Save current frame with grid overlay
    - 'f': Toggle fullscreen mode
    - 'q': Quit editor
"""

# Add src directory to path to import editor_prototype
from src.editor_prototype import main

if __name__ == "__main__":
    main()
