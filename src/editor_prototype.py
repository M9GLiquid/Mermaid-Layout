"""
Grid editor that overlays an editable occupancy grid on top of a rectified GPS camera feed.

The editor fetches frames from an Axis network camera, applies rectification using the GPS
overlay API to produce a top-down view, and allows users to mark grid cells as obstacles
or home positions. All grid storage flows through `grid_api`, ensuring consistency with
downstream systems like A* pathfinding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import sys
import cv2  # type: ignore
import numpy as np

# Add parent directory to path to import from api folder
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import utility functions
from utils import import_api

# Import overlay-api.py using simplified utility
overlay_api = import_api(project_root / "overlay" / "overlay-api.py", "overlay_api")
GPSOverlay = overlay_api.GPSOverlay

from src.axis_test import fetch_axis_snapshot, CAMERA_IP, USERNAME, PASSWORD
from src.grid_api import (
    Grid,
    load_grid,
    save_grid,
    DEFAULT_GRID_PATH,
    FREE,
    OBSTACLE,
    HOME,
)

# Lookup table mapping numeric cell states to printable symbols. This keeps console
# snapshots readable while still operating on integers internally.
DISPLAY_SYMBOL: Dict[int, str] = {
    FREE: ".",
    OBSTACLE: "#",
    HOME: "H",
}


@dataclass
class EditorState:
    """
    Bundle of data that needs to be accessed by both the render loop and mouse handler.

    The overlay object provides coordinate transformations and grid dimensions, while the
    mutable grid stores the current occupancy state. Cached cell sizes speed up mouse
    coordinate calculations.
    """

    grid: Grid
    overlay: GPSOverlay
    rows: int
    cols: int
    arena_bounds: Dict[str, float]  # Original bounds in rectified canvas space
    offset_x: int = 0  # Offset from API to convert image pixel to canvas coordinate
    offset_y: int = 0 
    rectified_width: int = 0
    rectified_height: int = 0
    cell_width: float = 1.0
    cell_height: float = 1.0
    grid_path: Path = DEFAULT_GRID_PATH

def _get_rectified_frame(overlay: GPSOverlay, snapshot_path: Optional[Path] = None) -> Tuple[np.ndarray, int, int, Dict[str, int]]:
    """
    Fetch a frame from the Axis camera (or load saved snapshot) and apply rectification.
    
    Uses the overlay API's transform_image() method to handle all rectification steps:
    fisheye correction, homography transformation, and proper coordinate mapping.
    
    Args:
        overlay: GPSOverlay instance with calibration data
        snapshot_path: Optional path to save/load snapshot. Defaults to "snapshot_raw.png"
    
    Returns:
        Tuple of (rectified_frame, width, height, offset_info) where:
        - rectified_frame: BGR image array of the rectified view
        - width, height: Dimensions of the rectified output canvas
        - offset_info: Dict with 'offset_x' and 'offset_y' to convert image pixels to canvas coordinates
    """
    if snapshot_path is None:
        snapshot_path = Path("snapshot_raw.png")
    else:
        snapshot_path = Path(snapshot_path)
    
    # Try to load saved snapshot first
    if snapshot_path.exists():
        print(f"Loading saved snapshot from {snapshot_path}")
        raw_frame = cv2.imread(str(snapshot_path))
        if raw_frame is None:
            print("Warning: Failed to load snapshot, fetching from camera...")
            raw_frame = fetch_axis_snapshot(
                camera_ip=CAMERA_IP,
                username=USERNAME,
                password=PASSWORD,
                resolution=None
            )
            # Save the fetched frame for next time
            cv2.imwrite(str(snapshot_path), raw_frame)
            print(f"Saved snapshot to {snapshot_path}")
        else:
            print("Using saved snapshot (no camera connection needed)")
    else:
        # Fetch raw frame from Axis camera (full resolution, no cropping)
        print("No saved snapshot found, fetching from camera...")
        raw_frame = fetch_axis_snapshot(
            camera_ip=CAMERA_IP,
            username=USERNAME,
            password=PASSWORD,
            resolution=None  # Let camera return full resolution without cropping
        )
        # Save the fetched frame for next time
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(snapshot_path), raw_frame)
        print(f"Saved snapshot to {snapshot_path} for future use")
    
    # Debug: Check actual frame size
    actual_height, actual_width = raw_frame.shape[:2]
    print(f"Frame size: {actual_width} × {actual_height}")
    print(f"Expected server size: {overlay.server_size[0]} × {overlay.server_size[1]}")
    
    # Use overlay API to transform the image (handles fisheye correction + homography)
    # Use show_grid=True to get the API's grid lines - we'll overlay cell states on top
    print("Applying rectification using overlay API...")
    rectified, offset_info = overlay.transform_image(str(snapshot_path), show_grid=True)
    
    # Get output dimensions
    rectified_height, rectified_width = rectified.shape[:2]
    print(f"Rectified output size: {rectified_width} × {rectified_height}")
    print(f"Offset info: {offset_info}")
    
    # Save the rectified image for inspection
    rectified_path = Path("snapshot_rectified.png")
    cv2.imwrite(str(rectified_path), rectified)
    print(f"Saved rectified image to {rectified_path}")
    
    return rectified, rectified_width, rectified_height, offset_info

def _rectified_to_grid_cell(
    x: float, y: float, state: EditorState
) -> Optional[Tuple[int, int]]:
    """
    Convert a mouse click position in image pixel space to a grid cell using the overlay API.
    
    Converts image pixel coordinates to canvas coordinates, then uses 
    overlay.get_grid_cell_from_rectified() to get the correct grid cell index.
    This ensures we're using the exact same grid cell calculation as the overlay API.
    
    Args:
        x: Mouse X coordinate in display window (image pixel coordinates)
        y: Mouse Y coordinate in display window (image pixel coordinates)
        state: Editor state containing overlay and offset info
        
    Returns:
        Tuple of (row, col) if click is within arena bounds, None otherwise.
    """
    # Convert image pixel coordinates to canvas coordinates
    # x_canvas = x_img + offset_x (as per API documentation)
    x_canvas = x + state.offset_x
    y_canvas = y + state.offset_y
    
    # Use overlay API's get_grid_cell_from_rectified() method to get the grid cell
    cell_info = state.overlay.get_grid_cell_from_rectified(x_canvas, y_canvas)
    
    if not cell_info["in_bounds"]:
        return None
    
    # Return row, col
    return cell_info["row"], cell_info["col"]

def _seed_grid(rows: int, cols: int, persisted: Sequence[Sequence[int]]) -> Grid:
    """
    Build a grid with the requested dimensions, copying any persisted values that fit.

    When the saved grid uses different dimensions the mismatched cells are clipped,
    keeping the prototype resilient while allowing manual edits of the JSON file.
    """
    seeded = [[FREE for _ in range(cols)] for _ in range(rows)]
    max_row = min(rows, len(persisted))
    for row_idx in range(max_row):
        row = persisted[row_idx]
        max_col = min(cols, len(row))
        for col_idx in range(max_col):
            seeded[row_idx][col_idx] = int(row[col_idx])
    return seeded

def _draw_header_overlay(img: np.ndarray, state: EditorState) -> np.ndarray:
    """
    Draw header above the image (not overlaying it) with status and instructions.
    
    Similar to Mermaid-Overlay tools, displays grid dimensions, cell counts, and
    keyboard shortcuts in a dark gray header bar above the image.
    
    Header size and font scales with image resolution for better visibility on large displays.
    
    Args:
        img: The image to add header above
        state: Editor state containing grid and dimensions
        
    Returns:
        New image with header bar above the original image
    """
    h, w = img.shape[:2]
    
    # Calculate header height based on image resolution (5-8% of height, min 80px, max 150px)
    header_height = max(80, min(150, int(h * 0.07)))
    
    # Calculate font scales based on image width (scales with resolution)
    # Base scale factors that work well for 1920px width
    base_width = 1920.0
    scale_factor = w / base_width
    
    # Font sizes scale with resolution (increased instruction font for better visibility)
    status_font_scale = max(0.5, min(1.2, 0.6 * scale_factor))
    instruction_font_scale = max(0.5, min(1.2, 0.7 * scale_factor))  # Increased from 0.5 to 0.7
    
    # Thickness scales with font size (instruction text uses thicker lines for better white visibility)
    status_thickness = max(1, int(2 * scale_factor))
    instruction_thickness = max(2, int(3 * scale_factor))  # Thicker than status text to ensure white appearance
    
    # Padding scales with resolution
    padding_x = max(10, int(10 * scale_factor))
    padding_y_status = max(25, int(25 * scale_factor))
    padding_y_instruction = max(55, int(55 * scale_factor))
    
    # Create a new image with header space above
    canvas = np.zeros((h + header_height, w, 3), dtype=np.uint8)
    
    # Draw header bar with dark gray background
    cv2.rectangle(canvas, (0, 0), (w, header_height), (40, 40, 40), -1)
    cv2.rectangle(canvas, (0, header_height - 1), (w, header_height), (80, 80, 80), 1)
    
    # Count obstacles and home positions
    obstacle_count = sum(1 for row in state.grid for cell in row if cell == OBSTACLE)
    home_count = sum(1 for row in state.grid for cell in row if cell == HOME)
    
    # Status text with grid dimensions and cell counts
    status_text = f"Grid: {state.rows}x{state.cols} cells | Obstacles: {obstacle_count} | Home: {home_count}"
    
    # Instruction text with keyboard shortcuts
    instruction_text = "Left click: Cycle cells (FREE → OBSTACLE → HOME) | 's' Save | 'i' Save image | 'f' Fullscreen | 'q' Quit"
    
    # Draw status text in header (white, larger font, scaled)
    cv2.putText(
        canvas, 
        status_text, 
        (padding_x, padding_y_status), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        status_font_scale, 
        (255, 255, 255), 
        status_thickness, 
        cv2.LINE_AA
    )
    
    # Draw instruction text (white, larger font, scaled)
    cv2.putText(
        canvas, 
        instruction_text, 
        (padding_x, padding_y_instruction), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        instruction_font_scale, 
        (255, 255, 255),  # White color for maximum visibility
        instruction_thickness, 
        cv2.LINE_AA
    )
    
    # Place original image below header
    canvas[header_height:, :] = img
    
    return canvas

def _draw_grid_overlay(frame, state: EditorState) -> None:
    """
    Draw cell state overlays (obstacles and home markers) on top of the rectified frame.
    
    The grid lines are already drawn by the overlay API (show_grid=True), so we only
    need to overlay the cell states. Uses arena_bounds which are in rectified canvas space,
    and converts them to image pixel coordinates using the offset.
    """
    rows, cols = state.rows, state.cols
    bounds = state.arena_bounds  # Arena bounds in rectified canvas space
    
    # Convert canvas coordinates to image pixel coordinates
    # grid_left = int(left - min_x) in API, so image_x = canvas_x - offset_x
    grid_left = int(bounds["left"] - state.offset_x)
    grid_top = int(bounds["top"] - state.offset_y)
    grid_right = int(bounds["right"] - state.offset_x)
    grid_bottom = int(bounds["bottom"] - state.offset_y)
    
    # Calculate cell dimensions (same as API)
    cell_width = (grid_right - grid_left) / cols
    cell_height = (grid_bottom - grid_top) / rows
    
    # Cache for mouse handler (not used anymore but kept for compatibility)
    state.cell_width = cell_width
    state.cell_height = cell_height

    # Draw cell state overlays (obstacles and home markers) with improved visibility
    tinted = frame.copy()
    for row_idx in range(rows):
        for col_idx in range(cols):
            cell_value = state.grid[row_idx][col_idx]
            if cell_value == FREE:
                continue
            
            # Use the exact same formula as API's grid line drawing:
            # grid_left + (col * (grid_right - grid_left) / cols)
            left_x = grid_left + (col_idx * (grid_right - grid_left) / cols)
            right_x = grid_left + ((col_idx + 1) * (grid_right - grid_left) / cols)
            top_y = grid_top + (row_idx * (grid_bottom - grid_top) / rows)
            bottom_y = grid_top + ((row_idx + 1) * (grid_bottom - grid_top) / rows)
            
            top_left = (int(left_x), int(top_y))
            bottom_right = (int(right_x), int(bottom_y))
            
            # Define colors and borders for better visibility
            if cell_value == OBSTACLE:
                # Obstacles: darker red tint with black border
                fill_colour = (0, 0, 100)  # Dark red (BGR)
                border_colour = (0, 0, 0)  # Black border
                border_thickness = 2
            else:  # HOME
                # Home: orange/yellow tint with bright orange border
                fill_colour = (0, 150, 255)  # Orange (BGR)
                border_colour = (0, 100, 255)  # Bright orange border
                border_thickness = 3
            
            # Draw filled rectangle with higher opacity
            cv2.rectangle(
                tinted,
                top_left,
                bottom_right,
                color=fill_colour,
                thickness=-1,  # Filled rectangle
            )
            
            # Draw border for better visibility
            cv2.rectangle(
                tinted,
                top_left,
                bottom_right,
                color=border_colour,
                thickness=border_thickness,
            )

    # Blend tinted overlay back on top with higher opacity for better visibility
    # Increased from 0.4/0.6 to 0.5/0.5 for better visibility while keeping image visible
    cv2.addWeighted(tinted, 0.5, frame, 0.5, 0, dst=frame)

def _format_grid_snapshot(grid: Grid) -> str:
    """
    Return a string representation of the grid using the display symbols with colors.
    
    Colors:
    - 'H' (Home): Green
    - 'X' (Obstacle): Orange/Yellow
    - '.' (Free): Default color
    """
    # ANSI color codes
    GREEN = "\033[92m"  # Bright green
    ORANGE = "\033[93m"  # Bright yellow/orange (closest to orange)
    RESET = "\033[0m"  # Reset to default
    
    def format_cell(cell: int) -> str:
        """Format a single cell with appropriate color."""
        symbol = DISPLAY_SYMBOL.get(cell, "?")
        if symbol == "H":
            return f"{GREEN}{symbol}{RESET}"
        elif symbol == "X":
            return f"{ORANGE}{symbol}{RESET}"
        else:
            return symbol
    
    return "\n".join(
        " ".join(format_cell(cell) for cell in row) for row in grid
    )

def _handle_mouse(event, x, y, _flags, state: EditorState) -> None:
    """
    Cycle the clicked cell through FREE -> OBSTACLE -> HOME on each left click.
    
    Mouse coordinates are converted from display space to grid cell indices using the
    rectified coordinate system and arena bounds.
    """
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Convert display coordinates to grid cell
    cell_pos = _rectified_to_grid_cell(x, y, state)
    if cell_pos is None:
        return
    
    row, col = cell_pos
    current = state.grid[row][col]

    updated_value = {
        FREE: OBSTACLE,
        OBSTACLE: HOME,
        HOME: FREE,
    }[current]

    state.grid[row][col] = updated_value

    # ANSI color codes for individual cell update message
    GREEN = "\033[92m"  # Bright green
    ORANGE = "\033[93m"  # Bright yellow/orange
    RESET = "\033[0m"  # Reset to default
    
    symbol = DISPLAY_SYMBOL.get(updated_value, "?")
    if symbol == "H":
        colored_symbol = f"{GREEN}{symbol}{RESET}"
    elif symbol == "X":
        colored_symbol = f"{ORANGE}{symbol}{RESET}"
    else:
        colored_symbol = symbol
    
    print(f"[update] cell ({row}, {col}) -> {colored_symbol}")
    print(_format_grid_snapshot(state.grid))

def run_editor(
    overlay: GPSOverlay, *, grid_path: Path | str = DEFAULT_GRID_PATH
) -> None:
    """
    Launch the grid editor using the GPS overlay API and Axis network camera.

    The function fetches frames from the Axis camera, applies rectification to produce
    a top-down view, and overlays the editable grid. Grid dimensions are read from the
    overlay calibration data, ensuring consistency with the navigation system.
    """
    rows = overlay.grid_rows
    cols = overlay.grid_cols
    
    persisted = load_grid(grid_path)
    grid = _seed_grid(rows, cols, persisted)

    # Get initial rectified frame to determine display size and offset info
    rectified_frame, rect_width, rect_height, offset_info = _get_rectified_frame(overlay)

    state = EditorState(
        grid=grid,
        overlay=overlay,
        rows=rows,
        cols=cols,
        arena_bounds=overlay.arena_bounds,
        offset_x=offset_info["offset_x"],
        offset_y=offset_info["offset_y"],
        rectified_width=rect_width,
        rectified_height=rect_height,
        grid_path=Path(grid_path),
    )

    window_name = "Mermaid AI Grid Editor (Rectified)"
    # Create resizable window (WINDOW_NORMAL allows resizing)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _handle_mouse, state)
    
    # Maximize window on startup
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"Grid dimensions: {rows} rows × {cols} cols")
    print(f"Rectified frame size: {rect_width} × {rect_height}")
    print("Controls: left click cycles cells (FREE → OBSTACLE → HOME), 's' saves grid, 'i' saves image, 'f' toggle fullscreen, 'q' quits.")
    print("Note: Snapshot is saved to 'snapshot_raw.png' - delete it to fetch a fresh image from camera.")
    
    # Store the base rectified frame (we'll draw overlays on a copy each frame)
    base_rectified_frame = rectified_frame.copy()
    
    # Save the initial frame with grid and overlays (without header for cleaner saved image)
    initial_display_frame = base_rectified_frame.copy()
    _draw_grid_overlay(initial_display_frame, state)
    rectified_with_grid_path = Path("snapshot_rectified_with_grid.png")
    cv2.imwrite(str(rectified_with_grid_path), initial_display_frame)
    print(f"Saved rectified image with grid and cell overlays to {rectified_with_grid_path}")
    
    # Track fullscreen state
    is_fullscreen = True
    
    try:
        while True:
            # Create a fresh copy of the base frame for drawing overlays
            # This allows us to redraw the grid without re-fetching from the camera
            display_frame = base_rectified_frame.copy()

            # Draw grid overlay on the display frame
            _draw_grid_overlay(display_frame, state)
            
            # Add header overlay above the image
            display_frame_with_header = _draw_header_overlay(display_frame, state)
            
            cv2.imshow(window_name, display_frame_with_header)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting editor. Goodbye!")
                break
            if key == ord("s"):
                save_grid(state.grid, state.grid_path)
                print(f"Grid saved to {state.grid_path.resolve()}")
            if key == ord("i"):
                # Save current frame with grid and cell overlays (without header for cleaner saved image)
                save_frame = display_frame.copy()
                cv2.imwrite(str(rectified_with_grid_path), save_frame)
                print(f"Image with grid and overlays saved to {rectified_with_grid_path}")
            if key == ord("f"):
                # Toggle fullscreen
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("Fullscreen mode")
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    print("Windowed mode")
    finally:
        cv2.destroyWindow(window_name)

def main() -> None:
    """
    Entry point that loads the GPS overlay calibration and launches the editor.

    The overlay configuration file (`overlay/gps_overlay.json`) provides grid dimensions,
    arena bounds, and transformation matrices needed for rectification.
    """
    overlay_path = Path(__file__).parent.parent / "overlay" / "gps_overlay.json"
    
    if not overlay_path.exists():
        raise FileNotFoundError(
            f"GPS overlay configuration not found at {overlay_path}. "
            "Please ensure gps_overlay.json exists in the overlay/ folder."
        )
    
    overlay = GPSOverlay(str(overlay_path))
    run_editor(overlay)

if __name__ == "__main__":
    main()
