#!/usr/bin/env python3
"""
Test script for the layout-api module (moved under tests/).

Usage (from repo root):
    python tests/test_layout_api.py
"""

import sys
from pathlib import Path

# Ensure repo root and api path are on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
API_DIR = REPO_ROOT / "api"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

# Import utility functions
from utils import import_api, extract_attrs

# Import layout-api.py using simplified utility
layout_api = import_api(API_DIR / "layout-api.py", "layout_api")

# Extract functions
get_map, get_map_json, get_map_as_string, get_map_info, get_symbol, get_symbols = extract_attrs(
    layout_api, 'get_map', 'get_map_json', 'get_map_as_string', 'get_map_info', 'get_symbol', 'get_symbols'
)


def test_get_map_json():
    """Test getting raw JSON data."""
    print("=" * 60)
    print("Test 1: get_map_json()")
    print("=" * 60)
    
    json_data = get_map_json()
    
    if json_data:
        print("✓ Successfully loaded JSON data")
        print(f"  - Number of rows: {len(json_data)}")
        if json_data:
            print(f"  - Number of cols: {len(json_data[0])}")
        print(f"  - First row preview: {json_data[0][:10]}...")
    else:
        print("✗ No JSON data found (grid.json might not exist)")
    
    print()


def test_get_map_default():
    """Test getting map with default colored symbols."""
    print("=" * 60)
    print("Test 2: get_map() with default colored symbols")
    print("=" * 60)
    
    map_data = get_map()
    
    if map_data:
        print("✓ Successfully loaded map with colored symbols")
        print(f"  - Map dimensions: {len(map_data)} rows × {len(map_data[0])} cols")
        print(f"  - Symbols are colored by default (gray '.', red '#', green 'H')")
        print(f"  - First row: {map_data[0][:20]}...")
        print("  - Sample cells from first 3 rows:")
        for i, row in enumerate(map_data[:3]):
            print(f"    Row {i}: {row[:15]}...")
    else:
        print("✗ No map data found")
    
    print()


def test_get_map_custom_symbols():
    """Test getting map with custom symbols."""
    print("=" * 60)
    print("Test 3: get_map() with custom symbols")
    print("=" * 60)
    
    custom_symbols = {
        'FREE': '.',
        'OBSTACLE': '#',
        'HOME': 'H'
    }
    
    map_data = get_map(symbols=custom_symbols)
    
    if map_data:
        print("✓ Successfully loaded map with custom symbols")
        print(f"  - Custom symbols: {custom_symbols}")
        print(f"  - First row: {map_data[0][:61]}...")
        print("  - Sample cells from first 3 rows:")
        for i, row in enumerate(map_data[:3]):
            print(f"    Row {i}: {row[:15]}...")
    else:
        print("✗ No map data found")
    
    print()


def test_get_map_as_string():
    """Test getting map as formatted string with colored symbols."""
    print("=" * 60)
    print("Test 4: get_map_as_string() with colored symbols")
    print("=" * 60)
    
    map_str = get_map_as_string()
    
    if map_str:
        lines = map_str.split('\n')
        print("✓ Successfully generated string representation with colored symbols")
        print(f"  - Total lines: {len(lines)}")
        print(f"  - Symbols are colored (gray '.', red '#', green 'H')")
        print(f"  - First {min(5, len(lines))} lines:")
        for i, line in enumerate(lines[:5]):
            print(f"    {line[:60]}...")
    else:
        print("✗ No map data found")
    
    print()


def test_get_map_as_string_custom():
    """Test getting map as string with custom symbols and separator."""
    print("=" * 60)
    print("Test 5: get_map_as_string() with custom symbols and separator")
    print("=" * 60)
    
    custom_symbols = {
        'FREE': '.',
        'OBSTACLE': '#',
        'HOME': 'H'
    }
    
    map_str = get_map_as_string(symbols=custom_symbols, cell_separator=" ")
    
    if map_str:
        lines = map_str.split('\n')
        print("✓ Successfully generated string with custom symbols and separator")
        print(f"  - Total lines: {len(lines)}")
        print(f"  - First {min(5, len(lines))} lines:")
        for i, line in enumerate(lines[:5]):
            print(f"    {line[:60]}...")
    else:
        print("✗ No map data found")
    
    print()


def test_get_map_info():
    """Test getting map statistics and information."""
    print("=" * 60)
    print("Test 6: get_map_info()")
    print("=" * 60)
    
    info = get_map_info()
    
    print("✓ Map statistics:")
    print(f"  - Dimensions: {info['rows']} rows × {info['cols']} cols")
    print(f"  - Total cells: {info['total_cells']}")
    print(f"  - Free cells: {info['free_count']}")
    print(f"  - Obstacle cells: {info['obstacle_count']}")
    print(f"  - Home cells: {info['home_count']}")
    
    if info['total_cells'] > 0:
        free_pct = (info['free_count'] / info['total_cells']) * 100
        obstacle_pct = (info['obstacle_count'] / info['total_cells']) * 100
        home_pct = (info['home_count'] / info['total_cells']) * 100
        print(f"\n  Percentages:")
        print(f"  - Free: {free_pct:.1f}%")
        print(f"  - Obstacle: {obstacle_pct:.1f}%")
        print(f"  - Home: {home_pct:.1f}%")
    
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Layout API - Test Suite")
    print("=" * 60)
    print()
    
    test_get_map_json()
    test_get_map_default()
    test_get_map_custom_symbols()
    test_get_map_as_string()
    test_get_map_as_string_custom()
    test_get_map_info()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
