"""
Backend Color Embedding Integration Test
Tests the color embedding functionality in d3_data_manager
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "src/d3-app/src/backend"
sys.path.insert(0, str(backend_path))

from services.d3_data_manager import D3DataManager

def test_color_embedding():
    """Test color embedding functionality"""
    print("=" * 70)
    print("Color Embedding Backend Integration Test")
    print("=" * 70)
    
    # Initialize manager
    print("\n1. Initializing D3DataManager...")
    manager = D3DataManager()
    
    # Test with default coloring
    print("\n2. Loading data with DEFAULT coloring (color_mode='cluster')...")
    try:
        result_default = manager.get_initial_data(
            dataset="default",
            dr_method="umap",
            color_mode='cluster'
        )
        print(f"✓ Loaded {len(result_default['points'])} points")
        if 'color' in result_default['points'][0]:
            print(f"  Sample point color: {result_default['points'][0]['color']}")
        else:
            print("  No color field in points (expected for default mode)")
    except Exception as e:
        print(f"✗ Error with default coloring: {e}")
        return False
    
    # Test with distance-based coloring
    print("\n3. Loading data with DISTANCE-BASED coloring (color_mode='distance')...")
    try:
        result_distance = manager.get_initial_data(
            dataset="default",
            dr_method="umap",
            color_mode='distance'
        )
        print(f"✓ Loaded {len(result_distance['points'])} points")
        
        # Check for color fields
        colored_count = sum(1 for p in result_distance['points'] if 'color' in p)
        print(f"✓ Points with color: {colored_count} / {len(result_distance['points'])}")
        
        if colored_count > 0:
            sample_colors = [p['color'] for p in result_distance['points'] if 'color' in p][:5]
            print(f"  Sample colors: {sample_colors}")
        
    except Exception as e:
        print(f"✗ Error with distance-based coloring: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare the two
    print("\n4. Comparison:")
    print(f"  Default mode points: {len(result_default['points'])}")
    print(f"  Distance mode points: {len(result_distance['points'])}")
    
    print("\n✅ Backend integration test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_color_embedding()
    sys.exit(0 if success else 1)
