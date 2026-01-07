"""
Test script for GPU UMAP Zoom Server
Tests the dedicated gpu_umap_server.py on port 8001
"""

import requests
import numpy as np
import base64
import io
import json
from pathlib import Path

# Configuration
GPU_SERVER_URL = "http://localhost:8001"
HEALTH_ENDPOINT = f"{GPU_SERVER_URL}/health"
INFO_ENDPOINT = f"{GPU_SERVER_URL}/api/info"
ZOOM_ENDPOINT = f"{GPU_SERVER_URL}/api/zoom/redraw"

# Load sample data from d3_data_manager
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from d3_data_manager import d3_data_manager as data_manager
    DATA_AVAILABLE = True
except:
    print("⚠ Warning: Could not load d3_data_manager")
    DATA_AVAILABLE = False

def test_health_check():
    """Test server health check"""
    print("\n" + "=" * 60)
    print("Test 0: Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=10)
        if response.status_code == 200:
            result = response.json()
            gpu_available = result.get('gpu_available')
            status = "✓ HEALTHY" if result.get('status') == 'ok' else "✗ UNHEALTHY"
            print(f"{status}")
            print(f"GPU Available: {gpu_available}")
            return gpu_available
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to GPU server at port 8001")
        print("  Start the server with:")
        print("    cd src/d3-app/src/backend")
        print("    conda activate rapids-25.10")
        print("    uvicorn gpu_umap_server:app --host 0.0.0.0 --port 8001")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_server_info():
    """Test server info endpoint"""
    print("\n" + "=" * 60)
    print("Test Info: Server Information")
    print("=" * 60)
    
    try:
        response = requests.get(INFO_ENDPOINT, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"Name: {result.get('name')}")
            print(f"Version: {result.get('version')}")
            print(f"GPU Available: {result.get('gpu_available')}")
            print(f"Endpoints: {len(result.get('endpoints', []))}")
            return True
        else:
            print(f"✗ Info request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def _numpy_to_b64(array: np.ndarray) -> str:
    """NumPy配列をBase64にエンコード"""
    buff = io.BytesIO()
    np.save(buff, array, allow_pickle=False)
    return base64.b64encode(buff.getvalue()).decode('utf-8')


def _b64_to_numpy(data_b64: str) -> np.ndarray:
    """Base64からNumPy配列にデコード"""
    decoded = base64.b64decode(data_b64)
    return np.load(io.BytesIO(decoded))


def test_zoom_with_random_data():
    """Test zoom with random high-dimensional data"""
    print("\n" + "=" * 60)
    print("Test 1: Zoom with Random Data (50 points, 50D)")
    print("=" * 60)
    
    try:
        # Generate random high-dimensional vectors
        n_samples = 50
        n_features = 50
        vectors = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # Generate initial 2D coordinates
        initial_embedding = np.random.randn(n_samples, 2).astype(np.float32)
        
        # Encode to Base64
        vectors_b64 = _numpy_to_b64(vectors)
        initial_b64 = _numpy_to_b64(initial_embedding)
        
        request_body = {
            "vectors_b64": vectors_b64,
            "initial_embedding_b64": initial_b64,
            "n_components": 2,
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "euclidean",
            "n_epochs": 200
        }
        
        print(f"Sending request: {n_samples} points, {n_features} dimensions (with init)...")
        print(f"Request size: vectors_b64={len(vectors_b64)} chars, init_b64={len(initial_b64)} chars")
        
        response = requests.post(ZOOM_ENDPOINT, json=request_body, timeout=120)
        print(f"Status Code: {response.status_code}")
        
        # Debug: print raw response
        try:
            result = response.json()
        except json.JSONDecodeError as je:
            print(f"✗ JSON Decode Error: {je}")
            print(f"Response text: {response.text[:500]}")
            return False
        
        status = result.get('status')
        print(f"Status: {status}")
        
        if status == 'success':
            coords_b64 = result.get('coordinates')
            shape = result.get('shape')
            
            if coords_b64:
                coords = _b64_to_numpy(coords_b64)
                print(f"✓ Success! Received coordinates: {coords.shape}")
                print(f"  Coordinates range:")
                print(f"    X: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
                print(f"    Y: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
                return True
            else:
                print("✗ No coordinates in response")
                print(f"Response: {json.dumps(result, indent=2)[:500]}")
                return False
        else:
            print(f"✗ Error: {result.get('message')}")
            return False
    
    except requests.exceptions.ConnectionError as ce:
        print(f"✗ Connection Error: {ce}")
        print("  GPU server not running on port 8001?")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_zoom_without_initial_embedding():
    """Test zoom without initial embedding"""
    print("\n" + "=" * 60)
    print("Test 2: Zoom WITHOUT Initial Embedding (100 points, 50D)")
    print("=" * 60)
    
    try:
        # Generate random high-dimensional vectors only
        n_samples = 100
        n_features = 50
        vectors = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # Encode to Base64
        vectors_b64 = _numpy_to_b64(vectors)
        
        request_body = {
            "vectors_b64": vectors_b64,
            "n_components": 2,
            "n_neighbors": 20,
            "min_dist": 0.05,
            "metric": "euclidean",
            "n_epochs": 300
        }
        
        print(f"Sending request: {n_samples} points, {n_features} dimensions (no init)...")
        print(f"Request size: vectors_b64={len(vectors_b64)} chars")
        
        response = requests.post(ZOOM_ENDPOINT, json=request_body, timeout=120)
        print(f"Status Code: {response.status_code}")
        
        try:
            result = response.json()
        except json.JSONDecodeError as je:
            print(f"✗ JSON Decode Error: {je}")
            print(f"Response text: {response.text[:500]}")
            return False
        
        status = result.get('status')
        print(f"Status: {status}")
        
        if status == 'success':
            coords_b64 = result.get('coordinates')
            
            if coords_b64:
                coords = _b64_to_numpy(coords_b64)
                print(f"✓ Success! Received coordinates: {coords.shape}")
                print(f"  Coordinates range:")
                print(f"    X: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
                print(f"    Y: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
                return True
            else:
                print("✗ No coordinates in response")
                print(f"Response: {json.dumps(result, indent=2)[:500]}")
                return False
        else:
            print(f"✗ Error: {result.get('message')}")
            return False
    
    except requests.exceptions.ConnectionError as ce:
        print(f"✗ Connection Error: {ce}")
        print("  GPU server not running on port 8001?")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_zoom_with_real_data():
    """Test zoom with real data from d3_data_manager"""
    print("\n" + "=" * 60)
    print("Test 3: Zoom with Real Data from d3_data_manager")
    print("=" * 60)
    
    if not DATA_AVAILABLE:
        print("⚠ Skipped: d3_data_manager not available")
        return False
    
    try:
        # Get data from manager
        print("Loading data from d3_data_manager...")
        initial_data = data_manager.get_initial_data()
        
        # Extract vectors and embedding
        vectors = initial_data.get('X')
        embedding = initial_data.get('Y')
        
        if vectors is None or embedding is None:
            print("✗ Could not get vectors or embedding from d3_data_manager")
            return False
        
        # Use subset for testing
        n_subset = min(200, vectors.shape[0])
        vectors_subset = vectors[:n_subset].astype(np.float32)
        embedding_subset = embedding[:n_subset].astype(np.float32)
        
        print(f"Using {n_subset} points from {vectors.shape[0]} total")
        print(f"  Vectors shape: {vectors_subset.shape}")
        print(f"  Embedding shape: {embedding_subset.shape}")
        
        # Encode to Base64
        vectors_b64 = _numpy_to_b64(vectors_subset)
        embedding_b64 = _numpy_to_b64(embedding_subset)
        
        request_body = {
            "vectors_b64": vectors_b64,
            "initial_embedding_b64": embedding_b64,
            "n_components": 2,
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "euclidean",
            "n_epochs": 200
        }
        
        print("Sending request to GPU server...")
        print(f"Request size: vectors_b64={len(vectors_b64)} chars, init_b64={len(embedding_b64)} chars")
        
        response = requests.post(ZOOM_ENDPOINT, json=request_body, timeout=120)
        print(f"Status Code: {response.status_code}")
        
        try:
            result = response.json()
        except json.JSONDecodeError as je:
            print(f"✗ JSON Decode Error: {je}")
            print(f"Response text: {response.text[:500]}")
            return False
        
        status = result.get('status')
        print(f"Status: {status}")
        
        if status == 'success':
            coords_b64 = result.get('coordinates')
            
            if coords_b64:
                coords = _b64_to_numpy(coords_b64)
                print(f"✓ Success! Received coordinates: {coords.shape}")
                print(f"  Coordinates range:")
                print(f"    X: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
                print(f"    Y: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
                return True
            else:
                print("✗ No coordinates in response")
                print(f"Response: {json.dumps(result, indent=2)[:500]}")
                return False
        else:
            print(f"✗ Error: {result.get('message')}")
            return False
    
    except requests.exceptions.ConnectionError as ce:
        print(f"✗ Connection Error: {ce}")
        print("  GPU server not running on port 8001?")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dimension_mismatch_error():
    """Test error handling for dimension mismatch"""
    print("\n" + "=" * 60)
    print("Test 4: Error Handling - Dimension Mismatch")
    print("=" * 60)
    
    try:
        # Create mismatched dimensions
        vectors = np.random.randn(50, 50).astype(np.float32)
        initial_embedding = np.random.randn(100, 2).astype(np.float32)  # Wrong count!
        
        vectors_b64 = _numpy_to_b64(vectors)
        embedding_b64 = _numpy_to_b64(initial_embedding)
        
        request_body = {
            "vectors_b64": vectors_b64,
            "initial_embedding_b64": embedding_b64,
            "n_components": 2
        }
        
        print("Sending request with mismatched dimensions...")
        response = requests.post(ZOOM_ENDPOINT, json=request_body, timeout=60)
        
        result = response.json()
        status = result.get('status')
        print(f"Status: {status}")
        
        if status == 'error':
            print(f"✓ Error correctly caught: {result.get('message')}")
            return True
        else:
            print(f"✗ Expected error but got success")
            return False
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False



def test_zoom_api():
    """Run all zoom API tests"""
    
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  GPU UMAP Zoom Server Test Suite".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Test 0: Health Check
    gpu_ok = test_health_check()
    if not gpu_ok:
        print("\n⚠ GPU server is not running or unreachable")
        print("  Please start it with:")
        print("    cd src/d3-app/src/backend")
        print("    conda activate rapids-25.10")
        print("    python gpu_umap_server.py")
        return
    
    # Test Info
    test_server_info()
    
    # Run tests
    results = []
    results.append(("Random Data with Init (50 points)", test_zoom_with_random_data()))
    results.append(("No Initial Embedding (100 points)", test_zoom_without_initial_embedding()))
    results.append(("Real Data from d3_data_manager", test_zoom_with_real_data()))
    results.append(("Dimension Mismatch Error", test_dimension_mismatch_error()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    print("=" * 60)


if __name__ == "__main__":
    test_zoom_api()
