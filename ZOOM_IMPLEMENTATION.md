# Zoom Feature Implementation Guide

## Overview

This document describes the GPU-accelerated zoom feature that allows users to focus on a subset of data points and redraw the 2D projection with better spatial separation while preserving the "mental map" (current spatial ordering).

## Architecture

### Backend Flow

```
User selects points (Frontend)
    ↓
POST /api/zoom/redraw with point_ids
    ↓
d3_data_manager.zoom_redraw()
    ├── Load high-dimensional vectors for selected points (from vector.npy)
    ├── Get current 2D coordinates as initial positions (from _embedding2d)
    ├── Transfer to GPU (CuPy)
    ├── Run GPU UMAP with init parameter for initial positions
    ├── Transfer results back to CPU
    ├── Encode to Base64
    └── Return coordinates
    ↓
Response: Base64-encoded (N, 2) coordinates array
    ↓
Frontend decodes and updates DR visualization
```

### Key Components

#### 1. Backend Implementation (`d3_data_manager.py`)

**New Methods:**
- `_b64_to_numpy(data_b64)` - Static method to decode Base64 to NumPy
- `_numpy_to_b64(array)` - Static method to encode NumPy to Base64
- `zoom_redraw(point_ids, dr_method, ...)` - Main zoom computation method

**GPU UMAP Parameters:**
- `init`: Base64-decoded initial embedding (current 2D coordinates)
- `n_neighbors`: Controls local structure preservation (default: 15)
- `min_dist`: Minimum distance between points (default: 0.1)
- `n_epochs`: Number of optimization iterations (default: 200)

#### 2. FastAPI Endpoint (`main_d3.py`)

**POST /api/zoom/redraw**

Request:
```json
{
  "point_ids": [0, 1, 2, 3, ...],
  "dr_method": "umap",
  "n_neighbors": 15,
  "min_dist": 0.1,
  "n_epochs": 200
}
```

Response (Success):
```json
{
  "status": "success",
  "coordinates": "base64-encoded array",
  "shape": [N, 2],
  "point_ids": [0, 1, 2, 3, ...]
}
```

Response (Error):
```json
{
  "status": "error",
  "message": "Error description"
}
```

#### 3. Frontend Integration

**State Management** (useAppStore.tsx):
- `zoomTargetPoints`: Set of selected point indices
- `zoomTargetClusters`: Set of selected cluster indices  
- `isZoomActive`: Boolean flag indicating zoom computation in progress
- `setZoomTarget(pointIds?, clusterIds?)`: Set zoom targets
- `getZoomTargetPoints(data)`: Get flattened point IDs with priority logic
- `setZoomActive(isActive)`: Manage loading state

**API Integration** (Fetch.ts):
- New `fetchZoomRedraw(pointIds)` function
- Calls `/api/zoom/redraw` with proper error handling
- Returns decoded coordinates or throws error

**UI Component** (DRVisualization.tsx):
- "Zoom In" button (shows selected point count)
- Loading indicator during API call
- Updates DR coordinates on response
- Error notification if API call fails

## Base64 Encoding/Decoding Pattern

### Backend (Python)

```python
import base64
import io
import numpy as np

# Encoding: NumPy array → Base64 string
def _numpy_to_b64(array: np.ndarray) -> str:
    buff = io.BytesIO()
    np.save(buff, array, allow_pickle=False)
    return base64.b64encode(buff.getvalue()).decode('utf-8')

# Decoding: Base64 string → NumPy array
def _b64_to_numpy(data_b64: str) -> np.ndarray:
    decoded = base64.b64decode(data_b64)
    return np.load(io.BytesIO(decoded))
```

### Frontend (TypeScript)

```typescript
import * as base64 from 'base64-js';

// Encoding: Float32Array → Base64 string
function arrayToBase64(arr: Float32Array): string {
  const bytes = new Uint8Array(arr.buffer);
  return base64.fromByteArray(bytes);
}

// Decoding: Base64 string → Float32Array
function base64ToArray(b64: string): Float32Array {
  const bytes = base64.toByteArray(b64);
  return new Float32Array(bytes.buffer);
}
```

## Mental Map Preservation

The key to preserving spatial ordering is using `init` parameter in GPU UMAP:

```python
umap_model = cuMLUMAP(
    n_components=2,
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    metric="euclidean",
    random_state=42,
    init=init_gpu,  # Current 2D coordinates as initial positions
    n_epochs=n_epochs
)
```

When `init` is provided:
1. UMAP uses these coordinates as the starting point
2. Optimization keeps nearby points close to their initial positions
3. Global structure emerges with better separation
4. Spatial ordering from original projection is largely preserved

## Error Handling

### Backend Validation
- Point ID range checking
- File existence validation (vector.npy)
- GPU availability check
- Vector dimension matching

### API Error Responses
- 400: Invalid request (e.g., out-of-range point IDs)
- 500: Server error (GPU unavailable, file missing, computation failed)

### Frontend Error Handling
- Display error notification to user
- Clear zoom state on error
- Log error details to console
- Provide retry option

## Performance Considerations

### Time Complexity
- Data loading: O(N * D) where N = number of points, D = dimensions
- GPU UMAP: Typically 5-30 seconds depending on N and n_epochs
- Total response time: Usually under 30 seconds for N < 1000

### Memory Requirements
- GPU VRAM: Approximately 4 * N * D bytes for GPU arrays
- Estimated: ~1.2 GB for 10000 points in 300D space
- CPU RAM: Additional ~1 GB for intermediate data transfers

### Optimization Tips
1. Reduce `n_epochs` for faster results (e.g., 100-150 instead of 200)
2. Increase `n_neighbors` slightly for more global structure preservation
3. Use smaller subsets (< 1000 points) for real-time responsiveness
4. Consider batch processing for multiple zoom levels

## Testing

### Manual Testing
1. Start backend: `uvicorn main_d3:app --host 0.0.0.0 --port 8000`
2. Run test script: `python test_zoom_api.py`
3. Tests cover: small subset, large subset, invalid IDs, unsupported methods

### Frontend Testing
1. Load application with real data
2. Select points using lasso or brush
3. Click "Zoom In" button
4. Verify new coordinates appear
5. Check for visual mental map preservation
6. Test with various selection sizes (10, 100, 500, 1000+ points)

## Future Enhancements

1. **Multi-level Zoom**: Store zoom history and allow "zoom out" navigation
2. **Custom Parameters**: Expose UMAP parameters to UI for user tuning
3. **Parallel Computation**: Support multiple zoom requests simultaneously
4. **Progressive Rendering**: Stream results as GPU finishes computation
5. **Alternative DR Methods**: Add t-SNE, PCA support with same interface
6. **Cluster-based Zoom**: Zoom to selected clusters directly

## Dependencies

### Python (Backend)
- `cupy`: GPU array support
- `cuml`: GPU UMAP implementation
- `fastapi`: Web framework
- `pydantic`: Request/response validation
- `numpy`: Array operations

### TypeScript (Frontend)
- `react`: Component framework
- `d3`: Visualization library
- `base64-js`: Base64 encoding/decoding (or use native Uint8Array)

## Installation Requirements

### GPU Setup
```bash
# Install NVIDIA dependencies
# Requirements: CUDA 11.0+, cuDNN 8.0+, NVIDIA drivers 450+

# Install Python packages (Conda recommended)
conda create -n drzoom -c rapids -c conda-forge cuml cupy cudatoolkit=11.2
conda activate drzoom
pip install fastapi uvicorn pydantic numpy
```

### Fallback (CPU-only)
If GPU is unavailable, implement CPU fallback using scikit-learn UMAP:
```python
from umap import UMAP as CPUUmap
if not HAS_GPU:
    umap_model = CPUUmap(init=current_coords, ...)
```

## Configuration

Edit in `d3_data_manager.py`:
```python
"umap_zoom": {
    "default_n_neighbors": 15,
    "default_min_dist": 0.1,
    "default_n_epochs": 200,
    "max_points_allowed": 10000
}
```

## Debugging Tips

1. **Check GPU availability**: `python -c "import cupy; import cuml; print('GPU ready')"`
2. **Verify vector file**: `python -c "import numpy as np; v = np.load('vector.npy'); print(v.shape)"`
3. **Test Base64 round-trip**: `python -c "from d3_data_manager import D3DataManager; ... test encode/decode"`
4. **Monitor GPU memory**: `nvidia-smi --loop=1` in separate terminal
5. **Check API response**: Use Postman or curl to test endpoint directly

## References

- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [cuML UMAP](https://docs.rapids.ai/api/cuml/stable/api.html#umap)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Base64 in Python/JS](https://en.wikipedia.org/wiki/Base64)
