# Code Changes Summary - Zoom Feature Implementation

## Files Modified: 2
## Files Created: 1
## Total Lines Added: ~220

---

## 1. d3_data_manager.py

### Location: `src/d3-app/src/backend/services/d3_data_manager.py`

### Change 1: New Imports (Lines 1-19)

```python
# ADDED:
import base64
import io

try:
    import cupy as cp
    from cuml.manifold import UMAP as cuMLUMAP
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
```

**Purpose**: Import GPU libraries and Base64 utilities

### Change 2: New Methods in D3DataManager Class (End of file, before class end)

```python
# ========================================================================
# Zoom Feature: GPU-Accelerated UMAP Redraw with Initial Position Preservation
# ========================================================================

@staticmethod
def _b64_to_numpy(data_b64: str) -> np.ndarray:
    """Decode Base64 string to NumPy array"""
    decoded = base64.b64decode(data_b64)
    return np.load(io.BytesIO(decoded))

@staticmethod
def _numpy_to_b64(array: np.ndarray) -> str:
    """Encode NumPy array to Base64 string"""
    buff = io.BytesIO()
    np.save(buff, array, allow_pickle=False)
    return base64.b64encode(buff.getvalue()).decode('utf-8')

def zoom_redraw(
    self,
    point_ids: List[int],
    dr_method: str = "umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_epochs: int = 200
) -> Dict[str, Any]:
    """
    Redraw 2D projection for selected points using GPU-accelerated UMAP.
    Preserves mental map by using current coordinates as initial positions.
    
    Args:
        point_ids: List of point indices to zoom into
        dr_method: Dimensionality reduction method (currently only 'umap' supported)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        n_epochs: Number of UMAP epochs
        
    Returns:
        Dict with:
            - status: "success" or "error"
            - coordinates: Base64-encoded (N_selected, 2) embedding
            - shape: [N_selected, 2]
            - point_ids: Original point indices for mapping
    """
    if not HAS_GPU:
        return {
            "status": "error",
            "message": "GPU UMAP not available. Install cupy and cuml."
        }
    
    if dr_method != "umap":
        return {
            "status": "error",
            "message": f"Only 'umap' is supported for zoom redraw, got '{dr_method}'"
        }
    
    try:
        # Validate point_ids
        point_ids = [int(p) for p in point_ids]
        if not self._embedding2d is None:
            max_id = self._embedding2d.shape[0] - 1
            if any(p < 0 or p > max_id for p in point_ids):
                return {
                    "status": "error",
                    "message": f"Point IDs out of range [0, {max_id}]"
                }
        
        # Load high-dimensional vectors for selected points
        vectors_file = self.base_path / self.datasets_config["default"]["data_path"] / "vector.npy"
        if not vectors_file.exists():
            return {
                "status": "error",
                "message": f"Vector file not found at {vectors_file}"
            }
        
        all_vectors = np.load(vectors_file)  # (N, D)
        selected_vectors = all_vectors[point_ids]  # (N_selected, D)
        
        print(f"✓ Loaded {len(point_ids)} vectors from high-dimensional space: {selected_vectors.shape}")
        
        # Get current 2D coordinates as initial positions
        current_coords = self._embedding2d[point_ids]  # (N_selected, 2)
        print(f"✓ Extracted current coordinates for initial positions: {current_coords.shape}")
        
        # Transfer to GPU
        vectors_gpu = cp.asarray(selected_vectors, dtype=cp.float32)
        init_gpu = cp.asarray(current_coords, dtype=cp.float32)
        
        print(f"✓ Transferred data to GPU: vectors {vectors_gpu.shape}, init {init_gpu.shape}")
        
        # Create UMAP model with initial positions
        umap_model = cuMLUMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, len(point_ids) - 1),  # Adjust if fewer points
            min_dist=min_dist,
            metric="euclidean",
            random_state=42,
            init=init_gpu,  # Use current positions as mental map reference
            n_epochs=n_epochs,
            verbose=True
        )
        
        # Execute GPU UMAP
        embedding_gpu = umap_model.fit_transform(vectors_gpu)
        cp.cuda.runtime.deviceSynchronize()  # Ensure GPU computation completes
        
        # Transfer back to CPU
        embedding_cpu = cp.asnumpy(embedding_gpu)
        print(f"✓ UMAP computation complete: {embedding_cpu.shape}")
        
        # Encode result to Base64
        embedding_b64 = self._numpy_to_b64(embedding_cpu)
        
        return {
            "status": "success",
            "coordinates": embedding_b64,
            "shape": list(embedding_cpu.shape),
            "point_ids": point_ids
        }
    
    except ValueError as e:
        print(f"ValueError in zoom_redraw: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
    except Exception as e:
        print(f"Error in zoom_redraw: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Internal error during zoom redraw: {str(e)}"
        }
```

**Stats**: 186 lines added

---

## 2. main_d3.py

### Location: `src/d3-app/src/backend/main_d3.py`

### Change 1: New Pydantic Models (After existing models)

```python
class ZoomRedrawRequest(BaseModel):
    """Request model for zoom redraw with GPU UMAP"""
    point_ids: List[int]
    dr_method: str = "umap"
    n_neighbors: int = 15
    min_dist: float = 0.1
    n_epochs: int = 200


class ZoomRedrawResponse(BaseModel):
    """Response model for zoom redraw"""
    status: str  # "success" or "error"
    coordinates: Optional[str] = None  # Base64-encoded (N, 2) array
    shape: Optional[List[int]] = None
    point_ids: Optional[List[int]] = None
    message: Optional[str] = None
```

**Stats**: 16 lines added

### Change 2: New Endpoint (Before error handlers)

```python
@app.post("/api/zoom/redraw", response_model=ZoomRedrawResponse)
async def zoom_redraw(request: ZoomRedrawRequest):
    """
    Redraw 2D projection for selected points using GPU-accelerated UMAP.
    
    Preserves mental map by using current coordinates as initial positions.
    Useful for exploring a subset of points with better separation.
    
    Args:
        request: ZoomRedrawRequest with point_ids and UMAP parameters
        
    Returns:
        ZoomRedrawResponse with Base64-encoded coordinates or error message
    """
    try:
        result = data_manager.zoom_redraw(
            point_ids=request.point_ids,
            dr_method=request.dr_method,
            n_neighbors=request.n_neighbors,
            min_dist=request.min_dist,
            n_epochs=request.n_epochs
        )
        return result
    except Exception as e:
        print(f"Error in zoom_redraw endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
```

**Stats**: 27 lines added

**Total for main_d3.py**: 43 lines added

---

## 3. test_zoom_api.py (NEW FILE)

### Location: `src/d3-app/src/backend/test_zoom_api.py`

Complete test script with 4 test cases:
1. Small subset (50 points)
2. Large subset (200 points)
3. Invalid point IDs
4. Unsupported DR method

**Stats**: 200+ lines

---

## Diff Summary

```
d3_data_manager.py
├─ Line 1-20:   New imports (base64, io, cupy, cuml)
└─ Line ~1100:  New methods (_b64_to_numpy, _numpy_to_b64, zoom_redraw)
    └─ 186 lines added

main_d3.py
├─ Line ~70:    New Pydantic models (ZoomRedrawRequest, ZoomRedrawResponse)
│               └─ 16 lines added
└─ Line ~385:   New endpoint (/api/zoom/redraw)
    └─ 27 lines added

Total: 229 lines added
```

---

## Key Implementation Points

### 1. GPU Check
```python
try:
    import cupy as cp
    from cuml.manifold import UMAP as cuMLUMAP
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
```

### 2. Mental Map Preservation
```python
umap_model = cuMLUMAP(
    ...
    init=init_gpu,  # ← KEY: Use current coordinates as initial positions
    ...
)
```

### 3. Base64 Pattern
```python
# Encode: NumPy → Base64
buff = io.BytesIO()
np.save(buff, array, allow_pickle=False)
b64 = base64.b64encode(buff.getvalue()).decode('utf-8')

# Decode: Base64 → NumPy
decoded = base64.b64decode(data_b64)
array = np.load(io.BytesIO(decoded))
```

### 4. Data Flow
```
User request
    ↓
POST /api/zoom/redraw { point_ids: [...] }
    ↓
FastAPI endpoint
    ↓
data_manager.zoom_redraw()
    ├─ Load vectors
    ├─ Get current coords
    ├─ GPU transfer
    ├─ GPU UMAP
    ├─ CPU transfer
    └─ Base64 encode
    ↓
Response { status: "success", coordinates: "..." }
    ↓
Frontend decodes and updates visualization
```

---

## API Contract

### Request
```json
POST /api/zoom/redraw
{
  "point_ids": [0, 1, 2, ..., N],
  "dr_method": "umap",
  "n_neighbors": 15,
  "min_dist": 0.1,
  "n_epochs": 200
}
```

### Success Response
```json
{
  "status": "success",
  "coordinates": "gAN9cQA...(Base64 encoded)",
  "shape": [100, 2],
  "point_ids": [0, 1, 2, ..., N],
  "message": null
}
```

### Error Response
```json
{
  "status": "error",
  "coordinates": null,
  "shape": null,
  "point_ids": null,
  "message": "Error description"
}
```

---

## Testing Checklist

- [x] Code compiles without GPU-specific errors
- [x] API endpoint accepts requests
- [x] Base64 encoding/decoding works
- [x] Error handling for:
  - [ ] GPU unavailable
  - [ ] Invalid point IDs
  - [ ] Missing vector file
  - [ ] Unsupported DR method
- [ ] Performance acceptable (< 30s for 1000 points)
- [ ] Mental map visually preserved in test
