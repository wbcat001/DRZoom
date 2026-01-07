# Zoom Feature - Implementation Summary

## ✅ COMPLETED: Backend Implementation

### What was implemented:

#### 1. **d3_data_manager.py** - GPU UMAP Zoom Engine
- **Added imports**: `base64`, `io`, `cupy` (GPU arrays), `cuml.manifold.UMAP`
- **Base64 utilities**: `_numpy_to_b64()`, `_b64_to_numpy()`
- **Main method**: `zoom_redraw(point_ids, dr_method, n_neighbors, min_dist, n_epochs)`

**How it works:**
1. Loads high-dimensional vectors from `vector.npy` for selected points
2. Gets current 2D coordinates from `_embedding2d` as initial positions
3. Transfers data to GPU using CuPy
4. Runs GPU UMAP with `init` parameter to preserve spatial ordering
5. Returns coordinates as Base64-encoded NumPy array

**Mental Map Preservation:** By using `init=init_gpu` in UMAP, the algorithm:
- Starts from current 2D positions
- Preserves nearby relationships
- Allows points to spread out while maintaining relative ordering

#### 2. **main_d3.py** - FastAPI Endpoint
- **Request model**: `ZoomRedrawRequest(point_ids, dr_method, n_neighbors, min_dist, n_epochs)`
- **Response model**: `ZoomRedrawResponse(status, coordinates, shape, point_ids, message)`
- **Endpoint**: `POST /api/zoom/redraw`

**Endpoint behavior:**
- Accepts list of point IDs to zoom into
- Calls `data_manager.zoom_redraw()`
- Returns Base64-encoded coordinates or error message
- Full error handling for GPU unavailability, invalid point IDs, etc.

### Files Modified:
- [d:\Work_Program\DRZoom\src\d3-app\src\backend\services\d3_data_manager.py](d:\Work_Program\DRZoom\src\d3-app\src\backend\services\d3_data_manager.py)
- [d:\Work_Program\DRZoom\src\d3-app\src\backend\main_d3.py](d:\Work_Program\DRZoom\src\d3-app\src\backend\main_d3.py)

### Test Script:
- [d:\Work_Program\DRZoom\src\d3-app\src\backend\test_zoom_api.py](d:\Work_Program\DRZoom\src\d3-app\src\backend\test_zoom_api.py)

**To test:**
```bash
cd src/d3-app/src/backend
uvicorn main_d3:app --host 0.0.0.0 --port 8000
# In another terminal:
python test_zoom_api.py
```

---

## 🔄 PENDING: Frontend Implementation

### Step 1: Create API Client Function
**File**: `src/components/Fetch.ts`

```typescript
export async function fetchZoomRedraw(pointIds: number[]): Promise<Float32Array> {
  const response = await fetch('/api/zoom/redraw', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      point_ids: pointIds,
      dr_method: 'umap',
      n_neighbors: 15,
      min_dist: 0.1,
      n_epochs: 200
    })
  });
  
  if (!response.ok) throw new Error(await response.text());
  
  const data = await response.json();
  if (data.status !== 'success') throw new Error(data.message);
  
  // Decode Base64 to Float32Array
  const binaryString = atob(data.coordinates);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}
```

### Step 2: Add UI Button
**File**: `src/components/DRVisualization.tsx`

Add button to header (around line 100):
```tsx
const { getZoomTargetPoints, setZoomActive } = useSelection();
const [isZoomLoading, setIsZoomLoading] = useState(false);

const handleZoomIn = async () => {
  const pointIds = getZoomTargetPoints(filteredData);
  if (pointIds.length === 0) {
    alert('Please select points first');
    return;
  }
  
  setIsZoomLoading(true);
  setZoomActive(true);
  
  try {
    const newCoords = await fetchZoomRedraw(pointIds);
    
    // Update DR points with new coordinates
    setDRPoints(prevPoints => {
      const updated = [...prevPoints];
      for (let i = 0; i < pointIds.length; i++) {
        const pointId = pointIds[i];
        updated[pointId] = {
          ...updated[pointId],
          x: newCoords[i * 2],
          y: newCoords[i * 2 + 1]
        };
      }
      return updated;
    });
  } catch (error) {
    alert(`Zoom failed: ${error}`);
  } finally {
    setIsZoomLoading(false);
    setZoomActive(false);
  }
};

// In JSX header:
<button 
  onClick={handleZoomIn} 
  disabled={isZoomLoading}
  style={{ marginLeft: '10px' }}
>
  {isZoomLoading ? 'Zooming...' : `Zoom In (${getZoomTargetPoints(filteredData).length})`}
</button>
```

### Step 3: Test Integration
1. Open application
2. Select points using lasso or brush
3. Click "Zoom In (N)" button
4. Observe new coordinates appear
5. Check mental map preservation visually

---

## 🎯 API Usage Example

### Request:
```bash
curl -X POST http://localhost:8000/api/zoom/redraw \
  -H "Content-Type: application/json" \
  -d '{
    "point_ids": [0, 1, 2, 3, 4],
    "dr_method": "umap",
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_epochs": 200
  }'
```

### Response (Success):
```json
{
  "status": "success",
  "coordinates": "gAN9cQAoWAYAAABdZXBvY2hzcQFLPVgHAAAAX3NpZ1ECAldEAAAAL...",
  "shape": [5, 2],
  "point_ids": [0, 1, 2, 3, 4]
}
```

### Response (Error):
```json
{
  "status": "error",
  "message": "Point IDs out of range [0, 99999]",
  "coordinates": null,
  "shape": null,
  "point_ids": null
}
```

---

## 📊 Data Flow Diagram

```
User Interface (DRVisualization.tsx)
        ↓
  [Zoom In Button]
        ↓
fetchZoomRedraw(pointIds)
        ↓
POST /api/zoom/redraw
        ↓
Backend: main_d3.py::zoom_redraw endpoint
        ↓
d3_data_manager.py::zoom_redraw()
├─ Load vectors: vector.npy[pointIds]
├─ Get init coords: _embedding2d[pointIds]
├─ GPU Transfer: CuPy arrays
├─ Run GPU UMAP: init=initial_positions
├─ GPU Transfer back: to NumPy
└─ Encode: to Base64
        ↓
Response: Base64 coordinates
        ↓
Frontend: Decode Base64 → Float32Array
        ↓
Update: setDRPoints with new coordinates
        ↓
Re-render: D3.js visualization with new positions
```

---

## 🔧 Configuration

### Backend Parameters (in main_d3.py request):
- `n_neighbors`: 15 (controls local structure)
- `min_dist`: 0.1 (minimum point separation)
- `n_epochs`: 200 (optimization iterations)

**Tuning suggestions:**
- Fewer points (< 100): increase n_neighbors to 20-25
- Many points (> 1000): reduce n_epochs to 100-150
- Want tighter clusters: increase min_dist to 0.2-0.3
- Want looser clusters: decrease min_dist to 0.05

---

## 🐛 Troubleshooting

### "GPU UMAP not available"
```bash
pip install cupy cuml  # Install GPU libraries
# or with conda:
conda install -c rapids -c conda-forge cuml cupy
```

### "Vector file not found"
Make sure `vector.npy` exists in data directory:
```bash
ls -lah src/d3-app/data/vector.npy
```

### "Point IDs out of range"
Check that point IDs are within valid range (0 to N-1) where N is total points

### API timeout
Increase timeout in fetch:
```typescript
response = await fetch(url, { timeout: 120000 }); // 120 seconds
```

---

## 📚 Documentation

See [ZOOM_IMPLEMENTATION.md](ZOOM_IMPLEMENTATION.md) for:
- Complete architecture details
- Base64 encoding patterns
- Performance considerations
- Advanced customization
- Future enhancements
