# Nearby Highlighting Design & Implementation Options

## Current Implementation (Stroke-based)

**Status**: ✅ Completed and working

- Method: Red stroke outline on nearby cluster points
- Location: `applyNearbyStroke()` callback (lines 60-66)
- Trigger: When `selection.nearbyClusterIds` changes
- Applies to: All circles with cluster ID in nearby set
- Attributes:
  - `stroke: '#FF0000'` for nearby, `'none'` otherwise
  - `stroke-width: 0.5` for nearby, `0` otherwise

---

## Option: Opacity-based Highlighting

### Design Considerations

The opacity approach would make nearby points stand out by **increasing brightness/reducing dimming** rather than adding visual borders.

**Key Challenge**: Opacity is already controlled by the existing highlight system

Current opacity flow:
```
determinePointHighlight()
  ↓
getElementStyle(highlight, anySelectionActive)
  ├─ If highlight active → opacity = 1.0 (full)
  ├─ If anySelectionActive (other things selected) → opacity = 0.2 (dimmed)
  └─ Else → opacity = 1.0 (normal)
```

### Implementation Strategy

**Option A: Integrate Nearby into Main Opacity Calculation**

Modify `getElementStyle()` in `src/types/color.ts`:

1. Add `nearbyClusterIds` as parameter to `getElementStyle()`
2. Update signature:
   ```typescript
   export function getElementStyle(
     highlight: HighlightState,
     anySelectionActive: boolean,
     isNearby: boolean  // NEW
   ): SVGStyle
   ```
3. Opacity logic becomes:
   ```typescript
   opacity: (() => {
     if (isNearby) return 1.0;  // Always fully opaque for nearby
     if (highlight.isDRSelected || ...) return 1.0;
     if (anySelectionActive) return 0.2;
     return 1.0;
   })()
   ```

**Affected locations**:
- `DRVisualization.tsx` line 258 (circle rendering):
  ```tsx
  .attr('opacity', (d) => {
    const highlight = determinePointHighlight(...);
    const style = getElementStyle(
      highlight,
      anySelectionActive,
      selection.nearbyClusterIds.has(d.c)  // NEW param
    );
    return style.opacity || 1.0;
  })
  ```
- `DRVisualization.tsx` line 306 (search results opacity - same pattern)
- `color.ts` line 104 (function definition)
- All other `getElementStyle()` calls (search in codebase)

**Option B: Keep Separate, Override via `applyNearbyOpacity`**

Keep nearby logic independent (simpler but less integrated):

1. Rename `applyNearbyStroke()` → `applyNearbyOpacity()`
2. Replace `.attr('stroke'...)` with `.attr('opacity'...)`
3. Logic:
   ```typescript
   const applyNearbyOpacity = useCallback(() => {
     if (!circlesRef.current) return;
     circlesRef.current
       .attr('opacity', (d: Point) => {
         if (selection.nearbyClusterIds.has(d.c)) {
           return 1.0;  // Brighten nearby
         }
         // Fallback to original logic from rendering
         const highlight = determinePointHighlight(d.i, d, ...);
         const style = getElementStyle(highlight, anySelectionActive);
         return style.opacity || 1.0;
       });
   }, [selection.nearbyClusterIds, ...deps]);
   ```

**Drawback**: `anySelectionActive` is in outer scope; must be a closure capture or passed differently

---

## Recommendation

**Use Option A** (Integrate into `getElementStyle`):
- ✅ Single source of truth for opacity
- ✅ Cleaner dependency management
- ✅ Follows existing pattern
- ⚠️ Requires updating all `getElementStyle()` call sites

### Migration Steps

1. Update `color.ts` function signature and add `isNearby` parameter handling
2. Search codebase for all `getElementStyle(` calls:
   ```
   grep -r "getElementStyle(" src/
   ```
3. Update each call to pass `selection.nearbyClusterIds.has(d.c)` or equivalent
4. Replace `applyNearbyStroke()` with `applyNearbyOpacity()`:
   - Remove stroke logic
   - Add opacity update with same selection-based calculation
   - Keep as separate callback (for now) or merge into main rendering
5. Test that selection dimming still works and nearby points brighten correctly

---

## Testing Checklist

- [ ] Nearby points are fully opaque (1.0)
- [ ] Non-nearby points obey normal dimming rules (0.2 when something else selected)
- [ ] Highlight colors still visible
- [ ] No visual overlap with existing opacity transitions
