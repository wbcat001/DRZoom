import React, { useRef, useEffect, useState, useMemo, useCallback, useContext } from 'react';
import * as d3 from 'd3';
import { useSelection, useData, useUIState, useViewConfig, AppContext } from '../../store/useAppStore';
import { Point } from '../../types';
import { determinePointHighlight, isAnySelectionActive, createScaleFactors } from '../../utils';
import { getElementStyle } from '../../types/color';
import { apiClient } from '../../api/client';
import { LassoSelection } from '../../utils/lassoSelection';
import { encodeVectorsToBase64, decodeBase64ToCoordinates } from '../../utils/base64';
import SearchBar from '../SearchBar/SearchBar';
import './DRVisualization.css';

const DRVisualization: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const circlesRef = useRef<any>(null);
  const { selection, selectClusters, selectPoints, setDRSelectedClusters, setNearbyClusterIds } = useSelection();
  const { data, updatePointCoordinates, setZoomMode } = useData();
  const { config } = useViewConfig();
  const { setError } = useUIState();
  const [, setDimensions] = useState({ width: 0, height: 0 });
  const [interactionMode, setInteractionMode] = useState<'point' | 'cluster'>('point');
  const [ignoreNoise, setIgnoreNoise] = useState<boolean>(true);
  const [brushEnabled, setBrushEnabled] = useState<boolean>(false);
  const [lassoEnabled, setLassoEnabled] = useState<boolean>(false);
  const [showHoverWords, setShowHoverWords] = useState<boolean>(true);
  const [showAnnotations, setShowAnnotations] = useState<boolean>(true);
  const [containmentThreshold, setContainmentThreshold] = useState<number>(0.1);
  const [isZoomProcessing, setIsZoomProcessing] = useState<boolean>(false);
  const lassoRef = useRef<LassoSelection | null>(null);
  const prevSelectedPointIdsRef = useRef<string>('');
  const zoomTransformRef = useRef<any>(null);
  
  // Get zoom state from context
  const appContext = useContext(AppContext);
  const isZoomedMode = appContext?.state.isZoomed || false;
  const zoomedPointIds = appContext?.state.zoomedPointIds || [];
  
  // Filter display points based on zoom mode
  const displayData = useMemo(() => {
    console.log('displayData recalculating', {
      isZoomedMode,
      zoomedPointIds: zoomedPointIds.length,
      dataPointsTotal: data.points.length
    });
    if (isZoomedMode && zoomedPointIds.length > 0) {
      const filtered = {
        ...data,
        points: data.points.filter(p => zoomedPointIds.includes(p.i))
      };
      console.log('Zoom mode filtered to', filtered.points.length, 'points');
      return filtered;
    }
    console.log('Normal mode with all data points:', data.points.length);
    return data;
  }, [data, isZoomedMode, zoomedPointIds]);
  const [tooltip, setTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    point?: { id: number; label: string; cluster: number };
  }>({ visible: false, x: 0, y: 0 });

  const margin = { top: 20, right: 20, bottom: 40, left: 40 };

  // Map hovered dendrogram merge index to the actual cluster IDs involved (child1, child2, parent)
  const hoveredClusterIds = useMemo(() => {
    const ids = new Set<number>();
    const hoverIdx = selection.dendrogramHoveredCluster;
    if (hoverIdx === null) return ids;

    const linkageEntry = data.linkageMatrix[hoverIdx];
    if (!linkageEntry) return ids;

    const child1Id = data.clusterIdMap[linkageEntry.child1] ?? linkageEntry.child1;
    const child2Id = data.clusterIdMap[linkageEntry.child2] ?? linkageEntry.child2;
    const parentId = data.clusterIdMap[linkageEntry.parent] ?? linkageEntry.parent;

    ids.add(child1Id);
    ids.add(child2Id);
    ids.add(parentId);

    return ids;
  }, [selection.dendrogramHoveredCluster, data.linkageMatrix, data.clusterIdMap]);

  // Apply red stroke to nearby clusters (called when nearbyClusterIds changes)
  const applyNearbyStroke = useCallback(() => {
    if (!circlesRef.current) return;

    circlesRef.current
      .attr('stroke', (d: Point) => (selection.nearbyClusterIds.has(d.c) ? '#FF0000' : 'none'))
      .attr('stroke-width', (d: Point) => (selection.nearbyClusterIds.has(d.c) ? 0.5 : 0));
  }, [selection.nearbyClusterIds]);

  // Update dimensions on container resize
  useEffect(() => {
    if (!containerRef.current) return;

    const resizeObserver = new ResizeObserver(() => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (rect) {
        setDimensions({
          width: rect.width - margin.left - margin.right,
          height: rect.height - margin.top - margin.bottom
        });
      }
    });

    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, []);

  // Handle point selection change - compute associated clusters locally
  useEffect(() => {
    // Convert Set to sorted string for comparison
    const currentPointIds = Array.from(selection.selectedPointIds).sort((a, b) => a - b).join(',');
    
    // Skip if selection hasn't actually changed
    if (currentPointIds === prevSelectedPointIdsRef.current) {
      return;
    }
    
    prevSelectedPointIdsRef.current = currentPointIds;
    
    if (selection.selectedPointIds.size === 0) {
      setDRSelectedClusters([]);
      return;
    }

    // Get selected points from data
    const selectedPoints = Array.from(selection.selectedPointIds)
      .map(id => data.points.find(p => p.i === id))
      .filter((p): p is Point => p !== undefined && p.c !== -1); // Filter out undefined and noise

    if (selectedPoints.length === 0) {
      setDRSelectedClusters([]);
      return;
    }

    // Count points per cluster
    const clusterCounts = new Map<number, number>();
    selectedPoints.forEach(p => {
      clusterCounts.set(p.c, (clusterCounts.get(p.c) || 0) + 1);
    });

    // Get cluster sizes from metadata
    const clusterSizes = new Map<number, number>();
    Object.entries(data.clusterMetadata).forEach(([idStr, meta]) => {
      const id = Number(idStr);
      if (id !== -1) {
        clusterSizes.set(id, meta.z || 0);
      }
    });

    // Filter by containment ratio
    const selectedClusters: number[] = [];
    clusterCounts.forEach((count, clusterId) => {
      const totalSize = clusterSizes.get(clusterId) || 0;
      if (totalSize > 0) {
        const containmentRatio = count / totalSize;
        if (containmentRatio >= containmentThreshold) {
          selectedClusters.push(clusterId);
        }
      }
    });

    console.log('DR selection computed locally:', {
      totalSelectedPoints: selectedPoints.length,
      uniqueClusters: clusterCounts.size,
      filteredClusters: selectedClusters.length,
      threshold: containmentThreshold
    });

    setDRSelectedClusters(selectedClusters);
  }, [selection.selectedPointIds, data.points, data.clusterMetadata, containmentThreshold, setDRSelectedClusters]);

  // Update circle stroke when nearby clusters change
  useEffect(() => {
    applyNearbyStroke();
  }, [applyNearbyStroke]);

  // Handle zoom redraw for selected points
  const handleZoomRedraw = useCallback(async () => {
    if (selection.selectedPointIds.size === 0) {
      setError('No points selected. Please select points first.');
      return;
    }

    setIsZoomProcessing(true);
    try {
      // Get selected points data
      const selectedPoints = data.points.filter(p => selection.selectedPointIds.has(p.i));

      if (selectedPoints.length === 0) {
        throw new Error('No valid points found in selection');
      }

      // Extract high-dimensional vectors and current 2D coordinates
      let vectors = selectedPoints.map(p => (p as any).v).filter((v: any) => Array.isArray(v) && v.length > 0);
      const currentCoords = selectedPoints.map(p => [p.x, p.y]);

      // If vectors are not present on points, fetch from backend lazily
      if (vectors.length !== selectedPoints.length) {
        const datasetId = config.currentDataset || 'default';
        const vectorResponse = await apiClient.fetchPointVectors({
          point_ids: selectedPoints.map(p => p.i),
          dataset: datasetId
        });

        if (!vectorResponse.success || !vectorResponse.vectors) {
          throw new Error(vectorResponse.message || 'Failed to fetch high-dimensional vectors');
        }

        vectors = vectorResponse.vectors;
      }

      console.log(`Zoom redraw: ${selectedPoints.length} points`, {
        vectorDim: vectors[0]?.length ?? 0,
        coordsSample: currentCoords.slice(0, 3)
      });

      // Encode to Base64
      const vectors_b64 = encodeVectorsToBase64(vectors);
      const initial_embedding_b64 = encodeVectorsToBase64(currentCoords);

      // Call UMAP server
      const response = await apiClient.zoomRedraw({
        vectors_b64,
        initial_embedding_b64,
        n_components: 2,
        n_neighbors: Math.min(15, selectedPoints.length - 1),
        min_dist: 0.1,
        metric: 'euclidean',
        n_epochs: 200
      });

      if (response.status === 'success' && response.coordinates) {
        // Decode new coordinates
        const newCoords = decodeBase64ToCoordinates(
          response.coordinates,
          selectedPoints.length,
          2
        );

        console.log('Zoom redraw success:', {
          newCoordsSample: newCoords.slice(0, 3),
          shape: response.shape
        });

          // Update data store with new coordinates
          const updates = selectedPoints.map((point, idx) => ({
            pointId: point.i,
            x: newCoords[idx][0],
            y: newCoords[idx][1]
          }));

          updatePointCoordinates(updates);

          // Enter zoom mode - display only selected points
          setZoomMode(true, Array.from(selection.selectedPointIds));

          // Reset zoom/pan so updated points stay in view
          zoomTransformRef.current = null;
          
          // Force SVG zoom state reset by creating identity transform
          if (svgRef.current) {
            const svg = d3.select(svgRef.current);
            const zoom = d3.zoom<SVGSVGElement, unknown>()
              .on('zoom', () => {/* noop for reset */});
            svg.call((zoom.transform as any), d3.zoomIdentity);
            console.log('Reset SVG zoom transform to identity');
          }
          
          console.log(`Successfully updated ${updates.length} point coordinates`);
      } else {
        throw new Error(response.message || 'Zoom redraw failed');
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      console.error('Zoom redraw error:', errorMsg);
      setError(`Zoom redraw failed: ${errorMsg}`);
    } finally {
      setIsZoomProcessing(false);
    }
    }, [selection.selectedPointIds, data.points, updatePointCoordinates, setError, config.currentDataset]);

  // Main D3 rendering logic
  useEffect(() => {
    console.log('D3 rendering triggered', {
      hasRef: !!svgRef.current,
      hasContainer: !!containerRef.current,
      pointsCount: displayData.points.length,
      isZoomedMode,
      zoomedPointIds: zoomedPointIds.length
    });

    if (!svgRef.current || !containerRef.current || displayData.points.length === 0) {
      console.log('Early return from D3 rendering');
      return;
    }

    const svg = d3.select(svgRef.current);
    const container = containerRef.current;

    const rect = container.getBoundingClientRect();
    const width = rect.width - margin.left - margin.right;
    const height = rect.height - margin.top - margin.bottom;

    // Set SVG dimensions
    svg
      .attr('width', rect.width)
      .attr('height', rect.height);

    // Remove existing content
    svg.selectAll('g').remove();

    // Create scales - always recalculate to handle coordinate updates from zoom redraw
    console.log('Recalculating scales for data points (count=%d)', displayData.points.length);
    const scaleFactors = createScaleFactors(displayData.points, width, height, 0);
    
    // Debug: log scale domain and range
    const xDomain = (scaleFactors.xScale.domain ? scaleFactors.xScale.domain() : 'no domain');
    const yDomain = (scaleFactors.yScale.domain ? scaleFactors.yScale.domain() : 'no domain');
    console.log('Scale domains - X:', xDomain, 'Y:', yDomain);

    // Create zoom root and plot group (keep margin separate from zoom transform)
    const zoomRoot = svg.append('g');
    const g = zoomRoot.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Reapply stored zoom/pan transform - IMPORTANT: preserve k/x/y values without margin offset
    if (zoomTransformRef.current) {
      const currentTransform = zoomTransformRef.current;
      // Apply zoom scale and pan without margin - the margin is already in the inner g
      zoomRoot.attr('transform', `translate(${currentTransform.x},${currentTransform.y}) scale(${currentTransform.k})`);
    }

    // Add background
    g.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', '#f8f9fa')
      .attr('pointer-events', 'all');

    // Color scale for clusters
    const clusterIds = Array.from(new Set(displayData.points.map((p) => p.c)));
    const colorScale = d3.scaleOrdinal()
      .domain(clusterIds.map((id) => id.toString()))
      .range(d3.schemeCategory10);

    // Check if any selection is active
    const anySelectionActive = isAnySelectionActive(
      selection.selectedClusterIds,
      selection.selectedPointIds,
      selection.heatmapClickedClusters,
      hoveredClusterIds.size > 0 ? 1 : null
    );

    // Draw points
    const circles = g.selectAll('.data-point')
      .data(displayData.points, (d: any) => d.i)
      .join(
        (enter) =>
          enter
            .append('circle')
            .attr('class', 'data-point'),
        (update) => update,
        (exit) => exit.remove()
      )
      .attr('cx', (d) => scaleFactors.xScale(d.x))
      .attr('cy', (d) => scaleFactors.yScale(d.y))
      .attr('r', 1)
      .attr('fill', function (d: any) {
        const dendroHoverForPoint = hoveredClusterIds.has(d.c) ? d.c : null;
        const highlight = determinePointHighlight(
          d.i,
          d,
          selection.selectedClusterIds,
          selection.selectedPointIds,
          selection.heatmapClickedClusters,
          dendroHoverForPoint,
          selection.searchResultPointIds
        );
        const anyActive = isAnySelectionActive(
          selection.selectedClusterIds,
          selection.selectedPointIds,
          selection.heatmapClickedClusters,
          hoveredClusterIds.size > 0 ? 1 : null,
          selection.searchResultPointIds
        );
        const style = getElementStyle(highlight, anyActive);
        
        // Color priority: similarity-based > highlight style > cluster scale
        if (d.color) {
          return d.color as string;
        }
        if (style.fill) {
          return style.fill;
        }
        return colorScale(d.c.toString()) as string;
      } as any)
      .attr('opacity', (d) => {
        const dendroHoverForPoint = hoveredClusterIds.has(d.c) ? d.c : null;
        const highlight = determinePointHighlight(
          d.i,
          d,
          selection.selectedClusterIds,
          selection.selectedPointIds,
          selection.heatmapClickedClusters,
          dendroHoverForPoint,
          selection.searchResultPointIds
        );
        const style = getElementStyle(highlight, anySelectionActive);
        return style.opacity || 1.0;
      })
      .attr('stroke', 'none')
      .attr('stroke-width', 0)
      .style('cursor', (d: Point) => (ignoreNoise && d.c === -1 ? 'default' : 'pointer'))
      .on('mouseover', function (event, d: Point) {
        if (ignoreNoise && d.c === -1) return;
        const rect = container.getBoundingClientRect();
        setTooltip({
          visible: true,
          x: event.clientX - rect.left + 12,
          y: event.clientY - rect.top + 12,
          point: { id: d.i, label: d.l, cluster: d.c }
        });
        d3.select(this).attr('r', 5).attr('stroke', '#FF0000').attr('stroke-width', 2);

        // Fetch nearby clusters for this cluster
        const clusterId = d.c;
        fetch(`http://localhost:8000/api/clusters/${clusterId}/nearby`)
          .then((res) => res.json())
          .then((response) => {
            const nearbyIds = response.nearbyClusterIds || [];
            console.log(`[Nearby Clusters] Cluster ${clusterId}:`, nearbyIds);
            setNearbyClusterIds(nearbyIds);
          })
          .catch((error) => {
            console.error('Failed to fetch nearby clusters:', error);
            setNearbyClusterIds([]);
          });
      })
      .on('mousemove', function (event, d: Point) {
        if (ignoreNoise && d.c === -1) return;
        const rect = container.getBoundingClientRect();
        setTooltip((prev) => ({
          ...prev,
          visible: true,
          x: event.clientX - rect.left + 12,
          y: event.clientY - rect.top + 12,
          point: { id: d.i, label: d.l, cluster: d.c }
        }));
      })
      .on('mouseout', function () {
        d3.select(this).attr('r', 3).attr('stroke-width', 0);
        setTooltip({ visible: false, x: 0, y: 0 });
        setNearbyClusterIds([]);
      })
      .on('click', (_event, d: Point) => {
        if (ignoreNoise && d.c === -1) return;
        if (interactionMode === 'point') {
          selectPoints([d.i]);
        } else {
          selectClusters([d.c]);
          selectPoints([]);
        }

        // Copy the point's label to clipboard
        const textToCopy = d.l || String(d.i);
        try {
          if (navigator && 'clipboard' in navigator && typeof navigator.clipboard.writeText === 'function') {
            navigator.clipboard.writeText(textToCopy)
              .then(() => {
                console.log('Copied to clipboard:', textToCopy);
              })
              .catch((err) => {
                console.error('Clipboard copy failed:', err);
              });
          } else {
            const textarea = document.createElement('textarea');
            textarea.value = textToCopy;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            console.log('Copied to clipboard (fallback):', textToCopy);
          }
        } catch (err) {
          console.error('Clipboard copy error:', err);
        }
      });

    // Store circles reference and apply initial nearby stroke
    circlesRef.current = circles;
    applyNearbyStroke();

    // Add text annotations for search results
    if (selection.searchResultPointIds.size > 0) {
      const searchResults = data.points.filter(p => selection.searchResultPointIds.has(p.i));
      
      // Adjust font size based on zoom scale
      const zoomScale = zoomTransformRef.current?.k || 1;
      const baseFontSize = 11;
      const adjustedFontSize = Math.max(6, baseFontSize / zoomScale); // min 6px
      
      g.selectAll('.search-annotation')
        .data(searchResults)
        .enter()
        .append('text')
        .attr('class', 'search-annotation')
        .attr('x', (d) => scaleFactors.xScale(d.x))
        .attr('y', (d) => scaleFactors.yScale(d.y) - 8)
        .attr('text-anchor', 'middle')
        .attr('font-size', `${adjustedFontSize}px`)
        .attr('font-weight', '600')
        .attr('fill', '#9C27B0')
        .attr('stroke', '#fff')
        .attr('stroke-width', `${Math.max(1, 2 / zoomScale)}px`)
        .attr('paint-order', 'stroke')
        .attr('pointer-events', 'none')
        .text((d) => d.l);
    }

    // Add annotations for selected clusters (from dendrogram or DR selection)
    // Combine manual selections and DR-derived cluster selections
    const annotationClusterIds = new Set<number>([
      ...Array.from(selection.selectedClusterIds),
      ...Array.from(selection.drSelectedClusterIds)
    ]);

    if (showAnnotations && annotationClusterIds.size > 0) {
      const selectedClusterIds = Array.from(annotationClusterIds);

      // Adjust font size based on zoom scale
      const zoomScale = zoomTransformRef.current?.k || 1;
      const baseFontSize = 12;
      const adjustedFontSize = Math.max(6, baseFontSize / zoomScale); // min 6px

      selectedClusterIds.forEach((cid) => {
        const clusterPoints = data.points.filter((p) => p.c === cid);
        if (clusterPoints.length === 0) return;

        const centroidX = d3.mean(clusterPoints, (p) => scaleFactors.xScale(p.x)) as number;
        const centroidY = d3.mean(clusterPoints, (p) => scaleFactors.yScale(p.y)) as number;

        const rep = (data.clusterNames[cid] || data.clusterNames[String(cid)] || '') as string;
        if (!rep) return;

        g.append('text')
          .attr('class', 'cluster-annotation')
          .attr('x', centroidX)
          .attr('y', centroidY - 12)
          .attr('text-anchor', 'middle')
          .attr('font-size', `${adjustedFontSize}px`)
          .attr('font-weight', '600')
          .attr('fill', '#2C7BE5')
          .attr('stroke', '#fff')
          .attr('stroke-width', `${Math.max(1, 2 / zoomScale)}px`)
          .attr('paint-order', 'stroke')
          .attr('pointer-events', 'none')
          .text(rep);
      });
    }

    // Add zoom behavior (always available, not disabled during selection tool use)
    // The zoom behavior is independent of selection tools and will be managed by disabling/enabling event listeners
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .on('zoom', (event) => {
        // Only apply zoom transform if selection tools are disabled
        if (!brushEnabled && !lassoEnabled) {
          zoomRoot.attr('transform', event.transform);
          zoomTransformRef.current = event.transform;
        }
      });

    // Always attach zoom to SVG but it's only active when selection tools are disabled
    svg.call(zoom as any);
    
    if (zoomTransformRef.current && !brushEnabled && !lassoEnabled) {
      svg.call((zoom.transform as any), zoomTransformRef.current);
    }

    // Add rectangular brush for range selection
    if (brushEnabled && !lassoEnabled) {
      const brushGroup = g.append('g')
        .attr('class', 'brush-layer');
      
      const brush = d3.brush()
        .extent([[0, 0], [width, height]])
        .on('start brush', () => {
          // Apply styles during brush to ensure visibility
          brushGroup.selectAll('.selection')
            .style('fill', 'rgba(0, 123, 255, 0.25)')
            .style('stroke', '#007bff')
            .style('stroke-width', '2px');
        })
        .on('end', (event: any) => {
          const sel = event.selection;
          if (!sel) {
            return;
          }
          const [[x0, y0], [x1, y1]] = sel;
          const minX = Math.min(x0, x1);
          const maxX = Math.max(x0, x1);
          const minY = Math.min(y0, y1);
          const maxY = Math.max(y0, y1);
          const selectedIds: number[] = [];
          for (const p of data.points) {
            if (ignoreNoise && p.c === -1) continue;
            const px = scaleFactors.xScale(p.x);
            const py = scaleFactors.yScale(p.y);
            if (px >= minX && px <= maxX && py >= minY && py <= maxY) {
              selectedIds.push(p.i);
            }
          }
          console.log('Brush selected points:', selectedIds.length);
          selectPoints(selectedIds);
          // Clear brush after selection
          brushGroup.call(brush.move as any, null);
        });
      
      brushGroup.call(brush as any);
    }

    // Lasso selection using dedicated class
    if (lassoEnabled && !brushEnabled) {
      // Clean up previous lasso instance
      if (lassoRef.current) {
        lassoRef.current.disable();
      }
      
      // Create new lasso instance
      lassoRef.current = new LassoSelection({
        container: g,
        svg: svg,
        items: data.points,
        xScale: scaleFactors.xScale as any,
        yScale: scaleFactors.yScale as any,
        ignoreNoise,
        onStart: () => {
          console.log('Lasso selection started');
        },
        onDraw: (possibleIds) => {
          // Optional: preview selection during draw
          // console.log('Possible points:', possibleIds.length);
        },
        onEnd: (selectedIds) => {
          console.log('Lasso selection ended:', selectedIds.length, 'points');
          selectPoints(selectedIds);
        }
      });
      
      lassoRef.current.enable();
    } else if (lassoRef.current) {
      // Disable lasso when not in lasso mode
      lassoRef.current.disable();
      lassoRef.current = null;
    }

  }, [
    data,
    displayData,
    selection.selectedClusterIds,
    selection.selectedPointIds,
    selection.drSelectedClusterIds,
    selection.heatmapClickedClusters,
    selection.dendrogramHoveredCluster,
    selection.searchResultPointIds,
    hoveredClusterIds,
    margin,
    interactionMode,
    selectClusters,
    selectPoints,
    ignoreNoise,
    brushEnabled,
    lassoEnabled,
    showAnnotations,
    applyNearbyStroke
  ]);

  // Cleanup lasso on unmount
  useEffect(() => {
    return () => {
      if (lassoRef.current) {
        lassoRef.current.disable();
        lassoRef.current = null;
      }
    };
  }, []);

  return (
    <div ref={containerRef} className="panel dr-visualization-container">
      <div className="panel-header">
        <div className="header-content">
          <h4>DR Visualization</h4>
          <SearchBar />
          <div className="interaction-controls">
            <label className={`mode-toggle ${interactionMode === 'point' ? 'active' : ''}`}>
              <input
                type="radio"
                name="interaction-mode"
                value="point"
                checked={interactionMode === 'point'}
                onChange={() => setInteractionMode('point')}
              />
              <span>Point mode</span>
            </label>
            <label className={`mode-toggle ${interactionMode === 'cluster' ? 'active' : ''}`}>
              <input
                type="radio"
                name="interaction-mode"
                value="cluster"
                checked={interactionMode === 'cluster'}
                onChange={() => setInteractionMode('cluster')}
              />
              <span>Cluster mode</span>
            </label>
            <label className="mode-toggle">
              <input
                type="checkbox"
                checked={ignoreNoise}
                onChange={(e) => setIgnoreNoise(e.target.checked)}
              />
              <span>Ignore noise (-1)</span>
            </label>
            <label className="mode-toggle">
              <input
                type="checkbox"
                checked={brushEnabled}
                onChange={(e) => setBrushEnabled(e.target.checked)}
              />
              <span>Range select (brush)</span>
            </label>
            <label className="mode-toggle">
              <input
                type="checkbox"
                checked={lassoEnabled}
                onChange={(e) => setLassoEnabled(e.target.checked)}
              />
              <span>Lasso select</span>
            </label>
                <label className="mode-toggle">
                  <span style={{ marginRight: 6 }}>Min cluster coverage</span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={containmentThreshold}
                    onChange={(e) => setContainmentThreshold(Number(e.target.value))}
                    style={{ verticalAlign: 'middle' }}
                  />
                  <span style={{ marginLeft: 6 }}>{containmentThreshold.toFixed(2)}</span>
                </label>
            <label className="mode-toggle">
              <input
                type="checkbox"
                checked={showHoverWords}
                onChange={(e) => setShowHoverWords(e.target.checked)}
              />
              <span>Hover representative</span>
            </label>
            <label className="mode-toggle">
              <input
                type="checkbox"
                checked={showAnnotations}
                onChange={(e) => setShowAnnotations(e.target.checked)}
              />
              <span>Cluster annotations</span>
            </label>
            <button
              className="zoom-redraw-button"
              onClick={handleZoomRedraw}
              disabled={isZoomProcessing || selection.selectedPointIds.size === 0}
              style={{
                marginLeft: '10px',
                padding: '6px 12px',
                backgroundColor: selection.selectedPointIds.size === 0 ? '#ccc' : '#2C7BE5',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: selection.selectedPointIds.size === 0 || isZoomProcessing ? 'not-allowed' : 'pointer',
                fontSize: '12px',
                fontWeight: '500'
              }}
              title={
                selection.selectedPointIds.size === 0
                  ? 'Select points first'
                  : `Recalculate UMAP for ${selection.selectedPointIds.size} selected points`
              }
            >
              {isZoomProcessing ? 'Processing...' : `Zoom Redraw (${selection.selectedPointIds.size})`}
            </button>
          </div>
        </div>
      </div>
      <div className="panel-content">
        <svg ref={svgRef} className="dr-plot" />
        {tooltip.visible && tooltip.point && (
          <div
            className="dr-tooltip"
            style={{ left: tooltip.x, top: tooltip.y }}
          >
            <div>ID: {tooltip.point.id}</div>
            <div>Label: {tooltip.point.label}</div>
            <div>Cluster: {tooltip.point.cluster}</div>
            {showHoverWords && (
              (() => {
                const rep = (data.clusterNames[tooltip.point!.cluster] || data.clusterNames[String(tooltip.point!.cluster)] || '') as string;
                return rep ? (
                  <div>Rep: {rep}</div>
                ) : null;
              })()
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DRVisualization;
