import React, { useRef, useEffect, useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useSelection, useData, useAppContext } from '../../store/useAppStore';
import { computeDendrogramCoords, generateDendrogramSegments } from '../../utils/dendrogramCoords';
import { HIGHLIGHT_COLORS } from '../../types/color';
import { buildSimilarityMap } from '../../utils/similarity';
import { DendrogramSortMode } from '../../types/data';
import './Dendrogram.css';

const Dendrogram: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const zoomTransformRef = useRef<any>(null);  // Store zoom transform
  const { selection, setDendrogramHovered, selectClusters } = useSelection();
  const { data } = useData();
  const { state, dispatch } = useAppContext();

  const [proportionalWidth, setProportionalWidth] = useState(false);
  const [showLabels, setShowLabels] = useState(false);
  const [brushEnabled, setBrushEnabled] = useState(false);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [tooltip, setTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    cluster?: { 
      id: number; 
      name: string; 
      size: number; 
      stability: number; 
      words: string[]; 
      child1?: number; 
      child2?: number; 
      parent?: number;
      child1Words?: string[];
      child2Words?: string[];
      child1Name?: string;
      child2Name?: string;
    };
  }>({ visible: false, x: 0, y: 0 });

  const margin = { top: 40, right: 20, bottom: 40, left: 40 };

  // Compute parent map from linkage matrix (each node -> its parent node)
  const parentMap = useMemo(() => {
    const map: Record<number, number> = {};
    data.linkageMatrix.forEach((entry, idx) => {
      const parentIdx = data.linkageMatrix.length + idx;
      map[entry.child1] = parentIdx;
      map[entry.child2] = parentIdx;
    });
    return map;
  }, [data.linkageMatrix]);

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

  // Compute dendrogram coordinates
  const dendrogramData = useMemo(() => {
    console.log('dendrogramData computing', {
      linkageMatrixLength: data.linkageMatrix.length,
      dendrogramSortMode: state.dendrogramSortMode,
      hasClusterSimilarities: !!state.clusterSimilarities
    });

    if (data.linkageMatrix.length === 0) {
      console.log('dendrogramData: early return - empty linkageMatrix');
      return null;
    }

    try {
      const sortMode = state.dendrogramSortMode;
      const similarityMap = state.clusterSimilarities
        ? buildSimilarityMap(state.clusterSimilarities)
        : undefined;
      
      const coords = computeDendrogramCoords(
        data.linkageMatrix,
        data.linkageMatrix.length + 1,
        sortMode,
        similarityMap
      );
      const segments = generateDendrogramSegments(coords);
      console.log('dendrogramData: computed successfully', {
        coordsLength: coords.dcoord.length,
        segmentsLength: segments.length
      });
      return { coords, segments };
    } catch (error) {
      console.error('dendrogramData: Error computing dendrogram:', error);
      return null;
    }
  }, [data.linkageMatrix, state.dendrogramSortMode, state.clusterSimilarities]);

  // Reverse clusterIdMap: original cluster ID -> sequential index used in linkage/coords
  const reverseClusterIdMap = useMemo(() => {
    const rev = new Map<number, number>();
    Object.entries(data.clusterIdMap || {}).forEach(([seqStr, orig]) => {
      const seq = Number(seqStr);
      const origId = Number(orig);
      if (!Number.isNaN(seq) && !Number.isNaN(origId)) {
        rev.set(origId, seq);
      }
    });
    return rev;
  }, [data.clusterIdMap]);

  // Main D3 rendering logic
  useEffect(() => {
    console.log('Dendrogram useEffect triggered', {
      hasSvgRef: !!svgRef.current,
      hasContainerRef: !!containerRef.current,
      hasDendrogramData: !!dendrogramData,
      linkageMatrixLength: data.linkageMatrix.length
    });

    if (!svgRef.current || !containerRef.current || !dendrogramData) {
      console.log('Dendrogram early return:', {
        noSvg: !svgRef.current,
        noContainer: !containerRef.current,
        noData: !dendrogramData
      });
      return;
    }

    console.log('Dendrogram rendering START');

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

    // Create zoom root and plot group
    const zoomRoot = svg.append('g');
    const g = zoomRoot.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Reapply stored zoom/pan transform so toggling brush mode preserves view
    if (zoomTransformRef.current) {
      zoomRoot.attr('transform', zoomTransformRef.current as any);
    }

    // Get coordinate ranges
    const icoordFlat = dendrogramData.coords.icoord.flat();
    const dcoordFlat = dendrogramData.coords.dcoord.flat();

    const xMin = Math.min(...icoordFlat);
    const xMax = Math.max(...icoordFlat);
    const yMin = Math.min(...dcoordFlat);
    const yMax = Math.max(...dcoordFlat);

    // Create scales
    const xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, width]);
    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([height, 0]);

    // Precompute leaf positions (x) and corresponding original cluster IDs
    const leafXsCoord: number[] = [];
    for (let i = 0; i < dendrogramData.coords.dcoord.length; i++) {
      const yVals = dendrogramData.coords.dcoord[i];
      const xVals = dendrogramData.coords.icoord[i];
      // Left child is a leaf if y1 == 0
      if (yVals[0] === 0) {
        leafXsCoord.push(xVals[0]);
      }
      // Right child is a leaf if y4 == 0
      if (yVals[3] === 0) {
        leafXsCoord.push(xVals[3]);
      }
    }
    // Unique and sorted by coordinate
    const leafXsSorted = Array.from(new Set(leafXsCoord)).sort((a, b) => a - b);
    const leafScreenXs = leafXsSorted.map((x) => xScale(x));
    // Map sorted x order to leafOrder indices → original cluster IDs
    const leafSeqOrder = dendrogramData.coords.leafOrder;
    const leafOrigIds = leafSeqOrder.map((seqIdx) => (data.clusterIdMap[seqIdx] ?? seqIdx));

    const leafDetectBandPx = 30; // vertical band near bottom for leaf hover detection

    // Add background with leaf hover detection
    g.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', '#f8f9fa')
      .attr('pointer-events', 'all')
      .on('mousemove', (event) => {
        // Detect leaf under mouse when cursor is near the leaf band (bottom area)
        if (leafScreenXs.length === 0) return;
        const [mx, my] = d3.pointer(event, g.node() as any);
        if (my < height - leafDetectBandPx) return;
        const idx = d3.bisectCenter(leafScreenXs, mx);
        const origId = leafOrigIds[Math.max(0, Math.min(idx, leafOrigIds.length - 1))];

        const rect2 = container.getBoundingClientRect();
        const meta = (data.clusterMetadata[origId] || (data.clusterMetadata as any)[String(origId)]);
        const name = (data.clusterNames[origId] || (data.clusterNames as any)[String(origId)] || `Cluster ${origId}`) as string;
        const words = (data.clusterWords[origId] || (data.clusterWords as any)[String(origId)] || []) as string[];

        setTooltip({
          visible: true,
          x: event.clientX - rect2.left + 12,
          y: event.clientY - rect2.top - 60,
          cluster: {
            id: origId,
            name,
            size: meta?.z || 0,
            stability: meta?.s || 0,
            words: words.slice(0, 5)
          }
        });
      })
      .on('mouseleave', () => {
        setTooltip({ visible: false, x: 0, y: 0 });
      });

    // Draw dendrogram segments
    g.selectAll('.dendrogram-segment')
      .data(dendrogramData.segments, (d, i) => i)
      .enter()
      .append('line')
      .attr('class', 'dendrogram-segment')
      .attr('x1', (d) => xScale(d[0][0]))
      .attr('y1', (d) => yScale(d[0][1]))
      .attr('x2', (d) => xScale(d[1][0]))
      .attr('y2', (d) => yScale(d[1][1]))
      .attr('stroke', (d, i) => {
        const mergeIdx = Math.floor(i / 3);
        
        // Check if this merge's children are in the selected cluster IDs
        const linkageEntry = data.linkageMatrix[mergeIdx];
        if (linkageEntry) {
          const child1Id = data.clusterIdMap[linkageEntry.child1] ?? linkageEntry.child1;
          const child2Id = data.clusterIdMap[linkageEntry.child2] ?? linkageEntry.child2;
          
          const isChildSelected = selection.selectedClusterIds.has(child1Id) || selection.selectedClusterIds.has(child2Id);
          const isHeatmapClicked = selection.heatmapClickedClusters.has(child1Id) || selection.heatmapClickedClusters.has(child2Id);
          const isDRChildSelected = selection.drSelectedClusterIds.has(child1Id) || selection.drSelectedClusterIds.has(child2Id);
          const isHovered = selection.dendrogramHoveredCluster === mergeIdx;

          // Priority order: heatmap_click > selectedClusterIds (direct children) > drSelectedClusterIds (direct children) > hovered > default
          if (isHeatmapClicked) return HIGHLIGHT_COLORS.heatmap_click;
          if (isChildSelected) return HIGHLIGHT_COLORS.dr_selection;
          if (isDRChildSelected) return HIGHLIGHT_COLORS.dr_selected; // DR-derived selection
          if (isHovered) return HIGHLIGHT_COLORS.dendrogram_to_dr;
        }
        return HIGHLIGHT_COLORS.default;
      })
      .attr('stroke-width', (d, i) => {
        const mergeIdx = Math.floor(i / 3);
        
        const linkageEntry = data.linkageMatrix[mergeIdx];
        if (linkageEntry) {
          const child1Id = data.clusterIdMap[linkageEntry.child1] ?? linkageEntry.child1;
          const child2Id = data.clusterIdMap[linkageEntry.child2] ?? linkageEntry.child2;
          
          const isHeatmapClicked = selection.heatmapClickedClusters.has(child1Id) || selection.heatmapClickedClusters.has(child2Id);
          const isDRChildSelected = selection.drSelectedClusterIds.has(child1Id) || selection.drSelectedClusterIds.has(child2Id);
          
          if (isHeatmapClicked || isDRChildSelected) return 2;
        }
        return 1;
      })
      .attr('opacity', (d, i) => {
        const mergeIdx = Math.floor(i / 3);
        
        const linkageEntry = data.linkageMatrix[mergeIdx];
        if (linkageEntry) {
          const child1Id = data.clusterIdMap[linkageEntry.child1] ?? linkageEntry.child1;
          const child2Id = data.clusterIdMap[linkageEntry.child2] ?? linkageEntry.child2;
          
          const isDRChildSelected = selection.drSelectedClusterIds.has(child1Id) || selection.drSelectedClusterIds.has(child2Id);
          
          if (isDRChildSelected) return 0.9;
        }
        return 0.7;
      })
      .on('mouseover', function (event, d) {
        // Get linkage entry for this merge operation
        // d is the segment data with structure: [[x1, y1], [x2, y2]]
        // We need to find which linkage operation this segment belongs to
        const allSegments = g.selectAll('.dendrogram-segment').nodes();
        const thisIndex = allSegments.indexOf(this);
        
        // Each merge operation produces 3 segments, so mergeIdx = floor(thisIndex / 3)
        const mergeIdx = Math.floor(thisIndex / 3);
        
        if (mergeIdx >= data.linkageMatrix.length) {
          console.warn(`mergeIdx ${mergeIdx} out of range (linkageMatrix length: ${data.linkageMatrix.length})`);
          return;
        }
        
        // Get the linkage entry for this merge
        const linkageEntry = data.linkageMatrix[mergeIdx];
        const child1Idx = linkageEntry?.child1;
        const child2Idx = linkageEntry?.child2;
        const parentIdx = linkageEntry?.parent;
        
        // Convert to actual cluster IDs using clusterIdMap
        const child1 = child1Idx !== undefined ? (data.clusterIdMap[child1Idx] ?? child1Idx) : undefined;
        const child2 = child2Idx !== undefined ? (data.clusterIdMap[child2Idx] ?? child2Idx) : undefined;
        const parentId = parentIdx !== undefined ? (data.clusterIdMap[parentIdx] ?? parentIdx) : undefined;
        
        // Get child1's words
        const child1Words = child1 !== undefined ? (data.clusterWords[child1] || data.clusterWords[String(child1)] || []) : [];
        const child1Name = child1 !== undefined ? (data.clusterNames[child1] || data.clusterNames[String(child1)] || '') : '';
        
        // Get child2's words
        const child2Words = child2 !== undefined ? (data.clusterWords[child2] || data.clusterWords[String(child2)] || []) : [];
        const child2Name = child2 !== undefined ? (data.clusterNames[child2] || data.clusterNames[String(child2)] || '') : '';
        
        // Get parent's metadata
        const parentMeta = parentId !== undefined ? (data.clusterMetadata[parentId] || data.clusterMetadata[String(parentId)]) : undefined;
        const parentWords = parentId !== undefined ? (data.clusterWords[parentId] || data.clusterWords[String(parentId)] || []) : [];
        const parentName = parentId !== undefined ? (data.clusterNames[parentId] || data.clusterNames[String(parentId)] || `Cluster ${parentId}`) : '';
        
        console.log('Cluster hover:', {
          mergeIdx,
          child1,
          child1Words: child1Words.slice(0, 5),
          child2,
          child2Words: child2Words.slice(0, 5),
          parentId,
          parentName,
          parentWords: parentWords.slice(0, 5)
        });
        
        setDendrogramHovered(mergeIdx);
        
        const rect = container.getBoundingClientRect();
        setTooltip({
          visible: true,
          x: event.clientX - rect.left + 12,
          y: event.clientY - rect.top - 80,
          cluster: {
            id: parentId,
            name: parentName,
            size: parentMeta?.z || 0,
            stability: parentMeta?.s || 0,
            words: parentWords.slice(0, 5),
            child1,
            child2,
            parent: undefined,
            child1Words: child1Words.slice(0, 5),
            child2Words: child2Words.slice(0, 5),
            child1Name,
            child2Name
          }
        });
      })
      .on('mouseout', function () {
        setDendrogramHovered(null);
        setTooltip({ visible: false, x: 0, y: 0 });
      })
      .on('click', function (event, d) {
        // Get merge index for clicked segment
        const allSegments = g.selectAll('.dendrogram-segment').nodes();
        const thisIndex = allSegments.indexOf(this);
        const mergeIdx = Math.floor(thisIndex / 3);
        
        if (mergeIdx >= data.linkageMatrix.length) return;
        
        const linkageEntry = data.linkageMatrix[mergeIdx];
        const child1Idx = linkageEntry?.child1;
        const child2Idx = linkageEntry?.child2;
        const parentIdx = linkageEntry?.parent;
        
        const child1 = child1Idx !== undefined ? (data.clusterIdMap[child1Idx] ?? child1Idx) : undefined;
        const child2 = child2Idx !== undefined ? (data.clusterIdMap[child2Idx] ?? child2Idx) : undefined;
        const parentId = parentIdx !== undefined ? (data.clusterIdMap[parentIdx] ?? parentIdx) : undefined;
        
        const child1Words = child1 !== undefined ? (data.clusterWords[child1] || data.clusterWords[String(child1)] || []) : [];
        const child1Name = child1 !== undefined ? (data.clusterNames[child1] || data.clusterNames[String(child1)] || '') : '';
        const child2Words = child2 !== undefined ? (data.clusterWords[child2] || data.clusterWords[String(child2)] || []) : [];
        const child2Name = child2 !== undefined ? (data.clusterNames[child2] || data.clusterNames[String(child2)] || '') : '';
        const parentMeta = parentId !== undefined ? (data.clusterMetadata[parentId] || data.clusterMetadata[String(parentId)]) : undefined;
        const parentWords = parentId !== undefined ? (data.clusterWords[parentId] || data.clusterWords[String(parentId)] || []) : [];
        const parentName = parentId !== undefined ? (data.clusterNames[parentId] || data.clusterNames[String(parentId)] || `Cluster ${parentId}`) : '';
        
        // Format tooltip information for clipboard
        let clipboardText = `Dendrogram Merge Information\n`;
        clipboardText += `==============================\n\n`;
        clipboardText += `Parent: ${parentName}\n`;
        clipboardText += `ID: ${parentId}\n`;
        clipboardText += `Size: ${parentMeta?.z || 0}\n`;
        clipboardText += `Stability: ${parentMeta?.s?.toFixed(3) || 'N/A'}\n\n`;
        
        if (child1 !== undefined) {
          clipboardText += `Child1: ${child1}\n`;
          if (child1Name) clipboardText += `  Name: ${child1Name}\n`;
          if (child1Words.length > 0) {
            clipboardText += `  Words: ${child1Words.join(', ')}\n`;
          }
          clipboardText += `\n`;
        }
        
        if (child2 !== undefined) {
          clipboardText += `Child2: ${child2}\n`;
          if (child2Name) clipboardText += `  Name: ${child2Name}\n`;
          if (child2Words.length > 0) {
            clipboardText += `  Words: ${child2Words.join(', ')}\n`;
          }
          clipboardText += `\n`;
        }
        
        if (parentWords.length > 0) {
          clipboardText += `Parent Words: ${parentWords.join(', ')}\n`;
        }
        
        // Copy to clipboard
        navigator.clipboard.writeText(clipboardText)
          .then(() => {
            console.log('Copied dendrogram info to clipboard');
            // Optional: show a brief notification
          })
          .catch(err => {
            console.error('Failed to copy to clipboard:', err);
          });
      });

    // Add annotations for clusters selected via dendrogram or DR selection
    const annotationClusterIds = new Set<number>([
      ...Array.from(selection.selectedClusterIds),
      ...Array.from(selection.drSelectedClusterIds)
    ]);

    if (annotationClusterIds.size > 0) {
      const leafCount = data.linkageMatrix.length + 1;
      const rowsCount = dendrogramData.coords.icoord.length;

      const annotations: { x: number; y: number; label: string; idx: number }[] = [];

      let annIdx = 0;

      annotationClusterIds.forEach((origId) => {
        const seqIdx = reverseClusterIdMap.get(origId);
        if (seqIdx === undefined) return;

        // For merge nodes, seqIdx corresponds to linkage row index = seqIdx - leafCount
        const rowIdx = seqIdx - leafCount;
        if (rowIdx < 0 || rowIdx >= rowsCount) return;

        const xVals = dendrogramData.coords.icoord[rowIdx];
        const yVals = dendrogramData.coords.dcoord[rowIdx];
        if (!xVals || !yVals || xVals.length === 0 || yVals.length === 0) return;

        const xPos = d3.mean(xVals) ?? 0;
        const yPos = d3.max(yVals) ?? 0;

        const label = (data.clusterNames[origId] || data.clusterNames[String(origId)] || `Cluster ${origId}`) as string;
        // Stagger vertically by index to reduce overlap
        const offset = annIdx * -10; // -10px per annotation
        annotations.push({ x: xScale(xPos), y: yScale(yPos) - 8 + offset, label, idx: annIdx });
        annIdx += 1;
      });

      // Adjust font size based on zoom scale
      const zoomScale = zoomTransformRef.current?.k || 1;
      const baseFontSize = 10;
      const adjustedFontSize = Math.max(6, baseFontSize / zoomScale); // min 6px

      g.selectAll('.dendro-annotation')
        .data(annotations)
        .enter()
        .append('text')
        .attr('class', 'dendro-annotation')
        .attr('x', (d) => d.x)
        .attr('y', (d) => d.y)
        .attr('text-anchor', 'middle')
        .attr('font-size', `${adjustedFontSize}px`)
        .attr('font-weight', '500')
        .attr('fill', '#2C7BE5')
        .attr('stroke', '#fff')
        .attr('stroke-width', `${Math.max(1, 2 / zoomScale)}px`)
        .attr('paint-order', 'stroke')
        .attr('pointer-events', 'none')
        .text((d) => d.label);
    }

    // Add brush selection (rectangular)
    if (brushEnabled) {
      const brushGroup = g.append('g').attr('class', 'brush-layer');
      const brush = d3.brush()
        .extent([[0, 0], [width, height]])
        .on('start brush', () => {
          // Make brush selection visible with inline styles
          brushGroup.selectAll('.selection')
            .style('stroke', '#007bff')
            .style('stroke-width', '2px')
            .style('fill', 'rgba(0, 123, 255, 0.15)');
        })
        .on('end', (event) => {
          if (!event.selection) {
            // Clear selection if brush is removed
            selectClusters([]);
            console.log('Brush cleared, reset cluster selection');
            return;
          }

          const [[x0, y0], [x1, y1]] = event.selection as [[number, number], [number, number]];
          const minX = Math.min(x0, x1);
          const maxX = Math.max(x0, x1);
          const minY = Math.min(y0, y1);
          const maxY = Math.max(y0, y1);
          
          // Find merge operations whose segments intersect with brush rectangle
          const selectedMergeIndices = new Set<number>();
          dendrogramData.segments.forEach((seg, segIdx) => {
            const xA = xScale(seg[0][0]);
            const yA = yScale(seg[0][1]);
            const xB = xScale(seg[1][0]);
            const yB = yScale(seg[1][1]);

            const segMinX = Math.min(xA, xB);
            const segMaxX = Math.max(xA, xB);
            const segMinY = Math.min(yA, yB);
            const segMaxY = Math.max(yA, yB);

            const intersects = !(segMaxX < minX || segMinX > maxX || segMaxY < minY || segMinY > maxY);
            if (intersects) {
              const mergeIdx = Math.floor(segIdx / 3);
              selectedMergeIndices.add(mergeIdx);
            }
          });

          // Collect child cluster IDs from selected merges (only direct children, not recursive)
          const allChildClusterIds = new Set<number>();
          console.log(`Selected ${selectedMergeIndices.size} merge operations`);
          
          selectedMergeIndices.forEach(mergeIdx => {
            const linkageEntry = data.linkageMatrix[mergeIdx];
            if (linkageEntry) {
              // Get actual cluster IDs for child1 and child2
              const child1Id = data.clusterIdMap[linkageEntry.child1] ?? linkageEntry.child1;
              const child2Id = data.clusterIdMap[linkageEntry.child2] ?? linkageEntry.child2;
              
              console.log(`Merge #${mergeIdx}: child1=${child1Id}, child2=${child2Id}`);
              
              allChildClusterIds.add(child1Id);
              allChildClusterIds.add(child2Id);
            }
          });
          
          const selectedClusterIds = Array.from(allChildClusterIds);
          console.log('Dendrogram brush selected child clusters:', selectedClusterIds.length, selectedClusterIds);
          selectClusters(selectedClusterIds);
        });

      brushGroup.call(brush as any);
    }

    // Add zoom behavior (only when brush is disabled)
    if (!brushEnabled) {
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .on('zoom', (event) => {
          zoomRoot.attr('transform', event.transform);
          zoomTransformRef.current = event.transform;
        });

      svg.call(zoom as any);
      
      if (zoomTransformRef.current) {
        svg.call((zoom.transform as any), zoomTransformRef.current);
      }
    }

    console.log('Dendrogram rendering COMPLETE');
  }, [dendrogramData, selection, margin, brushEnabled, selectClusters, setDendrogramHovered]);

  return (
    <div ref={containerRef} className="panel dendrogram-container">
      <div className="panel-header">
        Dendrogram
        <div className="header-controls">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={proportionalWidth}
              onChange={(e) => setProportionalWidth(e.target.checked)}
            />
            <span>Size Weight</span>
          </label>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={showLabels}
              onChange={(e) => setShowLabels(e.target.checked)}
            />
            <span>Labels</span>
          </label>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={brushEnabled}
              onChange={(e) => setBrushEnabled(e.target.checked)}
            />
            <span>Brush Selection</span>
          </label>
          <label className="select-label">
            <span>Sort:</span>
            <select
              value={state.dendrogramSortMode}
              onChange={(e) => dispatch({ type: 'SET_DENDROGRAM_SORT_MODE', payload: e.target.value as DendrogramSortMode })}
            >
              <option value="default">Default</option>
              <option value="size">Size</option>
              <option value="similarity">Similarity</option>
            </select>
          </label>
        </div>
      </div>
      <div className="panel-content">
        <svg ref={svgRef} className="dendrogram-plot" />
        {tooltip.visible && tooltip.cluster && (
          <div
            className="dendrogram-tooltip"
            style={{ left: tooltip.x, top: tooltip.y }}
          >
            <div><strong>{tooltip.cluster.name}</strong></div>
            <div className="tooltip-meta">ID: {tooltip.cluster.id}</div>
            <div>Size: {tooltip.cluster.size}</div>
            <div>Stability: {tooltip.cluster.stability.toFixed(3)}</div>
            {tooltip.cluster.child1 !== undefined && (
              <div className="tooltip-section">
                <div className="tooltip-meta">Child1: {tooltip.cluster.child1}</div>
                {(tooltip.cluster as any).child1Words && (tooltip.cluster as any).child1Words.length > 0 && (
                  <div className="tooltip-words" style={{ marginLeft: '10px', fontSize: '0.9em' }}>
                    {(tooltip.cluster as any).child1Name && <span>{(tooltip.cluster as any).child1Name}: </span>}
                    {(tooltip.cluster as any).child1Words.join(', ')}
                  </div>
                )}
              </div>
            )}
            {tooltip.cluster.child2 !== undefined && (
              <div className="tooltip-section">
                <div className="tooltip-meta">Child2: {tooltip.cluster.child2}</div>
                {(tooltip.cluster as any).child2Words && (tooltip.cluster as any).child2Words.length > 0 && (
                  <div className="tooltip-words" style={{ marginLeft: '10px', fontSize: '0.9em' }}>
                    {(tooltip.cluster as any).child2Name && <span>{(tooltip.cluster as any).child2Name}: </span>}
                    {(tooltip.cluster as any).child2Words.join(', ')}
                  </div>
                )}
              </div>
            )}
            {tooltip.cluster.parent !== undefined && (
              <div className="tooltip-meta">Parent: {tooltip.cluster.parent}</div>
            )}
            {tooltip.cluster.words.length > 0 && (
              <div className="tooltip-words">
                Parent words: {tooltip.cluster.words.join(', ')}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Dendrogram;
