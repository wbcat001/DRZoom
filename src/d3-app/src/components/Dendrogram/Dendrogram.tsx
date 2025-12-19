import React, { useRef, useEffect, useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useAppContext, useSelection, useData } from '../../store/useAppStore.tsx';
import { computeDendrogramCoords, generateDendrogramSegments } from '../../utils/dendrogramCoords';
import { getElementStyle, HIGHLIGHT_COLORS } from '../../types/color';
import './Dendrogram.css';

const Dendrogram: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { state, dispatch } = useAppContext();
  const { selection, setDendrogramHovered } = useSelection();
  const { data } = useData();

  const [proportionalWidth, setProportionalWidth] = useState(false);
  const [showLabels, setShowLabels] = useState(false);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  const margin = { top: 40, right: 20, bottom: 40, left: 40 };

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
    if (data.linkageMatrix.length === 0) return null;

    try {
      const coords = computeDendrogramCoords(data.linkageMatrix, data.linkageMatrix.length + 1);
      const segments = generateDendrogramSegments(coords);
      return { coords, segments };
    } catch (error) {
      console.error('Error computing dendrogram:', error);
      return null;
    }
  }, [data.linkageMatrix]);

  // Main D3 rendering logic
  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !dendrogramData) return;

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

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add background
    g.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', '#f8f9fa')
      .attr('pointer-events', 'all');

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
        const clusterIdx = Math.floor(i / 3);
        const isSelected = selection.selectedClusterIds.has(clusterIdx);
        const isHeatmapClicked = selection.heatmapClickedClusters.has(clusterIdx);
        const isHovered = selection.dendrogramHoveredCluster === clusterIdx;

        if (isHeatmapClicked) return HIGHLIGHT_COLORS.heatmap_click;
        if (isSelected) return HIGHLIGHT_COLORS.dr_selection;
        if (isHovered) return HIGHLIGHT_COLORS.dendrogram_to_dr;
        return HIGHLIGHT_COLORS.default;
      })
      .attr('stroke-width', 1)
      .attr('opacity', 0.7)
      .on('mouseover', function () {
        const clusterIdx = Math.floor((this as any).__data__[2] / 3);
        setDendrogramHovered(clusterIdx);
      })
      .on('mouseout', function () {
        setDendrogramHovered(null);
      });

    // Add zoom behavior
    const zoom = d3.zoom<SVGGElement, unknown>()
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom as any);

  }, [dendrogramData, selection, margin]);

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
        </div>
      </div>
      <div className="panel-content">
        <svg ref={svgRef} className="dendrogram-plot" />
      </div>
    </div>
  );
};

export default Dendrogram;
