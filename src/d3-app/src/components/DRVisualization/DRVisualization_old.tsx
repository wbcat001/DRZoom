import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { useSelection, useData } from '../../store/useAppStore.tsx';
import { Point } from '../../types';
import { determinePointHighlight, isAnySelectionActive, createScaleFactors } from '../../utils';
import { getElementStyle } from '../../types/color';
import './DRVisualization.css';

const DRVisualization: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { selection, selectClusters, selectPoints } = useSelection();
  const { data } = useData();
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [interactionMode, setInteractionMode] = useState<'point' | 'cluster'>('point');
  const [ignoreNoise, setIgnoreNoise] = useState<boolean>(true);
  const [tooltip, setTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    point?: { id: number; label: string; cluster: number };
  }>({ visible: false, x: 0, y: 0 });

  const margin = { top: 20, right: 20, bottom: 40, left: 40 };

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

  // Main D3 rendering logic
  useEffect(() => {
    if (!svgRef.current || !containerRef.current || data.points.length === 0) return;

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

    // Create scales
    const scaleFactors = createScaleFactors(data.points, width, height, 0);

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add background
    g.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', '#f8f9fa')
      .attr('pointer-events', 'all');

    // Color scale for clusters
    const clusterIds = Array.from(new Set(data.points.map((p) => p.c)));
    const colorScale = d3.scaleOrdinal()
      .domain(clusterIds.map((id) => id.toString()))
      .range(d3.schemeCategory10);

    // Check if any selection is active
    const anySelectionActive = isAnySelectionActive(
      selection.selectedClusterIds,
      selection.selectedPointIds,
      selection.heatmapClickedClusters,
      selection.dendrogramHoveredCluster
    );

    // Draw points
    const circles = g.selectAll('.data-point')
      .data(data.points, (d: any) => d.i)
      .enter()
      .append('circle')
      .attr('class', 'data-point')
      .attr('cx', (d) => scaleFactors.xScale(d.x))
      .attr('cy', (d) => scaleFactors.yScale(d.y))
      .attr('r', 1)
      .attr('fill', (d) => {
        const highlight = determinePointHighlight(
          d.i,
          d,
          selection.selectedClusterIds,
          selection.selectedPointIds,
          selection.heatmapClickedClusters,
          selection.dendrogramHoveredCluster
        );
        const anyActive = isAnySelectionActive(
          selection.selectedClusterIds,
          selection.selectedPointIds,
          selection.heatmapClickedClusters,
          selection.dendrogramHoveredCluster
        );
        const style = getElementStyle(highlight, anyActive);
        return (style.fill || colorScale(d.c.toString())) as string;
      })
      .attr('opacity', (d) => {
        const highlight = determinePointHighlight(
          d.i,
          d,
          selection.selectedClusterIds,
          selection.selectedPointIds,
          selection.heatmapClickedClusters,
          selection.dendrogramHoveredCluster
        );
        const style = getElementStyle(highlight, anySelectionActive);
        return style.opacity || 1.0;
      })
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
        d3.select(this).attr('r', 5).attr('stroke', '#333').attr('stroke-width', 1);
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
      })
      .on('click', (_event, d: Point) => {
        if (ignoreNoise && d.c === -1) return;
        if (interactionMode === 'point') {
          selectPoints([d.i]);
        } else {
          selectClusters([d.c]);
          selectPoints([]);
        }
      });

    // Add zoom behavior
    const zoom = d3.zoom<SVGGElement, unknown>()
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom as any);

  }, [data.points, selection, margin, interactionMode, selectClusters, selectPoints, ignoreNoise]);

  return (
    <div ref={containerRef} className="panel dr-visualization-container">
      <div className="panel-header">
        <div className="header-content">
          <h4>DR Visualization</h4>
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
          </div>
        )}
      </div>
    </div>
  );
};

export default DRVisualization;
