import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { useAppContext, useSelection, useViewConfig } from '../../store/useAppStore.tsx';
import { apiClient } from '../../api/client';
import { HIGHLIGHT_COLORS } from '../../types/color';
import './ClusterHeatmap.css';

const ClusterHeatmap: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { state } = useAppContext();
  const { selection, setHeatmapClicked } = useSelection();
  const { config } = useViewConfig();
  const [heatmapData, setHeatmapData] = useState<any>(null);
  const [reverseColorScale, setReverseColorScale] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const margin = { top: 60, right: 60, bottom: 60, left: 60 };

  // Load heatmap data when metric or selected clusters change
  useEffect(() => {
    const loadHeatmapData = async () => {
      if (state.points.length === 0) return;

      try {
        setIsLoading(true);
        const response = await apiClient.getHeatmap({
          metric: config.currentMetric,
          topN: 200,
          clusterIds: Array.from(selection.selectedClusterIds).length > 0 
            ? Array.from(selection.selectedClusterIds) 
            : undefined
        });

        if (response.data) {
          setHeatmapData(response.data);
        }
      } catch (error) {
        console.error('Error loading heatmap:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadHeatmapData();
  }, [config.currentMetric, selection.selectedClusterIds, state.points.length]);

  // Main D3 rendering logic
  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !heatmapData || isLoading) return;

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

    const { matrix, clusterOrder } = heatmapData;
    const n = Math.min(matrix.length, clusterOrder.length);

    if (n === 0) {
      g.append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .text('No data available');
      return;
    }

    const cellSize = Math.min(width / n, height / n);

    // Color scale
    const maxValue = Math.max(...matrix.flat());
    const minValue = Math.min(...matrix.flat());

    const colorScale = reverseColorScale
      ? d3.scaleLinear<string>()
          .domain([minValue, maxValue])
          .range(['#fff', '#d62728'])
      : d3.scaleLinear<string>()
          .domain([minValue, maxValue])
          .range(['#fff', '#1f77b4']);

    // Create scales
    const xScale = d3.scaleBand()
      .domain(clusterOrder.map((id: number) => id.toString()))
      .range([0, n * cellSize])
      .padding(0.02);

    const yScale = d3.scaleBand()
      .domain(clusterOrder.map((id: number) => id.toString()))
      .range([0, n * cellSize])
      .padding(0.02);

    // Draw cells
    clusterOrder.forEach((rowClusterId: number, i: number) => {
      clusterOrder.forEach((colClusterId: number, j: number) => {
        if (i < matrix.length && j < matrix[i].length) {
          const value = matrix[i][j];
          const isSelected = selection.selectedClusterIds.has(rowClusterId) ||
                           selection.selectedClusterIds.has(colClusterId);

          g.append('rect')
            .attr('class', 'heatmap-cell')
            .attr('x', xScale(colClusterId.toString()) || 0)
            .attr('y', yScale(rowClusterId.toString()) || 0)
            .attr('width', xScale.bandwidth())
            .attr('height', yScale.bandwidth())
            .style('fill', colorScale(value))
            .style('stroke', isSelected ? HIGHLIGHT_COLORS.dr_selection : '#fff')
            .style('stroke-width', isSelected ? 2 : 1)
            .style('cursor', 'pointer')
            .on('click', () => {
              setHeatmapClicked([rowClusterId, colClusterId]);
            })
            .on('mouseover', function () {
              d3.select(this)
                .style('stroke', HIGHLIGHT_COLORS.dendrogram_to_dr)
                .style('stroke-width', 2);
            })
            .on('mouseout', function () {
              d3.select(this)
                .style('stroke', isSelected ? HIGHLIGHT_COLORS.dr_selection : '#fff')
                .style('stroke-width', isSelected ? 2 : 1);
            });

          // Add text if cell is large enough
          if (cellSize > 30) {
            g.append('text')
              .attr('class', 'heatmap-label')
              .attr('x', (xScale(colClusterId.toString()) || 0) + xScale.bandwidth() / 2)
              .attr('y', (yScale(rowClusterId.toString()) || 0) + yScale.bandwidth() / 2)
              .attr('text-anchor', 'middle')
              .attr('dominant-baseline', 'middle')
              .attr('font-size', '10px')
              .text(value.toFixed(2))
              .style('pointer-events', 'none');
          }
        }
      });
    });

    // Add axes labels if space permits
    if (cellSize > 20) {
      // X axis labels
      g.selectAll('.x-label')
        .data(clusterOrder)
        .enter()
        .append('text')
        .attr('class', 'x-label')
        .attr('x', (d: number) => (xScale(d.toString()) || 0) + xScale.bandwidth() / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('font-size', '11px')
        .text((d: number) => `C${d}`)
        .style('pointer-events', 'none');

      // Y axis labels
      g.selectAll('.y-label')
        .data(clusterOrder)
        .enter()
        .append('text')
        .attr('class', 'y-label')
        .attr('x', -10)
        .attr('y', (d: number) => (yScale(d.toString()) || 0) + yScale.bandwidth() / 2)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '11px')
        .text((d: number) => `C${d}`)
        .style('pointer-events', 'none');
    }

  }, [heatmapData, selection.selectedClusterIds, reverseColorScale, isLoading, margin]);

  return (
    <div ref={containerRef} className="panel cluster-heatmap-container">
      <div className="panel-header">
        Cluster Similarity Heatmap
        <div className="header-controls">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={reverseColorScale}
              onChange={(e) => setReverseColorScale(e.target.checked)}
            />
            <span>Reverse Colors</span>
          </label>
        </div>
      </div>
      <div className="panel-content">
        {isLoading && <div className="loading">Loading heatmap...</div>}
        <svg ref={svgRef} className="heatmap-plot" />
      </div>
    </div>
  );
};

export default ClusterHeatmap;
