import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import type { InteractionMode } from '../../types';
import './DRVisualization.css';

const DRVisualization: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [interactionMode, setInteractionMode] = useState<InteractionMode>('zoom');

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const svg = d3.select(svgRef.current);
    const container = containerRef.current;
    
    // Clear previous content
    svg.selectAll('*').remove();

    // Get container dimensions
    const rect = container.getBoundingClientRect();
    const margin = { top: 20, right: 20, bottom: 40, left: 40 };
    const width = rect.width - margin.left - margin.right;
    const height = rect.height - margin.top - margin.bottom;

    // Set SVG dimensions
    svg
      .attr('width', rect.width)
      .attr('height', rect.height);

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Sample data for demo
    const sampleData = Array.from({ length: 100 }, (_, i) => ({
      id: i,
      x: Math.random() * width,
      y: Math.random() * height,
      cluster: Math.floor(Math.random() * 5)
    }));

    // Color scale
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    // Draw points
    g.selectAll('.data-point')
      .data(sampleData)
      .enter()
      .append('circle')
      .attr('class', 'data-point')
      .attr('cx', d => d.x)
      .attr('cy', d => d.y)
      .attr('r', 3)
      .style('fill', d => colorScale(d.cluster.toString()))
      .style('stroke', 'white')
      .style('stroke-width', 1)
      .on('mouseover', function() {
        d3.select(this).attr('r', 5);
      })
      .on('mouseout', function() {
        d3.select(this).attr('r', 3);
      });

    // Setup zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        g.attr('transform', `translate(${margin.left + event.transform.x},${margin.top + event.transform.y}) scale(${event.transform.k})`);
      });

    // Setup brush behavior
    const brush = d3.brush()
      .on('start brush end', (event) => {
        if (!event.sourceEvent) return;
        const selection = event.selection;
        if (selection) {
          // Handle brush selection
          g.selectAll('.data-point')
            .classed('selected', (d: any) => {
              const [x0, y0] = selection[0];
              const [x1, y1] = selection[1];
              return d.x >= x0 && d.x <= x1 && d.y >= y0 && d.y <= y1;
            });
        }
      });

    // Apply interaction mode
    if (interactionMode === 'zoom') {
      svg.call(zoom);
      svg.select('.brush-layer').remove();
    } else {
      svg.on('.zoom', null);
      svg.append('g')
        .attr('class', 'brush-layer')
        .call(brush);
    }

  }, [interactionMode]);

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="header-content">
          <h4>DR Visualization</h4>
          <div className="interaction-controls">
            <label className="mode-toggle">
              <input
                type="radio"
                name="interaction-mode"
                value="brush"
                checked={interactionMode === 'brush'}
                onChange={() => setInteractionMode('brush')}
              />
              <span>Brush Selection</span>
            </label>
            <label className="mode-toggle">
              <input
                type="radio"
                name="interaction-mode"
                value="zoom"
                checked={interactionMode === 'zoom'}
                onChange={() => setInteractionMode('zoom')}
              />
              <span>Zoom/Pan</span>
            </label>
          </div>
        </div>
      </div>
      
      <div className="panel-content" ref={containerRef}>
        <svg ref={svgRef} className="dr-svg"></svg>
      </div>
    </div>
  );
};

export default DRVisualization;
