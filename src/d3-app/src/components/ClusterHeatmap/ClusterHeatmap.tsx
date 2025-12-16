import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import './ClusterHeatmap.css';

const ClusterHeatmap: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [similarityMetric, setSimilarityMetric] = useState('kl_divergence');
  const [reverseColorScale, setReverseColorScale] = useState(false);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const svg = d3.select(svgRef.current);
    const container = containerRef.current;
    
    // Clear previous content
    svg.selectAll('*').remove();

    // Get container dimensions
    const rect = container.getBoundingClientRect();
    const margin = { top: 60, right: 60, bottom: 60, left: 60 };
    const width = rect.width - margin.left - margin.right;
    const height = rect.height - margin.top - margin.bottom;

    // Set SVG dimensions
    svg
      .attr('width', rect.width)
      .attr('height', rect.height);

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Sample data - 5x5 similarity matrix
    const clusters = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'];
    const similarities = [
      [1.0, 0.3, 0.2, 0.1, 0.15],
      [0.3, 1.0, 0.4, 0.25, 0.2],
      [0.2, 0.4, 1.0, 0.35, 0.3],
      [0.1, 0.25, 0.35, 1.0, 0.45],
      [0.15, 0.2, 0.3, 0.45, 1.0]
    ];

    const cellSize = Math.min(width / clusters.length, height / clusters.length);

    // Color scale
    const colorScale = reverseColorScale 
      ? d3.scaleSequential(d3.interpolateReds).domain([1, 0])
      : d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);

    // Create scales
    const xScale = d3.scaleBand()
      .domain(clusters)
      .range([0, clusters.length * cellSize])
      .padding(0.05);

    const yScale = d3.scaleBand()
      .domain(clusters)
      .range([0, clusters.length * cellSize])
      .padding(0.05);

    // Draw cells
    clusters.forEach((rowCluster, i) => {
      clusters.forEach((colCluster, j) => {
        const value = similarities[i][j];
        
        g.append('rect')
          .attr('class', 'heatmap-cell')
          .attr('x', xScale(colCluster) || 0)
          .attr('y', yScale(rowCluster) || 0)
          .attr('width', xScale.bandwidth())
          .attr('height', yScale.bandwidth())
          .style('fill', colorScale(value))
          .style('stroke', 'white')
          .style('stroke-width', 2)
          .style('cursor', 'pointer')
          .on('click', function() {
            d3.select(this).style('stroke', d3.select(this).style('stroke') === 'rgb(255, 107, 53)' ? 'white' : '#ff6b35');
            console.log(`Clicked cell: ${rowCluster} vs ${colCluster}, similarity: ${value}`);
          })
          .on('mouseover', function(event) {
            // Tooltip
            const tooltip = d3.select('body').append('div')
              .attr('class', 'tooltip')
              .style('opacity', 0)
              .style('position', 'absolute')
              .style('background', 'rgba(0, 0, 0, 0.8)')
              .style('color', 'white')
              .style('padding', '8px')
              .style('border-radius', '4px')
              .style('font-size', '12px')
              .style('pointer-events', 'none');

            tooltip.transition()
              .duration(200)
              .style('opacity', .9);
            
            tooltip.html(`${rowCluster} vs ${colCluster}<br/>Similarity: ${value.toFixed(3)}`)
              .style('left', (event.pageX + 10) + 'px')
              .style('top', (event.pageY - 28) + 'px');
            
            d3.select(this).style('stroke-width', 4);
          })
          .on('mouseout', function() {
            d3.selectAll('.tooltip').remove();
            d3.select(this).style('stroke-width', 2);
          });

        // Add text labels for values
        g.append('text')
          .attr('x', (xScale(colCluster) || 0) + xScale.bandwidth() / 2)
          .attr('y', (yScale(rowCluster) || 0) + yScale.bandwidth() / 2)
          .attr('text-anchor', 'middle')
          .attr('alignment-baseline', 'middle')
          .style('font-size', '10px')
          .style('font-weight', 'bold')
          .style('fill', value > 0.5 ? 'white' : 'black')
          .style('pointer-events', 'none')
          .text(value.toFixed(2));
      });
    });

    // Add axis labels
    g.selectAll('.x-label')
      .data(clusters)
      .enter()
      .append('text')
      .attr('class', 'x-label')
      .attr('x', d => (xScale(d) || 0) + xScale.bandwidth() / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('font-weight', '500')
      .text(d => d);

    g.selectAll('.y-label')
      .data(clusters)
      .enter()
      .append('text')
      .attr('class', 'y-label')
      .attr('x', -10)
      .attr('y', d => (yScale(d) || 0) + yScale.bandwidth() / 2)
      .attr('text-anchor', 'end')
      .attr('alignment-baseline', 'middle')
      .style('font-size', '12px')
      .style('font-weight', '500')
      .text(d => d);

  }, [similarityMetric, reverseColorScale]);

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="heatmap-header">
          <h4>Cluster Similarity</h4>
          <div className="heatmap-controls">
            <select 
              className="metric-select"
              value={similarityMetric}
              onChange={(e) => setSimilarityMetric(e.target.value)}
            >
              <option value="kl_divergence">KL Divergence</option>
              <option value="bhattacharyya">Bhattacharyya</option>
              <option value="mahalanobis">Mahalanobis</option>
            </select>
            
            <label className="reverse-label">
              <input
                type="checkbox"
                checked={reverseColorScale}
                onChange={(e) => setReverseColorScale(e.target.checked)}
              />
              <span>Reverse</span>
            </label>
          </div>
        </div>
      </div>
      
      <div className="panel-content" ref={containerRef}>
        <svg ref={svgRef} className="heatmap-svg"></svg>
      </div>
    </div>
  );
};

export default ClusterHeatmap;
