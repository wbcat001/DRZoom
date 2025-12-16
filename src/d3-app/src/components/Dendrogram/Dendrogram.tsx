import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import './Dendrogram.css';

interface DendrogramData {
  nodes: Array<{ id: string; x: number; y: number; cluster_size?: number }>;
  links: Array<{ source: string; target: string; distance: number }>;
}

const Dendrogram: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [proportionalWidth, setProportionalWidth] = useState(false);
  const [showLabels, setShowLabels] = useState(false);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const svg = d3.select(svgRef.current);
    const container = containerRef.current;
    
    // Clear previous content
    svg.selectAll('*').remove();

    // Get container dimensions
    const rect = container.getBoundingClientRect();
    const margin = { top: 40, right: 20, bottom: 40, left: 40 };
    const width = rect.width - margin.left - margin.right;
    const height = rect.height - margin.top - margin.bottom;

    // Set SVG dimensions
    svg
      .attr('width', rect.width)
      .attr('height', rect.height);

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Sample dendrogram data
    const sampleData: DendrogramData = {
      nodes: [
        { id: 'root', x: width / 2, y: 0 },
        { id: 'cluster1', x: width * 0.25, y: height * 0.3, cluster_size: 25 },
        { id: 'cluster2', x: width * 0.75, y: height * 0.3, cluster_size: 30 },
        { id: 'leaf1', x: width * 0.15, y: height * 0.6, cluster_size: 10 },
        { id: 'leaf2', x: width * 0.35, y: height * 0.6, cluster_size: 15 },
        { id: 'leaf3', x: width * 0.65, y: height * 0.6, cluster_size: 12 },
        { id: 'leaf4', x: width * 0.85, y: height * 0.6, cluster_size: 18 }
      ],
      links: [
        { source: 'root', target: 'cluster1', distance: 0.8 },
        { source: 'root', target: 'cluster2', distance: 0.7 },
        { source: 'cluster1', target: 'leaf1', distance: 0.4 },
        { source: 'cluster1', target: 'leaf2', distance: 0.3 },
        { source: 'cluster2', target: 'leaf3', distance: 0.5 },
        { source: 'cluster2', target: 'leaf4', distance: 0.2 }
      ]
    };

    // Create node lookup
    const nodeMap = new Map(sampleData.nodes.map(d => [d.id, d]));

    // Draw links
    g.selectAll('.dendrogram-link')
      .data(sampleData.links)
      .enter()
      .append('path')
      .attr('class', 'dendrogram-link')
      .attr('d', d => {
        const source = nodeMap.get(d.source);
        const target = nodeMap.get(d.target);
        if (!source || !target) return '';
        
        // Create L-shaped path (typical dendrogram style)
        const midY = (source.y + target.y) / 2;
        return `M${source.x},${source.y} L${source.x},${midY} L${target.x},${midY} L${target.x},${target.y}`;
      })
      .style('fill', 'none')
      .style('stroke', '#333')
      .style('stroke-width', d => proportionalWidth ? d.distance * 3 : 2);

    // Draw nodes
    g.selectAll('.dendrogram-node')
      .data(sampleData.nodes)
      .enter()
      .append('circle')
      .attr('class', 'dendrogram-node')
      .attr('cx', d => d.x)
      .attr('cy', d => d.y)
      .attr('r', d => d.cluster_size ? Math.sqrt(d.cluster_size) : 4)
      .style('fill', '#4A90E2')
      .style('stroke', 'white')
      .style('stroke-width', 2)
      .on('click', function(_, d) {
        d3.select(this).style('fill', d3.select(this).style('fill') === 'rgb(255, 0, 0)' ? '#4A90E2' : '#ff0000');
        console.log('Node clicked:', d.id);
      })
      .on('mouseover', function() {
        d3.select(this).style('stroke', '#007bff').style('stroke-width', 3);
      })
      .on('mouseout', function() {
        d3.select(this).style('stroke', 'white').style('stroke-width', 2);
      });

    // Draw labels if enabled
    if (showLabels) {
      g.selectAll('.node-label')
        .data(sampleData.nodes)
        .enter()
        .append('text')
        .attr('class', 'node-label')
        .attr('x', d => d.x)
        .attr('y', d => d.y - 10)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('font-family', 'Arial, sans-serif')
        .text(d => d.id);
    }

  }, [proportionalWidth, showLabels]);

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="dendrogram-header">
          <h4>Cluster Dendrogram</h4>
          
          <div className="dendrogram-controls">
            <div className="control-group">
              <label className="radio-label">
                <input
                  type="radio"
                  name="dendro-mode"
                  value="node"
                  defaultChecked
                />
                <span>Node Selection</span>
              </label>
            </div>
            
            <div className="control-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={proportionalWidth}
                  onChange={(e) => setProportionalWidth(e.target.checked)}
                />
                <span>Proportional Width</span>
              </label>
            </div>
            
            <div className="control-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={showLabels}
                  onChange={(e) => setShowLabels(e.target.checked)}
                />
                <span>Show Labels</span>
              </label>
            </div>
          </div>
        </div>
      </div>
      
      <div className="panel-content" ref={containerRef}>
        <svg ref={svgRef} className="dendrogram-svg"></svg>
      </div>
    </div>
  );
};

export default Dendrogram;
