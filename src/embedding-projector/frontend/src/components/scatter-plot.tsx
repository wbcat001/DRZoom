import React, { useState, useEffect } from 'react';
import * as d3 from 'd3';
import type { ScatterPoint } from '../types';


type ScatterPlotProps = {
    data: ScatterPoint[];
}

const ScatterPlot: React.FC<ScatterPlotProps> = ({ data }) => {
    const ref = React.useRef<SVGSVGElement | null>(null);
    const refDiv = React.useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        if (!ref.current) return;
        if (!refDiv.current) return;

        const svg = d3.select(ref.current);
        svg.selectAll("*").remove(); // Clear previous content

        const width = 800;
        const height = 800;

        svg.attr("width", width).attr("height", height);

        const xScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d.x) as [number, number])
            .range([0, width]);

        const yScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d.y) as [number, number])
            .range([height, 0]);

        var tooltip = d3.select(refDiv.current)
            .append("div")
            .style("opacity", 0)
            .attr("class", "tooltip")
            .style("background-color", "white")
            .style("border", "solid")
            .style("border-width", "1px")
            .style("border-radius", "5px")
            .style("padding", "10px");

        var mouseover = function(event: any, d: ScatterPoint) {
            console.log("mouseover", d);
            tooltip.style("opacity", 1);
        }

        var mousemove = function(event: any, d: ScatterPoint) {
            tooltip
            .html(`X: ${d.x}, Y: ${d.y}${d.label ? `<br>Label: ${d.label}` : ''}`)
            .style("left", (event.x)/2 + "px") // It is important to put the +90: other wise the tooltip is exactly where the point is an it creates a weird effect
            .style("top", (event.y)/2 + "px")
        }

        var mouseleave = function(event: any, d: ScatterPoint) {
            tooltip.style("opacity", 0);
        }

        svg.selectAll("circle")
            .data(data)
            .enter()
            .append("circle")
            .attr("cx", d => xScale(d.x))
            .attr("cy", d => yScale(d.y))
            .attr("r", 5)
            .style("fill", "steelblue")
            .on("mouseover", mouseover)
            .on("mousemove", mousemove)
            .on("mouseleave", mouseleave);

    }, [data]);
    return (<div id="scatter-plot" ref={refDiv}>
            <svg ref={ref} />
    </div>)
}

export default ScatterPlot;