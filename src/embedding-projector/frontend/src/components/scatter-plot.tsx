import React, { useState, useEffect } from 'react';
import * as d3 from 'd3';
import type { ScatterPoint } from '../types';


type ScatterPlotProps = {
    data: ScatterPoint[];
    onClick?: (point: ScatterPoint) => void;
    selectedIndices?: number[];
}

const ScatterPlot: React.FC<ScatterPlotProps> = ({ data, onClick, selectedIndices = [0, 1, 2] }) => {
    const ref = React.useRef<SVGSVGElement | null>(null);
    const refDiv = React.useRef<HTMLDivElement | null>(null);
    const tooltipRef = React.useRef<HTMLDivElement | null>(null);

    // tooltip
    useEffect(() => {
        if (!tooltipRef.current) return;
        const tooltip = d3.select(tooltipRef.current)
            .append("div")
            .style("position", "absolute")
            .style("visibility", "hidden")
            .attr("class", "tooltip")
            .style("background-color", "red")
            .style("border", "solid")
            .style("border-width", "1px")
            .style("border-radius", "5px")
            .style("padding", "10px")
            .style("z-index", 1000)
            .style("pointer-events", "none");

            tooltipRef.current = tooltip.node() as HTMLDivElement;

        return () => {
            tooltip.remove();
            tooltipRef.current = null;
        }

    }, [])

    useEffect(() => {
        if (!ref.current) return;
        if (!refDiv.current) return;

        const tooltip = d3.select(tooltipRef.current);

        const svg = d3.select(ref.current);
        svg.selectAll("*").remove(); // Clear previous content
        const parentDiv = d3.select(refDiv.current);

        const width = parseInt(parentDiv.style("width")) || 800;
        // const height = parseInt(parentDiv.style("height")) || 800;
        const height = width
        console.log("width", width, "height", height);
        const margin = { top: 20, right: 30, bottom: 30, left: 40 };
        svg.attr("width", width).attr("height", height);

        const xScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d.x) as [number, number])
            .range([margin.left, width - margin.right]);

        const yScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d.y) as [number, number])
            .range([height - margin.bottom, margin.top]);


        var mouseover = function(event: any, d: ScatterPoint) {
        
            tooltip.style("visibility", "visible")
            .html(`X: ${d.x}, Y: ${d.y}${d.label ? `<br>Label: ${d.label}` : ''}`)
        }

        var mousemove = function(event: any, d: ScatterPoint) {
            tooltip
            // .html(`X: ${d.x}, Y: ${d.y}${d.label ? `<br>Label: ${d.label}` : ''}`)
            .style("left", (event.pageX)/2 + "px") // It is important to put the +90: other wise the tooltip is exactly where the point is an it creates a weird effect
            .style("top", (event.pageY)/2 + "px")
        }

        var mouseleave = function(event: any, d: ScatterPoint) {
            console.log("mouseleave", d);
            tooltip.style("visibility", "hidden");
        }

        var handleClick = function(event: any, d: ScatterPoint) {
            console.log("click", d);
            if (onClick) {
                onClick(d);
            }
        }

        const circles = svg.selectAll<SVGCircleElement, ScatterPoint>("circle")
        .data(data, (d:ScatterPoint) => d.index)

        // Enter
        const enterCircles = circles.enter()
        .append("circle")
        .attr("cx", d => xScale(d.x))
        .attr("cy", d => yScale(d.y))
        .attr("r", 4)
        .on("mouseover", mouseover)
        .on("mousemove", mousemove)
        .on("mouseleave", mouseleave)
        .on("click", handleClick)
        .style("fill", "steelblue")
        // .style("stroke", (d, i) => selectedIndices.includes(i) ? "#d63031" : "#333")
        // .style("stroke-width", (d, i) => selectedIndices.includes(i) ? 4 : 0);

        const allCircles = enterCircles.merge(circles);


        // Update
        allCircles.transition()
        .duration(100)
        .attr("r", (d, i) => selectedIndices.includes(i) ? 6 : 4)
        .style("stroke", (d, i) => selectedIndices.includes(i) ? "#d63031" : "#333")
        .style("stroke-width", (d, i) => selectedIndices.includes(i) ? 5 : 0);

        circles.exit()
        .transition()
        .duration(300)
        .attr("r", 0)
        .remove();


        // svg.selectAll("circle")
        //     .data(data)
        //     .enter()
        //     .append("circle")
        //     .attr("cx", d => xScale(d.x))
        //     .attr("cy", d => yScale(d.y))
        //     .attr("r", 3)
        //     .style("fill", "steelblue")
        //     .style("stroke", (d, i) => selectedIndices.includes(i) ? "#ff6b6b" : "steelblue")
        //     .style("stroke-width", (d, i) => selectedIndices.includes(i) ? 3 : 0)
        //     .style("stroke-opacity", 1)
        //     .on("mouseover", mouseover)
        //     .on("mousemove", mousemove)
        //     .on("mouseleave", mouseleave)
        //     .on("click", handleClick);

    }, [data, selectedIndices]);
    return (<div id="scatter-plot" style={{ position: "relative" }} ref={refDiv} className="w-full h-full">
        <div className="tooltip" ref={tooltipRef} style={{position: "relative"}}></div>
            <svg ref={ref} />
    </div>)
}

export default ScatterPlot;