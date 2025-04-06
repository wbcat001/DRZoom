import React, { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3';
import { FetchInitialData } from './Fetch';
interface DataPoint {
    index: number;
    x: number;
    y: number;
  }

type PlotProps = {
    fetchInitialDataPoints: () => Promise<DataPoint[]>;
    fetchDataPoints: (indexes: number[]) => Promise<DataPoint[]>;
}


const Plot: React.FC<PlotProps> = ({fetchInitialDataPoints, fetchDataPoints}) => {
    const svgRef = useRef<SVGSVGElement>(null)
    const [dataPoints, setDataPoints] = useState<DataPoint[]>([])


    useEffect(() => {
        fetchInitialDataPoints()
            .then((initialDataPoints) => {
                setDataPoints(initialDataPoints);
            }
            ).catch((error) => {
                console.error('Error fetching data:', error);
            }
            );
    }, [])
    
   

    useEffect(() => {
        const svg = d3.select(svgRef.current);
        const margin = { top:20, right: 20, bottom: 20, left: 20 };
        const width = 800 - margin.left - margin.right;
        const height = 800 - margin.top - margin.bottom;

        const x = d3.scaleLinear().domain([d3.min(dataPoints, (d) => d.x) as number, d3.max(dataPoints, (d) => d.x) as number]).range([0, width]);

        const y = d3.scaleLinear().domain([d3.min(dataPoints, (d) => d.y) as number, d3.max(dataPoints, (d) => d.y) as number]).range([height, 0]);

        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        var tooltip = d3.select("body")
            .append("div")
            .style("position", "absolute")
            .style("z-index", "10")
            .style("visibility", "hidden")
            .style("background", "#ffffff")
            .text("a simple tooltip");
        
        g.selectAll('.dot')
            .data(dataPoints, (d: unknown) => (d as DataPoint).index)
            .enter()
            .append('circle')
            .attr('class', 'dot')
            .attr('r', 3)
            .attr('cx', (d) => x(d.x))
            .attr('cy', (d) => y(d.y))
            .style('fill', 'blue')
            .text(function(d: DataPoint) {
              return d.index;
            })
            // マウスオーバーによるツールチップ表示
            .on("mouseover", function(event, d){
                // tooltip.text(event); 
                return tooltip.style("visibility", "visible")
                            .text("index: "+d.index+" x: "+d.x+" y: "+d.y);
                })
              .on("mousemove", function(event){return tooltip.style("top", (event.pageY-10)+"px").style("left",(event.pageX+10)+"px");})
              .on("mouseout", function(){return tooltip.style("visibility", "hidden");});

        

        // ズーム時の処理
        const zoomed = (event:d3.D3ZoomEvent<SVGSVGElement, unknown>) =>{


            const transform = event.transform;
            
            g.selectAll('.dot')
            .data(dataPoints)
            .transition()
            .attr('cx', (d: DataPoint) => transform.applyX(x(d.x)))
            .attr('cy', (d: DataPoint) => transform.applyY(y(d.y)));

            // 画面範囲内のデータを取得
            const selectedData = dataPoints.filter((d) => {
                const xPos = transform.applyX(x(d.x));
                const yPos = transform.applyY(y(d.y));
                return xPos >= 0 && xPos <= width && yPos >= 0 && yPos <= height;
              });
              
            // 送信用のデータ
            const selectedIndeces = selectedData.map((d) => d.index);

            fetchDataPoints(selectedIndeces).then((newDataPoints) => {
                updateDataPoints(newDataPoints);
            }
            ).catch((error) => {
                console.error('Error fetching data:', error);
            }
            );

        }

        // ズーム動作の定義
        const zoom = d3.zoom<SVGSVGElement, unknown>().scaleExtent([1, 10]).on('zoom', zoomed); // TODO:scaleの段階をどれだけ分けられるか調べる
        svg.call(zoom as any);

        // 新しいデータでのレイアウト更新
        const updateDataPoints = (newDataPoints: DataPoint[]) => {
            g.selectAll('.dot')
            .data(newDataPoints, (d) => (d as DataPoint).index)
            .transition()
            .duration(1000) // Animation duration
            .attr('cx', (d) => x(d.x))
            .attr('cy', (d) => y(d.y))
            .style('fill', 'red');
        }
    }, [])

    return (
            <svg ref={svgRef} width={800} height={800}></svg>
    )
} 

export default Plot;