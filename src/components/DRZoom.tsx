import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';


interface DataPoint {
  index: number;
  x: number;
  y: number;
}

// I will add comments


const App: React.FC = () => {
  // Sample time-series data
  const [data, setData] = useState<DataPoint[]>([])

  const svgRef = useRef<SVGSVGElement>(null)
  const [selectedIndexes, setSelectedIndexes] = useState<number[]>([]);
  const [loading, setLoading] = useState(true);


  // Fetch initial data from server
  useEffect(() => {
    const fetchInitialData = async () => {
        try {
            // resposne = requests.post("http://localhost:8000/pca/init",json={"options":"test"}) in python
            const res = await fetch('http://localhost:8000/pca/init', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ options: 'test' }),
    
            });
            const response = await res.json();
            
            console.log('PCA result:', response.data.length);
            console.log("Data point", response.data[0]);

            setData(response.data.map((d: any) => ({ index: d.index, x: d.data[0], y: d.data[1] })));
            // set initial indexes use map((d , index)
            setSelectedIndexes(response.data.map((d:any) => d.index));
           
            
            setLoading(false);
            
    
        }catch (error) {
            console.error('Error sending data:', error);
            setLoading(false);
        }
    }
    fetchInitialData();
    }, []);



  useEffect(() => {
    if (!data || loading) {
      return;
    }
    


    const svg = d3.select(svgRef.current);
    const margin = { top: 20, right: 20, bottom: 20, left: 20 };
    
    const width = 800 - margin.left - margin.right;
    const height = 800 - margin.top - margin.bottom;

    const x = d3.scaleLinear().domain([0, d3.max(data, (d) => d.x) as number]).range([0, width]);
    const y = d3.scaleLinear().domain([d3.min(data, (d) => d.y) as number, d3.max(data, (d) => d.y) as number]).range([height, 0]);

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    g.selectAll('.dot')
      .data(data, (d: unknown) => (d as DataPoint).index)
      .enter()
      .append('circle')
      .attr('class', 'dot')
      .attr('r', 3)
      .attr('cx', (d) => x(d.x))
      .attr('cy', (d) => y(d.y))
      .style('fill', 'blue');

    const zoom = d3.zoom<SVGSVGElement, unknown>().scaleExtent([1, 10]).on('zoom', zoomed);

    
    svg.call(zoom as any);
    

    function zoomed(event: d3.D3ZoomEvent<SVGSVGElement, unknown>) {
      const transform = event.transform;

      g.selectAll('.dot')
        .data(data)
        .transition()
        .attr('cx', (d: DataPoint) => transform.applyX(x(d.x)))
        .attr('cy', (d: DataPoint) => transform.applyY(y(d.y)));

      // Extract the selected data points based on the zoomed area
      const selectedData = data.filter((d) => {
        const xPos = transform.applyX(x(d.x));
        const yPos = transform.applyY(y(d.y));
        return xPos >= 0 && xPos <= width && yPos >= 0 && yPos <= height;
      });
      

      const selectedIndexes = selectedData.map((d) => d.index);
      console.log('Selected Indexes:', selectedData);
      setSelectedIndexes(selectedIndexes); // Store selected indexes in state
    //   console.log('Selected Indexes:', selectedIndexes.length);

      // Send the selected data to the Python API
      sendDataToAPI(selectedIndexes);
    }

    // Send selected data to the API
    async function sendDataToAPI(selectedIndexes: number[]) {
      try {
        // const response = await axios.post('http://localhost:8000/pca', {
        //   filter: selectedIndexes,
        // });
        console.log('Selected Indexes:', selectedIndexes.length);
        const res = await fetch('http://localhost:8000/pca/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ filter: selectedIndexes }),
            });
        // in fastAPI : return {"data":result.tolist()}
        const response = await res.json();


        console.log("update data", response.data[0])
        console.log("update data", response.data[10])
        
        // Update the dots based on PCA result
        updateData(response.data.map((d: any) => ({ index: d.index, x: d.data[0], y: d.data[1] })));
      } catch (error) {
        console.error('Error sending data:', error);
      }
    }

    // Update the dots based on PCA results
    function updateData(newData: DataPoint[]) {
      g.selectAll('.dot')
        .data(newData, (d) => (d as DataPoint).index)
        .transition()
        .duration(1000) // Animation duration
        .attr('cx', (d) => x(d.x))
        .attr('cy', (d) => y(d.y))
        .style('fill', 'red');
    }
    
    setData(data);
  }, [loading]);

  return (
    <div>
      <h2>semantic zoom</h2>
      <svg ref={svgRef} width={800} height={800}></svg>
    </div>
  );
};

export default App;
