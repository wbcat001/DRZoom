{
  const svgNode =  d3.select(svg`<svg width=300 height=200></svg>`)
  
  const circles = svgNode.selectAll("circle")
          .data(data)
          .enter()
          .append("circle")
          .attr("r", 7)
          .attr("cx", d=>d[1])
          .attr("cy", d=>d[2])
          .attr('fill','steelblue')
  
  // ----------------   LASSO STUFF . ----------------
  
  
  
  var lasso_start = function() {
    console.log('start')
      lasso.items()
          .attr("r",7) 
          .classed("not_possible",true)
          .classed("selected",false);
  };

  var lasso_draw = function() {
    console.log('draw')
      lasso.possibleItems()
          .classed("not_possible",false)
          .classed("possible",true);
      lasso.notPossibleItems()
          .classed("not_possible",true)
          .classed("possible",false);
  };

  var lasso_end = function() {
      console.log('end')
      lasso.items()
          .classed("not_possible",false)
          .classed("possible",false);
      lasso.selectedItems()
          .classed("selected",true)
          .attr("r",7);
      lasso.notSelectedItems()
          .attr("r",3.5);
  };
  
  const lasso = d3.lasso()
          .closePathDistance(305) 
          .closePathSelect(true) 
          .targetArea(svgNode)
          .items(circles) 
          .on("start",lasso_start) 
          .on("draw",lasso_draw) 
          .on("end",lasso_end); 

  svgNode.call(lasso);
  
  return svgNode.node();
}