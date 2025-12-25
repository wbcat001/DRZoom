/**
 * Lasso selection utility for D3 visualizations
 * Simplified TypeScript implementation inspired by d3-lasso
 */

import * as d3 from 'd3';

export interface LassoOptions {
  /** Container to attach lasso events */
  container: d3.Selection<SVGGElement, unknown, null, undefined>;
  /** SVG element for event handling */
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  /** Items to select (data points) */
  items: Array<{ i: number; x: number; y: number; c: number }>;
  /** Scale functions for coordinates */
  xScale: d3.ScaleLinear<number, number>;
  yScale: d3.ScaleLinear<number, number>;
  /** Callback when selection starts */
  onStart?: () => void;
  /** Callback during drawing */
  onDraw?: (possibleIds: number[]) => void;
  /** Callback when selection ends */
  onEnd?: (selectedIds: number[]) => void;
  /** Whether to ignore noise points */
  ignoreNoise?: boolean;
}

export class LassoSelection {
  private container: d3.Selection<SVGGElement, unknown, null, undefined>;
  private svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private items: Array<{ i: number; x: number; y: number; c: number }>;
  private xScale: d3.ScaleLinear<number, number>;
  private yScale: d3.ScaleLinear<number, number>;
  private onStart?: () => void;
  private onDraw?: (possibleIds: number[]) => void;
  private onEnd?: (selectedIds: number[]) => void;
  private ignoreNoise: boolean;
  
  private path: d3.Selection<SVGPathElement, unknown, null, undefined> | null = null;
  private points: [number, number][] = [];
  private isDrawing: boolean = false;

  constructor(options: LassoOptions) {
    this.container = options.container;
    this.svg = options.svg;
    this.items = options.items;
    this.xScale = options.xScale;
    this.yScale = options.yScale;
    this.onStart = options.onStart;
    this.onDraw = options.onDraw;
    this.onEnd = options.onEnd;
    this.ignoreNoise = options.ignoreNoise ?? true;
  }

  /**
   * Initialize lasso on the container
   */
  public enable(): void {
    // Create lasso path overlay
    this.path = this.container
      .append('path')
      .attr('class', 'lasso-path')
      .style('fill', 'rgba(255, 165, 0, 0.2)')
      .style('stroke', '#FFA500')
      .style('stroke-width', '2px')
      .style('stroke-dasharray', '5,5')
      .style('pointer-events', 'none');

    // Attach events to SVG for better event capture
    this.svg.on('mousedown.lasso', (event: MouseEvent) => {
      // Only start lasso on background (not on points)
      const target = event.target as Element;
      if (target.tagName === 'rect' || target.tagName === 'svg') {
        this.startLasso(event);
      }
    });

    this.svg.on('mousemove.lasso', (event: MouseEvent) => {
      if (this.isDrawing) {
        this.drawLasso(event);
      }
    });

    this.svg.on('mouseup.lasso', () => {
      if (this.isDrawing) {
        this.endLasso();
      }
    });

    this.svg.on('mouseleave.lasso', () => {
      if (this.isDrawing) {
        this.cancelLasso();
      }
    });
  }

  /**
   * Remove lasso from the container
   */
  public disable(): void {
    this.path?.remove();
    this.path = null;
    
    this.svg.on('mousedown.lasso', null);
    this.svg.on('mousemove.lasso', null);
    this.svg.on('mouseup.lasso', null);
    this.svg.on('mouseleave.lasso', null);
  }

  /**
   * Start lasso selection
   */
  private startLasso(event: MouseEvent): void {
    event.preventDefault();
    event.stopPropagation();
    
    this.isDrawing = true;
    const [x, y] = d3.pointer(event, this.container.node());
    this.points = [[x, y]];
    
    this.updatePath();
    this.onStart?.();
  }

  /**
   * Draw lasso path
   */
  private drawLasso(event: MouseEvent): void {
    const [x, y] = d3.pointer(event, this.container.node());
    this.points.push([x, y]);
    
    // Limit points to prevent performance issues
    if (this.points.length > 500) {
      this.points = this.points.filter((_, i) => i % 2 === 0);
    }
    
    this.updatePath();
    
    // Calculate possible selections
    const possibleIds = this.getPointsInPolygon();
    this.onDraw?.(possibleIds);
  }

  /**
   * End lasso selection
   */
  private endLasso(): void {
    this.isDrawing = false;
    
    // Close the polygon
    if (this.points.length > 2) {
      this.points.push(this.points[0]);
      this.updatePath();
    }
    
    // Get final selected points
    const selectedIds = this.getPointsInPolygon();
    this.onEnd?.(selectedIds);
    
    // Clear lasso
    this.clear();
  }

  /**
   * Cancel lasso selection
   */
  private cancelLasso(): void {
    this.isDrawing = false;
    this.clear();
  }

  /**
   * Update lasso path visualization
   */
  private updatePath(): void {
    if (!this.path || this.points.length === 0) return;
    
    const lineGenerator = d3.line<[number, number]>();
    const pathData = lineGenerator(this.points);
    
    this.path.attr('d', pathData ? `${pathData}Z` : null);
  }

  /**
   * Clear lasso path
   */
  private clear(): void {
    this.points = [];
    this.path?.attr('d', null);
  }

  /**
   * Get points inside the lasso polygon
   */
  private getPointsInPolygon(): number[] {
    if (this.points.length < 3) return [];
    
    const selectedIds: number[] = [];
    
    for (const item of this.items) {
      // Skip noise points if configured
      if (this.ignoreNoise && item.c === -1) continue;
      
      const px = this.xScale(item.x);
      const py = this.yScale(item.y);
      
      // Check if point is inside polygon
      if (this.isPointInPolygon([px, py], this.points)) {
        selectedIds.push(item.i);
      }
    }
    
    return selectedIds;
  }

  /**
   * Point-in-polygon test using ray casting algorithm
   */
  private isPointInPolygon(point: [number, number], polygon: [number, number][]): boolean {
    const [x, y] = point;
    let inside = false;
    
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const [xi, yi] = polygon[i];
      const [xj, yj] = polygon[j];
      
      const intersect = ((yi > y) !== (yj > y)) &&
        (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
      
      if (intersect) inside = !inside;
    }
    
    return inside;
  }
}
