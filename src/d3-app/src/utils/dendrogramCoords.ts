/**
 * Dendrogram coordinate calculation utilities
 * Ported from Python app_labeled.py to JavaScript/TypeScript
 * Computes coordinates for visualizing hierarchical clustering results
 */

import type { LinkageMatrix, DendrogramCoordinates, DendrogramSortMode } from '../types';

/**
 * Node information structure for dendrogram calculation
 */
interface DendrogramNode {
  x: number | null;
  y: number;
  size: number;
  left: number | null;
  right: number | null;
}

/**
 * Compute dendrogram coordinates from linkage matrix
 * Standard version without size weighting
 *
 * @param Z - Linkage matrix where each row is [child1, child2, distance, size]
 * @param nPoints - Number of leaf nodes (points/clusters at base)
 * @param sortBy - Sort mode: 'default' (no sort), 'size' (by cluster size), 'similarity' (by similarity distance)
 * @param similarityMap - Optional map for similarity-based sorting (required when sortBy='similarity')
 * @returns Object containing icoord, dcoord, and leafOrder arrays
 */
export function computeDendrogramCoords(
  Z: LinkageMatrix,
  nPoints: number,
  sortBy: DendrogramSortMode = 'default',
  similarityMap?: Map<string, number>
): DendrogramCoordinates {
  // number of nodes is 2*nPoints-1, but not directly used
  
  // Initialize leaf nodes
  const nodes: DendrogramNode[] = Array(nPoints).fill(null).map(() => ({
    x: null,
    y: 0.0,
    size: 1,
    left: null,
    right: null
  }));

  // Add internal nodes from linkage matrix
  for (let i = 0; i < nPoints - 1; i++) {
    const entry = Z[i];
    nodes.push({
      x: null,
      y: entry.distance,
      size: entry.size,
      left: entry.child1,
      right: entry.child2
    });
  }

  /**
   * Get leaf nodes order with specified sorting strategy
   */
  function getLeafOrderSorted(nodeIdx: number): number[] {
    const node = nodes[nodeIdx];
    if (nodeIdx < nPoints) {
      return [nodeIdx];
    }
    
    if (node.left === null || node.right === null) {
      return [];
    }

    const leftSize = nodes[node.left].size;
    const rightSize = nodes[node.right].size;

    let c1Idx = node.left;
    let c2Idx = node.right;

    // Apply sorting strategy
    if (sortBy === 'size') {
      // Size-based: larger cluster first
      if (leftSize < rightSize) {
        [c1Idx, c2Idx] = [c2Idx, c1Idx];
      }
    } else if (sortBy === 'similarity' && similarityMap) {
      // Similarity-based: recursively calculate average similarity to parent's sibling
      // If this is challenging, use a simpler heuristic: prefer subtree with higher inter-cluster similarity
      // For now, we use size as a proxy when similarity data is incomplete
      // TODO: Implement full similarity-based ordering
      const leftSim = getSubtreeSimilarity(node.left, similarityMap);
      const rightSim = getSubtreeSimilarity(node.right, similarityMap);
      // Higher similarity score → place first (closer together)
      if (leftSim < rightSim) {
        [c1Idx, c2Idx] = [c2Idx, c1Idx];
      }
    }
    // else: default ordering (keep original left-right)

    const orderLeft = getLeafOrderSorted(c1Idx);
    const orderRight = getLeafOrderSorted(c2Idx);
    return [...orderLeft, ...orderRight];
  }

  /**
   * Helper: Calculate average inter-cluster similarity for a subtree
   * Returns average distance (lower = more similar)
   */
  function getSubtreeSimilarity(nodeIdx: number, simMap: Map<string, number>): number {
    const leaves = getLeafOrderSorted(nodeIdx);
    if (leaves.length <= 1) return 0;
    
    // Calculate average pairwise similarity among leaves in this subtree
    let totalDist = 0;
    let count = 0;
    for (let i = 0; i < leaves.length; i++) {
      for (let j = i + 1; j < leaves.length; j++) {
        const key1 = `${leaves[i]}-${leaves[j]}`;
        const key2 = `${leaves[j]}-${leaves[i]}`;
        const dist = simMap.get(key1) ?? simMap.get(key2) ?? 1.0; // default to 1.0 if not found
        totalDist += dist;
        count++;
      }
    }
    return count > 0 ? totalDist / count : 0;
  }

  /**
   * Calculate X coordinates recursively
   */
  function calculateXCoord(nodeIdx: number, leafToX: Map<number, number>): number {
    const node = nodes[nodeIdx];
    
    if (nodeIdx < nPoints) {
      const xCoord = leafToX.get(nodeIdx) ?? 0;
      node.x = xCoord;
      return xCoord;
    }

    if (node.left === null || node.right === null) {
      return 0;
    }

    const xLeft = calculateXCoord(node.left, leafToX);
    const xRight = calculateXCoord(node.right, leafToX);
    const xCoord = (xLeft + xRight) / 2.0;
    node.x = xCoord;
    return xCoord;
  }

  // Get root node index (last node added)
  const rootNodeIdx = 2 * nPoints - 2;
  
  // Get leaf order
  const leafOrder = getLeafOrderSorted(rootNodeIdx);
  
  // Create leaf to x coordinate mapping
  const leafToX = new Map<number, number>();
  for (let i = 0; i < leafOrder.length; i++) {
    leafToX.set(leafOrder[i], 2 * i + 1);
  }

  // Calculate x coordinates for all nodes
  calculateXCoord(rootNodeIdx, leafToX);

  // Generate segment coordinates
  const icoord: number[][] = [];
  const dcoord: number[][] = [];

  for (let i = 0; i < nPoints - 1; i++) {
    const P = nPoints + i;
    const c1 = nodes[P].left;
    const c2 = nodes[P].right;

    if (c1 === null || c2 === null) {
      continue;
    }

    const yP = nodes[P].y;
    const yC1 = nodes[c1].y;
    const yC2 = nodes[c2].y;
    // parent x is implied by children average in layout
    const xC1 = nodes[c1].x ?? 0;
    const xC2 = nodes[c2].x ?? 0;

    icoord.push([xC1, xC1, xC2, xC2]);
    dcoord.push([yC1, yP, yP, yC2]);
  }

  return {
    icoord,
    dcoord,
    leafOrder
  };
}

/**
 * Compute dendrogram coordinates with size weighting
 * Larger clusters get more horizontal space
 *
 * @param Z - Linkage matrix
 * @param nPoints - Number of leaf nodes
 * @param leafSizes - Optional array of sizes for each leaf node
 * @returns Object containing icoord, dcoord, and leafOrder arrays
 */
export function computeDendrogramCoordsWithSize(
  Z: LinkageMatrix,
  nPoints: number,
  leafSizes?: number[]
): DendrogramCoordinates {
  // number of nodes is 2*nPoints-1, but not directly used

  // Initialize leaf nodes
  const nodes: DendrogramNode[] = Array(nPoints).fill(null).map((_, i) => ({
    x: null,
    y: 0.0,
    size: leafSizes && i < leafSizes.length ? leafSizes[i] : 1,
    left: null,
    right: null
  }));

  // Add internal nodes from linkage matrix
  for (let i = 0; i < nPoints - 1; i++) {
    const entry = Z[i];
    nodes.push({
      x: null,
      y: entry.distance,
      size: entry.size,
      left: entry.child1,
      right: entry.child2
    });
  }

  /**
   * Get leaf nodes order sorted by cluster size (descending)
   */
  function getLeafOrderSorted(nodeIdx: number): number[] {
    const node = nodes[nodeIdx];
    if (nodeIdx < nPoints) {
      return [nodeIdx];
    }

    if (node.left === null || node.right === null) {
      return [];
    }

    const leftSize = nodes[node.left].size;
    const rightSize = nodes[node.right].size;

    let c1Idx = node.left;
    let c2Idx = node.right;

    if (leftSize < rightSize) {
      [c1Idx, c2Idx] = [c2Idx, c1Idx];
    }

    const orderLeft = getLeafOrderSorted(c1Idx);
    const orderRight = getLeafOrderSorted(c2Idx);
    return [...orderLeft, ...orderRight];
  }

  /**
   * Calculate X coordinates with size consideration
   */
  function calculateXCoordWithSize(nodeIdx: number): number {
    const node = nodes[nodeIdx];
    
    if (nodeIdx < nPoints) {
      return node.x ?? 0;
    }

    if (node.left === null || node.right === null) {
      return 0;
    }

    const xLeft = calculateXCoordWithSize(node.left);
    const xRight = calculateXCoordWithSize(node.right);
    const xCoord = (xLeft + xRight) / 2.0;
    node.x = xCoord;
    return xCoord;
  }

  // Get root node index
  const rootNodeIdx = 2 * nPoints - 2;

  // Get leaf order
  const leafOrder = getLeafOrderSorted(rootNodeIdx);

  // Assign size-aware widths to leaf nodes
  const minWidth = 0.5;
  const maxWidth = 8.0;

  const leafSizesArr = leafOrder.map((idx) => nodes[idx].size);
  const minSize = Math.min(...leafSizesArr);
  const maxSize = Math.max(...leafSizesArr);

  let currentX = 0.0;

  for (const leafIdx of leafOrder) {
    const leafSize = nodes[leafIdx].size;
    let width: number;

    if (maxSize > minSize) {
      // Log-scale normalization
      const logNormalized =
        (Math.log(leafSize) - Math.log(minSize)) /
        (Math.log(maxSize) - Math.log(minSize));
      width = minWidth + (maxWidth - minWidth) * logNormalized;
    } else {
      // All same size
      width = (minWidth + maxWidth) / 2.0;
    }

    // Set leaf node center position
    nodes[leafIdx].x = currentX + width / 2.0;
    currentX += width;
  }

  // Calculate upper node positions (average of children)
  calculateXCoordWithSize(rootNodeIdx);

  // Generate segment coordinates
  const icoord: number[][] = [];
  const dcoord: number[][] = [];

  for (let i = 0; i < nPoints - 1; i++) {
    const P = nPoints + i;
    const c1 = nodes[P].left;
    const c2 = nodes[P].right;

    if (c1 === null || c2 === null) {
      continue;
    }

    const yP = nodes[P].y;
    const yC1 = nodes[c1].y;
    const yC2 = nodes[c2].y;
    // parent x is implied by children average in layout
    const xC1 = nodes[c1].x ?? 0;
    const xC2 = nodes[c2].x ?? 0;

    icoord.push([xC1, xC1, xC2, xC2]);
    dcoord.push([yC1, yP, yP, yC2]);
  }

  return {
    icoord,
    dcoord,
    leafOrder
  };
}

/**
 * Convert dendrogram coordinates to SVG path segments
 * Each segment represents one line in the dendrogram visualization
 */
export function generateDendrogramSegments(
  coords: DendrogramCoordinates
): Array<[[number, number], [number, number]]> {
  const segments: Array<[[number, number], [number, number]]> = [];

  for (let i = 0; i < coords.icoord.length; i++) {
    const [x1, x2, x3, x4] = coords.icoord[i];
    const [y1, y2, y3, y4] = coords.dcoord[i];

    segments.push([[x1, y1], [x2, y2]]);
    segments.push([[x2, y2], [x3, y3]]);
    segments.push([[x4, y4], [x3, y3]]);
  }

  return segments;
}
