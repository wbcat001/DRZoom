/**
 * Color and styling utilities
 * Provides functions for determining colors based on selection state
 */

import type { HighlightState } from '../types/color';

/**
 * Determine if a cluster should be highlighted based on current selections
 */
export function determineClusterHighlight(
  clusterId: number,
  selectedClusterIds: Set<number>,
  heatmapClickedClusters: Set<number>,
  dendrogramHovered: number | null
): HighlightState {
  return {
    isDRSelected: selectedClusterIds.has(clusterId),
    isHeatmapClicked: heatmapClickedClusters.has(clusterId),
    isDendrogramSelected: dendrogramHovered === clusterId,
    isSearchResult: false,
    isHovered: false
  };
}

/**
 * Determine if a point should be highlighted based on current selections
 */
export function determinePointHighlight(
  pointId: number,
  point: any,
  selectedClusterIds: Set<number>,
  selectedPointIds: Set<number>,
  heatmapClickedClusters: Set<number>,
  dendrogramHovered: number | null,
  searchResultPointIds: Set<number> = new Set()
): HighlightState {
  const isPointSelected = selectedPointIds.has(pointId);
  const isClusterSelected = selectedClusterIds.has(point.c);
  const isClusterHeatmapClicked = heatmapClickedClusters.has(point.c);
  const isClusterDendrogramHovered = dendrogramHovered === point.c;
  const isSearchResult = searchResultPointIds.has(pointId);

  return {
    isDRSelected: isPointSelected || isClusterSelected,
    isHeatmapClicked: isClusterHeatmapClicked,
    isDendrogramSelected: isClusterDendrogramHovered,
    isSearchResult: isSearchResult,
    isHovered: false
  };
}

/**
 * Check if any selection is active
 */
export function isAnySelectionActive(
  selectedClusterIds: Set<number>,
  selectedPointIds: Set<number>,
  heatmapClickedClusters: Set<number>,
  dendrogramHovered: number | null,
  searchResultPointIds: Set<number> = new Set()
): boolean {
  return (
    selectedClusterIds.size > 0 ||
    selectedPointIds.size > 0 ||
    heatmapClickedClusters.size > 0 ||
    dendrogramHovered !== null ||
    searchResultPointIds.size > 0
  );
}

/**
 * Calculate scale factors for SVG rendering
 */
export interface ScaleFactors {
  xScale: (value: number) => number;
  yScale: (value: number) => number;
  radiusScale: (size: number) => number;
}

export function createScaleFactors(
  data: any[],
  width: number,
  height: number,
  padding: number = 40
): ScaleFactors {
  if (data.length === 0) {
    return {
      xScale: (x) => x,
      yScale: (y) => y,
      radiusScale: (_size) => 3
    };
  }

  const xValues = data.map((d) => d.x);
  const yValues = data.map((d) => d.y);

  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);

  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  const plotWidth = width - 2 * padding;
  const plotHeight = height - 2 * padding;

  return {
    xScale: (x: number) => padding + ((x - xMin) / xRange) * plotWidth,
    yScale: (y: number) => height - padding - ((y - yMin) / yRange) * plotHeight,
    radiusScale: (size: number) => Math.max(2, Math.min(6, 3 + Math.log(size + 1) / 2))
  };
}

/**
 * Format numbers for display
 */
export function formatNumber(num: number, decimals: number = 2): string {
  return num.toFixed(decimals);
}

/**
 * Format cluster size with abbreviations
 */
export function formatClusterSize(size: number): string {
  if (size >= 1000000) {
    return `${(size / 1000000).toFixed(1)}M`;
  }
  if (size >= 1000) {
    return `${(size / 1000).toFixed(1)}K`;
  }
  return size.toString();
}

/**
 * Get cluster label or default name
 */
export function getClusterLabel(
  clusterId: number,
  clusterNames: Record<number, string>
): string {
  return clusterNames[clusterId] || `Cluster ${clusterId}`;
}

/**
 * Get top N words from cluster word list
 */
export function getTopWords(
  clusterId: number,
  clusterWords: Record<number, string[]>,
  topN: number = 5
): string[] {
  const words = clusterWords[clusterId] || [];
  return words.slice(0, topN);
}
