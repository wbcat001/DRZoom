/**
 * Cluster similarity utilities
 * Handles cluster similarity data transformation and lookup
 */

import type { ClusterSimilarityEntry } from '../types';

/**
 * Build a similarity map from array format to Map for O(1) lookup
 * 
 * @param pairs Array of [clusterId1, clusterId2, distance] entries
 * @returns Map with normalized keys (smaller ID first) mapping to distance
 */
export function buildSimilarityMap(pairs: ClusterSimilarityEntry[]): Map<string, number> {
  const similarityMap = new Map<string, number>();
  
  for (const [id1, id2, dist] of pairs) {
    // Normalize key: smaller ID first
    const key = id1 < id2 ? `${id1}-${id2}` : `${id2}-${id1}`;
    similarityMap.set(key, dist);
  }
  
  return similarityMap;
}

/**
 * Get similarity distance between two clusters
 * 
 * @param similarityMap Map created by buildSimilarityMap
 * @param id1 First cluster ID
 * @param id2 Second cluster ID
 * @returns Distance between clusters, or null if not found
 */
export function getSimilarity(similarityMap: Map<string, number>, id1: number, id2: number): number | null {
  // Normalize key to handle both orderings
  const key = id1 < id2 ? `${id1}-${id2}` : `${id2}-${id1}`;
  return similarityMap.get(key) ?? null;
}
