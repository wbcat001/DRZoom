/**
 * Data types for HDBSCAN Cluster Explorer
 * These types represent the core data structures used throughout the application
 */

/**
 * Individual data point in the DR (Dimensionality Reduction) scatter plot
 * @property i - Unique point index
 * @property x - X coordinate in reduced dimension space
 * @property y - Y coordinate in reduced dimension space
 * @property c - Cluster ID this point belongs to
 * @property l - Label/name of this point
 */
export interface Point {
  i: number;      // Index
  x: number;      // X coordinate
  y: number;      // Y coordinate
  c: number;      // Cluster ID
  l: string;      // Label
}

/**
 * Linkage matrix entry (scipy-compatible format)
 * Represents a merge operation in hierarchical clustering
 * @property child1 - Index of first child cluster
 * @property child2 - Index of second child cluster
 * @property distance - Distance at which merge occurred
 * @property size - Total number of points in merged cluster
 */
export interface LinkageMatrixEntry {
  child1: number;
  child2: number;
  distance: number;
  size: number;
}

export type LinkageMatrix = LinkageMatrixEntry[];

/**
 * Cluster metadata information
 * @property s - Stability score (0-1)
 * @property h - Strahler number (hierarchy depth measure)
 * @property z - Cluster size (number of points)
 */
export interface ClusterMetadata {
  s: number;      // Stability
  h: number;      // Strahler number
  z: number;      // Size
}

export type ClusterMetaMap = Record<number, ClusterMetadata>;

/**
 * Similarity matrix entry
 * Represents distance/similarity between two clusters
 */
export interface SimilarityData {
  matrix: number[][];
  clusterOrder: number[];
  metric: 'kl_divergence' | 'bhattacharyya_coefficient' | 'mahalanobis_distance';
}

/**
 * Cluster information with metadata and words
 */
export interface ClusterInfo {
  id: number;
  label: string;
  words: string[];
  stability: number;
  strahler: number;
  size: number;
}

/**
 * Point detail information
 */
export interface PointDetail {
  id: number;
  label: string;
  coordinates: { x: number; y: number };
  clusterId: number;
  nearbyPoints: Point[];
}

/**
 * Selection statistics
 */
export interface SelectionStats {
  selectedPointCount: number;
  selectedClusterCount: number;
  selectedClusters: number[];
  containmentRatio: Record<number, number>; // cluster_id -> containment_ratio
}

/**
 * Dendrogram coordinate information
 */
export interface DendrogramCoordinates {
  icoord: number[][];  // X coordinates of line segments
  dcoord: number[][];  // Y coordinates of line segments
  leafOrder: number[]; // Order of leaves
}

/**
 * Complete initial data payload from backend
 */
export interface InitialDataPayload {
  points: Point[];
  zMatrix: LinkageMatrix;
  clusterMeta: ClusterMetaMap;
  clusterNames: Record<number, string>;
  clusterWords: Record<number, string[]>;
  datasetInfo: {
    name: string;
    pointCount: number;
    clusterCount: number;
    description: string;
  };
}
