import type { ClusterMetaMap, LinkageMatrix, Point } from '../types';

// Simple Gaussian RNG (Box-Muller)
function randn(): number {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

interface MockDataset {
  points: Point[];
  linkageMatrix: LinkageMatrix;
  clusterMetadata: ClusterMetaMap;
  clusterNames: Record<number, string>;
  clusterWords: Record<number, string[]>;
}

function buildLinkage(clusterSizes: number[]): LinkageMatrix {
  const linkage: LinkageMatrix = [];
  let parentId = clusterSizes.length;
  let prevParent = -1;
  let prevSize = 0;

  for (let i = 0; i < clusterSizes.length - 1; i++) {
    const child1 = i === 0 ? 0 : prevParent;
    const child2 = i + 1;

    const size = i === 0
      ? clusterSizes[0] + clusterSizes[1]
      : prevSize + clusterSizes[i + 1];

    linkage.push({
      child1,
      child2,
      distance: (i + 1) / clusterSizes.length,
      size
    });

    prevParent = parentId;
    prevSize = size;
    parentId += 1;
  }

  return linkage;
}

export function generateMockDataset(pointCount: number, clusterCount = 8): MockDataset {
  const centers: Array<[number, number]> = [];
  for (let i = 0; i < clusterCount; i++) {
    centers.push([Math.random() * 8 - 4, Math.random() * 8 - 4]);
  }

  const baseClusterSize = Math.floor(pointCount / clusterCount);
  const clusterSizes = Array(clusterCount).fill(baseClusterSize);
  for (let i = 0; i < pointCount - baseClusterSize * clusterCount; i++) {
    clusterSizes[i % clusterCount] += 1;
  }

  const points: Point[] = [];
  let index = 0;
  for (let cId = 0; cId < clusterCount; cId++) {
    const [cx, cy] = centers[cId];
    const size = clusterSizes[cId];
    for (let i = 0; i < size; i++) {
      const x = cx + randn() * 0.4;
      const y = cy + randn() * 0.4;
      points.push({
        i: index,
        x,
        y,
        c: cId,
        l: `p_${cId}_${i}`
      });
      index += 1;
    }
  }

  const clusterMetadata: ClusterMetaMap = {};
  const clusterNames: Record<number, string> = {};
  const clusterWords: Record<number, string[]> = {};
  const wordBank = [
    ['vector', 'space', 'embedding'],
    ['graph', 'edge', 'node'],
    ['topic', 'model', 'lda'],
    ['cluster', 'density', 'core'],
    ['metric', 'distance', 'similarity'],
    ['tree', 'branch', 'leaf'],
    ['probability', 'entropy', 'divergence'],
    ['layout', 'force', 'spring'],
    ['sample', 'variance', 'mean'],
    ['search', 'index', 'neighbor']
  ];

  for (let cId = 0; cId < clusterCount; cId++) {
    clusterMetadata[cId] = {
      s: 0.9 - (cId * 0.05),
      h: 1,
      z: clusterSizes[cId]
    };
    clusterNames[cId] = `Cluster ${cId}`;
    clusterWords[cId] = wordBank[cId % wordBank.length];
  }

  const linkageMatrix = buildLinkage(clusterSizes);

  return { points, linkageMatrix, clusterMetadata, clusterNames, clusterWords };
}
