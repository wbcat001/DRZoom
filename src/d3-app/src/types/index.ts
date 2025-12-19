/**
 * Export all types from types module
 */

export * from './data';
export * from './api';
export * from './color';
export * from './state';

export type InteractionMode = 'brush' | 'zoom';
export type TabType = 'point-details' | 'selection-stats' | 'cluster-size' | 'system-log';

export const HIGHLIGHT_COLORS = {
  default: '#4A90E2',
  defaultDimmed: '#B8D4F0',
  drSelection: '#FFA500',
  heatmapClick: '#FF0000',
  heatmapToDr: '#FF1493',
  dendrogramToDr: '#32CD32',
} as const;
