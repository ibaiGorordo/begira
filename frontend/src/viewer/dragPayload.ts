import type { ElementInfo } from './api'

export const HIERARCHY_DRAG_MIME = 'application/x-begira-hierarchy-item'

export type HierarchyElementDragPayload = {
  source: 'hierarchy-element'
  elementId: string
  elementType: ElementInfo['type']
}

export function buildHierarchyElementDragPayload(element: Pick<ElementInfo, 'id' | 'type'>): HierarchyElementDragPayload {
  return {
    source: 'hierarchy-element',
    elementId: element.id,
    elementType: element.type,
  }
}

export function parseHierarchyElementDragPayload(raw: string): HierarchyElementDragPayload | null {
  try {
    const parsed = JSON.parse(raw) as Partial<HierarchyElementDragPayload>
    if (parsed.source !== 'hierarchy-element') return null
    if (typeof parsed.elementId !== 'string' || parsed.elementId.length === 0) return null
    if (parsed.elementType !== 'pointcloud' && parsed.elementType !== 'gaussians' && parsed.elementType !== 'camera' && parsed.elementType !== 'image') {
      return null
    }
    return {
      source: 'hierarchy-element',
      elementId: parsed.elementId,
      elementType: parsed.elementType,
    }
  } catch {
    return null
  }
}
