export type ElementType = 'pointcloud' | 'gaussians'

export type ElementInfo = {
  id: string
  type: ElementType
  name: string
  revision: number
  createdAt?: number
  bounds?: { min: [number, number, number]; max: [number, number, number] }
  summary?: Record<string, unknown>
}

export type PointCloudElementMeta = {
  id: string
  type: 'pointcloud'
  name: string
  revision: number
  pointCount: number
  pointSize: number
  bounds: { min: [number, number, number]; max: [number, number, number] }
  endianness: 'little' | 'big'
  interleaved: boolean
  bytesPerPoint: number
  schema: {
    position: { type: 'float32'; components: 3 }
    color?: { type: 'uint8'; components: 3; normalized: true }
  }
  payloads: { points: { url: string; contentType: string } }
}

export type GaussianSplatElementMeta = {
  id: string
  type: 'gaussians'
  name: string
  revision: number
  count: number
  pointSize: number
  bounds: { min: [number, number, number]; max: [number, number, number] }
  endianness: 'little' | 'big'
  bytesPerGaussian: number
  schema: {
    position: { type: 'float32'; components: 3 }
    sh0: { type: 'float32'; components: 3 }
    opacity: { type: 'float32'; components: 1 }
    scale: { type: 'float32'; components: 3 }
    rotation: { type: 'float32'; components: 4 }
  }
  payloads: { gaussians: { url: string; contentType: string } }
}

export type Events = {
  globalRevision: number
}

export type ViewerSettings = {
  coordinateConvention?: string
}

export type UpdatePointCloudSettingsRequest = {
  pointSize?: number
}

export async function fetchEvents(): Promise<Events> {
  const res = await fetch('/api/events')
  if (!res.ok) throw new Error(`Failed to get events: ${res.status} ${res.statusText}`)
  return (await res.json()) as Events
}

export async function fetchElements(): Promise<ElementInfo[]> {
  const res = await fetch('/api/elements')
  if (!res.ok) throw new Error(`Failed to list elements: ${res.status} ${res.statusText}`)
  return (await res.json()) as ElementInfo[]
}

export async function fetchElementMeta(elementId: string): Promise<any> {
  const res = await fetch(`/api/elements/${encodeURIComponent(elementId)}/meta`)
  if (!res.ok) throw new Error(`Failed to get meta: ${res.status} ${res.statusText}`)
  return await res.json()
}

export async function fetchPointCloudElementMeta(elementId: string): Promise<PointCloudElementMeta> {
  const meta = (await fetchElementMeta(elementId)) as PointCloudElementMeta
  if (meta.type !== 'pointcloud') throw new Error(`Element ${elementId} is not a pointcloud (type=${(meta as any).type})`)
  return meta
}

export async function fetchGaussianElementMeta(elementId: string): Promise<GaussianSplatElementMeta> {
  const meta = (await fetchElementMeta(elementId)) as GaussianSplatElementMeta
  if (meta.type !== 'gaussians') throw new Error(`Element ${elementId} is not a gaussians (type=${(meta as any).type})`)
  return meta
}

export async function fetchBinaryPayload(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to get payload: ${res.status} ${res.statusText}`)
  return await res.arrayBuffer()
}

export async function updatePointCloudSettings(elementId: string, req: UpdatePointCloudSettingsRequest): Promise<void> {
  // For now, the only mutable pointcloud setting is pointSize.
  // This just re-uploads meta via a dedicated endpoint once it exists.
  // Temporary behavior: no-op if server doesn't support settings for elements.
  const res = await fetch(`/api/elements/${encodeURIComponent(elementId)}/meta`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(req),
  })

  // Older packaged builds may not have PATCH; treat 405/404 as unsupported.
  if (res.status === 404 || res.status === 405) return
  if (!res.ok) throw new Error(`Failed to update settings: ${res.status} ${res.statusText}`)
}

export async function fetchViewerSettings(): Promise<ViewerSettings> {
  const res = await fetch('/api/viewer/settings')
  if (!res.ok) throw new Error(`Failed to get viewer settings: ${res.status} ${res.statusText}`)
  return (await res.json()) as ViewerSettings
}
