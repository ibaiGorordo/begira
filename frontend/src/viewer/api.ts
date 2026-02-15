export type ElementType = 'pointcloud' | 'gaussians' | 'camera' | 'image'
export type TimelineAxis = 'frame' | 'timestamp'

export type SampleQuery = {
  frame?: number
  timestamp?: number
}

export type TimelineInfo = {
  defaultAxis: TimelineAxis
  axes: Array<{
    axis: TimelineAxis
    min: number | null
    max: number | null
    hasData: boolean
  }>
  latest: {
    frame: number | null
    timestamp: number | null
  }
}

function appendSampleToUrl(rawUrl: string, sample?: SampleQuery): string {
  if (!sample) return rawUrl
  const hasFrame = sample.frame !== undefined
  const hasTimestamp = sample.timestamp !== undefined
  if (hasFrame && hasTimestamp) {
    throw new Error('Sample query cannot include both frame and timestamp')
  }

  const url = new URL(rawUrl, window.location.origin)
  if (hasFrame) {
    url.searchParams.set('frame', String(Math.round(Number(sample.frame))))
  }
  if (hasTimestamp) {
    url.searchParams.set('timestamp', String(Number(sample.timestamp)))
  }
  return `${url.pathname}${url.search}`
}

export type ElementInfo = {
  id: string
  type: ElementType
  name: string
  revision: number
  stateRevision?: number
  dataRevision?: number
  createdAt?: number
  bounds?: { min: [number, number, number]; max: [number, number, number] }
  summary?: Record<string, unknown>
  visible?: boolean
  // Camera-only fields returned from /api/elements.
  fov?: number
  near?: number
  far?: number
  position?: [number, number, number]
  rotation?: [number, number, number, number]
}

export type PointCloudElementMeta = {
  id: string
  type: 'pointcloud'
  name: string
  revision: number
  stateRevision: number
  dataRevision: number
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
  position?: [number, number, number]
  rotation?: [number, number, number, number]
  visible?: boolean
}

export type GaussianSplatElementMeta = {
  id: string
  type: 'gaussians'
  name: string
  revision: number
  stateRevision: number
  dataRevision: number
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
  position?: [number, number, number]
  rotation?: [number, number, number, number]
  visible?: boolean
}

export type CameraElementMeta = {
  id: string
  type: 'camera'
  name: string
  revision: number
  stateRevision: number
  dataRevision: number
  fov: number
  near: number
  far: number
  position?: [number, number, number]
  rotation?: [number, number, number, number]
  visible?: boolean
}

export type ImageElementMeta = {
  id: string
  type: 'image'
  name: string
  revision: number
  stateRevision: number
  dataRevision: number
  width: number
  height: number
  channels: number
  mimeType: string
  payloads: { image: { url: string; contentType: string } }
  position?: [number, number, number]
  rotation?: [number, number, number, number]
  visible?: boolean
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

export type UpdateElementMetaRequest = {
  pointSize?: number
  position?: [number, number, number]
  rotation?: [number, number, number, number]
  visible?: boolean
  deleted?: boolean
  fov?: number
  near?: number
  far?: number
  frame?: number
  timestamp?: number
  static?: boolean
}

export async function fetchEvents(): Promise<Events> {
  const res = await fetch('/api/events')
  if (!res.ok) throw new Error(`Failed to get events: ${res.status} ${res.statusText}`)
  return (await res.json()) as Events
}

export async function fetchTimelineInfo(): Promise<TimelineInfo> {
  const res = await fetch('/api/timeline')
  if (!res.ok) throw new Error(`Failed to get timeline info: ${res.status} ${res.statusText}`)
  return (await res.json()) as TimelineInfo
}

export async function fetchElements(sample?: SampleQuery): Promise<ElementInfo[]> {
  const res = await fetch(appendSampleToUrl('/api/elements', sample))
  if (!res.ok) throw new Error(`Failed to list elements: ${res.status} ${res.statusText}`)
  return (await res.json()) as ElementInfo[]
}

export async function fetchElementMeta(elementId: string, sample?: SampleQuery): Promise<any> {
  const res = await fetch(appendSampleToUrl(`/api/elements/${encodeURIComponent(elementId)}/meta`, sample))
  if (!res.ok) throw new Error(`Failed to get meta: ${res.status} ${res.statusText}`)
  return await res.json()
}

export async function fetchPointCloudElementMeta(elementId: string, sample?: SampleQuery): Promise<PointCloudElementMeta> {
  const meta = (await fetchElementMeta(elementId, sample)) as PointCloudElementMeta
  if (meta.type !== 'pointcloud') throw new Error(`Element ${elementId} is not a pointcloud (type=${(meta as any).type})`)
  return meta
}

export async function fetchGaussianElementMeta(elementId: string, sample?: SampleQuery): Promise<GaussianSplatElementMeta> {
  const meta = (await fetchElementMeta(elementId, sample)) as GaussianSplatElementMeta
  if (meta.type !== 'gaussians') throw new Error(`Element ${elementId} is not a gaussians (type=${(meta as any).type})`)
  return meta
}

export async function fetchCameraElementMeta(elementId: string, sample?: SampleQuery): Promise<CameraElementMeta> {
  const meta = (await fetchElementMeta(elementId, sample)) as CameraElementMeta
  if (meta.type !== 'camera') throw new Error(`Element ${elementId} is not a camera (type=${(meta as any).type})`)
  return meta
}

export async function fetchImageElementMeta(elementId: string, sample?: SampleQuery): Promise<ImageElementMeta> {
  const meta = (await fetchElementMeta(elementId, sample)) as ImageElementMeta
  if (meta.type !== 'image') throw new Error(`Element ${elementId} is not an image (type=${(meta as any).type})`)
  return meta
}

export async function fetchBinaryPayload(url: string, sample?: SampleQuery): Promise<ArrayBuffer> {
  const res = await fetch(appendSampleToUrl(url, sample))
  if (!res.ok) throw new Error(`Failed to get payload: ${res.status} ${res.statusText}`)
  return await res.arrayBuffer()
}

export async function updatePointCloudSettings(elementId: string, req: UpdatePointCloudSettingsRequest): Promise<void> {
  const res = await fetch(`/api/elements/${encodeURIComponent(elementId)}/meta`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(req),
  })

  // Older packaged builds may not have PATCH; treat 405/404 as unsupported.
  if (res.status === 404 || res.status === 405) return
  if (!res.ok) throw new Error(`Failed to update settings: ${res.status} ${res.statusText}`)
}

export async function updateElementMeta(elementId: string, req: UpdateElementMetaRequest): Promise<void> {
  const res = await fetch(`/api/elements/${encodeURIComponent(elementId)}/meta`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) throw new Error(`Failed to update element: ${res.status} ${res.statusText}`)
}

export async function deleteElement(elementId: string): Promise<void> {
  const res = await fetch(`/api/elements/${encodeURIComponent(elementId)}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`Failed to delete element: ${res.status} ${res.statusText}`)
}

export async function resetProject(): Promise<void> {
  const res = await fetch('/api/reset', { method: 'POST' })
  if (!res.ok) throw new Error(`Failed to reset project: ${res.status} ${res.statusText}`)
}

export async function createCamera(req: {
  name: string
  position?: [number, number, number]
  rotation?: [number, number, number, number]
  fov?: number
  near?: number
  far?: number
  frame?: number
  timestamp?: number
  static?: boolean
  elementId?: string | null
}): Promise<{ id: string }> {
  const res = await fetch('/api/elements/cameras', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) throw new Error(`Failed to create camera: ${res.status} ${res.statusText}`)
  const data = (await res.json()) as { id: string }
  return { id: data.id }
}

export async function fetchViewerSettings(): Promise<ViewerSettings> {
  const res = await fetch('/api/viewer/settings')
  if (!res.ok) throw new Error(`Failed to get viewer settings: ${res.status} ${res.statusText}`)
  return (await res.json()) as ViewerSettings
}
