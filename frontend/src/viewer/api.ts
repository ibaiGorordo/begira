export type PointCloudInfo = {
  id: string
  name: string
  pointCount: number
  revision: number
  createdAt?: number
  pointSize?: number
  bounds?: { min: [number, number, number]; max: [number, number, number] }
}

export type PointCloudMeta = {
  id: string
  name: string
  pointCount: number
  revision: number
  pointSize: number
  bounds: { min: [number, number, number]; max: [number, number, number] }
  endianness: 'little' | 'big'
  interleaved: boolean
  bytesPerPoint: number
  schema: {
    position: { type: 'float32'; components: 3 }
    color?: { type: 'uint8'; components: 3; normalized: true }
  }
  payload: { url: string; contentType: string }
}

export type Events = {
  globalRevision: number
}

export type UpdatePointCloudSettingsRequest = {
  pointSize?: number
}

export async function fetchEvents(): Promise<Events> {
  const res = await fetch('/api/events')
  if (!res.ok) throw new Error(`Failed to get events: ${res.status} ${res.statusText}`)
  return (await res.json()) as Events
}

export async function fetchPointCloudList(): Promise<PointCloudInfo[]> {
  const res = await fetch('/api/pointclouds')
  if (!res.ok) throw new Error(`Failed to list pointclouds: ${res.status} ${res.statusText}`)
  return (await res.json()) as PointCloudInfo[]
}

export async function fetchPointCloudMeta(cloudId: string): Promise<PointCloudMeta> {
  const res = await fetch(`/api/pointclouds/${encodeURIComponent(cloudId)}/meta`)
  if (!res.ok) throw new Error(`Failed to get meta: ${res.status} ${res.statusText}`)
  return (await res.json()) as PointCloudMeta
}

export async function fetchPointCloudPayload(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to get points payload: ${res.status} ${res.statusText}`)
  return await res.arrayBuffer()
}

export async function updatePointCloudSettings(cloudId: string, req: UpdatePointCloudSettingsRequest): Promise<void> {
  const res = await fetch(`/api/pointclouds/${encodeURIComponent(cloudId)}/settings`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) throw new Error(`Failed to update settings: ${res.status} ${res.statusText}`)
}
