import { useEffect, useMemo, useState } from 'react'
import { fetchBinaryPayload, fetchPointCloudElementMeta, PointCloudElementMeta } from './api'
import { decodePointCloud, DecodedPointCloud } from './pointcloudGeometry'

export type UsePointCloudState =
  | { status: 'loading' }
  | { status: 'error'; error: Error }
  | { status: 'ready'; meta: PointCloudElementMeta; decoded: DecodedPointCloud }

export function usePointCloud(cloudId: string): UsePointCloudState {
  const [meta, setMeta] = useState<PointCloudElementMeta | null>(null)
  const [payload, setPayload] = useState<ArrayBuffer | null>(null)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    if (!cloudId) {
      setMeta(null)
      setPayload(null)
      setError(null)
      return
    }

    let cancelled = false
    setMeta(null)
    setPayload(null)
    setError(null)

    fetchPointCloudElementMeta(cloudId)
      .then((m) => {
        if (cancelled) return
        setMeta(m)
        return fetchBinaryPayload(m.payloads.points.url)
      })
      .then((buf) => {
        if (cancelled || !buf) return
        setPayload(buf)
      })
      .catch((e: unknown) => {
        if (cancelled) return
        setError(e instanceof Error ? e : new Error(String(e)))
      })

    return () => {
      cancelled = true
    }
  }, [cloudId])

  // Refresh meta periodically so settings updates apply without re-downloading payload.
  useEffect(() => {
    let cancelled = false
    if (!cloudId) return

    const tick = async () => {
      try {
        const m = await fetchPointCloudElementMeta(cloudId)
        if (cancelled) return
        setMeta((prev) => {
          if (!prev || prev.revision !== m.revision || prev.pointSize !== m.pointSize) return m
          return prev
        })
      } catch {
        // ignore
      }
    }

    const id = window.setInterval(() => void tick(), 750)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [cloudId])

  const decoded = useMemo(() => {
    if (!meta || !payload) return null
    return decodePointCloud(meta, payload)
  }, [meta, payload])

  if (error) return { status: 'error', error }
  if (!cloudId || !meta || !payload || !decoded) return { status: 'loading' }
  return { status: 'ready', meta, decoded }
}
