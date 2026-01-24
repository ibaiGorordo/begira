import { useEffect, useMemo, useState } from 'react'
import { fetchPointCloudMeta, fetchPointCloudPayload, PointCloudMeta } from './api'
import { decodePointCloud, DecodedPointCloud } from './pointcloudGeometry'

export type UsePointCloudState =
  | { status: 'loading' }
  | { status: 'error'; error: Error }
  | { status: 'ready'; meta: PointCloudMeta; decoded: DecodedPointCloud }

export function usePointCloud(cloudId: string): UsePointCloudState {
  const [meta, setMeta] = useState<PointCloudMeta | null>(null)
  const [payload, setPayload] = useState<ArrayBuffer | null>(null)
  const [error, setError] = useState<Error | null>(null)

  // Load meta+payload when cloudId changes.
  useEffect(() => {
    // If no cloud is selected/requested, don't fetch.
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

    fetchPointCloudMeta(cloudId)
      .then((m) => {
        if (cancelled) return
        setMeta(m)
        return fetchPointCloudPayload(m.payload.url)
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

  // Refresh meta periodically (cheap) so settings updates (like pointSize) apply
  // without forcing a points payload re-download.
  useEffect(() => {
    let cancelled = false
    if (!cloudId) return

    const tick = async () => {
      try {
        const m = await fetchPointCloudMeta(cloudId)
        if (cancelled) return
        setMeta((prev) => {
          // Only update if revision changed to avoid rerenders.
          if (!prev || prev.revision !== m.revision || prev.pointSize !== m.pointSize) return m
          return prev
        })
      } catch {
        // ignore; main load effect surfaces errors
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
