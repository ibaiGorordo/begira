import { useEffect, useMemo, useRef, useState } from 'react'
import { fetchBinaryPayload, fetchPointCloudElementMeta, type PointCloudElementMeta, type SampleQuery } from '../shared/api'
import { decodePointCloud, DecodedPointCloud } from './pointcloudGeometry'

export type UsePointCloudState =
  | { status: 'loading' }
  | { status: 'error'; error: Error }
  | { status: 'ready'; meta: PointCloudElementMeta; decoded: DecodedPointCloud }

function sampleDep(sample?: SampleQuery): string {
  return `${sample?.timeline ?? 'na'}|${sample?.time ?? 'na'}|${sample?.frame ?? 'na'}|${sample?.timestamp ?? 'na'}`
}

export function usePointCloud(cloudId: string, sample?: SampleQuery, enabled = true): UsePointCloudState {
  const [meta, setMeta] = useState<PointCloudElementMeta | null>(null)
  const [payload, setPayload] = useState<ArrayBuffer | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const payloadKeyRef = useRef<string | null>(null)
  const prevIdRef = useRef<string | null>(null)
  const dep = sampleDep(sample)

  useEffect(() => {
    if (!cloudId) {
      setMeta(null)
      setPayload(null)
      setError(null)
      payloadKeyRef.current = null
      prevIdRef.current = null
      return
    }
    if (!enabled) return

    let cancelled = false
    setError(null)
    if (prevIdRef.current !== cloudId) {
      prevIdRef.current = cloudId
      setMeta(null)
      setPayload(null)
      payloadKeyRef.current = null
    }

    const sync = async () => {
      try {
        const m = await fetchPointCloudElementMeta(cloudId, sample)
        if (cancelled) return

        setMeta(m)
        const payloadKey = `${m.payloads.points.url}|${m.dataRevision}`
        if (payloadKey !== payloadKeyRef.current) {
          const buf = await fetchBinaryPayload(m.payloads.points.url, sample)
          if (cancelled) return
          payloadKeyRef.current = payloadKey
          setPayload(buf)
        }
      } catch (e: unknown) {
        if (cancelled) return
        setError(e instanceof Error ? e : new Error(String(e)))
      }
    }

    void sync()
    const id = window.setInterval(() => void sync(), 750)

    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [cloudId, dep, enabled])

  const decoded = useMemo(() => {
    if (!meta || !payload) return null
    return decodePointCloud(meta, payload)
  }, [meta, payload])

  if (error) return { status: 'error', error }
  if (!cloudId || !meta || !payload || !decoded) return { status: 'loading' }
  return { status: 'ready', meta, decoded }
}
