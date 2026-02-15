import { useEffect, useMemo, useRef, useState } from 'react'
import { fetchBinaryPayload, fetchGaussianElementMeta, type GaussianSplatElementMeta, type SampleQuery } from '../shared/api'
import { decodeGaussians, DecodedGaussians } from './gaussiansGeometry'

export type UseGaussiansState =
  | { status: 'loading' }
  | { status: 'error'; error: Error }
  | { status: 'ready'; meta: GaussianSplatElementMeta; decoded: DecodedGaussians }

function sampleDep(sample?: SampleQuery): string {
  return `${sample?.timeline ?? 'na'}|${sample?.time ?? 'na'}|${sample?.frame ?? 'na'}|${sample?.timestamp ?? 'na'}`
}

export function useGaussians(elementId: string, sample?: SampleQuery, enabled = true): UseGaussiansState {
  const [meta, setMeta] = useState<GaussianSplatElementMeta | null>(null)
  const [payload, setPayload] = useState<ArrayBuffer | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const payloadKeyRef = useRef<string | null>(null)
  const prevIdRef = useRef<string | null>(null)
  const dep = sampleDep(sample)

  useEffect(() => {
    if (!elementId) {
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
    if (prevIdRef.current !== elementId) {
      prevIdRef.current = elementId
      setMeta(null)
      setPayload(null)
      payloadKeyRef.current = null
    }

    const sync = async () => {
      try {
        const m = await fetchGaussianElementMeta(elementId, sample)
        if (cancelled) return

        setMeta(m)
        const payloadKey = `${m.payloads.gaussians.url}|${m.dataRevision}`
        if (payloadKey !== payloadKeyRef.current) {
          const buf = await fetchBinaryPayload(m.payloads.gaussians.url, sample)
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
  }, [elementId, dep, enabled])

  const decoded = useMemo(() => {
    if (!meta || !payload) return null
    return decodeGaussians(meta, payload)
  }, [meta, payload])

  if (error) return { status: 'error', error }
  if (!elementId || !meta || !payload || !decoded) return { status: 'loading' }
  return { status: 'ready', meta, decoded }
}
