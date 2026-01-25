import { useEffect, useMemo, useState } from 'react'
import { fetchBinaryPayload, fetchGaussianElementMeta, GaussianSplatElementMeta } from './api'
import { decodeGaussians, DecodedGaussians } from './gaussiansGeometry'

export type UseGaussiansState =
  | { status: 'loading' }
  | { status: 'error'; error: Error }
  | { status: 'ready'; meta: GaussianSplatElementMeta; decoded: DecodedGaussians }

export function useGaussians(elementId: string): UseGaussiansState {
  const [meta, setMeta] = useState<GaussianSplatElementMeta | null>(null)
  const [payload, setPayload] = useState<ArrayBuffer | null>(null)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    if (!elementId) {
      setMeta(null)
      setPayload(null)
      setError(null)
      return
    }

    let cancelled = false
    setMeta(null)
    setPayload(null)
    setError(null)

    fetchGaussianElementMeta(elementId)
      .then((m) => {
        if (cancelled) return
        setMeta(m)
        return fetchBinaryPayload(m.payloads.gaussians.url)
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
  }, [elementId])

  useEffect(() => {
    let cancelled = false
    if (!elementId) return

    const tick = async () => {
      try {
        const m = await fetchGaussianElementMeta(elementId)
        if (cancelled) return
        setMeta((prev) => {
          if (!prev || prev.revision !== m.revision) return m
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
  }, [elementId])

  const decoded = useMemo(() => {
    if (!meta || !payload) return null
    return decodeGaussians(meta, payload)
  }, [meta, payload])

  if (error) return { status: 'error', error }
  if (!elementId || !meta || !payload || !decoded) return { status: 'loading' }
  return { status: 'ready', meta, decoded }
}
