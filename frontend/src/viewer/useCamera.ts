import { useEffect, useRef, useState } from 'react'
import { fetchCameraElementMeta, type CameraElementMeta, type SampleQuery } from './api'

export type UseCameraState =
  | { status: 'loading' }
  | { status: 'error'; error: Error }
  | { status: 'ready'; meta: CameraElementMeta }

function sampleDep(sample?: SampleQuery): string {
  return `${sample?.timeline ?? 'na'}|${sample?.time ?? 'na'}|${sample?.frame ?? 'na'}|${sample?.timestamp ?? 'na'}`
}

export function useCamera(elementId: string, sample?: SampleQuery, enabled = true): UseCameraState {
  const [meta, setMeta] = useState<CameraElementMeta | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const lastIdRef = useRef<string | null>(null)
  const dep = sampleDep(sample)

  useEffect(() => {
    if (!elementId) {
      setMeta(null)
      setError(null)
      return
    }
    if (!enabled) return

    let cancelled = false
    setError(null)
    if (lastIdRef.current !== elementId) {
      lastIdRef.current = elementId
      setMeta(null)
    }

    const sync = async () => {
      try {
        const m = await fetchCameraElementMeta(elementId, sample)
        if (cancelled) return
        setMeta(m)
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

  if (error) return { status: 'error', error }
  if (!meta) return { status: 'loading' }
  return { status: 'ready', meta }
}
