import { useEffect, useState } from 'react'
import { fetchCameraElementMeta, type CameraElementMeta } from './api'

export type UseCameraState =
  | { status: 'loading' }
  | { status: 'error'; error: Error }
  | { status: 'ready'; meta: CameraElementMeta }

export function useCamera(elementId: string): UseCameraState {
  const [meta, setMeta] = useState<CameraElementMeta | null>(null)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    if (!elementId) {
      setMeta(null)
      setError(null)
      return
    }

    let cancelled = false
    setMeta(null)
    setError(null)

    fetchCameraElementMeta(elementId)
      .then((m) => {
        if (cancelled) return
        setMeta(m)
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
        const m = await fetchCameraElementMeta(elementId)
        if (cancelled) return
        setMeta((prev) => {
          if (!prev || prev.revision !== m.revision) return m
          if (String(prev.visible) !== String(m.visible)) return m
          if (JSON.stringify(prev.position) !== JSON.stringify(m.position)) return m
          if (JSON.stringify(prev.rotation) !== JSON.stringify(m.rotation)) return m
          if (prev.fov !== m.fov || prev.near !== m.near || prev.far !== m.far) return m
          return prev
        })
      } catch {
        // ignore periodic errors
      }
    }

    const id = window.setInterval(() => void tick(), 1500)
    void tick()
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [elementId])

  if (error) return { status: 'error', error }
  if (!meta) return { status: 'loading' }
  return { status: 'ready', meta }
}
