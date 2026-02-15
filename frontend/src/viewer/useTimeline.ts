import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchTimelineInfo, type SampleQuery, type TimelineAxis, type TimelineInfo } from './api'

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v))
}

function toSample(axis: TimelineAxis, value: number): SampleQuery {
  if (axis === 'frame') return { frame: Math.round(value) }
  return { timestamp: value }
}

export function useTimeline({ enabled = true }: { enabled?: boolean } = {}) {
  const [axis, setAxisRaw] = useState<TimelineAxis>('frame')
  const [value, setValueRaw] = useState(0)
  const [info, setInfo] = useState<TimelineInfo | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isScrubbing, setIsScrubbing] = useState(false)
  const [playbackFps, setPlaybackFpsRaw] = useState(24)
  const initialized = useRef(false)
  const axisRef = useRef<TimelineAxis>('frame')
  const valueRef = useRef(0)
  const isPlayingRef = useRef(false)
  const isScrubbingRef = useRef(false)
  const refreshRequestRef = useRef(0)
  const resumeAfterScrubRef = useRef(false)

  useEffect(() => {
    axisRef.current = axis
  }, [axis])

  useEffect(() => {
    valueRef.current = value
  }, [value])

  useEffect(() => {
    isPlayingRef.current = isPlaying
  }, [isPlaying])

  useEffect(() => {
    isScrubbingRef.current = isScrubbing
  }, [isScrubbing])

  const bounds = useMemo(() => {
    const axisInfo = info?.axes.find((a) => a.axis === axis)
    if (!axisInfo || axisInfo.min === null || axisInfo.max === null) return null
    return {
      min: axisInfo.min,
      max: axisInfo.max,
      hasData: axisInfo.hasData,
    }
  }, [axis, info])

  const sampleQuery = useMemo<SampleQuery>(() => toSample(axis, value), [axis, value])

  const sampleKey = useMemo(
    () => `${sampleQuery.frame !== undefined ? `f:${sampleQuery.frame}` : `t:${sampleQuery.timestamp}`}`,
    [sampleQuery]
  )

  const refresh = useCallback(async ({ reinitialize = false }: { reinitialize?: boolean } = {}): Promise<SampleQuery> => {
    const requestId = ++refreshRequestRef.current
    const fetched = await fetchTimelineInfo()
    if (requestId !== refreshRequestRef.current) {
      return toSample(axisRef.current, valueRef.current)
    }
    setInfo(fetched)

    const currentAxis = axisRef.current
    const axisInfo = fetched.axes.find((a) => a.axis === currentAxis)
    if (!axisInfo || axisInfo.min === null || axisInfo.max === null) {
      return toSample(currentAxis, valueRef.current)
    }

    let nextValue = valueRef.current
    if (reinitialize || !initialized.current) {
      const latest = currentAxis === 'frame' ? fetched.latest.frame : fetched.latest.timestamp
      nextValue = latest ?? axisInfo.min
      initialized.current = true
      valueRef.current = nextValue
      setValueRaw(nextValue)
      return toSample(currentAxis, nextValue)
    }

    // Do not fight user input/playback; only clamp idle cursor if needed.
    if (!isPlayingRef.current && !isScrubbingRef.current) {
      const clamped = clamp(valueRef.current, axisInfo.min, axisInfo.max)
      if (clamped !== valueRef.current) {
        nextValue = clamped
        valueRef.current = clamped
        setValueRaw(clamped)
      }
    }

    return toSample(currentAxis, nextValue)
  }, [])

  useEffect(() => {
    if (!enabled) return
    let cancelled = false

    const sync = async () => {
      try {
        await refresh()
      } catch {
        if (cancelled) return
      }
    }

    void sync()
    const id = window.setInterval(() => void sync(), 1000)

    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [axis, enabled, refresh])

  useEffect(() => {
    if (!bounds?.hasData) return
    setValueRaw((prev) => clamp(prev, bounds.min, bounds.max))
  }, [bounds?.hasData, bounds?.min, bounds?.max])

  useEffect(() => {
    if (!enabled || !isPlaying || isScrubbing || !bounds?.hasData) return
    const fps = Math.max(0.5, playbackFps)
    const intervalMs = Math.max(10, Math.round(1000 / fps))
    const id = window.setInterval(() => {
      let reachedEnd = false
      setValueRaw((prev) => {
        const delta = axis === 'frame' ? 1 : 1 / fps
        const next = prev + delta
        if (next >= bounds.max) {
          reachedEnd = true
          valueRef.current = bounds.max
          return bounds.max
        }
        valueRef.current = next
        return next
      })
      if (reachedEnd) {
        setIsPlaying(false)
      }
    }, intervalMs)
    return () => window.clearInterval(id)
  }, [axis, bounds, enabled, isPlaying, isScrubbing, playbackFps])

  useEffect(() => {
    if (enabled) return
    setIsPlaying(false)
  }, [enabled])

  const setAxis = useCallback(
    (nextAxis: TimelineAxis) => {
      setAxisRaw(nextAxis)
      const axisInfo = info?.axes.find((a) => a.axis === nextAxis)
      if (!axisInfo || axisInfo.min === null || axisInfo.max === null) return
      const latest = nextAxis === 'frame' ? info?.latest.frame : info?.latest.timestamp
      const next = latest ?? axisInfo.min
      valueRef.current = clamp(next, axisInfo.min, axisInfo.max)
      setValueRaw(clamp(next, axisInfo.min, axisInfo.max))
    },
    [info]
  )

  const setValue = useCallback(
    (nextValue: number) => {
      if (!bounds) {
        valueRef.current = nextValue
        setValueRaw(nextValue)
        return
      }
      const clamped = clamp(nextValue, bounds.min, bounds.max)
      valueRef.current = clamped
      setValueRaw(clamped)
    },
    [bounds]
  )

  const step = useCallback(
    (delta: number) => {
      if (!bounds?.hasData) return
      const fps = Math.max(0.5, playbackFps)
      const stepSize = axis === 'frame' ? 1 : 1 / fps
      setValueRaw((prev) => {
        const next = clamp(prev + delta * stepSize, bounds.min, bounds.max)
        valueRef.current = next
        return next
      })
    },
    [axis, bounds, playbackFps]
  )

  const setPlaybackFps = useCallback((fps: number) => {
    setPlaybackFpsRaw(Number.isFinite(fps) && fps > 0 ? Math.max(0.5, Math.min(240, fps)) : 24)
  }, [])

  const beginScrub = useCallback(() => {
    if (isScrubbingRef.current) return
    resumeAfterScrubRef.current = isPlayingRef.current
    setIsScrubbing(true)
    if (isPlayingRef.current) {
      setIsPlaying(false)
    }
  }, [])

  const endScrub = useCallback(() => {
    if (!isScrubbingRef.current) return
    setIsScrubbing(false)
    if (resumeAfterScrubRef.current) {
      setIsPlaying(true)
    }
    resumeAfterScrubRef.current = false
  }, [])

  const togglePlay = useCallback(() => {
    if (!bounds?.hasData) return
    setIsPlaying((prev) => {
      if (prev) return false
      setValueRaw((current) => {
        if (current >= bounds.max - 1e-9) {
          valueRef.current = bounds.min
          return bounds.min
        }
        valueRef.current = current
        return current
      })
      return true
    })
  }, [bounds])

  const reinitialize = useCallback(() => {
    initialized.current = false
    setIsPlaying(false)
  }, [])

  return {
    axis,
    value,
    bounds,
    info,
    isPlaying,
    isScrubbing,
    playbackFps,
    sampleQuery,
    sampleKey,
    setAxis,
    setValue,
    setIsPlaying,
    togglePlay,
    beginScrub,
    endScrub,
    step,
    setPlaybackFps,
    refresh,
    reinitialize,
  }
}
