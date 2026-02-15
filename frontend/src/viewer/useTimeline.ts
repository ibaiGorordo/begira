import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchTimelineInfo, type SampleQuery, type TimelineAxis, type TimelineInfo } from './api'

type AxisInfo = TimelineInfo['axes'][number]

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v))
}

function toSample(axis: TimelineAxis | null, axisInfo: AxisInfo | null, value: number): SampleQuery | undefined {
  if (!axis || !axisInfo || !axisInfo.hasData) return undefined
  if (axisInfo.kind === 'sequence') {
    const seq = Math.round(value)
    if (axis === 'frame') return { frame: seq }
    return { timeline: axis, time: seq, frame: seq }
  }
  if (axis === 'timestamp') return { timestamp: value }
  return { timeline: axis, time: value }
}

function sampleKeyFor(sample: SampleQuery | undefined): string {
  if (!sample) return 'none'
  if (sample.timeline !== undefined && sample.time !== undefined) return `tl:${sample.timeline}:${sample.time}`
  if (sample.frame !== undefined) return `f:${sample.frame}`
  if (sample.timestamp !== undefined) return `t:${sample.timestamp}`
  return 'none'
}

export function useTimeline({ enabled = true }: { enabled?: boolean } = {}) {
  const [axis, setAxisRaw] = useState<TimelineAxis | null>(null)
  const [value, setValueRaw] = useState(0)
  const [info, setInfo] = useState<TimelineInfo | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isScrubbing, setIsScrubbing] = useState(false)
  const [loopPlayback, setLoopPlayback] = useState(false)
  const [playbackFps, setPlaybackFpsRaw] = useState(24)

  const initialized = useRef(false)
  const axisRef = useRef<TimelineAxis | null>(null)
  const valueRef = useRef(0)
  const isPlayingRef = useRef(false)
  const isScrubbingRef = useRef(false)
  const refreshRequestRef = useRef(0)
  const resumeAfterScrubRef = useRef(false)
  const boundsRef = useRef<{ min: number; max: number; hasData: boolean } | null>(null)
  const axisInfoRef = useRef<AxisInfo | null>(null)
  const playbackFpsRef = useRef(24)
  const loopPlaybackRef = useRef(false)
  const playbackTimerRef = useRef<number | null>(null)

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

  const axisInfo = useMemo<AxisInfo | null>(() => {
    if (!axis || !info) return null
    return info.axes.find((a) => a.axis === axis) ?? null
  }, [axis, info])

  const bounds = useMemo(() => {
    if (!axisInfo || axisInfo.min === null || axisInfo.max === null) return null
    return {
      min: axisInfo.min,
      max: axisInfo.max,
      hasData: axisInfo.hasData,
    }
  }, [axisInfo])

  const sampleQuery = useMemo<SampleQuery | undefined>(() => toSample(axis, axisInfo, value), [axis, axisInfo, value])
  const sampleKey = useMemo(() => sampleKeyFor(sampleQuery), [sampleQuery])
  const axes = useMemo(() => (info?.axes ?? []).filter((a) => a.hasData), [info])

  useEffect(() => {
    boundsRef.current = bounds
  }, [bounds])

  useEffect(() => {
    axisInfoRef.current = axisInfo
  }, [axisInfo])

  useEffect(() => {
    playbackFpsRef.current = playbackFps
  }, [playbackFps])

  useEffect(() => {
    loopPlaybackRef.current = loopPlayback
  }, [loopPlayback])

  const refresh = useCallback(async ({ reinitialize = false }: { reinitialize?: boolean } = {}): Promise<SampleQuery | undefined> => {
    const requestId = ++refreshRequestRef.current
    const fetched = await fetchTimelineInfo()
    if (requestId !== refreshRequestRef.current) {
      return toSample(axisRef.current, axisInfo ?? null, valueRef.current)
    }
    setInfo(fetched)

    const available = (fetched.axes ?? []).filter((a) => a.hasData && a.min !== null && a.max !== null)
    if (available.length === 0) {
      setAxisRaw(null)
      valueRef.current = 0
      setValueRaw(0)
      setIsPlaying(false)
      return undefined
    }

    let activeAxis = axisRef.current
    const activeInfo = activeAxis ? available.find((a) => a.axis === activeAxis) ?? null : null
    if (!activeInfo) {
      const defaultInfo = fetched.defaultAxis ? available.find((a) => a.axis === fetched.defaultAxis) ?? null : null
      activeAxis = defaultInfo?.axis ?? available[0].axis
      setAxisRaw(activeAxis)
    }

    const infoForAxis = available.find((a) => a.axis === activeAxis) ?? available[0]
    if (!infoForAxis) return undefined
    if (activeAxis !== infoForAxis.axis) {
      activeAxis = infoForAxis.axis
      setAxisRaw(activeAxis)
    }

    let nextValue = valueRef.current
    if (reinitialize || !initialized.current) {
      const latest = fetched.latest?.[String(activeAxis)]
      nextValue = latest ?? infoForAxis.min ?? 0
      initialized.current = true
      valueRef.current = nextValue
      setValueRaw(nextValue)
      return toSample(activeAxis, infoForAxis, nextValue)
    }

    if (!isPlayingRef.current && !isScrubbingRef.current) {
      const minV = infoForAxis.min ?? 0
      const maxV = infoForAxis.max ?? minV
      const clamped = clamp(valueRef.current, minV, maxV)
      if (clamped !== valueRef.current) {
        nextValue = clamped
        valueRef.current = clamped
        setValueRaw(clamped)
      }
    }

    return toSample(activeAxis, infoForAxis, nextValue)
  }, [axisInfo])

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
  }, [enabled, refresh])

  useEffect(() => {
    if (!bounds?.hasData) return
    setValueRaw((prev) => clamp(prev, bounds.min, bounds.max))
  }, [bounds?.hasData, bounds?.min, bounds?.max])

  const clearPlaybackTimer = useCallback(() => {
    if (playbackTimerRef.current !== null) {
      window.clearTimeout(playbackTimerRef.current)
      playbackTimerRef.current = null
    }
  }, [])

  const playbackTick = useCallback((): boolean => {
    const b = boundsRef.current
    const a = axisInfoRef.current
    if (!b?.hasData || !a) return false
    const fps = Math.max(0.5, playbackFpsRef.current)
    let reachedEnd = false
    setValueRaw((prev) => {
      const delta = a.kind === 'sequence' ? 1 : 1 / fps
      const next = prev + delta
      if (next >= b.max) {
        reachedEnd = true
        if (loopPlaybackRef.current) {
          valueRef.current = b.min
          return b.min
        }
        valueRef.current = b.max
        return b.max
      }
      valueRef.current = next
      return next
    })
    if (reachedEnd && !loopPlaybackRef.current) {
      isPlayingRef.current = false
      setIsPlaying(false)
      clearPlaybackTimer()
      return false
    }
    return true
  }, [clearPlaybackTimer])

  const schedulePlayback = useCallback(() => {
    clearPlaybackTimer()

    const tick = () => {
      if (!isPlayingRef.current) {
        clearPlaybackTimer()
        return
      }
      if (!isScrubbingRef.current) {
        const keepGoing = playbackTick()
        if (!keepGoing) return
      }
      const fps = Math.max(0.5, playbackFpsRef.current)
      const delayMs = Math.max(8, Math.round(1000 / fps))
      playbackTimerRef.current = window.setTimeout(tick, delayMs)
    }

    const fps = Math.max(0.5, playbackFpsRef.current)
    const delayMs = Math.max(8, Math.round(1000 / fps))
    playbackTimerRef.current = window.setTimeout(tick, delayMs)
  }, [clearPlaybackTimer, playbackTick])

  useEffect(() => {
    if (enabled) return
    isPlayingRef.current = false
    setIsPlaying(false)
    clearPlaybackTimer()
  }, [clearPlaybackTimer, enabled])

  useEffect(() => {
    if (!isPlaying) {
      clearPlaybackTimer()
      return
    }
    schedulePlayback()
    return () => clearPlaybackTimer()
  }, [clearPlaybackTimer, isPlaying, schedulePlayback])

  const setAxis = useCallback(
    (nextAxis: TimelineAxis) => {
      if (!nextAxis) return
      setAxisRaw(nextAxis)
      const nextAxisInfo = info?.axes.find((a) => a.axis === nextAxis && a.hasData)
      if (!nextAxisInfo || nextAxisInfo.min === null || nextAxisInfo.max === null) return
      const latest = info?.latest?.[nextAxis]
      const next = latest ?? nextAxisInfo.min
      const clamped = clamp(next, nextAxisInfo.min, nextAxisInfo.max)
      valueRef.current = clamped
      setValueRaw(clamped)
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
      if (!bounds?.hasData || !axisInfo) return
      const fps = Math.max(0.5, playbackFps)
      const stepSize = axisInfo.kind === 'sequence' ? 1 : 1 / fps
      setValueRaw((prev) => {
        const next = clamp(prev + delta * stepSize, bounds.min, bounds.max)
        valueRef.current = next
        return next
      })
    },
    [axisInfo, bounds, playbackFps]
  )

  const setPlaybackFps = useCallback((fps: number) => {
    setPlaybackFpsRaw(Number.isFinite(fps) && fps > 0 ? Math.max(0.5, Math.min(240, fps)) : 24)
  }, [])

  const beginScrub = useCallback(() => {
    if (isScrubbingRef.current) return
    isScrubbingRef.current = true
    resumeAfterScrubRef.current = isPlayingRef.current
    setIsScrubbing(true)
    if (isPlayingRef.current) {
      isPlayingRef.current = false
      setIsPlaying(false)
    }
  }, [])

  const endScrub = useCallback(() => {
    if (!isScrubbingRef.current) {
      resumeAfterScrubRef.current = false
      return
    }
    isScrubbingRef.current = false
    setIsScrubbing(false)
    if (resumeAfterScrubRef.current) {
      isPlayingRef.current = true
      setIsPlaying(true)
    }
    resumeAfterScrubRef.current = false
  }, [])

  useEffect(() => {
    if (!isScrubbing) return
    const stop = () => endScrub()
    window.addEventListener('pointerup', stop, true)
    window.addEventListener('pointercancel', stop, true)
    window.addEventListener('mouseup', stop, true)
    window.addEventListener('touchend', stop, true)
    return () => {
      window.removeEventListener('pointerup', stop, true)
      window.removeEventListener('pointercancel', stop, true)
      window.removeEventListener('mouseup', stop, true)
      window.removeEventListener('touchend', stop, true)
    }
  }, [endScrub, isScrubbing])

  const togglePlay = useCallback(() => {
    if (!bounds?.hasData) return
    if (isScrubbingRef.current) {
      isScrubbingRef.current = false
      setIsScrubbing(false)
      resumeAfterScrubRef.current = false
    }
    setIsPlaying((prev) => {
      if (prev) {
        isPlayingRef.current = false
        clearPlaybackTimer()
        return false
      }
      setValueRaw((current) => {
        if (current >= bounds.max - 1e-9) {
          valueRef.current = bounds.min
          return bounds.min
        }
        valueRef.current = current
        return current
      })
      isPlayingRef.current = true
      return true
    })
  }, [bounds, clearPlaybackTimer])

  const reinitialize = useCallback(() => {
    initialized.current = false
    isPlayingRef.current = false
    setIsPlaying(false)
    clearPlaybackTimer()
  }, [clearPlaybackTimer])

  return {
    axis,
    axes,
    axisInfo,
    value,
    bounds,
    info,
    isPlaying,
    isScrubbing,
    loopPlayback,
    playbackFps,
    sampleQuery,
    sampleKey,
    setAxis,
    setValue,
    setIsPlaying,
    setLoopPlayback,
    togglePlay,
    beginScrub,
    endScrub,
    step,
    setPlaybackFps,
    refresh,
    reinitialize,
  }
}
