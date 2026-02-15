import { useMemo, useRef, type PointerEvent as ReactPointerEvent } from 'react'
import type { TimelineAxis, TimelineKind } from './api'

type TimelineBounds = {
  min: number
  max: number
  hasData: boolean
}

type AxisOption = {
  axis: TimelineAxis
  kind: TimelineKind
  min: number | null
  max: number | null
  hasData: boolean
}

function formatTick(kind: TimelineKind, value: number): string {
  if (kind === 'sequence') return `#${Math.round(value)}`
  return value.toFixed(2)
}

export default function TimelineBar({
  axis,
  axes,
  value,
  bounds,
  isPlaying,
  loopPlayback,
  playbackFps,
  onAxisChange,
  onValueChange,
  onToggleLoop,
  onTogglePlay,
  onScrubStart,
  onScrubEnd,
  onStep,
  onPlaybackFpsChange,
}: {
  axis: TimelineAxis | null
  axes: AxisOption[]
  value: number
  bounds: TimelineBounds | null
  isPlaying: boolean
  loopPlayback: boolean
  playbackFps: number
  onAxisChange: (axis: TimelineAxis) => void
  onValueChange: (value: number) => void
  onToggleLoop: (loop: boolean) => void
  onTogglePlay: () => void
  onScrubStart: () => void
  onScrubEnd: () => void
  onStep: (delta: number) => void
  onPlaybackFpsChange: (value: number) => void
}) {
  const selectedAxis = axis ? axes.find((a) => a.axis === axis) ?? null : null
  const hasData = !!bounds && bounds.hasData && !!selectedAxis
  const min = bounds?.min ?? 0
  const max = bounds?.max ?? 0
  const clampedValue = Math.min(max, Math.max(min, value))
  const range = Math.max(1e-9, max - min)
  const playheadRatio = hasData ? (clampedValue - min) / range : 0
  const playheadPercent = `${Math.max(0, Math.min(1, playheadRatio)) * 100}%`
  const scrubStateRef = useRef<{ active: boolean; pointerId: number | null }>({ active: false, pointerId: null })

  const majorTickCount = 10
  const minorPerSegment = 4

  const majorTicks = useMemo(
    () =>
      Array.from({ length: majorTickCount + 1 }, (_, i) => {
        const t = i / majorTickCount
        const x = min + t * (max - min)
        return { t, value: x }
      }),
    [max, min],
  )

  const minorTicks = useMemo(
    () =>
      Array.from({ length: majorTickCount * minorPerSegment + 1 }, (_, i) => {
        if (i % minorPerSegment === 0) return null
        const t = i / (majorTickCount * minorPerSegment)
        return { t }
      }).filter(Boolean) as Array<{ t: number }>,
    [],
  )

  const applyPointerToValue = (clientX: number, rect: DOMRect) => {
    if (!hasData || rect.width <= 1) return
    const x = Math.max(0, Math.min(rect.width, clientX - rect.left))
    const t = x / rect.width
    const raw = min + t * (max - min)
    const next = selectedAxis?.kind === 'sequence' ? Math.round(raw) : raw
    onValueChange(next)
  }

  const beginPointerScrub = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!hasData) return
    const container = event.currentTarget
    scrubStateRef.current.active = true
    scrubStateRef.current.pointerId = event.pointerId
    container.setPointerCapture(event.pointerId)
    onScrubStart()
    applyPointerToValue(event.clientX, container.getBoundingClientRect())
  }

  const movePointerScrub = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!scrubStateRef.current.active || !hasData) return
    applyPointerToValue(event.clientX, event.currentTarget.getBoundingClientRect())
  }

  const endPointerScrub = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!scrubStateRef.current.active) return
    scrubStateRef.current.active = false
    if (
      scrubStateRef.current.pointerId !== null &&
      event.currentTarget.hasPointerCapture(scrubStateRef.current.pointerId)
    ) {
      event.currentTarget.releasePointerCapture(scrubStateRef.current.pointerId)
    }
    scrubStateRef.current.pointerId = null
    onScrubEnd()
  }

  return (
    <div className="timeline-bar">
      <div className="timeline-left-controls">
        <button className="timeline-btn icon nav" onClick={() => onStep(-1)} disabled={!hasData} title="Step back (Left Arrow)">
          <svg className="timeline-nav-icon" viewBox="0 0 16 10" aria-hidden>
            <path d="M7.5 2 L3.5 5 L7.5 8 Z" fill="currentColor" />
            <line x1="8.2" y1="5" x2="13.0" y2="5" />
          </svg>
        </button>
        <button className="timeline-btn icon" onClick={onTogglePlay} disabled={!hasData} title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}>
          {isPlaying ? '⏸' : '▶'}
        </button>
        <button className="timeline-btn icon nav" onClick={() => onStep(1)} disabled={!hasData} title="Step forward (Right Arrow)">
          <svg className="timeline-nav-icon" viewBox="0 0 16 10" aria-hidden>
            <line x1="3.0" y1="5" x2="7.8" y2="5" />
            <path d="M8.5 2 L12.5 5 L8.5 8 Z" fill="currentColor" />
          </svg>
        </button>

        <select
          className="timeline-axis-select"
          value={axis ?? ''}
          onChange={(e) => onAxisChange(e.target.value)}
          disabled={axes.length === 0}
        >
          {axes.length === 0 ? (
            <option value="">No timeline</option>
          ) : (
            axes.map((entry) => (
              <option key={entry.axis} value={entry.axis}>
                {entry.axis}
              </option>
            ))
          )}
        </select>

        <div className="timeline-fps-inline">
          <input
            className="timeline-input small"
            type="number"
            min={0.5}
            max={240}
            step={0.5}
            value={playbackFps}
            onChange={(e) => {
              const v = Number(e.target.value)
              if (!Number.isFinite(v)) return
              onPlaybackFpsChange(v)
            }}
          />
          <span>FPS</span>
        </div>

        <label className="timeline-loop">
          <input
            type="checkbox"
            checked={loopPlayback}
            onChange={(e) => onToggleLoop(e.target.checked)}
          />
          Loop
        </label>
      </div>

      <div className="timeline-track-block">
        <div className="timeline-ruler">
          {hasData &&
            minorTicks.map((tick) => (
              <div key={`minor-${tick.t}`} className="timeline-ruler-tick minor" style={{ left: `${tick.t * 100}%` }} />
            ))}
          {hasData &&
            majorTicks.map((tick) => (
              <div key={`major-${tick.t}`} className="timeline-ruler-tick major" style={{ left: `${tick.t * 100}%` }}>
                <span className="timeline-ruler-label">{formatTick(selectedAxis.kind, tick.value)}</span>
              </div>
            ))}
          {hasData && <div className="timeline-playhead-line ruler" style={{ left: playheadPercent }} />}
        </div>

        <div
          className={`timeline-scrub-surface${hasData ? '' : ' disabled'}`}
          onPointerDown={beginPointerScrub}
          onPointerMove={movePointerScrub}
          onPointerUp={endPointerScrub}
          onPointerCancel={endPointerScrub}
        >
          <div className="timeline-scrub-track" />
          {hasData && <div className="timeline-playhead-line track" style={{ left: playheadPercent }} />}
          {hasData && <div className="timeline-scrub-thumb" style={{ left: playheadPercent }} />}
        </div>
      </div>

      <div className="timeline-readout">
        <input
          className="timeline-input"
          type="number"
          value={selectedAxis?.kind === 'sequence' ? Math.round(clampedValue) : clampedValue.toFixed(3)}
          disabled={!hasData}
          onChange={(e) => {
            const v = Number(e.target.value)
            if (!Number.isFinite(v)) return
            onValueChange(v)
          }}
        />
      </div>
    </div>
  )
}
