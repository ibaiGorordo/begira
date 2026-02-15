import type { TimelineAxis } from './api'

type TimelineBounds = {
  min: number
  max: number
  hasData: boolean
}

export default function TimelineBar({
  axis,
  value,
  bounds,
  isPlaying,
  playbackFps,
  onAxisChange,
  onValueChange,
  onTogglePlay,
  onScrubStart,
  onScrubEnd,
  onStep,
  onPlaybackFpsChange,
}: {
  axis: TimelineAxis
  value: number
  bounds: TimelineBounds | null
  isPlaying: boolean
  playbackFps: number
  onAxisChange: (axis: TimelineAxis) => void
  onValueChange: (value: number) => void
  onTogglePlay: () => void
  onScrubStart: () => void
  onScrubEnd: () => void
  onStep: (delta: number) => void
  onPlaybackFpsChange: (value: number) => void
}) {
  const hasData = !!bounds && bounds.hasData
  const min = bounds?.min ?? 0
  const max = bounds?.max ?? 0
  const clampedValue = Math.min(max, Math.max(min, value))
  const range = Math.max(1e-9, max - min)
  const sliderStep = axis === 'frame' ? 1 : range / 1000

  return (
    <div className="timeline-bar">
      <div className="timeline-left">
        <label className="timeline-label">Axis</label>
        <select value={axis} onChange={(e) => onAxisChange(e.target.value as TimelineAxis)}>
          <option value="frame">Frame</option>
          <option value="timestamp">Timestamp</option>
        </select>
      </div>

      <div className="timeline-center">
        <button className="timeline-btn" onClick={() => onStep(-1)} disabled={!hasData} title="Step back">
          ◀
        </button>
        <button className="timeline-btn" onClick={onTogglePlay} disabled={!hasData} title={isPlaying ? 'Pause' : 'Play'}>
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <button className="timeline-btn" onClick={() => onStep(1)} disabled={!hasData} title="Step forward">
          ▶
        </button>

        <input
          className="timeline-slider"
          type="range"
          min={min}
          max={max}
          step={sliderStep}
          value={clampedValue}
          disabled={!hasData}
          onPointerDown={onScrubStart}
          onPointerUp={onScrubEnd}
          onPointerCancel={onScrubEnd}
          onBlur={onScrubEnd}
          onChange={(e) => onValueChange(Number(e.target.value))}
        />

        <input
          className="timeline-input"
          type="number"
          value={axis === 'frame' ? Math.round(clampedValue) : clampedValue.toFixed(3)}
          disabled={!hasData}
          onChange={(e) => {
            const v = Number(e.target.value)
            if (!Number.isFinite(v)) return
            onValueChange(v)
          }}
        />
      </div>

      <div className="timeline-right">
        <label className="timeline-label">FPS</label>
        <input
          className="timeline-input"
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
      </div>
    </div>
  )
}
