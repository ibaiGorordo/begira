type Props = {
  visible: boolean
  enabled: boolean
  followSyncActive: boolean
  selectedFrame: number | null
  keyCount: number
  pullEnabled: boolean
  pullInfluenceFrames: number
  onTogglePull: (next: boolean) => void
  onChangePullInfluenceFrames: (next: number) => void
  onAddKey: () => void
  onDeleteKey: () => void
  onSmooth: () => void
}

export default function AnimateToolbar({
  visible,
  enabled,
  followSyncActive,
  selectedFrame,
  keyCount,
  pullEnabled,
  pullInfluenceFrames,
  onTogglePull,
  onChangePullInfluenceFrames,
  onAddKey,
  onDeleteKey,
  onSmooth,
}: Props) {
  if (!visible) return null

  return (
    <div className="animate-toolbar-wrap">
      <div className="animate-toolbar">
        <button className="toolbar-btn" type="button" onClick={onAddKey} disabled={!enabled}>
          Add Key
        </button>
        <button className="toolbar-btn" type="button" onClick={onDeleteKey} disabled={!enabled || selectedFrame === null}>
          Delete Key
        </button>
        <button className="toolbar-btn" type="button" onClick={onSmooth} disabled={!enabled || keyCount < 3}>
          Smooth
        </button>

        <label className="animate-toggle">
          <input type="checkbox" checked={pullEnabled} onChange={(e) => onTogglePull(e.target.checked)} disabled={!enabled} />
          Pull
        </label>

        <label className="animate-inline">
          Influence
          <input
            type="range"
            min={0}
            max={48}
            step={1}
            value={pullInfluenceFrames}
            onChange={(e) => onChangePullInfluenceFrames(Math.max(0, Math.round(Number(e.target.value) || 0)))}
            disabled={!enabled || !pullEnabled}
          />
          <span className="animate-slider-value">{pullInfluenceFrames}f</span>
        </label>

        <div className="animate-readout">
          frame: {selectedFrame ?? 'none'} | keys: {keyCount}
        </div>
      </div>
      {followSyncActive && (
        <div className="animate-toolbar-hint">
          Animation editing is disabled while camera follow/view sync is active.
        </div>
      )}
    </div>
  )
}
