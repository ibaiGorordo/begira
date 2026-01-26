import type { ElementInfo } from './api'

export type HierarchyProps = {
  elements: ElementInfo[] | null
  selectedId: string | null
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
  onToggleVisibility: (id: string, visible: boolean) => void
  onDelete: (id: string) => void
  onAddCamera: () => void
}

export default function Hierarchy({
  elements,
  selectedId,
  onSelect,
  onFocus,
  onToggleVisibility,
  onDelete,
  onAddCamera,
}: HierarchyProps) {
  const items = elements ?? []

  return (
    <div style={{ padding: 12, width: 260 }}>
      <strong>Scene</strong>
      <div style={{ marginTop: 10, fontSize: 12, opacity: 0.75 }}>Elements</div>
      <div style={{ marginTop: 8 }}>
        <button
          onClick={onAddCamera}
          style={{
            padding: '6px 8px',
            borderRadius: 6,
            border: '1px solid #1b2235',
            background: '#0f1630',
            color: '#e8ecff',
            cursor: 'pointer',
            fontSize: 12,
          }}
        >
          Add Camera
        </button>
      </div>

      <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
        {items.map((c) => {
          const selected = c.id === selectedId
          const isVisible = c.visible !== false
          const count = Number((c.summary as any)?.pointCount ?? (c.summary as any)?.count ?? 0)
          const unit = c.type === 'pointcloud' ? 'pts' : c.type === 'gaussians' ? 'splats' : ''
          return (
            <div key={c.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                type="checkbox"
                title={isVisible ? 'Hide' : 'Show'}
                checked={isVisible}
                onChange={(e) => onToggleVisibility(c.id, e.target.checked)}
              />
              <button
                onClick={() => onSelect(c.id)}
                onDoubleClick={() => onFocus(c.id)}
                style={{
                  flex: 1,
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 6,
                  border: '1px solid #1b2235',
                  background: selected ? '#172242' : '#0f1630',
                  color: '#e8ecff',
                  cursor: 'pointer',
                }}
                title={c.id}
              >
                <div style={{ fontSize: 13, fontWeight: 600 }}>{c.name}</div>
                <div style={{ fontSize: 11, opacity: 0.5, textTransform: 'uppercase' }}>{c.type}</div>
                {c.type !== 'camera' && (
                  <div style={{ fontSize: 12, opacity: 0.75 }}>{count.toLocaleString()} {unit}</div>
                )}
              </button>
              <button
                onClick={() => onDelete(c.id)}
                title="Remove"
                style={{
                  padding: '4px 6px',
                  borderRadius: 6,
                  border: '1px solid #1b2235',
                  background: '#0f1630',
                  color: '#e8ecff',
                  cursor: 'pointer',
                }}
              >
                ×
              </button>
            </div>
          )
        })}

        {elements && items.length === 0 && <div style={{ opacity: 0.75 }}>(no elements)</div>}
        {!elements && <div style={{ opacity: 0.75 }}>(loading…)</div>}
      </div>

      {selectedId === null && elements && elements.length > 0 && (
        <div style={{ marginTop: 10, fontSize: 12, opacity: 0.65 }}>(nothing selected)</div>
      )}

      <div style={{ marginTop: 14, fontSize: 12, opacity: 0.65 }}>Tip: Click to select, double-click to focus the camera.</div>
    </div>
  )
}
