import type { ElementInfo } from './api'

export type HierarchyProps = {
  elements: ElementInfo[] | null
  selectedId: string | null
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
}

export default function Hierarchy({ elements, selectedId, onSelect, onFocus }: HierarchyProps) {
  const pointclouds = (elements ?? []).filter((e) => e.type === 'pointcloud')

  return (
    <div style={{ padding: 12, width: 260 }}>
      <strong>Scene</strong>
      <div style={{ marginTop: 10, fontSize: 12, opacity: 0.75 }}>Elements</div>

      <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
        {pointclouds.map((c) => {
          const selected = c.id === selectedId
          const pointCount = Number((c.summary as any)?.pointCount ?? 0)
          return (
            <button
              key={c.id}
              onClick={() => onSelect(c.id)}
              onDoubleClick={() => onFocus(c.id)}
              style={{
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
              <div style={{ fontSize: 12, opacity: 0.75 }}>{pointCount.toLocaleString()} pts</div>
            </button>
          )
        })}

        {elements && pointclouds.length === 0 && <div style={{ opacity: 0.75 }}>(no elements)</div>}
        {!elements && <div style={{ opacity: 0.75 }}>(loading…)</div>}
      </div>

      {selectedId === null && elements && elements.length > 0 && (
        <div style={{ marginTop: 10, fontSize: 12, opacity: 0.65 }}>(nothing selected)</div>
      )}

      <div style={{ marginTop: 14, fontSize: 12, opacity: 0.65 }}>Tip: click to select, double-click to focus the camera.</div>
    </div>
  )
}
