import { useEffect, useState } from 'react'
import type { ElementInfo } from './api'

export type HierarchyProps = {
  elements: ElementInfo[] | null
  selectedId: string | null
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
}

export default function Hierarchy({ elements, selectedId, onSelect, onFocus }: HierarchyProps) {
  const items = elements ?? []
  const [, setTick] = useState(0)

  useEffect(() => {
    const handler = (e: any) => {
      if (!e?.detail?.id) return
      setTick((t) => t + 1)
    }
    try {
      window.addEventListener('begira_visibility_changed', handler)
    } catch {}
    return () => {
      try {
        window.removeEventListener('begira_visibility_changed', handler)
      } catch {}
    }
  }, [])

  const setVisibilityForElement = (id: string, visible: boolean) => {
    try {
      const anyWin = window as any
      anyWin.__begira_visibility = anyWin.__begira_visibility || {}
      anyWin.__begira_visibility[id] = visible
      try {
        const ev = new CustomEvent('begira_visibility_changed', { detail: { id } })
        window.dispatchEvent(ev)
      } catch {}
    } catch {}
  }

  return (
    <div style={{ padding: 12, width: 260 }}>
      <strong>Scene</strong>
      <div style={{ marginTop: 10, fontSize: 12, opacity: 0.75 }}>Elements</div>

      <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
        {items.map((c) => {
          const selected = c.id === selectedId
          const visibilityMap = (window as any).__begira_visibility || {}
          const isVisible = visibilityMap[c.id] !== false
          const count = Number((c.summary as any)?.pointCount ?? (c.summary as any)?.count ?? 0)
          const unit = c.type === 'pointcloud' ? 'pts' : 'splats'
          return (
            <div key={c.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                type="checkbox"
                title={isVisible ? 'Hide' : 'Show'}
                checked={isVisible}
                onChange={(e) => setVisibilityForElement(c.id, e.target.checked)}
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
                <div style={{ fontSize: 12, opacity: 0.75 }}>{count.toLocaleString()} {unit}</div>
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

      <div style={{ marginTop: 14, fontSize: 12, opacity: 0.65 }}>Tip: click to select, double-click to focus the camera.</div>
    </div>
  )
}
