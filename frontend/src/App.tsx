import { useEffect, useMemo, useRef, useState } from 'react'
import PointCloudCanvas from './viewer/PointCloudCanvas'
import { fetchEvents, fetchPointCloudList, PointCloudInfo } from './viewer/api'
import Inspector from './viewer/Inspector'
import Hierarchy from './viewer/Hierarchy'

function CollapseHandle({ side, open, onToggle }: { side: 'left' | 'right'; open: boolean; onToggle: () => void }) {
  const isLeft = side === 'left'
  const label = isLeft ? (open ? 'Collapse scene' : 'Expand scene') : open ? 'Collapse inspector' : 'Expand inspector'
  const glyph = isLeft ? (open ? '◀' : '▶') : open ? '▶' : '◀'

  return (
    <div
      style={{
        width: 18,
        minWidth: 18,
        position: 'relative',
        background: '#0b1020',
        borderLeft: isLeft ? 0 : '1px solid #1b2235',
        borderRight: isLeft ? '1px solid #1b2235' : 0,
      }}
    >
      <button
        type="button"
        aria-label={label}
        title={label}
        onClick={onToggle}
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: 22,
          height: 54,
          padding: 0,
          borderRadius: 999,
          border: '1px solid #1b2235',
          background: '#0f1630',
          color: '#e8ecff',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          opacity: 0.9,
        }}
        onMouseEnter={(e) => {
          ;(e.currentTarget as HTMLButtonElement).style.opacity = '1'
        }}
        onMouseLeave={(e) => {
          ;(e.currentTarget as HTMLButtonElement).style.opacity = '0.9'
        }}
      >
        <span style={{ fontSize: 12, lineHeight: 1 }}>{glyph}</span>
      </button>
    </div>
  )
}

export default function App() {
  const [clouds, setClouds] = useState<PointCloudInfo[] | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [leftOpen, setLeftOpen] = useState(true)
  const [rightOpen, setRightOpen] = useState(true)
  const [focusTarget, setFocusTarget] = useState<string | null>(null)
  const lastRev = useRef<number>(-1)

  useEffect(() => {
    fetchPointCloudList()
      .then((items) => {
        setClouds(items)
        // Nothing selected by default.
        setSelectedId(null)

        // But do set an initial camera focus target to the first-logged cloud.
        const ordered =
          items.length > 0 && items.every((c) => typeof c.createdAt === 'number')
            ? [...items].sort((a, b) => (a.createdAt! - b.createdAt!))
            : items
        if (ordered.length > 0) setFocusTarget(ordered[0].id)
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  useEffect(() => {
    let cancelled = false

    const tick = async () => {
      try {
        const ev = await fetchEvents()
        if (cancelled) return

        if (ev.globalRevision !== lastRev.current) {
          lastRev.current = ev.globalRevision
          const items = await fetchPointCloudList()
          if (cancelled) return
          setClouds(items)

          if (items.length === 0) {
            setSelectedId(null)
            return
          }

          // Keep current selection if it still exists; otherwise clear.
          if (selectedId !== null && !items.some((c) => c.id === selectedId)) {
            setSelectedId(null)
          }
        }
      } catch (e: unknown) {
        if (cancelled) return
        setError(e instanceof Error ? e.message : String(e))
      }
    }

    const id = window.setInterval(() => void tick(), 1000)
    void tick()

    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [selectedId])

  const selected = useMemo(() => {
    if (!clouds || selectedId === null) return null
    return clouds.find((c) => c.id === selectedId) ?? null
  }, [clouds, selectedId])

  const orderedClouds = useMemo(() => {
    const items = clouds ?? []
    // Prefer API-provided createdAt (true log order). Fallback: keep current list order.
    if (items.every((c) => typeof c.createdAt === 'number')) {
      return [...items].sort((a, b) => (a.createdAt! - b.createdAt!))
    }
    return items
  }, [clouds])

  return (
    <div className="app" style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <div className="header" style={{ borderBottom: '1px solid #1b2235', background: '#0b1020', color: '#e8ecff' }}>
        <strong>begira</strong>
        <span style={{ opacity: 0.7 }}>point cloud viewer</span>
        <div style={{ flex: 1 }} />
        {error && <span style={{ color: 'crimson' }}>{error}</span>}
      </div>

      <div className="viewer" style={{ display: 'flex', flex: 1, minHeight: 0, background: '#0b1020' }}>
        {leftOpen && (
          <div style={{ borderRight: '1px solid #1b2235', background: '#0f1630', color: '#e8ecff' }}>
            <Hierarchy
              clouds={clouds}
              selectedId={selectedId}
              onSelect={setSelectedId}
              onFocus={(id) => setFocusTarget(id)}
            />
          </div>
        )}

        <CollapseHandle side="left" open={leftOpen} onToggle={() => setLeftOpen((v) => !v)} />

        <div style={{ flex: 1, minWidth: 0 }}>
          <PointCloudCanvas
            cloudIds={orderedClouds.map((c) => c.id)}
            selectedId={selectedId}
            onSelect={setSelectedId}
            focusTarget={focusTarget}
            onFocus={(id) => setFocusTarget(id)}
            cloudMetaBounds={orderedClouds.map((c) => c.bounds).filter(Boolean) as any}
          />
        </div>

        <CollapseHandle side="right" open={rightOpen} onToggle={() => setRightOpen((v) => !v)} />

        {rightOpen && (
          <div style={{ borderLeft: '1px solid #1b2235', background: '#0f1630', color: '#e8ecff' }}>
            <Inspector selected={selected} />
          </div>
        )}
      </div>
    </div>
  )
}
