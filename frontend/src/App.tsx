import { useEffect, useMemo, useRef, useState } from 'react'
import PointCloudCanvas from './viewer/PointCloudCanvas'
import { fetchElements, fetchEvents, type ElementInfo } from './viewer/api'
import Inspector from './viewer/Inspector'
import Hierarchy, { type HierarchyProps } from './viewer/Hierarchy'

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
  const [elements, setElements] = useState<ElementInfo[] | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [leftOpen, setLeftOpen] = useState(true)
  const [rightOpen, setRightOpen] = useState(true)
  const [focusTarget, setFocusTarget] = useState<string | null>(null)
  const lastRev = useRef<number>(-1)
  const didInitFocus = useRef(false)

  const pointclouds = useMemo(() => (elements ?? []).filter((e) => e.type === 'pointcloud'), [elements])
  const gaussians = useMemo(() => (elements ?? []).filter((e) => e.type === 'gaussians'), [elements])

  useEffect(() => {
    fetchElements()
      .then((items) => {
        setElements(items)
        setSelectedId(null)

        // Default focus: first element that appears (only once).
        if (!didInitFocus.current) {
          const ordered =
            items.length > 0 && items.every((c) => typeof c.createdAt === 'number')
              ? [...items].sort((a, b) => (a.createdAt! - b.createdAt!))
              : items
          if (ordered.length > 0) {
            didInitFocus.current = true
            setFocusTarget(ordered[0].id)
          }
        }
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
          const items = await fetchElements()
          if (cancelled) return
          setElements(items)

          if (items.length === 0) {
            setSelectedId(null)
            return
          }

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
    if (!elements || selectedId === null) return null
    return elements.find((c) => c.id === selectedId) ?? null
  }, [elements, selectedId])

  const orderedPointclouds = useMemo(() => {
    const items = pointclouds
    if (items.every((c) => typeof c.createdAt === 'number')) {
      return [...items].sort((a, b) => (a.createdAt! - b.createdAt!))
    }
    return items
  }, [pointclouds])

  const orderedGaussians = useMemo(() => {
    const items = gaussians
    if (items.every((c) => typeof c.createdAt === 'number')) {
      return [...items].sort((a, b) => (a.createdAt! - b.createdAt!))
    }
    return items
  }, [gaussians])

  return (
    <div className="app" style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <div className="header" style={{ borderBottom: '1px solid #1b2235', background: '#0b1020', color: '#e8ecff' }}>
        <strong>begira</strong>
        <span style={{ opacity: 0.7 }}>viewer</span>
        <div style={{ flex: 1 }} />
        {error && <span style={{ color: 'crimson' }}>{error}</span>}
      </div>

      <div className="viewer" style={{ display: 'flex', flex: 1, minHeight: 0, background: '#0b1020' }}>
        {leftOpen && (
          <div style={{ borderRight: '1px solid #1b2235', background: '#0f1630', color: '#e8ecff' }}>
            <Hierarchy
              {...({
                elements,
                selectedId,
                onSelect: setSelectedId,
                onFocus: (id: string) => setFocusTarget(id),
              } satisfies HierarchyProps)}
            />
          </div>
        )}

        <CollapseHandle side="left" open={leftOpen} onToggle={() => setLeftOpen((v) => !v)} />

        <div style={{ flex: 1, minWidth: 0 }}>
          <PointCloudCanvas
            cloudIds={orderedPointclouds.map((c) => c.id)}
            gaussianIds={orderedGaussians.map((c) => c.id)}
            selectedId={selectedId}
            onSelect={setSelectedId}
            focusTarget={focusTarget}
            onFocus={(id) => {
              // `id` from the canvas is used only to *clear* a pending focus request.
              // Camera focus requests come from double-click actions (Hierarchy / scene).
              if (id === null) setFocusTarget(null)
              else setFocusTarget(id)
            }}
            cloudMetaBounds={orderedPointclouds.map((c) => c.bounds).filter(Boolean) as any}
            gaussianMetaBounds={orderedGaussians.map((c) => c.bounds).filter(Boolean) as any}
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
