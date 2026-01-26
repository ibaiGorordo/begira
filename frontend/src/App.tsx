import { useEffect, useMemo, useRef, useState } from 'react'
import PointCloudCanvas from './viewer/PointCloudCanvas'
import { fetchElementMeta, fetchElements, fetchEvents, updateElementMeta, deleteElement, resetProject, createCamera, type ElementInfo } from './viewer/api'
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
  const [activeCameraId, setActiveCameraId] = useState<string | null>(null)
  const [transformMode, setTransformMode] = useState<'translate' | 'rotate'>('translate')
  const lastRev = useRef<number>(-1)
  const didInitFocus = useRef(false)
  const orderRef = useRef<Map<string, number>>(new Map())
  const nextOrderRef = useRef(0)
  const undoStack = useRef<Array<{ label: string; do: () => Promise<void>; undo: () => Promise<void> }>>([])
  const redoStack = useRef<Array<{ label: string; do: () => Promise<void>; undo: () => Promise<void> }>>([])
  const [historyCounts, setHistoryCounts] = useState({ undo: 0, redo: 0 })
  const resettingRef = useRef(false)
  const [isResetting, setIsResetting] = useState(false)

  const pointclouds = useMemo(() => (elements ?? []).filter((e) => e.type === 'pointcloud'), [elements])
  const gaussians = useMemo(() => (elements ?? []).filter((e) => e.type === 'gaussians'), [elements])
  const cameras = useMemo(() => (elements ?? []).filter((e) => e.type === 'camera'), [elements])

  const orderElements = (items: ElementInfo[]) => {
    const orderMap = orderRef.current
    let next = nextOrderRef.current
    for (const item of items) {
      if (!orderMap.has(item.id)) {
        orderMap.set(item.id, next)
        next += 1
      }
    }
    nextOrderRef.current = next
    return [...items].sort((a, b) => (orderMap.get(a.id) ?? 0) - (orderMap.get(b.id) ?? 0))
  }

  useEffect(() => {
    fetchElements()
      .then((items) => {
        const ordered = orderElements(items)
        setElements(ordered)
        setSelectedId(null)

        // Default focus: first element that appears (only once).
        if (!didInitFocus.current) {
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
          const ordered = orderElements(items)
          setElements(ordered)

          if (ordered.length === 0) {
            setSelectedId(null)
            return
          }

          if (selectedId !== null && !ordered.some((c) => c.id === selectedId)) {
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

  useEffect(() => {
    if (activeCameraId && !elements?.some((e) => e.id === activeCameraId)) {
      setActiveCameraId(null)
    }
  }, [activeCameraId, elements])

  const bumpHistory = () => {
    setHistoryCounts({
      undo: undoStack.current.length,
      redo: redoStack.current.length,
    })
  }

  const runAction = async (action: { label: string; do: () => Promise<void>; undo: () => Promise<void> }) => {
    if (resettingRef.current) return
    await action.do()
    if (resettingRef.current) return
    undoStack.current.push(action)
    redoStack.current = []
    bumpHistory()
  }

  const runUserAction = async (action: { label: string; do: () => Promise<void>; undo: () => Promise<void> }) => {
    if (resettingRef.current) return
    await runAction(action)
  }

  const undo = async () => {
    if (resettingRef.current) return
    const action = undoStack.current.pop()
    if (!action) {
      bumpHistory()
      return
    }
    await action.undo()
    redoStack.current.push(action)
    bumpHistory()
  }

  const redo = async () => {
    if (resettingRef.current) return
    const action = redoStack.current.pop()
    if (!action) {
      bumpHistory()
      return
    }
    await action.do()
    undoStack.current.push(action)
    bumpHistory()
  }

  const setLocalPose = (id: string, position: [number, number, number], rotation: [number, number, number, number]) => {
    try {
      const anyWin = window as any
      if (!anyWin.__begira_local_pose) anyWin.__begira_local_pose = {}
      anyWin.__begira_local_pose[id] = { position: [...position], rotation: [...rotation] }
    } catch {}
  }

  const clearLocalPose = (id: string) => {
    try {
      const anyWin = window as any
      if (anyWin.__begira_local_pose) {
        delete anyWin.__begira_local_pose[id]
      }
    } catch {}
  }

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase()
      if (tag === 'input' || tag === 'textarea' || (e.target as HTMLElement | null)?.isContentEditable) return
      const mod = e.metaKey || e.ctrlKey
      if (!mod) return
      if (e.key.toLowerCase() === 'z' && !e.shiftKey) {
        e.preventDefault()
        void undo()
      } else if (e.key.toLowerCase() === 'z' && e.shiftKey) {
        e.preventDefault()
        void redo()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  const applyVisibility = async (id: string, visible: boolean) => {
    const prev = elements?.find((e) => e.id === id)
    await runUserAction({
      label: visible ? 'Show' : 'Hide',
      do: async () => {
        await updateElementMeta(id, { visible })
        setElements((cur) => (cur ? cur.map((e) => (e.id === id ? { ...e, visible } : e)) : cur))
      },
      undo: async () => {
        await updateElementMeta(id, { visible: prev?.visible !== false })
        setElements((cur) => (cur ? cur.map((e) => (e.id === id ? { ...e, visible: prev?.visible !== false } : e)) : cur))
      },
    })
  }

  const removeElement = async (id: string) => {
    const prev = elements?.find((e) => e.id === id) ?? null
    const prevIndex = elements ? elements.findIndex((e) => e.id === id) : -1
    await runUserAction({
      label: 'Remove',
      do: async () => {
        await deleteElement(id)
        setElements((cur) => (cur ? cur.filter((e) => e.id !== id) : cur))
        if (selectedId === id) {
          setSelectedId(null)
          setFocusTarget(null)
        }
      },
      undo: async () => {
        await updateElementMeta(id, { deleted: false })
        if (!prev) return
        setElements((cur) => {
          if (!cur) return cur
          const next = [...cur]
          const idx = prevIndex >= 0 ? prevIndex : next.length
          next.splice(idx, 0, prev)
          return next
        })
      },
    })
  }

  const resetAll = async () => {
    resettingRef.current = true
    setIsResetting(true)
    setHistoryCounts({ undo: 0, redo: 0 })
    await resetProject()
    const items = await fetchElements()
    setElements(orderElements(items))
    setActiveCameraId(null)
    try {
      const anyWin = window as any
      anyWin.__begira_visual_override = {}
      anyWin.__begira_local_pose = {}
      window.dispatchEvent(new CustomEvent('begira_reset'))
    } catch {}
    undoStack.current = []
    redoStack.current = []
    setHistoryCounts({ undo: 0, redo: 0 })
    resettingRef.current = false
    setIsResetting(false)
  }

  const addCameraFromView = async () => {
    try {
      const cam = (window as any).__begira_view_camera as { position: [number, number, number]; rotation: [number, number, number, number]; fov: number; near: number; far: number } | undefined
      const payload = cam
        ? { name: `Camera ${cameras.length + 1}`, position: cam.position, rotation: cam.rotation, fov: cam.fov, near: cam.near, far: cam.far }
        : { name: `Camera ${cameras.length + 1}` }
      await createCamera(payload)
    } catch (e: unknown) {
      console.error(e)
    }
  }

  const onTransformCommit = async (id: string, position: [number, number, number], rotation: [number, number, number, number]) => {
    if (resettingRef.current) return
    const prev = await fetchElementMeta(id)
    const prevPos: [number, number, number] = (prev.position ?? [0, 0, 0]) as [number, number, number]
    const prevRot: [number, number, number, number] = (prev.rotation ?? [0, 0, 0, 1]) as [number, number, number, number]
    const qLen = Math.hypot(rotation[0], rotation[1], rotation[2], rotation[3]) || 1
    const normRot: [number, number, number, number] = [
      rotation[0] / qLen,
      rotation[1] / qLen,
      rotation[2] / qLen,
      rotation[3] / qLen,
    ]
    await runUserAction({
      label: 'Transform',
      do: async () => {
        await updateElementMeta(id, { position, rotation: normRot })
        setLocalPose(id, position, normRot)
      },
      undo: async () => {
        await updateElementMeta(id, { position: prevPos, rotation: prevRot })
        setLocalPose(id, prevPos, prevRot)
      },
    })
  }

  return (
    <div className="app" style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <div className="header" style={{ borderBottom: '1px solid #1b2235', background: '#0b1020', color: '#e8ecff' }}>
        <strong>begira</strong>
        <span style={{ opacity: 0.7 }}>viewer</span>
        <div style={{ flex: 1 }} />
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button
            onClick={() => void undo()}
            disabled={historyCounts.undo === 0 || isResetting}
            style={{
              padding: '4px 8px',
              borderRadius: 6,
              border: '1px solid #1b2235',
              background: '#0f1630',
              color: '#e8ecff',
              cursor: 'pointer',
              fontSize: 12,
              opacity: historyCounts.undo === 0 || isResetting ? 0.5 : 1,
            }}
          >
            Undo
          </button>
          <button
            onClick={() => void redo()}
            disabled={historyCounts.redo === 0 || isResetting}
            style={{
              padding: '4px 8px',
              borderRadius: 6,
              border: '1px solid #1b2235',
              background: '#0f1630',
              color: '#e8ecff',
              cursor: 'pointer',
              fontSize: 12,
              opacity: historyCounts.redo === 0 || isResetting ? 0.5 : 1,
            }}
          >
            Redo
          </button>
          <button
            onClick={() => void resetAll()}
            style={{
              padding: '4px 8px',
              borderRadius: 6,
              border: '1px solid #1b2235',
              background: '#0f1630',
              color: '#e8ecff',
              cursor: 'pointer',
              fontSize: 12,
            }}
          >
            Reset
          </button>
        </div>
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
                onToggleVisibility: (id: string, visible: boolean) => void applyVisibility(id, visible),
                onDelete: (id: string) => void removeElement(id),
                onAddCamera: () => void addCameraFromView(),
              } satisfies HierarchyProps)}
            />
          </div>
        )}

        <CollapseHandle side="left" open={leftOpen} onToggle={() => setLeftOpen((v) => !v)} />

        <div style={{ flex: 1, minWidth: 0 }}>
          <PointCloudCanvas
            cloudIds={pointclouds.map((c) => c.id)}
            gaussianIds={gaussians.map((c) => c.id)}
            cameraIds={cameras.map((c) => c.id)}
            selectedId={selectedId}
            onSelect={setSelectedId}
            focusTarget={focusTarget}
            onFocus={(id) => {
              // `id` from the canvas is used only to *clear* a pending focus request.
              // Camera focus requests come from double-click actions (Hierarchy / scene).
              if (id === null) setFocusTarget(null)
              else setFocusTarget(id)
            }}
            cloudMetaBounds={pointclouds.map((c) => c.bounds).filter(Boolean) as any}
            gaussianMetaBounds={gaussians.map((c) => c.bounds).filter(Boolean) as any}
            activeCameraId={activeCameraId}
            transformMode={transformMode}
            onTransformCommit={(id, position, rotation) => void onTransformCommit(id, position, rotation)}
          />
        </div>

        <CollapseHandle side="right" open={rightOpen} onToggle={() => setRightOpen((v) => !v)} />

        {rightOpen && (
          <div style={{ borderLeft: '1px solid #1b2235', background: '#0f1630', color: '#e8ecff' }}>
            <Inspector
              selected={selected}
              activeCameraId={activeCameraId}
              onSetActiveCamera={setActiveCameraId}
              onDelete={(id) => void removeElement(id)}
              transformMode={transformMode}
              onTransformModeChange={setTransformMode}
              onPoseCommit={(id, position, rotation) => void onTransformCommit(id, position, rotation)}
            />
          </div>
        )}
      </div>
    </div>
  )
}
