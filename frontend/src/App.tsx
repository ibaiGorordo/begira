import { useEffect, useMemo, useRef, useState } from 'react'
import { fetchElementMeta, fetchElements, fetchEvents, updateElementMeta, deleteElement, resetProject, createCamera, type ElementInfo } from './viewer/api'
import Inspector from './viewer/Inspector'
import Hierarchy, { type HierarchyProps, type HierarchyViewInfo } from './viewer/Hierarchy'
import DockWorkspace, { type DockImageView, type DockWorkspaceHandle } from './viewer/DockWorkspace'

const MAIN_3D_VIEW_ID = 'view-3d-main'
const IMAGE_VIEW_PREFIX = 'view-image-'

function imageViewId(imageId: string): string {
  return `${IMAGE_VIEW_PREFIX}${imageId}`
}

function imageIdFromViewId(viewId: string): string | null {
  if (!viewId.startsWith(IMAGE_VIEW_PREFIX)) return null
  return viewId.slice(IMAGE_VIEW_PREFIX.length)
}

function centerFromBounds(bounds?: { min: [number, number, number]; max: [number, number, number] }): [number, number, number] | null {
  if (!bounds) return null
  return [
    (bounds.min[0] + bounds.max[0]) * 0.5,
    (bounds.min[1] + bounds.max[1]) * 0.5,
    (bounds.min[2] + bounds.max[2]) * 0.5,
  ]
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v))
}

function CollapseHandle({
  side,
  open,
  onToggle,
  onResizeStart,
}: {
  side: 'left' | 'right'
  open: boolean
  onToggle: () => void
  onResizeStart: (e: React.MouseEvent<HTMLDivElement>) => void
}) {
  const isLeft = side === 'left'
  const label = isLeft ? (open ? 'Collapse scene' : 'Expand scene') : open ? 'Collapse inspector' : 'Expand inspector'
  const glyph = isLeft ? (open ? '◀' : '▶') : open ? '▶' : '◀'

  return (
    <div
      onMouseDown={onResizeStart}
      style={{
        width: 14,
        minWidth: 14,
        flex: '0 0 14px',
        position: 'relative',
        zIndex: 4,
        background: '#0b1020',
        borderLeft: isLeft ? 0 : '1px solid #1b2235',
        borderRight: isLeft ? '1px solid #1b2235' : 0,
        cursor: open ? 'col-resize' : 'default',
      }}
    >
      <div
        style={{
          position: 'absolute',
          top: 0,
          bottom: 0,
          left: '50%',
          width: 1,
          transform: 'translateX(-50%)',
          background: '#1b2235',
          opacity: open ? 0.8 : 0.45,
          pointerEvents: 'none',
        }}
      />
      <button
        type="button"
        aria-label={label}
        title={label}
        onMouseDown={(e) => e.stopPropagation()}
        onClick={onToggle}
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: 20,
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
  const workspaceRef = useRef<DockWorkspaceHandle | null>(null)
  const [leftWidth, setLeftWidth] = useState(300)
  const [rightWidth, setRightWidth] = useState(300)
  const [threeDViewVisible, setThreeDViewVisible] = useState(true)
  const [imageViewOrder, setImageViewOrder] = useState<string[]>([])
  const [imageViewState, setImageViewState] = useState<Record<string, { visible: boolean; deleted: boolean }>>({})

  const pointclouds = useMemo(() => (elements ?? []).filter((e) => e.type === 'pointcloud'), [elements])
  const gaussians = useMemo(() => (elements ?? []).filter((e) => e.type === 'gaussians'), [elements])
  const cameras = useMemo(() => (elements ?? []).filter((e) => e.type === 'camera'), [elements])
  const images = useMemo(() => (elements ?? []).filter((e) => e.type === 'image'), [elements])
  const imagesById = useMemo(() => new Map(images.map((img) => [img.id, img])), [images])

  useEffect(() => {
    const ids = images.map((img) => img.id)
    const idSet = new Set(ids)

    setImageViewOrder((prev) => {
      const kept = prev.filter((id) => idSet.has(id))
      for (const id of ids) {
        if (!kept.includes(id)) kept.push(id)
      }
      return kept
    })

    setImageViewState((prev) => {
      const next: Record<string, { visible: boolean; deleted: boolean }> = {}
      for (const id of ids) {
        next[id] = prev[id] ?? { visible: true, deleted: false }
      }
      return next
    })
  }, [images])

  const imageViews = useMemo(() => {
    const out: Array<{ id: string; imageId: string; name: string; visible: boolean; deleted: boolean; elementVisible: boolean }> = []
    for (const imageId of imageViewOrder) {
      const img = imagesById.get(imageId)
      if (!img) continue
      const state = imageViewState[imageId] ?? { visible: true, deleted: false }
      out.push({
        id: imageViewId(imageId),
        imageId,
        name: img.name,
        visible: state.visible,
        deleted: state.deleted,
        elementVisible: img.visible !== false,
      })
    }
    return out
  }, [imageViewOrder, imageViewState, imagesById])

  const dockImageViews = useMemo<DockImageView[]>(
    () =>
      imageViews
        .filter((v) => !v.deleted && v.visible && v.elementVisible)
        .map((v) => ({
          id: v.id,
          imageId: v.imageId,
          name: v.name,
          visible: true,
        })),
    [imageViews]
  )

  const hierarchyViews = useMemo<HierarchyViewInfo[]>(() => {
    const threeDElementIds = (elements ?? []).filter((e) => e.type !== 'image').map((e) => e.id)
    return [
      {
        id: MAIN_3D_VIEW_ID,
        name: '3D',
        kind: '3d',
        visible: threeDViewVisible,
        canDelete: false,
        canReorder: false,
        elementIds: threeDElementIds,
      },
      ...imageViews
        .filter((v) => !v.deleted)
        .map((v) => ({
          id: v.id,
          name: v.name,
          kind: 'image' as const,
          visible: v.visible && v.elementVisible,
          canDelete: true,
          canReorder: true,
          elementIds: [v.imageId],
        })),
    ]
  }, [elements, imageViews, threeDViewVisible])

  const lookAtTargets = useMemo(
    () =>
      (elements ?? [])
        .filter((e) => e.type !== 'image')
        .map((e) => {
          const p = e.position ?? centerFromBounds(e.bounds) ?? [0, 0, 0]
          return {
            id: e.id,
            name: e.name,
            type: e.type,
            position: [p[0], p[1], p[2]] as [number, number, number],
          }
        }),
    [elements]
  )

  const setImageViewFlags = (imageId: string, patch: Partial<{ visible: boolean; deleted: boolean }>) => {
    setImageViewState((prev) => {
      const cur = prev[imageId] ?? { visible: true, deleted: false }
      return { ...prev, [imageId]: { ...cur, ...patch } }
    })
  }

  const openImageView = (imageId: string) => {
    const img = imagesById.get(imageId)
    if (!img) return
    setImageViewOrder((prev) => (prev.includes(imageId) ? prev : [...prev, imageId]))
    setImageViewFlags(imageId, { visible: true, deleted: false })
    if (img.visible === false) void applyVisibility(imageId, true)
    setSelectedId(imageId)
    setFocusTarget(null)
    window.setTimeout(() => {
      workspaceRef.current?.focusImage(imageId)
    }, 0)
  }

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

  const beginResize = (side: 'left' | 'right', e: React.MouseEvent<HTMLDivElement>) => {
    if ((side === 'left' && !leftOpen) || (side === 'right' && !rightOpen)) return
    if (e.button !== 0) return
    e.preventDefault()

    const startX = e.clientX
    const startW = side === 'left' ? leftWidth : rightWidth
    const minW = 220
    const maxW = 520

    const onMove = (ev: MouseEvent) => {
      const dx = ev.clientX - startX
      const raw = side === 'left' ? startW + dx : startW - dx
      const nextW = clamp(raw, minW, maxW)
      if (side === 'left') setLeftWidth(nextW)
      else setRightWidth(nextW)
    }

    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      document.body.style.userSelect = ''
      document.body.style.cursor = ''
    }

    document.body.style.userSelect = 'none'
    document.body.style.cursor = 'col-resize'
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
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

      <div className="viewer" style={{ display: 'flex', flex: 1, minHeight: 0, background: '#0b1020', position: 'relative' }}>
        {leftOpen && (
          <div
            style={{
              width: leftWidth,
              minWidth: leftWidth,
              maxWidth: leftWidth,
              borderRight: '1px solid #1b2235',
              background: '#0f1630',
              color: '#e8ecff',
              flex: '0 0 auto',
              position: 'relative',
              zIndex: 5,
            }}
          >
            <Hierarchy
              {...({
                elements,
                selectedId,
                onSelect: setSelectedId,
                onFocus: (id: string) => {
                  const el = elements?.find((e) => e.id === id)
                  if (el?.type === 'image') {
                    openImageView(id)
                    return
                  }
                  setFocusTarget(id)
                },
                onToggleVisibility: (id: string, visible: boolean) => void applyVisibility(id, visible),
                onDelete: (id: string) => void removeElement(id),
                onAddCamera: () => void addCameraFromView(),
                views: hierarchyViews,
                onActivateView: (viewId: string) => {
                  if (viewId === MAIN_3D_VIEW_ID) return
                  const imageId = imageIdFromViewId(viewId)
                  if (!imageId) return
                  openImageView(imageId)
                },
                onToggleViewVisibility: (viewId: string, visible: boolean) => {
                  if (viewId === MAIN_3D_VIEW_ID) {
                    setThreeDViewVisible(visible)
                    return
                  }
                  const imageId = imageIdFromViewId(viewId)
                  if (!imageId) return
                  setImageViewFlags(imageId, { visible })
                  void applyVisibility(imageId, visible)
                },
                onDeleteView: (viewId: string) => {
                  const imageId = imageIdFromViewId(viewId)
                  if (!imageId) return
                  setImageViewFlags(imageId, { deleted: true, visible: false })
                  setImageViewOrder((prev) => prev.filter((id) => id !== imageId))
                },
                onMoveView: (viewId: string, direction: 'up' | 'down') => {
                  const imageId = imageIdFromViewId(viewId)
                  if (!imageId) return
                  setImageViewOrder((prev) => {
                    const idx = prev.indexOf(imageId)
                    if (idx < 0) return prev
                    const nextIdx = direction === 'up' ? idx - 1 : idx + 1
                    if (nextIdx < 0 || nextIdx >= prev.length) return prev
                    const out = [...prev]
                    const tmp = out[idx]
                    out[idx] = out[nextIdx]
                    out[nextIdx] = tmp
                    return out
                  })
                },
              } satisfies HierarchyProps)}
            />
          </div>
        )}

        <CollapseHandle
          side="left"
          open={leftOpen}
          onToggle={() => setLeftOpen((v) => !v)}
          onResizeStart={(e) => beginResize('left', e)}
        />

        <div style={{ flex: 1, minWidth: 0, minHeight: 0, overflow: 'hidden', position: 'relative', zIndex: 1 }}>
          <DockWorkspace
            ref={workspaceRef}
            pointclouds={pointclouds}
            gaussians={gaussians}
            cameras={cameras}
            images={images}
            selectedId={selectedId}
            onSelect={(id) => {
              setSelectedId(id)
              setFocusTarget(null)
            }}
            focusTarget={focusTarget}
            onFocus={(id) => setFocusTarget(id)}
            activeCameraId={activeCameraId}
            transformMode={transformMode}
            onTransformModeChange={setTransformMode}
            onTransformCommit={(id, position, rotation) => void onTransformCommit(id, position, rotation)}
            show3D={threeDViewVisible}
            imageViews={dockImageViews}
          />
        </div>

        <CollapseHandle
          side="right"
          open={rightOpen}
          onToggle={() => setRightOpen((v) => !v)}
          onResizeStart={(e) => beginResize('right', e)}
        />

        {rightOpen && (
          <div
            style={{
              width: rightWidth,
              minWidth: rightWidth,
              maxWidth: rightWidth,
              borderLeft: '1px solid #1b2235',
              background: '#0f1630',
              color: '#e8ecff',
              flex: '0 0 auto',
              position: 'relative',
              zIndex: 5,
            }}
          >
            <Inspector
              selected={selected}
              activeCameraId={activeCameraId}
              onSetActiveCamera={setActiveCameraId}
              onDelete={(id) => void removeElement(id)}
              transformMode={transformMode}
              onTransformModeChange={setTransformMode}
              onPoseCommit={(id, position, rotation) => void onTransformCommit(id, position, rotation)}
              lookAtTargets={lookAtTargets}
            />
          </div>
        )}
      </div>
    </div>
  )
}
