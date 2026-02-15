import { useEffect, useMemo, useRef, useState } from 'react'
import {
  createCamera,
  fetchElementMeta,
  fetchElements,
  fetchEvents,
  updateElementMeta,
  deleteElement,
  resetProject,
  type ElementInfo,
  type SampleQuery,
} from './viewer/api'
import Inspector from './viewer/Inspector'
import Hierarchy, { type HierarchyProps, type HierarchyViewInfo } from './viewer/Hierarchy'
import DockWorkspace, { type DockImageView, type DockThreeDView, type DockWorkspaceHandle } from './viewer/DockWorkspace'
import TimelineBar from './viewer/TimelineBar'
import { useTimeline } from './viewer/useTimeline'
import { usePageActivity } from './viewer/usePageActivity'
import { HIERARCHY_DRAG_MIME, parseHierarchyElementDragPayload } from './viewer/dragPayload'

const MAIN_3D_VIEW_ID = 'view-3d-main'
const EXTRA_3D_VIEW_PREFIX = 'view-3d-'
const IMAGE_VIEW_PREFIX = 'view-image-'
const CAMERA_VIEW_PREFIX = 'view-camera-'

function imageViewId(imageId: string): string {
  return `${IMAGE_VIEW_PREFIX}${imageId}`
}

function imageIdFromViewId(viewId: string): string | null {
  if (!viewId.startsWith(IMAGE_VIEW_PREFIX)) return null
  return viewId.slice(IMAGE_VIEW_PREFIX.length)
}

function cameraViewId(cameraId: string): string {
  return `${CAMERA_VIEW_PREFIX}${cameraId}`
}

function cameraIdFromViewId(viewId: string): string | null {
  if (!viewId.startsWith(CAMERA_VIEW_PREFIX)) return null
  return viewId.slice(CAMERA_VIEW_PREFIX.length)
}

function isExtraThreeDViewId(viewId: string): boolean {
  return viewId.startsWith(EXTRA_3D_VIEW_PREFIX) && viewId !== MAIN_3D_VIEW_ID
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

type ViewCameraSnapshot = {
  position: [number, number, number]
  rotation: [number, number, number, number]
  fov?: number
  near?: number
  far?: number
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
      className="panel-handle"
      onMouseDown={onResizeStart}
      style={{
        zIndex: 7,
        borderLeft: isLeft ? 0 : '1px solid var(--line)',
        borderRight: isLeft ? '1px solid var(--line)' : 0,
        cursor: open ? 'col-resize' : 'default',
      }}
    >
      <button
        type="button"
        aria-label={label}
        title={label}
        onMouseDown={(e) => e.stopPropagation()}
        onClick={onToggle}
        style={{
          opacity: open ? 1 : 0.85,
        }}
      >
        <span style={{ fontSize: 12, lineHeight: 1 }}>{glyph}</span>
      </button>
    </div>
  )
}

function EdgeReopenButton({
  side,
  onClick,
}: {
  side: 'left' | 'right'
  onClick: () => void
}) {
  const isLeft = side === 'left'
  return (
    <button
      type="button"
      className={`edge-reopen edge-reopen-${side}`}
      aria-label={isLeft ? 'Expand scene panel' : 'Expand inspector panel'}
      title={isLeft ? 'Expand scene panel' : 'Expand inspector panel'}
      onClick={onClick}
    >
      <span style={{ fontSize: 12, lineHeight: 1 }}>{isLeft ? '▶' : '◀'}</span>
    </button>
  )
}

export default function App() {
  const pageActive = usePageActivity()
  const [elements, setElements] = useState<ElementInfo[] | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [leftOpen, setLeftOpen] = useState(true)
  const [rightOpen, setRightOpen] = useState(true)
  const [focusTarget, setFocusTarget] = useState<string | null>(null)
  const [activeCameraId, setActiveCameraId] = useState<string | null>(null)
  const [transformMode, setTransformMode] = useState<'translate' | 'rotate' | 'animate'>('translate')
  const lastRev = useRef<number>(-1)
  const timelineSampleRef = useRef<SampleQuery | undefined>(undefined)
  const sampleFetchAliveRef = useRef(true)
  const sampleFetchRunningRef = useRef(false)
  const sampleFetchQueuedTokenRef = useRef(0)
  const sampleFetchQueuedSampleRef = useRef<SampleQuery | undefined>(undefined)
  const lastOpenCameraViewSeq = useRef<number>(0)
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
  const [extraThreeDViewOrder, setExtraThreeDViewOrder] = useState<string[]>([])
  const [extraThreeDViewState, setExtraThreeDViewState] = useState<Record<string, { visible: boolean; deleted: boolean; name: string; initialCamera: ViewCameraSnapshot | null }>>({})
  const [imageViewOrder, setImageViewOrder] = useState<string[]>([])
  const [imageViewState, setImageViewState] = useState<Record<string, { visible: boolean; deleted: boolean }>>({})
  const [cameraViewOrder, setCameraViewOrder] = useState<string[]>([])
  const [cameraViewState, setCameraViewState] = useState<Record<string, { visible: boolean; deleted: boolean }>>({})
  const timeline = useTimeline({ enabled: true })

  const pointclouds = useMemo(() => (elements ?? []).filter((e) => e.type === 'pointcloud'), [elements])
  const gaussians = useMemo(() => (elements ?? []).filter((e) => e.type === 'gaussians'), [elements])
  const boxes = useMemo(() => (elements ?? []).filter((e) => e.type === 'box3d'), [elements])
  const ellipsoids = useMemo(() => (elements ?? []).filter((e) => e.type === 'ellipsoid3d'), [elements])
  const cameras = useMemo(() => (elements ?? []).filter((e) => e.type === 'camera'), [elements])
  const images = useMemo(() => (elements ?? []).filter((e) => e.type === 'image'), [elements])
  const camerasById = useMemo(() => new Map(cameras.map((cam) => [cam.id, cam])), [cameras])
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

  useEffect(() => {
    const ids = cameras.map((cam) => cam.id)
    const idSet = new Set(ids)

    setCameraViewOrder((prev) => {
      const kept = prev.filter((id) => idSet.has(id))
      return kept.length === prev.length ? prev : kept
    })

    setCameraViewState((prev) => {
      let changed = false
      const next: Record<string, { visible: boolean; deleted: boolean }> = {}
      for (const [id, state] of Object.entries(prev)) {
        if (!idSet.has(id)) {
          changed = true
          continue
        }
        next[id] = state
      }
      return changed ? next : prev
    })
  }, [cameras])

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

  const cameraViews = useMemo(() => {
    const out: Array<{
      id: string
      cameraId: string
      name: string
      visible: boolean
      deleted: boolean
      elementVisible: boolean
    }> = []
    for (const cameraId of cameraViewOrder) {
      const cam = camerasById.get(cameraId)
      if (!cam) continue
      const state = cameraViewState[cameraId] ?? { visible: false, deleted: false }
      out.push({
        id: cameraViewId(cameraId),
        cameraId,
        name: `${cam.name} view`,
        visible: state.visible,
        deleted: state.deleted,
        elementVisible: cam.visible !== false,
      })
    }
    return out
  }, [cameraViewOrder, cameraViewState, camerasById])

  const extraThreeDViews = useMemo(() => {
    const out: Array<{ id: string; name: string; visible: boolean; deleted: boolean; initialCamera: ViewCameraSnapshot | null }> = []
    for (const viewId of extraThreeDViewOrder) {
      const state = extraThreeDViewState[viewId]
      if (!state) continue
      out.push({
        id: viewId,
        name: state.name,
        visible: state.visible,
        deleted: state.deleted,
        initialCamera: state.initialCamera ?? null,
      })
    }
    return out
  }, [extraThreeDViewOrder, extraThreeDViewState])

  const dockThreeDViews = useMemo<DockThreeDView[]>(
    () =>
      extraThreeDViews
        .filter((v) => !v.deleted && v.visible)
        .map((v) => ({
          id: v.id,
          name: v.name,
          visible: true,
          initialCamera: v.initialCamera ?? null,
        })),
    [extraThreeDViews],
  )

  const dockCameraViews = useMemo(
    () =>
      cameraViews
        .filter((v) => !v.deleted && v.visible && v.elementVisible)
        .map((v) => ({
          id: v.id,
          cameraId: v.cameraId,
          name: v.name,
          visible: true,
        })),
    [cameraViews]
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
      ...extraThreeDViews
        .filter((v) => !v.deleted)
        .map((v) => ({
          id: v.id,
          name: v.name,
          kind: '3d' as const,
          visible: v.visible,
          canDelete: true,
          canReorder: false,
          elementIds: threeDElementIds,
        })),
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
      ...cameraViews
        .filter((v) => !v.deleted)
        .map((v) => ({
          id: v.id,
          name: v.name,
          kind: 'camera' as const,
          visible: v.visible && v.elementVisible,
          canDelete: true,
          canReorder: false,
          elementIds: [v.cameraId],
        })),
    ]
  }, [cameraViews, elements, extraThreeDViews, imageViews, threeDViewVisible])

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

  const setCameraViewFlags = (cameraId: string, patch: Partial<{ visible: boolean; deleted: boolean }>) => {
    setCameraViewState((prev) => {
      const cur = prev[cameraId] ?? { visible: false, deleted: false }
      return { ...prev, [cameraId]: { ...cur, ...patch } }
    })
  }

  const setExtraThreeDViewFlags = (viewId: string, patch: Partial<{ visible: boolean; deleted: boolean; name: string; initialCamera: ViewCameraSnapshot | null }>) => {
    setExtraThreeDViewState((prev) => {
      const cur = prev[viewId]
      if (!cur) return prev
      return { ...prev, [viewId]: { ...cur, ...patch } }
    })
  }

  const getCurrentViewSnapshot = (): ViewCameraSnapshot | null => {
    try {
      const anyWin = window as any
      const cam = anyWin.__begira_view_camera as
        | {
            position?: [number, number, number]
            rotation?: [number, number, number, number]
            fov?: number
            near?: number
            far?: number
          }
        | undefined
      if (!cam || !Array.isArray(cam.position) || !Array.isArray(cam.rotation)) return null
      if (cam.position.length !== 3 || cam.rotation.length !== 4) return null
      const snapshot: ViewCameraSnapshot = {
        position: [Number(cam.position[0]), Number(cam.position[1]), Number(cam.position[2])],
        rotation: [Number(cam.rotation[0]), Number(cam.rotation[1]), Number(cam.rotation[2]), Number(cam.rotation[3])],
      }
      if (typeof cam.fov === 'number' && Number.isFinite(cam.fov)) snapshot.fov = cam.fov
      if (typeof cam.near === 'number' && Number.isFinite(cam.near)) snapshot.near = cam.near
      if (typeof cam.far === 'number' && Number.isFinite(cam.far)) snapshot.far = cam.far
      return snapshot
    } catch {
      return null
    }
  }

  const addThreeDView = () => {
    const id = `${EXTRA_3D_VIEW_PREFIX}${Math.random().toString(16).slice(2, 10)}`
    const existingVisibleCount = 1 + extraThreeDViews.filter((v) => !v.deleted).length
    const name = `3D ${existingVisibleCount + 1}`
    const initialCamera = getCurrentViewSnapshot()
    setExtraThreeDViewOrder((prev) => [...prev, id])
    setExtraThreeDViewState((prev) => ({
      ...prev,
      [id]: { visible: true, deleted: false, name, initialCamera },
    }))
    setFocusTarget(null)
    window.setTimeout(() => {
      workspaceRef.current?.focusThreeDView(id)
    }, 0)
  }

  const addCamera = async () => {
    try {
      const viewCam = getCurrentViewSnapshot()
      const nextIndex = cameras.length + 1
      const req: {
        name: string
        position?: [number, number, number]
        rotation?: [number, number, number, number]
        fov?: number
        near?: number
        far?: number
      } = { name: `camera_${nextIndex}` }
      if (viewCam?.position) req.position = viewCam.position
      if (viewCam?.rotation) req.rotation = viewCam.rotation
      if (typeof viewCam?.fov === 'number' && Number.isFinite(viewCam.fov)) req.fov = viewCam.fov
      if (typeof viewCam?.near === 'number' && Number.isFinite(viewCam.near)) req.near = viewCam.near
      if (typeof viewCam?.far === 'number' && Number.isFinite(viewCam.far)) req.far = viewCam.far
      const created = await createCamera(req)
      const items = await fetchElements(timeline.sampleQuery)
      const ordered = orderElements(items)
      setElements(ordered)
      setSelectedId(created.id)
      setFocusTarget(created.id)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const openCameraView = (cameraId: string) => {
    setCameraViewOrder((prev) => (prev.includes(cameraId) ? prev : [...prev, cameraId]))
    setCameraViewFlags(cameraId, { visible: true, deleted: false })
    setSelectedId(cameraId)
    // Opening a camera view should not force the main 3D viewport into that camera.
    setFocusTarget(null)
    window.setTimeout(() => {
      workspaceRef.current?.focusCamera(cameraId)
    }, 0)
  }

  const hasHierarchyDragPayload = (dataTransfer: DataTransfer | null): boolean => {
    if (!dataTransfer) return false
    return Array.from(dataTransfer.types).includes(HIERARCHY_DRAG_MIME)
  }

  const handleWorkspaceDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    if (!hasHierarchyDragPayload(event.dataTransfer)) return
    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
  }

  const handleWorkspaceDrop = (event: React.DragEvent<HTMLDivElement>) => {
    if (!hasHierarchyDragPayload(event.dataTransfer)) return
    event.preventDefault()
    const raw = event.dataTransfer.getData(HIERARCHY_DRAG_MIME)
    const payload = parseHierarchyElementDragPayload(raw)
    if (!payload || payload.elementType !== 'camera') return
    openCameraView(payload.elementId)
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

  const applyFetchedElements = (items: ElementInfo[]) => {
    const ordered = orderElements(items)
    setElements(ordered)
    setSelectedId((cur) => {
      if (!cur) return cur
      return ordered.some((e) => e.id === cur) ? cur : null
    })

    if (!didInitFocus.current && ordered.length > 0) {
      didInitFocus.current = true
      setFocusTarget(ordered[0].id)
    }
  }

  const pumpSampleFetchQueue = async () => {
    if (sampleFetchRunningRef.current) return
    sampleFetchRunningRef.current = true
    try {
      while (sampleFetchAliveRef.current) {
        const token = sampleFetchQueuedTokenRef.current
        const sample = sampleFetchQueuedSampleRef.current
        if (token === 0) break
        sampleFetchQueuedTokenRef.current = 0
        const items = await fetchElements(sample)
        if (!sampleFetchAliveRef.current) return
        applyFetchedElements(items)
      }
    } catch (e: unknown) {
      if (!sampleFetchAliveRef.current) return
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      sampleFetchRunningRef.current = false
      if (sampleFetchAliveRef.current && sampleFetchQueuedTokenRef.current !== 0) {
        void pumpSampleFetchQueue()
      }
    }
  }

  const enqueueSampleFetch = (sample: SampleQuery | undefined) => {
    sampleFetchQueuedSampleRef.current = sample
    sampleFetchQueuedTokenRef.current += 1
    void pumpSampleFetchQueue()
  }

  useEffect(() => {
    sampleFetchAliveRef.current = true
    return () => {
      sampleFetchAliveRef.current = false
      sampleFetchQueuedTokenRef.current = 0
    }
  }, [])

  useEffect(() => {
    timelineSampleRef.current = timeline.sampleQuery
  }, [timeline.sampleKey])

  useEffect(() => {
    enqueueSampleFetch(timeline.sampleQuery)
  }, [timeline.sampleKey])

  useEffect(() => {
    if (!pageActive) return
    let cancelled = false
    const tick = async () => {
      try {
        const ev = await fetchEvents()
        if (cancelled) return
        const openCameraCmd = ev.viewerCommands?.openCameraView
        if (
          openCameraCmd &&
          Number.isFinite(openCameraCmd.seq) &&
          Number(openCameraCmd.seq) > lastOpenCameraViewSeq.current &&
          typeof openCameraCmd.cameraId === 'string' &&
          openCameraCmd.cameraId.length > 0
        ) {
          lastOpenCameraViewSeq.current = Number(openCameraCmd.seq)
          openCameraView(openCameraCmd.cameraId)
        }
        if (ev.globalRevision === lastRev.current) return
        lastRev.current = ev.globalRevision
        enqueueSampleFetch(timelineSampleRef.current)
      } catch {
        // Ignore transient poll errors.
      }
    }
    const id = window.setInterval(() => void tick(), 1000)
    void tick()
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [pageActive])

  const selected = useMemo(() => {
    if (!elements || selectedId === null) return null
    return elements.find((c) => c.id === selectedId) ?? null
  }, [elements, selectedId])

  useEffect(() => {
    if (activeCameraId && !elements?.some((e) => e.id === activeCameraId)) {
      setActiveCameraId(null)
    }
  }, [activeCameraId, elements])

  useEffect(() => {
    try {
      const anyWin = window as any
      anyWin.__begira_local_pose = {}
      window.dispatchEvent(new CustomEvent('begira_local_pose_cleared'))
    } catch {}
  }, [timeline.sampleKey])

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

  const getLocalPose = (id: string): { position: [number, number, number]; rotation: [number, number, number, number] } | null => {
    try {
      const anyWin = window as any
      const pose = anyWin.__begira_local_pose?.[id]
      if (!pose) return null
      const pos = pose.position
      const rot = pose.rotation
      if (!Array.isArray(pos) || pos.length !== 3) return null
      if (!Array.isArray(rot) || rot.length !== 4) return null
      return {
        position: [Number(pos[0]), Number(pos[1]), Number(pos[2])],
        rotation: [Number(rot[0]), Number(rot[1]), Number(rot[2]), Number(rot[3])],
      }
    } catch {
      return null
    }
  }

  const extractFreshElementPatch = (id: string, meta: any): Partial<ElementInfo> => {
    const patch: Partial<ElementInfo> = {}
    const local = getLocalPose(id)
    if (local) {
      patch.position = local.position
      patch.rotation = local.rotation
      return patch
    }

    if (Array.isArray(meta?.position) && meta.position.length === 3) {
      patch.position = [Number(meta.position[0]), Number(meta.position[1]), Number(meta.position[2])]
    }
    if (Array.isArray(meta?.rotation) && meta.rotation.length === 4) {
      patch.rotation = [Number(meta.rotation[0]), Number(meta.rotation[1]), Number(meta.rotation[2]), Number(meta.rotation[3])]
    }
    if (typeof meta?.fov === 'number' && Number.isFinite(meta.fov)) patch.fov = meta.fov
    if (typeof meta?.near === 'number' && Number.isFinite(meta.near)) patch.near = meta.near
    if (typeof meta?.far === 'number' && Number.isFinite(meta.far)) patch.far = meta.far
    if (typeof meta?.revision === 'number' && Number.isFinite(meta.revision)) patch.revision = meta.revision
    return patch
  }

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase()
      if (tag === 'input' || tag === 'textarea' || (e.target as HTMLElement | null)?.isContentEditable) return
      const mod = e.metaKey || e.ctrlKey
      if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault()
        timeline.togglePlay()
        return
      }
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        timeline.step(-1)
        return
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault()
        timeline.step(1)
        return
      }
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
  }, [timeline, undo, redo])

  const applyVisibility = async (id: string, visible: boolean) => {
    const prev = elements?.find((e) => e.id === id)
    await runUserAction({
      label: visible ? 'Show' : 'Hide',
      do: async () => {
        await updateElementMeta(id, { visible })
        const freshMeta = visible ? await fetchElementMeta(id, timeline.sampleQuery).catch(() => null) : null
        const patch = freshMeta ? extractFreshElementPatch(id, freshMeta) : {}
        setElements((cur) => (cur ? cur.map((e) => (e.id === id ? { ...e, ...patch, visible } : e)) : cur))
      },
      undo: async () => {
        const undoVisible = prev?.visible !== false
        await updateElementMeta(id, { visible: undoVisible })
        const freshMeta = undoVisible ? await fetchElementMeta(id, timeline.sampleQuery).catch(() => null) : null
        const patch = freshMeta ? extractFreshElementPatch(id, freshMeta) : {}
        setElements((cur) => (cur ? cur.map((e) => (e.id === id ? { ...e, ...patch, visible: undoVisible } : e)) : cur))
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
    timeline.reinitialize()
    setHistoryCounts({ undo: 0, redo: 0 })
    await resetProject()
    let refreshSample = timeline.sampleQuery
    try {
      refreshSample = await timeline.refresh({ reinitialize: true })
    } catch {
      // keep current timeline cursor on refresh failures
    }
    const items = await fetchElements(refreshSample)
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

  const onTransformCommit = async (id: string, position: [number, number, number], rotation: [number, number, number, number]) => {
    if (resettingRef.current) return
    const prev = await fetchElementMeta(id, timeline.sampleQuery)
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
    <div className="app">
      <div className="header">
        <div className="brand-mark" />
        <div className="brand-title">
          <strong>begira</strong>
          <span>viewer</span>
        </div>
        <div className="header-right">
          {error && <span className="header-error">{error}</span>}
          <div className="header-actions">
            <button
              className="toolbar-btn"
              onClick={() => void undo()}
              disabled={historyCounts.undo === 0 || isResetting}
              style={{ opacity: historyCounts.undo === 0 || isResetting ? 0.5 : 1 }}
            >
              Undo
            </button>
            <button
              className="toolbar-btn"
              onClick={() => void redo()}
              disabled={historyCounts.redo === 0 || isResetting}
              style={{ opacity: historyCounts.redo === 0 || isResetting ? 0.5 : 1 }}
            >
              Redo
            </button>
            <button className="toolbar-btn danger" onClick={() => void resetAll()}>
              Reset
            </button>
          </div>
        </div>
      </div>

      <div className="viewer">
        {leftOpen && (
          <div
            className="side-panel"
            style={{
              width: leftWidth,
              minWidth: leftWidth,
              maxWidth: leftWidth,
              flex: '0 0 auto',
            }}
          >
            <div className="panel-scroll">
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
                  views: hierarchyViews,
                  onAddCamera: () => void addCamera(),
                  onAdd3DView: addThreeDView,
                  onActivateView: (viewId: string) => {
                    if (viewId === MAIN_3D_VIEW_ID || isExtraThreeDViewId(viewId)) {
                      workspaceRef.current?.focusThreeDView(viewId)
                      return
                    }
                    const imageId = imageIdFromViewId(viewId)
                    if (imageId) {
                      openImageView(imageId)
                      return
                    }
                    const cameraId = cameraIdFromViewId(viewId)
                    if (cameraId) openCameraView(cameraId)
                  },
                  onToggleViewVisibility: (viewId: string, visible: boolean) => {
                    if (viewId === MAIN_3D_VIEW_ID) {
                      setThreeDViewVisible(visible)
                      return
                    }
                    if (isExtraThreeDViewId(viewId)) {
                      setExtraThreeDViewFlags(viewId, { visible })
                      return
                    }
                    const imageId = imageIdFromViewId(viewId)
                    if (imageId) {
                      setImageViewFlags(imageId, { visible })
                      void applyVisibility(imageId, visible)
                      return
                    }
                    const cameraId = cameraIdFromViewId(viewId)
                    if (!cameraId) return
                    setCameraViewFlags(cameraId, { visible })
                  },
                  onDeleteView: (viewId: string) => {
                    if (isExtraThreeDViewId(viewId)) {
                      setExtraThreeDViewFlags(viewId, { deleted: true, visible: false })
                      setExtraThreeDViewOrder((prev) => prev.filter((id) => id !== viewId))
                      return
                    }
                    const imageId = imageIdFromViewId(viewId)
                    if (imageId) {
                      setImageViewFlags(imageId, { deleted: true, visible: false })
                      setImageViewOrder((prev) => prev.filter((id) => id !== imageId))
                      return
                    }
                    const cameraId = cameraIdFromViewId(viewId)
                    if (!cameraId) return
                    setCameraViewFlags(cameraId, { deleted: true, visible: false })
                    setCameraViewOrder((prev) => prev.filter((id) => id !== cameraId))
                  },
                  onMoveView: (viewId: string, direction: 'up' | 'down') => {
                    const imageId = imageIdFromViewId(viewId)
                    if (imageId) {
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
                      return
                    }
                  },
                } satisfies HierarchyProps)}
              />
            </div>
          </div>
        )}

        <CollapseHandle
          side="left"
          open={leftOpen}
          onToggle={() => setLeftOpen((v) => !v)}
          onResizeStart={(e) => beginResize('left', e)}
        />

        <div className="workspace-shell" style={{ flex: 1 }} onDragOver={handleWorkspaceDragOver} onDrop={handleWorkspaceDrop}>
          <DockWorkspace
            ref={workspaceRef}
            pointclouds={pointclouds}
            gaussians={gaussians}
            boxes={boxes}
            ellipsoids={ellipsoids}
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
            onRunUserAction={(action) => runUserAction(action)}
            onSelectTimelineFrame={(frame) => {
              const currentAxisInfo = timeline.axes.find((a) => a.axis === timeline.axis)
              if (currentAxisInfo?.kind !== 'sequence') {
                const sequenceAxis = timeline.axes.find((a) => a.kind === 'sequence')
                if (sequenceAxis) {
                  timeline.setAxis(sequenceAxis.axis)
                  window.requestAnimationFrame(() => timeline.setValue(frame))
                  return
                }
              }
              timeline.setValue(frame)
            }}
            show3D={threeDViewVisible}
            extraThreeDViews={dockThreeDViews}
            imageViews={dockImageViews}
            cameraViews={dockCameraViews}
            sample={timeline.sampleQuery}
          />
        </div>

        {rightOpen ? (
          <CollapseHandle
            side="right"
            open={rightOpen}
            onToggle={() => setRightOpen((v) => !v)}
            onResizeStart={(e) => beginResize('right', e)}
          />
        ) : (
          <EdgeReopenButton side="right" onClick={() => setRightOpen(true)} />
        )}

        {rightOpen && (
          <div
            className="side-panel right"
            style={{
              width: rightWidth,
              minWidth: rightWidth,
              maxWidth: rightWidth,
              flex: '0 0 auto',
            }}
          >
            <div className="panel-scroll">
              <Inspector
                selected={selected}
                activeCameraId={activeCameraId}
                onDelete={(id) => void removeElement(id)}
                transformMode={transformMode}
                onTransformModeChange={setTransformMode}
                onPoseCommit={(id, position, rotation) => void onTransformCommit(id, position, rotation)}
                lookAtTargets={lookAtTargets}
                sample={timeline.sampleQuery}
                enabled={pageActive}
                onOpenCameraView={openCameraView}
              />
            </div>
          </div>
        )}
      </div>

      <TimelineBar
        axis={timeline.axis}
        axes={timeline.axes}
        value={timeline.value}
        bounds={timeline.bounds}
        isPlaying={timeline.isPlaying}
        loopPlayback={timeline.loopPlayback}
        playbackFps={timeline.playbackFps}
        onAxisChange={timeline.setAxis}
        onValueChange={timeline.setValue}
        onToggleLoop={timeline.setLoopPlayback}
        onTogglePlay={timeline.togglePlay}
        onScrubStart={timeline.beginScrub}
        onScrubEnd={timeline.endScrub}
        onStep={timeline.step}
        onPlaybackFpsChange={timeline.setPlaybackFps}
      />
    </div>
  )
}
