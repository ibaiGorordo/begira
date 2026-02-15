import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import type { ElementInfo, SampleQuery } from './api'
import {
  fetchPointCloudElementMeta,
  fetchGaussianElementMeta,
  fetchCameraElementMeta,
  fetchImageElementMeta,
  fetchElementMeta,
  updatePointCloudSettings,
  updateElementMeta,
  deleteElement,
} from './api'
import { COLORMAPS, DEFAULT_DEPTH_COLORMAP, DEFAULT_HEIGHT_COLORMAP, type ColormapId } from './colormaps'

type Props = {
  selected: ElementInfo | null
  activeCameraId: string | null
  onSetActiveCamera: (id: string | null) => void
  onDelete: (id: string) => void
  transformMode: 'translate' | 'rotate'
  onTransformModeChange: (mode: 'translate' | 'rotate') => void
  onPoseCommit: (id: string, position: [number, number, number], rotation: [number, number, number, number]) => void
  onOpenCameraView: (cameraId: string) => void
  sample?: SampleQuery
  enabled?: boolean
  lookAtTargets: Array<{
    id: string
    name: string
    type: ElementInfo['type']
    position: [number, number, number]
  }>
}

export default function Inspector({
  selected,
  activeCameraId,
  onSetActiveCamera,
  onDelete,
  transformMode,
  onTransformModeChange,
  onPoseCommit,
  onOpenCameraView,
  sample,
  enabled = true,
  lookAtTargets,
}: Props) {
  const [pointSize, setPointSize] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const lastSent = useRef<number | null>(null)
  const lastPoseSent = useRef<string | null>(null)
  const lastCameraSent = useRef<string | null>(null)
  const lastSelectedId = useRef<string | null>(null)
  const suppressPoseCommitRef = useRef(false)
  const manualPoseEditRef = useRef(false)
  const round3 = (v: number) => Math.round(v * 1000) / 1000

  const isPointCloud = selected?.type === 'pointcloud'
  const isGaussians = selected?.type === 'gaussians'
  const isCamera = selected?.type === 'camera'
  const isImage = selected?.type === 'image'

  // LOD override state
  const [lodOverride, setLodOverride] = useState<string | undefined>(undefined)

  // Visual override state (client-only preview)
  type ColorMode = 'logged' | 'solid' | 'height' | 'depth'
  const [colorMode, setColorMode] = useState<ColorMode>('logged')
  const [solidColor, setSolidColor] = useState<string>('#ff8a33') // hex string for input[type=color]
  const [colorMap, setColorMap] = useState<ColormapId>(DEFAULT_HEIGHT_COLORMAP)
  const [isVisible, setIsVisible] = useState(true)
  const [position, setPosition] = useState<[number, number, number] | null>(null)
  const [rotation, setRotation] = useState<[number, number, number, number] | null>(null)
  const [rotationEuler, setRotationEuler] = useState<[number, number, number] | null>(null)
  const dragRef = useRef<{
    startX: number
    startY: number
    startVal: number
    index: number
    active: boolean
  } | null>(null)
  const [fov, setFov] = useState<number | null>(null)
  const [near, setNear] = useState<number | null>(null)
  const [far, setFar] = useState<number | null>(null)
  const [imgWidth, setImgWidth] = useState<number | null>(null)
  const [imgHeight, setImgHeight] = useState<number | null>(null)
  const [imgChannels, setImgChannels] = useState<number | null>(null)
  const [imgMime, setImgMime] = useState<string | null>(null)
  const [lookAtTargetId, setLookAtTargetId] = useState<string>('')

  const availableLookAtTargets = lookAtTargets.filter((t) => t.id !== selected?.id && t.type !== 'image')

  useEffect(() => {
    setErr(null)
    lastSent.current = null
    if (!selected) {
      setPointSize(null)
      return
    }

    const fetchMeta = isPointCloud
      ? fetchPointCloudElementMeta
      : isGaussians
        ? fetchGaussianElementMeta
        : isCamera
          ? fetchCameraElementMeta
          : isImage
            ? fetchImageElementMeta
          : null
    if (!fetchMeta) {
      setPointSize(null)
      return
    }

    if (!enabled) return
    fetchMeta(selected.id, sample)
      .then((m: any) => {
        suppressPoseCommitRef.current = true
        setPointSize(m.pointSize ?? null)
        setPosition(m.position ?? [0, 0, 0])
        const quat: [number, number, number, number] = (m.rotation ?? [0, 0, 0, 1]) as any
        setRotation(quat)
        try {
          const q = new THREE.Quaternion(quat[0], quat[1], quat[2], quat[3]).normalize()
          const euler = new THREE.Euler().setFromQuaternion(q, 'XYZ')
          setRotationEuler([
            THREE.MathUtils.radToDeg(euler.x),
            THREE.MathUtils.radToDeg(euler.y),
            THREE.MathUtils.radToDeg(euler.z),
          ])
        } catch {
          setRotationEuler([0, 0, 0])
        }
        setIsVisible(m.visible !== false)
        setFov(m.fov ?? null)
        setNear(m.near ?? null)
        setFar(m.far ?? null)
        setImgWidth(m.width ?? null)
        setImgHeight(m.height ?? null)
        setImgChannels(m.channels ?? null)
        setImgMime(m.mimeType ?? null)
        window.setTimeout(() => {
          suppressPoseCommitRef.current = false
        }, 0)
      })
      .catch((e: unknown) => setErr(e instanceof Error ? e.message : String(e)))

    // initialize client-side values from window globals if present
    try {
      const anyWin = window as any
      const lod = (anyWin.__begira_lod_override && anyWin.__begira_lod_override[selected.id]) ?? undefined
      setLodOverride(lod)
      // visual override (client-only)
      const vis = anyWin.__begira_visual_override && anyWin.__begira_visual_override[selected.id]
      if (vis) {
        const mode = (vis.colorMode ?? 'logged') as ColorMode
        setColorMode(mode)
        const fallback =
          mode === 'depth' ? DEFAULT_DEPTH_COLORMAP : mode === 'height' ? DEFAULT_HEIGHT_COLORMAP : DEFAULT_HEIGHT_COLORMAP
        setColorMap((vis.colorMap as ColormapId) ?? fallback)
        if (vis.solidColor) {
          // assume [r,g,b] floats 0..1
          const c = vis.solidColor
          function toHex(x: number) {
            const v = Math.round(Math.max(0, Math.min(1, x)) * 255)
            return v.toString(16).padStart(2, '0')
          }
          const hex = '#' + toHex(c[0]) + toHex(c[1]) + toHex(c[2])
          setSolidColor(hex)
        }
      } else {
        setColorMode('logged')
        setColorMap(DEFAULT_HEIGHT_COLORMAP)
      }
    } catch {}
  }, [enabled, selected?.id, isPointCloud, isGaussians, isCamera, isImage, sample?.frame, sample?.timestamp])

  useEffect(() => {
    try {
      const anyWin = window as any
      const map = anyWin.__begira_local_pose || {}
      if (lastSelectedId.current && lastSelectedId.current !== selected?.id) {
        delete map[lastSelectedId.current]
      }
      lastSelectedId.current = selected?.id ?? null
      anyWin.__begira_local_pose = map
    } catch {}
  }, [selected?.id])

  useEffect(() => {
    if (!enabled || !selected) return
    try {
      const anyWin = window as any
      if (!anyWin.__begira_local_pose) return
      delete anyWin.__begira_local_pose[selected.id]
    } catch {}
  }, [enabled, selected?.id, sample?.frame, sample?.timestamp])

  useEffect(() => {
    const handler = () => {
      manualPoseEditRef.current = false
      lastPoseSent.current = null
      lastCameraSent.current = null
    }
    window.addEventListener('begira_reset', handler)
    return () => window.removeEventListener('begira_reset', handler)
  }, [])

  useEffect(() => {
    if (!selected || pointSize === null) return
    if (!isPointCloud && !isGaussians) return

    const v = pointSize
    const handle = window.setTimeout(async () => {
      if (lastSent.current !== null && Math.abs(lastSent.current - v) < 1e-9) return

      setBusy(true)
      setErr(null)
      try {
        await updatePointCloudSettings(selected.id, { pointSize: v })
        lastSent.current = v
      } catch (e: unknown) {
        setErr(e instanceof Error ? e.message : String(e))
      } finally {
        setBusy(false)
      }
    }, 200)

    return () => window.clearTimeout(handle)
  }, [pointSize, selected?.id, isPointCloud, isGaussians])

  // Handlers for the new UI that update window globals the scene reads
  const setLodOverrideForElement = (id: string, v: string | undefined) => {
    try {
      const anyWin = window as any
      anyWin.__begira_lod_override = anyWin.__begira_lod_override || {}
      if (v === undefined) delete anyWin.__begira_lod_override[id]
      else anyWin.__begira_lod_override[id] = v
      // Also publish to apply function to ensure visibility updates (no-op, kept for backward compat)
    } catch {}
  }

  const setVisualOverrideForElement = (
    id: string,
    mode: ColorMode,
    hexColor?: string | null,
    mapId: ColormapId = DEFAULT_HEIGHT_COLORMAP,
  ) => {
    try {
      const anyWin = window as any
      anyWin.__begira_visual_override = anyWin.__begira_visual_override || {}
      if (mode === 'logged') {
        // clear override
        delete anyWin.__begira_visual_override[id]
      } else {
        const obj: any = { colorMode: mode }
        if (mode === 'solid' && hexColor) {
          // convert #rrggbb to [r,g,b] floats
          const c = hexColor.replace('#', '')
          const r = parseInt(c.substring(0, 2), 16) / 255
          const g = parseInt(c.substring(2, 4), 16) / 255
          const b = parseInt(c.substring(4, 6), 16) / 255
          obj.solidColor = [r, g, b]
        }
        if (mode === 'height' || mode === 'depth') {
          obj.colorMap = mapId
        }
        anyWin.__begira_visual_override[id] = obj
      }
      // notify viewers to re-read overrides and re-render
      try {
        if (typeof window !== 'undefined' && (window as any).dispatchEvent) {
          const ev = new CustomEvent('begira_visual_override_changed', { detail: { id } })
          window.dispatchEvent(ev)
        }
      } catch {}
    } catch {}
  }

  const setVisibilityForElement = (id: string, visible: boolean) => {
    try {
      setIsVisible(visible)
      void updateElementMeta(id, { visible })
    } catch {}
  }

  useEffect(() => {
    if (!selected || !position || !rotation) return
    if (suppressPoseCommitRef.current) return
    if (!manualPoseEditRef.current) return
    const key = `${position.join(',')}|${rotation.join(',')}`
    if (lastPoseSent.current === key) return
    const handle = window.setTimeout(() => {
      const q = new THREE.Quaternion(rotation[0], rotation[1], rotation[2], rotation[3]).normalize()
      onPoseCommit(selected.id, position, [q.x, q.y, q.z, q.w])
      lastPoseSent.current = key
      manualPoseEditRef.current = false
    }, 50)
    return () => window.clearTimeout(handle)
  }, [position, rotation, selected?.id, onPoseCommit])

  useEffect(() => {
    const handler = (e: any) => {
      if (!selected) return
      if (e?.detail?.id !== selected.id) return
      const pos = e.detail.position as [number, number, number] | undefined
      const rot = e.detail.rotation as [number, number, number, number] | undefined
      if (!pos || !rot) return
      suppressPoseCommitRef.current = true
      setPosition((prev) => (JSON.stringify(prev) === JSON.stringify(pos) ? prev : pos))
      setRotation((prev) => (JSON.stringify(prev) === JSON.stringify(rot) ? prev : rot))
      try {
        const q = new THREE.Quaternion(rot[0], rot[1], rot[2], rot[3]).normalize()
        const euler = new THREE.Euler().setFromQuaternion(q, 'XYZ')
        const nextEuler: [number, number, number] = [
          THREE.MathUtils.radToDeg(euler.x),
          THREE.MathUtils.radToDeg(euler.y),
          THREE.MathUtils.radToDeg(euler.z),
        ]
        setRotationEuler((prev) => (JSON.stringify(prev) === JSON.stringify(nextEuler) ? prev : nextEuler))
      } catch {}
      window.setTimeout(() => {
        suppressPoseCommitRef.current = false
      }, 0)
    }
    window.addEventListener('begira_local_pose_changed', handler)
    return () => window.removeEventListener('begira_local_pose_changed', handler)
  }, [selected?.id])

  useEffect(() => {
    let cancelled = false
    if (!selected) return

    const tick = async () => {
      try {
        if (cancelled || !selected) return
        const anyWin = window as any
        const localPose = anyWin.__begira_local_pose?.[selected.id]
        if (localPose && localPose.position && localPose.rotation) {
          const pos = localPose.position as [number, number, number]
          const rot = localPose.rotation as [number, number, number, number]
          setPosition((prev) => (JSON.stringify(prev) === JSON.stringify(pos) ? prev : pos))
          setRotation((prev) => (JSON.stringify(prev) === JSON.stringify(rot) ? prev : rot))
          try {
            const q = new THREE.Quaternion(rot[0], rot[1], rot[2], rot[3]).normalize()
            const euler = new THREE.Euler().setFromQuaternion(q, 'XYZ')
            const nextEuler: [number, number, number] = [
              THREE.MathUtils.radToDeg(euler.x),
              THREE.MathUtils.radToDeg(euler.y),
              THREE.MathUtils.radToDeg(euler.z),
            ]
            setRotationEuler((prev) => (JSON.stringify(prev) === JSON.stringify(nextEuler) ? prev : nextEuler))
          } catch {}
          return
        }
        const meta = await fetchElementMeta(selected.id, sample)
        if (cancelled) return
        suppressPoseCommitRef.current = true
        if (meta.position) {
          const pos = meta.position as [number, number, number]
          setPosition((prev) => (JSON.stringify(prev) === JSON.stringify(pos) ? prev : pos))
        }
        if (meta.rotation) {
          const rot = meta.rotation as [number, number, number, number]
          setRotation((prev) => (JSON.stringify(prev) === JSON.stringify(rot) ? prev : rot))
          try {
            const q = new THREE.Quaternion(rot[0], rot[1], rot[2], rot[3]).normalize()
            const euler = new THREE.Euler().setFromQuaternion(q, 'XYZ')
            const nextEuler: [number, number, number] = [
              THREE.MathUtils.radToDeg(euler.x),
              THREE.MathUtils.radToDeg(euler.y),
              THREE.MathUtils.radToDeg(euler.z),
            ]
            setRotationEuler((prev) => (JSON.stringify(prev) === JSON.stringify(nextEuler) ? prev : nextEuler))
          } catch {}
        }
        if (meta.visible !== undefined) setIsVisible(meta.visible !== false)
        if (meta.pointSize !== undefined) setPointSize(meta.pointSize)
        if (meta.fov !== undefined) setFov(meta.fov)
        if (meta.near !== undefined) setNear(meta.near)
        if (meta.far !== undefined) setFar(meta.far)
        if (meta.width !== undefined) setImgWidth(meta.width)
        if (meta.height !== undefined) setImgHeight(meta.height)
        if (meta.channels !== undefined) setImgChannels(meta.channels)
        if (meta.mimeType !== undefined) setImgMime(meta.mimeType)
        window.setTimeout(() => {
          suppressPoseCommitRef.current = false
        }, 0)
      } catch {}
    }

    const id = window.setInterval(() => void tick(), 500)
    void tick()
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [selected?.id, sample?.frame, sample?.timestamp])

  useEffect(() => {
    if (!selected || !position || !rotation) return
    if (!manualPoseEditRef.current) return
    try {
      const anyWin = window as any
      if (!anyWin.__begira_local_pose) anyWin.__begira_local_pose = {}
      anyWin.__begira_local_pose[selected.id] = {
        position: [...position],
        rotation: [...rotation],
      }
    } catch {}
  }, [position, rotation, selected?.id])

  useEffect(() => {
    if (!selected || !isCamera) return
    if (fov === null || near === null || far === null) return
    const key = `${fov}|${near}|${far}`
    if (lastCameraSent.current === key) return
    const handle = window.setTimeout(() => {
      void updateElementMeta(selected.id, { fov, near, far })
      lastCameraSent.current = key
    }, 50)
    return () => window.clearTimeout(handle)
  }, [fov, near, far, selected?.id, isCamera])

  useEffect(() => {
    if (!isCamera) {
      setLookAtTargetId('')
      return
    }
    if (availableLookAtTargets.length === 0) {
      setLookAtTargetId('')
      return
    }
    setLookAtTargetId((prev) => {
      if (prev && availableLookAtTargets.some((t) => t.id === prev)) return prev
      return availableLookAtTargets[0].id
    })
  }, [isCamera, selected?.id, availableLookAtTargets])

  const lookAtSelectedTarget = () => {
    if (!selected || !isCamera || !position) return
    if (!lookAtTargetId) return
    const target = availableLookAtTargets.find((t) => t.id === lookAtTargetId)
    if (!target) return

    const from = new THREE.Vector3(position[0], position[1], position[2])
    const to = new THREE.Vector3(target.position[0], target.position[1], target.position[2])
    const dir = to.clone().sub(from)
    if (dir.lengthSq() < 1e-12) return
    dir.normalize()

    const up = new THREE.Vector3(0, 1, 0)
    if (Math.abs(dir.dot(up)) > 0.98) up.set(0, 0, 1)

    const cam = new THREE.PerspectiveCamera()
    cam.position.copy(from)
    cam.up.copy(up)
    cam.lookAt(to)
    cam.updateMatrixWorld()

    const q = cam.quaternion.clone().normalize()
    const nextRotation: [number, number, number, number] = [q.x, q.y, q.z, q.w]
    const euler = new THREE.Euler().setFromQuaternion(q, 'XYZ')
    const nextEuler: [number, number, number] = [
      THREE.MathUtils.radToDeg(euler.x),
      THREE.MathUtils.radToDeg(euler.y),
      THREE.MathUtils.radToDeg(euler.z),
    ]

    manualPoseEditRef.current = true
    setRotation(nextRotation)
    setRotationEuler(nextEuler)
  }

  const startDragNumber =
    (
      index: number,
      current: number,
      apply: (index: number, value: number) => void,
      stepBase = 0.01,
    ) =>
    (e: React.MouseEvent<HTMLInputElement>) => {
    if (e.button !== 0) return
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startVal: current,
      index,
      active: false,
    }
    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current) return
      const dy = ev.clientY - dragRef.current.startY
      if (!dragRef.current.active && Math.abs(dy) < 2) return
      dragRef.current.active = true
      const step = ev.shiftKey ? stepBase * 0.1 : stepBase
      const nextVal = round3(dragRef.current.startVal - dy * step)
      manualPoseEditRef.current = true
      apply(dragRef.current.index, nextVal)
    }
    const onUp = () => {
      dragRef.current = null
      document.body.style.userSelect = ''
      document.body.style.cursor = ''
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    document.body.style.userSelect = 'none'
    document.body.style.cursor = 'ns-resize'
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  if (!selected) {
    return (
      <div className="inspector-layout">
        <h2 className="inspect-title">Inspector</h2>
        <div className="inspect-card">
          <div className="panel-subtitle">Select an element to inspect metadata and controls.</div>
        </div>
      </div>
    )
  }

  return (
    <div className="inspector-layout">
      <div>
        <h2 className="inspect-title">Inspector</h2>
        <div className="panel-subtitle">Element controls and metadata</div>
      </div>

      <div className="inspect-card">
        <div className="inspect-grid">
          <div>
            <div className="inspect-label">Name</div>
            <div className="inspect-value">{selected.name}</div>
          </div>
          <div>
            <div className="inspect-label">Type</div>
            <div className="inspect-value">{selected.type}</div>
          </div>
        </div>
        <div style={{ marginTop: 8 }}>
          <div className="inspect-label">Id</div>
          <div className="inspect-value inspect-mono">{selected.id}</div>
        </div>
      </div>

      <div className="inspect-card">
        <div className="pill-row" style={{ justifyContent: 'space-between' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
            <input type="checkbox" checked={isVisible} onChange={(e) => setVisibilityForElement(selected.id, e.target.checked)} />
            Visible
          </label>
          <button className="toolbar-btn danger" onClick={() => onDelete(selected.id)}>
            Remove
          </button>
        </div>
      </div>

      {isCamera && (
        <div className="inspect-card">
          <div className="inspect-label">Look At Object</div>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <select
              value={lookAtTargetId}
              onChange={(e) => setLookAtTargetId(e.target.value)}
              disabled={availableLookAtTargets.length === 0}
              style={{ flex: 1, minWidth: 0 }}
            >
              {availableLookAtTargets.length === 0 && <option value="">(no available 3D objects)</option>}
              {availableLookAtTargets.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name} ({t.type})
                </option>
              ))}
            </select>
            <button type="button" className="toolbar-btn" onClick={lookAtSelectedTarget} disabled={!lookAtTargetId || availableLookAtTargets.length === 0}>
              Look At
            </button>
          </div>
        </div>
      )}

      <div className="inspect-card">
        <div className="inspect-label">Transform Mode</div>
        <div className="pill-row">
          <button className={`pill-btn${transformMode === 'translate' ? ' active' : ''}`} onClick={() => onTransformModeChange('translate')}>
            Move
          </button>
          <button className={`pill-btn${transformMode === 'rotate' ? ' active' : ''}`} onClick={() => onTransformModeChange('rotate')}>
            Rotate
          </button>
          {isCamera && (
            <button className={`pill-btn${activeCameraId === selected.id ? ' active' : ''}`} onClick={() => onSetActiveCamera(activeCameraId === selected.id ? null : selected.id)}>
              {activeCameraId === selected.id ? 'Stop View Sync' : 'Sync 3D View'}
            </button>
          )}
        </div>

        <div style={{ marginTop: 10 }}>
          <div className="inspect-label">Position</div>
          <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
            {[0, 1, 2].map((i) => (
              <input
                key={`pos-${i}`}
                type="number"
                step={0.001}
                value={position ? round3(position[i]) : 0}
                onMouseDown={startDragNumber(
                  i,
                  position ? round3(position[i]) : 0,
                  (idx, val) =>
                    setPosition((prev) => {
                      const next: [number, number, number] = prev ? ([...prev] as any) : [0, 0, 0]
                      next[idx] = round3(val)
                      return next
                    }),
                  0.01,
                )}
                onChange={(e) => {
                  const v = round3(parseFloat(e.target.value))
                  manualPoseEditRef.current = true
                  setPosition((prev) => {
                    const next: [number, number, number] = prev ? ([...prev] as any) : [0, 0, 0]
                    next[i] = Number.isFinite(v) ? round3(v) : 0
                    return next
                  })
                }}
                style={{ width: 86 }}
              />
            ))}
          </div>
        </div>

        <div style={{ marginTop: 8 }}>
          <div className="inspect-label">Rotation (deg)</div>
          <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
            {[0, 1, 2].map((i) => (
              <input
                key={`rot-${i}`}
                type="number"
                step={0.001}
                value={rotationEuler ? round3(rotationEuler[i]) : 0}
                onMouseDown={startDragNumber(
                  i,
                  rotationEuler ? round3(rotationEuler[i]) : 0,
                  (idx, val) => {
                    manualPoseEditRef.current = true
                    setRotationEuler((prev) => {
                      const next: [number, number, number] = prev ? ([...prev] as any) : [0, 0, 0]
                      next[idx] = round3(val)
                      return next
                    })
                    const nextEuler: [number, number, number] = rotationEuler ? ([...rotationEuler] as any) : [0, 0, 0]
                    nextEuler[idx] = round3(val)
                    const euler = new THREE.Euler(
                      THREE.MathUtils.degToRad(nextEuler[0]),
                      THREE.MathUtils.degToRad(nextEuler[1]),
                      THREE.MathUtils.degToRad(nextEuler[2]),
                      'XYZ',
                    )
                    const q = new THREE.Quaternion().setFromEuler(euler).normalize()
                    setRotation([q.x, q.y, q.z, q.w])
                  },
                  0.1,
                )}
                onChange={(e) => {
                  const v = round3(parseFloat(e.target.value))
                  manualPoseEditRef.current = true
                  setRotationEuler((prev) => {
                    const next: [number, number, number] = prev ? ([...prev] as any) : [0, 0, 0]
                    next[i] = Number.isFinite(v) ? round3(v) : 0
                    return next
                  })
                  const nextEuler: [number, number, number] = rotationEuler ? ([...rotationEuler] as any) : [0, 0, 0]
                  nextEuler[i] = Number.isFinite(v) ? round3(v) : 0
                  const euler = new THREE.Euler(
                    THREE.MathUtils.degToRad(nextEuler[0]),
                    THREE.MathUtils.degToRad(nextEuler[1]),
                    THREE.MathUtils.degToRad(nextEuler[2]),
                    'XYZ',
                  )
                  const q = new THREE.Quaternion().setFromEuler(euler).normalize()
                  setRotation([q.x, q.y, q.z, q.w])
                }}
                style={{ width: 86 }}
              />
            ))}
          </div>
        </div>
      </div>

      {isCamera && (
        <div className="inspect-card">
          <div className="inspect-label">Camera</div>
          <div style={{ marginTop: 6 }}>
            <button type="button" className="toolbar-btn" onClick={() => onOpenCameraView(selected.id)}>
              Open Camera View
            </button>
          </div>
          <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
            <label style={{ fontSize: 12 }}>
              FOV
              <input type="number" step={1} value={fov ?? 60} onChange={(e) => setFov(parseFloat(e.target.value))} style={{ width: 86, marginLeft: 6 }} />
            </label>
            <label style={{ fontSize: 12 }}>
              Near
              <input type="number" step={0.01} value={near ?? 0.01} onChange={(e) => setNear(parseFloat(e.target.value))} style={{ width: 86, marginLeft: 6 }} />
            </label>
            <label style={{ fontSize: 12 }}>
              Far
              <input type="number" step={1} value={far ?? 1000} onChange={(e) => setFar(parseFloat(e.target.value))} style={{ width: 86, marginLeft: 6 }} />
            </label>
          </div>
        </div>
      )}

      {(isPointCloud || isGaussians || isImage) && (
        <div className="inspect-card">
          {isPointCloud && selected.summary?.pointCount !== undefined && (
            <div style={{ marginBottom: 6 }}>
              <div className="inspect-label">Points</div>
              <div className="inspect-value">{selected.summary.pointCount as number}</div>
            </div>
          )}

          {isGaussians && selected.summary?.count !== undefined && (
            <div style={{ marginBottom: 6 }}>
              <div className="inspect-label">Gaussians</div>
              <div className="inspect-value">{selected.summary.count as number}</div>
            </div>
          )}

          {isImage && (
            <>
              <div className="inspect-label">Image</div>
              <div className="inspect-value">
                {imgWidth ?? '?'} x {imgHeight ?? '?'}
              </div>
              <div style={{ marginTop: 6 }} className="inspect-label">
                Channels
              </div>
              <div className="inspect-value">{imgChannels ?? '?'}</div>
              <div style={{ marginTop: 6 }} className="inspect-label">
                Mime
              </div>
              <div className="inspect-value inspect-mono">{imgMime ?? '?'}</div>
            </>
          )}
        </div>
      )}

      {isPointCloud && (
        <div className="inspect-card">
          <div className="inspect-label">{isPointCloud ? 'Point Size' : 'Scaling'}</div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 6 }}>
            <input
              type="range"
              min={isPointCloud ? 0.001 : 0.01}
              max={isPointCloud ? 3 : 10}
              step={0.001}
              value={pointSize ?? (isPointCloud ? 0.02 : 1.0)}
              onChange={(e) => setPointSize(parseFloat(e.target.value))}
              style={{ flex: 1 }}
            />
            <input
              type="number"
              min={0.0001}
              max={100}
              step={0.01}
              value={pointSize ?? (isPointCloud ? 0.02 : 1.0)}
              onChange={(e) => setPointSize(parseFloat(e.target.value))}
              style={{ width: 90 }}
            />
          </div>
          <div className="panel-subtitle" style={{ marginTop: 6 }}>
            {busy ? 'Updating...' : ' '}
          </div>
        </div>
      )}

      {(isPointCloud || isGaussians) && (
        <div className="inspect-card">
          <div className="inspect-label">Color</div>
          <div className="inspect-label" style={{ marginTop: 8 }}>
            Mode
          </div>
          <div className="pill-row">
            {(['logged', 'solid', 'height', 'depth'] as ColorMode[]).map((v) => {
              const active = colorMode === v
              return (
                <button
                  key={v}
                  className={`pill-btn${active ? ' active' : ''}`}
                  onClick={() => {
                    setColorMode(v)
                    if (v === 'height') {
                      setColorMap(DEFAULT_HEIGHT_COLORMAP)
                      setVisualOverrideForElement(selected.id, v, solidColor, DEFAULT_HEIGHT_COLORMAP)
                      return
                    }
                    if (v === 'depth') {
                      setColorMap(DEFAULT_DEPTH_COLORMAP)
                      setVisualOverrideForElement(selected.id, v, solidColor, DEFAULT_DEPTH_COLORMAP)
                      return
                    }
                    setVisualOverrideForElement(selected.id, v, solidColor, colorMap)
                  }}
                >
                  {v}
                </button>
              )
            })}
          </div>

          {colorMode === 'solid' && (
            <div style={{ marginTop: 10, display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                aria-label="Solid color"
                type="color"
                value={solidColor}
                onChange={(e) => {
                  setSolidColor(e.target.value)
                  setVisualOverrideForElement(selected.id, 'solid', e.target.value, colorMap)
                }}
              />
              <div className="inspect-value">{solidColor}</div>
            </div>
          )}

          {(colorMode === 'height' || colorMode === 'depth') && (
            <div style={{ marginTop: 12 }}>
              <label className="inspect-label">Colormap</label>
              <select
                value={colorMap}
                onChange={(e) => {
                  const next = e.target.value as ColormapId
                  setColorMap(next)
                  setVisualOverrideForElement(selected.id, colorMode, solidColor, next)
                }}
                style={{ marginTop: 6, width: '100%' }}
              >
                {COLORMAPS.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.label}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      )}

      {isGaussians && (
        <div className="inspect-card">
          <div className="inspect-label">Client Controls</div>
          <div className="inspect-label" style={{ marginTop: 10 }}>
            LOD Override
          </div>
          <div className="pill-row">
            {(['auto', 'high', 'medium', 'low'] as const).map((v) => {
              const active = (v === 'auto' && !lodOverride) || lodOverride === v
              return (
                <button
                  key={v}
                  className={`pill-btn${active ? ' active' : ''}`}
                  onClick={() => {
                    const val = v === 'auto' ? undefined : v
                    setLodOverride(val)
                    setLodOverrideForElement(selected.id, val as any)
                  }}
                >
                  {v}
                </button>
              )
            })}
          </div>
        </div>
      )}

      {err && <div className="error-text">{err}</div>}
    </div>
  )
}
