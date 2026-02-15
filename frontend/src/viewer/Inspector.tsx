import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import type { ElementInfo, SampleQuery } from './api'
import {
  deleteCameraAnimation,
  fetchPointCloudElementMeta,
  fetchGaussianElementMeta,
  fetchCameraAnimation,
  putCameraAnimation,
  fetchCameraElementMeta,
  fetchImageElementMeta,
  fetchElementMeta,
  updatePointCloudSettings,
  updateElementMeta,
  deleteElement,
  type CameraAnimationTrack,
} from './api'
import { COLORMAPS, DEFAULT_DEPTH_COLORMAP, DEFAULT_HEIGHT_COLORMAP, type ColormapId } from './colormaps'

type Props = {
  selected: ElementInfo | null
  activeCameraId: string | null
  onDelete: (id: string) => void
  transformMode: 'translate' | 'rotate' | 'animate'
  onTransformModeChange: (mode: 'translate' | 'rotate' | 'animate') => void
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
  const isBox3D = selected?.type === 'box3d'
  const isEllipsoid3D = selected?.type === 'ellipsoid3d'

  // LOD override state
  const [lodOverride, setLodOverride] = useState<string | undefined>(undefined)

  // Visual override state (client-only preview)
  type ColorMode = 'logged' | 'solid' | 'height' | 'depth'
  const [colorMode, setColorMode] = useState<ColorMode>('logged')
  const [solidColor, setSolidColor] = useState<string>('#ff8a33') // hex string for input[type=color]
  const [colorMap, setColorMap] = useState<ColormapId>(DEFAULT_HEIGHT_COLORMAP)
  const [showBounds, setShowBounds] = useState(false)
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
  const [animationTrack, setAnimationTrack] = useState<CameraAnimationTrack | null>(null)
  const [animationBusy, setAnimationBusy] = useState(false)
  const [animationTargetId, setAnimationTargetId] = useState<string>('')
  const [animationStartFrame, setAnimationStartFrame] = useState<number>(0)
  const [animationEndFrame, setAnimationEndFrame] = useState<number>(120)
  const [animationStep, setAnimationStep] = useState<number>(1)
  const [animationTurns, setAnimationTurns] = useState<number>(1.0)
  const [animationRadius, setAnimationRadius] = useState<string>('5')
  const [animationPhaseDeg, setAnimationPhaseDeg] = useState<number>(0.0)

  const availableLookAtTargets = lookAtTargets.filter((t) => t.id !== selected?.id && t.type !== 'image')
  const availableLookAtTargetIds = availableLookAtTargets.map((t) => t.id).join('|')
  const followSyncActive = !!(isCamera && selected && activeCameraId === selected.id)

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
            : fetchElementMeta

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
        setShowBounds(Boolean(vis.showBounds))
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
        setShowBounds(false)
      }
    } catch {}
  }, [enabled, selected?.id, isPointCloud, isGaussians, isCamera, isImage, sample?.frame, sample?.timestamp, sample?.timeline, sample?.time])

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
  }, [enabled, selected?.id, sample?.frame, sample?.timestamp, sample?.timeline, sample?.time])

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
    showBoundsOverride?: boolean,
  ) => {
    try {
      const anyWin = window as any
      anyWin.__begira_visual_override = anyWin.__begira_visual_override || {}
      const prev = (anyWin.__begira_visual_override[id] ?? {}) as any
      const showBounds = showBoundsOverride !== undefined ? Boolean(showBoundsOverride) : Boolean(prev.showBounds)
      const obj: any = { ...prev, colorMode: mode, showBounds }
      if (mode === 'solid' && hexColor) {
        // convert #rrggbb to [r,g,b] floats
        const c = hexColor.replace('#', '')
        const r = parseInt(c.substring(0, 2), 16) / 255
        const g = parseInt(c.substring(2, 4), 16) / 255
        const b = parseInt(c.substring(4, 6), 16) / 255
        obj.solidColor = [r, g, b]
      } else {
        delete obj.solidColor
      }
      if (mode === 'height' || mode === 'depth') {
        obj.colorMap = mapId
      } else {
        delete obj.colorMap
      }

      if (mode === 'logged' && !showBounds) {
        delete anyWin.__begira_visual_override[id]
      } else {
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
  }, [selected?.id, sample?.frame, sample?.timestamp, sample?.timeline, sample?.time])

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
    setLookAtTargetId((prev) => {
      if (prev && availableLookAtTargets.some((t) => t.id === prev)) return prev
      return ''
    })
  }, [isCamera, selected?.id, availableLookAtTargets])

  useEffect(() => {
    if (!isCamera || !selected || !enabled) {
      setAnimationTrack(null)
      return
    }

    let cancelled = false

    const sync = async () => {
      try {
        const track = await fetchCameraAnimation(selected.id)
        if (cancelled) return
        setAnimationTrack(track)
        if (!track) {
          setAnimationTargetId((prev) => {
            if (prev && availableLookAtTargets.some((t) => t.id === prev)) return prev
            return availableLookAtTargets[0]?.id ?? ''
          })
          setAnimationRadius('5')
          return
        }
        setAnimationTargetId(track.targetId)
        setAnimationStartFrame(Number(track.startFrame))
        setAnimationEndFrame(Number(track.endFrame))
        setAnimationStep(Math.max(1, Number(track.step)))
        setAnimationTurns(Number(track.params.turns ?? 1.0))
        setAnimationRadius(track.params.radius !== undefined ? String(track.params.radius) : '')
        setAnimationPhaseDeg(Number(track.params.phaseDeg ?? 0.0))
      } catch {}
    }

    void sync()
    const onChanged = (ev: Event) => {
      const detail = (ev as CustomEvent<{ cameraId?: string }>).detail
      if (detail?.cameraId && detail.cameraId !== selected.id) return
      void sync()
    }
    window.addEventListener('begira_camera_animation_changed', onChanged as EventListener)

    return () => {
      cancelled = true
      window.removeEventListener('begira_camera_animation_changed', onChanged as EventListener)
    }
  }, [enabled, isCamera, selected?.id, availableLookAtTargetIds])

  useEffect(() => {
    if (!isCamera) return
    setAnimationTargetId((prev) => {
      if (prev && availableLookAtTargets.some((t) => t.id === prev)) return prev
      return availableLookAtTargets[0]?.id ?? ''
    })
  }, [availableLookAtTargetIds, isCamera, selected?.id])

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

  const generateCameraAnimation = async (mode: 'follow' | 'orbit') => {
    if (!selected || !isCamera || !animationTargetId) return
    setAnimationBusy(true)
    setErr(null)
    try {
      const req: {
        mode: 'follow' | 'orbit'
        targetId: string
        startFrame: number
        endFrame: number
        step?: number
        turns?: number
        radius?: number
        phaseDeg?: number
      } = {
        mode,
        targetId: animationTargetId,
        startFrame: Math.max(0, Math.round(animationStartFrame)),
        endFrame: Math.max(Math.round(animationStartFrame), Math.round(animationEndFrame)),
        step: Math.max(1, Math.round(animationStep)),
      }
      if (mode === 'orbit') {
        req.turns = Number.isFinite(animationTurns) ? animationTurns : 1.0
        req.phaseDeg = Number.isFinite(animationPhaseDeg) ? animationPhaseDeg : 0.0
        const radiusVal = Number(animationRadius)
        if (animationRadius.trim() !== '' && Number.isFinite(radiusVal) && radiusVal > 0) {
          req.radius = radiusVal
        }
      }
      const track = await putCameraAnimation(selected.id, req)
      setAnimationTrack(track)
      window.dispatchEvent(
        new CustomEvent('begira_camera_animation_changed', {
          detail: { cameraId: selected.id },
        }),
      )
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setAnimationBusy(false)
    }
  }

  const clearCameraAnimationTrack = async () => {
    if (!selected || !isCamera) return
    setAnimationBusy(true)
    setErr(null)
    try {
      await deleteCameraAnimation(selected.id)
      setAnimationTrack(null)
      window.dispatchEvent(
        new CustomEvent('begira_camera_animation_changed', {
          detail: { cameraId: selected.id },
        }),
      )
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setAnimationBusy(false)
    }
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
              <option value="">(select 3D object)</option>
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
        <div className="inspect-label">Transform</div>
        <div className="panel-subtitle">Use the 3D viewport toolbar to switch Move, Rotate, and Animate modes.</div>

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

      {isCamera && (
        <div className="inspect-card">
          <div className="inspect-label">Animation</div>
          {followSyncActive && (
            <div className="panel-subtitle" style={{ marginTop: 6 }}>
              Animation editing is disabled while camera follow/view sync is active. Stop view sync to author orbit keys.
            </div>
          )}
          <div style={{ marginTop: 8, display: 'grid', gap: 8 }}>
            <label style={{ fontSize: 12 }}>
              Target
              <select
                value={animationTargetId}
                onChange={(e) => setAnimationTargetId(e.target.value)}
                style={{ width: '100%', marginTop: 4 }}
                disabled={animationBusy || availableLookAtTargets.length === 0 || followSyncActive}
              >
                {availableLookAtTargets.length === 0 && <option value="">(no available 3D objects)</option>}
                {availableLookAtTargets.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.name} ({t.type})
                  </option>
                ))}
              </select>
            </label>

            <div style={{ display: 'flex', gap: 6 }}>
              <label style={{ fontSize: 12, flex: 1 }}>
                Start frame
                <input
                  type="number"
                  step={1}
                  value={animationStartFrame}
                  onChange={(e) => setAnimationStartFrame(Math.round(Number(e.target.value) || 0))}
                  style={{ width: '100%', marginTop: 4 }}
                  disabled={animationBusy || followSyncActive}
                />
              </label>
              <label style={{ fontSize: 12, flex: 1 }}>
                End frame
                <input
                  type="number"
                  step={1}
                  value={animationEndFrame}
                  onChange={(e) => setAnimationEndFrame(Math.round(Number(e.target.value) || 0))}
                  style={{ width: '100%', marginTop: 4 }}
                  disabled={animationBusy || followSyncActive}
                />
              </label>
              <label style={{ fontSize: 12, width: 86 }}>
                Step
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={animationStep}
                  onChange={(e) => setAnimationStep(Math.max(1, Math.round(Number(e.target.value) || 1)))}
                  style={{ width: '100%', marginTop: 4 }}
                  disabled={animationBusy || followSyncActive}
                />
              </label>
            </div>

            <div style={{ display: 'grid', gap: 6 }}>
              <div style={{ display: 'flex', gap: 6 }}>
                <label style={{ fontSize: 12, flex: 1 }}>
                  Turns
                  <input
                    type="number"
                    step={0.1}
                    value={animationTurns}
                    onChange={(e) => setAnimationTurns(Number(e.target.value))}
                    style={{ width: '100%', marginTop: 4 }}
                    disabled={animationBusy || followSyncActive}
                  />
                </label>
                <label style={{ fontSize: 12, flex: 1 }}>
                  Radius
                  <input
                    type="number"
                    step={0.01}
                    value={animationRadius}
                    onChange={(e) => setAnimationRadius(e.target.value)}
                    placeholder="auto"
                    style={{ width: '100%', marginTop: 4 }}
                    disabled={animationBusy || followSyncActive}
                  />
                </label>
              </div>
              <label style={{ fontSize: 12 }}>
                Phase (deg)
                <input
                  type="number"
                  step={1}
                  value={animationPhaseDeg}
                  onChange={(e) => setAnimationPhaseDeg(Number(e.target.value))}
                  style={{ width: '100%', marginTop: 4 }}
                  disabled={animationBusy || followSyncActive}
                />
              </label>
            </div>

            <div className="pill-row" style={{ justifyContent: 'space-between' }}>
              <button
                type="button"
                className="toolbar-btn"
                onClick={() => void generateCameraAnimation('orbit')}
                disabled={animationBusy || !animationTargetId || followSyncActive}
              >
                {animationBusy ? 'Working...' : 'Generate Orbit'}
              </button>
              <button
                type="button"
                className="toolbar-btn"
                onClick={() => void clearCameraAnimationTrack()}
                disabled={animationBusy || !animationTrack || followSyncActive}
              >
                Clear
              </button>
            </div>

            {animationTrack && (
              <div className="panel-subtitle">
                Active: {animationTrack.mode} [{animationTrack.startFrame}, {animationTrack.endFrame}]
                {animationTrack.mode === 'orbit' ? `, keys ${animationTrack.controlKeys.length}` : ''}
              </div>
            )}
            {animationTrack?.mode === 'follow' && (
              <div className="panel-subtitle">Follow is treated as view-sync behavior. Use Orbit animation for trajectory authoring.</div>
            )}
            <div className="panel-subtitle">Use the top 3D toolbar mode `Animate` to edit orbit keys in the viewport.</div>
          </div>
        </div>
      )}

      {(isPointCloud || isGaussians || isImage || isBox3D || isEllipsoid3D) && (
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

          {isBox3D && Array.isArray((selected as any).size) && (
            <>
              <div className="inspect-label">Box Size</div>
              <div className="inspect-value">{(selected as any).size.map((v: number) => Number(v).toFixed(3)).join(' x ')}</div>
            </>
          )}

          {isEllipsoid3D && Array.isArray((selected as any).radii) && (
            <>
              <div className="inspect-label">Ellipsoid Radii</div>
              <div className="inspect-value">{(selected as any).radii.map((v: number) => Number(v).toFixed(3)).join(' x ')}</div>
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

          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, marginTop: 10 }}>
            <input
              type="checkbox"
              checked={showBounds}
              onChange={(e) => {
                const checked = e.target.checked
                setShowBounds(checked)
                setVisualOverrideForElement(selected.id, colorMode, solidColor, colorMap, checked)
              }}
            />
            Show boundaries
          </label>

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
