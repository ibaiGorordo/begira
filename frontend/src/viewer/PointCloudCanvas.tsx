import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, TransformControls } from '@react-three/drei'
import { useEffect, useLayoutEffect, useMemo, useRef, useState, useCallback } from 'react'
import * as THREE from 'three'
import WASDControls from './WASDControls'
import MultiPointCloudScene, { PointCloudRenderMode } from './MultiPointCloudScene'
import GaussianSplatScene from './GaussianSplatScene'
import CameraScene, { CAMERA_GIZMO_OVERLAY_LAYER, type CameraVisual } from './CameraScene'
import {
  createClickGesture,
  isClickGesture,
  onPointerDownGesture,
  onPointerMoveGesture,
  onPointerUpGesture,
} from './interaction'
import { usePointCloud } from './usePointCloud'
import DebugOverlay, { isDebugOverlayEnabledFromUrl } from './DebugOverlay'
import { getCoordinateConvention, parseCoordinateConventionFromUrl, type CoordinateConventionId } from './coordinateConventions'
import { fetchViewerSettings, type SampleQuery } from './api'
import { usePageActivity } from './usePageActivity'

type Bounds3 = { min: THREE.Vector3; max: THREE.Vector3 }

// Viewer-space is always Three.js native Y-up.
const VIEW_UP = new THREE.Vector3(0, 1, 0)

function transformBounds(bounds: Bounds3, q: THREE.Quaternion): Bounds3 {
  const corners: THREE.Vector3[] = []
  const { min, max } = bounds
  for (const x of [min.x, max.x]) {
    for (const y of [min.y, max.y]) {
      for (const z of [min.z, max.z]) {
        corners.push(new THREE.Vector3(x, y, z).applyQuaternion(q))
      }
    }
  }

  const outMin = new THREE.Vector3(Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY)
  const outMax = new THREE.Vector3(Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY)
  for (const c of corners) {
    outMin.min(c)
    outMax.max(c)
  }
  return { min: outMin, max: outMax }
}

function fitCameraToBounds(
  camera: THREE.PerspectiveCamera,
  size: { width: number; height: number },
  bounds: { min: THREE.Vector3; max: THREE.Vector3 },
  opts?: { up?: THREE.Vector3; forward?: THREE.Vector3; topDown?: boolean }
) {
  const center = bounds.min.clone().add(bounds.max).multiplyScalar(0.5)
  const bboxSize = bounds.max.clone().sub(bounds.min)

  const aspect = size.width / Math.max(1, size.height)
  const vFov = THREE.MathUtils.degToRad(camera.fov)
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * aspect)

  const halfX = bboxSize.x * 0.5
  const halfY = bboxSize.y * 0.5
  const halfZ = bboxSize.z * 0.5

  const distX = halfX / Math.tan(hFov / 2)
  const distY = halfY / Math.tan(vFov / 2)
  const baseDist = Math.max(distX, distY, halfZ)

  const distance = (baseDist || 1) * 1.4

  const up = (opts?.up?.clone() ?? new THREE.Vector3(0, 1, 0)).normalize()
  if (up.lengthSq() === 0) up.set(0, 1, 0).normalize()

  const forwardSeed = opts?.forward?.clone() ?? new THREE.Vector3(0, 0, 1)
  const fwd = forwardSeed.lengthSq() === 0 ? new THREE.Vector3(0, 0, 1) : forwardSeed.normalize()
  if (Math.abs(fwd.dot(up)) > 0.95) fwd.set(1, 0, 0)

  const side = new THREE.Vector3().crossVectors(up, fwd).normalize()
  const forward = new THREE.Vector3().crossVectors(side, up).normalize()

  const topDown = opts?.topDown === true
  const dir = topDown
    ? up.clone()
    : forward.clone().add(up.clone().multiplyScalar(0.45)).add(side.clone().multiplyScalar(0.85)).normalize()

  camera.position.copy(center.clone().add(dir.multiplyScalar(distance)))
  camera.near = Math.max(0.001, distance / 1000)
  camera.far = distance * 2000
  camera.lookAt(center)
  camera.updateProjectionMatrix()

  return { center }
}

function FrameCamera({
  bounds,
  focusToken,
  forward,
  topDown,
}: {
  bounds: { min: THREE.Vector3; max: THREE.Vector3 } | null
  focusToken: number
  forward: THREE.Vector3
  topDown: boolean
}) {
  const { camera, size } = useThree()

  const lastBoundsRef = useRef<{ min: THREE.Vector3; max: THREE.Vector3 } | null>(null)
  useEffect(() => {
    lastBoundsRef.current = bounds
  }, [bounds])

  useLayoutEffect(() => {
    const b = lastBoundsRef.current
    if (!b) return
    if (!Number.isFinite(size.width) || !Number.isFinite(size.height) || size.width <= 1 || size.height <= 1) return

    const handle = requestAnimationFrame(() => {
      const perspective = camera as THREE.PerspectiveCamera
      const { center } = fitCameraToBounds(perspective, size, b, {
        up: (camera as THREE.Camera).up,
        forward,
        topDown,
      })

      const anyCam = camera as unknown as { controls?: { target: THREE.Vector3; update: () => void } }
      if (anyCam.controls) {
        anyCam.controls.target.copy(center)
        anyCam.controls.update()
      } else {
        ;(camera as any).userData = (camera as any).userData ?? {}
        ;(camera as any).userData.pendingOrbitTarget = center.clone()
      }
    })

    return () => cancelAnimationFrame(handle)
    // Intentionally NOT depending on `bounds` identity: selection/meta refreshes can recreate bounds.
  }, [camera, focusToken, size.height, size.width, forward, topDown])

  return null
}

function ActiveCameraDriver({
  meta,
  enabled,
  worldToView,
}: {
  meta: { position?: [number, number, number]; rotation?: [number, number, number, number]; fov: number; near: number; far: number } | null
  enabled: boolean
  worldToView: THREE.Quaternion
}) {
  const { camera } = useThree()
  const savedState = useRef<{
    position: THREE.Vector3
    quaternion: THREE.Quaternion
    fov: number
    near: number
    far: number
    target: THREE.Vector3 | null
  } | null>(null)
  const wasEnabled = useRef(false)

  const restoreIfNeeded = useCallback(() => {
    const prev = savedState.current
    if (!prev) return
    const cam = camera as THREE.PerspectiveCamera
    cam.position.copy(prev.position)
    cam.quaternion.copy(prev.quaternion)
    cam.fov = prev.fov
    cam.near = prev.near
    cam.far = prev.far
    cam.updateProjectionMatrix()

    const anyCam = camera as unknown as { controls?: { target: THREE.Vector3; update: () => void } }
    if (anyCam.controls && prev.target) {
      anyCam.controls.target.copy(prev.target)
      anyCam.controls.update()
    }

    savedState.current = null
  }, [camera])

  useLayoutEffect(() => {
    const shouldDrive = enabled && !!meta
    const cam = camera as THREE.PerspectiveCamera
    const anyCam = camera as unknown as { controls?: { target: THREE.Vector3; update: () => void } }

    if (shouldDrive && !savedState.current) {
      const target = anyCam.controls ? anyCam.controls.target.clone() : null
      savedState.current = {
        position: cam.position.clone(),
        quaternion: cam.quaternion.clone(),
        fov: cam.fov,
        near: cam.near,
        far: cam.far,
        target,
      }
    } else if (!shouldDrive && wasEnabled.current) {
      restoreIfNeeded()
    }
    wasEnabled.current = shouldDrive

    if (!shouldDrive || !meta) return
    const posWorld = meta.position ?? [0, 0, 0]
    const rotWorld = meta.rotation ?? [0, 0, 0, 1]

    // Camera metadata is authored in world coordinates; scene is rendered in view coordinates.
    const posView = new THREE.Vector3(posWorld[0], posWorld[1], posWorld[2]).applyQuaternion(worldToView)
    const qWorld = new THREE.Quaternion(rotWorld[0], rotWorld[1], rotWorld[2], rotWorld[3]).normalize()
    const qView = worldToView.clone().multiply(qWorld).normalize()

    cam.position.copy(posView)
    cam.quaternion.copy(qView)
    cam.fov = THREE.MathUtils.clamp(Number.isFinite(meta.fov) ? meta.fov : 60, 5, 175)
    cam.near = Math.max(0.0001, Number.isFinite(meta.near) ? meta.near : 0.01)
    cam.far = Math.max(cam.near + 0.01, Number.isFinite(meta.far) ? meta.far : 1000)
    cam.updateProjectionMatrix()
  }, [camera, enabled, meta, worldToView])

  // Re-apply every frame while active so no other control path can override
  // camera pose/projection (e.g. damping controls).
  useFrame(() => {
    if (!enabled || !meta) return
    const posWorld = meta.position ?? [0, 0, 0]
    const rotWorld = meta.rotation ?? [0, 0, 0, 1]

    const posView = new THREE.Vector3(posWorld[0], posWorld[1], posWorld[2]).applyQuaternion(worldToView)
    const qWorld = new THREE.Quaternion(rotWorld[0], rotWorld[1], rotWorld[2], rotWorld[3]).normalize()
    const qView = worldToView.clone().multiply(qWorld).normalize()

    const cam = camera as THREE.PerspectiveCamera
    cam.position.copy(posView)
    cam.quaternion.copy(qView)
    cam.fov = THREE.MathUtils.clamp(Number.isFinite(meta.fov) ? meta.fov : 60, 5, 175)
    cam.near = Math.max(0.0001, Number.isFinite(meta.near) ? meta.near : 0.01)
    cam.far = Math.max(cam.near + 0.01, Number.isFinite(meta.far) ? meta.far : 1000)
    cam.updateProjectionMatrix()
  })

  useEffect(() => {
    return () => restoreIfNeeded()
  }, [restoreIfNeeded])

  return null
}

function ViewCameraReporter() {
  const { camera } = useThree()
  const lastSent = useRef(0)
  useFrame(() => {
    const now = performance.now()
    if (now - lastSent.current < 200) return
    lastSent.current = now
    const cam = camera as THREE.PerspectiveCamera
    ;(window as any).__begira_view_camera = {
      position: [cam.position.x, cam.position.y, cam.position.z],
      rotation: [cam.quaternion.x, cam.quaternion.y, cam.quaternion.z, cam.quaternion.w],
      fov: cam.fov,
      near: cam.near,
      far: cam.far,
    }
  })
  return null
}

function OverlayCameraRenderPass() {
  const { gl, scene, camera, raycaster } = useThree()
  useEffect(() => {
    camera.layers.enable(CAMERA_GIZMO_OVERLAY_LAYER)
    raycaster.layers.enable(CAMERA_GIZMO_OVERLAY_LAYER)
  }, [camera, raycaster])

  useFrame(() => {
    const prevAutoClear = gl.autoClear
    const prevMask = camera.layers.mask

    gl.autoClear = true
    camera.layers.set(0)
    gl.render(scene, camera)

    gl.autoClear = false
    gl.clearDepth()
    camera.layers.set(CAMERA_GIZMO_OVERLAY_LAYER)
    gl.render(scene, camera)

    camera.layers.mask = prevMask
    gl.autoClear = prevAutoClear
  }, 1)
  return null
}

function FollowCameraDriver({
  cameraMeta,
  enabled,
  worldToView,
}: {
  cameraMeta: CameraVisual | null
  enabled: boolean
  worldToView: THREE.Quaternion
}) {
  const { camera } = useThree()

  useLayoutEffect(() => {
    if (!enabled || !cameraMeta) return

    const cam = camera as THREE.PerspectiveCamera
    const posWorld = cameraMeta.position ?? [0, 0, 0]
    const rotWorld = cameraMeta.rotation ?? [0, 0, 0, 1]
    const posView = new THREE.Vector3(posWorld[0], posWorld[1], posWorld[2]).applyQuaternion(worldToView)
    const qWorld = new THREE.Quaternion(rotWorld[0], rotWorld[1], rotWorld[2], rotWorld[3]).normalize()
    const qView = worldToView.clone().multiply(qWorld).normalize()

    cam.position.copy(posView)
    cam.quaternion.copy(qView)
    if (Number.isFinite(cameraMeta.fov)) {
      cam.fov = THREE.MathUtils.clamp(cameraMeta.fov as number, 5, 175)
    }
    const near = Number.isFinite(cameraMeta.near) ? Math.max(0.0001, cameraMeta.near as number) : cam.near
    const far = Number.isFinite(cameraMeta.far) ? Math.max(near + 0.01, cameraMeta.far as number) : cam.far
    cam.near = near
    cam.far = far
    cam.updateProjectionMatrix()

    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(qView).normalize()
    const targetDistance = THREE.MathUtils.clamp(Math.max(near * 8, 1.0), 0.25, 10)
    const target = posView.clone().add(forward.multiplyScalar(targetDistance))

    const anyCam = camera as unknown as { controls?: { target: THREE.Vector3; update: () => void } }
    if (anyCam.controls) {
      anyCam.controls.target.copy(target)
      anyCam.controls.update()
    } else {
      ;(camera as any).userData = (camera as any).userData ?? {}
      ;(camera as any).userData.pendingOrbitTarget = target.clone()
    }
  }, [camera, cameraMeta, enabled, worldToView])

  useFrame(() => {
    if (!enabled || !cameraMeta) return

    const cam = camera as THREE.PerspectiveCamera
    const posWorld = cameraMeta.position ?? [0, 0, 0]
    const rotWorld = cameraMeta.rotation ?? [0, 0, 0, 1]
    const posView = new THREE.Vector3(posWorld[0], posWorld[1], posWorld[2]).applyQuaternion(worldToView)
    const qWorld = new THREE.Quaternion(rotWorld[0], rotWorld[1], rotWorld[2], rotWorld[3]).normalize()
    const qView = worldToView.clone().multiply(qWorld).normalize()

    cam.position.copy(posView)
    cam.quaternion.copy(qView)
    if (Number.isFinite(cameraMeta.fov)) {
      cam.fov = THREE.MathUtils.clamp(cameraMeta.fov as number, 5, 175)
    }
    const near = Number.isFinite(cameraMeta.near) ? Math.max(0.0001, cameraMeta.near as number) : cam.near
    const far = Number.isFinite(cameraMeta.far) ? Math.max(near + 0.01, cameraMeta.far as number) : cam.far
    cam.near = near
    cam.far = far
    cam.updateProjectionMatrix()
  })

  return null
}

function useSceneBounds(): {
  bounds: { min: THREE.Vector3; max: THREE.Vector3 } | null
  setBoundsFromMeta: (metaBounds: { min: [number, number, number]; max: [number, number, number] }[]) => void
} {
  const api = useMemo(() => {
    const state: { bounds: { min: THREE.Vector3; max: THREE.Vector3 } | null } = { bounds: null }

    const setBoundsFromMeta = (all: { min: [number, number, number]; max: [number, number, number] }[]) => {
      if (all.length === 0) {
        state.bounds = null
        return
      }
      const min = new THREE.Vector3(Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY)
      const max = new THREE.Vector3(Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY)

      for (const b of all) {
        min.min(new THREE.Vector3(...b.min))
        max.max(new THREE.Vector3(...b.max))
      }
      state.bounds = { min, max }
    }

    return { state, setBoundsFromMeta }
  }, [])

  return { bounds: api.state.bounds, setBoundsFromMeta: api.setBoundsFromMeta }
}

export default function PointCloudCanvas({
  cloudIds,
  gaussianIds = [],
  cameraIds = [],
  cameraVisuals = [],
  selectedId,
  onSelect,
  focusTarget,
  onFocus,
  cloudMetaBounds,
  gaussianMetaBounds = [],
  cameraMetaBounds = [],
  activeCameraId,
  transformMode = 'translate',
  sample,
  secondaryView = false,
  onTransformCommit,
}: {
  cloudIds: string[]
  gaussianIds?: string[]
  cameraIds?: string[]
  cameraVisuals?: CameraVisual[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  focusTarget: string | null
  onFocus: (id: string | null) => void
  cloudMetaBounds: { min: [number, number, number]; max: [number, number, number] }[]
  gaussianMetaBounds?: { min: [number, number, number]; max: [number, number, number] }[]
  cameraMetaBounds?: { min: [number, number, number]; max: [number, number, number] }[]
  activeCameraId?: string | null
  transformMode?: 'translate' | 'rotate'
  sample?: SampleQuery
  secondaryView?: boolean
  onTransformCommit?: (id: string, position: [number, number, number], rotation: [number, number, number, number]) => void
}) {
  const background = useMemo(() => new THREE.Color('#0b1020'), [])
  const pageActive = usePageActivity()

  const [conventionId, setConventionId] = useState<CoordinateConventionId>(() => {
    return parseCoordinateConventionFromUrl(window.location.search) ?? 'rh-z-up'
  })

  // Read settings once on mount (URL wins). If you change server settings at runtime, you must reload.
  useEffect(() => {
    const urlOverride = parseCoordinateConventionFromUrl(window.location.search)
    if (urlOverride) {
      setConventionId(urlOverride)
      return
    }

    let cancelled = false
    fetchViewerSettings()
      .then((s) => {
        if (cancelled) return
        if (s.coordinateConvention === 'rh-y-up' || s.coordinateConvention === 'rh-z-up') {
          setConventionId(s.coordinateConvention)
        }
      })
      .catch(() => {
        /* ignore */
      })

    return () => {
      cancelled = true
    }
  }, [])

  const convention = useMemo(() => getCoordinateConvention(conventionId), [conventionId])

  // Debug overlay:
  const [debugOverlayEnabled, setDebugOverlayEnabled] = useState(() => isDebugOverlayEnabledFromUrl())
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase()
      if (tag === 'input' || tag === 'textarea' || (e.target as HTMLElement | null)?.isContentEditable) return
      if (e.key === '`') setDebugOverlayEnabled((v) => !v)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  // Rendering mode:
  const [renderMode] = useState<PointCloudRenderMode>(() => {
    const v = new URLSearchParams(window.location.search).get('mode')
    if (v === 'fast' || v === 'circles' || v === 'quality') return v
    return 'circles'
  })
  const effectiveRenderMode: PointCloudRenderMode = secondaryView ? 'fast' : renderMode

  const orbitRef = useRef<any>(null)
  const [controlsReady, setControlsReady] = useState(false)

  // Focus:
  const [focusId, setFocusId] = useState<string | null>(null)
  const focusIdRef = useRef<string | null>(null)
  useEffect(() => {
    focusIdRef.current = focusId
  }, [focusId])

  useEffect(() => {
    if (selectedId === null) {
      setFocusId(null)
    }
  }, [selectedId])

  const [focusToken, setFocusToken] = useState(0)
  const bumpFocus = useCallback((source?: string) => {
    setFocusToken((t) => {
      void source
      return t + 1
    })
  }, [])
  const [followCameraId, setFollowCameraId] = useState<string | null>(null)

  // IMPORTANT: refit whenever convention changes.
  useEffect(() => {
    if (cloudIds.length === 0 && gaussianIds.length === 0) return
    bumpFocus('convention-change')
  }, [conventionId, cloudIds.length, gaussianIds.length, bumpFocus])

  // Choose a forward direction that makes the change visible.
  const viewForward = useMemo(() => {
    if (conventionId === 'rh-y-up') return new THREE.Vector3(0, 0, -1)
    return new THREE.Vector3(0, 0, 1)
  }, [conventionId])

  const topDownOnFocus = false

  const cameraVisualById = useMemo(() => {
    const byId = new Map<string, CameraVisual>()
    for (const c of cameraVisuals) byId.set(c.id, c)
    return byId
  }, [cameraVisuals])
  const followCameraMeta = followCameraId ? cameraVisualById.get(followCameraId) ?? null : null
  const activeCameraMeta = useMemo(() => {
    if (!activeCameraId) return null
    const c = cameraVisualById.get(activeCameraId)
    if (!c) return null
    return {
      position: c.position,
      rotation: c.rotation,
      fov: Number.isFinite(c.fov as number) ? (c.fov as number) : 60,
      near: Number.isFinite(c.near as number) ? (c.near as number) : 0.01,
      far: Number.isFinite(c.far as number) ? (c.far as number) : 1000,
    }
  }, [activeCameraId, cameraVisualById])

  useEffect(() => {
    if (!followCameraId) return
    if (cameraVisualById.has(followCameraId)) return
    setFollowCameraId(null)
  }, [cameraVisualById, followCameraId])

  // Initialize focus when elements first appear.
  const prevElementCount = useRef(0)
  const allElementIds = useMemo(() => [...cloudIds, ...gaussianIds, ...cameraIds], [cloudIds, gaussianIds, cameraIds])
  useEffect(() => {
    const prev = prevElementCount.current
    prevElementCount.current = allElementIds.length

    if (allElementIds.length === 0) {
      setFocusId(null)
      return
    }

    // Only set initial focus once, when elements first appear.
    if (prev === 0 && allElementIds.length > 0) {
      setFocusId(allElementIds[0])
      bumpFocus('startup:first-element')
      return
    }

    // If the currently focused id disappears, clear focus to avoid camera jumps.
    setFocusId((cur) => {
      if (!cur) return null
      return allElementIds.includes(cur) ? cur : null
    })
  }, [allElementIds, bumpFocus])

  // External focus request (double-click from hierarchy or scene).
  // IMPORTANT: this should be the only time selection triggers a refit.
  useEffect(() => {
    if (!focusTarget) return
    if (!allElementIds.includes(focusTarget)) {
      // Clear invalid/old focus requests.
      onFocus(null)
      return
    }

    const focusedCamera = cameraVisualById.get(focusTarget)
    if (focusedCamera) {
      setFocusId(focusTarget)
      setFollowCameraId(focusTarget)
      onFocus(null)
      return
    }

    // Always refit, even if focusTarget equals current focus id.
    setFocusId(focusTarget)
    bumpFocus('effect:focusTarget')

    // Consume the request so it doesn't refire on every render/poll tick.
    onFocus(null)
  }, [allElementIds, focusTarget, bumpFocus, cameraVisualById, onFocus])

  // First cloud decoded bounds:
  const firstCloudId = cloudIds[0] ?? ''
  const firstCloudState = usePointCloud(firstCloudId, sample, pageActive)
  const firstDecodedBounds = useMemo(() => {
    if (firstCloudState.status !== 'ready') return null
    return firstCloudState.decoded.bounds
  }, [firstCloudState])

  // Scene bounds from meta:
  const scene = useSceneBounds()
  useEffect(() => {
    scene.setBoundsFromMeta([...cloudMetaBounds, ...gaussianMetaBounds, ...cameraMetaBounds])
  }, [cloudMetaBounds, gaussianMetaBounds, cameraMetaBounds, scene])

  const focusBounds = useMemo(() => {
    let b: Bounds3 | null = null

    if (!focusId) return null
    if (focusId === firstCloudId && firstDecodedBounds) b = firstDecodedBounds
    else {
      // Search in clouds
      const cIdx = cloudIds.findIndex((x) => x === focusId)
      if (cIdx >= 0) {
        const mb = cloudMetaBounds[cIdx]
        if (mb) b = { min: new THREE.Vector3(...mb.min), max: new THREE.Vector3(...mb.max) }
      } else {
        // Search in gaussians
        const gIdx = gaussianIds.findIndex((x) => x === focusId)
        if (gIdx >= 0) {
          const mb = gaussianMetaBounds[gIdx]
          if (mb) b = { min: new THREE.Vector3(...mb.min), max: new THREE.Vector3(...mb.max) }
        } else {
          // Search in cameras
          const camIdx = cameraIds.findIndex((x) => x === focusId)
          if (camIdx >= 0) {
            const mb = cameraMetaBounds[camIdx]
            if (mb) b = { min: new THREE.Vector3(...mb.min), max: new THREE.Vector3(...mb.max) }
          }
        }
      }
      if (!b) b = scene.bounds
    }

    if (!b) return null
    return transformBounds(b, convention.worldToView)
  }, [
    cameraIds,
    cameraMetaBounds,
    cloudIds,
    gaussianIds,
    cloudMetaBounds,
    gaussianMetaBounds,
    convention.worldToView,
    firstCloudId,
    firstDecodedBounds,
    focusId,
    scene.bounds,
  ])

  const canvasGesture = useRef(createClickGesture())
  const objectMapRef = useRef<Map<string, THREE.Object3D>>(new Map())
  const [objectsVersion, setObjectsVersion] = useState(0)
  const lastTransformRef = useRef<string | null>(null)
  const lastLocalPoseRef = useRef<string | null>(null)
  const registerObject = useCallback((id: string, obj: THREE.Object3D | null) => {
    if (obj) objectMapRef.current.set(id, obj)
    else objectMapRef.current.delete(id)
    setObjectsVersion((v) => v + 1)
  }, [])
  const selectedObject = useMemo(() => {
    if (!selectedId) return null
    return objectMapRef.current.get(selectedId) ?? null
  }, [selectedId, objectsVersion])

  const activeCameraReady = !!activeCameraMeta
  const followCameraActive = !activeCameraReady && !!followCameraMeta
  const hiddenCameraIds = useMemo(() => {
    const ids = new Set<string>()
    if (activeCameraId) ids.add(activeCameraId)
    if (followCameraId) ids.add(followCameraId)
    return ids
  }, [activeCameraId, followCameraId])

  useEffect(() => {
    if (orbitRef.current) {
      orbitRef.current.enabled = !activeCameraReady
    }
  }, [activeCameraReady])

  return (
    <Canvas
      // Do NOT key the canvas by conventionId. That causes a full remount and camera reset.
      frameloop={pageActive ? 'demand' : 'never'}
      camera={{ fov: 60, near: 0.01, far: 1000, position: [2.5, 2.0, 2.5] }}
      gl={{ powerPreference: 'low-power', antialias: false }}
      dpr={secondaryView ? [0.75, 1.0] : [1, 1.5]}
      onCreated={({ gl, camera }) => {
        gl.setClearColor(background, 1)
        gl.autoClear = true

        gl.outputColorSpace = THREE.SRGBColorSpace
        gl.toneMapping = THREE.NoToneMapping
        gl.toneMappingExposure = 1.0

        ;(camera as unknown as { controls?: any }).controls = undefined
        ;(camera as any).userData = (camera as any).userData ?? {}
        ;(camera as any).userData.pendingOrbitTarget = null

        // ALWAYS keep Three.js native up for camera/controls
        ;(camera as THREE.Camera).up.copy(VIEW_UP)
      }}
      onPointerDown={(e) => onPointerDownGesture(canvasGesture.current, e.nativeEvent)}
      onPointerMove={(e) => onPointerMoveGesture(canvasGesture.current, e.nativeEvent)}
      onPointerUp={(e) => {
        const wasClick = isClickGesture(canvasGesture.current, e.nativeEvent)
        onPointerUpGesture(canvasGesture.current)
        if (!wasClick) return
        setFollowCameraId(null)
        onSelect(null)
      }}
    >
      <DebugOverlay enabled={debugOverlayEnabled && !secondaryView} />
      <ViewCameraReporter />
      <OverlayCameraRenderPass />
      <FollowCameraDriver cameraMeta={followCameraMeta} enabled={followCameraActive} worldToView={convention.worldToView} />

      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 8, 5]} intensity={0.8} />

      {controlsReady && focusId && !activeCameraReady && !followCameraActive && (
        <FrameCamera bounds={focusBounds} focusToken={focusToken} forward={viewForward} topDown={topDownOnFocus} />
      )}

      <ActiveCameraDriver meta={activeCameraMeta} enabled={activeCameraReady} worldToView={convention.worldToView} />

      <WASDControls enabled={!activeCameraReady && !followCameraActive} speed={2.0} />

      {/* Viewer-space grid */}
      {!secondaryView && <gridHelper args={[10, 10, '#29324a', '#1b2235']} />}

      {/* World rotated per convention */}
      <group quaternion={convention.worldToView}>
        {!secondaryView && <axesHelper args={[1.5]} />}
        <MultiPointCloudScene
          cloudIds={cloudIds}
          selectedId={selectedId}
          renderMode={effectiveRenderMode}
          sample={sample}
          enabled={pageActive}
          onSelect={(id) => {
            setFollowCameraId(null)
            onSelect(id)
          }}
          onFocus={(id) => {
            // Explicit user intent: always refit on double-click, even if it's already focused.
            setFollowCameraId(null)
            setFocusId(id)
            bumpFocus('cloud:doubleclick-focus')
            onFocus(id)
          }}
          onRegisterObject={registerObject}
        />
        <GaussianSplatScene
          elementIds={gaussianIds}
          selectedId={selectedId}
          sample={sample}
          enabled={pageActive}
          onSelect={(id) => {
            setFollowCameraId(null)
            onSelect(id)
          }}
          onFocus={(id) => {
            setFollowCameraId(null)
            setFocusId(id)
            bumpFocus('gaussians:doubleclick-focus')
            onFocus(id)
          }}
          onRegisterObject={registerObject}
        />
        <CameraScene
          cameras={cameraVisuals}
          selectedId={selectedId}
          hiddenIds={Array.from(hiddenCameraIds)}
          onSelect={(id) => onSelect(id)}
          onFocus={(id) => {
            setFocusId(id)
            setFollowCameraId(id)
            onFocus(id)
          }}
          onRegisterObject={registerObject}
        />
      </group>

      {!activeCameraReady && (
        <OrbitControls
          // Do NOT key by conventionId for the same reason as Canvas.
          makeDefault
          minPolarAngle={0}
          maxPolarAngle={Math.PI}
          minDistance={0}
          maxDistance={Infinity}
          enableDamping
          dampingFactor={0.07}
          onStart={() => {
            setFollowCameraId(null)
          }}
          ref={(ctrl) => {
            orbitRef.current = ctrl
            if (!ctrl) return

            const cam = (ctrl as any)?.object
            if (cam) {
              ;(cam as THREE.Camera).up.copy(VIEW_UP)
              ;(cam as any).controls = ctrl

              const pending = (cam as any).userData?.pendingOrbitTarget as THREE.Vector3 | null | undefined
              if (pending) {
                ctrl.target.copy(pending)
                ctrl.update()
                ;(cam as any).userData.pendingOrbitTarget = null
              }
            }
            setControlsReady(true)
          }}
        />
      )}

      {selectedObject && onTransformCommit && !followCameraActive && (
        <TransformControls
          object={selectedObject}
          mode={transformMode}
          space="world"
          onObjectChange={() => {
            if (!selectedId || !selectedObject) return
            const pos = selectedObject.position
            const quat = selectedObject.quaternion
            const offset = (selectedObject as any).userData?.centerOffset as THREE.Vector3 | undefined
            const ox = offset?.x ?? 0
            const oy = offset?.y ?? 0
            const oz = offset?.z ?? 0
            const qLen = Math.hypot(quat.x, quat.y, quat.z, quat.w) || 1
            const rotation: [number, number, number, number] = [
              quat.x / qLen,
              quat.y / qLen,
              quat.z / qLen,
              quat.w / qLen,
            ]
            const payload: [number, number, number, number, number, number, number] = [
              pos.x - ox,
              pos.y - oy,
              pos.z - oz,
              rotation[0],
              rotation[1],
              rotation[2],
              rotation[3],
            ]
            const key = payload.map((v) => v.toFixed(6)).join(',')
            if (lastLocalPoseRef.current === key) return
            lastLocalPoseRef.current = key
            try {
              const anyWin = window as any
              if (!anyWin.__begira_local_pose) anyWin.__begira_local_pose = {}
              anyWin.__begira_local_pose[selectedId] = {
                position: [pos.x - ox, pos.y - oy, pos.z - oz],
                rotation,
              }
              window.dispatchEvent(
                new CustomEvent('begira_local_pose_changed', {
                  detail: { id: selectedId, position: [pos.x - ox, pos.y - oy, pos.z - oz], rotation },
                }),
              )
            } catch {}
          }}
          onMouseDown={() => {
            if (orbitRef.current) orbitRef.current.enabled = false
          }}
          onMouseUp={() => {
            if (orbitRef.current) orbitRef.current.enabled = true
            const pos = selectedObject.position
            const quat = selectedObject.quaternion
            const offset = (selectedObject as any).userData?.centerOffset as THREE.Vector3 | undefined
            const ox = offset?.x ?? 0
            const oy = offset?.y ?? 0
            const oz = offset?.z ?? 0
            const payload: [number, number, number, number, number, number, number] = [
              pos.x - ox,
              pos.y - oy,
              pos.z - oz,
              quat.x,
              quat.y,
              quat.z,
              quat.w,
            ]
            const key = payload.map((v) => v.toFixed(6)).join(',')
            if (lastTransformRef.current === key) return
            lastTransformRef.current = key
            onTransformCommit(selectedId!, [pos.x - ox, pos.y - oy, pos.z - oz], [quat.x, quat.y, quat.z, quat.w])
          }}
        />
      )}
    </Canvas>
  )
}
