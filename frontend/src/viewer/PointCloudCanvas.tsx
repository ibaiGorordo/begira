import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { useEffect, useLayoutEffect, useMemo, useRef, useState, useCallback } from 'react'
import * as THREE from 'three'
import WASDControls from './WASDControls'
import MultiPointCloudScene, { PointCloudRenderMode } from './MultiPointCloudScene'
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
import { fetchViewerSettings } from './api'

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

  useLayoutEffect(() => {
    if (!bounds) return
    if (!Number.isFinite(size.width) || !Number.isFinite(size.height) || size.width <= 1 || size.height <= 1) return

    const handle = requestAnimationFrame(() => {
      const perspective = camera as THREE.PerspectiveCamera
      const { center } = fitCameraToBounds(perspective, size, bounds, {
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
  }, [camera, focusToken, size.height, size.width, forward, topDown, bounds])

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
  selectedId,
  onSelect,
  focusTarget,
  onFocus,
  cloudMetaBounds,
}: {
  cloudIds: string[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  focusTarget: string | null
  onFocus: (id: string | null) => void
  cloudMetaBounds: { min: [number, number, number]; max: [number, number, number] }[]
}) {
  const background = useMemo(() => new THREE.Color('#0b1020'), [])

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

  const orbitRef = useRef<any>(null)
  const [controlsReady, setControlsReady] = useState(false)

  // Focus:
  const [focusId, setFocusId] = useState<string | null>(null)
  const focusIdRef = useRef<string | null>(null)
  useEffect(() => {
    focusIdRef.current = focusId
  }, [focusId])

  const [focusToken, setFocusToken] = useState(0)
  const bumpFocus = useCallback((source?: string) => {
    setFocusToken((t) => {
      void source
      return t + 1
    })
  }, [])

  // IMPORTANT: refit whenever convention changes.
  useEffect(() => {
    if (cloudIds.length === 0) return
    bumpFocus('convention-change')
  }, [conventionId, cloudIds.length, bumpFocus])

  // Choose a forward direction that makes the change visible.
  const viewForward = useMemo(() => {
    if (conventionId === 'rh-y-up') return new THREE.Vector3(0, 0, -1)
    return new THREE.Vector3(0, 0, 1)
  }, [conventionId])

  const topDownOnFocus = false

  // Initialize focus when clouds first appear.
  const prevCloudCount = useRef(0)
  useEffect(() => {
    const prev = prevCloudCount.current
    prevCloudCount.current = cloudIds.length

    if (cloudIds.length === 0) {
      setFocusId(null)
      return
    }

    if (prev === 0 && cloudIds.length > 0) {
      setFocusId(cloudIds[0])
      bumpFocus('startup:first-cloud')
      return
    }

    setFocusId((cur) => {
      if (!cur) return cloudIds[0]
      return cloudIds.includes(cur) ? cur : cloudIds[0]
    })
  }, [cloudIds, bumpFocus])

  // External focus request:
  useEffect(() => {
    if (!focusTarget) return
    if (!cloudIds.includes(focusTarget)) return

    const prev = focusIdRef.current
    if (prev !== focusTarget) {
      setFocusId(focusTarget)
      bumpFocus('effect:focusTarget')
    }
    onFocus(null)
  }, [cloudIds, focusTarget, bumpFocus, onFocus])

  // First cloud decoded bounds:
  const firstCloudId = cloudIds[0] ?? ''
  const firstCloudState = usePointCloud(firstCloudId)
  const firstDecodedBounds = useMemo(() => {
    if (firstCloudState.status !== 'ready') return null
    return firstCloudState.decoded.bounds
  }, [firstCloudState])

  // Scene bounds from meta:
  const scene = useSceneBounds()
  useEffect(() => {
    scene.setBoundsFromMeta(cloudMetaBounds)
  }, [cloudMetaBounds, scene])

  const focusBounds = useMemo(() => {
    let b: Bounds3 | null = null

    if (!focusId) b = scene.bounds
    else if (focusId === firstCloudId && firstDecodedBounds) b = firstDecodedBounds
    else {
      const idx = cloudIds.findIndex((x) => x === focusId)
      if (idx >= 0) {
        const mb = cloudMetaBounds[idx]
        if (mb) b = { min: new THREE.Vector3(...mb.min), max: new THREE.Vector3(...mb.max) }
      }
      if (!b) b = scene.bounds
    }

    if (!b) return null
    return transformBounds(b, convention.worldToView)
  }, [cloudIds, cloudMetaBounds, convention.worldToView, firstCloudId, firstDecodedBounds, focusId, scene.bounds])

  const canvasGesture = useRef(createClickGesture())

  return (
    <Canvas
      key={conventionId}
      camera={{ fov: 60, near: 0.01, far: 1000, position: [2.5, 2.0, 2.5] }}
      dpr={[1, 1.5]}
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
        onSelect(null)
      }}
    >
      <DebugOverlay enabled={debugOverlayEnabled} />

      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 8, 5]} intensity={0.8} />

      {controlsReady && (
        <FrameCamera bounds={focusBounds} focusToken={focusToken} forward={viewForward} topDown={topDownOnFocus} />
      )}

      <WASDControls enabled speed={2.0} />

      {/* Viewer-space grid */}
      <gridHelper args={[10, 10, '#29324a', '#1b2235']} />

      {/* World rotated per convention */}
      <group key={conventionId} quaternion={convention.worldToView}>
        <axesHelper args={[1.5]} />
        <MultiPointCloudScene
          cloudIds={cloudIds}
          selectedId={selectedId}
          renderMode={renderMode}
          onSelect={(id) => onSelect(id)}
          onFocus={(id) => {
            const prev = focusIdRef.current
            if (prev !== id) {
              setFocusId(id)
              bumpFocus('cloud:doubleclick-focus')
            }
            onFocus(id)
          }}
        />
      </group>

      <OrbitControls
        key={conventionId}
        makeDefault
        minPolarAngle={0}
        maxPolarAngle={Math.PI}
        minDistance={0}
        maxDistance={Infinity}
        enableDamping
        dampingFactor={0.07}
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
    </Canvas>
  )
}
