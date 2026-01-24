import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { useEffect, useLayoutEffect, useMemo, useRef, useState, useCallback } from 'react'
import * as THREE from 'three'
import WASDControls from './WASDControls'
import MultiPointCloudScene from './MultiPointCloudScene'
import {
  createClickGesture,
  isClickGesture,
  onPointerDownGesture,
  onPointerMoveGesture,
  onPointerUpGesture,
} from './interaction'
import { usePointCloud } from './usePointCloud'

function fitCameraToBounds(
  camera: THREE.PerspectiveCamera,
  size: { width: number; height: number },
  bounds: { min: THREE.Vector3; max: THREE.Vector3 }
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

  // Slightly larger default margin than before; point clouds often look best a bit more zoomed out.
  const distance = (baseDist || 1) * 1.4

  const dir = new THREE.Vector3(1, 0.85, 1).normalize()
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
}: {
  bounds: { min: THREE.Vector3; max: THREE.Vector3 } | null
  focusToken: number
}) {
  const { camera, size } = useThree()

  // Fit only on explicit focus requests.
  useLayoutEffect(() => {
    if (!bounds) return

    // Avoid fitting against a transient 0-sized canvas (common during initial layout).
    if (!Number.isFinite(size.width) || !Number.isFinite(size.height) || size.width <= 1 || size.height <= 1) return

    const handle = requestAnimationFrame(() => {
      const perspective = camera as THREE.PerspectiveCamera
      const { center } = fitCameraToBounds(perspective, size, bounds)

      const anyCam = camera as unknown as { controls?: { target: THREE.Vector3; update: () => void }; userData?: any }
      if (anyCam.controls) {
        anyCam.controls.target.copy(center)
        anyCam.controls.update()
      } else {
        ;(camera as any).userData = (camera as any).userData ?? {}
        ;(camera as any).userData.pendingOrbitTarget = center.clone()
      }
    })

    return () => cancelAnimationFrame(handle)
    // IMPORTANT: don't depend on bounds here; focusToken is the sole trigger.
  }, [camera, focusToken, size.height, size.width])

  return null
}

function useSceneBounds(): {
  bounds: { min: THREE.Vector3; max: THREE.Vector3 } | null
  setBoundsFromMeta: (metaBounds: { min: [number, number, number]; max: [number, number, number] }[]) => void
} {
  const api = useMemo(() => {
    const state: {
      bounds: { min: THREE.Vector3; max: THREE.Vector3 } | null
    } = { bounds: null }

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

  return {
    bounds: api.state.bounds,
    setBoundsFromMeta: api.setBoundsFromMeta,
  }
}

// Local helper: avoid depending on module export resolution for isDoubleClick
function isDoubleClickLocal(nowMs: number, lastClickMs: number | null, maxGapMs = 350): boolean {
  if (lastClickMs === null) return false
  return nowMs - lastClickMs <= maxGapMs
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

  const [controlsReady, setControlsReady] = useState(false)
  const orbitRef = useRef<any>(null)

  // Camera focus target (what we want to frame). Default is the first cloud.
  const [focusId, setFocusId] = useState<string | null>(null)
  const focusIdRef = useRef<string | null>(null)
  useEffect(() => {
    focusIdRef.current = focusId
  }, [focusId])

  // Instead of a boolean auto-frame flag (which can be accidentally re-enabled),
  // use a monotonically increasing token: bump it only for explicit focus requests.
  const [focusToken, setFocusToken] = useState(0)

  const bumpFocus = useCallback((source?: string) => {
    setFocusToken((t) => {
      const next = t + 1
      // Debug logging for focus token bumps was useful during development but is noisy in production.
      // Keep the source parameter for potential future debugging.
      void source
      return next
    })
  }, [])

  // Keep OrbitControls -> focus interaction wiring.
  useEffect(() => {
    const ctrl = orbitRef.current
    if (!ctrl) return

    // Any manual orbiting means we should NOT refit unless explicitly requested.
    const onStart = () => {
      /* no-op: having this listener makes it easy to extend if needed */
    }
    ctrl.addEventListener('start', onStart)
    return () => {
      ctrl.removeEventListener('start', onStart)
    }
  }, [controlsReady])

  // Initialize focus when clouds first appear.
  const prevCloudCount = useRef(0)
  useEffect(() => {
    const prev = prevCloudCount.current
    prevCloudCount.current = cloudIds.length

    if (cloudIds.length === 0) {
      setFocusId(null)
      return
    }

    // First cloud appears at startup.
    if (prev === 0 && cloudIds.length > 0) {
      setFocusId(cloudIds[0])
      bumpFocus('startup:first-cloud')
      return
    }

    // If focused cloud disappeared, fall back to first.
    setFocusId((cur) => {
      if (!cur) return cloudIds[0]
      return cloudIds.includes(cur) ? cur : cloudIds[0]
    })
  }, [cloudIds])

  // External focus request (e.g. hierarchy double-click): focus that cloud.
  useEffect(() => {
    if (!focusTarget) return
    if (!cloudIds.includes(focusTarget)) return

    const prev = focusIdRef.current
    if (prev !== focusTarget) {
      setFocusId(focusTarget)
      bumpFocus('effect:focusTarget')
    }

    // Consume the focus request so it doesn't keep re-triggering on every rerender.
    onFocus(null)
  }, [cloudIds, focusTarget, bumpFocus, onFocus])

  // Load first cloud to obtain decoded bounds for initial framing.
  const firstCloudId = cloudIds[0] ?? ''
  const firstCloudState = usePointCloud(firstCloudId)
  const firstDecodedBounds = useMemo(() => {
    if (firstCloudState.status !== 'ready') return null
    return firstCloudState.decoded.bounds
  }, [firstCloudState])

  // Scene aggregate bounds derived from meta bounds.
  const scene = useSceneBounds()
  useEffect(() => {
    scene.setBoundsFromMeta(cloudMetaBounds)
  }, [cloudMetaBounds, scene])

  const focusBounds = useMemo(() => {
    if (!focusId) return scene.bounds

    if (focusId === firstCloudId && firstDecodedBounds) return firstDecodedBounds

    const idx = cloudIds.findIndex((x) => x === focusId)
    if (idx >= 0) {
      const b = cloudMetaBounds[idx]
      if (b) return { min: new THREE.Vector3(...b.min), max: new THREE.Vector3(...b.max) }
    }

    return scene.bounds
  }, [cloudIds, cloudMetaBounds, firstCloudId, firstDecodedBounds, focusId, scene.bounds])

  const lastSceneClickMs = useRef<number | null>(null)
  const canvasGesture = useRef(createClickGesture())

  return (
    <Canvas
      camera={{ fov: 60, near: 0.01, far: 1000, position: [2.5, 2.0, 2.5] }}
      onCreated={({ gl, camera }) => {
        gl.setClearColor(background)
        ;(camera as unknown as { controls?: any }).controls = undefined
        ;(camera as any).userData = (camera as any).userData ?? {}
        ;(camera as any).userData.pendingOrbitTarget = null
      }}
      onPointerDown={(e) => {
        onPointerDownGesture(canvasGesture.current, e.nativeEvent)
      }}
      onPointerMove={(e) => {
        onPointerMoveGesture(canvasGesture.current, e.nativeEvent)
      }}
      onPointerUp={(e) => {
        const wasClick = isClickGesture(canvasGesture.current, e.nativeEvent)
        onPointerUpGesture(canvasGesture.current)
        if (!wasClick) return

        // Single click on empty space clears selection but must NOT re-center the camera.
        onSelect(null)
      }}
    >
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 8, 5]} intensity={0.8} />

      {controlsReady && <FrameCamera bounds={focusBounds} focusToken={focusToken} />}

      <WASDControls enabled speed={2.0} />

      <MultiPointCloudScene
        cloudIds={cloudIds}
        selectedId={selectedId}
        onSelect={(id) => {
          // Selecting should not refit camera.
          onSelect(id)
        }}
        onFocus={(id) => {
          // Explicit focus only (double-click on a cloud).
          const prev = focusIdRef.current
          if (prev !== id) {
            setFocusId(id)
            bumpFocus('cloud:doubleclick-focus')
          }
          onFocus(id)
        }}
      />

      <gridHelper args={[10, 10, '#29324a', '#1b2235']} />
      <OrbitControls
        makeDefault
        ref={(ctrl) => {
          orbitRef.current = ctrl
          if (!ctrl) return

          const cam = (ctrl as any)?.object
          if (!cam) return

          ;(cam as any).controls = ctrl

          const pending = (cam as any).userData?.pendingOrbitTarget as THREE.Vector3 | null | undefined
          if (pending) {
            ctrl.target.copy(pending)
            ctrl.update()
            ;(cam as any).userData.pendingOrbitTarget = null
          }

          setControlsReady(true)
        }}
      />
    </Canvas>
  )
}
