import * as THREE from 'three'
import { useEffect, useMemo, useRef } from 'react'
import { useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { getCircleSpriteTexture } from './spriteCircle'
import { usePointCloud } from './usePointCloud'
import {
  createClickGesture,
  isClickGesture,
  isDoubleClick,
  onPointerDownGesture,
  onPointerMoveGesture,
  onPointerUpGesture,
} from './interaction'
import { useThree } from '@react-three/fiber'
import { DEFAULT_DEPTH_COLORMAP, DEFAULT_HEIGHT_COLORMAP, buildColormapLUT, sampleColormapLUT, type ColormapId } from './colormaps'

export type PointCloudRenderMode = 'fast' | 'circles' | 'quality'

function Cloud({
  cloudId,
  selected,
  onSelect,
  onFocus,
  renderMode,
}: {
  cloudId: string
  selected: boolean
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
  renderMode: PointCloudRenderMode
}) {
  const state = usePointCloud(cloudId)
  const circleMap = useMemo(() => getCircleSpriteTexture(), [])

  const gesture = useRef(createClickGesture())
  const lastClickMs = useRef<number | null>(null)

  useEffect(() => {
    gesture.current = createClickGesture()
    lastClickMs.current = null
  }, [cloudId])

  // read visual override from window globals
  const anyWin = typeof window !== 'undefined' ? (window as any) : ({} as any)
  const vis = anyWin.__begira_visual_override && anyWin.__begira_visual_override[cloudId]
  const colorMode: 'logged' | 'solid' | 'height' | 'depth' = (vis && vis.colorMode) || 'logged'
  const defaultMap = colorMode === 'depth' ? DEFAULT_DEPTH_COLORMAP : DEFAULT_HEIGHT_COLORMAP
  const colorMap: ColormapId = (vis && vis.colorMap) || defaultMap
  const lut = useMemo(() => buildColormapLUT(colorMap, 256), [colorMap])
  const visibilityMap = (window as any).__begira_visibility || {}
  const isVisible = visibilityMap[cloudId] !== false

  // listen for inspector override changes so we re-render
  const [, setTick] = useState(0)
  useEffect(() => {
    const handler = (e: any) => {
      if (!e?.detail?.id) return
      if (e.detail.id !== cloudId) return
      setTick((t) => t + 1)
    }
    try {
      window.addEventListener('begira_visual_override_changed', handler)
      window.addEventListener('begira_visibility_changed', handler)
    } catch {}
    return () => {
      try {
        window.removeEventListener('begira_visual_override_changed', handler)
        window.removeEventListener('begira_visibility_changed', handler)
      } catch {}
    }
  }, [cloudId])

  const { camera } = useThree()
  const pointsRef = useRef<THREE.Points | null>(null)

  // helper: compute height colors once and attach as geometry attribute
  const computeHeightColors = () => {
    if (colorMode !== 'height') return
    try {
      const decoded = (state as any).decoded
      if (!decoded || !decoded.geometry) return
      const geom = decoded.geometry
      const pos = geom.attributes.position
      if (!pos) return
      const n = pos.count
      if (n === 0) return

      // We'll transform each position into view-space using modelView = camera.matrixWorldInverse * points.matrixWorld
      if (!pointsRef.current) return
      // Ensure world matrix up to date
      pointsRef.current.updateMatrixWorld(true)

      // Determine authored/world-up axis in world coordinates.
      // The points are parented under a top-level group that applies the coordinate-convention rotation
      // (convention.worldToView). We detect whether authored-up is Z or Y by seeing which local axis maps
      // closest to the viewer's display up (0,1,0) after the parent's quaternion is applied.
      const parent = pointsRef.current.parent
      if (!parent) return
      const parentQuat = new THREE.Quaternion()
      parent.getWorldQuaternion(parentQuat)

      const viewUp = new THREE.Vector3(0, 1, 0)
      const zAxis = new THREE.Vector3(0, 0, 1).applyQuaternion(parentQuat)
      const yAxis = new THREE.Vector3(0, 1, 0).applyQuaternion(parentQuat)

      // pick authored local up: whichever local axis maps closer to viewUp indicates that axis was the authored-up
      const authoredLocalUp = zAxis.distanceTo(viewUp) < yAxis.distanceTo(viewUp) ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0)

      // authoredUp in world coordinates (apply parent's world rotation)
      const authoredUpWorld = authoredLocalUp.clone().applyQuaternion(parentQuat).normalize()

      // compute bounds along the authored-up axis in world space
      let minH = Infinity
      let maxH = -Infinity
      const v = new THREE.Vector3()
      const worldPos = new THREE.Vector3()
      const m = pointsRef.current.matrixWorld
      for (let i = 0; i < n; i++) {
        v.set(pos.array[i * 3 + 0], pos.array[i * 3 + 1], pos.array[i * 3 + 2])
        worldPos.copy(v).applyMatrix4(m)
        const h = worldPos.dot(authoredUpWorld)
        if (h < minH) minH = h
        if (h > maxH) maxH = h
      }
      const range = Math.max(1e-6, maxH - minH)

      const colors = new Float32Array(n * 3)
      for (let i = 0; i < n; i++) {
        v.set(pos.array[i * 3 + 0], pos.array[i * 3 + 1], pos.array[i * 3 + 2])
        worldPos.copy(v).applyMatrix4(m)
        const dot = worldPos.dot(authoredUpWorld)
        const t = Math.min(1, Math.max(0, (dot - minH) / range))
        const [r, g, b] = sampleColormapLUT(lut, t)
        colors[i * 3] = r
        colors[i * 3 + 1] = g
        colors[i * 3 + 2] = b
      }

      // store original color attribute if present so we can restore later
      if (!geom.userData) geom.userData = {}
      // store original color attribute; store `null` explicitly when missing so we can restore/cleanly remove later
      if (geom.userData._orig_color === undefined) geom.userData._orig_color = geom.attributes.color ?? null

      geom.setAttribute('color', new THREE.BufferAttribute(colors, 3))
      geom.attributes.color.needsUpdate = true
    } catch (e) {
      // swallow
    }
  }

  useEffect(() => {
    computeHeightColors()
    // recompute once when geometry or camera reference changes
  }, [colorMode, colorMap, lut, (state as any).decoded?.geometry, camera])

  // Also recompute periodically during frames while in height mode so gradient follows camera movement
  const frameCounter = useRef(0)
  useFrame(() => {
    if (colorMode === 'height') {
      frameCounter.current = (frameCounter.current + 1) % 8
      if (frameCounter.current === 0) computeHeightColors()
    }
    if (colorMode === 'depth') {
      // throttle depth recompute
      const now = performance.now()
      if (now - lastDepthUpdateRef.current > DEPTH_UPDATE_MS) {
        lastDepthUpdateRef.current = now
        computeDepthColors()
      }
    }
  })

  // helper: when switching back to logged, restore original colors if we replaced them
  useEffect(() => {
    if (colorMode !== 'logged') return
    try {
      const decoded = (state as any).decoded
      if (!decoded || !decoded.geometry || !decoded.geometry.userData) return
      const geom = decoded.geometry
      // If the original color was null (meaning there was no color attribute originally), remove the attribute.
      if (geom.userData._orig_color === null) {
        if (geom.attributes.color) {
          geom.deleteAttribute('color')
        }
      } else if (geom.userData._orig_color) {
        geom.setAttribute('color', geom.userData._orig_color)
      }
      // clear the stored original marker so future height toggles re-capture if needed
      delete geom.userData._orig_color
    } catch (e) {}
  }, [colorMode, colorMap, lut, (state as any).decoded?.geometry])

  // For depth mode we compute colors on the CPU and throttle updates
  const lastDepthUpdateRef = useRef<number>(0)
  const DEPTH_UPDATE_MS = 150
  useEffect(() => {
    // No-op placeholder for color mode side-effects
  }, [colorMode])

  // material ref so we can force updates when overrides change
  const materialRef = useRef<THREE.PointsMaterial | null>(null)
  useEffect(() => {
    // When the colorMode changes, mark material for recompilation once.
    try {
      if (materialRef.current) {
        materialRef.current.needsUpdate = true
      }
    } catch {}
  }, [colorMode])

  // When we switch into depth mode, compute colors immediately so the UI updates on button press
  useEffect(() => {
    if (colorMode === 'depth') {
      try {
        if (materialRef.current) materialRef.current.needsUpdate = true
        computeDepthColors()
      } catch {}
    }
  }, [colorMode, colorMap, lut, (state as any).decoded?.geometry])

  // We now compute depth colors on the CPU and write a color attribute, so we don't need
  // the previous shader-based depth injection. Leave onBeforeCompile as a no-op.
  const onBeforeCompile = (_shader: any) => {
    /* no-op: depth coloring is handled via per-vertex color attribute */
  }

  // Compute depth colors on the CPU and set geometry color attribute. This mirrors the
  // gaussians approach (we compute colors from positions and write into the geometry).
  const computeDepthColors = () => {
    if (colorMode !== 'depth') return
    try {
      const decoded = (state as any).decoded
      if (!decoded || !decoded.geometry) return
      const geom = decoded.geometry
      const pos = geom.attributes.position
      if (!pos) return
      const n = pos.count
      if (n === 0) return
      if (!pointsRef.current) return

      // preserve original color attribute if not stored
      if (!geom.userData) geom.userData = {}
      if (geom.userData._orig_color === undefined) geom.userData._orig_color = geom.attributes.color ?? null

      // ensure matrices are current
      pointsRef.current.updateMatrixWorld(true)
      if (typeof camera.updateMatrixWorld === 'function') camera.updateMatrixWorld(true)
      // modelView matrix used by the shader
      const modelView = new THREE.Matrix4()
      modelView.multiplyMatrices(camera.matrixWorldInverse, pointsRef.current.matrixWorld)

      // center in local space (use boundingSphere if present, else subsampled average)
      let centerLocal = new THREE.Vector3(0, 0, 0)
      if (geom.boundingSphere) {
        centerLocal = geom.boundingSphere.center.clone()
      } else {
        const MAX_SAMPLES = 2000
        const step = Math.max(1, Math.floor(n / MAX_SAMPLES))
        let sx = 0, sy = 0, sz = 0, sc = 0
        for (let i = 0; i < n; i += step) {
          sx += pos.array[i * 3 + 0]
          sy += pos.array[i * 3 + 1]
          sz += pos.array[i * 3 + 2]
          sc++
        }
        if (sc > 0) centerLocal.set(sx / sc, sy / sc, sz / sc)
      }

      const centerMV = new THREE.Vector4(centerLocal.x, centerLocal.y, centerLocal.z, 1.0).applyMatrix4(modelView)
      const centerDepth = -centerMV.z

      // estimate min/max using subsampling for performance
      const MAX_SAMPLES = 2000
      const step = Math.max(1, Math.floor(n / MAX_SAMPLES))
      let minD = Number.POSITIVE_INFINITY
      let maxD = Number.NEGATIVE_INFINITY
      const v = new THREE.Vector3()
      const mv = new THREE.Vector4()
      for (let i = 0; i < n; i += step) {
        v.set(pos.array[i * 3 + 0], pos.array[i * 3 + 1], pos.array[i * 3 + 2])
        mv.set(v.x, v.y, v.z, 1.0).applyMatrix4(modelView)
        const d = -mv.z - centerDepth
        if (d < minD) minD = d
        if (d > maxD) maxD = d
      }
      const range = Math.max(1e-6, maxD - minD)

      const colors = new Float32Array(n * 3)
      for (let i = 0; i < n; i++) {
        v.set(pos.array[i * 3 + 0], pos.array[i * 3 + 1], pos.array[i * 3 + 2])
        mv.set(v.x, v.y, v.z, 1.0).applyMatrix4(modelView)
        const d = -mv.z - centerDepth
        const t = Math.min(1, Math.max(0, (d - minD) / range))
        const [r, g, b] = sampleColormapLUT(lut, t)
        colors[i * 3] = r
        colors[i * 3 + 1] = g
        colors[i * 3 + 2] = b
      }

      geom.setAttribute('color', new THREE.BufferAttribute(colors, 3))
      geom.attributes.color.needsUpdate = true
    } catch (e) {
      // ignore
    }
  }

  // Ensure geometry is ready
  if (state.status !== 'ready') return null

  const baseSize = Math.max(0.001, state.meta.pointSize ?? 0.02)
  const fast = renderMode === 'fast'
  const circles = renderMode === 'circles'

  return (
    <points
      ref={pointsRef}
      visible={isVisible}
      geometry={state.decoded.geometry}
      // IMPORTANT: raycasting against large THREE.Points can become extremely expensive,
      // especially while moving the camera (pointer events trigger raycasts internally).
      // We disable it to keep interaction smooth for large point clouds.
      raycast={() => null}
      onPointerDown={(e) => {
        onPointerDownGesture(gesture.current, e.nativeEvent)
      }}
      onPointerMove={(e) => {
        onPointerMoveGesture(gesture.current, e.nativeEvent)
      }}
      onPointerUp={(e) => {
        const wasClick = isClickGesture(gesture.current, e.nativeEvent)
        onPointerUpGesture(gesture.current)
        if (!wasClick) return

        e.stopPropagation()

        const now = performance.now()
        const dbl = isDoubleClick(now, lastClickMs.current)
        lastClickMs.current = now

        // If it's already selected and it's just a single click, do nothing.
        // This avoids re-triggering any selection/focus side-effects that can reset camera.
        if (!dbl && selected) return

        onSelect(cloudId)
        if (dbl) onFocus(cloudId)
      }}
    >
      <pointsMaterial
        ref={materialRef}
        // 'fast'  : opaque square points (fastest)
        // 'circles': opaque circle sprites via alphaTest (still fast, no blending)
        // 'quality': transparent blended sprites (prettiest, slowest)
        size={fast ? Math.max(0.001, baseSize * 0.75) : baseSize}
        sizeAttenuation
        // Vertex colors depend on effective visual mode. For 'logged' prefer payload colors;
        // for 'height' and 'depth' we create a color attribute above; for 'solid' disable vertexColors.
        vertexColors={colorMode === 'solid' ? false : ((state as any).decoded?.hasColor || colorMode === 'height' || colorMode === 'depth')}
        map={circles || renderMode === 'quality' ? circleMap : null}
        alphaTest={circles ? 0.5 : fast ? 0.0 : 0.05}
        transparent={renderMode === 'quality'}
        opacity={renderMode === 'quality' ? (selected ? 0.9 : 0.6) : 1.0}
        depthWrite={renderMode !== 'quality'}
        depthTest
        blending={renderMode === 'quality' ? THREE.NormalBlending : THREE.NoBlending}
        // apply solid color when requested
        color={colorMode === 'solid' && vis && vis.solidColor ? new THREE.Color(vis.solidColor[0], vis.solidColor[1], vis.solidColor[2]) : undefined}
        // ensure material recompiles when mode changes (depth shader injection)
        key={colorMode}
        // onBeforeCompile to inject depth-based coloring when requested
        onBeforeCompile={onBeforeCompile}
      />
    </points>
  )
}

export default function MultiPointCloudScene({
  cloudIds,
  selectedId,
  onSelect,
  onFocus,
  renderMode = 'fast',
}: {
  cloudIds: string[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
  renderMode?: PointCloudRenderMode
}) {
  return (
    <>
      {cloudIds.map((id) => (
        <Cloud
          key={id}
          cloudId={id}
          selected={id === selectedId}
          onSelect={onSelect}
          onFocus={onFocus}
          renderMode={renderMode}
        />
      ))}
    </>
  )
}
