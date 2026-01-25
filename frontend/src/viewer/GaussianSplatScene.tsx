import { useEffect, useRef } from 'react'
import { useGaussians } from './useGaussians'
import { SplatMesh } from '@sparkjsdev/spark'
import { useThree, useFrame } from '@react-three/fiber'
import * as THREE from 'three'

function Gaussians({
  elementId,
  selected: _selected,
  onSelect,
  onFocus,
}: {
  elementId: string
  selected: boolean
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
}) {
  const state = useGaussians(elementId)
  const groupRef = useRef<THREE.Group>(null)
  const { camera, size } = useThree()
  // We'll keep two meshes for simple LOD switching
  const splatHighRef = useRef<SplatMesh | null>(null)
  const splatLowRef = useRef<SplatMesh | null>(null)
  const splatMedRef = useRef<SplatMesh | null>(null)
  const blobUrlRef = useRef<{ high?: string; low?: string }>({})
  const loadingRef = useRef(false)
  const boundingSphereRef = useRef<THREE.Sphere | null>(null)

  // LOD configuration (tweakable)
  // We'll use a screen-relative projected radius (NDC units) of the object's bounding sphere
  // NDC radius runs roughly in [0..1] where 1 is half the screen extent; tune these
  const HIGH_NDC = 0.25 // if object's projected radius >= this -> high LOD
  const MEDIUM_NDC = 0.083 // if >= this -> medium LOD
  const FRAME_THROTTLE = 8 // check distance every N frames (throttle to reduce work)
  const SAMPLE_STEP_MEDIUM = 3 // decimation for medium LOD
  const SAMPLE_STEP_LOW = 6 // decimation factor for low LOD (more aggressive)

  // Utility: build a PLY blob from geometry; sampleStep allows decimation
  const createPLYBlob = (sampleStep = 1) => {
    const geom = (state as any).decoded.geometry
    const positions = geom.attributes.position
    const sh0 = geom.attributes.sh0
    const opacity = geom.attributes.opacity
    const scales = geom.attributes.scale
    const rotations = geom.attributes.rotation
    const originalCount = positions.count

    // Deterministic random sampling to avoid regular holes from strided sampling
    // For sampleStep <= 1 take all points; otherwise pick approx originalCount/sampleStep
    const indices: number[] = []
    if (sampleStep <= 1) {
      for (let i = 0; i < originalCount; i++) indices.push(i)
    } else {
      const target = Math.max(1, Math.floor(originalCount / sampleStep))
      // seeded xorshift RNG based on elementId and sampleStep so selection is stable
      let seed = 2166136261 >>> 0
      for (let i = 0; i < elementId.length; i++) seed = (seed ^ elementId.charCodeAt(i)) >>> 0
      seed = (seed + sampleStep * 374761393) >>> 0
      function rand(): number {
        // xorsift32
        seed ^= seed << 13
        seed ^= seed >>> 17
        seed ^= seed << 5
        return (seed >>> 0) / 4294967295
      }

      // reservoir-like selection: choose random indices without replacement
      const used = new Set<number>()
      while (indices.length < target && indices.length < originalCount) {
        const idx = Math.floor(rand() * originalCount)
        if (!used.has(idx)) {
          used.add(idx)
          indices.push(idx)
        }
      }
      // ensure deterministic order for better compression and stability
      indices.sort((a, b) => a - b)
    }
    const count = indices.length

    // Build PLY header
    let header = 'ply\n'
    header += 'format binary_little_endian 1.0\n'
    header += `element vertex ${count}\n`
    header += 'property float x\n'
    header += 'property float y\n'
    header += 'property float z\n'
    header += 'property float f_dc_0\n'
    header += 'property float f_dc_1\n'
    header += 'property float f_dc_2\n'
    header += 'property float opacity\n'
    header += 'property float scale_0\n'
    header += 'property float scale_1\n'
    header += 'property float scale_2\n'
    header += 'property float rot_0\n'
    header += 'property float rot_1\n'
    header += 'property float rot_2\n'
    header += 'property float rot_3\n'
    header += 'end_header\n'

    const headerBytes = new TextEncoder().encode(header)
    const dataBytes = count * 14 * 4

    const plyBuffer = new Uint8Array(headerBytes.length + dataBytes)
    plyBuffer.set(headerBytes, 0)

    const dataView = new DataView(plyBuffer.buffer, headerBytes.length)
    for (let idx = 0; idx < count; idx++) {
      const i = indices[idx]
      const offset = idx * 14 * 4
      dataView.setFloat32(offset + 0, positions.array[i * 3 + 0], true)
      dataView.setFloat32(offset + 4, positions.array[i * 3 + 1], true)
      dataView.setFloat32(offset + 8, positions.array[i * 3 + 2], true)
      dataView.setFloat32(offset + 12, sh0.array[i * 3 + 0], true)
      dataView.setFloat32(offset + 16, sh0.array[i * 3 + 1], true)
      dataView.setFloat32(offset + 20, sh0.array[i * 3 + 2], true)
      dataView.setFloat32(offset + 24, opacity.array[i], true)
      dataView.setFloat32(offset + 28, scales.array[i * 3 + 0], true)
      dataView.setFloat32(offset + 32, scales.array[i * 3 + 1], true)
      dataView.setFloat32(offset + 36, scales.array[i * 3 + 2], true)
      dataView.setFloat32(offset + 40, rotations.array[i * 4 + 0], true)
      dataView.setFloat32(offset + 44, rotations.array[i * 4 + 1], true)
      dataView.setFloat32(offset + 48, rotations.array[i * 4 + 2], true)
      dataView.setFloat32(offset + 52, rotations.array[i * 4 + 3], true)
    }

    return new Blob([plyBuffer], { type: 'application/octet-stream' })
  }

  // Create both LOD blobs and SplatMesh instances once when ready
  useEffect(() => {
    if (state.status !== 'ready' || !groupRef.current || loadingRef.current) return
    loadingRef.current = true

    const setup = async () => {
      try {
        // Create high LOD blob
        if (!blobUrlRef.current.high) {
          const blobHigh = createPLYBlob(1)
          blobUrlRef.current.high = URL.createObjectURL(blobHigh)
        }

        // Create medium and low LOD blobs (only if decimation will reduce points)
        if (!blobUrlRef.current.low || !(blobUrlRef.current as any).med) {
          const positions = (state as any).decoded.geometry.attributes.position
          if (positions.count > SAMPLE_STEP_MEDIUM * 2) {
            // medium
            const blobMed = createPLYBlob(SAMPLE_STEP_MEDIUM)
            ;(blobUrlRef.current as any).med = URL.createObjectURL(blobMed)
          } else {
            ;(blobUrlRef.current as any).med = blobUrlRef.current.high
          }

          if (positions.count > SAMPLE_STEP_LOW * 2) {
            const blobLow = createPLYBlob(SAMPLE_STEP_LOW)
            blobUrlRef.current.low = URL.createObjectURL(blobLow)
          } else {
            blobUrlRef.current.low = blobUrlRef.current.high
          }
        }

        // Compute a simple bounding sphere for frustum culling (from positions)
        if (!boundingSphereRef.current) {
          try {
            const posAttr = (state as any).decoded.geometry.attributes.position
            const n = posAttr.count
            let cx = 0
            let cy = 0
            let cz = 0
            for (let i = 0; i < n; i++) {
              cx += posAttr.array[i * 3 + 0]
              cy += posAttr.array[i * 3 + 1]
              cz += posAttr.array[i * 3 + 2]
            }
            cx /= n
            cy /= n
            cz /= n
            let maxSq = 0
            for (let i = 0; i < n; i++) {
              const dx = posAttr.array[i * 3 + 0] - cx
              const dy = posAttr.array[i * 3 + 1] - cy
              const dz = posAttr.array[i * 3 + 2] - cz
              const d2 = dx * dx + dy * dy + dz * dz
              if (d2 > maxSq) maxSq = d2
            }
            boundingSphereRef.current = new THREE.Sphere(new THREE.Vector3(cx, cy, cz), Math.sqrt(maxSq))
          } catch (e) {
            // ignore failures, optional optimization
            boundingSphereRef.current = null
          }
        }

        // Create SplatMesh instances. Spark internally manages async parsing, so create both and add to group
        if (!splatHighRef.current) {
          splatHighRef.current = new SplatMesh({
            url: blobUrlRef.current.high!,
            // @ts-ignore
            splatScale: 1.0,
            // @ts-ignore
            alphaTest: 0.08,
            // @ts-ignore
            renderMode: 'default',
            // @ts-ignore
            onLoad: () => {
              // noop
            },
            // @ts-ignore
            onError: (err: any) => console.error('[Gaussians] High LOD Error:', err),
          })
          splatHighRef.current.visible = false
          groupRef.current!.add(splatHighRef.current)
        }

        // create medium LOD
        if (!splatMedRef.current) {
          splatMedRef.current = new SplatMesh({
            url: (blobUrlRef.current as any).med!,
            // @ts-ignore
            splatScale: 1.2,
            // @ts-ignore
            alphaTest: 0.11,
            // @ts-ignore
            renderMode: 'performance',
            // @ts-ignore
            onLoad: () => {},
            // @ts-ignore
            onError: (err: any) => console.error('[Gaussians] Med LOD Error:', err),
          })
          splatMedRef.current.visible = false
          groupRef.current!.add(splatMedRef.current)
        }

        if (!splatLowRef.current) {
          splatLowRef.current = new SplatMesh({
            url: blobUrlRef.current.low!,
            // make low LOD slightly larger so it reads fine at distance
            // @ts-ignore
            splatScale: 2.0,
            // @ts-ignore
            alphaTest: 0.14,
            // @ts-ignore
            renderMode: 'performance',
            // @ts-ignore
            onLoad: () => {},
            // @ts-ignore
            onError: (err: any) => console.error('[Gaussians] Low LOD Error:', err),
          })
          splatLowRef.current.visible = false
          groupRef.current!.add(splatLowRef.current)
        }

        // Initial visibility: show low by default until distance check runs
        if (splatLowRef.current) splatLowRef.current.visible = true
      } catch (err) {
        console.error('[Gaussians] setup failed:', err)
      } finally {
        loadingRef.current = false
      }
    }

    setup()

    return () => {
      // cleanup on unmount / state change
      loadingRef.current = false
      if (splatHighRef.current && groupRef.current) {
        groupRef.current.remove(splatHighRef.current)
        try {
          splatHighRef.current.dispose()
        } catch (e) {
          // ignore
        }
        splatHighRef.current = null
      }
      if (splatMedRef.current && groupRef.current) {
        groupRef.current.remove(splatMedRef.current)
        try {
          splatMedRef.current.dispose()
        } catch (e) {}
        splatMedRef.current = null
      }
      if (splatLowRef.current && groupRef.current) {
        groupRef.current.remove(splatLowRef.current)
        try {
          splatLowRef.current.dispose()
        } catch (e) {
          // ignore
        }
        splatLowRef.current = null
      }
      boundingSphereRef.current = null
      if (blobUrlRef.current.high) {
        URL.revokeObjectURL(blobUrlRef.current.high)
      }
      if ((blobUrlRef.current as any).med && (blobUrlRef.current as any).med !== blobUrlRef.current.high) {
        URL.revokeObjectURL((blobUrlRef.current as any).med)
      }
      if (blobUrlRef.current.low && blobUrlRef.current.low !== blobUrlRef.current.high) {
        URL.revokeObjectURL(blobUrlRef.current.low)
      }
      blobUrlRef.current = {}
    }
  // Only re-run when decoded geometry instance changes
  }, [state.status])

  // Throttled per-frame LOD selection based on camera distance
  const frameCounter = useRef(0)
  // helper to publish LOD status to global registry for the debug overlay
  const publishLOD = (level: 'high' | 'medium' | 'low') => {
    try {
      const anyWin = window as any
      if (!anyWin.__begira_lod_status) anyWin.__begira_lod_status = {}
      anyWin.__begira_lod_status[elementId] = level
    } catch (e) {
      // ignore
    }
  }
  useFrame(() => {
    frameCounter.current = (frameCounter.current + 1) % FRAME_THROTTLE
    if (frameCounter.current !== 0) return
    const group = groupRef.current
    if (!group || (!splatHighRef.current && !splatLowRef.current) || !camera) return

    // Frustum culling: if bounding sphere present and not in frustum, hide both
    if (boundingSphereRef.current) {
      const projView = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
      const frustum = new THREE.Frustum()
      frustum.setFromProjectionMatrix(projView)

      // Transform sphere center to world space and scale radius by group's world scale
      const localSphere = boundingSphereRef.current
      const centerWorld = localSphere.center.clone().applyMatrix4(group.matrixWorld)

      // Decompose group's world matrix to get scale components
      const tmpPos = new THREE.Vector3()
      const tmpQuat = new THREE.Quaternion()
      const tmpScale = new THREE.Vector3()
      group.matrixWorld.decompose(tmpPos, tmpQuat, tmpScale)
      const maxScale = Math.max(tmpScale.x, tmpScale.y, tmpScale.z, 1)
      const worldRadius = localSphere.radius * maxScale

      const worldSphere = new THREE.Sphere(centerWorld, worldRadius)
      if (!frustum.intersectsSphere(worldSphere)) {
        if (splatHighRef.current) splatHighRef.current.visible = false
        if (splatLowRef.current) splatLowRef.current.visible = false
        return
      }
    }

    // compute representative position for distance: group's world position
    const worldPos = new THREE.Vector3()
    group.getWorldPosition(worldPos)

    // If we have a bounding sphere, compute its projected radius in NDC (normalized device coords)
    // NDC coordinates run -1..1 across the viewport; the radius in NDC is a screen-relative metric.
    let ndcRadius = 0
    if (boundingSphereRef.current && camera) {
      const localSphere = boundingSphereRef.current
      const centerWorld = localSphere.center.clone().applyMatrix4(group.matrixWorld)

      // world radius scaled by group's max world scale
      const tmpPos = new THREE.Vector3()
      const tmpQuat = new THREE.Quaternion()
      const tmpScale = new THREE.Vector3()
      group.matrixWorld.decompose(tmpPos, tmpQuat, tmpScale)
      const maxScale = Math.max(tmpScale.x, tmpScale.y, tmpScale.z, 1)
      const worldRadius = localSphere.radius * maxScale

      const camPos = camera.getWorldPosition(new THREE.Vector3())
      const camToCenter = centerWorld.clone().sub(camPos)
      const dist = camToCenter.length()
      if (dist > 0 && worldRadius > 0) {
        // pick a perpendicular direction and a point on the sphere's rim
        let perp = new THREE.Vector3(1, 0, 0)
        if (Math.abs(camToCenter.clone().normalize().dot(perp)) > 0.9) perp = new THREE.Vector3(0, 1, 0)
        perp = camToCenter.clone().cross(perp).normalize()
        const pointOnSphere = centerWorld.clone().add(perp.multiplyScalar(worldRadius))

        // project center and point, then compute NDC distance
        const ndcCenter = centerWorld.clone().project(camera)
        const ndcPoint = pointOnSphere.clone().project(camera)
        const dx = ndcPoint.x - ndcCenter.x
        const dy = ndcPoint.y - ndcCenter.y
        ndcRadius = Math.sqrt(dx * dx + dy * dy)
      }
    }

    // check for global or per-element overrides
    const anyWin = window as any
    const perElemOverride = anyWin.__begira_lod_override?.[elementId]
    const globalOverride = anyWin.__begira_lod_override_global

    // Decide among high / medium / low using screen-space radius thresholds
    let chosen: 'high' | 'medium' | 'low' = 'low'
    if (perElemOverride === 'high') chosen = 'high'
    else if (perElemOverride === 'medium') chosen = 'medium'
    else if (perElemOverride === 'low') chosen = 'low'
    else if (globalOverride === 'high') chosen = 'high'
    else if (globalOverride === 'medium') chosen = 'medium'
    else if (globalOverride === 'low') chosen = 'low'
    else {
      // choose based on normalized NDC radius (screen-relative)
      if (ndcRadius >= HIGH_NDC) chosen = 'high'
      else if (ndcRadius >= MEDIUM_NDC) chosen = 'medium'
      else chosen = 'low'
    }

    if (splatHighRef.current) splatHighRef.current.visible = chosen === 'high'
    if (splatMedRef.current) splatMedRef.current.visible = chosen === 'medium'
    if (splatLowRef.current) splatLowRef.current.visible = chosen === 'low'


    // publish the effective LOD for the debug overlay
    publishLOD(chosen)
  })

  if (state.status !== 'ready') return null

  return (
    <group
      ref={groupRef}
      onClick={(e) => {
        e.stopPropagation()
        onSelect(elementId)
        if (e.detail === 2) onFocus(elementId)
      }}
    />
  )
}

export default function GaussianSplatScene({
  elementIds,
  selectedId,
  onSelect,
  onFocus,
}: {
  elementIds: string[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
}) {
  return (
    <>
      {elementIds.map((id) => (
        <Gaussians
          key={id}
          elementId={id}
          selected={id === selectedId}
          onSelect={onSelect}
          onFocus={onFocus}
        />
      ))}
    </>
  )
}
