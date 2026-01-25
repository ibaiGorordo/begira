import { useEffect, useRef } from 'react'
import { useGaussians } from './useGaussians'
import { SplatMesh, dyno, type SplatTransformer } from '@sparkjsdev/spark'
import { COLORMAPS, DEFAULT_DEPTH_COLORMAP, DEFAULT_HEIGHT_COLORMAP, sampleColormap, type ColormapDef, type ColormapId } from './colormaps'
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

  // NEW: in-memory cache of per-element decoded arrays so we can mutate colors/pose before building blobs
  const elementCacheRef = useRef<{
    original?: {
      positions: Float32Array
      sh0: Float32Array | null
      opacity: Float32Array | null
      scales: Float32Array | null
      rotations: Float32Array | null
      count: number
    }
    working?: {
      positions: Float32Array
      sh0: Float32Array | null
      opacity: Float32Array | null
      scales: Float32Array | null
      rotations: Float32Array | null
      count: number
    }
  }>({})

  const boundingSphereInitializedRef = useRef(false)

  // LOD configuration (tweakable)
  // We'll use a screen-relative projected radius (NDC units) of the object's bounding sphere
  // NDC radius runs roughly in [0..1] where 1 is half the screen extent; tune these
  const HIGH_NDC = 0.25 // if object's projected radius >= this -> high LOD
  const MEDIUM_NDC = 0.083 // if >= this -> medium LOD
  const FRAME_THROTTLE = 30 // check distance every N frames (throttle to reduce work)
  const SAMPLE_STEP_MEDIUM = 3 // decimation for medium LOD
  const SAMPLE_STEP_LOW = 6 // decimation factor for low LOD (more aggressive)

  // Utility: build a PLY blob from geometry; sampleStep allows decimation
  // Modified to read from the in-memory working cache when available
  const createPLYBlob = (sampleStep = 1) => {
    // Prefer working cache, fall back to decoded geometry
    let positionsAttr: { array: Float32Array; count: number } | undefined
    let sh0Attr: { array: Float32Array } | null = null
    let opacityAttr: { array: Float32Array } | null = null
    let scalesAttr: { array: Float32Array } | null = null
    let rotationsAttr: { array: Float32Array } | null = null

    const cache = elementCacheRef.current.working
    if (cache) {
      positionsAttr = { array: cache.positions, count: cache.count }
      if (cache.sh0) sh0Attr = { array: cache.sh0 }
      if (cache.opacity) opacityAttr = { array: cache.opacity }
      if (cache.scales) scalesAttr = { array: cache.scales }
      if (cache.rotations) rotationsAttr = { array: cache.rotations }
    } else {
      const geom = (state as any).decoded.geometry
      positionsAttr = geom.attributes.position
      sh0Attr = geom.attributes.sh0 ?? null
      opacityAttr = geom.attributes.opacity ?? null
      scalesAttr = geom.attributes.scale ?? null
      rotationsAttr = geom.attributes.rotation ?? null
    }

    if (!positionsAttr) throw new Error('No positions attribute available')

    const positions = positionsAttr
    const sh0 = sh0Attr
    const opacity = opacityAttr
    const scales = scalesAttr
    const rotations = rotationsAttr
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

    // Build PLY header (sh0 stored as three floats per-vertex)
    let header = 'ply\n'
    header += 'format binary_little_endian 1.0\n'
    header += `element vertex ${count}\n`
    header += 'property float x\n'
    header += 'property float y\n'
    header += 'property float z\n'
    // Use 3DGS / shader-compatible names for 0th-order SH coefficients
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
    header += 'property float orig_index\n'
    header += 'end_header\n'

    const headerBytes = new TextEncoder().encode(header)
    // now 15 floats per vertex
    const dataBytes = count * 15 * 4

    const plyBuffer = new Uint8Array(headerBytes.length + dataBytes)
    plyBuffer.set(headerBytes, 0)

    const dataView = new DataView(plyBuffer.buffer, headerBytes.length)
    for (let idx = 0; idx < count; idx++) {
      const i = indices[idx]
      const offset = idx * 15 * 4
      dataView.setFloat32(offset + 0, positions.array[i * 3 + 0], true)
      dataView.setFloat32(offset + 4, positions.array[i * 3 + 1], true)
      dataView.setFloat32(offset + 8, positions.array[i * 3 + 2], true)
      dataView.setFloat32(offset + 12, sh0 ? sh0.array[i * 3 + 0] : 0, true)
      dataView.setFloat32(offset + 16, sh0 ? sh0.array[i * 3 + 1] : 0, true)
      dataView.setFloat32(offset + 20, sh0 ? sh0.array[i * 3 + 2] : 0, true)
      dataView.setFloat32(offset + 24, opacity ? opacity.array[i] : 1, true)
      dataView.setFloat32(offset + 28, scales ? scales.array[i * 3 + 0] : 1, true)
      dataView.setFloat32(offset + 32, scales ? scales.array[i * 3 + 1] : 1, true)
      dataView.setFloat32(offset + 36, scales ? scales.array[i * 3 + 2] : 1, true)
      dataView.setFloat32(offset + 40, rotations ? rotations.array[i * 4 + 0] : 0, true)
      dataView.setFloat32(offset + 44, rotations ? rotations.array[i * 4 + 1] : 0, true)
      dataView.setFloat32(offset + 48, rotations ? rotations.array[i * 4 + 2] : 0, true)
      dataView.setFloat32(offset + 52, rotations ? rotations.array[i * 4 + 3] : 1, true)
      // store original index so we can map back to cache.working later
      dataView.setFloat32(offset + 56, i, true)
    }

    return new Blob([plyBuffer], { type: 'application/octet-stream' })
  }

  const makeDepthRampModifier = (
    splatToView: SplatTransformer,
    minDepth: dyno.DynoVal<'float'>,
    maxDepth: dyno.DynoVal<'float'>,
    stops: [number, number, number, number][],
  ) => {
    const { dynoBlock } = dyno
    const { splitGsplat, combineGsplat, Gsplat } = dyno
    const { normalizedDepth } = dyno
    const { split } = dyno
    const { neg, clamp, mix, sub, div } = dyno
    const { and, greaterThanEqual, lessThanEqual, select } = dyno
    const { dynoConst } = dyno

    return dynoBlock({ gsplat: Gsplat }, { gsplat: Gsplat }, ({ gsplat }) => {
      if (!gsplat) {
        throw new Error('No gsplat input')
      }
      let { center } = splitGsplat(gsplat).outputs
      center = splatToView.apply(center)
      const { z } = split(center).outputs
      let depth = normalizedDepth(neg(z), minDepth, maxDepth)
      depth = clamp(depth, dynoConst('float', 0), dynoConst('float', 1))

      const buildFromStops = (segmentStops: [number, number, number, number][]) => {
        let r: dyno.DynoVal<'float'> = dynoConst('float', segmentStops[0][1])
        let g: dyno.DynoVal<'float'> = dynoConst('float', segmentStops[0][2])
        let b: dyno.DynoVal<'float'> = dynoConst('float', segmentStops[0][3])
        for (let i = 0; i < segmentStops.length - 1; i++) {
          const [t0, r0, g0, b0] = segmentStops[i]
          const [t1, r1, g1, b1] = segmentStops[i + 1]
          const t0c = dynoConst('float', t0)
          const t1c = dynoConst('float', t1)
          const span = dynoConst('float', Math.max(1e-6, t1 - t0))
          const local = clamp(div(sub(depth, t0c), span), dynoConst('float', 0), dynoConst('float', 1))
          const segR = mix(dynoConst('float', r0), dynoConst('float', r1), local)
          const segG = mix(dynoConst('float', g0), dynoConst('float', g1), local)
          const segB = mix(dynoConst('float', b0), dynoConst('float', b1), local)
          const inSeg = and(greaterThanEqual(depth, t0c), lessThanEqual(depth, t1c))
          r = select(inSeg, segR, r)
          g = select(inSeg, segG, g)
          b = select(inSeg, segB, b)
        }
        return { r, g, b }
      }

      const { r, g, b } = buildFromStops(stops)
      return { gsplat: combineGsplat({ gsplat, r, g, b }) }
    })
  }

  const getDepthRangeForCamera = () => {
    try {
      const group = groupRef.current
      const localSphere = boundingSphereRef.current
      if (!group || !localSphere || !camera) {
        return { minDepth: 0, maxDepth: 1 }
      }

      const centerWorld = localSphere.center.clone().applyMatrix4(group.matrixWorld)
      const tmpPos = new THREE.Vector3()
      const tmpQuat = new THREE.Quaternion()
      const tmpScale = new THREE.Vector3()
      group.matrixWorld.decompose(tmpPos, tmpQuat, tmpScale)
      const maxScale = Math.max(tmpScale.x, tmpScale.y, tmpScale.z, 1)
      const worldRadius = localSphere.radius * maxScale

      const camPos = camera.getWorldPosition(new THREE.Vector3())
      const camDir = camera.getWorldDirection(new THREE.Vector3())
      const depthCenter = centerWorld.clone().sub(camPos).dot(camDir)
      let minDepth = depthCenter - worldRadius
      let maxDepth = depthCenter + worldRadius
      const near = typeof camera.near === 'number' ? camera.near : 0.01
      const far = typeof camera.far === 'number' ? camera.far : maxDepth
      minDepth = Math.max(minDepth, near)
      maxDepth = Math.min(maxDepth, far)
      if (minDepth === maxDepth) maxDepth = minDepth + 1e-3
      return { minDepth, maxDepth }
    } catch {
      return { minDepth: 0, maxDepth: 1 }
    }
  }

  const applyDepthColorToMeshes = () => {
    const { minDepth, maxDepth } = getDepthRangeForCamera()
    const updateUniform = (uniform: dyno.DynoFloat<string>, value: number) => {
      uniform.value = value
      uniform.uniform.value = value
    }
    const getActiveColorMap = (): ColormapId => {
      try {
        const vis = (window as any).__begira_visual_override?.[elementId]
        return (vis && (vis.colorMap as ColormapId)) || DEFAULT_DEPTH_COLORMAP
      } catch {
        return DEFAULT_DEPTH_COLORMAP
      }
    }
    const colorMapId = getActiveColorMap()
    const colorMapDef: ColormapDef = COLORMAPS.find((c) => c.id === colorMapId) ?? COLORMAPS[0]
    const ensureModifier = (mesh: SplatMesh) => {
      const data = mesh.userData as any
      if (!data._depthRamp) {
        const safeId = `${elementId}_${mesh.uuid}`.replace(/[^a-zA-Z0-9_]/g, '_')
        const minDepthUniform = dyno.dynoFloat(0, `minDepth_${safeId}`)
        const maxDepthUniform = dyno.dynoFloat(1, `maxDepth_${safeId}`)
        data._depthRamp = {
          minDepth: minDepthUniform,
          maxDepth: maxDepthUniform,
          colorMapId: colorMapDef.id,
          modifier: makeDepthRampModifier(mesh.context.worldToView, minDepthUniform, maxDepthUniform, colorMapDef.stops),
        }
        mesh.enableWorldToView = true
        mesh.worldModifier = data._depthRamp.modifier
        mesh.updateGenerator()
      } else if (data._depthRamp.colorMapId !== colorMapDef.id) {
        data._depthRamp.colorMapId = colorMapDef.id
        data._depthRamp.modifier = makeDepthRampModifier(
          mesh.context.worldToView,
          data._depthRamp.minDepth,
          data._depthRamp.maxDepth,
          colorMapDef.stops,
        )
        mesh.worldModifier = data._depthRamp.modifier
        mesh.updateGenerator()
      }
      return data._depthRamp
    }
    const applyTo = (mesh: SplatMesh | null) => {
      if (!mesh) return
      try {
        const state = ensureModifier(mesh)
        updateUniform(state.minDepth, minDepth)
        updateUniform(state.maxDepth, maxDepth)
      } catch (e) {}
    }
    applyTo(splatHighRef.current)
    applyTo(splatMedRef.current)
    applyTo(splatLowRef.current)
  }

  const clearDepthColorFromMeshes = () => {
    const clear = (mesh: SplatMesh | null) => {
      if (!mesh) return
      try {
        mesh.worldModifier = undefined
        mesh.enableWorldToView = false
        mesh.updateGenerator()
        if (mesh.userData) delete (mesh.userData as any)._depthRamp
      } catch (e) {}
    }
    clear(splatHighRef.current)
    clear(splatMedRef.current)
    clear(splatLowRef.current)
  }

  // Apply a visual override to the in-memory working cache. Supports: 'logged', 'solid', 'height', 'depth'
  const applyVisualOverrideToCache = (mode: string, solidColorHex?: string, colorMapId: ColormapId = DEFAULT_HEIGHT_COLORMAP) => {
    try {
      const decoded = (state as any).decoded
      if (!decoded || !decoded.geometry) return
      // initialize cache from decoded geometry if missing
      if (!elementCacheRef.current.original) {
        const geom = decoded.geometry
        const positions = new Float32Array(geom.attributes.position.array)
        const sh0 = geom.attributes.sh0 ? new Float32Array(geom.attributes.sh0.array) : null
        const opacity = geom.attributes.opacity ? new Float32Array(geom.attributes.opacity.array) : null
        const scales = geom.attributes.scale ? new Float32Array(geom.attributes.scale.array) : null
        const rotations = geom.attributes.rotation ? new Float32Array(geom.attributes.rotation.array) : null
        elementCacheRef.current.original = {
          positions,
          sh0,
          opacity,
          scales,
          rotations,
          count: geom.attributes.position.count,
        }
        // start working copy as a fresh clone
        elementCacheRef.current.working = {
          positions: new Float32Array(positions),
          sh0: sh0 ? new Float32Array(sh0) : null,
          opacity: opacity ? new Float32Array(opacity) : null,
          scales: scales ? new Float32Array(scales) : null,
          rotations: rotations ? new Float32Array(rotations) : null,
          count: geom.attributes.position.count,
        }
      }

      const cache = elementCacheRef.current
      if (!cache.original || !cache.working) return

      // Helper: reset working sh0 to original
      const resetSh0 = () => {
        if (cache.original!.sh0) {
          cache.working!.sh0 = new Float32Array(cache.original!.sh0)
        } else {
          cache.working!.sh0 = null
        }
      }

      if (mode === 'logged') {
        resetSh0()
        return
      }

      if (mode === 'solid') {
        // parse hex color if provided, else fallback to white
        let rgb = [1, 1, 1]
        if (solidColorHex) {
          const s = solidColorHex.replace('#', '')
          if (s.length === 6) {
            rgb = [parseInt(s.slice(0, 2), 16) / 255, parseInt(s.slice(2, 4), 16) / 255, parseInt(s.slice(4, 6), 16) / 255]
          }
        }
        // set sh0 triples to the rgb value (simple approach: set DC triplet to rgb)
        if (!cache.working!.sh0) cache.working!.sh0 = new Float32Array(cache.working!.count * 3)
        for (let i = 0; i < cache.working!.count; i++) {
          cache.working!.sh0[i * 3 + 0] = rgb[0]
          cache.working!.sh0[i * 3 + 1] = rgb[1]
          cache.working!.sh0[i * 3 + 2] = rgb[2]
        }
        return
      }

      if (mode === 'height') {
        // compute height along world up axis (use camera.up which is set based on coordinate convention)
        const group = groupRef.current
        if (!group) return
        const geomCount = cache.working!.count
        const wPos = new THREE.Vector3()
        const worldUp = camera?.up ? camera.up.clone().normalize() : new THREE.Vector3(0, 0, 1)
        // find min and max along worldUp
        let min = Number.POSITIVE_INFINITY
        let max = Number.NEGATIVE_INFINITY
        const tmpVec = new THREE.Vector3()
        for (let i = 0; i < geomCount; i++) {
          wPos.set(cache.working!.positions[i * 3 + 0], cache.working!.positions[i * 3 + 1], cache.working!.positions[i * 3 + 2])
          // transform by group's world matrix
          wPos.applyMatrix4(group.matrixWorld)
          const v = wPos.dot(worldUp)
          if (v < min) min = v
          if (v > max) max = v
        }
        const range = Math.max(1e-6, max - min)
        if (!cache.working!.sh0) cache.working!.sh0 = new Float32Array(geomCount * 3)
        for (let i = 0; i < geomCount; i++) {
          tmpVec.set(cache.working!.positions[i * 3 + 0], cache.working!.positions[i * 3 + 1], cache.working!.positions[i * 3 + 2])
          tmpVec.applyMatrix4(group.matrixWorld)
          const v = tmpVec.dot(worldUp)
          const t = Math.min(1, Math.max(0, (v - min) / range))
          const [r, g, b] = sampleColormap(colorMapId, t)
          cache.working!.sh0[i * 3 + 0] = r
          cache.working!.sh0[i * 3 + 1] = g
          cache.working!.sh0[i * 3 + 2] = b
        }
        return
      }

      if (mode === 'depth') {
        applyDepthColorToMeshes()
        return
      }
    } catch (e) {
      // ignore failures
    }
  }

  // Helper: rebuild blob URLs from the current working cache and recreate SplatMesh instances
  // Non-destructive full rebuild: create new meshes invisible and swap them in after they finish loading
  const rebuildMeshesFromCache = async () => {
    if (loadingRef.current) return
    loadingRef.current = true
    try {
      const parent = groupRef.current
      if (!parent) return

      // create blobs
      const blobHigh = createPLYBlob(1)
      const blobMed = createPLYBlob(SAMPLE_STEP_MEDIUM)
      const blobLow = createPLYBlob(SAMPLE_STEP_LOW)

      const highUrl = URL.createObjectURL(blobHigh)
      const medUrl = URL.createObjectURL(blobMed)
      const lowUrl = URL.createObjectURL(blobLow)

      const makeSplatPromise = (url: string, opts: any) =>
        new Promise<SplatMesh>((resolve, reject) => {
          try {
            const m = new SplatMesh({
              url,
              ...opts,
              onLoad: () => resolve(m),
              onError: (err: any) => reject(err),
            })
            m.visible = false
            parent.add(m)
          } catch (e) {
            reject(e)
          }
        })

      // Build new meshes concurrently
      const pHigh = makeSplatPromise(highUrl, { splatScale: 1.0, alphaTest: 0.08, renderMode: 'default' })
      const pMed = makeSplatPromise(medUrl, { splatScale: 1.2, alphaTest: 0.11, renderMode: 'performance' })
      const pLow = makeSplatPromise(lowUrl, { splatScale: 2.0, alphaTest: 0.14, renderMode: 'performance' })

      let newHigh: SplatMesh | null = null
      let newMed: SplatMesh | null = null
      let newLow: SplatMesh | null = null

      try {
        ;[newHigh, newMed, newLow] = await Promise.all([pHigh, pMed, pLow])
      } catch (e) {
        // if any failed, remove any partial new meshes and fall back to previous state
        try {
          if (newHigh) { parent.remove(newHigh); try { newHigh.dispose() } catch {} }
          if (newMed) { parent.remove(newMed); try { newMed.dispose() } catch {} }
          if (newLow) { parent.remove(newLow); try { newLow.dispose() } catch {} }
        } catch {}
        // cleanup created blob urls
        try { URL.revokeObjectURL(highUrl) } catch {}
        try { URL.revokeObjectURL(medUrl) } catch {}
        try { URL.revokeObjectURL(lowUrl) } catch {}
        throw e
      }

      // Swap in: make each new mesh visible if the old one was visible, then remove and dispose the old
      try {
        const oldHigh = splatHighRef.current
        const oldMed = splatMedRef.current
        const oldLow = splatLowRef.current

        if (newHigh) {
          newHigh.visible = !!oldHigh && oldHigh.visible
          if (oldHigh && parent) {
            parent.remove(oldHigh)
            try { oldHigh.dispose() } catch {}
          }
          splatHighRef.current = newHigh
        }

        if (newMed) {
          newMed.visible = !!oldMed && oldMed.visible
          if (oldMed && parent) {
            parent.remove(oldMed)
            try { oldMed.dispose() } catch {}
          }
          splatMedRef.current = newMed
        }

        if (newLow) {
          newLow.visible = !!oldLow && oldLow.visible
          if (oldLow && parent) {
            parent.remove(oldLow)
            try { oldLow.dispose() } catch {}
          }
          splatLowRef.current = newLow
        }

        try {
          const vis = (window as any).__begira_visual_override?.[elementId]
          if (vis && vis.colorMode === 'depth') applyDepthColorToMeshes()
        } catch {}

        // revoke previous blob urls after successful swap
        try { if (blobUrlRef.current.high) URL.revokeObjectURL(blobUrlRef.current.high) } catch {}
        try { if ((blobUrlRef.current as any).med && (blobUrlRef.current as any).med !== blobUrlRef.current.high) URL.revokeObjectURL((blobUrlRef.current as any).med) } catch {}
        try { if (blobUrlRef.current.low && blobUrlRef.current.low !== blobUrlRef.current.high) URL.revokeObjectURL(blobUrlRef.current.low) } catch {}

        blobUrlRef.current.high = highUrl
        ;(blobUrlRef.current as any).med = medUrl
        blobUrlRef.current.low = lowUrl
      } catch (e) {
        // if swap failed, clean up newly created meshes
        try {
          if (newHigh) { parent.remove(newHigh); try { newHigh.dispose() } catch {} }
          if (newMed) { parent.remove(newMed); try { newMed.dispose() } catch {} }
          if (newLow) { parent.remove(newLow); try { newLow.dispose() } catch {} }
        } catch {}
        throw e
      }
    } catch (e) {
      console.error('[Gaussians] rebuild failed:', e)
    } finally {
      loadingRef.current = false
    }
  }

  // Rebuild only the low LOD (cheaper) in a non-destructive way. Used while the camera is moving to reduce flicker.
  const buildingLowRef = useRef(false)
  const rebuildLowFromCache = async () => {
    if (buildingLowRef.current) return
    buildingLowRef.current = true
    try {
      const parent = groupRef.current
      if (!parent) return
      const blobLow = createPLYBlob(SAMPLE_STEP_LOW)
      const lowUrl = URL.createObjectURL(blobLow)

      const makeSplatPromise = (url: string, opts: any) =>
        new Promise<SplatMesh>((resolve, reject) => {
          try {
            const m = new SplatMesh({
              url,
              ...opts,
              onLoad: () => resolve(m),
              onError: (err: any) => reject(err),
            })
            m.visible = false
            parent.add(m)
          } catch (e) { reject(e) }
        })

      let newLow: SplatMesh | null = null
      try {
        newLow = await makeSplatPromise(lowUrl, { splatScale: 2.0, alphaTest: 0.14, renderMode: 'performance' })
      } catch (e) {
        try { if (newLow) { parent.remove(newLow); try { newLow.dispose() } catch {} } } catch {}
        try { URL.revokeObjectURL(lowUrl) } catch {}
        throw e
      }

      try {
        const oldLow = splatLowRef.current
        if (newLow) {
          newLow.visible = !!oldLow && oldLow.visible
          if (oldLow && parent) {
            parent.remove(oldLow)
            try { oldLow.dispose() } catch {}
          }
          splatLowRef.current = newLow
        }
        try { if (blobUrlRef.current.low && blobUrlRef.current.low !== blobUrlRef.current.high) URL.revokeObjectURL(blobUrlRef.current.low) } catch {}
        blobUrlRef.current.low = lowUrl

        try {
          const vis = (window as any).__begira_visual_override?.[elementId]
          if (vis && vis.colorMode === 'depth') applyDepthColorToMeshes()
        } catch {}
      } catch (e) {
        try { if (newLow) { parent.remove(newLow); try { newLow.dispose() } catch {} } } catch {}
        throw e
      }
    } catch (e) {
      console.error('[Gaussians] rebuild low failed:', e)
    } finally {
      buildingLowRef.current = false
    }
  }

  // Create both LOD blobs and SplatMesh instances once when ready
  useEffect(() => {
    if (state.status !== 'ready' || !groupRef.current || loadingRef.current) return
    loadingRef.current = true

    const setup = async () => {
      try {
        // Initialize cache from decoded geometry so future overrides operate on it
        try {
          const decoded = (state as any).decoded
          if (decoded && decoded.geometry && !elementCacheRef.current.original) {
            const geom = decoded.geometry
            const positions = new Float32Array(geom.attributes.position.array)
            const sh0 = geom.attributes.sh0 ? new Float32Array(geom.attributes.sh0.array) : null
            const opacity = geom.attributes.opacity ? new Float32Array(geom.attributes.opacity.array) : null
            const scales = geom.attributes.scale ? new Float32Array(geom.attributes.scale.array) : null
            const rotations = geom.attributes.rotation ? new Float32Array(geom.attributes.rotation.array) : null
            elementCacheRef.current.original = {
              positions,
              sh0,
              opacity,
              scales,
              rotations,
              count: geom.attributes.position.count,
            }
            elementCacheRef.current.working = {
              positions: new Float32Array(positions),
              sh0: sh0 ? new Float32Array(sh0) : null,
              opacity: opacity ? new Float32Array(opacity) : null,
              scales: scales ? new Float32Array(scales) : null,
              rotations: rotations ? new Float32Array(rotations) : null,
              count: geom.attributes.position.count,
            }
          }
        } catch (e) {
          // ignore
        }

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
        if (!boundingSphereRef.current && elementCacheRef.current.working) {
          try {
            const posAttr = elementCacheRef.current.working.positions
            const n = elementCacheRef.current.working.count
            let cx = 0
            let cy = 0
            let cz = 0
            for (let i = 0; i < n; i++) {
              cx += posAttr[i * 3 + 0]
              cy += posAttr[i * 3 + 1]
              cz += posAttr[i * 3 + 2]
            }
            cx /= n
            cy /= n
            cz /= n
            let maxSq = 0
            for (let i = 0; i < n; i++) {
              const dx = posAttr[i * 3 + 0] - cx
              const dy = posAttr[i * 3 + 1] - cy
              const dz = posAttr[i * 3 + 2] - cz
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
              // Try to apply any solid tint currently requested (backwards-compat fallback)
              try {
                const anyWin = typeof window !== 'undefined' ? (window as any) : ({} as any)
                const vis = anyWin.__begira_visual_override && anyWin.__begira_visual_override[elementId]
                if (vis && vis.colorMode === 'solid' && vis.solidColor) {
                  const rgb = vis.solidColor as [number, number, number]
                  const children = groupRef.current?.children ?? []
                  // Apply to each SplatMesh and descendants
                  children.forEach((c: any) => {
                    const applyTintToObject = (obj: any, rgb: [number, number, number]) => {
                      if (!obj) return
                      // If SplatMesh exposes a public API to recolor, prefer it
                      try {
                        if (typeof obj.setTint === 'function') {
                          obj.setTint(new THREE.Color(rgb[0], rgb[1], rgb[2]))
                          return
                        }
                      } catch {}

                      // Walk children and attempt to set uniforms/material colors
                      const tryApply = (m: any) => {
                        try {
                          if (!m) return
                          if (m.material) {
                            if (m.material.uniforms && m.material.uniforms.uTint) {
                              m.material.uniforms.uTint.value.setRGB(rgb[0], rgb[1], rgb[2])
                              m.material.uniforms.uTint.needsUpdate = true
                            } else if (m.material.color) {
                              m.material.color.setRGB(rgb[0], rgb[1], rgb[2])
                              m.material.needsUpdate = true
                            }
                          }
                        } catch (e) {
                          // ignore
                        }
                      }

                      // If object itself is a mesh-like with children, traverse
                      if (obj.children && obj.children.length > 0) {
                        obj.children.forEach((c: any) => {
                          tryApply(c)
                          // some children may have deeper nesting
                          if (c.children && c.children.length > 0) c.children.forEach((cc: any) => tryApply(cc))
                        })
                      } else {
                        tryApply(obj)
                      }
                    }

                    applyTintToObject(c, rgb)
                  })
                }
              } catch (e) {}
              try {
                const anyWin = typeof window !== 'undefined' ? (window as any) : ({} as any)
                const vis = anyWin.__begira_visual_override && anyWin.__begira_visual_override[elementId]
                if (vis && vis.colorMode === 'depth') applyDepthColorToMeshes()
              } catch {}
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
            onLoad: () => {
              // apply tint if present
              try {
                const anyWin = typeof window !== 'undefined' ? (window as any) : ({} as any)
                const vis = anyWin.__begira_visual_override && anyWin.__begira_visual_override[elementId]
                if (vis && vis.colorMode === 'solid' && vis.solidColor) {
                  const rgb = vis.solidColor as [number, number, number]
                  const children = groupRef.current?.children ?? []
                  // Apply to each SplatMesh and descendants
                  children.forEach((c: any) => {
                    const applyTintToObject = (obj: any, rgb: [number, number, number]) => {
                      if (!obj) return
                      // If SplatMesh exposes a public API to recolor, prefer it
                      try {
                        if (typeof obj.setTint === 'function') {
                          obj.setTint(new THREE.Color(rgb[0], rgb[1], rgb[2]))
                          return
                        }
                      } catch {}

                      // Walk children and attempt to set uniforms/material colors
                      const tryApply = (m: any) => {
                        try {
                          if (!m) return
                          if (m.material) {
                            if (m.material.uniforms && m.material.uniforms.uTint) {
                              m.material.uniforms.uTint.value.setRGB(rgb[0], rgb[1], rgb[2])
                              m.material.uniforms.uTint.needsUpdate = true
                            } else if (m.material.color) {
                              m.material.color.setRGB(rgb[0], rgb[1], rgb[2])
                              m.material.needsUpdate = true
                            }
                          }
                        } catch (e) {
                          // ignore
                        }
                      }

                      // If object itself is a mesh-like with children, traverse
                      if (obj.children && obj.children.length > 0) {
                        obj.children.forEach((c: any) => {
                          tryApply(c)
                          // some children may have deeper nesting
                          if (c.children && c.children.length > 0) c.children.forEach((cc: any) => tryApply(cc))
                        })
                      } else {
                        tryApply(obj)
                      }
                    }

                    applyTintToObject(c, rgb)
                  })
                }
              } catch (e) {}
              try {
                const anyWin = typeof window !== 'undefined' ? (window as any) : ({} as any)
                const vis = anyWin.__begira_visual_override && anyWin.__begira_visual_override[elementId]
                if (vis && vis.colorMode === 'depth') applyDepthColorToMeshes()
              } catch {}
            },
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
            onLoad: () => {
              // apply tint if present
              try {
                const anyWin = typeof window !== 'undefined' ? (window as any) : ({} as any)
                const vis = anyWin.__begira_visual_override && anyWin.__begira_visual_override[elementId]
                if (vis && vis.colorMode === 'solid' && vis.solidColor) {
                  const rgb = vis.solidColor as [number, number, number]
                  const children = groupRef.current?.children ?? []
                  // Apply to each SplatMesh and descendants
                  children.forEach((c: any) => {
                    const applyTintToObject = (obj: any, rgb: [number, number, number]) => {
                      if (!obj) return
                      // If SplatMesh exposes a public API to recolor, prefer it
                      try {
                        if (typeof obj.setTint === 'function') {
                          obj.setTint(new THREE.Color(rgb[0], rgb[1], rgb[2]))
                          return
                        }
                      } catch {}

                      // Walk children and attempt to set uniforms/material colors
                      const tryApply = (m: any) => {
                        try {
                          if (!m) return
                          if (m.material) {
                            if (m.material.uniforms && m.material.uniforms.uTint) {
                              m.material.uniforms.uTint.value.setRGB(rgb[0], rgb[1], rgb[2])
                              m.material.uniforms.uTint.needsUpdate = true
                            } else if (m.material.color) {
                              m.material.color.setRGB(rgb[0], rgb[1], rgb[2])
                              m.material.needsUpdate = true
                            }
                          }
                        } catch (e) {
                          // ignore
                        }
                      }

                      // If object itself is a mesh-like with children, traverse
                      if (obj.children && obj.children.length > 0) {
                        obj.children.forEach((c: any) => {
                          tryApply(c)
                          // some children may have deeper nesting
                          if (c.children && c.children.length > 0) c.children.forEach((cc: any) => tryApply(cc))
                        })
                      } else {
                        tryApply(obj)
                      }
                    }

                    applyTintToObject(c, rgb)
                  })
                }
              } catch (e) {}
              try {
                const anyWin = typeof window !== 'undefined' ? (window as any) : ({} as any)
                const vis = anyWin.__begira_visual_override && anyWin.__begira_visual_override[elementId]
                if (vis && vis.colorMode === 'depth') applyDepthColorToMeshes()
              } catch {}
            },
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

    // Listen for visual override changes and apply them by rebuilding the cached blobs/meshes
    const onVisOverrideChanged = (ev: any) => {
      try {
        if (!ev || !ev.detail || ev.detail.id !== elementId) return
        const anyWin = window as any
        const vis = anyWin.__begira_visual_override && anyWin.__begira_visual_override[elementId]
        if (!vis || !vis.colorMode) {
          // reset to logged
          clearDepthColorFromMeshes()
          applyVisualOverrideToCache('logged')
          // rebuild meshes
          void rebuildMeshesFromCache()
          return
        }
        if (vis.colorMode === 'solid') {
          clearDepthColorFromMeshes()
          const hex = vis.solidColor ? '#' + [vis.solidColor[0], vis.solidColor[1], vis.solidColor[2]].map((v: number) => Math.round(v * 255).toString(16).padStart(2, '0')).join('') : undefined
          applyVisualOverrideToCache('solid', hex)
        } else if (vis.colorMode === 'height') {
          clearDepthColorFromMeshes()
          const cmap = (vis.colorMap as ColormapId) || DEFAULT_HEIGHT_COLORMAP
          applyVisualOverrideToCache('height', undefined, cmap)
        } else if (vis.colorMode === 'depth') {
          applyDepthColorToMeshes()
          return
        } else {
          clearDepthColorFromMeshes()
          applyVisualOverrideToCache('logged')
        }
        void rebuildMeshesFromCache()
      } catch {}
    }
    try {
      window.addEventListener('begira_visual_override_changed', onVisOverrideChanged)
    } catch {}

    return () => {
      try {
        window.removeEventListener('begira_visual_override_changed', onVisOverrideChanged)
      } catch {}

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

  // Throttle updates for depth-colormap mode so it reacts to camera moves
  const lastDepthUpdateRef = useRef<number>(0)
  const DEPTH_UPDATE_MS = 33

  useFrame(() => {
    const group = groupRef.current
    if (!group || !camera) return

    const visibilityMap = (window as any).__begira_visibility || {}
    const isVisible = visibilityMap[elementId] !== false
    group.visible = isVisible
    if (!isVisible) {
      if (splatHighRef.current) splatHighRef.current.visible = false
      if (splatMedRef.current) splatMedRef.current.visible = false
      if (splatLowRef.current) splatLowRef.current.visible = false
      return
    }

    // Update depth ramp uniforms while in depth mode.
    try {
      const vis = (window as any).__begira_visual_override?.[elementId]
      if (vis && vis.colorMode === 'depth') {
        const now = performance.now()
        if (now - lastDepthUpdateRef.current > DEPTH_UPDATE_MS) {
          lastDepthUpdateRef.current = now
          applyDepthColorToMeshes()
        }
      }
    } catch (e) {
      // ignore
    }

    frameCounter.current = (frameCounter.current + 1) % FRAME_THROTTLE
    if (frameCounter.current !== 0) return

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
