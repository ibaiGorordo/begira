import * as THREE from 'three'
import { PointCloudElementMeta } from '../shared/api'

export type DecodedPointCloud = {
  geometry: THREE.BufferGeometry
  bounds: { min: THREE.Vector3; max: THREE.Vector3 }
  hasColor: boolean
}

export function decodePointCloud(meta: PointCloudElementMeta, buf: ArrayBuffer): DecodedPointCloud {
  if (meta.endianness !== 'little') {
    // JS typed arrays are native endian; for now we only support little.
    throw new Error(`Unsupported endianness: ${meta.endianness}`)
  }

  const { pointCount, bytesPerPoint } = meta
  const expectedBytes = pointCount * bytesPerPoint
  if (buf.byteLength !== expectedBytes) {
    throw new Error(`Unexpected payload size: got ${buf.byteLength}, expected ${expectedBytes}`)
  }

  const hasColor = Boolean(meta.schema.color)

  const positions = new Float32Array(pointCount * 3)
  const colors = hasColor ? new Uint8Array(pointCount * 3) : null

  // Fast path: positions-only payload is already tightly packed as float32.
  if (!hasColor && bytesPerPoint === 12) {
    positions.set(new Float32Array(buf))
  } else if (hasColor && bytesPerPoint === 15) {
    // Hot path: XYZ(float32) + RGB(uint8) interleaved.
    // Use DataView for float32 (endianness-safe) and plain byte copies for color.
    const dv = new DataView(buf)
    const bytes = new Uint8Array(buf)

    // hasColor implies colors is allocated in this branch
    for (let i = 0; i < pointCount; i++) {
      const base = i * 15
      const dst = i * 3
      positions[dst] = dv.getFloat32(base + 0, true)
      positions[dst + 1] = dv.getFloat32(base + 4, true)
      positions[dst + 2] = dv.getFloat32(base + 8, true)

      // RGB at 12..14
      colors![dst] = bytes[base + 12]
      colors![dst + 1] = bytes[base + 13]
      colors![dst + 2] = bytes[base + 14]
    }
  } else {
    // Generic slower path for any future schema/stride.
    const dv = new DataView(buf)
    const bytes = new Uint8Array(buf)

    for (let i = 0; i < pointCount; i++) {
      const base = i * bytesPerPoint
      const dst = i * 3
      positions[dst] = dv.getFloat32(base + 0, true)
      positions[dst + 1] = dv.getFloat32(base + 4, true)
      positions[dst + 2] = dv.getFloat32(base + 8, true)

      if (colors) {
        colors[dst] = bytes[base + 12]
        colors[dst + 1] = bytes[base + 13]
        colors[dst + 2] = bytes[base + 14]
      }
    }
  }

  function srgbToLinear(u: number) {
    // u in [0,1]
    return u <= 0.04045 ? u / 12.92 : Math.pow((u + 0.055) / 1.055, 2.4)
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))

  if (colors) {
    // colorsUint8: Uint8Array length = N*3
    const out = new Float32Array(colors.length)
    for (let i = 0; i < colors.length; i++) {
      const u = colors[i] / 255
      out[i] = srgbToLinear(u)
    }

    geometry.setAttribute('color', new THREE.BufferAttribute(out, 3))
  }

  geometry.computeBoundingSphere()

  const min = new THREE.Vector3(...meta.bounds.min)
  const max = new THREE.Vector3(...meta.bounds.max)

  return {
    geometry,
    bounds: { min, max },
    hasColor,
  }
}
