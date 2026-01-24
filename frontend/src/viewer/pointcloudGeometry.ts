import * as THREE from 'three'
import { PointCloudMeta } from './api'

export type DecodedPointCloud = {
  geometry: THREE.BufferGeometry
  bounds: { min: THREE.Vector3; max: THREE.Vector3 }
  hasColor: boolean
}

export function decodePointCloud(meta: PointCloudMeta, buf: ArrayBuffer): DecodedPointCloud {
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

  // Interleaved layout: [XYZ float32][RGB uint8] per point.
  const positions = new Float32Array(pointCount * 3)
  let colors: Uint8Array | null = null
  if (hasColor) colors = new Uint8Array(pointCount * 3)

  const dv = new DataView(buf)
  const stride = bytesPerPoint

  for (let i = 0; i < pointCount; i++) {
    const base = i * stride
    const pOff = i * 3

    positions[pOff + 0] = dv.getFloat32(base + 0, true)
    positions[pOff + 1] = dv.getFloat32(base + 4, true)
    positions[pOff + 2] = dv.getFloat32(base + 8, true)

    if (colors) {
      const cOff = pOff
      colors[cOff + 0] = dv.getUint8(base + 12)
      colors[cOff + 1] = dv.getUint8(base + 13)
      colors[cOff + 2] = dv.getUint8(base + 14)
    }
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  if (colors) geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3, true))

  geometry.computeBoundingSphere()

  const min = new THREE.Vector3(...meta.bounds.min)
  const max = new THREE.Vector3(...meta.bounds.max)

  return {
    geometry,
    bounds: { min, max },
    hasColor,
  }
}

