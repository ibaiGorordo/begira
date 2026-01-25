import * as THREE from 'three'
import { GaussianSplatElementMeta } from './api'

export type DecodedGaussians = {
  geometry: THREE.BufferGeometry
  bounds: { min: THREE.Vector3; max: THREE.Vector3 }
  count: number
}

export function decodeGaussians(meta: GaussianSplatElementMeta, buf: ArrayBuffer): DecodedGaussians {
  if (meta.endianness !== 'little') {
    throw new Error(`Unsupported endianness: ${meta.endianness}`)
  }

  const { count, bytesPerGaussian } = meta
  const expectedBytes = count * bytesPerGaussian
  if (buf.byteLength !== expectedBytes) {
    throw new Error(`Unexpected payload size: got ${buf.byteLength}, expected ${expectedBytes}`)
  }

  const positions = new Float32Array(count * 3)
  const sh0 = new Float32Array(count * 3)
  const opacity = new Float32Array(count)
  const scales = new Float32Array(count * 3)
  const rotations = new Float32Array(count * 4)

  // 14 floats interleaved
  const f32 = new Float32Array(buf, 0, buf.byteLength / 4)
  for (let i = 0; i < count; i++) {
    const base = i * 14
    const dst3 = i * 3
    const dst4 = i * 4

    if (base + 13 >= f32.length) break

    // positions
    positions[dst3] = f32[base + 0]
    positions[dst3 + 1] = f32[base + 1]
    positions[dst3 + 2] = f32[base + 2]

    // sh0
    sh0[dst3] = f32[base + 3]
    sh0[dst3 + 1] = f32[base + 4]
    sh0[dst3 + 2] = f32[base + 5]

    // opacity
    opacity[i] = f32[base + 6]

    // scale
    scales[dst3] = f32[base + 7]
    scales[dst3 + 1] = f32[base + 8]
    scales[dst3 + 2] = f32[base + 9]

    // rotation
    rotations[dst4] = f32[base + 10]
    rotations[dst4 + 1] = f32[base + 11]
    rotations[dst4 + 2] = f32[base + 12]
    rotations[dst4 + 3] = f32[base + 13]
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('sh0', new THREE.BufferAttribute(sh0, 3))
  geometry.setAttribute('opacity', new THREE.BufferAttribute(opacity, 1))
  geometry.setAttribute('scale', new THREE.BufferAttribute(scales, 3))
  geometry.setAttribute('rotation', new THREE.BufferAttribute(rotations, 4))

  geometry.computeBoundingSphere()

  const min = new THREE.Vector3(...meta.bounds.min)
  const max = new THREE.Vector3(...meta.bounds.max)

  return {
    geometry,
    bounds: { min, max },
    count,
  }
}
