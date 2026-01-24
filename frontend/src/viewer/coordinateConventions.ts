import * as THREE from 'three'

export type CoordinateConventionId = 'rh-y-up' | 'rh-z-up'

export type CoordinateConvention = {
  id: CoordinateConventionId
  /** Display-space up vector. */
  up: THREE.Vector3
  /** Rotation that maps points authored in the app's "world" coordinates into display coordinates. */
  worldToView: THREE.Quaternion
  /** Ground plane normal in display space. Useful for aligning helpers like grids. */
  groundNormal: THREE.Vector3
}

const _vUpY = new THREE.Vector3(0, 1, 0)
const _vUpZ = new THREE.Vector3(0, 0, 1)

function quatFromBasis(viewForward: THREE.Vector3, viewUp: THREE.Vector3): THREE.Quaternion {
  // Build a right-handed orthonormal basis in view space.
  // forward = viewForward, up = viewUp, right = forward x up.
  const f = viewForward.clone().normalize()
  const u0 = viewUp.clone().normalize()

  // Ensure up isn't parallel to forward.
  if (Math.abs(f.dot(u0)) > 0.999) {
    // Pick a different up seed.
    u0.set(0, 0, 1)
    if (Math.abs(f.dot(u0)) > 0.999) u0.set(0, 1, 0)
  }

  const r = new THREE.Vector3().crossVectors(f, u0).normalize()
  const u = new THREE.Vector3().crossVectors(r, f).normalize()

  // Construct rotation matrix with columns = [right, up, forward].
  const m = new THREE.Matrix4()
  m.makeBasis(r, u, f)

  return new THREE.Quaternion().setFromRotationMatrix(m)
}

export const COORDINATE_CONVENTIONS: Record<CoordinateConventionId, CoordinateConvention> = {
  'rh-y-up': {
    id: 'rh-y-up',
    up: _vUpY.clone(),
    worldToView: new THREE.Quaternion(),     // identity
    groundNormal: _vUpY.clone(),
  },

  'rh-z-up': {
    id: 'rh-z-up',
    up: _vUpZ.clone(),
    // Rotate authored Z-up world into Three's Y-up view space:
    worldToView: new THREE.Quaternion().setFromEuler(
      new THREE.Euler(-Math.PI / 2, 0, 0, 'XYZ')
    ),
    groundNormal: _vUpZ.clone(),
  },
}

export function parseCoordinateConventionFromUrl(search: string): CoordinateConventionId | null {
  const p = new URLSearchParams(search)

  const up = p.get('up')
  if (up === 'y') return 'rh-y-up'
  if (up === 'z') return 'rh-z-up'

  const c = p.get('convention')
  if (c === 'rh-y-up' || c === 'rh-z-up') return c

  return null
}

export function getCoordinateConvention(id: CoordinateConventionId): CoordinateConvention {
  return COORDINATE_CONVENTIONS[id]
}

/**
 * Helper that creates a unit vector with the same direction as `up`, but guaranteed normalized.
 */
export function normalizedUp(up: THREE.Vector3): THREE.Vector3 {
  const out = up.clone()
  if (out.lengthSq() === 0) return _vUpY.clone()
  return out.normalize()
}
