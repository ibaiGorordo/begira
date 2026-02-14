import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

export const CAMERA_GIZMO_OVERLAY_LAYER = 1
const CAMERA_GIZMO_WORLD_SCALE = 1.6

export type CameraVisual = {
  id: string
  position?: [number, number, number]
  rotation?: [number, number, number, number]
  fov?: number
  near?: number
  far?: number
  visible?: boolean
}

function CameraGizmo({
  camera,
  selected,
  onSelect,
  onFocus,
  onRegisterObject,
}: {
  camera: CameraVisual
  selected: boolean
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
  onRegisterObject?: (id: string, obj: THREE.Object3D | null) => void
}) {
  const cameraId = camera.id
  const groupRef = useRef<THREE.Group | null>(null)
  const lastLocalPoseRef = useRef<string | null>(null)
  const frustumColor = selected ? '#9dd1ff' : '#8ec5f7'

  const frustumGeom = useMemo(() => {
    const fovDeg = Math.min(175.0, Math.max(5.0, camera.fov ?? 60.0))
    const aspect = 16 / 9
    const depth = 0.55
    const halfY = Math.tan(THREE.MathUtils.degToRad(fovDeg * 0.5)) * depth
    const halfX = halfY * aspect

    const o = new THREE.Vector3(0, 0, 0)
    const tl = new THREE.Vector3(-halfX, halfY, -depth)
    const tr = new THREE.Vector3(halfX, halfY, -depth)
    const br = new THREE.Vector3(halfX, -halfY, -depth)
    const bl = new THREE.Vector3(-halfX, -halfY, -depth)
    const topTip = new THREE.Vector3(0, halfY * 1.35, -depth)

    const segments = [
      o, tl,
      o, tr,
      o, br,
      o, bl,
      tl, tr,
      tr, br,
      br, bl,
      bl, tl,
      tl, topTip,
      tr, topTip,
    ]

    const out = new Float32Array(segments.length * 3)
    for (let i = 0; i < segments.length; i++) {
      out[i * 3 + 0] = segments[i].x
      out[i * 3 + 1] = segments[i].y
      out[i * 3 + 2] = segments[i].z
    }

    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(out, 3))
    return g
  }, [camera.fov])

  useEffect(() => {
    if (!onRegisterObject) return
    if (!groupRef.current) return
    onRegisterObject(cameraId, groupRef.current)
    return () => onRegisterObject(cameraId, null)
  }, [cameraId, onRegisterObject])

  useEffect(() => {
    if (!groupRef.current) return
    if ((window as any).__begira_local_pose?.[cameraId]) return
    const pos = camera.position ?? [0, 0, 0]
    const rot = camera.rotation ?? [0, 0, 0, 1]
    groupRef.current.position.set(pos[0], pos[1], pos[2])
    groupRef.current.quaternion.set(rot[0], rot[1], rot[2], rot[3]).normalize()
  }, [camera.position, camera.rotation, cameraId])

  useEffect(() => {
    if (!groupRef.current) return
    groupRef.current.traverse((obj) => {
      obj.layers.set(CAMERA_GIZMO_OVERLAY_LAYER)
    })
    // Keep a stable world-space size so gizmos don't grow/shrink with viewport zoom.
    groupRef.current.scale.setScalar(CAMERA_GIZMO_WORLD_SCALE)
  }, [])

  useFrame(() => {
    try {
      const local = (window as any).__begira_local_pose?.[cameraId]
      if (local && local.position && local.rotation && groupRef.current) {
        const key = `${local.position.join(',')}|${local.rotation.join(',')}`
        if (lastLocalPoseRef.current !== key) {
          lastLocalPoseRef.current = key
          groupRef.current.position.set(local.position[0], local.position[1], local.position[2])
          groupRef.current.quaternion.set(local.rotation[0], local.rotation[1], local.rotation[2], local.rotation[3]).normalize()
        }
      }
    } catch {}
  })

  useEffect(() => {
    return () => {
      frustumGeom.dispose()
    }
  }, [frustumGeom])

  if (camera.visible === false) {
    return null
  }

  return (
    <group
      ref={groupRef}
      renderOrder={1000000}
      frustumCulled={false}
      onClick={(e) => {
        e.stopPropagation()
        onSelect(cameraId)
        if (e.detail === 2) onFocus(cameraId)
      }}
    >
      <lineSegments geometry={frustumGeom} frustumCulled={false} renderOrder={1000000}>
        <lineBasicMaterial
          color={frustumColor}
          depthTest={false}
          depthWrite={false}
          transparent
          opacity={1}
          toneMapped={false}
        />
      </lineSegments>
    </group>
  )
}

export default function CameraScene({
  cameras,
  selectedId,
  onSelect,
  onFocus,
  onRegisterObject,
}: {
  cameras: CameraVisual[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
  onRegisterObject?: (id: string, obj: THREE.Object3D | null) => void
}) {
  return (
    <>
      {cameras.map((camera) => (
        <CameraGizmo
          key={camera.id}
          camera={camera}
          selected={camera.id === selectedId}
          onSelect={onSelect}
          onFocus={onFocus}
          onRegisterObject={onRegisterObject}
        />
      ))}
    </>
  )
}
