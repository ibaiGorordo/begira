import { useEffect, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { useCamera } from './useCamera'

function CameraGizmo({
  cameraId,
  selected,
  onSelect,
  onFocus,
  onRegisterObject,
}: {
  cameraId: string
  selected: boolean
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
  onRegisterObject?: (id: string, obj: THREE.Object3D | null) => void
}) {
  const state = useCamera(cameraId)
  const meta = state.status === 'ready' ? state.meta : null
  const groupRef = useRef<THREE.Group | null>(null)
  const camRef = useRef<THREE.PerspectiveCamera | null>(null)
  const helperRef = useRef<THREE.CameraHelper | null>(null)
  const lastLocalPoseRef = useRef<string | null>(null)

  useEffect(() => {
    if (!onRegisterObject) return
    onRegisterObject(cameraId, groupRef.current)
    return () => onRegisterObject(cameraId, null)
  }, [cameraId, onRegisterObject])

  useEffect(() => {
    if (!meta || !groupRef.current) return
    if ((window as any).__begira_local_pose?.[cameraId]) return
    const pos = meta.position ?? [0, 0, 0]
    const rot = meta.rotation ?? [0, 0, 0, 1]
    groupRef.current.position.set(pos[0], pos[1], pos[2])
    groupRef.current.quaternion.set(rot[0], rot[1], rot[2], rot[3]).normalize()
  }, [meta])

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
    if (!camRef.current || !meta) return
    camRef.current.fov = meta.fov
    camRef.current.near = meta.near
    camRef.current.far = meta.far
    camRef.current.updateProjectionMatrix()
    if (helperRef.current) helperRef.current.update()
  }, [meta])

  const [helper, setHelper] = useState<THREE.CameraHelper | null>(null)

  useEffect(() => {
    if (!camRef.current) return
    const h = new THREE.CameraHelper(camRef.current)
    helperRef.current = h
    setHelper(h)
    return () => {
      try {
        h.dispose()
      } catch {}
    }
  }, [])

  if (!meta) return null
  if (meta.visible === false) return null

  return (
    <group
      ref={groupRef}
      onClick={(e) => {
        e.stopPropagation()
        onSelect(cameraId)
        if (e.detail === 2) onFocus(cameraId)
      }}
    >
      <perspectiveCamera ref={camRef} />
      {helper && <primitive object={helper} />}
      {selected && <axesHelper args={[0.5]} />}
    </group>
  )
}

export default function CameraScene({
  cameraIds,
  selectedId,
  onSelect,
  onFocus,
  onRegisterObject,
}: {
  cameraIds: string[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
  onRegisterObject?: (id: string, obj: THREE.Object3D | null) => void
}) {
  return (
    <>
      {cameraIds.map((id) => (
        <CameraGizmo
          key={id}
          cameraId={id}
          selected={id === selectedId}
          onSelect={onSelect}
          onFocus={onFocus}
          onRegisterObject={onRegisterObject}
        />
      ))}
    </>
  )
}
