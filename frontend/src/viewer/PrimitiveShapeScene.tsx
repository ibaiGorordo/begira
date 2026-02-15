import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

const UNIT_BOX_GEOMETRY = new THREE.BoxGeometry(1, 1, 1)
const UNIT_BOX_EDGES = new THREE.EdgesGeometry(UNIT_BOX_GEOMETRY)
const UNIT_SPHERE_GEOMETRY = new THREE.SphereGeometry(1, 14, 10)
const UNIT_SPHERE_EDGES = new THREE.EdgesGeometry(UNIT_SPHERE_GEOMETRY)

type Vec3 = [number, number, number]
type Quat = [number, number, number, number]

export type BoxVisual = {
  id: string
  position?: Vec3
  rotation?: Quat
  size?: Vec3
  color?: Vec3
  visible?: boolean
  interactive?: boolean
  variant?: 'default' | 'boundary'
}

export type EllipsoidVisual = {
  id: string
  position?: Vec3
  rotation?: Quat
  radii?: Vec3
  color?: Vec3
  visible?: boolean
}

export type WireBoxOverlay = {
  id: string
  center: Vec3
  size: Vec3
  color?: Vec3
}

function colorFromTuple(color: Vec3 | undefined, fallbackHex: string): THREE.Color {
  if (!color) return new THREE.Color(fallbackHex)
  return new THREE.Color(color[0], color[1], color[2])
}

function BoxItem({
  box,
  selected,
  onSelect,
  onFocus,
  onRegisterObject,
}: {
  box: BoxVisual
  selected: boolean
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
  onRegisterObject?: (id: string, obj: THREE.Object3D | null) => void
}) {
  const groupRef = useRef<THREE.Group | null>(null)
  const lastLocalPoseRef = useRef<string | null>(null)
  const interactive = box.interactive !== false
  const isBoundary = box.variant === 'boundary'
  const color = useMemo(
    () => colorFromTuple(box.color, isBoundary ? '#8ec5f7' : selected ? '#9dd1ff' : '#8ec5f7'),
    [box.color, isBoundary, selected],
  )

  useEffect(() => {
    if (!interactive) return
    if (!onRegisterObject) return
    if (!groupRef.current) return
    onRegisterObject(box.id, groupRef.current)
    return () => onRegisterObject(box.id, null)
  }, [box.id, interactive, onRegisterObject])

  useEffect(() => {
    if (!groupRef.current) return
    if ((window as any).__begira_local_pose?.[box.id]) return
    const p = box.position ?? [0, 0, 0]
    const r = box.rotation ?? [0, 0, 0, 1]
    const s = box.size ?? [1, 1, 1]
    groupRef.current.position.set(p[0], p[1], p[2])
    groupRef.current.quaternion.set(r[0], r[1], r[2], r[3]).normalize()
    groupRef.current.scale.set(Math.max(1e-6, s[0]), Math.max(1e-6, s[1]), Math.max(1e-6, s[2]))
  }, [box.id, box.position, box.rotation, box.size])

  useFrame(() => {
    try {
      const local = (window as any).__begira_local_pose?.[box.id]
      if (!local || !local.position || !local.rotation || !groupRef.current) return
      const key = `${local.position.join(',')}|${local.rotation.join(',')}`
      if (lastLocalPoseRef.current === key) return
      lastLocalPoseRef.current = key
      groupRef.current.position.set(local.position[0], local.position[1], local.position[2])
      groupRef.current.quaternion.set(local.rotation[0], local.rotation[1], local.rotation[2], local.rotation[3]).normalize()
    } catch {
      // ignore local pose bridge failures
    }
  })

  if (box.visible === false) return null

  return (
    <group
      ref={groupRef}
      onClick={(e) => {
        if (!interactive) return
        e.stopPropagation()
        onSelect(box.id)
        if (e.detail === 2) onFocus(box.id)
      }}
      onDoubleClick={(e) => {
        if (!interactive) return
        e.stopPropagation()
        onFocus(box.id)
      }}
    >
      <lineSegments geometry={UNIT_BOX_EDGES}>
        <lineBasicMaterial
          color={color}
          toneMapped={false}
          transparent={isBoundary}
          opacity={isBoundary ? 0.95 : 1.0}
          depthTest={!isBoundary}
          depthWrite={!isBoundary}
        />
      </lineSegments>
    </group>
  )
}

function EllipsoidItem({
  ellipsoid,
  selected,
  onSelect,
  onFocus,
  onRegisterObject,
}: {
  ellipsoid: EllipsoidVisual
  selected: boolean
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
  onRegisterObject?: (id: string, obj: THREE.Object3D | null) => void
}) {
  const groupRef = useRef<THREE.Group | null>(null)
  const lastLocalPoseRef = useRef<string | null>(null)
  const color = useMemo(
    () => colorFromTuple(ellipsoid.color, selected ? '#9bd6b0' : '#89cda3'),
    [ellipsoid.color, selected],
  )

  useEffect(() => {
    if (!onRegisterObject) return
    if (!groupRef.current) return
    onRegisterObject(ellipsoid.id, groupRef.current)
    return () => onRegisterObject(ellipsoid.id, null)
  }, [ellipsoid.id, onRegisterObject])

  useEffect(() => {
    if (!groupRef.current) return
    if ((window as any).__begira_local_pose?.[ellipsoid.id]) return
    const p = ellipsoid.position ?? [0, 0, 0]
    const r = ellipsoid.rotation ?? [0, 0, 0, 1]
    const rad = ellipsoid.radii ?? [0.5, 0.5, 0.5]
    groupRef.current.position.set(p[0], p[1], p[2])
    groupRef.current.quaternion.set(r[0], r[1], r[2], r[3]).normalize()
    groupRef.current.scale.set(Math.max(1e-6, rad[0]), Math.max(1e-6, rad[1]), Math.max(1e-6, rad[2]))
  }, [ellipsoid.id, ellipsoid.position, ellipsoid.rotation, ellipsoid.radii])

  useFrame(() => {
    try {
      const local = (window as any).__begira_local_pose?.[ellipsoid.id]
      if (!local || !local.position || !local.rotation || !groupRef.current) return
      const key = `${local.position.join(',')}|${local.rotation.join(',')}`
      if (lastLocalPoseRef.current === key) return
      lastLocalPoseRef.current = key
      groupRef.current.position.set(local.position[0], local.position[1], local.position[2])
      groupRef.current.quaternion.set(local.rotation[0], local.rotation[1], local.rotation[2], local.rotation[3]).normalize()
    } catch {
      // ignore local pose bridge failures
    }
  })

  if (ellipsoid.visible === false) return null

  return (
    <group
      ref={groupRef}
      onClick={(e) => {
        e.stopPropagation()
        onSelect(ellipsoid.id)
        if (e.detail === 2) onFocus(ellipsoid.id)
      }}
      onDoubleClick={(e) => {
        e.stopPropagation()
        onFocus(ellipsoid.id)
      }}
    >
      <lineSegments geometry={UNIT_SPHERE_EDGES}>
        <lineBasicMaterial color={color} toneMapped={false} />
      </lineSegments>
    </group>
  )
}

export function BoxScene({
  boxes,
  selectedId,
  onSelect,
  onFocus,
  onRegisterObject,
}: {
  boxes: BoxVisual[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
  onRegisterObject?: (id: string, obj: THREE.Object3D | null) => void
}) {
  return (
    <>
      {boxes.map((box) => (
        <BoxItem
          key={box.id}
          box={box}
          selected={box.id === selectedId}
          onSelect={onSelect}
          onFocus={onFocus}
          onRegisterObject={onRegisterObject}
        />
      ))}
    </>
  )
}

export function EllipsoidScene({
  ellipsoids,
  selectedId,
  onSelect,
  onFocus,
  onRegisterObject,
}: {
  ellipsoids: EllipsoidVisual[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
  onRegisterObject?: (id: string, obj: THREE.Object3D | null) => void
}) {
  return (
    <>
      {ellipsoids.map((ellipsoid) => (
        <EllipsoidItem
          key={ellipsoid.id}
          ellipsoid={ellipsoid}
          selected={ellipsoid.id === selectedId}
          onSelect={onSelect}
          onFocus={onFocus}
          onRegisterObject={onRegisterObject}
        />
      ))}
    </>
  )
}

export function WireBoxOverlayScene({ boxes }: { boxes: WireBoxOverlay[] }) {
  const inheritedBoxes: BoxVisual[] = boxes.map((box) => ({
    id: box.id,
    position: box.center,
    size: box.size,
    color: box.color,
    interactive: false,
    variant: 'boundary',
    visible: true,
  }))

  return (
    <BoxScene
      boxes={inheritedBoxes}
      selectedId={null}
      onSelect={() => {
        // non-interactive special case
      }}
      onFocus={() => {
        // non-interactive special case
      }}
    />
  )
}
