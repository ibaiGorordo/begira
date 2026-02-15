import { Html } from '@react-three/drei'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { CAMERA_GIZMO_OVERLAY_LAYER } from './CameraScene'
import type { CameraAnimationTrack, CameraAnimationTrajectory } from './api'

type Props = {
  track: CameraAnimationTrack | null
  trajectory: CameraAnimationTrajectory | null
  selectedKeyFrame: number | null
  editable: boolean
  onSelectKey: (frame: number | null) => void
  onShiftClickPath?: (point: [number, number, number]) => void
  onRegisterKeyObject?: (frame: number, obj: THREE.Object3D | null) => void
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v))
}

function keyColorForIndex(index: number, count: number): string {
  if (count <= 1) return '#ffd48e'
  const t = clamp(index / (count - 1), 0, 1)
  // Blue -> cyan -> green -> yellow -> orange
  const hue = 0.62 - t * 0.55
  const color = new THREE.Color().setHSL(hue, 0.85, 0.58)
  return `#${color.getHexString()}`
}

function buildTrajectoryGeometry(points: Array<[number, number, number]>): THREE.BufferGeometry | null {
  if (points.length < 2) return null
  const segmentCount = points.length - 1
  const buf = new Float32Array(segmentCount * 2 * 3)
  for (let i = 0; i < segmentCount; i += 1) {
    const a = points[i]
    const b = points[i + 1]
    const base = i * 6
    buf[base + 0] = a[0]
    buf[base + 1] = a[1]
    buf[base + 2] = a[2]
    buf[base + 3] = b[0]
    buf[base + 4] = b[1]
    buf[base + 5] = b[2]
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(buf, 3))
  return g
}

export default function CameraTrajectoryScene({
  track,
  trajectory,
  selectedKeyFrame,
  editable,
  onSelectKey,
  onShiftClickPath,
  onRegisterKeyObject,
}: Props) {
  const [hoveredKeyFrame, setHoveredKeyFrame] = useState<number | null>(null)
  const groupRef = useRef<THREE.Group | null>(null)

  const lineGeometry = useMemo(() => {
    if (!trajectory) return null
    return buildTrajectoryGeometry(trajectory.positions)
  }, [trajectory])

  useEffect(() => {
    return () => {
      if (lineGeometry) lineGeometry.dispose()
    }
  }, [lineGeometry])

  useEffect(() => {
    if (!groupRef.current) return
    groupRef.current.traverse((obj) => {
      obj.layers.set(CAMERA_GIZMO_OVERLAY_LAYER)
    })
  }, [track, trajectory, lineGeometry])

  const frameToPosition = useMemo(() => {
    const out = new Map<number, [number, number, number]>()
    if (!trajectory) return out
    const n = Math.min(trajectory.frames.length, trajectory.positions.length)
    for (let i = 0; i < n; i += 1) {
      out.set(Number(trajectory.frames[i]), trajectory.positions[i])
    }
    return out
  }, [trajectory])

  const keyMarkers = useMemo(() => {
    if (!track || track.mode !== 'orbit') return [] as Array<{ frame: number; position: [number, number, number]; color: string }>
    const out: Array<{ frame: number; position: [number, number, number]; color: string }> = []
    const count = track.controlKeys.length
    for (let i = 0; i < count; i += 1) {
      const key = track.controlKeys[i]
      const p = frameToPosition.get(Number(key.frame))
      if (!p) continue
      out.push({
        frame: Number(key.frame),
        position: p,
        color: keyColorForIndex(i, count),
      })
    }
    return out
  }, [frameToPosition, track])

  const markerRadius = useMemo(() => {
    if (!trajectory || trajectory.positions.length < 2) return 0.012
    const min = new THREE.Vector3(Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY)
    const max = new THREE.Vector3(Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY)
    for (const p of trajectory.positions) {
      min.min(new THREE.Vector3(p[0], p[1], p[2]))
      max.max(new THREE.Vector3(p[0], p[1], p[2]))
    }
    const diag = max.sub(min).length()
    return clamp(diag * 0.0018, 0.008, 0.03)
  }, [trajectory])

  if (!track || !trajectory || !lineGeometry) return null

  return (
    <group ref={groupRef}>
      <lineSegments
        geometry={lineGeometry}
        frustumCulled={false}
        renderOrder={4000}
        onPointerDown={(e) => {
          const mouseEvent = e.nativeEvent as PointerEvent
          if (!mouseEvent.shiftKey) return
          e.stopPropagation()
          onShiftClickPath?.([e.point.x, e.point.y, e.point.z])
        }}
      >
        <lineBasicMaterial color="#98c8ff" transparent opacity={0.85} depthTest depthWrite={false} toneMapped={false} />
      </lineSegments>
      {keyMarkers.map((key) => {
        const selected = selectedKeyFrame === key.frame
        const hovered = hoveredKeyFrame === key.frame
        return (
          <mesh
            key={`key-${key.frame}`}
            ref={(obj) => onRegisterKeyObject?.(key.frame, obj)}
            position={key.position}
            renderOrder={4010}
            frustumCulled={false}
            onPointerDown={(e) => {
              e.stopPropagation()
            }}
            onPointerUp={(e) => {
              e.stopPropagation()
            }}
            onClick={(e) => {
              e.stopPropagation()
              onSelectKey(key.frame)
            }}
            onPointerOver={(e) => {
              e.stopPropagation()
              setHoveredKeyFrame(key.frame)
            }}
            onPointerOut={(e) => {
              e.stopPropagation()
              setHoveredKeyFrame((prev) => (prev === key.frame ? null : prev))
            }}
          >
            <sphereGeometry args={[markerRadius, 18, 18]} />
            <meshBasicMaterial
              color={selected ? '#ffe7a8' : key.color}
              depthTest
              depthWrite={false}
              transparent
              opacity={selected || hovered ? 0.98 : 0.86}
              toneMapped={false}
            />
            {(selected || hovered) && (
              <Html position={[0, markerRadius * 1.9, 0]} center sprite style={{ pointerEvents: 'none' }}>
                <div
                  style={{
                    padding: '2px 6px',
                    borderRadius: 6,
                    fontSize: 11,
                    fontWeight: 600,
                    background: 'rgba(9, 15, 33, 0.9)',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    color: '#dce7ff',
                    whiteSpace: 'nowrap',
                  }}
                >
                  frame {key.frame}
                </div>
              </Html>
            )}
          </mesh>
        )
      })}
    </group>
  )
}
