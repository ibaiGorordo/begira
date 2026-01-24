import * as THREE from 'three'
import { useEffect, useMemo, useRef } from 'react'
import { getCircleSpriteTexture } from './spriteCircle'
import { usePointCloud } from './usePointCloud'
import {
  createClickGesture,
  isClickGesture,
  isDoubleClick,
  onPointerDownGesture,
  onPointerMoveGesture,
  onPointerUpGesture,
} from './interaction'

function Cloud({
  cloudId,
  selected,
  onSelect,
  onFocus,
}: {
  cloudId: string
  selected: boolean
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
}) {
  const state = usePointCloud(cloudId)
  const circleMap = useMemo(() => getCircleSpriteTexture(), [])

  const gesture = useRef(createClickGesture())
  const lastClickMs = useRef<number | null>(null)

  useEffect(() => {
    gesture.current = createClickGesture()
    lastClickMs.current = null
  }, [cloudId])

  if (state.status !== 'ready') return null

  return (
    <points
      geometry={state.decoded.geometry}
      onPointerDown={(e) => {
        onPointerDownGesture(gesture.current, e.nativeEvent)
      }}
      onPointerMove={(e) => {
        onPointerMoveGesture(gesture.current, e.nativeEvent)
      }}
      onPointerUp={(e) => {
        const wasClick = isClickGesture(gesture.current, e.nativeEvent)
        onPointerUpGesture(gesture.current)
        if (!wasClick) return

        e.stopPropagation()

        const now = performance.now()
        const dbl = isDoubleClick(now, lastClickMs.current)
        lastClickMs.current = now

        // If it's already selected and it's just a single click, do nothing.
        // This avoids re-triggering any selection/focus side-effects that can reset camera.
        if (!dbl && selected) return

        onSelect(cloudId)
        if (dbl) onFocus(cloudId)
      }}
    >
      <pointsMaterial
        size={Math.max(0.001, state.meta.pointSize ?? 0.02)}
        sizeAttenuation
        vertexColors={state.decoded.hasColor}
        map={circleMap}
        alphaTest={0.05}
        transparent
        opacity={selected ? 0.9 : 0.6}
        depthWrite={false}
        blending={THREE.NormalBlending}
      />
    </points>
  )
}

export default function MultiPointCloudScene({
  cloudIds,
  selectedId,
  onSelect,
  onFocus,
}: {
  cloudIds: string[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
}) {
  return (
    <>
      {cloudIds.map((id) => (
        <Cloud key={id} cloudId={id} selected={id === selectedId} onSelect={onSelect} onFocus={onFocus} />
      ))}
    </>
  )
}
