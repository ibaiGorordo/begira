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

export type PointCloudRenderMode = 'fast' | 'circles' | 'quality'

function Cloud({
  cloudId,
  selected,
  onSelect,
  onFocus,
  renderMode,
}: {
  cloudId: string
  selected: boolean
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
  renderMode: PointCloudRenderMode
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

  const baseSize = Math.max(0.001, state.meta.pointSize ?? 0.02)
  const fast = renderMode === 'fast'
  const circles = renderMode === 'circles'

  return (
    <points
      geometry={state.decoded.geometry}
      // IMPORTANT: raycasting against large THREE.Points can become extremely expensive,
      // especially while moving the camera (pointer events trigger raycasts internally).
      // We disable it to keep interaction smooth for large point clouds.
      raycast={() => null}
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
        // 'fast'  : opaque square points (fastest)
        // 'circles': opaque circle sprites via alphaTest (still fast, no blending)
        // 'quality': transparent blended sprites (prettiest, slowest)
        size={fast ? Math.max(0.001, baseSize * 0.75) : baseSize}
        sizeAttenuation
        vertexColors={state.decoded.hasColor}
        map={circles || renderMode === 'quality' ? circleMap : null}
        alphaTest={circles ? 0.5 : fast ? 0.0 : 0.05}
        transparent={renderMode === 'quality'}
        opacity={renderMode === 'quality' ? (selected ? 0.9 : 0.6) : 1.0}
        depthWrite={renderMode !== 'quality'}
        depthTest
        blending={renderMode === 'quality' ? THREE.NormalBlending : THREE.NoBlending}
      />
    </points>
  )
}

export default function MultiPointCloudScene({
  cloudIds,
  selectedId,
  onSelect,
  onFocus,
  renderMode = 'fast',
}: {
  cloudIds: string[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
  renderMode?: PointCloudRenderMode
}) {
  return (
    <>
      {cloudIds.map((id) => (
        <Cloud
          key={id}
          cloudId={id}
          selected={id === selectedId}
          onSelect={onSelect}
          onFocus={onFocus}
          renderMode={renderMode}
        />
      ))}
    </>
  )
}
