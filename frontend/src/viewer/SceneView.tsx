import * as THREE from 'three'
import { useMemo } from 'react'
import { usePointCloud } from './usePointCloud'

export default function SceneView({ cloudId, onPicked }: { cloudId: string; onPicked: (picked: boolean) => void }) {
  const state = usePointCloud(cloudId)

  const circleMap = useMemo(() => {
    // imported lazily to avoid circular deps
    return null
  }, [])

  if (state.status !== 'ready') return null

  // Note: we attach onPointerDown to the points object for selection.
  // Clicking empty space is handled by the Canvas onPointerMissed.
  return (
    <points
      geometry={state.decoded.geometry}
      onPointerDown={(e) => {
        e.stopPropagation()
        onPicked(true)
      }}
    >
      <pointsMaterial
        size={Math.max(0.001, state.meta.pointSize ?? 0.02)}
        sizeAttenuation
        vertexColors={state.decoded.hasColor}
        transparent
        opacity={0.6}
        depthWrite={false}
        blending={THREE.NormalBlending}
      />
    </points>
  )
}

