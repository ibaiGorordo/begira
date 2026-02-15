import * as THREE from 'three'
import type { SampleQuery } from './api'
import GaussianSplatItem from './GaussianSplatItem'

export default function GaussianSplatScene({
  elementIds,
  selectedId,
  onSelect,
  onFocus,
  sample,
  enabled = true,
  onRegisterObject,
}: {
  elementIds: string[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
  sample?: SampleQuery
  enabled?: boolean
  onRegisterObject?: (id: string, obj: THREE.Object3D | null) => void
}) {
  return (
    <>
      {elementIds.map((id) => (
        <GaussianSplatItem
          key={id}
          elementId={id}
          selected={id === selectedId}
          onSelect={onSelect}
          onFocus={onFocus}
          sample={sample}
          enabled={enabled}
          onRegisterObject={onRegisterObject}
        />
      ))}
    </>
  )
}
