import { useMemo } from 'react'
import type { ElementInfo } from './api'

type Props = {
  images: ElementInfo[]
  selectedId: string | null
  onSelect: (id: string) => void
}

function asNumber(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v
  return null
}

export default function ImageView({ images, selectedId, onSelect }: Props) {
  const selectedImage = useMemo(() => {
    if (images.length === 0) return null
    const hit = selectedId ? images.find((e) => e.id === selectedId) : null
    return hit ?? images[0]
  }, [images, selectedId])

  if (images.length === 0) {
    return (
      <div style={{ height: '100%', display: 'grid', placeItems: 'center', color: '#e8ecff', opacity: 0.7 }}>
        No image elements logged yet.
      </div>
    )
  }

  if (!selectedImage) return null

  const summary = selectedImage.summary ?? {}
  const width = asNumber((summary as any).width)
  const height = asNumber((summary as any).height)
  const channels = asNumber((summary as any).channels)
  const mainUrl = `/api/elements/${encodeURIComponent(selectedImage.id)}/payloads/image?rev=${selectedImage.revision}`

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, color: '#e8ecff' }}>
      <div style={{ padding: '10px 12px', borderBottom: '1px solid #1b2235', background: '#0d1328' }}>
        <div style={{ fontSize: 13, fontWeight: 600 }}>{selectedImage.name}</div>
        <div style={{ fontSize: 12, opacity: 0.75 }}>
          {width && height ? `${width}x${height}` : 'unknown size'}
          {channels ? `, ${channels}ch` : ''}
        </div>
      </div>

      <div style={{ flex: 1, minHeight: 0, display: 'grid', placeItems: 'center', background: '#070c18' }}>
        <img
          src={mainUrl}
          alt={selectedImage.name}
          style={{
            maxWidth: '100%',
            maxHeight: '100%',
            objectFit: 'contain',
            imageRendering: 'auto',
          }}
        />
      </div>

      <div
        style={{
          borderTop: '1px solid #1b2235',
          background: '#0d1328',
          padding: 8,
          display: 'flex',
          gap: 8,
          overflowX: 'auto',
        }}
      >
        {images.map((img) => {
          const active = img.id === selectedImage.id
          const thumbUrl = `/api/elements/${encodeURIComponent(img.id)}/payloads/image?rev=${img.revision}`
          return (
            <button
              key={img.id}
              type="button"
              onClick={() => onSelect(img.id)}
              style={{
                border: active ? '1px solid #6ea8fe' : '1px solid #1b2235',
                borderRadius: 8,
                background: '#0f1630',
                color: '#e8ecff',
                padding: 6,
                minWidth: 128,
                maxWidth: 128,
                textAlign: 'left',
              }}
            >
              <img
                src={thumbUrl}
                alt={img.name}
                style={{
                  width: '100%',
                  height: 72,
                  objectFit: 'cover',
                  borderRadius: 4,
                  display: 'block',
                }}
              />
              <div
                style={{
                  marginTop: 6,
                  fontSize: 11,
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
                title={img.name}
              >
                {img.name}
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
