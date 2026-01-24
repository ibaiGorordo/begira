import * as THREE from 'three'

let _circleTexture: THREE.Texture | null = null

export function getCircleSpriteTexture(): THREE.Texture {
  if (_circleTexture) return _circleTexture

  const size = 128
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Failed to create 2D canvas context')

  ctx.clearRect(0, 0, size, size)

  // Softer circular sprite (anti-aliased). We keep the center slightly dimmer
  // to avoid the "too bright" look when many points overlap.
  const r = size / 2
  const grd = ctx.createRadialGradient(r, r, 0, r, r, r)
  grd.addColorStop(0.0, 'rgba(255,255,255,0.75)')
  grd.addColorStop(0.6, 'rgba(255,255,255,0.65)')
  grd.addColorStop(0.9, 'rgba(255,255,255,0.15)')
  grd.addColorStop(1.0, 'rgba(255,255,255,0.0)')

  ctx.fillStyle = grd
  ctx.beginPath()
  ctx.arc(r, r, r, 0, Math.PI * 2)
  ctx.fill()

  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.needsUpdate = true

  _circleTexture = tex
  return tex
}
