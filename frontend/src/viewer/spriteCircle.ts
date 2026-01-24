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

  // Hard-edged circular alpha mask.
  // This is meant for the 'circles' render mode where we use alphaTest (opaque, no blending).
  const r = size / 2
  ctx.fillStyle = 'rgba(255,255,255,1.0)'
  ctx.beginPath()
  ctx.arc(r, r, r * 0.98, 0, Math.PI * 2)
  ctx.fill()

  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.needsUpdate = true

  _circleTexture = tex
  return tex
}
