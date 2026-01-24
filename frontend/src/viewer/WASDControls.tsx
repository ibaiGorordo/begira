import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'

type Keys = {
  w: boolean
  a: boolean
  s: boolean
  d: boolean
  q: boolean
  e: boolean
  shift: boolean
}

function isTypingTarget(ev: KeyboardEvent): boolean {
  const t = ev.target as HTMLElement | null
  if (!t) return false
  const tag = t.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || (t as any).isContentEditable === true
}

export default function WASDControls({ enabled, speed = 2 }: { enabled: boolean; speed?: number }) {
  const { camera } = useThree()

  const keys = useRef<Keys>({ w: false, a: false, s: false, d: false, q: false, e: false, shift: false })

  const forward = useMemo(() => new THREE.Vector3(), [])
  const right = useMemo(() => new THREE.Vector3(), [])
  const up = useMemo(() => new THREE.Vector3(0, 1, 0), [])
  const move = useMemo(() => new THREE.Vector3(), [])

  useEffect(() => {
    if (!enabled) return

    const down = (ev: KeyboardEvent) => {
      if (isTypingTarget(ev)) return

      if (ev.key === 'w' || ev.key === 'W') keys.current.w = true
      if (ev.key === 'a' || ev.key === 'A') keys.current.a = true
      if (ev.key === 's' || ev.key === 'S') keys.current.s = true
      if (ev.key === 'd' || ev.key === 'D') keys.current.d = true
      if (ev.key === 'q' || ev.key === 'Q') keys.current.q = true
      if (ev.key === 'e' || ev.key === 'E') keys.current.e = true
      if (ev.key === 'Shift') keys.current.shift = true
    }

    const upFn = (ev: KeyboardEvent) => {
      if (ev.key === 'w' || ev.key === 'W') keys.current.w = false
      if (ev.key === 'a' || ev.key === 'A') keys.current.a = false
      if (ev.key === 's' || ev.key === 'S') keys.current.s = false
      if (ev.key === 'd' || ev.key === 'D') keys.current.d = false
      if (ev.key === 'q' || ev.key === 'Q') keys.current.q = false
      if (ev.key === 'e' || ev.key === 'E') keys.current.e = false
      if (ev.key === 'Shift') keys.current.shift = false
    }

    window.addEventListener('keydown', down)
    window.addEventListener('keyup', upFn)

    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', upFn)
    }
  }, [enabled])

  useFrame((_, delta) => {
    if (!enabled) return

    const k = keys.current
    const anyCam = camera as unknown as { controls?: { target: THREE.Vector3; update: () => void } }

    const pos = camera.position
    const target = anyCam.controls?.target

    // Scale movement by current "zoom" (distance to OrbitControls target).
    // This matches the feel of tools like Rerun: far away = move fast, close = move slow.
    const dist = target ? pos.distanceTo(target) : 1

    // Base: 2 units/sec at dist=1. Grows linearly with distance.
    // Clamp so we don't get stuck slow or go insanely fast.
    const distScale = THREE.MathUtils.clamp(dist, 0.1, 5000)

    const actualSpeed = (k.shift ? 4 : 1) * speed * distScale

    camera.getWorldDirection(forward)
    right.copy(forward).cross(up).normalize()

    move.set(0, 0, 0)
    if (k.w) move.add(forward)
    if (k.s) move.sub(forward)
    if (k.d) move.add(right)
    if (k.a) move.sub(right)
    if (k.e) move.add(up)
    if (k.q) move.sub(up)

    if (move.lengthSq() === 0) return

    move.normalize().multiplyScalar(actualSpeed * delta)

    pos.add(move)
    if (target) target.add(move)

    anyCam.controls?.update()
  })

  return null
}
