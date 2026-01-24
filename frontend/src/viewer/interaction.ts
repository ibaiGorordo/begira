export type ClickGesture = {
  isDown: boolean
  downX: number
  downY: number
  downTime: number
  moved: boolean
}

export function createClickGesture(): ClickGesture {
  return { isDown: false, downX: 0, downY: 0, downTime: 0, moved: false }
}

export function onPointerDownGesture(g: ClickGesture, e: PointerEvent) {
  g.isDown = true
  g.downX = e.clientX
  g.downY = e.clientY
  g.downTime = performance.now()
  g.moved = false
}

export function onPointerMoveGesture(g: ClickGesture, e: PointerEvent, moveThresholdPx = 6) {
  if (!g.isDown) return
  const dx = e.clientX - g.downX
  const dy = e.clientY - g.downY
  if (dx * dx + dy * dy > moveThresholdPx * moveThresholdPx) g.moved = true
}

export function isClickGesture(g: ClickGesture, e: PointerEvent, maxDurationMs = 250): boolean {
  if (!g.isDown) return false
  const dt = performance.now() - g.downTime
  const dx = e.clientX - g.downX
  const dy = e.clientY - g.downY
  const dist2 = dx * dx + dy * dy
  return dt <= maxDurationMs && dist2 <= 6 * 6 && !g.moved
}

export function onPointerUpGesture(g: ClickGesture) {
  g.isDown = false
}

export function isDoubleClick(nowMs: number, lastClickMs: number | null, maxGapMs = 350): boolean {
  if (lastClickMs === null) return false
  return nowMs - lastClickMs <= maxGapMs
}
