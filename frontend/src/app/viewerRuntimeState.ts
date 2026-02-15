type Pose = {
  position: [number, number, number]
  rotation: [number, number, number, number]
}

type VisualOverride = {
  colorMode?: 'logged' | 'solid' | 'height' | 'depth'
  solidColor?: [number, number, number]
  colorMap?: string
  showBounds?: boolean
}

type RuntimeWindow = Window & {
  __begira_local_pose?: Record<string, Pose>
  __begira_visual_override?: Record<string, VisualOverride>
  __begira_lod_override?: Record<string, string | undefined>
}

function runtimeWindow(): RuntimeWindow {
  return window as RuntimeWindow
}

export function getLocalPose(id: string): Pose | null {
  const pose = runtimeWindow().__begira_local_pose?.[id]
  return pose ?? null
}

export function setLocalPose(id: string, pose: Pose): void {
  const w = runtimeWindow()
  w.__begira_local_pose = w.__begira_local_pose ?? {}
  w.__begira_local_pose[id] = pose
}

export function clearLocalPose(id: string): void {
  const w = runtimeWindow()
  if (!w.__begira_local_pose) return
  delete w.__begira_local_pose[id]
}

export function clearAllLocalPose(): void {
  runtimeWindow().__begira_local_pose = {}
}

export function getVisualOverride(id: string): VisualOverride | undefined {
  return runtimeWindow().__begira_visual_override?.[id]
}

export function setVisualOverride(id: string, value: VisualOverride): void {
  const w = runtimeWindow()
  w.__begira_visual_override = w.__begira_visual_override ?? {}
  w.__begira_visual_override[id] = value
}

export function clearVisualOverride(id: string): void {
  const w = runtimeWindow()
  if (!w.__begira_visual_override) return
  delete w.__begira_visual_override[id]
}

export function clearAllVisualOverrides(): void {
  runtimeWindow().__begira_visual_override = {}
}

export function setLodOverride(id: string, value: string | undefined): void {
  const w = runtimeWindow()
  w.__begira_lod_override = w.__begira_lod_override ?? {}
  if (value === undefined) {
    delete w.__begira_lod_override[id]
  } else {
    w.__begira_lod_override[id] = value
  }
}

export function getLodOverride(id: string): string | undefined {
  return runtimeWindow().__begira_lod_override?.[id]
}
