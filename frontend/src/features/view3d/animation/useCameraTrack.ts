import { useEffect, useState } from 'react'

import {
  fetchCameraAnimation,
  fetchCameraAnimationTrajectory,
  type CameraAnimationTrack,
  type CameraAnimationTrajectory,
} from '../../../viewer/api'

type State = {
  track: CameraAnimationTrack | null
  trajectory: CameraAnimationTrajectory | null
}

export function useCameraTrack(cameraId: string | null): State {
  const [state, setState] = useState<State>({ track: null, trajectory: null })

  useEffect(() => {
    if (!cameraId) {
      setState({ track: null, trajectory: null })
      return
    }
    let alive = true
    const run = async () => {
      const track = await fetchCameraAnimation(cameraId)
      if (!alive) return
      if (!track) {
        setState({ track: null, trajectory: null })
        return
      }
      const trajectory = await fetchCameraAnimationTrajectory(cameraId, {
        startFrame: track.startFrame,
        endFrame: track.endFrame,
        stride: 1,
      })
      if (!alive) return
      setState({ track, trajectory })
    }
    void run()
    return () => {
      alive = false
    }
  }, [cameraId])

  return state
}
