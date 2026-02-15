import type { ReactNode } from 'react'

type Props = {
  children?: ReactNode
}

export default function CameraDrivers({ children }: Props) {
  return <>{children ?? null}</>
}
