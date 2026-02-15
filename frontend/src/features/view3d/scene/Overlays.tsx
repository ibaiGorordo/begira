import type { ReactNode } from 'react'

type Props = {
  children?: ReactNode
}

export default function Overlays({ children }: Props) {
  return <>{children ?? null}</>
}
