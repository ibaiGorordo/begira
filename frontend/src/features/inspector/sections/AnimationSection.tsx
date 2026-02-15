import type { ReactNode } from 'react'

type Props = {
  children?: ReactNode
}

export default function AnimationSection({ children }: Props) {
  return <>{children ?? null}</>
}
