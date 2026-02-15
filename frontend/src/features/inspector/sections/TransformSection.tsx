import type { ReactNode } from 'react'

type Props = {
  children?: ReactNode
}

export default function TransformSection({ children }: Props) {
  return <>{children ?? null}</>
}
