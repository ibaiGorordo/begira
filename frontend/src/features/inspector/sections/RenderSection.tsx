import type { ReactNode } from 'react'

type Props = {
  children?: ReactNode
}

export default function RenderSection({ children }: Props) {
  return <>{children ?? null}</>
}
