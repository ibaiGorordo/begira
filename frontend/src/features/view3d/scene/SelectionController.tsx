import type { ReactNode } from 'react'

type Props = {
  children?: ReactNode
}

export default function SelectionController({ children }: Props) {
  return <>{children ?? null}</>
}
