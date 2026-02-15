import { useState } from 'react'

export function usePanels(initialLeft = true, initialRight = true) {
  const [leftOpen, setLeftOpen] = useState(initialLeft)
  const [rightOpen, setRightOpen] = useState(initialRight)

  return {
    leftOpen,
    rightOpen,
    setLeftOpen,
    setRightOpen,
  }
}
