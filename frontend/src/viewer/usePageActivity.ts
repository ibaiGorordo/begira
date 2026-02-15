import { useEffect, useState } from 'react'

function readIsActive(): boolean {
  if (typeof document === 'undefined') return true
  const visible = document.visibilityState === 'visible'
  const focused = typeof document.hasFocus === 'function' ? document.hasFocus() : true
  return visible && focused
}

export function usePageActivity(): boolean {
  const [active, setActive] = useState<boolean>(() => readIsActive())

  useEffect(() => {
    const sync = () => setActive(readIsActive())
    sync()
    document.addEventListener('visibilitychange', sync)
    window.addEventListener('focus', sync)
    window.addEventListener('blur', sync)
    window.addEventListener('pageshow', sync)
    window.addEventListener('pagehide', sync)
    return () => {
      document.removeEventListener('visibilitychange', sync)
      window.removeEventListener('focus', sync)
      window.removeEventListener('blur', sync)
      window.removeEventListener('pageshow', sync)
      window.removeEventListener('pagehide', sync)
    }
  }, [])

  return active
}

