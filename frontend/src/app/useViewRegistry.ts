import { useMemo, useState } from 'react'

type ViewState = {
  visible: boolean
  deleted: boolean
}

export function useViewRegistry() {
  const [views, setViews] = useState<Record<string, ViewState>>({})

  const api = useMemo(
    () => ({
      views,
      setView(viewId: string, patch: Partial<ViewState>) {
        setViews((prev) => {
          const cur = prev[viewId] ?? { visible: true, deleted: false }
          return { ...prev, [viewId]: { ...cur, ...patch } }
        })
      },
      removeView(viewId: string) {
        setViews((prev) => {
          if (!(viewId in prev)) return prev
          const next = { ...prev }
          delete next[viewId]
          return next
        })
      },
    }),
    [views],
  )

  return api
}
