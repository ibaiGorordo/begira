import { useMemo, useRef, useState } from 'react'

export type UndoableAction = {
  label: string
  do: () => Promise<void>
  undo: () => Promise<void>
}

export function useHistory() {
  const undoStack = useRef<UndoableAction[]>([])
  const redoStack = useRef<UndoableAction[]>([])
  const [counts, setCounts] = useState({ undo: 0, redo: 0 })

  const bump = () => {
    setCounts({
      undo: undoStack.current.length,
      redo: redoStack.current.length,
    })
  }

  const api = useMemo(
    () => ({
      counts,
      async run(action: UndoableAction) {
        await action.do()
        undoStack.current.push(action)
        redoStack.current = []
        bump()
      },
      async undo() {
        const action = undoStack.current.pop()
        if (!action) {
          bump()
          return
        }
        await action.undo()
        redoStack.current.push(action)
        bump()
      },
      async redo() {
        const action = redoStack.current.pop()
        if (!action) {
          bump()
          return
        }
        await action.do()
        undoStack.current.push(action)
        bump()
      },
      clear() {
        undoStack.current = []
        redoStack.current = []
        bump()
      },
    }),
    [counts],
  )

  return api
}
