import { useEffect, useRef, useState } from 'react'
import type { ElementInfo } from './api'
import { fetchPointCloudElementMeta, updatePointCloudSettings } from './api'

type Props = {
  selected: ElementInfo | null
}

export default function Inspector({ selected }: Props) {
  const [pointSize, setPointSize] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const lastSent = useRef<number | null>(null)

  const isPointCloud = selected?.type === 'pointcloud'

  useEffect(() => {
    setErr(null)
    lastSent.current = null
    if (!selected || !isPointCloud) {
      setPointSize(null)
      return
    }

    fetchPointCloudElementMeta(selected.id)
      .then((m) => setPointSize(m.pointSize))
      .catch((e: unknown) => setErr(e instanceof Error ? e.message : String(e)))
  }, [selected?.id, isPointCloud])

  useEffect(() => {
    if (!selected || !isPointCloud || pointSize === null) return

    const v = pointSize
    const handle = window.setTimeout(async () => {
      if (lastSent.current !== null && Math.abs(lastSent.current - v) < 1e-9) return

      setBusy(true)
      setErr(null)
      try {
        await updatePointCloudSettings(selected.id, { pointSize: v })
        lastSent.current = v
      } catch (e: unknown) {
        setErr(e instanceof Error ? e.message : String(e))
      } finally {
        setBusy(false)
      }
    }, 200)

    return () => window.clearTimeout(handle)
  }, [pointSize, selected?.id, isPointCloud])

  if (!selected) {
    return (
      <div style={{ padding: 12, borderLeft: '1px solid #1b2235', width: 280 }}>
        <strong>Inspector</strong>
        <div style={{ marginTop: 8, opacity: 0.75 }}>Select an element.</div>
      </div>
    )
  }

  return (
    <div style={{ padding: 12, borderLeft: '1px solid #1b2235', width: 280 }}>
      <strong>Inspector</strong>
      <div style={{ marginTop: 10, fontSize: 12, opacity: 0.8 }}>id</div>
      <div style={{ fontFamily: 'monospace', fontSize: 12, wordBreak: 'break-all' }}>{selected.id}</div>

      <div style={{ marginTop: 10, fontSize: 12, opacity: 0.8 }}>type</div>
      <div>{selected.type}</div>

      <div style={{ marginTop: 10, fontSize: 12, opacity: 0.8 }}>name</div>
      <div>{selected.name}</div>

      {isPointCloud && (
        <div style={{ marginTop: 14 }}>
          <label style={{ display: 'block', fontSize: 12, opacity: 0.9 }}>Point size</label>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 6 }}>
            <input
              type="range"
              min={0.001}
              max={3}
              step={0.001}
              value={pointSize ?? 0.02}
              onChange={(e) => setPointSize(parseFloat(e.target.value))}
              style={{ flex: 1 }}
            />
            <input
              type="number"
              min={0.0001}
              max={3}
              step={0.001}
              value={pointSize ?? 0.02}
              onChange={(e) => setPointSize(parseFloat(e.target.value))}
              style={{ width: 90 }}
            />
          </div>
          <div style={{ marginTop: 8, fontSize: 12, opacity: 0.75 }}>{busy ? 'Updating…' : ' '}</div>
        </div>
      )}

      {err && <div style={{ marginTop: 10, color: 'crimson', fontSize: 12 }}>{err}</div>}
    </div>
  )
}
