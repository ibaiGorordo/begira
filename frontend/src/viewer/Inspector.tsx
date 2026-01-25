import { useEffect, useRef, useState } from 'react'
import type { ElementInfo } from './api'
import { fetchPointCloudElementMeta, fetchGaussianElementMeta, updatePointCloudSettings } from './api'

type Props = {
  selected: ElementInfo | null
}

export default function Inspector({ selected }: Props) {
  const [pointSize, setPointSize] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const lastSent = useRef<number | null>(null)

  const isPointCloud = selected?.type === 'pointcloud'
  const isGaussians = selected?.type === 'gaussians'

  // LOD override state
  const [lodOverride, setLodOverride] = useState<string | undefined>(undefined)

  useEffect(() => {
    setErr(null)
    lastSent.current = null
    if (!selected) {
      setPointSize(null)
      return
    }

    const fetchMeta = isPointCloud ? fetchPointCloudElementMeta : isGaussians ? fetchGaussianElementMeta : null
    if (!fetchMeta) {
      setPointSize(null)
      return
    }

    fetchMeta(selected.id)
      .then((m: any) => setPointSize(m.pointSize))
      .catch((e: unknown) => setErr(e instanceof Error ? e.message : String(e)))

    // initialize client-side values from window globals if present
    try {
      const anyWin = window as any
      const lod = (anyWin.__begira_lod_override && anyWin.__begira_lod_override[selected.id]) ?? undefined
      setLodOverride(lod)
    } catch {}
  }, [selected?.id, isPointCloud, isGaussians])

  useEffect(() => {
    if (!selected || pointSize === null) return
    if (!isPointCloud && !isGaussians) return

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
  }, [pointSize, selected?.id, isPointCloud, isGaussians])

  // Handlers for the new UI that update window globals the scene reads
  const setLodOverrideForElement = (id: string, v: string | undefined) => {
    try {
      const anyWin = window as any
      anyWin.__begira_lod_override = anyWin.__begira_lod_override || {}
      if (v === undefined) delete anyWin.__begira_lod_override[id]
      else anyWin.__begira_lod_override[id] = v
      // Also publish to apply function to ensure visibility updates (no-op, kept for backward compat)
    } catch {}
  }

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

      {isPointCloud && selected.summary?.pointCount !== undefined && (
        <>
          <div style={{ marginTop: 10, fontSize: 12, opacity: 0.8 }}>points</div>
          <div>{selected.summary.pointCount as number}</div>
        </>
      )}

      {isGaussians && selected.summary?.count !== undefined && (
        <>
          <div style={{ marginTop: 10, fontSize: 12, opacity: 0.8 }}>gaussians</div>
          <div>{selected.summary.count as number}</div>
        </>
      )}

      {isPointCloud && (
        <div style={{ marginTop: 14 }}>
          <label style={{ display: 'block', fontSize: 12, opacity: 0.9 }}>
            {isPointCloud ? 'Point size' : 'Scaling'}
          </label>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 6 }}>
            <input
              type="range"
              min={isPointCloud ? 0.001 : 0.01}
              max={isPointCloud ? 3 : 10}
              step={0.001}
              value={pointSize ?? (isPointCloud ? 0.02 : 1.0)}
              onChange={(e) => setPointSize(parseFloat(e.target.value))}
              style={{ flex: 1 }}
            />
            <input
              type="number"
              min={0.0001}
              max={100}
              step={0.01}
              value={pointSize ?? (isPointCloud ? 0.02 : 1.0)}
              onChange={(e) => setPointSize(parseFloat(e.target.value))}
              style={{ width: 90 }}
            />
          </div>
          <div style={{ marginTop: 8, fontSize: 12, opacity: 0.75 }}>{busy ? 'Updating…' : ' '}</div>
        </div>
      )}

      {/* Client-side controls for gaussians */}
      {isGaussians && (
        <div style={{ marginTop: 14 }}>
          <label style={{ display: 'block', fontSize: 12, opacity: 0.9 }}>Client controls</label>

          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 12, opacity: 0.8 }}>LOD override</div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 6 }}>
              {(['auto', 'high', 'medium', 'low'] as const).map((v) => {
                const active = (v === 'auto' && !lodOverride) || (lodOverride === v)
                return (
                  <button
                    key={v}
                    onClick={() => {
                      const val = v === 'auto' ? undefined : v
                      setLodOverride(val)
                      setLodOverrideForElement(selected.id, val as any)
                    }}
                    style={{
                      padding: '6px 8px',
                      borderRadius: 6,
                      border: '1px solid rgba(255,255,255,0.06)',
                      background: active ? 'rgba(255,255,255,0.06)' : 'transparent',
                      color: '#e8ecff',
                      cursor: 'pointer',
                    }}
                  >
                    {v}
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      )}

      {err && <div style={{ marginTop: 10, color: 'crimson', fontSize: 12 }}>{err}</div>}
    </div>
  )
}
