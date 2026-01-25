import { useEffect, useRef, useState } from 'react'
import type { ElementInfo } from './api'
import { fetchPointCloudElementMeta, fetchGaussianElementMeta, updatePointCloudSettings } from './api'
import { COLORMAPS, DEFAULT_DEPTH_COLORMAP, DEFAULT_HEIGHT_COLORMAP, type ColormapId } from './colormaps'

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

  // Visual override state (client-only preview)
  type ColorMode = 'logged' | 'solid' | 'height' | 'depth'
  const [colorMode, setColorMode] = useState<ColorMode>('logged')
  const [solidColor, setSolidColor] = useState<string>('#ff8a33') // hex string for input[type=color]
  const [colorMap, setColorMap] = useState<ColormapId>(DEFAULT_HEIGHT_COLORMAP)
  const [isVisible, setIsVisible] = useState(true)

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
      // visual override (client-only)
      const vis = anyWin.__begira_visual_override && anyWin.__begira_visual_override[selected.id]
      if (vis) {
        const mode = (vis.colorMode ?? 'logged') as ColorMode
        setColorMode(mode)
        const fallback =
          mode === 'depth' ? DEFAULT_DEPTH_COLORMAP : mode === 'height' ? DEFAULT_HEIGHT_COLORMAP : DEFAULT_HEIGHT_COLORMAP
        setColorMap((vis.colorMap as ColormapId) ?? fallback)
        if (vis.solidColor) {
          // assume [r,g,b] floats 0..1
          const c = vis.solidColor
          function toHex(x: number) {
            const v = Math.round(Math.max(0, Math.min(1, x)) * 255)
            return v.toString(16).padStart(2, '0')
          }
          const hex = '#' + toHex(c[0]) + toHex(c[1]) + toHex(c[2])
          setSolidColor(hex)
        }
      } else {
        setColorMode('logged')
        setColorMap(DEFAULT_HEIGHT_COLORMAP)
      }
      const visibilityMap = anyWin.__begira_visibility || {}
      setIsVisible(visibilityMap[selected.id] !== false)
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

  const setVisualOverrideForElement = (
    id: string,
    mode: ColorMode,
    hexColor?: string | null,
    mapId: ColormapId = DEFAULT_HEIGHT_COLORMAP,
  ) => {
    try {
      const anyWin = window as any
      anyWin.__begira_visual_override = anyWin.__begira_visual_override || {}
      if (mode === 'logged') {
        // clear override
        delete anyWin.__begira_visual_override[id]
      } else {
        const obj: any = { colorMode: mode }
        if (mode === 'solid' && hexColor) {
          // convert #rrggbb to [r,g,b] floats
          const c = hexColor.replace('#', '')
          const r = parseInt(c.substring(0, 2), 16) / 255
          const g = parseInt(c.substring(2, 4), 16) / 255
          const b = parseInt(c.substring(4, 6), 16) / 255
          obj.solidColor = [r, g, b]
        }
        if (mode === 'height' || mode === 'depth') {
          obj.colorMap = mapId
        }
        anyWin.__begira_visual_override[id] = obj
      }
      // notify viewers to re-read overrides and re-render
      try {
        if (typeof window !== 'undefined' && (window as any).dispatchEvent) {
          const ev = new CustomEvent('begira_visual_override_changed', { detail: { id } })
          window.dispatchEvent(ev)
        }
      } catch {}
    } catch {}
  }

  const setVisibilityForElement = (id: string, visible: boolean) => {
    try {
      const anyWin = window as any
      anyWin.__begira_visibility = anyWin.__begira_visibility || {}
      anyWin.__begira_visibility[id] = visible
      setIsVisible(visible)
      try {
        const ev = new CustomEvent('begira_visibility_changed', { detail: { id } })
        window.dispatchEvent(ev)
      } catch {}
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

      {(isPointCloud || isGaussians) && (
        <div style={{ marginTop: 14 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, opacity: 0.9 }}>
            <input
              type="checkbox"
              checked={isVisible}
              onChange={(e) => setVisibilityForElement(selected.id, e.target.checked)}
            />
            Visible
          </label>
        </div>
      )}

      {(isPointCloud || isGaussians) && (
        <div style={{ marginTop: 14 }}>
          <label style={{ display: 'block', fontSize: 12, opacity: 0.9 }}>Color</label>

          <div style={{ marginTop: 8, fontSize: 12, opacity: 0.8 }}>Mode</div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 6, flexWrap: 'wrap' }}>
            {(['logged', 'solid', 'height', 'depth'] as ColorMode[]).map((v) => {
              const active = colorMode === v
              return (
                <button
                  key={v}
                  onClick={() => {
                    setColorMode(v)
                    if (v === 'height') {
                      setColorMap(DEFAULT_HEIGHT_COLORMAP)
                      setVisualOverrideForElement(selected.id, v, solidColor, DEFAULT_HEIGHT_COLORMAP)
                      return
                    }
                    if (v === 'depth') {
                      setColorMap(DEFAULT_DEPTH_COLORMAP)
                      setVisualOverrideForElement(selected.id, v, solidColor, DEFAULT_DEPTH_COLORMAP)
                      return
                    }
                    setVisualOverrideForElement(selected.id, v, solidColor, colorMap)
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

          {colorMode === 'solid' && (
            <div style={{ marginTop: 10, display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                aria-label="Solid color"
                type="color"
                value={solidColor}
                onChange={(e) => {
                  setSolidColor(e.target.value)
                  setVisualOverrideForElement(selected.id, 'solid', e.target.value, colorMap)
                }}
              />
              <div style={{ fontSize: 12, opacity: 0.8 }}>{solidColor}</div>
            </div>
          )}

          {(colorMode === 'height' || colorMode === 'depth') && (
            <div style={{ marginTop: 12 }}>
              <label style={{ display: 'block', fontSize: 12, opacity: 0.85 }}>Colormap</label>
              <select
                value={colorMap}
                onChange={(e) => {
                  const next = e.target.value as ColormapId
                  setColorMap(next)
                  setVisualOverrideForElement(selected.id, colorMode, solidColor, next)
                }}
                style={{
                  marginTop: 6,
                  width: '100%',
                  background: '#0f1630',
                  color: '#e8ecff',
                  border: '1px solid #1b2235',
                  borderRadius: 6,
                  padding: '6px 8px',
                }}
              >
                {COLORMAPS.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.label}
                  </option>
                ))}
              </select>
            </div>
          )}
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
