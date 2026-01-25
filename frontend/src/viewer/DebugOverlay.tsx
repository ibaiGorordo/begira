import { Html } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { useEffect, useMemo, useRef, useState } from 'react'

export function isDebugOverlayEnabledFromUrl(url: Location | URL = window.location): boolean {
  const search = url instanceof URL ? url.search : url.search
  const params = new URLSearchParams(search)
  const v = params.get('debug')
  if (v === null) return false
  return v === '' || v === '1' || v.toLowerCase() === 'true'
}

type OverlayStats = {
  fps: number
  ms: number
  drawCalls: number
  triangles: number
  points: number
  lines: number
  geometries: number
  textures: number
  programs: number | null
  dpr: number
  renderer: 'webgl' | 'unknown'
  vendor: string | null
  device: string | null
  jsHeapMB: number | null
  jsHeapLimitMB: number | null
}

function formatMB(mb: number | null): string {
  if (mb === null || !Number.isFinite(mb)) return 'n/a'
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`
  return `${mb.toFixed(0)} MB`
}

function safeRound(n: number): number {
  if (!Number.isFinite(n)) return 0
  return Math.round(n)
}

function getJsHeapStats(): { usedMB: number | null; limitMB: number | null } {
  const anyPerf = performance as unknown as {
    memory?: { usedJSHeapSize: number; jsHeapSizeLimit: number }
  }
  const mem = anyPerf.memory
  if (!mem) return { usedMB: null, limitMB: null }
  const usedMB = mem.usedJSHeapSize / (1024 * 1024)
  const limitMB = mem.jsHeapSizeLimit / (1024 * 1024)
  return {
    usedMB: Number.isFinite(usedMB) ? usedMB : null,
    limitMB: Number.isFinite(limitMB) ? limitMB : null,
  }
}

function getUnmaskedRendererInfo(gl: WebGLRenderingContext | WebGL2RenderingContext): {
  vendor: string | null
  renderer: string | null
} {
  // Works on Chrome/Firefox; not always available on Safari.
  const ext = gl.getExtension('WEBGL_debug_renderer_info') as any
  if (!ext) return { vendor: null, renderer: null }

  try {
    const vendor = gl.getParameter(ext.UNMASKED_VENDOR_WEBGL)
    const renderer = gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)
    return {
      vendor: typeof vendor === 'string' ? vendor : null,
      renderer: typeof renderer === 'string' ? renderer : null,
    }
  } catch {
    return { vendor: null, renderer: null }
  }
}

export default function DebugOverlay({ enabled }: { enabled: boolean }) {
  const { gl } = useThree()

  // Only sample when enabled to avoid any overhead.
  const [stats, setStats] = useState<OverlayStats | null>(null)

  // FPS derived from an exponentially-weighted moving average of frame delta.
  const emaDtMs = useRef<number | null>(null)
  const lastPublishMs = useRef<number>(0)

  const staticInfo = useMemo(() => {
    const renderer = (gl as any)?.isWebGLRenderer ? ('webgl' as const) : ('unknown' as const)
    const ctx = (gl as any).getContext?.() as WebGLRenderingContext | WebGL2RenderingContext | null | undefined

    const dpr = typeof gl.getPixelRatio === 'function' ? gl.getPixelRatio() : window.devicePixelRatio

    let vendor: string | null = null
    let device: string | null = null

    if (ctx) {
      const info = getUnmaskedRendererInfo(ctx)
      vendor = info.vendor
      device = info.renderer
    }

    return { renderer, vendor, device, dpr }
  }, [gl])

  useEffect(() => {
    if (!enabled) {
      setStats(null)
      emaDtMs.current = null
      return
    }
  }, [enabled])

  useFrame((_, dtSeconds) => {
    if (!enabled) return

    const dtMs = dtSeconds * 1000
    const alpha = 0.08 // smoothing factor

    if (emaDtMs.current === null) emaDtMs.current = dtMs
    else emaDtMs.current = emaDtMs.current * (1 - alpha) + dtMs * alpha

    const now = performance.now()
    // Update UI at ~4Hz to keep React overhead low.
    if (now - lastPublishMs.current < 250) return
    lastPublishMs.current = now

    const info = (gl as THREE.WebGLRenderer).info
    const render = info?.render
    const memory = info?.memory

    const jsHeap = getJsHeapStats()

    const next: OverlayStats = {
      fps: emaDtMs.current > 0 ? 1000 / emaDtMs.current : 0,
      ms: emaDtMs.current ?? 0,
      drawCalls: safeRound(render?.calls ?? 0),
      triangles: safeRound(render?.triangles ?? 0),
      points: safeRound(render?.points ?? 0),
      lines: safeRound(render?.lines ?? 0),
      geometries: safeRound(memory?.geometries ?? 0),
      textures: safeRound(memory?.textures ?? 0),
      programs: Array.isArray((gl as any).info?.programs) ? (gl as any).info.programs.length : null,
      dpr: staticInfo.dpr,
      renderer: staticInfo.renderer,
      vendor: staticInfo.vendor,
      device: staticInfo.device,
      jsHeapMB: jsHeap.usedMB,
      jsHeapLimitMB: jsHeap.limitMB,
    }

    setStats(next)
  })

  // LOD stats state (updated along with renderer stats)
  const [lodCounts, setLodCounts] = useState<{ high: number; medium: number; low: number }>({ high: 0, medium: 0, low: 0 })
  const [globalOverride, setGlobalOverride] = useState<string | undefined>((window as any).__begira_lod_override_global)
  // helper to fetch LOD table from window
  const sampleLodStatus = () => {
    try {
      const anyWin = window as any
      const table = anyWin.__begira_lod_status || {}
      let high = 0
      let medium = 0
      let low = 0
      for (const k in table) {
        const v = table[k]
        if (v === 'high') high++
        else if (v === 'medium') medium++
        else low++
      }
      setLodCounts({ high, medium, low })
      // keep local globalOverride state in sync with external console changes
      setGlobalOverride((anyWin.__begira_lod_override_global as string) ?? undefined)
    } catch {
      setLodCounts({ high: 0, medium: 0, low: 0 })
    }
  }

  // Also update LOD counts on the same cadence as stats
  useFrame(() => {
    if (!enabled) return
    // inexpensive - call sampleLodStatus at the same 4Hz rhythm
    sampleLodStatus()
  })

  if (!enabled || !stats) return null

  return (
    <Html
      // Fullscreen overlay anchored to top-left.
      fullscreen
      style={{
        pointerEvents: 'none',
        fontFamily:
          'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
        fontSize: 12,
        lineHeight: 1.35,
        color: '#e8ecff',
      }}
    >
      <div
        style={{
          position: 'absolute',
          top: 10,
          left: 10,
          padding: '8px 10px',
          borderRadius: 8,
          background: 'rgba(11, 16, 32, 0.75)',
          border: '1px solid rgba(41, 50, 74, 0.7)',
          backdropFilter: 'blur(6px)',
          WebkitBackdropFilter: 'blur(6px)',
          whiteSpace: 'pre',
        }}
      >
        <div style={{ fontWeight: 700, marginBottom: 6 }}>begira debug (toggle: press ` | URL: ?debug=1)</div>
        <div>FPS: {stats.fps.toFixed(1)} ({stats.ms.toFixed(2)} ms)</div>
        <div>Calls: {stats.drawCalls}  Tris: {stats.triangles}  Points: {stats.points}  Lines: {stats.lines}</div>
        <div>Geoms: {stats.geometries}  Tex: {stats.textures}  Programs: {stats.programs ?? 'n/a'}</div>
        <div>JS heap: {formatMB(stats.jsHeapMB)} / {formatMB(stats.jsHeapLimitMB)}</div>
        <div>Renderer: {stats.renderer}  DPR: {stats.dpr.toFixed(2)}</div>
        <div>GPU: {stats.vendor ?? 'n/a'} | {stats.device ?? 'n/a'}</div>

        <div style={{ marginTop: 6, display: 'flex', gap: 12, alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ fontWeight: 600 }}>LOD counts:</span>
            <span style={{ color: '#6ee7b7' }}>high: {lodCounts.high}</span>
            <span style={{ color: '#facc15' }}>medium: {lodCounts.medium}</span>
            <span style={{ color: '#f87171' }}>low: {lodCounts.low}</span>
          </div>

          <div style={{ marginLeft: 'auto', pointerEvents: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
            {/** Buttons show active state with color -- auto/high/medium/low **/}
            {(() => {
              const btnCommon: React.CSSProperties = { padding: '4px 8px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.06)', cursor: 'pointer', background: 'transparent', color: '#e8ecff' }
              const makeBtn = (label: string, value: string | undefined, color?: string) => {
                const active = globalOverride === value || (value === undefined && typeof globalOverride === 'undefined')
                const style: React.CSSProperties = { ...btnCommon }
                if (active) style.background = color ?? 'rgba(255,255,255,0.08)'
                return (
                  <button
                    key={label}
                    onClick={() => {
                      try {
                        ;(window as any).__begira_lod_override_global = value
                      } catch {}
                      setGlobalOverride(value)
                    }}
                    style={style}
                  >
                    {label}
                  </button>
                )
              }

              return (
                <>
                  {makeBtn('auto', undefined)}
                  {makeBtn('high', 'high', 'rgba(110,231,183,0.18)')}
                  {makeBtn('medium', 'medium', 'rgba(250,204,21,0.14)')}
                  {makeBtn('low', 'low', 'rgba(248,113,113,0.14)')}
                </>
              )
            })()}

            <button
              onClick={() => {
                try {
                  ;(window as any).__begira_lod_status = {}
                  ;(window as any).__begira_lod_override = {}
                  ;(window as any).__begira_lod_override_global = undefined
                } catch {}
                setGlobalOverride(undefined)
              }}
              style={{ marginLeft: 8, padding: '4px 8px', borderRadius: 6, cursor: 'pointer', background: 'transparent', color: '#e8ecff', border: '1px solid rgba(255,255,255,0.06)' }}
            >
              clear per-elem
            </button>
          </div>
        </div>
      </div>
    </Html>
  )
}
