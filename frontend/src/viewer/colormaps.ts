export type ColormapId = 'viridis' | 'plasma' | 'inferno' | 'magma' | 'turbo'

export type ColormapStop = [number, number, number, number]

export type ColormapDef = {
  id: ColormapId
  label: string
  stops: ColormapStop[]
}

export const DEFAULT_HEIGHT_COLORMAP: ColormapId = 'turbo'
export const DEFAULT_DEPTH_COLORMAP: ColormapId = 'plasma'

export const COLORMAPS: ColormapDef[] = [
  {
    id: 'viridis',
    label: 'Viridis',
    stops: [
      [0.0, 0.267, 0.005, 0.329],
      [0.25, 0.283, 0.141, 0.458],
      [0.5, 0.254, 0.265, 0.53],
      [0.75, 0.207, 0.372, 0.553],
      [1.0, 0.993, 0.906, 0.144],
    ],
  },
  {
    id: 'plasma',
    label: 'Plasma',
    stops: [
      [0.0, 0.051, 0.029, 0.528],
      [0.25, 0.373, 0.055, 0.649],
      [0.5, 0.678, 0.159, 0.507],
      [0.75, 0.902, 0.371, 0.212],
      [1.0, 0.94, 0.975, 0.131],
    ],
  },
  {
    id: 'inferno',
    label: 'Inferno',
    stops: [
      [0.0, 0.002, 0.005, 0.014],
      [0.25, 0.296, 0.039, 0.329],
      [0.5, 0.588, 0.185, 0.287],
      [0.75, 0.894, 0.411, 0.15],
      [1.0, 0.988, 0.998, 0.645],
    ],
  },
  {
    id: 'magma',
    label: 'Magma',
    stops: [
      [0.0, 0.001, 0.0, 0.014],
      [0.25, 0.251, 0.049, 0.318],
      [0.5, 0.541, 0.174, 0.446],
      [0.75, 0.843, 0.349, 0.384],
      [1.0, 0.987, 0.991, 0.749],
    ],
  },
  {
    id: 'turbo',
    label: 'Turbo',
    stops: [
      [0.0, 0.19, 0.071, 0.232],
      [0.25, 0.0, 0.604, 0.984],
      [0.5, 0.251, 0.99, 0.415],
      [0.75, 0.988, 0.822, 0.142],
      [1.0, 0.694, 0.0, 0.149],
    ],
  },
]

export const COLORMAP_IDS = COLORMAPS.map((c) => c.id)

export const getColormap = (id: ColormapId): ColormapDef => {
  return COLORMAPS.find((c) => c.id === id) ?? COLORMAPS[0]
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, v))

const sampleStops = (stops: ColormapStop[], t: number): [number, number, number] => {
  const u = clamp01(t)
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, r0, g0, b0] = stops[i]
    const [t1, r1, g1, b1] = stops[i + 1]
    if (u >= t0 && u <= t1) {
      const span = Math.max(1e-6, t1 - t0)
      const k = (u - t0) / span
      return [r0 * (1 - k) + r1 * k, g0 * (1 - k) + g1 * k, b0 * (1 - k) + b1 * k]
    }
  }
  const last = stops[stops.length - 1]
  return [last[1], last[2], last[3]]
}

export const sampleColormap = (id: ColormapId, t: number): [number, number, number] => {
  return sampleStops(getColormap(id).stops, t)
}

export const buildColormapLUT = (id: ColormapId, size = 256): Float32Array => {
  const lut = new Float32Array(size * 3)
  for (let i = 0; i < size; i++) {
    const t = size === 1 ? 0 : i / (size - 1)
    const [r, g, b] = sampleColormap(id, t)
    const offset = i * 3
    lut[offset] = r
    lut[offset + 1] = g
    lut[offset + 2] = b
  }
  return lut
}

export const sampleColormapLUT = (lut: Float32Array, t: number): [number, number, number] => {
  const size = Math.max(1, Math.floor(lut.length / 3))
  const idx = Math.min(size - 1, Math.max(0, Math.floor(t * (size - 1))))
  const offset = idx * 3
  return [lut[offset], lut[offset + 1], lut[offset + 2]]
}
