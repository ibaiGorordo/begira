import type { ElementInfo } from './api'

export type HierarchyViewInfo = {
  id: string
  name: string
  kind: '3d' | 'image'
  visible: boolean
  canDelete: boolean
  canReorder: boolean
  elementIds: string[]
}

export type HierarchyProps = {
  elements: ElementInfo[] | null
  selectedId: string | null
  onSelect: (id: string | null) => void
  onFocus: (id: string) => void
  onToggleVisibility: (id: string, visible: boolean) => void
  onDelete: (id: string) => void
  onAddCamera: () => void
  views: HierarchyViewInfo[]
  onActivateView: (viewId: string) => void
  onToggleViewVisibility: (viewId: string, visible: boolean) => void
  onDeleteView: (viewId: string) => void
  onMoveView: (viewId: string, direction: 'up' | 'down') => void
}

export default function Hierarchy({
  elements,
  selectedId,
  onSelect,
  onFocus,
  onToggleVisibility,
  onDelete,
  onAddCamera,
  views,
  onActivateView,
  onToggleViewVisibility,
  onDeleteView,
  onMoveView,
}: HierarchyProps) {
  const items = elements ?? []
  const byId = new Map(items.map((e) => [e.id, e]))
  const reorderableViewIds = views.filter((v) => v.canReorder).map((v) => v.id)

  return (
    <div style={{ padding: 12, width: '100%', boxSizing: 'border-box' }}>
      <strong>Scene</strong>
      <div style={{ marginTop: 8 }}>
        <button
          onClick={onAddCamera}
          style={{
            padding: '6px 8px',
            borderRadius: 6,
            border: '1px solid #1b2235',
            background: '#0f1630',
            color: '#e8ecff',
            cursor: 'pointer',
            fontSize: 12,
          }}
        >
          Add Camera
        </button>
      </div>

      <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
        {views.map((view) => {
          const reorderIndex = reorderableViewIds.indexOf(view.id)
          const isFirstReorderable = reorderIndex <= 0
          const isLastReorderable = reorderIndex < 0 || reorderIndex >= reorderableViewIds.length - 1

          return (
            <div key={view.id} style={{ border: '1px solid #1b2235', borderRadius: 8, background: '#0f1630' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: 6 }}>
                <input
                  type="checkbox"
                  checked={view.visible}
                  title={view.visible ? 'Hide view' : 'Show view'}
                  onChange={(e) => onToggleViewVisibility(view.id, e.target.checked)}
                />
                <button
                  type="button"
                  onClick={() => onActivateView(view.id)}
                  style={{
                    flex: 1,
                    textAlign: 'left',
                    padding: '6px 8px',
                    borderRadius: 6,
                    border: '1px solid #1b2235',
                    background: '#172242',
                    color: '#e8ecff',
                    cursor: 'pointer',
                  }}
                  title={view.id}
                >
                  <div style={{ fontSize: 13, fontWeight: 700 }}>{view.name}</div>
                  <div style={{ fontSize: 11, opacity: 0.55, textTransform: 'uppercase' }}>{view.kind} view</div>
                </button>
                <button
                  type="button"
                  title="Move up"
                  disabled={!view.canReorder || isFirstReorderable}
                  onClick={() => onMoveView(view.id, 'up')}
                  style={{ padding: '2px 6px', minWidth: 26 }}
                >
                  ↑
                </button>
                <button
                  type="button"
                  title="Move down"
                  disabled={!view.canReorder || isLastReorderable}
                  onClick={() => onMoveView(view.id, 'down')}
                  style={{ padding: '2px 6px', minWidth: 26 }}
                >
                  ↓
                </button>
                <button
                  type="button"
                  title="Delete view"
                  disabled={!view.canDelete}
                  onClick={() => onDeleteView(view.id)}
                  style={{ padding: '2px 6px', minWidth: 26 }}
                >
                  ×
                </button>
              </div>

              <div style={{ borderTop: '1px solid #1b2235', padding: '6px 6px 6px 16px', display: 'grid', gap: 6 }}>
                {view.elementIds.map((elementId) => {
                  const element = byId.get(elementId)
                  if (!element) {
                    return (
                      <div key={elementId} style={{ fontSize: 12, opacity: 0.6 }}>
                        (missing element)
                      </div>
                    )
                  }

                  const selected = selectedId === element.id
                  const isVisible = element.visible !== false
                  const count = Number((element.summary as any)?.pointCount ?? (element.summary as any)?.count ?? 0)
                  const unit = element.type === 'pointcloud' ? 'pts' : element.type === 'gaussians' ? 'splats' : ''
                  const imageW = Number((element.summary as any)?.width ?? 0)
                  const imageH = Number((element.summary as any)?.height ?? 0)

                  return (
                    <div key={element.id} style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                      <input
                        type="checkbox"
                        title={isVisible ? 'Hide element' : 'Show element'}
                        checked={isVisible}
                        onChange={(e) => onToggleVisibility(element.id, e.target.checked)}
                      />
                      <button
                        onClick={() => onSelect(element.id)}
                        onDoubleClick={() => onFocus(element.id)}
                        style={{
                          flex: 1,
                          textAlign: 'left',
                          padding: '7px 8px',
                          borderRadius: 6,
                          border: '1px solid #1b2235',
                          background: selected ? '#172242' : '#0f1630',
                          color: '#e8ecff',
                          cursor: 'pointer',
                        }}
                        title={element.id}
                      >
                        <div style={{ fontSize: 13, fontWeight: 600 }}>{element.name}</div>
                        <div style={{ fontSize: 11, opacity: 0.55, textTransform: 'uppercase' }}>{element.type}</div>
                        {element.type !== 'camera' && element.type !== 'image' && (
                          <div style={{ fontSize: 12, opacity: 0.75 }}>{count.toLocaleString()} {unit}</div>
                        )}
                        {element.type === 'image' && (
                          <div style={{ fontSize: 12, opacity: 0.75 }}>
                            {imageW > 0 && imageH > 0 ? `${imageW}x${imageH}` : 'image'}
                          </div>
                        )}
                      </button>
                      <button
                        onClick={() => onDelete(element.id)}
                        title="Remove element"
                        style={{ padding: '2px 6px', minWidth: 26 }}
                      >
                        ×
                      </button>
                    </div>
                  )
                })}
                {view.elementIds.length === 0 && <div style={{ fontSize: 12, opacity: 0.6 }}>(no elements)</div>}
              </div>
            </div>
          )
        })}
      </div>

      {views.length === 0 && <div style={{ marginTop: 8, opacity: 0.75 }}>(no views)</div>}

      <div style={{ marginTop: 14, fontSize: 12, opacity: 0.65 }}>Tip: Double-click an image element to activate its linked 2D view.</div>
    </div>
  )
}
