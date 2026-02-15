import type { ElementInfo } from '../shared/api'
import { buildHierarchyElementDragPayload, HIERARCHY_DRAG_MIME } from './dragPayload'

export type HierarchyViewInfo = {
  id: string
  name: string
  kind: '3d' | 'image' | 'camera'
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
  views: HierarchyViewInfo[]
  onActivateView: (viewId: string) => void
  onToggleViewVisibility: (viewId: string, visible: boolean) => void
  onDeleteView: (viewId: string) => void
  onMoveView: (viewId: string, direction: 'up' | 'down') => void
  onAddCamera: () => void
  onAdd3DView: () => void
}

type IconKind = 'view3d' | 'view2d' | 'pointcloud' | 'gaussians' | 'camera' | 'image' | 'box3d' | 'ellipsoid3d'

function iconKindForElement(type: ElementInfo['type']): IconKind {
  if (type === 'pointcloud') return 'pointcloud'
  if (type === 'gaussians') return 'gaussians'
  if (type === 'camera') return 'camera'
  if (type === 'box3d') return 'box3d'
  if (type === 'ellipsoid3d') return 'ellipsoid3d'
  return 'image'
}

function MinimalIcon({ kind }: { kind: IconKind }) {
  if (kind === 'view3d') {
    return (
      <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
        <path d="M9.8 10.2 L15.8 10.2" stroke="#cc6b64" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M9.8 10.2 L9.8 4.2" stroke="#4f9f62" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M9.8 10.2 L5.3 14.6" stroke="#4d74b8" strokeWidth="1.5" strokeLinecap="round" />
        <circle cx="9.8" cy="10.2" r="1.2" fill="#4b5f7f" />
      </svg>
    )
  }
  if (kind === 'view2d') {
    return (
      <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
        <path d="M4 15.5 H16" stroke="#4d74b8" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M4.5 16 V4" stroke="#4f9f62" strokeWidth="1.5" strokeLinecap="round" />
        <circle cx="4.5" cy="15.5" r="1.2" fill="#4b5f7f" />
      </svg>
    )
  }
  if (kind === 'pointcloud') {
    return (
      <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
        <circle cx="6" cy="12.5" r="1.4" fill="#4d74b8" />
        <circle cx="9.4" cy="9.8" r="1.4" fill="#5a7fbe" />
        <circle cx="12.7" cy="12" r="1.4" fill="#688ac5" />
        <circle cx="8.3" cy="6.4" r="1.4" fill="#7395cc" />
        <circle cx="13.8" cy="7.4" r="1.4" fill="#7fa0d2" />
      </svg>
    )
  }
  if (kind === 'gaussians') {
    return (
      <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
        <ellipse cx="6.8" cy="12.6" rx="2.45" ry="1.65" transform="rotate(-20 6.8 12.6)" fill="#8e73c6" />
        <ellipse cx="10.1" cy="7.2" rx="2.6" ry="1.75" transform="rotate(14 10.1 7.2)" fill="#5f8fcb" />
        <ellipse cx="13.3" cy="12.1" rx="2.4" ry="1.6" transform="rotate(28 13.3 12.1)" fill="#5ca985" />
      </svg>
    )
  }
  if (kind === 'camera') {
    return (
      <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
        <rect x="3.6" y="6.9" width="12.8" height="8.4" rx="2" stroke="#3f6fa3" strokeWidth="1.2" />
        <path d="M7.1 6.9 L8.4 5.3 H11.6 L12.9 6.9" stroke="#3f6fa3" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx="10" cy="11.1" r="2.2" stroke="#3f6fa3" strokeWidth="1.2" />
        <circle cx="14.1" cy="8.6" r="0.8" fill="#3f6fa3" />
      </svg>
    )
  }
  if (kind === 'box3d') {
    return (
      <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
        <path d="M5.6 7.3 L10 4.8 L14.4 7.3 L10 9.8 L5.6 7.3 Z" stroke="#4d74b8" strokeWidth="1.2" strokeLinejoin="round" />
        <path d="M5.6 7.3 V12.8 L10 15.2 L14.4 12.8 V7.3" stroke="#4d74b8" strokeWidth="1.2" strokeLinejoin="round" />
        <path d="M10 9.8 V15.2" stroke="#4d74b8" strokeWidth="1.2" />
      </svg>
    )
  }
  if (kind === 'ellipsoid3d') {
    return (
      <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
        <ellipse cx="10" cy="10" rx="5.5" ry="3.6" stroke="#7e64bf" strokeWidth="1.2" />
        <ellipse cx="10" cy="10" rx="3.1" ry="5.2" stroke="#7e64bf" strokeWidth="1.2" />
        <ellipse cx="10" cy="10" rx="5.2" ry="2.2" transform="rotate(32 10 10)" stroke="#7e64bf" strokeWidth="1.2" />
      </svg>
    )
  }
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <rect x="3.8" y="4.4" width="12.4" height="11.2" rx="2" stroke="#8a6a2f" strokeWidth="1.3" />
      <path d="M5.8 13.2 L8.8 9.9 L11.3 12.1 L13.7 9.4 L16.1 12.8" stroke="#8a6a2f" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="8" cy="7.5" r="1.1" fill="#8a6a2f" />
    </svg>
  )
}

export default function Hierarchy({
  elements,
  selectedId,
  onSelect,
  onFocus,
  onToggleVisibility,
  onDelete,
  views,
  onActivateView,
  onToggleViewVisibility,
  onDeleteView,
  onMoveView,
  onAddCamera,
  onAdd3DView,
}: HierarchyProps) {
  const items = elements ?? []
  const byId = new Map(items.map((e) => [e.id, e]))
  const closeAddMenu = (event: React.MouseEvent<HTMLButtonElement>) => {
    const root = event.currentTarget.closest('details')
    if (root) root.removeAttribute('open')
  }
  const reorderableViewIds = views.filter((v) => v.canReorder).map((v) => v.id)
  const handleElementDragStart = (event: React.DragEvent<HTMLButtonElement>, element: ElementInfo) => {
    if (element.type !== 'camera') return
    const payload = buildHierarchyElementDragPayload(element)
    event.dataTransfer.setData(HIERARCHY_DRAG_MIME, JSON.stringify(payload))
    event.dataTransfer.setData('text/plain', `${element.type}:${element.id}`)
    event.dataTransfer.effectAllowed = 'copy'
  }

  return (
    <div className="scene-layout">
      <div>
        <h2 className="panel-title">Scene</h2>
        <div className="panel-subtitle">Scene hierarchy and views</div>
      </div>

      <div className="section-card">
        <div className="section-head" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
          <span>Hierarchy</span>
          <details className="scene-add-menu">
            <summary className="icon-btn" title="Add">
              +
            </summary>
            <div className="scene-add-menu-popover">
              <button
                type="button"
                className="toolbar-btn"
                onClick={(event) => {
                  closeAddMenu(event)
                  onAddCamera()
                }}
              >
                Add Camera
              </button>
              <button
                type="button"
                className="toolbar-btn"
                onClick={(event) => {
                  closeAddMenu(event)
                  onAdd3DView()
                }}
              >
                Add 3D View
              </button>
            </div>
          </details>
        </div>
        <div style={{ padding: 8, display: 'grid', gap: 8 }}>
          {views.map((view) => {
            const reorderIndex = reorderableViewIds.indexOf(view.id)
            const isFirstReorderable = reorderIndex <= 0
            const isLastReorderable = reorderIndex < 0 || reorderIndex >= reorderableViewIds.length - 1
            const hasSelectedChild = view.elementIds.includes(selectedId ?? '')

            return (
              <div key={view.id} className="view-item">
                <div className="view-head">
                  <input
                    type="checkbox"
                    checked={view.visible}
                    title={view.visible ? 'Hide view' : 'Show view'}
                    onChange={(e) => onToggleViewVisibility(view.id, e.target.checked)}
                  />
                  <button
                    type="button"
                    className={`view-btn${hasSelectedChild ? ' selected' : ''}`}
                    onClick={() => onActivateView(view.id)}
                    title={view.id}
                  >
                    <div className="view-title-line">
                      <span className={`type-icon-wrap view-kind ${view.kind === '3d' ? 'kind-3d' : view.kind === 'camera' ? 'type-camera' : 'kind-2d'}`}>
                        <MinimalIcon kind={view.kind === '3d' ? 'view3d' : view.kind === 'camera' ? 'camera' : 'view2d'} />
                      </span>
                      <div className="primary">{view.name}</div>
                    </div>
                  </button>
                  <button
                    className="icon-btn"
                    type="button"
                    title="Move up"
                    disabled={!view.canReorder || isFirstReorderable}
                    onClick={() => onMoveView(view.id, 'up')}
                  >
                    ↑
                  </button>
                  <button
                    className="icon-btn"
                    type="button"
                    title="Move down"
                    disabled={!view.canReorder || isLastReorderable}
                    onClick={() => onMoveView(view.id, 'down')}
                  >
                    ↓
                  </button>
                  <button
                    className="icon-btn"
                    type="button"
                    title="Delete view"
                    disabled={!view.canDelete}
                    onClick={() => onDeleteView(view.id)}
                  >
                    ×
                  </button>
                </div>

                <div style={{ borderTop: '1px solid var(--line)', padding: '6px 6px 6px 14px', display: 'grid', gap: 4 }}>
                  {view.elementIds.map((elementId) => {
                    const element = byId.get(elementId)
                    if (!element) {
                      return (
                        <div key={elementId} className="panel-subtitle">
                          (missing element)
                        </div>
                      )
                    }

                    const selected = selectedId === element.id
                    const isVisible = element.visible !== false

                    return (
                      <div key={element.id} className="item-row">
                        <input
                          type="checkbox"
                          title={isVisible ? 'Hide element' : 'Show element'}
                          checked={isVisible}
                          onChange={(e) => onToggleVisibility(element.id, e.target.checked)}
                        />
                        <button
                          className={`item-btn${selected ? ' selected' : ''}${element.type === 'camera' ? ' draggable' : ''}`}
                          onClick={() => onSelect(element.id)}
                          onDoubleClick={() => onFocus(element.id)}
                          title={element.id}
                          draggable={element.type === 'camera'}
                          onDragStart={(event) => handleElementDragStart(event, element)}
                        >
                          <div className="item-title-line">
                            <span className={`type-icon-wrap type-${element.type}`}>
                              <MinimalIcon kind={iconKindForElement(element.type)} />
                            </span>
                            <div className="primary">{element.name}</div>
                          </div>
                        </button>
                        <button className="icon-btn" onClick={() => onDelete(element.id)} title="Remove element">
                          ×
                        </button>
                      </div>
                    )
                  })}
                  {view.elementIds.length === 0 && <div className="panel-subtitle">(no elements)</div>}
                </div>
              </div>
            )
          })}
          {views.length === 0 && <div className="panel-subtitle">(no views)</div>}
        </div>
      </div>

      <div className="panel-subtitle">Tip: Double-click an image or camera element to activate its linked view.</div>
    </div>
  )
}
