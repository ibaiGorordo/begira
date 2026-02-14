import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react'
import { Actions, DockLocation, Layout, Model, type IJsonModel, type Node, type TabNode, type TabSetNode } from 'flexlayout-react'
import PointCloudCanvas from './PointCloudCanvas'
import type { ElementInfo } from './api'

const ROOT_ROW_ID = 'workspace-root-row'
const THREE_D_TABSET_ID = 'workspace-tabset-3d'
const THREE_D_TAB_ID = 'view-3d-main'

type TransformMode = 'translate' | 'rotate'

export type DockImageView = {
  id: string
  imageId: string
  name: string
  visible: boolean
}

type DockWorkspaceProps = {
  pointclouds: ElementInfo[]
  gaussians: ElementInfo[]
  cameras: ElementInfo[]
  images: ElementInfo[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  focusTarget: string | null
  onFocus: (id: string | null) => void
  activeCameraId: string | null
  transformMode: TransformMode
  onTransformModeChange: (mode: TransformMode) => void
  onTransformCommit: (id: string, position: [number, number, number], rotation: [number, number, number, number]) => void
  show3D: boolean
  imageViews: DockImageView[]
}

export type DockWorkspaceHandle = {
  focusImage: (imageId: string) => void
}

const initialLayout: IJsonModel = {
  global: {
    tabEnableClose: true,
    tabSetEnableClose: false,
    tabSetEnableDeleteWhenEmpty: true,
    tabSetEnableMaximize: true,
    tabSetEnableDrop: true,
    tabEnableRename: false,
  },
  layout: {
    type: 'row',
    id: ROOT_ROW_ID,
    children: [
      {
        type: 'tabset',
        id: THREE_D_TABSET_ID,
        enableDeleteWhenEmpty: true,
        weight: 100,
        children: [
          {
            type: 'tab',
            id: THREE_D_TAB_ID,
            component: '3d',
            name: '3D',
            enableClose: false,
          },
        ],
      },
    ],
  },
}

function asNumber(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v
  return null
}

function getTabConfigImageId(tab: TabNode): string | null {
  const cfg = tab.getConfig() as { imageId?: string } | undefined
  return typeof cfg?.imageId === 'string' ? cfg.imageId : null
}

function findImageTab(model: Model, imageId: string): TabNode | null {
  let hit: TabNode | null = null
  model.visitNodes((node: Node) => {
    if (hit) return
    if (node.getType() !== 'tab') return
    const tab = node as TabNode
    if (tab.getComponent() !== 'image') return
    if (getTabConfigImageId(tab) !== imageId) return
    hit = tab
  })
  return hit
}

function findImageTabset(model: Model): TabSetNode | null {
  let hit: TabSetNode | null = null
  model.visitNodes((node: Node) => {
    if (hit) return
    if (node.getType() !== 'tabset') return
    const tabset = node as TabSetNode
    const hasImageTab = tabset
      .getChildren()
      .some((child) => child.getType() === 'tab' && (child as TabNode).getComponent() === 'image')
    if (hasImageTab) hit = tabset
  })
  return hit
}

function findAnyTabset(model: Model): TabSetNode | null {
  let hit: TabSetNode | null = null
  model.visitNodes((node: Node) => {
    if (hit) return
    if (node.getType() !== 'tabset') return
    hit = node as TabSetNode
  })
  return hit
}

function ThreeDTabPane({
  pointclouds,
  gaussians,
  cameras,
  selectedId,
  onSelect,
  focusTarget,
  onFocus,
  activeCameraId,
  transformMode,
  onTransformModeChange,
  onTransformCommit,
}: Omit<DockWorkspaceProps, 'show3D' | 'imageViews' | 'images'> & { images: ElementInfo[] }) {
  const cameraVisuals = useMemo(
    () =>
      cameras.map((c) => ({
        id: c.id,
        position: c.position,
        rotation: c.rotation,
        fov: c.fov,
        near: c.near,
        far: c.far,
        visible: c.visible,
      })),
    [cameras]
  )

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div className="workspace-overlay">
        <button
          className={transformMode === 'translate' ? 'active' : ''}
          onClick={() => onTransformModeChange('translate')}
        >
          Move
        </button>
        <button
          className={transformMode === 'rotate' ? 'active' : ''}
          onClick={() => onTransformModeChange('rotate')}
        >
          Rotate
        </button>
      </div>

      <PointCloudCanvas
        cloudIds={pointclouds.map((c) => c.id)}
        gaussianIds={gaussians.map((c) => c.id)}
        cameraIds={cameras.map((c) => c.id)}
        cameraVisuals={cameraVisuals}
        selectedId={selectedId}
        onSelect={onSelect}
        focusTarget={focusTarget}
        onFocus={onFocus}
        cloudMetaBounds={pointclouds.map((c) => c.bounds).filter(Boolean) as { min: [number, number, number]; max: [number, number, number] }[]}
        gaussianMetaBounds={gaussians.map((c) => c.bounds).filter(Boolean) as { min: [number, number, number]; max: [number, number, number] }[]}
        cameraMetaBounds={cameras.map((c) => c.bounds).filter(Boolean) as { min: [number, number, number]; max: [number, number, number] }[]}
        activeCameraId={activeCameraId}
        transformMode={transformMode}
        onTransformCommit={(id, position, rotation) => onTransformCommit(id, position, rotation)}
      />
    </div>
  )
}

function ImageTabPane({
  node,
  imagesById,
}: {
  node: TabNode
  imagesById: Map<string, ElementInfo>
}) {
  const imageId = getTabConfigImageId(node)
  const selectedImage = imageId ? imagesById.get(imageId) ?? null : null

  if (!selectedImage) {
    return (
      <div className="empty-pane">
        Linked image is not available.
      </div>
    )
  }

  const summary = selectedImage.summary ?? {}
  const width = asNumber((summary as Record<string, unknown>).width)
  const height = asNumber((summary as Record<string, unknown>).height)
  const channels = asNumber((summary as Record<string, unknown>).channels)
  const imageUrl = `/api/elements/${encodeURIComponent(selectedImage.id)}/payloads/image?rev=${selectedImage.revision}`

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
      <div className="image-tab-header">
        <div className="name">{selectedImage.name}</div>
        <div className="meta">
          {width && height ? `${width}x${height}` : 'unknown size'}
          {channels ? `, ${channels}ch` : ''}
        </div>
      </div>

      <div className="image-pane" style={{ flex: 1, minHeight: 0 }}>
        <img
          src={imageUrl}
          alt={selectedImage.name}
          style={{
            maxWidth: '100%',
            maxHeight: '100%',
            objectFit: 'contain',
            imageRendering: 'auto',
          }}
        />
      </div>
    </div>
  )
}

function imageTabJson(view: DockImageView) {
  return {
    type: 'tab' as const,
    id: view.id,
    component: 'image',
    name: view.name,
    config: { imageId: view.imageId },
    enableClose: false,
  }
}

const DockWorkspace = forwardRef<DockWorkspaceHandle, DockWorkspaceProps>(function DockWorkspace(props, ref) {
  const [model] = useState(() => Model.fromJson(initialLayout))
  const sanitizingRef = useRef(false)

  const imagesById = useMemo(() => new Map(props.images.map((img) => [img.id, img])), [props.images])

  useEffect(() => {
    model.setOnCreateTabSet(() => ({
      type: 'tabset',
      enableDeleteWhenEmpty: true,
      enableClose: false,
      enableDrop: true,
      enableDrag: true,
      enableMaximize: true,
    }))
  }, [model])

  const sanitizeEmptyTabsets = useCallback(() => {
    if (sanitizingRef.current) return
    sanitizingRef.current = true
    try {
      let changed = true
      while (changed) {
        changed = false
        const emptyTabsetIds: string[] = []
        model.visitNodes((node: Node) => {
          if (node.getType() !== 'tabset') return
          const tabset = node as TabSetNode
          if (tabset.getChildren().length === 0) {
            emptyTabsetIds.push(tabset.getId())
          }
        })

        if (emptyTabsetIds.length === 0) break
        changed = true
        for (const id of emptyTabsetIds) {
          model.doAction(Actions.deleteTabset(id))
        }
      }
    } finally {
      sanitizingRef.current = false
    }
  }, [model])

  useEffect(() => {
    const threeDTab = model.getNodeById(THREE_D_TAB_ID)

    if (!props.show3D) {
      if (threeDTab) model.doAction(Actions.deleteTab(THREE_D_TAB_ID))
      sanitizeEmptyTabsets()
      return
    }

    if (!threeDTab) {
      const anyTabset = findAnyTabset(model)
      const targetId = anyTabset?.getId() ?? ROOT_ROW_ID
      const dock = anyTabset ? DockLocation.LEFT : DockLocation.CENTER
      model.doAction(
        Actions.addNode(
          {
            type: 'tab' as const,
            id: THREE_D_TAB_ID,
            component: '3d',
            name: '3D',
            enableClose: false,
          },
          targetId,
          dock,
          -1,
          true,
        )
      )
    }
    sanitizeEmptyTabsets()
  }, [model, props.show3D, sanitizeEmptyTabsets])

  useEffect(() => {
    const desired = props.imageViews.filter((v) => v.visible)
    const desiredByImageId = new Map(desired.map((v) => [v.imageId, v]))

    const existingImageTabs: TabNode[] = []
    model.visitNodes((node: Node) => {
      if (node.getType() !== 'tab') return
      const tab = node as TabNode
      if (tab.getComponent() === 'image') existingImageTabs.push(tab)
    })

    for (const tab of existingImageTabs) {
      const imageId = getTabConfigImageId(tab)
      const expected = imageId ? desiredByImageId.get(imageId) : undefined
      if (!expected || expected.id !== tab.getId()) {
        model.doAction(Actions.deleteTab(tab.getId()))
      }
    }

    let imageTabset = findImageTabset(model)

    if (desired.length === 0) {
      if (imageTabset) {
        const remainingImageTabs = imageTabset
          .getChildren()
          .filter((child) => child.getType() === 'tab' && (child as TabNode).getComponent() === 'image')
        if (remainingImageTabs.length === 0) {
          model.doAction(Actions.deleteTabset(imageTabset.getId()))
        }
      }
      return
    }

    let imageTabsetId = imageTabset?.getId() ?? null

    for (let i = 0; i < desired.length; i += 1) {
      const view = desired[i]
      const existing = model.getNodeById(view.id)
      if (!existing || existing.getType() !== 'tab' || (existing as TabNode).getComponent() !== 'image') {
        if (imageTabsetId) {
          model.doAction(Actions.addNode(imageTabJson(view), imageTabsetId, DockLocation.CENTER, i, false))
        } else {
          const hostTabset = findAnyTabset(model)
          const hostId = hostTabset?.getId() ?? ROOT_ROW_ID
          const hostDock = hostTabset ? DockLocation.RIGHT : DockLocation.CENTER
          const added = model.doAction(Actions.addNode(imageTabJson(view), hostId, hostDock, -1, false))
          const parent = added?.getParent()
          if (parent && parent.getType() === 'tabset') {
            imageTabsetId = parent.getId()
          }
        }
      } else {
        const tab = existing as TabNode
        if (tab.getName() !== view.name || getTabConfigImageId(tab) !== view.imageId) {
          model.doAction(
            Actions.updateNodeAttributes(view.id, {
              name: view.name,
              config: { imageId: view.imageId },
            })
          )
        }
      }
    }

    imageTabset = findImageTabset(model)
    if (!imageTabset) return

    const tabsetId = imageTabset.getId()
    const currentIds = imageTabset
      .getChildren()
      .filter((n) => n.getType() === 'tab' && (n as TabNode).getComponent() === 'image')
      .map((n) => n.getId())

    for (let i = 0; i < desired.length; i += 1) {
      const desiredId = desired[i].id
      const currentIndex = currentIds.indexOf(desiredId)
      if (currentIndex === -1 || currentIndex === i) continue
      model.doAction(Actions.moveNode(desiredId, tabsetId, DockLocation.CENTER, i, false))
      currentIds.splice(currentIndex, 1)
      currentIds.splice(i, 0, desiredId)
    }

    sanitizeEmptyTabsets()
  }, [model, props.imageViews, sanitizeEmptyTabsets])

  const focusImage = useCallback(
    (imageId: string) => {
      const tab = findImageTab(model, imageId)
      if (!tab) return
      model.doAction(Actions.selectTab(tab.getId()))
      props.onSelect(imageId)
    },
    [model, props]
  )

  useImperativeHandle(
    ref,
    () => ({
      focusImage,
    }),
    [focusImage]
  )

  const factory = useCallback(
    (node: TabNode) => {
      const component = node.getComponent()
      if (component === '3d') return <ThreeDTabPane {...props} />
      if (component === 'image') return <ImageTabPane node={node} imagesById={imagesById} />
      return <div className="empty-pane">Unknown view component: {component}</div>
    },
    [imagesById, props]
  )

  return (
    <div className="begira-dock-workspace" style={{ width: '100%', height: '100%', minHeight: 0, minWidth: 0, position: 'relative', overflow: 'hidden' }}>
      <Layout
        model={model}
        factory={factory}
        onModelChange={() => {
          sanitizeEmptyTabsets()
        }}
      />
    </div>
  )
})

export default DockWorkspace
