# begira (WIP)

Minimal 3D viewer with a Python API and a React/Three.js frontend.
<img width="2154" height="1190" alt="image" src="https://github.com/user-attachments/assets/52a2d0f2-d8a2-45d8-ac5f-9afaf3c9fa05" />

## Features
- Point clouds and Gaussian splats with orbit controls
- Image logging (OpenCV/Pillow) with 2D image view
- Color modes: logged, solid, height, depth
- Selectable colormaps for height/depth
- LOD for gaussians and focus on selection

## Repo layout
- `src/begira/` → Python library
  - `src/begira/core/` → domain model/state/timeline/registry
  - `src/begira/api/` → HTTP API app, parsing, serializers, routes
  - `src/begira/sdk/` → Python client + handles
  - `src/begira/runtime/` → app/server/web mounting
  - `src/begira/io/` → image + ply loaders/encoders
- `frontend/` → web viewer (Vite + React + three)
  - `frontend/src/app/` → app shell + app-level runtime state
  - `frontend/src/features/` → hierarchy/inspector/timeline/workspace/3d-view features
  - `frontend/src/shared/api/` → API types and fetch clients
  - `frontend/src/viewer/` → viewer rendering modules
- `examples/` → small runnable scripts

## Quickstart

Install the Python package (editable is fine for dev):

```bash
python -m pip install -e .
```

Install directly from Git:

```bash
python -m pip install "begira @ git+https://github.com/ibaiGorordo/begira.git"
```

This works without local npm steps as long as `src/begira/_frontend/dist/` is
present in the repository (prebuilt frontend bundle tracked in git).

Run the example:

```bash
python examples/hello_begira.py
```

Or launch directly:

```bash
python -m begira --port 8000
```

## Frontend build (packaged UI)

To build the UI that the Python server will serve:

```bash
cd frontend
npm install
npm run build
```

This writes the build directly into:
`src/begira/_frontend/dist/`

## Dev mode (hot reload)

Backend:

```bash
uvicorn begira.runtime.app:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm run dev
```

Open the Vite URL (usually http://localhost:5173).
Use the top-right view buttons (`3D`, `Images`, `Split`) to switch between 3D and 2D image views.

## Packaging the frontend

The built UI is bundled under `src/begira/_frontend/dist/`.
You can run the helper script to install frontend deps and build in one step:

```bash
python scripts/sync_frontend_dist.py
```

Then build distributions:

```bash
python -m build
```

## API (current)
- `GET /api/events` → `{ globalRevision }` (polling)
- `GET /api/elements` → list renderable elements
- `GET /api/elements/{id}/meta` → element metadata
- `PATCH /api/elements/{id}/meta` → update element settings (pointcloud `pointSize`)
- `GET /api/elements/{id}/payloads/{name}` → binary payloads

Pointcloud payload format:
- XYZ: 3×float32 little‑endian
- optional RGB: 3×uint8

## Log A Camera From Python

```python
import begira

viewer = begira.run(port=8000)

points = viewer.log_points("points", positions, colors)
image = viewer.log_image("frame", bgr_image)  # OpenCV ndarray (H,W,3)
camera = viewer.log_camera(
    "main_camera",
    position=(2.0, 1.5, 3.0),
    fov=60.0,
    near=0.01,
    far=1000.0,
)

camera.look_at(points, 1.0)  # place camera 1m from target
points.disable()   # hide
points.enable()    # show
points.delete()    # soft-delete

# live metadata-backed properties
print(points.name, points.position, points.orientation)
print(points.rotation_matrix)
print(points.count)         # pointcloud/gaussians
print(points.kind, points.visible, points.deleted, points.bounds)
print(camera.fov, camera.near, camera.far)
print(image.width, image.height, image.channels, image.mime_type)
```
