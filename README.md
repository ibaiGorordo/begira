# begira

Minimal point-cloud viewer (Three.js + React) with a Python API.

## Repo layout (recommended)

- `src/begira/` → the Python library (API server + launcher)
- `frontend/` → the web viewer (Vite + React + three)
  - `frontend/src/` → viewer code
  - `frontend/dist/` → built static files (served by Python; not committed)
- `examples/` → small runnable scripts

This is a common pattern for projects like "Python server + embedded web UI": keep a real frontend project (Node toolchain) but ship/serve *its build output* from the Python app.

## What you get
- **Python library**: exposes `begira.run()` (Viser/Rerun-style one-call launcher)
- **FastAPI backend**: serves a sample point cloud as a single binary payload.
- **TypeScript frontend**: Vite + React + react-three-fiber rendering the cloud with orbit controls.

## Examples

- `examples/hello_begira.py` — starts the viewer with one script.

## Run (single command)

One-time: build the frontend bundle (so Python can serve it):

```bash
cd /Users/ibaigorordo/Desktop/myprojects/begira/frontend
npm install
npm run build
```

Install the Python package (editable is fine for dev):

```bash
cd /Users/ibaigorordo/Desktop/myprojects/begira
python -m pip install -e .
```

Now launch everything from Python:

```bash
python -c "import begira; begira.run()"
```

Or run as a module (blocks like a normal server):

```bash
python -m begira --port 8000
```

## Run (dev, two processes)

If you prefer hot-reload frontend development:

### Backend

```bash
cd /Users/ibaigorordo/Desktop/myprojects/begira
uvicorn begira.server:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd /Users/ibaigorordo/Desktop/myprojects/begira/frontend
npm run dev
```

Open the printed Vite URL (usually http://localhost:5173).

## API contract (current)
- `GET /api/pointclouds` → list available clouds
- `GET /api/pointclouds/{id}/meta` → metadata describing how to parse the payload
- `GET /api/pointclouds/{id}/points` → `application/octet-stream` interleaved bytes per point:
  - XYZ: 3× float32 (little-endian)
  - RGB: 3× uint8 (optional, present in `sample`)

## Next steps
- Add file loaders (PLY/LAS/LAZ) and a registry.
- Add decimation / LOD.
- Add picking/hover readouts and axes helpers.

