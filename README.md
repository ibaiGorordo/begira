# begira

Minimal point-cloud viewer (Three.js + React) with a Python API.

<img width="2154" height="1190" alt="image" src="https://github.com/user-attachments/assets/52a2d0f2-d8a2-45d8-ac5f-9afaf3c9fa05" />


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

# If you're using uv-managed environments, prefer:
uv pip install -e .

# Otherwise, with a normal venv:
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

### Frontend note

If you installed `begira` as a Python package, the full React/Three.js viewer UI is included in the wheel and served automatically.

If you're developing on the repo and want to rebuild the UI:

```bash
cd frontend
npm install
npm run build
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

## Using the `uv` tool (optional)

If you use the `uv` developer tool (which uses `uv.lock` to provide reproducible environments), you can install and run the project with the locked environment:

```bash
# Install the uv CLI (one-time)
python -m pip install --upgrade uv

# Create and install the locked environment defined by uv.lock
uv install

# Run the package CLI via uv (uses the environment from uv.lock)
uv run begira

# Run the server using uv (equivalent to 'uvicorn begira.server:app')
uv run uvicorn begira.server:app --reload --host 127.0.0.1 --port 8000

# Run tests inside the locked environment
uv run pytest -q
```

This will use the `uv.lock` file included in the repository to install exact dependency versions and run commands in that environment.

## Testing the "fresh install" experience (no Node/npm)

To test what an end user sees right after installing the Python package (no `npm` build), you want to install from a built wheel into a clean environment.

Example using `uv`:

```bash
# From the repo root
rm -rf /tmp/begira-test && mkdir -p /tmp/begira-test

# Build a wheel (should already include the built frontend assets)
uv pip install -U build
uv run python -m build -w

# Create a clean env and install the wheel
cd /tmp/begira-test
uv venv
uv pip install /Users/gorordoibai/Desktop/myprojects/begira/dist/*.whl

# Run the example (should open the full UI without running npm)
uv run python -c "import begira; s=begira.run(open_browser=False); print(s.url)"
```

If the root page errors with "frontend assets are missing", it means the wheel was built without including the Vite build output.

## Maintainers: bundling the full frontend into the Python package

The full Vite build output is packaged under:

- `src/begira/_frontend/dist/`

Before publishing to PyPI, build and sync the frontend into that folder:

```bash
python scripts/sync_frontend_dist.py
```

Then build your distributions:

```bash
python -m build
```

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
