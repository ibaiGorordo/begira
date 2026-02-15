# Begira Architecture

## Backend
- `src/begira/core/`: domain entities, timeline model, animation model, registry service.
- `src/begira/api/`: FastAPI app composition, request parsing, response serialization, route mounting.
- `src/begira/sdk/`: user-facing Python SDK client and object handles.
- `src/begira/runtime/`: server runner and frontend static mounting.
- `src/begira/io/`: file/image loaders and payload encoders.

Compatibility: legacy module paths in `src/begira/*.py` re-export the new package layout.

## Frontend
- `frontend/src/app/`: app shell, app-level state helpers, runtime shared state.
- `frontend/src/features/`: feature-organized modules (`hierarchy`, `inspector`, `timeline`, `workspace`, `view3d`).
- `frontend/src/shared/api/`: typed API contracts and request functions.
- `frontend/src/shared/utils/`: shared utilities.
- `frontend/src/viewer/`: compatibility layer for prior import paths.

## Runtime Data Flow
1. SDK writes element payload/meta to API.
2. API updates registry temporal state.
3. Viewer polls events + timeline + element/meta payloads.
4. Viewer renders 3D/2D views through workspace layout.

## Compatibility Guarantees
- `begira.run`, `BegiraClient`, and handle classes remain stable.
- Existing `/api/*` endpoints remain unchanged.
- Existing frontend runtime request/response shapes remain unchanged.
